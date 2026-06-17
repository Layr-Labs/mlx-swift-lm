// Copyright © 2026 Apple Inc.
//
// Regression test for DAR-325: a Metal live-resource COUNT leak on the
// single-token decode path of the continuous-batching KV caches
// (`BatchKVCache.swift`).
//
// ── TRUE ROOT CAUSE (established empirically; see the DAR-325 investigation) ──
//
// Each decode step the caches mutate two *metadata* MLXArrays into fresh lazy
// graphs that are never collapsed by the generation loop:
//   • `batchOffset += 1`   (both BatchKVCache and BatchRotatingKVCache)
//   • `leftPadding -= trim` (BatchRotatingKVCache only, once the window is full)
// Each `+ Int32(...)` / `- Int32(...)` allocates a tiny scalar buffer that the
// growing lazy add-chain pins live. The chain only collapses when something
// *consumes* the array (forcing eval).
//
// The keys/values are NOT a leak: every layer's attention consumes the returned
// `(k, v)` each step, which collapses their slice-of-`concatenated(...)` graph.
// (The earlier "heavy" fix that `contiguous()`-copied + `eval`'d keys/values was
// therefore unnecessary work.)
//
// The metadata leaks because Gemma 4's decode forward consumes each metadata
// array from only ONE representative cache:
//   • RoPE applies a single shared `perRowOffset` captured from the FIRST cache
//     (Gemma4.swift:1264-1266, threaded into every layer at :1300) — so only
//     cache[0].batchOffset is consumed; every other cache's batchOffset chain
//     accretes one scalar/step.
//   • The sliding mask is built ONCE from `localCache[firstSlidingCacheIdx]`
//     (Gemma4.swift:1244, reused for every sliding layer at :1283) and its
//     `makeMask` calls `leftPadding.max().item()` — so only that one sliding
//     cache's leftPadding chain collapses; the others accrete one scalar/step.
//
// Net live-resource growth in production gemma-4 (25 sliding + ~6 full caches):
//   ≈ 2·(25−1)  [sliding: batchOffset + leftPadding]
//   +  (6−1)    [full:    batchOffset only]
//   ≈ 53 / step, with FLAT bytes (scalars are tiny), until `numResources` hits
//   the iogpu ceiling (~499000) and the process aborts with
//   `[metal::malloc] Resource limit exceeded`.
//
// ── THE FIX (minimal, proven) ──
// On the single-token decode path only, `asyncEval` the leaking metadata so its
// lazy chain detaches and the prior step's scalar buffer becomes reclaimable:
//   • BatchRotatingKVCache.update(): `asyncEval(batchOffset, leftPadding)`
//   • BatchKVCache.update():          `asyncEval(batchOffset)`
// `asyncEval` (no hard GPU sync) suffices; keys/values are left untouched; the
// bounded prefill path (`stepCount > 1`) is untouched.
//
// ── FAITHFULNESS RULES (violating any hides the leak) ──
//   1. Reproduce the MULTI-cache shape: a single cache cannot expose the
//      production mechanism (one cache would have its own metadata consumed).
//   2. Consume `(k, v)` from EVERY cache each step (mirrors per-layer attention)
//      but consume `batchOffset`/`leftPadding` from only ONE representative
//      cache (mirrors the shared RoPE offset + shared sliding mask). Eval a
//      tiny PROBE derived from those — never the cache state directly.
//   3. Measure `numResources` AFTER `clearCache()` so only LIVE (graph-pinned,
//      non-reclaimable) buffers count; take the min over a few samples to reject
//      transient cross-suite allocations.
//   4. Raise `cacheLimit`/`memoryLimit` so the byte-driven trim can't mask the
//      COUNT; restore on exit.

import Foundation
import MLX
import MLXLMCommon
import Testing

@Suite("BatchKVCache decode resource-count leak (DAR-325)", .serialized)
struct BatchRotatingKVCacheResourceTests {

    private let steps = 4000
    private let warmup = 500
    private let H = 2
    private let D = 4

    /// LIVE Metal resource count: trim reclaimable buffers first, then take the
    /// min over a few rapid samples so a concurrently-running suite's transient
    /// live buffers do not inflate the reading. The DAR-325 leak is graph-pinned
    /// state that `clearCache()` cannot reclaim, so it survives this.
    private func liveResourceCount(samples: Int = 4) -> Int {
        var m = Int.max
        for _ in 0 ..< samples {
            MLX.Memory.clearCache()
            m = min(m, MLX.Memory.numResources)
        }
        return m
    }

    private func token() -> MLXArray { MLXArray.ones([1, H, 1, D], dtype: .float32) }

    /// Per-step average live-resource growth between warmup (steady state) and
    /// the end of the run — the leak rate. `representativeProbe` lets the caller
    /// model the shared single-consumer (RoPE offset / sliding mask) that only
    /// touches cache[0].
    private func decodeSlope(
        caches: [any KVCache],
        representativeProbe: (any KVCache) -> Void
    ) -> (growth: Int, slope: Double) {
        var warmupCount = 0
        for step in 0 ..< steps {
            autoreleasepool {
                var probes: [MLXArray] = []
                probes.reserveCapacity(caches.count * 2)
                for c in caches {
                    let (k, v) = c.update(keys: token(), values: token())
                    // Per-layer attention consumes every cache's (k, v).
                    probes.append(k.sum())
                    probes.append(v.sum())
                }
                // Shared single-consumer touches only the representative cache.
                representativeProbe(caches[0])
                eval(probes)
            }
            if step == warmup { warmupCount = liveResourceCount() }
        }
        let growth = liveResourceCount() - warmupCount
        return (growth, Double(growth) / Double(steps - warmup))
    }

    private func skipIfNoMetal(_ label: String) -> Bool {
        if MLX.Memory.resourceLimit <= 0 {
            print("[DAR-325] \(label): no Metal backend; skipping resource-count assertion")
            return true
        }
        return false
    }

    private func withRaisedLimits(_ body: () -> Void) {
        let savedCache = MLX.Memory.cacheLimit
        let savedMemory = MLX.Memory.memoryLimit
        defer {
            MLX.Memory.cacheLimit = savedCache
            MLX.Memory.memoryLimit = savedMemory
        }
        MLX.Memory.memoryLimit = 80 << 30
        MLX.Memory.cacheLimit = 80 << 30
        body()
    }

    /// A genuine leak grows ~linearly (tens per step → tens of thousands total);
    /// the fix and the control stay flat. The threshold (< 0.5/step) sits two
    /// orders of magnitude under the pre-fix slope (~48/step sliding, ~24/step
    /// full) and well above any clearCache-suppressed cross-suite noise.
    private let maxSlope = 0.5

    // MARK: - Subject: BatchRotatingKVCache (gemma sliding layers)

    @Test("sliding (BatchRotatingKVCache): multi-cache decode is resource-count stable")
    func batchRotatingMultiCacheDecodeIsStable() {
        if skipIfNoMetal("BatchRotatingKVCache") { return }
        withRaisedLimits {
            let N = 25  // gemma-4 sliding-layer count
            let maxSize = 64
            let caches: [any KVCache] = (0 ..< N).map { _ in
                BatchRotatingKVCache(maxSize: maxSize, leftPadding: [0])
            }
            let (growth, slope) = decodeSlope(caches: caches) { rep in
                guard let rep = rep as? BatchRotatingKVCache else { return }
                // Shared RoPE: one cache's batchOffset. Shared sliding mask:
                // makeMask's `leftPadding.max().item()` on the same cache.
                eval(rep.batchOffset.sum())
                _ = rep.leftPadding.max().item(Int32.self)
            }
            print(
                "[DAR-325] BatchRotatingKVCache x\(N): growth=\(growth) over "
                    + "\(steps - warmup) steps, slope=\(slope)/step")
            #expect(
                slope < maxSlope,
                """
                BatchRotatingKVCache leaks Metal resources during decode: \
                \(growth) live buffers over \(steps - warmup) single-token steps \
                (slope \(slope)/step). Both batchOffset and leftPadding chains must \
                be collapsed on the decode path. This is the DAR-325 \
                `[metal::malloc] Resource limit exceeded` crash.
                """)
        }
    }

    // MARK: - Sibling: BatchKVCache (gemma full-attention layers)
    //
    // gemma-4 shares ONE perRowOffset across full AND sliding layers, so the
    // full-attention caches leak batchOffset too (they never mutate leftPadding,
    // so there is no leftPadding leak there — and gpt-oss, which consumes each
    // cache's own batchOffset via the generic applyRotaryPosition, is unaffected
    // either way).

    @Test("full (BatchKVCache): multi-cache decode is resource-count stable")
    func batchFullMultiCacheDecodeIsStable() {
        if skipIfNoMetal("BatchKVCache") { return }
        withRaisedLimits {
            let N = 8
            let caches: [any KVCache] = (0 ..< N).map { _ in BatchKVCache(leftPadding: [0]) }
            let (growth, slope) = decodeSlope(caches: caches) { rep in
                guard let rep = rep as? BatchKVCache else { return }
                eval(rep.batchOffset.sum())  // shared RoPE offset (one cache)
            }
            print(
                "[DAR-325] BatchKVCache x\(N): growth=\(growth) over "
                    + "\(steps - warmup) steps, slope=\(slope)/step")
            #expect(
                slope < maxSlope,
                """
                BatchKVCache leaks Metal resources during decode: \(growth) live \
                buffers over \(steps - warmup) single-token steps (slope \
                \(slope)/step). batchOffset must be collapsed on the decode path.
                """)
        }
    }

    // MARK: - Control: RotatingKVCache (genuinely leak-free)
    //
    // Single-stream RotatingKVCache writes in place into a pre-allocated ring
    // (updateInPlace, KVCache.swift) and tracks offset as a plain Int — no
    // per-step MLXArray metadata chain. It is flat BEFORE and AFTER the fix, so
    // it pins the measurement methodology: the metric itself does not
    // manufacture a leak.

    @Test("control (RotatingKVCache): single-stream decode is resource-count stable")
    func rotatingControlIsStable() {
        if skipIfNoMetal("RotatingKVCache") { return }
        withRaisedLimits {
            let caches: [any KVCache] = (0 ..< 4).map { _ in
                RotatingKVCache(maxSize: 64, keep: 0)
            }
            let (growth, slope) = decodeSlope(caches: caches) { _ in }
            print(
                "[DAR-325] RotatingKVCache control: growth=\(growth) over "
                    + "\(steps - warmup) steps, slope=\(slope)/step")
            #expect(
                slope < maxSlope,
                """
                RotatingKVCache control unexpectedly grew (\(growth)); the \
                measurement methodology is manufacturing a leak.
                """)
        }
    }
}
