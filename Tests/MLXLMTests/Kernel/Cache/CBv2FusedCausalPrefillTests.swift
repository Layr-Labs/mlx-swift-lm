// CBv2FusedCausalPrefillTests.swift
//
// The fused causal prompt route (`CBv2AttentionV1.usesFusedCausalPrefill`):
// a full-attention prompt chunk whose head geometry MLX serves with a fused
// kernel is attended with ONE symbolic `.causal` SDPA call instead of query
// blocks, on both KV backends. Its failure modes are silent — a diagonal
// aligned top-left instead of bottom-right hides history from every query,
// and a shape MLX actually composes would materialize the unblocked score
// tensor — so this suite checks:
//
//  1. Routing: `mlxFusesCausalAttention` never claims a shape MLX composes
//     (head_dim 192/512, head_dim 256 below 1024 queries, off NAX or in
//     fp32, mismatched value width, the qL <= 8 vector range).
//  2. Eligibility: windows, softcaps, spans and bidirectional chunks keep
//     the blocked path; the kill switch is honoured.
//  3. Parity: multi-chunk prefill with history, GQA and GPT-OSS-style sinks
//     on the paged and contiguous backends against an fp32 reference built
//     from ABSOLUTE positions, and against the blocked BOOL-mask calls the
//     route replaces.
//  4. Causality by perturbation on the paged route.
//  5. head_dim 256 at a 1024-query chunk after history: the NAX head-dim
//     split kernel on M5, the composed blocks elsewhere.
//
// Tiny synthetic tensors, seeded RNG, no model download.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

private func fullKind(
    headDim: Int, kvHeads: Int, queryHeads: Int, hasSinks: Bool = false
) -> CBv2LayerKind {
    CBv2LayerKind(
        attention: .full, hasSinks: hasSinks,
        headDim: headDim, kvHeads: kvHeads, queryHeads: queryHeads)
}

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    precondition(a.shape == b.shape, "shape mismatch \(a.shape) vs \(b.shape)")
    let diff = MLX.abs(a.asType(.float32) - b.asType(.float32)).max()
    eval(diff)
    return diff.item(Float.self)
}

private func bitIdentical(_ a: MLXArray, _ b: MLXArray) -> Bool {
    guard a.shape == b.shape else { return false }
    let same = MLX.all(a .== b)
    eval(same)
    return same.item(Bool.self)
}

/// fp32 attention over ABSOLUTE positions: key column `j` is position `j`,
/// query row `i` is position `queryStart + i`, and `k <= q` is the whole
/// mask. Shares nothing with the route under test.
private func absoluteReference(
    queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
    queryStart: Int, sinks: MLXArray?
) -> MLXArray {
    let qpos = MLXArray(Int32(queryStart) ..< Int32(queryStart + queries.dim(2)))
        .expandedDimensions(axis: 1)
    let kpos = MLXArray(Int32(0) ..< Int32(keys.dim(2))).expandedDimensions(axis: 0)
    return PagedAttentionReference.composedAttention(
        queries: queries, keys: keys, values: values, scale: scale,
        boolMask: kpos .<= qpos, sinks: sinks)
}

/// The computation the route replaces on the paged backend: 128-query
/// blocks, each one fused-or-composed SDPA with its absolute BOOL mask over
/// keys `[0, block end)`.
private func blockedBoolMaskReference(
    queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
    queryStart: Int, sinks: MLXArray?, block: Int = 128
) -> MLXArray {
    let l = queries.dim(2)
    var outputs: [MLXArray] = []
    var offset = 0
    while offset < l {
        let count = min(block, l - offset)
        let end = queryStart + offset + count
        let qpos = MLXArray(Int32(queryStart + offset) ..< Int32(end))
            .expandedDimensions(axis: 1)
        let kpos = MLXArray(Int32(0) ..< Int32(end)).expandedDimensions(axis: 0)
        outputs.append(
            MLXFast.scaledDotProductAttention(
                queries: queries[0..., 0..., offset ..< (offset + count), 0...],
                keys: keys[0..., 0..., 0 ..< end, 0...],
                values: values[0..., 0..., 0 ..< end, 0...],
                scale: scale, mask: .array(kpos .<= qpos), sinks: sinks))
        offset += count
    }
    return concatenated(outputs, axis: 2)
}

/// One full-attention layer on one backend, fed prompt chunks in order
/// through the production `updateAndAttend`.
private final class PrefillArm {
    enum Storage { case paged, contiguous }

    let cache: CBv2AttendingLayerCache
    private let release: () -> Void

    init(_ storage: Storage, kind: CBv2LayerKind, dtype: DType, maxChunk: Int, maxLength: Int)
        throws
    {
        switch storage {
        case .paged:
            let backend = try PagedKVBackend(
                layerKinds: [kind],
                config: PagedKVPoolConfig(
                    capacityBytes: 256 << 20, dtype: dtype, maxPrefillChunk: maxChunk,
                    nominalMaxSequenceLength: maxLength))
            let state = try backend.makeSequenceState(
                layerKinds: [kind], promptLength: maxLength, maxLength: maxLength)
            cache = backend.makeLayerCaches()[0]
            cache.setRows([state[0]!])
            release = { backend.release(state) }
        case .contiguous:
            let backend = CBv2ContiguousKVBackend(
                config: CBv2ContiguousBackendConfig(bytesCapacity: 1 << 30, kvDType: dtype))
            let state = try backend.makeSequenceState(
                layerKinds: [kind], promptLength: maxLength, maxLength: maxLength)
            cache = CBv2LayerCache(layerIndex: 0, kind: kind, rows: [state[0]!])
            release = { backend.release(state) }
        }
    }

    deinit { release() }

    func step(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, scale: Float, sinks: MLXArray?)
        -> MLXArray
    {
        cache.updateAndAttend(queries: q, keys: k, values: v, scale: scale, sinks: sinks)
    }
}

/// `[1, heads, n, headDim]` prompt tensors for a multi-chunk prefill.
private struct Prompt {
    let queries: MLXArray
    let keys: MLXArray
    let values: MLXArray

    init(tokens: Int, kind: CBv2LayerKind, dtype: DType, seed: UInt64) {
        MLXRandom.seed(seed)
        queries = MLXRandom.normal([1, kind.queryHeads, tokens, kind.headDim]).asType(dtype)
        keys = MLXRandom.normal([1, kind.kvHeads, tokens, kind.headDim]).asType(dtype)
        values = MLXRandom.normal([1, kind.kvHeads, tokens, kind.valueHeadDim]).asType(dtype)
        eval(queries, keys, values)
    }

    func slice(_ array: MLXArray, _ range: Range<Int>) -> MLXArray {
        array[0..., 0..., range, 0...]
    }
}

extension KernelTests {

    @Suite("CBv2FusedCausalPrefill", .serialized)
    struct CBv2FusedCausalPrefillTests {

        // MARK: - Routing mirror

        @Test func routingMirrorNeverClaimsAComposedShape() {
            func fuses(_ qL: Int, _ d: Int, dv: Int? = nil, dtype: DType = .bfloat16, nax: Bool)
                -> Bool
            {
                CBv2AttentionV1.mlxFusesCausalAttention(
                    queryLength: qL, queryHeadDim: d, valueHeadDim: dv ?? d, dtype: dtype,
                    naxAvailable: nax)
            }
            for nax in [false, true] {
                for d in [64, 72, 80, 96, 128] {
                    for dtype in [DType.float32, .float16, .bfloat16] {
                        #expect(fuses(9, d, dtype: dtype, nax: nax), "d=\(d) qL=9")
                        #expect(fuses(4096, d, dtype: dtype, nax: nax), "d=\(d) qL=4096")
                    }
                    #expect(!fuses(8, d, nax: nax), "qL <= 8 is the vector kernel range")
                    #expect(!fuses(1, d, nax: nax))
                }
                // MLX: "Unfused path is faster" for 192, no full kernel for 512,
                // and q/v widths must match.
                #expect(!fuses(4096, 192, nax: nax))
                #expect(!fuses(4096, 512, nax: nax))
                #expect(!fuses(4096, 192, dv: 128, nax: nax))
                #expect(!fuses(4096, 128, dv: 64, nax: nax))
            }
            // head_dim 256: NAX head-dim-split kernel only, from 1024 queries.
            #expect(fuses(1024, 256, nax: true))
            #expect(fuses(4096, 256, dtype: .float16, nax: true))
            #expect(!fuses(1023, 256, nax: true))
            #expect(!fuses(4096, 256, dtype: .float32, nax: true))
            #expect(!fuses(4096, 256, nax: false))
        }

        @Test func onlyPlainFullAttentionChunksAreEligible() {
            let q64 = MLXArray.zeros([1, 4, 256, 64], dtype: .bfloat16)
            let v64 = MLXArray.zeros([1, 2, 256, 64], dtype: .bfloat16)
            let full = fullKind(headDim: 64, kvHeads: 2, queryHeads: 4)
            func eligible(
                _ kind: CBv2LayerKind, _ q: MLXArray, _ v: MLXArray,
                softcap: Float? = nil, spans: Bool = false
            ) -> Bool {
                CBv2AttentionV1.usesFusedCausalPrefill(
                    kind: kind, queries: q, values: v, softcap: softcap, hasSpanContext: spans)
            }

            // The default test process runs on the GPU stream; the kill switch
            // (`DARKBLOOM_TTFT_FUSED_CAUSAL_PREFILL=0`) turns every case off.
            let enabled = CBv2AttentionV1.fusedCausalPrefillEnabled
            #expect(eligible(full, q64, v64) == enabled)
            var sinks = full
            sinks.hasSinks = true
            #expect(
                eligible(sinks, q64, v64) == enabled, "sinks are supported by the fused kernels")

            var sliding = full
            sliding.attention = .slidingWindow(128)
            #expect(!eligible(sliding, q64, v64), "windows need an array mask")
            var bidirectional = full
            bidirectional.isBidirectional = true
            #expect(!eligible(bidirectional, q64, v64))
            #expect(!eligible(full, q64, v64, softcap: 30), "softcap takes the composed reference")
            #expect(!eligible(full, q64, v64, spans: true), "span overlays need an array mask")

            // Gemma 4's head geometries never qualify; Qwen's head_dim 256 does
            // only on NAX from 1024 queries.
            let q512 = MLXArray.zeros([1, 16, 2048, 512], dtype: .bfloat16)
            let v512 = MLXArray.zeros([1, 2, 2048, 512], dtype: .bfloat16)
            #expect(!eligible(fullKind(headDim: 512, kvHeads: 2, queryHeads: 16), q512, v512))
            let qwen = fullKind(headDim: 256, kvHeads: 4, queryHeads: 16)
            let q256 = MLXArray.zeros([1, 16, 2048, 256], dtype: .bfloat16)
            let v256 = MLXArray.zeros([1, 4, 2048, 256], dtype: .bfloat16)
            #expect(eligible(qwen, q256, v256) == (enabled && MiMoV26NAXGatherQMM.naxAvailable))
            let q256Short = MLXArray.zeros([1, 16, 512, 256], dtype: .bfloat16)
            let v256Short = MLXArray.zeros([1, 4, 512, 256], dtype: .bfloat16)
            #expect(!eligible(qwen, q256Short, v256Short))
        }

        // MARK: - Numerics

        /// GPT-OSS geometry (head_dim 64, GQA 8, learned sinks) over a
        /// three-chunk prompt, so later chunks attend gathered history: both
        /// backends must match the absolute-position reference and the blocked
        /// BOOL-mask calls the route replaces.
        @Test(arguments: [DType.float32, .bfloat16])
        func sinkModelPrefillMatchesReferencesOnBothBackends(dtype: DType) throws {
            let kind = fullKind(headDim: 64, kvHeads: 2, queryHeads: 16, hasSinks: true)
            let scale: Float = 1 / Float(64).squareRoot()
            let chunks = [384, 256, 200]
            let total = chunks.reduce(0, +)
            let prompt = Prompt(tokens: total, kind: kind, dtype: dtype, seed: 7)
            MLXRandom.seed(11)
            let sinks = MLXRandom.normal([kind.queryHeads]).asType(dtype)

            let paged = try PrefillArm(
                .paged, kind: kind, dtype: dtype, maxChunk: chunks.max()!, maxLength: total)
            let contiguous = try PrefillArm(
                .contiguous, kind: kind, dtype: dtype, maxChunk: chunks.max()!, maxLength: total)
            let tolerance: Float = dtype == .float32 ? 2e-3 : 3e-2

            var start = 0
            for chunk in chunks {
                let range = start ..< (start + chunk)
                let q = prompt.slice(prompt.queries, range)
                let k = prompt.slice(prompt.keys, range)
                let v = prompt.slice(prompt.values, range)
                let gotPaged = paged.step(q, k, v, scale: scale, sinks: sinks)
                let gotContiguous = contiguous.step(q, k, v, scale: scale, sinks: sinks)

                let visible = 0 ..< range.upperBound
                let reference = absoluteReference(
                    queries: q, keys: prompt.slice(prompt.keys, visible),
                    values: prompt.slice(prompt.values, visible), scale: scale,
                    queryStart: start, sinks: sinks)
                let blocked = blockedBoolMaskReference(
                    queries: q, keys: prompt.slice(prompt.keys, visible),
                    values: prompt.slice(prompt.values, visible), scale: scale,
                    queryStart: start, sinks: sinks)
                let dPaged = maxAbsDiff(gotPaged, reference)
                let dContiguous = maxAbsDiff(gotContiguous, reference)
                let dBlocked = maxAbsDiff(gotPaged, blocked)
                print(
                    "[fused-causal] \(dtype) chunk \(range): |paged-ref| \(dPaged) "
                        + "|contig-ref| \(dContiguous) |paged-blocked| \(dBlocked)")
                #expect(gotPaged.shape == [1, kind.queryHeads, chunk, kind.headDim])
                #expect(dPaged < tolerance, "paged chunk \(range) vs absolute reference")
                #expect(dContiguous < tolerance, "contiguous chunk \(range) vs absolute reference")
                #expect(dBlocked < tolerance, "paged chunk \(range) vs blocked BOOL-mask calls")
                #expect(maxAbsDiff(gotPaged, gotContiguous) < tolerance)
                start += chunk
            }
        }

        /// Strict causality on the paged route: perturbing key and value at
        /// position `p + 1` must leave every query `<= p` bit-identical and
        /// move the queries that see it.
        @Test func pagedRouteNeverLeaksFutureKeys() throws {
            let kind = fullKind(headDim: 64, kvHeads: 2, queryHeads: 8, hasSinks: true)
            let scale: Float = 0.125
            let history = 160
            let chunk = 320
            let p = history + 200
            let prompt = Prompt(tokens: history + chunk, kind: kind, dtype: .float32, seed: 3)
            MLXRandom.seed(5)
            let sinks = MLXRandom.normal([kind.queryHeads])

            func run(perturbAt position: Int?) throws -> MLXArray {
                // Non-mutating: a slice assignment would rewrite the shared
                // `prompt` arrays in place (`_updateInternal`).
                var keys = prompt.keys
                var values = prompt.values
                if let position {
                    let positions = MLXArray(Int32(0) ..< Int32(prompt.keys.dim(2)))
                    let bump =
                        (positions .== Int32(position)).asType(.float32)
                        .reshaped([1, 1, -1, 1]) * 3
                    keys = prompt.keys + bump
                    values = prompt.values - bump
                    eval(keys, values)
                }
                let arm = try PrefillArm(
                    .paged, kind: kind, dtype: .float32, maxChunk: chunk, maxLength: history + chunk
                )
                _ = arm.step(
                    prompt.slice(prompt.queries, 0 ..< history), prompt.slice(keys, 0 ..< history),
                    prompt.slice(values, 0 ..< history), scale: scale, sinks: sinks)
                let out = arm.step(
                    prompt.slice(prompt.queries, history ..< (history + chunk)),
                    prompt.slice(keys, history ..< (history + chunk)),
                    prompt.slice(values, history ..< (history + chunk)), scale: scale, sinks: sinks)
                eval(out)
                return out
            }

            let baseline = try run(perturbAt: nil)
            let perturbed = try run(perturbAt: p + 1)
            let unaffected = 0 ..< (p + 1 - history)
            let affected = (p + 1 - history) ..< chunk
            #expect(
                bitIdentical(
                    baseline[0..., 0..., unaffected, 0...], perturbed[0..., 0..., unaffected, 0...]),
                "queries <= \(p) must not see key \(p + 1)")
            #expect(
                maxAbsDiff(
                    baseline[0..., 0..., affected, 0...], perturbed[0..., 0..., affected, 0...])
                    > 1e-3,
                "queries >= \(p + 1) must see key \(p + 1)")
        }

        /// Qwen full-attention geometry (head_dim 256, GQA 4, bf16): a 500-query
        /// chunk (always blocked) followed by a 1030-query chunk, which takes the
        /// NAX head-dim-split kernel where NAX exists and the composed blocks
        /// elsewhere. Both chunks must match the absolute reference on both
        /// backends, which pins MLX's bottom-right `.causal` offset under
        /// history for the D256 kernel.
        ///
        /// `kL` = 1530 is not a multiple of the kernel's 32-key tile and the
        /// contiguous buffer has slack past it, so on NAX MLX widens that view's
        /// `kL` to 1536 and reads the slack rows; causal masking must zero them.
        /// Paged attends a fresh concatenation, which MLX never widens.
        @Test func headDim256ChunkAfterHistoryMatchesReference() throws {
            let kind = fullKind(headDim: 256, kvHeads: 1, queryHeads: 4)
            let scale: Float = 1 / Float(256).squareRoot()
            let chunks = [500, 1030]
            let total = chunks.reduce(0, +)
            let prompt = Prompt(tokens: total, kind: kind, dtype: .bfloat16, seed: 13)
            let paged = try PrefillArm(
                .paged, kind: kind, dtype: .bfloat16, maxChunk: 1030, maxLength: total + 256)
            let contiguous = try PrefillArm(
                .contiguous, kind: kind, dtype: .bfloat16, maxChunk: 1030, maxLength: total + 256)
            var start = 0
            for chunk in chunks {
                let range = start ..< (start + chunk)
                let q = prompt.slice(prompt.queries, range)
                let gotPaged = paged.step(
                    q, prompt.slice(prompt.keys, range), prompt.slice(prompt.values, range),
                    scale: scale, sinks: nil)
                let gotContiguous = contiguous.step(
                    q, prompt.slice(prompt.keys, range), prompt.slice(prompt.values, range),
                    scale: scale, sinks: nil)
                let reference = absoluteReference(
                    queries: q, keys: prompt.slice(prompt.keys, 0 ..< range.upperBound),
                    values: prompt.slice(prompt.values, 0 ..< range.upperBound), scale: scale,
                    queryStart: start, sinks: nil)
                let dPaged = maxAbsDiff(gotPaged, reference)
                let dContiguous = maxAbsDiff(gotContiguous, reference)
                let dBackends = maxAbsDiff(gotPaged, gotContiguous)
                print(
                    "[fused-causal] d256 chunk \(range) nax=\(MiMoV26NAXGatherQMM.naxAvailable): "
                        + "|paged-ref| \(dPaged) |contig-ref| \(dContiguous) "
                        + "|paged-contig| \(dBackends)")
                #expect(dPaged < 3e-2, "paged chunk \(range)")
                #expect(dContiguous < 3e-2, "contiguous chunk \(range)")
                #expect(dBackends < 3e-2, "paged vs contiguous chunk \(range)")
                if MiMoV26NAXGatherQMM.naxAvailable && chunk >= 1024 {
                    // One kernel on both backends; the six slack columns only the
                    // contiguous view exposes are causally masked, so even a few
                    // unmasked zero keys (which still inflate the softmax
                    // denominator) would show up here.
                    #expect(dBackends == 0, "widened contiguous kL must not change the output")
                }
                start += chunk
            }
        }
    }
}
