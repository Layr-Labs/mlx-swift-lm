// CBv2RingReadFoldB1Tests.swift
//
// CPU tests for RING-READ-FOLD-B1. The fold's attention is a GPU metal_kernel
// (a one-row transcription of the B=8 `attendRing` two-pass, GPU-only). Its
// attention output is a NEAR-TIE — not bit-exact — vs the pre-fold path's
// `MLXFast.scaledDotProductAttention`: same keys, different block partition, one
// greedy flip in ~6,000 tokens (the orchestrator's token diff; HumanEval-gated,
// the D512-two-pass class). What IS bit-exact — and what these CPU tests cover —
// is the READ and the gating:
//
//  1. the READ the fold performs — the in-kernel modular temporal walk
//     `slot = (start + i) % window` from `decodeRingView.start` — is
//     byte-identical to the `temporalOrder` window the concat+SDPA path
//     attends, at every ring phase (before/after wrap), so the fold reads the
//     exact same K/V the path it replaces does; and
//  2. eligibility gates BEFORE the ring write (so the post-write attend can
//     never fail), at the real kernel shapes.
//
// The GPU flash-attention math over that walk is the B=8 kernel's, unchanged.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

private final class RingFoldBundleSentinel {}

/// Repo convention: nojit mlx-swift needs a metallib next to the test exe.
/// Run with `MLX_METALLIB_SOURCE=<a built mlx.metallib>`; no-op if unset.
private func ensureRingFoldMetallib() {
    let env = ProcessInfo.processInfo.environment
    guard let source = env["MLX_METALLIB_SOURCE"], !source.isEmpty,
        FileManager.default.fileExists(atPath: source),
        let exeDir = Bundle(for: RingFoldBundleSentinel.self)
            .executableURL?.deletingLastPathComponent()
    else { return }
    for name in ["mlx.metallib", "default.metallib"] {
        let dst = exeDir.appendingPathComponent(name)
        if !FileManager.default.fileExists(atPath: dst.path) {
            try? FileManager.default.copyItem(at: URL(fileURLWithPath: source), to: dst)
        }
    }
}

private func ringToken(step: Int, kvHeads: Int, headDim: Int) -> MLXArray {
    let base = Float(step) * 1000.0
    let heads = MLXArray((0 ..< kvHeads).map { Float($0) * 10.0 }).reshaped([1, kvHeads, 1, 1])
    let dims = MLXArray((0 ..< headDim).map { Float($0) * 0.01 }).reshaped([1, 1, 1, headDim])
    return broadcast(MLXArray(base) + heads + dims, to: [1, kvHeads, 1, headDim])
        .asType(.bfloat16)
}

private func equalExact(_ a: MLXArray, _ b: MLXArray) -> Bool {
    a.shape == b.shape && allClose(a, b, rtol: 0, atol: 0).item(Bool.self)
}

@Suite("CBv2RingReadFoldB1: modular walk == temporalOrder")
struct CBv2RingReadFoldReadTests {

    /// Decline the KVQ q4 mirror (its GPU-only pack kernel would fire on the
    /// CPU test device when a headDim-256 ring is written). This is the
    /// single-prompt answer the product path also takes at B=1 — the mirror has
    /// no reachable reader below a batch of 8.

    /// Across many decode steps (the ring start rotating through every phase,
    /// including the wrap), the temporal window rebuilt from
    /// `decodeRingView.start` by the fold's modular walk equals — bit-exact —
    /// the `temporalOrder` window the plain `update` returns.
    @Test func foldWalkMatchesTemporalOrder() {
      ensureRingFoldMetallib()
      Device.withDefaultDevice(.cpu) {
        let kvHeads = 2, headDim = 4, window = 16
        let reference = CBv2WindowedSequenceKV(window: window, kvHeads: kvHeads, headDim: headDim)
        let candidate = CBv2WindowedSequenceKV(window: window, kvHeads: kvHeads, headDim: headDim)

        // Fill both rings identically (fold only engages on a FULL ring).
        for step in 0 ..< window {
            let k = ringToken(step: step, kvHeads: kvHeads, headDim: headDim)
            let v = ringToken(step: step + 500, kvHeads: kvHeads, headDim: headDim)
            _ = reference.update(keys: k, values: v)
            _ = candidate.update(keys: k, values: v)
        }
        #expect(candidate.retainedCount == window)

        // Decode past several full wraps; compare the fold's read to the plain
        // temporal-order return at each phase.
        for step in window ..< (window + 3 * window + 5) {
            let k = ringToken(step: step, kvHeads: kvHeads, headDim: headDim)
            let v = ringToken(step: step + 500, kvHeads: kvHeads, headDim: headDim)

            let (refK, refV) = reference.update(keys: k, values: v)

            candidate.decodeRingWrite(keys: k, values: v)
            guard let view = candidate.decodeRingView else {
                Issue.record("decodeRingView nil at step \(step)")
                return
            }
            // The fold's in-kernel walk: slot = (start + i) % window, i in 0..<window.
            let idx = MLXArray((0 ..< window).map { Int32((view.start + $0) % window) })
            let foldK = take(view.keys, idx, axis: 2)
            let foldV = take(view.values, idx, axis: 2)

            #expect(equalExact(foldK, refK), "keys differ at step \(step) start \(view.start)")
            #expect(equalExact(foldV, refV), "values differ at step \(step) start \(view.start)")
        }
      }
    }
}

@Suite("CBv2RingReadFoldB1: eligibility")
struct CBv2RingReadFoldEligibilityTests {

    /// See the read suite: keep the KVQ mirror off so filling a headDim-256
    /// ring on the CPU device never reaches its GPU-only pack kernel.

    private static let window = 1024, kvHeads = 8, headDim = 256, queryHeads = 16

    /// The kernel-shape gate is pure (no cache, no mirror): matching shapes
    /// admit, any mismatch refuses.
    @Test func kernelShapeGate() {
      Device.withDefaultDevice(.cpu) {
        let q = broadcast(MLXArray(Float(0.5)), to: [1, Self.queryHeads, 1, Self.headDim])
            .asType(.bfloat16)
        let ring = broadcast(MLXArray(Float(1)), to: [1, Self.kvHeads, Self.window, Self.headDim])
            .asType(.bfloat16)
        #expect(
            CBv2RaggedTwoPassDecodeAttentionV1.ringB1Eligible(
                queries: q, keys: ring, values: ring, slidingWindowLength: Self.window)
                == CBv2RingReadFoldB1.enabled)
        // Wrong query heads, wrong ring length, and non-window slidingWindow all refuse.
        let qBad = broadcast(MLXArray(Float(0.5)), to: [1, 8, 1, Self.headDim]).asType(.bfloat16)
        #expect(
            CBv2RaggedTwoPassDecodeAttentionV1.ringB1Eligible(
                queries: qBad, keys: ring, values: ring, slidingWindowLength: Self.window)
                == false)
        let ringShort = broadcast(MLXArray(Float(1)), to: [1, Self.kvHeads, 512, Self.headDim])
            .asType(.bfloat16)
        #expect(
            CBv2RaggedTwoPassDecodeAttentionV1.ringB1Eligible(
                queries: q, keys: ringShort, values: ringShort, slidingWindowLength: 512)
                == false)
        #expect(
            CBv2RaggedTwoPassDecodeAttentionV1.ringB1Eligible(
                queries: q, keys: ring, values: ring, slidingWindowLength: 999)
                == false)
      }
    }

    private func slidingKind() -> CBv2LayerKind {
        CBv2LayerKind(
            attention: .slidingWindow(Self.window), headDim: Self.headDim,
            kvHeads: Self.kvHeads, queryHeads: Self.queryHeads)
    }

    private func fullRing() -> CBv2WindowedSequenceKV {
        let ring = CBv2WindowedSequenceKV(
            window: Self.window, kvHeads: Self.kvHeads, headDim: Self.headDim)
        // One window-aligned chunk fills the ring.
        let k = broadcast(MLXArray(Float(1)), to: [1, Self.kvHeads, Self.window, Self.headDim])
            .asType(.bfloat16)
        _ = ring.update(keys: k, values: k)
        return ring
    }

    private func decodeToken() -> MLXArray {
        broadcast(MLXArray(Float(2)), to: [1, Self.kvHeads, 1, Self.headDim]).asType(.bfloat16)
    }

    private func query() -> MLXArray {
        broadcast(MLXArray(Float(0.5)), to: [1, Self.queryHeads, 1, Self.headDim]).asType(.bfloat16)
    }

    @Test func engagesForFullPlainSlidingRing() {
      ensureRingFoldMetallib()
      Device.withDefaultDevice(.cpu) {
        let ring = fullRing()
        #expect(ring.retainedCount == Self.window)
        let ok = CBv2RingReadFoldB1.eligible(
            ring: ring, kind: slidingKind(), queries: query(),
            keys: decodeToken(), values: decodeToken(),
            scale: 1.0, sinks: nil, softcap: nil)
        // Only true when the fold's default switch is on (it is by default).
        #expect(ok == CBv2RingReadFoldB1.enabled)
      }
    }

    @Test func refusesNotYetFullRing() {
      Device.withDefaultDevice(.cpu) {
        let ring = CBv2WindowedSequenceKV(
            window: Self.window, kvHeads: Self.kvHeads, headDim: Self.headDim)
        for step in 0 ..< 10 {
            _ = ring.update(keys: decodeToken(), values: decodeToken())
            _ = step
        }
        #expect(ring.retainedCount < Self.window)
        #expect(
            CBv2RingReadFoldB1.eligible(
                ring: ring, kind: slidingKind(), queries: query(),
                keys: decodeToken(), values: decodeToken(),
                scale: 1.0, sinks: nil, softcap: nil) == false)
      }
    }

    @Test func refusesScaleSinksSoftcap() {
      Device.withDefaultDevice(.cpu) {
        let ring = fullRing()
        let sinks = broadcast(MLXArray(Float(0)), to: [Self.queryHeads]).asType(.bfloat16)
        #expect(
            CBv2RingReadFoldB1.eligible(
                ring: ring, kind: slidingKind(), queries: query(),
                keys: decodeToken(), values: decodeToken(),
                scale: 2.0, sinks: nil, softcap: nil) == false)
        #expect(
            CBv2RingReadFoldB1.eligible(
                ring: fullRing(), kind: slidingKind(), queries: query(),
                keys: decodeToken(), values: decodeToken(),
                scale: 1.0, sinks: sinks, softcap: nil) == false)
        #expect(
            CBv2RingReadFoldB1.eligible(
                ring: fullRing(), kind: slidingKind(), queries: query(),
                keys: decodeToken(), values: decodeToken(),
                scale: 1.0, sinks: nil, softcap: 30.0) == false)
      }
    }

    @Test func refusesFullAttentionKind() {
      Device.withDefaultDevice(.cpu) {
        let ring = fullRing()
        let fullKind = CBv2LayerKind(
            attention: .full, headDim: Self.headDim,
            kvHeads: Self.kvHeads, queryHeads: Self.queryHeads)
        #expect(
            CBv2RingReadFoldB1.eligible(
                ring: ring, kind: fullKind, queries: query(),
                keys: decodeToken(), values: decodeToken(),
                scale: 1.0, sinks: nil, softcap: nil) == false)
      }
    }
}
