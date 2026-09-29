// Copyright © 2026 Eigen Labs.
// Source-only regression: preserve managed q128 / K-prefix visibility.
// Run only under an explicitly owned M5/NAX lane. No execution is claimed.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class MiMoV26NAXAttentionVisibilityTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["DARKBLOOM_TEST_MIMO_NAX_ATTENTION"] == "1",
            MiMoV26NAXAttention.requested
        else {
            throw XCTSkip("Requires owned lane and process-start NAX attention/test opt-ins")
        }
        guard MiMoV26NAXGatherQMM.gpuStream(.default), MiMoV26NAXGatherQMM.naxAvailable else {
            throw XCTSkip("Requires actual NAX-capable GPU stream/build")
        }
        guard CBv2AttentionV1.queryBlockSize == 128 else {
            throw XCTSkip("This visibility witness requires the unchanged q128 baseline")
        }
    }

    private let scale: Float = 1 / Float(192).squareRoot()

    private func withCaches(
        dtype: DType, length: Int,
        _ body: (CBv2LayerCache, CBv2LayerCache, any CBv2SequenceKV, any CBv2SequenceKV) throws ->
            Void
    ) throws {
        let kind = CBv2LayerKind(
            attention: .full, headDim: 192, valueHeadDim: 128,
            kvHeads: 4, queryHeads: 64)
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 128 << 20, kvDType: dtype))
        let baselineState = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: length)
        let candidateState = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: length)
        defer {
            backend.release(baselineState)
            backend.release(candidateState)
        }
        let baselineRow = try XCTUnwrap(baselineState[0])
        let candidateRow = try XCTUnwrap(candidateState[0])
        let baseline = CBv2LayerCache(layerIndex: 0, kind: kind)
        let candidate = CBv2LayerCache(layerIndex: 0, kind: kind, mimoV26NAXAttention: true)
        baseline.setRows([baselineRow])
        candidate.setRows([candidateRow])
        try body(baseline, candidate, baselineRow, candidateRow)
    }

    /// Independent expression of the pre-port full-cache q128/K-prefix path.
    /// This helper never asks the new NAX dispatcher to select its oracle.
    private func slicedReference(q: MLXArray, k: MLXArray, v: MLXArray) -> MLXArray {
        let length = q.dim(2)
        let history = k.dim(2) - length
        var outputs: [MLXArray] = []
        for start in stride(from: 0, to: length, by: 128) {
            let count = min(128, length - start)
            let end = history + start + count
            outputs.append(
                MLXFast.scaledDotProductAttention(
                    queries: q[0..., 0..., start ..< (start + count), 0...],
                    keys: k[0..., 0..., ..<end, 0...],
                    values: v[0..., 0..., ..<end, 0...], scale: scale,
                    mask: count == 1 ? .none : .causal, sinks: nil))
        }
        return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 2)
    }

    private func exact(_ actual: MLXArray, _ expected: MLXArray, label: String) {
        eval(actual, expected)
        XCTAssertEqual(actual.dtype, expected.dtype)
        XCTAssertEqual(actual.shape, expected.shape)
        let a = actual.view(dtype: .uint16).asArray(UInt16.self)
        let b = expected.view(dtype: .uint16).asArray(UInt16.self)
        var maxULP: UInt32 = 0
        for (x, y) in zip(a, b) {
            let xi = x & 0x8000 == 0 ? UInt32(x) + 0x8000 : UInt32(~x)
            let yi = y & 0x8000 == 0 ? UInt32(y) + 0x8000 : UInt32(~y)
            maxULP = max(maxULP, xi > yi ? xi - yi : yi - xi)
        }
        print("MiMoNAX visibility \(label) maxStorageULP=\(maxULP)")
        XCTAssertEqual(maxULP, 0, label)
        XCTAssertEqual(a, b, label)
    }

    private func exactState(_ a: any CBv2SequenceKV, _ b: any CBv2SequenceKV, length: Int) {
        XCTAssertEqual(a.absoluteOffset, length)
        XCTAssertEqual(b.absoluteOffset, length)
        XCTAssertEqual(a.retainedCount, length)
        XCTAssertEqual(b.retainedCount, length)
        let sa = a.snapshot()
        let sb = b.snapshot()
        XCTAssertEqual(sa.offset, sb.offset)
        exact(sa.keys, sb.keys, label: "unchanged K and one update")
        exact(sa.values, sb.values, label: "unchanged V and one update")
    }

    func testFiniteFP16OverflowCannotReadValuesBeyondFirstBlockPrefix() throws {
        let length = 256
        let q = (MLXArray.ones([1, 64, length, 192], dtype: .float16) * 2048).asType(.float16)
        let k = (MLXArray.ones([1, 4, length, 192], dtype: .float16) * -2048).asType(.float16)
        // Finite native inputs deliberately overflow the materialized native QK.
        let score = matmul(
            q[0 ..< 1, 0 ..< 1, 0 ..< 1, 0...] * MLXArray(scale).asType(.float16),
            k[0 ..< 1, 0 ..< 1, 0 ..< 1, 0...].swappedAxes(-1, -2))
        let nativeScore = score.asType(.float32).item(Float.self)
        XCTAssertTrue(nativeScore.isInfinite && nativeScore < 0)
        for futureValue: Float in [1, -2] {
            let v = concatenated(
                [
                    MLXArray.zeros([1, 4, 128, 128], dtype: .float16),
                    MLXArray.ones([1, 4, 128, 128], dtype: .float16) * futureValue,
                ], axis: 2
            ).asType(.float16)
            try withCaches(dtype: .float16, length: length) { baseline, candidate, a, b in
                let before = MiMoV26NAXAttention.encodedCalls()
                let reference = baseline.updateAndAttend(
                    queries: q, keys: k, values: v,
                    scale: scale, sinks: nil)
                XCTAssertEqual(MiMoV26NAXAttention.encodedCalls(), before)
                let actual = candidate.updateAndAttend(
                    queries: q, keys: k, values: v,
                    scale: scale, sinks: nil)
                XCTAssertEqual(
                    MiMoV26NAXAttention.encodedCalls() - before, 2,
                    "Both original q128 blocks must engage, not one full call")
                let sliced = slicedReference(q: q, k: k, v: v)
                let first = actual[0..., 0..., 0 ..< 1, 0...]
                let baselineFirst = reference[0..., 0..., 0 ..< 1, 0...]
                let slicedFirst = sliced[0..., 0..., 0 ..< 1, 0...]
                exact(first, baselineFirst, label: "overflow actual cache first query")
                exact(first, slicedFirst, label: "overflow independent sliced oracle")
                XCTAssertTrue(first.asType(.float32).asArray(Float.self).allSatisfy { $0 == 0 })

                // Negative control: the ordinary unblocked contract legitimately
                // includes those finite-min-masked future columns. It differs.
                let full = try XCTUnwrap(
                    MiMoV26NAXAttention.tryAttention(
                        queries: q, keys: k, values: v, scale: scale, mask: .causal, sinks: nil))
                let fullFirst = full[0..., 0..., 0 ..< 1, 0...]
                let magnitude = abs(fullFirst.asType(.float32)).max().item(Float.self)
                XCTAssertGreaterThan(
                    magnitude, 0.25, "Fixture must discriminate the removed shortcut")
                // Last fully-visible rows can legitimately be all -inf/NaN.
                // Deliberately inspect only the discriminating first row.
                exactState(a, b, length: length)
            }
        }
    }

    func testHealthyUniformScoresKeepOriginalPrefixSlicesAndShortCalls() throws {
        for dtype in [DType.bfloat16, .float16] {
            for length in [64, 128, 129, 256, 300] {
                let q = MLXArray.zeros([1, 64, length, 192], dtype: dtype)
                let k = MLXArray.zeros([1, 4, length, 192], dtype: dtype)
                let values = MLXArray((0 ..< length).map { Float($0 % 2) }, [1, 1, length, 1])
                    .asType(dtype)
                let v = broadcast(values, to: [1, 4, length, 128])
                let expectedEncodings = (0 ..< length).filter { $0 % 128 == 0 }
                    .filter { min(128, length - $0) > 8 }.count
                try withCaches(dtype: dtype, length: length) { baseline, candidate, a, b in
                    let before = MiMoV26NAXAttention.encodedCalls()
                    let reference = baseline.updateAndAttend(
                        queries: q, keys: k, values: v,
                        scale: scale, sinks: nil)
                    XCTAssertEqual(MiMoV26NAXAttention.encodedCalls(), before)
                    let actual = candidate.updateAndAttend(
                        queries: q, keys: k, values: v,
                        scale: scale, sinks: nil)
                    XCTAssertEqual(MiMoV26NAXAttention.encodedCalls() - before, expectedEncodings)
                    exact(actual, reference, label: "healthy cache \(dtype) L\(length)")
                    exact(
                        actual, slicedReference(q: q, k: k, v: v),
                        label: "healthy sliced oracle \(dtype) L\(length)")
                    XCTAssertTrue(
                        actual.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
                    exactState(a, b, length: length)
                }
            }
        }
    }
}
