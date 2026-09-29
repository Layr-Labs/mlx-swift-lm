// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0
// Source-only candidate tests. Numerical diagnostics alone are not qualification.

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLMCommon

final class MiMoV26NAXAttentionTests: XCTestCase {
    private func requireGPU() throws {
        guard ProcessInfo.processInfo.environment["DARKBLOOM_TEST_MIMO_NAX_ATTENTION"] == "1"
        else { throw XCTSkip("Requires explicitly owned NAX GPU lane") }
        guard MiMoV26NAXGatherQMM.gpuStream(.default), MiMoV26NAXGatherQMM.naxAvailable
        else { throw XCTSkip("Current stream/build/device cannot execute NAX") }
    }

    private func fixture(_ length: Int, _ keys: Int, dtype: DType, heads: Int = 8, kvHeads: Int = 2)
        -> (q: MLXArray, k: MLXArray, v: MLXArray)
    {
        (
            q: (MLXRandom.normal([1, heads, length, 192], key: MLXRandom.key(192)) * 0.5).asType(
                dtype),
            k: (MLXRandom.normal([1, kvHeads, keys, 192], key: MLXRandom.key(193)) * 0.5).asType(
                dtype),
            v: (MLXRandom.normal([1, kvHeads, keys, 128], key: MLXRandom.key(194)) * 0.5).asType(
                dtype)
        )
    }

    private func run(
        _ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, sinks: MLXArray? = nil
    )
        throws -> MLXArray
    {
        let scale = 1 / Float(192).squareRoot()
        let plan = try XCTUnwrap(
            MiMoV26NAXAttention.makePlan(
                queries: q, keys: k, values: v, scale: scale, mask: mask,
                sinks: sinks, production: false))
        return MiMoV26NAXAttention.launch(
            queries: q, keys: k, values: v, scale: scale, plan: plan)
    }

    private func report(_ a: MLXArray, _ b: MLXArray, label: String) {
        eval(a, b)
        XCTAssertEqual(a.dtype, b.dtype)
        XCTAssertEqual(a.shape, b.shape)
        let aa = a.asType(.float32).asArray(Float.self)
        let bb = b.asType(.float32).asArray(Float.self)
        var maxULP: UInt32 = 0
        var maxAbsolute: Float = 0
        for (x, y) in zip(aa, bb) {
            XCTAssertTrue(x.isFinite && y.isFinite)
            let xb =
                a.dtype == .bfloat16
                ? UInt16(truncatingIfNeeded: x.bitPattern >> 16) : Float16(x).bitPattern
            let yb =
                b.dtype == .bfloat16
                ? UInt16(truncatingIfNeeded: y.bitPattern >> 16) : Float16(y).bitPattern
            let xi = (xb & 0x8000) == 0 ? UInt32(xb) + 0x8000 : UInt32(~xb)
            let yi = (yb & 0x8000) == 0 ? UInt32(yb) + 0x8000 : UInt32(~yb)
            maxULP = max(maxULP, xi > yi ? xi - yi : yi - xi)
            maxAbsolute = max(maxAbsolute, abs(x - y))
        }
        print("MiMoNAXAttention \(label) dtype=\(a.dtype) maxULP=\(maxULP) maxAbs=\(maxAbsolute)")
        // Full-model greedy/state and arithmetic attribution gates are separate.
        // A finite or tolerance-close result is deliberately not called lossless.
    }

    func testShapeDtypeMaskAndOwnerGates() {
        let f = fixture(64, 128, dtype: .bfloat16)
        func plan(
            _ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
            _ mask: MLXFast.ScaledDotProductAttentionMaskMode = .causal,
            _ sinks: MLXArray? = nil, production: Bool = false
        ) -> MiMoV26NAXAttention.Plan? {
            MiMoV26NAXAttention.makePlan(
                queries: q, keys: k, values: v,
                scale: 0.07, mask: mask, sinks: sinks, production: production)
        }
        XCTAssertNotNil(plan(f.q, f.k, f.v))
        XCTAssertNil(plan(f.q, f.k, f.v, production: true))
        XCTAssertNil(plan(f.q.asType(.float32), f.k.asType(.float32), f.v.asType(.float32)))
        XCTAssertNil(plan(f.q[.ellipsis, ..<8, 0...], f.k, f.v))
        XCTAssertNil(plan(f.q, f.k, f.v, .array(MLXArray.zeros([64, 128]))))
        XCTAssertNil(plan(f.q, f.k, f.v, .array(MLXArray.zeros([3, 64, 128], dtype: .bool))))
        XCTAssertNil(plan(f.q, f.k, f.v, .causal, MLXArray.zeros([8], dtype: .float32)))
    }

    func testUniformProbabilityRoundingAndMaskedTailExactly() throws {
        try requireGPU()
        for dtype in [DType.bfloat16, .float16] {
            let q = MLXArray.zeros([1, 8, 64, 192], dtype: dtype)
            for (keyCount, hasSink, fullyMasked) in [
                (128, false, false), (127, true, false),
                (127, false, true),
            ] {
                let k = MLXArray.zeros([1, 2, keyCount, 192], dtype: dtype)
                let v = MLXArray.ones([1, 2, keyCount, 128], dtype: dtype)
                let sinks = hasSink ? MLXArray.zeros([8], dtype: dtype) : nil
                let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                    fullyMasked
                    ? .array(MLXArray.zeros([64, keyCount], dtype: .bool)) : .none
                let actual = try run(q, k, v, mask: mask, sinks: sinks)
                let expected =
                    MLXArray.ones(actual.shape, dtype: dtype)
                    * (hasSink ? Float(127.0 / 128.0) : Float(1))
                report(
                    actual, expected,
                    label: "uniform k=\(keyCount) sink=\(hasSink) masked=\(fullyMasked)")
                XCTAssertEqual(actual.asData().data, expected.asData().data)
            }
        }
    }

    func testLazyInnerHeadStrideAndSequenceStrideAreBitExact() throws {
        try requireGPU()
        for dtype in [DType.bfloat16, .float16] {
            // No eval occurs before the kernel receives these lazy views.
            let qWide = MLXRandom.normal([1, 200, 8, 384], key: MLXRandom.key(200)).asType(dtype)
            let kWide = MLXRandom.normal([1, 2, 900, 384], key: MLXRandom.key(201)).asType(dtype)
            let vWide = MLXRandom.normal([1, 2, 900, 256], key: MLXRandom.key(202)).asType(dtype)
            let q = qWide.transposed(0, 2, 1, 3)[.ellipsis, .stride(by: 2)]
            let k = kWide[0..., 0..., .stride(by: 2), .stride(by: 2)]
            let v = vWide[0..., 0..., .stride(by: 2), .stride(by: 2)]
            let actual = try run(q, k, v, mask: .causal)
            let expected = try run(contiguous(q), contiguous(k), contiguous(v), mask: .causal)
            report(actual, expected, label: "lazy head/sequence strides")
            XCTAssertEqual(actual.asData().data, expected.asData().data)
            let rows = actual.transposed(0, 2, 1, 3).reshaped([1, 200, 8 * 128])
            let referenceRows = expected.transposed(0, 2, 1, 3).reshaped(rows.shape)
            XCTAssertEqual(rows.asData().data, referenceRows.asData().data)
        }
    }

    func testBaselineNumericalDiagnosticsForTailsSinksAndWindow() throws {
        try requireGPU()
        for dtype in [DType.bfloat16, .float16] {
            for (length, keys) in [(16, 512), (63, 127), (64, 128), (255, 4095), (300, 700)] {
                let f = fixture(length, keys, dtype: dtype)
                let sink = MLXRandom.normal([8], key: MLXRandom.key(301)).asType(dtype)
                for window in [false, true] {
                    let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                        window
                        ? .array(
                            createCausalMask(n: length, offset: keys - length, windowSize: 128))
                        : .causal
                    let actual = try run(f.q, f.k, f.v, mask: mask, sinks: sink)
                    let baseline = MLXFast.scaledDotProductAttention(
                        queries: f.q, keys: f.k, values: f.v,
                        scale: 1 / Float(192).squareRoot(), mask: mask, sinks: sink)
                    report(
                        actual, baseline, label: "baseline q=\(length) k=\(keys) window=\(window)")
                }
            }
        }
    }

    func testManagedFullAndWindowKeepExactKVStateAndExcludeSerialMTP() throws {
        try requireGPU()
        guard MiMoV26NAXAttention.requested else {
            throw XCTSkip(
                "Start process with DARKBLOOM_MIMO_V26_NAX_ATTENTION=1 for managed dispatch")
        }
        for window in [false, true] {
            let kind = CBv2LayerKind(
                attention: window ? .slidingWindow(128) : .full, hasSinks: window,
                headDim: 192, valueHeadDim: 128, kvHeads: 4, queryHeads: 64)
            let backend = CBv2ContiguousKVBackend(
                config: .init(bytesCapacity: 128 << 20, kvDType: .bfloat16))
            let baselineState = try backend.makeSequenceState(
                layerKinds: [kind], promptLength: 0, maxLength: 1024)
            let candidateState = try backend.makeSequenceState(
                layerKinds: [kind], promptLength: 0, maxLength: 1024)
            defer {
                backend.release(baselineState)
                backend.release(candidateState)
            }
            let baseline = CBv2LayerCache(layerIndex: 0, kind: kind)
            let candidate = CBv2LayerCache(layerIndex: 0, kind: kind, mimoV26NAXAttention: true)
            baseline.setRows([try XCTUnwrap(baselineState[0])])
            candidate.setRows([try XCTUnwrap(candidateState[0])])
            let f = fixture(300, 300, dtype: .bfloat16, heads: 64, kvHeads: 4)
            let sinks = window ? MLXArray.zeros([64], dtype: .bfloat16) : nil
            let before = MiMoV26NAXAttention.encodedCalls()
            let a = baseline.updateAndAttend(
                queries: f.q, keys: f.k, values: f.v, scale: 0.07, sinks: sinks)
            XCTAssertEqual(MiMoV26NAXAttention.encodedCalls(), before)
            let b = candidate.updateAndAttend(
                queries: f.q, keys: f.k, values: f.v, scale: 0.07, sinks: sinks)
            XCTAssertGreaterThan(MiMoV26NAXAttention.encodedCalls(), before)
            report(b, a, label: "managed window=\(window)")
            let sa = baselineState[0]!.snapshot()
            let sb = candidateState[0]!.snapshot()
            eval(sa.keys, sa.values, sb.keys, sb.values)
            XCTAssertEqual(sa.offset, sb.offset)
            XCTAssertEqual(sa.keys.asData().data, sb.keys.asData().data)
            XCTAssertEqual(sa.values.asData().data, sb.values.asData().data)
            candidate.mtpSerializesRectangularAttention = true
            let verify = fixture(16, 16, dtype: .bfloat16, heads: 64, kvHeads: 4)
            let serialBefore = MiMoV26NAXAttention.encodedCalls()
            let serial = candidate.updateAndAttend(
                queries: verify.q, keys: verify.k, values: verify.v, scale: 0.07, sinks: sinks)
            eval(serial)
            XCTAssertEqual(MiMoV26NAXAttention.encodedCalls(), serialBefore)
        }
    }
}
