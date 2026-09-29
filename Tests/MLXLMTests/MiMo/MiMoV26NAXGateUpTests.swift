// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0
// Joint MXFP4 gather source candidate: tests prepared, not executed.

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLMCommon

final class MiMoV26NAXGateUpTests: XCTestCase {
    private func requireGPU() throws {
        guard ProcessInfo.processInfo.environment["DARKBLOOM_TEST_MIMO_NAX_GATE_UP"] == "1"
        else { throw XCTSkip("Requires explicitly owned NAX GPU lane") }
        guard MiMoV26NAXGatherQMM.gpuStream(.default), MiMoV26NAXGatherQMM.naxAvailable
        else { throw XCTSkip("Current stream/build/device cannot execute NAX") }
    }

    private func weights(
        _ experts: Int, _ output: Int, _ input: Int,
        dtype: DType, seed: UInt64
    ) -> (MLXArray, MLXArray) {
        let w = (MLXRandom.normal([experts, output, input], key: MLXRandom.key(seed)) * 0.05)
            .asType(dtype)
        let (packed, scales, biases) = MLX.quantized(w, groupSize: 32, bits: 4, mode: .mxfp4)
        XCTAssertNil(biases)
        return (packed, scales)
    }

    private func stock(
        _ x: MLXArray, _ indices: MLXArray,
        _ weights: (MLXArray, MLXArray)
    ) -> MLXArray {
        MLX.gatherQuantizedMM(
            x, weights.0, scales: weights.1, biases: nil,
            rhsIndices: indices, transpose: true, groupSize: 32, bits: 4,
            mode: .mxfp4, sortedIndices: true)
    }

    private func exact(_ actual: MLXArray, _ reference: MLXArray, _ label: String) {
        eval(actual, reference)
        XCTAssertEqual(actual.dtype, reference.dtype)
        XCTAssertEqual(actual.shape, reference.shape)
        let a = actual.asType(.float32).asArray(Float.self)
        let b = reference.asType(.float32).asArray(Float.self)
        var maxULP: UInt32 = 0
        for (x, y) in zip(a, b) {
            XCTAssertTrue(x.isFinite && y.isFinite)
            let xb =
                actual.dtype == .bfloat16
                ? UInt16(truncatingIfNeeded: x.bitPattern >> 16) : Float16(x).bitPattern
            let yb =
                reference.dtype == .bfloat16
                ? UInt16(truncatingIfNeeded: y.bitPattern >> 16) : Float16(y).bitPattern
            let xi = xb & 0x8000 == 0 ? UInt32(xb) + 0x8000 : UInt32(~xb)
            let yi = yb & 0x8000 == 0 ? UInt32(yb) + 0x8000 : UInt32(~yb)
            maxULP = max(maxULP, xi > yi ? xi - yi : yi - xi)
        }
        print("MiMoNAXGateUp \(label) dtype=\(actual.dtype) maxULP=\(maxULP)")
        XCTAssertEqual(maxULP, 0)
        XCTAssertEqual(actual.asData().data, reference.asData().data)
    }

    func testJointOutputsMatchBothIndependentOracles() throws {
        try requireGPU()
        let routes = [
            [70, 0, 5, 33, 64, 17, 1, 130],
            [300, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 300],
        ]
        for dtype in [DType.bfloat16, .float16] {
            let gate = weights(8, 128, 256, dtype: dtype, seed: 100)
            let up = weights(8, 128, 256, dtype: dtype, seed: 101)
            for counts in routes {
                let ids = counts.enumerated().flatMap {
                    Array(repeating: UInt32($0.offset), count: $0.element)
                }
                let indices = MLXArray(ids)
                let x = MLXRandom.normal([ids.count, 1, 256], key: MLXRandom.key(102)).asType(dtype)
                let plan = MiMoV26NAXGatherQMM.Plan(
                    rows: ids.count, experts: 8, input: 256, output: 128)
                for bm in [64, 128] {
                    let joint = MiMoV26NAXGateUp.launch(
                        x: x, indices: indices,
                        gateWeight: gate.0, gateScales: gate.1,
                        upWeight: up.0, upScales: up.1, plan: plan, tileRows: bm)
                    exact(joint.gate, stock(x, indices, gate), "stock gate bm=\(bm)")
                    exact(joint.up, stock(x, indices, up), "stock up bm=\(bm)")
                    let separateGate = MiMoV26NAXGatherQMM.launch(
                        x: x, indices: indices,
                        weight: gate.0, scales: gate.1, plan: plan, tileRows: bm,
                        doubleBuffer: false)
                    let separateUp = MiMoV26NAXGatherQMM.launch(
                        x: x, indices: indices,
                        weight: up.0, scales: up.1, plan: plan, tileRows: bm, doubleBuffer: false)
                    exact(joint.gate, separateGate, "single NAX gate bm=\(bm)")
                    exact(joint.up, separateUp, "single NAX up bm=\(bm)")
                }
            }
        }
    }

    func testOversizedSortedRowsAgainstSafeStockSlices() throws {
        try requireGPU()
        let counts = Array(repeating: 2304, count: 15) + [4000]
        let ids = counts.enumerated().flatMap {
            Array(repeating: UInt32($0.offset), count: $0.element)
        }
        let indices = MLXArray(ids)
        let x = MLXRandom.normal([ids.count, 1, 128], key: MLXRandom.key(103)).asType(.bfloat16)
        let gate = weights(16, 64, 128, dtype: .bfloat16, seed: 104)
        let up = weights(16, 64, 128, dtype: .bfloat16, seed: 105)
        let half = ids.count / 2
        func reference(_ w: (MLXArray, MLXArray)) -> MLXArray {
            concatenated([
                stock(x[..<half], indices[..<half], w),
                stock(x[half...], indices[half...], w),
            ])
        }
        let plan = MiMoV26NAXGatherQMM.Plan(rows: ids.count, experts: 16, input: 128, output: 64)
        for bm in [64, 128] {
            let result = MiMoV26NAXGateUp.launch(
                x: x, indices: indices,
                gateWeight: gate.0, gateScales: gate.1,
                upWeight: up.0, upScales: up.1, plan: plan, tileRows: bm)
            exact(result.gate, reference(gate), "oversized gate bm=\(bm)")
            exact(result.up, reference(up), "oversized up bm=\(bm)")
        }
    }

    func testSingleGatherSortAndExistingGLUCompositionArePreserved() throws {
        try requireGPU()
        let tokens = 300
        let topK = 8
        let experts = 16
        let input = 128
        let hidden = 64
        let ids = (0 ..< tokens).flatMap { token in
            (0 ..< topK).map { UInt32((token * 17 + $0 * 3) % experts) }
        }
        let indices = MLXArray(ids, [1, tokens, topK])
        let x = MLXRandom.normal([1, tokens, input], key: MLXRandom.key(106)).asType(.bfloat16)
        let (sortedX, sortedIDs, inverse) = gatherSort(
            x: expandedDimensions(x, axes: [-2, -3]), indices: indices)
        XCTAssertEqual(sortedX.shape, [tokens * topK, 1, input])
        let gate = weights(experts, hidden, input, dtype: .bfloat16, seed: 107)
        let up = weights(experts, hidden, input, dtype: .bfloat16, seed: 108)
        let down = weights(experts, input, hidden, dtype: .bfloat16, seed: 109)
        let plan = MiMoV26NAXGatherQMM.Plan(
            rows: sortedIDs.size, experts: experts, input: input, output: hidden)
        let joint = MiMoV26NAXGateUp.launch(
            x: sortedX, indices: sortedIDs,
            gateWeight: gate.0, gateScales: gate.1, upWeight: up.0, upScales: up.1, plan: plan)
        let refGate = stock(sortedX, sortedIDs, gate)
        let refUp = stock(sortedX, sortedIDs, up)
        let activated = compiledSiluProduct(joint.gate, joint.up)
        let referenceActivated = compiledSiluProduct(refGate, refUp)
        exact(activated, referenceActivated, "unchanged compiledSiluProduct")
        let actual = scatterUnsort(
            x: stock(activated, sortedIDs, down),
            invOrder: inverse, shape: indices.shape)
        let reference = scatterUnsort(
            x: stock(referenceActivated, sortedIDs, down),
            invOrder: inverse, shape: indices.shape)
        exact(actual, reference, "unchanged down and inverse mapping")
    }

    func testNonProductionGeometryDeclinesWithoutEncoding() {
        let gate = QuantizedSwitchLinear(
            SwitchLinear(inputDims: 128, outputDims: 64, numExperts: 8),
            groupSize: 32, bits: 4, mode: .mxfp4)
        let up = QuantizedSwitchLinear(
            SwitchLinear(inputDims: 128, outputDims: 64, numExperts: 8),
            groupSize: 32, bits: 4, mode: .mxfp4)
        let before = MiMoV26NAXGateUp.encodedCalls()
        XCTAssertNil(
            MiMoV26NAXGateUp.tryProjection(
                x: MLXArray.zeros([64, 1, 128], dtype: .bfloat16),
                indices: MLXArray.zeros([64], dtype: .uint32), gate: gate, up: up, sorted: true))
        XCTAssertEqual(MiMoV26NAXGateUp.encodedCalls(), before)
    }

    func testSwiGLUEpiloguePreservesEveryRoundedStage() throws {
        try requireGPU()
        for dtype in [DType.bfloat16, .float16] {
            let gate = weights(8, 128, 256, dtype: dtype, seed: 201)
            let up = weights(8, 128, 256, dtype: dtype, seed: 202)
            for counts in [[33, 0, 1, 130, 17, 5, 64, 3], [0, 0, 0, 0, 0, 0, 0, 300]] {
                let ids = counts.enumerated().flatMap {
                    Array(repeating: UInt32($0.offset), count: $0.element)
                }
                let indices = MLXArray(ids)
                let plan = MiMoV26NAXGatherQMM.Plan(
                    rows: ids.count, experts: 8, input: 256, output: 128)
                for scale: Float in [0.1, 1, 8, 16] {
                    let x =
                        (MLXRandom.normal([ids.count, 1, 256], key: MLXRandom.key(203)) * scale)
                        .asType(dtype)
                    for bm in [64, 128] {
                        let fused = MiMoV26NAXGateUp.launchActivation(
                            x: x, indices: indices,
                            gateWeight: gate.0, gateScales: gate.1,
                            upWeight: up.0, upScales: up.1, plan: plan, tileRows: bm)
                        let separate = MiMoV26NAXGateUp.launch(
                            x: x, indices: indices,
                            gateWeight: gate.0, gateScales: gate.1,
                            upWeight: up.0, upScales: up.1, plan: plan, tileRows: bm)
                        exact(
                            fused, compiledSiluProduct(separate.gate, separate.up),
                            "epilogue separate NAX bm=\(bm) scale=\(scale)")
                        exact(
                            fused,
                            compiledSiluProduct(stock(x, indices, gate), stock(x, indices, up)),
                            "epilogue independent stock bm=\(bm) scale=\(scale)")
                    }
                }
            }
        }
    }

    func testSwiGLUEpilogueDoesNotChangeNonfiniteStorageBits() throws {
        try requireGPU()
        let rows = 65
        let indices = MLXArray.zeros([rows], dtype: .uint32)
        let plan = MiMoV26NAXGatherQMM.Plan(rows: rows, experts: 1, input: 128, output: 64)
        for dtype in [DType.bfloat16, .float16] {
            let gate = weights(1, 64, 128, dtype: dtype, seed: 204)
            let up = weights(1, 64, 128, dtype: dtype, seed: 205)
            for value: Float in [.infinity, -.infinity, .nan, 0, -0.0] {
                let x = MLXArray.full([rows, 1, 128], values: MLXArray(value), dtype: dtype)
                let result = MiMoV26NAXGateUp.launchActivation(
                    x: x, indices: indices,
                    gateWeight: gate.0, gateScales: gate.1, upWeight: up.0, upScales: up.1,
                    plan: plan)
                let reference = compiledSiluProduct(stock(x, indices, gate), stock(x, indices, up))
                eval(result, reference)
                XCTAssertEqual(
                    result.asData(access: .copy).data, reference.asData(access: .copy).data)
            }
        }
    }

    func testMappedSwiGLUMatchesMaterializedRowsBitForBit() throws {
        try requireGPU()
        let tokens = 73
        let input = 128
        let output = 64
        let experts = 8
        let counts = [65, 0, 3, 131, 1, 17, 0, 83]
        let ids = counts.enumerated().flatMap {
            Array(repeating: UInt32($0.offset), count: $0.element)
        }
        let map = MLXArray((0 ..< ids.count).map { UInt32(($0 * 7 + 3) % tokens) })
        let indices = MLXArray(ids)
        let plan = MiMoV26NAXGatherQMM.Plan(
            rows: ids.count, experts: experts, input: input, output: output)
        for dtype in [DType.bfloat16, .float16] {
            let x = MLXRandom.normal([tokens, 1, input], key: MLXRandom.key(210)).asType(dtype)
            let copied = x[map]
            let gate = weights(experts, output, input, dtype: dtype, seed: 211)
            let up = weights(experts, output, input, dtype: dtype, seed: 212)
            for bm in [64, 128] {
                let mapped = MiMoV26NAXGateUp.launchActivation(
                    x: x, indices: indices,
                    gateWeight: gate.0, gateScales: gate.1, upWeight: up.0, upScales: up.1,
                    plan: plan, tileRows: bm, rowMap: map)
                let materialized = MiMoV26NAXGateUp.launchActivation(
                    x: copied, indices: indices,
                    gateWeight: gate.0, gateScales: gate.1, upWeight: up.0, upScales: up.1,
                    plan: plan, tileRows: bm)
                exact(mapped, materialized, "mapped/repeated rows bm=\(bm)")
                exact(
                    mapped,
                    compiledSiluProduct(stock(copied, indices, gate), stock(copied, indices, up)),
                    "mapped/stock bm=\(bm)")
            }
        }
    }
}
