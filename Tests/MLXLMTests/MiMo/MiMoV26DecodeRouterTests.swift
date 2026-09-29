// Copyright © 2026 Eigen Labs.
// Source-only native test candidate; requires the exclusive native lane.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM

final class MiMoV26DecodeRouterEligibilityTests: XCTestCase {
    func testOnlyAlignedShortGPURouterShapesAreEligible() {
        func supported(
            _ shape: [Int], _ weight: [Int], _ type: DType = .bfloat16,
            _ operand: DType = .bfloat16, _ device: DeviceType = .gpu
        ) -> Bool {
            MiMoV26DecodeRouter.supports(
                shape: shape, weightShape: weight,
                inputDType: type, weightDType: type, operandDType: operand, device: device)
        }
        for rows in 1 ... 7 { XCTAssertTrue(supported([1, rows, 4096], [256, 4096])) }
        XCTAssertTrue(supported([1, 2, 1024], [64, 1024], .float32, .float32))
        XCTAssertFalse(supported([2, 1, 4096], [256, 4096]))
        XCTAssertFalse(supported([1, 8, 4096], [256, 4096]))
        XCTAssertFalse(supported([1, 1, 4000], [256, 4000]))
        XCTAssertFalse(supported([1, 1, 1024], [256, 1024]))
        XCTAssertFalse(supported([1, 1, 4096], [255, 4096]))
        XCTAssertFalse(supported([1, 1, 4096], [256, 2048]))
        XCTAssertFalse(supported([1, 1, 4096], [256, 4096], .uint32))
        XCTAssertFalse(supported([1, 1, 4096], [256, 4096], .bfloat16, .float16))
        XCTAssertFalse(supported([1, 1, 4096], [256, 4096], .bfloat16, .bfloat16, .cpu))
    }
}

final class MiMoV26DecodeRouterTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["MLX_TEST_MIMO_DECODE_KERNELS"] == "1" else {
            throw XCTSkip("Requires explicit native-lane run: MLX_TEST_MIMO_DECODE_KERNELS=1")
        }
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Metal dispatch cannot be qualified by a CPU fallback")
        }
    }

    private func signal(_ shape: [Int], salt: Int, positive: Bool = false) -> MLXArray {
        MLXArray(
            (0 ..< shape.reduce(1, *)).map { i -> Float in
                let value = sin(Float((i * 13 + salt * 19) % 4093))
                return positive ? 0.01 + abs(value) * 0.01 : value * 0.1
            }, shape)
    }

    private func maxULP(_ a: MLXArray, _ b: MLXArray) -> UInt32 {
        eval(a, b)
        XCTAssertEqual(a.shape, b.shape)
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(b.dtype, .float32)
        let left = a.asArray(Float.self)
        let right = b.asArray(Float.self)
        XCTAssertTrue(left.allSatisfy(\.isFinite) && right.allSatisfy(\.isFinite))
        func ordered(_ bits: UInt32) -> UInt32 {
            bits & 0x8000_0000 == 0 ? bits | 0x8000_0000 : ~bits
        }
        return zip(left, right).map { x, y in
            let u = ordered(x.bitPattern)
            let v = ordered(y.bitPattern)
            return u > v ? u - v : v - u
        }.max() ?? 0
    }

    private func perRow(_ x: MLXArray, _ w: MLXArray, operand: DType) -> MLXArray {
        let matrix = w.asType(operand).asType(.float32).T
        return concatenated(
            (0 ..< x.dim(1)).map { row in
                matmul(x[0..., row ..< row + 1, 0...].asType(operand).asType(.float32), matrix)
            }, axis: 1)
    }

    func testNativeRouterShapeMatchesIndependentDecodeGEMVAtEveryRowCount() throws {
        let weight = signal([256, 4096], salt: 3).asType(.bfloat16)
        for rows in 1 ... 7 {
            let x = signal([1, 4096, rows], salt: rows).asType(.bfloat16).transposed(0, 2, 1)
            let actual = try XCTUnwrap(
                MiMoV26DecodeRouter.logits(
                    x, weight: weight, operandDType: .bfloat16, enabled: true))
            let expected = perRow(x, weight, operand: .bfloat16)
            XCTAssertEqual(maxULP(actual, expected), 0, "rows \(rows), per-row FP32 GEMV")
            XCTAssertEqual(
                actual.view(dtype: .uint32).asArray(UInt32.self),
                expected.view(dtype: .uint32).asArray(UInt32.self))
        }
    }

    func testDeclaredOperandCastAndWeightReloadArePreserved() throws {
        let x = signal([1, 3, 1024], salt: 11)
        let initial = signal([64, 1024], salt: 7)
        for operand: DType in [.bfloat16, .float32] {
            for weight in [initial, initial * 1.01 + 0.001] {
                let actual = try XCTUnwrap(
                    MiMoV26DecodeRouter.logits(
                        x, weight: weight, operandDType: operand, enabled: true))
                XCTAssertEqual(maxULP(actual, perRow(x, weight, operand: operand)), 0)
            }
        }
        let rounded = perRow(x, initial, operand: .bfloat16)
        let unrounded = perRow(x, initial, operand: .float32)
        XCTAssertGreaterThan(
            maxULP(rounded, unrounded), 0, "fixture must detect a skipped BF16 cast")
        XCTAssertNil(
            MiMoV26DecodeRouter.logits(x, weight: initial, operandDType: .bfloat16, enabled: false))
    }

    private func select(_ logits: MLXArray, _ bias: MLXArray) -> (MLXArray, MLXArray) {
        let scores = sigmoid(logits)
        let indices = argPartition(-(scores + bias), kth: 7, axis: -1)[.ellipsis, ..<8]
        var weights = takeAlong(scores, indices, axis: -1)
        weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        return (indices, weights)
    }

    func testVerifyReportsBatchedULPAndPreservesFixtureExpertSelection() throws {
        // The original <=2 normalized-weight bound is contradicted by an
        // independent STOCK per-row-vs-batched witness on the exact fixture.
        // Require candidate bits to match the original per-row FP32 pipeline;
        // retain the original batched selection and logit guard separately.
        let w = signal([256, 4096], salt: 23, positive: true).asType(.bfloat16)
        let bias = MLXArray((0 ..< 256).map { Float($0) / 1024 })
        for rows in 1 ... 7 {
            let x = signal([1, rows, 4096], salt: 29, positive: true).asType(.bfloat16)
            let actual = try XCTUnwrap(
                MiMoV26DecodeRouter.logits(
                    x, weight: w, operandDType: .bfloat16, enabled: true))
            // Independent stock reference: no candidate helper/custom kernel.
            let independent = perRow(x, w, operand: .bfloat16)
            let candidatePerRowLogitsULP = maxULP(actual, independent)
            XCTAssertEqual(
                candidatePerRowLogitsULP, 0, "positive fixture exact stock per-row logits")
            XCTAssertEqual(
                actual.view(dtype: .uint32).asArray(UInt32.self),
                independent.view(dtype: .uint32).asArray(UInt32.self))

            let batched = matmul(x.asType(.float32), w.asType(.float32).T)
            let logitsULP = maxULP(actual, batched)
            let nativeBaselineLogitsULP = maxULP(independent, batched)
            let a = select(actual, bias)
            let rowOracle = select(independent, bias)
            let batchOracle = select(batched, bias)
            let candidatePerRowScoresULP = maxULP(a.1, rowOracle.1)
            XCTAssertEqual(
                candidatePerRowScoresULP, 0, "positive fixture exact stock normalized weights")
            XCTAssertEqual(
                a.1.view(dtype: .uint32).asArray(UInt32.self),
                rowOracle.1.view(dtype: .uint32).asArray(UInt32.self))
            XCTAssertEqual(
                a.0.asArray(UInt32.self), rowOracle.0.asArray(UInt32.self),
                "exact stock per-row routed experts")
            XCTAssertEqual(
                a.0.asArray(UInt32.self), batchOracle.0.asArray(UInt32.self),
                "exact original batched routed experts")
            let scoresULP = maxULP(a.1, batchOracle.1)
            let nativeBaselineScoresULP = maxULP(rowOracle.1, batchOracle.1)
            // Context only: no new 3-ULP (or other) score tolerance.
            print(
                "MiMo router rows=\(rows) candidatePerRowLogitULP=\(candidatePerRowLogitsULP) candidatePerRowScoreULP=\(candidatePerRowScoresULP) candidateBatchLogitULP=\(logitsULP) nativeBatchLogitULP=\(nativeBaselineLogitsULP) candidateBatchScoreULP=\(scoresULP) nativeBatchScoreULP=\(nativeBaselineScoresULP)"
            )
            if rows == 1 {
                XCTAssertEqual(logitsULP, 0)
                XCTAssertEqual(scoresULP, 0)
            } else {
                XCTAssertLessThanOrEqual(logitsULP, 64, "positive fixture only")
            }
        }
    }
}
