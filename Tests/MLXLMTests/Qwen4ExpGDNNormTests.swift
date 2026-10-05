// Qwen4ExpGDNNormTests.swift
//
// The gated deltanet query/key normalization must be the original model's
// `l2norm`: `x / sqrt(sum(x * x) + eps)`, with the epsilon added to the SUM of
// squares. An RMS norm adds the epsilon to the mean, which is the same as
// `head_dim * eps` on the sum. At small magnitudes the two differ a lot.
//
// The expected values are computed on the host in Double, from the formula.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class Qwen4ExpGDNNormTests: XCTestCase {

    private static let headDim = 128
    private static let eps = 1e-6

    /// One head of `headDim` values, `[1, 1, 1, headDim]`.
    private func head(_ values: [Float]) -> MLXArray {
        MLXArray(values).reshaped(1, 1, 1, Self.headDim)
    }

    /// `x / sqrt(sum(x * x) + eps)`, in Double.
    private func l2norm(_ values: [Float]) -> [Double] {
        let sumOfSquares = values.reduce(0.0) { $0 + Double($1) * Double($1) }
        let scale = 1 / (sumOfSquares + Self.eps).squareRoot()
        return values.map { Double($0) * scale }
    }

    private func assertClose(
        _ actual: MLXArray, _ expected: [Double], file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let values = actual.asType(.float32).asArray(Float.self)
        XCTAssertEqual(values.count, expected.count, file: file, line: line)
        for (a, e) in zip(values, expected) {
            XCTAssertEqual(Double(a), e, accuracy: abs(e) * 1e-5, file: file, line: line)
        }
    }

    private func check(_ qValues: [Float], _ kValues: [Float]) {
        let (q, k) = Qwen4ExpGDNNorm.queryKey(
            q: head(qValues), k: head(kValues), dtype: .float32)
        let invScale = 1 / Double(Self.headDim).squareRoot()
        assertClose(q, l2norm(qValues).map { $0 * invScale })
        assertClose(k, l2norm(kValues))
    }

    /// Small, constant values: the sum of squares is 1.28e-6, so the epsilon
    /// is about half of the norm's denominator. The RMS form would add
    /// 1.28e-4 instead and give a value about 7.5 times smaller.
    func testSmallValuesUseTheEpsilonOnTheSum() {
        check(
            [Float](repeating: 1e-4, count: Self.headDim),
            [Float](repeating: -1e-4, count: Self.headDim))
    }

    /// Mixed signs and magnitudes near the epsilon scale.
    func testMixedValuesMatchTheFormula() {
        let q = (0 ..< Self.headDim).map { Float(sin(Double($0) * 0.7)) * 2e-4 }
        let k = (0 ..< Self.headDim).map { Float(cos(Double($0) * 1.3)) * 5e-5 }
        check(q, k)
    }
}
