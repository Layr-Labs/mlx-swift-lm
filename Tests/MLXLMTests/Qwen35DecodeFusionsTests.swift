// Copyright © 2026 Eigen Labs / Darkbloom.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM

/// The qwen3_5 compiled chains must be the unfused expressions' bits.
final class Qwen35DecodeFusionsTests: XCTestCase {
    func testSwitchIsOnByDefaultAndOffByEnvironmentOrHardware() {
        XCTAssertTrue(Qwen35DecodeFusions.enabled(environment: [:], compiledDecodeSupported: true))
        XCTAssertTrue(
            Qwen35DecodeFusions.enabled(
                environment: [Qwen35DecodeFusions.envFlag: "1"],
                compiledDecodeSupported: true))
        for off in ["0", "false", "OFF", "no"] {
            XCTAssertFalse(
                Qwen35DecodeFusions.enabled(
                    environment: [Qwen35DecodeFusions.envFlag: off],
                    compiledDecodeSupported: true))
        }
        XCTAssertFalse(
            Qwen35DecodeFusions.enabled(environment: [:], compiledDecodeSupported: false))
    }

    private func bits(_ x: MLXArray) -> [UInt32] {
        x.dtype == .float32
            ? x.asArray(Float.self).map(\.bitPattern)
            : x.asType(.float32).asArray(Float.self).map(\.bitPattern)
    }

    /// Inputs with rounding ties, large magnitudes and both signs.
    private func inputs(_ shape: [Int], seed: UInt64, dtype: DType) -> MLXArray {
        let key = MLXRandom.key(seed)
        let base = MLXRandom.normal(shape, key: key) * 6
        let edges = MLXArray([Float(0), -0.0, 1e-30, -40, 40, 88, -88, 0.5]).asType(dtype)
        let flat = base.asType(dtype).reshaped(-1)
        flat[0 ..< 8] = edges
        return flat.reshaped(shape)
    }

    func testGatedDeltaGatesMatchTheUnfusedExpressionsBitForBit() {
        for (dtype, seed) in [(DType.bfloat16, UInt64(1)), (.float16, 2), (.float32, 3)] {
            for shape in [[1, 1, 48], [1, 7, 24], [2, 3, 16]] {
                let a = inputs(shape, seed: seed, dtype: dtype)
                let b = inputs(shape, seed: seed + 100, dtype: dtype)
                let heads = shape[2]
                let aLog = inputs([heads], seed: seed + 200, dtype: .float32) / 3
                let dtBias = inputs([heads], seed: seed + 300, dtype: dtype)
                let fused = Qwen35DecodeFusions.gdnGates(a: a, b: b, aLog: aLog, dtBias: dtBias)
                let beta = sigmoid(b).asType(.float32)
                // The separate ops (`computeGatedDeltaG` is itself compiled).
                let g = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
                eval(fused.g, fused.beta, g, beta)
                XCTAssertEqual(fused.g.dtype, g.dtype)
                XCTAssertEqual(fused.beta.dtype, beta.dtype)
                XCTAssertEqual(bits(fused.g), bits(g), "g \(dtype) \(shape)")
                XCTAssertEqual(bits(fused.beta), bits(beta), "beta \(dtype) \(shape)")
            }
        }
    }

    func testGatedSiluNormTailMatchesTheGatedNormBitForBit() {
        for (dtype, weightType) in [
            (DType.bfloat16, DType.bfloat16), (.bfloat16, .float32), (.float16, .float32),
            (.float32, .float32),
        ] {
            let norm = Qwen3NextRMSNormGated(dimensions: 128, eps: 1e-6, gateActivation: .silu)
            norm.update(
                parameters: ModuleParameters.unflattened([
                    ("weight", (inputs([128], seed: 7, dtype: .float32) / 6 + 1).asType(weightType))
                ]))
            let x = inputs([1, 3, 4, 128], seed: 11, dtype: dtype)
            let gate = inputs([1, 3, 4, 128], seed: 13, dtype: dtype)
            let reference = norm(x, gate: gate)
            let normed = MLXFast.rmsNorm(x, weight: norm.weight, eps: norm.eps)
            let fused = Qwen35DecodeFusions.gatedSiluNormTail(
                normed: normed, gate: gate, dtype: x.dtype)
            eval(reference, fused)
            XCTAssertEqual(fused.dtype, reference.dtype)
            XCTAssertEqual(bits(fused), bits(reference), "\(dtype) weight \(weightType)")
        }
    }
}
