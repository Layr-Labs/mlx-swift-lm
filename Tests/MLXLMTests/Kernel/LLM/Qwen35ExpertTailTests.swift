// Copyright © 2026 Eigen Labs / Darkbloom.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM

extension KernelTests {

    /// `Qwen35ExpertTail` writes the bits of the ops it replaces: the routed
    /// experts' `weightedExpertSum`, the shared expert's `sigmoid(gate) *
    /// shared`, their sum, the residual add and the next `RMSNorm`.
    @Suite(.serialized)
    struct Qwen35ExpertTailTests {
        private func bytes(_ x: MLXArray) -> Data {
            eval(x)
            return x.asData(access: .copy).data
        }

        private func normal(_ shape: [Int], _ seed: UInt64, _ dtype: DType, scale: Float = 1)
            -> MLXArray
        {
            (MLXRandom.normal(shape, key: MLXRandom.key(seed)) * scale).asType(dtype)
        }

        private func withZeros(_ x: MLXArray, _ seed: UInt64) -> MLXArray {
            let pick = MLXRandom.uniform(0 ..< 1, x.shape, key: MLXRandom.key(seed)) .< 0.02
            let sign = MLXRandom.uniform(0 ..< 1, x.shape, key: MLXRandom.key(seed + 1)) .< 0.5
            let zero = MLX.where(sign, MLXArray(-0.0), MLXArray(0.0)).asType(x.dtype)
            return MLX.where(pick, zero, x)
        }

        private func check(
            rows count: Int, experts: Int, width: Int, dtype: DType, gate: MLXArray? = nil,
            seed: UInt64
        ) throws {
            let rows = withZeros(normal([count, experts, width], seed, dtype, scale: 2), seed + 1)
            let weights = softmax(normal([count, experts], seed + 2, dtype, scale: 2), axis: -1)
                .asType(dtype)
            let sharedGate = gate ?? normal([count, 1], seed + 3, dtype, scale: 3)
            let shared = withZeros(normal([count, width], seed + 4, dtype, scale: 2), seed + 5)
            let x = withZeros(normal([count, width], seed + 6, dtype, scale: 8), seed + 7)
            let weight = (normal([width], seed + 8, .float32, scale: 0.2) + 1).asType(dtype)
            let parts = Qwen35ExpertTail.Parts(
                rows: rows, weights: weights, sharedGate: sharedGate, shared: shared)
            let fused = Qwen35ExpertTail.apply(x, parts, weight: weight, eps: 1e-6)
            // The unfused ops: `weightedExpertSum`, the shared combine, the
            // residual add and `MLXFast.rmsNorm`.
            let routed = weightedExpertSum(rows, weights)
            let combined = routed + sigmoid(sharedGate) * shared
            let h = x + combined
            let normed = MLXFast.rmsNorm(h, weight: weight, eps: 1e-6)
            let label = "\(dtype) rows \(count) experts \(experts) width \(width)"
            #expect(bytes(fused.sum) == bytes(h), "sum \(label)")
            #expect(bytes(fused.norm) == bytes(normed), "norm \(label)")
            // The compiled combine the qwen3_5 path takes.
            let compiled = Qwen4ExpFusions.sharedExpertCombine(routed, sharedGate, shared)
            #expect(bytes(x + compiled) == bytes(fused.sum), "compiled combine \(label)")
        }

        @Test func tailKeepsTheUnfusedBits() throws {
            for dtype in [DType.bfloat16, .float16] {
                for width in [2048, 1024, 4096, 128] {
                    guard Qwen35ExpertTail.supports(width: width, experts: 8, dtype: dtype) else {
                        #expect(Device.defaultDevice().deviceType != .gpu, "\(dtype) \(width)")
                        continue
                    }
                    for (count, experts) in [(1, 8), (3, 8), (2, 4), (1, 6)] {
                        try check(
                            rows: count, experts: experts, width: width, dtype: dtype,
                            seed: UInt64(width * 3 + count * 7 + experts))
                    }
                }
            }
        }

        /// Every 16-bit value as the shared expert's gate.
        @Test func everyGateValueKeepsTheUnfusedBits() throws {
            for dtype in [DType.bfloat16, .float16] {
                guard Qwen35ExpertTail.supports(width: 8, experts: 8, dtype: dtype) else {
                    #expect(Device.defaultDevice().deviceType != .gpu, "\(dtype)")
                    continue
                }
                let values: [Float] = (0 ..< 65_536).map { pattern in
                    dtype == .bfloat16
                        ? Float(bitPattern: UInt32(pattern) << 16)
                        : Float(Float16(bitPattern: UInt16(pattern)))
                }
                let gate = MLXArray(values).asType(dtype).reshaped(65_536, 1)
                try check(rows: 65_536, experts: 8, width: 8, dtype: dtype, gate: gate, seed: 91)
            }
        }

        @Test func switchOffAndOddWidthsKeepTheUnfusedOps() {
            #expect(!Qwen35ExpertTail.supports(width: 2880, experts: 8, dtype: .bfloat16))
            #expect(!Qwen35ExpertTail.supports(width: 2048, experts: 8, dtype: .float32))
            // More than 8 rows share the column reduction's lanes.
            #expect(!Qwen35ExpertTail.supports(width: 2048, experts: 10, dtype: .bfloat16))
            Qwen35ExpertTail.$enabledOverride.withValue(false) {
                #expect(!Qwen35ExpertTail.supports(width: 2048, experts: 8, dtype: .bfloat16))
            }
        }
    }
}
