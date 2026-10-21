// Copyright 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

extension KernelTests {
    /// Sorted expert-gather matmul at GLM-5.3 prefill shapes, affine Q4/g64
    /// versus NVFP4/g16 banks. Benchmark only (`GLM53_NVFP4_GATHER_BENCH=1`).
    @Suite(.serialized)
    struct GLM5NextNVFP4GatherBenchTests {
        private static var enabled: Bool {
            ProcessInfo.processInfo.environment["GLM53_NVFP4_GATHER_BENCH"] == "1"
        }

        private static func best(_ repetitions: Int, _ body: () -> MLXArray) -> Double {
            var best = Double.infinity
            for _ in 0 ..< repetitions {
                let start = ContinuousClock.now
                eval(body())
                let d = start.duration(to: .now)
                best = min(
                    best,
                    Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15)
            }
            return best
        }

        @Test(.enabled(if: enabled)) func sortedGatherAffineVersusNVFP4() throws {
            try Device.withDefaultDevice(.gpu) {
                let (experts, rows, inputs) = (288, 2048, 4096)
                let dense = (MLXRandom.normal([experts, rows, inputs]) * 0.02).asType(.bfloat16)
                eval(dense)
                let affine = MLX.quantized(dense, groupSize: 64, bits: 4, mode: .affine)
                let fp4 = MLX.quantized(dense, groupSize: 16, bits: 4, mode: .nvfp4)
                eval(affine.wq, affine.scales, affine.biases!, fp4.wq, fp4.scales)
                print(
                    "GLM53_NVFP4_GATHER fp4 wq=\(fp4.wq.shape) \(fp4.wq.dtype) scales=\(fp4.scales.shape) \(fp4.scales.dtype) biases=\(String(describing: fp4.biases?.shape))"
                )
                for tokens in [32, 512, 2048] {
                    let assignments = tokens * 8
                    for skew in [false, true] {
                        // Skewed routing: half the assignments on 8 experts.
                        let uniform = MLXRandom.randInt(Int32(0) ..< Int32(experts), [assignments])
                        let hot = MLXRandom.randInt(Int32(0) ..< Int32(8), [assignments])
                        let pick = MLXRandom.uniform(low: 0, high: 1, [assignments]) .< 0.5
                        let indices = (skew ? MLX.where(pick, hot, uniform) : uniform).asType(
                            .uint32)
                        let sorted = MLX.sorted(indices)
                        let x = (MLXRandom.normal([assignments, 1, inputs])).asType(.bfloat16)
                        eval(sorted, x)
                        let tAffine = Self.best(5) {
                            gatherQuantizedMM(
                                x, affine.wq, scales: affine.scales, biases: affine.biases,
                                rhsIndices: sorted, transpose: true, groupSize: 64, bits: 4,
                                mode: .affine, sortedIndices: true)
                        }
                        let tFP4 = Self.best(5) {
                            gatherQuantizedMM(
                                x, fp4.wq, scales: fp4.scales, biases: nil,
                                rhsIndices: sorted, transpose: true, groupSize: 16, bits: 4,
                                mode: .nvfp4, sortedIndices: true)
                        }
                        let tFP4Unsorted = Self.best(3) {
                            gatherQuantizedMM(
                                x, fp4.wq, scales: fp4.scales, biases: nil,
                                rhsIndices: sorted, transpose: true, groupSize: 16, bits: 4,
                                mode: .nvfp4, sortedIndices: false)
                        }
                        print(
                            String(
                                format:
                                    "GLM53_NVFP4_GATHER tokens=%d assignments=%d skew=%d affine_sorted_ms=%.3f nvfp4_sorted_ms=%.3f nvfp4_unsorted_ms=%.3f",
                                tokens, assignments, skew ? 1 : 0, tAffine, tFP4, tFP4Unsorted))
                    }
                }
            }
        }
    }
}
