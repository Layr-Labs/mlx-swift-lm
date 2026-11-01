// Copyright 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

extension KernelTests {
    /// Device memory-bandwidth ceilings for GLM decode planning: streaming
    /// reductions and copies, and one-token Q4 g64 GEMVs at GLM-5.3 shapes,
    /// alone and as independent batches in one evaluation. Benchmark only
    /// (`GLM53_BW_BENCH=1`); it asserts nothing about speed.
    @Suite(.serialized)
    struct GLM5NextBandwidthBenchTests {
        private static var enabled: Bool {
            ProcessInfo.processInfo.environment["GLM53_BW_BENCH"] == "1"
        }

        private static func best(_ repetitions: Int, _ body: () -> [MLXArray]) -> Double {
            var best = Double.infinity
            for _ in 0 ..< repetitions {
                let start = ContinuousClock.now
                eval(body())
                let duration = start.duration(to: .now)
                let ms =
                    Double(duration.components.seconds) * 1000
                    + Double(duration.components.attoseconds) / 1e15
                best = min(best, ms)
            }
            return best
        }

        @Test(.enabled(if: enabled)) func deviceBandwidthCeilings() throws {
            try Device.withDefaultDevice(.gpu) {
                let gigabytes = 4
                let elements = gigabytes << 29
                let big = MLXRandom.uniform(low: -1, high: 1, [elements]).asType(.bfloat16)
                eval(big)
                let bytes = Double(elements * 2)
                let sumMs = Self.best(5) { [big.sum()] }
                print(
                    String(
                        format: "GLM53_BW stream_sum gb=%.1f ms=%.3f gbps=%.1f", bytes / 1e9, sumMs,
                        bytes / sumMs / 1e6))
                let copyMs = Self.best(5) { [big + 1] }
                print(
                    String(
                        format: "GLM53_BW stream_copy gb=%.1f ms=%.3f gbps=%.1f", 2 * bytes / 1e9,
                        copyMs, 2 * bytes / copyMs / 1e6))

                // (name, rows, inputs, independent copies per evaluation)
                let shapes: [(String, Int, Int, Int)] = [
                    ("lm_head", 154_880, 4096, 1), ("kda_fused_in", 24_896, 4096, 34),
                    ("kda_o_proj", 4096, 8192, 34), ("dsa_q_a", 1536, 4096, 11),
                    ("dsa_q_b", 16_384, 1536, 11), ("dsa_o_proj", 4096, 16_384, 11),
                    ("expert_gate", 2048, 4096, 42), ("expert_down", 4096, 2048, 42),
                ]
                for (name, rows, inputs, copies) in shapes {
                    let weights = (0 ..< copies).map { index -> (MLXArray, MLXArray, MLXArray?) in
                        let w =
                            MLXRandom.normal([rows, inputs], key: MLXRandom.key(UInt64(index)))
                            * 0.02
                        let q = MLX.quantized(w.asType(.bfloat16), groupSize: 64, bits: 4)
                        eval(q.wq, q.scales, q.biases!)
                        return (q.wq, q.scales, q.biases)
                    }
                    let x = MLXRandom.normal([1, inputs]).asType(.bfloat16)
                    eval(x)
                    let perMatrix = Double(rows * inputs) * (0.5 + 4.0 / 64)
                    func gemv(_ i: Int) -> MLXArray {
                        quantizedMM(
                            x, weights[i].0, scales: weights[i].1, biases: weights[i].2,
                            transpose: true, groupSize: 64, bits: 4)
                    }
                    let single = Self.best(20) { [gemv(0)] }
                    let all = Self.best(10) { (0 ..< copies).map(gemv) }
                    print(
                        String(
                            format:
                                "GLM53_BW gemv %@ rows=%d inputs=%d mb=%.1f single_ms=%.4f single_gbps=%.1f batch=%d batch_ms=%.3f batch_gbps=%.1f per_matrix_us=%.1f",
                            name, rows, inputs, perMatrix / 1e6, single, perMatrix / single / 1e6,
                            copies, all, Double(copies) * perMatrix / all / 1e6,
                            all * 1000 / Double(copies)))
                    Memory.clearCache()
                }
            }
        }
    }
}
