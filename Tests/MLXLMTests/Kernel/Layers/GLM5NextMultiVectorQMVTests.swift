// Copyright 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {
    /// `GLM5NextMultiVectorQMV`: every vector's output is bitwise the layer
    /// applied to that vector alone (MLX's one-row `qmv_fast`), at 1-8 vectors,
    /// affine 4 and 8 bits, several group sizes, and the GLM head's geometry.
    @Suite(.serialized)
    struct GLM5NextMultiVectorQMVTests {
        private static func bits(_ array: MLXArray) -> [UInt32] {
            array.asType(.float32).asArray(Float.self).map(\.bitPattern)
        }

        private static func check(n: Int, k: Int, bits: Int, groupSize: Int, vectors: [Int], seed: UInt64)
            throws
        {
            let weight = (MLXRandom.normal([n, k], key: MLXRandom.key(seed)) * 0.02).asType(.bfloat16)
            let layer = QuantizedLinear(weight: weight, bias: nil, groupSize: groupSize, bits: bits)
            for count in vectors {
                let x = (MLXRandom.normal([count, k], key: MLXRandom.key(seed + UInt64(count))) * 0.5)
                    .asType(.bfloat16)
                let batched = try #require(GLM5NextMultiVectorQMV.apply(layer, x))
                let alone = (0 ..< count).map { layer(x[$0 ..< $0 + 1].reshaped(1, 1, k)).reshaped(n) }
                eval([batched] + alone)
                for v in 0 ..< count {
                    #expect(
                        Self.bits(batched[v]) == Self.bits(alone[v]),
                        "N \(n) K \(k) b\(bits) g\(groupSize) V \(count) vector \(v)")
                }
            }
        }

        @Test func everyVectorIsItsOwnOneRowProductOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                try Self.check(n: 1024, k: 4096, bits: 4, groupSize: 64, vectors: Array(1 ... 8), seed: 1)
                try Self.check(n: 512, k: 2048, bits: 4, groupSize: 32, vectors: [1, 3, 8], seed: 2)
                try Self.check(n: 512, k: 4096, bits: 4, groupSize: 128, vectors: [2, 5], seed: 3)
                try Self.check(n: 1024, k: 4096, bits: 8, groupSize: 64, vectors: [1, 4, 7], seed: 4)
                // The GLM head (154,880 x 4096, affine 4-bit, group 64).
                try Self.check(n: 154_880, k: 4096, bits: 4, groupSize: 64, vectors: [1, 4], seed: 5)
            }
        }

        @Test func layersItCannotReadAreRefused() throws {
            try Device.withDefaultDevice(.gpu) {
                let x = MLXArray.zeros([2, 4096], dtype: .bfloat16)
                #expect(GLM5NextMultiVectorQMV.apply(Linear(4096, 64, bias: false), x) == nil)
                let quantized = QuantizedLinear(
                    weight: (MLXRandom.normal([64, 4096]) * 0.02).asType(.bfloat16), bias: nil, groupSize: 64,
                    bits: 4)
                #expect(GLM5NextMultiVectorQMV.apply(quantized, MLXArray.zeros([9, 4096], dtype: .bfloat16)) == nil)
                #expect(GLM5NextMultiVectorQMV.apply(quantized, MLXArray.zeros([2, 4096], dtype: .float32)) == nil)
                #expect(GLM5NextMultiVectorQMV.apply(quantized, x) != nil)
            }
        }

        /// Opt-in timing (`GLM53_MULTI_QMV_BENCH=1`): the GLM head for V vectors
        /// in one launch against V one-row launches, median microseconds.
        @Test func headTiming() throws {
            guard ProcessInfo.processInfo.environment["GLM53_MULTI_QMV_BENCH"] == "1" else { return }
            try Device.withDefaultDevice(.gpu) {
                let (n, k) = (154_880, 4096)
                let layer = QuantizedLinear(
                    weight: (MLXRandom.normal([n, k], key: MLXRandom.key(9)) * 0.02).asType(.bfloat16), bias: nil, groupSize: 64,
                    bits: 4)
                eval(layer)
                func time(_ body: () -> [MLXArray]) -> Double {
                    for _ in 0 ..< 3 { eval(body()) }
                    var samples: [Double] = []
                    for _ in 0 ..< 9 {
                        let start = DispatchTime.now().uptimeNanoseconds
                        var outputs: [MLXArray] = []
                        for _ in 0 ..< 8 { outputs += body() }
                        eval(outputs)
                        samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3 / 8)
                    }
                    return samples.sorted()[4]
                }
                for count in [1, 2, 4, 8] {
                    let x = (MLXRandom.normal([count, k], key: MLXRandom.key(UInt64(count))) * 0.5).asType(.bfloat16)
                    eval(x)
                    let batched = time { [GLM5NextMultiVectorQMV.apply(layer, x)!] }
                    let separate = time { (0 ..< count).map { layer(x[$0 ..< $0 + 1].reshaped(1, 1, k)) } }
                    let wide = time { [layer(x.reshaped(1, count, k))] }
                    print(String(format: "GLM53_MULTI_QMV V=%d multi_us=%.1f separate_us=%.1f qmv_wide_us=%.1f",
                        count, batched, separate, wide))
                }
            }
        }
    }
}
