// Copyright © 2026 Eigen Labs / Darkbloom.

import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {
    /// `GatedDeltaConvolution` writes the bits of the unfused ops it replaces:
    /// the concatenation `[state; inputs]`, MLX's depthwise `conv1d` and the
    /// compiled `silu` (MLX's `Sigmoid` functor, whose exponential is
    /// `metal::precise::exp`).
    @Suite(.serialized)
    struct GatedDeltaConvolutionTests {
        private var verified: Bool {
            Device.defaultDevice().deviceType == .gpu
                && ResidualRMSNorm.verifiedFamilies.contains(AffineQuantizedVectors.family())
        }

        private func bytes(_ x: MLXArray) -> Data {
            eval(x)
            return x.asData(access: .copy).data
        }

        @Test func fusedStepKeepsTheUnfusedBits() throws {
            guard verified else {
                let x = MLXArray.zeros([1, 1, 64])
                #expect(
                    GatedDeltaConvolution.apply(
                        state: MLXArray.zeros([1, 3, 64]), inputs: x,
                        weight: MLXArray.zeros([64, 4, 1])) == nil)
                return
            }
            for dtype in [DType.bfloat16, .float16, .float32] {
                // Qwen 3.5's convolution: 4 taps; channels of the 9B, 27B and A3B and a partial thread group.
                for channels in [8192, 10240, 6144, 1000] {
                    for (batch, steps) in [(1, 1), (4, 1), (2, 3), (1, 8)] {
                        let seed = UInt64(channels * 31 + batch * 7 + steps)
                        let taps = 4
                        let state =
                            (MLXRandom.normal(
                                [batch, taps - 1, channels], key: MLXRandom.key(seed)) * 2).asType(
                                dtype)
                        let inputs =
                            (MLXRandom.normal(
                                [batch, steps, channels], key: MLXRandom.key(seed + 1)) * 2).asType(
                                dtype)
                        let weight =
                            (MLXRandom.normal(
                                [channels, taps, 1], key: MLXRandom.key(seed + 2)) / 2).asType(
                                dtype)
                        let fused = try #require(
                            GatedDeltaConvolution.apply(
                                state: state, inputs: inputs, weight: weight))
                        let convInput = concatenated([state, inputs], axis: 1)
                        let output = silu(
                            conv1d(convInput, weight, stride: 1, padding: 0, groups: channels))
                        #expect(bytes(fused.convInput) == bytes(convInput))
                        #expect(
                            bytes(fused.output) == bytes(output),
                            "\(dtype) channels \(channels) batch \(batch) steps \(steps)")
                    }
                }
            }
        }

        private func normal(_ shape: [Int], _ seed: UInt64, _ dtype: DType, scale: Float = 2)
            -> MLXArray
        {
            (MLXRandom.normal(shape, key: MLXRandom.key(seed)) * scale).asType(dtype)
        }

        /// The convolution and q/k norm in one kernel write what the two
        /// kernels and the unfused ops write: `[state; inputs]`, the scaled
        /// RMS norms of every q and k head, and the value channels.
        @Test func convolutionAndQKNormKeepTheUnfusedBits() throws {
            guard verified else {
                #expect(
                    GatedDeltaConvolutionQKNorm.apply(
                        state: MLXArray.zeros([1, 3, 512]), inputs: MLXArray.zeros([1, 1, 512]),
                        weight: MLXArray.zeros([512, 4, 1]), heads: 1, dim: 128,
                        qScale: MLXArray(Float(1)), kScale: MLXArray(Float(1)), eps: 1e-6) == nil)
                return
            }
            for dtype in [DType.bfloat16, .float16, .float32] {
                // (key heads, head width, value channels): the 27B, the 9B and
                // A3B, and a narrow head.
                for (heads, dim, valueChannels) in [(16, 128, 6144), (16, 128, 4096), (2, 64, 192)]
                {
                    let channels = 2 * heads * dim + valueChannels
                    for (batch, steps) in [(1, 1), (4, 1), (2, 3), (1, 8)] {
                        let seed = UInt64(channels * 13 + batch * 5 + steps)
                        let state = normal([batch, 3, channels], seed, dtype)
                        // The projection's column slice, read through its strides.
                        let wide = normal([batch, steps, channels + 64], seed + 1, dtype)
                        let inputs = wide[0..., 0..., 32 ..< (32 + channels)]
                        let weight = normal([channels, 4, 1], seed + 2, dtype, scale: 0.5)
                        let qScale = MLXArray(pow(Float(dim), -1)).asType(dtype)
                        let kScale = MLXArray(pow(Float(dim), -0.5)).asType(dtype)
                        let fused = try #require(
                            GatedDeltaConvolutionQKNorm.apply(
                                state: state, inputs: inputs, weight: weight, heads: heads,
                                dim: dim, qScale: qScale, kScale: kScale, eps: 1e-6))
                        // The unfused ops.
                        let convInput = concatenated([state, inputs], axis: 1)
                        let convOut = silu(
                            conv1d(convInput, weight, stride: 1, padding: 0, groups: channels))
                        let parts = split(
                            convOut, indices: [heads * dim, 2 * heads * dim], axis: -1)
                        let q =
                            qScale
                            * MLXFast.rmsNorm(
                                parts[0].reshaped(batch, steps, heads, dim),
                                weight: MLXArray.mlxNone, eps: 1e-6)
                        let k =
                            kScale
                            * MLXFast.rmsNorm(
                                parts[1].reshaped(batch, steps, heads, dim),
                                weight: MLXArray.mlxNone, eps: 1e-6)
                        let label = "\(dtype) \(heads)x\(dim)+\(valueChannels) \(batch)x\(steps)"
                        #expect(bytes(fused.convInput) == bytes(convInput), "\(label)")
                        #expect(bytes(fused.q) == bytes(q), "q \(label)")
                        #expect(bytes(fused.k) == bytes(k), "k \(label)")
                        #expect(bytes(fused.v) == bytes(parts[2]), "v \(label)")
                        // The two kernels it replaces.
                        let conv = try #require(
                            GatedDeltaConvolution.apply(
                                state: state, inputs: inputs, weight: weight))
                        let pair = try #require(
                            GatedDeltaQKNorm.apply(
                                convOut: conv.output, heads: heads, dim: dim, qScale: qScale,
                                kScale: kScale, eps: 1e-6))
                        #expect(bytes(fused.q) == bytes(pair.q), "pair q \(label)")
                        #expect(bytes(fused.k) == bytes(pair.k), "pair k \(label)")
                    }
                }
            }
        }

        /// The gated output norm in one kernel writes what `Qwen3NextRMSNormGated`
        /// and the fast norm plus compiled tail write, for every 16-bit gate value.
        @Test func outputNormKeepsTheUnfusedBits() throws {
            guard verified else {
                #expect(
                    GatedDeltaOutputNorm.apply(
                        MLXArray.zeros([1, 1, 2, 128]), gate: MLXArray.zeros([1, 1, 2, 128]),
                        weight: MLXArray.ones([128]), eps: 1e-6) == nil)
                return
            }
            for dtype in [DType.bfloat16, .float16, .float32] {
                for dim in [128, 64] {
                    let norm = Qwen3NextRMSNormGated(
                        dimensions: dim, eps: 1e-6, gateActivation: .silu)
                    let weight = (normal([dim], 7, .float32, scale: 0.2) + 1).asType(dtype)
                    norm.update(parameters: ModuleParameters.unflattened([("weight", weight)]))
                    var shapes = [
                        [1, 1, 48, dim], [4, 1, 48, dim], [1, 3, 32, dim], [2, 1, 8, dim],
                    ]
                    if dtype != .float32 {
                        shapes.append([1, 1, 65_536 / dim, dim])
                    }
                    for shape in shapes {
                        let seed = UInt64(shape.reduce(1, *) + dim)
                        let x = normal(shape, seed, dtype)
                        var gate: MLXArray
                        if shape[2] * dim == 65_536 {
                            // Every 16-bit pattern as a gate value.
                            let values: [Float] = (0 ..< 65_536).map { pattern in
                                dtype == .bfloat16
                                    ? Float(bitPattern: UInt32(pattern) << 16)
                                    : Float(Float16(bitPattern: UInt16(pattern)))
                            }
                            gate = MLXArray(values).asType(dtype).reshaped(shape)
                        } else {
                            // A column slice of a wider projection, as `z` is.
                            let rows = shape[0] * shape[1]
                            let wide = normal(
                                [rows, shape[2] * dim + 32], seed + 1, dtype, scale: 4)
                            gate = wide[0..., 16 ..< (16 + shape[2] * dim)].reshaped(shape)
                        }
                        let fused = try #require(
                            GatedDeltaOutputNorm.apply(x, gate: gate, weight: weight, eps: 1e-6))
                        let reference = norm(x, gate: gate)
                        let tail = Qwen35DecodeFusions.gatedSiluNormTail(
                            normed: MLXFast.rmsNorm(x, weight: weight, eps: 1e-6), gate: gate,
                            dtype: dtype)
                        #expect(bytes(fused) == bytes(reference), "\(dtype) \(shape)")
                        #expect(bytes(fused) == bytes(tail), "tail \(dtype) \(shape)")
                    }
                }
            }
        }
    }
}
