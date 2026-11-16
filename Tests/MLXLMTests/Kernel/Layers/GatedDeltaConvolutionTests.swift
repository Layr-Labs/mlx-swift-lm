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
    }
}
