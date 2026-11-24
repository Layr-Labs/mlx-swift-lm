// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import Testing

@testable import MLXLMCommon

extension KernelTests {

    /// `SamplerV2.exponentialNoise(fromUniform:)`, the Exp(1) noise of the
    /// CBv2 exponential race, on the Metal device.
    @Suite
    struct CBv2ExponentialNoiseTests {

        /// MLX's Float32 uniforms are `k · 2^-32`. k ≤ 128 rounds `1 - u` to
        /// 1, so log1p returns `-u` exactly; from k = 129 it takes the
        /// division branch through `log(1 + x)`. The last draw is the largest
        /// uniform, `1 - 2^-24`.
        static let draws: [Float] = {
            let k: [Int] = [0, 1, 128, 129, 192, 256, 1 << 12, 1 << 16, 1 << 20, (1 << 24) - 1]
            return k.map { Float($0) * 0x1p-32 } + [1 - 0x1p-24]
        }()

        /// Each draw against Double `-log1p(-max(u, 2^-33))`, within 8 ulp
        /// (the fast-math bounds of log and division with room). A log that
        /// rounds a small draw's log1p to zero is millions of ulp off.
        @Test func noiseMatchesDoubleLog1pOnMetal() throws {
            let noise = Device.withDefaultDevice(.gpu) {
                SamplerV2.exponentialNoise(fromUniform: MLXArray(Self.draws)).asArray(Float.self)
            }
            try #require(noise.count == Self.draws.count)
            for (u, e) in zip(Self.draws, noise) {
                let expected = Float(-Foundation.log1p(-max(Double(u), 0x1p-33)))
                #expect(e.isFinite && e >= 0x1p-33, "u \(u): noise \(e)")
                #expect(
                    abs(e - expected) <= 8 * expected.ulp,
                    "u \(u): noise \(e), expected \(expected)")
            }
        }

        /// The premise of `testTinyUniformDrawDoesNotPromoteTailToken`: at its
        /// key, token 31,133 draws `u = 70 · 2^-32`, so `1 - u` rounds to 1 in
        /// Float32 and the noise is `u` exactly.
        @Test func tailRegressionKeyDrawsItsTinyUniform() {
            let noise = Device.withDefaultDevice(.gpu) {
                SamplerV2(fallbackSeed: 0).verifyExponentialNoise(
                    rows: [(seed: 42, id: 77, step: 137, greedy: false)], vocab: 201_088
                )[0, 31_133].item(Float.self)
            }
            #expect(noise == 70 * 0x1p-32)
        }
    }
}
