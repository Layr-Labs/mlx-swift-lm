import Foundation
import MLX
import Testing

@testable import MLXLMCommon

extension KernelTests {

    /// Tests of the bicubic and nearest interpolation Metal kernels in
    /// `InterpolationUtils.swift`, with small synthetic images.
    ///
    /// The nearest kernel is compared with a reference in Swift. The bicubic
    /// kernel is checked through properties of its weights: they sum to 1, a
    /// cubic kernel keeps a linear ramp linear away from the borders, and
    /// the kernel is 1 at distance 0 and 0 at distances 1 and 2.
    @Suite
    struct InterpolationUtilsTests {

        /// A `[B, C, H, W]` float32 image with values `f(b, c, y, x)`.
        static func image(
            batch: Int = 1, channels: Int = 1, height: Int, width: Int,
            _ f: (Int, Int, Int, Int) -> Float
        ) -> MLXArray {
            var values: [Float] = []
            for b in 0 ..< batch {
                for c in 0 ..< channels {
                    for y in 0 ..< height {
                        for x in 0 ..< width {
                            values.append(f(b, c, y, x))
                        }
                    }
                }
            }
            return MLXArray(values).reshaped(batch, channels, height, width)
        }

        static func random(batch: Int, channels: Int, height: Int, width: Int) -> MLXArray {
            let x = MLXRandom.normal([batch, channels, height, width], key: MLXRandom.key(3))
            eval(x)
            return x
        }

        // Tolerance: the kernels compute in float32. Sums of 16 weighted
        // values of size 1 give errors near 1e-6.
        static let tolerance: Float = 1e-5

        @Test(arguments: [(5, 7, 10, 3), (6, 6, 3, 9), (4, 4, 4, 4)])
        func nearestMatchesTheFloorIndexReference(
            inH: Int, inW: Int, outH: Int, outW: Int
        ) {
            let x = Self.random(batch: 2, channels: 3, height: inH, width: inW)
            let output = nearestInterpolate(x, size: (outH, outW))
            #expect(output.shape == [2, 3, outH, outW])

            let values = x.asArray(Float.self)
            var expected: [Float] = []
            for bc in 0 ..< 6 {
                for y in 0 ..< outH {
                    for xOut in 0 ..< outW {
                        let yIn = min(
                            Int((Float(y) * Float(inH) / Float(outH)).rounded(.down)), inH - 1)
                        let xIn = min(
                            Int((Float(xOut) * Float(inW) / Float(outW)).rounded(.down)), inW - 1)
                        expected.append(values[(bc * inH + yIn) * inW + xIn])
                    }
                }
            }
            #expect(output.asArray(Float.self) == expected)
        }

        @Test func nearestScaleFactorSetsTheOutputSize() {
            let x = Self.random(batch: 1, channels: 2, height: 4, width: 6)
            #expect(nearestInterpolate(x, scaleFactor: 2).shape == [1, 2, 8, 12])
            #expect(nearestInterpolate(x, scaleFactor: (0.5, 1.5)).shape == [1, 2, 2, 9])
            // Scale 2 repeats each pixel twice in each direction.
            let up = nearestInterpolate(x, scaleFactor: 2)
            #expect(
                SyntheticModel.maxAbsDifference(up[0..., 0..., .stride(by: 2), .stride(by: 2)], x)
                    == 0)
            #expect(
                SyntheticModel.maxAbsDifference(
                    up[0..., 0..., 1..., 1...][0..., 0..., .stride(by: 2), .stride(by: 2)], x) == 0)
        }

        @Test func sameSizeBicubicIsTheIdentity() {
            let x = Self.random(batch: 2, channels: 3, height: 5, width: 7)
            #expect(
                SyntheticModel.maxAbsDifference(bicubicInterpolate(x, size: (5, 7)), x)
                    <= Self.tolerance)
        }

        @Test(arguments: [false, true])
        func bicubicKeepsAConstantImage(antialias: Bool) {
            let x = Self.image(batch: 2, channels: 2, height: 6, width: 8) { b, c, _, _ in
                Float(1 + b + 2 * c)
            }
            for size in [(12, 16), (3, 4), (5, 11)] {
                let output = bicubicInterpolate(x, size: size, antialias: antialias)
                #expect(output.shape == [2, 2, size.0, size.1])
                let expected = broadcast(x[0..., 0..., ..<1, ..<1], to: output.shape)
                #expect(
                    SyntheticModel.maxAbsDifference(output, expected) <= Self.tolerance,
                    "size \(size)")
            }
        }

        /// A cubic convolution kernel reproduces linear functions. Away from
        /// the borders, where all 4 taps are inside the image, an upsampled
        /// ramp is the ramp at the mapped coordinate.
        @Test func bicubicUpsamplingKeepsALinearRampLinear() {
            let inW = 8
            let outW = 16
            let x = Self.image(height: 1, width: inW) { _, _, _, x in Float(x) * 0.5 }
            let output = bicubicInterpolate(x, size: (1, outW)).asArray(Float.self)
            for xOut in 0 ..< outW {
                let xIn = (Float(xOut) + 0.5) / Float(outW) * Float(inW) - 0.5
                guard xIn >= 1, xIn <= Float(inW - 2) else { continue }
                #expect(abs(output[xOut] - xIn * 0.5) <= 1e-4, "x \(xOut)")
            }
        }

        @Test func alignCornersKeepsTheCorners() {
            let x = Self.random(batch: 1, channels: 1, height: 4, width: 5)
            let output = bicubicInterpolate(x, size: (7, 9), alignCorners: true)
            for (yOut, yIn) in [(0, 0), (6, 3)] {
                for (xOut, xIn) in [(0, 0), (8, 4)] {
                    #expect(
                        SyntheticModel.maxAbsDifference(
                            output[0, 0, yOut, xOut], x[0, 0, yIn, xIn]) <= Self.tolerance)
                }
            }
        }

        /// Antialiasing widens the filter when the image gets smaller, so the
        /// result comes closer to the mean of each 4 x 4 block than the
        /// result without antialiasing.
        @Test func antialiasDownsamplingComesCloserToTheBlockMean() {
            let x = Self.random(batch: 1, channels: 2, height: 16, width: 16)
            let blockMean = x.reshaped(1, 2, 4, 4, 4, 4).mean(axes: [3, 5])
            let smooth = bicubicInterpolate(x, size: (4, 4), antialias: true)
            let sharp = bicubicInterpolate(x, size: (4, 4), antialias: false)
            let smoothError = SyntheticModel.maxAbsDifference(smooth, blockMean)
            let sharpError = SyntheticModel.maxAbsDifference(sharp, blockMean)
            #expect(smoothError < sharpError, "\(smoothError) against \(sharpError)")
        }

        @Test func bicubicKeepsTheInputDType() {
            let x = Self.random(batch: 1, channels: 3, height: 4, width: 4).asType(.bfloat16)
            let output = bicubicInterpolate(x, scaleFactor: 2)
            #expect(output.dtype == .bfloat16)
            #expect(output.shape == [1, 3, 8, 8])
            #expect(nearestInterpolate(x, scaleFactor: 2).dtype == .bfloat16)
        }
    }
}
