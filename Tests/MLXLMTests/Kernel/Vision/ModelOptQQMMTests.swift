import Foundation
import MLX
import MLXLMCommon
import Testing

extension KernelTests {
    /// Independent scalar oracle for ModelOpt 0.37 NVFP4QTensor.quantize/_cast_fp4,
    /// pinned in evidence/modelopt-nvfp4-0.37.0.py. No MLX quantization in the oracle.
    struct ModelOptQQMMTests {
        private let fp4: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]

        private func nearestEven(_ x: Float, values: [Float]) -> Float {
            var best = 0
            for i in values.indices {
                let distance = abs(values[i] - x)
                let previous = abs(values[best] - x)
                if distance < previous || (distance == previous && i % 2 == 0) { best = i }
            }
            return values[best]
        }

        private func e4m3(_ x: Float) -> Float {
            // torch.float8_e4m3fn conversion: finite range ends at 448; the
            // round-to-nearest-even overflow threshold is 464 (tie stays 448).
            if x > 464 || !x.isFinite { return .nan }
            let values: [Float] = (0 ... 126).map { bits in
                let exponent = bits >> 3
                let mantissa = bits & 7
                return exponent == 0
                    ? Float(mantissa) / 512
                    : Float(1 + Float(mantissa) / 8) * pow(Float(2), Float(exponent - 7))
            }
            return nearestEven(x, values: values)
        }

        private func reference(_ x: [Float], inputScale: Float) -> [Float] {
            precondition(x.count % 16 == 0)
            return stride(from: 0, to: x.count, by: 16).flatMap { start -> [Float] in
                let block = Array(x[start ..< (start + 16)])
                let maximum = block.map(abs).max()!
                let rawScale = maximum / (6 * inputScale)
                // ModelOpt replaces an exactly zero block scale with one before E4M3.
                let scale = e4m3(rawScale == 0 ? 1 : rawScale)
                return block.map { value in
                    let normalized = abs(value / (scale * inputScale))
                    guard scale.isFinite else { return Float.nan }
                    let rounded = nearestEven(normalized, values: fp4)
                    return (value < 0 ? -rounded : rounded) * scale * inputScale
                }
            }
        }

        private func diagonalWeights(scale: Float) throws -> ModelOptNVFP4Weights {
            var bytes = [UInt8](repeating: 0, count: 16 * 8)
            // E2M1 code 2 = 1. Each output selects one activation coordinate.
            for row in 0 ..< 16 { bytes[row * 8 + row / 2] = row % 2 == 0 ? 2 : 32 }
            return try ModelOptNVFP4Weights(
                packedBytes: MLXArray(bytes).reshaped(16, 8),
                e4m3ScaleBytes: MLXArray([UInt8](repeating: 56, count: 16)).reshaped(16, 1),
                tensorScale: scale)  // E4M3 code 56 = 1.
        }

        @Test func fp4TiesAndSeparateCalibratedScales() throws {
            let row: [Float] = [
                0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6,
                0, 0.5, 1, 1.5, 2, 3, 4, 6,
            ]
            let x = (row + row.map { -$0 }).map { $0 * 2 }
            let weights = try diagonalWeights(scale: 0.5)
            let result = try weights.calibratedMatmul(MLXArray(x).reshaped(2, 16), inputScale: 2)
            let expected = reference(x, inputScale: 2).map { $0 * 0.5 }
            #expect(result.shape == [2, 16])
            #expect(result.dtype == .float32)
            for (actual, wanted) in zip(result.asArray(Float.self), expected) {
                #expect(abs(actual - wanted) < 1e-5)
            }
        }

        @Test func e4m3ScaleTiesAndFiniteFp4Saturation() throws {
            // Max/6 = 1.0625 and 1.1875 hit opposite E4M3 rounding ties.
            // Rounding 1.0625 down to 1 makes 6.375 exceed FP4's finite max.
            let x: [Float] =
                [6.375] + Array(repeating: 0.75, count: 15)
                + [7.125] + Array(repeating: -1.75, count: 15)
            let weights = try diagonalWeights(scale: 0.25)
            let result = try weights.calibratedMatmul(MLXArray(x).reshaped(2, 16), inputScale: 1)
            let expected = reference(x, inputScale: 1).map { $0 * 0.25 }
            for (actual, wanted) in zip(result.asArray(Float.self), expected) {
                #expect(abs(actual - wanted) < 1e-5)
            }
        }

        @Test func scaleOverflowMatchesModelOptRatherThanSilentlySaturating() throws {
            let x = Array(repeating: Float(4096), count: 32)
            #expect(reference(x, inputScale: 1).allSatisfy { $0.isNaN })
            let weights = try diagonalWeights(scale: 1)
            let result = try weights.calibratedMatmul(MLXArray(x).reshaped(2, 16), inputScale: 1)
            // The adapter must restore ModelOpt NaN semantics; native QQMM alone
            // failed this conformance gate in the recorded initial run.
            #expect(result.asArray(Float.self).allSatisfy { $0.isNaN })
        }

        @Test func invalidInputMetadataIsRejectedBeforeExecution() throws {
            let weights = try diagonalWeights(scale: 1)
            for shape in [[16], [1, 15], [0, 16], [1, 1, 16]] {
                #expect(throws: ModelOptNVFP4Weights.ValidationError.self) {
                    try weights.calibratedMatmul(MLXArray.zeros(shape), inputScale: 1)
                }
            }
            #expect(throws: ModelOptNVFP4Weights.ValidationError.self) {
                try weights.calibratedMatmul(MLXArray.zeros([1, 16], dtype: .uint8), inputScale: 1)
            }
            for scale in [Float(0), -1, .nan, .infinity, Float.greatestFiniteMagnitude, 1e-40] {
                #expect(throws: ModelOptNVFP4Weights.ValidationError.self) {
                    try weights.calibratedMatmul(MLXArray.zeros([1, 16]), inputScale: scale)
                }
            }
        }

        @Test func overflowBoundaryZeroAndNonfiniteRows() throws {
            let weights = try diagonalWeights(scale: 1)
            let maxima: [Float] = [464 * 6, (464 * 6).nextUp, 0, .infinity, -.infinity, .nan]
            let x = maxima.flatMap { value in [value] + Array(repeating: Float(0), count: 15) }
            let result = try weights.calibratedMatmul(MLXArray(x).reshaped(6, 16), inputScale: 1)
                .asArray(Float.self)
            #expect(result[0] == 448 * 6)
            #expect(result[1 ..< 16].allSatisfy { $0 == 0 })
            #expect(result[16 ..< 32].allSatisfy { $0.isNaN })
            #expect(result[32 ..< 48].allSatisfy { $0 == 0 })
            #expect(result[48 ..< 96].allSatisfy { $0.isNaN })
        }
    }
}
