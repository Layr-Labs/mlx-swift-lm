import Foundation
import MLX
import MLXLMCommon
import Testing

extension KernelTests {
    struct ModelOptFP8CalibratedMatmulTests {
        private func decode(_ byte: UInt8) -> Float {
            let code = Int(byte & 127)
            let exponent = code >> 3
            let mantissa = code & 7
            if code == 127 { return .nan }
            let magnitude: Float =
                exponent == 0
                ? Float(mantissa) / 512
                : (1 + Float(mantissa) / 8) * pow(Float(2), Float(exponent - 7))
            return byte & 128 == 0 ? magnitude : -magnitude
        }

        private func castE4M3(_ x: Float) -> Float {
            // Independent scalar round-to-nearest-even, including subnormals.
            // ModelOpt's pinned eager PyTorch cast does not clamp overflow.
            guard x.isFinite, abs(x) <= 464 else { return .nan }
            var best = 0
            for code in 0 ... 126 {
                let distance = abs(decode(UInt8(code)) - abs(x))
                let previous = abs(decode(UInt8(best)) - abs(x))
                if distance < previous || (distance == previous && code % 2 == 0) { best = code }
            }
            let value = decode(UInt8(best))
            return x.sign == .minus ? -value : value
        }

        private func bf16(_ value: Float) -> Float {
            if value.isNaN { return .nan }
            let bits = value.bitPattern
            return Float(bitPattern: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) & 0xffff_0000)
        }

        private func oracle(
            _ x: [Float], width: Int, weights: [UInt8],
            inputScale: Float, weightScale: Float, useBF16: Bool = false
        ) -> [Float] {
            let scale: Float = 448 / (inputScale * 448)
            let inverse: Float = 1 / scale
            let activations = x.map { value -> Float in
                let decoded = castE4M3(value * scale) * inverse
                return useBF16 ? bf16(decoded) : decoded
            }
            let decoded = weights.map { code -> Float in
                let value = decode(code) * weightScale
                return useBF16 ? bf16(value) : value
            }
            return stride(from: 0, to: x.count, by: width).flatMap { start in
                stride(from: 0, to: weights.count, by: width).map { row -> Float in
                    let dot = (0 ..< width).reduce(Float(0)) {
                        $0 + activations[start + $1] * decoded[row + $1]
                    }
                    return useBF16 ? bf16(dot) : dot
                }
            }
        }

        private func identity() -> [UInt8] {
            (0 ..< 256).map { index in index / 16 == index % 16 ? 56 : 0 }
        }

        @Test func tiesSubnormalsAndDistinctScalesMatchIndependentOracle() throws {
            let values: [Float] = [
                0, -0.0, 1.0 / 1024, -1.0 / 1024, 3.0 / 1024, -3.0 / 1024,
                1.0625, -1.0625, 1.1875, -1.1875, 432, -432, 448, -448, 464, -464,
            ]
            let input = values.map { $0 * 2 }
            let bytes = identity()
            let weights = try ModelOptFP8Weights(
                e4m3Bytes: MLXArray(bytes).reshaped(16, 16), tensorScale: 0.25)
            let output = try weights.calibratedMatmul(
                MLXArray(input).reshaped(1, 16), inputScale: 2, weightDType: .float32,
                outputRows: 0 ..< 16)
            let expected = oracle(
                input, width: 16, weights: bytes, inputScale: 2, weightScale: 0.25)
            #expect(output.shape == [1, 16] && output.dtype == .float32)
            for (actual, reference) in zip(output.asArray(Float.self), expected) {
                #expect(actual == reference)
            }
        }

        @Test func overflowAndNaNPropagateInsteadOfSaturating() throws {
            let weights = try ModelOptFP8Weights(
                e4m3Bytes: MLXArray(identity()).reshaped(16, 16), tensorScale: 1)
            for invalid in [Float(464).nextUp, -Float(464).nextUp, .infinity, -.infinity, .nan] {
                let values = [invalid] + Array(repeating: Float(1), count: 15)
                let output = try weights.calibratedMatmul(
                    MLXArray(values).reshaped(1, 16), inputScale: 1, weightDType: .float32,
                    outputRows: 0 ..< 16)
                // IEEE NaN times even a zero weight contaminates every output.
                #expect(output.asArray(Float.self).allSatisfy { $0.isNaN })
            }
        }

        @Test func fp32ScaleMultiplyPrecedesBfloat16CastAndRowsRemainBounded() throws {
            let bytes: [UInt8] = [
                56, 57, 58, 59, 60, 61, 62, 63,
                184, 185, 186, 187, 188, 189, 190, 191,
                64, 65, 66, 67, 68, 69, 70, 71,
            ]
            let values: [Float] = [0.5, 0.75, 1, 1.25, -0.5, -0.75, 1.5, 2]
            let weights = try ModelOptFP8Weights(
                e4m3Bytes: MLXArray(bytes).reshaped(3, 8), tensorScale: 0.0031)
            let input = MLXArray(values).reshaped(1, 8).asType(.bfloat16)
            let output = try weights.calibratedMatmul(
                input, inputScale: 0.1, weightDType: .bfloat16, outputRows: 1 ..< 3)
            let expected = oracle(
                values, width: 8, weights: Array(bytes[8 ..< 24]),
                inputScale: 0.1, weightScale: 0.0031, useBF16: true)
            #expect(output.shape == [1, 2] && output.dtype == .bfloat16)
            for (actual, reference) in zip(output.asArray(Float.self), expected) {
                // One output BF16 ULP allows backend accumulation-order rounding.
                #expect(abs(actual - reference) <= max(abs(reference) / 128, 1e-7))
            }
            // A source-format NaN weight must survive the FP32 multiply/cast.
            var nanBytes = bytes
            nanBytes[8] = 127
            let nanWeights = try ModelOptFP8Weights(
                e4m3Bytes: MLXArray(nanBytes).reshaped(3, 8), tensorScale: 0.0031)
            let nanOutput = try nanWeights.calibratedMatmul(
                input, inputScale: 0.1, weightDType: .bfloat16, outputRows: 1 ..< 2)
            #expect(nanOutput.asArray(Float.self)[0].isNaN)
        }

        @Test func originalWeightDtypeIsExplicitAndMixedDtypesAreRejected() throws {
            let weights = try ModelOptFP8Weights(
                e4m3Bytes: MLXArray(identity()).reshaped(16, 16), tensorScale: 0.0031)
            for (activation, originalWeight) in [
                (DType.float32, DType.bfloat16),
                (.bfloat16, .float32), (.float16, .bfloat16),
                (.uint8, .uint8), (.float32, .uint8),
            ] {
                #expect(throws: ModelOptFP8Weights.ValidationError.self) {
                    try weights.calibratedMatmul(
                        MLXArray.zeros([1, 16], dtype: activation),
                        inputScale: 0.1, weightDType: originalWeight, outputRows: 0 ..< 1)
                }
            }
        }

        @Test func invalidShapesScalesAndWorkspaceAreRejected() throws {
            let weights = try ModelOptFP8Weights(
                e4m3Bytes: MLXArray(identity()).reshaped(16, 16), tensorScale: 1)
            for shape in [[16], [1, 15], [0, 16], [1, 1, 16]] {
                #expect(throws: ModelOptFP8Weights.ValidationError.self) {
                    try weights.calibratedMatmul(
                        MLXArray.zeros(shape), inputScale: 1, weightDType: .float32,
                        outputRows: 0 ..< 1)
                }
            }
            for scale in [
                Float(0), -1, .nan, .infinity, .greatestFiniteMagnitude, .leastNonzeroMagnitude,
            ] {
                #expect(throws: ModelOptFP8Weights.ValidationError.self) {
                    try weights.calibratedMatmul(
                        MLXArray.zeros([1, 16]), inputScale: scale, weightDType: .float32,
                        outputRows: 0 ..< 1)
                }
            }
            #expect(throws: ModelOptFP8Weights.ValidationError.self) {
                try weights.calibratedMatmul(
                    MLXArray.zeros([1, 16], dtype: .uint8), inputScale: 1, weightDType: .float32,
                    outputRows: 0 ..< 1)
            }
            for rows in [0 ..< 0, -1 ..< 1, 0 ..< 17] {
                #expect(throws: ModelOptFP8Weights.ValidationError.self) {
                    try weights.calibratedMatmul(
                        MLXArray.zeros([1, 16]), inputScale: 1, weightDType: .float32,
                        outputRows: rows)
                }
            }
            let large = try ModelOptFP8Weights(
                e4m3Bytes: MLXArray.zeros([1025, 2048], dtype: .uint8), tensorScale: 1)
            #expect(throws: ModelOptFP8Weights.ValidationError.self) {
                try large.calibratedMatmul(
                    MLXArray.zeros([1, 2048]), inputScale: 1, weightDType: .float32,
                    outputRows: 0 ..< 1025)
            }
        }
    }
}
