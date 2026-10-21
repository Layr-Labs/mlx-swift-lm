import Foundation
import MLX
import MLXLMCommon
import Testing

extension KernelTests {
    struct ModelOptFP8WeightsTests {
        @Test func everyE4M3CodeAndDirectScaleMatchesScalarOracle() throws {
            let weights = try ModelOptFP8Weights(
                e4m3Bytes: MLXArray((0 ... 255).map(UInt8.init)).reshaped(16, 16), tensorScale: 0.25
            )
            let actual = try weights.decodedRows(0 ..< 16, dtype: .float32).asArray(Float.self)
            for bits in 0 ... 255 {
                let code = bits & 127
                let exponent = code >> 3
                let mantissa = code & 7
                if code == 127 {
                    #expect(actual[bits].isNaN)
                    continue
                }
                let magnitude: Float =
                    exponent == 0
                    ? Float(mantissa) / 512
                    : (1 + Float(mantissa) / 8) * pow(Float(2), Float(exponent - 7))
                let expected = (bits >= 128 ? -magnitude : magnitude) * 0.25
                #expect(actual[bits] == expected)
                if magnitude == 0 { #expect(actual[bits].sign == expected.sign) }
            }
            let subset = try weights.decodedRows(2 ..< 4, dtype: .bfloat16)
            #expect(subset.shape == [2, 16] && subset.dtype == .bfloat16)
            #expect(subset.asArray(Float.self) == Array(actual[32 ..< 64]))
        }

        @Test func malformedMetadataAndUnboundedDecodeAreRejected() throws {
            #expect(throws: ModelOptFP8Weights.ValidationError.self) {
                try ModelOptFP8Weights(e4m3Bytes: MLXArray.zeros([2, 16]), tensorScale: 1)
            }
            for scale in [Float(0), -1, .nan, .infinity] {
                #expect(throws: ModelOptFP8Weights.ValidationError.self) {
                    try ModelOptFP8Weights(
                        e4m3Bytes: MLXArray.zeros([2, 16], dtype: .uint8), tensorScale: scale)
                }
            }
            let weights = try ModelOptFP8Weights(
                e4m3Bytes: MLXArray.zeros([2, 16], dtype: .uint8), tensorScale: 1)
            for rows in [0 ..< 0, -1 ..< 1, 0 ..< 3] {
                #expect(throws: ModelOptFP8Weights.ValidationError.self) {
                    try weights.decodedRows(rows)
                }
            }
            #expect(throws: ModelOptFP8Weights.ValidationError.self) {
                try weights.decodedRows(0 ..< 1, dtype: .uint8)
            }
            let large = try ModelOptFP8Weights(
                e4m3Bytes: MLXArray.zeros([4097, 4096], dtype: .uint8), tensorScale: 1)
            #expect(throws: ModelOptFP8Weights.ValidationError.self) {
                try large.decodedRows(0 ..< 4097)
            }
        }
    }
}
