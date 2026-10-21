import Foundation
import MLX
import MLXLMCommon
import Testing

extension KernelTests {
    struct NemotronSuperFP8KVCodecTests {
        //Independent scalar enumeration oracle, no MLX conversions or implementation
        //bit arithmetic. E4M3 positive codes0...126 enumerate every finite magnitude.
        private func finite(_ code: Int) -> Float {
            let exponent = code >> 3
            let mantissa = code & 7
            return exponent == 0
                ? Float(mantissa) / 512
                : (1 + Float(mantissa) / 8) * pow(Float(2), Float(exponent - 7))
        }
        private func encode(_ value: Float) -> UInt8 {
            if value.isNaN { return 127 }
            let sign: UInt8 = value.sign == .minus ? 128 : 0
            if abs(value) >= 448 { return sign | 126 }
            var best = 0
            for code in 1 ... 126 {
                let distance = abs(finite(code) - abs(value))
                let previous = abs(finite(best) - abs(value))
                if distance < previous || (distance == previous && code % 2 == 0) { best = code }
            }
            return sign | UInt8(best)
        }

        @Test func everyBF16EncodingMatchesScalarReference() throws {
            let values = (0 ..< 65536).map { Float(bitPattern: UInt32($0) << 16) }
            let input = MLXArray(values).asType(.bfloat16)
            let actual = try NemotronSuperFP8KVCodec.encode(input).asArray(UInt8.self)
            #expect(actual == values.map(encode))
        }

        @Test func fp32MidpointsNeighborsSaturationAndSignedZero() throws {
            var values: [Float] = [
                0, -0.0, .infinity, -.infinity, .nan, -Float.nan, 464, -464, 465, -465,
            ]
            for code in 0 ..< 126 {
                let midpoint = (finite(code) + finite(code + 1)) / 2
                values += [
                    midpoint.nextDown, midpoint, midpoint.nextUp,
                    -midpoint.nextDown, -midpoint, -midpoint.nextUp,
                ]
            }
            let actual = try NemotronSuperFP8KVCodec.encode(MLXArray(values)).asArray(UInt8.self)
            #expect(actual == values.map(encode))
        }

        @Test func allByteCodesDecodeAndPreserveNaNClass() throws {
            let bytes = MLXArray((0 ... 255).map { UInt8($0) })
            for dtype in [DType.float32, .float16, .bfloat16] {
                let decoded = try NemotronSuperFP8KVCodec.decode(bytes, dtype: dtype)
                #expect(decoded.dtype == dtype)
                let values = decoded.asType(.float32).asArray(Float.self)
                for code in 0 ... 255 {
                    if code & 127 == 127 {
                        #expect(values[code].isNaN)
                    } else {
                        let expected = finite(code & 127) * (code & 128 == 0 ? Float(1) : -1)
                        #expect(values[code].bitPattern == expected.bitPattern)
                    }
                }
            }
        }

        @Test func boundsAndDTypesFailBeforeNativeConversion() throws {
            #expect(throws: (any Error).self) {
                try NemotronSuperFP8KVCodec.encode(MLXArray([Int32(1)]))
            }
            #expect(throws: (any Error).self) {
                try NemotronSuperFP8KVCodec.decode(MLXArray([Float(1)]))
            }
            #expect(throws: (any Error).self) {
                try NemotronSuperFP8KVCodec.decode(MLXArray([UInt8(1)]), dtype: .int32)
            }
            #expect(throws: (any Error).self) {
                try NemotronSuperFP8KVCodec.encode(
                    MLXArray.zeros([NemotronSuperFP8KVCodec.maximumElements + 1]))
            }
        }
    }
}
