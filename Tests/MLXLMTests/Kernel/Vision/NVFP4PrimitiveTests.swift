import MLX
import Testing

extension KernelTests {
    /// Native primitive evidence only; this does not validate ModelOpt checkpoint conversion.
    struct NVFP4PrimitiveTests {
        @Test func allNibblesAndBothScales() {
            let packed = MLXArray([UInt32(0x7654_3210), UInt32(0xfedc_ba98)]).reshaped(1, 2)
            // E4M3 byte 0x40 = 2.0. MLX's globalScale represents amax,
            // with inverse encoding scale amax/(448*6), not a direct multiplier.
            let scales = MLXArray([UInt8(0x40)]).reshaped(1, 1)
            let decoded = dequantized(
                packed, scales: scales, biases: nil, groupSize: 16, bits: 4, mode: .nvfp4,
                globalScale: MLXArray(Float(448 * 6 * 3)), dtype: .float32)
            let e2m1: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6]
            #expect(decoded.asArray(Float.self) == e2m1.map { $0 * 2 * 3 })
        }
    }
}
