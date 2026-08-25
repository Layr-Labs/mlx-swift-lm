import MLX
import Testing

@testable import Qwen38DFlash2

@Suite("Qwen 3.8 verify-width quantized projections")
struct Qwen38QuantizedProjectionKernelTests {
    @Test("exact M4 split-K matches affine Q4/G32 stock")
    func m4NumericParity() {
        MLXRandom.seed(38_004)
        let weight = MLXRandom.normal([32, 256]).asType(.bfloat16)
        let (packed, scales, biases) = quantized(
            weight, groupSize: 32, bits: 4, mode: .affine)
        let input = MLXRandom.normal([4, 256]).asType(.bfloat16)
        let expected = quantizedMM(
            input, packed, scales: scales, biases: biases,
            transpose: true, groupSize: 32, bits: 4, mode: .affine)
        let actual = qwen38M4QuantizedMM(
            input: input,
            weight: packed,
            scales: scales,
            biases: biases!,
            groupSize: 32)
        eval(expected, actual)
        #expect(actual.shape == expected.shape)
        #expect(
            abs(actual.asType(.float32) - expected.asType(.float32))
                .max().item(Float.self) < 0.25)
    }

    @Test("exact M5 and barrier-free M6 match affine Q4/G32 stock")
    func m5M6NumericParity() {
        MLXRandom.seed(38_056)
        let weight = MLXRandom.normal([12, 64]).asType(.bfloat16)
        let (packed, scales, biases) = quantized(
            weight, groupSize: 32, bits: 4, mode: .affine)

        for (rows, parts, direct) in [(5, 2, false), (6, 1, true)] {
            let input = MLXRandom.normal([rows, 64]).asType(.bfloat16)
            let expected = quantizedMM(
                input, packed, scales: scales, biases: biases,
                transpose: true, groupSize: 32, bits: 4, mode: .affine)
            let actual = qwen38M56QuantizedMM(
                input: input,
                weight: packed,
                scales: scales,
                biases: biases!,
                groupSize: 32,
                kParts: parts,
                barrierFree: direct)
            eval(expected, actual)
            #expect(actual.shape == expected.shape)
            #expect(
                abs(actual.asType(.float32) - expected.asType(.float32))
                    .max().item(Float.self) < 0.25)
        }
    }

    @Test("M7-padded and M8 Metal 4 tensor projections match affine Q4/G32 stock")
    func m7M8NumericParity() {
        MLXRandom.seed(38_078)
        let weight = MLXRandom.normal([32, 256]).asType(.bfloat16)
        let (packed, scales, biases) = quantized(
            weight, groupSize: 32, bits: 4, mode: .affine)

        for rows in [7, 8] {
            let input = MLXRandom.normal([rows, 256]).asType(.bfloat16)
            let expected = quantizedMM(
                input, packed, scales: scales, biases: biases,
                transpose: true, groupSize: 32, bits: 4, mode: .affine)
            let actual = qwen38M78NAXQuantizedMM(
                input: input,
                weight: packed,
                scales: scales,
                biases: biases!,
                groupSize: 32,
                simdgroups: 8)
            eval(expected, actual)
            #expect(actual.shape == expected.shape)
            #expect(
                abs(actual.asType(.float32) - expected.asType(.float32))
                    .max().item(Float.self) <= 0.25)
        }
    }

    @Test("M16-padded fallback matches affine Q4/G32 stock")
    func m16PaddedNumericParity() {
        MLXRandom.seed(38_016)
        let weight = MLXRandom.normal([32, 256]).asType(.bfloat16)
        let (packed, scales, biases) = quantized(
            weight, groupSize: 32, bits: 4, mode: .affine)

        for rows in [7, 8] {
            let input = MLXRandom.normal([rows, 256]).asType(.bfloat16)
            let expected = quantizedMM(
                input, packed, scales: scales, biases: biases,
                transpose: true, groupSize: 32, bits: 4, mode: .affine)
            let actual = qwen38M16NAXQuantizedMM(
                input: input,
                weight: packed,
                scales: scales,
                biases: biases!,
                groupSize: 32)
            eval(expected, actual)
            #expect(actual.shape == expected.shape)
            #expect(
                abs(actual.asType(.float32) - expected.asType(.float32))
                    .max().item(Float.self) <= 0.25)
        }
    }
}
