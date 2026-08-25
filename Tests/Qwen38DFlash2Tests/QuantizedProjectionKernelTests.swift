import MLX
import Testing

@testable import Qwen38DFlash2

@Suite("Qwen 3.8 verify-width quantized projections", .serialized)
struct Qwen38QuantizedProjectionKernelTests {
    private enum ProductionKernel {
        case m4
        case m56(parts: Int, barrierFree: Bool)
        case m78(simdgroups: Int)
        case m16
    }

    private func assertProductionGeometryParity(
        rows: Int,
        k: Int,
        n: Int,
        kernel: ProductionKernel,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let weight = MLXRandom.normal([n, k]).asType(.bfloat16)
        let (packed, scales, biases) = quantized(
            weight, groupSize: 64, bits: 4, mode: .affine)
        let input = MLXRandom.normal([rows, k]).asType(.bfloat16)
        let expected = quantizedMM(
            input, packed, scales: scales, biases: biases,
            transpose: true, groupSize: 64, bits: 4, mode: .affine)
        let actual: MLXArray
        switch kernel {
        case .m4:
            actual = qwen38M4QuantizedMM(
                input: input, weight: packed, scales: scales,
                biases: biases!, groupSize: 64)
        case .m56(let parts, let barrierFree):
            actual = qwen38M56QuantizedMM(
                input: input, weight: packed, scales: scales,
                biases: biases!, groupSize: 64,
                kParts: parts, barrierFree: barrierFree)
        case .m78(let simdgroups):
            actual = qwen38M78NAXQuantizedMM(
                input: input, weight: packed, scales: scales,
                biases: biases!, groupSize: 64, simdgroups: simdgroups)
        case .m16:
            actual = qwen38M16NAXQuantizedMM(
                input: input, weight: packed, scales: scales,
                biases: biases!, groupSize: 64)
        }
        eval(expected, actual)
        let maximumAbsoluteError = abs(
            actual.asType(.float32) - expected.asType(.float32)
        ).max().item(Float.self)
        #expect(
            actual.shape == expected.shape
                && maximumAbsoluteError <= 0.25,
            "M=\(rows) K=\(k) N=\(n) max_abs=\(maximumAbsoluteError)",
            sourceLocation: sourceLocation)
        Memory.clearCache()
    }

    @Test("source-final M4 compile-time-K split matches affine Q4/G32 stock")
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

    @Test("production M4 split geometries match affine Q4/G64 stock")
    func productionM4NumericParity() {
        MLXRandom.seed(38_404)
        assertProductionGeometryParity(rows: 4, k: 5_120, n: 17_408, kernel: .m4)
        assertProductionGeometryParity(rows: 4, k: 5_120, n: 1_024, kernel: .m4)
    }

    @Test("production M5 and M6 geometries match affine Q4/G64 stock")
    func productionM5M6NumericParity() {
        MLXRandom.seed(38_506)
        assertProductionGeometryParity(
            rows: 5, k: 6_144, n: 5_120,
            kernel: .m56(parts: 2, barrierFree: false))
        assertProductionGeometryParity(
            rows: 6, k: 5_120, n: 10_240,
            kernel: .m56(parts: 1, barrierFree: true))
        assertProductionGeometryParity(
            rows: 6, k: 5_120, n: 17_408,
            kernel: .m56(parts: 1, barrierFree: true))
        assertProductionGeometryParity(
            rows: 6, k: 17_408, n: 5_120,
            kernel: .m56(parts: 2, barrierFree: false))
    }

    @Test("production M7, M8, and padded-M16 geometries match affine Q4/G64 stock")
    func productionM7M8M16NumericParity() {
        MLXRandom.seed(38_816)
        assertProductionGeometryParity(
            rows: 7, k: 6_144, n: 5_120, kernel: .m78(simdgroups: 4))
        assertProductionGeometryParity(
            rows: 7, k: 5_120, n: 6_144, kernel: .m78(simdgroups: 4))
        assertProductionGeometryParity(
            rows: 8, k: 5_120, n: 1_024, kernel: .m78(simdgroups: 4))
        assertProductionGeometryParity(
            rows: 8, k: 5_120, n: 10_240, kernel: .m78(simdgroups: 4))
        assertProductionGeometryParity(
            rows: 8, k: 5_120, n: 17_408, kernel: .m78(simdgroups: 4))
        assertProductionGeometryParity(
            rows: 8, k: 5_120, n: 5_120, kernel: .m16)
    }

}
