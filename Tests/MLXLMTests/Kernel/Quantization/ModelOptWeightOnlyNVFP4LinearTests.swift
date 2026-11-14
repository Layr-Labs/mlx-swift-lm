// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

extension KernelTests {
    /// The weight-only NVFP4 projection against a host decode written from the
    /// producer's documented formula: `e2m1[code] * e4m3(block scale) * weight_scale_2`,
    /// element `2i` in the low nibble of byte `i` (NVIDIA ModelOpt `NVFP4QTensor`).
    struct ModelOptWeightOnlyNVFP4LinearTests {
        static let e2m1: [Float] = [
            0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6,
        ]

        /// E4M3FN: 1 sign, 4 exponent (bias 7), 3 mantissa bits; no infinities.
        static func e4m3(_ byte: UInt8) -> Float {
            let sign: Float = byte & 0x80 == 0 ? 1 : -1
            let exponent = Int(byte >> 3) & 0xF
            let mantissa = Float(byte & 0x7)
            if byte & 0x7F == 0x7F { return .nan }
            if exponent == 0 {
                return sign * mantissa / 8 * Float(sign: .plus, exponent: -6, significand: 1)
            }
            return sign * (1 + mantissa / 8)
                * Float(sign: .plus, exponent: exponent - 7, significand: 1)
        }

        static func hostWeight(
            packed: [UInt8], scales: [UInt8], tensorScale: Float, rows: Int, columns: Int
        ) -> [Float] {
            var dense = [Float](repeating: 0, count: rows * columns)
            for row in 0 ..< rows {
                for column in 0 ..< columns {
                    let byte = packed[row * (columns / 2) + column / 2]
                    let code = column % 2 == 0 ? byte & 0xF : byte >> 4
                    let block = e4m3(scales[row * (columns / 16) + column / 16])
                    dense[row * columns + column] = e2m1[Int(code)] * block * tensorScale
                }
            }
            return dense
        }

        private func randomLayer(
            rows: Int, columns: Int, tensorScale: Float, seed: UInt64
        ) throws -> (ModelOptWeightOnlyNVFP4Linear, [Float]) {
            var generator = SplitMix64(seed: seed)
            let packed = (0 ..< rows * columns / 2).map { _ in
                UInt8(truncatingIfNeeded: generator.next())
            }
            // Positive finite block scales, as the producer clamps them to [2^-9, 448].
            let scales = (0 ..< rows * columns / 16).map { _ in
                UInt8(truncatingIfNeeded: generator.next() % 0x7E) + 1
            }
            let layer = try ModelOptWeightOnlyNVFP4Linear(
                packedBytes: MLXArray(packed).reshaped(rows, columns / 2),
                e4m3ScaleBytes: MLXArray(scales).reshaped(rows, columns / 16),
                tensorScale: tensorScale)
            return (
                layer,
                Self.hostWeight(
                    packed: packed, scales: scales, tensorScale: tensorScale, rows: rows,
                    columns: columns)
            )
        }

        @Test func nibbleOrderAndCodeTableMatchTheProducer() throws {
            // One row of 16 inputs holding codes 0...15 in element order.
            let packed = (0 ..< 8).map { UInt8(($0 * 2 + 1) << 4 | ($0 * 2)) }
            let layer = try ModelOptWeightOnlyNVFP4Linear(
                packedBytes: MLXArray(packed).reshaped(1, 8),
                e4m3ScaleBytes: MLXArray([UInt8(0x40)]).reshaped(1, 1),  // 2.0
                tensorScale: 0.25)
            let output = layer(MLXArray.eye(16, dtype: .float32)).asArray(Float.self)
            #expect(output == Self.e2m1.map { $0 * 2 * 0.25 })
            #expect(Self.e4m3(0x38) == 1 && Self.e4m3(0x40) == 2 && Self.e4m3(0x7E) == 448)
            #expect(Self.e4m3(0x01) == Float(sign: .plus, exponent: -9, significand: 1))
        }

        @Test func dequantizedWeightEqualsTheHostDecode() throws {
            let (layer, host) = try randomLayer(rows: 12, columns: 64, tensorScale: 3.1e-4, seed: 1)
            let dense = layer.dequantizedWeight(dtype: .float32)
            #expect(dense.shape == [12, 64])
            // FP32 products of two- and four-bit significands with one FP32
            // scalar: the two orders of multiplication differ by at most 1 ulp.
            for (value, expected) in zip(dense.asArray(Float.self), host) {
                #expect(abs(value - expected) <= abs(expected) * 2.4e-7)
            }
        }

        @Test func productsMatchTheDenseHostMatrix() throws {
            let rows = 24
            let columns = 96
            let (layer, host) = try randomLayer(
                rows: rows, columns: columns, tensorScale: 7.5e-5, seed: 2)
            let dense = MLXArray(host).reshaped(rows, columns)
            MLXRandom.seed(3)
            for shape in [[1, columns], [5, columns], [2, 7, columns]] {
                let x = MLXRandom.normal(shape)
                let expected = matmul(x, dense.T)
                let output = layer(x)
                #expect(output.shape == expected.shape && output.dtype == .float32)
                // FP32 accumulation in another order over 96 terms.
                let scale = abs(expected).max().item(Float.self)
                #expect(abs(output - expected).max().item(Float.self) <= scale * 2e-5)

                // BF16 activations: the kernel rounds its FP32 sum to BF16 once
                // and the FP32 tensor-scale multiply rounds once more.
                let narrow = layer(x.asType(.bfloat16))
                #expect(narrow.dtype == .bfloat16)
                let reference = matmul(x.asType(.bfloat16).asType(.float32), dense.T)
                #expect(
                    abs(narrow.asType(.float32) - reference).max().item(Float.self)
                        <= scale * 0.012)
            }
        }

        @Test func scaleIsAppliedVerbatimInFP32() throws {
            let (layer, _) = try randomLayer(rows: 8, columns: 32, tensorScale: 1.7e-3, seed: 4)
            MLXRandom.seed(5)
            let x = MLXRandom.normal([3, 32]).asType(.bfloat16)
            let unscaled = layer.unscaledProduct(x)
            let expected = (unscaled.asType(.float32) * MLXArray(Float(1.7e-3))).asType(.bfloat16)
            #expect(arrayEqual(layer(x), expected).item(Bool.self))
            #expect(layer.weightScale2.asArray(Float.self) == [1.7e-3])
        }

        @Test func refusesInvalidGeometryAndScales() throws {
            let packed = MLXArray.zeros([4, 8], dtype: .uint8)
            let scales = MLXArray.zeros([4, 1], dtype: .uint8)
            for scale in [0, -1, Float.nan, Float.infinity, Float.greatestFiniteMagnitude] {
                #expect(throws: ModelOptWeightOnlyNVFP4Linear.Failure.invalidScale) {
                    try ModelOptWeightOnlyNVFP4Linear(
                        packedBytes: packed, e4m3ScaleBytes: scales, tensorScale: scale)
                }
            }
            #expect(throws: ModelOptWeightOnlyNVFP4Linear.Failure.invalidLayout) {
                try ModelOptWeightOnlyNVFP4Linear(
                    packedBytes: packed, e4m3ScaleBytes: MLXArray.zeros([4, 2], dtype: .uint8),
                    tensorScale: 1)
            }
            #expect(throws: ModelOptWeightOnlyNVFP4Linear.Failure.invalidLayout) {
                try ModelOptWeightOnlyNVFP4Linear(
                    packedBytes: packed.asType(.uint32), e4m3ScaleBytes: scales, tensorScale: 1)
            }
            #expect(throws: ModelOptWeightOnlyNVFP4Linear.Failure.invalidLayout) {
                try ModelOptWeightOnlyNVFP4Linear(
                    packedBytes: MLXArray.zeros([4, 12], dtype: .uint8),
                    e4m3ScaleBytes: scales, tensorScale: 1)
            }
            let layer = try ModelOptWeightOnlyNVFP4Linear(
                packedBytes: packed, e4m3ScaleBytes: scales, tensorScale: 1)
            #expect(
                Set(layer.parameters().flattened().map(\.0)) == [
                    "weight", "weight_scale", "weight_scale_2",
                ])
            #expect(layer.trainableParameters().flattened().isEmpty)
            #expect(throws: (any Error).self) {
                try layer.update(
                    parameters: .unflattened(["weight_scale_2": MLXArray([Float(0)])]),
                    verify: [])
            }
        }
    }
}

/// Small deterministic generator for byte fixtures.
private struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
