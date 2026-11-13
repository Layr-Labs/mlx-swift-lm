// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

/// Synthetic weight-only NVFP4 tensors for the row-exact product. Not
/// checkpoint data.
struct WeightOnlyNVFP4Fixture {
    let outputs: Int
    let inputs: Int
    /// uint32 `[outputs, inputs / 8]`.
    let weight: MLXArray
    /// uint8 `[outputs, inputs / 16]`.
    let blockScales: MLXArray

    /// Uniform E2M1 codes; E4M3 scales between 0.25 (0x28) and 4 (0x48).
    init(outputs: Int, inputs: Int, seed: UInt64) {
        self.outputs = outputs
        self.inputs = inputs
        weight = MLXRandom.randInt(
            Int32(0) ..< 256, [outputs, inputs / 2], key: MLXRandom.key(seed)
        ).asType(.uint8).view(dtype: .uint32)
        blockScales = MLXRandom.randInt(
            Int32(0x28) ..< 0x49, [outputs, inputs / 16], key: MLXRandom.key(seed + 1)
        ).asType(.uint8)
        eval(weight, blockScales)
    }

    func activations(rows: Int, dtype: DType, seed: UInt64 = 11) -> MLXArray {
        let x = (MLXRandom.normal([rows, inputs], key: MLXRandom.key(seed)) * 0.5).asType(dtype)
        eval(x)
        return x
    }

    /// The product with the scalar kernel taking `scalarRows` rows (nil: the
    /// device's default).
    func product(_ x: MLXArray, scalarRows: Int? = nil, tensorScale: MLXArray? = nil) -> MLXArray {
        let saved = ModelOptWeightOnlyNVFP4RowExact.scalarRows
        defer { ModelOptWeightOnlyNVFP4RowExact.scalarRows = saved }
        if let scalarRows { ModelOptWeightOnlyNVFP4RowExact.scalarRows = scalarRows }
        let result = ModelOptWeightOnlyNVFP4RowExact.product(
            x, weight: weight, blockScales: blockScales, tensorScale: tensorScale)
        eval(result)
        return result
    }
}

/// The raw 16-bit words of each row of `[R, ...]`.
func weightOnlyNVFP4RowWords(_ array: MLXArray) -> [[UInt16]] {
    let rows = array.dim(0)
    let flat = array.reshaped(rows, -1).view(dtype: .uint16).asArray(UInt16.self)
    let width = flat.count / rows
    return (0 ..< rows).map { Array(flat[($0 * width) ..< (($0 + 1) * width)]) }
}

extension KernelTests {
    @Suite(.serialized)
    struct ModelOptWeightOnlyNVFP4RowExactTests {
        static let rowCounts = [1, 2, 3, 8, 9, 16]

        /// `(outputs, inputs)`: tiny matrices, outputs off the 32-row tile,
        /// inputs off the 32-input block and the 128-input chunk, and more
        /// than one chunk per partition.
        static let shapes = [
            (1, 16), (3, 16), (7, 48), (8, 112), (9, 128), (31, 144), (33, 272), (250, 1040),
            (64, 2064), (40, 4096),
        ]

        /// Every row of an R-row product has the bytes of its one-row product,
        /// from the matrix kernel alone and, where the scalar kernel is
        /// measured, from every split between the two kernels.
        @Test(arguments: shapes.indices)
        func rowsHaveTheBytesOfOneRowProducts(index: Int) {
            let (outputs, inputs) = Self.shapes[index]
            let fixture = WeightOnlyNVFP4Fixture(
                outputs: outputs, inputs: inputs, seed: UInt64(100 + index))
            // Every shape in bfloat16; the other dtypes on two of them.
            for dtype in [DType.bfloat16, .float32, .float16]
            where dtype == .bfloat16 || [(7, 48), (250, 1040)].contains(where: {
                $0 == (outputs, inputs)
            }) {
                let x = fixture.activations(rows: 16, dtype: dtype)
                let alone = (0 ..< 16).map {
                    weightOnlyNVFP4RowWords(fixture.product(x[$0 ..< $0 + 1], scalarRows: 0))[0]
                }
                var splits = [0]
                if ModelOptWeightOnlyNVFP4RowExact.scalarKernelMeasured { splits += [1, 2, 4] }
                for scalarRows in splits {
                    for rows in Self.rowCounts {
                        let together = weightOnlyNVFP4RowWords(
                            fixture.product(x[0 ..< rows], scalarRows: scalarRows))
                        #expect(
                            together == Array(alone[0 ..< rows]),
                            "[\(outputs), \(inputs)] \(dtype) rows \(rows) scalar \(scalarRows)")
                    }
                }
            }
        }

        /// A row's bytes do not depend on where it is in the call.
        @Test func rowsDoNotDependOnTheirPosition() {
            let fixture = WeightOnlyNVFP4Fixture(outputs: 70, inputs: 1296, seed: 7)
            let x = fixture.activations(rows: 16, dtype: .bfloat16)
            let order = MLXArray((0 ..< 16).map { Int32(($0 * 7 + 3) % 16) })
            let straight = weightOnlyNVFP4RowWords(fixture.product(x))
            let moved = weightOnlyNVFP4RowWords(fixture.product(x[order]))
            for (position, source) in order.asArray(Int32.self).enumerated() {
                #expect(moved[position] == straight[Int(source)])
            }
        }

        /// The kernels follow the documented order: a host loop of fused
        /// multiply-adds in that order gives the same bytes. Every E2M1 code
        /// and every finite E4M3 code occurs, signed and subnormal scales too.
        @Test(arguments: [(5, 16), (3, 48), (4, 304), (2, 2064)])
        func resultsEqualTheDocumentedOrderOnTheHost(outputs: Int, inputs: Int) {
            var state: UInt64 = UInt64(outputs * 131 + inputs) &* 0x9E37_79B9_7F4A_7C15 | 1
            func next() -> UInt64 {
                state ^= state << 13
                state ^= state >> 7
                state ^= state << 17
                return state
            }
            let codes = (0 ..< outputs * inputs / 2).map { _ in UInt8(truncatingIfNeeded: next()) }
            let scaleCodes = (0 ..< outputs * inputs / 16).map { _ -> UInt8 in
                let code = UInt8(truncatingIfNeeded: next())
                return code & 0x7F == 0x7F ? code ^ 0x01 : code
            }
            let rows = 9
            // Multiples of 1/64 below 4: exact in every activation dtype.
            let values = (0 ..< rows * inputs).map { _ in
                Float(Int(next() % 513) - 256) / 64
            }
            let weight = MLXArray(codes).reshaped(outputs, inputs / 2).view(dtype: .uint32)
            let blockScales = MLXArray(scaleCodes).reshaped(outputs, inputs / 16)
            let tensorScale: Float = 0.00371
            let magnitudes: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
            func e4m3(_ code: UInt8) -> Float {
                let exponent = Int((code & 0x7F) >> 3)
                let fraction = Float(code & 7) / 8
                let value =
                    exponent == 0
                    ? fraction * Float(sign: .plus, exponent: -6, significand: 1)
                    : (1 + fraction) * Float(sign: .plus, exponent: exponent - 7, significand: 1)
                return code & 0x80 == 0 ? value : -value
            }
            func weightValue(_ row: Int, _ k: Int) -> Float {
                let byte = codes[row * inputs / 2 + k / 2]
                let code = k % 2 == 0 ? byte & 0xF : byte >> 4
                let value = magnitudes[Int(code & 7)]
                return (code & 8 == 0 ? value : -value) * e4m3(scaleCodes[row * inputs / 16 + k / 16])
            }
            // Float32 sums of the documented order for one row and output.
            func sum(_ row: Int, _ output: Int) -> Float {
                var partitions = [Float](repeating: 0, count: 8)
                for chunk in 0 ..< (inputs + 127) / 128 {
                    var acc = partitions[chunk % 8]
                    for block in 0 ..< 4 {
                        let base = chunk * 128 + block * 32
                        for i in 0 ..< 4 {
                            for c in 0 ..< 8 where base + 4 * c + i < inputs {
                                let k = base + 4 * c + i
                                acc = acc.addingProduct(
                                    weightValue(output, k), values[row * inputs + k])
                            }
                        }
                    }
                    partitions[chunk % 8] = acc
                }
                return partitions.dropFirst().reduce(partitions[0], +)
            }
            func bfloat16(_ value: Float) -> Float {
                let bits = value.bitPattern
                let rounded = (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) & 0xFFFF_0000
                return Float(bitPattern: rounded)
            }
            for dtype in [DType.float32, .bfloat16, .float16] {
                func round(_ value: Float) -> Float {
                    switch dtype {
                    case .bfloat16: bfloat16(value)
                    case .float16: Float(Float16(value))
                    default: value
                    }
                }
                let x = MLXArray(values).reshaped(rows, inputs).asType(dtype)
                var plain: [Float] = []
                var scaled: [Float] = []
                for row in 0 ..< rows {
                    for output in 0 ..< outputs {
                        let rounded = round(sum(row, output))
                        plain.append(rounded)
                        scaled.append(round(rounded * tensorScale))
                    }
                }
                var splits = [0]
                if ModelOptWeightOnlyNVFP4RowExact.scalarKernelMeasured { splits.append(rows) }
                for scalarRows in splits {
                    let saved = ModelOptWeightOnlyNVFP4RowExact.scalarRows
                    ModelOptWeightOnlyNVFP4RowExact.scalarRows = scalarRows
                    defer { ModelOptWeightOnlyNVFP4RowExact.scalarRows = saved }
                    // One row at a time for the scalar kernel, all rows for the other.
                    func device(_ scale: MLXArray?) -> [UInt32] {
                        let pieces =
                            scalarRows == 0
                            ? [x] : (0 ..< rows).map { x[$0 ..< $0 + 1] }
                        return pieces.flatMap {
                            ModelOptWeightOnlyNVFP4RowExact.product(
                                $0, weight: weight, blockScales: blockScales, tensorScale: scale
                            ).asType(.float32).view(dtype: .uint32).asArray(UInt32.self)
                        }
                    }
                    #expect(
                        device(nil) == plain.map(\.bitPattern),
                        "[\(outputs), \(inputs)] \(dtype) scalar rows \(scalarRows)")
                    #expect(
                        device(MLXArray([tensorScale])) == scaled.map(\.bitPattern),
                        "[\(outputs), \(inputs)] \(dtype) scaled, scalar rows \(scalarRows)")
                }
            }
        }

        /// The tensor scale inside the kernel gives the bytes of the separate
        /// float32 multiply that a linear layer applies to the unscaled product.
        @Test func tensorScaleEqualsTheSeparateMultiply() {
            let fixture = WeightOnlyNVFP4Fixture(outputs: 257, inputs: 1168, seed: 23)
            let scale = MLXArray([Float(0.0041)])
            for dtype in [DType.bfloat16, .float32, .float16] {
                for rows in [1, 6, 16] {
                    let x = fixture.activations(rows: rows, dtype: dtype)
                    let unscaled = fixture.product(x)
                    let separate = (unscaled.asType(.float32) * scale).asType(dtype)
                    #expect(
                        weightOnlyNVFP4RowWords(fixture.product(x, tensorScale: scale))
                            == weightOnlyNVFP4RowWords(separate))
                }
            }
        }

        /// A NaN or an infinity in one row leaves every other row's bytes.
        @Test func nonFiniteRowsStayInTheirRows() {
            let fixture = WeightOnlyNVFP4Fixture(outputs: 96, inputs: 1040, seed: 31)
            let x = fixture.activations(rows: 16, dtype: .bfloat16)
            var values = x.asType(.float32).asArray(Float.self)
            values[2 * 1040 + 517] = .nan
            values[11 * 1040 + 3] = .infinity
            let poisoned = MLXArray(values).reshaped(16, 1040).asType(.bfloat16)
            let clean = weightOnlyNVFP4RowWords(fixture.product(x))
            for rows in [3, 8, 16] {
                let result = fixture.product(poisoned[0 ..< rows])
                let words = weightOnlyNVFP4RowWords(result)
                for row in 0 ..< rows where row != 2 && row != 11 {
                    #expect(words[row] == clean[row])
                }
                #expect(isNaN(result[2]).all().item(Bool.self))
            }
            // The one-row call of a non-finite row has the same bytes too.
            for row in [2, 11] {
                #expect(
                    weightOnlyNVFP4RowWords(fixture.product(poisoned[row ..< row + 1]))[0]
                        == weightOnlyNVFP4RowWords(fixture.product(poisoned))[row])
            }
        }

        @Test func leadingDimensionsAreKept() {
            let fixture = WeightOnlyNVFP4Fixture(outputs: 48, inputs: 512, seed: 41)
            let x = fixture.activations(rows: 6, dtype: .bfloat16)
            let nested = fixture.product(x.reshaped(2, 3, 512))
            #expect(nested.shape == [2, 3, 48])
            #expect(
                weightOnlyNVFP4RowWords(nested.reshaped(6, 48))
                    == weightOnlyNVFP4RowWords(fixture.product(x)))
        }

        @Test func supportedCalls() {
            let supports = ModelOptWeightOnlyNVFP4RowExact.supports
            #expect(supports(1, 6656, .bfloat16))
            #expect(supports(16, 16, .float32))
            #expect(!supports(0, 6656, .bfloat16))
            #expect(!supports(17, 6656, .bfloat16))
            #expect(!supports(8, 6664, .bfloat16))
            #expect(!supports(8, 6656, .int32))
        }
    }
}
