import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

@Suite(.tags(.integration), .serialized)
struct Nemotron35BatchDecodeProjectionTests {
    private func bytes(_ value: MLXArray) -> Data {
        eval(value)
        return value.asData(access: .copy).data
    }

    private func input(rows: Int, width: Int = 256, tokens: Int = 1) -> MLXArray {
        MLXArray((0..<rows * tokens * width).map {
            sin(Float($0) * 0.37) + Float($0 % 7 - 3) / 128
        }, [rows, tokens, width])
    }

    private func packed(bits: Int, group: Int, width: Int = 256, outputs: Int = 35) -> QuantizedLinear {
        let weights = (0..<outputs * width * bits / 32).map {
            UInt32(truncatingIfNeeded: $0 &* 1_664_525 &+ 1_013_904_223)
        }
        let scales = MLXArray((0..<outputs * width / group).map { Float($0 % 19 + 1) / 128 },
            [outputs, width / group])
        let offsets = MLXArray((0..<outputs * width / group).map { Float($0 % 17 - 8) / 128 },
            [outputs, width / group])
        return QuantizedLinear(weight: MLXArray(weights, [outputs, width * bits / 32]),
            bias: MLXArray((0..<outputs).map { Float($0 - 10) / 64 }),
            scales: scales, biases: offsets, groupSize: group, bits: bits)
    }

    @Test
    func testPackedAffineRowsMatchOriginalSingletonBytesWithBias() {
        Device.withDefaultDevice(.cpu) {
            for bits in [2, 3, 4, 8] {
                for group in [64, 128] {
                    let layer = packed(bits: bits, group: group)
                    for rows in [1, 2, 4, 8] {
                        let x = input(rows: rows)
                        let expected = concatenated((0..<rows).map { layer(x[$0..<$0 + 1]) }, axis: 0)
                        let actual = Nemotron35BatchDecode.$active.withValue(true) {
                            Nemotron35BatchDecode.linear(layer, x)
                        }
                        #expect(actual.shape == expected.shape)
                        #expect(actual.dtype == expected.dtype)
                        #expect(bytes(actual) == bytes(expected), "bits=\(bits), group=\(group), rows=\(rows)")
                    }
                }
            }
        }
    }

    @Test
    func testRecurrentRowsOwnCompactNativeBits() throws {
        try Device.withDefaultDevice(.cpu) {
            let x = MLXArray((0..<8 * 128).map { UInt32(truncatingIfNeeded: $0 * 1664525 + 1013904223) },
                [8, 1, 128])
            eval(x)
            for row in 0..<8 {
                let retained = Nemotron35BatchDecode.$active.withValue(true) {
                    Nemotron35BatchDecode.ownedRow(x, row: row)
                }
                eval(retained)
                #expect(bytes(retained) == bytes(x[row..<row + 1]))
                let info = try #require(try retained.evaluatedBufferInfo())
                #expect(info.dataElements == retained.size)
                #expect(info.dataOffset == 0)
                #expect(info.isRowContiguous)
                #expect(info.allocatedBytes <= (try Memory.allocationFootprintUpperBound(byteCount: retained.nbytes)))
            }
        }
    }

    @Test
    func testDisabledSingletonAndPrefillCallOriginalForwardOnce() {
        Device.withDefaultDevice(.cpu) {
            let layer = ObservedLinear()
            for (rows, tokens, active) in [(4, 1, false), (1, 1, true), (4, 3, true)] {
                layer.shapes = []
                let x = input(rows: rows, tokens: tokens)
                let got = Nemotron35BatchDecode.$active.withValue(active) {
                    Nemotron35BatchDecode.linear(layer, x)
                }
                #expect(layer.shapes == [x.shape])
                #expect(bytes(got) == bytes(x + 7))
            }
        }
    }

    @Test
    func testCustomQuantizedOverrideAndFloatingModuleRemainRowwise() {
        Device.withDefaultDevice(.cpu) {
            let source = packed(bits: 4, group: 64)
            let custom = ObservedQuantized(weight: source.weight, scales: source.scales,
                biases: source.biases, groupSize: source.groupSize, bits: source.bits)
            let floating = ObservedLinear()
            let x = input(rows: 4)
            for layer in [custom, floating] as [Linear] {
                let expected = concatenated((0..<4).map { layer(x[$0..<$0 + 1]) }, axis: 0)
                custom.shapes = []; floating.shapes = []
                let got = Nemotron35BatchDecode.$active.withValue(true) {
                    Nemotron35BatchDecode.linear(layer, x)
                }
                let shapes = layer === custom ? custom.shapes : floating.shapes
                #expect(shapes == Array(repeating: [1, 1, 256], count: 4))
                #expect(bytes(got) == bytes(expected))
            }
        }
    }

    @Test
    func testQuantizedAndCustomTiedHeadPreserveOriginalRows() {
        Device.withDefaultDevice(.cpu) {
            let values = MLXArray((0..<35 * 256).map { sin(Float($0) * 0.11) }, [35, 256])
            let packed = QuantizedEmbedding(weight: values, groupSize: 64, bits: 4)
            let custom = ObservedEmbedding(weight: values)
            let x = input(rows: 4)
            for embedding in [packed, custom] as [Embedding] {
                let expected = concatenated((0..<4).map { embedding.asLinear(x[$0..<$0 + 1]) }, axis: 0)
                custom.shapes = []
                let got = Nemotron35BatchDecode.$active.withValue(true) {
                    Nemotron35BatchDecode.embedding(embedding, x)
                }
                #expect(bytes(got) == bytes(expected))
                if embedding === custom {
                    #expect(custom.shapes == Array(repeating: [1, 1, 256], count: 4))
                }
            }
        }
    }

    @Test
    func testUnqualifiedPackingKeepsOriginalRowForward() {
        Device.withDefaultDevice(.cpu) {
            let layer = packed(bits: 4, group: 32)
            let x = input(rows: 4)
            let expected = concatenated((0..<4).map { layer(x[$0..<$0 + 1]) }, axis: 0)
            let got = Nemotron35BatchDecode.$active.withValue(true) {
                Nemotron35BatchDecode.linear(layer, x)
            }
            #expect(bytes(got) == bytes(expected))
        }
    }


}

private final class ObservedLinear: Linear {
    var shapes: [[Int]] = []
    init() { super.init(weight: MLXArray.zeros([256, 256]), bias: nil) }
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        shapes.append(x.shape)
        return x + 7
    }
}

private final class ObservedQuantized: QuantizedLinear {
    var shapes: [[Int]] = []
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        shapes.append(x.shape)
        return super.callAsFunction(x * 2) + 11
    }
}

private final class ObservedEmbedding: Embedding {
    var shapes: [[Int]] = []
    override func asLinear(_ x: MLXArray) -> MLXArray {
        shapes.append(x.shape)
        return super.asLinear(x * 3) + 5
    }
}
