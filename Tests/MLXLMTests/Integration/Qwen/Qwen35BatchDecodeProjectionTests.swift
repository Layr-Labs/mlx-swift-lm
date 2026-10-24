import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

@Suite(.tags(.integration), .serialized)
struct Qwen35BatchDecodeProjectionTests {
    private func bytes(_ value: MLXArray) -> Data {
        eval(value)
        return value.asData(access: .copy).data
    }

    private func input(rows: Int, width: Int = 256, tokens: Int = 1) -> MLXArray {
        MLXArray(
            (0 ..< rows * tokens * width).map {
                sin(Float($0) * 0.37) + Float($0 % 7 - 3) / 128
            }, [rows, tokens, width])
    }

    private func packed(bits: Int, group: Int, width: Int = 256, outputs: Int = 35)
        -> QuantizedLinear
    {
        let weights = (0 ..< outputs * width * bits / 32).map {
            UInt32(truncatingIfNeeded: $0 &* 1_664_525 &+ 1_013_904_223)
        }
        let scales = MLXArray(
            (0 ..< outputs * width / group).map { Float($0 % 19 + 1) / 128 },
            [outputs, width / group])
        let offsets = MLXArray(
            (0 ..< outputs * width / group).map { Float($0 % 17 - 8) / 128 },
            [outputs, width / group])
        return QuantizedLinear(
            weight: MLXArray(weights, [outputs, width * bits / 32]),
            bias: MLXArray((0 ..< outputs).map { Float($0 - 10) / 64 }),
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
                        let expected = concatenated(
                            (0 ..< rows).map { layer(x[$0 ..< $0 + 1]) }, axis: 0)
                        let actual = Qwen35BatchDecodeProjection.$active.withValue(true) {
                            Qwen35BatchDecodeProjection.linear(layer, x)
                        }
                        #expect(actual.shape == expected.shape)
                        #expect(actual.dtype == expected.dtype)
                        #expect(
                            bytes(actual) == bytes(expected),
                            "bits=\(bits), group=\(group), rows=\(rows)")
                    }
                }
            }
        }
    }

    @Test
    func testScopeRequiresExplicitDecodeAndRestoresOuterValue() {
        Device.withDefaultDevice(.cpu) {
            #expect(!(Qwen35BatchDecodeProjection.active))
            let decode = MLXArray.zeros([4, 1], dtype: .int32)
            Qwen35BatchDecodeProjection.withModel(decode) {
                #expect(!(Qwen35BatchDecodeProjection.active))
            }
            Qwen35BatchDecodeScope.$enabled.withValue(true) {
                for shape in [[1, 1], [1, 8], [4, 2], [9, 1]] {
                    Qwen35BatchDecodeProjection.withModel(MLXArray.zeros(shape, dtype: .int32)) {
                        #expect(!(Qwen35BatchDecodeProjection.active))
                    }
                }
                Qwen35BatchDecodeProjection.withModel(decode) {
                    #expect(Qwen35BatchDecodeProjection.active)
                    Qwen35BatchDecodeProjection.withModel(MLXArray.zeros([4, 2], dtype: .int32)) {
                        #expect(!(Qwen35BatchDecodeProjection.active))
                    }
                    #expect(Qwen35BatchDecodeProjection.active)
                }
            }
            #expect(!(Qwen35BatchDecodeProjection.active))
            #expect(!(Qwen35BatchDecodeScope.enabled))
        }
    }

    /// A scoped verify rectangle projects every row as that row's own
    /// `[1, S, H]` window: quantized matmul selects its kernel from the total
    /// row count, so one `[B·S, H]` product is not a single request's
    /// arithmetic once B·S reaches the device's vector limit.
    @Test
    func testVerifyRowsProjectEachRowAsItsOwnWindowOnlyWhenScoped() {
        Device.withDefaultDevice(.cpu) {
            Qwen35BatchDecodeScope.$enabled.withValue(true) {
                // Rectangles need the verify scope; decode columns do not.
                Qwen35BatchDecodeProjection.withModel(MLXArray.zeros([4, 3], dtype: .int32)) {
                    #expect(!(Qwen35BatchDecodeProjection.active))
                }
                Qwen35BatchDecodeScope.$verifyRows.withValue(true) {
                    for shape in [[2, 2], [4, 3], [8, 5], [4, 1]] {
                        Qwen35BatchDecodeProjection.withModel(MLXArray.zeros(shape, dtype: .int32))
                        {
                            #expect(Qwen35BatchDecodeProjection.active, "\(shape)")
                        }
                    }
                    for shape in [[1, 3], [9, 3], [4, 6], [4, 7]] {
                        Qwen35BatchDecodeProjection.withModel(MLXArray.zeros(shape, dtype: .int32))
                        {
                            #expect(!(Qwen35BatchDecodeProjection.active), "\(shape)")
                        }
                    }
                }
            }
            Qwen35BatchDecodeScope.$verifyRows.withValue(true) {
                // The verify scope alone enables nothing.
                Qwen35BatchDecodeProjection.withModel(MLXArray.zeros([4, 3], dtype: .int32)) {
                    #expect(!(Qwen35BatchDecodeProjection.active))
                }
            }
            for bits in [4, 8] {
                let layer = packed(bits: bits, group: 64)
                for (rows, tokens) in [(2, 2), (4, 3), (8, 5)] {
                    let x = input(rows: rows, tokens: tokens)
                    let expected = concatenated(
                        (0 ..< rows).map { layer(x[$0 ..< $0 + 1]) }, axis: 0)
                    let actual = Qwen35BatchDecodeScope.$verifyRows.withValue(true) {
                        Qwen35BatchDecodeProjection.$active.withValue(true) {
                            Qwen35BatchDecodeProjection.linear(layer, x)
                        }
                    }
                    #expect(actual.shape == [rows, tokens, 35])
                    #expect(
                        bytes(actual) == bytes(expected),
                        "bits=\(bits), rows=\(rows), tokens=\(tokens)")
                }
            }
            #expect(!(Qwen35BatchDecodeScope.verifyRows))
        }
    }

    /// Shared verify calls mirror MLX's vector limit per architecture and
    /// apply only on the GPU: on the CPU (and for the quad kernel's 64/128
    /// widths or another quantizer) every window keeps its own call, and the
    /// scoped rectangle still equals its row-by-row result there.
    @Test
    func testSharedVerifyWindowsFollowTheVectorLimitAndStayRowwiseOnTheCPU() {
        let expected: [(String, [Int])] = [
            ("applegpu_g15d", [12, 18, 32]), ("applegpu_g17s", [13, 25, 33]),
            ("applegpu_g16s", [13, 15, 13]), ("applegpu_g14s", [6, 10, 14]),
            ("applegpu_g14d", [12, 18, 32]), ("applegpu_g12x", [10, 12, 18]),
        ]
        for (architecture, limits) in expected {
            let got = [(5120, 6144), (4096, 4096), (2048, 2048)].map {
                Qwen35QuantizedVectorLimit.limit(
                    inputs: $0.0, outputs: $0.1, architecture: architecture)
            }
            #expect(got == limits, "\(architecture)")
        }
        Device.withDefaultDevice(.cpu) {
            #expect(
                Qwen35QuantizedVectorLimit.sharedWindows(
                    MLXArray.zeros([8, 3, 256]), outputs: 35, mode: .affine) == 1)
            let layer = packed(bits: 4, group: 64)
            let x = input(rows: 6, tokens: 3)
            let expected = concatenated((0 ..< 6).map { layer(x[$0 ..< $0 + 1]) }, axis: 0)
            let actual = Qwen35BatchDecodeScope.$verifyRows.withValue(true) {
                Qwen35BatchDecodeProjection.$active.withValue(true) {
                    Qwen35BatchDecodeProjection.linear(layer, x)
                }
            }
            #expect(bytes(actual) == bytes(expected))
            let grouped = Qwen35QuantizedVectorLimit.rows(x, by: 4) { layer($0) }
            #expect(grouped.shape == expected.shape)
        }
        Device.withDefaultDevice(.gpu) {
            for columns in 2 ... 5 {
                let shared = Qwen35QuantizedVectorLimit.sharedWindows(
                    MLXArray.zeros([8, columns, 5120]), outputs: 6144, mode: .affine)
                #expect(shared >= 1)
                #expect(
                    shared * columns
                        < Qwen35QuantizedVectorLimit.limit(inputs: 5120, outputs: 6144))
            }
            #expect(
                Qwen35QuantizedVectorLimit.sharedWindows(
                    MLXArray.zeros([8, 3, 128]), outputs: 6144, mode: .affine) == 1)
            #expect(
                Qwen35QuantizedVectorLimit.sharedWindows(
                    MLXArray.zeros([8, 3, 5120]), outputs: 6144, mode: .mxfp4) == 1)
        }
    }

    @Test
    func testDisabledSingletonAndPrefillCallOriginalForwardOnce() {
        Device.withDefaultDevice(.cpu) {
            let layer = ObservedLinear()
            for (rows, tokens, active) in [(4, 1, false), (1, 1, true), (4, 3, true)] {
                layer.shapes = []
                let x = input(rows: rows, tokens: tokens)
                let got = Qwen35BatchDecodeProjection.$active.withValue(active) {
                    Qwen35BatchDecodeProjection.linear(layer, x)
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
            let custom = ObservedQuantized(
                weight: source.weight, scales: source.scales,
                biases: source.biases, groupSize: source.groupSize, bits: source.bits)
            let floating = ObservedLinear()
            let x = input(rows: 4)
            for layer in [custom, floating] as [Linear] {
                let expected = concatenated((0 ..< 4).map { layer(x[$0 ..< $0 + 1]) }, axis: 0)
                custom.shapes = []
                floating.shapes = []
                let got = Qwen35BatchDecodeProjection.$active.withValue(true) {
                    Qwen35BatchDecodeProjection.linear(layer, x)
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
            let values = MLXArray((0 ..< 35 * 256).map { sin(Float($0) * 0.11) }, [35, 256])
            let packed = QuantizedEmbedding(weight: values, groupSize: 64, bits: 4)
            let custom = ObservedEmbedding(weight: values)
            let x = input(rows: 4)
            for embedding in [packed, custom] as [Embedding] {
                let expected = concatenated(
                    (0 ..< 4).map { embedding.asLinear(x[$0 ..< $0 + 1]) }, axis: 0)
                custom.shapes = []
                let got = Qwen35BatchDecodeProjection.$active.withValue(true) {
                    Qwen35BatchDecodeProjection.embedding(embedding, x)
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
            #expect(
                !(Qwen35BatchDecodeProjection.supports(
                    layer.mode, bits: layer.bits,
                    groupSize: layer.groupSize, input: x, scales: layer.scales)))
            let expected = concatenated((0 ..< 4).map { layer(x[$0 ..< $0 + 1]) }, axis: 0)
            let got = Qwen35BatchDecodeProjection.$active.withValue(true) {
                Qwen35BatchDecodeProjection.linear(layer, x)
            }
            #expect(bytes(got) == bytes(expected))
        }
    }

    @Test
    func testHadamardLayoutAndTiedHeadUseTheirOriginalTransformOnce() throws {
        try Device.withDefaultDevice(.cpu) {
            let source = packed(bits: 2, group: 128)
            let transform = try SignedBlockHadamard(
                blockSize: 64,
                signs: (0 ..< 256).map { $0 % 3 == 0 ? Float(-1) : Float(1) })
            let layer = try HadamardQuantizedLinear(
                weight: source.weight, bias: source.bias,
                scales: source.scales, biases: source.biases, groupSize: 128, bits: 2,
                transform: transform,
                gdnLayout: HadamardGDNLayout(width: 256, keyHeads: 2, valueHeads: 4))
            let head = try HadamardQuantizedEmbedding(
                weight: source.weight, scales: source.scales,
                biases: source.biases, groupSize: 128, bits: 2, transform: transform)
            let x = input(rows: 4)
            let expectedLinear = concatenated((0 ..< 4).map { layer(x[$0 ..< $0 + 1]) }, axis: 0)
            let expectedHead = concatenated(
                (0 ..< 4).map { head.asLinear(x[$0 ..< $0 + 1]) }, axis: 0)
            let (actualLinear, actualHead) = Qwen35BatchDecodeProjection.$active.withValue(true) {
                (
                    Qwen35BatchDecodeProjection.linear(layer, x),
                    Qwen35BatchDecodeProjection.embedding(head, x)
                )
            }
            #expect(bytes(actualLinear) == bytes(expectedLinear))
            #expect(bytes(actualHead) == bytes(expectedHead))
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
