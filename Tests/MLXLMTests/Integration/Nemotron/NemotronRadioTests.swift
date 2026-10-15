import MLX
import MLXNN
@testable import MLXVLM
import Testing

@Suite(.tags(.integration))
struct NemotronRadioTests {
    private let small = NemotronRadioDimensions(width: 2, layers: 1, heads: 1,
        channels: 1, patch: 2, positionSide: 2, prefix: 1, mlpWidth: 3)

    @Test func patchOrderIsChannelThenSpatial() {
        let input = MLXArray((0..<32).map(Float.init)).reshaped(1, 2, 4, 4)
        let patches = NemotronRadioEmbeddings.imagePatches(input, patch: 2)
        #expect(patches.shape == [1, 4, 8])
        #expect(patches.asArray(Float.self) == [
            0, 1, 4, 5, 16, 17, 20, 21, 2, 3, 6, 7, 18, 19, 22, 23,
            8, 9, 12, 13, 24, 25, 28, 29, 10, 11, 14, 15, 26, 27, 30, 31,
        ])
    }

    @Test func squareBilinearThenCrop() {
        let table = MLXArray([Float(0), 10, 20, 30]).reshaped(1, 4, 1)
        let resized = NemotronRadioEmbeddings.croppedPositions(table, side: 2,
            rows: 2, columns: 4, dtype: .float32)
        #expect(resized.asArray(Float.self) == [0, 2.5, 7.5, 10, 5, 7.5, 12.5, 15])
        let downsampled = NemotronRadioEmbeddings.croppedPositions(table, side: 2,
            rows: 1, columns: 1, dtype: .bfloat16)
        #expect(downsampled.dtype == .bfloat16)
        #expect(downsampled.asType(.float32).item(Float.self) == 15)
    }

    @Test func rawFusedQKVExtractionAndStrictMissing() throws {
        let tower = NemotronRadio(small)
        let parameters = Dictionary(uniqueKeysWithValues: tower.parameters().flattened())
        var weights = parameters.mapValues { MLXArray.zeros($0.shape) }
        for name in ["query", "key", "value"] {
            weights.removeValue(forKey: "blocks.0.\(name).weight")
            weights.removeValue(forKey: "blocks.0.\(name).bias")
        }
        let root = "vision_model.radio_model.model.blocks.0.attn.qkv."
        weights[root + "weight"] = MLXArray((0..<12).map(Float.init)).reshaped(6, 2)
        weights[root + "bias"] = MLXArray((0..<6).map(Float.init))
        try tower.loadImageWeights(from: weights)
        #expect(tower.blocks[0].query.weight.asArray(Float.self) == [0, 1, 2, 3])
        #expect(tower.blocks[0].key.weight.asArray(Float.self) == [4, 5, 6, 7])
        #expect(tower.blocks[0].value.bias?.asArray(Float.self) == [4, 5])
        weights.removeValue(forKey: root + "weight")
        #expect(throws: NemotronRadio.WeightError.self) { try tower.loadImageWeights(from: weights) }
    }

    @Test func zeroResidualTowerPreservesProjectedPatchesWithoutRenormalizingPixels() throws {
        let tower = NemotronRadio(small)
        var weights = Dictionary(uniqueKeysWithValues: tower.parameters().flattened())
            .mapValues { MLXArray.zeros($0.shape) }
        weights["embeddings.projection.weight"] = MLXArray([Float(1), 0, 0, 0, 0, 0, 0, 1]).reshaped(2, 4)
        weights["embeddings.cls_register_token"] = MLXArray([Float(99), 98]).reshaped(1, 2)
        try tower.loadImageWeights(from: weights)
        let pixels = MLXArray((0..<16).map(Float.init)).reshaped(1, 1, 4, 4)
        let output = tower(normalizedPixels: pixels)
        #expect(output.shape == [1, 4, 2])
        #expect(output.asArray(Float.self) == [0, 5, 2, 7, 8, 13, 10, 15])
    }

    // Independent CPU PyTorch 2.12 reference: nonzero attention, MLP and 2→3 positional resize.
    @Test func nonzeroTowerMatchesIndependentTorchReference() throws {
        for dtype: DType in [.float32, .bfloat16] {
            let tower = NemotronRadio(.init(width: 4, layers: 1, heads: 2,
                channels: 1, patch: 2, positionSide: 2, prefix: 1, mlpWidth: 6))
            let parameters = Dictionary(uniqueKeysWithValues: tower.parameters().flattened())
            var weights: [String: MLXArray] = [:]
            for (index, key) in parameters.keys.sorted().enumerated() {
                let shape = parameters[key]!.shape
                let count = shape.reduce(1, *)
                let values = (0..<count).map { Float(((index + 1) * 7 + $0 * 3) % 23 - 11) / 32 }
                weights[key] = MLXArray(values).reshaped(shape).asType(dtype)
            }
            try tower.loadImageWeights(from: weights)
            let pixels = MLXArray((0..<24).map { Float($0 % 11 - 5) / 8 })
                .reshaped(1, 1, 4, 6).asType(dtype)
            let actual = tower(normalizedPixels: pixels).asType(.float32).asArray(Float.self)
            let fp32: [Float] = [-0.09251272678375244, -0.06425459682941437, -0.4223155379295349, 0.39523208141326904, -0.26454609632492065, -0.22057481110095978, -0.5631136894226074, 0.2699863910675049, -0.13769692182540894, -0.5919232368469238, -0.4475262761116028, -0.1157076507806778, -0.07696899771690369, -0.04085339605808258, -0.3911547064781189, 0.43416839838027954, -0.2489943504333496, -0.19716906547546387, -0.5319430232048035, 0.30893564224243164, 0.3132234215736389, -0.6476621031761169, -0.019215703010559082, -0.1948658525943756]
            let bf16: [Float] = [-0.091796875, -0.0634765625, -0.421875, 0.39453125, -0.263671875, -0.2197265625, -0.5625, 0.26953125, -0.13671875, -0.59375, -0.44921875, -0.1162109375, -0.076171875, -0.0400390625, -0.390625, 0.43359375, -0.248046875, -0.1962890625, -0.53125, 0.30859375, 0.314453125, -0.6484375, -0.017578125, -0.1943359375]
            let expected = dtype == .float32 ? fp32 : bf16
            // BF16 permits two output ULPs at the largest reference magnitude.
            let tolerance: Float = dtype == .float32 ? 0.000005 : 0.0078125
            #expect(actual.count == expected.count)
            for (a, b) in zip(actual, expected) { #expect(abs(a - b) <= tolerance) }
        }
    }

    @Test func integerWeightsAreRejectedBeforeMutation() throws {
        let tower = NemotronRadio(small)
        let original = Dictionary(uniqueKeysWithValues: tower.parameters().flattened())
        for useFused in [false, true] {
            var weights = original.mapValues { MLXArray.zeros($0.shape) }
            if useFused {
                for name in ["query", "key", "value"] {
                    weights.removeValue(forKey: "blocks.0.\(name).weight")
                }
                weights["vision_model.radio_model.model.blocks.0.attn.qkv.weight"] =
                    MLXArray.zeros([6, 2], dtype: .uint8)
            } else {
                weights["embeddings.projection.weight"] = MLXArray.zeros([2, 4], dtype: .uint8)
            }
            #expect(throws: NemotronRadio.WeightError.self) { try tower.loadImageWeights(from: weights) }
            let after = Dictionary(uniqueKeysWithValues: tower.parameters().flattened())
            for key in original.keys {
                #expect(after[key]!.asArray(Float.self) == original[key]!.asArray(Float.self))
            }
        }
    }
}
