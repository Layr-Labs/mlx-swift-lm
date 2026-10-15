import MLX
import MLXNN
import Testing

@testable import MLXVLM

@Suite(.tags(.integration))
struct NemotronSuperImageCompositionTests {
    private func component(dtype: DType = .float32) -> NemotronSuperImageComposition {
        let d = NemotronRadioDimensions(
            width: 2, layers: 1, heads: 1,
            channels: 1, patch: 2, positionSide: 2, prefix: 1, mlpWidth: 3)
        let component = NemotronSuperImageComposition(
            tower: NemotronRadio(d),
            projector: NemotronSuperVisionProjector(visionWidth: 2, hiddenWidth: 3, outputWidth: 2),
            width: 2, imageTokenID: 18, visionDType: dtype)
        component.update(
            parameters: ModuleParameters.unflattened(
                component.parameters().flattened().map { ($0.0, $0.1.asType(dtype)) }))
        return component
    }

    @Test func multipleImagesKeepTextAndOrderWithNativeCast() throws {
        let component = component()
        let text = MLXArray((0 ..< 12).map(Float.init)).reshaped(1, 6, 2).asType(.bfloat16)
        let first = MLXArray([Float(20), 21, 22, 23]).reshaped(1, 2, 2)
        let second = MLXArray([Float(30), 31]).reshaped(1, 1, 2)
        let output = try component.substitute(
            prompt: [1, 18, 18, 2, 18, 3],
            textEmbeddings: text, imageFeatures: [first, second])
        #expect(output.dtype == .bfloat16)
        #expect(
            output.asType(.float32).asArray(Float.self) == [
                0, 1, 20, 21, 22, 23, 6, 7, 30, 31, 10, 11,
            ])
    }

    @Test func imageGraphComposesTowerAndProjector() throws {
        let component = component()
        let pixels = MLXArray((0 ..< 16).map { Float($0) / 16 }).reshaped(1, 1, 4, 4)
        let actual = try component.imageFeatures(normalizedPixels: pixels)
        let expected = component.projector(
            component.tower(normalizedPixels: pixels), height: 2, width: 2)
        #expect(actual.shape == [1, 1, 2])
        #expect(arrayEqual(actual, expected).item(Bool.self))
        let text = MLXArray.zeros([1, 3, 2])
        let result = try component.substitute(
            prompt: [1, 18, 2], textEmbeddings: text,
            imageFeatures: [actual])
        #expect(arrayEqual(result[0..., 1 ..< 2, 0...], actual).item(Bool.self))
    }

    @Test func pixelsCastToDeclaredVisionDtypeBeforeTower() throws {
        let component = component(dtype: .bfloat16)
        let pixels = MLXArray((0 ..< 16).map { Float($0) / 17 }).reshaped(1, 1, 4, 4)
        let actual = try component.imageFeatures(normalizedPixels: pixels)
        let expected = component.projector(
            component.tower(normalizedPixels: pixels.asType(.bfloat16)), height: 2, width: 2)
        #expect(actual.dtype == .bfloat16)
        #expect(arrayEqual(actual, expected).item(Bool.self))
    }

    @Test func rejectsUnsupportedGeometryAndPlaceholderMismatch() {
        let component = component()
        #expect(throws: NemotronSuperImageComposition.Failure.self) {
            try component.imageFeatures(normalizedPixels: MLXArray.zeros([2, 1, 4, 4]))
        }
        #expect(throws: NemotronSuperImageComposition.Failure.self) {
            try component.imageFeatures(normalizedPixels: MLXArray.zeros([1, 1, 2, 4]))
        }
        #expect(throws: NemotronSuperImageComposition.Failure.self) {
            try component.substitute(
                prompt: [1, 18], textEmbeddings: MLXArray.zeros([1, 2, 2]),
                imageFeatures: [MLXArray.zeros([1, 2, 2])])
        }
        #expect(throws: NemotronSuperImageComposition.Failure.self) {
            try component.substitute(
                prompt: [1, 18], textEmbeddings: MLXArray.zeros([1, 2, 2]),
                imageFeatures: [MLXArray.zeros([1, 1, 3])])
        }
    }

    private func replacementWeights(_ component: NemotronSuperImageComposition) -> [String:
        MLXArray]
    {
        Dictionary(
            uniqueKeysWithValues:
                component.tower.parameters().flattened().map {
                    ($0.0, MLXArray.ones($0.1.shape, dtype: $0.1.dtype))
                }
                + component.projector.parameters().flattened().map {
                    ($0.0, MLXArray.ones($0.1.shape, dtype: $0.1.dtype))
                })
    }

    @Test func invalidProjectorLeavesBothWeightSetsUnchanged() throws {
        let model = component()
        let before = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var incoming = replacementWeights(model)
        incoming.removeValue(forKey: "mlp1.linear2.weight")
        #expect(throws: (any Error).self) { try model.loadImageWeights(from: incoming) }
        for (name, value) in model.parameters().flattened() {
            #expect(arrayEqual(value, try #require(before[name])).item(Bool.self))
        }
        incoming = replacementWeights(model)
        incoming["mlp1.linear2.weight"] = try #require(incoming["mlp1.linear2.weight"]).asType(
            .int32)
        #expect(throws: (any Error).self) { try model.loadImageWeights(from: incoming) }
        for (name, value) in model.parameters().flattened() {
            #expect(arrayEqual(value, try #require(before[name])).item(Bool.self))
        }
    }

    @Test func validatedImageWeightsCommitTogether() throws {
        let model = component()
        try model.loadImageWeights(from: replacementWeights(model))
        for (_, value) in model.parameters().flattened() {
            #expect(
                arrayEqual(value, MLXArray.ones(value.shape, dtype: value.dtype)).item(Bool.self))
        }
        let before = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var invalid = replacementWeights(model)
        invalid["embeddings.projection.weight"] = MLXArray.zeros([1])
        #expect(throws: (any Error).self) { try model.loadImageWeights(from: invalid) }
        for (name, value) in model.parameters().flattened() {
            #expect(arrayEqual(value, try #require(before[name])).item(Bool.self))
        }
    }

}
