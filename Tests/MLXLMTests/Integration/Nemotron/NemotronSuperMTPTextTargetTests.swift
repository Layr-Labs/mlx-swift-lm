import MLX
import MLXNN
import Testing
@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

struct NemotronSuperMTPTextTargetTests {
    private func component() throws -> NemotronSuperModel {
        let args = NemotronHConfiguration(vocabSize: 32, hiddenSize: 16, numHiddenLayers: 3,
            numAttentionHeads: 1, numKeyValueHeads: 1, mambaNumHeads: 1, mambaHeadDim: 16,
            ssmStateSize: 4, convKernel: 4, nGroups: 1, intermediateSize: 16,
            moeIntermediateSize: 16, moeSharedExpertIntermediateSize: 16,
            nRoutedExperts: 2, numExpertsPerTok: 1, hybridOverridePattern: "M*E", headDim: 16,
            nSharedExperts: 1, moeLatentSize: 16)
        let tower = NemotronRadio(NemotronRadioDimensions(width: 2, layers: 1, heads: 1,
            channels: 1, patch: 2, positionSide: 2, prefix: 1, mlpWidth: 3))
        let images = NemotronSuperImageComposition(tower: tower,
            projector: NemotronSuperVisionProjector(visionWidth: 2, hiddenWidth: 3, outputWidth: 16),
            width: 16, imageTokenID: 18)
        return try NemotronSuperModel(languageModel: NemotronHModel(args), imageComposition: images)
    }

    private func withRequest(_ model: NemotronSuperModel,
        body: ([KVCache], CBv2RecurrentStateEvaluation) throws -> [MLXArray]) throws -> [MLXArray] {
        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 20, kvDType: .float32))
        let bank = CBv2LayerCacheBank(caches: model.languageModel.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
        let rows = try backend.makeSequenceState(layerKinds: model.cbv2LayerKinds, promptLength: 3, maxLength: 8)
        defer { backend.release(rows) }
        let caches = try bank.layerCaches(rowStates: [rows]).map { try #require($0 as? KVCache) }
        let state = try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
        let binding = try state.bind()
        let outputs = try body(caches, binding)
        try MLX.withError { errors in eval(outputs + (try binding.evaluate())); try errors.check() }
        try binding.commit()
        return outputs
    }

    @Test func hiddenTargetMatchesPlainTextAndNarrowPrefill() throws {
        let model = try component(), target = NemotronSuperMTPTextTarget(model)
        let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let plain = try withRequest(model) { caches, state in
            [model.languageModel.cbv2Forward(tokens, caches: caches, recurrentState: [state])]
        }
        let captured = try withRequest(model) { caches, state in
            let result = target.cbv2ForwardWithHidden(tokens, caches: caches,
                recurrentState: [state], positionIds: nil)
            return [result.logits, result.lastHidden]
        }
        #expect(captured[0].shape == [1, 3, 32] && captured[1].shape == [1, 3, 16])
        #expect(abs(captured[0] - plain[0]).max().item(Float.self) < 1e-5)
        let narrowed = try withRequest(model) { caches, state in
            let result = target.cbv2ForwardWithHiddenForPrefill(tokens, caches: caches,
                recurrentState: [state], positionIds: nil, requirement: .lastPositionLogits)
            return [result.logits, result.lastHidden]
        }
        #expect(narrowed[0].shape == [1, 1, 32])
        #expect(abs(narrowed[0] - captured[0][0..., 2..., 0...]).max().item(Float.self) < 1e-5)
        #expect(abs(narrowed[1] - captured[1]).max().item(Float.self) < 1e-5)
        #expect(target.cbv2MTPTargetIdentity == ObjectIdentifier(model.languageModel))
        #expect(!((target as AnyObject) is any CBv2RecurrentCaptureMTPForwardable))
        #expect(!((NemotronSuperRecurrentAdapter(model) as AnyObject) is any CBv2RecurrentMTPForwardable))
    }

    @Test func mediaAndExplicitPositionsAreRejected() throws {
        let model = try component(), target = NemotronSuperMTPTextTarget(model)
        #expect(throws: NemotronHModel.SuperMTPTextError.self) {
            try NemotronSuperMTPTextTarget.validateTextPrefill(hasEmbeddings: true)
        }
        let input = LMInput(text: .init(tokens: MLXArray([Int32(1)])),
            image: .init(pixels: MLXArray.zeros([1, 1, 4, 4])))
        #expect(throws: NemotronHModel.SuperMTPTextError.self) {
            try target.prepare(input, cache: [], windowSize: nil)
        }
        #expect(throws: NemotronHModel.SuperMTPTextError.self) {
            try model.languageModel.superMTPTextForward(MLXArray([Int32(1)]).reshaped(1, 1),
                caches: [], recurrentState: [], positionIds: MLXArray([Int32(0)]))
        }
    }
}
