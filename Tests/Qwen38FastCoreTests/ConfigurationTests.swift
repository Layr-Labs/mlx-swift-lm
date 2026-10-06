import Testing

@testable import Qwen38FastCore

@Suite("Qwen 3.8 runner shape contract")
struct ConfigurationTests {
    @Test("target contract preserves real Qwen 3.8 geometry and packing")
    func targetContract() {
        let target = Qwen38TargetContract.production

        #expect(target.modelType == "qwen3_5_text")
        #expect(target.hiddenSize == 5_120)
        #expect(target.hiddenLayers == 64)
        #expect(target.intermediateSize == 17_408)
        #expect(target.attentionHeads == 24)
        #expect(target.keyValueHeads == 4)
        #expect(target.headDimension == 256)
        #expect(target.linearKeyHeads == 16)
        #expect(target.linearValueHeads == 48)
        #expect(target.linearKeyHeadDimension == 128)
        #expect(target.linearValueHeadDimension == 128)
        #expect(target.fullAttentionInterval == 4)
        #expect(target.vocabularySize == 248_320)
        #expect(target.activationDType == "bfloat16")
        #expect(target.defaultQuantization == .affine(bits: 4, groupSize: 32))
        #expect(target.requiredQuantizationIslands.contains(.affine(bits: 8, groupSize: 64)))
    }

    @Test("draft contract preserves DFlash2 physical and feature geometry")
    func draftContract() {
        let draft = Qwen38DraftContract.production

        #expect(draft.blockSize == 8)
        #expect(draft.targetLayerIDs == [5, 19, 33, 47, 61])
        #expect(draft.hiddenLayers == 5)
        #expect(draft.hiddenSize == 5_120)
        #expect(draft.intermediateSize == 17_408)
        #expect(draft.attentionHeads == 32)
        #expect(draft.keyValueHeads == 8)
        #expect(draft.headDimension == 128)
        #expect(draft.vocabularySize == 248_320)
        #expect(draft.maskTokenID == 248_070)
        #expect(draft.quantization == .affine(bits: 4, groupSize: 64))
    }
}
