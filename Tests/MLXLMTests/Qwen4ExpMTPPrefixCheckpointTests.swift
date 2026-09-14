import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("Qwen4 MTP trusted prefix checkpoint", .serialized)
struct Qwen4ExpMTPPrefixCheckpointTests {
    // Shared by media-cache tests so target-only restoration uses the same
    // established production assistant fixture, not a second mock drafter.
    func fixture() throws -> Qwen4ExpInlineMTPAssistant {
        var args = Qwen4ExpTextConfiguration()
        args.hiddenSize = 32
        args.hiddenLayers = 1
        args.attentionHeads = 2
        args.kvHeads = 1
        args.headDim = 16
        args.linearNumValueHeads = 2
        args.linearNumKeyHeads = 1
        args.linearKeyHeadDim = 8
        args.linearValueHeadDim = 8
        args.vocabularySize = 64
        args.maxPositionEmbeddings = 128
        args.fullAttentionInterval = 1
        args.layerTypes = ["qwen_sparse_attention"]
        args.hcCount = 2
        args.hcLowrank = 8
        args.pleLayerIds = []
        args.pleEmbedDim = 32
        args.indexerNHeads = 2
        args.indexerKVHeads = 1
        args.indexerHeadDim = 8
        args.indexerBudget = 16
        args.indexerCompressRatio = 4
        args.numExperts = 1
        args.numExpertsPerTok = 1
        args.sharedExpertIntermediateSize = 16
        args.moeIntermediateSize = 16
        args.mropeSection = [2, 1, 1]
        args.partialRotaryFactor = 0.25
        args.mtpNumHiddenLayers = 1
        let target = Qwen4ExpTextModel(args)
        let assistant = try Qwen4ExpInlineMTPAssistant(configuration: args,
            blockSize: 3, target: target, verificationMode: .rectangularExact)
        // Drain random initializer graphs before accumulating test forwards.
        eval(target.parameters(), assistant.parameters())
        return assistant
    }

    private func observation() -> CBv2MTPCommittedTargetObservation {
        .init(tokens: MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4]),
            hidden: MLXArray((0..<256).map { Float($0) / 100 }, [1, 4, 64]).asType(.bfloat16))
    }

    private func draft(_ assistant: Qwen4ExpInlineMTPAssistant, _ state: any CBv2MTPRequestState,
                       seed: Int32 = 5, carry: MLXArray? = nil)
        -> (tokens: [Int32], hidden: [UInt32])
    {
        let output = assistant.draftStep(tokens: MLXArray([seed]).reshaped([1, 1]),
            hidden: carry ?? observation().hidden[0..., 3..<4, 0...], shortlist: nil, requestState: state)
        eval([output.tokens, output.hidden] + assistant.evaluationTargets(for: state))
        return (output.tokens.asArray(Int32.self), output.hidden.asArray(Float.self).map(\.bitPattern))
    }

    @Test func restoredHistoryMatchesColdDraftAndDoesNotShareMutableCache() throws {
        let assistant = try fixture()
        let donor = assistant.makeRequestState()
        let observed = observation()
        assistant.observeCommittedTarget(observed, requestState: donor)
        let checkpoint = try #require(assistant.capturePrefixCheckpoint(requestState: donor, targetInputCount: 4))
        eval(checkpoint.evaluationTargets)
        let warm = try #require(assistant.restorePrefixCheckpoint(checkpoint))
        let sibling = try #require(assistant.restorePrefixCheckpoint(checkpoint))
        #expect(warm.committedInputCount == 3 && sibling.committedInputCount == 3)
        let expected = draft(assistant, donor)
        let actual = draft(assistant, warm)
        #expect(expected.tokens == actual.tokens && expected.hidden == actual.hidden)
        #expect(sibling.stagedInputCount == 0 && sibling.committedInputCount == 3)
        #expect(assistant.capturePrefixCheckpoint(requestState: warm, targetInputCount: 4) == nil)
        assistant.discardRound(requestState: donor)
        assistant.discardRound(requestState: warm)
        // Discard drops speculative head writes, but preserves the canonical
        // seed transition (token 5). The next round must supply NEW token 6,
        // not replay token 5 and accidentally test duplicate input history.
        let reference = assistant.makeRequestState()
        assistant.observeCommittedTarget(observed, requestState: reference)
        let nextHidden = MLXArray.ones([1, 1, 64], dtype: .bfloat16) * MLXArray(Float(0.5)).asType(.bfloat16)
        assistant.observeCommittedTarget(.init(tokens: MLXArray([Int32(5)]).reshaped([1, 1]),
            hidden: nextHidden), requestState: reference)
        #expect(warm.committedInputCount == reference.committedInputCount)
        let next = draft(assistant, warm, seed: 6, carry: nextHidden)
        let referenceNext = draft(assistant, reference, seed: 6, carry: nextHidden)
        #expect(next.tokens == referenceNext.tokens && next.hidden == referenceNext.hidden)
        for state in [donor, warm, sibling, reference] {
            assistant.discardRound(requestState: state)
            assistant.releaseRequestState(state)
            #expect(state.materializedBytes == 0)
        }
        #expect(assistant.capturePrefixCheckpoint(requestState: donor, targetInputCount: 4) == nil)
    }

    @Test func serializedHistoryRebindsToFreshAssistantWithoutRecomputingIt() throws {
        let assistant = try fixture()
        let donor = assistant.makeRequestState()
        assistant.observeCommittedTarget(observation(), requestState: donor)
        let checkpoint = try #require(assistant.capturePrefixCheckpoint(requestState: donor, targetInputCount: 4))
        let arrays = try #require(assistant.encodePrefixCheckpoint(checkpoint))
        let payload = arrays.map { ($0.asData().data, $0.shape, $0.dtype) }
        assistant.releaseRequestState(donor)
        let other = try fixture()
        #expect(other.restorePrefixCheckpoint(checkpoint) == nil)
        #expect(other.encodePrefixCheckpoint(checkpoint) == nil)
        #expect(other.prefixCheckpointCodecID == assistant.prefixCheckpointCodecID)
        let imported = payload.map { MLXArray($0.0, $0.1, dtype: $0.2) }
        eval(imported)
        let decoded = try #require(other.decodePrefixCheckpoint(tensors: imported, prefixTokens: [1, 2, 3, 4]))
        let warm = try #require(other.restorePrefixCheckpoint(decoded))
        let cold = other.makeRequestState()
        other.observeCommittedTarget(observation(), requestState: cold)
        let expected = draft(other, cold), actual = draft(other, warm)
        #expect(actual.tokens == expected.tokens && actual.hidden == expected.hidden)
        for state in [warm, cold] {
            other.discardRound(requestState: state)
            other.releaseRequestState(state)
        }
    }

    @Test func malformedForeignAndUnevaluatedHistoryIsRefused() throws {
        let assistant = try fixture()
        let state = assistant.makeRequestState()
        assistant.observeCommittedTarget(observation(), requestState: state)
        let other = try fixture()
        #expect(other.capturePrefixCheckpoint(requestState: state, targetInputCount: 4) == nil)
        #expect(assistant.capturePrefixCheckpoint(requestState: state, targetInputCount: 3) == nil)
        #expect(assistant.prefixCheckpointTensorDescriptors(targetInputCount: 1) == nil)
        #expect(assistant.prefixCheckpointTensorDescriptors(targetInputCount: 129) == nil)
        let checkpoint = try #require(assistant.capturePrefixCheckpoint(requestState: state, targetInputCount: 4))
        let arrays = try #require(assistant.encodePrefixCheckpoint(checkpoint))
        eval(arrays)
        #expect(assistant.decodePrefixCheckpoint(tensors: arrays, prefixTokens: [1, 2, 3, 9]) == nil)
        #expect(assistant.decodePrefixCheckpoint(tensors: arrays, prefixTokens: [64, 2, 3, 4]) == nil)
        #expect(assistant.decodePrefixCheckpoint(tensors: Array(arrays.dropLast()), prefixTokens: [1, 2, 3, 4]) == nil)
        let wrongDType = [arrays[0].asType(.float32), arrays[1], arrays[2]]
        #expect(assistant.decodePrefixCheckpoint(tensors: wrongDType, prefixTokens: [1, 2, 3, 4]) == nil)
        let lazy = MLX.where(MLXArray(true), arrays[1], arrays[1])
        #expect(try lazy.evaluatedBufferInfo() == nil)
        #expect(assistant.decodePrefixCheckpoint(tensors: [arrays[0], lazy, arrays[2]], prefixTokens: [1, 2, 3, 4]) == nil)
        #expect(try lazy.evaluatedBufferInfo() == nil)
        assistant.releaseRequestState(state)
    }
}
