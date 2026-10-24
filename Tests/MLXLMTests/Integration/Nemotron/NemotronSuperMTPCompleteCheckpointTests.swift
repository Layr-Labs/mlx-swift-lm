import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// The Super assistant's durable prompt-history codec on synthetic weights: a
/// checkpoint encoded to its declared tensors, carried through host bytes as a
/// complete SSD checkpoint is, decodes to the same codes and queue, and a
/// request restored from it drafts exactly as the cold one. Decode refuses a
/// foreign prompt, a code that is not an integer byte and a mismatched layout.
/// The adapter and the MTP target declare their attention KV types.
@Suite(.tags(.integration))
struct NemotronSuperMTPCompleteCheckpointTests {
    typealias Fixture = NemotronSuperMTPFixture
    typealias Tests = NemotronSuperMTPAssistantTests

    static let prompt = (0 ..< 23).map { ($0 * 3 + 7) % 60 + 1 }

    /// Each array through host bytes, as an authenticated import yields it.
    static func throughHost(_ arrays: [MLXArray]) -> [MLXArray] {
        eval(arrays)
        return arrays.map { MLXArray($0.asData(access: .copy).data, $0.shape, dtype: $0.dtype) }
    }

    @Test(arguments: [9, 16, 17, 20])
    func encodedCheckpointRestoresThroughHostBytesExactly(count: Int) throws {
        let stack = try Fixture.stack(kind: .random, depth: 2)
        let assistant = stack.assistant
        let hidden = Tests.rows(Self.prompt.count, seed: 61)
        func observe(_ state: any CBv2MTPRequestState, _ range: Range<Int>) {
            assistant.observeCommittedTarget(
                .init(
                    tokens: Tests.tokens(Array(Self.prompt[range])),
                    hidden: hidden[0..., range, 0...]),
                requestState: state)
        }
        func finish(_ state: any CBv2MTPRequestState) throws -> [MLXArray] {
            let draft = assistant.draftStep(
                tokens: Tests.tokens([5]), hidden: hidden[0..., 22 ..< 23, 0...], shortlist: nil,
                requestState: state)
            eval([draft.tokens, draft.hidden] + assistant.evaluationTargets(for: state))
            return [draft.tokens, draft.hidden]
                + (try #require(try Tests.typed(state).history)).committedCodes()
        }
        let cold = assistant.makeRequestState()
        try assistant.configureRequestState(cold, maximumSequenceLength: 64)
        observe(cold, 0 ..< 5)
        observe(cold, 5 ..< count)
        let checkpoint = try #require(
            assistant.capturePrefixCheckpoint(requestState: cold, targetInputCount: count))
        let descriptors = try #require(
            assistant.prefixCheckpointTensorDescriptors(targetInputCount: count))
        let drained = (count - 1) / 8 * 8
        let queued = (count - 1) % 8
        #expect(descriptors.count == (drained > 0 ? 2 : 0) + (queued > 0 ? 2 : 0) + 1)
        let encoded = try #require(assistant.encodePrefixCheckpoint(checkpoint))
        #expect(encoded.count == descriptors.count)
        for (array, descriptor) in zip(encoded, descriptors) {
            #expect(array.shape == descriptor.shape && array.dtype == descriptor.dtype.mlxDType)
        }
        // Widened codes cost two bytes per code; the queue keeps its types.
        #expect(
            descriptors.reduce(0) { $0 + $1.byteCount }
                == 2 * 2 * drained * 16 + queued * (Fixture.hidden * 4 + 4) + Fixture.hidden * 4)
        let decoded = try #require(
            assistant.decodePrefixCheckpoint(
                tensors: Self.throughHost(encoded),
                prefixTokens: Array(Self.prompt.prefix(count)))
                as? NemotronSuperMTPAssistant.PrefixCheckpoint)
        let original = try #require(checkpoint as? NemotronSuperMTPAssistant.PrefixCheckpoint)
        eval(original.evaluationTargets)
        Tests.exact(decoded.evaluationTargets, original.evaluationTargets)
        #expect(decoded.targetInputCount == count && decoded.historyRows == drained)

        observe(cold, count ..< 23)
        let expected = try finish(cold)
        let restored = try #require(assistant.restorePrefixCheckpoint(decoded))
        try assistant.configureRequestState(restored, maximumSequenceLength: 64)
        #expect(try Tests.typed(restored).history?.offset == drained)
        observe(restored, count ..< 23)
        Tests.exact(try finish(restored), expected)
        assistant.releaseRequestState(cold)
        assistant.releaseRequestState(restored)
    }

    @Test func decodeRefusesForeignPromptsCodesAndLayouts() throws {
        let stack = try Fixture.stack(kind: .random, depth: 2)
        let assistant = stack.assistant
        let count = 20
        let hidden = Tests.rows(Self.prompt.count, seed: 62)
        let state = assistant.makeRequestState()
        try assistant.configureRequestState(state, maximumSequenceLength: 64)
        assistant.observeCommittedTarget(
            .init(
                tokens: Tests.tokens(Array(Self.prompt.prefix(count))),
                hidden: hidden[0..., 0 ..< count, 0...]),
            requestState: state)
        let checkpoint = try #require(
            assistant.capturePrefixCheckpoint(requestState: state, targetInputCount: count))
        let encoded = Self.throughHost(try #require(assistant.encodePrefixCheckpoint(checkpoint)))
        let prefix = Array(Self.prompt.prefix(count))
        #expect(assistant.decodePrefixCheckpoint(tensors: encoded, prefixTokens: prefix) != nil)
        // A queued token that is not the prompt's own.
        var foreign = prefix
        foreign[count - 1] = (foreign[count - 1] % 60) + 1
        #expect(assistant.decodePrefixCheckpoint(tensors: encoded, prefixTokens: foreign) == nil)
        // A widened code that is not an integer byte.
        var corrupt = encoded
        corrupt[0] = Self.throughHost([encoded[0] + MLXArray(Float16(0.5))])[0]
        #expect(assistant.decodePrefixCheckpoint(tensors: corrupt, prefixTokens: prefix) == nil)
        corrupt[0] = Self.throughHost([encoded[0] + MLXArray(Float16(256))])[0]
        #expect(assistant.decodePrefixCheckpoint(tensors: corrupt, prefixTokens: prefix) == nil)
        // Another count's layout, and a missing tensor.
        #expect(
            assistant.decodePrefixCheckpoint(
                tensors: encoded, prefixTokens: Array(prefix.prefix(17))) == nil)
        #expect(
            assistant.decodePrefixCheckpoint(
                tensors: Array(encoded.dropLast()), prefixTokens: prefix) == nil)
        // No layout below two inputs.
        #expect(assistant.prefixCheckpointTensorDescriptors(targetInputCount: 1) == nil)
        assistant.releaseRequestState(state)
    }

    /// Complete checkpoints keep the attention KV at the composite's own
    /// types, which the plain adapter and the MTP target declare alike.
    @Test func adapterAndTargetDeclareTheirAttentionKVTypes() throws {
        let stack = try Fixture.stack(kind: .random, depth: 2)
        let adapter = try NemotronSuperRecurrentAdapter(
            stack.model, maximumDecodeRows: 1, exactPrefixCheckpoints: true)
        let types = try #require(adapter.cbv2CompleteCheckpointKVDTypes)
        #expect(types.count == adapter.cbv2LayerKinds.count && !types.isEmpty)
        #expect(types == stack.model.languageModel.observedAttentionKVDTypes())
        let target = try NemotronSuperMTPTextTarget(
            stack.model, maximumRequests: 1, exactPrefixCheckpoints: true)
        #expect(target.cbv2CompleteCheckpointKVDTypes == types)
        #expect(
            stack.assistant.prefixCheckpointCodecID.hasPrefix("nemotron-super-mtp-fp8-history-v1:"))
    }
}
