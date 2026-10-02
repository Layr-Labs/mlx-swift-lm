import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {

    /// Request-state and prefix-checkpoint tests of `NemotronH35MTPAssistant`
    /// with a tiny float32 target.
    ///
    /// The older XCTest suites run the draft rounds and one checkpoint round
    /// trip. These tests drive the remaining paths: the draft-limit clamp,
    /// the cost properties, the bound checks of `configureRequestState`, a
    /// head without prompt priming, the backlog flag, foreign states and
    /// checkpoints, and the checkpoint descriptors.
    @Suite
    struct NemotronH35MTPRuntimeTests {

        static let vocabularySize = 128
        static let hiddenSize = 64

        /// One Mamba layer, hidden 64, one attention head of 64 for the MTP
        /// head, 4 experts.
        static func target() throws -> NemotronH35Model {
            let base: [String: Any] = [
                "model_type": "nemotron_h", "vocab_size": vocabularySize,
                "hidden_size": hiddenSize, "num_hidden_layers": 1,
                "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 64,
                "mamba_num_heads": 4, "mamba_head_dim": 16, "ssm_state_size": 16,
                "conv_kernel": 4, "n_groups": 2, "intermediate_size": 128,
                "moe_intermediate_size": 64, "moe_shared_expert_intermediate_size": 64,
                "n_routed_experts": 4, "num_experts_per_tok": 2,
                "layers_block_type": ["mamba"], "mamba_ssm_cache_dtype": "float32",
                "num_nextn_predict_layers": 1, "mtp_layers_block_type": ["attention", "moe"],
            ]
            let model = NemotronH35Model(
                try SyntheticModel.configuration(NemotronH35Configuration.self, base))
            SyntheticModel.randomize(model, seed: 1)
            return model
        }

        static func tokens(_ values: [Int]) -> MLXArray {
            MLXArray(values.map { Int32($0) }).reshaped(1, values.count)
        }

        /// `count` hidden rows with distinct values.
        static func hidden(_ count: Int) -> MLXArray {
            (MLXArray(0 ..< (count * hiddenSize)).asType(.float32) * 0.01)
                .reshaped(1, count, hiddenSize)
        }

        static func observe(
            _ assistant: NemotronH35MTPAssistant, _ state: any CBv2MTPRequestState,
            _ values: [Int]
        ) {
            assistant.observeCommittedTarget(
                .init(tokens: tokens(values), hidden: hidden(values.count)),
                requestState: state)
        }

        static func typed(_ state: any CBv2MTPRequestState) throws
            -> NemotronH35MTPAssistant.State
        {
            try #require(state as? NemotronH35MTPAssistant.State)
        }

        // MARK: - Properties and bounds

        @Test func draftLimitAndCostPropertiesFollowTheGeometry() throws {
            let target = try Self.target()
            // A limit outside 1...7 falls back to 7.
            for (requested, expected) in [(0, 7), (8, 7), (1, 1), (3, 3)] {
                let assistant = NemotronH35MTPAssistant(
                    target: target, maximumDraftTokens: requested)
                #expect(assistant.maximumDraftTokens == expected, "requested \(requested)")
                #expect(assistant.requestStateTokenAllocationPadding == expected + 1)
            }
            let assistant = NemotronH35MTPAssistant(target: target, maximumDraftTokens: 3)
            #expect(assistant.maximumSpeculativeBatch == 1)
            #expect(assistant.requestStateTokenGranularity == 256)
            // Logical bytes: 2 x 1 KV head x 64 x 4 + 4 x 64 x 4 + 16 = 1552.
            // That is below the floor of one 20 KiB pool segment.
            #expect(assistant.requestStateBytesPerToken == 20 * 1024)
            // Each prepare call returns a new, empty capture object.
            #expect(assistant.prepare(rows: []) !== assistant.prepare(rows: []))

            let state = try Self.typed(assistant.makeRequestState())
            #expect(state.cacheSnapshot.isEmpty)
            #expect(state.cacheOffset == 0)
            #expect(state.materializedBytes == 0)
        }

        @Test func configureRequestStateRejectsBadBounds() throws {
            let assistant = NemotronH35MTPAssistant(
                target: try Self.target(), maximumDraftTokens: 3)
            let state = assistant.makeRequestState()
            #expect(throws: CBv2KVError.self) {
                try assistant.configureRequestState(state, maximumSequenceLength: 0)
            }
            #expect(throws: CBv2KVError.self) {
                try assistant.configureRequestState(state, maximumSequenceLength: Int.max)
            }
            #expect(try Self.typed(state).paged == nil)

            // The bound includes a padding of the draft limit + 1 = 4.
            try assistant.configureRequestState(state, maximumSequenceLength: 64)
            #expect(try Self.typed(state).paged?.maximumSequenceLength == 68)
            // A smaller bound keeps the existing storage.
            try assistant.configureRequestState(state, maximumSequenceLength: 32)
            #expect(try Self.typed(state).paged?.maximumSequenceLength == 68)
            // A larger bound cannot grow the existing storage.
            #expect(throws: CBv2KVError.self) {
                try assistant.configureRequestState(state, maximumSequenceLength: 65)
            }
            assistant.releaseRequestState(state)
            #expect(state.materializedBytes == 0)
        }

        // MARK: - History

        /// Without prompt priming, rows observed before the first draft only
        /// set the frontier. The first draft then equals a draft from a state
        /// that observed nothing.
        @Test func withoutPromptPrimingTheFirstDraftIgnoresObservedHistory() throws {
            let assistant = NemotronH35MTPAssistant(
                target: try Self.target(), maximumDraftTokens: 3, primePromptHistory: false)
            let primed = assistant.makeRequestState()
            Self.observe(assistant, primed, [5, 8, 13, 21])
            let typed = try Self.typed(primed)
            #expect(typed.pendingTokens.isEmpty)
            #expect(typed.pendingHidden.isEmpty)
            #expect(typed.frontier?.shape == [1, 1, Self.hiddenSize])
            #expect(primed.committedInputCount == 4)

            let empty = assistant.makeRequestState()
            try assistant.configureRequestState(primed, maximumSequenceLength: 64)
            try assistant.configureRequestState(empty, maximumSequenceLength: 64)
            let seed = Self.tokens([34])
            let frontier = Self.hidden(1)
            let a = assistant.draftStep(
                tokens: seed, hidden: frontier, shortlist: nil, requestState: primed)
            let b = assistant.draftStep(
                tokens: seed, hidden: frontier, shortlist: nil, requestState: empty)
            eval(
                [a.tokens, a.hidden, b.tokens, b.hidden]
                    + assistant.evaluationTargets(for: primed)
                    + assistant.evaluationTargets(for: empty))
            // Only the seed row is in the head cache.
            #expect(typed.cacheOffset == 1)
            #expect(try Self.typed(empty).cacheOffset == 1)
            // Same ops on the same data: exact.
            #expect(SyntheticModel.maxAbsDifference(a.hidden, b.hidden) == 0)
            #expect(a.tokens.asArray(Int32.self) == b.tokens.asArray(Int32.self))

            assistant.discardRound(requestState: primed)
            assistant.discardRound(requestState: empty)
            assistant.releaseRequestState(primed)
            assistant.releaseRequestState(empty)
        }

        /// After the first round, a target backlog above the draft limit is
        /// counted as prefill work. An empty observation changes nothing, and
        /// a discard with no staged draft is a no-op.
        @Test func backlogAboveTheDraftLimitCountsAsPrefillWork() throws {
            let assistant = NemotronH35MTPAssistant(
                target: try Self.target(), maximumDraftTokens: 2)
            let state = assistant.makeRequestState()
            try assistant.configureRequestState(state, maximumSequenceLength: 64)
            // Not started yet.
            #expect(state.hasPendingPrefillForCostAccounting)

            assistant.discardRound(requestState: state)
            #expect(state.stagedInputCount == 0)
            #expect(try Self.typed(state).cacheOffset == 0)

            let draft = assistant.draftStep(
                tokens: Self.tokens([4]), hidden: Self.hidden(1), shortlist: nil,
                requestState: state)
            eval([draft.tokens, draft.hidden] + assistant.evaluationTargets(for: state))
            assistant.finalizeRound(
                requestState: state, confirmedInputTokens: 1,
                committedDraftTokens: MLXArray.zeros([1, 0], dtype: .int32),
                committedTargetHidden: MLXArray.zeros([1, 0, Self.hiddenSize]))
            #expect(state.committedInputCount == 1)
            #expect(!state.hasPendingPrefillForCostAccounting)

            assistant.observeCommittedTarget(
                .init(
                    tokens: MLXArray.zeros([1, 0], dtype: .int32),
                    hidden: MLXArray.zeros([1, 0, Self.hiddenSize])),
                requestState: state)
            #expect(state.committedInputCount == 1)
            #expect(try Self.typed(state).frontier?.shape == nil)

            // Two rows: one queued pair. The last row becomes the frontier.
            Self.observe(assistant, state, [5, 6])
            #expect(try Self.typed(state).pendingTokens.reduce(0) { $0 + $1.dim(1) } == 1)
            #expect(!state.hasPendingPrefillForCostAccounting)

            // Three more rows: 1 + 1 + 2 = 4 queued pairs, above the limit 2.
            Self.observe(assistant, state, [7, 8, 9])
            #expect(try Self.typed(state).pendingTokens.reduce(0) { $0 + $1.dim(1) } == 4)
            #expect(state.hasPendingPrefillForCostAccounting)
            #expect(state.committedInputCount == 6)
            assistant.releaseRequestState(state)
        }

        @Test func releaseIgnoresAStateOfAnotherAssistant() throws {
            let target = try Self.target()
            let owner = NemotronH35MTPAssistant(target: target)
            let other = NemotronH35MTPAssistant(target: target)
            let state = owner.makeRequestState()
            Self.observe(owner, state, [4, 7, 9])
            let bytes = state.materializedBytes
            #expect(bytes > 0)

            other.releaseRequestState(state)
            #expect(state.materializedBytes == bytes)
            #expect(state.committedInputCount == 3)
            #expect(try !Self.typed(state).released)

            owner.releaseRequestState(state)
            #expect(state.materializedBytes == 0)
            #expect(try Self.typed(state).released)
        }

        // MARK: - Prefix checkpoints

        @Test func prefixCheckpointCodecIDAndDescriptors() throws {
            let assistant = NemotronH35MTPAssistant(target: try Self.target())
            #expect(
                assistant.prefixCheckpointCodecID
                    == "nemotron35-mtp-paged-trusted-history-v1:h64:v128:"
                    + "\(DType.float32):kv\(DType.float32)")
            // One target input has no shifted history.
            #expect(assistant.prefixCheckpointTensorDescriptors(targetInputCount: 1) == nil)
            #expect(assistant.prefixCheckpointTensorDescriptors(targetInputCount: 0) == nil)

            let descriptors = try #require(
                assistant.prefixCheckpointTensorDescriptors(targetInputCount: 5))
            #expect(
                descriptors.map(\.role)
                    == [
                        CBv2CheckpointTensorRole.assistantHidden, .assistantTokens,
                        .assistantFrontier,
                    ])
            #expect(descriptors.map(\.shape) == [[1, 4, 64], [1, 4], [1, 1, 64]])
            #expect(
                descriptors.map(\.dtype) == [CBv2CheckpointDType.float32, .int32, .float32])
            #expect(descriptors.map(\.byteCount) == [1024, 16, 256])
        }

        @Test func prefixCheckpointRejectsAWrongOwnerCountOrToken() throws {
            let target = try Self.target()
            let assistant = NemotronH35MTPAssistant(target: target)
            let other = NemotronH35MTPAssistant(target: target)
            let state = assistant.makeRequestState()
            Self.observe(assistant, state, [5, 8, 13, 21])

            // The count must be the committed input count.
            #expect(
                assistant.capturePrefixCheckpoint(requestState: state, targetInputCount: 3) == nil)
            // Another assistant does not own the state.
            #expect(other.capturePrefixCheckpoint(requestState: state, targetInputCount: 4) == nil)

            let checkpoint = try #require(
                assistant.capturePrefixCheckpoint(requestState: state, targetInputCount: 4))
            #expect(checkpoint.targetInputCount == 4)
            // Hidden [1, 3, 64] float32 + tokens [1, 3] int32 + frontier
            // [1, 1, 64] float32 = 768 + 12 + 256 bytes.
            #expect(checkpoint.materializedBytes == 1036)
            #expect(
                checkpoint.evaluationTargets.map(\.shape) == [[1, 3, 64], [1, 3], [1, 1, 64]])

            // Another assistant cannot restore or encode the checkpoint.
            #expect(other.restorePrefixCheckpoint(checkpoint) == nil)
            #expect(other.encodePrefixCheckpoint(checkpoint) == nil)

            let encoded = try #require(assistant.encodePrefixCheckpoint(checkpoint))
            eval(encoded)
            // A prefix token outside the vocabulary.
            #expect(
                assistant.decodePrefixCheckpoint(tensors: encoded, prefixTokens: [5, 8, 13, 200])
                    == nil)
            // A prefix of 1 token has no descriptors.
            #expect(assistant.decodePrefixCheckpoint(tensors: encoded, prefixTokens: [5]) == nil)

            // The owner restores a new state with the trusted history.
            let restored = try #require(assistant.restorePrefixCheckpoint(checkpoint))
            let typed = try Self.typed(restored)
            #expect(restored.committedInputCount == 4)
            #expect(typed.pendingTokens.map(\.shape) == [[1, 3]])
            #expect(typed.pendingTokens.first?.asArray(Int32.self) == [Int32(8), 13, 21])
            #expect(typed.frontier?.shape == [1, 1, 64])
            #expect(typed.cacheOffset == 0)

            // A released state gives no checkpoint.
            assistant.releaseRequestState(state)
            #expect(
                assistant.capturePrefixCheckpoint(requestState: state, targetInputCount: 4) == nil)
            assistant.releaseRequestState(restored)

            // A state with one observed token has no shifted history.
            let single = assistant.makeRequestState()
            Self.observe(assistant, single, [5])
            #expect(
                assistant.capturePrefixCheckpoint(requestState: single, targetInputCount: 1) == nil)
            assistant.releaseRequestState(single)
        }
    }
}
