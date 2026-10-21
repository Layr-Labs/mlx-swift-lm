import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Request-state contract of the Super assistant on synthetic weights:
/// shifted trusted history, chained-row rewind and authoritative replay,
/// discard, the calibration fence and the bounded admission declaration.
@Suite(.tags(.integration))
struct NemotronSuperMTPAssistantTests {
    typealias Fixture = NemotronSuperMTPFixture

    static func tokens(_ values: [Int]) -> MLXArray {
        MLXArray(values.map(Int32.init)).reshaped(1, values.count)
    }

    static func rows(_ count: Int, seed: UInt64) -> MLXArray {
        MLXRandom.normal([1, count, Fixture.hidden], key: MLXRandom.key(seed))
    }

    static func typed(_ state: any CBv2MTPRequestState) throws -> NemotronSuperMTPAssistant.State {
        try #require(state as? NemotronSuperMTPAssistant.State)
    }

    static func exact(_ lhs: [MLXArray], _ rhs: [MLXArray]) {
        #expect(lhs.count == rhs.count)
        for (a, b) in zip(lhs, rhs) {
            #expect(a.shape == b.shape && arrayEqual(a, b).item(Bool.self))
        }
    }

    /// Codes a fresh history holds after exactly the authoritative pairs:
    /// chunked prompt priming, then one-row appends.
    static func reference(
        _ stack: Fixture.Stack, prompt: [Int], promptHidden: MLXArray,
        rows: [(token: MLXArray, hidden: MLXArray)]
    ) throws -> [MLXArray] {
        let assistant = stack.assistant
        let history = try NemotronSuperFP8KVHistory(
            heads: 1, headDimension: 16, capacityTokens: assistant.historyCapacity,
            dtype: .float32)
        let quote = assistant.workspace.graph
        let language = stack.model.languageModel
        let ids = tokens(Array(prompt.dropFirst()))
        for start in stride(from: 0, to: ids.dim(1), by: quote.historyChunkRows) {
            let end = min(ids.dim(1), start + quote.historyChunkRows)
            try stack.head.appendTrustedHistory(
                hidden: promptHidden[0..., start ..< end, 0...],
                embedding: language.mtpEmbedding(ids[0..., start ..< end]),
                history: history, quote: quote)
        }
        for row in rows {
            try stack.head.appendTrustedHistory(
                hidden: row.hidden, embedding: language.mtpEmbedding(row.token.reshaped(1, 1)),
                history: history, quote: quote)
        }
        eval(history.innerState())
        return history.committedCodes()
    }

    @Test func policyAndExactTargetBinding() throws {
        let stack = try Fixture.stack(kind: .random)
        let assistant = stack.assistant
        #expect(assistant.requiredVerificationMode == .serialTarget)
        #expect(assistant.maximumDraftTokens == 1 && assistant.maximumSpeculativeBatch == 1)
        #expect(!assistant.supportsTargetPrefixAcceptance)
        #expect(assistant.requiresCommittedObservationFence)
        #expect(assistant.requestStateBytesPerToken == 0)
        #expect(assistant.mtpTargetIdentity == stack.target.cbv2MTPTargetIdentity)
        #expect(stack.target.cbv2Capabilities.supportsMTP)
        let language = stack.model.languageModel
        #expect(throws: NemotronSuperMTPAssistant.Failure.depth) {
            try NemotronSuperMTPAssistant(
                target: language, head: stack.head,
                settings: .init(
                    maximumDraftTokens: 2, maximumSequenceLength: 64, backendScratchBytes: 0))
        }
        #expect(throws: NemotronSuperMTPAssistant.Failure.depth) {
            try NemotronSuperMTPAssistant(
                target: language, head: stack.head, settings: Fixture.settings(depth: 4))
        }
        #expect(throws: NemotronSuperMTPAssistant.Failure.sequenceBound) {
            try NemotronSuperMTPAssistant(
                target: language, head: stack.head,
                settings: .init(maximumSequenceLength: 0, backendScratchBytes: 0))
        }
        // A plain dense output head is supported; an affine-quantized one is not.
        let dense = NemotronHModel(Fixture.configuration())
        _ = try NemotronSuperMTPAssistant(
            target: dense, head: stack.head, settings: Fixture.settings())
        try dense.update(
            modules: .unflattened([
                (
                    "lm_head",
                    QuantizedLinear(
                        weight: MLXArray.zeros([Fixture.vocabulary, Fixture.hidden]), bias: nil,
                        groupSize: 32, bits: 4) as Module
                )
            ]), verify: .none)
        #expect(throws: NemotronSuperMTPAssistant.Failure.outputHead) {
            try NemotronSuperMTPAssistant(
                target: dense, head: stack.head, settings: Fixture.settings())
        }
    }

    @Test func observationQueuesShiftedTrustedPairs() throws {
        let stack = try Fixture.stack(kind: .random)
        let assistant = stack.assistant
        let state = assistant.makeRequestState()
        let s = try Self.typed(state)
        let hidden = Self.rows(3, seed: 1)
        assistant.observeCommittedTarget(
            .init(tokens: Self.tokens([4, 5, 6]), hidden: hidden), requestState: state)
        #expect(s.pendingTokens.map { $0.shape } == [[1, 2]])
        #expect(arrayEqual(s.pendingHidden[0], hidden[0..., 0 ..< 2, 0...]).item(Bool.self))
        #expect(arrayEqual(try #require(s.frontier), hidden[0..., 2 ..< 3, 0...]).item(Bool.self))
        let next = Self.rows(1, seed: 2)
        assistant.observeCommittedTarget(
            .init(tokens: Self.tokens([7]), hidden: next), requestState: state)
        // The previous frontier pairs with the newly observed token.
        #expect(s.pendingTokens.count == 2 && s.pendingTokens[1].asArray(Int32.self) == [7])
        #expect(arrayEqual(s.pendingHidden[1], hidden[0..., 2 ..< 3, 0...]).item(Bool.self))
        #expect(state.committedInputCount == 4)
        #expect(state.hasPendingPrefillForCostAccounting)
        assistant.releaseRequestState(state)
        #expect(state.materializedBytes == 0)
    }

    /// Depth three with every accepted-prefix length. Chained rows are dropped
    /// and accepted inputs replay with authoritative target hidden, so the
    /// history equals a fresh history built from trusted pairs only.
    @Test(arguments: [0, 1, 2, 3])
    func chainedRowsRewindAndAcceptedInputsReplayExactly(accepted: Int) throws {
        let stack = try Fixture.stack(kind: .random, depth: 3)
        let assistant = stack.assistant
        let state = assistant.makeRequestState()
        try assistant.configureRequestState(state, maximumSequenceLength: 64)
        let prompt = [3, 9, 1, 4, 22, 7, 3, 8, 30, 12, 2]
        let promptHidden = Self.rows(prompt.count, seed: 10)
        assistant.observeCommittedTarget(
            .init(tokens: Self.tokens(prompt), hidden: promptHidden), requestState: state)
        let frontier = promptHidden[0..., (prompt.count - 1)..., 0...]
        let seed = Self.tokens([5])
        var token = seed
        var hidden = frontier
        var drafts: [MLXArray] = []
        for _ in 0 ..< 3 {
            let step = assistant.draftStep(
                tokens: token, hidden: hidden, shortlist: nil, requestState: state)
            drafts.append(step.tokens.reshaped(1, 1))
            token = step.tokens.reshaped(1, 1)
            hidden = step.hidden
        }
        let roots = assistant.evaluationTargets(for: state)
        eval(roots)
        try assistant.requestStateDidFinishEvaluation(state)
        let s = try Self.typed(state)
        #expect(state.stagedInputCount == 3)
        #expect(try #require(s.history).offset == prompt.count - 1 + 3)
        let targetHidden = Self.rows(accepted + 1, seed: 20)
        let committed =
            accepted == 0
            ? MLXArray.zeros([1, 0], dtype: .int32)
            : concatenated(Array(drafts.prefix(accepted)), axis: 1)
        assistant.finalizeRound(
            requestState: state, confirmedInputTokens: 1 + accepted,
            committedDraftTokens: committed,
            committedTargetHidden: targetHidden[0..., 0 ..< accepted, 0...])
        #expect(try #require(s.history).offset == prompt.count)
        #expect(state.committedInputCount == prompt.count + 1 + accepted)
        let nextSeed = Self.tokens([accepted + 40])
        let carry = targetHidden[0..., accepted ..< (accepted + 1), 0...]
        let next = assistant.draftStep(
            tokens: nextSeed, hidden: carry, shortlist: nil, requestState: state)
        eval([next.tokens] + assistant.evaluationTargets(for: state))
        var trusted: [(token: MLXArray, hidden: MLXArray)] = [(seed, frontier)]
        for index in 0 ..< accepted {
            trusted.append((drafts[index], targetHidden[0..., index ..< (index + 1), 0...]))
        }
        trusted.append((nextSeed, carry))
        Self.exact(
            try #require(s.history).committedCodes(),
            try Self.reference(stack, prompt: prompt, promptHidden: promptHidden, rows: trusted))
        // Discarding the second round also abandons its replayed rows; the
        // accepted pairs return to the queue for the next attempt.
        assistant.discardRound(requestState: state)
        #expect(try #require(s.history).offset == prompt.count)
        #expect(s.pendingTokens.reduce(0) { $0 + $1.dim(1) } == accepted)
        assistant.releaseRequestState(state)
    }

    /// Whole canonical blocks reach the FP8 history as they are observed, in
    /// the chunks the first draft step would use, whatever the observation
    /// boundaries; only the partial block stays queued, within its fixed
    /// charge. After a round, accepted rows reach the history at the next
    /// observation one at a time, as the next draft step would write them.
    @Test(arguments: [[23], [3, 9, 1, 10], [8, 8, 7], [1, 1, 6, 15]])
    func observationDrainsTrustedRowsExactly(pieces: [Int]) throws {
        let stack = try Fixture.stack(kind: .random, depth: 2)
        let assistant = stack.assistant
        let state = assistant.makeRequestState()
        try assistant.configureRequestState(state, maximumSequenceLength: 64)
        let s = try Self.typed(state)
        let prompt = (0 ..< 23).map { ($0 * 5 + 2) % 60 + 1 }
        let promptHidden = Self.rows(prompt.count, seed: 50)
        var start = 0
        for count in pieces {
            let range = start ..< (start + count)
            assistant.observeCommittedTarget(
                .init(
                    tokens: Self.tokens(Array(prompt[range])),
                    hidden: promptHidden[0..., range, 0...]),
                requestState: state)
            start += count
            let pairs = start - 1
            #expect(try #require(s.history).offset == pairs / 8 * 8)
            #expect(s.pendingTokens.reduce(0) { $0 + $1.dim(1) } == pairs % 8)
            #expect(s.pendingHidden.reduce(0) { $0 + $1.dim(1) } == pairs % 8)
            let queued = (s.pendingTokens + s.pendingHidden).reduce(0) { $0 + $1.nbytes }
            #expect(queued <= 8 * (Fixture.hidden * 4 + 4))
            eval(assistant.evaluationTargets(for: state))
        }
        let frontier = promptHidden[0..., 22 ..< 23, 0...]
        let seed = Self.tokens([5])
        let draft = assistant.draftStep(
            tokens: seed, hidden: frontier, shortlist: nil, requestState: state)
        eval([draft.tokens] + assistant.evaluationTargets(for: state))
        try assistant.requestStateDidFinishEvaluation(state)
        #expect(try #require(s.history).offset == 23 && s.pendingTokens.isEmpty)
        let targetHidden = Self.rows(2, seed: 51)
        assistant.finalizeRound(
            requestState: state, confirmedInputTokens: 2,
            committedDraftTokens: draft.tokens.reshaped(1, 1),
            committedTargetHidden: targetHidden[0..., 0 ..< 1, 0...])
        // The trusted seed row stays; the accepted pair queues.
        #expect(try #require(s.history).offset == 23 && s.pendingTokens.count == 1)
        // A plain step's observation: the accepted pair goes to the history now.
        assistant.observeCommittedTarget(
            .init(tokens: Self.tokens([9]), hidden: Self.rows(1, seed: 52)),
            requestState: state)
        #expect(try #require(s.history).offset == 24 && s.pendingTokens.isEmpty)
        let carry = targetHidden[0..., 1 ..< 2, 0...]
        let nextSeed = Self.tokens([41])
        let next = assistant.draftStep(
            tokens: nextSeed, hidden: carry, shortlist: nil, requestState: state)
        eval([next.tokens] + assistant.evaluationTargets(for: state))
        Self.exact(
            try #require(s.history).committedCodes(),
            try Self.reference(
                stack, prompt: prompt, promptHidden: promptHidden,
                rows: [
                    (seed, frontier),
                    (draft.tokens.reshaped(1, 1), targetHidden[0..., 0 ..< 1, 0...]),
                    (nextSeed, carry),
                ]))
        assistant.discardRound(requestState: state)
        #expect(try #require(s.history).offset == 24 && s.pendingTokens.isEmpty)
        assistant.releaseRequestState(state)
    }

    /// A checkpoint holds the drained blocks' FP8 codes and the partial block,
    /// not the prompt's hidden rows; a restored request continues to the same
    /// draft and draft history as the cold one, bit for bit.
    @Test(arguments: [9, 16, 17, 20])
    func prefixCheckpointRestoresDrainedBlocksExactly(count: Int) throws {
        let stack = try Fixture.stack(kind: .random, depth: 2)
        let assistant = stack.assistant
        let prompt = (0 ..< 23).map { ($0 * 3 + 7) % 60 + 1 }
        let hidden = Self.rows(prompt.count, seed: 60)
        func observe(_ state: any CBv2MTPRequestState, _ range: Range<Int>) {
            assistant.observeCommittedTarget(
                .init(tokens: Self.tokens(Array(prompt[range])), hidden: hidden[0..., range, 0...]),
                requestState: state)
        }
        func finish(_ state: any CBv2MTPRequestState) throws -> [MLXArray] {
            let draft = assistant.draftStep(
                tokens: Self.tokens([5]), hidden: hidden[0..., 22 ..< 23, 0...], shortlist: nil,
                requestState: state)
            eval([draft.tokens, draft.hidden] + assistant.evaluationTargets(for: state))
            return [draft.tokens, draft.hidden]
                + (try #require(try Self.typed(state).history)).committedCodes()
        }
        let cold = assistant.makeRequestState()
        try assistant.configureRequestState(cold, maximumSequenceLength: 64)
        observe(cold, 0 ..< 5)
        observe(cold, 5 ..< count)
        let checkpoint = try #require(
            assistant.capturePrefixCheckpoint(requestState: cold, targetInputCount: count)
                as? NemotronSuperMTPAssistant.PrefixCheckpoint)
        eval(checkpoint.evaluationTargets)
        let drained = (count - 1) / 8 * 8
        let queued = (count - 1) % 8
        #expect(checkpoint.historyRows == drained)
        #expect((checkpoint.tokens?.dim(1) ?? 0) == queued)
        #expect(
            checkpoint.materializedBytes
                == 2 * drained * 16 + queued * (Fixture.hidden * 4 + 4) + Fixture.hidden * 4)
        observe(cold, count ..< 23)
        let expected = try finish(cold)
        let restored = try #require(assistant.restorePrefixCheckpoint(checkpoint))
        try assistant.configureRequestState(restored, maximumSequenceLength: 64)
        #expect(try Self.typed(restored).history?.offset == drained)
        observe(restored, count ..< 23)
        Self.exact(try finish(restored), expected)
        assistant.releaseRequestState(cold)
        assistant.releaseRequestState(restored)
    }

    @Test func discardRestoresQueuedHistoryAndRedraftsIdentically() throws {
        let stack = try Fixture.stack(kind: .random, depth: 2)
        let assistant = stack.assistant
        let state = assistant.makeRequestState()
        try assistant.configureRequestState(state, maximumSequenceLength: 64)
        let prompt = [2, 4, 6, 8, 10]
        let promptHidden = Self.rows(prompt.count, seed: 30)
        assistant.observeCommittedTarget(
            .init(tokens: Self.tokens(prompt), hidden: promptHidden), requestState: state)
        let frontier = promptHidden[0..., 4 ..< 5, 0...]
        func chain() throws -> [MLXArray] {
            let first = assistant.draftStep(
                tokens: Self.tokens([11]), hidden: frontier, shortlist: nil, requestState: state)
            let second = assistant.draftStep(
                tokens: first.tokens.reshaped(1, 1), hidden: first.hidden, shortlist: nil,
                requestState: state)
            eval([first.hidden, second.hidden] + assistant.evaluationTargets(for: state))
            return [first.tokens, first.hidden, second.tokens, second.hidden]
                + (try Self.typed(state)).history!.committedCodes()
        }
        let original = try chain()
        assistant.discardRound(requestState: state)
        let s = try Self.typed(state)
        #expect(state.stagedInputCount == 0 && s.history?.offset == 0)
        #expect(s.pendingTokens.count == 1 && s.frontier != nil && !s.started)
        Self.exact(try chain(), original)
        assistant.releaseRequestState(state)
    }

    @Test func refusedGraphEndsRequestAtTheFenceWithoutMutation() throws {
        let stack = try Fixture.stack(kind: .random)
        let assistant = stack.assistant
        let state = assistant.makeRequestState()
        try assistant.configureRequestState(state, maximumSequenceLength: 16)
        assistant.observeCommittedTarget(
            .init(tokens: Self.tokens([4, 5]), hidden: Self.rows(2, seed: 3)),
            requestState: state)
        let draft = assistant.draftStep(
            tokens: Self.tokens([3]), hidden: Self.rows(1, seed: 4), shortlist: nil,
            requestState: state)
        eval([draft.tokens] + assistant.evaluationTargets(for: state))
        try assistant.requestStateDidFinishEvaluation(state)
        #expect(try Self.typed(state).validity.isEmpty)
        assistant.discardRound(requestState: state)
        let s = try Self.typed(state)
        #expect(s.history?.offset == 0 && s.pendingTokens.count == 1)
        // Corrupt the output head's registered calibration after admission:
        // the fastpath wrapper then refuses to build its graph.
        let fp8 = try #require(stack.model.languageModel.lmHead as? ModelOptCalibratedLinear)
        let shape = Dictionary(uniqueKeysWithValues: fp8.parameters().flattened())["input_scale"]!
            .shape
        #expect(throws: ModelOptCalibratedLinear.Failure.invalidMetadata) {
            try fp8.update(
                parameters: .unflattened([("input_scale", MLXArray(Float.nan).reshaped(shape))]),
                verify: [.noUnusedKeys, .shapeMismatch])
        }
        let refused = assistant.draftStep(
            tokens: Self.tokens([3]), hidden: Self.rows(1, seed: 4), shortlist: nil,
            requestState: state)
        eval([refused.tokens] + assistant.evaluationTargets(for: state))
        #expect(state.stagedInputCount == 1 && s.history?.offset == 0)
        #expect(throws: NemotronSuperMTPAssistant.Failure.graphRefused) {
            try assistant.requestStateDidFinishEvaluation(state)
        }
        assistant.discardRound(requestState: state)
        #expect(s.pendingTokens.count == 1)
        assistant.releaseRequestState(state)
    }

    @Test func configurationBoundsForeignStateAndRelease() throws {
        let stack = try Fixture.stack(kind: .random, maximumSequenceLength: 32)
        let assistant = stack.assistant
        let state = assistant.makeRequestState()
        #expect(throws: NemotronSuperMTPAssistant.Failure.sequenceBound) {
            try assistant.configureRequestState(state, maximumSequenceLength: 0)
        }
        #expect(throws: NemotronSuperMTPAssistant.Failure.sequenceBound) {
            try assistant.configureRequestState(state, maximumSequenceLength: 33)
        }
        try assistant.configureRequestState(state, maximumSequenceLength: 16)
        let s = try Self.typed(state)
        #expect(s.history?.capacityTokens == 18)
        try assistant.configureRequestState(state, maximumSequenceLength: 8)
        #expect(throws: NemotronSuperMTPAssistant.Failure.sequenceBound) {
            try assistant.configureRequestState(state, maximumSequenceLength: 17)
        }
        let other = try Fixture.stack(kind: .random).assistant
        #expect(throws: NemotronSuperMTPAssistant.Failure.state) {
            try other.configureRequestState(state, maximumSequenceLength: 8)
        }
        assistant.releaseRequestState(state)
        assistant.releaseRequestState(state)
        #expect(state.materializedBytes == 0 && s.history == nil)
        #expect(throws: NemotronSuperMTPAssistant.Failure.state) {
            try assistant.configureRequestState(state, maximumSequenceLength: 8)
        }
    }

    /// The configured-window declaration (`windowRequestAllocation`), which the
    /// engine now admits split into a fixed part and per-token buffers
    /// (`NemotronSuperMTPSequenceChargeTests`).
    @Test func boundedDeclarationCoversHistoryBacklogAndRoundWorkspace() throws {
        let stack = try Fixture.stack(kind: .random, depth: 3, maximumSequenceLength: 300)
        let assistant = stack.assistant
        let limits = CBv2MTPAllocationLimits(maximumPrefillTokens: 16, maximumDraftTokens: 3)
        let spec = try #require(assistant.windowRequestAllocation(limits: limits))
        let history = try NemotronSuperFP8KVHistory(
            heads: 1, headDimension: 16, capacityTokens: assistant.historyCapacity,
            dtype: .float32)
        let resident = spec.resident.reduce(0) { $0 + $1.logicalBytes * $1.allocationCount }
        #expect(
            spec.resident.prefix(2).reduce(0) { $0 + $1.logicalBytes * $1.allocationCount }
                == history.maximumResidentBytes)
        // History, its depth + 1 copies, the partial block twice (8 rows of
        // FP32 hidden and an Int32 token) and the frontier carries.
        #expect(
            resident
                == 5 * history.maximumResidentBytes + 2 * 8 * (Fixture.hidden * 4 + 4)
                + 4 * Fixture.hidden * 4)
        let working = spec.working.reduce(0) { $0 + $1.logicalBytes * $1.allocationCount }
        #expect(working == assistant.workspace.totalBytes)
        #expect(spec.working[0].allocationCount == 3)
        for declaration in [assistant.windowRequestAllocation, assistant.boundedRequestAllocation] {
            #expect(declaration(.init(maximumPrefillTokens: 16, maximumDraftTokens: 4)) == nil)
            #expect(declaration(.init(maximumPrefillTokens: 0, maximumDraftTokens: 1)) == nil)
        }
        let shallow = try #require(
            assistant.windowRequestAllocation(
                limits: .init(maximumPrefillTokens: 16, maximumDraftTokens: 1)))
        #expect(shallow.working[0].allocationCount == 1)
        print(
            "SYNTHETIC super-mtp fixture workspace: draftStep=\(assistant.workspace.draftStepBytes) "
                + "outputHead=\(assistant.workspace.outputHeadArrayBytes) "
                + "total=\(assistant.workspace.totalBytes) resident=\(resident)")
    }
}
