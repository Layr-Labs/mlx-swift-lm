import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// A one-shot visit gate, not a semaphore permit left borrowed at destruction.

private func withMiMoConstructionScope<Value>(
    _ body: (NativeConstructionScope) throws -> Value
) rethrows -> Value {
    let work = NativeConstructionScope()
    defer {
        // Unexpected failed completion is restart-only, including in this
        // dedicated native test process. Never deallocate its sole SDK owner.
        if work.snapshot.isRetainedFault { _ = Unmanaged.passRetained(work) }
    }
    return try body(work)
}

private final class MiMoV26SecondFenceVisit: @unchecked Sendable {
    private let lock = NSLock()
    private var visits = 0
    func shouldBlock() -> Bool {
        lock.withLock {
            guard visits < 2 else { return false }
            visits += 1
            return visits == 2
        }
    }
}

private final class MiMoV26MTPReadoutSpy: Linear {
    var shapes: [[Int]] = []
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        shapes.append(x.shape)
        return super.callAsFunction(x)
    }
}

/// Test-only proposal override. Every native head still executes and advances
/// its own request state. A target continuation controls exactly where the
/// generic verifier accepts/rejects, independently of random tiny-head quality.
private final class MiMoV26ScriptedAssistant: CBv2MTPRequestStatefulDrafter {
    let native: MiMoV26MTPAssistant
    let script: [Int]?
    let promptLength: Int
    let rejectPosition: Int
    var states: [MiMoV26MTPState] = []
    var observations = 0
    var observationFenceReads = 0
    var fenceHook: ((MiMoV26MTPState) throws -> Void)?
    init(
        _ native: MiMoV26MTPAssistant, script: [Int]? = nil,
        promptLength: Int = 0, rejectPosition: Int = 3
    ) {
        self.native = native
        self.script = script
        self.promptLength = promptLength
        self.rejectPosition = rejectPosition
    }
    var mtpTargetIdentity: ObjectIdentifier? { native.mtpTargetIdentity }
    var requiredVerificationMode: CBv2MTPVerificationMode? { .serialTarget }
    var maximumDraftTokens: Int? { 3 }
    var maximumSpeculativeBatch: Int? { 1 }
    var requiresCommittedObservationFence: Bool { true }
    var requestStateBytesPerToken: Int { native.requestStateBytesPerToken }
    var requestStateTokenGranularity: Int { native.requestStateTokenGranularity }
    var requestStateTokenAllocationPadding: Int { native.requestStateTokenAllocationPadding }
    func makeRequestState() -> any CBv2MTPRequestState {
        let state = native.makeRequestState() as! MiMoV26MTPState
        states.append(state)
        return state
    }
    func configureRequestState(_ state: any CBv2MTPRequestState, maximumSequenceLength: Int) throws
    {
        try native.configureRequestState(state, maximumSequenceLength: maximumSequenceLength)
    }
    func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState
    ) {
        observations += 1
        native.observeCommittedTarget(observation, requestState: requestState)
    }
    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture { native.prepare(rows: rows) }
    func draftStep(tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture)
        -> (tokens: MLXArray, hidden: MLXArray)
    {
        native.draftStep(tokens: tokens, hidden: hidden, prepared: prepared)
    }
    func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        let result = native.draftStep(
            tokens: tokens, hidden: hidden, shortlist: shortlist,
            requestState: requestState)
        guard let script else { return result }
        let state = requestState as! MiMoV26MTPState
        let depth = state.stagedInputCount - 1
        let index = state.observedCount - promptLength + depth + 1
        precondition(script.indices.contains(index))
        let token = (script[index] + (depth == rejectPosition ? 1 : 0)) % 32
        return (MLXArray([Int32(token)]), result.hidden)
    }
    func evaluationTargets(for state: any CBv2MTPRequestState) -> [MLXArray] {
        if state.stagedInputCount == 0 { observationFenceReads += 1 }
        return native.evaluationTargets(for: state)
    }
    func requestStateDidFinishEvaluation(_ state: any CBv2MTPRequestState) throws {
        if let state = state as? MiMoV26MTPState { try fenceHook?(state) }
        try native.requestStateDidFinishEvaluation(state)
    }
    func finalizeRound(
        requestState: any CBv2MTPRequestState, confirmedInputTokens: Int,
        committedDraftTokens: MLXArray, committedTargetHidden: MLXArray
    ) {
        native.finalizeRound(
            requestState: requestState, confirmedInputTokens: confirmedInputTokens,
            committedDraftTokens: committedDraftTokens, committedTargetHidden: committedTargetHidden
        )
    }
    func discardRound(requestState: any CBv2MTPRequestState) {
        native.discardRound(requestState: requestState)
    }
    func releaseRequestState(_ state: any CBv2MTPRequestState) { native.releaseRequestState(state) }
}

final class MiMoV26MTPEngineTests: XCTestCase {
    private func ids(_ values: [Int]) -> MLXArray {
        MLXArray(values.map(Int32.init), [1, values.count])
    }
    private func prompt(_ length: Int) -> [Int] { (0 ..< length).map { 1 + ($0 * 7) % 31 } }
    private func state(_ assistant: MiMoV26MTPAssistant) throws -> MiMoV26MTPState {
        let state = assistant.makeRequestState() as! MiMoV26MTPState
        try assistant.configureRequestState(state, maximumSequenceLength: 64)
        return state
    }
    private func observe(
        _ assistant: MiMoV26MTPAssistant, _ state: MiMoV26MTPState,
        _ tokens: MLXArray, _ hidden: MLXArray
    ) {
        assistant.observeCommittedTarget(.init(tokens: tokens, hidden: hidden), requestState: state)
        eval(assistant.evaluationTargets(for: state))
        XCTAssertNoThrow(try assistant.requestStateDidFinishEvaluation(state))
    }
    private func propose(
        _ assistant: MiMoV26MTPAssistant, _ state: MiMoV26MTPState,
        seed: Int, hidden: MLXArray, depth: Int = 3
    ) -> [Int] {
        var token = ids([seed])
        var feature = hidden
        var result: [Int] = []
        for _ in 0 ..< depth {
            let output = assistant.draftStep(
                tokens: token, hidden: feature, shortlist: nil, requestState: state)
            eval([output.tokens, output.hidden] + assistant.evaluationTargets(for: state))
            XCTAssertNoThrow(try assistant.requestStateDidFinishEvaluation(state))
            result.append(Int(output.tokens.item(Int32.self)))
            token = output.tokens.reshaped([1, 1])
            feature = output.hidden
        }
        return result
    }

    func testChunkedPrimingAndEveryDepthMatchScalarNonChainingReference() throws {
        let (target, mtp) = try MiMoV26MTPChecks.fixture()
        let assistant = try MiMoV26MTPAssistant(target: target, predictor: mtp)
        for length in [1, 2, 3, 11, 31] {
            let tokens = prompt(length)
            let output = try target.forward(inputIDs: ids(tokens))
            let allHidden = output.normalizedHiddenStates.asArray(Float.self)
            let scalarHidden = (0 ..< length).map { Array(allHidden[$0 * 4 ..< ($0 + 1) * 4]) }
            for chunk in [1, 2, 7, 64] {
                let state = try state(assistant)
                for start in stride(from: 0, to: length, by: chunk) {
                    let end = min(length, start + chunk)
                    observe(
                        assistant, state, ids(Array(tokens[start ..< end])),
                        output.normalizedHiddenStates[0..., start ..< end, 0...])
                    XCTAssertLessThanOrEqual(state.retainedFeatureRows, 3)
                    for shapes in state.retainedHistoryShapes {
                        for shape in shapes { XCTAssertLessThanOrEqual(shape[2], 3) }
                    }
                }
                XCTAssertEqual(state.headInputCounts, (0 ..< 3).map { max(0, length - $0 - 1) })
                var chain = tokens + [7]
                var hidden = output.normalizedHiddenStates[0..., (-1)..., 0...]
                for depth in 0 ..< 3 {
                    let proposal = assistant.draftStep(
                        tokens: ids([chain.last!]), hidden: hidden,
                        shortlist: nil, requestState: state)
                    eval(
                        [proposal.tokens, proposal.hidden] + assistant.evaluationTargets(for: state)
                    )
                    try assistant.requestStateDidFinishEvaluation(state)
                    let expected = MiMoV26MTPChecks.reference(
                        mtp.layers[depth], hidden: scalarHidden,
                        tokens: Array(chain[(depth + 1)...]), target: target,
                        firstTokenPosition: depth + 1)
                    let lastLogits = Array(expected.logits.suffix(32))
                    let expectedToken = lastLogits.indices.max { lastLogits[$0] < lastLogits[$1] }!
                    XCTAssertEqual(
                        Int(proposal.tokens.item(Int32.self)), expectedToken,
                        "length=\(length) chunk=\(chunk) depth=\(depth)")
                    try MiMoV26MTPChecks.near(
                        proposal.hidden,
                        output.normalizedHiddenStates[0..., (-1)..., 0...], "target feature reuse")
                    chain.append(Int(proposal.tokens.item(Int32.self)))
                    hidden = proposal.hidden
                }
                XCTAssertEqual(state.headProposalCounts, [1, 1, 1])
                assistant.discardRound(requestState: state)
                XCTAssertEqual(state.headInputCounts, (0 ..< 3).map { max(0, length - $0 - 1) })
                assistant.releaseRequestState(state)
                XCTAssertEqual(state.materializedBytes, 0)
            }
        }
    }

    func testAcceptedPrefixesRollbackWrapAndCarryFallbackMatchTrustedReplay() throws {
        let (target, mtp) = try MiMoV26MTPChecks.fixture()
        let assistant = try MiMoV26MTPAssistant(target: target, predictor: mtp)
        for accepted in 0 ... 3 {
            let tokens = prompt(11)
            let targetCache = target.newCache()
            let initial = try target.forward(inputIDs: ids(tokens), cache: targetCache)
            let native = try state(assistant)
            let oracle = try state(assistant)
            observe(assistant, native, ids(tokens), initial.normalizedHiddenStates)
            observe(assistant, oracle, ids(tokens), initial.normalizedHiddenStates)
            let seed = Int(argMax(initial.logits[0, -1], axis: -1).item(Int32.self))
            let drafts = propose(
                assistant, native, seed: seed,
                hidden: initial.normalizedHiddenStates[0..., (-1)..., 0...])
            var outputs: [MiMoV26TextOutput] = []
            for token in [seed] + drafts.prefix(accepted) {
                let output = try target.forward(inputIDs: ids([token]), cache: targetCache)
                eval(output.normalizedHiddenStates, output.logits)
                outputs.append(output)
            }
            let acceptedHidden =
                accepted == 0
                ? MLXArray.zeros([1, 0, 4])
                : concatenated(outputs.prefix(accepted).map(\.normalizedHiddenStates), axis: 1)
            assistant.finalizeRound(
                requestState: native, confirmedInputTokens: accepted + 1,
                committedDraftTokens: ids(Array(drafts.prefix(accepted))),
                committedTargetHidden: acceptedHidden)
            let canonical = [seed] + drafts.prefix(accepted)
            observe(
                assistant, oracle, ids(Array(canonical)),
                concatenated(outputs.map(\.normalizedHiddenStates), axis: 1))
            XCTAssertEqual(native.committedInputCount, oracle.committedInputCount)
            let last = outputs.last!
            let next = Int(argMax(last.logits[0, -1], axis: -1).item(Int32.self))
            // A separate admitted request exercises fallback from the same
            // canonical prefix. No unpriced public snapshot owner is exported.
            let fallback = try state(assistant)
            observe(assistant, fallback, ids(tokens), initial.normalizedHiddenStates)
            XCTAssertEqual(
                propose(
                    assistant, fallback, seed: seed,
                    hidden: initial.normalizedHiddenStates[0..., (-1)..., 0...]), drafts)
            assistant.finalizeRound(
                requestState: fallback, confirmedInputTokens: accepted + 1,
                committedDraftTokens: ids(Array(drafts.prefix(accepted))),
                committedTargetHidden: acceptedHidden)
            let actual = propose(assistant, native, seed: next, hidden: last.normalizedHiddenStates)
            let expected = propose(
                assistant, oracle, seed: next, hidden: last.normalizedHiddenStates)
            XCTAssertEqual(actual, expected, "accepted prefix \(accepted)")
            XCTAssertEqual(native.headInputCounts, oracle.headInputCounts)
            assistant.discardRound(requestState: native)
            assistant.discardRound(requestState: oracle)
            // Verification-to-plain fallback emits a historyCarry callback,
            // followed by the ordinary target input observation.
            observe(assistant, fallback, ids([next]), last.normalizedHiddenStates)
            let nextOutput = try target.forward(inputIDs: ids([next]), cache: targetCache)
            observe(assistant, fallback, ids([next]), nextOutput.normalizedHiddenStates)
            observe(assistant, oracle, ids([next]), nextOutput.normalizedHiddenStates)
            let nextSeed = Int(argMax(nextOutput.logits[0, -1], axis: -1).item(Int32.self))
            XCTAssertEqual(
                propose(
                    assistant, fallback, seed: nextSeed, hidden: nextOutput.normalizedHiddenStates),
                propose(
                    assistant, oracle, seed: nextSeed, hidden: nextOutput.normalizedHiddenStates))
            for state in [native, oracle, fallback] { assistant.releaseRequestState(state) }
        }
    }

    private func measuredBacking(_ assistant: MiMoV26MTPAssistant, _ state: MiMoV26MTPState) throws
        -> Int
    {
        try assistant.evaluationTargets(for: state).reduce(0) { total, array in
            let info = try XCTUnwrap(array.evaluatedBufferInfo())
            return total + info.allocatedBytes
        }
    }

    func testRequestIsolationMeasuredOwnershipAndRelease() throws {
        let (target, mtp) = try MiMoV26MTPChecks.fixture()
        let assistant = try MiMoV26MTPAssistant(target: target, predictor: mtp)
        let first = try state(assistant)
        let second = try state(assistant)
        let output = try target.forward(inputIDs: ids(prompt(20)))
        observe(assistant, first, ids(prompt(20)), output.normalizedHiddenStates)
        observe(assistant, second, ids(prompt(20)), output.normalizedHiddenStates)
        XCTAssertFalse(first.hasUnmeasuredResidency)
        XCTAssertEqual(first.materializedBytes, try measuredBacking(assistant, first))
        let before = second.materializedBytes
        let counters = second.headInputCounts
        let knownBeforeDraft = first.materializedBytes
        let one = assistant.draftStep(
            tokens: ids([7]), hidden: output.normalizedHiddenStates[0..., (-1)..., 0...],
            shortlist: nil, requestState: first)
        XCTAssertTrue(first.hasUnmeasuredResidency)
        XCTAssertEqual(
            first.materializedBytes, knownBeforeDraft,
            "prior frozen owners retire only at successor fence")
        eval([one.tokens, one.hidden] + assistant.evaluationTargets(for: first))
        try assistant.requestStateDidFinishEvaluation(first)
        XCTAssertFalse(first.hasUnmeasuredResidency)
        XCTAssertEqual(first.materializedBytes, try measuredBacking(assistant, first))
        XCTAssertEqual(second.materializedBytes, before)
        XCTAssertEqual(second.headInputCounts, counters)
        let beforeDiscard = first.materializedBytes
        assistant.discardRound(requestState: first)
        XCTAssertTrue(first.hasUnmeasuredResidency)
        XCTAssertEqual(
            first.materializedBytes, beforeDiscard,
            "discard cannot refund retained measured contexts")
        eval(assistant.evaluationTargets(for: first))
        try assistant.requestStateDidFinishEvaluation(first)
        let expected = propose(
            assistant, first, seed: 7, hidden: output.normalizedHiddenStates[0..., (-1)..., 0...])
        assistant.discardRound(requestState: first)
        XCTAssertEqual(
            propose(
                assistant, first, seed: 7,
                hidden: output.normalizedHiddenStates[0..., (-1)..., 0...]), expected)
        // Prefix checkpoints go only through the priced historical codec.
        XCTAssertTrue(assistant is any CBv2HistoricalMTPPrefixCheckpointCoding)
        assistant.releaseRequestState(first)
        assistant.releaseRequestState(first)
        XCTAssertTrue(first.isReleased)
        XCTAssertEqual(first.materializedBytes, 0)
        XCTAssertEqual(first.measuredRootCount, 0)
        XCTAssertFalse(first.hasUnmeasuredResidency)
        XCTAssertTrue(assistant.evaluationTargets(for: first).isEmpty)
        XCTAssertThrowsError(try assistant.requestStateDidFinishEvaluation(first))
        assistant.releaseRequestState(second)
    }

    func testFenceRejectsDuplicateAndAliasedBackingWithoutInventedIdentity() throws {
        let (target, mtp) = try MiMoV26MTPChecks.fixture()
        let assistant = try MiMoV26MTPAssistant(target: target, predictor: mtp)
        let tokens = prompt(11)
        let output = try target.forward(inputIDs: ids(tokens))
        for duplicateDescriptor in [true, false] {
            let candidate = try state(assistant)
            observe(assistant, candidate, ids(tokens), output.normalizedHiddenStates)
            candidate.willMutate()
            if duplicateDescriptor {
                candidate.pendingHidden = candidate.tail
            } else {
                // Different MLX descriptors, same backing. Shape/extents alone
                // cannot justify counting both allocations.
                candidate.tail = candidate.cache!.innerState()[0].reshaped([1, 3, 4])
            }
            eval(assistant.evaluationTargets(for: candidate))
            XCTAssertThrowsError(try assistant.requestStateDidFinishEvaluation(candidate))
            XCTAssertTrue(candidate.hasUnmeasuredResidency)
            assistant.releaseRequestState(candidate)
            XCTAssertEqual(candidate.materializedBytes, 0)
            XCTAssertEqual(candidate.measuredRootCount, 0)
        }
    }

    func testNamedReloadAndExplicitSessionInvalidationFailClosed() throws {
        let (target, mtp) = try MiMoV26MTPChecks.fixture()
        let assistant = try MiMoV26MTPAssistant(target: target, predictor: mtp)
        let adapter = try MiMoV26CBv2Adapter(target: target, assistant: assistant)
        let existing = try state(assistant)
        XCTAssertTrue(adapter.cbv2Capabilities.supportsMTP)
        try mtp.loadConvertedWeights(MiMoV26MTPChecks.fixtureWeights(mtp, prefix: "mtp."))
        XCTAssertNil(assistant.mtpTargetIdentity)
        XCTAssertFalse(adapter.cbv2Capabilities.supportsMTP)
        XCTAssertThrowsError(
            try assistant.configureRequestState(existing, maximumSequenceLength: 64))
        assistant.releaseRequestState(existing)

        let newAssistant = try MiMoV26MTPAssistant(target: target, predictor: mtp)
        let valid = try state(newAssistant)
        XCTAssertEqual(newAssistant.mtpTargetIdentity, ObjectIdentifier(target))
        let (replacement, replacementMTP) = try MiMoV26MTPChecks.fixture()
        XCTAssertThrowsError(try MiMoV26MTPAssistant(target: replacement, predictor: mtp))
        XCTAssertThrowsError(try MiMoV26CBv2Adapter(target: replacement, assistant: newAssistant))
        newAssistant.releaseRequestState(valid)
        newAssistant.invalidateLoadedSession()
        // Generic parent/child/target Module mutation is unsupported while a
        // session is active. A host that performs it must invalidate first; the
        // old assistant never becomes valid again after such updates.
        try mtp.update(parameters: mtp.parameters(), verify: .all)
        try mtp.layers[0].update(parameters: mtp.layers[0].parameters(), verify: .all)
        try target.update(parameters: target.parameters(), verify: .all)
        XCTAssertNil(newAssistant.mtpTargetIdentity)
        XCTAssertThrowsError(try MiMoV26CBv2Adapter(target: target, assistant: newAssistant))
        let fresh = try MiMoV26MTPAssistant(target: replacement, predictor: replacementMTP)
        XCTAssertEqual(fresh.mtpTargetIdentity, ObjectIdentifier(replacement))
    }

    func testAffine4Group64BF16AssistantUsesAllThreeNativeHeads() throws {
        let target = try MiMoV26TextModel(MiMoV26MTPChecks.affineConfiguration())
        try target.update(
            parameters: .unflattened(
                MiMoV26MTPChecks.fixtureWeights(target)
                    .mapValues { $0.asType(.bfloat16) }), verify: .all)
        let mtp = try MiMoV26MTP(target: target)
        try mtp.loadConvertedWeights(MiMoV26MTPChecks.packedFixture(mtp).stored)
        let assistant = try MiMoV26MTPAssistant(target: target, predictor: mtp)
        let state = try state(assistant)
        let tokens = prompt(9)
        let output = try target.forward(inputIDs: ids(tokens))
        observe(assistant, state, ids(tokens), output.normalizedHiddenStates)
        _ = propose(
            assistant, state, seed: 7, hidden: output.normalizedHiddenStates[0..., (-1)..., 0...])
        XCTAssertEqual(state.headProposalCounts, [1, 1, 1])
        XCTAssertEqual(state.retainedFeatureRows, 3)
        for depth in state.retainedHistoryShapes {
            for shape in depth { XCTAssertLessThanOrEqual(shape[2], 3) }
        }
        assistant.releaseRequestState(state)
        XCTAssertEqual(state.materializedBytes, 0)
    }

    func testCopyPreservesFloatingBitsAndDetachesLargeParent() throws {
        for dtype: DType in [.float32, .bfloat16] {
            let data: Data
            if dtype == .float32 {
                let bits: [UInt32] = [
                    0, 0x8000_0000, 1, 0x007f_ffff, 0x7fc0_1234, 0x7f80_0000, 0xff80_0000,
                ]
                data = (0 ..< 36).map { bits[$0 % bits.count] }.withUnsafeBufferPointer {
                    Data(buffer: $0)
                }
            } else {
                let bits: [UInt16] = [0, 0x8000, 1, 0x007f, 0x7fc1, 0x7f80, 0xff80]
                data = (0 ..< 36).map { bits[$0 % bits.count] }.withUnsafeBufferPointer {
                    Data(buffer: $0)
                }
            }
            let parent = MLXArray(data, [1, 9, 4], dtype: dtype)
            let selected = parent[0..., 6 ..< 9, 0...]
            let copy = mimoV26MTPCopy(selected)
            eval(copy)
            StreamOrDevice.default.stream.synchronize()
            let info = try XCTUnwrap(copy.evaluatedBufferInfo())
            XCTAssertTrue(info.isUnique)
            XCTAssertTrue(info.isRowContiguous)
            XCTAssertEqual(info.dataOffset, 0)
            XCTAssertEqual(info.dataElements, 12)
            XCTAssertLessThanOrEqual(
                info.allocatedBytes,
                try Memory.allocationFootprintUpperBound(byteCount: 12 * dtype.size))
            XCTAssertEqual(copy.asData(access: .copy).data, Data(data.suffix(12 * dtype.size)))
        }
    }

    func testOfficialKVWidthsResidencyAtWindowAndChunkBoundaries() throws {
        for dtype: DType in [.float32, .bfloat16] {
            var fields =
                try JSONSerialization.jsonObject(
                    with: JSONEncoder().encode(MiMoV26MTPChecks.config())) as! [String: Any]
            fields["max_position_embeddings"] = 256
            fields["sliding_window"] = 128
            fields["sliding_window_size"] = 128
            fields["head_dim"] = 192
            fields["swa_head_dim"] = 192
            fields["v_head_dim"] = 128
            fields["swa_v_head_dim"] = 128
            fields["dtype"] = dtype == .float32 ? "float32" : "bfloat16"
            fields["moe_router_dtype"] = fields["dtype"]
            let config = try JSONDecoder().decode(
                MiMoV26Configuration.self,
                from: JSONSerialization.data(withJSONObject: fields))
            let target = try MiMoV26TextModel(config)
            try target.update(
                parameters: .unflattened(
                    MiMoV26MTPChecks.fixtureWeights(target)
                        .mapValues { $0.asType(dtype) }), verify: .all)
            let mtp = try MiMoV26MTP(target: target)
            try mtp.loadConvertedWeights(
                MiMoV26MTPChecks.fixtureWeights(mtp, prefix: "mtp.")
                    .mapValues { $0.asType(dtype) })
            let assistant = try MiMoV26MTPAssistant(target: target, predictor: mtp)
            eval(target, mtp)
            for length in [1, 2, 3, 127, 128, 129] {
                let tokens = prompt(length)
                let targetOutput = try target.forward(inputIDs: ids(tokens))
                eval(targetOutput.normalizedHiddenStates)
                for chunk in [1, 64, 129] {
                    let state = assistant.makeRequestState() as! MiMoV26MTPState
                    try assistant.configureRequestState(state, maximumSequenceLength: 256)
                    for start in stride(from: 0, to: length, by: chunk) {
                        let end = min(length, start + chunk)
                        let previous = state.materializedBytes
                        assistant.observeCommittedTarget(
                            .init(
                                tokens: ids(Array(tokens[start ..< end])),
                                hidden: targetOutput.normalizedHiddenStates[
                                    0..., start ..< end, 0...]), requestState: state)
                        XCTAssertTrue(state.hasUnmeasuredResidency)
                        XCTAssertEqual(state.materializedBytes, previous)
                        eval(assistant.evaluationTargets(for: state))
                        try assistant.requestStateDidFinishEvaluation(state)
                        XCTAssertFalse(state.hasUnmeasuredResidency)
                        XCTAssertEqual(
                            state.materializedBytes, try measuredBacking(assistant, state))
                        XCTAssertEqual(
                            state.measuredRootCount, assistant.evaluationTargets(for: state).count)
                    }
                    print(
                        "MTP_RESIDENCY dtype=\(dtype) prompt=\(length) chunk=\(chunk) measured=\(state.materializedBytes) logical=\(state.logicalRootBytes) active=\(Memory.activeMemory) cache=\(Memory.cacheMemory)"
                    )
                    assistant.releaseRequestState(state)
                    XCTAssertEqual(state.materializedBytes, 0)
                    XCTAssertEqual(state.measuredRootCount, 0)
                    XCTAssertTrue(assistant.evaluationTargets(for: state).isEmpty)
                }
            }
        }
    }

    func testAdapterExplicitLoadedOwnerAndHiddenPrefillReadoutContract() throws {
        let (target, mtp) = try MiMoV26MTPChecks.fixture()
        let unloaded = try MiMoV26MTP(target: target)
        XCTAssertThrowsError(try MiMoV26MTPAssistant(target: target, predictor: unloaded))
        let native = try MiMoV26MTPAssistant(target: target, predictor: mtp)
        let otherTarget = try MiMoV26TextModel(MiMoV26MTPChecks.config())
        XCTAssertThrowsError(try MiMoV26CBv2Adapter(target: otherTarget, assistant: native))
        XCTAssertFalse(try MiMoV26CBv2Adapter(target: target).cbv2Capabilities.supportsMTP)
        let adapter = try MiMoV26CBv2Adapter(target: target, assistant: native)
        XCTAssertTrue(adapter.cbv2Capabilities.supportsMTP)
        XCTAssertTrue(adapter.supportsRequestStatefulMTP)
        XCTAssertNil(adapter.mtpCaptureLayers)
        _ = try withMiMoConstructionScope { constructionWork in
            try adapter.probeNativeKVTypes(retaining: constructionWork)
        }
        let backend = try adapter.makeBackend(bytesCapacity: 1 << 20)
        let rows = try backend.makeSequenceState(
            layerKinds: adapter.layerKinds, promptLength: 9, maxLength: 32)
        let caches = adapter.makeCaches()
        try adapter.bindRows([rows], caches: caches)
        let spy = MiMoV26MTPReadoutSpy(4, 32, bias: false)
        try spy.update(parameters: target.lmHead!.parameters(), verify: .all)
        target.update(modules: ModuleChildren.unflattened([("lm_head", spy as Module)]))
        let requestState = try state(native)
        let first = adapter.forwardWithHiddenForPrefill(
            tokens: ids(prompt(6)), caches: caches,
            requirement: .evaluationOnly)
        observe(native, requestState, ids(prompt(6)), first.lastHidden)
        eval(first.logits)
        XCTAssertEqual(first.logits.shape, [1, 1])
        XCTAssertTrue(
            spy.shapes.isEmpty, "prompt priming constructed target or predictor vocabulary logits")
        XCTAssertEqual(first.lastHidden.shape, [1, 6, 4])
        let final = adapter.forwardWithHiddenForPrefill(
            tokens: ids([2, 3, 4]), caches: caches,
            requirement: .lastPositionLogits)
        observe(native, requestState, ids([2, 3, 4]), final.lastHidden)
        eval(final.logits)
        XCTAssertEqual(final.logits.shape, [1, 32])
        XCTAssertEqual(spy.shapes, [[1, 4]])
        let reference = try target.forward(inputIDs: ids(prompt(6) + [2, 3, 4]))
        try MiMoV26MTPChecks.near(
            final.lastHidden, reference.normalizedHiddenStates[0..., 6..., 0...],
            "CBv2 post-norm features")
        try MiMoV26MTPChecks.near(
            final.logits, reference.logits[0..., -1, 0...], "CBv2 prefill logits")
        native.releaseRequestState(requestState)
        backend.release(rows)
        XCTAssertEqual(backend.bytesInUse, 0)
    }

    private func engine(
        target: MiMoV26TextModel, assistant: MiMoV26MTPAssistant?,
        drafter: (any CBv2MTPDrafter)? = nil, chunk: Int = 3, depth: Int = 3
    )
        throws -> (EngineV2, MiMoV26CBv2Backend)
    {
        let adapter = try MiMoV26CBv2Adapter(target: target, assistant: assistant)
        _ = try withMiMoConstructionScope { constructionWork in
            try adapter.probeNativeKVTypes(retaining: constructionWork)
        }
        let backend = try adapter.makeBackend(bytesCapacity: 32 << 20)
        let engine = EngineV2(
            model: adapter, layerKinds: adapter.layerKinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: adapter.makeCaches()),
            schedulerConfig: .init(
                maxConcurrentRequests: 2, maxBatchedTokensPerStep: 32,
                prefillChunkSize: chunk, maxWaiting: 8),
            mtpDrafter: drafter ?? assistant,
            mtpConfig: .init(
                enabled: assistant != nil, maxDraftTokens: depth,
                maxSpeculativeBatch: 1, fixedDraftTokens: depth, verificationMode: .serialTarget))
        return (engine, backend)
    }
    private func request(_ id: UInt64, _ tokens: [Int], _ budget: Int) -> CBv2Request {
        .init(
            id: .init(id), promptTokens: tokens, sampling: .init(temperature: 0), maxTokens: budget)
    }
    private func baseline(
        _ target: MiMoV26TextModel, _ request: CBv2Request,
        chunk: Int = 3
    ) async throws -> CBv2SchedCollected {
        let (engine, backend) = try engine(target: target, assistant: nil, chunk: chunk)
        let result = await cbv2SchedCollect(try engine.submit(request))
        await engine.shutdown()
        XCTAssertEqual(backend.bytesInUse, 0)
        return result
    }

    func testRealEngineNativeOFFONAgreementAcrossChunksDepthAndBudget() async throws {
        let (target, mtp) = try MiMoV26MTPChecks.fixture()
        let native = try MiMoV26MTPAssistant(target: target, predictor: mtp)
        for (length, chunk, depth, budget) in [
            (1, 1, 1, 1), (2, 1, 2, 2), (11, 3, 3, 16), (17, 7, 2, 13),
        ] {
            let req = request(1, prompt(length), budget)
            let off = try await baseline(target, req, chunk: chunk)
            let tracking = MiMoV26ScriptedAssistant(native)
            let (engine, backend) = try engine(
                target: target, assistant: native, drafter: tracking, chunk: chunk, depth: depth)
            let on = await cbv2SchedCollect(try engine.submit(req))
            let metrics = try XCTUnwrap(engine.mtpMetricsSnapshot())
            await engine.shutdown()
            XCTAssertEqual(on.tokens, off.tokens)
            XCTAssertEqual(on.finishReason, off.finishReason)
            XCTAssertEqual(on.tokens.count, budget)
            if budget > 2 {
                XCTAssertGreaterThan(metrics.rounds, 0)
                XCTAssertGreaterThan(tracking.observations, 1)
                XCTAssertGreaterThan(tracking.observationFenceReads, 0)
                XCTAssertTrue(tracking.states.contains { $0.headProposalCounts[depth - 1] > 0 })
            }
            XCTAssertTrue(tracking.states.allSatisfy { $0.isReleased && $0.materializedBytes == 0 })
            XCTAssertEqual(backend.bytesInUse, 0)
            XCTAssertEqual(metrics.rectangularVerificationRounds, 0)
        }
    }

    func testRealEngineAcceptRejectPositionsZeroThroughThreeAndEOS() async throws {
        let (target, mtp) = try MiMoV26MTPChecks.fixture()
        let native = try MiMoV26MTPAssistant(target: target, predictor: mtp)
        let tokens = prompt(11)
        let req = request(1, tokens, 24)
        let off = try await baseline(target, req)
        for rejection in 0 ... 3 {
            let tracking = MiMoV26ScriptedAssistant(
                native, script: off.tokens,
                promptLength: tokens.count, rejectPosition: rejection)
            let (engine, backend) = try engine(target: target, assistant: native, drafter: tracking)
            let on = await cbv2SchedCollect(try engine.submit(req))
            let metrics = try XCTUnwrap(engine.mtpMetricsSnapshot())
            await engine.shutdown()
            XCTAssertEqual(on.tokens, off.tokens, "first rejected position \(rejection)")
            XCTAssertGreaterThan(metrics.rounds, 0)
            if rejection == 0 {
                XCTAssertEqual(metrics.acceptedTokens, 0)
            } else {
                XCTAssertGreaterThan(metrics.perPositionAccepted[rejection - 1], 0)
            }
            if rejection < 3 { XCTAssertEqual(metrics.perPositionAccepted[rejection], 0) }
            XCTAssertEqual(backend.bytesInUse, 0)
        }
        var eos = req
        eos.stopTokens = [off.tokens[2]]
        let eosOff = try await baseline(target, eos)
        let tracking = MiMoV26ScriptedAssistant(
            native, script: off.tokens, promptLength: tokens.count)
        let (engine, backend) = try engine(target: target, assistant: native, drafter: tracking)
        let eosOn = await cbv2SchedCollect(try engine.submit(eos))
        await engine.shutdown()
        XCTAssertEqual(eosOn.tokens, eosOff.tokens)
        XCTAssertEqual(eosOn.finishReason, .stop)
        XCTAssertEqual(backend.bytesInUse, 0)
    }

    func testRealEngineRequestJoinIsolationAndCancellationDrain() async throws {
        let (target, mtp) = try MiMoV26MTPChecks.fixture()
        let native = try MiMoV26MTPAssistant(target: target, predictor: mtp)
        let first = request(1, prompt(11), 13)
        let second = request(2, prompt(19), 11)
        let expectedFirst = try await baseline(target, first)
        let expectedSecond = try await baseline(target, second)
        let tracking = MiMoV26ScriptedAssistant(native)
        let (engine, backend) = try engine(target: target, assistant: native, drafter: tracking)
        let firstStream = try engine.submit(first)
        let secondStream = try engine.submit(second)
        let firstTask = Task { await cbv2SchedCollect(firstStream) }
        let secondTask = Task { await cbv2SchedCollect(secondStream) }
        let resultFirst = await firstTask.value
        let resultSecond = await secondTask.value
        XCTAssertEqual(resultFirst.tokens, expectedFirst.tokens)
        XCTAssertEqual(resultSecond.tokens, expectedSecond.tokens)
        let cancellation = request(3, prompt(13), 40)
        let stream = try engine.submit(cancellation)
        var cancelled = false
        var emitted = 0
        var finish: CBv2FinishReason?
        for await event in stream {
            switch event {
            case .delta(_, let tokens, _):
                emitted += tokens.count
                if emitted >= 6 && !cancelled {
                    cancelled = true
                    engine.cancel(cancellation.id)
                }
            case .finished(let reason, _): finish = reason
            }
        }
        await engine.shutdown()
        XCTAssertTrue(cancelled)
        XCTAssertEqual(finish, .cancelled)
        XCTAssertEqual(backend.bytesInUse, 0)
        XCTAssertTrue(tracking.states.allSatisfy { $0.isReleased && $0.materializedBytes == 0 })
        XCTAssertEqual(Set(tracking.states.map(ObjectIdentifier.init)).count, tracking.states.count)
    }

    func testRealEngineSamplingSafetyFallbacksDoNotDraft() async throws {
        let (target, mtp) = try MiMoV26MTPChecks.fixture()
        let native = try MiMoV26MTPAssistant(target: target, predictor: mtp)
        for sampling in [
            CBv2SamplingParams(temperature: 0.7, seed: 7),
            CBv2SamplingParams(temperature: 0, repetitionPenalty: 1.1),
            CBv2SamplingParams(temperature: 0, logitBias: [3: 1]),
        ] {
            let tracking = MiMoV26ScriptedAssistant(native)
            let (engine, backend) = try engine(target: target, assistant: native, drafter: tracking)
            var req = request(1, prompt(7), 8)
            req.sampling = sampling
            let result = await cbv2SchedCollect(try engine.submit(req))
            let metrics = try XCTUnwrap(engine.mtpMetricsSnapshot())
            await engine.shutdown()
            XCTAssertEqual(result.tokens.count, 8)
            XCTAssertEqual(metrics.rounds, 0)
            XCTAssertEqual(metrics.draftedTokens, 0)
            XCTAssertTrue(tracking.states.isEmpty)
            XCTAssertEqual(backend.bytesInUse, 0)
        }
    }

    func testEngineCompletionFenceRetainsKnownOwnersThroughCancelDrainAndFailure() async throws {
        let (target, mtp) = try MiMoV26MTPChecks.fixture()
        let native = try MiMoV26MTPAssistant(target: target, predictor: mtp)
        for shutdown in [false, true] {
            let tracking = MiMoV26ScriptedAssistant(native)
            let visit = MiMoV26SecondFenceVisit()
            let entered = DispatchSemaphore(value: 0)
            let resume = DispatchSemaphore(value: 0)
            tracking.fenceHook = { state in
                if visit.shouldBlock() {
                    XCTAssertTrue(state.hasUnmeasuredResidency)
                    XCTAssertGreaterThan(state.materializedBytes, 0)
                    entered.signal()
                    _ = resume.wait(timeout: .now() + 10)
                }
            }
            let (engine, backend) = try engine(target: target, assistant: native, drafter: tracking)
            let stream = try engine.submit(request(1, prompt(11), 16))
            let collected = Task { await cbv2SchedCollect(stream) }
            let blocked = await withCheckedContinuation { continuation in
                DispatchQueue.global().async {
                    continuation.resume(returning: entered.wait(timeout: .now() + 5) == .success)
                }
            }
            XCTAssertTrue(blocked)
            let owner = try XCTUnwrap(tracking.states.first)
            let held = owner.materializedBytes
            XCTAssertGreaterThan(held, 0)
            XCTAssertTrue(owner.hasUnmeasuredResidency)
            let drain = shutdown ? Task { await engine.shutdown() } : nil
            if !shutdown { engine.cancel(.init(1)) }
            XCTAssertEqual(
                owner.materializedBytes, held,
                "queued cancel/drain cannot retire the measured owner")
            resume.signal()
            let result = await collected.value
            if let drain {
                await drain.value
            } else {
                XCTAssertEqual(result.finishReason, .cancelled)
                await engine.shutdown()
            }
            XCTAssertEqual(backend.bytesInUse, 0)
            XCTAssertTrue(
                tracking.states.allSatisfy {
                    $0.isReleased && $0.materializedBytes == 0 && $0.measuredRootCount == 0
                })
        }
        let failing = MiMoV26ScriptedAssistant(native)
        failing.fenceHook = { _ in throw MLXError.caught("intentional MTP completion-owner refusal")
        }
        let (engine, backend) = try engine(target: target, assistant: native, drafter: failing)
        let result = await cbv2SchedCollect(try engine.submit(request(2, prompt(11), 16)))
        if case .error? = result.finishReason {
        } else {
            XCTFail("owner proof failure did not fail closed")
        }
        await engine.shutdown()
        XCTAssertEqual(backend.bytesInUse, 0)
        XCTAssertTrue(failing.states.allSatisfy { $0.isReleased && $0.materializedBytes == 0 })
    }
}
