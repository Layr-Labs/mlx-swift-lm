// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A request-stateful drafter that is not advisory: it forwards to the real
/// drafter and conforms to nothing but the base protocol, like every drafter
/// that existed before `CBv2MTPAdvisoryHistoryDrafter`.
private final class PersistentHistoryDrafter: CBv2MTPRequestStatefulDrafter {
    let native: MuseGlimmerDFlashDrafter
    private(set) var observations = 0
    init(_ native: MuseGlimmerDFlashDrafter) { self.native = native }

    var mtpTargetIdentity: ObjectIdentifier? { native.mtpTargetIdentity }
    var requiredVerificationMode: CBv2MTPVerificationMode? { native.requiredVerificationMode }
    var maximumDraftTokens: Int? { native.maximumDraftTokens }
    var maximumSpeculativeBatch: Int? { native.maximumSpeculativeBatch }
    var supportsTargetPrefixAcceptance: Bool { true }
    var requiresCommittedObservationFence: Bool { true }
    func makeRequestState() -> any CBv2MTPRequestState { native.makeRequestState() }
    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture { native.prepare(rows: rows) }
    func draftStep(tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture)
        -> (tokens: MLXArray, hidden: MLXArray)
    {
        native.draftStep(tokens: tokens, hidden: hidden, prepared: prepared)
    }
    func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation, requestState: any CBv2MTPRequestState
    ) {
        observations += 1
        native.observeCommittedTarget(observation, requestState: requestState)
    }
    func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        native.draftStep(
            tokens: tokens, hidden: hidden, shortlist: shortlist, requestState: requestState)
    }
    func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] {
        native.evaluationTargets(for: requestState)
    }
    func requestStateDidFinishEvaluation(_ requestState: any CBv2MTPRequestState) throws {
        try native.requestStateDidFinishEvaluation(requestState)
    }
    func finalizeRound(
        requestState: any CBv2MTPRequestState, confirmedInputTokens: Int,
        committedDraftTokens: MLXArray, committedTargetHidden: MLXArray
    ) {
        native.finalizeRound(
            requestState: requestState, confirmedInputTokens: confirmedInputTokens,
            committedDraftTokens: committedDraftTokens,
            committedTargetHidden: committedTargetHidden)
    }
    func discardRound(requestState: any CBv2MTPRequestState) {
        native.discardRound(requestState: requestState)
    }
    func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        native.releaseRequestState(requestState)
    }
}

/// Complete prefix checkpoints with the DFlash drafter active. Its context is
/// advisory, so the engine captures and adopts checkpoints with no assistant
/// state, exactly as without a drafter; the restored request's drafter starts
/// at the restored boundary. Drafters that are not advisory keep the refusal
/// they had.
@Suite(.serialized, .tags(.integration))
struct MuseGlimmerDFlashCheckpointTests {
    typealias Fixture = MuseGlimmerDFlashEngineFixture
    typealias Scripted = MuseGlimmerDFlashScriptedDrafter

    static let chunk = Fixture.chunk
    static let maxTokens = 36
    static var promptLength: Int { 3 * chunk + 5 }

    private func request(
        _ id: UInt64, _ prompt: [Int], _ sampling: CBv2SamplingParams, receipt: UInt64
    ) -> CBv2Request {
        CBv2Request(
            id: .init(id), promptTokens: prompt, sampling: sampling, maxTokens: Self.maxTokens,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(receipt))
    }

    private func bytes(_ archive: CompleteCheckpointFixtureStore.Archive) -> [Data] {
        archive.chunks.sorted { ($0.tensor, $0.offset) < ($1.tensor, $1.offset) }.map(\.bytes)
    }

    /// A cold run on a fresh engine; returns its tokens and what it saved.
    private func cold(
        _ target: MuseGlimmerDFlashTinyTarget, drafter: (any CBv2MTPDrafter)?,
        prompt: [Int], sampling: CBv2SamplingParams,
        verification: CBv2MTPVerificationMode = .serialTarget
    ) async throws -> (tokens: [Int], store: CompleteCheckpointFixtureStore, rounds: Int) {
        let store = CompleteCheckpointFixtureStore(segmentBytes: 4096)
        let (engine, backend) = try Fixture.engine(
            target: target, drafter: drafter, verification: verification, store: store)
        let result = await cbv2SchedCollect(
            try engine.submit(request(1, prompt, sampling, receipt: 1001)))
        let rounds = engine.mtpMetricsSnapshot()?.rounds ?? 0
        #expect(result.finishReason == .length && result.tokens.count == Self.maxTokens)
        #expect(engine.admissionForTesting.bytesReserved == 0)
        await engine.shutdown()
        #expect(backend.bytesWired == 0)
        return (result.tokens, store, rounds)
    }

    @Test(
        .timeLimit(.minutes(4)),
        arguments: [CBv2MTPVerificationMode.serialTarget, .rectangular])
    func restoredRequestEqualsColdRequestAndKeepsSpeculating(
        verification: CBv2MTPVerificationMode
    ) async throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let prompt = Fixture.prompt(Self.promptLength, salt: 9)
        for sampling in [Fixture.greedy, Fixture.sampled] {
            // The reference: no drafter at all.
            let plain = try await cold(target, drafter: nil, prompt: prompt, sampling: sampling)
            #expect(plain.rounds == 0)
            let positions = plain.store.saved.map(\.manifest.position)
            #expect(positions == [3 * Self.chunk, Self.chunk])

            // Cold with the drafter: same tokens, same checkpoints, drafts ran.
            let script = [prompt + plain.tokens]
            let coldDrafter = Scripted(
                try Fixture.drafter(target: target), mode: .right, scripts: script,
                vocabulary: target.config.vocabSize)
            let drafted = try await cold(
                target, drafter: coldDrafter, prompt: prompt, sampling: sampling,
                verification: verification)
            #expect(drafted.tokens == plain.tokens)
            #expect(drafted.rounds > 0)
            #expect(drafted.store.saved.map(\.manifest.position) == positions)
            #expect(drafted.store.saved.allSatisfy { $0.manifest.assistantCodecID == nil })
            for (with, without) in zip(drafted.store.saved, plain.store.saved) {
                // The checkpoint holds the target's bytes and nothing else.
                #expect(with.manifest.backendLayout == without.manifest.backendLayout)
                #expect(with.manifest.tensors.count == without.manifest.tensors.count)
                #expect(bytes(with) == bytes(without))
            }
            #expect(coldDrafter.observedPositions.first == 0)

            // Warm: adopt each boundary, from either donor, with any drafts.
            for boundary in positions {
                for donor in [drafted.store, plain.store] {
                    for mode in [Scripted.Mode.right, .native, .wrong(at: 1)] {
                        let reopened = CompleteCheckpointFixtureStore(
                            archives: donor.saved.filter { $0.manifest.position == boundary })
                        let scripted = Scripted(
                            try Fixture.drafter(target: target), mode: mode, scripts: script,
                            vocabulary: target.config.vocabSize)
                        let (engine, backend) = try Fixture.engine(
                            target: target, drafter: scripted, verification: verification,
                            store: reopened)
                        let warmRequest = request(1, prompt, sampling, receipt: 2002)
                        #expect(try reopened.stage(engine: engine, request: warmRequest))
                        let warm = await cbv2SchedCollect(try engine.submit(warmRequest))
                        let label = "boundary \(boundary) mode \(mode) \(verification)"
                        #expect(warm.tokens == plain.tokens, "\(label)")
                        #expect(warm.usage?.prefixCachePrefillTokensSaved == boundary)
                        #expect(warm.usage?.prefixCacheReplayTokens == 0)
                        // The drafter never saw the restored prefix: its first
                        // row is the boundary, and its context grows from there.
                        #expect(scripted.observedPositions.first == boundary, "\(label)")
                        #expect(scripted.observedPositions.allSatisfy { $0 >= boundary })
                        let metrics = try #require(engine.mtpMetricsSnapshot())
                        #expect(metrics.rounds > 0, "speculation must resume: \(label)")
                        #expect(
                            zip(scripted.blockAnchors, scripted.blockContextRows).allSatisfy {
                                $1 == min($0 - boundary, 32)
                            }, "\(label) rows \(scripted.blockContextRows)")
                        if mode == .right {
                            // Right drafts are all kept: anchors and positions
                            // after the restore are the target's.
                            #expect(metrics.acceptedTokens == metrics.draftedTokens, "\(label)")
                        }
                        engine.loopForTesting.onEngineQueueSync {
                            #expect(engine.loopForTesting.mtp?.requestStateCountForTesting == 0)
                        }
                        #expect(scripted.states.allSatisfy { $0.released })
                        #expect(engine.admissionForTesting.bytesReserved == 0)
                        await engine.shutdown()
                        #expect(backend.bytesWired == 0)
                    }
                }
            }

            // An engine without a drafter adopts what the drafter engine saved.
            let reopened = CompleteCheckpointFixtureStore(archives: drafted.store.saved)
            let (engine, _) = try Fixture.engine(target: target, drafter: nil, store: reopened)
            let warmRequest = request(1, prompt, sampling, receipt: 3003)
            #expect(try reopened.stage(engine: engine, request: warmRequest))
            let warm = await cbv2SchedCollect(try engine.submit(warmRequest))
            #expect(warm.tokens == plain.tokens)
            #expect(warm.usage?.prefixCachePrefillTokensSaved == 3 * Self.chunk)
            await engine.shutdown()
        }
    }

    /// Drafters that are not advisory keep exactly the behavior they had: the
    /// generic paged engine builds no checkpoint codec for them, saves
    /// nothing, and hands them rows through the original observation call.
    @Test(.timeLimit(.minutes(3)))
    func persistentHistoryDraftersKeepTheirRefusal() async throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let prompt = Fixture.prompt(Self.promptLength, salt: 10)
        let plain = try await cold(target, drafter: nil, prompt: prompt, sampling: Fixture.greedy)
        #expect(!plain.store.saved.isEmpty)

        // (1) A drafter that does not conform to the advisory protocol.
        let persistent = PersistentHistoryDrafter(try Fixture.drafter(target: target))
        // (2) One that conforms and declares its state not advisory.
        let declined = Scripted(
            try Fixture.drafter(target: target), mode: .native,
            vocabulary: target.config.vocabSize)
        declined.requestStateIsAdvisory = false
        for drafter in [persistent as any CBv2MTPDrafter, declined] {
            let store = CompleteCheckpointFixtureStore(
                archives: plain.store.saved, segmentBytes: 4096)
            let (engine, _) = try Fixture.engine(target: target, drafter: drafter, store: store)
            let driver = try #require(engine.loopForTesting.mtp)
            #expect(driver.tracksPersistentHistory && !driver.historyIsAdvisory)
            #expect(driver.checkpointsRequireAssistantHistory)
            #expect(engine.completeCheckpointCodec == nil)
            #expect(engine.completeCheckpointCapture == nil)
            let coldRequest = request(1, prompt, Fixture.greedy, receipt: 1001)
            // Nothing can be staged for an engine without the codec.
            #expect(throws: (any Error).self) {
                _ = try store.stage(engine: engine, request: coldRequest)
            }
            let result = await cbv2SchedCollect(try engine.submit(coldRequest))
            #expect(result.tokens == plain.tokens)
            #expect(result.usage?.prefixCachePrefillTokensSaved == 0)
            #expect(store.saved.count == plain.store.saved.count, "nothing new was saved")
            #expect(try #require(engine.mtpMetricsSnapshot()).rounds > 0)
            await engine.shutdown()
        }
        #expect(persistent.observations > 0)
        #expect(declined.observedPositions.isEmpty, "positions go to advisory drafters only")
        #expect(!declined.blockAnchors.isEmpty)
    }

    /// The seam's three switches, on the driver alone.
    @Test func advisoryHistoryNeedsTheDeclarationAndExactAcceptance() async throws {
        let target = MuseGlimmerDFlashTinyTarget()
        func driver(
            _ drafter: any CBv2MTPDrafter, fixed: Int? = 7, acceptance: CBv2MTPAcceptance = .exact
        ) throws -> CBv2MTPRoundDriver {
            try #require(
                CBv2MTPRoundDriver.build(
                    model: target, drafter: drafter,
                    config: CBv2MTPConfig(
                        enabled: true, maxDraftTokens: 7, fixedDraftTokens: fixed,
                        verificationMode: .serialTarget, acceptance: acceptance)))
        }
        let advisory = try Fixture.drafter(target: target)
        #expect(advisory.requestStateIsAdvisory)
        var d = try driver(advisory)
        #expect(d.tracksPersistentHistory && d.historyIsAdvisory)
        #expect(!d.checkpointsRequireAssistantHistory)
        // Typical acceptance can keep a draft the target did not choose, so the
        // DFlash drafter refuses it (decision D6): no driver, and the engine
        // says why.
        #expect(advisory.requiresExactAcceptance)
        #expect(
            CBv2MTPRoundDriver.build(
                model: target, drafter: advisory,
                config: CBv2MTPConfig(
                    enabled: true, maxDraftTokens: 7, fixedDraftTokens: 7,
                    verificationMode: .serialTarget, acceptance: .typical(delta: 0.2))) == nil)
        let (typical, _) = try Fixture.engine(
            target: target, drafter: advisory, acceptance: .typical(delta: 0.2))
        #expect(typical.mtpInactiveReason == "drafter requires exact acceptance")
        await typical.shutdown()
        // The switch itself, on an advisory drafter that waives the refusal.
        let tolerant = MuseGlimmerDFlashScriptedDrafter(advisory, mode: .native, vocabulary: 128)
        tolerant.waivesExactAcceptance = true
        d = try driver(tolerant, acceptance: .typical(delta: 0.2))
        #expect(d.tracksPersistentHistory && !d.historyIsAdvisory)
        #expect(d.checkpointsRequireAssistantHistory)
        // Target-only: no history at all, as before.
        d = try driver(advisory, fixed: 0)
        #expect(!d.tracksPersistentHistory && !d.checkpointsRequireAssistantHistory)
        // Not advisory: the new property is the old one.
        d = try driver(PersistentHistoryDrafter(advisory))
        #expect(d.tracksPersistentHistory && !d.historyIsAdvisory)
        #expect(d.checkpointsRequireAssistantHistory == d.tracksPersistentHistory)
        d = try driver(PersistentHistoryDrafter(advisory), fixed: 0)
        #expect(d.checkpointsRequireAssistantHistory == d.tracksPersistentHistory)

        // The engine builds the plain historical codec, with no assistant.
        let store = CompleteCheckpointFixtureStore()
        let (engine, _) = try Fixture.engine(target: target, drafter: advisory, store: store)
        let codec = try #require(engine.completeCheckpointCodec)
        #expect(codec.assistant == nil)
        #expect(
            codec.backendLayout == CBv2CompleteCheckpointManifest.historicalAttentionLayout)
        let (reference, _) = try Fixture.engine(target: target, drafter: nil, store: store)
        #expect(reference.completeCheckpointCodec?.backendLayout == codec.backendLayout)
    }
}
