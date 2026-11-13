// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The DFlash drafter through the real CBv2 engine on the paged backend, over
/// a tiny random target. The invariant under test: a draft is kept only when
/// it equals the target's own token, so the emitted tokens are those of the
/// same engine without a drafter, whatever the drafter proposes.
@Suite(.serialized, .tags(.integration))
struct MuseGlimmerDFlashEngineTests {
    typealias Fixture = MuseGlimmerDFlashEngineFixture
    typealias Scripted = MuseGlimmerDFlashScriptedDrafter

    static let maxTokens = 40
    /// Longer than the target window (17) and the drafter window (32).
    static let promptLength = 45

    private func run(_ engine: EngineV2, _ request: CBv2Request) async throws
        -> CBv2SchedCollected
    {
        await cbv2SchedCollect(try engine.submit(request))
    }

    /// Tokens of the same request on an engine without a drafter. A sampled
    /// request's noise is keyed by its id, so the id is part of the request.
    private func baseline(
        _ target: MuseGlimmerDFlashTinyTarget, prompt: [Int], sampling: CBv2SamplingParams,
        maxTokens: Int = MuseGlimmerDFlashEngineTests.maxTokens, id: UInt64 = 1
    ) async throws -> [Int] {
        let (engine, backend) = try Fixture.engine(target: target, drafter: nil)
        let result = try await run(
            engine, Fixture.request(id, prompt: prompt, maxTokens: maxTokens, sampling: sampling))
        await engine.shutdown()
        #expect(result.finishReason == .length && result.tokens.count == maxTokens)
        #expect(backend.bytesReserved == 0)
        return result.tokens
    }

    private func expectNoState(_ engine: EngineV2, _ scripted: Scripted) {
        engine.loopForTesting.onEngineQueueSync {
            #expect(engine.loopForTesting.mtp?.requestStateCountForTesting == 0)
        }
        #expect(!scripted.states.isEmpty && scripted.states.allSatisfy { $0.released })
        #expect(scripted.states.allSatisfy { !$0.context.isAllocated && $0.memo == nil })
    }

    static let modes: [Scripted.Mode] =
        [.native, .right] + (0 ..< 7).map { Scripted.Mode.wrong(at: $0) }

    @Test(
        .timeLimit(.minutes(3)),
        arguments: [CBv2MTPVerificationMode.serialTarget, .rectangular])
    func greedyOutputEqualsTheTargetAloneWhateverIsDrafted(
        verification: CBv2MTPVerificationMode
    ) async throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let prompt = Fixture.prompt(Self.promptLength, salt: 1)
        let expected = try await baseline(target, prompt: prompt, sampling: Fixture.greedy)
        for mode in Self.modes {
            let scripted = Scripted(
                try Fixture.drafter(target: target), mode: mode, scripts: [prompt + expected],
                vocabulary: target.config.vocabSize)
            let (engine, backend) = try Fixture.engine(
                target: target, drafter: scripted, verification: verification)
            #expect(engine.mtpInactiveReason == nil)
            let result = try await run(
                engine,
                Fixture.request(
                    1, prompt: prompt, maxTokens: Self.maxTokens, sampling: Fixture.greedy))
            let metrics = try #require(engine.mtpMetricsSnapshot())
            #expect(result.tokens == expected, "\(mode) \(verification)")
            #expect(result.finishReason == .length)
            #expect(metrics.rounds > 0 && metrics.draftedTokens > 0, "\(mode) never drafted")
            let rectangular = verification == .rectangular
            #expect(metrics.rectangularVerificationRounds == (rectangular ? metrics.rounds : 0))
            #expect(metrics.serialVerificationRounds == (rectangular ? 0 : metrics.rounds))
            // One block per round, anchored at consecutive committed positions.
            #expect(scripted.blockAnchors.count == metrics.rounds)
            #expect(scripted.blockAnchors == scripted.blockAnchors.sorted())
            #expect(scripted.blockAnchors.allSatisfy { $0 > prompt.count })
            #expect(
                zip(scripted.blockAnchors, scripted.blockContextRows).allSatisfy {
                    $1 == min($0, 32)
                }, "context rows \(scripted.blockContextRows) at \(scripted.blockAnchors)")
            let positions = metrics.perPositionAccepted
            switch mode {
            case .right:
                // Every draft of every round is the target's own token.
                #expect(metrics.acceptedTokens == metrics.draftedTokens)
                #expect(metrics.emittedTokens == metrics.acceptedTokens + metrics.rounds)
            case .wrong(let index):
                // Full rounds accept exactly the drafts before the wrong one.
                #expect(positions.count > index ? positions[index] == 0 : true)
                if index > 0 { #expect(positions[index - 1] > 0) }
                #expect(metrics.acceptedTokens <= metrics.rounds * index)
            case .native:
                #expect(metrics.acceptedTokens <= metrics.draftedTokens)
            }
            expectNoState(engine, scripted)
            await engine.shutdown()
            #expect(backend.bytesReserved == 0 && backend.bytesWired == 0)
        }
    }

    @Test(
        .timeLimit(.minutes(3)),
        arguments: [CBv2MTPVerificationMode.serialTarget, .rectangular])
    func sampledOutputEqualsTheTargetAlone(verification: CBv2MTPVerificationMode) async throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let prompt = Fixture.prompt(Self.promptLength, salt: 2)
        let expected = try await baseline(target, prompt: prompt, sampling: Fixture.sampled)
        let greedy = try await baseline(target, prompt: prompt, sampling: Fixture.greedy)
        #expect(expected != greedy, "the sampled request must not be greedy in disguise")
        for mode in [Scripted.Mode.native, .right, .wrong(at: 0), .wrong(at: 3)] {
            let scripted = Scripted(
                try Fixture.drafter(target: target), mode: mode, scripts: [prompt + expected],
                vocabulary: target.config.vocabSize)
            let (engine, backend) = try Fixture.engine(
                target: target, drafter: scripted, verification: verification)
            let result = try await run(
                engine,
                Fixture.request(
                    1, prompt: prompt, maxTokens: Self.maxTokens, sampling: Fixture.sampled))
            let metrics = try #require(engine.mtpMetricsSnapshot())
            #expect(result.tokens == expected, "\(mode) \(verification)")
            #expect(metrics.rounds > 0, "sampled rows must speculate")
            if mode == .right {
                // Drafts equal to the target's own samples are all kept.
                #expect(metrics.acceptedTokens == metrics.draftedTokens)
            }
            expectNoState(engine, scripted)
            await engine.shutdown()
            #expect(backend.bytesReserved == 0)
        }
    }

    /// Every depth the engine may choose, including the tail rounds a length
    /// bound shortens.
    @Test(.timeLimit(.minutes(3)))
    func everyDraftDepthIsExact() async throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let prompt = Fixture.prompt(20, salt: 3)
        for maxTokens in [1, 2, 9, 23] {
            let expected = try await baseline(
                target, prompt: prompt, sampling: Fixture.greedy, maxTokens: maxTokens)
            for depth in [1, 2, 4, 7] {
                let scripted = Scripted(
                    try Fixture.drafter(
                        target: target,
                        settings: .init(maximumDraftTokens: depth, maximumSpeculativeRows: 2)),
                    mode: .right, scripts: [prompt + expected],
                    vocabulary: target.config.vocabSize)
                let (engine, _) = try Fixture.engine(
                    target: target, drafter: scripted, depth: depth)
                let result = try await run(
                    engine,
                    Fixture.request(
                        1, prompt: prompt, maxTokens: maxTokens, sampling: Fixture.greedy))
                #expect(result.tokens == expected, "depth \(depth) maxTokens \(maxTokens)")
                let metrics = try #require(engine.mtpMetricsSnapshot())
                #expect(metrics.draftedTokens <= metrics.rounds * depth)
                await engine.shutdown()
            }
        }
    }

    @Test(.timeLimit(.minutes(3)))
    func cancellationMidRoundLeavesNoState() async throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let prompt = Fixture.prompt(Self.promptLength, salt: 4)
        let expected = try await baseline(target, prompt: prompt, sampling: Fixture.greedy)
        for verification in [CBv2MTPVerificationMode.serialTarget, .rectangular] {
            let scripted = Scripted(
                try Fixture.drafter(target: target), mode: .right, scripts: [prompt + expected],
                vocabulary: target.config.vocabSize)
            let (engine, backend) = try Fixture.engine(
                target: target, drafter: scripted, verification: verification)
            for cancelAtBlock in [1, 2, 3] {
                let id = CBv2RequestID(UInt64(100 + cancelAtBlock))
                let blocksBefore = scripted.blockAnchors.count
                // Cancel while the round's graph is being built.
                scripted.onBlock = { count in
                    if count == blocksBefore + cancelAtBlock { engine.cancel(id) }
                }
                let (events, retirement) = try engine.submitWithRetirement(
                    CBv2Request(
                        id: id, promptTokens: prompt, sampling: Fixture.greedy, maxTokens: 400))
                var finish: CBv2FinishReason?
                var tokens: [Int] = []
                for await event in events {
                    switch event {
                    case .delta(_, let delta, _): tokens.append(contentsOf: delta)
                    case .finished(let reason, _): finish = reason
                    }
                }
                await retirement.wait()
                #expect(finish == .cancelled)
                #expect(scripted.blockAnchors.count == blocksBefore + cancelAtBlock)
                // What was emitted before the cancel is still the target's.
                #expect(tokens == Array(expected.prefix(tokens.count)))
                engine.loopForTesting.onEngineQueueSync {
                    #expect(engine.loopForTesting.mtp?.requestStateCountForTesting == 0)
                    #expect(engine.loopForTesting.scheduler.record(for: id) == nil)
                }
                #expect(scripted.states.allSatisfy { $0.released })
                #expect(backend.bytesReserved == 0)
                #expect(engine.admissionForTesting.bytesReserved == 0)
            }
            scripted.onBlock = nil
            // The engine serves the next request exactly.
            let resumed = try await run(
                engine,
                Fixture.request(
                    200, prompt: prompt, maxTokens: Self.maxTokens, sampling: Fixture.greedy))
            #expect(resumed.tokens == expected)
            expectNoState(engine, scripted)
            await engine.shutdown()
            #expect(backend.bytesReserved == 0 && backend.bytesWired == 0)
        }
    }

    /// Two requests of different lengths in one engine: each keeps its own
    /// context and both stay exact, greedy beside sampled.
    @Test(
        .timeLimit(.minutes(3)),
        arguments: [CBv2MTPVerificationMode.serialTarget, .rectangular])
    func twoConcurrentRowsKeepSeparateContexts(
        verification: CBv2MTPVerificationMode
    ) async throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let prompts = [Fixture.prompt(37, salt: 5), Fixture.prompt(52, salt: 6)]
        let samplings = [Fixture.greedy, Fixture.sampled]
        var expected: [[Int]] = []
        for index in 0 ..< 2 {
            expected.append(
                try await baseline(
                    target, prompt: prompts[index], sampling: samplings[index],
                    id: UInt64(index + 1)))
        }
        for mode in [Scripted.Mode.native, .right, .wrong(at: 2)] {
            let scripted = Scripted(
                try Fixture.drafter(target: target), mode: mode,
                scripts: zip(prompts, expected).map { $0 + $1 },
                vocabulary: target.config.vocabSize)
            let (engine, backend) = try Fixture.engine(
                target: target, drafter: scripted, verification: verification)
            let streams = try (0 ..< 2).map { index in
                try engine.submit(
                    Fixture.request(
                        UInt64(index + 1), prompt: prompts[index], maxTokens: Self.maxTokens,
                        sampling: samplings[index]))
            }
            async let first = cbv2SchedCollect(streams[0])
            async let second = cbv2SchedCollect(streams[1])
            let results = await [first, second]
            #expect(results[0].tokens == expected[0], "\(mode) \(verification) row 0")
            #expect(results[1].tokens == expected[1], "\(mode) \(verification) row 1")
            let metrics = try #require(engine.mtpMetricsSnapshot())
            #expect(metrics.rounds > 0)
            #expect(scripted.states.count == 2)
            expectNoState(engine, scripted)
            await engine.shutdown()
            #expect(backend.bytesReserved == 0 && backend.bytesWired == 0)
        }
    }

    /// The engine never drafts for a request with image spans while a
    /// request-stateful drafter is installed: the request gets no drafter
    /// state and runs on the target alone. Text requests keep speculating.
    @Test(.timeLimit(.minutes(3)))
    func multimodalRequestsRunOnTheTargetAlone() async throws {
        #expect(
            !EngineLoopV2.mtpMultimodalHistoryEligible(
                hasSpans: true, tracksPersistentHistory: true))
        let target = MuseGlimmerDFlashTinyTarget()
        let prompt = Fixture.prompt(Self.promptLength, salt: 7)
        let image = MLXRandom.normal([6, 64], key: MLXRandom.key(8))
        eval(image)
        func media(_ id: UInt64) -> CBv2Request {
            CBv2Request(
                id: CBv2RequestID(id), promptTokens: prompt, sampling: Fixture.greedy,
                maxTokens: Self.maxTokens,
                multimodal: CBv2MultimodalInput(
                    spans: [CBv2ImageSpan(tokenOffset: 3, length: 6)], attention: .causal
                ) { [image] })
        }
        let (plain, _) = try Fixture.engine(target: target, drafter: nil)
        let expected = try await run(plain, media(1))
        await plain.shutdown()
        #expect(expected.finishReason == .length && expected.tokens.count == Self.maxTokens)
        let text = try await baseline(target, prompt: prompt, sampling: Fixture.greedy)
        #expect(text != expected.tokens, "the image must change the continuation")

        let scripted = Scripted(
            try Fixture.drafter(target: target), mode: .right, scripts: [prompt + text],
            vocabulary: target.config.vocabSize)
        let (engine, backend) = try Fixture.engine(target: target, drafter: scripted)
        let result = try await run(engine, media(2))
        #expect(result.tokens == expected.tokens)
        #expect(scripted.states.isEmpty && scripted.blockAnchors.isEmpty)
        #expect(try #require(engine.mtpMetricsSnapshot()).rounds == 0)
        // The same engine still speculates for text.
        let spoken = try await run(
            engine,
            Fixture.request(3, prompt: prompt, maxTokens: Self.maxTokens, sampling: Fixture.greedy))
        #expect(spoken.tokens == text)
        #expect(try #require(engine.mtpMetricsSnapshot()).rounds > 0)
        expectNoState(engine, scripted)
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
    }

    /// Target rows that do not fit never stop the engine thread: the
    /// request ends with an error at the step's fence and the engine serves
    /// the next one.
    @Test(.timeLimit(.minutes(3)))
    func misshapenTargetRowsEndTheRequestNotTheEngine() async throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let prompt = Fixture.prompt(Self.promptLength, salt: 11)
        let expected = try await baseline(target, prompt: prompt, sampling: Fixture.greedy)
        let scripted = Scripted(
            try Fixture.drafter(target: target), mode: .right, scripts: [prompt + expected],
            vocabulary: target.config.vocabSize)
        let (engine, backend) = try Fixture.engine(target: target, drafter: scripted)
        target.widensHidden = true
        let failed = try await run(
            engine,
            Fixture.request(1, prompt: prompt, maxTokens: Self.maxTokens, sampling: Fixture.greedy))
        guard case .error = failed.finishReason else {
            Issue.record("expected an error finish, got \(String(describing: failed.finishReason))")
            return
        }
        #expect(failed.tokens.isEmpty)
        target.widensHidden = false
        let served = try await run(
            engine,
            Fixture.request(2, prompt: prompt, maxTokens: Self.maxTokens, sampling: Fixture.greedy))
        #expect(served.tokens == expected)
        expectNoState(engine, scripted)
        await engine.shutdown()
        #expect(backend.bytesReserved == 0 && backend.bytesWired == 0)

        // The same faults on the drafter alone: reported at the fence.
        let drafter = try Fixture.drafter(target: target)
        func tokens(_ count: Int) -> MLXArray {
            MLXArray([Int32](repeating: 3, count: count)).reshaped([1, count])
        }
        var state = drafter.makeRequestState()
        drafter.observeCommittedTarget(
            .init(tokens: tokens(4), hidden: MLXArray.zeros([1, 4, 192])), requestState: state)
        #expect(throws: MuseGlimmerDFlashError.state("observation")) {
            try drafter.requestStateDidFinishEvaluation(state)
        }
        #expect(state.committedInputCount == 0)
        // More draft steps than the round's depth.
        state = drafter.makeRequestState()
        drafter.observeCommittedTarget(
            .init(tokens: tokens(4), hidden: MLXArray.zeros([1, 4, 128])), requestState: state)
        try drafter.requestStateDidFinishEvaluation(state)
        for index in 0 ..< 8 {
            let step = drafter.draftStep(
                tokens: tokens(1), hidden: MLXArray.zeros([1, 1, 128]), shortlist: nil,
                requestState: state)
            #expect(step.tokens.shape == [1] && step.hidden.shape == [1, 1, 64], "step \(index)")
        }
        #expect(throws: MuseGlimmerDFlashError.state("draft step")) {
            try drafter.requestStateDidFinishEvaluation(state)
        }
        drafter.discardRound(requestState: state)
        #expect(state.stagedInputCount == 0)
        drafter.releaseRequestState(state)
        #expect(throws: MuseGlimmerDFlashError.self) {
            try drafter.requestStateDidFinishEvaluation(state)
        }
    }

    // MARK: Binding and admission

    @Test func drafterBindsOnlyToAMatchingTarget() throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let drafter = try Fixture.drafter(target: target)
        #expect(drafter.mtpTargetIdentity == target.mtpTargetIdentity)
        #expect(drafter.maximumDraftTokens == 7 && drafter.supportsTargetPrefixAcceptance)
        #expect(drafter.requiresCommittedObservationFence)
        let config = CBv2MTPConfig(
            enabled: true, maxDraftTokens: 7, fixedDraftTokens: 7, verificationMode: .serialTarget)
        let driver = try #require(
            CBv2MTPRoundDriver.build(model: target, drafter: drafter, config: config))
        #expect(driver.usesRequestStatefulDrafter && driver.targetPrefixAcceptance)
        #expect(driver.config.maxDraftTokens == 7)
        // Another instance of the same architecture is a foreign target.
        #expect(
            CBv2MTPRoundDriver.build(
                model: MuseGlimmerDFlashTinyTarget(), drafter: drafter, config: config) == nil)

        // A draft model that does not fit the target is refused at binding.
        var narrow = try Fixture.drafterConfiguration()
        narrow.targetLayerIDs = [0]
        #expect(throws: MuseGlimmerDFlashError.target("context width")) {
            try MuseGlimmerDFlashDrafter(
                target: target, model: try MuseGlimmerDFlashFixture.model(narrow))
        }
        var mask = try Fixture.drafterConfiguration()
        mask.maskTokenID = 128
        #expect(throws: MuseGlimmerDFlashError.target("mask token")) {
            try MuseGlimmerDFlashDrafter(
                target: target, model: try MuseGlimmerDFlashFixture.model(mask))
        }
        for settings in [
            MuseGlimmerDFlashDrafter.Settings(maximumDraftTokens: 8),
            .init(maximumDraftTokens: 0), .init(maximumSpeculativeRows: 9),
        ] {
            #expect(throws: MuseGlimmerDFlashError.configuration("settings")) {
                try MuseGlimmerDFlashDrafter(
                    target: target,
                    model: try MuseGlimmerDFlashFixture.model(try Fixture.drafterConfiguration()),
                    settings: settings)
            }
        }
    }

    /// The request charge is the ring, whatever the request's length; the
    /// block's work is charged once per engine.
    @Test func admissionChargesTheRingOnceAndNothingPerToken() throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let drafter = try Fixture.drafter(target: target)
        let c = drafter.model.configuration
        let ring = MuseGlimmerDFlashContextState.residentBytes(configuration: c, dtype: .float32)
        #expect(ring == 2 * 2 * 2 * 32 * 16 * 4)
        #expect(drafter.requestStateBytesPerToken == 0)
        let limits = CBv2MTPAllocationLimits(maximumPrefillTokens: 64, maximumDraftTokens: 7)
        let spec = try #require(drafter.boundedRequestAllocation(limits: limits))
        #expect(spec.resident == [.init(logicalBytes: ring / 2, allocationCount: 2)])
        #expect(spec.resident.allSatisfy { $0.logicalBytesPerToken == 0 })
        #expect(
            spec.working.allSatisfy { $0.logicalBytesPerToken == 0 } && spec.hostBytesPerToken == 0)
        // A longer prompt chunk commits no more than one window of rows.
        let longer = try #require(
            drafter.boundedRequestAllocation(
                limits: .init(maximumPrefillTokens: 4096, maximumDraftTokens: 7)))
        #expect(longer.working == spec.working)
        #expect(spec.ownerWorking.first?.allocationCount == 2)
        #expect(spec.ownerWorking.first?.logicalBytes == drafter.blockBytes)

        let (engine, _) = try Fixture.engine(target: target, drafter: drafter)
        let admission = engine.admissionForTesting
        let plain = try Fixture.engine(target: target, drafter: nil).0.admissionForTesting
        let short = admission.allocatedBytes(forTokens: 64)
        let long = admission.allocatedBytes(forTokens: 4096)
        // The drafter adds the same fixed bytes at any length.
        #expect(
            short - plain.allocatedBytes(forTokens: 64)
                == long - plain.allocatedBytes(forTokens: 4096))
        #expect(short - plain.allocatedBytes(forTokens: 64) >= ring)

        // The published geometry: 20,480 bytes per position, 2,048 positions.
        let published = MuseGlimmerDFlashContextState.residentBytes(
            configuration: .museGlimmer30B, dtype: .bfloat16)
        #expect(published == 20_480 * 2048)
    }

    /// The state's reported bytes are its ring, allocated at the first row.
    @Test func requestStateReportsItsRing() throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let drafter = try Fixture.drafter(target: target)
        let state = drafter.makeRequestState()
        #expect(state.materializedBytes == 0 && state.committedInputCount == 0)
        let hidden = MLXRandom.normal([1, 5, 128], key: MLXRandom.key(3))
        drafter.observeCommittedTarget(
            .init(tokens: MLXArray([Int32(1), 2, 3, 4, 5]).reshaped([1, 5]), hidden: hidden),
            requestState: state)
        eval(drafter.evaluationTargets(for: state))
        #expect(
            state.materializedBytes
                == MuseGlimmerDFlashContextState.residentBytes(
                    configuration: drafter.model.configuration, dtype: .float32))
        #expect(state.committedInputCount == 5 && state.stagedInputCount == 0)
        drafter.releaseRequestState(state)
        #expect(state.materializedBytes == 0)
        #expect(drafter.evaluationTargets(for: state).isEmpty)
    }
}
