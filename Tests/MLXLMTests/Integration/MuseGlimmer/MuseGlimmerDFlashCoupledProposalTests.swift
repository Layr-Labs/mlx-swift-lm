// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Coupled proposals (`CBv2MTPCoupledProposalDrafter`) through the real CBv2
/// engine over the tiny random target. A sampled row's drafts are raced with
/// the noise of the target's verify draw at each output index. The emitted
/// tokens must be the target's alone whatever the race proposes; the engine
/// must hand each round exactly that draw's keys; greedy rows and drafters
/// that do not opt in must see nothing.
@Suite(.serialized, .tags(.integration))
struct MuseGlimmerDFlashCoupledProposalTests {
    typealias Fixture = MuseGlimmerDFlashEngineFixture
    typealias Scripted = MuseGlimmerDFlashScriptedDrafter

    static let maxTokens = 40
    static let promptLength = 45
    static let coupled = MuseGlimmerDFlashDrafter.Settings(
        maximumSpeculativeRows: 2, couplesSampledProposals: true)

    private func collect(
        _ target: MuseGlimmerDFlashTinyTarget, drafter: (any CBv2MTPDrafter)?, prompt: [Int],
        sampling: CBv2SamplingParams, verification: CBv2MTPVerificationMode = .serialTarget,
        sampler: CBv2StepSampler = CBv2DefaultSampler()
    ) async throws -> (tokens: [Int], metrics: CBv2MTPMetrics?) {
        let (engine, backend) = try Fixture.engine(
            target: target, drafter: drafter, verification: verification, sampler: sampler)
        let result = await cbv2SchedCollect(
            try engine.submit(
                Fixture.request(1, prompt: prompt, maxTokens: Self.maxTokens, sampling: sampling)))
        let metrics = engine.mtpMetricsSnapshot()
        await engine.shutdown()
        #expect(result.finishReason == .length && result.tokens.count == Self.maxTokens)
        #expect(backend.bytesReserved == 0)
        return (result.tokens, metrics)
    }

    /// Fixed draft rows `[7, vocab]` to apply a recorded selection to.
    static func probe(_ vocabulary: Int) -> MLXArray {
        let values: [Float] = (0 ..< (7 * vocabulary)).map {
            Float(($0 * 73 + 19) % 509 - 254) / 47
        }
        return MLXArray(values, [7, vocabulary])
    }

    static func ints(_ a: MLXArray) -> [Int32] {
        a.reshaped([-1]).asType(.int32).asArray(Int32.self)
    }

    @Test(
        .timeLimit(.minutes(3)),
        arguments: [CBv2MTPVerificationMode.serialTarget, .rectangular])
    func sampledOutputIsTheTargetsWithCoupledProposals(
        verification: CBv2MTPVerificationMode
    ) async throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let prompt = Fixture.prompt(Self.promptLength, salt: 2)
        let expected = try await collect(
            target, drafter: nil, prompt: prompt, sampling: Fixture.sampled
        ).tokens
        let scripted = Scripted(
            try Fixture.drafter(target: target, settings: Self.coupled), mode: .native,
            vocabulary: target.config.vocabSize)
        let run = try await collect(
            target, drafter: scripted, prompt: prompt, sampling: Fixture.sampled,
            verification: verification)
        let metrics = try #require(run.metrics)
        #expect(run.tokens == expected, "\(verification)")
        #expect(metrics.rounds > 0 && metrics.draftedTokens > 0)
        // One selection per round, before its block, and every one usable.
        #expect(scripted.selections.count == metrics.rounds)
        #expect(scripted.selections.count == scripted.blockAnchors.count)
        #expect(scripted.selections.allSatisfy { $0 != nil })
    }

    /// An unseeded row is keyed by its engine's own sampler, which the
    /// proposals use too: two engines with the same fallback seed emit the
    /// same tokens with and without coupled proposals.
    @Test(.timeLimit(.minutes(3)))
    func unseededRowsRaceWithTheEnginesOwnNoise() async throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let prompt = Fixture.prompt(Self.promptLength, salt: 6)
        var unseeded = Fixture.sampled
        unseeded.seed = nil
        let expected = try await collect(
            target, drafter: nil, prompt: prompt, sampling: unseeded,
            sampler: CBv2DefaultSampler(fallbackSeed: 77)
        ).tokens
        let other = try await collect(
            target, drafter: nil, prompt: prompt, sampling: unseeded,
            sampler: CBv2DefaultSampler(fallbackSeed: 78)
        ).tokens
        #expect(other != expected, "the fallback seed must reach the draw")
        let scripted = Scripted(
            try Fixture.drafter(target: target, settings: Self.coupled), mode: .native,
            vocabulary: target.config.vocabSize)
        let run = try await collect(
            target, drafter: scripted, prompt: prompt, sampling: unseeded,
            sampler: CBv2DefaultSampler(fallbackSeed: 77))
        #expect(run.tokens == expected)
        #expect(try #require(run.metrics).rounds > 0)
        #expect(!scripted.selections.isEmpty && scripted.selections.allSatisfy { $0 != nil })
    }

    /// Round r's selection races draft index j with the keys of output index
    /// `anchor + 1 + j - prompt`, the index whose target draw verifies it.
    @Test(.timeLimit(.minutes(3)))
    func eachRoundRacesWithItsVerifyDrawsKeys() async throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let vocabulary = target.config.vocabSize
        let prompt = Fixture.prompt(Self.promptLength, salt: 3)
        let scripted = Scripted(
            try Fixture.drafter(target: target, settings: Self.coupled), mode: .native,
            vocabulary: vocabulary)
        _ = try await collect(target, drafter: scripted, prompt: prompt, sampling: Fixture.sampled)
        let probe = Self.probe(vocabulary)
        let oracle = CBv2DefaultSampler(fallbackSeed: 0)
        func race(base: Int) -> [Int32] {
            Self.ints(
                oracle.mtpCoupledProposal(
                    logits: probe.reshaped([7, 1, vocabulary]),
                    params: Array(repeating: Fixture.sampled, count: 7),
                    requestIDs: Array(repeating: CBv2RequestID(1), count: 7),
                    stepBases: (0 ..< 7).map { base + $0 }))
        }
        #expect(!scripted.blockAnchors.isEmpty)
        var shiftedDiffers = false
        for (selection, anchor) in zip(scripted.selections, scripted.blockAnchors) {
            let chosen = try #require(selection)
            let given = Self.ints(try #require(chosen.propose(probe)))
            let base = anchor + 1 - prompt.count
            #expect(given == race(base: base), "anchor \(anchor)")
            if given != race(base: base + 1) { shiftedDiffers = true }
        }
        #expect(shiftedDiffers, "the probe must tell neighbouring keys apart")
    }

    /// The drafter proposes what its selection gives, and the output is still
    /// the target's own.
    @Test(.timeLimit(.minutes(3)))
    func theDrafterProposesWhatItsSelectionGives() async throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let vocabulary = target.config.vocabSize
        let prompt = Fixture.prompt(Self.promptLength, salt: 4)
        let expected = try await collect(
            target, drafter: nil, prompt: prompt, sampling: Fixture.sampled
        ).tokens
        let scripted = Scripted(
            try Fixture.drafter(target: target, settings: Self.coupled), mode: .native,
            vocabulary: vocabulary)
        let fixed: [Int32] = [5, 9, 2, 77, 31, 0, 13]
        scripted.selectionOverride = { _ in
            CBv2MTPProposalSelection { logits in MLXArray(Array(fixed.prefix(logits.dim(0)))) }
        }
        let run = try await collect(
            target, drafter: scripted, prompt: prompt, sampling: Fixture.sampled)
        #expect(run.tokens == expected)
        #expect(!scripted.proposals.isEmpty)
        for round in scripted.proposals {
            let tokens = round.flatMap { Self.ints($0) }
            #expect(tokens == Array(fixed.prefix(tokens.count)))
        }
        // A selection without an answer leaves the argmax: the same proposals
        // as no selection at all.
        let plain = Scripted(
            try Fixture.drafter(target: target), mode: .native, vocabulary: vocabulary)
        _ = try await collect(target, drafter: plain, prompt: prompt, sampling: Fixture.sampled)
        let declined = Scripted(
            try Fixture.drafter(target: target, settings: Self.coupled), mode: .native,
            vocabulary: vocabulary)
        declined.selectionOverride = { _ in CBv2MTPProposalSelection { _ in nil } }
        _ = try await collect(target, drafter: declined, prompt: prompt, sampling: Fixture.sampled)
        #expect(
            declined.proposals.map { $0.flatMap { Self.ints($0) } }
                == plain.proposals.map { $0.flatMap { Self.ints($0) } })
    }

    /// Greedy rows get no selection; a drafter that does not opt in is never
    /// asked.
    @Test(.timeLimit(.minutes(3)))
    func greedyRowsAndOptedOutDraftersSeeNothing() async throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let prompt = Fixture.prompt(Self.promptLength, salt: 5)
        let greedyExpected = try await collect(
            target, drafter: nil, prompt: prompt, sampling: Fixture.greedy
        ).tokens
        let greedy = Scripted(
            try Fixture.drafter(target: target, settings: Self.coupled), mode: .native,
            vocabulary: target.config.vocabSize)
        let greedyRun = try await collect(
            target, drafter: greedy, prompt: prompt, sampling: Fixture.greedy)
        #expect(greedyRun.tokens == greedyExpected)
        #expect(!greedy.selections.isEmpty && greedy.selections.allSatisfy { $0 == nil })

        let off = Scripted(
            try Fixture.drafter(target: target), mode: .native,
            vocabulary: target.config.vocabSize)
        #expect(!off.couplesSampledProposals)
        _ = try await collect(target, drafter: off, prompt: prompt, sampling: Fixture.sampled)
        #expect(off.selections.isEmpty)
        #expect(try #require(off.proposals.first).count > 0)
    }

    /// The race's float32 working set is charged only when it is on.
    @Test func admissionChargesTheRaceOnlyWhenOn() throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let off = try Fixture.drafter(target: target)
        let on = try Fixture.drafter(target: target, settings: Self.coupled)
        #expect(off.coupledProposalBytes == 0)
        #expect(on.coupledProposalBytes == 11 * 7 * target.config.vocabSize * 4)
        #expect(on.blockBytes == off.blockBytes + on.coupledProposalBytes)
    }

    /// The step-sampler seam: the default sampler's race, nil elsewhere.
    @Test func theSamplerSeamIsTheDefaultSamplersRace() throws {
        try Device.withDefaultDevice(.cpu) {
            let vocabulary = 129
            let logits = Self.probe(vocabulary).reshaped([7, 1, vocabulary])
            let params = Array(repeating: Fixture.sampled, count: 7)
            let ids = Array(repeating: CBv2RequestID(3), count: 7)
            let bases = (0 ..< 7).map { 11 + $0 }
            let sampler = CBv2DefaultSampler(fallbackSeed: 5)
            let seam = try #require(
                (sampler as any CBv2StepSampler).mtpCoupledDraftProposal(
                    logits: logits, params: params, requestIDs: ids, stepBases: bases))
            #expect(
                Self.ints(seam)
                    == Self.ints(
                        sampler.mtpCoupledProposal(
                            logits: logits, params: params, requestIDs: ids, stepBases: bases)))
            #expect(
                (CBv2GreedySampler() as any CBv2StepSampler).mtpCoupledDraftProposal(
                    logits: logits, params: params, requestIDs: ids, stepBases: bases) == nil)
        }
    }
}
