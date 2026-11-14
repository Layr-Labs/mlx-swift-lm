// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Batched requests (`CBv2MTPBatchedRequestDrafter`): the requests of one
/// step share the drafter's context commit, block forward and head product.
/// The emitted tokens must be the target's alone; the batched passes must
/// leave every request's context and proposals as the per-request calls do;
/// and a drafter that does not opt in keeps the per-request calls.
@Suite(.serialized, .tags(.integration))
struct MuseGlimmerDFlashBatchedRequestTests {
    typealias Fixture = MuseGlimmerDFlashEngineFixture
    typealias Scripted = MuseGlimmerDFlashScriptedDrafter

    static let maxTokens = 40
    static let batched = MuseGlimmerDFlashDrafter.Settings(
        maximumSpeculativeRows: 2, batchesRequests: true)
    static let batchedCoupled = MuseGlimmerDFlashDrafter.Settings(
        maximumSpeculativeRows: 2, couplesSampledProposals: true, batchesRequests: true)

    /// The requests submitted together; tokens in submission order.
    private func collect(
        _ target: MuseGlimmerDFlashTinyTarget, drafter: (any CBv2MTPDrafter)?, prompts: [[Int]],
        samplings: [CBv2SamplingParams], verification: CBv2MTPVerificationMode = .serialTarget,
        speculativeRows: Int = 2
    ) async throws -> (tokens: [[Int]], metrics: CBv2MTPMetrics?) {
        let (engine, backend) = try Fixture.engine(
            target: target, drafter: drafter, verification: verification,
            concurrentRequests: prompts.count, speculativeRows: speculativeRows)
        let streams = try prompts.indices.map { index in
            try engine.submit(
                Fixture.request(
                    UInt64(index + 1), prompt: prompts[index], maxTokens: Self.maxTokens,
                    sampling: samplings[index]))
        }
        var results: [[Int]] = []
        for stream in streams {
            let result = await cbv2SchedCollect(stream)
            #expect(result.finishReason == .length && result.tokens.count == Self.maxTokens)
            results.append(result.tokens)
        }
        let metrics = engine.mtpMetricsSnapshot()
        await engine.shutdown()
        #expect(backend.bytesReserved == 0 && backend.bytesWired == 0)
        return (results, metrics)
    }

    /// Two rows that speculate together start their rounds in one call and
    /// emit the target's tokens, whatever is proposed and accepted.
    @Test(
        .timeLimit(.minutes(4)),
        arguments: [CBv2MTPVerificationMode.serialTarget, .rectangular])
    func batchedRoundsKeepTheTargetsOutput(verification: CBv2MTPVerificationMode) async throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let prompts = [Fixture.prompt(37, salt: 5), Fixture.prompt(52, salt: 6)]
        let samplings = [Fixture.sampled, Fixture.greedy]
        let expected = try await collect(
            target, drafter: nil, prompts: prompts, samplings: samplings
        ).tokens
        let scripts = zip(prompts, expected).map { $0 + $1 }
        for settings in [Self.batched, Self.batchedCoupled] {
            for mode in [Scripted.Mode.native, .right, .wrong(at: 2)] {
                let scripted = Scripted(
                    try Fixture.drafter(target: target, settings: settings), mode: mode,
                    scripts: scripts, vocabulary: target.config.vocabSize)
                let run = try await collect(
                    target, drafter: scripted, prompts: prompts, samplings: samplings,
                    verification: verification)
                let label = "\(mode) \(verification) coupled \(settings.couplesSampledProposals)"
                #expect(run.tokens == expected, "\(label)")
                #expect(try #require(run.metrics).rounds > 0, "\(label)")
                #expect(scripted.roundStarts.contains(2), "\(label)")
                #expect(scripted.states.allSatisfy { $0.released && $0.pending == nil })
            }
        }
    }

    /// Three rows over two speculative slots decode plainly; each step's
    /// observations reach the drafter in one call, and the output stays
    /// the target's. Without the setting, every row keeps its own calls.
    @Test(.timeLimit(.minutes(4)))
    func plainStepsObserveTheirRowsInOneCall() async throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let prompts = [
            Fixture.prompt(33, salt: 7), Fixture.prompt(41, salt: 8), Fixture.prompt(48, salt: 9),
        ]
        let samplings = [Fixture.sampled, Fixture.greedy, Fixture.sampled]
        let expected = try await collect(
            target, drafter: nil, prompts: prompts, samplings: samplings
        ).tokens
        let on = Scripted(
            try Fixture.drafter(target: target, settings: Self.batched), mode: .native,
            vocabulary: target.config.vocabSize)
        let batchedRun = try await collect(
            target, drafter: on, prompts: prompts, samplings: samplings)
        #expect(batchedRun.tokens == expected)
        #expect(on.observationBatches.contains { $0 >= 3 })

        let off = Scripted(
            try Fixture.drafter(target: target), mode: .native,
            vocabulary: target.config.vocabSize)
        #expect(!off.batchesRequests)
        let plainRun = try await collect(
            target, drafter: off, prompts: prompts, samplings: samplings)
        #expect(plainRun.tokens == expected)
        #expect(off.observationBatches.isEmpty && off.roundStarts.isEmpty)
        #expect(!off.observedPositions.isEmpty)
    }

    /// One batched commit leaves each context as its own commits do, through
    /// duplicates, a gap and a chunk longer than the window; one batched
    /// forward drafts each block as its own forward does.
    @Test func batchedPassesMatchPerStatePasses() throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let model = try Fixture.drafter(target: target).model
        let c = model.configuration
        func rows(_ count: Int, _ key: UInt64) -> MLXArray {
            MLXRandom.normal([count, c.contextWidth], key: MLXRandom.key(key))
        }
        // Per state: (rows, position; nil continues).
        let histories: [[(rows: MLXArray, position: Int?)]] = [
            [(rows(5, 1), 0), (rows(3, 2), nil)],
            [(rows(40, 3), 0), (rows(4, 4), 38), (rows(1, 5), nil)],
            [(rows(4, 6), 0), (rows(6, 7), 10)],
        ]
        let single = histories.map { _ in model.makeContextState() }
        for (state, history) in zip(single, histories) {
            for entry in history {
                model.commit(
                    features: entry.rows, at: entry.position ?? state.nextPosition, to: state)
            }
        }
        let batchedStates = histories.map { _ in model.makeContextState() }
        var commits: [MuseGlimmerDFlashDraftModel.ContextCommit] = []
        for step in 0 ..< 3 {
            for (state, history) in zip(batchedStates, histories) where step < history.count {
                commits.append(
                    .init(features: history[step].rows, at: history[step].position, to: state))
            }
        }
        model.commit(commits)
        eval(single.flatMap(\.arrays) + batchedStates.flatMap(\.arrays))
        for (index, (a, b)) in zip(single, batchedStates).enumerated() {
            #expect(a.nextPosition == b.nextPosition && a.firstPosition == b.firstPosition)
            #expect(a.heldPositions == b.heldPositions, "state \(index)")
            let x = try #require(a.visibleContext())
            let y = try #require(b.visibleContext())
            #expect(allClose(x.keys, y.keys, atol: 1e-5).item(Bool.self), "keys \(index)")
            #expect(allClose(x.values, y.values, atol: 1e-5).item(Bool.self), "values \(index)")
        }
        #expect(single.map(\.nextPosition) == [8, 43, 16])
        #expect(single[1].count == c.slidingWindow)

        let embeddings = (0 ..< 3).map {
            MLXRandom.normal([c.blockSize, c.hiddenSize], key: MLXRandom.key(UInt64(10 + $0)))
        }
        let together = model.draftBlocks(embeddings: embeddings, states: batchedStates)
        for (index, state) in single.enumerated() {
            let alone = model.draftBlock(embeddings: embeddings[index], state: state)
            #expect(allClose(alone, together[index], atol: 1e-4).item(Bool.self), "block \(index)")
            let heads = [alone, together[index]].map {
                argMax(target.dflashHeadProduct($0[1 ..< 8, 0...]), axis: -1).asType(.int32)
                    .asArray(Int32.self)
            }
            #expect(heads[0] == heads[1], "proposals \(index)")
        }
    }

    /// Accepted rows held for the next round's pass end in the context the
    /// immediate commits give, and the rounds propose the same tokens.
    @Test func deferredAcceptedRowsMatchImmediateCommits() throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let width = target.dflashContextWidth
        func hidden(_ count: Int, _ key: UInt64) -> MLXArray {
            MLXRandom.normal([1, count, width], key: MLXRandom.key(key))
        }
        func tokens(_ count: Int) -> MLXArray {
            MLXArray([Int32](repeating: 3, count: count)).reshaped([1, count])
        }
        var proposals: [[[Int32]]] = []
        var contexts: [MuseGlimmerDFlashContextState] = []
        for settings in [
            MuseGlimmerDFlashDrafter.Settings(maximumSpeculativeRows: 2), Self.batched,
        ] {
            let drafter = try Fixture.drafter(target: target, settings: settings)
            let state = drafter.makeRequestState() as! MuseGlimmerDFlashDrafter.State
            drafter.observeCommittedTarget(
                .init(tokens: tokens(20), hidden: hidden(20, 1)), startingAt: 0,
                requestState: state)
            var rounds: [[Int32]] = []
            for (round, accepted) in [2, 0, 3].enumerated() {
                var drafted: [Int32] = []
                for _ in 0 ..< 7 {
                    let step = drafter.draftStep(
                        tokens: MLXArray([Int32(5 + round)]).reshaped([1, 1]),
                        hidden: hidden(1, UInt64(100 + round)), shortlist: nil,
                        requestState: state)
                    drafted += step.tokens.asArray(Int32.self)
                }
                try drafter.requestStateDidFinishEvaluation(state)
                rounds.append(drafted)
                drafter.finalizeRound(
                    requestState: state, confirmedInputTokens: 1 + accepted,
                    committedDraftTokens: MLXArray.zeros([1, accepted], dtype: .int32),
                    committedTargetHidden: hidden(accepted, UInt64(200 + round)))
            }
            #expect((state.pending?.dim(0) ?? 0) == (settings.batchesRequests ? 3 : 0))
            // A plain step after the rounds: the carry, then its own row.
            drafter.observeCommittedTarget(
                .init(tokens: tokens(1), hidden: hidden(1, 300)), startingAt: 27,
                requestState: state)
            drafter.observeCommittedTarget(
                .init(tokens: tokens(1), hidden: hidden(1, 301)), startingAt: 28,
                requestState: state)
            try drafter.requestStateDidFinishEvaluation(state)
            #expect(state.pending == nil)
            #expect(state.committedInputCount == 29 && state.context.nextPosition == 29)
            eval(drafter.evaluationTargets(for: state))
            proposals.append(rounds)
            contexts.append(state.context)
        }
        #expect(proposals[0] == proposals[1])
        let off = try #require(contexts[0].visibleContext())
        let on = try #require(contexts[1].visibleContext())
        #expect(contexts[0].heldPositions == contexts[1].heldPositions)
        #expect(allClose(off.keys, on.keys, atol: 1e-5).item(Bool.self))
        #expect(allClose(off.values, on.values, atol: 1e-5).item(Bool.self))
    }

    /// Pending rows are charged per request only when batching is on; the
    /// setting is off by default.
    @Test func admissionChargesPendingRowsOnlyWhenOn() throws {
        let target = MuseGlimmerDFlashTinyTarget()
        let off = try Fixture.drafter(target: target)
        let on = try Fixture.drafter(target: target, settings: Self.batched)
        #expect(off.pendingBytes == 0)
        #expect(on.pendingBytes == 2 * 7 * target.dflashContextWidth * 4)
        let limits = CBv2MTPAllocationLimits(maximumPrefillTokens: 64, maximumDraftTokens: 7)
        let a = try #require(off.boundedRequestAllocation(limits: limits))
        let b = try #require(on.boundedRequestAllocation(limits: limits))
        #expect(b.resident == a.resident + [.init(logicalBytes: on.pendingBytes)])
        #expect(b.working == a.working && b.ownerWorking == a.ownerWorking)
        #expect(!MuseGlimmerDFlashDrafter.Settings().batchesRequests)
        #expect(!MuseGlimmerDFlashDrafter.Settings.verifying(target).batchesRequests)
        #expect(
            MuseGlimmerDFlashDrafter.Settings.verifying(target, batchesRequests: true)
                .batchesRequests)
    }
}
