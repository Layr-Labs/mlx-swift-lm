import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Forwards every call to an Inkling 2 DSpark drafter and records what the
/// engine hands it: each committed-target observation (tokens and feature
/// rows, in call order, per request state), each round's drafts, and each
/// state's committed input count when it is released.
final class Inkling2DepthZeroRecordingDrafter: CBv2MTPBatchedBlockDrafter, @unchecked Sendable {
    struct StateLog {
        var observedTokens: [MLXArray] = []
        var observedHidden: [MLXArray] = []
        var drafts: [MLXArray] = []
        var finalizedHidden: [MLXArray] = []
        var committedInputCount = 0
        var maximumPendingRows = 0
    }

    let inner: Inkling2DSparkDrafter
    private(set) var logs: [ObjectIdentifier: StateLog] = [:]
    private(set) var order: [ObjectIdentifier] = []
    private(set) var discardedRounds = 0

    init(_ inner: Inkling2DSparkDrafter) { self.inner = inner }

    private func update(_ state: any CBv2MTPRequestState, _ body: (inout StateLog) -> Void) {
        let key = ObjectIdentifier(state)
        if logs[key] == nil { order.append(key) }
        body(&logs[key, default: StateLog()])
    }

    /// Logs in request-state creation order (requests admit in id order).
    var orderedLogs: [StateLog] { order.compactMap { logs[$0] } }

    var mtpTargetIdentity: ObjectIdentifier? { inner.mtpTargetIdentity }
    var requiredVerificationMode: CBv2MTPVerificationMode? { inner.requiredVerificationMode }
    var maximumDraftTokens: Int? { inner.maximumDraftTokens }
    var maximumSpeculativeBatch: Int? { inner.maximumSpeculativeBatch }
    var supportsTargetPrefixAcceptance: Bool { inner.supportsTargetPrefixAcceptance }
    var supportsEarlyDraftSubmission: Bool { inner.supportsEarlyDraftSubmission }
    var prefersBatchedRectangularAttention: Bool { inner.prefersBatchedRectangularAttention }
    var commitsAcceptedPrefixPerRow: Bool { inner.commitsAcceptedPrefixPerRow }
    var requestStateBytesPerToken: Int { inner.requestStateBytesPerToken }
    var requestStateTokenGranularity: Int { inner.requestStateTokenGranularity }
    var requestStateTokenAllocationPadding: Int { inner.requestStateTokenAllocationPadding }
    var requestStateAllocationSpecs: [CBv2AuxiliaryAllocationSpec]? {
        inner.requestStateAllocationSpecs
    }
    var requiresCommittedObservationFence: Bool { inner.requiresCommittedObservationFence }
    var supportsOrdinaryDecodeHistory: Bool { inner.supportsOrdinaryDecodeHistory }
    var supportsDepthZeroChaining: Bool { inner.supportsDepthZeroChaining }
    var draftShortlistSize: Int? { inner.draftShortlistSize }
    var submitsEveryDraftStepEarly: Bool { inner.submitsEveryDraftStepEarly }

    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        inner.prepare(rows: rows)
    }
    func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        inner.draftStep(tokens: tokens, hidden: hidden, prepared: prepared)
    }
    func makeRequestState() -> any CBv2MTPRequestState { inner.makeRequestState() }
    func configureRequestState(
        _ requestState: any CBv2MTPRequestState, maximumSequenceLength: Int
    ) throws {
        try inner.configureRequestState(requestState, maximumSequenceLength: maximumSequenceLength)
    }
    func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState
    ) {
        inner.observeCommittedTarget(observation, requestState: requestState)
        let pending = (requestState as? Inkling2DSparkDrafter.State)?.pendingRows ?? 0
        update(requestState) {
            $0.observedTokens.append(observation.tokens)
            $0.observedHidden.append(observation.hidden)
            $0.maximumPendingRows = max($0.maximumPendingRows, pending)
        }
    }
    func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        let result = inner.draftStep(
            tokens: tokens, hidden: hidden, shortlist: shortlist, requestState: requestState)
        update(requestState) { $0.drafts.append(result.tokens) }
        return result
    }
    func draftBlock(
        tokens: [MLXArray], hidden: [MLXArray], requestStates: [any CBv2MTPRequestState],
        depth: Int
    ) -> MLXArray {
        let drafts = inner.draftBlock(
            tokens: tokens, hidden: hidden, requestStates: requestStates, depth: depth)
        for (row, state) in requestStates.enumerated() {
            update(state) { $0.drafts.append(drafts[row ..< row + 1]) }
        }
        return drafts
    }
    func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] {
        inner.evaluationTargets(for: requestState)
    }
    func requestStateDidFinishEvaluation(_ requestState: any CBv2MTPRequestState) throws {
        try inner.requestStateDidFinishEvaluation(requestState)
    }
    func finalizeRound(
        requestState: any CBv2MTPRequestState, confirmedInputTokens: Int,
        committedDraftTokens: MLXArray, committedTargetHidden: MLXArray
    ) {
        inner.finalizeRound(
            requestState: requestState, confirmedInputTokens: confirmedInputTokens,
            committedDraftTokens: committedDraftTokens,
            committedTargetHidden: committedTargetHidden)
        update(requestState) { $0.finalizedHidden.append(committedTargetHidden) }
    }
    func discardRound(requestState: any CBv2MTPRequestState) {
        discardedRounds += 1
        inner.discardRound(requestState: requestState)
    }
    func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        let count = requestState.committedInputCount
        update(requestState) { $0.committedInputCount = count }
        inner.releaseRequestState(requestState)
    }
}

/// Bit patterns of a lazy array's values (evaluated here, after the engine drained).
private func bits(_ arrays: [MLXArray]) -> [UInt32] {
    arrays.flatMap { $0.asType(.float32).asArray(Float.self).map(\.bitPattern) }
}

private func ints(_ arrays: [MLXArray]) -> [Int32] {
    arrays.flatMap { $0.asType(.int32).asArray(Int32.self) }
}

/// Depth-zero plans of a drafter that takes ordinary-decode history run as
/// chained ordinary decode instead of seed steps
/// (`CBv2MTPRoundDriver.depthZeroChaining`). The tiny Inkling 2 target and
/// random DSpark drafter of `Inkling2DSparkTargetEngineTests` run a forced
/// schedule (rounds in even 6-token blocks, depth zero in odd ones, depth zero
/// from 24 tokens on) with chaining off (seed steps, the old path) and on:
/// tokens, every round's drafts and features, and every observation the
/// drafter receives must be bit-identical, and the chained run must chain and
/// need no seed step (one row). The only difference allowed: a row's final
/// step, which a seed step observes at launch while the chained step finds the
/// row finished (its history released) at finalize. The schedule keeps the
/// tail round-free because a chained boundary cannot see a co-batched row
/// finish before its in-flight token lands (a round there would run one step
/// later than with seed steps; tokens are unaffected either way).
@Suite(.serialized, .tags(.integration))
struct Inkling2DSparkDepthZeroChainedTests {
    struct Run {
        var tokens: [[Int]]
        var metrics: CBv2MTPMetrics?
        var chainedSteps: Int
        var logs: [Inkling2DepthZeroRecordingDrafter.StateLog]
        var discardedRounds: Int
    }

    static var dtype: DType { .bfloat16 }

    static func engine(
        _ model: Inkling2Model, drafter: (any CBv2MTPDrafter)?, rows: Int, mtp: CBv2MTPConfig
    ) throws -> EngineV2 {
        let kinds = model.cbv2LayerKinds
        let types = Array(repeating: dtype, count: kinds.count)
        let paged = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 128 << 20, dtype: dtype, maxPrefillChunk: 32,
                nominalMaxSequenceLength: 256, segmentSizeBytes: 1 << 20, layerDTypes: types))
        let storage = paged.makeLayerCaches()
        let caches = model.newCacheV2 { index, _ in storage[index] }
        return EngineV2(
            model: CBv2SteppableLanguageModelAdapter(model), layerKinds: kinds, backend: paged,
            cacheProvider: CBv2LayerCacheBank(caches: caches), sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: rows, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 16, enablePrefixCache: false),
            admissionConfig: .init(watermarkFraction: 0),
            mtpDrafter: drafter,
            mtpConfig: drafter == nil ? .init() : mtp)
    }

    static func prompts(_ rows: Int) -> [[Int]] {
        (0 ..< rows).map {
            SyntheticModel.tokens(count: 9 + 13 * $0, vocabularySize: 60, seed: 40 + $0)
        }
    }

    /// Rounds in even 6-token blocks of the shortest row, depth zero in odd
    /// ones and from 24 tokens on.
    static let schedule: @Sendable ([Int]) -> Int? = { counts in
        let count = counts.min() ?? 0
        return count >= 24 || (count / 6) % 2 == 1 ? 0 : nil
    }

    static func run(
        _ model: Inkling2Model, drafter: Inkling2DSparkDrafter?, rows: Int, tokens: Int,
        mtp: CBv2MTPConfig, chaining: Bool, scheduled: Bool
    ) async throws -> Run {
        let recording = drafter.map(Inkling2DepthZeroRecordingDrafter.init)
        let engine = try engine(model, drafter: recording, rows: rows, mtp: mtp)
        if let driver = engine.loopForTesting.mtp {
            driver.depthZeroChaining = chaining
            if scheduled { driver.planDepthLimitForTesting = schedule }
        }
        let streams = try prompts(rows).enumerated().map { index, prompt in
            try engine.submit(
                .init(
                    id: .init(UInt64(index + 1)), promptTokens: prompt,
                    sampling: .init(temperature: 0), maxTokens: tokens,
                    prefixCacheEnabled: false))
        }
        var outputs: [[Int]] = []
        for stream in streams {
            outputs.append(
                try await cbv2CollectFinished(stream, timeoutSeconds: 300, expectedTokens: tokens)
                    .tokens)
        }
        let metrics = engine.mtpMetricsSnapshot()
        let chained = engine.chainedStepCount
        await engine.shutdown()
        return Run(
            tokens: outputs, metrics: metrics, chainedSteps: chained,
            logs: recording?.orderedLogs ?? [], discardedRounds: recording?.discardedRounds ?? 0)
    }

    static func expectSameHistory(_ seeded: Run, _ chained: Run) {
        #expect(seeded.logs.count == chained.logs.count)
        for (s, c) in zip(seeded.logs, chained.logs) {
            #expect(ints(s.drafts) == ints(c.drafts), "drafts differ")
            #expect(bits(s.finalizedHidden) == bits(c.finalizedHidden), "round features differ")
            // Every observation of the chained run, in order and bit for bit;
            // the seed run may add the row's final step (observed at launch).
            let shared = c.observedTokens.count
            #expect((shared ... shared + 1).contains(s.observedTokens.count))
            #expect(
                ints(Array(s.observedTokens.prefix(shared))) == ints(c.observedTokens),
                "observed tokens differ")
            #expect(
                bits(Array(s.observedHidden.prefix(shared))) == bits(c.observedHidden),
                "observed features differ")
            let extra = s.observedTokens.dropFirst(shared).reduce(0) { $0 + $1.dim(1) }
            #expect(s.committedInputCount == c.committedInputCount + extra)
            #expect(
                (c.maximumPendingRows ... c.maximumPendingRows + extra).contains(
                    s.maximumPendingRows))
        }
    }

    @Test(
        "forced depth-zero stretches chain with identical tokens, drafts and history",
        arguments: [1, 8])
    func forcedScheduleChainsExactly(rows: Int) async throws {
        let model = try Inkling2DSparkTargetEngineTests.target(dtype: Self.dtype)
        let drafter = try Inkling2DSparkTargetEngineTests.drafter(target: model, dtype: Self.dtype)
        let fixed = Inkling2DSparkDrafter.defaultMTPConfig(maxDraftTokens: 4, adaptive: false)
        let tokens = 40

        let off = try await Self.run(
            model, drafter: nil, rows: rows, tokens: tokens, mtp: fixed, chaining: true,
            scheduled: false)
        let seeded = try await Self.run(
            model, drafter: drafter, rows: rows, tokens: tokens, mtp: fixed, chaining: false,
            scheduled: true)
        let chained = try await Self.run(
            model, drafter: drafter, rows: rows, tokens: tokens, mtp: fixed, chaining: true,
            scheduled: true)

        #expect(off.tokens.allSatisfy { $0.count == tokens })
        #expect(seeded.tokens == off.tokens)
        #expect(chained.tokens == off.tokens)
        let seededMetrics = try #require(seeded.metrics)
        let chainedMetrics = try #require(chained.metrics)
        print(
            "seeded rounds=\(seededMetrics.rounds) drafted=\(seededMetrics.draftedTokens) "
                + "accepted=\(seededMetrics.acceptedTokens) observations="
                + "\(seeded.logs.map(\.observedTokens.count)) chained observations="
                + "\(chained.logs.map(\.observedTokens.count))")
        #expect(chainedMetrics.rounds > 0)
        #expect(chainedMetrics.rounds == seededMetrics.rounds)
        #expect(chainedMetrics.acceptedTokens == seededMetrics.acceptedTokens)
        #expect(chainedMetrics.draftedTokens == seededMetrics.draftedTokens)
        // Seed steps carried every depth-zero step; chained, none is needed:
        // a carry stored at the stretch's last finalize starts the next round.
        #expect(seededMetrics.seedSteps > 0)
        if rows == 1 {
            #expect(chainedMetrics.seedSteps == 0)
        } else {
            #expect(chainedMetrics.seedSteps < seededMetrics.seedSteps)
        }
        #expect(chained.chainedSteps > seeded.chainedSteps)
        Self.expectSameHistory(seeded, chained)
        #expect(chained.discardedRounds == 0 && seeded.discardedRounds == 0)
        print(
            "rows=\(rows) rounds=\(chainedMetrics.rounds) accepted=\(chainedMetrics.acceptedTokens) "
                + "seed steps \(seededMetrics.seedSteps) -> \(chainedMetrics.seedSteps), "
                + "chained steps \(seeded.chainedSteps) -> \(chained.chainedSteps), "
                + "max pending rows \(chained.logs.map(\.maximumPendingRows))")
    }

    @Test(
        "adaptive and fixed depth: drafter on equals off, chaining on or off, row 0 of 8 equals solo"
    )
    func adaptiveAndFixedEqualDrafterOff() async throws {
        let model = try Inkling2DSparkTargetEngineTests.target(dtype: Self.dtype)
        let drafter = try Inkling2DSparkTargetEngineTests.drafter(target: model, dtype: Self.dtype)
        let tokens = 32
        let configs: [(String, CBv2MTPConfig)] = [
            ("adaptive", Inkling2DSparkDrafter.defaultMTPConfig(adaptive: true)),
            ("fixed4", Inkling2DSparkDrafter.defaultMTPConfig(maxDraftTokens: 4, adaptive: false)),
            // Every plan at depth zero: the whole request is a chained stretch.
            (
                "depth0",
                CBv2MTPConfig(
                    enabled: true, maxDraftTokens: 7, maxSpeculativeBatch: 8,
                    fixedDraftTokens: 7, fixedDraftTokensByDecodeRows: [0],
                    verificationMode: .rectangular, maxAutomaticRectangularTokens: 64,
                    speculativeBatchLimit: 8)
            ),
        ]
        var solo: [Int]?
        for rows in [1, 8] {
            let off = try await Self.run(
                model, drafter: nil, rows: rows, tokens: tokens, mtp: .init(), chaining: true,
                scheduled: false)
            if rows == 1 { solo = off.tokens[0] } else { #expect(off.tokens[0] == solo) }
            for (name, config) in configs {
                for chaining in [false, true] {
                    let on = try await Self.run(
                        model, drafter: drafter, rows: rows, tokens: tokens, mtp: config,
                        chaining: chaining, scheduled: false)
                    #expect(on.tokens == off.tokens, "\(name) rows=\(rows) chaining=\(chaining)")
                    #expect(on.tokens[0] == solo, "\(name) rows=\(rows) row 0 vs solo")
                    let metrics = try #require(on.metrics)
                    if name == "depth0" {
                        #expect(metrics.rounds == 0)
                        if chaining {
                            // Only steps beside a prompt chunk (MTP prefill path) seed.
                            if rows == 1 { #expect(metrics.seedSteps == 0) }
                            #expect(on.chainedSteps > tokens / 2)
                        } else {
                            #expect(metrics.seedSteps > 0)
                        }
                    }
                    print(
                        "\(name) rows=\(rows) chaining=\(chaining) rounds=\(metrics.rounds) "
                            + "accepted=\(metrics.acceptedTokens) seed=\(metrics.seedSteps) "
                            + "chained=\(on.chainedSteps) depths=\(metrics.depthSelections) "
                            + "fallbacks=\(metrics.controllerFallbacks)")
                }
            }
        }
    }
}

/// Review check (c)/(d): a depth-zero stretch is never absorbing. Without a
/// policy margin (an ordinary-decode carry) the marginal policy never picks a
/// smaller depth than with one, and while the head-cost ratio is below 0.5
/// the margin cannot decide between depth zero and a round at all. Rounds
/// return through the controller's probe cadence, which chained depth-zero
/// steps advance: at most 256 plans pass between probes, which also bounds
/// the feature rows a request queues between rounds.
@Suite(.serialized)
struct Inkling2DSparkDepthZeroPolicyTests {
    @Test(
        "a missing margin never lowers the marginal depth; below h = 0.5 it never decides depth zero"
    )
    func missingMarginNeverLowersDepth() {
        var generator = SystemRandomNumberGenerator()
        for _ in 0 ..< 4000 {
            let probabilities = (0 ..< 7).map { _ in Double.random(in: 0 ... 1, using: &generator) }
            let h = Double.random(in: 0 ... 1.5, using: &generator)
            let margin = Double.random(in: 0 ... 12, using: &generator)
            func depth(_ margin: Double?) -> Int {
                CBv2MTPMarginalDepthPolicy.selectDepth(
                    offeredDepth: 7, remainingTokens: 100, verificationLimit: 7,
                    acceptanceProbabilities: probabilities, previousTargetTopTwoMargin: margin,
                    headStepCostRatio: h)
            }
            #expect(depth(nil) >= depth(margin))
            if h < 0.5 { #expect((depth(nil) == 0) == (depth(margin) == 0)) }
        }
    }

    @Test("chained depth-zero plans advance the probe cadence; recovered acceptance drafts again")
    func depthZeroIsNotAbsorbing() throws {
        let model = try Inkling2DSparkTargetEngineTests.target(dtype: .bfloat16)
        let drafter = try Inkling2DSparkTargetEngineTests.drafter(target: model, dtype: .bfloat16)
        let driver = try #require(
            CBv2MTPRoundDriver.build(
                model: CBv2SteppableLanguageModelAdapter(model), drafter: drafter,
                config: Inkling2DSparkDrafter.defaultMTPConfig(adaptive: true)))
        #expect(driver.usesMarginalPolicy && driver.feedsHistoryFromOrdinaryDecode)
        let id = CBv2RequestID(1)
        // h = 0.1: C0 = 10 ms, Ck = 10 (1 + 0.1 k) ms.
        func cost(_ depth: Int) -> UInt64 { UInt64(10_000_000 + 1_000_000 * depth) }

        /// One plan as the engine makes it: controller and marginal policy,
        /// then a round (rejected in the bad phase, fully accepted in the good
        /// one), the isolated warm-up probe, or a chained depth-zero step.
        func plan(good: Bool) -> (depth: Int, isExploration: Bool) {
            driver.beginPlan(plannedDecodeRows: 1, canSpeculate: true, rowIDs: [id])
            var decision = driver.planDecision
            if driver.shouldApplyMarginalPolicyToPlan, decision.depth > 0 {
                let marginal = driver.marginalDepth(
                    for: id, offeredDepth: decision.depth, remainingTokens: 10_000,
                    verificationLimit: driver.maximumAutomaticDepth(plannedDecodeRows: 1),
                    decodeRowBucket: decision.decodeRowBucket)
                driver.clampPlanDepth(to: marginal, reason: "marginal_policy")
                decision = driver.planDecision
            }
            let k = decision.depth
            let probe = decision.isExploration && k == 0
            let measurement = CBv2MTPStepMeasurement(
                decision: decision, actualDepth: k, costEligible: true,
                chained: k == 0 && !probe, seedOnly: probe,
                workloadGeneration: driver.workloadGeneration)
            if k > 0 {
                driver.recordStepAcceptance(
                    drafted: k, accepted: good ? k : 0, observedDrafts: k,
                    decodeRowBucket: decision.decodeRowBucket, measurement: measurement)
            }
            driver.recordStepCost(
                measurement, wallTimeNanos: cost(k), finalizedPlainWork: k == 0,
                finalizedSeedIDs: probe ? [id] : [], finalizedVerification: k > 0,
                claimedSeedCostNanos: 0)
            if k > 0 {
                driver.observeRequestAcceptance(
                    id: id, draftedDepth: k, acceptedDepth: good ? k : 0,
                    rejectionObserved: !good, endedByTruncation: false)
            }
            return (k, decision.isExploration)
        }

        // Bad phase: every draft is rejected. Depth zero takes over, yet
        // rounds keep coming at the controller's probe cadence.
        var gap = 0
        var longestGap = 0
        var badRounds = 0
        var badDepthZero = 0
        for _ in 0 ..< 2000 {
            let step = plan(good: false)
            if step.depth > 0 {
                badRounds += 1
                longestGap = max(longestGap, gap)
                gap = 0
            } else {
                badDepthZero += 1
                gap += 1
            }
        }
        longestGap = max(longestGap, gap)
        #expect(badDepthZero > 1900)
        #expect(badRounds >= 7)
        #expect(longestGap <= 257)
        #expect(
            driver.marginalDepth(
                for: id, offeredDepth: 7, remainingTokens: 10_000, verificationLimit: 7,
                decodeRowBucket: 1) == 0)

        // Good phase: probes are accepted, acceptance recovers and the
        // marginal policy drafts on its own again (non-exploration rounds).
        var goodPlans = 0
        var policyRounds = 0
        while goodPlans < 3000 && policyRounds < 20 {
            let step = plan(good: true)
            goodPlans += 1
            if step.depth > 0 && !step.isExploration { policyRounds += 1 }
        }
        #expect(policyRounds >= 20)
        print(
            "bad: rounds=\(badRounds) depth0=\(badDepthZero) longest gap=\(longestGap); "
                + "good: \(goodPlans) plans to 20 policy rounds")
    }
}
