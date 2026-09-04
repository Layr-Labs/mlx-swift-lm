import Testing

@testable import MLXLMCommon

/// Pins the two regimes of `CBv2MTPDepthController`, because in the field they
/// look like a backend bug and are not one.
///
/// Gate G2's parity harness reported `rounds=5, drafted=5, accepted=4` on the
/// contiguous KV backend against `rounds=59, drafted=62, accepted=54` on paged,
/// for the SAME drafter, prompts and 144 emitted tokens — a ~12x asymmetry that
/// reads like paged over-speculating or contiguous being gated off. It is
/// neither. Both numbers are this controller's two attractors, selected by ONE
/// comparison in `decide()`:
///
///     goodput(d) = expectedCommitted(d) / ewmaCost(d)
///
/// Before position 1 has `acceptanceMinSamples` observations, `rate(at: 1)`
/// returns 1, so `expectedCommitted(1) == 2` and depth 1 is chosen exactly when
///
///     cost1 / cost0  <  2 / (1 + hysteresisFraction)  ==  1.9047...
///
/// Above that line the controller settles on depth 0; below it, nearly every
/// step speculates. Nothing else changes; the 12x is one bit.
///
/// Measured on gemma-4-26B-A4B-it-qat-4bit + its qat assistant, M4 Max, B=1,
/// 144 emitted tokens, contiguous KV (`MTPBackendAsymmetryLiveTests`):
///
///   adaptive : cost0 = 10.916 ms (1 sample), cost1 = 25.111 ms (5 samples)
///              -> ratio 2.300 -> "unprofitable" -> 111.82 tok/s
///   forced   : cost1 = 14.214 ms (75 samples, steady state)
///              -> ratio 1.302 -> profitable          -> 128.33 tok/s
///
/// The controller's own estimate is 1.77x the steady-state truth, which is
/// what puts it on the wrong side of the line. The inflation is structural,
/// not noise: a depth-zero plan invalidates every carry
/// (`EngineLoopV2.beginMTPPlan`), so every probe round must SEED first, and
/// `CBv2MTPRoundDriver.recordStepCost` folds that seed's wall time into the
/// round's cost sample (`attributed = wallTimeNanos &+ claimedSeedCostNanos`)
/// while `expectedCommitted` never credits the token the seed emitted. Cost
/// in, token out — so choosing depth 0 manufactures the evidence for choosing
/// depth 0 again. Forced mode needs 3 seeds for 75 rounds; the adaptive
/// controller pays 5 seeds for 5.
///
/// ## What changed, and why the counts in these tests moved
///
/// The unprofitable arm used to collapse to exactly 5 rounds on a doubling
/// cadence (8, 16, 32, 64, 128). That was one probe rule doing two jobs, and
/// it was wrong at the second. The probe was `min(selected + 1, limit)`, so
/// with `selected` resting at 0 it only ever asked about depth 1 — and
/// `limit` is `acceptance.frontier + 1`, a frontier that advances only once a
/// position has actually been drafted `acceptanceMinSamples` times. Declining
/// to explore deeper was therefore also declining to gather the evidence that
/// would justify exploring deeper, and the backoff finished the job by making
/// the tenth probe arrive after a thousand rounds. At THE TEST (Gemma 4 26B,
/// 17,408-token prompt) that produced `depthSelections {0: 1002, 1: 16}` over
/// 1,018 decisions while a FIXED depth 4 ran 1.40x serial on the same prompt.
///
/// The rule now probes the frontier (`min(frontier + 1, maxDepth)`), and backs
/// off only when a probe buys no new evidence at its own position. That is
/// sound because goodput is not monotone in depth: a round pays a fixed setup
/// and verify overhead which amortizes over the tokens it commits, so depth 1
/// can be a real loss while depth 4 is the best arm on the board. A hill-climb
/// that stops at the first losing step never finds that out. The cost is a
/// bounded learning phase — at most `maxDepth * acceptanceMinSamples`
/// productive probes — after which the original geometric backoff resumes to
/// its 256 cap, which `unprofitableCostLearnsTheEnvelopeThenBacksOff` pins end
/// to end.
///
/// Four things these tests defend:
///   1. depth 0 must keep re-probing on a BOUNDED cadence. Losing the backoff
///      either freezes MTP off forever or re-probes unboundedly.
///   2. that cadence must be able to reach every depth in the envelope, not
///      just `selected + 1`.
///   3. the profitable regime must actually speculate on nearly every step.
///   4. the seed-inflated estimate, at the REAL measured costs, is what flips
///      the verdict — so a change to seed attribution shows up here first.
@Suite("CBv2MTPDepthController probe cadence")
struct CBv2MTPProbeCadenceTests {

    /// One decode step's worth of controller interaction, mirroring the engine
    /// loop: `beginMTPPlan` selects a depth, a positive depth needs a seed step
    /// before it can verify, and `finalize` reports the completed step back.
    ///
    /// Depth-zero steps report `chained: true` once the warmup baseline exists,
    /// which is what the contiguous steady state does — the chained fast path
    /// keeps running and `EngineLoopV2.mtpWantsStep` only breaks it for the one
    /// non-chained baseline probe. Chained depth-zero work advances the probe
    /// cadence without contributing a cost sample, exactly as
    /// `recordFinalizedStep` specifies.
    private struct Outcome {
        var roundSteps: [Int] = []
        var roundDepths: [Int] = []
        var drafted = 0
        var accepted = 0

        var rounds: Int { roundSteps.count }
        var gaps: [Int] {
            zip(roundSteps.dropFirst(), roundSteps).map { $0 - $1 }
        }
    }

    /// `costRatio` is cost(depth 1) / cost(depth 0) — the only quantity that
    /// separates the two regimes. Deeper rounds cost `costRatio * depth`.
    private func drive(
        costRatio: Double,
        steps: Int,
        baselineNanos: UInt64 = 30_000_000
    ) -> (outcome: Outcome, controller: CBv2MTPDepthController) {
        let controller = CBv2MTPDepthController(
            maxDepth: CBv2MTPConfig.testedMaxDraftTokens, fixedDepth: nil)
        func cost(_ depth: Int) -> UInt64 {
            guard depth > 0 else { return baselineNanos }
            return UInt64(Double(baselineNanos) * costRatio * Double(depth))
        }

        var outcome = Outcome()
        var hasCarry = false
        for step in 0 ..< steps {
            let decision = controller.select(plannedDecodeRows: 1, canSpeculate: true)
            if decision.depth == 0 {
                let chained = !controller.requiresNonChainedDepthZeroProbe(decision)
                controller.recordFinalizedStep(
                    decision: decision,
                    actualDepth: 0,
                    wallTimeNanos: cost(0),
                    costEligible: true,
                    chained: chained,
                    finalizedPlainWork: true,
                    finalizedVerification: false)
                // A depth-zero plan invalidates every carry (`beginMTPPlan`).
                hasCarry = false
                continue
            }
            guard hasCarry else {
                // Seed step: hidden capture only, no verification, and the
                // driver routes its cost to the pending-seed ledger instead of
                // the controller.
                hasCarry = true
                continue
            }
            // Deterministic 7-of-8 acceptance so the regimes are reproducible.
            let accepted = step % 8 == 3 ? 0 : decision.depth
            controller.observeAcceptance(
                decodeRowBucket: decision.decodeRowBucket,
                drafted: decision.depth,
                accepted: accepted)
            controller.recordFinalizedStep(
                decision: decision,
                actualDepth: decision.depth,
                wallTimeNanos: cost(decision.depth),
                costEligible: true,
                chained: false,
                finalizedPlainWork: false,
                finalizedVerification: true)
            outcome.roundSteps.append(step)
            outcome.roundDepths.append(decision.depth)
            outcome.drafted += decision.depth
            outcome.accepted += accepted
            hasCarry = accepted == decision.depth
        }
        return (outcome, controller)
    }

    @Test("unprofitable cost measures the envelope once, then backs off to the cap")
    func unprofitableCostLearnsTheEnvelopeThenBacksOff() {
        let (outcome, controller) = drive(costRatio: 2.5, steps: 1600)

        // The verdict never changes: this workload is not worth speculating on
        // and the controller never settles on a positive depth.
        #expect(controller.activeDepthForTesting(decodeRowBucket: 1) == 0)
        #expect(outcome.rounds == 84)

        // The learning phase walks the envelope one position at a time, each
        // position held until it carries `acceptanceMinSamples` observations.
        // It is a monotone climb 1 -> maxDepth, not a hill-climb that stalls.
        let maxDepth = CBv2MTPConfig.testedMaxDraftTokens
        #expect(Set(outcome.roundDepths) == Set(1 ... maxDepth))
        let climb = Array(outcome.roundDepths.prefix(78))
        #expect(climb == climb.sorted())
        #expect(climb.first == 1)
        #expect(climb.last == maxDepth)
        for depth in 1 ... maxDepth {
            #expect(
                climb.filter { $0 == depth }.count >= 10,
                "depth \(depth) never reached acceptanceMinSamples")
        }

        // Once every position carries evidence the probe ROTATES the envelope
        // instead of climbing it, so a depth that measured badly once stays
        // revisitable.
        #expect(Array(outcome.roundDepths.suffix(6)) == [1, 2, 3, 4, 5, 6])

        // And the ORIGINAL doubling resumes and saturates: the last probes are
        // 257 steps apart, one round in 257, and the interval is at its cap.
        #expect(Array(outcome.gaps.suffix(6)) == [17, 33, 65, 129, 257, 257])
        #expect(controller.probeIntervalForTesting(decodeRowBucket: 1) == 256)

        // Bounded by construction: `maxDepth * acceptanceMinSamples` productive
        // probes is the whole budget, and it is paid once per bucket.
        #expect(outcome.rounds <= maxDepth * 10 + 16)
    }

    /// The same arm at the horizon the field harness actually ran (200 steps):
    /// still declining, still cheap, and now climbing.
    @Test("the learning phase is a small fraction of an unprofitable run")
    func unprofitableLearningPhaseStaysCheap() {
        let (outcome, controller) = drive(costRatio: 2.5, steps: 200)

        #expect(outcome.rounds == 23)
        #expect(outcome.drafted == 38)
        #expect(outcome.accepted == 34)
        #expect(controller.activeDepthForTesting(decodeRowBucket: 1) == 0)
        // 23 speculative rounds in 200 steps. The old rule spent 5 and learned
        // nothing past depth 1; these 23 have priced depths 1, 2 and 3.
        #expect(Set(outcome.roundDepths) == [1, 2, 3])
        #expect(Double(outcome.rounds) / 200.0 < 0.15)
    }

    @Test("profitable cost speculates on nearly every step")
    func profitableCostSpeculatesContinuously() {
        let (outcome, controller) = drive(costRatio: 1.5, steps: 200)

        #expect(outcome.rounds > 150)
        #expect(controller.activeDepthForTesting(decodeRowBucket: 1) == 1)
        // Once position 1 is trusted the frontier admits depth 2, so a few
        // rounds draft deeper than the settled depth.
        #expect(outcome.drafted > outcome.rounds)
        #expect(outcome.roundDepths.contains(2))
    }

    /// The whole asymmetry is this line. Two runs that straddle it differ by
    /// more than an order of magnitude in speculative work while agreeing on
    /// every emitted token.
    @Test("a single cost comparison separates the two regimes")
    func oneCostComparisonSeparatesTheRegimes() {
        // expectedCommitted(1) == 2 before position 1 is trusted, and a
        // challenger must clear the incumbent by `hysteresisFraction`.
        let threshold = 2.0 / 1.05

        let justProfitable = drive(costRatio: threshold * 0.98, steps: 200)
        let justUnprofitable = drive(costRatio: threshold * 1.02, steps: 200)

        #expect(justUnprofitable.outcome.rounds == 23)
        #expect(justProfitable.outcome.rounds > 150)
        // The regimes are still separated by the one comparison — the sharpest
        // reading is that only one of them SETTLES on speculation.
        #expect(justUnprofitable.controller.activeDepthForTesting(decodeRowBucket: 1) == 0)
        #expect(justProfitable.controller.activeDepthForTesting(decodeRowBucket: 1) == 1)
        #expect(justProfitable.outcome.rounds >= justUnprofitable.outcome.rounds * 7)
    }

    /// A depth-zero baseline must exist before any comparison is possible, and
    /// it may only be measured on a NON-chained step. If that probe stopped
    /// being requested the controller would never leave warmup and MTP would be
    /// silently off — the failure mode the paged `kv_unsupported` no-op already
    /// produced once this wave.
    @Test("the depth-zero baseline is demanded before any depth is selected")
    func warmupBaselineIsRequiredBeforeSelection() {
        let controller = CBv2MTPDepthController(
            maxDepth: CBv2MTPConfig.testedMaxDraftTokens, fixedDepth: nil)

        let first = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(first.depth == 0)
        #expect(first.reason == "warmup_baseline")
        #expect(controller.requiresNonChainedDepthZeroProbe(first))

        // A CHAINED depth-zero step cannot satisfy it: its interval overlaps
        // neighbouring graph construction, so it is not a comparable baseline.
        controller.recordFinalizedStep(
            decision: first, actualDepth: 0, wallTimeNanos: 30_000_000,
            costEligible: true, chained: true,
            finalizedPlainWork: true, finalizedVerification: false)
        #expect(controller.requiresNonChainedDepthZeroProbe(first))

        controller.recordFinalizedStep(
            decision: first, actualDepth: 0, wallTimeNanos: 30_000_000,
            costEligible: true, chained: false,
            finalizedPlainWork: true, finalizedVerification: false)
        #expect(!controller.requiresNonChainedDepthZeroProbe(first))

        // With a baseline in hand the controller immediately costs depth 1.
        let second = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(second.depth == 1)
        #expect(second.reason == "explore_cost")
    }

    /// The field case, at the wall-clock costs actually measured on gemma-4.
    ///
    /// This is a CHARACTERIZATION test: it pins today's wrong answer together
    /// with the right one, so the gap cannot widen unnoticed and so anyone who
    /// changes seed-cost attribution, the innovation clamp, or the hysteresis
    /// margin is told exactly what those knobs were deciding. If a fix lands,
    /// `seedInflatedEstimate` stops declining and THIS TEST MUST BE UPDATED —
    /// that failure is the fix working, not a regression.
    ///
    /// The frontier-probe change did NOT fix it: the seed is folded into every
    /// positive-depth cost sample, including the deeper probes, so measuring
    /// depths 2 and 3 does not rescue the comparison. The controller still
    /// declines. Seed attribution is the remaining defect.
    @Test("measured gemma-4 costs: the seed-inflated estimate is what says no")
    func seedInflationFlipsTheVerdictAtMeasuredCosts() {
        // contiguous, B=1, 144 tokens, M4 Max. See the suite comment.
        let baselineNanos: UInt64 = 10_916_000
        let seedInflatedDepthOne = 25_111_000.0
        let steadyStateDepthOne = 14_214_000.0

        let seedInflatedEstimate = drive(
            costRatio: seedInflatedDepthOne / Double(baselineNanos),
            steps: 200, baselineNanos: baselineNanos)
        let steadyStateTruth = drive(
            costRatio: steadyStateDepthOne / Double(baselineNanos),
            steps: 200, baselineNanos: baselineNanos)

        // What the engine does today: decline, and spend only the bounded
        // learning budget finding that out.
        #expect(seedInflatedEstimate.outcome.rounds == 23)
        #expect(seedInflatedEstimate.outcome.accepted == 34)
        #expect(
            seedInflatedEstimate.controller.activeDepthForTesting(decodeRowBucket: 1) == 0)

        // What the same controller does on the same hardware once depth 1 is
        // costed at its steady-state price — the price forced mode measures
        // over 75 samples, and the one worth +14.7% decode throughput.
        #expect(steadyStateTruth.outcome.rounds > 150)
        #expect(steadyStateTruth.controller.activeDepthForTesting(decodeRowBucket: 1) == 1)

        // Both sides of the comparison are honest measurements of the same
        // machine. Only the seed attribution separates them.
        #expect(seedInflatedDepthOne / steadyStateDepthOne > 1.7)
    }
}
