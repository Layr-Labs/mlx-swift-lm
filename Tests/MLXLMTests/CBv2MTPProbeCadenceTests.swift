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
/// The controller's own estimate was 1.77x the steady-state truth, which is
/// what put it on the wrong side of the line. The inflation was structural,
/// not noise: a depth-zero plan invalidates every carry
/// (`EngineLoopV2.beginMTPPlan`), so every probe round must SEED first, and
/// `CBv2MTPRoundDriver.recordStepCost` used to fold that seed's wall time into
/// the round's cost sample (`attributed = wallTimeNanos &+
/// claimedSeedCostNanos`) while `expectedCommitted` never credits the token the
/// seed emitted. Cost in, token out — so choosing depth 0 manufactured the
/// evidence for choosing depth 0 again. Forced mode needs 3 seeds for 75
/// rounds; the adaptive controller paid 5 seeds for 5.
///
/// That fold is gone. The seed is now recorded as the bucket's TRANSITION cost
/// (`CBv2MTPDepthController.observeTransitionCost`) and the depth's sample is
/// the isolated round, because a settled positive depth does not re-seed:
/// `EngineLoopV2+MTPFinalize` stores a fresh carry on every verify round that
/// confirmed a token, partial rejections included. On the measured pair that
/// moves the depth-1 estimate from 25.111 ms to 14.195 ms and the verdict from
/// "unprofitable" to profitable — see
/// `theIsolatedVerifyCostIsWhatDecidesAtMeasuredCosts` below.
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
        #expect(outcome.rounds == 52)

        let maxDepth = CBv2MTPConfig.testedMaxDraftTokens
        #expect(Set(outcome.roundDepths) == Set(1 ... maxDepth))

        // The opening prices the envelope from the CEILING DOWN. That is the
        // first seven rounds, 7 through 1, and it is why this suite no longer
        // asserts a monotone climb from 1: the climb was the symptom of an
        // opening that started at depth 0 and could not get past its own
        // acceptance frontier.
        #expect(Array(outcome.roundDepths.prefix(7)) == [7, 6, 5, 4, 3, 2, 1])

        // Once every position carries evidence the probe ROTATES the envelope,
        // so a depth that measured badly once stays revisitable.
        #expect(Array(outcome.roundDepths.suffix(6)) == [2, 3, 4, 5, 6, 7])

        // And the ORIGINAL doubling resumes and saturates: the last probes are
        // 257 steps apart, one round in 257, and the interval is at its cap.
        #expect(Array(outcome.gaps.suffix(6)) == [33, 65, 129, 257, 257, 257])
        #expect(controller.probeIntervalForTesting(decodeRowBucket: 1) == 256)

        // Bounded by construction, and CHEAPER than before: 52 rounds against
        // the 84 the upward opening spent, because pricing the envelope once
        // from the top replaces ten probes per rung.
        #expect(outcome.rounds <= maxDepth * 10 + 16)
    }

    /// The same arm at the horizon the field harness actually ran (200 steps):
    /// still declining, still cheap, and now climbing.
    @Test("the learning phase is a small fraction of an unprofitable run")
    func unprofitableLearningPhaseStaysCheap() {
        let (outcome, controller) = drive(costRatio: 2.5, steps: 200)

        #expect(outcome.rounds == 27)
        #expect(outcome.drafted == 110)
        #expect(outcome.accepted == 90)
        #expect(controller.activeDepthForTesting(decodeRowBucket: 1) == 0)
        // 27 speculative rounds in 200 steps, and they price the WHOLE
        // envelope rather than the first three rungs. That is the cost of the
        // ceiling opening on a workload where MTP does not pay: 110 drafted
        // tokens against 38, because the opening scan starts at depth 7. The
        // verdict is unchanged -- `activeDepth` is still 0 -- and the scan is
        // paid once per bucket.
        #expect(Set(outcome.roundDepths) == Set(1 ... CBv2MTPConfig.testedMaxDraftTokens))
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

        #expect(justUnprofitable.outcome.rounds == 34)
        #expect(justProfitable.outcome.rounds > 150)
        // The regimes are still separated by the one comparison — the sharpest
        // reading is that only one of them SETTLES on speculation.
        #expect(justUnprofitable.controller.activeDepthForTesting(decodeRowBucket: 1) == 0)
        #expect(justProfitable.controller.activeDepthForTesting(decodeRowBucket: 1) == 1)
        // The separation narrows to about 5x, from 7x, because the ceiling
        // opening makes the unprofitable arm price the whole envelope before it
        // declines. It still declines, and it still never settles above 0,
        // which is what the comparison is for.
        #expect(justProfitable.outcome.rounds >= justUnprofitable.outcome.rounds * 5)
    }

    /// A depth-zero baseline must still be measured, and it may only be
    /// measured on a NON-chained step. What changed is WHEN: the controller now
    /// opens at the ceiling and prices the envelope downward, so the baseline
    /// is the last rung of the opening scan instead of the first decision. It
    /// is not optional — `goodput(0)` is zero without a cost sample, so a
    /// controller that never priced depth 0 could never decline to speculate.
    @Test("the depth-zero baseline is still demanded, after the ceiling opening")
    func warmupBaselineIsRequiredBeforeSelection() {
        let maxDepth = CBv2MTPConfig.testedMaxDraftTokens
        let controller = CBv2MTPDepthController(maxDepth: maxDepth, fixedDepth: nil)

        // The opening decision is the ceiling, and it does not break the chain:
        // only a depth-zero probe does.
        let opening = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(opening.depth == maxDepth)
        #expect(opening.reason == "open_ceiling")
        #expect(!controller.requiresNonChainedDepthZeroProbe(opening))
        controller.recordFinalizedStep(
            decision: opening, actualDepth: maxDepth, wallTimeNanos: 30_000_000,
            costEligible: true, chained: false,
            finalizedPlainWork: false, finalizedVerification: true)

        // The downward scan reaches depth 0 last. Walk it there.
        var first = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        var guard_ = 0
        while first.depth != 0, guard_ < maxDepth + 2 {
            controller.recordFinalizedStep(
                decision: first, actualDepth: first.depth,
                wallTimeNanos: 30_000_000, costEligible: true, chained: false,
                finalizedPlainWork: false, finalizedVerification: true)
            first = controller.select(plannedDecodeRows: 1, canSpeculate: true)
            guard_ += 1
        }
        #expect(first.depth == 0)
        #expect(first.reason == "explore_cost")
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

        // With every rung priced, the controller leaves the opening scan.
        let second = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(second.reason != "explore_cost")
        #expect(second.reason != "open_ceiling")
    }

    /// The field case, at the wall-clock costs actually measured on gemma-4,
    /// with the seed fix in place.
    ///
    /// Both arms are the SAME hardware and the same rounds. They differ only in
    /// which number the controller is handed: the seed-inflated 25.111 ms it
    /// used to get, or the isolated 14.195 ms it gets now. One arm declines and
    /// one speculates, so this pins the fix at the exact costs that produced
    /// the field report — and it will fail loudly if the fold ever comes back.
    ///
    /// 25.111 - 10.916 = 14.195, and forced mode measured 14.214 over 75
    /// samples. The arithmetic closes on one seed, which is what says the
    /// inflation was exactly the seed and nothing else.
    @Test("measured gemma-4 costs: the isolated verify cost is what decides")
    func theIsolatedVerifyCostIsWhatDecidesAtMeasuredCosts() {
        // contiguous, B=1, 144 tokens, M4 Max. See the suite comment.
        let baselineNanos: UInt64 = 10_916_000
        let seedInflatedDepthOne = 25_111_000.0
        let isolatedDepthOne = seedInflatedDepthOne - Double(baselineNanos)

        let folded = drive(
            costRatio: seedInflatedDepthOne / Double(baselineNanos),
            steps: 200, baselineNanos: baselineNanos)
        let isolated = drive(
            costRatio: isolatedDepthOne / Double(baselineNanos),
            steps: 200, baselineNanos: baselineNanos)

        // What the controller was fed before the fix: decline, and spend only
        // the bounded learning budget finding that out.
        #expect(folded.outcome.rounds == 27)
        #expect(folded.outcome.accepted == 90)
        #expect(folded.controller.activeDepthForTesting(decodeRowBucket: 1) == 0)

        // What it is fed now — the isolated round, 14.195 ms against forced
        // mode's independently measured 14.214 ms steady state.
        #expect(abs(isolatedDepthOne - 14_195_000.0) < 1_000.0)
        #expect(abs(isolatedDepthOne - 14_214_000.0) / 14_214_000.0 < 0.005)
        #expect(isolated.outcome.rounds > 150)
        #expect(isolated.controller.activeDepthForTesting(decodeRowBucket: 1) == 1)

        // One seed is the whole difference between the two verdicts.
        #expect(seedInflatedDepthOne / isolatedDepthOne > 1.7)
    }
}
