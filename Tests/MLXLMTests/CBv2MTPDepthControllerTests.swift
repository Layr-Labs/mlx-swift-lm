import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2MTPDepthController")
struct CBv2MTPDepthControllerTests {
    @Test func automaticVerificationCapsDepthByRectangularWork() throws {
        let model = MTPControllerTestModel()
        let driver = try #require(
            CBv2MTPRoundDriver.build(
                model: model, drafter: MTPControllerTestDrafter(target: model),
                config: CBv2MTPConfig(
                    enabled: true, maxDraftTokens: 7, maxSpeculativeBatch: 8,
                    fixedDraftTokens: 7, verificationMode: .automatic,
                    maxAutomaticRectangularTokens: 8)))

        #expect(driver.maximumAutomaticDepth(plannedDecodeRows: 1) == 7)
        #expect(driver.maximumAutomaticDepth(plannedDecodeRows: 2) == 3)
        #expect(driver.maximumAutomaticDepth(plannedDecodeRows: 4) == 1)
        #expect(driver.maximumAutomaticDepth(plannedDecodeRows: 8) == 0)
        #expect(driver.previewDecision(plannedDecodeRows: 4, canSpeculate: true).depth == 1)
        let blocked = driver.previewDecision(plannedDecodeRows: 8, canSpeculate: true)
        #expect(blocked.depth == 0)
        #expect(blocked.reason == "automatic_rectangular_limit")
    }

    @Test func fixedOverrideIncludesZeroAndClampsToTestedMaximum() {
        let zero = CBv2MTPDepthController(maxDepth: 7, fixedDepth: 0)
        #expect(zero.select(plannedDecodeRows: 1, canSpeculate: true).depth == 0)

        let two = CBv2MTPDepthController(maxDepth: 7, fixedDepth: 2)
        #expect(two.select(plannedDecodeRows: 4, canSpeculate: true).depth == 2)

        let clamped = CBv2MTPDepthController(maxDepth: 99, fixedDepth: 99)
        #expect(clamped.maxDepth == 7)
        #expect(clamped.fixedDepth == 7)
    }

    @Test func decodeRowsUsePowerOfTwoBuckets() {
        #expect(CBv2MTPDepthController.decodeRowBucket(0) == 0)
        #expect(CBv2MTPDepthController.decodeRowBucket(1) == 1)
        #expect(CBv2MTPDepthController.decodeRowBucket(2) == 2)
        #expect(CBv2MTPDepthController.decodeRowBucket(3) == 4)
        #expect(CBv2MTPDepthController.decodeRowBucket(4) == 4)
        #expect(CBv2MTPDepthController.decodeRowBucket(8) == 8)
    }

    @Test func poorAcceptanceAndSteepCostSelectDepthZero() {
        let controller = CBv2MTPDepthController(maxDepth: 1, fixedDepth: nil)
        controller.observeCost(decodeRowBucket: 1, depth: 0, wallTimeNanos: 30_000_000)
        controller.observeCost(decodeRowBucket: 1, depth: 1, wallTimeNanos: 60_000_000)
        for _ in 0 ..< 20 {
            controller.observeAcceptance(decodeRowBucket: 1, drafted: 1, accepted: 0)
        }

        let decision = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(decision.depth == 0)
        #expect(decision.reason == "unprofitable")
    }

    @Test func flatCostAndStrongConditionalAcceptanceExploreDeeper() {
        let controller = CBv2MTPDepthController(maxDepth: 7, fixedDepth: nil)
        for depth in 0 ... 4 {
            controller.observeCost(
                decodeRowBucket: 2, depth: depth,
                wallTimeNanos: UInt64(100_000_000 + depth * 1_000_000))
        }
        for _ in 0 ..< 20 {
            controller.observeAcceptance(decodeRowBucket: 2, drafted: 4, accepted: 4)
        }

        let decision = controller.select(plannedDecodeRows: 2, canSpeculate: true)
        #expect(decision.depth >= 4)
    }

    @Test func conditionalAcceptanceOnlyUpdatesReachedPositions() {
        let controller = CBv2MTPDepthController(maxDepth: 7, fixedDepth: nil)
        for _ in 0 ..< 10 {
            controller.observeAcceptance(decodeRowBucket: 1, drafted: 4, accepted: 1)
        }
        _ = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        let snapshot = controller.snapshot()
        #expect(snapshot.conditionalAcceptance.count == 2)
        #expect(snapshot.conditionalAcceptance[0] == 1)
        #expect(snapshot.conditionalAcceptance[1] == 0)
    }

    @Test func bucketLearningIsIsolated() {
        let controller = CBv2MTPDepthController(maxDepth: 7, fixedDepth: nil)
        controller.observeCost(decodeRowBucket: 1, depth: 0, wallTimeNanos: 10)
        controller.observeCost(decodeRowBucket: 1, depth: 1, wallTimeNanos: 12)
        // Bucket 1 has cost samples, so it is past its opening. Bucket 3 has
        // never been seen and opens at the ceiling. The point is that the two
        // buckets do not share learning, which is unchanged.
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).reason != "open_ceiling")
        #expect(controller.select(plannedDecodeRows: 3, canSpeculate: true).reason == "open_ceiling")
    }

    @Test func oneWallCostOutlierIsClamped() throws {
        let controller = CBv2MTPDepthController(maxDepth: 7, fixedDepth: nil)
        for _ in 0 ..< 20 {
            controller.observeCost(decodeRowBucket: 1, depth: 0, wallTimeNanos: 16_000_000)
        }
        controller.observeCost(decodeRowBucket: 1, depth: 0, wallTimeNanos: 160_000_000)
        _ = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        let input = try #require(
            controller.snapshot().costInputs.first {
                $0.decodeRowBucket == 1 && $0.depth == 0
            })
        // alpha 0.3 * clamp 25% permits at most a 7.5% one-sample move.
        #expect(input.ewmaWallTimeNanos <= 17_200_001)
        #expect(input.samples == 21)
    }

    @Test func previewDoesNotConsumeExplorationCadence() {
        let controller = CBv2MTPDepthController(maxDepth: 7, fixedDepth: nil)
        let first = controller.preview(plannedDecodeRows: 1, canSpeculate: true)
        let second = controller.preview(plannedDecodeRows: 1, canSpeculate: true)
        #expect(first == second)
        #expect(controller.snapshot().decodeRowBucket == 0)
    }

    @Test func chainedDepthZeroElapsedCannotForceFalseSpeculation() throws {
        let driver = try makeDriver(maxDepth: 1)
        // The controller opens at the ceiling, which at maxDepth 1 is depth 1,
        // and prices depth 0 on the way down. Only the ORDER moved; the
        // depth-zero baseline is still required and still non-chained.
        let opening = begin(driver)
        #expect(opening.depth == 1)
        #expect(opening.reason == "open_ceiling")
        #expect(!driver.requiresNonChainedDepthZeroProbe(opening))
        let row = CBv2RequestID(1)
        record(
            driver, decision: opening, actualDepth: 0,
            wallTimeNanos: 20, seedOnly: true,
            finalizedSeedIDs: [row])
        let verify = begin(driver)
        let seedCost = driver.claimPendingSeedCost(
            decodeRowBucket: 1, finalizedVerifyIDs: [row])
        #expect(seedCost == 20)
        // 250 ns of ISOLATED verify against a 100 ns baseline. The seed is no
        // longer folded in, so the verify cost has to be unprofitable on its
        // own: depth 1 needs `cost1 / cost0 < 2 / (1 + hysteresisFraction)`,
        // i.e. under 190 ns. (Before the seed fix this read 190 + 20 = 210 and
        // the 20 ns seed was doing the deciding.)
        record(
            driver, decision: verify, actualDepth: 1,
            wallTimeNanos: 250, finalizedVerification: true,
            claimedSeedCostNanos: seedCost)
        // The seed cost is recorded, just not against depth 1.
        #expect(driver.transitionCostNanosForTesting(decodeRowBucket: 1) == 20)

        // Now the downward scan reaches depth 0, and THIS is the decision that
        // must break the chain.
        let baseline = begin(driver)
        #expect(baseline.depth == 0)
        #expect(baseline.reason == "explore_cost")
        #expect(driver.requiresNonChainedDepthZeroProbe(baseline))
        record(
            driver, decision: baseline, actualDepth: 0,
            wallTimeNanos: 100, finalizedPlainWork: true)

        let targetOnly = begin(driver)
        #expect(targetOnly.depth == 0)
        #expect(targetOnly.reason == "unprofitable")
        record(
            driver, decision: targetOnly, actualDepth: 0,
            wallTimeNanos: 1_000_000_000, chained: true,
            finalizedPlainWork: true)

        let baselineCost = try #require(
            driver.metricsSnapshot().costInputs.first {
                $0.decodeRowBucket == 1 && $0.depth == 0
            })
        #expect(baselineCost.samples == 1)
        #expect(baselineCost.totalWallTimeNanos == 100)
        #expect(begin(driver).depth == 0)
    }

    @Test func exploratorySeedBindsToVerifyAndActiveTransitionWaits() throws {
        let driver = try makeDriver(maxDepth: 1)
        // Seed the ceiling opening; the depth-zero baseline follows it.
        let opening = begin(driver)
        #expect(opening.depth == 1)
        let row = CBv2RequestID(7)
        record(
            driver, decision: opening, actualDepth: 0,
            wallTimeNanos: 30, seedOnly: true,
            finalizedSeedIDs: [row])
        #expect(driver.pendingSeedCostCountForTesting == 1)
        #expect(begin(driver).reason == "open_ceiling")

        let verification = driver.controllerDecision
        let seedCost = driver.claimPendingSeedCost(
            decodeRowBucket: 1, finalizedVerifyIDs: [row])
        record(
            driver, decision: verification, actualDepth: 1,
            wallTimeNanos: 20, finalizedVerification: true,
            claimedSeedCostNanos: seedCost)
        let depthOne = try #require(
            driver.metricsSnapshot().costInputs.first {
                $0.decodeRowBucket == 1 && $0.depth == 1
            })
        // The seed still BINDS to this verify -- that is what the cohort match
        // in `CBv2MTPSeedCostLedger` is for, and it is why the claim has to
        // happen before verify-row completion retires the request ids. What
        // changed is where the 30 ns lands: on the bucket's transition cost,
        // not on depth 1's steady-state sample, which is the isolated 20 ns.
        #expect(depthOne.totalWallTimeNanos == 20)
        #expect(driver.transitionCostNanosForTesting(decodeRowBucket: 1) == 30)
        #expect(driver.activeDepthForTesting(decodeRowBucket: 1) == 0)

        // The scan still owes depth 0. Price it, then the goodput comparison
        // becomes possible and the transition is a real selection.
        let baseline = begin(driver)
        #expect(baseline.depth == 0)
        #expect(baseline.reason == "explore_cost")
        record(
            driver, decision: baseline, actualDepth: 0,
            wallTimeNanos: 100, finalizedPlainWork: true)

        let transition = begin(driver)
        #expect(transition.depth == 1)
        #expect(!transition.isExploration)
        record(
            driver, decision: transition, actualDepth: 0,
            wallTimeNanos: 1_000, finalizedPlainWork: true)
        #expect(driver.activeDepthForTesting(decodeRowBucket: 1) == 0)

        let retry = begin(driver)
        #expect(retry.depth == 1)
        record(
            driver, decision: retry, actualDepth: 1,
            wallTimeNanos: 50, finalizedVerification: true)
        #expect(driver.activeDepthForTesting(decodeRowBucket: 1) == 1)
    }

    @Test func cancellationAndCarryLossClearPendingSeedCost() throws {
        let driver = try makeDriver(maxDepth: 1)
        let baseline = begin(driver)
        record(
            driver, decision: baseline, actualDepth: 0,
            wallTimeNanos: 100, finalizedPlainWork: true)
        let exploration = begin(driver)
        let cancelled = CBv2RequestID(11)
        record(
            driver, decision: exploration, actualDepth: 0,
            wallTimeNanos: 30, seedOnly: true,
            finalizedSeedIDs: [cancelled])
        #expect(driver.pendingSeedCostCountForTesting == 1)
        driver.requestDidFinish(cancelled)
        #expect(driver.pendingSeedCostCountForTesting == 0)
        #expect(
            driver.claimPendingSeedCost(
                decodeRowBucket: 1, finalizedVerifyIDs: [CBv2RequestID(12)]) == 0)

        let retry = begin(driver)
        let invalidated = CBv2RequestID(13)
        record(
            driver, decision: retry, actualDepth: 0,
            wallTimeNanos: 40, seedOnly: true,
            finalizedSeedIDs: [invalidated])
        driver.invalidateCarry(invalidated)
        #expect(driver.pendingSeedCostCountForTesting == 0)
    }

    @Test func noVerificationDoesNotAdvanceProbeBackoff() throws {
        let driver = try makeDriver(maxDepth: 1)
        // Open at the ceiling, then price depth 0 on the way down. The probe
        // cadence this test is about only starts once both rungs are costed.
        let opening = begin(driver)
        record(
            driver, decision: opening, actualDepth: 1,
            wallTimeNanos: 250, finalizedVerification: true)
        let baseline = begin(driver)
        #expect(baseline.depth == 0)
        record(
            driver, decision: baseline, actualDepth: 0,
            wallTimeNanos: 100, finalizedPlainWork: true)

        // Eight, not seven: the ceiling opening spends one more decision before
        // the first non-exploration round, and `roundsSinceProbe` only starts
        // counting from there.
        for _ in 0 ..< 8 {
            let targetOnly = begin(driver)
            #expect(targetOnly.depth == 0)
            record(
                driver, decision: targetOnly, actualDepth: 0,
                wallTimeNanos: 1_000_000, chained: true,
                finalizedPlainWork: true)
        }

        let probe = begin(driver)
        #expect(probe.depth == 1)
        #expect(probe.reason == "explore_deeper")
        #expect(driver.probeIntervalForTesting(decodeRowBucket: 1) == 8)
        record(
            driver, decision: probe, actualDepth: 0,
            wallTimeNanos: 1_000, finalizedPlainWork: true)

        let unchanged = begin(driver)
        #expect(unchanged.depth == 1)
        #expect(unchanged.reason == "explore_deeper")
        #expect(driver.probeIntervalForTesting(decodeRowBucket: 1) == 8)
        record(
            driver, decision: unchanged, actualDepth: 1,
            wallTimeNanos: 250, finalizedVerification: true)
        // This probe is the FIRST to land on depth 1 since the opening scan
        // priced depth 0, so it is new information at that position and the
        // cadence stays at base. Before the ceiling opening the depth-0
        // baseline came first, so the very first probe was already a repeat and
        // this test saw the doubling one cycle earlier.
        //
        // The doubling itself is NOT asserted here any more. It needs a second
        // probe at the same position with no new acceptance sample, and driving
        // the driver that far turns this test into a cadence test. The end-to-
        // end shape -- gaps 33, 65, 129, 257 and the interval at its 256 cap --
        // is pinned by
        // `CBv2MTPProbeCadenceTests.unprofitableCostLearnsTheEnvelopeThenBacksOff`.
        // What THIS test is for is unchanged and still asserted above: a probe
        // that produced no verification does not advance the backoff.
        #expect(driver.probeIntervalForTesting(decodeRowBucket: 1) == 8)
    }

    @Test func seedCostLedgerIsDepthAgnosticAndRejectsChangedCohorts() {
        var ledger = CBv2MTPSeedCostLedger()
        let first = CBv2RequestID(21)
        let second = CBv2RequestID(22)
        ledger.record(decodeRowBucket: 2, requestIDs: [first, second], nanos: 77)
        #expect(ledger.take(decodeRowBucket: 1, requestIDs: [first, second]) == 0)
        #expect(ledger.take(decodeRowBucket: 2, requestIDs: [first]) == 0)
        #expect(ledger.count == 0)

        // No selected-depth input exists: the same row bucket/cohort claims
        // the seed cost for whichever depth its next verification uses.
        ledger.record(decodeRowBucket: 2, requestIDs: [first, second], nanos: 88)
        #expect(ledger.take(decodeRowBucket: 2, requestIDs: [first, second]) == 88)
    }

    @Test func marginalPolicyClampsEveryDepthInputToZeroThroughFour() {
        let probabilities = [1.0, 1.0, 1.0, 1.0]
        func select(_ offered: Int, remaining: Int = 10, verification: Int = 10) -> Int {
            CBv2MTPMarginalDepthPolicy.selectDepth(
                offeredDepth: offered,
                remainingTokens: remaining,
                verificationLimit: verification,
                acceptanceProbabilities: probabilities,
                previousTargetTopTwoMargin: nil,
                headStepCostRatio: 0)
        }

        for depth in 0 ... 4 {
            #expect(select(depth) == depth)
        }
        #expect(select(99) == 4)
        #expect(select(-1) == 0)
        #expect(select(4, remaining: 1) == 0)
        #expect(select(4, remaining: 2) == 1)
        #expect(select(4, remaining: 99, verification: 2) == 2)
        #expect(select(4, remaining: 99, verification: -1) == 0)
    }

    @Test func marginalPolicyStopsOnEqualityAtEveryNextPosition() {
        let equalityCases: [([Double], Int)] = [
            ([0.5, 1, 1, 1], 0),
            ([1, 2.0 / 3.0, 1, 1], 1),
            ([1, 1, 0.75, 1], 2),
            ([1, 1, 1, 0.8], 3),
        ]
        for (probabilities, expectedDepth) in equalityCases {
            let depth = CBv2MTPMarginalDepthPolicy.selectDepth(
                offeredDepth: 4,
                remainingTokens: 5,
                verificationLimit: 4,
                acceptanceProbabilities: probabilities,
                previousTargetTopTwoMargin: nil,
                headStepCostRatio: 0.5)
            #expect(depth == expectedDepth)
        }

        #expect(
            CBv2MTPMarginalDepthPolicy.selectDepth(
                offeredDepth: 4,
                remainingTokens: 5,
                verificationLimit: 4,
                acceptanceProbabilities: [1, 1, 1, 1],
                previousTargetTopTwoMargin: nil,
                headStepCostRatio: 0.5) == 4)
    }

    @Test func marginalPolicyWiresMarginDivisorsAndPositionsIntoSelection() {
        let negativeMargin = -2.0
        let firstExponential = exp(negativeMargin / 2.0)
        let firstCap = firstExponential / (1.0 + firstExponential)
        #expect(
            CBv2MTPMarginalDepthPolicy.selectDepth(
                offeredDepth: 1,
                remainingTokens: 2,
                verificationLimit: 1,
                acceptanceProbabilities: [1, 1, 1, 1],
                previousTargetTopTwoMargin: negativeMargin,
                headStepCostRatio: firstCap) == 0)

        let positiveMargin = 4.0
        let firstProbability = 1.0 / (1.0 + exp(-positiveMargin / 2.0))
        let secondProbability = 1.0 / (1.0 + exp(-positiveMargin / 3.0))
        let correctReach = firstProbability * secondProbability
        let wrongReach = firstProbability * firstProbability
        let correctBoundary = correctReach / (1.0 + firstProbability - correctReach)
        let wrongBoundary = wrongReach / (1.0 + firstProbability - wrongReach)
        let discriminatingCost = (correctBoundary + wrongBoundary) / 2.0
        #expect(
            CBv2MTPMarginalDepthPolicy.selectDepth(
                offeredDepth: 2,
                remainingTokens: 3,
                verificationLimit: 2,
                acceptanceProbabilities: [1, 1, 1, 1],
                previousTargetTopTwoMargin: positiveMargin,
                headStepCostRatio: discriminatingCost) == 1)

        #expect(
            CBv2MTPMarginalDepthPolicy.selectDepth(
                offeredDepth: 3,
                remainingTokens: 4,
                verificationLimit: 3,
                acceptanceProbabilities: [1, 1, 1, 1],
                previousTargetTopTwoMargin: positiveMargin,
                headStepCostRatio: 0.46) == 3)
    }

    @Test func marginalPolicyIgnoresMissingAndNonfiniteMargins() {
        let margins: [Double?] = [nil, .nan, .infinity, -.infinity]
        for margin in margins {
            #expect(
                CBv2MTPMarginalDepthPolicy.selectDepth(
                    offeredDepth: 4,
                    remainingTokens: 5,
                    verificationLimit: 4,
                    acceptanceProbabilities: [0.7, 0.6, 1, 1],
                    previousTargetTopTwoMargin: margin,
                    headStepCostRatio: 0.3) == 4)
        }
        #expect(
            CBv2MTPMarginalDepthPolicy.selectDepth(
                offeredDepth: 1,
                remainingTokens: 2,
                verificationLimit: 1,
                acceptanceProbabilities: [.nan],
                previousTargetTopTwoMargin: nil,
                headStepCostRatio: 0) == 0)
    }

    @Test func negativeMarginAndInvalidCostRatioDoNotOverdraft() {
        #expect(
            CBv2MTPMarginalDepthPolicy.selectDepth(
                offeredDepth: 4,
                remainingTokens: 5,
                verificationLimit: 4,
                acceptanceProbabilities: [1, 1, 1, 1],
                previousTargetTopTwoMargin: -6,
                headStepCostRatio: 0.18) == 0)

        let withFallback = CBv2MTPMarginalDepthPolicy.selectDepth(
            offeredDepth: 4,
            remainingTokens: 5,
            verificationLimit: 4,
            acceptanceProbabilities: [0.2, 0.2, 0.2, 0.2],
            previousTargetTopTwoMargin: nil,
            headStepCostRatio: .nan)
        let withBootstrap = CBv2MTPMarginalDepthPolicy.selectDepth(
            offeredDepth: 4,
            remainingTokens: 5,
            verificationLimit: 4,
            acceptanceProbabilities: [0.2, 0.2, 0.2, 0.2],
            previousTargetTopTwoMargin: nil,
            headStepCostRatio: CBv2MTPRawCostEstimator.bootstrapHeadStepCostRatio)
        #expect(withFallback == withBootstrap)
    }

    @Test func rawCostEstimatorUsesC0CkSlopeAndBootstrap() {
        var estimator = CBv2MTPRawCostEstimator()
        #expect(
            estimator.headStepCostRatio
                == CBv2MTPRawCostEstimator.bootstrapHeadStepCostRatio)
        let admittedBaseline = estimator.observe(depth: 0, rawWallTimeNanos: 100)
        #expect(admittedBaseline)
        #expect(
            estimator.headStepCostRatio
                == CBv2MTPRawCostEstimator.bootstrapHeadStepCostRatio)
        let ignoredWarmup = estimator.observe(depth: 1, rawWallTimeNanos: 10_000)
        #expect(!ignoredWarmup)
        #expect(estimator.needsSteadyStateProbe(depth: 1))
        #expect(
            estimator.headStepCostRatio
                == CBv2MTPRawCostEstimator.bootstrapHeadStepCostRatio)
        let admittedDepthOne = estimator.observe(depth: 1, rawWallTimeNanos: 118)
        #expect(admittedDepthOne)
        #expect(!estimator.needsSteadyStateProbe(depth: 1))
        #expect(abs(estimator.headStepCostRatio - 0.18) < 1e-12)

        var multipleDepths = CBv2MTPRawCostEstimator()
        multipleDepths.observe(depth: 0, rawWallTimeNanos: 100)
        multipleDepths.observe(depth: 1, rawWallTimeNanos: 999)
        multipleDepths.observe(depth: 1, rawWallTimeNanos: 120)
        multipleDepths.observe(depth: 2, rawWallTimeNanos: 999)
        multipleDepths.observe(depth: 2, rawWallTimeNanos: 150)
        #expect(abs(multipleDepths.headStepCostRatio - 0.225) < 1e-12)

        var noMeasuredIncrement = CBv2MTPRawCostEstimator()
        noMeasuredIncrement.observe(depth: 0, rawWallTimeNanos: 100)
        noMeasuredIncrement.observe(depth: 2, rawWallTimeNanos: 999)
        noMeasuredIncrement.observe(depth: 2, rawWallTimeNanos: 80)
        #expect(noMeasuredIncrement.headStepCostRatio == 0)

        var extreme = CBv2MTPRawCostEstimator()
        extreme.observe(depth: 0, rawWallTimeNanos: 1)
        extreme.observe(depth: 1, rawWallTimeNanos: Double.greatestFiniteMagnitude)
        extreme.observe(depth: 1, rawWallTimeNanos: Double.greatestFiniteMagnitude)
        let extremeRatio = extreme.headStepCostRatio
        #expect(extremeRatio > 1)
        #expect(
            CBv2MTPMarginalDepthPolicy.selectDepth(
                offeredDepth: 4,
                remainingTokens: 5,
                verificationLimit: 4,
                acceptanceProbabilities: [1, 1, 1, 1],
                previousTargetTopTwoMargin: nil,
                headStepCostRatio: extremeRatio) == 0)
    }

    @Test func steadyStateProbeStillRejectsGenuinelyExpensiveHeads() {
        var estimator = CBv2MTPRawCostEstimator()
        #expect(!estimator.needsSteadyStateProbe(depth: 1))
        estimator.observe(depth: 0, rawWallTimeNanos: 100)
        #expect(estimator.needsSteadyStateProbe(depth: 1))
        estimator.observe(depth: 1, rawWallTimeNanos: 1, chained: true)
        #expect(estimator.needsSteadyStateProbe(depth: 1))
        estimator.observe(depth: 1, rawWallTimeNanos: 10_000)
        #expect(estimator.needsSteadyStateProbe(depth: 1))
        estimator.observe(depth: 1, rawWallTimeNanos: 500)
        #expect(!estimator.needsSteadyStateProbe(depth: 1))
        #expect(abs(estimator.headStepCostRatio - 4.0) < 1e-12)
        #expect(
            CBv2MTPMarginalDepthPolicy.selectDepth(
                offeredDepth: 4,
                remainingTokens: 5,
                verificationLimit: 4,
                acceptanceProbabilities: [1, 1, 1, 1],
                previousTargetTopTwoMargin: nil,
                headStepCostRatio: estimator.headStepCostRatio) == 0)
    }

    @Test func costConfirmationProbeIsBoundedByOrdinaryCapacity() {
        #expect(
            CBv2MTPMarginalDepthPolicy.boundedProbeDepth(
                offeredDepth: 4, remainingTokens: 5, verificationLimit: 4) == 1)
        #expect(
            CBv2MTPMarginalDepthPolicy.boundedProbeDepth(
                offeredDepth: 0, remainingTokens: 5, verificationLimit: 4) == 0)
        #expect(
            CBv2MTPMarginalDepthPolicy.boundedProbeDepth(
                offeredDepth: 4, remainingTokens: 1, verificationLimit: 4) == 0)
        #expect(
            CBv2MTPMarginalDepthPolicy.boundedProbeDepth(
                offeredDepth: 4, remainingTokens: 5, verificationLimit: 0) == 0)
    }

    @Test func rawCostEstimatorRejectsInvalidAndChainedIntervals() {
        var estimator = CBv2MTPRawCostEstimator()
        let rejectedZero = estimator.observe(depth: 0, rawWallTimeNanos: 0)
        let rejectedNegative = estimator.observe(depth: 0, rawWallTimeNanos: -1)
        let rejectedNaN = estimator.observe(depth: 0, rawWallTimeNanos: .nan)
        let rejectedInfinity = estimator.observe(depth: 0, rawWallTimeNanos: .infinity)
        let rejectedDepth = estimator.observe(depth: 5, rawWallTimeNanos: 100)
        let rejectedChained = estimator.observe(
            depth: 0, rawWallTimeNanos: 100, chained: true)
        #expect(!rejectedZero)
        #expect(!rejectedNegative)
        #expect(!rejectedNaN)
        #expect(!rejectedInfinity)
        #expect(!rejectedDepth)
        #expect(!rejectedChained)
        #expect(estimator.sampleCount(depth: 0) == 0)
        #expect(
            estimator.headStepCostRatio
                == CBv2MTPRawCostEstimator.bootstrapHeadStepCostRatio)
    }

    @Test func rawCostSlopeExcludesSeedAttributionAndClampsOutliers() {
        var estimator = CBv2MTPRawCostEstimator()
        estimator.observe(depth: 0, rawWallTimeNanos: 100)
        // The raw depth-one interval is 118. A separately attributed seed cost
        // of 50 has no parameter here and must not turn h from 0.18 into 0.68.
        estimator.observe(depth: 1, rawWallTimeNanos: 999)
        estimator.observe(depth: 1, rawWallTimeNanos: 118)
        #expect(abs(estimator.headStepCostRatio - 0.18) < 1e-12)

        var clamped = CBv2MTPRawCostEstimator()
        clamped.observe(depth: 0, rawWallTimeNanos: 100)
        clamped.observe(depth: 1, rawWallTimeNanos: 999)
        clamped.observe(depth: 1, rawWallTimeNanos: 100)
        clamped.observe(depth: 1, rawWallTimeNanos: 1_000)
        #expect(abs(clamped.headStepCostRatio - 0.075) < 1e-12)
    }

    @Test func requestAcceptanceUpdatesOnlyObservedPositions() {
        var state = CBv2MTPRequestAcceptanceState()
        let initial = state.probabilities
        #expect(initial.count == 4)
        for position in 0 ..< initial.count {
            #expect(abs(initial[position] - 0.85 * pow(0.98, Double(position))) < 1e-12)
        }

        state.observe(draftedDepth: 4, acceptedDepth: 2, rejectionObserved: true)
        #expect(abs(state.probabilities[0] - (initial[0] + 0.15 * (1 - initial[0]))) < 1e-12)
        #expect(abs(state.probabilities[1] - (initial[1] + 0.15 * (1 - initial[1]))) < 1e-12)
        #expect(abs(state.probabilities[2] - (initial[2] + 0.15 * (0 - initial[2]))) < 1e-12)
        #expect(state.probabilities[3] == initial[3])
    }

    @Test func truncationDoesNotRecordARejection() {
        var state = CBv2MTPRequestAcceptanceState()
        let initial = state.probabilities
        state.observe(
            draftedDepth: 4,
            acceptedDepth: 2,
            rejectionObserved: false,
            endedByTruncation: true)

        #expect(state.probabilities[0] > initial[0])
        #expect(state.probabilities[1] > initial[1])
        #expect(state.probabilities[2] == initial[2])
        #expect(state.probabilities[3] == initial[3])
    }

    @Test func fullAcceptanceTransfersOnlyBoundedNontruncatedOptimism() {
        var recovering = CBv2MTPRequestAcceptanceState()
        for _ in 0 ..< 12 {
            recovering.observe(
                draftedDepth: 3, acceptedDepth: 2, rejectionObserved: true)
        }
        let depressedNextPosition = recovering.probabilities[2]
        for _ in 0 ..< 12 {
            recovering.observe(
                draftedDepth: 2, acceptedDepth: 2, rejectionObserved: false)
        }
        #expect(recovering.probabilities[2] > depressedNextPosition)
        #expect(recovering.probabilities[2] <= 0.95)

        var partial = CBv2MTPRequestAcceptanceState()
        let partialNextPosition = partial.probabilities[2]
        partial.observe(draftedDepth: 2, acceptedDepth: 1, rejectionObserved: true)
        #expect(partial.probabilities[2] == partialNextPosition)

        var truncated = CBv2MTPRequestAcceptanceState()
        let truncatedNextPosition = truncated.probabilities[2]
        truncated.observe(
            draftedDepth: 2,
            acceptedDepth: 2,
            rejectionObserved: false,
            endedByTruncation: true)
        #expect(truncated.probabilities[2] == truncatedNextPosition)
    }

    @Test func acceptanceStateHasPerRequestValueIsolation() {
        var firstRequest = CBv2MTPRequestAcceptanceState()
        let secondRequest = firstRequest
        let secondInitial = secondRequest.probabilities

        firstRequest.observe(draftedDepth: 4, acceptedDepth: 0, rejectionObserved: true)
        #expect(firstRequest.probabilities[0] != secondRequest.probabilities[0])
        #expect(secondRequest.probabilities == secondInitial)
    }

    private func makeDriver(maxDepth: Int) throws -> CBv2MTPRoundDriver {
        let model = MTPControllerTestModel()
        return try #require(
            CBv2MTPRoundDriver.build(
                model: model,
                drafter: MTPControllerTestDrafter(target: model),
                config: CBv2MTPConfig(
                    enabled: true, maxDraftTokens: maxDepth,
                    maxSpeculativeBatch: 1, fixedDraftTokens: nil)))
    }

    private func begin(_ driver: CBv2MTPRoundDriver) -> CBv2MTPDepthDecision {
        driver.beginPlan(plannedDecodeRows: 1, canSpeculate: true)
        return driver.controllerDecision
    }

    private func record(
        _ driver: CBv2MTPRoundDriver,
        decision: CBv2MTPDepthDecision,
        actualDepth: Int,
        wallTimeNanos: UInt64,
        costEligible: Bool = true,
        chained: Bool = false,
        seedOnly: Bool = false,
        finalizedPlainWork: Bool = false,
        finalizedSeedIDs: Set<CBv2RequestID> = [],
        finalizedVerification: Bool = false,
        claimedSeedCostNanos: UInt64 = 0
    ) {
        driver.recordStepCost(
            CBv2MTPStepMeasurement(
                decision: decision, actualDepth: actualDepth,
                costEligible: costEligible, chained: chained,
                seedOnly: seedOnly),
            wallTimeNanos: wallTimeNanos,
            finalizedPlainWork: finalizedPlainWork,
            finalizedSeedIDs: finalizedSeedIDs,
            finalizedVerification: finalizedVerification,
            claimedSeedCostNanos: claimedSeedCostNanos)
    }
}

private final class MTPControllerTestPrepared: CBv2MTPPreparedCapture {}

private final class MTPControllerTestDrafter: CBv2MTPDrafter {
    let mtpTargetIdentity: ObjectIdentifier?

    init(target: MTPControllerTestModel) {
        self.mtpTargetIdentity = ObjectIdentifier(target)
    }

    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        MTPControllerTestPrepared()
    }

    func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        (tokens, hidden)
    }
}

private final class MTPControllerTestModel: CBv2MTPSteppableModel {
    let mtpCaptureLayers: CBv2MTPCaptureLayers? = .init(full: 0, sliding: 0)
    var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(self) }

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        fatalError("controller tests do not execute model graphs")
    }

    func forwardWithHidden(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        fatalError("controller tests do not execute model graphs")
    }
}
