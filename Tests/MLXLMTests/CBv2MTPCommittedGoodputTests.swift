import Testing

@testable import MLXLMCommon

@Suite("CBv2MTPCommittedGoodput")
struct CBv2MTPCommittedGoodputTests {
    @Test func seedAndVerifyOwnDisjointTimeAndAllCommittedOutputs() {
        var clock = CBv2MTPCommittedGoodputClock()
        let rows = [CBv2RequestID(1)]
        #expect(clock.observe(
            measurement: measurement(depth: 0), completedAtNanos: 100,
            isolatedWallTimeNanos: 12, rowIDs: rows, committedTokens: 1) == nil)
        #expect(clock.observe(
            measurement: measurement(depth: 1, seed: true), completedAtNanos: 108,
            isolatedWallTimeNanos: 7, rowIDs: rows, committedTokens: 1) == nil)
        let seeded = clock.observe(
            measurement: measurement(depth: 1), completedAtNanos: 124,
            isolatedWallTimeNanos: 14, rowIDs: rows, committedTokens: 2)
        #expect(seeded == .init(wallTimeNanos: 24, committedTokens: 3))
        let continued = clock.observe(
            measurement: measurement(depth: 1), completedAtNanos: 140,
            isolatedWallTimeNanos: 14, rowIDs: rows, committedTokens: 2)
        #expect(continued == .init(wallTimeNanos: 16, committedTokens: 2))
        #expect((seeded?.wallTimeNanos ?? 0) + (continued?.wallTimeNanos ?? 0) == 140 - 100)
    }

    @Test func cancellationCohortAndIdleResetCannotChargeAnotherRequest() {
        var clock = CBv2MTPCommittedGoodputClock()
        let first = [CBv2RequestID(1)]
        let second = [CBv2RequestID(2)]
        _ = clock.observe(
            measurement: measurement(depth: 1, seed: true), completedAtNanos: 100,
            isolatedWallTimeNanos: 8, rowIDs: first, committedTokens: 1)
        #expect(clock.observe(
            measurement: measurement(depth: 1), completedAtNanos: 500,
            isolatedWallTimeNanos: 16, rowIDs: second, committedTokens: 1)
                == .init(wallTimeNanos: 16, committedTokens: 1))
        clock.reset()
        #expect(clock.observe(
            measurement: measurement(depth: 1), completedAtNanos: 50_000,
            isolatedWallTimeNanos: 17, rowIDs: second, committedTokens: 2)
                == .init(wallTimeNanos: 17, committedTokens: 2))
        #expect(clock.observe(
            measurement: measurement(depth: 1), completedAtNanos: 50_010,
            isolatedWallTimeNanos: 10, rowIDs: [], committedTokens: 0) == nil)
        #expect(clock.observe(
            measurement: measurement(depth: 1), completedAtNanos: 90_000,
            isolatedWallTimeNanos: 18, rowIDs: second, committedTokens: 1)
                == .init(wallTimeNanos: 18, committedTokens: 1))
    }

    @Test(arguments: [1, 2, 4])
    func actualSeedOutputsDiscoverGainWithoutBatchScalingBias(rows: Int) throws {
        let controller = calibrated(rows: rows, baseline: 8_700_000)
        let probe = controller.select(plannedDecodeRows: rows, canSpeculate: true)
        // A one-time shape compile is not the steady estimator's anchor.
        #expect(!controller.recordCommittedVerification(
            decision: probe, wallTimeNanos: 1_000_000_000,
            committedTokens: 3 * rows, rowCount: rows))
        #expect(controller.select(plannedDecodeRows: rows, canSpeculate: true).depth == 1)
        #expect(controller.recordCommittedVerification(
            decision: probe, wallTimeNanos: 24_000_000,
            committedTokens: 3 * rows, rowCount: rows))
        let input = try #require(controller.snapshot().costInputs.first { $0.depth == 1 })
        #expect(input.samples == 1)
        #expect(input.ewmaWallTimeNanos == 24_000_000)
        #expect(input.ewmaNanosPerCommittedToken == 8_000_000)
        #expect(controller.select(plannedDecodeRows: rows, canSpeculate: true).reason == "goodput")
    }

    @Test func timeAndOutputEWMAsWeightSeedTransitionsEqually() throws {
        let controller = calibrated(rows: 1, baseline: 8_000_000)
        let probe = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        _ = controller.recordCommittedVerification(
            decision: probe, wallTimeNanos: 999_000_000, committedTokens: 2, rowCount: 1)
        _ = controller.recordCommittedVerification(
            decision: probe, wallTimeNanos: 16_000_000, committedTokens: 2, rowCount: 1)
        // A new seed adds both cost and output: matching EWMA weights yield
        // 19.6 ms / 2.3 tokens. Clamping only time would manufacture a gain.
        _ = controller.recordCommittedVerification(
            decision: probe, wallTimeNanos: 28_000_000, committedTokens: 3, rowCount: 1)
        let input = try #require(controller.snapshot().costInputs.first { $0.depth == 1 })
        #expect(input.ewmaWallTimeNanos == 19_600_000)
        #expect(input.ewmaNanosPerCommittedToken == 8_521_739)
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).depth == 0)
    }

    @Test func observedRejectionAndTruncationNeverBorrowUncommittedTokens() {
        let controller = calibrated(rows: 1, baseline: 8_000_000)
        let probe = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        _ = controller.recordCommittedVerification(
            decision: probe, wallTimeNanos: 1_000_000_000, committedTokens: 2, rowCount: 1)
        // Even perfect historical acceptance cannot turn one committed token
        // after an EOS/budget/common-width clamp into two outputs.
        for _ in 0 ..< 20 {
            controller.observeAcceptance(decodeRowBucket: 1, drafted: 1, accepted: 1)
        }
        _ = controller.recordCommittedVerification(
            decision: probe, wallTimeNanos: 12_000_000, committedTokens: 1, rowCount: 1)
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).depth == 0)
        #expect(!controller.recordCommittedVerification(
            decision: probe, wallTimeNanos: 1, committedTokens: 0, rowCount: 1))
    }

    @Test func unprofitableProbesBackOffAndRecoverWithoutJITAnchoring() {
        let controller = calibrated(rows: 1, baseline: 8_000_000)
        let probe = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        _ = controller.recordCommittedVerification(
            decision: probe, wallTimeNanos: 1_000_000_000, committedTokens: 2, rowCount: 1)
        // C0 isolated 12 ms, chained 8 ms. Even with 80% actual draft
        // acceptance, 20 ms verification is slower than ordinary decode.
        for tokens in [2, 2, 2, 2, 1] {
            _ = controller.recordCommittedVerification(
                decision: probe, wallTimeNanos: 20_000_000,
                committedTokens: tokens, rowCount: 1)
        }
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).depth == 0)
        var probes = 0
        var recovered = false
        for _ in 0 ..< 512 {
            let decision = controller.select(plannedDecodeRows: 1, canSpeculate: true)
            if decision.depth == 0 {
                _ = controller.recordFinalizedStep(
                    decision: decision, actualDepth: 0, wallTimeNanos: 999_000_000,
                    costEligible: true, chained: true,
                    finalizedPlainWork: true, finalizedVerification: false)
            } else {
                if decision.isExploration { probes += 1 }
                _ = controller.recordCommittedVerification(
                    decision: decision, wallTimeNanos: 12_000_000,
                    committedTokens: 2, rowCount: 1)
                if !decision.isExploration { recovered = true; break }
            }
        }
        #expect(probes > 1)
        #expect(recovered)
        #expect(controller.probeIntervalForTesting(decodeRowBucket: 1) == 8)
    }

    @Test func changingRequestRefreshesBaselineAndDoesNotRepeatShapeWarmup() {
        let controller = calibrated(rows: 1, baseline: 8_700_000)
        let probe = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        _ = controller.recordCommittedVerification(
            decision: probe, wallTimeNanos: 999_000_000, committedTokens: 2, rowCount: 1)
        _ = controller.recordCommittedVerification(
            decision: probe, wallTimeNanos: 15_000_000, committedTokens: 2, rowCount: 1)
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).depth == 1)
        controller.beginWorkload(rowIDs: [CBv2RequestID(99)])
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).reason == "warmup_chained_baseline")
        for _ in 0 ..< 3 {
            controller.observeCommittedDecodeInterval(decodeRowBucket: 1, wallTimeNanos: 8_000_000)
        }
        let fresh = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(controller.recordCommittedVerification(
            decision: fresh, wallTimeNanos: 17_000_000, committedTokens: 2, rowCount: 1))
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).depth == 0)
    }

    @Test func fasterOrdinaryDecodeDoesNotRequireFivePercentExitMargin() {
        let controller = calibrated(rows: 1, baseline: 8_700_000)
        let probe = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        _ = controller.recordCommittedVerification(
            decision: probe, wallTimeNanos: 999_000_000, committedTokens: 2, rowCount: 1)
        _ = controller.recordCommittedVerification(
            decision: probe, wallTimeNanos: 16_000_000, committedTokens: 2, rowCount: 1)
        let active = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        _ = controller.recordCommittedVerification(
            decision: active, wallTimeNanos: 16_000_000, committedTokens: 2, rowCount: 1)
        #expect(controller.activeDepthForTesting(decodeRowBucket: 1) == 1)
        for _ in 0 ..< 30 {
            controller.observeCommittedDecodeInterval(decodeRowBucket: 1, wallTimeNanos: 7_800_000)
        }
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).depth == 0)
    }

    private func calibrated(rows: Int, baseline: UInt64) -> CBv2MTPDepthController {
        let controller = CBv2MTPDepthController(
            maxDepth: 1, fixedDepth: nil, useCommittedDecodeBaseline: true)
        controller.beginWorkload(rowIDs: (0 ..< rows).map { CBv2RequestID(UInt64($0 + 1)) })
        controller.observeCost(decodeRowBucket: rows, depth: 0, wallTimeNanos: 12_000_000)
        for _ in 0 ..< 3 {
            controller.observeCommittedDecodeInterval(decodeRowBucket: rows, wallTimeNanos: baseline)
        }
        return controller
    }

    private func measurement(depth: Int, seed: Bool = false) -> CBv2MTPStepMeasurement {
        CBv2MTPStepMeasurement(
            decision: .init(depth: depth, decodeRowBucket: 1, reason: "test", isExploration: false),
            actualDepth: seed ? 0 : depth, costEligible: true, chained: false, seedOnly: seed)
    }
}
