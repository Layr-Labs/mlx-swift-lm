import XCTest

@testable import MLXLMCommon

final class CBv2CalibratedFirstContentTests: XCTestCase {
    private func cell() -> CBv2FirstContentCalibrationCell {
        .init(
            promptTokensMin: 1, promptTokensMax: 32_768,
            contextTokensMin: 1, contextTokensMax: 32_768, reusedPrefix: false,
            contention: "isolated", prefillTokensPerSecond: 800, decodeTokensPerSecond: 40,
            maxPrefillWorkTokens: 32_768, maxDecodeWorkTokens: 1_024, maxActiveRequests: 1,
            competitorProfileIDs: [], maxOtherModelRequests: 0,
            maxOtherModelServiceFraction: 0, errorRatio: 1.12, errorAdditiveMilliseconds: 150)
    }

    private func policy(_ cells: [CBv2FirstContentCalibrationCell]? = nil)
        -> CBv2FirstContentCalibration
    {
        .init(
            cells: cells ?? [cell()], evidenceGuard: .init(), sameModelRequests: 1,
            otherModelRequests: 0, otherModelServiceFraction: 0, competitorProfileIDs: [])
    }

    func testMeasuredBoundChargesOnlyIncomingFirstContentDecodeAllowance() throws {
        let work = CBv2FirstTokenScheduledWork(
            prefillTokens: 8_828, decodeTokens: 0,
            scheduledSteps: 5, mixedSteps: 0)
        let expected = (8_828.0 / 800 + 33.0 / 40) * 1.12 + 0.15
        for output in [128, 4_096, 1_000_000] {
            XCTAssertEqual(
                try XCTUnwrap(
                    policy().serviceSeconds(
                        work: work, promptTokens: 8_828,
                        reusedPrefix: false, activeRequests: 1, maxOutputTokens: output)), expected,
                accuracy: 1e-12)
        }
    }

    func testActualWorkAndCacheStateSelectCellsAfterQueueing() {
        let work = CBv2FirstTokenScheduledWork(
            prefillTokens: 4_096, decodeTokens: 8,
            scheduledSteps: 8, mixedSteps: 4)
        var measured = cell()
        measured.contention = "same_model"
        measured.maxActiveRequests = 2
        measured.reusedPrefix = true
        let calibration = policy([measured])
        XCTAssertNotNil(
            calibration.serviceSeconds(
                work: work, promptTokens: 8_192,
                reusedPrefix: true, activeRequests: 2, maxOutputTokens: 128))
        XCTAssertNil(
            calibration.serviceSeconds(
                work: work, promptTokens: 8_192,
                reusedPrefix: false, activeRequests: 2, maxOutputTokens: 128))
        XCTAssertNil(
            calibration.serviceSeconds(
                work: work, promptTokens: 8_192,
                reusedPrefix: true, activeRequests: 3, maxOutputTokens: 128))
        XCTAssertNil(
            calibration.serviceSeconds(
                work: work, promptTokens: 8_192,
                reusedPrefix: true, activeRequests: 2, maxOutputTokens: 128,
                existingSchedulerContextTokensMax: 65_536))
    }

    func testExternalEpochAndExactCompetitorEnvelopeFailBackConservatively() {
        let work = CBv2FirstTokenScheduledWork(
            prefillTokens: 4_096, decodeTokens: 0,
            scheduledSteps: 2, mixedSteps: 0)
        var measured = cell()
        measured.contention = "other_model"
        measured.maxActiveRequests = 2
        measured.competitorProfileIDs = ["reviewed-competitor"]
        measured.maxOtherModelRequests = 1
        measured.maxOtherModelServiceFraction = 0.25
        var calibration = policy([measured])
        calibration.otherModelRequests = 1
        calibration.otherModelServiceFraction = 0.25
        calibration.competitorProfileIDs = ["reviewed-competitor"]
        XCTAssertNotNil(
            calibration.serviceSeconds(
                work: work, promptTokens: 4_096,
                reusedPrefix: false, activeRequests: 1, maxOutputTokens: 128))
        calibration.competitorProfileIDs = ["another-runtime"]
        XCTAssertNil(
            calibration.serviceSeconds(
                work: work, promptTokens: 4_096,
                reusedPrefix: false, activeRequests: 1, maxOutputTokens: 128))
        calibration.competitorProfileIDs = ["reviewed-competitor"]
        calibration.evidenceGuard.invalidate()
        XCTAssertNil(
            calibration.serviceSeconds(
                work: work, promptTokens: 4_096,
                reusedPrefix: false, activeRequests: 1, maxOutputTokens: 128))
    }

    func testOverlappingCellsUseLargerBoundRegardlessOfOrdering() throws {
        var slower = cell()
        slower.errorAdditiveMilliseconds = 800
        let work = CBv2FirstTokenScheduledWork(
            prefillTokens: 4_096, decodeTokens: 0,
            scheduledSteps: 2, mixedSteps: 0)
        let first = try XCTUnwrap(
            policy([cell(), slower]).serviceSeconds(
                work: work, promptTokens: 4_096,
                reusedPrefix: false, activeRequests: 1, maxOutputTokens: 128))
        let second = try XCTUnwrap(
            policy([slower, cell()]).serviceSeconds(
                work: work, promptTokens: 4_096,
                reusedPrefix: false, activeRequests: 1, maxOutputTokens: 128))
        XCTAssertEqual(first, second)
        XCTAssertEqual(first, (4_096.0 / 800 + 33.0 / 40) * 1.12 + 0.8, accuracy: 1e-12)
    }

    func testQueuedProjectionDoesNotRefreshMeasurementExpiry() {
        var calibration = policy()
        calibration.validUntil = ContinuousClock.now.advanced(by: .seconds(-1))
        let work = CBv2FirstTokenScheduledWork(
            prefillTokens: 4_096, decodeTokens: 0,
            scheduledSteps: 2, mixedSteps: 0)
        XCTAssertNil(
            calibration.serviceSeconds(
                work: work, promptTokens: 4_096,
                reusedPrefix: false, activeRequests: 1, maxOutputTokens: 128))
    }

    func testEarlyDecodeContextMustFitTheMeasuredCell() {
        var measured = cell()
        measured.promptTokensMax = 4_096
        measured.contextTokensMax = 4_096
        for (prompt, output, qualifies) in [
            (4_096, 33, false), (4_063, 33, true), (4_063, 1_000_000, true),
            (4_095, 1, true), (4_095, 2, false),
        ] {
            let work = CBv2FirstTokenScheduledWork(
                prefillTokens: prompt, decodeTokens: 0, scheduledSteps: 1, mixedSteps: 0)
            let result = policy([measured]).serviceSeconds(
                work: work, promptTokens: prompt, reusedPrefix: false,
                activeRequests: 1, maxOutputTokens: output)
            XCTAssertEqual(result != nil, qualifies, "prompt=\(prompt), output=\(output)")
        }
        measured.promptTokensMax = Int.max
        measured.contextTokensMax = Int.max
        measured.maxPrefillWorkTokens = Int.max
        XCTAssertNil(
            policy([measured]).serviceSeconds(
                work: .init(
                    prefillTokens: Int.max, decodeTokens: 0, scheduledSteps: 1, mixedSteps: 0),
                promptTokens: Int.max, reusedPrefix: false, activeRequests: 1, maxOutputTokens: 1))
    }

    func testInjectedEngineClockExpiresCalibrationBeforeAtomicAdmission() async throws {
        let clock = CBv2SchedFakeClock()
        var measured = cell()
        measured.prefillTokensPerSecond = 4
        measured.decodeTokensPerSecond = 100
        measured.errorRatio = 1.1
        measured.errorAdditiveMilliseconds = 100
        var calibration = policy([measured])
        calibration.validUntil = clock.clock.now().advanced(by: .seconds(60))
        // The actual monotonic clock is still before expiry. Only the engine's
        // supported injected clock has advanced beyond this evidence window.
        clock.advance(seconds: 120)
        let harness = CBv2SchedHarness(
            schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 4,
                prefillChunkSize: 4, maxConcurrentPartialPrefills: 1, maxWaiting: 8),
            loopConfig: .init(clock: clock.clock))
        let deadline = clock.clock.now().advanced(by: .seconds(3))
        let admission = CBv2FirstTokenDeadlineAdmission(
            deadline: deadline,
            conservativePrefillTokensPerSecond: 1, conservativeDecodeTokensPerSecond: 1,
            calibration: calibration)
        let result = try await harness.engine.submit(
            .init(id: .init(85_002), promptTokens: Array(0 ..< 8), maxTokens: 1),
            firstTokenDeadline: admission)
        if case .deadlineUnreachable(.bounded(_, let duration)) = result {
            XCTAssertEqual(duration, .seconds(8))
            XCTAssertTrue(harness.model.forwardShapes.isEmpty)
            XCTAssertEqual(harness.backend.liveStates, 0)
        } else {
            XCTFail("expired calibration must use the legacy eight-second bound")
        }
        XCTAssertEqual(admission.deadline, deadline)
        await harness.engine.shutdown()
    }

    func testSameModelRetirementAndPreSubmitWorkCannotDisappearFromProjection() throws {
        var measured = cell()
        measured.contention = "same_model"
        measured.maxActiveRequests = 2
        measured.maxDecodeWorkTokens = 8_192
        var calibration = policy([measured])
        calibration.sameModelRequests = 2
        calibration.sameModelPrefillTokens = 8_192
        calibration.sameModelDecodeTokens = 4_096
        // The older owner is preparing or retiring and absent from scheduler
        // rows. Its retained service lease still bounds all original work.
        let projected = CBv2FirstTokenScheduledWork(
            prefillTokens: 4_096, decodeTokens: 0,
            scheduledSteps: 2, mixedSteps: 0)
        let seconds = try XCTUnwrap(
            calibration.serviceSeconds(
                work: projected, promptTokens: 4_096,
                reusedPrefix: false, activeRequests: 1, maxOutputTokens: 128))
        XCTAssertEqual(seconds, (12_288.0 / 800 + 4_129.0 / 40) * 1.12 + 0.15, accuracy: 1e-12)
    }

    func testAtomicAdmissionKeepsDeadlineAndInvalidatedEvidenceUsesLegacyRates() async throws {
        for invalidate in [false, true] {
            let harness = CBv2SchedHarness(
                schedulerConfig: CBv2SchedulerConfig(
                    maxConcurrentRequests: 1, maxBatchedTokensPerStep: 4,
                    prefillChunkSize: 4, maxConcurrentPartialPrefills: 1, maxWaiting: 8))
            var measured = cell()
            measured.prefillTokensPerSecond = 4
            measured.decodeTokensPerSecond = 100
            measured.errorRatio = 1.1
            measured.errorAdditiveMilliseconds = 100
            let calibration = policy([measured])
            if invalidate { calibration.evidenceGuard.invalidate() }
            let deadline = ContinuousClock.now.advanced(by: .seconds(3))
            let admission = CBv2FirstTokenDeadlineAdmission(
                deadline: deadline,
                conservativePrefillTokensPerSecond: 1, conservativeDecodeTokensPerSecond: 1,
                calibration: calibration)
            let result = try await harness.engine.submit(
                CBv2Request(id: .init(85_001), promptTokens: Array(0 ..< 8), maxTokens: 1),
                firstTokenDeadline: admission)
            XCTAssertEqual(admission.deadline, deadline)
            if invalidate {
                guard case .deadlineUnreachable(let projected) = result,
                    case .bounded(_, let duration) = projected
                else {
                    XCTFail("stale calibration must retain the legacy rejection")
                    continue
                }
                XCTAssertEqual(duration, .seconds(8))
                XCTAssertTrue(harness.model.forwardShapes.isEmpty)
                XCTAssertEqual(harness.backend.liveStates, 0)
            } else {
                guard case .admitted(let stream, let projected, _, let retirement) = result,
                    case .bounded(_, let duration) = projected
                else {
                    XCTFail("qualified bound fits the unchanged budget")
                    continue
                }
                XCTAssertEqual(duration, .seconds((8.0 / 4 + 1.0 / 100) * 1.1 + 0.1))
                for await _ in stream {}
                await retirement.wait()
            }
            await harness.engine.shutdown()
        }
    }
}
