// Copyright © 2026 Eigen Labs.

import XCTest

@testable import MLXLMCommon

final class CBv2UnboundedReasonTests: XCTestCase {
    private func scheduler(partialPrefills: Int = 1) -> SchedulerV2 {
        SchedulerV2(
            config: CBv2SchedulerConfig(
                maxConcurrentRequests: 2, maxBatchedTokensPerStep: 4,
                prefillChunkSize: 4, maxConcurrentPartialPrefills: partialPrefills,
                maxWaiting: 8))
    }

    private func request() -> CBv2Request {
        CBv2Request(id: CBv2RequestID(12_900), promptTokens: [1, 2, 3, 4], maxTokens: 1)
    }

    func testUnsupportedSchedulerAndMissingTargetHaveSeparateReasons() throws {
        let unsupported = scheduler(partialPrefills: 2)
        let target = request()
        try unsupported.enqueue(target)
        XCTAssertEqual(
            unsupported.firstTokenWorkProjection(for: target.id),
            .unbounded(reason: .unsupportedScheduler))
        XCTAssertEqual(
            scheduler().firstTokenWorkProjection(for: target.id),
            .unbounded(reason: .targetMissing))
    }

    func testInvalidInFlightEvidenceDoesNotBecomeMissingRateEvidence() throws {
        let scheduler = scheduler()
        let target = request()
        try scheduler.enqueue(target)
        XCTAssertEqual(
            scheduler.firstTokenWorkProjection(
                for: target.id, inFlightAssignments: [(target.id, 0)]),
            .unbounded(reason: .invalidInFlightAssignment))
        XCTAssertEqual(
            scheduler.firstTokenWorkProjection(
                for: target.id, inFlightAssignments: [(target.id, 1)]),
            .unbounded(reason: .inconsistentTokenCursor))
    }

    func testUnownedPendingSampleAndInvalidPrefixPreviewHaveSeparateReasons() throws {
        let scheduler = scheduler()
        let target = request()
        let record = try scheduler.enqueue(target)
        record.pendingSamples = 1
        XCTAssertEqual(
            scheduler.firstTokenWorkProjection(for: target.id),
            .unbounded(reason: .unownedPendingSample))
        record.pendingSamples = 0
        XCTAssertEqual(
            scheduler.firstTokenWorkProjection(
                for: target.id, unmaterializedPrefixAdoption: true),
            .unbounded(reason: .invalidPrefixReservation))
    }

    func testSchedulerReasonSurvivesEngineConversionBeforeRateChecks() async throws {
        let harness = CBv2SchedHarness()
        let target = request()
        let policy = CBv2FirstTokenDeadlineAdmission(
            deadline: .now.advanced(by: .seconds(30)),
            conservativePrefillTokensPerSecond: nil,
            conservativeDecodeTokensPerSecond: nil)
        let result = harness.engine.loopForTesting.onEngineQueueSync {
            harness.engine.loopForTesting.firstTokenProjectedWork(
                .unbounded(reason: .prefixGeometryBlocked), request: target,
                reusedPrefix: false, targetComputedTokens: 0, admission: policy)
        }
        XCTAssertEqual(result, .unbounded(reason: .prefixGeometryBlocked))
        await harness.engine.shutdown()
    }

    func testPhaseArithmeticFailurePrecedesLaterMissingRate() async throws {
        let harness = CBv2SchedHarness()
        let target = request()
        let policy = CBv2FirstTokenDeadlineAdmission(
            deadline: .now.advanced(by: .seconds(30)),
            conservativePrefillTokensPerSecond: .leastNonzeroMagnitude,
            conservativeDecodeTokensPerSecond: nil)
        let result = harness.engine.loopForTesting.onEngineQueueSync {
            harness.engine.loopForTesting.firstTokenProjectedWork(
                .bounded(
                    work: CBv2FirstTokenScheduledWork(
                        prefillTokens: 1, decodeTokens: 1, scheduledSteps: 1, mixedSteps: 1),
                    capacityOperations: []),
                request: target, reusedPrefix: false, targetComputedTokens: 0,
                admission: policy)
        }
        XCTAssertEqual(result, .unbounded(reason: .serviceDurationInvalid))
        await harness.engine.shutdown()
    }

    func testMissingPrefillRateRejectsBeforeForwardAndReleasesRequest() async throws {
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 2, maxBatchedTokensPerStep: 4,
                prefillChunkSize: 4, maxConcurrentPartialPrefills: 1, maxWaiting: 8))
        let policy = CBv2FirstTokenDeadlineAdmission(
            deadline: .now.advanced(by: .seconds(30)),
            conservativePrefillTokensPerSecond: nil,
            conservativeDecodeTokensPerSecond: 10)
        let result = try await harness.engine.submit(request(), firstTokenDeadline: policy)
        guard case .deadlineUnreachable(let work) = result else {
            XCTFail("missing required phase rate must remain a refusal")
            await harness.engine.shutdown()
            return
        }
        XCTAssertEqual(work, .unbounded(reason: .prefillRateUnavailable))
        XCTAssertTrue(harness.model.forwardShapes.isEmpty)
        XCTAssertEqual(harness.engine.capacity().activeRequests, 0)
        XCTAssertEqual(harness.engine.capacity().waitingRequests, 0)
        await harness.engine.shutdown()
    }
}
