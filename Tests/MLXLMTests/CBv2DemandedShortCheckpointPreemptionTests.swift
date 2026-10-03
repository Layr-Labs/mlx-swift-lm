import XCTest

@testable import MLXLMCommon

final class CBv2DemandedShortCheckpointPreemptionTests: XCTestCase {
    func testActualCapacityRequeueKeepsRestartedScheduleAndProjectionUnsplit() throws {
        var config = CBv2SchedulerConfig(
            maxConcurrentRequests: 1, maxBatchedTokensPerStep: 2_048,
            prefillChunkSize: 512, soloPrefillStripeTokens: 4_096,
            maxConcurrentPartialPrefills: 1, enablePrefixCache: true)
        config.demandedShortCheckpointMinimumTokens = 1_024
        let scheduler = SchedulerV2(config: config)
        let request = CBv2Request(
            id: .init(7_006), promptTokens: Array(repeating: 1, count: 2_800),
            maxTokens: 1, cacheSalt: "tenant-fixture", prefixCheckpointTargetTokens: 2_311)
        let record = try scheduler.enqueue(request)
        XCTAssertEqual(scheduler.plan().assignments.map(\.numTokens), [2_304])
        XCTAssertTrue(scheduler.requeueOnCapacity(request.id))
        XCTAssertEqual(record.preemptionCount, 1)
        XCTAssertEqual(record.numComputedTokens, 0)
        XCTAssertNil(record.prefixReusePlan)
        guard case .bounded(let work, _) = scheduler.firstTokenWorkProjection(for: request.id)
        else { return XCTFail("restarted text-work projection must remain bounded") }
        XCTAssertEqual(work.prefillTokens, 2_800)
        XCTAssertEqual(work.scheduledSteps, 1)
        XCTAssertEqual(scheduler.plan().assignments.map(\.numTokens), [2_800])
        XCTAssertEqual(record.plannedPrefillChunkSize, 4_096)
        XCTAssertEqual(record.numComputedTokens, 2_800)
    }
}
