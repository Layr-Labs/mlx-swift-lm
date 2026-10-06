import XCTest

@testable import MLXLMCommon

final class CBv2DemandedShortCheckpointCapacityTests: XCTestCase {
    func testRealAdmissionFallbackMarksOnlyTheExecutedDemandedBoundary() throws {
        let kinds = [
            CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
        ]
        let capacity = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 1_024 * 8,
            config: .init(watermarkFraction: 0, elementBytes: 4))
        var config = CBv2SchedulerConfig(
            maxConcurrentRequests: 1, maxBatchedTokensPerStep: 2_048,
            prefillChunkSize: 512, soloPrefillStripeTokens: 4_096,
            maxConcurrentPartialPrefills: 1, enablePrefixCache: true)
        config.demandedShortCheckpointMinimumTokens = 1_024
        let scheduler = SchedulerV2(config: config, capacity: capacity)
        let request = CBv2Request(
            id: .init(7_010), promptTokens: Array(repeating: 1, count: 2_800),
            maxTokens: 1, cacheSalt: "tenant-fixture", prefixCheckpointTargetTokens: 2_311)
        let record = try scheduler.enqueue(request)
        let admitted = scheduler.plan()
        XCTAssertEqual(admitted.assignments.map(\.numTokens), [512])
        XCTAssertTrue(admitted.demandedShortCheckpointRows.isEmpty)
        XCTAssertEqual(capacity.bytesReserved, 512 * 8)
        let running = scheduler.plan()
        XCTAssertEqual(running.assignments.map(\.numTokens), [512])
        XCTAssertTrue(running.demandedShortCheckpointRows.isEmpty)
        XCTAssertEqual(record.numComputedTokens, 1_024)
        XCTAssertEqual(record.preemptionCount, 0)
        XCTAssertEqual(capacity.bytesReserved, 1_024 * 8)

        // A real increase in granted capacity can now hold the requested
        // boundary. Its provenance survives; the subsequent tail is plain.
        capacity.updateBytesCapacity(4_096 * 8)
        let boundary = scheduler.plan()
        XCTAssertEqual(boundary.assignments.map(\.numTokens), [1_280])
        XCTAssertEqual(boundary.demandedShortCheckpointRows, [request.id])
        XCTAssertEqual(record.numComputedTokens, 2_304)
        let tail = scheduler.plan()
        XCTAssertEqual(tail.assignments.map(\.numTokens), [496])
        XCTAssertTrue(tail.demandedShortCheckpointRows.isEmpty)
        XCTAssertEqual(record.numComputedTokens, 2_800)
        scheduler.finish(id: request.id, reason: .cancelled)
        // Backend release belongs to the loop after scheduler retirement.
        capacity.releaseAll(id: request.id)
        XCTAssertEqual(capacity.bytesReserved, 0)
    }
}
