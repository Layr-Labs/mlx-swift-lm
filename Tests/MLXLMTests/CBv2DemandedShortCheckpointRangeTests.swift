import XCTest

@testable import MLXLMCommon

final class CBv2DemandedShortCheckpointRangeTests: XCTestCase {
    private func config() -> CBv2SchedulerConfig {
        var config = CBv2SchedulerConfig(
            maxConcurrentRequests: 1, maxBatchedTokensPerStep: 2_048,
            prefillChunkSize: 512, soloPrefillStripeTokens: 4_096,
            maxConcurrentPartialPrefills: 1, enablePrefixCache: true)
        config.demandedShortCheckpointMinimumTokens = 1_024
        return config
    }

    private func chunk(
        _ config: CBv2SchedulerConfig, prompt: Int = 2_800, hint: Int? = 2_311,
        computed: Int = 0, proposed: Int = 2_800, stripe: Int? = 4_096,
        reused: Bool = false, scoped: Bool = true, media: Bool = false
    ) -> Int {
        config.demandedShortCheckpointChunk(
            promptTokens: prompt, hintTokens: hint, computedTokens: computed,
            proposed: proposed, armedSoloStripeTokens: stripe, hasPrefixReuse: reused,
            hasCacheScope: scoped, isMultimodal: media)
    }

    func testOneDemandedAlignedInteriorBoundaryAndUnchangedTail() {
        XCTAssertEqual(chunk(config()), 2_304)
        XCTAssertEqual(chunk(config(), hint: 1_400), 1_280)
        XCTAssertEqual(chunk(config(), hint: Int.max), 2_560)
        XCTAssertEqual(chunk(config(), computed: 2_304, proposed: 496), 496)
        XCTAssertEqual(chunk(config(), proposed: 512), 512)
        XCTAssertEqual(chunk(config(), computed: 1_024, proposed: 1_776), 1_280)
    }

    func testNovelUnsupportedMediaAdopterAndLongRequestsKeepExistingGeometry() {
        var disabled = config()
        disabled.demandedShortCheckpointMinimumTokens = nil
        XCTAssertEqual(chunk(disabled), 2_800)
        disabled = config()
        disabled.enablePrefixCache = false
        XCTAssertEqual(chunk(disabled), 2_800)
        for hint in [nil, 0, 1_023] as [Int?] {
            XCTAssertEqual(chunk(config(), hint: hint), 2_800)
        }
        XCTAssertEqual(chunk(config(), stripe: nil), 2_800)
        XCTAssertEqual(chunk(config(), reused: true), 2_800)
        XCTAssertEqual(chunk(config(), scoped: false), 2_800)
        XCTAssertEqual(chunk(config(), media: true), 2_800)
        XCTAssertEqual(chunk(config(), prompt: 4_096, proposed: 4_096), 4_096)
        XCTAssertEqual(chunk(config(), prompt: 900, hint: 2_000, proposed: 900), 900)
    }

    func testSchedulerAndDeadlineProjectionChargeTheSameExtraStepWithoutRaisingTheCap() throws {
        let scheduler = SchedulerV2(config: config())
        let request = CBv2Request(
            id: .init(7_001), promptTokens: Array(repeating: 1, count: 2_800),
            maxTokens: 1, cacheSalt: "tenant-fixture", prefixCheckpointTargetTokens: 2_311)
        let record = try scheduler.enqueue(request)
        guard case .bounded(let work, _) = scheduler.firstTokenWorkProjection(for: request.id)
        else {
            return XCTFail("short demand projection must be bounded")
        }
        XCTAssertEqual(work.prefillTokens, 2_800)
        XCTAssertEqual(work.scheduledSteps, 2)
        let first = scheduler.plan()
        XCTAssertEqual(first.assignments.map(\.numTokens), [2_304])
        XCTAssertEqual(record.plannedPrefillChunkSize, 4_096)
        let second = scheduler.plan()
        XCTAssertEqual(second.assignments.map(\.numTokens), [496])
        XCTAssertEqual(record.plannedPrefillChunkSize, 4_096)
        XCTAssertEqual(record.numComputedTokens, 2_800)
    }

    func testNovelColdProjectionAndScheduleRemainOneStep() throws {
        let scheduler = SchedulerV2(config: config())
        let request = CBv2Request(
            id: .init(7_002), promptTokens: Array(repeating: 1, count: 2_800),
            maxTokens: 1, cacheSalt: "tenant-fixture", prefixCheckpointTargetTokens: 0)
        try scheduler.enqueue(request)
        guard case .bounded(let work, _) = scheduler.firstTokenWorkProjection(for: request.id)
        else {
            return XCTFail("novel projection must be bounded")
        }
        XCTAssertEqual(work.scheduledSteps, 1)
        XCTAssertEqual(scheduler.plan().assignments.map(\.numTokens), [2_800])
    }
}
