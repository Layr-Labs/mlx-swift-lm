import XCTest

@testable import MLXLMCommon

/// Real SchedulerV2 and causal CBv2Request metadata, without model/embedding
/// execution. CBv2SchedSim confirms planner assignments using the existing
/// scheduler-test convention; this does not claim a sealed native media run.
final class MiMoV26PrefillMediaStripeTests: XCTestCase {
    private enum FixtureError: Error { case embeddingsMustNotBeRead }

    private func scheduler(ceiling: Int? = 2048) -> SchedulerV2 {
        SchedulerV2(
            config: .init(
                maxConcurrentRequests: 4,
                maxBatchedTokensPerStep: 2048, prefillChunkSize: 512,
                soloPrefillStripeTokens: 8192, soloPrefillStripeMediaCeiling: ceiling,
                maxConcurrentPartialPrefills: 1, maxWaiting: 64))
    }

    private func request(
        _ id: UInt64, count: Int, media: Bool = false,
        maxTokens: Int = 1
    ) -> CBv2Request {
        var value = CBv2Request(
            id: .init(id), promptTokens: Array(repeating: 7, count: count),
            maxTokens: maxTokens)
        if media {
            value.multimodal = .init(spans: [.init(tokenOffset: 1, length: 1)], attention: .causal)
            {
                throw FixtureError.embeddingsMustNotBeRead
            }
        }
        return value
    }

    private func work(_ scheduler: SchedulerV2, for id: CBv2RequestID) throws
        -> CBv2FirstTokenScheduledWork
    {
        guard case .bounded(let work, _) = scheduler.firstTokenWorkProjection(for: id) else {
            XCTFail("expected an actual bounded scheduler projection")
            throw FixtureError.embeddingsMustNotBeRead
        }
        return work
    }

    private func confirm(_ scheduler: SchedulerV2, _ plan: CBv2StepPlan) -> [CBv2RequestID] {
        let sampled = CBv2SchedSim.confirm(scheduler, plan: plan)
        for id in sampled {
            if let row = scheduler.record(for: id), row.generatedTokenCount >= row.request.maxTokens
            {
                scheduler.finish(id: id, reason: .length)
            }
        }
        return sampled
    }

    func testPlainTextWidensButGenuineCausalMediaRetainsOriginalStripe() throws {
        let text = scheduler()
        let media = scheduler()
        let textRequest = request(501, count: 8192)
        let mediaRequest = request(502, count: 8192, media: true)
        try text.enqueue(textRequest)
        let mediaRow = try media.enqueue(mediaRequest)
        XCTAssertNotNil(mediaRow.request.multimodal)
        XCTAssertEqual(mediaRow.request.multimodal?.attention, .causal)
        XCTAssertTrue(
            mediaRow.multimodalBlocks.isEmpty,
            "causal media must reproduce the empty-bidirectional-block distinction")
        XCTAssertEqual(try work(text, for: textRequest.id).scheduledSteps, 1)
        XCTAssertEqual(try work(media, for: mediaRequest.id).scheduledSteps, 4)
        XCTAssertEqual(text.plan().assignments.map(\.numTokens), [8192])
        for _ in 0 ..< 4 {
            let plan = media.plan()
            XCTAssertEqual(plan.assignments.map(\.numTokens), [2048])
            _ = confirm(media, plan)
        }
        XCTAssertNil(media.record(for: mediaRequest.id))
    }

    func testNilCeilingPreservesLegacyCausalMediaStripe() throws {
        let legacy = scheduler(ceiling: nil)
        let media = request(503, count: 8192, media: true)
        try legacy.enqueue(media)
        XCTAssertEqual(try work(legacy, for: media.id).scheduledSteps, 1)
        XCTAssertEqual(legacy.plan().assignments.map(\.numTokens), [8192])
    }

    func testOriginalNoStripeAndStricterMediaCeilingsStayBounded() throws {
        for ceiling in [0, 512, 1024] {
            let scheduled = scheduler(ceiling: ceiling)
            let media = request(UInt64(600 + ceiling), count: 8192, media: true)
            try scheduled.enqueue(media)
            XCTAssertEqual(scheduled.plan().assignments.map(\.numTokens), [max(512, ceiling)])
        }
        var original = CBv2SchedulerConfig(prefillChunkSize: 512, soloPrefillStripeTokens: nil)
        XCTAssertNil(original.resolvedSoloPrefillStripeTokens(isMultimodal: true))
        original.soloPrefillStripeTokens = 2048
        original.soloPrefillStripeMediaCeiling = 1024
        XCTAssertEqual(original.resolvedSoloPrefillStripeTokens(isMultimodal: true), 1024)
        XCTAssertEqual(original.resolvedSoloPrefillStripeTokens(isMultimodal: false), 2048)
    }

    func testFinalTextStripeDoesNotTransferWideBudgetToMediaSuccessor() throws {
        let scheduled = scheduler()
        let text = request(504, count: 8300)
        let media = request(505, count: 9000, media: true)
        try scheduled.enqueue(text)
        try scheduled.enqueue(media)
        let first = scheduled.plan()
        XCTAssertEqual(first.assignments.map(\.numTokens), [8192])
        _ = confirm(scheduled, first)
        let next = scheduled.plan()
        XCTAssertEqual(next.assignments.first(where: { $0.id == text.id })?.numTokens, 108)
        XCTAssertEqual(next.assignments.first(where: { $0.id == media.id })?.numTokens, 512)
        _ = confirm(scheduled, next)
        XCTAssertEqual(scheduled.plan().assignments.map(\.numTokens), [2048])
    }

    func testDecodeNeighborDisarmsWideStripeForBothTextAndCausalMedia() throws {
        for isMedia in [false, true] {
            let scheduled = scheduler()
            let decoder = request(506, count: 1, maxTokens: 4)
            try scheduled.enqueue(decoder)
            _ = confirm(scheduled, scheduled.plan())
            let target = request(507, count: 8192, media: isMedia)
            try scheduled.enqueue(target)
            let plan = scheduled.plan()
            XCTAssertEqual(plan.assignments.first(where: { $0.id == decoder.id })?.numTokens, 1)
            XCTAssertEqual(plan.assignments.first(where: { $0.id == target.id })?.numTokens, 512)
        }
    }

    func testDeadlineProjectionAgreesWithActualMediaStepsAfterDecodeNeighbor() throws {
        let scheduled = scheduler()
        let decoder = request(508, count: 1, maxTokens: 2)
        try scheduled.enqueue(decoder)
        _ = confirm(scheduled, scheduled.plan())  // one remaining decode token
        let target = request(509, count: 8192, media: true)
        try scheduled.enqueue(target)
        let projection = try work(scheduled, for: target.id)
        XCTAssertEqual(projection.prefillTokens, 8192)
        XCTAssertEqual(projection.decodeTokens, 1)
        XCTAssertEqual(projection.mixedSteps, 1)
        var steps = 0
        var chunks: [Int] = []
        var targetSampled = false
        while steps < 32 && !targetSampled {
            let plan = scheduled.plan()
            XCTAssertFalse(plan.assignments.isEmpty)
            if let assignment = plan.assignments.first(where: { $0.id == target.id }) {
                chunks.append(assignment.numTokens)
            }
            steps += 1
            targetSampled = confirm(scheduled, plan).contains(target.id)
        }
        XCTAssertTrue(targetSampled)
        XCTAssertEqual(chunks, [512, 2048, 2048, 2048, 1536])
        XCTAssertEqual(projection.scheduledSteps, steps)
    }

    func testPerEngineMixedQuotaPreservesMediaCeilingAndProjection() throws {
        // Upstream's per-engine mixed quota must compose with the native
        // media ceiling; neither scheduling nor its pure forecast may drop it.
        let scheduled = SchedulerV2(
            config: .init(
                maxConcurrentRequests: 4,
                maxBatchedTokensPerStep: 2048, prefillChunkSize: 512,
                soloPrefillStripeTokens: 8192, soloPrefillStripeMediaCeiling: 2048,
                maxConcurrentPartialPrefills: 1, mixedStepPrefillTokenCap: 128))
        let decoder = request(510, count: 1, maxTokens: 2)
        try scheduled.enqueue(decoder)
        _ = confirm(scheduled, scheduled.plan())
        let target = request(511, count: 8192, media: true)
        try scheduled.enqueue(target)
        let projection = try work(scheduled, for: target.id)
        XCTAssertEqual(projection.prefillTokens, 8192)
        XCTAssertEqual(projection.decodeTokens, 1)
        XCTAssertEqual(projection.mixedSteps, 1)
        var steps = 0
        var chunks: [Int] = []
        var sampled = false
        while steps < 32 && !sampled {
            let plan = scheduled.plan()
            XCTAssertFalse(plan.assignments.isEmpty)
            if let assignment = plan.assignments.first(where: { $0.id == target.id }) {
                chunks.append(assignment.numTokens)
            }
            steps += 1
            sampled = confirm(scheduled, plan).contains(target.id)
        }
        XCTAssertTrue(sampled)
        XCTAssertEqual(chunks, [128, 2048, 2048, 2048, 1920])
        XCTAssertEqual(projection.scheduledSteps, steps)
    }
}
