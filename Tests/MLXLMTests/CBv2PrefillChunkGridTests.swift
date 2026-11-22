// Copyright © 2026 Eigen Labs.
//
// The model-scoped prefill grid (`SchedulerV2.prefillChunkAlignment`, from
// `CBv2ModelCapabilities.prefillChunkAlignment`): a gridded prompt is cut only
// at multiples of the alignment and at its end, each cell runs whole or waits
// for its first claim, and decode rows go first. Default off. Pure scheduler
// bookkeeping: no MLX arrays and no engine thread, except the default check.

import Foundation
import XCTest

@testable import MLXLMCommon

final class CBv2PrefillChunkGridTests: XCTestCase {

    // MARK: Fixtures

    private func scheduler(
        alignment: Int?, budget: Int = 2048, chunk: Int = 256, stripe: Int? = nil,
        partialPrefills: Int? = nil, mixedCap: Int? = nil
    ) -> SchedulerV2 {
        let scheduler = SchedulerV2(
            config: .init(
                maxConcurrentRequests: 6, maxBatchedTokensPerStep: budget,
                prefillChunkSize: chunk, soloPrefillStripeTokens: stripe,
                maxConcurrentPartialPrefills: partialPrefills, mixedStepPrefillTokenCap: mixedCap,
                maxWaiting: 16))
        scheduler.prefillChunkAlignment = alignment
        return scheduler
    }

    @discardableResult
    private func enqueue(
        _ scheduler: SchedulerV2, length: Int, maxTokens: Int = 8, token: Int = 7
    ) throws -> CBv2Request {
        let request = CBv2SchedFixtures.request(
            prompt: Array(repeating: token, count: length), maxTokens: maxTokens)
        try scheduler.enqueue(request)
        return request
    }

    /// A short prompt driven to decode-ready.
    private func decodingRow(_ scheduler: SchedulerV2, maxTokens: Int = 200) throws -> CBv2Request {
        let request = try enqueue(scheduler, length: 10, maxTokens: maxTokens, token: 3)
        CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())
        XCTAssertEqual(scheduler.record(for: request.id)?.isDecodeReady, true)
        return request
    }

    private func tokens(_ plan: CBv2StepPlan, _ id: CBv2RequestID) -> Int {
        plan.assignments.first { $0.id == id }?.numTokens ?? 0
    }

    /// Each plan's tokens for `id` until its prompt is computed.
    private func cuts(
        _ scheduler: SchedulerV2, _ id: CBv2RequestID, limit: Int = 40
    ) -> [Int] {
        var cuts = [Int]()
        for _ in 0 ..< limit {
            guard let rec = scheduler.record(for: id),
                rec.numComputedTokens < rec.request.promptTokens.count
            else { break }
            let plan = scheduler.plan()
            cuts.append(tokens(plan, id))
            CBv2SchedSim.confirm(scheduler, plan: plan)
        }
        return cuts
    }

    // MARK: Default off

    func testDefaultOffKeepsOrdinaryCuts() async throws {
        XCTAssertNil(CBv2ModelCapabilities().prefillChunkAlignment)
        XCTAssertNil(CBv2ModelCapabilities.attentionOnly.prefillChunkAlignment)
        XCTAssertNil(CBv2ModelCapabilities.initialRecurrentTarget.prefillChunkAlignment)
        XCTAssertNil(SchedulerV2(config: .init()).prefillChunkAlignment)
        let harness = CBv2SchedHarness()
        XCTAssertNil(harness.engine.loopForTesting.scheduler.prefillChunkAlignment)
        await harness.engine.shutdown()

        // Without a grid the step budget still cuts a prompt short beside a
        // decode row, and the decode row keeps its place.
        let plain = scheduler(alignment: nil, budget: 256)
        let decode = try decodingRow(plain)
        let prompt = try enqueue(plain, length: 600)
        let plan = plain.plan()
        XCTAssertEqual(plan.assignments.map(\.id), [decode.id, prompt.id])
        XCTAssertEqual(plan.assignments.map(\.numTokens), [1, 255])
    }

    // MARK: Cuts

    func testCellEndsAtTheGridThePromptEndOrTheRowEnd() {
        func cell(_ start: Int, prompt: Int = 600, end: Int = 600) -> Int? {
            SchedulerV2.prefillGridCell(
                alignment: 256, start: start, promptTokens: prompt, end: end, plan: nil,
                hasBlocks: false)
        }
        XCTAssertEqual(cell(0), 256)
        XCTAssertEqual(cell(256), 256)
        XCTAssertEqual(cell(300), 212)
        XCTAssertEqual(cell(512), 88)
        // A recompute after preemption cuts at the grid, the prompt end, then
        // the row end.
        XCTAssertEqual(cell(500, end: 620), 12)
        XCTAssertEqual(cell(512, end: 620), 88)
        XCTAssertEqual(cell(600, end: 620), 20)
        XCTAssertNil(cell(599))  // one token left: a decode row
        XCTAssertNil(
            SchedulerV2.prefillGridCell(
                alignment: nil, start: 0, promptTokens: 600, end: 600, plan: nil,
                hasBlocks: false))
        XCTAssertNil(
            SchedulerV2.prefillGridCell(
                alignment: 256, start: 0, promptTokens: 600, end: 600, plan: nil,
                hasBlocks: true))
    }

    /// The same prompt gets the same cuts alone, under the solo stripe, and
    /// beside a decode row; neither the plain chunk nor the stripe applies.
    func testCutsDependOnThePromptAlone() throws {
        let alone = scheduler(alignment: 256, chunk: 512, stripe: 1024)
        let soloPrompt = try enqueue(alone, length: 600)
        XCTAssertEqual(cuts(alone, soloPrompt.id), [256, 256, 88])

        let beside = scheduler(alignment: 256, chunk: 512, stripe: 1024)
        let decode = try decodingRow(beside)
        let prompt = try enqueue(beside, length: 600)
        XCTAssertEqual(cuts(beside, prompt.id), [256, 256, 88])
        XCTAssertEqual(beside.record(for: decode.id)?.generatedTokenCount, 4)

        // Ungridded, the stripe and the plain chunk cut it differently.
        let plain = scheduler(alignment: nil, chunk: 512, stripe: 1024)
        let plainPrompt = try enqueue(plain, length: 600)
        XCTAssertEqual(cuts(plain, plainPrompt.id), [600])
    }

    /// A step holds one cell even when the configured step limit is smaller.
    func testAStepHoldsOneCell() throws {
        let small = scheduler(alignment: 256, budget: 128, chunk: 128)
        let prompt = try enqueue(small, length: 600)
        XCTAssertEqual(cuts(small, prompt.id), [256, 256, 88])
    }

    // MARK: Whole or none, and fairness

    /// Under a budget of one cell, a cell never runs partly: it waits one
    /// step, then takes the whole next step by first claim. Decode rows go
    /// first otherwise, so each side waits at most one step at a time.
    func testCellsRunWholeAndNeitherSideStarves() throws {
        let tight = scheduler(alignment: 256, budget: 256)
        let decode = try decodingRow(tight)
        let prompt = try enqueue(tight, length: 1300)
        var promptCuts = [Int]()
        var decodeWaits = 0
        var promptWaits = 0
        var steps = 0
        while let rec = tight.record(for: prompt.id),
            rec.numComputedTokens < rec.request.promptTokens.count, steps < 40
        {
            let start = rec.numComputedTokens
            let plan = tight.plan()
            let n = tokens(plan, prompt.id)
            if n > 0 {
                XCTAssertEqual(start % 256, 0, "a cut off the grid at \(start)")
                promptCuts.append(n)
                promptWaits = 0
            } else {
                promptWaits += 1
                XCTAssertLessThanOrEqual(promptWaits, 1, "prompt starved at step \(steps)")
            }
            if tokens(plan, decode.id) > 0 {
                decodeWaits = 0
            } else {
                decodeWaits += 1
                XCTAssertLessThanOrEqual(decodeWaits, 1, "decode starved at step \(steps)")
            }
            CBv2SchedSim.confirm(tight, plan: plan)
            steps += 1
        }
        XCTAssertEqual(promptCuts, [256, 256, 256, 256, 256, 20])
        // The ungridded scheduler would have split every chunk around the decode row.
        XCTAssertGreaterThan(tight.record(for: decode.id)?.generatedTokenCount ?? 0, 4)
    }

    /// A long prompt admitted before a short one keeps taking whole cells,
    /// but the short one, once decoding, is planned first.
    func testDecodeRowsArePlannedBeforeEarlierPromptRows() throws {
        let grid = scheduler(alignment: 256, budget: 300)
        let long = try enqueue(grid, length: 2000)
        let short = try enqueue(grid, length: 20, maxTokens: 64, token: 5)
        var plan = grid.plan()
        XCTAssertEqual(plan.assignments.map(\.numTokens), [256, 20])
        CBv2SchedSim.confirm(grid, plan: plan)
        plan = grid.plan()
        XCTAssertEqual(plan.assignments.map(\.id), [short.id, long.id])
        XCTAssertEqual(plan.assignments.map(\.numTokens), [1, 256])
        // The planning order does not leak: the running order is unchanged.
        XCTAssertEqual(grid.running.map(\.id), [long.id, short.id])
    }

    /// A quota below one cell never splits it: the cell runs by first claim,
    /// exempt from the quota, every other step.
    func testQuotaBelowACellDefersToTheClaim() throws {
        let capped = scheduler(alignment: 256, mixedCap: 100)
        let decode = try decodingRow(capped)
        let prompt = try enqueue(capped, length: 600)
        var observed = [Int]()
        for _ in 0 ..< 8 {
            guard capped.record(for: prompt.id)?.isDecodeReady == false else { break }
            let plan = capped.plan()
            observed.append(tokens(plan, prompt.id))
            XCTAssertEqual(tokens(plan, decode.id) > 0 || tokens(plan, prompt.id) > 0, true)
            CBv2SchedSim.confirm(capped, plan: plan)
        }
        XCTAssertEqual(observed.filter { $0 > 0 }, [256, 256, 88])
        XCTAssertFalse(observed.contains { $0 > 0 && $0 < 88 })
    }

    // MARK: Deadline projection

    /// The first-token projection plans the same whole cells, and fails
    /// closed where `plan()` would defer a cell to a claim it does not model.
    func testProjectionMirrorsCellsAndFailsClosedOnAWait() throws {
        let roomy = scheduler(alignment: 256, partialPrefills: 1)
        try decodingRow(roomy)
        let target = try enqueue(roomy, length: 600)
        guard case .bounded(let work, _) = roomy.firstTokenWorkProjection(for: target.id) else {
            return XCTFail("cells that fit must project bounded work")
        }
        XCTAssertEqual(cuts(roomy, target.id), [256, 256, 88])
        XCTAssertEqual(work.prefillTokens, 600)
        XCTAssertEqual(work.scheduledSteps, 3)

        let tight = scheduler(alignment: 256, budget: 256, partialPrefills: 1)
        try decodingRow(tight)
        let waiting = try enqueue(tight, length: 600)
        XCTAssertEqual(
            tight.firstTokenWorkProjection(for: waiting.id),
            .unbounded(reason: .prefixGeometryBlocked))
    }
}
