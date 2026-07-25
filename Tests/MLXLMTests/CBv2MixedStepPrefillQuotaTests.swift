// Copyright © 2026 Eigen Labs.
//
// WS-B: the OPT-IN mixed-step prefill quota
// (`SchedulerV2.mixedStepPrefillTokenCap`).
//
// A mixed step pairs one decoding row's single token with up to
// `maxBatchedTokensPerStep / prefillChunkSize` full prefill chunks under one
// asyncEval + readback boundary, so the decoder's inter-token latency tracks
// the whole forward. The quota bounds the prefill half of such a step.
//
// These tests are pure scheduler bookkeeping — no MLX arrays, no weights, no
// engine thread. They pin: (1) default-off byte-identical planning, (2) the
// cap binding on mixed steps only, (3) decode never capped, (4) no leaked
// optimistic advance / capacity reservation for a skipped row, (5) liveness.

import Foundation
import XCTest

@testable import MLXLMCommon

final class CBv2MixedStepPrefillQuotaTests: XCTestCase {

    // MARK: Fixtures

    private func makeScheduler(
        maxConcurrent: Int = 4, budget: Int = 2048, chunk: Int = 512, maxWaiting: Int = 64,
        capacity: CBv2StepCapacity? = nil
    ) -> SchedulerV2 {
        SchedulerV2(
            config: CBv2SchedulerConfig(
                maxConcurrentRequests: maxConcurrent, maxBatchedTokensPerStep: budget,
                prefillChunkSize: chunk, maxWaiting: maxWaiting),
            capacity: capacity)
    }

    /// Enqueue a short prompt and drive it to decode-ready (prompt fully
    /// computed, one sampled token confirmed ⇒ `remainingTokens == 1`).
    @discardableResult
    private func makeDecodingRow(
        _ scheduler: SchedulerV2, promptLength: Int = 100, maxTokens: Int = 64
    ) throws -> CBv2Request {
        let request = CBv2SchedFixtures.request(
            prompt: Array(repeating: 7, count: promptLength), maxTokens: maxTokens)
        try scheduler.enqueue(request)
        let plan = scheduler.plan()
        XCTAssertEqual(plan.assignments.map(\.numTokens), [promptLength])
        XCTAssertEqual(CBv2SchedSim.confirm(scheduler, plan: plan), [request.id])
        XCTAssertEqual(scheduler.record(for: request.id)?.isDecodeReady, true)
        return request
    }

    private func enqueuePrefills(
        _ scheduler: SchedulerV2, count: Int, promptLength: Int = 600
    ) throws -> [CBv2Request] {
        try (0 ..< count).map { i in
            let request = CBv2SchedFixtures.request(
                prompt: Array(repeating: i + 1, count: promptLength), maxTokens: 8)
            try scheduler.enqueue(request)
            return request
        }
    }

    /// Tokens the plan gave to PREFILL rows. `decodeIDs` must be sampled
    /// BEFORE `plan()` — a row that finishes its prompt this step becomes a
    /// decode row afterwards, and its chunk was still prefill work.
    private func prefillTokens(
        _ plan: CBv2StepPlan, decodeIDs: Set<CBv2RequestID>
    ) -> Int {
        plan.assignments.filter { !decodeIDs.contains($0.id) }.reduce(0) { $0 + $1.numTokens }
    }

    /// The rows `plan()` will treat as decode work this step.
    private func decodeIDs(_ scheduler: SchedulerV2) -> Set<CBv2RequestID> {
        Set(
            scheduler.running
                .filter { !$0.isPaused && !$0.cancelRequested && $0.isDecodeReady }
                .map(\.id))
    }

    // MARK: 1. Default (nil) — byte-identical to today

    /// Pinned baseline: 1 decode + 3 fresh 600-token prompts under a 2048
    /// budget / 512 chunk produce ONE step of 1 + 512 + 512 + 512 = 1537
    /// tokens. This is exactly the pathology the cap exists to bound, and it
    /// must remain the default behavior.
    func testCapNilPlansOneDecodePlusThreeFullPrefillChunks() throws {
        let scheduler = makeScheduler()
        XCTAssertNil(scheduler.mixedStepPrefillTokenCap, "cap must default to disabled")
        let decode = try makeDecodingRow(scheduler)
        let prefills = try enqueuePrefills(scheduler, count: 3)

        let plan = scheduler.plan()
        XCTAssertEqual(
            plan.assignments.map(\.id), [decode.id] + prefills.map(\.id))
        XCTAssertEqual(plan.assignments.map(\.numTokens), [1, 512, 512, 512])
        XCTAssertEqual(plan.assignments.reduce(0) { $0 + $1.numTokens }, 1537)
        XCTAssertTrue(plan.preemptions.isEmpty)
        XCTAssertEqual(scheduler.runningCount, 4)
        XCTAssertEqual(scheduler.waitingCount, 0)
    }

    /// Explicit nil assignment is the same as never touching the property.
    func testExplicitNilCapMatchesUntouchedScheduler() throws {
        let baseline = makeScheduler()
        let explicit = makeScheduler()
        explicit.mixedStepPrefillTokenCap = nil

        for scheduler in [baseline, explicit] {
            let decode = try makeDecodingRow(scheduler)
            _ = try enqueuePrefills(scheduler, count: 3)
            let plan = scheduler.plan()
            XCTAssertEqual(plan.assignments.first?.id, decode.id)
            XCTAssertEqual(plan.assignments.map(\.numTokens), [1, 512, 512, 512])
        }
    }

    // MARK: 2. Mixed step — prefill capped, decode intact

    func testMixedStepPrefillTokensBoundedByCap() throws {
        let scheduler = makeScheduler()
        scheduler.mixedStepPrefillTokenCap = 512
        let decode = try makeDecodingRow(scheduler)
        let prefills = try enqueuePrefills(scheduler, count: 3)

        let plan = scheduler.plan()
        // Decode is untouched: present, first, exactly 1 token.
        XCTAssertEqual(plan.assignments.first?.id, decode.id)
        XCTAssertEqual(plan.assignments.first?.numTokens, 1)
        // Prefill is bounded by the cap even though 1535 budget tokens remain.
        XCTAssertLessThanOrEqual(prefillTokens(plan, decodeIDs: [decode.id]), 512)
        XCTAssertEqual(plan.assignments.map(\.id), [decode.id, prefills[0].id])
        XCTAssertEqual(plan.assignments.map(\.numTokens), [1, 512])
        // The rows the cap skipped keep their queue position, in FCFS order.
        XCTAssertEqual(scheduler.waiting.map(\.id), [prefills[1].id, prefills[2].id])
    }

    /// The cap is a token budget, not an all-or-nothing per-row gate: a cap
    /// that is not a multiple of the chunk size is filled exactly.
    func testCapIsFilledExactlyAcrossRows() throws {
        let scheduler = makeScheduler()
        scheduler.mixedStepPrefillTokenCap = 700
        let decode = try makeDecodingRow(scheduler)
        let prefills = try enqueuePrefills(scheduler, count: 3)

        let plan = scheduler.plan()
        XCTAssertEqual(plan.assignments.map(\.id), [decode.id, prefills[0].id, prefills[1].id])
        XCTAssertEqual(plan.assignments.map(\.numTokens), [1, 512, 188])
        XCTAssertEqual(prefillTokens(plan, decodeIDs: [decode.id]), 700)
    }

    /// The cap binds on RUNNING mid-prefill rows too, not just admissions.
    func testCapBindsOnRunningMidPrefillRow() throws {
        let scheduler = makeScheduler()
        scheduler.mixedStepPrefillTokenCap = 100
        let decode = try makeDecodingRow(scheduler, promptLength: 50)
        let long = CBv2SchedFixtures.request(
            prompt: Array(repeating: 3, count: 2000), maxTokens: 8)
        try scheduler.enqueue(long)

        // Step 1 admits `long` from waiting under the cap.
        var plan = scheduler.plan()
        XCTAssertEqual(plan.assignments.map(\.numTokens), [1, 100])
        CBv2SchedSim.confirm(scheduler, plan: plan)

        // Step 2: `long` is RUNNING and mid-prefill — still capped.
        plan = scheduler.plan()
        XCTAssertEqual(plan.assignments.map(\.id), [decode.id, long.id])
        XCTAssertEqual(plan.assignments.map(\.numTokens), [1, 100])
        XCTAssertEqual(scheduler.record(for: long.id)?.numComputedTokens, 200)
    }

    /// Decode is never starved by the cap — not even at cap 0, and not even
    /// when MTP widens the decode row to 1 + k.
    func testDecodeIsNeverCapped() throws {
        let scheduler = makeScheduler()
        scheduler.mixedStepPrefillTokenCap = 0
        scheduler.speculationPlanner = { _ in 3 }
        let decode = try makeDecodingRow(scheduler)
        let prefills = try enqueuePrefills(scheduler, count: 2)

        let plan = scheduler.plan()
        XCTAssertEqual(plan.assignments.map(\.id), [decode.id])
        XCTAssertEqual(plan.assignments.map(\.numTokens), [4])  // 1 + k, untouched
        XCTAssertTrue(plan.speculationFallbacks.isEmpty)
        XCTAssertEqual(scheduler.waiting.map(\.id), prefills.map(\.id))
    }

    // MARK: 3. Pure-prefill steps are uncapped

    func testPurePrefillStepIgnoresCap() throws {
        let scheduler = makeScheduler()
        scheduler.mixedStepPrefillTokenCap = 128
        let prefills = try enqueuePrefills(scheduler, count: 3)

        let plan = scheduler.plan()
        XCTAssertEqual(plan.assignments.map(\.id), prefills.map(\.id))
        XCTAssertEqual(plan.assignments.map(\.numTokens), [512, 512, 512])
        XCTAssertEqual(plan.assignments.reduce(0) { $0 + $1.numTokens }, 1536)
    }

    /// A paused decode row is not schedulable, so the step is pure prefill
    /// and the quota never arms (same predicate the running pass skips on).
    func testPausedDecodeRowDoesNotArmTheCap() throws {
        let scheduler = makeScheduler()
        scheduler.mixedStepPrefillTokenCap = 128
        let decode = try makeDecodingRow(scheduler)
        scheduler.pause(decode.id)
        let prefills = try enqueuePrefills(scheduler, count: 2)

        let plan = scheduler.plan()
        XCTAssertEqual(plan.assignments.map(\.id), prefills.map(\.id))
        XCTAssertEqual(plan.assignments.map(\.numTokens), [512, 512])
    }

    // MARK: 4. A skipped row leaves nothing behind

    func testSkippedRowHasNoAdvanceAndNoReservation() throws {
        let capacity = CBv2SchedMockCapacity(tokenLimit: 100_000)
        let scheduler = makeScheduler(capacity: capacity)
        scheduler.mixedStepPrefillTokenCap = 512
        let decode = try makeDecodingRow(scheduler)
        let prefills = try enqueuePrefills(scheduler, count: 3)
        let reservedAfterWarmup = capacity.totalReserved

        let plan = scheduler.plan()
        XCTAssertEqual(plan.assignments.map(\.numTokens), [1, 512])

        for skipped in [prefills[1], prefills[2]] {
            let rec = try XCTUnwrap(scheduler.record(for: skipped.id))
            XCTAssertEqual(rec.numComputedTokens, 0, "optimistic advance leaked")
            XCTAssertEqual(rec.status, .waiting, "skipped row must not be admitted")
            XCTAssertNil(capacity.reserved[skipped.id], "capacity reservation leaked")
            XCTAssertFalse(
                capacity.releaseAllCalls.contains(skipped.id),
                "skipped row must not be preempted/cancelled")
        }
        XCTAssertTrue(plan.preemptions.isEmpty)
        // Only the decode token (1) and the one admitted chunk (512) were
        // charged on top of the warm-up prefill.
        XCTAssertEqual(capacity.totalReserved, reservedAfterWarmup + 1 + 512)
        XCTAssertEqual(capacity.reserved[prefills[0].id], 512)
        XCTAssertEqual(capacity.reserved[decode.id], 100 + 1)
    }

    /// The plan's own rollback contract still holds under the cap: unwinding
    /// an unexecuted capped plan returns the ledger to its pre-plan state.
    func testRollbackOfCappedPlanRestoresLedger() throws {
        let capacity = CBv2SchedMockCapacity(tokenLimit: 100_000)
        let scheduler = makeScheduler(capacity: capacity)
        scheduler.mixedStepPrefillTokenCap = 512
        let decode = try makeDecodingRow(scheduler)
        let prefills = try enqueuePrefills(scheduler, count: 3)
        let before = capacity.totalReserved

        let plan = scheduler.plan()
        scheduler.rollback(plan)

        XCTAssertEqual(capacity.totalReserved, before)
        XCTAssertEqual(scheduler.record(for: decode.id)?.numComputedTokens, 100)
        XCTAssertEqual(scheduler.record(for: prefills[0].id)?.numComputedTokens, 0)
        XCTAssertNil(capacity.reserved[prefills[1].id])
    }

    // MARK: 5. Liveness — a capped row always gets a later step

    /// FIFO drain: the quota only defers prefill, and it is consumed in queue
    /// order, so every skipped row reaches the head and is scheduled.
    func testCappedRowIsScheduledOnALaterStep() throws {
        let scheduler = makeScheduler()
        scheduler.mixedStepPrefillTokenCap = 512
        let decode = try makeDecodingRow(scheduler)
        let prefills = try enqueuePrefills(scheduler, count: 3)

        var scheduledSteps: [CBv2RequestID: Int] = [:]
        for step in 0 ..< 8 {
            // Sampled before planning: rows that complete their prompt this
            // step join the decode set only from the NEXT step on.
            let decodes = decodeIDs(scheduler)
            let plan = scheduler.plan()
            XCTAssertEqual(
                plan.assignments.first(where: { $0.id == decode.id })?.numTokens, 1,
                "decode must be scheduled every step, at exactly 1 token")
            for id in decodes {
                XCTAssertEqual(
                    plan.assignments.first(where: { $0.id == id })?.numTokens, 1,
                    "every decode row keeps its single token under the cap")
            }
            XCTAssertLessThanOrEqual(prefillTokens(plan, decodeIDs: decodes), 512)
            for (id, _) in plan.assignments where scheduledSteps[id] == nil {
                scheduledSteps[id] = step
            }
            CBv2SchedSim.confirm(scheduler, plan: plan)
        }

        // Deferred, never dropped: each row lands on a later step, in order.
        XCTAssertEqual(scheduledSteps[prefills[0].id], 0)
        XCTAssertNotNil(scheduledSteps[prefills[1].id])
        XCTAssertNotNil(scheduledSteps[prefills[2].id])
        XCTAssertLessThan(scheduledSteps[prefills[1].id]!, scheduledSteps[prefills[2].id]!)
        for request in prefills {
            let rec = try XCTUnwrap(scheduler.record(for: request.id))
            XCTAssertGreaterThan(rec.numComputedTokens, 0)
        }
    }

    /// The degenerate cap (0) blocks prefill entirely while a decode row
    /// lives — but decode rows are finite, and no NEW decode row can appear
    /// while prefill is blocked, so the step goes pure-prefill (uncapped) as
    /// soon as the last decoder finishes.
    func testZeroCapUnblocksWhenTheDecodeRowFinishes() throws {
        let scheduler = makeScheduler()
        scheduler.mixedStepPrefillTokenCap = 0
        let decode = try makeDecodingRow(scheduler)
        let prefills = try enqueuePrefills(scheduler, count: 2)

        for _ in 0 ..< 3 {
            let plan = scheduler.plan()
            XCTAssertEqual(plan.assignments.map(\.id), [decode.id])
            CBv2SchedSim.confirm(scheduler, plan: plan)
        }
        XCTAssertEqual(scheduler.waitingCount, 2)

        scheduler.finish(id: decode.id, reason: .length)

        let plan = scheduler.plan()
        XCTAssertEqual(plan.assignments.map(\.id), prefills.map(\.id))
        XCTAssertEqual(plan.assignments.map(\.numTokens), [512, 512])
    }
}
