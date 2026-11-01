// Copyright © 2026 Eigen Labs.
//
// First-token projection under the decode-only interleave
// (`decodeStepsBetweenPrefillChunks`): pure scheduler projections, checked
// against the scheduler's own plans. No MLX arrays, engine or model forward.
// The engine's deadline verdict is CBv2DecodeInterleaveDeadlineTests
// (Integration/Engine).

import Foundation
import Testing

@testable import MLXLMCommon

extension UnitTests {
    @Suite("First-token projection with decode-only steps between prompt chunks")
    struct CBv2DecodeInterleaveProjectionTests {
        private func scheduler(decodeSteps: Int, maxConcurrent: Int = 8, budget: Int = 64)
            -> SchedulerV2
        {
            var config = CBv2SchedulerConfig(
                maxConcurrentRequests: maxConcurrent, maxBatchedTokensPerStep: budget,
                prefillChunkSize: 4, maxConcurrentPartialPrefills: 1, maxWaiting: 64)
            config.decodeStepsBetweenPrefillChunks = decodeSteps
            return SchedulerV2(config: config)
        }

        private func request(prompt: Int, maxTokens: Int) -> CBv2Request {
            CBv2SchedFixtures.request(prompt: Array(1 ... prompt), maxTokens: maxTokens)
        }

        /// Admit one-token prompts and confirm their first samples, so each is
        /// a decode-ready running row, as the engine loop would.
        private func addDecoders(_ scheduler: SchedulerV2, count: Int, maxTokens: Int) throws {
            for _ in 0 ..< count {
                try scheduler.enqueue(request(prompt: 1, maxTokens: maxTokens))
            }
            CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())
        }

        private func work(_ projection: CBv2FirstTokenWorkProjection) throws
            -> CBv2FirstTokenScheduledWork
        {
            guard case .bounded(let work, _) = projection else {
                Issue.record("expected a bounded projection, got \(projection)")
                throw CBv2DecodeInterleaveTestFailure()
            }
            return work
        }

        /// Plans and confirms until `target` samples, retiring rows at their
        /// output limit like the engine, and charges every step the way the
        /// projection does. `launched` is a plan already made but not yet
        /// confirmed.
        private func simulate(
            _ scheduler: SchedulerV2, until target: CBv2RequestID,
            launched: CBv2StepPlan? = nil
        ) -> CBv2FirstTokenScheduledWork {
            var prefill = 0
            var decode = 0
            var steps = 0
            var mixed = 0
            var shapes: [CBv2DecodeStepShape: Int] = [:]
            var next = launched
            for _ in 0 ..< 1_000 {
                let plan = next ?? scheduler.plan()
                next = nil
                var stepPrefill = 0
                var stepDecode = 0
                var decodeRows = 0
                var widest = 0
                for (id, count) in plan.assignments {
                    guard let rec = scheduler.record(for: id) else { continue }
                    let start = rec.numComputedTokens - count
                    let promptCount = min(count, max(0, rec.request.promptTokens.count - start))
                    stepPrefill += promptCount
                    stepDecode += count - promptCount
                    if count > promptCount {
                        decodeRows += 1
                        widest = max(widest, count - promptCount)
                    }
                }
                steps += 1
                prefill += stepPrefill
                decode += stepDecode
                if stepPrefill > 0, stepDecode > 0 { mixed += 1 }
                if decodeRows > 0 {
                    shapes[CBv2DecodeStepShape(rows: decodeRows, width: widest), default: 0] += 1
                }
                if CBv2SchedSim.confirm(scheduler, plan: plan).contains(target) { break }
                for rec in scheduler.running
                where rec.generatedTokenCount >= rec.request.maxTokens {
                    scheduler.finish(id: rec.id, reason: .length)
                }
            }
            return CBv2FirstTokenScheduledWork(
                prefillTokens: prefill, decodeTokens: decode, scheduledSteps: steps,
                mixedSteps: mixed, decodeSteps: shapes)
        }

        @Test(
            "held decode-only steps sit between the target's prompt chunks",
            arguments: [0, 8, 24])
        func heldStepsSitBetweenChunks(decodeSteps: Int) throws {
            let scheduler = scheduler(decodeSteps: decodeSteps)
            try addDecoders(scheduler, count: 1, maxTokens: 200)
            let target = request(prompt: 12, maxTokens: 1)
            try scheduler.enqueue(target)

            // Three 4-token chunks beside the decode row, and the held steps
            // after the first two.
            let projected = try work(scheduler.firstTokenWorkProjection(for: target.id))
            let steps = 3 + 2 * decodeSteps
            #expect(projected.prefillTokens == 12)
            #expect(projected.decodeTokens == steps)
            #expect(projected.scheduledSteps == steps)
            #expect(projected.mixedSteps == 3)
            #expect(projected.decodeSteps == [.init(rows: 1, width: 1): steps])
            #expect(simulate(scheduler, until: target.id) == projected)
        }

        @Test(
            "a chunk that just ran holds the target behind the owed steps",
            arguments: [0, 8, 24])
        func owedStepsPrecedeTheFirstChunk(decodeSteps: Int) throws {
            let scheduler = scheduler(decodeSteps: decodeSteps)
            try addDecoders(scheduler, count: 1, maxTokens: 200)
            // A one-chunk prompt beside the decode row owes the interleave.
            try scheduler.enqueue(request(prompt: 4, maxTokens: 200))
            CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())
            #expect(scheduler.decodeOnlyStepsOwed == decodeSteps)
            let target = request(prompt: 8, maxTokens: 1)
            try scheduler.enqueue(target)

            let projected = try work(scheduler.firstTokenWorkProjection(for: target.id))
            let steps = 2 + 2 * decodeSteps
            #expect(projected.prefillTokens == 8)
            #expect(projected.decodeTokens == 2 * steps)
            #expect(projected.scheduledSteps == steps)
            #expect(projected.mixedSteps == 2)
            #expect(projected.decodeSteps == [.init(rows: 2, width: 1): steps])
            #expect(simulate(scheduler, until: target.id) == projected)
        }

        @Test(
            "the launched plan has already settled the owed count",
            arguments: [0, 8, 24])
        func launchedPlanSettlesTheOwedCount(decodeSteps: Int) throws {
            let scheduler = scheduler(decodeSteps: decodeSteps)
            try addDecoders(scheduler, count: 1, maxTokens: 200)
            let target = request(prompt: 12, maxTokens: 1)
            try scheduler.enqueue(target)
            // The target's first chunk is in flight beside the decode row.
            let launched = scheduler.plan()
            #expect(scheduler.decodeOnlyStepsOwed == decodeSteps)

            let projected = try work(
                scheduler.firstTokenWorkProjection(
                    for: target.id, inFlightAssignments: launched.assignments))
            let steps = 3 + 2 * decodeSteps
            #expect(projected.scheduledSteps == steps)
            #expect(projected.decodeTokens == steps)
            #expect(projected.mixedSteps == 3)
            #expect(simulate(scheduler, until: target.id, launched: launched) == projected)
        }

        @Test("a decode row that retires during the held steps releases the next chunk")
        func retiredDecodeRowEndsTheHold() throws {
            let scheduler = scheduler(decodeSteps: 8)
            // One generated token so far; three more reach the limit.
            try addDecoders(scheduler, count: 1, maxTokens: 4)
            let target = request(prompt: 12, maxTokens: 1)
            try scheduler.enqueue(target)

            // Chunk 1 beside the row, two held steps, the row retires, then
            // chunks 2 and 3 run back to back.
            let projected = try work(scheduler.firstTokenWorkProjection(for: target.id))
            #expect(projected.scheduledSteps == 5)
            #expect(projected.prefillTokens == 12)
            #expect(projected.decodeTokens == 3)
            #expect(projected.mixedSteps == 1)
            #expect(simulate(scheduler, until: target.id) == projected)
        }

        @Test("held steps keep the speculative width the live plan verifies")
        func heldStepsKeepTheSpeculativeWidth() throws {
            let scheduler = scheduler(decodeSteps: 8)
            try addDecoders(scheduler, count: 1, maxTokens: 200)
            scheduler.speculationPlanner = { _ in 3 }
            scheduler.speculationDraftTokenUpperBound = 3
            let target = request(prompt: 12, maxTokens: 1)
            try scheduler.enqueue(target)

            // Every round credits one token and charges its 1+3 width.
            let projected = try work(scheduler.firstTokenWorkProjection(for: target.id))
            #expect(projected.scheduledSteps == 19)
            #expect(projected.decodeTokens == 4 * 19)
            #expect(projected.decodeSteps == [.init(rows: 1, width: 4): 19])
        }

        @Test(
            "a full running set drains its stretch, then the rest of the hold",
            arguments: [(0, 4, 7), (8, 9, 12)])
        func fullSlotStretchCountsTowardTheHold(
            decodeSteps: Int, steps: Int, decodeTokens: Int
        ) throws {
            let scheduler = scheduler(decodeSteps: decodeSteps, maxConcurrent: 2)
            let finishing = request(prompt: 1, maxTokens: 4)
            try scheduler.enqueue(finishing)
            CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())
            try scheduler.enqueue(request(prompt: 4, maxTokens: 200))
            CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())
            let target = request(prompt: 4, maxTokens: 1)
            try scheduler.enqueue(target)

            // Two full-slot steps until `finishing` reaches its limit and the
            // one-step-late chained successor all count toward the hold; the
            // remaining held steps run the surviving row alone, then the
            // target's chunk.
            let projected = try work(scheduler.firstTokenWorkProjection(for: target.id))
            #expect(projected.scheduledSteps == steps)
            #expect(projected.prefillTokens == 4)
            #expect(projected.decodeTokens == decodeTokens)
            #expect(projected.mixedSteps == 1)
            #expect(
                projected.decodeSteps
                    == [.init(rows: 2, width: 1): 3, .init(rows: 1, width: 1): steps - 3])
        }
    }
}

private struct CBv2DecodeInterleaveTestFailure: Error {}
