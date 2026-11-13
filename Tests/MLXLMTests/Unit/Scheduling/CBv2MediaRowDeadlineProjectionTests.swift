// Copyright © 2026 Eigen Labs.
//
// First-token projection beside a row with bidirectional media blocks
// (Gemma 4 images): refused while the row still has prompt work, priced as
// ordinary decode once it has sampled from its whole prompt. Pure scheduler
// projections, checked against the scheduler's own plans. No MLX arrays.

import Foundation
import Testing

@testable import MLXLMCommon

extension UnitTests {
    @Suite("First-token projection beside a bidirectional media row")
    struct CBv2MediaRowDeadlineProjectionTests {
        /// Production shape: serialized prefill, and block rows take a smaller
        /// plain chunk than text.
        private func scheduler() -> SchedulerV2 {
            var config = CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 64,
                prefillChunkSize: 16, maxConcurrentPartialPrefills: 1, maxWaiting: 64)
            config.bidirectionalMediaPrefillChunkCeiling = 4
            return SchedulerV2(config: config)
        }

        /// 20 prompt tokens with one image block at [4, 12).
        private func image(
            prompt: Int = 20, block: CBv2ImageSpan = CBv2ImageSpan(tokenOffset: 4, length: 8),
            attention: CBv2MultimodalAttention = .bidirectionalSpans
        ) -> CBv2Request {
            var request = CBv2SchedFixtures.request(
                prompt: Array(1 ... prompt), maxTokens: 64)
            request.multimodal = CBv2MultimodalInput(spans: [block], attention: attention) { [] }
            return request
        }

        private func text(prompt: Int = 40, maxTokens: Int = 1) -> CBv2Request {
            CBv2SchedFixtures.request(prompt: Array(1 ... prompt), maxTokens: maxTokens)
        }

        /// Plan and confirm until `id` samples.
        private func runUntilSampled(_ scheduler: SchedulerV2, _ id: CBv2RequestID) {
            for _ in 0 ..< 100 {
                if CBv2SchedSim.confirm(scheduler, plan: scheduler.plan()).contains(id) { return }
            }
            Issue.record("\(id) never sampled")
        }

        /// The verdict a text request would get now, then the request is
        /// discarded as the engine discards a refused one.
        private func probe(
            _ scheduler: SchedulerV2, inFlight: CBv2StepPlan? = nil
        ) throws -> CBv2FirstTokenWorkProjection {
            let target = text()
            try scheduler.enqueue(target)
            defer { scheduler.finish(id: target.id, reason: .cancelled) }
            return scheduler.firstTokenWorkProjection(
                for: target.id, inFlightAssignments: inFlight?.assignments ?? [])
        }

        private func work(_ projection: CBv2FirstTokenWorkProjection) throws
            -> CBv2FirstTokenScheduledWork
        {
            guard case .bounded(let work, _) = projection else {
                Issue.record("expected a bounded projection, got \(projection)")
                throw CBv2MediaRowProjectionTestFailure()
            }
            return work
        }

        /// Plans and confirms until `target` samples and charges every step the
        /// way the projection does. `launched` is a plan already made but not
        /// yet confirmed.
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
            }
            return CBv2FirstTokenScheduledWork(
                prefillTokens: prefill, decodeTokens: decode, scheduledSteps: steps,
                mixedSteps: mixed, decodeSteps: shapes)
        }

        @Test("text beside a decoding image row is priced like text beside a text decode")
        func decodingImageRowIsPricedAsDecode() throws {
            let scheduler = scheduler()
            let media = image()
            try scheduler.enqueue(media)
            runUntilSampled(scheduler, media.id)

            let target = text()
            try scheduler.enqueue(target)
            let projected = try work(scheduler.firstTokenWorkProjection(for: target.id))
            // Three text chunks of 16, 16 and 8, each beside one image decode.
            #expect(projected.prefillTokens == 40)
            #expect(projected.decodeTokens == 3)
            #expect(projected.scheduledSteps == 3)
            #expect(projected.mixedSteps == 3)
            #expect(projected.decodeSteps == [.init(rows: 1, width: 1): 3])
            #expect(simulate(scheduler, until: target.id) == projected)
        }

        @Test("a launched image decode step is charged in full")
        func launchedImageDecodeIsCharged() throws {
            let scheduler = scheduler()
            let media = image()
            try scheduler.enqueue(media)
            runUntilSampled(scheduler, media.id)
            let launched = scheduler.plan()
            #expect(launched.assignments.map(\.numTokens) == [1])

            let target = text()
            try scheduler.enqueue(target)
            let projected = try work(
                scheduler.firstTokenWorkProjection(
                    for: target.id, inFlightAssignments: launched.assignments))
            #expect(projected.prefillTokens == 40)
            #expect(projected.decodeTokens == 4)
            #expect(projected.scheduledSteps == 4)
            #expect(simulate(scheduler, until: target.id, launched: launched) == projected)
        }

        @Test("text is refused while the image row has prompt work")
        func imageRowInPrefillRefusesText() throws {
            let scheduler = scheduler()
            let media = image()
            try scheduler.enqueue(media)
            #expect(try probe(scheduler) == .unbounded(reason: .multimodalWork), "waiting")

            var plan = scheduler.plan()
            #expect(plan.assignments.map(\.numTokens) == [4])
            #expect(try probe(scheduler, inFlight: plan) == .unbounded(reason: .multimodalWork))
            CBv2SchedSim.confirm(scheduler, plan: plan)
            #expect(try probe(scheduler) == .unbounded(reason: .multimodalWork), "before block")

            plan = scheduler.plan()
            #expect(plan.assignments.map(\.numTokens) == [8], "the block rides one chunk")
            CBv2SchedSim.confirm(scheduler, plan: plan)
            #expect(try probe(scheduler) == .unbounded(reason: .multimodalWork), "after block")

            // The final prompt chunk is launched: the row is decode-ready only
            // optimistically, so the confirmed cursor still refuses.
            CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())
            plan = scheduler.plan()
            let record = try #require(scheduler.record(for: media.id))
            #expect(record.numComputedTokens == record.effectiveTokenCount)
            #expect(try probe(scheduler, inFlight: plan) == .unbounded(reason: .multimodalWork))

            #expect(CBv2SchedSim.confirm(scheduler, plan: plan) == [media.id])
            _ = try work(try probe(scheduler))
        }

        @Test("a one-token image block at the prompt end is still prompt work")
        func finalImageTokenRefusesText() throws {
            let scheduler = scheduler()
            let media = image(prompt: 5, block: CBv2ImageSpan(tokenOffset: 4, length: 1))
            try scheduler.enqueue(media)
            CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())
            let record = try #require(scheduler.record(for: media.id))
            #expect(record.numComputedTokens == 4)
            #expect(record.isDecodeReady, "decode-shaped, but the token is the image")
            #expect(try probe(scheduler) == .unbounded(reason: .multimodalWork))

            runUntilSampled(scheduler, media.id)
            _ = try work(try probe(scheduler))
        }

        @Test("a preempted image row refuses text until it decodes again")
        func preemptedImageRowRefusesUntilDecodeReady() throws {
            let scheduler = scheduler()
            let media = image()
            try scheduler.enqueue(media)
            runUntilSampled(scheduler, media.id)
            for _ in 0 ..< 3 { CBv2SchedSim.confirm(scheduler, plan: scheduler.plan()) }
            #expect(scheduler.record(for: media.id)?.generatedTokenCount == 4)
            _ = try work(try probe(scheduler))

            #expect(scheduler.requeueOnCapacity(media.id))
            #expect(try probe(scheduler) == .unbounded(reason: .multimodalWork), "requeued")
            for _ in 0 ..< 4 { CBv2SchedSim.confirm(scheduler, plan: scheduler.plan()) }
            // Prompt recomputed; three generated tokens still replay at the
            // block row's own chunk size, which the projection does not mirror.
            let record = try #require(scheduler.record(for: media.id))
            #expect(record.numComputedTokens == 20)
            #expect(record.remainingTokens == 4)
            #expect(try probe(scheduler) == .unbounded(reason: .multimodalWork), "replaying")

            runUntilSampled(scheduler, media.id)
            #expect(scheduler.record(for: media.id)?.isDecodeReady == true)
            _ = try work(try probe(scheduler))
        }

        /// The target's projected decode steps beside `first` once it decodes,
        /// under an MTP planner whose live depth gate allows no drafts.
        private func decodeStepsBesideSpeculating(_ first: CBv2Request) throws
            -> [CBv2DecodeStepShape: Int]?
        {
            let scheduler = scheduler()
            try scheduler.enqueue(first)
            runUntilSampled(scheduler, first.id)
            scheduler.speculationPlanner = { _ in 3 }
            scheduler.speculationDraftTokenUpperBound = 3
            scheduler.speculationDepthCeiling = { _ in 0 }
            let target = text(prompt: 16)
            try scheduler.enqueue(target)
            return try work(
                scheduler.firstTokenWorkProjection(for: target.id, mirrorsSpeculationDepth: true)
            ).decodeSteps
        }

        @Test("a decoding image row keeps the configured 1+k MTP charge")
        func decodingImageRowKeepsConservativeSpeculation() throws {
            let textDecode = try decodeStepsBesideSpeculating(text(prompt: 20, maxTokens: 64))
            #expect(textDecode == [.init(rows: 1, width: 1): 1])
            let imageDecode = try decodeStepsBesideSpeculating(image())
            #expect(imageDecode == [.init(rows: 1, width: 4): 1])
        }

        @Test("causal media is projected through its prefill, as before")
        func causalMediaIsUnchanged() throws {
            let scheduler = scheduler()
            try scheduler.enqueue(image(attention: .causal))
            _ = try work(try probe(scheduler))
        }
    }
}

private struct CBv2MediaRowProjectionTestFailure: Error {}
