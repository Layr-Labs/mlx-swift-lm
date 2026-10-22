// Copyright © 2026 Eigen Labs.

import Foundation
import Testing

@testable import MLXLMCommon

extension UnitTests {

    /// `CBv2SchedulerConfig.bidirectionalMediaPrefillChunkCeiling`: a caller may
    /// widen `prefillChunkSize` for text while requests with bidirectional media
    /// blocks keep a smaller plain chunk. Blocks still ride one chunk whole,
    /// causal media and text are untouched, and an unset or non-binding ceiling
    /// leaves planning unchanged. The scheduler is planned directly; no arrays.
    @Suite
    struct BidirectionalMediaChunkCeilingTests {

        private func makeScheduler(
            budget: Int = 64, chunk: Int = 16, ceiling: Int? = 4, partialCap: Int? = nil
        ) -> SchedulerV2 {
            var config = CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: budget,
                prefillChunkSize: chunk, maxConcurrentPartialPrefills: partialCap, maxWaiting: 8)
            config.bidirectionalMediaPrefillChunkCeiling = ceiling
            return SchedulerV2(config: config)
        }

        private func request(
            _ id: UInt64, tokens: Int, spans: [CBv2ImageSpan] = [],
            attention: CBv2MultimodalAttention = .bidirectionalSpans, maxTokens: Int = 4
        ) -> CBv2Request {
            var request = CBv2Request(
                id: CBv2RequestID(id), promptTokens: [Int](repeating: 7, count: tokens),
                maxTokens: maxTokens)
            if !spans.isEmpty {
                request.multimodal = CBv2MultimodalInput(spans: spans, attention: attention) { [] }
            }
            return request
        }

        private func assigned(_ plan: CBv2StepPlan, _ id: UInt64) -> Int? {
            plan.assignments.first { $0.id == CBv2RequestID(id) }?.numTokens
        }

        @Test func blockBearingRequestTakesTheCeilingBesideAWideTextChunk() throws {
            let scheduler = makeScheduler()
            try scheduler.enqueue(request(1, tokens: 40))
            try scheduler.enqueue(
                request(2, tokens: 40, spans: [CBv2ImageSpan(tokenOffset: 20, length: 3)]))

            let first = scheduler.plan()
            #expect(assigned(first, 1) == 16, "text keeps the configured plain chunk")
            #expect(assigned(first, 2) == 4, "a block-bearing request takes the ceiling")
            CBv2SchedSim.confirm(scheduler, plan: first)

            let second = scheduler.plan()
            #expect(assigned(second, 1) == 16)
            #expect(assigned(second, 2) == 4)
        }

        @Test func aBlockLongerThanTheCeilingStillRidesOneChunk() throws {
            let scheduler = makeScheduler()
            try scheduler.enqueue(
                request(1, tokens: 24, spans: [CBv2ImageSpan(tokenOffset: 4, length: 10)]))

            let lead = scheduler.plan()
            #expect(assigned(lead, 1) == 4, "text before the block, ending at its start")
            CBv2SchedSim.confirm(scheduler, plan: lead)

            let block = scheduler.plan()
            #expect(assigned(block, 1) == 10, "the chunk snaps over the whole block")
            CBv2SchedSim.confirm(scheduler, plan: block)

            let tail = scheduler.plan()
            #expect(assigned(tail, 1) == 4, "trailing text returns to the ceiling")
        }

        @Test(arguments: [nil, 16, 32, 0, -4] as [Int?])
        func unsetOrNonBindingCeilingKeepsThePlainChunk(ceiling: Int?) throws {
            let scheduler = makeScheduler(ceiling: ceiling)
            try scheduler.enqueue(
                request(1, tokens: 40, spans: [CBv2ImageSpan(tokenOffset: 30, length: 3)]))
            #expect(assigned(scheduler.plan(), 1) == 16)
        }

        @Test func causalMediaAndTextIgnoreTheCeiling() throws {
            let scheduler = makeScheduler()
            try scheduler.enqueue(
                request(
                    1, tokens: 40, spans: [CBv2ImageSpan(tokenOffset: 20, length: 3)],
                    attention: .causal))
            try scheduler.enqueue(request(2, tokens: 40))
            let plan = scheduler.plan()
            #expect(assigned(plan, 1) == 16, "causal media has no bidirectional block")
            #expect(assigned(plan, 2) == 16)
        }

        @Test func deferredBlockRowResumesAtTheCeiling() throws {
            // A decode row leaves less than the block's length, so the block row
            // is deferred and then admitted ahead of the running pass. Its later
            // text chunks keep the ceiling rather than the wide plain chunk.
            let scheduler = makeScheduler(budget: 8, chunk: 8, ceiling: 2)
            try scheduler.enqueue(request(1, tokens: 3, maxTokens: 32))
            CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())

            try scheduler.enqueue(
                request(2, tokens: 14, spans: [CBv2ImageSpan(tokenOffset: 0, length: 8)]))
            let starved = scheduler.plan()
            #expect(assigned(starved, 1) == 1)
            #expect(assigned(starved, 2) == nil, "the block cannot fit the leftover budget")
            CBv2SchedSim.confirm(scheduler, plan: starved)

            let deferred = scheduler.plan()
            #expect(assigned(deferred, 2) == 8, "deferred admission carries the whole block")
            #expect(scheduler.record(for: CBv2RequestID(2))?.plannedPrefillChunkSize == 2)
            CBv2SchedSim.confirm(scheduler, plan: deferred)

            let tail = scheduler.plan()
            #expect(assigned(tail, 1) == 1)
            #expect(assigned(tail, 2) == 2, "trailing text keeps the ceiling")
        }

        @Test func serializedPrefillKeepsTheCeilingBesideDecode() throws {
            // Production serializes prompt work; the block-bearing successor of a
            // decoding text row prefills at the ceiling, not the wide chunk.
            let scheduler = makeScheduler(budget: 64, chunk: 16, ceiling: 4, partialCap: 1)
            try scheduler.enqueue(request(1, tokens: 16, maxTokens: 32))
            CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())
            try scheduler.enqueue(
                request(2, tokens: 20, spans: [CBv2ImageSpan(tokenOffset: 12, length: 3)]))
            let plan = scheduler.plan()
            #expect(assigned(plan, 1) == 1)
            #expect(assigned(plan, 2) == 4)
        }
    }
}
