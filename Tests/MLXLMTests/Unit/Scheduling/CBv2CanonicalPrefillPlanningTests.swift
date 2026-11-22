// Copyright © 2026 Eigen Labs.

import Foundation
import Testing

@testable import MLXLMCommon

extension UnitTests {

    /// The canonical prefill grid in the scheduler's plan. With
    /// `canonicalPrefillGridTokens` set, every prompt chunk of a text request
    /// is one canonical cell, whatever the plain chunk, the solo stripe, the
    /// step budget or the company. The scheduler is planned directly; no arrays.
    @Suite
    struct CBv2CanonicalPrefillPlanningTests {

        private static let rule = CBv2CanonicalBoundaryRule(open: [900], close: [[901]])

        private static func fill(_ count: Int) -> [Int] { [Int](repeating: 7, count: count) }

        /// A user message, the assistant open, a reply and its close, a
        /// follow-up: 30 tokens with boundaries at 5 and 19.
        private static let twoTurnPrompt = fill(4) + [900] + fill(13) + [901] + fill(11)

        private func makeScheduler(
            budget: Int = 64, chunk: Int = 16, stripe: Int? = nil, grid: Int? = 8,
            rule: CBv2CanonicalBoundaryRule? = Self.rule
        ) -> SchedulerV2 {
            var config = CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: budget,
                prefillChunkSize: chunk, soloPrefillStripeTokens: stripe, maxWaiting: 16)
            config.canonicalPrefillGridTokens = grid
            config.canonicalBoundaryRule = rule
            return SchedulerV2(config: config)
        }

        private func request(_ id: UInt64, tokens: [Int]) -> CBv2Request {
            CBv2Request(id: CBv2RequestID(id), promptTokens: tokens, maxTokens: 4)
        }

        private func assigned(_ plan: CBv2StepPlan, _ id: UInt64) -> Int? {
            plan.assignments.first { $0.id == CBv2RequestID(id) }?.numTokens
        }

        /// The prompt chunks of request 1, planned alone to the end of its prompt.
        private func promptChunks(_ scheduler: SchedulerV2, tokens: [Int]) throws -> [Int] {
            try scheduler.enqueue(request(1, tokens: tokens))
            var chunks: [Int] = []
            while chunks.reduce(0, +) < tokens.count, chunks.count < tokens.count {
                let plan = scheduler.plan()
                chunks.append(try #require(assigned(plan, 1)))
                CBv2SchedSim.confirm(scheduler, plan: plan)
            }
            return chunks
        }

        @Test func chunksEndAtGridMultiplesAndMessageBoundaries() throws {
            #expect(Self.rule.boundaries(in: Self.twoTurnPrompt) == [5, 19])
            let chunks = try promptChunks(makeScheduler(), tokens: Self.twoTurnPrompt)
            #expect(chunks == [5, 3, 8, 3, 5, 6])
        }

        @Test func theGridWithoutARuleCutsAtGridMultiplesOnly() throws {
            let chunks = try promptChunks(makeScheduler(rule: nil), tokens: Self.twoTurnPrompt)
            #expect(chunks == [8, 8, 8, 6])
        }

        @Test func withoutTheGridTheRuleChangesNothing() throws {
            let chunks = try promptChunks(makeScheduler(grid: nil), tokens: Self.twoTurnPrompt)
            #expect(chunks == [16, 14])
        }

        @Test func aSoloStripeDoesNotWidenACell() throws {
            let plain = try promptChunks(
                makeScheduler(stripe: 32, grid: nil), tokens: Self.twoTurnPrompt)
            #expect(plain == [30], "the stripe is armed without the grid")
            let chunks = try promptChunks(makeScheduler(stripe: 32), tokens: Self.twoTurnPrompt)
            #expect(chunks == [5, 3, 8, 3, 5, 6])
        }

        @Test func aMediaPromptKeepsOrdinaryChunks() throws {
            let scheduler = makeScheduler()
            var media = request(1, tokens: Self.fill(40))
            media.multimodal = CBv2MultimodalInput(
                spans: [CBv2ImageSpan(tokenOffset: 20, length: 3)], attention: .bidirectionalSpans
            ) { [] }
            try scheduler.enqueue(media)
            #expect(assigned(scheduler.plan(), 1) == 16)
        }

        @Test func aWaitingRequestIsNotAdmittedWithPartOfACell() throws {
            let scheduler = makeScheduler(budget: 12, rule: nil)
            try scheduler.enqueue(request(1, tokens: Self.fill(24)))
            try scheduler.enqueue(request(2, tokens: Self.fill(24)))
            let plan = scheduler.plan()
            #expect(assigned(plan, 1) == 8)
            #expect(assigned(plan, 2) == nil, "the 4 tokens left hold no whole cell")
        }

        @Test func aRunningRowWaitsForAStepThatHoldsItsWholeCell() throws {
            let scheduler = makeScheduler(budget: 12)
            try scheduler.enqueue(request(1, tokens: Self.fill(16)))
            // Boundary at 2: the first cell fits beside request 1, the second does not.
            try scheduler.enqueue(request(2, tokens: [7, 900] + Self.fill(10)))

            let first = scheduler.plan()
            #expect(assigned(first, 1) == 8)
            #expect(assigned(first, 2) == 2)
            CBv2SchedSim.confirm(scheduler, plan: first)

            let second = scheduler.plan()
            #expect(assigned(second, 1) == 8)
            #expect(assigned(second, 2) == nil, "the 4 tokens left do not hold its 6-token cell")
            CBv2SchedSim.confirm(scheduler, plan: second)

            let third = scheduler.plan()
            #expect(assigned(third, 2) == 6)
        }

        @Test func everyPromptChunkIsOneCanonicalCellUnderRandomLoad() throws {
            var generator = CBv2SchedSplitMix64(seed: 20_261_110)
            for _ in 0 ..< 200 {
                let grid = Int.random(in: 2 ... 8, using: &generator)
                let scheduler = makeScheduler(
                    budget: Int.random(in: grid ... 40, using: &generator),
                    chunk: Int.random(in: 1 ... 24, using: &generator),
                    stripe: Bool.random(using: &generator) ? 48 : nil, grid: grid)
                let requests = Int.random(in: 1 ... 6, using: &generator)
                for id in 1 ... requests {
                    var tokens: [Int] = []
                    for _ in 0 ..< Int.random(in: 1 ... 60, using: &generator) {
                        // About one token in six opens or closes a message.
                        tokens.append([7, 7, 7, 7, 900, 901].randomElement(using: &generator)!)
                    }
                    try scheduler.enqueue(request(UInt64(id), tokens: tokens))
                }

                var finished = 0
                for _ in 0 ..< 400 where finished < requests {
                    let plan = scheduler.plan()
                    for (id, count) in plan.assignments {
                        let record = try #require(scheduler.record(for: id))
                        let promptLength = record.request.promptTokens.count
                        let end = record.numComputedTokens
                        guard end <= promptLength else { continue }
                        let partition = try #require(record.canonicalPartition)
                        let start = end - count
                        #expect(start == 0 || partition.isCanonical(start))
                        #expect(end == partition.cellEnd(from: start, limit: promptLength))
                    }
                    for id in CBv2SchedSim.confirm(scheduler, plan: plan) {
                        _ = scheduler.finish(id: id, reason: .stop)
                        finished += 1
                    }
                }
                #expect(finished == requests, "every prompt finishes")
            }
        }
    }
}
