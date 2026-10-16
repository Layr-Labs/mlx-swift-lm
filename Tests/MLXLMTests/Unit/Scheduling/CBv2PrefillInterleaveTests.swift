import Foundation
import Testing

@testable import MLXLMCommon

extension UnitTests {
    /// The opt-in prompt-row ceiling (`maxPrefillRowsPerStep`) and the
    /// decode-only interleave (`decodeStepsBetweenPrefillChunks`). Pure
    /// scheduler bookkeeping: chunk geometry never changes, prompt work is
    /// only deferred, nothing leaks on rollback and no prompt starves.
    struct CBv2PrefillInterleaveTests {
        static func scheduler(
            rows: Int? = 1, decodeSteps: Int = 0, budget: Int = 64, chunk: Int = 16
        ) -> SchedulerV2 {
            var config = CBv2SchedulerConfig(
                maxConcurrentRequests: 8, maxBatchedTokensPerStep: budget,
                prefillChunkSize: chunk, maxWaiting: 16)
            config.maxPrefillRowsPerStep = rows
            config.decodeStepsBetweenPrefillChunks = decodeSteps
            return SchedulerV2(config: config)
        }

        /// Ids only need to be unique within one scheduler.
        static func request(_ id: UInt64, prompt: Int, maxTokens: Int = 32) -> CBv2Request {
            CBv2Request(
                id: CBv2RequestID(id), promptTokens: Array(repeating: 3, count: prompt),
                maxTokens: maxTokens)
        }

        /// Confirms one sample for every assignment that finished its tokens.
        static func confirm(_ scheduler: SchedulerV2, _ plan: CBv2StepPlan) {
            for (id, _) in plan.assignments {
                guard let rec = scheduler.record(for: id),
                    rec.numComputedTokens == rec.effectiveTokenCount
                else { continue }
                scheduler.markPendingSamples(ids: [id])
                scheduler.recordSampled(id: id, token: 5)
            }
        }

        static func promptRows(_ scheduler: SchedulerV2, _ plan: CBv2StepPlan) -> [Int] {
            plan.assignments.compactMap { id, count in
                guard let rec = scheduler.record(for: id) else { return nil }
                // Counted after the optimistic advance: prompt work ended at or
                // before the prompt boundary.
                return rec.numComputedTokens <= rec.request.promptTokens.count ? count : nil
            }
        }

        @Test func burstPrefillsOneRequestPerStepWithCanonicalChunks() throws {
            let scheduler = Self.scheduler()
            let prompts = [40, 7, 16, 5]
            for (id, prompt) in prompts.enumerated() {
                try scheduler.enqueue(Self.request(UInt64(id + 1), prompt: prompt))
            }
            var chunks: [[Int]] = []
            for _ in 0 ..< 12 {
                let plan = scheduler.plan()
                let prompt = Self.promptRows(scheduler, plan)
                #expect(prompt.count <= 1)
                if !prompt.isEmpty { chunks.append(prompt) }
                Self.confirm(scheduler, plan)
            }
            // FCFS, every chunk the configured size or the prompt's remainder.
            #expect(chunks.flatMap { $0 } == [16, 16, 8, 7, 16, 5])
            let decodeReady = scheduler.running.allSatisfy(\.isDecodeReady)
            #expect(decodeReady)
        }

        @Test func defaultConfigurationKeepsSeveralPromptsPerStep() throws {
            let scheduler = Self.scheduler(rows: nil)
            for (id, prompt) in [7, 5, 9].enumerated() {
                try scheduler.enqueue(Self.request(UInt64(id + 1), prompt: prompt))
            }
            let plan = scheduler.plan()
            #expect(Self.promptRows(scheduler, plan) == [7, 5, 9])
        }

        @Test func decodeRowsRunBetweenPromptChunks() throws {
            let scheduler = Self.scheduler(decodeSteps: 2)
            let decoding = Self.request(1, prompt: 4)
            try scheduler.enqueue(decoding)
            Self.confirm(scheduler, scheduler.plan())
            #expect(scheduler.record(for: decoding.id)?.isDecodeReady == true)
            let long = Self.request(2, prompt: 48)
            try scheduler.enqueue(long)
            var pattern: [Int] = []
            for _ in 0 ..< 9 {
                let plan = scheduler.plan()
                #expect(plan.assignments.contains { $0.id == decoding.id && $0.numTokens == 1 })
                let prompt = plan.assignments.first { $0.id == long.id }.flatMap { assignment in
                    let rec = scheduler.record(for: long.id)
                    return rec.map { $0.numComputedTokens <= 48 } == true
                        ? assignment.numTokens : nil
                }
                pattern.append(prompt ?? 0)
                Self.confirm(scheduler, plan)
            }
            // One chunk, two decode-only steps, repeated; the last chunk samples.
            #expect(pattern == [16, 0, 0, 16, 0, 0, 16, 0, 0])
            #expect(scheduler.record(for: long.id)?.isDecodeReady == true)
        }

        @Test func promptsAreNeverHeldWithoutDecodeRows() throws {
            let scheduler = Self.scheduler(decodeSteps: 5)
            let first = Self.request(1, prompt: 32)
            try scheduler.enqueue(first)
            let chunks = (0 ..< 2).map { _ -> Int in
                let plan = scheduler.plan()
                Self.confirm(scheduler, plan)
                return plan.assignments.first?.numTokens ?? 0
            }
            #expect(chunks == [16, 16])
        }

        /// First-token projection does not model either limit, so it refuses
        /// to forecast rather than promise an optimistic step count.
        @Test func firstTokenProjectionRefusesUnderEitherLimit() throws {
            for (rows, decodeSteps) in [(Int?.some(1), 0), (nil, 2), (nil, 0)] {
                let scheduler = Self.scheduler(rows: rows, decodeSteps: decodeSteps)
                var config = scheduler.config
                config.maxConcurrentPartialPrefills = 1
                let serialized = SchedulerV2(config: config)
                let request = Self.request(1, prompt: 40)
                try serialized.enqueue(request)
                let projection = serialized.firstTokenWorkProjection(for: request.id)
                var refused = false
                if case .unbounded(reason: .unsupportedScheduler) = projection { refused = true }
                #expect(refused == (rows != nil || decodeSteps > 0))
            }
        }

        @Test func rollbackRestoresTheInterleaveDebt() throws {
            let scheduler = Self.scheduler(decodeSteps: 1)
            let decoding = Self.request(1, prompt: 4)
            try scheduler.enqueue(decoding)
            Self.confirm(scheduler, scheduler.plan())
            let long = Self.request(2, prompt: 48)
            try scheduler.enqueue(long)
            let carried = scheduler.plan()
            #expect(carried.assignments.contains { $0.id == long.id && $0.numTokens == 16 })
            scheduler.rollback(carried)
            // The rolled-back plan never ran: its chunk is offered again now.
            let retried = scheduler.plan()
            #expect(retried.assignments.contains { $0.id == long.id && $0.numTokens == 16 })
            Self.confirm(scheduler, retried)
            let held = scheduler.plan()
            #expect(!held.assignments.contains { $0.id == long.id })
        }
    }
}
