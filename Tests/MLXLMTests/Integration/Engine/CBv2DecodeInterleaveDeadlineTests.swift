// Copyright © 2026 Eigen Labs.
//
// First-token deadline admission under the decode-only interleave
// (`decodeStepsBetweenPrefillChunks`), on an EngineV2 stack over mocks (no
// weights). The pure projections are CBv2DecodeInterleaveProjectionTests
// (Unit/Scheduling).

import Foundation
import Testing

@testable import MLXLMCommon

@Suite(.tags(.integration))
struct CBv2DecodeInterleaveDeadlineTests {
    @Test(
        "a deadline that covers the held decode steps is admitted, a shorter one refused",
        arguments: [0, 8, 24])
    func heldStepsArePricedAtAdmission(decodeSteps: Int) async throws {
        var config = CBv2SchedulerConfig(
            maxConcurrentRequests: 3, maxBatchedTokensPerStep: 6, prefillChunkSize: 4,
            maxConcurrentPartialPrefills: 1, maxWaiting: 4)
        config.decodeStepsBetweenPrefillChunks = decodeSteps
        let harness = CBv2SchedHarness(schedulerConfig: config)
        let loop = harness.engine.loopForTesting
        let decoders = (0 ..< 2).map { _ in
            CBv2SchedFixtures.request(prompt: [1], maxTokens: 64)
        }
        try loop.onEngineQueueSync {
            // Freeze right after the loop launches one decode step for both
            // rows; admissions still run on the engine queue.
            loop.suspendStepExecutionAtCountForTesting = 1
            for decoder in decoders { try loop.scheduler.enqueue(decoder) }
            CBv2SchedSim.confirm(loop.scheduler, plan: loop.scheduler.plan())
        }
        let decodeLaunched = await cbv2SchedWait {
            loop.onEngineQueueSync { loop.stepCount == 1 }
        }
        #expect(decodeLaunched)

        // The launched decode step, chunk 1 beside both rows, the held steps
        // and chunk 2: 8 prompt tokens at 4 tok/s, then one second for each
        // step at the decode rate measured for two-row steps.
        let steps = decodeSteps + 3
        let expected = CBv2FirstTokenProjectedWork.bounded(
            work: CBv2FirstTokenScheduledWork(
                prefillTokens: 8, decodeTokens: 2 * steps, scheduledSteps: steps,
                mixedSteps: 2, decodeSteps: [.init(rows: 2, width: 1): steps]),
            serviceDuration: .seconds(2 + steps))
        func admission(after duration: Duration) -> CBv2FirstTokenDeadlineAdmission {
            CBv2FirstTokenDeadlineAdmission(
                deadline: .now.advanced(by: duration), conservativePrefillTokensPerSecond: 4,
                conservativeDecodeTokensPerSecond: 1, decodeRateShape: .init(batchRows: 2))
        }
        func release() {
            loop.onEngineQueueSync {
                for decoder in decoders { loop.finishRequest(decoder.id, reason: .cancelled) }
                loop.suspendStepExecutionAtCountForTesting = nil
            }
        }

        let short = CBv2SchedFixtures.request(prompt: Array(2 ... 9), maxTokens: 1)
        let refused = try await harness.engine.submit(
            short, firstTokenDeadline: admission(after: .seconds(1 + steps)))
        guard case .deadlineUnreachable(let refusedWork) = refused else {
            Issue.record("a deadline one second short of the projection must be refused")
            release()
            await harness.engine.shutdown()
            return
        }
        #expect(refusedWork == expected)

        let covered = CBv2SchedFixtures.request(prompt: Array(2 ... 9), maxTokens: 1)
        let admitted = try await harness.engine.submit(
            covered, firstTokenDeadline: admission(after: .seconds(30 + steps)))
        guard case .admitted(let stream, let admittedWork, _, _) = admitted else {
            Issue.record("a deadline that covers the projection must be admitted")
            release()
            await harness.engine.shutdown()
            return
        }
        #expect(admittedWork == expected)

        release()
        let output = await cbv2SchedCollect(stream)
        #expect(output.finishReason == .length)
        await harness.engine.shutdown()
    }
}
