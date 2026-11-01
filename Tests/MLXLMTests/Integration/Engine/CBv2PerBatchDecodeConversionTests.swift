// Copyright © 2026 Eigen Labs.
//
// Per-batch decode charging through the engine's first-token deadline
// conversion, on an EngineV2 stack over mocks (no weights). The pure
// projection and arithmetic cases are CBv2PerBatchDecodeChargeTests
// (Unit/Scheduling).

import Foundation
import Testing

@testable import MLXLMCommon

@Suite(.tags(.integration))
struct CBv2PerBatchDecodeConversionTests {
    @Test("the engine prices per-batch decode at steps, and legacy at rows")
    func engineConversionUsesThePolicyCharge() async throws {
        let harness = CBv2SchedHarness()
        let target = CBv2SchedFixtures.request(prompt: Array(1 ... 8), maxTokens: 1)
        func seconds(_ work: CBv2FirstTokenScheduledWork, _ shape: CBv2DecodeRateShape?) -> Double?
        {
            let policy = CBv2FirstTokenDeadlineAdmission(
                deadline: .now.advanced(by: .seconds(60)),
                conservativePrefillTokensPerSecond: 8, conservativeDecodeTokensPerSecond: 6,
                decodeRateShape: shape)
            let projected = harness.engine.loopForTesting.onEngineQueueSync {
                harness.engine.loopForTesting.firstTokenProjectedWork(
                    .bounded(work: work, capacityOperations: []),
                    request: target, reusedPrefix: false, targetComputedTokens: 0,
                    admission: policy)
            }
            guard case .bounded(_, let duration) = projected else { return nil }
            return Double(duration.components.seconds)
                + Double(duration.components.attoseconds) / 1e18
        }
        let plain = CBv2FirstTokenScheduledWork(
            prefillTokens: 8, decodeTokens: 24, scheduledSteps: 6, mixedSteps: 6,
            decodeSteps: [.init(rows: 4, width: 1): 6])
        // Prefill 8/8 = 1 s. Legacy decode 24/6 = 4 s; per-batch at the
        // measured rows is six steps, 1 s; measured solo it is 4 s again.
        #expect(try #require(seconds(plain, nil)) == 5)
        #expect(try #require(seconds(plain, .init(batchRows: 4))) == 2)
        #expect(try #require(seconds(plain, .init(batchRows: 1))) == 5)
        // Three solo depth-2 rounds measured at 1.5 confirmed tokens each:
        // 4.5/6 s, against 9/6 s for the legacy per-row width.
        let rounds = CBv2FirstTokenScheduledWork(
            prefillTokens: 8, decodeTokens: 9, scheduledSteps: 3, mixedSteps: 3,
            decodeSteps: [.init(rows: 1, width: 3): 3])
        #expect(try #require(seconds(rounds, nil)) == 2.5)
        #expect(
            try #require(seconds(rounds, .init(batchRows: 1, tokensPerStep: 1.5, stepWidth: 3)))
                == 1.75)
        await harness.engine.shutdown()
    }
}
