import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Ordinary request ownership retirement", .serialized, .tags(.integration))
struct OrdinaryRequestRetirementTests {
    @Test("ordinary requests expose real retirement without a native shutdown contract")
    func ordinarySubmission() async throws {
        let harness = CBv2SchedHarness()
        #expect(harness.engine.nativeShutdownExecutionContractID == nil)
        let submission = try harness.engine.submitWithRetirement(
            CBv2SchedFixtures.request(prompt: [1, 2, 3], maxTokens: 4))
        let result = await cbv2SchedCollect(submission.events)
        await submission.retirement.wait()
        #expect(result.finishReason == .length)
        #expect(result.tokens.count == 4)
        #expect(harness.backend.liveStates == 0)
        #expect(harness.engine.capacity().activeRequests == 0)
        await harness.engine.shutdown()
    }

    @Test("watchdog terminal and waiter cancellation do not acknowledge a live row")
    func earlyTerminalRetainsOwnership() async throws {
        let layers = CBv2SchedFixtures.tinyLayerKinds()
        let backend = CBv2SchedMockBackend()
        let model = RetirementGatedModel()
        defer { model.release() }
        let engine = EngineV2(
            model: model, layerKinds: layers, backend: backend,
            cacheProvider: CBv2SchedMockCacheProvider(layerKinds: layers),
            sampler: CBv2GreedySampler(), detokenizerFactory: CBv2SchedScriptedDetokFactory(),
            loopConfig: .init(stepTimeout: 0.05, watchdogInterval: 0.01))
        let submission = try engine.submitWithRetirement(
            CBv2SchedFixtures.request(prompt: [1], maxTokens: 100))
        let retired = RetirementFlag()
        let waiter = Task {
            await submission.retirement.wait()
            retired.set()
        }
        let result = await cbv2SchedCollect(submission.events)
        if case .terminal(let cause, _) = result.finishReason {
            #expect(cause == .watchdog)
        } else {
            Issue.record("expected an early watchdog terminal")
        }
        #expect(backend.liveStates > 0)
        waiter.cancel()
        try await Task.sleep(nanoseconds: 50_000_000)
        #expect(!retired.value, "cancelling an ownership wait cannot free a live engine row")
        model.release()
        await waiter.value
        #expect(retired.value && backend.liveStates == 0)
        await engine.shutdown()
    }

    @Test("zero work acknowledges immediately and closed engines reject new work")
    func immediateAndClosed() async throws {
        let harness = CBv2SchedHarness()
        let submission = try harness.engine.submitWithRetirement(
            CBv2SchedFixtures.request(prompt: [1], maxTokens: 0))
        await submission.retirement.wait()
        #expect(harness.backend.makeCalls == 0)
        await harness.engine.shutdown()
        #expect(throws: (any Error).self) {
            try harness.engine.submitWithRetirement(
                CBv2SchedFixtures.request(prompt: [1], maxTokens: 1))
        }
    }
}

private final class RetirementFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var completed = false
    func set() { lock.withLock { completed = true } }
    var value: Bool { lock.withLock { completed } }
}

private final class RetirementGatedModel: CBv2SteppableModel, @unchecked Sendable {
    private let model = CBv2SchedScriptedModel()
    private let condition = NSCondition()
    private var released = false
    func release() {
        condition.lock()
        released = true
        condition.broadcast()
        condition.unlock()
    }
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        condition.lock()
        while !released { condition.wait() }
        condition.unlock()
        return model.forward(tokens: tokens, caches: caches)
    }
}
