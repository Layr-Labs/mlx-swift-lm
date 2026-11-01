import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// EngineV2 serving under the Super co-scheduling policy: one prompt chunk
/// per step at the configured chunk size, a step budget that never cuts that
/// chunk, and optional decode-only steps between chunks. A burst of requests,
/// some with prompts of several chunks, must give each request the tokens it
/// gets served alone with the same chunk size, for row-exact batched decode
/// and for batched speculation. Synthetic weights.
@Suite(.tags(.integration))
struct NemotronSuperCoScheduledServingTests {
    typealias Batched = NemotronSuperBatchedEngineTests
    typealias Speculative = NemotronSuperMTPBatchedEngineTests

    static let chunk = 8
    static let rows = 4

    static func policy(depth: Int, decodeSteps: Int) -> CBv2SchedulerConfig {
        var config = CBv2SchedulerConfig(
            maxConcurrentRequests: rows,
            maxBatchedTokensPerStep: chunk + rows * (1 + depth),
            prefillChunkSize: chunk, maxWaiting: 8)
        config.maxPrefillRowsPerStep = 1
        config.decodeStepsBetweenPrefillChunks = decodeSteps
        return config
    }

    static func engine(
        _ target: any LanguageModel,
        kinds: [CBv2LayerKind], caches: [any CBv2AttendingLayerCache], kvDType: DType,
        config: CBv2SchedulerConfig, drafter: NemotronSuperMTPAssistant?, depth: Int,
        sampler: any CBv2StepSampler = CBv2GreedySampler()
    ) -> (EngineV2, CBv2ContiguousKVBackend) {
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 64 << 20, kvDType: kvDType))
        let engine = EngineV2(
            model: CBv2SteppableLanguageModelAdapter(target), layerKinds: kinds,
            backend: backend, cacheProvider: CBv2LayerCacheBank(caches: caches),
            sampler: sampler, schedulerConfig: config,
            admissionConfig: .init(watermarkFraction: 0), mtpDrafter: drafter,
            mtpConfig: .init(
                enabled: drafter != nil, maxDraftTokens: depth, maxSpeculativeBatch: rows,
                fixedDraftTokens: depth, verificationMode: .rectangularExact))
        return (engine, backend)
    }

    /// Every request submitted at once; each collected to its finish.
    static func burst(_ engine: EngineV2) async throws -> [NemotronSuperMTPEngineTests.Collected] {
        let indices = Speculative.requests.indices.filter { $0 < rows }
        let streams = try indices.map { try engine.submit(Speculative.request($0, base: 300)) }
        return try await withThrowingTaskGroup(
            of: (Int, NemotronSuperMTPEngineTests.Collected).self
        ) { group in
            for (index, stream) in zip(indices, streams) {
                group.addTask { (index, try await Batched.collect(stream) { _ in }) }
            }
            var results: [Int: NemotronSuperMTPEngineTests.Collected] = [:]
            for try await (index, collected) in group { results[index] = collected }
            return indices.map { results[$0] ?? .init() }
        }
    }

    @Test(.timeLimit(.minutes(5)), arguments: [(false, 0), (false, 2), (true, 0), (true, 3)])
    func burstUnderTheServingPolicyMatchesSoloDecode(speculative: Bool, decodeSteps: Int)
        async throws
    {
        let testCase = Speculative.Case(
            head: .predictive, depth: 2, quantized: true, rows: Self.rows)
        let stack = try Speculative.stack(testCase)
        let solo = try await Speculative.solo(stack.model, kvDType: .bfloat16)
        let (engine, backend): (EngineV2, CBv2ContiguousKVBackend)
        if speculative {
            (engine, backend) = Self.engine(
                stack.target, kinds: stack.target.cbv2LayerKinds,
                caches: stack.target.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) },
                kvDType: .bfloat16, config: Self.policy(depth: 2, decodeSteps: decodeSteps),
                drafter: stack.assistant, depth: 2)
        } else {
            let adapter = try NemotronSuperRecurrentAdapter(
                stack.model, maximumDecodeRows: Self.rows, decodeProducts: .rowExact)
            (engine, backend) = Self.engine(
                adapter, kinds: adapter.cbv2LayerKinds,
                caches: adapter.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) },
                kvDType: .bfloat16, config: Self.policy(depth: 0, decodeSteps: decodeSteps),
                drafter: nil, depth: 0)
        }
        let results = try await Self.burst(engine)
        for (index, result) in results.enumerated() {
            #expect(
                result.tokens == solo[index].tokens,
                "request \(index) speculative \(speculative) decode steps \(decodeSteps)")
            #expect(result.finish == .length)
        }
        if speculative {
            let metrics = try #require(engine.mtpMetricsSnapshot())
            #expect(metrics.rounds > 0 && metrics.rectangularVerificationRounds > 0)
        }
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
    }
}
