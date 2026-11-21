import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Durable prefix checkpoints with the DSpark drafter on: the checkpoint
/// carries the target windows, the conv state and the drafter's own context
/// history (`native-paged-historical-recurrent-mtp-v1`), restores into a fresh
/// engine, and continues exactly as the cold run, which equals the run with
/// the drafter off. Another tenant's salt never stages it.
@Suite(.serialized, .tags(.integration))
struct Inkling2DSparkPrefixEngineTests {
    static func engine(
        drafting: Bool, store: CompleteCheckpointFixtureStore?
    ) throws -> (EngineV2, PagedKVBackend, Inkling2DSparkDrafter?) {
        let dtype = DType.float32
        let model = try Inkling2DSparkTargetEngineTests.target(dtype: dtype)
        let drafter =
            drafting
            ? try Inkling2DSparkTargetEngineTests.drafter(target: model, dtype: dtype) : nil
        let kinds = model.cbv2LayerKinds
        let types = Array(repeating: dtype, count: kinds.count)
        let paged = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 128 << 20, dtype: dtype, maxPrefillChunk: 64,
                nominalMaxSequenceLength: 512, segmentSizeBytes: 1 << 20, layerDTypes: types))
        let storage = paged.makeLayerCaches()
        let caches = model.newCacheV2 { index, _ in storage[index] }
        let engine = EngineV2(
            model: CBv2SteppableLanguageModelAdapter(model), layerKinds: kinds, backend: paged,
            cacheProvider: CBv2LayerCacheBank(caches: caches), sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 2, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 64, maxWaiting: 8, enablePrefixCache: store != nil),
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store,
            completeCheckpointKVDTypes: store == nil ? nil : types,
            mtpDrafter: drafter,
            mtpConfig: drafter == nil
                ? .init() : Inkling2DSparkDrafter.defaultMTPConfig(adaptive: false))
        return (engine, paged, drafter)
    }

    static func request(_ id: UInt64, prompt: [Int], salt: String = "dspark-fixture")
        -> CBv2Request
    {
        .init(
            id: .init(id), promptTokens: prompt, sampling: .init(temperature: 0, seed: 0),
            maxTokens: 12, cacheSalt: salt, prefixCacheReceiptID: .init(id + 1000),
            prefixCheckpointTargetTokens: 256)
    }

    @Test func restoredPrefixWithDrafterEqualsColdAndDrafterOff() async throws {
        let prompt = SyntheticModel.tokens(count: 300, vocabularySize: 60, seed: 77)

        let (off, _, _) = try Self.engine(drafting: false, store: nil)
        let offResult = try await cbv2CollectFinished(
            try off.submit(Self.request(9, prompt: prompt)), timeoutSeconds: 300)
        await off.shutdown()

        let seed = CompleteCheckpointFixtureStore(segmentBytes: 4093)
        let (cold, coldBackend, _) = try Self.engine(drafting: true, store: seed)
        #expect(cold.loopForTesting.mtp != nil)
        let coldResult = try await cbv2CollectFinished(
            try cold.submit(Self.request(1, prompt: prompt)), timeoutSeconds: 300)
        seed.finishPublicationCallbacks(engine: cold)
        await cold.shutdown()
        try #require(coldResult.finishReason == .length && coldResult.tokens.count == 12)
        #expect(coldResult.tokens == offResult.tokens)
        #expect(coldBackend.bytesReserved == 0)
        let archives = seed.saved.filter { $0.manifest.position == 256 }
        try #require(!archives.isEmpty)
        #expect(
            archives[0].manifest.backendLayout
                == CBv2CompleteCheckpointManifest.pagedHistoricalRecurrentMTPLayout)
        #expect(archives[0].manifest.assistantCodecID?.hasPrefix("inkling2-dspark") == true)

        let restored = CompleteCheckpointFixtureStore(archives: archives, segmentBytes: 4093)
        let (warm, warmBackend, _) = try Self.engine(drafting: true, store: restored)
        #expect(
            try !restored.stage(
                engine: warm, request: Self.request(2, prompt: prompt, salt: "other")))
        let input = Self.request(3, prompt: prompt)
        try #require(try restored.stage(engine: warm, request: input))
        let warmResult = try await cbv2CollectFinished(try warm.submit(input), timeoutSeconds: 300)
        restored.finishPublicationCallbacks(engine: warm)
        let metrics = warm.mtpMetricsSnapshot()
        await warm.shutdown()
        #expect(warmResult.tokens == coldResult.tokens)
        #expect(warmResult.usage?.prefixCachePrefillTokensSaved == 256)
        #expect((metrics?.rounds ?? 0) > 0)
        #expect(warmBackend.bytesReserved == 0 && warm.admissionForTesting.bytesReserved == 0)
        #expect(warm.loopForTesting.mtp?.requestStateCountForTesting == 0)
    }
}
