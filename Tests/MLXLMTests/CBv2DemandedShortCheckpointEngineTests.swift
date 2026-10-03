import XCTest

@testable import MLXLMCommon

/// Real recurrent graphs and byte-only archive reconstruction; real selected
/// Qwen+MTP encrypted-store qualification lives in the provider integration suite.
final class CBv2DemandedShortCheckpointEngineTests: XCTestCase {
    private func engine(_ store: CompleteCheckpointFixtureStore, minimum: Int? = 1_024)
        -> (EngineV2, CBv2ContiguousKVBackend)
    {
        let kinds = [
            CBv2LayerKind(
                attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1, modelLayerIndex: 1)
        ]
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 64 << 20, kvDType: .float32))
        var scheduler = CBv2SchedulerConfig(
            maxConcurrentRequests: 1, maxBatchedTokensPerStep: 4_096,
            prefillChunkSize: max(32, CBv2AttentionV1.queryBlockSize),
            soloPrefillStripeTokens: 4_096,
            maxWaiting: 4, enablePrefixCache: true)
        scheduler.demandedShortCheckpointMinimumTokens = minimum
        let engine = EngineV2(
            model: CompleteCheckpointFixtureModel(), layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds), sampler: CBv2GreedySampler(),
            schedulerConfig: scheduler, admissionConfig: .init(watermarkFraction: 0),
            completePrefixCache: store)
        return (engine, backend)
    }

    func testDemandedShortEndpointRestoresExactNativeContinuationAfterRestart() async throws {
        let store = CompleteCheckpointFixtureStore()
        let (donor, donorBackend) = engine(store)
        let prompt = (0 ..< 2_501).map { $0 % 7 + 1 }
        let original = await cbv2SchedCollect(
            try donor.submit(
                CBv2Request(
                    id: .init(27), promptTokens: prompt, maxTokens: 64, cacheSalt: "tenant",
                    prefixCacheReceiptID: .init(1027), prefixCheckpointTargetTokens: 2_100)))
        XCTAssertEqual(original.finishReason, .length)
        XCTAssertEqual(store.saved.map(\.manifest.position), [2_048])
        store.finishPublicationCallbacks(engine: donor)
        XCTAssertEqual(donor.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(donorBackend.bytesReserved, 0)
        await donor.shutdown()

        let reopened = CompleteCheckpointFixtureStore(archives: store.saved)
        let fork = Array(prompt.prefix(2_048)) + Array(repeating: 11, count: 257)
        let (warm, warmBackend) = engine(reopened)
        let request = CBv2Request(
            id: .init(28), promptTokens: fork, maxTokens: 64, cacheSalt: "tenant",
            prefixCacheReceiptID: .init(1028))
        XCTAssertTrue(try reopened.stage(engine: warm, request: request))
        let restored = await cbv2SchedCollect(try warm.submit(request))
        XCTAssertEqual(restored.usage?.prefixCachePrefillTokensSaved, 2_048)
        XCTAssertEqual(restored.usage?.prefixCacheReplayTokens, 0)
        reopened.finishPublicationCallbacks(engine: warm)
        XCTAssertEqual(warm.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(warmBackend.bytesReserved, 0)
        await warm.shutdown()

        let (cold, coldBackend) = engine(CompleteCheckpointFixtureStore(), minimum: nil)
        let control = await cbv2SchedCollect(
            try cold.submit(
                CBv2Request(
                    id: .init(29), promptTokens: fork, maxTokens: 64, cacheSalt: "tenant")))
        XCTAssertEqual(restored.tokens, control.tokens)
        XCTAssertEqual(coldBackend.bytesReserved, 0)
        await cold.shutdown()
    }

}
