import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Actual Flint layers, learned sinks and sliding RoPE through production
/// engines. Only the encoded-byte store is a fixture; it retains no MLX arrays.
@Suite(.serialized, .tags(.integration))
struct FlintPrefixTests {
    private func model() throws -> FlintModel {
        let url = try #require(
            Bundle.module.url(forResource: "flint-oracle", withExtension: "json"))
        let fixture = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        var config = try #require(fixture["config"] as? [String: Any])
        // The native paged kernel supports head dimensions >= 64.
        for key in [
            "hidden_size", "intermediate_size", "head_dim", "moe_intermediate_size",
            "shared_expert_intermediate_size",
        ] { config[key] = 64 }
        let c = try JSONDecoder().decode(
            FlintConfiguration.self,
            from: JSONSerialization.data(withJSONObject: config))
        let m = FlintModel(c)
        var weights = [String: MLXArray]()
        for (index, entry) in m.parameters().flattened().sorted(by: { $0.0 < $1.0 }).enumerated() {
            let (name, value) = entry
            let noise = MLXRandom.normal(
                value.shape, key: MLXRandom.key(UInt64(20_261_021 + index)))
            if name.contains("norm") {
                weights[name] = 1 + noise * 0.01
            } else {
                weights[name] = noise / sqrt(Float(value.shape.last ?? 1))
            }
        }
        try m.installFlintWeights(weights)
        eval(m)
        return m
    }

    private func engine(_ m: FlintModel, paged: Bool, store: CompleteCheckpointFixtureStore)
        throws -> (EngineV2, any CBv2KVBackend)
    {
        let kinds = m.cbv2LayerKinds
        let types = Array(repeating: DType.float32, count: kinds.count)
        let backend: any CBv2KVBackend
        let bank: CBv2LayerCacheBank
        if paged {
            let pages = try PagedKVBackend(
                layerKinds: kinds,
                config: .init(
                    capacityBytes: 128 << 20, dtype: .float32, maxPrefillChunk: 256,
                    nominalMaxSequenceLength: 2048, segmentSizeBytes: 1 << 20, layerDTypes: types))
            backend = pages
            bank = CBv2LayerCacheBank(caches: pages.makeLayerCaches())
        } else {
            backend = CBv2ContiguousKVBackend(
                config: .init(bytesCapacity: 128 << 20, kvDType: .float32))
            bank = CBv2LayerCacheBank(layerKinds: kinds)
        }
        return (
            EngineV2(
                model: CBv2SteppableLanguageModelAdapter(m), layerKinds: kinds,
                backend: backend, cacheProvider: bank, sampler: CBv2GreedySampler(),
                schedulerConfig: .init(
                    maxConcurrentRequests: 2, maxBatchedTokensPerStep: 256,
                    prefillChunkSize: 256, soloPrefillStripeTokens: 256, maxWaiting: 4,
                    enablePrefixCache: true),
                admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store,
                completeCheckpointKVDTypes: types), backend
        )
    }

    private func request(_ id: UInt64, salt: String = "flint-fixture") -> CBv2Request {
        .init(
            id: .init(id), promptTokens: (0 ..< 1543).map { ($0 * 7 + 3) % 32 },
            sampling: .init(temperature: 0, seed: 0), maxTokens: 8, cacheSalt: salt,
            prefixCacheReceiptID: .init(id + 1000), prefixCheckpointTargetTokens: 1024)
    }

    @Test(arguments: [false, true])
    func restoreAfterWindowWrapMatchesColdAndRejectsForeignSalt(paged: Bool) async throws {
        let m = try model()
        let scalarCache = m.newCache(parameters: nil)
        var logits = m(
            MLXArray(request(1).promptTokens.map(Int32.init)).reshaped(1, -1), cache: scalarCache)
        var expected = [Int]()
        for _ in 0 ..< 8 {
            let next = argMax(logits[0, -1], axis: -1).item(Int.self)
            expected.append(next)
            logits = m(MLXArray([Int32(next)]).reshaped(1, 1), cache: scalarCache)
        }
        let seed = CompleteCheckpointFixtureStore(segmentBytes: 4093)
        let (cold, coldBackend) = try engine(m, paged: paged, store: seed)
        let coldResult = await cbv2SchedCollect(try cold.submit(request(1)))
        seed.finishPublicationCallbacks(engine: cold)
        await cold.shutdown()
        try #require(coldResult.finishReason == .length)
        #expect(coldResult.tokens.count == 8)
        #expect(coldResult.tokens == expected)
        #expect(coldBackend.bytesReserved == 0)
        let archives = seed.saved.filter { $0.manifest.position == 1024 }
        try #require(!archives.isEmpty)
        // Recreate both the model and engine, and retain only serialized bytes.
        let restored = CompleteCheckpointFixtureStore(archives: archives, segmentBytes: 4093)
        let (warm, warmBackend) = try engine(model(), paged: paged, store: restored)
        #expect(try !restored.stage(engine: warm, request: request(2, salt: "another-tenant")))
        let input = request(3)
        try #require(try restored.stage(engine: warm, request: input))
        let warmResult = await cbv2SchedCollect(try warm.submit(input))
        restored.finishPublicationCallbacks(engine: warm)
        // Shutdown must also release a staged import never consumed by submit.
        try #require(try restored.stage(engine: warm, request: request(4)))
        await warm.shutdown()
        #expect(warmResult.finishReason == .length && warmResult.tokens == coldResult.tokens)
        #expect(warmResult.usage?.prefixCachePrefillTokensSaved == 1024)
        #expect(warmResult.usage?.prefixCacheReplayTokens == 0)
        #expect(warmBackend.bytesReserved == 0 && warm.admissionForTesting.bytesReserved == 0)
    }
}
