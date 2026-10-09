import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Packed KV legacy and complete prefix capabilities", .serialized)
struct CBv2QuantizedPrefixCapabilityTests {
    private var chunk: Int { max(32, CBv2AttentionV1.queryBlockSize) }
    private var promptCount: Int { max(1_024, chunk * 8) + 1 }
    /// Real paged attention with generated public inputs. Zero queries make
    /// each result the mean of visible values, with a stable greedy margin.
    private final class Model: CBv2SteppableModel,
        CBv2HistoricalAttentionCheckpointProviding, CBv2CompleteCheckpointKVTypeProviding
    {
        let cbv2SupportsHistoricalAttentionCheckpoint = true
        let cbv2CompleteCheckpointKVDTypes: [DType]? = [.float32]
        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
            let batch = tokens.dim(0)
            let count = tokens.dim(1)
            let values = broadcast(
                tokens.asType(.float32).reshaped([batch, 1, count, 1]) / Float(32),
                to: [batch, 1, count, 64])
            let attended = caches[0].updateAndAttend(
                queries: MLXArray.zeros([batch, 2, count, 64]), keys: values, values: values,
                scale: 0.125, sinks: nil)
            let hidden = mean(attended, axes: [1, 3]).reshaped([batch, count, 1])
            let target = MLX.round(hidden * Float(128)).asType(.int32) % 31
            return MLX.where(MLXArray(Int32(0) ..< Int32(32)) .== target, Float(10), Float(-10))
        }
    }

    private func backend(nativeOwners: Set<Int> = [], window: Int? = nil) throws -> PagedKVBackend {
        let kind = CBv2LayerKind(
            attention: window.map { .slidingWindow($0) } ?? .full,
            headDim: 64, kvHeads: 1, queryHeads: 2)
        return try PagedKVBackend(
            layerKinds: [kind],
            config: .init(
                capacityBytes: 64 << 20, dtype: .float32, maxPrefillChunk: chunk,
                nominalMaxSequenceLength: promptCount + chunk, segmentSizeBytes: 64 << 10,
                layerDTypes: [.float32], quantization: .init(), nativeLayerIndices: nativeOwners))
    }

    private func engine(
        backend: PagedKVBackend, legacy: PrefixCacheV2,
        complete: CompleteCheckpointFixtureStore? = nil
    ) -> EngineV2 {
        EngineV2(
            model: Model(), layerKinds: backend.layerKinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: chunk,
                prefillChunkSize: chunk, soloPrefillStripeTokens: chunk,
                maxWaiting: 4, enablePrefixCache: true),
            admissionConfig: .init(watermarkFraction: 0), prefixCache: legacy,
            completePrefixCache: complete)
    }

    private func request(_ id: UInt64) -> CBv2Request {
        .init(
            id: .init(id), promptTokens: (0 ..< promptCount).map { ($0 * 7) % 29 },
            sampling: .init(temperature: 0, seed: 0), maxTokens: 2, cacheSalt: "tenant",
            prefixCacheReceiptID: .init(id + 1_000))
    }

    @Test func capabilityUsesActualOwnersAndPreservesNativeExemptions() throws {
        let packed = try backend()
        #expect(packed.usesQuantizedStorage && packed.prefixReuseBackend == .unknown)
        #expect(!packed.supportsOrdinaryDecodeChaining)
        let native = try backend(nativeOwners: [0])
        #expect(!native.usesQuantizedStorage && native.prefixReuseBackend == .pagedFP16)
        #expect(native.supportsOrdinaryDecodeChaining)
        let recentOnlyWindow = try backend(window: 128)
        #expect(!recentOnlyWindow.usesQuantizedStorage)
        #expect(recentOnlyWindow.prefixReuseBackend == .pagedFP16)
        #expect(recentOnlyWindow.supportsOrdinaryDecodeChaining)
    }

    @Test func genericEngineDisablesLegacyLookupAndNativeDonations() async throws {
        let pages = try backend()
        let legacy = PrefixCacheV2(config: .init(blockSize: 16, maxBytes: 1 << 20))
        let engine = engine(backend: pages, legacy: legacy)
        #expect(!engine.prefixReuseCapability.isSupported)
        let first = await cbv2SchedCollect(try engine.submit(request(1)))
        let second = await cbv2SchedCollect(try engine.submit(request(2)))
        await engine.shutdown()
        #expect(first.finishReason == .length && second.finishReason == .length)
        #expect(first.tokens == second.tokens)
        #expect(second.usage?.prefixCachePrefillTokensSaved == 0)
        #expect(
            legacy.stats()
                == .init(hits: 0, misses: 0, tokensSaved: 0, entryCount: 0, bytesInUse: 0))
        #expect(pages.bytesReserved == 0 && pages.bytesInUse == 0 && pages.bytesWired == 0)
        #expect(engine.admissionForTesting.bytesReserved == 0)
    }

    @Test func packedCompleteFramesStillRestoreThroughGenericEngine() async throws {
        let seed = CompleteCheckpointFixtureStore(segmentBytes: 257)
        let coldPages = try backend()
        let coldLegacy = PrefixCacheV2(config: .init(blockSize: 16, maxBytes: 1 << 20))
        let coldEngine = engine(backend: coldPages, legacy: coldLegacy, complete: seed)
        #expect(!coldEngine.prefixReuseCapability.isSupported)
        #expect(
            coldEngine.completeCheckpointCodec != nil && coldEngine.completePrefixCache === seed)
        #expect(coldEngine.completeCheckpointCodec?.historicalLayout != nil)
        let cold = await cbv2SchedCollect(try coldEngine.submit(request(10)))
        seed.finishPublicationCallbacks(engine: coldEngine)
        await coldEngine.shutdown()
        #expect(cold.finishReason == .length)
        #expect(!seed.saved.isEmpty)
        #expect(coldLegacy.stats().entryCount == 0 && coldLegacy.stats().bytesInUse == 0)

        let reopened = CompleteCheckpointFixtureStore(archives: seed.saved, segmentBytes: 257)
        let warmPages = try backend()
        let warmLegacy = PrefixCacheV2(config: .init(blockSize: 16, maxBytes: 1 << 20))
        let warmEngine = engine(backend: warmPages, legacy: warmLegacy, complete: reopened)
        let resumed = request(11)
        #expect(try reopened.stage(engine: warmEngine, request: resumed))
        let warm = await cbv2SchedCollect(try warmEngine.submit(resumed))
        await warmEngine.shutdown()
        #expect(warm.finishReason == .length && warm.tokens == cold.tokens)
        #expect((warm.usage?.prefixCachePrefillTokensSaved ?? 0) > 0)
        #expect(warmLegacy.stats().entryCount == 0 && warmLegacy.stats().bytesInUse == 0)
        #expect(
            warmPages.bytesReserved == 0 && warmPages.bytesInUse == 0 && warmPages.bytesWired == 0)
        #expect(warmEngine.admissionForTesting.bytesReserved == 0)
    }
}
