import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2 quantized storage admission", .serialized)
struct CBv2QuantizedAdmissionTests {
    private var kinds: [CBv2LayerKind] {
        [
            .init(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2),
            .init(attention: .slidingWindow(16), headDim: 64, kvHeads: 1, queryHeads: 2),
            .init(attention: .full, sharesKVWithLayer: 0,
                  headDim: 64, kvHeads: 1, queryHeads: 2)
        ]
    }

    private func ledger(capacity: Int, rates: [Int] = [80, 256, 80]) -> AdmissionV2 {
        AdmissionV2(layerKinds: kinds, bytesCapacity: capacity, config: .init(
            watermarkFraction: 0, elementBytes: 2, layerElementBytes: [2, 2, 2],
            fixedBytesPerRequest: 100, auxiliaryBytesPerToken: 2,
            auxiliaryTokenGranularity: 16, auxiliaryTokenAllocationPadding: 3,
            layerBytesPerToken: rates), residency: CBv2PagedKVResidency(config: .init(
                pageSize: 16, capacityBytes: capacity, maxPrefillChunk: 16,
                maxBufferLength: 1 << 20)))
    }

    @Test("packed groups retain native windows and exact page and assistant overhead")
    func chargesStorageNotIntegerElementWidths() throws {
        // One 64-wide K/V owner: two 32-byte code rows, each with FP32
        // scale+offset (8 bytes), gives 80 bytes/token. Window stays 256.
        let admission = ledger(capacity: 12_228)
        #expect(admission.fullKVBytesPerToken == 80)
        #expect(admission.estimatedBytes(forTokens: 33) == 6_932)
        #expect(admission.fixedWindowBytesShortfall(afterReservingTokens: 33) == 5_296)
        #expect(admission.allocatedBytes(forTokens: 33) == 12_228)
        #expect(admission.maximumKVRequestOverheadBytes == 9_392)
        #expect(admission.canEverFit(promptTokens: 32, maxTokens: 1))
        #expect(!admission.canEverFit(promptTokens: 48, maxTokens: 1))
        let id = CBv2RequestID(1)
        try admission.reserve(id: id, additionalTokens: 33)
        #expect(admission.bytesReserved == 12_228)
        try admission.reserve(id: id, additionalTokens: 1) // same physical pages
        #expect(admission.bytesReserved == 12_228)
        #expect(throws: CBv2KVError.self) {
            try admission.reserve(id: id, additionalTokens: 15)
        }
        #expect(admission.bytesReserved == 12_228)
        admission.updateBytesCapacity(13_540)
        try admission.reserve(id: id, additionalTokens: 15)
        #expect(admission.bytesReserved == 13_540)
        admission.unreserve(id: id, tokens: 16)
        #expect(admission.bytesReserved == 12_228)
        admission.releaseAll(id: id)
        #expect(admission.bytesReserved == 0)
    }

    @Test("deadline projection and conservative chained-decode guard use packed byte rates")
    func projectionsMatchLiveReservations() throws {
        let admission = ledger(capacity: 12_228)
        let first = CBv2ProjectedCapacityReservation(
            id: .init(1), additionalTokens: 33, additionalBytes: 0)
        let nextPage = CBv2ProjectedCapacityReservation(
            id: .init(1), additionalTokens: 16, additionalBytes: 0)
        #expect(admission.canGuarantee(projectedOperations: [.reserve(first)]))
        #expect(!admission.canGuarantee(projectedOperations: [.reserve(first), .reserve(nextPage)]))
        #expect(admission.hasHeadroom(additionalTokens: 1))
        try admission.reserve(id: .init(1), additionalTokens: 33)
        #expect(!admission.hasHeadroom(additionalTokens: 1))
        #expect(admission.canGuarantee(projectedOperations: [.release(.init(1)), .reserve(first)]))
        admission.releaseAll(id: .init(1))
    }

    @Test("explicit native reconstruction scratch remains additive to packed storage")
    func scratchNeverOffsetsTargetBacking() throws {
        let admission = ledger(capacity: 16_324)
        let physical = admission.bindBackendPhysicalFloor(initialBytes: 0)
        defer { physical.close() }
        try admission.reserve(id: .init(1), additionalTokens: 33)
        // 16 native BF16 K/V rows cost 4096 bytes, not 16*80 packed bytes.
        let scratch = try admission.reserveTransient(bytes: 4_096)
        #expect(admission.bytesReserved == 16_324)
        #expect(throws: CBv2KVError.self) { try admission.reserveTransient(bytes: 1) }
        scratch.release()
        #expect(admission.bytesReserved == 12_228)
        admission.releaseAll(id: .init(1))
    }

    @Test("malformed exact rates never fall back to native or free storage")
    func invalidRatesFailClosed() {
        for rates in [[80], [-1, 256, 80], [0, 256, 80], [Int.max, 256, 80]] {
            let admission = ledger(capacity: 1 << 20, rates: rates)
            #expect(!admission.canEverFit(promptTokens: 16, maxTokens: 16))
            #expect(!admission.hasHeadroom(additionalTokens: 1))
        }
    }

    @Test("engine derives admission rates from the actual packed and native groups")
    func engineOverridesInaccurateCallerRate() async throws {
        let quantization = PagedKVQuantizationConfig(rotationBlockSize: 64)
        let backend = try PagedKVBackend(layerKinds: kinds, config: .init(
            capacityBytes: 1 << 20, maxPrefillChunk: 16,
            segmentSizeBytes: 32_768, layerDTypes: [.bfloat16, .bfloat16, .bfloat16],
            quantization: quantization))
        let engine = EngineV2(
            model: CBv2SchedScriptedModel(), layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            sampler: CBv2GreedySampler(), admissionConfig: .init(
                watermarkFraction: 0, layerBytesPerToken: [1, 1, 1]))
        #expect(engine.admissionForTesting.fullKVBytesPerToken == 80)
        #expect(engine.admissionForTesting.allocatedBytes(forTokens: 33) > 12_032,
                "native attention workspace must be prepaid in addition to packed target KV")
        #expect(engine.resolvedKVRequestOverheadBytes == 9_392)
        await engine.shutdown()
    }

    @Test("quantized engines never read legacy native snapshot prefix entries")
    func snapshotPrefixCacheIsDisabled() async throws {
        let full = [kinds[0]]
        let backend = try PagedKVBackend(layerKinds: full, config: .init(
            capacityBytes: 1 << 20, maxPrefillChunk: 16,
            segmentSizeBytes: 32_768, layerDTypes: [.bfloat16],
            quantization: .init(rotationBlockSize: 64)))
        let cache = CBv2SchedScriptedPrefixCache(matched: 16)
        let engine = EngineV2(
            model: CBv2SchedScriptedModel(), layerKinds: full, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            sampler: CBv2GreedySampler(),
            schedulerConfig: .init(prefillChunkSize: 16, enablePrefixCache: true),
            prefixCache: cache)
        let request = CBv2SchedFixtures.request(prompt: Array(repeating: 1, count: 32), maxTokens: 1)
        _ = try engine.submit(request)
        #expect(cache.lookups == 0)
        #expect(cache.endAdoptions == 0)
        engine.cancel(request.id)
        await engine.shutdown()
    }

    @Test("workspace covers image block snapping and solo stripes beyond the plain prefill chunk")
    func oversizedPrefillShapesUseTheSchedulerCeiling() async throws {
        let full = [kinds[0]]
        for stripe: Int? in [nil, 4_096] {
            let backend = try PagedKVBackend(layerKinds: full, config: .init(
                capacityBytes: 1 << 30, maxPrefillChunk: 512, maxBufferLength: 1 << 30,
                segmentSizeBytes: 32_768, layerDTypes: [.bfloat16],
                quantization: .init(rotationBlockSize: 64)))
            let engine = EngineV2(
                model: CBv2SchedScriptedModel(), layerKinds: full, backend: backend,
                cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
                sampler: CBv2GreedySampler(), schedulerConfig: .init(
                    maxConcurrentRequests: 1, maxBatchedTokensPerStep: 2_048,
                    prefillChunkSize: 512, soloPrefillStripeTokens: stripe))
            let queries = max(2_048, stripe ?? 0)
            let actual = try PagedQuantizedAttentionWorkspace.reservationBytes(
                queryCount: queries, blockSize: 8, queryHeads: 2, headDim: 64,
                pageSize: 16, maxAttendLength: 8_192, maximumSegmentCount: 513,
                broadcastTopology: true, nativeOutputBytes: queries * 2 * 64 * 4)
            let decode = try PagedQuantizedAttentionWorkspace.reservationBytes(
                queryCount: 1, blockSize: 1, queryHeads: 2, headDim: 64,
                pageSize: 16, maxAttendLength: 8_192, maximumSegmentCount: 18,
                nativeOutputBytes: 2 * 64 * 4)
            let prepaid = try #require(engine.routingWorkspaceBytes(forTokens: 8_192))
            #expect(prepaid >= actual + decode,
                "the final oversized prefill may overlap one pure-decode successor")
            await engine.shutdown()
        }
    }
}
