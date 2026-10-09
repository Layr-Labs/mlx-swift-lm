import MLX
import Testing

@testable import MLXLMCommon

@Suite("Packed EngineV2 admission derivation", .serialized)
struct CBv2QuantizedEngineAdmissionTests {
    private func kind(window: Int? = nil, dimension: Int = 64, heads: Int = 1, shares: Int? = nil)
        -> CBv2LayerKind
    {
        .init(
            attention: window.map { .slidingWindow($0) } ?? .full,
            sharesKVWithLayer: shares, headDim: dimension, kvHeads: heads, queryHeads: heads * 2)
    }

    private func backend(
        kinds: [CBv2LayerKind]? = nil, quantized: Bool = true,
        types: [DType]? = nil, nativeOwners: Set<Int> = [], capacity: Int = 256 << 10,
        segmentBytes: Int = 64 << 10
    ) throws -> PagedKVBackend {
        try PagedKVBackend(
            layerKinds: kinds ?? [kind()],
            config: .init(
                capacityBytes: capacity, dtype: .float32, maxPrefillChunk: 256,
                nominalMaxSequenceLength: 4096, segmentSizeBytes: segmentBytes,
                layerDTypes: types, quantization: quantized ? .init() : nil,
                nativeLayerIndices: nativeOwners))
    }

    private func engine(
        _ backend: PagedKVBackend, admission: AdmissionV2.Config = .init(watermarkFraction: 0),
        model: CBv2SteppableModel = CBv2SchedScriptedModel()
    ) -> EngineV2 {
        EngineV2(
            model: model, layerKinds: backend.layerKinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            sampler: CBv2GreedySampler(), admissionConfig: admission)
    }

    @Test func sdkDefaultUsesPackedRatesWithoutCallerDerivation() async throws {
        let backend = try backend()
        let engine = EngineV2(
            model: CBv2SchedScriptedModel(), layerKinds: backend.layerKinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            sampler: CBv2GreedySampler())
        let admission = engine.admissionForTesting
        #expect(admission.estimatedBytes(forTokens: 1024) == 80 * 1024)
        #expect(admission.canEverFit(promptTokens: 1024, maxTokens: 16))
        let native = AdmissionV2(
            layerKinds: backend.layerKinds, bytesCapacity: backend.bytesCapacity,
            config: .init(elementBytes: 4), residency: backend.kvResidency)
        #expect(!native.canEverFit(promptTokens: 1024, maxTokens: 16))
        await engine.shutdown()
    }

    @Test func exactPagesCannotConsumeTheRequiredNativeTailMinimum() async throws {
        let backend = try backend()
        let engine = engine(backend)
        let config = try backend.pool.admissionStorageConfig(.init(watermarkFraction: 0))
        let admission = engine.admissionForTesting
        let pages = admission.allocatedBytes(forTokens: 1024)
        #expect(config.minimumRequestTransientBytes > 0)
        engine.updateKVBytesCapacity(pages)
        #expect(admission.admissibleBytesCapacity == pages)
        #expect(!admission.canEverFit(promptTokens: 1024, maxTokens: 0))
        engine.updateKVBytesCapacity(pages + config.minimumRequestTransientBytes)
        #expect(admission.canEverFit(promptTokens: 1024, maxTokens: 0))
        await engine.shutdown()
    }

    @Test func providerPrederivationAndAutomaticDerivationChargeTheSameMinimum() async throws {
        let automaticBackend = try backend()
        let prederivedBackend = try backend()
        var caller = AdmissionV2.Config(watermarkFraction: 0, fixedBytesPerRequest: 123)
        caller.minimumRequestTransientBytes = 4096
        let prederived = try prederivedBackend.pool.admissionStorageConfig(caller)
        let repeated = try prederivedBackend.pool.admissionStorageConfig(prederived)
        #expect(repeated.minimumRequestTransientBytes == prederived.minimumRequestTransientBytes)
        #expect(repeated.layerBytesPerToken == prederived.layerBytesPerToken)
        let automatic = engine(automaticBackend, admission: caller)
        let explicit = engine(prederivedBackend, admission: prederived)
        let pages = 80 * 1024 + 123
        let ceiling = pages + prederived.minimumRequestTransientBytes
        automatic.updateKVBytesCapacity(ceiling)
        explicit.updateKVBytesCapacity(ceiling)
        #expect(automatic.resolvedFixedBytesPerRequest == 123)
        #expect(explicit.resolvedFixedBytesPerRequest == 123)
        #expect(automatic.admissionForTesting.allocatedBytes(forTokens: 1024) == pages)
        #expect(explicit.admissionForTesting.allocatedBytes(forTokens: 1024) == pages)
        #expect(automatic.admissionForTesting.canEverFit(promptTokens: 1024, maxTokens: 0))
        #expect(explicit.admissionForTesting.canEverFit(promptTokens: 1024, maxTokens: 0))
        automatic.updateKVBytesCapacity(ceiling - 1)
        explicit.updateKVBytesCapacity(ceiling - 1)
        #expect(!automatic.admissionForTesting.canEverFit(promptTokens: 1024, maxTokens: 0))
        #expect(!explicit.admissionForTesting.canEverFit(promptTokens: 1024, maxTokens: 0))
        await automatic.shutdown()
        await explicit.shutdown()
    }

    @Test func callerMinimumAssignmentInvalidatesOnlyTheDerivedCredit() throws {
        let backend = try backend(capacity: 8 << 20)
        var caller = AdmissionV2.Config(
            watermarkFraction: 0.125, elementBytes: 4, layerElementBytes: [4],
            fixedBytesPerRequest: 123, auxiliaryBytesPerToken: 17,
            auxiliaryTokenGranularity: 16, auxiliaryTokenAllocationPadding: 5)
        caller.minimumRequestTransientBytes = 4096
        var derived = try backend.pool.admissionStorageConfig(caller)
        let tail = derived.minimumRequestTransientBytes - 4096
        derived.minimumRequestTransientBytes = 23
        #expect(derived.pagedKVTransientFeasibilityBytes == 0)
        let refreshed = try backend.pool.admissionStorageConfig(derived)
        #expect(refreshed.minimumRequestTransientBytes == 23 + tail)
        #expect(
            try backend.pool.admissionStorageConfig(refreshed).minimumRequestTransientBytes
                == refreshed.minimumRequestTransientBytes)
        #expect(refreshed.watermarkFraction == caller.watermarkFraction)
        #expect(refreshed.elementBytes == caller.elementBytes)
        #expect(refreshed.layerElementBytes == caller.layerElementBytes)
        #expect(refreshed.fixedBytesPerRequest == caller.fixedBytesPerRequest)
        #expect(refreshed.auxiliaryBytesPerToken == caller.auxiliaryBytesPerToken)
        #expect(refreshed.auxiliaryTokenGranularity == caller.auxiliaryTokenGranularity)
        #expect(refreshed.auxiliaryTokenAllocationPadding == caller.auxiliaryTokenAllocationPadding)
        let native = try self.backend(quantized: false, capacity: 8 << 20)
        let nativeConfig = try native.pool.admissionStorageConfig(refreshed)
        #expect(nativeConfig.minimumRequestTransientBytes == 23)
        #expect(nativeConfig.layerBytesPerToken == [512])
    }

    @Test func actualMixedOwnerTableKeepsShortWindowsAssistantOwnersAndBorrowersNative()
        async throws
    {
        let kinds = [
            kind(), kind(window: 128), kind(window: 1024, dimension: 256, heads: 2),
            kind(dimension: 512, heads: 2), kind(dimension: 512, heads: 2, shares: 3),
        ]
        let nativePageBytes =
            try #require(kinds[3].kvGeometry?.bytesPerToken(elementBytes: DType.float32.size))
            * CBv2PagedDefaults.pageSize
        let backend = try backend(
            kinds: kinds, types: [.float32, .bfloat16, .float16, .float32, .float32],
            nativeOwners: [3], capacity: 8 << 20, segmentBytes: 2 * nativePageBytes)
        #expect(backend.pool.config.segmentSizeBytes == 2 * nativePageBytes)
        let config = try backend.pool.admissionStorageConfig(.init())
        #expect(config.layerBytesPerToken == [80, 256, 640, 8192, 0])
        let engine = engine(backend)
        #expect(
            engine.admissionForTesting.estimatedBytes(forTokens: 2048)
                == 80 * 2048 + 256 * 128 + 640 * 1024 + 8192 * 2048)
        await engine.shutdown()
    }

    @Test func nativeEngineRetainsCallerPhysicalRateAndMinimum() async throws {
        let backend = try backend(quantized: false)
        var caller = AdmissionV2.Config(watermarkFraction: 0)
        caller.layerBytesPerToken = [1024]
        caller.minimumRequestTransientBytes = 91
        let engine = engine(backend, admission: caller)
        #expect(engine.admissionForTesting.estimatedBytes(forTokens: 16) == 1024 * 16)
        engine.updateKVBytesCapacity(1024 * 16)
        #expect(!engine.admissionForTesting.canEverFit(promptTokens: 16, maxTokens: 0))
        engine.updateKVBytesCapacity(1024 * 16 + 91)
        #expect(engine.admissionForTesting.canEverFit(promptTokens: 16, maxTokens: 0))
        await engine.shutdown()
    }

    @Test func recurrentGenerationsRemainChargedAndDerivationOverflowFailsClosed() async throws {
        let model = QuantizedAdmissionRecurrentTarget()
        let policy = try #require(Memory.allocationFootprintPolicy())
        let state = try #require(model.recurrentStateSpec)
        let expected =
            try state.allocationBytesPerGeneration(policy: policy)
            * CBv2RecurrentStateSpec.maximumLiveGenerations
        let ordinaryBackend = try backend()
        let ordinary = engine(ordinaryBackend, model: model)
        #expect(ordinary.resolvedFixedBytesPerRequest == expected)
        #expect(ordinary.admissionForTesting.estimatedBytes(forTokens: 16) == 80 * 16 + expected)
        var overflowing = AdmissionV2.Config(watermarkFraction: 0)
        overflowing.minimumRequestTransientBytes = Int.max
        let refusedBackend = try backend()
        #expect(throws: (any Error).self) {
            try refusedBackend.pool.admissionStorageConfig(overflowing)
        }
        let refused = engine(refusedBackend, admission: overflowing, model: model)
        #expect(refused.resolvedFixedBytesPerRequest == expected)
        #expect(!refused.admissionForTesting.canEverFit(promptTokens: 16, maxTokens: 1))
        await ordinary.shutdown()
        await refused.shutdown()
    }
}

private final class QuantizedAdmissionRecurrentTarget: CBv2RecurrentSteppableModel {
    var cbv2Capabilities: CBv2ModelCapabilities {
        .init(
            supportsPrefixReuse: false, supportsPagedKV: true,
            supportsCompiledDecode: false, supportsPackedPrefill: false, supportsMTP: false)
    }
    let recurrentStateSpec: CBv2RecurrentStateSpec? = .init(layers: [
        .init(
            modelLayerIndex: 0, convShape: [1, 8], convDType: .float32,
            ssmShape: [1, 16], ssmDType: .float32)
    ])
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        preconditionFailure("admission fixture must not forward")
    }
    func forward(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray { preconditionFailure("admission fixture must not forward") }
}
