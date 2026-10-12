import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Runtime KV packed storage and original recent rows", .serialized)
struct CBv2RuntimeKVStorageTests {
    private func fixture(
        window: Int? = nil, dtype: DType = .float32,
        dimension: Int = 64, heads: Int = 1,
        capacity: Int = 8 << 20,
        quantization: PagedKVQuantizationConfig? = .init(), nativeOwners: Set<Int> = []
    ) throws -> (PagedKVBackend, CBv2LayerKind) {
        let kind = CBv2LayerKind(
            attention: window.map { .slidingWindow($0) } ?? .full,
            headDim: dimension, kvHeads: heads, queryHeads: heads * 2)
        let backend = try PagedKVBackend(
            layerKinds: [kind],
            config: .init(
                capacityBytes: capacity, dtype: dtype, maxPrefillChunk: 256,
                nominalMaxSequenceLength: 4096, maxBufferLength: 16 << 20,
                segmentSizeBytes: 64 << 10, quantization: quantization,
                nativeLayerIndices: nativeOwners))
        return (backend, kind)
    }
    private func tensor(
        start: Int, count: Int, heads: Int = 1, dimension: Int = 64,
        dtype: DType = .float32
    ) -> MLXArray {
        let values = (0 ..< heads * count * dimension).map { index in
            sin(Float(index + start * dimension) * 0.031) + Float(index % 13) * 0.013
        }
        return MLXArray(values, [heads, count, dimension]).asType(dtype)
    }
    private func raw(_ array: MLXArray) -> Data { array.asData(access: .copy).data }

    private func loop(_ backend: PagedKVBackend) -> EngineLoopV2 {
        EngineLoopV2(
            model: CBv2SchedScriptedModel(), layerKinds: backend.layerKinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            sampler: CBv2GreedySampler(), detokenizerFactory: CBv2SchedScriptedDetokFactory(),
            scheduler: SchedulerV2(config: .init()), capacity: nil,
            config: .init(), gauges: CBv2EngineGauges(kvBytesCapacity: backend.bytesCapacity))
    }

    @Test(arguments: [nil, 64, 128] as [Int?])
    func nativeOwnersSkipPackedStepRetirement(window: Int?) throws {
        var reference: [Data] = []
        for requested in [false, true] {
            let (backend, kind) = try fixture(
                window: window, quantization: requested ? .init() : nil,
                nativeOwners: window == nil ? [0] : [])
            let loop = loop(backend)
            #expect((backend.pool.config.quantization != nil) == requested)
            #expect(!backend.usesQuantizedStorage && backend.supportsOrdinaryDecodeChaining)
            // This is the gate shared by finalization, teacher scoring and failure cleanup.
            #expect(loop.quantizedStoragePool == nil)
            let state = try backend.makeSequenceState(
                layerKinds: [kind], promptLength: 256, maxLength: 512)
            defer {
                StreamOrDevice.default.stream.synchronize()
                (loop.cacheProvider as? CBv2CompositionInvalidating)?.releaseBoundRows()
                backend.release(state)
            }
            let row = try #require(state[0] as? PagedSequenceKV)
            let cache = try #require(
                loop.cacheProvider.layerCaches(rowStates: [state]).first as? PagedLayerCache)
            var outputs: [Data] = []
            for start in [0, 256] {
                let count = start == 0 ? 256 : 1
                let input = tensor(start: start, count: count).expandedDimensions(axis: 0)
                let output = cache.updateAndAttend(
                    queries: MLXArray.zeros([1, 2, count, 64]), keys: input, values: input,
                    scale: 0.125, sinks: nil)
                try withError { fault in
                    eval([output] + cache.innerState())
                    try fault.check()
                }
                try backend.pool.writeValidation.check()
                #expect(loop.quantizedStoragePool == nil)
                #expect(backend.pool.pendingQuantizedScratch.isEmpty)
                #expect(backend.pool.quantizedAttentionCaches.isEmpty)
                #expect(backend.pool.quantizedRecentRows.isEmpty)
                #expect(row.nativeRecentOwner == nil && row.retiredNativeRecentOwners.isEmpty)
                try loop.quantizedStoragePool?.finishQuantizedStorageStep()
                outputs.append(raw(output))
            }
            #expect(row.absoluteOffset == 257)
            if requested {
                #expect(outputs == reference, "A fully native requested profile keeps native math")
            } else {
                reference = outputs
            }
        }
    }

    @Test(arguments: [false, true])
    func packedOwnersKeepStepRetirementInMixedLayouts(mixed: Bool) throws {
        var kinds = [CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2)]
        if mixed {
            kinds.append(
                .init(attention: .slidingWindow(128), headDim: 64, kvHeads: 1, queryHeads: 2))
        }
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 8 << 20, dtype: .float32, maxPrefillChunk: 256,
                nominalMaxSequenceLength: 4096, maxBufferLength: 16 << 20,
                segmentSizeBytes: 64 << 10, quantization: .init()))
        let loop = loop(backend)
        #expect(backend.usesQuantizedStorage && !backend.supportsOrdinaryDecodeChaining)
        #expect(loop.quantizedStoragePool === backend.pool)
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 256, maxLength: 512)
        defer {
            StreamOrDevice.default.stream.synchronize()
            backend.pool.discardQuantizedStorageStepAfterSynchronization()
            (loop.cacheProvider as? CBv2CompositionInvalidating)?.releaseBoundRows()
            backend.release(state)
        }
        let row = try #require(state[0] as? PagedSequenceKV)
        let caches = try #require(
            loop.cacheProvider.layerCaches(rowStates: [state]) as? [PagedLayerCache])
        for start in [0, 256] {
            let count = start == 0 ? 256 : 1
            let input = tensor(start: start, count: count).expandedDimensions(axis: 0)
            let outputs = caches.map { cache in
                cache.updateAndAttend(
                    queries: MLXArray.zeros([1, 2, count, 64]), keys: input, values: input,
                    scale: 0.125, sinks: nil)
            }
            try backend.pool.writeValidation.check()
            let leases = backend.pool.pendingQuantizedScratch
            #expect(!leases.isEmpty && leases.allSatisfy { !$0.evaluationTargets.isEmpty })
            try withError { fault in
                eval(outputs + caches.flatMap { $0.innerState() })
                try fault.check()
            }
            try #require(loop.quantizedStoragePool).finishQuantizedStorageStep()
            #expect(backend.pool.pendingQuantizedScratch.isEmpty)
            #expect(leases.allSatisfy { $0.evaluationTargets.isEmpty })
            #expect(row.nativeRecentStart == start + count - 128)
            #expect(row.nativeRecentKeys?.dim(1) == 128)
            #expect(row.retiredNativeRecentOwners.isEmpty)
            #expect(loop.quantizedStoragePool === backend.pool)
        }
    }

    @Test func malformedGeometryAndRotationOverflowRefuseInsteadOfTrapping() throws {
        let format = PagedKVQuantizationConfig()
        #expect(format.resolvedRotationBlockSize(headDim: Int.min) == 0)
        #expect(throws: CBv2KVError.self) { try format.validate(headDim: Int.min) }
        #expect(throws: CBv2KVError.self) {
            try PagedKVQuantizationReference.encode(
                Array(repeating: Float.greatestFiniteMagnitude, count: 64), config: format,
                isKey: true)
        }
        let input = Array(repeating: Float(1.25), count: 64)
        let restored = try PagedKVQuantizationReference.roundTrip(
            input, config: format, isKey: false)
        #expect(restored == input)
    }

    @Test func exactByteRatesPreserveWindowAndAssistantOwners() throws {
        let format = PagedKVQuantizationConfig()
        #expect(try format.bytesPerToken(kvHeads: 2, headDim: 256) == 640)
        #expect(try format.bytesPerToken(kvHeads: 2, headDim: 192, valueHeadDim: 128) == 400)
        #expect(format.resolvedRotationBlockSize(headDim: 192) == 64)
        #expect(format.resolvedRotationBlockSize(headDim: 80) == 16)
        #expect(format.resolvedRotationBlockSize(headDim: 160) == 32)
        #expect(
            try JSONDecoder().decode(
                PagedKVQuantizationConfig.self,
                from: JSONEncoder().encode(format)) == format)
        let kinds = [
            CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2),
            CBv2LayerKind(attention: .slidingWindow(128), headDim: 64, kvHeads: 1, queryHeads: 2),
            CBv2LayerKind(attention: .slidingWindow(1024), headDim: 256, kvHeads: 2, queryHeads: 4),
            CBv2LayerKind(attention: .full, headDim: 512, kvHeads: 2, queryHeads: 4),
        ]
        let pool = try PagedKVPool(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 8 << 20, segmentSizeBytes: 256 << 10,
                quantization: format, nativeLayerIndices: [3]))
        #expect(pool.usesQuantization(layerIndex: 0))
        #expect(!pool.usesQuantization(layerIndex: 1))
        #expect(pool.usesQuantization(layerIndex: 2))
        #expect(!pool.usesQuantization(layerIndex: 3))
        let config = try pool.admissionStorageConfig(.init(watermarkFraction: 0))
        #expect(config.layerBytesPerToken == [80, 256, 640, 4096])
        #expect(config.minimumRequestTransientBytes > 0)
        #expect(throws: CBv2KVError.self) {
            try PagedKVPool(
                layerKinds: kinds,
                config: .init(
                    capacityBytes: 8 << 20, segmentSizeBytes: 64 << 10,
                    quantization: format, nativeLayerIndices: [4]))
        }
        #expect(throws: CBv2KVError.self) {
            try PagedKVPool(
                layerKinds: [
                    CBv2LayerKind(
                        attention: .full,
                        headDim: 192, valueHeadDim: 128, kvHeads: 2, queryHeads: 4)
                ],
                config: .init(
                    capacityBytes: 8 << 20, segmentSizeBytes: 64 << 10, quantization: format))
        }
        let shared = [
            kinds[0],
            CBv2LayerKind(
                attention: .full, sharesKVWithLayer: 0,
                headDim: 64, kvHeads: 1, queryHeads: 2),
        ]
        let nativeShared = try PagedKVPool(
            layerKinds: shared,
            config: .init(
                capacityBytes: 8 << 20, segmentSizeBytes: 64 << 10,
                quantization: format, nativeLayerIndices: [0]))
        #expect(!nativeShared.usesQuantization(layerIndex: 0))
        #expect(!nativeShared.usesQuantization(layerIndex: 1))
        #expect(try nativeShared.admissionStorageConfig(.init()).layerBytesPerToken == [256, 0])
        #expect(throws: CBv2KVError.self) {
            try PagedKVPool(
                layerKinds: shared,
                config: .init(
                    capacityBytes: 8 << 20, segmentSizeBytes: 64 << 10,
                    quantization: format, nativeLayerIndices: [1]))
        }
    }

    @Test func wholePromptStaysNativeUntilFenceThenOnlyRecent128OwnsNativeBytes() throws {
        let (backend, kind) = try fixture(dimension: 256, heads: 2)
        let state = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 256, maxLength: 512)
        defer { backend.release(state) }
        let row = try #require(state[0] as? PagedSequenceKV)
        let keys = tensor(start: 0, count: 256, heads: 2, dimension: 256)
        let values = keys * 0.7 + 0.11
        row.write(keys: keys, values: values)
        let pending = row.snapshot()
        #expect(raw(pending.keys) == raw(keys.expandedDimensions(axis: 0)))
        #expect(raw(pending.values) == raw(values.expandedDimensions(axis: 0)))
        let duringPrompt = try #require(backend.pool.segmentStorageSnapshot)
        #expect(backend.pool.nativeRecentBytesInUse > 0)
        #expect(
            backend.bytesInUse > duringPrompt.committedBytes,
            "A live native prompt band can exceed page backing without belonging to it")
        #expect(duringPrompt.livePageBytes <= duringPrompt.reservedPageBytes)
        #expect(
            duringPrompt.livePageBytes + duringPrompt.poisonBytes <= duringPrompt.committedBytes)
        #expect(
            duringPrompt.reservedPageBytes + duringPrompt.poisonBytes
                + duringPrompt.slackBytes + duringPrompt.allocatorPaddingBytes
                == duringPrompt.committedBytes)
        #expect(
            backend.bytesInUse == duringPrompt.livePageBytes + backend.pool.nativeRecentBytesInUse)
        eval(row.nativeRecentEvaluationRoots)
        try backend.pool.finishQuantizedRecentStep()
        #expect(row.nativeRecentStart == 128)
        #expect(row.nativeRecentKeys?.shape == [2, 128, 256])
        #expect(row.retiredNativeRecentOwners.isEmpty)
        let afterCompaction = try #require(backend.pool.segmentStorageSnapshot)
        #expect(afterCompaction.livePageBytes == duringPrompt.livePageBytes)
        #expect(afterCompaction.committedBytes == duringPrompt.committedBytes)
        #expect(
            backend.bytesInUse == afterCompaction.livePageBytes
                + backend.pool.nativeRecentBytesInUse)
        let current = row.snapshot()
        #expect(
            raw(current.keys[0..., 0..., 128..., 0...])
                == raw(keys[0..., 128..., 0...].expandedDimensions(axis: 0)))
        #expect(
            raw(current.values[0..., 0..., 128..., 0...])
                == raw(values[0..., 128..., 0...].expandedDimensions(axis: 0)))
        #expect(
            raw(current.keys[0..., 0..., ..<128, 0...])
                != raw(keys[0..., ..<128, 0...].expandedDimensions(axis: 0)))
        for group in backend.pool.groups.values {
            for segment in group.segments.values {
                #expect(segment.storage.dtype == .uint8)
                #expect(segment.storage.nbytes == segment.pages.count * group.pageBytes)
            }
        }
        let nativeAllBytes = keys.nbytes + values.nbytes
        #expect(backend.pool.nativeRecentBytesInUse < nativeAllBytes)
        #expect(backend.pool.bytesInUse < nativeAllBytes)
    }

    @Test func rejectionKeepsFormerConfirmedRecentRowsBitExact() throws {
        let (backend, kind) = try fixture()
        let state = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 256, maxLength: 512)
        defer { backend.release(state) }
        let row = try #require(state[0] as? PagedSequenceKV)
        let keys = tensor(start: 0, count: 256)
        row.write(keys: keys, values: keys)
        eval(row.nativeRecentEvaluationRoots)
        try backend.pool.finishQuantizedRecentStep()
        row.beginSpeculativeWrite()
        let pending = tensor(start: 256, count: 4)
        row.write(keys: pending, values: pending)
        eval(row.nativeRecentEvaluationRoots)
        #expect(row.nativeRecentStart == 128)
        #expect(row.nativeRecentKeys?.dim(1) == 132)
        // A step fence must not discard native history before this transaction commits.
        try backend.pool.finishQuantizedRecentStep()
        #expect(row.nativeRecentStart == 128)
        row.rollback(3)
        row.commitSpeculativeWrite()
        try backend.pool.finishQuantizedRecentStep()
        let expected = concatenated([keys[0..., 129..., 0...], pending[0..., ..<1, 0...]], axis: 1)
        #expect(row.absoluteOffset == 257)
        #expect(row.nativeRecentStart == 129)
        #expect(raw(try #require(row.nativeRecentKeys)) == raw(expected))
        #expect(
            raw(row.snapshot().keys[0..., 0..., 129..., 0...])
                == raw(expected.expandedDimensions(axis: 0)))
    }

    @Test func windowWrapKeepsRecentNativeAndRetiresEveryOwner() throws {
        let (backend, kind) = try fixture(window: 256)
        let state = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 1024)
        let row = try #require(state[0] as? PagedSequenceKV)
        for step in 0 ..< 4 {
            let input = tensor(start: step * 256, count: 256)
            row.write(keys: input, values: input)
            eval(row.nativeRecentEvaluationRoots)
            try backend.pool.finishQuantizedRecentStep()
            #expect(row.nativeRecentStart == (step + 1) * 256 - 128)
            #expect(raw(try #require(row.nativeRecentKeys)) == raw(input[0..., 128..., 0...]))
            #expect(row.retiredNativeRecentOwners.isEmpty)
        }
        #expect(row.retainedCount == 256)
        #expect(row.snapshot().keys.shape == [1, 1, 256, 64])
        backend.release(state)
        #expect(backend.pool.nativeRecentBytesInUse == 0)
        #expect(backend.pool.bytesInUse == 0)
        #expect(backend.pool.bytesReserved == 0)
    }

    @Test func partialCompactionFailureKeepsItsChargeUntilRowRetirement() throws {
        struct FailedAfterFirstCopy: Error {}
        let (backend, kind) = try fixture()
        let state = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 512)
        let row = try #require(state[0] as? PagedSequenceKV)
        let admission = AdmissionV2(
            layerKinds: [kind], bytesCapacity: 8 << 20,
            config: .init(watermarkFraction: 0))
        backend.pool.memoryAdmission = admission
        let input = tensor(start: 0, count: 256)
        row.write(keys: input, values: input)
        eval(row.nativeRecentEvaluationRoots)
        let before = admission.transientBytesReserved
        #expect(before > 0)
        backend.pool.quantizedRecentEvaluate = { roots in
            eval([roots[0]])
            throw FailedAfterFirstCopy()
        }
        #expect(throws: FailedAfterFirstCopy.self) { try backend.pool.finishQuantizedRecentStep() }
        #expect(row.nativeRecentStart == 0)
        #expect(row.retiredNativeRecentOwners.count == 1)
        #expect(admission.transientBytesReserved > before)
        backend.release(state)
        #expect(admission.transientBytesReserved == 0)
        #expect(backend.pool.nativeRecentBytesInUse == 0)
    }

    @Test func nativeTailAllocationRefusesBeforeWritingPackedBytes() throws {
        let (backend, kind) = try fixture()
        let state = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 512)
        defer { backend.release(state) }
        let row = try #require(state[0] as? PagedSequenceKV)
        let admission = AdmissionV2(
            layerKinds: [kind], bytesCapacity: 1024,
            config: .init(watermarkFraction: 0))
        backend.pool.memoryAdmission = admission
        let input = tensor(start: 0, count: 256)
        row.write(keys: input, values: input)
        #expect(backend.pool.writeValidation.isFaulted)
        #expect(row.absoluteOffset == 0)
        #expect(row.nativeRecentOwner == nil)
        #expect(admission.transientBytesReserved == 0)
        #expect(throws: CBv2KVError.self) { try backend.pool.writeValidation.check() }
    }

    @Test func legacyTensorPrefixRefusesBeforeMutatingAQuantizedPool() throws {
        let (backend, kind) = try fixture()
        let kinds = [kind]
        let capability = CBv2PrefixReuseCapability.derive(layerKinds: kinds, backend: .pagedFP16)
        let plan = try #require(capability.plan(matchedBoundary: 3, maximumSequenceLength: 32))
        let native = MLXArray.ones([1, 1, 3, 64], dtype: .float32)
        let prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?] = [(native, native, 3)]
        #expect(throws: CBv2KVError.self) {
            try backend.makeSequenceState(
                adopting: prefix, plan: plan, layerKinds: kinds, maxLength: 32)
        }
        #expect(backend.bytesReserved == 0 && backend.bytesWired == 0)
        #expect(backend.pool.groups.values.allSatisfy { $0.segments.isEmpty })
        #expect(!backend.pool.writeValidation.isFaulted)

        let nativeBackend = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 8 << 20, dtype: .float32, maxPrefillChunk: 256,
                segmentSizeBytes: 64 << 10))
        let restored = try nativeBackend.makeSequenceState(
            adopting: prefix, plan: plan, layerKinds: kinds, maxLength: 32)
        #expect(restored[0]?.absoluteOffset == 3)
        eval(
            restored.compactMap {
                ($0 as? PagedSequenceKV).map { nativeBackend.pool.group($0.groupKey).writeFence }
            })
        StreamOrDevice.default.stream.synchronize()
        nativeBackend.release(restored)
        #expect(nativeBackend.bytesReserved == 0)
    }
}
