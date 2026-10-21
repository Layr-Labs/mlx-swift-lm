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
        segmentBytes: Int = 64 << 10, maxPrefillChunk: Int = 256, recentTokenCount: Int = 128
    ) throws -> PagedKVBackend {
        try PagedKVBackend(
            layerKinds: kinds ?? [kind()],
            config: .init(
                capacityBytes: capacity, dtype: .float32, maxPrefillChunk: maxPrefillChunk,
                nominalMaxSequenceLength: 4096, segmentSizeBytes: segmentBytes,
                layerDTypes: types,
                quantization: quantized ? .init(recentTokenCount: recentTokenCount) : nil,
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

    private func admittedRow(_ backend: PagedKVBackend, promptTokens: Int, maxTokens: Int) throws
        -> (row: PagedSequenceKV, state: [CBv2SequenceKV?], admission: AdmissionV2, minimum: Int)
    {
        let config = try backend.pool.admissionStorageConfig(.init(watermarkFraction: 0))
        let admission = AdmissionV2(
            layerKinds: backend.layerKinds, bytesCapacity: backend.bytesCapacity,
            config: config, residency: backend.kvResidency)
        let length = promptTokens + maxTokens
        let overhead = try #require(
            backend.pool.minimumSegmentedOverhead(tokens: length, layerKinds: backend.layerKinds))
        let minimum = try #require(config.pagedKVNativeRecentFeasibility)
            .peakBytes(promptTokens: promptTokens, maxTokens: maxTokens)
        let capacity =
            admission.allocatedBytes(forTokens: length) + overhead
            + minimum
        backend.updateBytesCapacity(capacity)
        admission.updateBytesCapacity(capacity - 1)
        #expect(
            !admission.canEverFit(
                promptTokens: promptTokens, maxTokens: maxTokens, additionalBackendBytes: overhead))
        admission.updateBytesCapacity(capacity)
        try #require(
            admission.canEverFit(
                promptTokens: promptTokens, maxTokens: maxTokens, additionalBackendBytes: overhead))
        backend.pool.bindAdmission(admission)
        #expect(admission.bytesReserved == 0 && backend.bytesWired == 0)
        try admission.reserve(id: .init(1), additionalTokens: length)
        let state = try backend.makeSequenceState(
            layerKinds: backend.layerKinds, promptLength: promptTokens, maxLength: length)
        let row = try #require(state[0] as? PagedSequenceKV)
        #expect(admission.transientBytesReserved == 0)
        return (row, state, admission, minimum)
    }

    @Test func sdkDefaultUsesPackedRatesWithoutCallerDerivation() async throws {
        let backend = try backend(capacity: 3 << 20)
        let engine = EngineV2(
            model: CBv2SchedScriptedModel(), layerKinds: backend.layerKinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            sampler: CBv2GreedySampler())
        let admission = engine.admissionForTesting
        #expect(admission.estimatedBytes(forTokens: 8192) == 80 * 8192)
        #expect(admission.canEverFit(promptTokens: 8192, maxTokens: 16))
        let native = AdmissionV2(
            layerKinds: backend.layerKinds, bytesCapacity: backend.bytesCapacity,
            config: .init(elementBytes: 4), residency: backend.kvResidency)
        #expect(!native.canEverFit(promptTokens: 8192, maxTokens: 16))
        await engine.shutdown()
    }

    @Test func exactPagesCannotConsumeTheRequiredNativeTailMinimum() async throws {
        let backend = try backend()
        let engine = engine(backend)
        let config = try backend.pool.admissionStorageConfig(.init(watermarkFraction: 0))
        let admission = engine.admissionForTesting
        let pages = admission.allocatedBytes(forTokens: 1024)
        let minimum = try #require(config.pagedKVNativeRecentFeasibility)
            .peakBytes(promptTokens: 1024, maxTokens: 0)
        #expect(minimum > 0 && minimum < config.minimumRequestTransientBytes)
        engine.updateKVBytesCapacity(pages)
        #expect(admission.admissibleBytesCapacity == pages)
        #expect(!admission.canEverFit(promptTokens: 1024, maxTokens: 0))
        engine.updateKVBytesCapacity(pages + minimum - 1)
        #expect(!admission.canEverFit(promptTokens: 1024, maxTokens: 0))
        engine.updateKVBytesCapacity(pages + minimum)
        #expect(admission.canEverFit(promptTokens: 1024, maxTokens: 0))
        await engine.shutdown()
    }

    @Test func formerCompactTailBudgetIsRejectedBeforeAnyWrite() throws {
        let backend = try backend(capacity: 212_991)
        var old = AdmissionV2.Config(watermarkFraction: 0)
        old.layerBytesPerToken = [80]
        // Reproduce the historical compact-tail charge, independent of the
        // current allocator upper-bound policy. Recomputing this old budget
        // with today's policy no longer models the formerly admitted request.
        old.minimumRequestTransientBytes = 131_072
        let overhead = try #require(
            backend.pool.minimumSegmentedOverhead(tokens: 512, layerKinds: backend.layerKinds))
        #expect(old.minimumRequestTransientBytes == 131_072)
        #expect(80 * 512 + overhead == 81_919)
        let formerlyAccepted = AdmissionV2(
            layerKinds: backend.layerKinds, bytesCapacity: backend.bytesCapacity,
            config: old, residency: backend.kvResidency)
        #expect(
            formerlyAccepted.canEverFit(
                promptTokens: 256, maxTokens: 256, additionalBackendBytes: overhead))

        let config = try backend.pool.admissionStorageConfig(.init(watermarkFraction: 0))
        let admission = AdmissionV2(
            layerKinds: backend.layerKinds, bytesCapacity: backend.bytesCapacity,
            config: config, residency: backend.kvResidency)
        backend.pool.bindAdmission(admission)
        #expect(!admission.canEverFit(promptTokens: 256, maxTokens: 256))
        #expect(
            !admission.canEverFit(
                promptTokens: 256, maxTokens: 256, additionalBackendBytes: overhead))
        #expect(admission.bytesReserved == 0 && admission.transientBytesReserved == 0)
        #expect(backend.bytesWired == 0 && backend.pool.nativeRecentBytesInUse == 0)
        try backend.pool.writeValidation.check()
    }

    @Test(arguments: [1, 7], [0, 1])
    func shortRequestsDoNotPayConfiguredPrefillOrEightGenerations(
        promptTokens: Int, maximumOutput: Int
    ) throws {
        let backend = try backend(capacity: 8 << 20, maxPrefillChunk: 4096)
        let config = try backend.pool.admissionStorageConfig(.init(watermarkFraction: 0))
        let (row, state, admission, minimum) = try admittedRow(
            backend, promptTokens: promptTokens, maxTokens: maximumOutput)
        defer {
            backend.release(state)
            admission.releaseAll(id: .init(1))
        }
        var scalarOnly = config
        scalarOnly.pagedKVNativeRecentFeasibility = nil
        let configuredMaximum = AdmissionV2(
            layerKinds: backend.layerKinds, bytesCapacity: admission.bytesCapacity,
            config: scalarOnly, residency: backend.kvResidency)
        #expect(!configuredMaximum.canEverFit(promptTokens: promptTokens, maxTokens: maximumOutput))
        #expect(minimum < config.minimumRequestTransientBytes)
        if promptTokens == 1 && maximumOutput == 0 {
            let firstCopy =
                2 * (try CBv2CheckpointAllocationFootprint.bound(64 * 4))
                + 2 * (try CBv2CheckpointAllocationFootprint.bound(1))
                + (try CBv2CheckpointAllocationFootprint.bound(MemoryLayout<Int64>.stride))
            #expect(minimum == firstCopy)
        }
        let evaluate = backend.pool.quantizedRecentEvaluate
        var retirements = 0
        backend.pool.quantizedRecentEvaluate = { roots in
            retirements += 1
            try evaluate(roots)
        }
        defer { backend.pool.quantizedRecentEvaluate = evaluate }
        // Token-budget splitting is legal even though the whole prompt fits C.
        for _ in 0 ..< (promptTokens + maximumOutput) {
            let token = MLXArray.ones([1, 1, 64])
            row.write(keys: token, values: token)
            try backend.pool.writeValidation.check()
            #expect(admission.transientBytesReserved <= minimum)
            eval([backend.pool.group(row.groupKey).writeFence] + row.nativeRecentEvaluationRoots)
            try backend.pool.finishQuantizedRecentStep()
        }
        #expect(row.absoluteOffset == promptTokens + maximumOutput)
        #expect(row.nativeRecentKeys?.dim(1) == promptTokens + maximumOutput)
        #expect(retirements == 0)
        #expect(admission.bytesReserved <= admission.admissibleBytesCapacity)
    }

    @Test(arguments: [256, 1024], [0, 128])
    func nativeOwnerFloorFundsFullPromptAndOverlappingRetirement(chunk: Int, recent: Int) throws {
        let backend = try backend(
            capacity: 8 << 20, segmentBytes: 2 * 80 * CBv2PagedDefaults.pageSize,
            maxPrefillChunk: chunk, recentTokenCount: recent)
        let promptTokens = 15 + 2 * chunk
        let (row, state, admission, minimum) = try admittedRow(
            backend, promptTokens: promptTokens, maxTokens: chunk)
        let evaluate = backend.pool.quantizedRecentEvaluate
        var retirementPeaks = [Int]()
        backend.pool.quantizedRecentEvaluate = { roots in
            retirementPeaks.append(admission.transientBytesReserved)
            #expect(admission.transientBytesReserved == backend.pool.nativeRecentBytesInUse)
            #expect(admission.transientBytesReserved <= minimum)
            #expect(admission.bytesReserved <= admission.admissibleBytesCapacity)
            try evaluate(roots)
        }
        defer {
            backend.pool.quantizedRecentEvaluate = evaluate
            backend.release(state)
            admission.releaseAll(id: .init(1))
        }
        let keys = MLXArray(
            (0 ..< promptTokens * 64).map { Float($0) / 1024 }, [1, promptTokens, 64])
        let values = keys * 0.5
        var start = 0
        // The short first chunk leaves an uncompacted owner when R > 0. Every
        // full chunk then starts at slot 15 and touches one extra segment.
        for end in [15, 15 + chunk, promptTokens] {
            row.write(
                keys: keys[0..., start ..< end, 0...], values: values[0..., start ..< end, 0...])
            try backend.pool.writeValidation.check()
            #expect(row.absoluteOffset == end)
            #expect(row.nativeRecentKeys?.dim(1) == min(start, recent) + end - start)
            #expect(admission.transientBytesReserved == backend.pool.nativeRecentBytesInUse)
            #expect(admission.transientBytesReserved <= minimum)
            eval([backend.pool.group(row.groupKey).writeFence] + row.nativeRecentEvaluationRoots)
            try backend.pool.finishQuantizedRecentStep()
            #expect(row.nativeRecentKeys?.dim(1) == min(end, recent))
            #expect(row.retiredNativeRecentOwners.isEmpty)
            start = end
        }
        #expect(!retirementPeaks.isEmpty)
        #expect(row.absoluteOffset == promptTokens)
        #expect(row.nativeRecentStart == promptTokens - recent)
        if recent > 0 {
            #expect(
                try #require(row.nativeRecentKeys).asArray(Float.self)
                    == keys[0..., (promptTokens - recent)..., 0...].asArray(Float.self))
            #expect(
                try #require(row.nativeRecentValues).asArray(Float.self)
                    == values[0..., (promptTokens - recent)..., 0...].asArray(Float.self))
        }
    }

    @Test(arguments: [0, 128], [2, 8])
    func nativeOwnerFloorFundsEverySerialMTPGenerationThroughRollback(
        recent: Int, maximumOutput: Int
    ) throws {
        let backend = try backend(capacity: 8 << 20, recentTokenCount: recent)
        let promptTokens = max(1, recent - 1)
        let (row, state, admission, minimum) = try admittedRow(
            backend, promptTokens: promptTokens, maxTokens: maximumOutput)
        let evaluate = backend.pool.quantizedRecentEvaluate
        defer {
            backend.pool.quantizedRecentEvaluate = evaluate
            backend.release(state)
            admission.releaseAll(id: .init(1))
        }
        let keys = MLXArray.ones([1, promptTokens, 64]) * 0.25
        row.write(keys: keys, values: keys)
        try backend.pool.writeValidation.check()
        eval([backend.pool.group(row.groupKey).writeFence] + row.nativeRecentEvaluationRoots)
        try backend.pool.finishQuantizedRecentStep()
        let oldBytes = admission.transientBytesReserved
        let oldCount = min(promptTokens, recent)
        row.beginSpeculativeWrite()
        let span = min(CBv2PagedSpeculation.maxSpeculativeSpan, maximumOutput)
        for column in 1 ... span {
            let token = MLXArray.ones([1, 1, 64]) * Float(column)
            row.write(keys: token, values: token)
            try backend.pool.writeValidation.check()
            eval([backend.pool.group(row.groupKey).writeFence] + row.nativeRecentEvaluationRoots)
            let held = admission.transientBytesReserved
            #expect(row.retiredNativeRecentOwners.count == column)
            #expect(row.nativeRecentKeys?.dim(1) == oldCount + column)
            #expect(held > oldBytes && held == backend.pool.nativeRecentBytesInUse)
            #expect(held <= minimum)
            try backend.pool.finishQuantizedRecentStep()
            #expect(admission.transientBytesReserved == held)
        }
        let pendingBytes = admission.transientBytesReserved
        row.rollback(span - 1)
        #expect(admission.transientBytesReserved == pendingBytes)
        row.commitSpeculativeWrite()
        var retirementPeak = 0
        backend.pool.quantizedRecentEvaluate = { roots in
            retirementPeak = admission.transientBytesReserved
            #expect(retirementPeak > pendingBytes && retirementPeak <= minimum)
            #expect(retirementPeak == backend.pool.nativeRecentBytesInUse)
            #expect(admission.bytesReserved <= admission.admissibleBytesCapacity)
            try evaluate(roots)
        }
        try backend.pool.finishQuantizedRecentStep()
        #expect(retirementPeak > 0)
        #expect(row.absoluteOffset == promptTokens + 1)
        #expect(row.nativeRecentKeys?.dim(1) == min(promptTokens + 1, recent))
        #expect(row.retiredNativeRecentOwners.isEmpty)
        #expect(admission.transientBytesReserved < pendingBytes)
        if recent > 0 {
            let expected = concatenated([keys, MLXArray.ones([1, 1, 64])], axis: 1)
            #expect(
                try #require(row.nativeRecentKeys).asArray(Float.self)
                    == expected.asArray(Float.self))
            #expect(
                try #require(row.nativeRecentValues).asArray(Float.self)
                    == expected.asArray(Float.self))
        }
    }

    @Test func phaseMaximumIsTakenAfterSummingOnlyQuantizedOwners() throws {
        // These geometries cross phase maxima under both 4 KiB and 16 KiB
        // allocator policies; preserve the strict phase assertions below.
        let chunk = 208
        let recent = 16
        let span = CBv2PagedSpeculation.maxSpeculativeSpan
        let page = CBv2PagedDefaults.pageSize
        let offset = try CBv2CheckpointAllocationFootprint.bound(MemoryLayout<Int64>.stride)
        let metadata = 2 * (try CBv2CheckpointAllocationFootprint.bound(1))
        let oldOffsets = ((recent + page - 2) / page + 1) * offset
        let chunkOffsets = ((chunk + page - 2) / page + 1) * offset
        var prefill = [Int]()
        var serial = [Int]()
        for (dimension, heads, elementBytes) in [(64, 1, 2), (512, 2, 4)] {
            func arrays(_ count: Int) throws -> Int {
                2
                    * (try CBv2CheckpointAllocationFootprint.bound(
                        count * dimension * heads * elementBytes))
            }
            let overlap = try 2 * (arrays(recent) + metadata) + oldOffsets
            prefill.append(try overlap + arrays(recent + chunk) + chunkOffsets)
            serial.append(
                try (1 ... span).reduce(overlap) { total, column in
                    try total + arrays(recent + column) + offset
                })
        }
        #expect(prefill[0] < serial[0] && prefill[1] > serial[1])
        let backend = try backend(
            kinds: [
                // A window no wider than the recent band stays unquantized.
                kind(), kind(dimension: 512, heads: 2), kind(window: recent),
                kind(dimension: 512, heads: 2, shares: 1),
            ], types: [.float16, .float32, .float32, .float32], capacity: 16 << 20,
            maxPrefillChunk: chunk, recentTokenCount: recent)
        let config = try backend.pool.admissionStorageConfig(.init(watermarkFraction: 0))
        #expect(
            config.minimumRequestTransientBytes == max(prefill.reduce(0, +), serial.reduce(0, +)))
        #expect(
            config.minimumRequestTransientBytes
                < zip(prefill, serial).reduce(0) { $0 + max($1.0, $1.1) })
        #expect(config.layerBytesPerToken?[3] == 0)
        #expect(!backend.pool.usesQuantization(layerIndex: 2))
    }

    @Test func nativeOwnersAndSharedBorrowersAddNoNativeGenerationMinimum() throws {
        var caller = AdmissionV2.Config(watermarkFraction: 0, fixedBytesPerRequest: 123)
        caller.minimumRequestTransientBytes = 4096
        let owner = try backend(capacity: 8 << 20)
        let shared = try backend(kinds: [kind(), kind(shares: 0)], capacity: 8 << 20)
        let ownerConfig = try owner.pool.admissionStorageConfig(caller)
        let sharedConfig = try shared.pool.admissionStorageConfig(caller)
        #expect(
            sharedConfig.minimumRequestTransientBytes == ownerConfig.minimumRequestTransientBytes)
        #expect(sharedConfig.layerBytesPerToken == [80, 0])
        let native = try backend(
            kinds: [kind(), kind(shares: 0)], nativeOwners: [0], capacity: 8 << 20)
        let nativeConfig = try native.pool.admissionStorageConfig(sharedConfig)
        #expect(nativeConfig.minimumRequestTransientBytes == caller.minimumRequestTransientBytes)
        #expect(nativeConfig.pagedKVTransientFeasibilityBytes == 0)
        #expect(nativeConfig.pagedKVNativeRecentFeasibility == nil)
        #expect(nativeConfig.fixedBytesPerRequest == 123)
        #expect(nativeConfig.layerBytesPerToken == [512, 0])
        let unquantized = try backend(quantized: false)
        let unquantizedConfig = try unquantized.pool.admissionStorageConfig(ownerConfig)
        #expect(
            unquantizedConfig.minimumRequestTransientBytes == caller.minimumRequestTransientBytes)
        #expect(unquantizedConfig.pagedKVTransientFeasibilityBytes == 0)
        #expect(unquantizedConfig.pagedKVNativeRecentFeasibility == nil)
    }

    @Test func emptyNegativeAndOverflowingRequestsFailSafely() throws {
        let backend = try backend(capacity: 8 << 20)
        let config = try backend.pool.admissionStorageConfig(.init(watermarkFraction: 0))
        let policy = try #require(config.pagedKVNativeRecentFeasibility)
        #expect(try policy.peakBytes(promptTokens: 0, maxTokens: 0) == 0)
        let admission = AdmissionV2(
            layerKinds: backend.layerKinds, bytesCapacity: 0,
            config: config, residency: backend.kvResidency)
        #expect(admission.canEverFit(promptTokens: 0, maxTokens: 0))
        #expect(!admission.canEverFit(promptTokens: -1, maxTokens: 0))
        #expect(!admission.canEverFit(promptTokens: Int.max, maxTokens: 1))
        #expect(!admission.canEverFit(promptTokens: Int.max, maxTokens: 0))
        #expect(!admission.canEverFit(promptTokens: 1, maxTokens: Int.max))
        #expect(throws: (any Error).self) { try policy.peakBytes(promptTokens: -1, maxTokens: 0) }
        #expect(throws: (any Error).self) { try policy.peakBytes(promptTokens: 0, maxTokens: -1) }
        #expect(admission.canEverFit(promptTokens: 0, maxTokens: Int.min))
        #expect(admission.bytesReserved == 0 && backend.bytesWired == 0)
    }

    @Test func invalidPrefillAndOverflowingGenerationBoundsFailClosed() throws {
        for chunk in [0, -1] {
            #expect(throws: CBv2KVError.self) { try backend(maxPrefillChunk: chunk) }
        }
        #expect(throws: CBv2KVError.self) { try backend(recentTokenCount: -1) }
        for chunk in [Int.max, Int.max - 128] {
            let backend = try backend(maxPrefillChunk: chunk)
            #expect(throws: (any Error).self) {
                try backend.pool.admissionStorageConfig(.init(watermarkFraction: 0))
            }
            #expect(backend.bytesWired == 0)
        }
    }

    @Test func providerPrederivationAndAutomaticDerivationChargeTheSameMinimum() async throws {
        let automaticBackend = try backend()
        let prederivedBackend = try backend()
        var caller = AdmissionV2.Config(watermarkFraction: 0, fixedBytesPerRequest: 123)
        caller.minimumRequestTransientBytes = 4096
        let prederived = try prederivedBackend.pool.admissionStorageConfig(caller)
        let repeated = try prederivedBackend.pool.admissionStorageConfig(prederived)
        #expect(repeated.minimumRequestTransientBytes == prederived.minimumRequestTransientBytes)
        #expect(
            repeated.pagedKVTransientFeasibilityBytes == prederived.pagedKVTransientFeasibilityBytes
        )
        #expect(repeated.layerBytesPerToken == prederived.layerBytesPerToken)
        var previousQuote = caller
        previousQuote.minimumRequestTransientBytes += 131_072
        previousQuote.pagedKVTransientFeasibilityBytes = 131_072
        #expect(
            try prederivedBackend.pool.admissionStorageConfig(previousQuote)
                .minimumRequestTransientBytes == prederived.minimumRequestTransientBytes)
        let automatic = engine(automaticBackend, admission: caller)
        let explicit = engine(prederivedBackend, admission: prederived)
        let pages = 80 * 1024 + 123
        let sizedMinimum = try #require(prederived.pagedKVNativeRecentFeasibility)
            .peakBytes(promptTokens: 1024, maxTokens: 0)
        #expect(
            try #require(repeated.pagedKVNativeRecentFeasibility)
                .peakBytes(promptTokens: 1024, maxTokens: 0) == sizedMinimum)
        let ceiling = pages + caller.minimumRequestTransientBytes + sizedMinimum
        #expect(ceiling < pages + prederived.minimumRequestTransientBytes)
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
        var replacement = prederived
        replacement.minimumRequestTransientBytes = 23
        #expect(replacement.pagedKVTransientFeasibilityBytes == 0)
        #expect(replacement.pagedKVNativeRecentFeasibility == nil)
        let replaced = AdmissionV2(
            layerKinds: prederivedBackend.layerKinds, bytesCapacity: pages + 23,
            config: replacement, residency: prederivedBackend.kvResidency)
        #expect(replaced.canEverFit(promptTokens: 1024, maxTokens: 0))
        replaced.updateBytesCapacity(pages + 22)
        #expect(!replaced.canEverFit(promptTokens: 1024, maxTokens: 0))
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
        #expect(derived.pagedKVNativeRecentFeasibility == nil)
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
