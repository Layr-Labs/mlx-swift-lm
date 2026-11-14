// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The native tower through production engines on the segmented paged backend.
/// The model is tiny; attention has the production head geometry. Only the
/// checkpoint byte store is a fixture.
@Suite(.serialized, .tags(.integration))
struct MuseGlimmerCBv2Tests {
    static let window = 320
    static let chunk = 256

    /// Float32 towers compare tokens with the eager path strictly; BF16 towers
    /// (the production dtype) compare paged runs with each other byte for byte.
    private func model(_ dtype: DType = .bfloat16) throws -> MuseGlimmerTextModel {
        try MuseGlimmerFixture.model(
            MuseGlimmerFixture.pagedText(window: Self.window), dtype: dtype)
    }

    private func pagedBackend(_ m: MuseGlimmerTextModel) throws -> PagedKVBackend {
        return try PagedKVBackend(
            layerKinds: m.cbv2LayerKinds,
            config: .init(
                capacityBytes: 192 << 20, dtype: m.activationDType, maxPrefillChunk: Self.chunk,
                nominalMaxSequenceLength: 2048, segmentSizeBytes: 1 << 20,
                layerDTypes: m.kvStorageDTypes))
    }

    private func engine(
        _ m: MuseGlimmerTextModel, adapter: (any CBv2SteppableModel)? = nil,
        store: CompleteCheckpointFixtureStore? = nil, concurrency: Int = 2
    ) throws -> (EngineV2, PagedKVBackend) {
        m.prefillChunkAlignment = Self.chunk
        let backend = try pagedBackend(m)
        #expect(
            EngineV2.backendCapabilityViolation(capabilities: m.cbv2Capabilities, backend: backend)
                == nil)
        let engine = EngineV2(
            model: adapter ?? CBv2SteppableLanguageModelAdapter(m), layerKinds: m.cbv2LayerKinds,
            backend: backend, cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: concurrency, maxBatchedTokensPerStep: Self.chunk,
                prefillChunkSize: Self.chunk, soloPrefillStripeTokens: Self.chunk, maxWaiting: 4,
                enablePrefixCache: store != nil),
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store)
        #expect(engine.loopForTesting.scheduler.prefillChunkAlignment == Self.chunk)
        return (engine, backend)
    }

    private func request(
        _ id: UInt64, length: Int, salt: Int = 0, maxTokens: Int = 12, cacheSalt: String? = nil,
        checkpointAt target: Int? = nil
    ) -> CBv2Request {
        .init(
            id: .init(id),
            promptTokens: MuseGlimmerFixture.tokens(length, vocabulary: 96, salt: salt).map(
                Int.init),
            sampling: .init(temperature: 0, seed: 0), maxTokens: maxTokens, cacheSalt: cacheSalt,
            prefixCacheReceiptID: target == nil ? nil : .init(id + 1000),
            prefixCheckpointTargetTokens: target)
    }

    private func eager(_ m: MuseGlimmerTextModel, _ request: CBv2Request) throws -> [Int] {
        try MuseGlimmerFixture.eagerGreedy(
            m, prompt: request.promptTokens.map(Int32.init), chunk: Self.chunk,
            steps: request.maxTokens)
    }

    @Test func layerKindsAndCapabilitiesDescribeAPagedOnlyTarget() throws {
        let m = try model()
        let kinds = m.cbv2LayerKinds
        #expect(kinds.count == 6)
        #expect(
            kinds.map(\.attention) == [
                .slidingWindow(Self.window), .full, .slidingWindow(Self.window),
                .slidingWindow(Self.window), .slidingWindow(Self.window), .full,
            ])
        let capabilities = m.cbv2Capabilities
        #expect(capabilities.supportsPagedKV && capabilities.requiresNativePagedKV)
        #expect(capabilities.requiresPagedKV && !capabilities.supportsPackedPrefill)
        #expect(capabilities.prefillChunkAlignment == 1024)
        #expect(MuseGlimmerTextModel.defaultPrefillChunkAlignment == 1024)
        #expect(m.cbv2SupportsHistoricalAttentionCheckpoint)
        #expect(m.cbv2CompleteCheckpointKVDTypes == Array(repeating: .bfloat16, count: 6))
        #expect(m.kvStorageDTypes == Array(repeating: .bfloat16, count: 6))
        #expect(!m.cbv2SupportsPackedPrefill)
        let adapter = CBv2SteppableLanguageModelAdapter(m)
        #expect(adapter.cbv2Capabilities == capabilities && !adapter.supportsPackedPrefill)
        #expect(adapter.decodeCommandBufferLimits == MuseGlimmerDecodeCommandBuffers.limits)
        // No prompt chunk shares a forward with decode or verify rows (exact
        // windows depend on it): neither adapter packs prompt chunks.
        let drafting: AnyObject = MuseGlimmerCBv2Adapter(m)
        #expect(!(drafting is any CBv2PackedPrefillSteppableModel))
    }

    @Test func nonPagedAndUnprobedBackendsAreRefused() throws {
        let m = try model()
        // A pool whose prompt write is shorter than one grid cell is refused.
        #expect(
            EngineV2.backendCapabilityViolation(
                capabilities: m.cbv2Capabilities, backend: try pagedBackend(m)) != nil)
        m.prefillChunkAlignment = Self.chunk
        let capabilities = m.cbv2Capabilities
        let contiguous = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 8 << 20, kvDType: .bfloat16))
        #expect(
            EngineV2.backendCapabilityViolation(capabilities: capabilities, backend: contiguous)
                != nil)
        // The fixed-precision pool (no segments, no observed dtypes) is refused too.
        let fixed = try PagedKVBackend(
            layerKinds: m.cbv2LayerKinds,
            config: .init(
                capacityBytes: 64 << 20, dtype: .bfloat16, maxPrefillChunk: Self.chunk,
                nominalMaxSequenceLength: 2048))
        #expect(
            EngineV2.backendCapabilityViolation(capabilities: capabilities, backend: fixed) != nil)
        #expect(
            EngineV2.backendCapabilityViolation(
                capabilities: capabilities, backend: try pagedBackend(m)) == nil)
        // The new requirement is off for every other model: the defaults and
        // a native-paged-only model still accept contiguous storage.
        #expect(!CBv2ModelCapabilities.attentionOnly.requiresPagedKV)
        #expect(!CBv2ModelCapabilities.initialRecurrentTarget.requiresPagedKV)
        var nativeOnly = CBv2ModelCapabilities.attentionOnly
        nativeOnly.requiresNativePagedKV = true
        for existing in [CBv2ModelCapabilities.attentionOnly, .initialRecurrentTarget, nativeOnly] {
            #expect(
                EngineV2.backendCapabilityViolation(capabilities: existing, backend: contiguous)
                    == nil)
        }
    }

    /// 1,100 tokens against a 320-token window in 256-token chunks: every
    /// sliding ring has wrapped before the first sampled token.
    @Test func pagedGreedyTokensEqualTheEagerPathBeyondTheWindow() async throws {
        let m = try model(.float32)
        let input = request(1, length: 1100, maxTokens: 32)
        let expected = try eager(m, input)
        let (engine, backend) = try engine(m)
        let result = await cbv2SchedCollect(try engine.submit(input))
        await engine.shutdown()
        #expect(result.finishReason == .length)
        #expect(result.tokens == expected)
        #expect(Set(expected).count > 3)
        #expect(!backend.pool.writeValidation.isFaulted && backend.bytesReserved == 0)
    }

    /// Window semantics, not just the choice of token: logits of the paged
    /// caches against the eager rotating caches while both cross the window
    /// in prefill (700 tokens, 256 per chunk) and in decode (400 more steps).
    @Test func pagedLogitsTrackEagerLogitsAcrossTheWindowBoundary() throws {
        let m = try model(.float32)
        let prompt = MuseGlimmerFixture.tokens(700, vocabulary: 96, salt: 3)
        let backend = try pagedBackend(m)
        let caches = backend.makeLayerCaches()
        let state = try backend.makeSequenceState(
            layerKinds: m.cbv2LayerKinds, promptLength: prompt.count, maxLength: 1200)
        defer {
            eval(caches.flatMap { $0.innerState() })
            for cache in caches { cache.setRows([]) }
            backend.release(state)
        }
        for (index, cache) in caches.enumerated() { cache.setRows([state[index]!]) }
        let eagerCache = try m.newCache(parameters: nil)
        var worst: Float = 0
        var next = Int32(0)
        func step(_ tokens: MLXArray) {
            let paged = m(tokens, cache: caches)
            let eager = m(tokens, cache: eagerCache)
            eval(
                [paged, eager] + caches.flatMap { $0.innerState() }
                    + eagerCache.flatMap { $0.innerState() })
            worst = max(worst, abs(paged - eager).max().item(Float.self))
            next = argMax(eager[0, -1], axis: -1).item(Int32.self)
        }
        for start in stride(from: 0, to: prompt.count, by: Self.chunk) {
            step(
                MLXArray(Array(prompt[start ..< min(start + Self.chunk, prompt.count)])).reshaped(
                    1, -1))
        }
        for _ in 0 ..< 400 { step(MLXArray([next]).reshaped(1, 1)) }
        // Float32 sums in two kernels' orders; logits are of order 1 to 10.
        #expect(worst < 2e-3)
        #expect(!backend.pool.writeValidation.isFaulted)
    }

    @Test(arguments: [DType.float32, DType.bfloat16])
    func concurrentRequestsEqualTheirSoloOutputs(dtype: DType) async throws {
        let m = try model(dtype)
        let inputs = [
            request(1, length: 700, salt: 1, maxTokens: 24),
            request(2, length: 190, salt: 2, maxTokens: 24),
        ]
        var solo = [[Int]]()
        for input in inputs {
            let (engine, _) = try engine(m, concurrency: 1)
            let result = await cbv2SchedCollect(try engine.submit(input))
            await engine.shutdown()
            #expect(result.finishReason == .length)
            solo.append(result.tokens)
            let reference = try MuseGlimmerFixture.eagerGreedyWithMargins(
                m, prompt: input.promptTokens.map(Int32.init), chunk: Self.chunk,
                steps: input.maxTokens)
            if dtype == .float32 {
                #expect(result.tokens == reference.tokens)
            } else {
                MuseGlimmerFixture.expectSameUntilNearTie(result.tokens, reference)
            }
        }
        let (engine, backend) = try engine(m, concurrency: 2)
        let first = try engine.submit(inputs[0])
        let second = try engine.submit(inputs[1])
        async let a = cbv2SchedCollect(first)
        async let b = cbv2SchedCollect(second)
        let together = await [a, b]
        await engine.shutdown()
        #expect(together.map(\.tokens) == solo)
        #expect(together.allSatisfy { $0.finishReason == .length })
        #expect(!backend.pool.writeValidation.isFaulted && backend.bytesReserved == 0)
    }

    @Test(arguments: [DType.float32, DType.bfloat16])
    func completeCheckpointRestoreEqualsAColdRun(dtype: DType) async throws {
        let m = try model(dtype)
        let cold = request(1, length: 1543, maxTokens: 8, cacheSalt: "muse", checkpointAt: 1024)
        let seed = CompleteCheckpointFixtureStore(segmentBytes: 4093)
        let (coldEngine, coldBackend) = try engine(m, store: seed)
        let coldResult = await cbv2SchedCollect(try coldEngine.submit(cold))
        seed.finishPublicationCallbacks(engine: coldEngine)
        await coldEngine.shutdown()
        try #require(coldResult.finishReason == .length)
        if dtype == .float32 { #expect(coldResult.tokens == (try eager(m, cold))) }
        #expect(coldBackend.bytesReserved == 0)
        let archives = seed.saved.filter { $0.manifest.position == 1024 }
        try #require(!archives.isEmpty)

        // A new model and engine see only the serialized bytes.
        let restored = CompleteCheckpointFixtureStore(archives: archives, segmentBytes: 4093)
        let (warmEngine, warmBackend) = try engine(try model(dtype), store: restored)
        let foreign = request(
            2, length: 1543, maxTokens: 8, cacheSalt: "another-tenant", checkpointAt: 1024)
        #expect(try !restored.stage(engine: warmEngine, request: foreign))
        let warm = request(3, length: 1543, maxTokens: 8, cacheSalt: "muse", checkpointAt: 1024)
        try #require(try restored.stage(engine: warmEngine, request: warm))
        let warmResult = await cbv2SchedCollect(try warmEngine.submit(warm))
        restored.finishPublicationCallbacks(engine: warmEngine)
        await warmEngine.shutdown()
        #expect(warmResult.finishReason == .length && warmResult.tokens == coldResult.tokens)
        #expect(warmResult.usage?.prefixCachePrefillTokensSaved == 1024)
        #expect(warmResult.usage?.prefixCacheReplayTokens == 0)
        #expect(warmBackend.bytesReserved == 0 && warmEngine.admissionForTesting.bytesReserved == 0)
    }

    /// The prompt-output seam writes the same K/V as the full forward and
    /// returns the full forward's last-row logits.
    @Test func narrowedPrefillKeepsKVAndTheFrontierLogits() throws {
        let m = try model()
        let prompt = MuseGlimmerFixture.tokens(600, vocabulary: 96)
        func run(narrowed: Bool) throws -> (MLXArray, [Data]) {
            let backend = try pagedBackend(m)
            let caches = backend.makeLayerCaches()
            let state = try backend.makeSequenceState(
                layerKinds: m.cbv2LayerKinds, promptLength: prompt.count,
                maxLength: prompt.count + 8)
            defer {
                eval(caches.flatMap { $0.innerState() })
                for cache in caches { cache.setRows([]) }
                backend.release(state)
            }
            for (index, cache) in caches.enumerated() { cache.setRows([state[index]!]) }
            var last = MLXArray.zeros([1])
            for start in stride(from: 0, to: prompt.count, by: Self.chunk) {
                let end = min(start + Self.chunk, prompt.count)
                let tokens = MLXArray(Array(prompt[start ..< end])).reshaped(1, -1)
                if narrowed {
                    last = m.cbv2Prefill(
                        tokens, inputEmbedding: nil, cache: caches,
                        requirement: end == prompt.count ? .lastPositionLogits : .evaluationOnly)
                } else {
                    last = m(tokens, cache: caches)[0..., -1, 0...]
                }
                eval([last] + caches.flatMap { $0.innerState() })
            }
            var kv = [Data]()
            for row in state {
                let snapshot = try #require(row).snapshot()
                eval(snapshot.keys, snapshot.values)
                kv.append(snapshot.keys.asData(access: .copy).data)
                kv.append(snapshot.values.asData(access: .copy).data)
            }
            return (last, kv)
        }
        let (fullLogits, fullKV) = try run(narrowed: false)
        let (narrowLogits, narrowKV) = try run(narrowed: true)
        #expect(narrowKV == fullKV)
        #expect(narrowLogits.shape == [1, 96])
        // The head reads one row instead of 88: a dense product, so the rows
        // may differ in the last BF16 bit; the choice must not.
        let difference = abs(narrowLogits.asType(.float32) - fullLogits.asType(.float32))
        #expect(difference.max().item(Float.self) <= 0.05)
        #expect(
            argMax(narrowLogits, axis: -1).item(Int.self)
                == argMax(fullLogits, axis: -1).item(Int.self))
    }
}
