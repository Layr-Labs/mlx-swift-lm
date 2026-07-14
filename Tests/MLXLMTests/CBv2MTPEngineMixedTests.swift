import Foundation
import MLX
import MLXRandom
import Testing
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

private final class MTPTestSequenceKV: CBv2SequenceKV {
    let supportsSpeculativeWrites: Bool
    var absoluteOffset: Int = 0
    var retainedCount: Int { absoluteOffset }
    var byteCount: Int { 0 }

    init(supportsSpeculativeWrites: Bool) {
        self.supportsSpeculativeWrites = supportsSpeculativeWrites
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        absoluteOffset += keys.dim(-2)
        return (keys, values)
    }

    func snapshot() -> (keys: MLXArray, values: MLXArray, offset: Int) {
        let empty = MLXArray.zeros([1, 1, 0, 1], dtype: .float32)
        return (empty, empty, absoluteOffset)
    }

    func rollback(_ n: Int) { absoluteOffset -= n }
}

private final class MTPRejectingPrepared: CBv2MTPPreparedCapture {}

/// Returns the input token as every draft. The deterministic cycle target
/// predicts input+1, so every verify round rejects at position zero.
private final class MTPRejectingDrafter: CBv2MTPDrafter {
    let mtpTargetIdentity: ObjectIdentifier?

    init(target: Gemma4TextModel) {
        self.mtpTargetIdentity = ObjectIdentifier(target)
    }

    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        MTPRejectingPrepared()
    }

    func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        (tokens.squeezed(axis: 1), hidden)
    }
}

private final class MTPReleaseTrackingCacheProvider: CBv2LayerCacheProvider,
    CBv2CompositionInvalidating, @unchecked Sendable
{
    private let bank: CBv2LayerCacheBank
    private let lock = NSLock()
    private var retainsBoundRows = false
    private var staleObserver: (() -> Bool)?
    private var _releasesWhileBoundAndStale = 0

    init(layerKinds: [CBv2LayerKind]) {
        self.bank = CBv2LayerCacheBank(layerKinds: layerKinds)
    }

    var uniformAttentionSoftcap: Float?? { bank.uniformAttentionSoftcap }
    var supportsMultimodalSpans: Bool { bank.supportsMultimodalSpans }

    var releasesWhileBoundAndStale: Int {
        lock.lock()
        defer { lock.unlock() }
        return _releasesWhileBoundAndStale
    }

    func observeStaleness(_ observer: @escaping () -> Bool) {
        lock.lock()
        staleObserver = observer
        lock.unlock()
    }

    func layerCaches(rowStates: [[CBv2SequenceKV?]]) -> [CBv2AttendingLayerCache] {
        let caches = bank.layerCaches(rowStates: rowStates)
        lock.lock()
        retainsBoundRows = true
        lock.unlock()
        return caches
    }

    func invalidateBoundComposition() {
        // Invalidation alone does not clear the caches' strong row arrays.
        bank.invalidateBoundComposition()
    }

    func releaseBoundRows() {
        lock.lock()
        let wasRetaining = retainsBoundRows
        let observer = staleObserver
        lock.unlock()
        let wasAlreadyStale = observer?() ?? false

        bank.releaseBoundRows()
        lock.lock()
        if wasRetaining && wasAlreadyStale {
            _releasesWhileBoundAndStale += 1
        }
        retainsBoundRows = false
        lock.unlock()
    }
}

@Suite("CBv2MTPEngineMixed", .serialized)
struct CBv2MTPEngineMixedTests {
    private let vocabSize = 256
    private let hiddenSize = 64
    private let slidingWindow = 16
    private let fixedDepth = 2

    private func targetConfig(
        tieWordEmbeddings: Bool = true
    ) throws -> Gemma4TextConfiguration {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": \(hiddenSize),
                "num_hidden_layers": 6,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "head_dim": 32,
                "global_head_dim": 32,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 2,
                "layer_types": ["sliding_attention", "full_attention",
                                "full_attention", "sliding_attention",
                                "sliding_attention", "full_attention"],
                "sliding_window": \(slidingWindow),
                "final_logit_softcapping": 30.0,
                "tie_word_embeddings": \(tieWordEmbeddings),
                "vocab_size": \(vocabSize),
                "vocab_size_per_layer_input": \(vocabSize),
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false
            }
            """
        return try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    private func drafterConfig() throws -> Gemma4AssistantConfiguration {
        let json = """
            {
                "model_type": "gemma4_assistant",
                "backbone_hidden_size": \(hiddenSize),
                "use_ordered_embeddings": false,
                "num_centroids": 16,
                "centroid_intermediate_top_k": 4,
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 32,
                    "num_hidden_layers": 2,
                    "intermediate_size": 64,
                    "num_attention_heads": 2,
                    "head_dim": 32,
                    "global_head_dim": 32,
                    "num_key_value_heads": 1,
                    "num_kv_shared_layers": 2,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "sliding_window": \(slidingWindow),
                    "final_logit_softcapping": null,
                    "tie_word_embeddings": true,
                    "vocab_size": \(vocabSize),
                    "vocab_size_per_layer_input": \(vocabSize),
                    "rms_norm_eps": 1e-6,
                    "hidden_size_per_layer_input": 0,
                    "use_double_wide_mlp": false
                }
            }
            """
        return try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: Data(json.utf8))
    }

    private struct Fixture {
        let target: Gemma4TextModel
        let drafter: Gemma4AssistantDraftModel
    }

    private func makeFixture(
        seed: UInt64 = 0x5EED, deterministicTarget: Bool = false
    ) throws -> Fixture {
        MLXRandom.seed(seed)
        let target = Gemma4TextModel(
            try targetConfig(tieWordEmbeddings: !deterministicTarget))
        let drafter = try Gemma4AssistantDraftModel(config: drafterConfig())
        if deterministicTarget { stabilizeCBv2MTPGreedyCycleTarget(target) }
        eval(target, drafter)
        return Fixture(target: target, drafter: drafter)
    }

    private func makeEngine(
        _ fixture: Fixture, mtp: Bool,
        maxSpeculativeBatch: Int = 4, maxConcurrent: Int = 4,
        prefillChunkSize: Int = 16, bytesCapacity: Int = 1 << 28,
        admissionConfig: AdmissionV2.Config = .init(),
        compiledEnabled: Bool = false,
        backend: CBv2KVBackend? = nil,
        cacheProvider: CBv2LayerCacheProvider? = nil,
        prefixCache: CBv2PrefixCache? = nil,
        earlyPrefixDonation: Bool = false,
        eventBufferCapacity: Int = 256,
        mtpDrafter: (any CBv2MTPDrafter)? = nil
    ) throws -> EngineV2 {
        let kinds = fixture.target.cbv2LayerKinds
        let backend = backend
            ?? CBv2ContiguousKVBackend(config: .init(bytesCapacity: bytesCapacity))
        let provider = cacheProvider ?? CBv2LayerCacheBank(layerKinds: kinds)
        let drafter: (any CBv2MTPDrafter)?
        if mtp, let mtpDrafter {
            drafter = mtpDrafter
        } else if mtp {
            drafter = try Gemma4CBv2MTPDrafter(
                drafter: fixture.drafter, target: fixture.target)
        } else {
            drafter = nil
        }
        let mtpConfig = makeMTPConfig(
            enabled: mtp, maxSpeculativeBatch: maxSpeculativeBatch)
        return EngineV2(
            model: CBv2SteppableLanguageModelAdapter(fixture.target),
            layerKinds: kinds, backend: backend, cacheProvider: provider,
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: maxConcurrent,
                maxBatchedTokensPerStep: 256,
                prefillChunkSize: prefillChunkSize, maxWaiting: 16,
                enablePrefixCache: prefixCache != nil),
            loopConfig: CBv2EngineLoopConfig(
                eventBufferCapacity: eventBufferCapacity,
                enableEarlyPrefixDonation: earlyPrefixDonation),
            admissionConfig: admissionConfig,
            prefixCache: prefixCache,
            compiledDecodeConfig: CBv2CompiledDecodeConfig(
                enabled: compiledEnabled, buckets: [1, 2], kvCapacity: 256),
            mtpDrafter: drafter,
            mtpConfig: mtpConfig)
    }

    private func makeMTPConfig(
        enabled: Bool, maxSpeculativeBatch: Int
    ) -> CBv2MTPConfig {
        var config = CBv2MTPConfig(
            enabled: enabled, maxDraftTokens: fixedDepth,
            maxSpeculativeBatch: maxSpeculativeBatch,
            fixedDraftTokens: fixedDepth)
        config.runtimeChipNameOverrideForTesting = "Apple M4 Max"
        return config
    }

    private func request(
        id: UInt64, prompt: [Int], maxTokens: Int,
        temperature: Float = 0, topLogprobs: Int = 0, seed: UInt64? = nil
    ) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt,
            sampling: CBv2SamplingParams(
                temperature: temperature, seed: seed, topLogprobs: topLogprobs),
            maxTokens: maxTokens)
    }

    private func run(
        _ engine: EngineV2, _ request: CBv2Request
    ) async throws -> CBv2SchedCollected {
        await cbv2SchedCollect(try engine.submit(request))
    }

    private func baseline(
        _ fixture: Fixture, _ request: CBv2Request
    ) async throws -> CBv2SchedCollected {
        let engine = try makeEngine(fixture, mtp: false)
        let result = try await run(engine, request)
        await engine.shutdown()
        return result
    }

    private func admissionProbe(_ fixture: Fixture) -> AdmissionV2 {
        AdmissionV2(
            layerKinds: fixture.target.cbv2LayerKinds, bytesCapacity: 1 << 40,
            config: .init(watermarkFraction: 0))
    }

    @Test func verifyRowsWithChunkedPrefillNeighborStayExact() async throws {
        let fixture = try makeFixture()
        let requests = [
            request(
                id: 1,
                prompt: makePromptTokens(length: 16, seed: 111, vocabSize: vocabSize),
                maxTokens: 32),
            request(
                id: 2,
                prompt: makePromptTokens(length: 22, seed: 112, vocabSize: vocabSize),
                maxTokens: 28),
            request(
                id: 3,
                prompt: makePromptTokens(length: 128, seed: 113, vocabSize: vocabSize),
                maxTokens: 12),
        ]
        var expected: [CBv2SchedCollected] = []
        for item in requests { expected.append(try await baseline(fixture, item)) }

        let engine = try makeEngine(
            fixture, mtp: true, maxSpeculativeBatch: 4, prefillChunkSize: 8)
        async let a = run(engine, requests[0])
        async let b = run(engine, requests[1])
        async let c = run(engine, requests[2])
        let tuple = try await (a, b, c)
        let values = [tuple.0, tuple.1, tuple.2]
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()

        for index in values.indices { #expect(values[index].tokens == expected[index].tokens) }
        #expect(metrics.rounds > 0)
        #expect(engine.preemptionCount == 0)
    }

    @Test func tightSpeculativeReservationFallsBackWithoutPreemption() async throws {
        let fixture = try makeFixture()
        let promptLength = 20
        let maxTokens = 4
        let prompt = makePromptTokens(length: promptLength, seed: 221, vocabSize: vocabSize)
        let item = request(id: 1, prompt: prompt, maxTokens: maxTokens)
        let expected = try await baseline(fixture, item)
        let probe = admissionProbe(fixture)
        let dummyTokens = 64
        let dummyBytes = probe.estimatedBytes(forTokens: dummyTokens)
        let capacity =
            dummyBytes + probe.estimatedBytes(forTokens: promptLength + maxTokens - 1)

        let engine = try makeEngine(
            fixture, mtp: true, bytesCapacity: capacity,
            admissionConfig: .init(watermarkFraction: 0))
        try engine.admissionForTesting.reserve(
            id: CBv2RequestID(0xDEAD), additionalTokens: dummyTokens)
        let value = try await run(engine, item)
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()

        #expect(value.tokens == expected.tokens)
        #expect(engine.preemptionCount == 0)
        #expect(metrics.rounds == 0)
        #expect(metrics.controllerFallbacks["step_kv_headroom", default: 0] > 0)
    }

    @Test func stepTokenBudgetFallbackIsDistinguishedFromKVHeadroom() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 20, seed: 222, vocabSize: vocabSize)
        let expected = try await baseline(
            fixture, request(id: 1, prompt: prompt, maxTokens: 8))
        let kinds = fixture.target.cbv2LayerKinds
        let drafter = try Gemma4CBv2MTPDrafter(
            drafter: fixture.drafter, target: fixture.target)
        let engine = EngineV2(
            model: CBv2SteppableLanguageModelAdapter(fixture.target),
            layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 2,
                prefillChunkSize: 2, maxWaiting: 4),
            compiledDecodeConfig: .init(enabled: false),
            mtpDrafter: drafter,
            mtpConfig: makeMTPConfig(enabled: true, maxSpeculativeBatch: 1))
        let value = try await run(
            engine, request(id: 1, prompt: prompt, maxTokens: 8))
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()

        #expect(value.tokens == expected.tokens)
        #expect(metrics.controllerFallbacks["step_token_budget", default: 0] > 0)
        #expect(metrics.controllerFallbacks["step_kv_headroom", default: 0] == 0)
    }

    @Test func capacityShrinkDuringRoundsStaysExact() async throws {
        let fixture = try makeFixture()
        let promptA = makePromptTokens(length: 18, seed: 331, vocabSize: vocabSize)
        let promptB = makePromptTokens(length: 14, seed: 332, vocabSize: vocabSize)
        let requestA = request(id: 1, prompt: promptA, maxTokens: 36)
        let requestB = request(id: 2, prompt: promptB, maxTokens: 32)
        let expectedA = try await baseline(fixture, requestA)
        let expectedB = try await baseline(fixture, requestB)
        let probe = admissionProbe(fixture)
        let shrunk = 2 * (
            probe.estimatedBytes(forTokens: promptA.count + requestA.maxTokens)
                + probe.estimatedBytes(forTokens: promptB.count + requestB.maxTokens))

        let engine = try makeEngine(fixture, mtp: true)
        async let b = run(engine, requestB)
        let stream = try engine.submit(requestA)
        var valueA = CBv2SchedCollected()
        var didShrink = false
        for await event in stream {
            switch event {
            case .delta(_, let tokens, _):
                valueA.tokens.append(contentsOf: tokens)
                if !didShrink, valueA.tokens.count >= 6 {
                    didShrink = true
                    engine.updateKVBytesCapacity(shrunk)
                }
            case .finished(let reason, let usage):
                valueA.finishReason = reason
                valueA.usage = usage
            }
        }
        let valueB = try await b
        await engine.shutdown()
        #expect(didShrink)
        #expect(valueA.tokens == expectedA.tokens)
        #expect(valueB.tokens == expectedB.tokens)
        #expect(engine.preemptionCount == 0)
    }

    @Test func samplingAndLogprobRowsStayPlain() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 16, seed: 441, vocabSize: vocabSize)

        let temperature = try makeEngine(fixture, mtp: true)
        let sampled = try await run(
            temperature,
            request(
                id: 1, prompt: prompt, maxTokens: 12,
                temperature: 0.7, seed: 42))
        let sampledMetrics = try #require(temperature.mtpMetricsSnapshot())
        await temperature.shutdown()
        #expect(sampled.finishReason == .length)
        #expect(sampledMetrics.rounds == 0)
        #expect(sampledMetrics.seedSteps == 0)

        let logprobs = try makeEngine(fixture, mtp: true)
        let stream = try logprobs.submit(
            request(id: 2, prompt: prompt, maxTokens: 12, topLogprobs: 3))
        var tokens: [Int] = []
        var allHaveLogprobs = true
        for await event in stream {
            if case .delta(_, let delta, let reports) = event {
                tokens.append(contentsOf: delta)
                allHaveLogprobs = allHaveLogprobs
                    && reports?.count == delta.count
                    && reports?.allSatisfy { $0.topLogprobs.count == 3 } == true
            }
        }
        let logprobMetrics = try #require(logprobs.mtpMetricsSnapshot())
        await logprobs.shutdown()
        #expect(tokens.count == 12)
        #expect(allHaveLogprobs)
        #expect(logprobMetrics.rounds == 0)
    }

    @Test func mixedEligibleAndLogprobRowsKeepWholeDecodeBatchPlain() async throws {
        let fixture = try makeFixture()
        let promptA = makePromptTokens(length: 20, seed: 451, vocabSize: vocabSize)
        let promptB = makePromptTokens(length: 20, seed: 452, vocabSize: vocabSize)
        let greedy = request(id: 1, prompt: promptA, maxTokens: 16)
        let withLogprobs = request(
            id: 2, prompt: promptB, maxTokens: 16, topLogprobs: 2)
        let expectedA = try await baseline(fixture, greedy)
        let expectedB = try await baseline(fixture, withLogprobs)

        let engine = try makeEngine(fixture, mtp: true, maxSpeculativeBatch: 2)
        // Submit both rows before consuming either stream so every decode plan
        // sees the same mixed eligibility as target-only batching.
        let streamA = try engine.submit(greedy)
        let streamB = try engine.submit(withLogprobs)
        async let valueA = cbv2SchedCollect(streamA)
        async let valueB = cbv2SchedCollect(streamB)
        let values = await (valueA, valueB)
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()

        #expect(values.0.tokens == expectedA.tokens)
        #expect(values.1.tokens == expectedB.tokens)
        #expect(metrics.rounds == 0)
        #expect(metrics.controllerFallbacks["ineligible", default: 0] > 0)
    }

    @Test func compiledPlainDecodeCoexistsWithEagerMTPRounds() async throws {
        let fixture = try makeFixture()
        let plainPrompt = makePromptTokens(length: 16, seed: 551, vocabSize: vocabSize)
        let mtpPrompt = makePromptTokens(length: 20, seed: 552, vocabSize: vocabSize)
        let mtpRequest = request(id: 2, prompt: mtpPrompt, maxTokens: 24)
        let expected = try await baseline(fixture, mtpRequest)
        let engine = try makeEngine(fixture, mtp: true, compiledEnabled: true)

        _ = try await run(
            engine,
            request(
                id: 1, prompt: plainPrompt, maxTokens: 16,
                temperature: 0.7, seed: 7))
        #expect(engine.compiledDecodeStats?.compiledSteps ?? 0 > 0)
        let value = try await run(engine, mtpRequest)
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()
        #expect(value.tokens == expected.tokens)
        #expect(metrics.rounds > 0)
    }

    @Test func rejectedMTPBindingsReleaseBeforeCompiledOnlyStretch() async throws {
        let fixture = try makeFixture(deterministicTarget: true)
        let provider = MTPReleaseTrackingCacheProvider(
            layerKinds: fixture.target.cbv2LayerKinds)
        let engine = try makeEngine(
            fixture, mtp: true, prefillChunkSize: 8, compiledEnabled: true,
            cacheProvider: provider, eventBufferCapacity: 16,
            mtpDrafter: MTPRejectingDrafter(target: fixture.target))
        provider.observeStaleness { [weak engine] in
            engine?.loopForTesting.eagerCompositionStale ?? false
        }
        let greedy = request(
            id: 1,
            prompt: makePromptTokens(length: 16, seed: 0xB01, vocabSize: vocabSize),
            maxTokens: 64)
        let ineligible = request(
            id: 2,
            prompt: makePromptTokens(length: 64, seed: 0xB02, vocabSize: vocabSize),
            maxTokens: 64, temperature: 0.7, seed: 9)
        let greedyStream = try engine.submit(greedy)
        let ineligibleStream = try engine.submit(ineligible)

        let released = await cbv2SchedWait(timeoutSeconds: 3) {
            let metrics = engine.mtpMetricsSnapshot()
            return (metrics?.rounds ?? 0) > 0
                && metrics?.acceptedTokens == 0
                && (engine.compiledDecodeStats?.compiledSteps ?? 0) > 0
                && provider.releasesWhileBoundAndStale > 0
        }

        engine.cancel(greedy.id)
        engine.cancel(ineligible.id)
        async let greedyResult = cbv2SchedCollect(greedyStream)
        async let ineligibleResult = cbv2SchedCollect(ineligibleStream)
        _ = await (greedyResult, ineligibleResult)
        await engine.shutdown()

        #expect(
            released,
            "compiled decode must release eager rows retained by a rejected MTP round")
    }

    @Test func earlyDonationOnUnsafeHybridFallsBackToFullReplay() async throws {
        let fixture = try makeFixture()
        // The donated prompt matches, but this interleaved hybrid has a
        // storage-owning full layer downstream of sliding attention. Partial
        // adoption would permanently cache replay-boundary pollution, so the
        // safe policy recomputes the whole matched prefix.
        let prompt = makePromptTokens(length: 129, seed: 660, vocabSize: vocabSize)
        let cache = PrefixCacheV2(
            config: .init(blockSize: 8, modelName: "mtp-prefix-confirmed"))
        let engine = try makeEngine(
            fixture, mtp: true, prefillChunkSize: 16,
            prefixCache: cache, earlyPrefixDonation: true)

        let donor = request(id: 1, prompt: prompt, maxTokens: 128)
        let stream = try engine.submit(donor)
        var iterator = stream.makeAsyncIterator()
        guard case .delta? = await iterator.next() else {
            await engine.shutdown()
            Issue.record("MTP donor must reach its first confirmed token")
            return
        }
        let donated = await cbv2SchedWait { cache.stats().entryCount > 0 }
        #expect(donated, "early donation must materialize confirmed prompt KV")
        engine.cancel(donor.id)
        while let event = await iterator.next() {
            if case .finished = event { break }
        }

        let adopted = try await run(
            engine, request(id: 2, prompt: prompt, maxTokens: 8))
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()
        #expect(adopted.finishReason == .length)
        #expect(adopted.usage?.prefixCacheOutcome == .skippedPolicy)
        #expect(adopted.usage?.prefixCacheMatchedTokens == 128)
        #expect(adopted.usage?.prefixCachePrefillTokensSaved == 0)
        #expect(adopted.usage?.prefixCacheHitTokens == 0)
        #expect(metrics.rounds > 0)
    }

    @Test func terminalDonationAfterSynchronizedRollbackReplaysExactly() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 129, seed: 662, vocabSize: vocabSize)
        let cache = PrefixCacheV2(
            config: .init(blockSize: 8, modelName: "mtp-terminal-confirmed"))
        let engine = try makeEngine(
            fixture, mtp: true, prefillChunkSize: 16, prefixCache: cache)

        let cold = try await run(
            engine, request(id: 1, prompt: prompt, maxTokens: 12))
        let donated = await cbv2SchedWait { cache.stats().entryCount > 0 }
        #expect(donated)
        let warm = try await run(
            engine, request(id: 2, prompt: prompt, maxTokens: 12))
        await engine.shutdown()

        #expect(warm.usage?.prefixCacheOutcome == .skippedPolicy)
        #expect(warm.usage?.prefixCacheMatchedTokens ?? 0 >= 128)
        #expect(warm.usage?.prefixCachePrefillTokensSaved == 0)
        #expect(warm.usage?.prefixCacheHitTokens == 0)
        #expect(warm.tokens == cold.tokens)
    }

    @Test func quantizedFullKVRollbackMatchesQuantizedTargetOnly() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 24, seed: 665, vocabSize: vocabSize)
        let item = request(id: 1, prompt: prompt, maxTokens: 20)

        func backend() -> CBv2ContiguousKVBackend {
            CBv2ContiguousKVBackend(
                config: .init(
                    bytesCapacity: 1 << 28,
                    quantization: (groupSize: 32, bits: 8)))
        }

        let off = try makeEngine(fixture, mtp: false, backend: backend())
        let expected = try await run(off, item)
        await off.shutdown()

        let on = try makeEngine(fixture, mtp: true, backend: backend())
        let value = try await run(on, item)
        let metrics = try #require(on.mtpMetricsSnapshot())
        await on.shutdown()
        #expect(value.tokens == expected.tokens)
        #expect(metrics.rounds > 0)
    }

    @Test func unsupportedKVRowsFailOpenAtTheEngineStorageGate() {
        let supported = MTPTestSequenceKV(supportsSpeculativeWrites: true)
        let unsupported = MTPTestSequenceKV(supportsSpeculativeWrites: false)
        #expect(EngineLoopV2.mtpStorageEligible([supported, nil]))
        #expect(!EngineLoopV2.mtpStorageEligible([supported, unsupported]))
    }
}
