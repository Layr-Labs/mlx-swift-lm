// CBv2MTPEngineParityTests.swift
//
// Token authority through the real EngineV2. The adversarial drafter proves
// that the first mismatch emits the target correction, never the draft.

import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import Testing

@testable import MLXLLM

private final class CBv2ParityScriptCursor: CBv2MTPPreparedCapture {
    let baseIndices: [Int]
    var step = 0

    init(baseIndices: [Int]) { self.baseIndices = baseIndices }
}

/// Position-keyed deterministic drafter. offset 0 is an oracle; offset 1 is
/// guaranteed to disagree with the probed target continuation.
private final class CBv2ParityScriptedDrafter: CBv2MTPDrafter {
    private let script: [Int]
    private let promptLength: Int
    private let offset: Int
    private let vocabSize: Int
    let mtpTargetIdentity: ObjectIdentifier?

    init(
        script: [Int], promptLength: Int, offset: Int, vocabSize: Int,
        target: Gemma4TextModel
    ) {
        self.script = script
        self.promptLength = promptLength
        self.offset = offset
        self.vocabSize = vocabSize
        self.mtpTargetIdentity = ObjectIdentifier(target)
    }

    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        CBv2ParityScriptCursor(
            baseIndices: rows.map { $0.anchor - promptLength + 1 })
    }

    func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        let cursor = prepared as! CBv2ParityScriptCursor
        defer { cursor.step += 1 }
        let ids = cursor.baseIndices.map { base -> Int32 in
            let index = base + cursor.step
            let value = index < script.count ? script[index] : 0
            return Int32((value + offset) % vocabSize)
        }
        return (MLXArray(ids), hidden)
    }
}

private final class CBv2ParityConstantSampler: CBv2StepSampler {
    func sample(
        logits: MLXArray, params: [CBv2SamplingParams], requestIDs: [CBv2RequestID],
        stepIndex: Int, pendingSampledTokens: MLXArray?,
        rowContext: () -> [CBv2SamplerRow]
    ) -> MLXArray {
        MLXArray.zeros([params.count], dtype: .int32)
    }
}

@Suite("CBv2MTPEngineParity", .serialized)
struct CBv2MTPEngineParityTests {
    private let vocabSize = 256
    private let hiddenSize = 64
    private let slidingWindow = 16
    private let k = 2

    private func targetConfig() throws -> Gemma4TextConfiguration {
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
                "tie_word_embeddings": true,
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

    private func makeFixture(seed: UInt64 = 0x9A7E) throws -> Fixture {
        MLXRandom.seed(seed)
        let target = Gemma4TextModel(try targetConfig())
        let drafter = try Gemma4AssistantDraftModel(config: drafterConfig())
        eval(target, drafter)
        return Fixture(target: target, drafter: drafter)
    }

    private func makeEngine(
        target: Gemma4TextModel, drafter: (any CBv2MTPDrafter)?,
        maxSpeculativeBatch: Int = 4, maxConcurrent: Int = 4,
        fixedDepth: Int? = nil,
        sampler: CBv2StepSampler = CBv2DefaultSampler()
    ) -> EngineV2 {
        let kinds = target.cbv2LayerKinds
        let depth = fixedDepth ?? k
        return EngineV2(
            model: CBv2SteppableLanguageModelAdapter(target),
            layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            sampler: sampler,
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: maxConcurrent, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 16),
            compiledDecodeConfig: CBv2CompiledDecodeConfig(enabled: false),
            mtpDrafter: drafter,
            mtpConfig: CBv2MTPConfig(
                enabled: drafter != nil, maxDraftTokens: depth,
                maxSpeculativeBatch: maxSpeculativeBatch,
                fixedDraftTokens: depth))
    }

    private func realDrafter(_ fixture: Fixture) throws -> Gemma4CBv2MTPDrafter {
        try Gemma4CBv2MTPDrafter(drafter: fixture.drafter, target: fixture.target)
    }

    private func request(id: UInt64, prompt: [Int], maxTokens: Int) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt,
            sampling: CBv2SamplingParams(temperature: 0), maxTokens: maxTokens)
    }

    private func run(
        _ engine: EngineV2, _ request: CBv2Request
    ) async throws -> CBv2SchedCollected {
        await cbv2SchedCollect(try engine.submit(request))
    }

    private func baseline(
        _ fixture: Fixture, prompt: [Int], maxTokens: Int, id: UInt64 = 1
    ) async throws -> CBv2SchedCollected {
        let engine = makeEngine(target: fixture.target, drafter: nil)
        let result = try await run(engine, request(id: id, prompt: prompt, maxTokens: maxTokens))
        await engine.shutdown()
        return result
    }

    private func expectAccounting(_ metrics: CBv2MTPMetrics) {
        #expect(metrics.active)
        #expect((0 ... k).contains(metrics.selectedDepth))
        #expect(metrics.depthSelections[k, default: 0] > 0)
        #expect(metrics.proposedTokens == metrics.draftedTokens)
        #expect(metrics.draftedTokens >= metrics.rounds)
        #expect(metrics.draftedTokens <= metrics.rounds * k)
        #expect(metrics.acceptedTokens <= metrics.draftedTokens)
        #expect(metrics.acceptedTokens <= metrics.emittedTokens)
        #expect(metrics.emittedTokens >= metrics.rounds)
        #expect(metrics.emittedTokens <= metrics.acceptedTokens + metrics.rounds)
        #expect(metrics.perPositionAccepted.reduce(0, +) == metrics.acceptedTokens)
        for position in metrics.perPositionAccepted.indices.dropFirst() {
            #expect(
                metrics.perPositionAccepted[position]
                    <= metrics.perPositionAccepted[position - 1])
        }
    }

    @Test func b1ParityAcrossRaggedPromptLengths() async throws {
        let fixture = try makeFixture()
        for (index, length) in [5, 13, 24, 41].enumerated() {
            let prompt = makePromptTokens(
                length: length, seed: UInt64(100 + index), vocabSize: vocabSize)
            let off = try await baseline(fixture, prompt: prompt, maxTokens: 24)
            let engine = makeEngine(
                target: fixture.target, drafter: try realDrafter(fixture))
            let on = try await run(engine, request(id: 1, prompt: prompt, maxTokens: 24))
            let metrics = try #require(engine.mtpMetricsSnapshot())
            await engine.shutdown()

            #expect(on.tokens == off.tokens, "prompt length \(length)")
            #expect(metrics.rounds > 0)
            expectAccounting(metrics)
        }
    }

    @Test func b2AndB4RaggedBatchesMatchSoloTargetDecode() async throws {
        let fixture = try makeFixture()
        for batchSize in [2, 4] {
            let prompts = (0 ..< batchSize).map { row in
                makePromptTokens(
                    length: 9 + row * 7, seed: UInt64(200 + batchSize * 10 + row),
                    vocabSize: vocabSize)
            }
            var expected: [CBv2SchedCollected] = []
            for (row, prompt) in prompts.enumerated() {
                expected.append(
                    try await baseline(
                        fixture, prompt: prompt, maxTokens: 20, id: UInt64(row + 1)))
            }

            let engine = makeEngine(
                target: fixture.target, drafter: try realDrafter(fixture),
                maxSpeculativeBatch: 4)
            let streams = try prompts.enumerated().map { row, prompt in
                try engine.submit(
                    request(id: UInt64(row + 1), prompt: prompt, maxTokens: 20))
            }
            var results: [CBv2SchedCollected] = []
            for stream in streams { results.append(await cbv2SchedCollect(stream)) }
            let metrics = try #require(engine.mtpMetricsSnapshot())
            await engine.shutdown()
            for row in prompts.indices {
                #expect(results[row].tokens == expected[row].tokens, "B=\(batchSize), row=\(row)")
            }
            #expect(metrics.rounds > 0)
            expectAccounting(metrics)
        }
    }

    @Test func perfectDrafterHasExactAcceptanceAccounting() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 14, seed: 401, vocabSize: vocabSize)
        let probe = try await baseline(fixture, prompt: prompt, maxTokens: 20)
        let maxTokens = 14
        let engine = makeEngine(
            target: fixture.target,
            drafter: CBv2ParityScriptedDrafter(
                script: probe.tokens, promptLength: prompt.count, offset: 0,
                vocabSize: vocabSize, target: fixture.target))
        let on = try await run(engine, request(id: 1, prompt: prompt, maxTokens: maxTokens))
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()

        #expect(on.tokens == Array(probe.tokens.prefix(maxTokens)))
        let expectedRounds = (maxTokens - 2) / (k + 1)
        #expect(metrics.rounds == expectedRounds)
        #expect(metrics.acceptedTokens == metrics.draftedTokens)
        #expect(metrics.emittedTokens == metrics.acceptedTokens + metrics.rounds)
        #expect(metrics.perPositionAccepted == Array(repeating: expectedRounds, count: k))
        expectAccounting(metrics)
    }

    @Test func adversarialDrafterEmitsTargetCorrection() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 21, seed: 402, vocabSize: vocabSize)
        let probe = try await baseline(fixture, prompt: prompt, maxTokens: 20)
        let maxTokens = 12
        let engine = makeEngine(
            target: fixture.target,
            drafter: CBv2ParityScriptedDrafter(
                script: probe.tokens, promptLength: prompt.count, offset: 1,
                vocabSize: vocabSize, target: fixture.target))
        let on = try await run(engine, request(id: 1, prompt: prompt, maxTokens: maxTokens))
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()

        #expect(
            on.tokens == Array(probe.tokens.prefix(maxTokens)),
            "a wrong draft must be replaced by the target correction")
        #expect(metrics.acceptedTokens == 0)
        #expect(metrics.perPositionAccepted == Array(repeating: 0, count: k))
        #expect(metrics.emittedTokens == metrics.rounds)
        expectAccounting(metrics)
    }

    @Test func fullAcceptanceTailKeepsWholeB4TargetBatchPlain() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 18, seed: 490, vocabSize: vocabSize)
        let targetOnly = makeEngine(target: fixture.target, drafter: nil)
        async let offA = run(targetOnly, request(id: 11, prompt: prompt, maxTokens: 16))
        async let offB = run(targetOnly, request(id: 12, prompt: prompt, maxTokens: 16))
        async let offC = run(targetOnly, request(id: 13, prompt: prompt, maxTokens: 16))
        async let offD = run(targetOnly, request(id: 14, prompt: prompt, maxTokens: 16))
        let targetValues = try await (offA, offB, offC, offD)
        await targetOnly.shutdown()
        let script = targetValues.0.tokens
        let engine = makeEngine(
            target: fixture.target,
            drafter: CBv2ParityScriptedDrafter(
                script: script, promptLength: prompt.count, offset: 0,
                vocabSize: vocabSize, target: fixture.target),
            fixedDepth: 4)

        async let a = run(engine, request(id: 1, prompt: prompt, maxTokens: 8))
        async let b = run(engine, request(id: 2, prompt: prompt, maxTokens: 8))
        async let c = run(engine, request(id: 3, prompt: prompt, maxTokens: 8))
        async let d = run(engine, request(id: 4, prompt: prompt, maxTokens: 8))
        let values = try await (a, b, c, d)
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()

        #expect(values.0.tokens == Array(targetValues.0.tokens.prefix(8)))
        #expect(values.1.tokens == Array(targetValues.1.tokens.prefix(8)))
        #expect(values.2.tokens == Array(targetValues.2.tokens.prefix(8)))
        #expect(values.3.tokens == Array(targetValues.3.tokens.prefix(8)))
        #expect(metrics.controllerFallbacks["tail_depth", default: 0] > 0)
    }

    @Test func customSamplerKeepsTokenAuthorityByDisablingMTP() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 14, seed: 491, vocabSize: vocabSize)
        let item = request(id: 1, prompt: prompt, maxTokens: 8)

        let targetOnly = makeEngine(
            target: fixture.target, drafter: nil,
            sampler: CBv2ParityConstantSampler())
        let expected = try await run(targetOnly, item)
        await targetOnly.shutdown()

        let requestedMTP = makeEngine(
            target: fixture.target, drafter: try realDrafter(fixture),
            sampler: CBv2ParityConstantSampler())
        #expect(requestedMTP.mtpMetricsSnapshot() == nil)
        let actual = try await run(requestedMTP, item)
        await requestedMTP.shutdown()

        #expect(expected.tokens == Array(repeating: 0, count: item.maxTokens))
        #expect(actual.tokens == expected.tokens)
    }
}
