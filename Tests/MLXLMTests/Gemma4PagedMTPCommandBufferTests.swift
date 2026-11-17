// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLLM
@_spi(Benchmarking) @testable import MLXLMCommon

/// Gemma 4's MTP rounds run under its op-count command buffers
/// (`speculativeCommandBufferLimits`) on the paged KV backend that production
/// uses, so a round's verification writes and the reads after them share a
/// command buffer. Serial verification stays token-exact with the MTP-off
/// engine; rectangular verification (production's `[B, 2]` window, written in
/// bulk into segments of two pages, so each row spans several) gives the same
/// tokens with and without the limits.
@Suite("Gemma 4 paged MTP rounds under op-count command buffers", .serialized)
struct Gemma4PagedMTPCommandBufferTests {
    private let vocabSize = 256
    private let hiddenSize = 64
    private let slidingWindow = 16
    // The paged kernels take head dims 64, 128, 256 and 512.

    private func targetConfig() throws -> Gemma4TextConfiguration {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": \(hiddenSize),
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "head_dim": 64,
                "global_head_dim": 64,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 0,
                "layer_types": ["sliding_attention", "full_attention",
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
                    "head_dim": 64,
                    "global_head_dim": 64,
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

    private func makeEngine(
        target: Gemma4TextModel, drafter: Gemma4AssistantDraftModel?,
        verification: CBv2MTPVerificationMode = .serialTarget, segmentBytes: Int? = nil
    ) throws -> EngineV2 {
        let kinds = target.cbv2LayerKinds
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: PagedKVPoolConfig(
                capacityBytes: 64 << 20, dtype: .float32, maxPrefillChunk: 16,
                nominalMaxSequenceLength: 512, segmentSizeBytes: segmentBytes))
        let mtpDrafter = try drafter.map {
            try Gemma4CBv2MTPDrafter(drafter: $0, target: target)
        }
        return EngineV2(
            model: CBv2SteppableLanguageModelAdapter(target),
            layerKinds: kinds,
            backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 16),
            mtpDrafter: mtpDrafter,
            mtpConfig: CBv2MTPConfig(
                enabled: drafter != nil, maxDraftTokens: 1, maxSpeculativeBatch: 4,
                fixedDraftTokens: 1, verificationMode: verification,
                maxAutomaticRectangularTokens: 8))
    }

    private func greedy(id: UInt64, prompt: [Int], maxTokens: Int) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt,
            sampling: CBv2SamplingParams(temperature: 0), maxTokens: maxTokens)
    }

    @Test func greedyRoundsStayTokenExact() async throws {
        MLXRandom.seed(0x6E44)
        let target = Gemma4TextModel(try targetConfig())
        let drafter = try Gemma4AssistantDraftModel(config: drafterConfig())
        eval(target, drafter)
        let requested = ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_MTP_COMMAND_BUFFERS"]
        let limits = CBv2SteppableLanguageModelAdapter(target).speculativeCommandBufferLimits
        #expect(
            limits
                == (Gemma4DecodeCommandBuffers.speculativeEnabled(requested)
                    ? Gemma4DecodeCommandBuffers.limits : nil))
        // Prompts longer than the window and generations that wrap it again.
        let prompts = [
            makePromptTokens(length: 24, seed: 41, vocabSize: vocabSize),
            makePromptTokens(length: 40, seed: 42, vocabSize: vocabSize),
        ]
        var baselines: [[Int]] = []
        for (index, prompt) in prompts.enumerated() {
            let off = try makeEngine(target: target, drafter: nil)
            let out = await cbv2SchedCollect(
                try off.submit(greedy(id: UInt64(index + 1), prompt: prompt, maxTokens: 48)))
            await off.shutdown()
            #expect(out.tokens.count == 48)
            baselines.append(out.tokens)
        }
        for (index, prompt) in prompts.enumerated() {
            let on = try makeEngine(target: target, drafter: drafter)
            let out = await cbv2SchedCollect(
                try on.submit(greedy(id: UInt64(index + 1), prompt: prompt, maxTokens: 48)))
            let metrics = try #require(on.mtpMetricsSnapshot())
            await on.shutdown()
            #expect(metrics.rounds >= 1, "no MTP round ran")
            #expect(
                out.tokens == baselines[index],
                "MTP-on output diverged: on=\(out.tokens) off=\(baselines[index])")
        }
    }

    /// Production verifies a `[B, 2]` window (rectangular), which is not
    /// M=1-exact against plain decode, so the oracle is the same rounds with
    /// MLX's default command buffers: the partitioning must not change a token.
    @Test func rectangularRoundsGiveTheSameTokensWithAndWithoutTheLimits() async throws {
        MLXRandom.seed(0x6E45)
        let target = Gemma4TextModel(try targetConfig())
        let drafter = try Gemma4AssistantDraftModel(config: drafterConfig())
        eval(target, drafter)
        defer { Gemma4DecodeCommandBuffers.speculativeOverrideForTesting = nil }
        let prompts = [
            makePromptTokens(length: 24, seed: 51, vocabSize: vocabSize),
            makePromptTokens(length: 40, seed: 52, vocabSize: vocabSize),
        ]
        var outputs: [Bool: [[Int]]] = [:]
        for limited in [false, true] {
            Gemma4DecodeCommandBuffers.speculativeOverrideForTesting = limited
            let adapter = CBv2SteppableLanguageModelAdapter(target)
            #expect((adapter.speculativeCommandBufferLimits != nil) == limited)
            var tokens: [[Int]] = []
            for (index, prompt) in prompts.enumerated() {
                let engine = try makeEngine(
                    target: target, drafter: drafter, verification: .rectangular,
                    segmentBytes: 2 * 16 * 64 * 4 * 2)
                let out = await cbv2SchedCollect(
                    try engine.submit(
                        greedy(id: UInt64(index + 1), prompt: prompt, maxTokens: 48)))
                let metrics = try #require(engine.mtpMetricsSnapshot())
                await engine.shutdown()
                #expect(metrics.rounds >= 1, "no MTP round ran")
                #expect(out.tokens.count == 48)
                tokens.append(out.tokens)
            }
            outputs[limited] = tokens
        }
        #expect(outputs[true] == outputs[false])
    }
}
