// Copyright © 2026 Apple Inc.
//
// Smoke tests for the Gemma 4 MTP benchmark primitives. Verifies the
// measurement harness runs end-to-end, produces finite timings, and
// returns a token count matching `maxTokens`. Actual throughput numbers
// are hardware-dependent and are not asserted; a separate driver
// (outside the test target) collects them against real models.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXRandom
import MLXSpeculative
import Testing

@Suite("Gemma4MTPBenchmark")
struct Gemma4MTPBenchmarkTests {

    /// Tiny target for smoke tests — matches the E2B-shaped config used
    /// in the parity suite so we know the harness runs cleanly.
    private func tinyTargetConfig() throws -> Gemma4TextConfiguration {
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 256,
            "num_hidden_layers": 12,
            "intermediate_size": 512,
            "num_attention_heads": 4,
            "head_dim": 32,
            "global_head_dim": 32,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 6,
            "sliding_window": 128,
            "sliding_window_pattern": 5,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "vocab_size": 1024,
            "vocab_size_per_layer_input": 1024,
            "rms_norm_eps": 1e-6,
            "hidden_size_per_layer_input": 0
        }
        """
        return try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    private func tinyDrafterConfig() throws -> Gemma4AssistantConfiguration {
        let json = """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 256,
            "use_ordered_embeddings": false,
            "num_centroids": 32,
            "centroid_intermediate_top_k": 4,
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "head_dim": 32,
                "global_head_dim": 32,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 4,
                "sliding_window": 64,
                "final_logit_softcapping": null,
                "tie_word_embeddings": true,
                "vocab_size": 1024,
                "vocab_size_per_layer_input": 1024,
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false,
                "layer_types": ["sliding_attention", "sliding_attention",
                                "sliding_attention", "full_attention"]
            }
        }
        """
        return try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: Data(json.utf8))
    }

    @Test func baselineHarnessRunsAndProducesFiniteTimings() async throws {
        MLXRandom.seed(42)
        let target = Gemma4TextModel(try tinyTargetConfig())
        eval(target)
        let prompt = MLXArray([Int32](repeating: 7, count: 8))

        let r = measureBaselineThroughput(
            target: target, promptTokens: prompt, maxTokens: 8)

        #expect(r.generatedTokens == 8)
        #expect(r.prefillSeconds.isFinite && r.prefillSeconds >= 0)
        #expect(r.generationSeconds.isFinite && r.generationSeconds >= 0)
        #expect(r.acceptLengths == nil, "baseline has no accept histogram")
    }

    @Test func mtpHarnessRunsAndProducesAcceptLengths() async throws {
        MLXRandom.seed(42)
        let target = Gemma4TextModel(try tinyTargetConfig())
        let drafter = Gemma4AssistantDraftModel(config: try tinyDrafterConfig())
        eval(target, drafter)
        let prompt = MLXArray([Int32](repeating: 7, count: 8))

        let r = try await measureMTPThroughput(
            target: target, drafter: drafter,
            promptTokens: prompt, maxTokens: 12, blockSize: 3)

        #expect(r.generatedTokens == 12)
        #expect(r.prefillSeconds.isFinite && r.prefillSeconds >= 0)
        #expect(r.generationSeconds.isFinite && r.generationSeconds >= 0)
        #expect(r.acceptLengths != nil)
        if let a = r.acceptLengths {
            #expect(!a.isEmpty, "MTP must produce at least one round's accept count")
            // Each round emits accepted+1 tokens; sum must equal generated count.
            let emittedPerRound = a.map { $0 + 1 }
            #expect(
                emittedPerRound.reduce(0, +) == r.generatedTokens
                    || emittedPerRound.reduce(0, +) == r.generatedTokens + 1,
                "round accept lengths should reconstruct generated token count (±1 for bonus)"
            )
        }
    }
}
