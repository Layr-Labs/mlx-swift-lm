// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXSpeculative
import Testing

@Suite("Gemma4MTPRoundLoop B=1")
struct Gemma4MTPRoundLoopTests {

    /// Minimal Gemma 4 target: 10 layers, 5 kv-shared, hidden=64, vocab=64.
    /// Matches the shapes the drafter expects in its forward.
    private func smallTarget() throws -> Gemma4TextModel {
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": 64,
            "num_hidden_layers": 10,
            "intermediate_size": 128,
            "num_attention_heads": 2,
            "head_dim": 32,
            "global_head_dim": 32,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 5,
            "sliding_window": 64,
            "sliding_window_pattern": 5,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "vocab_size": 64,
            "vocab_size_per_layer_input": 64,
            "rms_norm_eps": 1e-6,
            "hidden_size_per_layer_input": 0
        }
        """
        let data = Data(json.utf8)
        let config = try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: data)
        return Gemma4TextModel(config)
    }

    /// Minimal drafter config matching the target's hidden_size and vocab.
    private func smallDrafter() throws -> Gemma4AssistantDraftModel {
        let json = """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 64,
            "use_ordered_embeddings": false,
            "num_centroids": 8,
            "centroid_intermediate_top_k": 2,
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
                "vocab_size": 64,
                "vocab_size_per_layer_input": 64,
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false,
                "layer_types": ["sliding_attention", "sliding_attention",
                                "sliding_attention", "full_attention"]
            }
        }
        """
        let data = Data(json.utf8)
        let cfg = try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: data)
        return Gemma4AssistantDraftModel(config: cfg)
    }

    /// Run prefill on the target to produce firstBonus/firstHidden/firstSharedKV
    /// for a given prompt. Drives the target's forwardForMTP over the prompt.
    private func prefill(
        target: Gemma4TextModel, promptTokens: [Int32]
    ) -> (firstBonus: Int, firstHidden: MLXArray, firstSharedKV: Gemma4SharedKV,
          cache: [KVCache]) {
        let tokens = MLXArray(promptTokens)[.newAxis, .ellipsis]  // [1, L]
        let cache = target.newCache(parameters: nil)
        let out = target.forwardForMTP(tokens, cache: cache)
        // Greedy bonus from the last position.
        let lastLogits = out.logits[0..., -1, 0...]
        let bonus = Int(lastLogits.argMax(axis: -1).item(Int32.self))
        // Last-position hidden (the drafter's next-step input).
        let lastHidden = out.lastHidden[0..., -1 ..< out.lastHidden.dim(1), 0...]
        return (bonus, lastHidden, out.capturedSharedKV, cache)
    }

    @Test func mtpEmitsTokensAndTerminates() async throws {
        let target = try smallTarget()
        let drafter = try smallDrafter()
        eval(target, drafter)

        let prompt: [Int32] = [1, 2, 3, 4, 5]
        let prefillResult = prefill(target: target, promptTokens: prompt)

        // Collect generated tokens as ints for easy inspection.
        var produced: [Int] = []
        let stream = try runGemma4MTPRounds(
            target: target,
            drafter: drafter,
            targetCache: prefillResult.cache,
            firstBonus: prefillResult.firstBonus,
            firstHidden: prefillResult.firstHidden,
            firstSharedKV: prefillResult.firstSharedKV,
            maxTokens: 12,
            blockSize: 3
        )
        for try await gen in stream {
            if case .chunk(let s) = gen, let tok = Int(s) { produced.append(tok) }
        }
        // Should emit at least the first bonus + some more tokens, up to
        // the maxTokens cap. Exact count depends on acceptance rate.
        #expect(produced.count >= 1)
        #expect(produced.count <= 12)
    }

    @Test func mtpRespectsMaxTokens() async throws {
        let target = try smallTarget()
        let drafter = try smallDrafter()
        eval(target, drafter)

        let prefillResult = prefill(target: target, promptTokens: [1, 2, 3])

        var count = 0
        let stream = try runGemma4MTPRounds(
            target: target,
            drafter: drafter,
            targetCache: prefillResult.cache,
            firstBonus: prefillResult.firstBonus,
            firstHidden: prefillResult.firstHidden,
            firstSharedKV: prefillResult.firstSharedKV,
            maxTokens: 5,
            blockSize: 3
        )
        for try await gen in stream {
            if case .chunk = gen { count += 1 }
        }
        #expect(count <= 5)
    }
}
