// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXSpeculative
import Testing

@Suite("Gemma4AssistantDraftModel skeleton")
struct Gemma4AssistantDraftModelTests {

    /// Build a minimal drafter config (4 layers, all kv-shared, tied embeds,
    /// no centroid head).
    private func drafterConfig(
        backbone: Int = 64,
        vocab: Int = 64,
        useOrderedEmbeddings: Bool = false,
        attentionKeqV: Bool = false,
        numGlobalKVHeads: Int? = nil
    ) throws -> Gemma4AssistantConfiguration {
        let kvHeadsField = numGlobalKVHeads.map { "\"num_global_key_value_heads\": \($0)," } ?? ""
        let json = """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": \(backbone),
            "use_ordered_embeddings": \(useOrderedEmbeddings),
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
                \(kvHeadsField)
                "sliding_window": 64,
                "attention_k_eq_v": \(attentionKeqV),
                "final_logit_softcapping": null,
                "tie_word_embeddings": true,
                "vocab_size": \(vocab),
                "vocab_size_per_layer_input": \(vocab),
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false,
                "layer_types": ["sliding_attention", "sliding_attention",
                                "sliding_attention", "full_attention"]
            }
        }
        """
        let data = Data(json.utf8)
        return try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: data)
    }

    /// Build a matching small target config (same hidden_size = backbone,
    /// same vocab).
    private func targetConfig(
        hiddenSize: Int = 64,
        vocab: Int = 64,
        attentionKeqV: Bool = false,
        numGlobalKVHeads: Int? = nil
    ) throws -> Gemma4TextConfiguration {
        let kvHeadsField = numGlobalKVHeads.map { "\"num_global_key_value_heads\": \($0)," } ?? ""
        let json = """
        {
            "model_type": "gemma4_text",
            "hidden_size": \(hiddenSize),
            "num_hidden_layers": 10,
            "intermediate_size": 128,
            "num_attention_heads": 2,
            "head_dim": 32,
            "global_head_dim": 32,
            "num_key_value_heads": 1,
            \(kvHeadsField)
            "num_kv_shared_layers": 5,
            "sliding_window": 64,
            "sliding_window_pattern": 5,
            "attention_k_eq_v": \(attentionKeqV),
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
            "vocab_size": \(vocab),
            "vocab_size_per_layer_input": \(vocab),
            "rms_norm_eps": 1e-6,
            "hidden_size_per_layer_input": 0
        }
        """
        let data = Data(json.utf8)
        return try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: data)
    }

    @Test func drafterInitializes() throws {
        let cfg = try drafterConfig()
        let drafter = Gemma4AssistantDraftModel(config: cfg)
        eval(drafter)
        #expect(drafter.config.backboneHiddenSize == 64)
        #expect(drafter.maskedEmbedder == nil)     // useOrderedEmbeddings: false
    }

    @Test func drafterWithCentroidHead() throws {
        let cfg = try drafterConfig(useOrderedEmbeddings: true)
        let drafter = Gemma4AssistantDraftModel(config: cfg)
        eval(drafter)
        #expect(drafter.maskedEmbedder != nil)
    }

    @Test func bindSucceedsOnMatchedTarget() throws {
        let cfg = try drafterConfig()
        let drafter = Gemma4AssistantDraftModel(config: cfg)
        let tCfg = try targetConfig()
        let target = Gemma4TextModel(tCfg)
        eval(drafter, target)
        try drafter.bind(target: target)
    }

    @Test func bindIsIdempotentOnSameTarget() throws {
        let cfg = try drafterConfig()
        let drafter = Gemma4AssistantDraftModel(config: cfg)
        let target = Gemma4TextModel(try targetConfig())
        eval(drafter, target)
        try drafter.bind(target: target)
        try drafter.bind(target: target)   // second call is a no-op
    }

    @Test func rebindToDifferentTargetThrows() throws {
        let cfg = try drafterConfig()
        let drafter = Gemma4AssistantDraftModel(config: cfg)
        let t1 = Gemma4TextModel(try targetConfig())
        let t2 = Gemma4TextModel(try targetConfig())
        eval(drafter, t1, t2)
        try drafter.bind(target: t1)
        #expect(throws: Gemma4MTPError.rebindForbidden) {
            try drafter.bind(target: t2)
        }
    }

    @Test func hiddenSizeMismatchThrows() throws {
        // Drafter backbone = 64, target hidden = 128 → mismatch.
        let cfg = try drafterConfig(backbone: 64)
        let drafter = Gemma4AssistantDraftModel(config: cfg)
        let target = Gemma4TextModel(try targetConfig(hiddenSize: 128))
        eval(drafter, target)
        #expect(throws: (any Error).self) {
            try drafter.bind(target: target)
        }
    }

    @Test func vocabMismatchThrows() throws {
        let cfg = try drafterConfig(vocab: 64)
        let drafter = Gemma4AssistantDraftModel(config: cfg)
        let target = Gemma4TextModel(try targetConfig(vocab: 128))
        eval(drafter, target)
        #expect(throws: (any Error).self) {
            try drafter.bind(target: target)
        }
    }

    @Test func kEqVCompatCheckPasses() throws {
        // Both drafter and target have attention_k_eq_v = true + matching
        // num_global_key_value_heads.
        let cfg = try drafterConfig(
            attentionKeqV: true, numGlobalKVHeads: 2)
        let drafter = Gemma4AssistantDraftModel(config: cfg)
        let target = Gemma4TextModel(try targetConfig(
            attentionKeqV: true, numGlobalKVHeads: 2))
        eval(drafter, target)
        try drafter.bind(target: target)
    }

    @Test func kEqVMismatchThrows() throws {
        // Drafter has k_eq_v=true, target has k_eq_v=false → throw.
        let cfg = try drafterConfig(
            attentionKeqV: true, numGlobalKVHeads: 2)
        let drafter = Gemma4AssistantDraftModel(config: cfg)
        let target = Gemma4TextModel(try targetConfig(
            attentionKeqV: false))
        eval(drafter, target)
        #expect(throws: (any Error).self) {
            try drafter.bind(target: target)
        }
    }
}

@Suite("Gemma4AssistantDraftModel forward")
struct Gemma4AssistantDraftModelForwardTests {

    private func drafterConfig(
        useOrderedEmbeddings: Bool = false
    ) throws -> Gemma4AssistantConfiguration {
        let json = """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 64,
            "use_ordered_embeddings": \(useOrderedEmbeddings),
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
        return try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: data)
    }

    /// Build dummy shared-KV (K/V for each layer-type). Shapes:
    ///   fullAttention: [B, num_key_value_heads, T, global_head_dim]
    ///   slidingAttention: [B, num_key_value_heads, T, head_dim]
    /// Here: B=1, kvHeads=1, T=8, both head_dim = 32.
    private func makeSharedKV(kvLen: Int = 8) -> Gemma4SharedKV {
        let B = 1, H = 1, D = 32
        func kv() -> (MLXArray, MLXArray) {
            let k = MLXArray.ones([B, H, kvLen, D], dtype: .float32) * 0.1
            let v = MLXArray.ones([B, H, kvLen, D], dtype: .float32) * 0.2
            return (k, v)
        }
        return Gemma4SharedKV(fullAttention: kv(), slidingAttention: kv())
    }

    @Test func forwardProducesExpectedShapes() throws {
        let cfg = try drafterConfig()
        let drafter = Gemma4AssistantDraftModel(config: cfg)
        eval(drafter)
        let inputsEmbeds = MLXArray.zeros([1, 1, 2 * 64], dtype: .float32)
        let sharedKV = makeSharedKV()
        let (lastHidden, logits) = drafter(
            inputsEmbeds: inputsEmbeds,
            sharedKV: sharedKV,
            positionOffset: .scalar(8)
        )
        #expect(lastHidden.shape == [1, 1, 64])   // [B, 1, backbone]
        #expect(logits.shape == [1, 1, 64])       // [B, 1, vocab]
    }

    @Test func forwardWithCentroidHeadHasCorrectShapes() throws {
        let cfg = try drafterConfig(useOrderedEmbeddings: true)
        let drafter = Gemma4AssistantDraftModel(config: cfg)
        eval(drafter)
        let inputsEmbeds = MLXArray.zeros([1, 1, 2 * 64], dtype: .float32)
        let sharedKV = makeSharedKV()
        let (lastHidden, logits) = drafter(
            inputsEmbeds: inputsEmbeds,
            sharedKV: sharedKV,
            positionOffset: .scalar(8)
        )
        #expect(lastHidden.shape == [1, 1, 64])
        #expect(logits.shape == [1, 1, 64])
    }

    @Test func forwardLogitsAreFinite() throws {
        let cfg = try drafterConfig()
        let drafter = Gemma4AssistantDraftModel(config: cfg)
        eval(drafter)
        let inputsEmbeds = MLXArray.zeros([1, 1, 2 * 64], dtype: .float32)
        let sharedKV = makeSharedKV()
        let (_, logits) = drafter(
            inputsEmbeds: inputsEmbeds,
            sharedKV: sharedKV,
            positionOffset: .scalar(8)
        )
        // Softcap NOT applied on drafter path, so logits can exceed ±30.
        // This test just verifies they're not NaN / ±inf on zero input.
        let absMax = MLX.abs(logits).max().item(Float.self)
        #expect(absMax.isFinite)
    }
}
