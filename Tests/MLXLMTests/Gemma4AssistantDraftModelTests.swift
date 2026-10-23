// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

@testable import MLXLLM

// MLXLLM (the fork's CBv2 drafter) and MLXVLM (the upstream drafter) both
// declare these names. Unqualified names in this file mean the MLXLLM types;
// the upstream tests at the end name the MLXVLM types explicitly.
private typealias Gemma4AssistantDraftModel = MLXLLM.Gemma4AssistantDraftModel
private typealias Gemma4AssistantConfiguration = MLXLLM.Gemma4AssistantConfiguration
private typealias Gemma4TextConfiguration = MLXLLM.Gemma4TextConfiguration

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
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        eval(drafter)
        #expect(drafter.config.backboneHiddenSize == 64)
        #expect(drafter.maskedEmbedder == nil)     // useOrderedEmbeddings: false
    }

    @Test func drafterWithCentroidHead() throws {
        let cfg = try drafterConfig(useOrderedEmbeddings: true)
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        eval(drafter)
        #expect(drafter.maskedEmbedder != nil)
    }

    @Test func bindSucceedsOnMatchedTarget() throws {
        let cfg = try drafterConfig()
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let tCfg = try targetConfig()
        let target = Gemma4TextModel(tCfg)
        eval(drafter, target)
        try drafter.bind(target: target)
    }

    @Test func bindIsIdempotentOnSameTarget() throws {
        let cfg = try drafterConfig()
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let target = Gemma4TextModel(try targetConfig())
        eval(drafter, target)
        try drafter.bind(target: target)
        try drafter.bind(target: target)   // second call is a no-op
    }

    @Test func rebindToDifferentTargetThrows() throws {
        let cfg = try drafterConfig()
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
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
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let target = Gemma4TextModel(try targetConfig(hiddenSize: 128))
        eval(drafter, target)
        #expect(throws: (any Error).self) {
            try drafter.bind(target: target)
        }
    }

    @Test func vocabMismatchThrows() throws {
        let cfg = try drafterConfig(vocab: 64)
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
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
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let target = Gemma4TextModel(try targetConfig(
            attentionKeqV: true, numGlobalKVHeads: 2))
        eval(drafter, target)
        try drafter.bind(target: target)
    }

    @Test func kEqVMismatchThrows() throws {
        // Drafter has k_eq_v=true, target has k_eq_v=false → throw.
        let cfg = try drafterConfig(
            attentionKeqV: true, numGlobalKVHeads: 2)
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let target = Gemma4TextModel(try targetConfig(
            attentionKeqV: false))
        eval(drafter, target)
        #expect(throws: (any Error).self) {
            try drafter.bind(target: target)
        }
    }

    @Test func bindRejectsMismatchedEffectiveKVHeadsWhenKeqVDisabled() throws {
        // Global KV heads are honored independent of k_eq_v: a drafter whose
        // full layers use 1 KV head cannot verify a target using 2.
        let drafter = try Gemma4AssistantDraftModel(config: drafterConfig())
        let target = Gemma4TextModel(
            try targetConfig(attentionKeqV: false, numGlobalKVHeads: 2))
        eval(drafter, target)

        do {
            try drafter.bind(target: target)
            Issue.record("binding unexpectedly succeeded")
        } catch let Gemma4MTPError.incompatibleDrafter(field, _, _) {
            #expect(field == "fullAttention.effectiveKVHeads")
        } catch {
            Issue.record("threw unexpected error \(error)")
        }
    }

    @Test func bindAcceptsMatchedEffectiveKVHeadsWhenKeqVDisabled() throws {
        // Matched effective full-layer KV head counts bind cleanly even with
        // k_eq_v disabled on both sides.
        let drafter = try Gemma4AssistantDraftModel(
            config: drafterConfig(attentionKeqV: false, numGlobalKVHeads: 2))
        let target = Gemma4TextModel(
            try targetConfig(attentionKeqV: false, numGlobalKVHeads: 2))
        eval(drafter, target)

        try drafter.bind(target: target)
    }

    @Test func slidingWindowMismatchReportsExactGeometry() throws {
        let drafter = try Gemma4AssistantDraftModel(config: drafterConfig())
        var targetConfiguration = try targetConfig()
        targetConfiguration.slidingWindow = 128
        let target = Gemma4TextModel(targetConfiguration)
        eval(drafter, target)

        #expect(
            throws: Gemma4MTPError.incompatibleDrafter(
                field: "slidingAttention.slidingWindow", drafter: "64", target: "128")
        ) {
            try drafter.bind(target: target)
        }
    }

    @Test func captureGeometryMismatchesThrowBeforeDrafting() throws {
        typealias DrafterMutation = (inout Gemma4AssistantConfiguration) -> Void
        typealias TargetMutation = (inout Gemma4TextConfiguration) -> Void
        let noDrafterChange: DrafterMutation = { _ in }
        let noTargetChange: TargetMutation = { _ in }
        let cases: [(String, String, DrafterMutation, TargetMutation)] = [
            (
                "sliding head dimension", "slidingAttention.headDimension",
                noDrafterChange, { $0.headDim = 16 }),
            (
                "sliding effective KV heads", "slidingAttention.effectiveKVHeads",
                noDrafterChange, { $0.numKeyValueHeads = 2 }),
            (
                "full head dimension", "fullAttention.headDimension",
                noDrafterChange, { $0.globalHeadDim = 16 }),
            (
                "full K equals V", "fullAttention.attentionKeqV",
                noDrafterChange, {
                    $0.attentionKeqV = true
                    $0.numGlobalKeyValueHeads = 1
                }),
            (
                "global KV heads", "fullAttention.effectiveKVHeads",
                {
                    $0.textConfig.attentionKeqV = true
                    $0.textConfig.numGlobalKeyValueHeads = 1
                },
                {
                    $0.attentionKeqV = true
                    $0.numGlobalKeyValueHeads = 2
                }),
            (
                "full effective KV heads", "fullAttention.effectiveKVHeads",
                {
                    $0.textConfig.layerTypes = Array(
                        repeating: "full_attention",
                        count: $0.textConfig.numHiddenLayers)
                },
                { $0.numKeyValueHeads = 2 }),
            (
                "missing sliding capture", "captureLayer.sliding_attention",
                noDrafterChange, {
                    $0.layerTypes =
                        Array(repeating: "full_attention", count: 5)
                        + Array(repeating: "sliding_attention", count: 5)
                }),
            (
                "missing full capture", "captureLayer.full_attention",
                noDrafterChange, {
                    $0.layerTypes =
                        Array(repeating: "sliding_attention", count: 5)
                        + Array(repeating: "full_attention", count: 5)
                }),
            (
                "matched geometry", "", noDrafterChange, noTargetChange),
        ]

        for (name, expectedField, mutateDrafter, mutateTarget) in cases {
            var drafterConfiguration = try drafterConfig()
            var targetConfiguration = try targetConfig()
            mutateDrafter(&drafterConfiguration)
            mutateTarget(&targetConfiguration)
            let drafter = try Gemma4AssistantDraftModel(config: drafterConfiguration)
            let target = Gemma4TextModel(targetConfiguration)
            eval(drafter, target)

            if expectedField.isEmpty {
                try drafter.bind(target: target)
                continue
            }
            do {
                try drafter.bind(target: target)
                Issue.record("\(name) unexpectedly bound")
            } catch let Gemma4MTPError.incompatibleDrafter(field, _, _) {
                #expect(field == expectedField, "\(name) reported \(field)")
            } catch {
                Issue.record("\(name) threw unexpected error \(error)")
            }
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
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
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
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
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
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
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

@Suite("Gemma4AssistantDraftModel sanitize")
struct Gemma4AssistantDraftModelSanitizeTests {

    private func drafterConfig(
        tieWordEmbeddings: Bool = true
    ) throws -> Gemma4AssistantConfiguration {
        let json = """
        {
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 64,
            "use_ordered_embeddings": true,
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
                "tie_word_embeddings": \(tieWordEmbeddings),
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

    @Test func castsTokenOrderingToInt32() throws {
        let cfg = try drafterConfig()
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let int64Ordering = MLXArray([Int64(0), 1, 2, 3, 4, 5, 6, 7,
                                      8, 9, 10, 11, 12, 13, 14, 15,
                                      16, 17, 18, 19, 20, 21, 22, 23,
                                      24, 25, 26, 27, 28, 29, 30, 31,
                                      32, 33, 34, 35, 36, 37, 38, 39,
                                      40, 41, 42, 43, 44, 45, 46, 47,
                                      48, 49, 50, 51, 52, 53, 54, 55,
                                      56, 57, 58, 59, 60, 61, 62, 63])
        var weights: [String: MLXArray] = [
            "masked_embedding.token_ordering": int64Ordering,
        ]
        weights = try drafter.sanitize(weights: weights)
        let cast = weights["masked_embedding.token_ordering"]
        #expect(cast != nil)
        #expect(cast!.dtype == .int32)
    }

    @Test func dropsLmHeadWhenTied() throws {
        let cfg = try drafterConfig(tieWordEmbeddings: true)
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let lmHead = MLXArray.zeros([64, 64], dtype: .float32)
        let preProj = MLXArray.zeros([64, 128], dtype: .float32)
        var weights: [String: MLXArray] = [
            "lm_head.weight": lmHead,
            "pre_projection.weight": preProj,
        ]
        weights = try drafter.sanitize(weights: weights)
        #expect(weights["lm_head.weight"] == nil)         // dropped
        #expect(weights["pre_projection.weight"] != nil)  // kept
    }

    @Test func keepsLmHeadWhenUntied() throws {
        let cfg = try drafterConfig(tieWordEmbeddings: false)
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let lmHead = MLXArray.zeros([64, 64], dtype: .float32)
        var weights: [String: MLXArray] = [
            "lm_head.weight": lmHead,
        ]
        weights = try drafter.sanitize(weights: weights)
        #expect(weights["lm_head.weight"] != nil)
    }

    @Test func failsLoudOnUnexpectedKProj() throws {
        let cfg = try drafterConfig()
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let kProj = MLXArray.zeros([64, 64], dtype: .float32)
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.k_proj.weight": kProj,
        ]
        #expect(throws: (any Error).self) {
            _ = try drafter.sanitize(weights: weights)
        }
    }

    @Test func failsLoudOnUnexpectedVProj() throws {
        let cfg = try drafterConfig()
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let vProj = MLXArray.zeros([64, 64], dtype: .float32)
        let weights: [String: MLXArray] = [
            "model.layers.2.self_attn.v_proj.weight": vProj,
        ]
        #expect(throws: (any Error).self) {
            _ = try drafter.sanitize(weights: weights)
        }
    }

    @Test func preservesOtherWeights() throws {
        let cfg = try drafterConfig()
        let drafter = try Gemma4AssistantDraftModel(config: cfg)
        let qProj = MLXArray.zeros([64, 64], dtype: .float32)
        let preProj = MLXArray.zeros([64, 128], dtype: .float32)
        let postProj = MLXArray.zeros([64, 64], dtype: .float32)
        let embed = MLXArray.zeros([64, 64], dtype: .float32)
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.q_proj.weight": qProj,
            "pre_projection.weight": preProj,
            "post_projection.weight": postProj,
            "model.embed_tokens.weight": embed,
        ]
        let out = try drafter.sanitize(weights: weights)
        #expect(out.count == 4)  // all four kept
    }
}

@Suite("Gemma4AssistantDraftModel loading")
struct Gemma4AssistantDraftModelLoadingTests {

    /// Integration test gated on env var. Loads a pre-downloaded drafter
    /// directory from disk using the local-path variant of `load`.
    @Test(.enabled(if: ProcessInfo.processInfo.environment[
        "MLX_SWIFT_LM_INTEGRATION_DATA_DIR"] != nil))
    func loadsFromDirectory() async throws {
        guard let dirPath = ProcessInfo.processInfo.environment[
            "MLX_SWIFT_LM_INTEGRATION_DATA_DIR"] else { return }
        // Expect a sub-directory e.g. "gemma-4-E4B-it-assistant-bf16" under
        // the integration data dir; caller is responsible for pre-downloading.
        let drafterDir = URL(fileURLWithPath: dirPath)
            .appending(path: "gemma-4-E4B-it-assistant-bf16")
        let drafter = try await Gemma4AssistantDraftModel.load(from: drafterDir)
        // A basic shape check: the first layer's q_proj weight should exist.
        // We don't assert exact values here; the parity tests do that.
        let params = drafter.parameters()
        // Flatten to check a known key exists.
        let flat = params.flattened()
        let keys = Set(flat.map(\.0))
        #expect(keys.contains("model.layers.0.self_attn.q_proj.weight"))
    }
}

// MARK: - Upstream MLXVLM drafter tests

// MARK: - Config decode (no MLX kernels required)

@Test
func testGemma4AssistantConfigurationDecodesSyntheticJSON() throws {
    let json = """
        {
          "model_type": "gemma4_assistant",
          "backbone_hidden_size": 5376,
          "tie_word_embeddings": true,
          "use_ordered_embeddings": false,
          "num_centroids": 2048,
          "centroid_intermediate_top_k": 32,
          "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 16,
            "head_dim": 256,
            "global_head_dim": 512,
            "vocab_size": 262144,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "sliding_window": 512,
            "sliding_window_pattern": 5,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "rope_traditional": false,
            "use_double_wide_mlp": false,
            "enable_moe_block": false,
            "attention_k_eq_v": true,
            "intermediate_size": 8192,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
            "rope_parameters": {},
            "tie_word_embeddings": true
          }
        }
        """
    let data = Data(json.utf8)
    let cfg = try JSONDecoder().decode(MLXVLM.Gemma4AssistantConfiguration.self, from: data)
    #expect(cfg.modelType == "gemma4_assistant")
    #expect(cfg.backboneHiddenSize == 5376)
    #expect(cfg.tieWordEmbeddings == true)
    #expect(cfg.useOrderedEmbeddings == false)
    #expect(cfg.numCentroids == 2048)
    #expect(cfg.centroidIntermediateTopK == 32)
    #expect(cfg.textConfiguration.hiddenSize == 1024)
    #expect(cfg.textConfiguration.hiddenLayers == 4)
    #expect(cfg.textConfiguration.vocabularySize == 262_144)
    #expect(cfg.textConfiguration.layerTypes.count == 4)
}

// MARK: - Sanitize behavior (no MLX kernels — uses MLXArray scalar metadata only)

@Test
func testSanitizeDropsLmHeadWhenTied() {
    let cfg = syntheticConfig(tieWordEmbeddings: true)
    let model = MLXVLM.Gemma4AssistantDraftModel(cfg)
    let weights: [String: MLXArray] = [
        "model.embed_tokens.weight": MLXArray.zeros([10, 4]),
        "lm_head.weight": MLXArray.zeros([10, 4]),
        "lm_head.scales": MLXArray.zeros([1]),
        "lm_head.biases": MLXArray.zeros([1]),
        "pre_projection.weight": MLXArray.zeros([4, 8]),
    ]
    let sanitized = model.sanitize(weights: weights)
    #expect(!sanitized.keys.contains { $0.hasPrefix("lm_head.") })
    #expect(sanitized["model.embed_tokens.weight"] != nil)
    #expect(sanitized["pre_projection.weight"] != nil)
}

@Test
func testSanitizeKeepsLmHeadWhenNotTied() {
    let cfg = syntheticConfig(tieWordEmbeddings: false)
    let model = MLXVLM.Gemma4AssistantDraftModel(cfg)
    let weights: [String: MLXArray] = [
        "model.embed_tokens.weight": MLXArray.zeros([10, 4]),
        "lm_head.weight": MLXArray.zeros([10, 4]),
        "lm_head.scales": MLXArray.zeros([1]),
        "lm_head.biases": MLXArray.zeros([1]),
    ]
    let sanitized = model.sanitize(weights: weights)
    #expect(sanitized["lm_head.weight"] != nil)
    #expect(sanitized["lm_head.scales"] != nil)
    #expect(sanitized["lm_head.biases"] != nil)
}

// MARK: - Synthetic shape test (no checkpoint needed)

@Test
func testGemma4AssistantDraftModelInstantiatesAndShape() {
    let cfg = syntheticConfig(tieWordEmbeddings: true)
    let model = MLXVLM.Gemma4AssistantDraftModel(cfg)
    #expect(model.config.modelType == "gemma4_assistant")
    #expect(model.config.backboneHiddenSize == 4)
    #expect(model.config.tieWordEmbeddings == true)
    // The inner Embedding and Linears are constructed at init.
    // We don't run inference here (would need actual weights + metal kernels).
}

@Test
func testGemma4AssistantRejectsNonGemmaTarget() {
    let model = MLXVLM.Gemma4AssistantDraftModel(syntheticConfig(tieWordEmbeddings: true))
    #expect(!model.isCompatible(with: NonGemmaTarget()))
}

// MARK: - MaskedEmbedder forward (use_ordered_embeddings)

/// Independent correctness pin for the centroid-routed sparse LM head.
///
/// With an identity `token_ordering` (cluster `c` owns tokens `[c·vsc, …)`), the
/// masked head must, at every *selected* vocab position, reproduce the plain dense
/// logit `hidden · embᵀ`; exactly `topK·vsc` positions are selected and every other
/// position must carry `mask_value`. The dense reference is computed independently
/// of the embedder, so a wrong gather/scatter/matmul fails this — no checkpoint or
/// Python fixture required.
@Test
func testMaskedEmbedderSelectedLogitsMatchDense() throws {
    let (h, v, c, k) = (4, 8, 4, 2)  // vsc = v/c = 2, N = k·vsc = 4
    let vsc = v / c
    let n = k * vsc

    let emb = Gemma4AssistantMaskedEmbedder(config: orderedSyntheticConfig())
    // Identity ordering so canonical token IDs equal vocab positions.
    try emb.update(
        parameters: ModuleParameters.unflattened([
            "token_ordering": MLXArray((0 ..< v).map { Int32($0) })
        ]), verify: [])

    let hidden = MLXArray([0.5, -1.0, 2.0, 0.25] as [Float], [1, 1, h])
    let lmHead = (MLXArray(0 ..< (v * h)).asType(.float32) * 0.1 - 1.0).reshaped([v, h])

    let masked = emb(hidden, tiedEmbedding: Embedding(weight: lmHead)).reshaped([v])
    let dense = matmul(hidden, lmHead.swappedAxes(-1, -2)).reshaped([v])

    let maskValue = masked.min()
    let selected = masked .!= maskValue
    // Exactly N positions survive the centroid pruning.
    #expect(selected.sum().item(Int.self) == n)
    // Every surviving position equals its dense logit.
    let agree = MLX.where(selected, abs(masked - dense) .< 1e-4, MLXArray(true))
    #expect(agree.all().item(Bool.self))
}

/// Regression pin for #383: the tied embedding of a *quantized* E-series drafter
/// is a `QuantizedEmbedding` whose `.weight` is packed `[vocab, H·bits/32]` — the
/// old forward fed that raw weight into a gather/reshape-to-H and crashed. Passing
/// the module (not `.weight`) gathers through its dequantizing forward. Reference
/// is the dense logit against the *dequantized* rows, so selected positions must
/// match exactly (identical dequantized rows, identical matmul).
@Test
func testMaskedEmbedderQuantizedTiedEmbedding() throws {
    // Hidden must be a multiple of the quant group size (64).
    let (h, v, c, k) = (64, 128, 8, 2)  // vsc = v/c = 16, N = k·vsc = 32
    let vsc = v / c
    let n = k * vsc

    let emb = Gemma4AssistantMaskedEmbedder(
        config: orderedSyntheticConfig(hidden: h, vocab: v, centroids: c, topK: k))
    try emb.update(
        parameters: ModuleParameters.unflattened([
            "token_ordering": MLXArray((0 ..< v).map { Int32($0) })
        ]), verify: [])

    let hidden = (MLXArray(0 ..< h).asType(.float32) * 0.01 - 0.3).reshaped([1, 1, h])
    let lmHead = (MLXArray(0 ..< (v * h)).asType(.float32) * 0.001 - 0.05).reshaped([v, h])
    let quantEmb = QuantizedEmbedding(weight: lmHead, groupSize: 64, bits: 4)

    let masked = emb(hidden, tiedEmbedding: quantEmb).reshaped([v])
    // Dense reference over dequantized rows (the head must reproduce these).
    let dequant = quantEmb(MLXArray(0 ..< v))  // [v, h]
    let dense = matmul(hidden, dequant.swappedAxes(-1, -2)).reshaped([v])

    let maskValue = masked.min()
    let selected = masked .!= maskValue
    #expect(selected.sum().item(Int.self) == n)
    let agree = MLX.where(selected, abs(masked - dense) .< 1e-3, MLXArray(true))
    #expect(agree.all().item(Bool.self))
}

// MARK: - Helpers

/// Tiny `use_ordered_embeddings=true` config. Defaults (hidden 4, vocab 8, 4
/// centroids, top-K 2) suit the dense pin; the quantized pin bumps hidden to a
/// multiple of the quantization group size.
private func orderedSyntheticConfig(
    hidden: Int = 4, vocab: Int = 8, centroids: Int = 4, topK: Int = 2
) -> MLXVLM.Gemma4AssistantConfiguration {
    let textJSON =
        """
        {
          "model_type": "gemma4_text", "hidden_size": \(hidden), "num_hidden_layers": 1,
          "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 2,
          "global_head_dim": 2, "vocab_size": \(vocab), "num_kv_shared_layers": 0,
          "hidden_size_per_layer_input": 0, "sliding_window": 4, "sliding_window_pattern": 1,
          "max_position_embeddings": 16, "rms_norm_eps": 1e-6, "rope_traditional": false,
          "use_double_wide_mlp": false, "enable_moe_block": false, "attention_k_eq_v": true,
          "intermediate_size": 8, "layer_types": ["full_attention"], "rope_parameters": {},
          "tie_word_embeddings": true
        }
        """
    let json =
        """
        {
          "model_type": "gemma4_assistant", "backbone_hidden_size": \(hidden),
          "tie_word_embeddings": true, "use_ordered_embeddings": true,
          "num_centroids": \(centroids), "centroid_intermediate_top_k": \(topK),
          "text_config": \(textJSON)
        }
        """
    return try! JSONDecoder().decode(MLXVLM.Gemma4AssistantConfiguration.self, from: Data(json.utf8))
}

private func syntheticConfig(tieWordEmbeddings: Bool) -> MLXVLM.Gemma4AssistantConfiguration {
    // Build a tiny synthetic config sufficient for sanitize / instantiation tests.
    // Uses JSON round-trip to avoid manually constructing Gemma4TextConfiguration
    // (which has many defaulted fields and a custom decoder).
    let textJSON =
        """
        {
          "model_type": "gemma4_text",
          "hidden_size": 4,
          "num_hidden_layers": 1,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 2,
          "global_head_dim": 2,
          "vocab_size": 10,
          "num_kv_shared_layers": 0,
          "hidden_size_per_layer_input": 0,
          "sliding_window": 4,
          "sliding_window_pattern": 1,
          "max_position_embeddings": 16,
          "rms_norm_eps": 1e-6,
          "rope_traditional": false,
          "use_double_wide_mlp": false,
          "enable_moe_block": false,
          "attention_k_eq_v": true,
          "intermediate_size": 8,
          "layer_types": ["full_attention"],
          "rope_parameters": {},
          "tie_word_embeddings": \(tieWordEmbeddings)
        }
        """
    let json =
        """
        {
          "model_type": "gemma4_assistant",
          "backbone_hidden_size": 4,
          "tie_word_embeddings": \(tieWordEmbeddings),
          "use_ordered_embeddings": false,
          "num_centroids": 2,
          "centroid_intermediate_top_k": 1,
          "text_config": \(textJSON)
        }
        """
    return try! JSONDecoder().decode(
        MLXVLM.Gemma4AssistantConfiguration.self, from: Data(json.utf8))
}

// MARK: - gemma4_unified_assistant (the 12B drafter)

/// Decode pin for the unified-assistant config shape: mirrors
/// `mlx-community/gemma-4-12B-it-assistant-4bit`'s `config.json`
/// (`model_type: gemma4_unified_assistant`, `text_config.model_type:
/// gemma4_unified_text`, `attention_k_eq_v`, `num_global_key_value_heads`).
@Test
func testGemma4UnifiedAssistantConfigurationDecodes() throws {
    let json = """
        {
          "model_type": "gemma4_unified_assistant",
          "backbone_hidden_size": 3840,
          "tie_word_embeddings": true,
          "use_ordered_embeddings": false,
          "num_centroids": 2048,
          "centroid_intermediate_top_k": 32,
          "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_global_key_value_heads": 1,
            "head_dim": 256,
            "global_head_dim": 512,
            "vocab_size": 262144,
            "num_kv_shared_layers": 4,
            "hidden_size_per_layer_input": 0,
            "sliding_window": 1024,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "attention_k_eq_v": true,
            "intermediate_size": 8192,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
            "rope_parameters": {},
            "tie_word_embeddings": true
          }
        }
        """
    let cfg = try JSONDecoder().decode(
        MLXVLM.Gemma4AssistantConfiguration.self, from: Data(json.utf8))
    #expect(cfg.modelType == "gemma4_unified_assistant")
    #expect(cfg.backboneHiddenSize == 3840)
    #expect(cfg.useOrderedEmbeddings == false)
    #expect(cfg.textConfiguration.modelType == "gemma4_unified_text")
    #expect(cfg.textConfiguration.attentionKEqV == true)
    #expect(cfg.textConfiguration.globalKVHeads == 1)
    #expect(cfg.textConfiguration.hiddenSize == 1024)

    // Instantiation must succeed with the unified text config.
    let model = MLXVLM.Gemma4AssistantDraftModel(cfg)
    #expect(model.config.modelType == "gemma4_unified_assistant")
}

/// End-to-end synthetic pin for the `gemma4_unified` target path: a tiny
/// `Gemma4Unified` (a) emits drafter state through the MTP-aware
/// `callAsFunction(_:cache:state:)` entry point, and (b) is accepted by
/// `draftBlock` as the target (the pre-fix guard only recognized `Gemma4`
/// and trapped on `Gemma4Unified`, which the iterator surfaced as sticky
/// passthrough — MTP silently disabled for the 12B).
@Test
func testDraftBlockAcceptsGemma4UnifiedTarget() throws {
    let unifiedJSON = """
        {
          "model_type": "gemma4_unified",
          "vocab_size": 32,
          "image_token_id": 31,
          "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 8,
            "global_head_dim": 8,
            "vocab_size": 32,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "sliding_window": 8,
            "attention_k_eq_v": true,
            "layer_types": ["sliding_attention", "full_attention"],
            "rope_parameters": {},
            "tie_word_embeddings": true
          },
          "vision_config": null,
          "audio_config": null
        }
        """
    let targetConfig = try JSONDecoder.json5().decode(
        Gemma4UnifiedConfiguration.self, from: Data(unifiedJSON.utf8))
    let target = Gemma4Unified(targetConfig)

    // Prime drafter state through the MTP entry point (what the iterator does).
    let cache = try target.newCache(parameters: nil)
    var emitState = LMOutput.State()
    emitState[mtpEmitFlagKey] = true
    let tokens = MLXArray((0 ..< 8).map { Int32($0 % 32) }).reshaped([1, 8])
    let out = target(LMInput.Text(tokens: tokens), cache: cache, state: emitState)
    eval(out.logits)

    #expect(out.state != nil, "Gemma4Unified must emit drafter state when the flag is set")
    guard let state = out.state,
        let lastHidden = state[mtpLastHiddenStatesKey],
        let sharedKV = state[mtpSharedKVStatesKey]
    else {
        Issue.record("missing drafter state keys on Gemma4Unified LMOutput")
        return
    }
    #expect(Set(sharedKV.keys) == ["full_attention", "sliding_attention"])

    // Tiny drafter with geometry matching the target's KV (cross-attention
    // consumes the target's sharedKV pool) and backbone hidden size.
    let drafterJSON = """
        {
          "model_type": "gemma4_unified_assistant",
          "backbone_hidden_size": 8,
          "tie_word_embeddings": true,
          "use_ordered_embeddings": false,
          "num_centroids": 2,
          "centroid_intermediate_top_k": 1,
          "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 8,
            "global_head_dim": 8,
            "vocab_size": 32,
            "num_kv_shared_layers": 2,
            "hidden_size_per_layer_input": 0,
            "sliding_window": 8,
            "attention_k_eq_v": true,
            "layer_types": ["sliding_attention", "full_attention"],
            "rope_parameters": {},
            "tie_word_embeddings": true
          }
        }
        """
    let drafterConfig = try JSONDecoder().decode(
        MLXVLM.Gemma4AssistantConfiguration.self, from: Data(drafterJSON.utf8))
    let drafter = MLXVLM.Gemma4AssistantDraftModel(drafterConfig)
    #expect(drafter.isCompatible(with: target))

    let lastHiddenSlice = lastHidden[0..., (-1)..., 0...]
    let proposed = drafter.draftBlock(
        target: target,
        lastToken: MLXArray([Int32(3)]),
        lastHidden: lastHiddenSlice,
        sharedKV: sharedKV,
        positionDeltas: nil,
        queryOffset: cache.first?.offset ?? 8,
        blockSize: 3,
        sampler: ArgMaxSampler()
    )
    eval(proposed)
    #expect(proposed.shape == [1, 2])
}

private final class NonGemmaTarget: Module, LanguageModel, KVCacheDimensionProvider {
    var kvHeads: [Int] { [] }

    func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill _: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([1, 1, 1])
    }
}
