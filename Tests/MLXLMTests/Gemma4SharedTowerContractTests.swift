import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXVLM

/// Contract tests pinning the deleted inline VLM text tower's semantics onto
/// the canonical `Gemma4TextModel` and configuration types. Each test guards a
/// behavior whose loss was found during review of the shared-tower cutover:
/// fp16 attention promotion, rank-1 token acceptance, full-layer global KV
/// head counts independent of k_eq_v, and nested quantization round-tripping.
@Suite("Gemma 4 shared-tower contracts", .serialized)
struct Gemma4SharedTowerContractTests {

    private func tinyConfigJSON(extraFields: [String] = []) -> String {
        var fields = [
            "\"model_type\": \"gemma4_text\"",
            "\"hidden_size\": 32",
            "\"num_hidden_layers\": 2",
            "\"intermediate_size\": 64",
            "\"num_attention_heads\": 2",
            "\"head_dim\": 16",
            "\"global_head_dim\": 16",
            "\"num_key_value_heads\": 2",
            "\"num_kv_shared_layers\": 0",
            "\"layer_types\": [\"sliding_attention\", \"full_attention\"]",
            "\"sliding_window\": 16",
            "\"final_logit_softcapping\": 30.0",
            "\"hidden_size_per_layer_input\": 0",
            "\"use_double_wide_mlp\": false",
            "\"tie_word_embeddings\": true",
            "\"vocab_size\": 64",
            "\"vocab_size_per_layer_input\": 64",
            "\"rms_norm_eps\": 1e-6",
        ]
        fields.append(contentsOf: extraFields)
        return "{ \(fields.joined(separator: ", ")) }"
    }

    private func tinyConfig(extraFields: [String] = []) throws -> Gemma4TextConfiguration {
        let json = tinyConfigJSON(extraFields: extraFields)
        return try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    private func tinyModel(_ config: Gemma4TextConfiguration) -> Gemma4TextModel {
        Gemma4TextModel(config)
    }

    private func param(
        _ params: [(String, MLXArray)], _ key: String
    ) -> MLXArray? {
        params.first { $0.0 == key }?.1
    }

    // MARK: F2 — rank-1 token acceptance

    /// The deleted inline twin accepted [N] token ids on cache-reuse turns and
    /// expanded them to [1, N]. `Gemma4TextModel` must keep that contract at
    /// both public entry points instead of trapping on `dim(1)`.
    @Test("rank-1 token ids produce the batched result, not a trap")
    func rankOneTokenAcceptance() throws {
        let config = try tinyConfig()
        let model = tinyModel(config)
        let ids = MLXArray([Int32(3), 5, 7, 11])
        let singleton = model(ids, cache: nil as [KVCache]?)
        let batched = model(ids.expandedDimensions(axis: 0), cache: nil as [KVCache]?)
        eval(singleton, batched)
        #expect(singleton.shape == batched.shape)
        #expect(singleton.dtype == batched.dtype)
        let equal = allClose(singleton, batched, rtol: 0, atol: 0)
        eval(equal)
        #expect(equal.item(Bool.self))
    }

    // MARK: F3 — global KV heads independent of k_eq_v

    /// A full-attention layer with `num_global_key_value_heads` different from
    /// `num_key_value_heads` uses the global count even when `attention_k_eq_v`
    /// is false — k_eq_v only elides `v_proj`. The pre-fix tower resolved the
    /// global count only under k_eq_v, mis-sizing projections and caches for
    /// such checkpoints.
    /// Asserted through the public module tree: projection widths encode the
    /// head-count rule (`rows = nKvHeads * effectiveHeadDim`), and `kvHeads`
    /// feeds cache sizing directly. Layer index 1 is the declared
    /// full-attention layer; the fixture uses head_dim == global_head_dim ==
    /// 16, so the distinguishing row counts are 32 (2 heads) vs 16 (1 head).
    @Test("full layer honors global KV head count without k_eq_v")
    func fullLayerGlobalKVHeadsWithoutKEqV() throws {
        let config = try tinyConfig(extraFields: [
            "\"attention_k_eq_v\": false",
            "\"num_global_key_value_heads\": 1",
            "\"num_key_value_heads\": 2",
        ])
        let model = tinyModel(config)
        #expect(model.kvHeads == [2, 1])

        let params = model.parameters().flattened()
        // Sliding layer 0: 2 KV heads × headDim 16 = 32 rows.
        #expect(param(params, "model.layers.0.self_attn.k_proj.weight")?.shape == [32, 32])
        // Full layer 1: global KV heads (1) × globalHeadDim 16 = 16 rows, and
        // k_eq_v=false keeps a real v_proj of the same width.
        #expect(param(params, "model.layers.1.self_attn.k_proj.weight")?.shape == [16, 32])
        #expect(param(params, "model.layers.1.self_attn.v_proj.weight")?.shape == [16, 32])
    }

    @Test("full layer still honors global KV head count under k_eq_v")
    func fullLayerGlobalKVHeadsWithKEqV() throws {
        let config = try tinyConfig(extraFields: [
            "\"attention_k_eq_v\": true",
            "\"num_global_key_value_heads\": 1",
            "\"num_key_value_heads\": 2",
        ])
        let model = tinyModel(config)
        #expect(model.kvHeads == [2, 1])
        let params = model.parameters().flattened()
        // k_eq_v=true: the global count still rules, and v_proj is elided.
        #expect(param(params, "model.layers.1.self_attn.k_proj.weight")?.shape == [16, 32])
        #expect(param(params, "model.layers.1.self_attn.v_proj.weight") == nil)
    }

    // MARK: F1 — fp16 attention promotion

    /// fp16 activations are promoted to fp32 around SDPA (vmlx #52). Forging
    /// the Q/K projection weights to magnitude ~600 guarantees fp16-range
    /// overflow in any fused or composed score computation (|q·k| ~ 10^6 ≫
    /// 65504), so a non-promoting build would return non-finite logits; the
    /// promoted tower must stay finite end-to-end.
    @Test("fp16 forward with fp16-overflowing Q×K scores stays finite")
    func fp16AttentionPromotion() throws {
        let config = try tinyConfig()
        let model = tinyModel(config)

        var fp16: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() {
            var value = value.asType(.float16)
            if key.hasSuffix(".self_attn.q_proj.weight") || key.hasSuffix(".self_attn.k_proj.weight") {
                value = (value.asType(.float32) * 600).asType(.float16)
            }
            fp16[key] = value
        }
        model.update(parameters: ModuleParameters.unflattened(fp16))

        let out = model(MLXArray([Int32(3), 5, 7, 11]), cache: nil as [KVCache]?)
        eval(out)
        #expect(abs(out).max().item(Float.self).isFinite)
        // Soft-capped logits are bounded by the configured cap.
        #expect(abs(out).max().item(Float.self) <= 30.0)
    }

    // MARK: F4 — nested quantization round trip

    private func vlmJSON(textQuantization: String?, rootQuantization: String?) -> String {
        let textQ = textQuantization.map { ", \"quantization\": \($0)" } ?? ""
        let rootQ = rootQuantization.map { ", \"quantization\": \($0)" } ?? ""
        return """
            {
                "model_type": "gemma4",
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 32,
                    "num_hidden_layers": 2,
                    "intermediate_size": 64,
                    "num_attention_heads": 2,
                    "head_dim": 16,
                    "global_head_dim": 16,
                    "num_key_value_heads": 2,
                    "vocab_size": 64
                    \(textQ)
                },
                "vision_config": {
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "image_size": 8,
                    "patch_size": 4
                }
                \(rootQ)
            }
            """
    }

    @Test("nested text_config quantization survives decode-encode-decode")
    func nestedQuantizationRoundTrip() throws {
        let decoder = JSONDecoder()
        let encoder = JSONEncoder()

        let first = try decoder.decode(
            MLXVLM.Gemma4Configuration.self, from: Data(vlmJSON(
                textQuantization: "{\"bits\": 4, \"group_size\": 64}",
                rootQuantization: nil).utf8))
        #expect(first.quantization == nil)
        #expect(first.textConfig.quantizationBits == 4)
        #expect(first.textConfig.quantizationGroupSize == 64)

        let encoded = try encoder.encode(first)
        let second = try decoder.decode(MLXVLM.Gemma4Configuration.self, from: encoded)
        #expect(second.textConfig.quantizationBits == 4)
        #expect(second.textConfig.quantizationGroupSize == 64)

        // A third cycle must be idempotent (no drift, no duplication).
        let third = try decoder.decode(
            MLXVLM.Gemma4Configuration.self, from: try encoder.encode(second))
        #expect(third.textConfig.quantizationBits == 4)
        #expect(third.textConfig.quantizationGroupSize == 64)
    }

    @Test("root quantization keeps precedence and round-trips")
    func rootQuantizationRoundTrip() throws {
        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        let first = try decoder.decode(
            MLXVLM.Gemma4Configuration.self, from: Data(vlmJSON(
                textQuantization: "{\"bits\": 8, \"group_size\": 128}",
                rootQuantization: "{\"quant_method\": \"affine\", \"bits\": 4, \"group_size\": 64}").utf8))
        // Root metadata overlays nested text config at decode.
        #expect(first.textConfig.quantizationBits == 4)
        #expect(first.textConfig.quantizationGroupSize == 64)
        let second = try decoder.decode(
            MLXVLM.Gemma4Configuration.self, from: try encoder.encode(first))
        #expect(second.textConfig.quantizationBits == 4)
        #expect(second.textConfig.quantizationGroupSize == 64)
    }
}
