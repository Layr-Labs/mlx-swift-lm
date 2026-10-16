import Foundation
import MLXVLM
import Testing

@testable import MLXLLM

/// Deliberately tiny complete source schema. This helper creates JSON/headers
/// only, so it is safe in the no-MLX UnitTests suite.
enum GLM5NextInventoryFixture {
    static func configuration(layers: Int = 4, q4: Bool = false) throws -> GLM5NextConfiguration {
        let d = q4 ? 64 : 4
        let a = q4 ? 64 : 2
        let types = (0 ..< layers).map {
            $0 % 4 == 3 ? "deepseek_sparse_attention" : "linear_attention"
        }
        let linear = types.indices.filter { types[$0] == "linear_attention" }
        let sparse = types.indices.filter { types[$0] == "deepseek_sparse_attention" }
        let dense = min(layers, 3)
        let text: [String: Any] = [
            "model_type": "glm5_next_text", "vocab_size": 32, "hidden_size": d,
            "intermediate_size": q4 ? 64 : 8, "moe_intermediate_size": q4 ? 64 : 4,
            "num_hidden_layers": layers, "num_attention_heads": 1, "num_key_value_heads": 1,
            "head_dim": 0, "qk_head_dim": a, "qk_nope_head_dim": a, "qk_rope_head_dim": 0,
            "v_head_dim": a, "q_lora_rank": d, "kv_lora_rank": q4 ? 64 : 2,
            "mla_use_nope": true, "layer_types": types,
            "mlp_layer_types": (0 ..< layers).map { $0 < dense ? "dense" : "sparse" },
            "indexer_types": Array(repeating: "full", count: layers),
            "linear_attn_config": [
                "num_heads": 1, "head_dim": a, "gate_lower_bound": -5,
                "short_conv_kernel_size": 4, "kda_layers": linear, "full_attn_layers": sparse,
            ],
            "index_head_dim": a, "index_n_heads": 1, "index_topk": 4, "index_kpool": 4,
            "index_kpool_compress": true, "index_kpool_always_select_tail": true,
            "indexer_rope_interleave": true, "index_share_for_mtp_iteration": true,
            "mhc": true, "hc_mult": 4, "hc_eps": 1e-6, "hc_sinkhorn_iters": 20,
            "first_k_dense_replace": dense, "n_routed_experts": 2, "n_shared_experts": 1,
            "num_experts_per_tok": 1, "n_group": 1, "topk_group": 1,
            "norm_topk_prob": true, "routed_scaling_factor": 2.5,
            "scoring_func": "sigmoid", "topk_method": "noaux_tc", "moe_router_dtype": "float32",
            "output_router_logits": false, "router_aux_loss_coef": 0.001,
            "num_nextn_predict_layers": 1, "hidden_act": "silu", "swiglu_limit": 10,
            "rms_norm_eps": 1e-5, "attention_bias": false, "attention_dropout": 0,
            "max_position_embeddings": 256, "initializer_range": 0.02, "use_cache": true,
            "tie_word_embeddings": false, "dtype": "float32", "pad_token_id": 0,
            "eos_token_id": [1, 2, 3],
        ]
        var root: [String: Any] = [
            "model_type": "glm5_next", "architectures": ["Glm5NextForConditionalGeneration"],
            "tie_word_embeddings": false, "text_config": text,
            "image_token_id": 10, "video_token_id": 11, "image_start_token_id": 12,
            "image_end_token_id": 13, "video_start_token_id": 14, "video_end_token_id": 15,
            "vision_config": [
                "model_type": "glm5_next_vision", "depth": 1, "hidden_size": 8,
                "intermediate_size": 12, "num_heads": 2, "image_size": 8, "patch_size": 2,
                "temporal_patch_size": 2, "spatial_merge_size": 2, "in_channels": 3,
                "out_hidden_size": d, "projection_intermediate_size": 8, "hidden_act": "silu",
                "swiglu_limit": 10, "rms_norm_eps": 1e-5, "attention_bias": true,
                "attention_dropout": 0, "initializer_range": 0.02,
            ],
        ]
        if q4 { root["quantization"] = ["bits": 4, "group_size": 64, "mode": "affine"] }
        return try JSONDecoder().decode(
            GLM5NextConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
    }
}

extension UnitTests {
    @Suite struct GLM5NextWeightInventoryTests {
        private func configuration(vision overrides: [String: Any]) throws -> GLM5NextConfiguration
        {
            let encoded = try JSONEncoder().encode(GLM5NextInventoryFixture.configuration())
            var root = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
            var vision = try #require(root["vision_config"] as? [String: Any])
            vision.merge(overrides) { _, replacement in replacement }
            root["vision_config"] = vision
            return try JSONDecoder().decode(
                GLM5NextConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
        }

        @Test(arguments: [false, true])
        func visionInventoryMatchesLoaderWithEitherBiasPolicy(attentionBias: Bool) throws {
            let config = try configuration(vision: ["attention_bias": attentionBias])
            let headers = try GLM5NextWeightInventory.expectedTensors(
                configuration: config, storage: .float32, shard: "tiny.safetensors")
            let inventory = try GLM5NextWeightInventory.validate(
                configuration: config, tensors: headers, storage: .float32)
            let shapes = Dictionary(
                uniqueKeysWithValues: inventory.entries(in: .vision).map {
                    ($0.tensor.name, $0.tensor.shape)
                })
            let loaderShapes = Dictionary(
                uniqueKeysWithValues:
                    GLM5NextVisionTower.expectedTensorShapes(configuration: config.visionConfig).map
                {
                    ("model.visual." + $0.key, $0.value)
                })
            #expect(shapes == loaderShapes)
            #expect((shapes["model.visual.blocks.0.attn.qkv.bias"] != nil) == attentionBias)
            #expect(shapes.count == (attentionBias ? 25 : 20))
        }

        @Test(arguments: [4_000_000_000_000_000_000, 2_000_000_000])
        func overflowingVisionWeightGeometryThrowsBeforeSchemaExpansion(hiddenSize: Int) throws {
            #expect(throws: GLM5NextConfigurationError.self) {
                try configuration(vision: [
                    "hidden_size": hiddenSize, "num_heads": 1, "intermediate_size": 1,
                    "spatial_merge_size": 1, "patch_size": 1, "temporal_patch_size": 1,
                    "in_channels": 1,
                ])
            }
        }

        @Test func completeHeaderInventorySeparatesTextMTPAndVision() throws {
            let config = try GLM5NextInventoryFixture.configuration()
            let tensors = try GLM5NextWeightInventory.expectedTensors(
                configuration: config, storage: .float32, shard: "tiny.safetensors")
            let inventory = try GLM5NextWeightInventory.validate(
                configuration: config, tensors: tensors, storage: .float32)
            #expect(inventory.entries["model.language_model.layers.4.eh_proj.weight"]?.part == .mtp)
            #expect(
                inventory.entries["model.language_model.layers.3.self_attn.kv_b_proj.weight"]?
                    .tensor.shape == [4, 2])
            #expect(
                inventory.entries["model.visual.patch_embed.proj.weight"]?.tensor.shape == [
                    8, 3, 2, 2, 2,
                ])
            #expect(inventory.entries(in: .mtp).count == 31)
            #expect(inventory.entries(in: .vision).count == 25)
            #expect(
                inventory.bytes(in: .mtp) > 0
                    && inventory.bytes(in: .text) > inventory.bytes(in: .mtp))
            #expect(inventory.entries.values.allSatisfy { $0.bytes > 0 })
            #expect(
                GLM5NextWeightInventory.nativeName(
                    for: "model.language_model.layers.4.mlp.experts.1.down_proj.weight")
                    == "model.layers.4.mlp.experts.down_proj.weight")
            #expect(
                GLM5NextWeightInventory.nativeName(for: "model.visual.downsample.weight")
                    == "model.visual.downsample.weight")
        }

        @Test func missingExtraDuplicateAndWrongDtypeAreRejected() throws {
            let config = try GLM5NextInventoryFixture.configuration()
            let headers = try GLM5NextWeightInventory.expectedTensors(
                configuration: config, storage: .float32, shard: "tiny.safetensors")
            #expect(throws: GLM5NextLoadError.self) {
                try GLM5NextWeightInventory.validate(
                    configuration: config, tensors: Array(headers.dropLast()), storage: .float32)
            }
            #expect(throws: GLM5NextLoadError.self) {
                try GLM5NextWeightInventory.validate(
                    configuration: config, tensors: headers + [headers[0]], storage: .float32)
            }
            let extra = GLM5NextWeightInventory.Tensor(
                name: "model.language_model.layers.99.fake.weight", shape: [1], dtype: "F32",
                shard: "tiny.safetensors")
            #expect(throws: GLM5NextLoadError.self) {
                try GLM5NextWeightInventory.validate(
                    configuration: config, tensors: headers + [extra], storage: .float32)
            }
            var wrong = headers
            wrong[0] = .init(
                name: headers[0].name, shape: headers[0].shape, dtype: "BF16",
                shard: headers[0].shard)
            #expect(throws: GLM5NextLoadError.self) {
                try GLM5NextWeightInventory.validate(
                    configuration: config, tensors: wrong, storage: .float32)
            }
        }

        @Test func packedQ4RequiresAllOffsetsAndPreservesEmbeddedHead() throws {
            let config = try GLM5NextInventoryFixture.configuration(q4: true)
            let headers = try GLM5NextWeightInventory.expectedTensors(
                configuration: config, storage: .vontraQ4, shard: "q4.safetensors")
            let inventory = try GLM5NextWeightInventory.validate(
                configuration: config, tensors: headers, storage: .vontraQ4)
            #expect(inventory.entries["lm_head.weight"]?.tensor.shape == [32, 8])
            #expect(inventory.entries["lm_head.weight"]?.tensor.dtype == "U32")
            #expect(
                inventory.entries["model.language_model.layers.4.eh_proj.weight"]?.tensor.shape == [
                    64, 16,
                ])
            let incomplete = headers.filter {
                $0.name != "model.language_model.layers.4.eh_proj.biases"
            }
            #expect(throws: GLM5NextLoadError.self) {
                try GLM5NextWeightInventory.validate(
                    configuration: config, tensors: incomplete, storage: .vontraQ4)
            }
        }
    }
}
