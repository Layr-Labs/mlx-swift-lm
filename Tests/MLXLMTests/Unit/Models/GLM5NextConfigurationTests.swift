import Foundation
import MLXLMCommon
import Testing

@testable import MLXLLM

extension UnitTests {
    @Suite
    struct GLM5NextConfigurationTests {
        // All architecture fields from the pinned checkpoint, transcribed independently.
        // https://huggingface.co/zai-org/GLM-5.3-Flash/raw/91b69215804f2d2036a8867f6ce44848e278dcda/config.json
        // Only quantization_config is omitted here: its 1,509 module exclusions are
        // storage metadata, tested separately below with representative source entries.
        private static let pinnedArchitecture = #"""
            {
              "architectures": ["Glm5NextForConditionalGeneration"],
              "model_type": "glm5_next",
              "transformers_version": "5.16.0",
              "tie_word_embeddings": false,
              "image_token_id": 154854,
              "video_token_id": 154855,
              "image_start_token_id": 154830,
              "image_end_token_id": 154831,
              "video_start_token_id": 154832,
              "video_end_token_id": 154833,
              "text_config": {
                "model_type": "glm5_next_text",
                "attention_bias": false,
                "attention_dropout": 0.0,
                "dtype": "bfloat16",
                "eos_token_id": [154820, 154827, 154829],
                "first_k_dense_replace": 3,
                "hc_eps": 1e-6,
                "hc_mult": 4,
                "hc_sinkhorn_iters": 20,
                "head_dim": 0,
                "hidden_act": "silu",
                "hidden_size": 4096,
                "index_head_dim": 128,
                "index_kpool": 4,
                "index_kpool_always_select_tail": true,
                "index_kpool_compress": true,
                "index_n_heads": 32,
                "index_topk": 2048,
                "index_share_for_mtp_iteration": true,
                "indexer_rope_interleave": true,
                "indexer_types": [
                  "full", "full", "full", "full", "full", "full", "full", "full",
                  "full", "full", "full", "full", "full", "full", "full", "full",
                  "full", "full", "full", "full", "full", "full", "full", "full",
                  "full", "full", "full", "full", "full", "full", "full", "full",
                  "full", "full", "full", "full", "full", "full", "full", "full",
                  "full", "full", "full", "full", "full"
                ],
                "initializer_range": 0.02,
                "intermediate_size": 12288,
                "kv_lora_rank": 512,
                "layer_types": [
                  "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
                  "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
                  "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
                  "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
                  "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
                  "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
                  "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
                  "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
                  "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
                  "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
                  "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
                  "linear_attention"
                ],
                "linear_attn_config": {
                  "num_heads": 64,
                  "gate_lower_bound": -5.0,
                  "head_dim": 128,
                  "short_conv_kernel_size": 4,
                  "kda_layers": [
                    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
                    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37,
                    38, 40, 41, 42, 44
                  ],
                  "full_attn_layers": [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
                },
                "max_position_embeddings": 1048576,
                "mhc": true,
                "mla_use_nope": true,
                "mlp_layer_types": [
                  "dense", "dense", "dense", "sparse", "sparse", "sparse", "sparse", "sparse",
                  "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse",
                  "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse",
                  "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse",
                  "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse",
                  "sparse", "sparse", "sparse", "sparse", "sparse"
                ],
                "moe_intermediate_size": 2048,
                "moe_router_dtype": "float32",
                "n_group": 1,
                "n_routed_experts": 288,
                "n_shared_experts": 1,
                "norm_topk_prob": true,
                "num_attention_heads": 64,
                "num_experts_per_tok": 8,
                "num_hidden_layers": 45,
                "num_key_value_heads": 64,
                "num_nextn_predict_layers": 1,
                "output_router_logits": false,
                "pad_token_id": 154820,
                "q_lora_rank": 1536,
                "qk_head_dim": 256,
                "qk_nope_head_dim": 256,
                "qk_rope_head_dim": 0,
                "rms_norm_eps": 1e-5,
                "routed_scaling_factor": 2.5,
                "router_aux_loss_coef": 0.001,
                "scoring_func": "sigmoid",
                "swiglu_limit": 10.0,
                "tie_word_embeddings": false,
                "topk_group": 1,
                "topk_method": "noaux_tc",
                "use_cache": true,
                "v_head_dim": 256,
                "vocab_size": 154880
              },
              "vision_config": {
                "attention_bias": true,
                "attention_dropout": 0.0,
                "depth": 24,
                "hidden_act": "silu",
                "hidden_size": 1024,
                "image_size": 448,
                "in_channels": 3,
                "initializer_range": 0.02,
                "intermediate_size": 4096,
                "model_type": "glm5_next_vision",
                "num_heads": 16,
                "out_hidden_size": 4096,
                "patch_size": 14,
                "projection_intermediate_size": 10240,
                "rms_norm_eps": 1e-5,
                "spatial_merge_size": 2,
                "swiglu_limit": 10.0,
                "temporal_patch_size": 2
              }
            }
            """#

        private func fixture() throws -> [String: Any] {
            try #require(
                JSONSerialization.jsonObject(with: Data(Self.pinnedArchitecture.utf8))
                    as? [String: Any])
        }

        // Exact semantic snapshot of both native namespaces in:
        // mlx-community/GLM-5.3-Flash-4bit@fa04cbc3622b16aa7475404060b05e506d81bd4e/config.json
        // The 175 entries follow this complete pattern, including the MTP block at layer 45.
        // This reconstructs fixture data only; it is not a model default or loader policy.
        private func candidateFixture() throws -> [String: Any] {
            var root = try fixture()
            var native: [String: Any] = ["group_size": 64, "bits": 4]
            for layer in 0 ... 45 {
                let prefix = "model.layers.\(layer).mlp"
                let downProjection = layer < 3 ? "down_proj" : "switch_mlp.down_proj"
                native["\(prefix).\(downProjection)"] = ["group_size": 64, "bits": 5]
                if layer >= 3 {
                    for projection in ["gate_proj", "up_proj", "down_proj"] {
                        native["\(prefix).shared_experts.\(projection)"] = [
                            "group_size": 64, "bits": 6,
                        ]
                    }
                }
            }
            root["quantization"] = native
            root["quantization_config"] = native
            return root
        }

        private func decode(_ root: [String: Any]) throws -> GLM5NextConfiguration {
            try JSONDecoder().decode(
                GLM5NextConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
        }

        private func setting(_ root: [String: Any], _ path: String, to value: Any) throws
            -> [String: Any]
        {
            var result = root
            let components = path.split(separator: ".", maxSplits: 1).map(String.init)
            if components.count == 1 {
                result[path] = value
            } else {
                let child = try #require(root[components[0]] as? [String: Any])
                result[components[0]] = try setting(child, components[1], to: value)
            }
            return result
        }

        @Test func pinnedArchitecturePreservesEveryFieldOnRoundTrip() throws {
            let root = try fixture()
            let config = try decode(root)
            let encoded = try #require(
                JSONSerialization.jsonObject(with: JSONEncoder().encode(config)) as? [String: Any])
            #expect(NSDictionary(dictionary: root).isEqual(to: encoded))
            #expect(
                try JSONDecoder().decode(
                    GLM5NextConfiguration.self, from: JSONEncoder().encode(config)) == config)
            #expect(config.modelType == "glm5_next")
            #expect(config.textConfig.hiddenSize == 4096)
            #expect(config.textConfig.vocabularySize == 154880)
            #expect(config.textConfig.hiddenLayers == 45)
            #expect(config.textConfig.headDim == 0)
            #expect(config.textConfig.qkNopeHeadDim == 256 && config.textConfig.qkRopeHeadDim == 0)
            #expect(config.textConfig.attentionHeads == 64 && config.textConfig.kvHeads == 64)
            #expect(config.textConfig.qLoraRank == 1536 && config.textConfig.kvLoraRank == 512)
            #expect(config.textConfig.mhc && config.textConfig.hcMult == 4)
            #expect(config.textConfig.hcEps == 1e-6 && config.textConfig.hcSinkhornIters == 20)
            #expect(config.textConfig.linearAttention.numHeads == 64)
            #expect(config.textConfig.linearAttention.headDim == 128)
            #expect(config.textConfig.linearAttention.gateLowerBound == -5)
            #expect(config.textConfig.linearAttention.shortConvKernelSize == 4)
            #expect(config.textConfig.linearAttention.kdaLayers.count == 34)
            #expect(config.textConfig.linearAttention.kdaLayers.last == 44)
            #expect(
                config.textConfig.linearAttention.fullAttentionLayers == [
                    3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43,
                ])
            #expect(
                config.textConfig.nRoutedExperts == 288 && config.textConfig.nSharedExperts == 1)
            #expect(
                config.textConfig.numExpertsPerTok == 8
                    && config.textConfig.routedScalingFactor == 2.5)
            #expect(config.textConfig.numNextNPredictLayers == 1)
            #expect(config.textConfig.indexShareForMTPIteration)
            #expect(config.textConfig.eosTokenIds == .multiple([154820, 154827, 154829]))
            #expect(
                config.visionConfig.hiddenSize == 1024 && config.visionConfig.outHiddenSize == 4096)
            #expect(
                config.visionConfig.depth == 24
                    && config.visionConfig.projectionIntermediateSize == 10240)
            #expect(config.quantizationConfig == nil)
        }

        @Test func sourceQuantizationMetadataIsPreservedNotUsedAsAnExecutionPolicy() throws {
            let metadata: [String: Any] = [
                "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": [
                    "attn_mha", "attn_mqa", "model.layers.45.eh_proj", "visual.post_layernorm",
                ],
            ]
            let root = try setting(fixture(), "quantization_config", to: metadata)
            let config = try decode(root)
            guard case .sourceFP8(let source)? = config.quantizationConfig else {
                Issue.record("Expected source FP8 metadata")
                return
            }
            #expect(source.weightBlockSize == [128, 128])
            #expect(source.modulesToNotConvert.count == 4)
            #expect(config.nativeQuantization == nil)
            let encoded = try #require(
                JSONSerialization.jsonObject(with: JSONEncoder().encode(config)) as? [String: Any])
            #expect(NSDictionary(dictionary: root).isEqual(to: encoded))
        }

        @Test func actualNativeCandidateRetainsBothNamespacesAndEveryMixedPrecisionOverride() throws
        {
            let root = try candidateFixture()
            let config = try decode(root)
            let native = try #require(config.nativeQuantization)
            guard case .native(let alias)? = config.quantizationConfig else {
                Issue.record("Expected native quantization_config, not source FP8 metadata")
                return
            }
            #expect(native == alias && config.quantization == native)
            #expect(native.defaultQuantization.bits == 4)
            #expect(native.defaultQuantization.groupSize == 64)
            #expect(native.defaultQuantization.mode == .affine)
            #expect(native.perModule.count == 175 && native.skippedModules.isEmpty)
            #expect(native.perModule.values.filter { $0.bits == 5 }.count == 46)
            #expect(native.perModule.values.filter { $0.bits == 6 }.count == 129)
            #expect(native.perModule.values.allSatisfy { $0.groupSize == 64 && $0.mode == .affine })
            let lookup = native.perLayerQuantization
            #expect(
                lookup.quantization(layer: "model.layers.45.mlp.switch_mlp.down_proj")?.bits == 5)
            #expect(
                lookup.quantization(layer: "model.layers.45.mlp.shared_experts.gate_proj")?.bits
                    == 6)
            #expect(lookup.quantization(layer: "model.layers.0.self_attn.q_proj")?.bits == 4)
            let base = try JSONDecoder().decode(
                BaseConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
            #expect(base.perLayerQuantization?.quantization == native.defaultQuantization)
            for (module, policy) in native.perModule {
                #expect(base.perLayerQuantization?.quantization(layer: module) == policy)
            }
            let encoded = try #require(
                JSONSerialization.jsonObject(with: JSONEncoder().encode(config)) as? [String: Any])
            #expect(NSDictionary(dictionary: root).isEqual(to: encoded))
            #expect(
                try JSONDecoder().decode(
                    GLM5NextConfiguration.self, from: JSONEncoder().encode(config)) == config)
        }

        @Test(arguments: ["quantization", "quantization_config"])
        func nativeNamespaceIsPreservedWithoutInventingAnAlias(_ namespace: String) throws {
            var root = try candidateFixture()
            root.removeValue(
                forKey: namespace == "quantization" ? "quantization_config" : "quantization")
            let config = try decode(root)
            #expect(config.nativeQuantization?.perModule.count == 175)
            let encoded = try #require(
                JSONSerialization.jsonObject(with: JSONEncoder().encode(config)) as? [String: Any])
            #expect(NSDictionary(dictionary: root).isEqual(to: encoded))
        }

        @Test(arguments: ["default", "override", "missingOverride", "skip"])
        func conflictingNativeNamespacesFailInsteadOfChoosingOne(_ mutation: String) throws {
            var root = try candidateFixture()
            var alias = try #require(root["quantization_config"] as? [String: Any])
            let module = "model.layers.45.mlp.switch_mlp.down_proj"
            switch mutation {
            case "default": alias["bits"] = 6
            case "override": alias[module] = ["group_size": 64, "bits": 6]
            case "missingOverride": alias.removeValue(forKey: module)
            case "skip": alias[module] = false
            default:
                Issue.record("Unknown mutation")
                return
            }
            root["quantization_config"] = alias
            #expect(
                throws: GLM5NextConfigurationError.invalid(
                    field: "quantization/quantization_config",
                    reason: "conflicting native default, module, or skip policies")
            ) { try decode(root) }
        }

        @Test func explicitAffineSpellingAndStorageMetadataSurviveWithoutChangingPolicy() throws {
            var root = try candidateFixture()
            var native = try #require(root["quantization"] as? [String: Any])
            var alias = native
            native["model.embed_tokens"] = false
            alias["model.embed_tokens"] = false
            alias["mode"] = "affine"
            alias["model.layers.45.mlp.switch_mlp.down_proj"] = [
                "group_size": 64, "bits": 5, "mode": "affine",
            ]
            alias["quant_method"] = "mlx"
            alias["linear_class"] = "QuantizedLinear"
            alias["quantization_mode"] = "affine"
            root["quantization"] = native
            root["quantization_config"] = alias
            let config = try decode(root)
            #expect(
                config.nativeQuantization?.perLayerQuantization.quantization(
                    layer: "model.embed_tokens") == nil)
            let encoded = try #require(
                JSONSerialization.jsonObject(with: JSONEncoder().encode(config)) as? [String: Any])
            #expect(NSDictionary(dictionary: root).isEqual(to: encoded))
        }

        @Test func sourceFP8AndNativeMetadataRemainDistinctWhenBothArePresent() throws {
            var root = try candidateFixture()
            let source: [String: Any] = [
                "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                "weight_block_size": [128, 128], "modules_to_not_convert": ["lm_head"],
            ]
            root["quantization_config"] = source
            let config = try decode(root)
            #expect(config.nativeQuantization?.perModule.count == 175)
            guard case .sourceFP8(let metadata)? = config.quantizationConfig else {
                Issue.record("Source metadata was mistaken for native policy")
                return
            }
            #expect(metadata.quantizationMethod == "fp8")
            let encoded = try #require(
                JSONSerialization.jsonObject(with: JSONEncoder().encode(config)) as? [String: Any])
            #expect(NSDictionary(dictionary: root).isEqual(to: encoded))
        }

        @Test func moduleNamespacesAreNotSilentlyRewritten() throws {
            var root = try candidateFixture()
            var native = try #require(root["quantization"] as? [String: Any])
            let original = "model.layers.45.mlp.shared_experts.gate_proj"
            let alternate = "model.language_model.layers.45.mlp.shared_experts.gate_proj"
            native[alternate] = ["group_size": 64, "bits": 5]
            root["quantization"] = native
            root["quantization_config"] = native
            let config = try decode(root)
            #expect(config.nativeQuantization?.perModule[original]?.bits == 6)
            #expect(config.nativeQuantization?.perModule[alternate]?.bits == 5)
            let encoded = try #require(
                JSONSerialization.jsonObject(with: JSONEncoder().encode(config)) as? [String: Any])
            #expect(NSDictionary(dictionary: root).isEqual(to: encoded))
        }

        @Test(arguments: [
            "bits", "group", "mode", "stringBits", "nullMode", "trueOverride",
            "missingGroup", "badOverride", "unknownOverrideField", "mixedFP8", "nonstringMetadata",
        ])
        func malformedNativeStorageMetadataIsRejected(_ mutation: String) throws {
            var root = try candidateFixture()
            var native = try #require(root["quantization"] as? [String: Any])
            let module = "model.layers.45.mlp.switch_mlp.down_proj"
            switch mutation {
            case "bits": native["bits"] = 7
            case "group": native["group_size"] = 0
            case "mode": native["mode"] = "not-a-mode"
            case "stringBits": native["bits"] = "4"
            case "nullMode": native["mode"] = NSNull()
            case "trueOverride": native[module] = true
            case "missingGroup": native[module] = ["bits": 5]
            case "badOverride": native[module] = ["group_size": 64, "bits": 9]
            case "unknownOverrideField":
                native[module] = ["group_size": 64, "bits": 5, "bit": 4]
            case "mixedFP8": native["quant_method"] = "fp8"
            case "nonstringMetadata": native["linear_class"] = false
            default:
                Issue.record("Unknown mutation")
                return
            }
            root["quantization"] = native
            root["quantization_config"] = native
            #expect(throws: (any Error).self) { try decode(root) }
        }

        @Test func consistentAlternateGeometryIsNotReplacedByCheckpointConstants() throws {
            var root = try fixture()
            for (path, value) in [
                ("text_config.hidden_size", 1024), ("text_config.intermediate_size", 3072),
                ("text_config.moe_intermediate_size", 512), ("text_config.num_attention_heads", 8),
                ("text_config.num_key_value_heads", 8), ("text_config.q_lora_rank", 192),
                ("text_config.kv_lora_rank", 64), ("text_config.qk_head_dim", 32),
                ("text_config.qk_nope_head_dim", 32), ("text_config.v_head_dim", 48),
                ("text_config.n_routed_experts", 16), ("text_config.n_group", 4),
                ("text_config.topk_group", 2), ("text_config.num_experts_per_tok", 4),
                ("text_config.hc_mult", 2), ("text_config.num_nextn_predict_layers", 2),
                ("text_config.linear_attn_config.num_heads", 12),
                ("text_config.linear_attn_config.head_dim", 24),
                ("vision_config.out_hidden_size", 1024),
            ] {
                root = try setting(root, path, to: value)
            }
            let config = try decode(root)
            #expect(config.textConfig.hiddenSize == 1024)
            #expect(config.textConfig.attentionHeads == 8)
            #expect(config.textConfig.linearAttention.numHeads == 12)
            #expect(config.textConfig.vHeadDim == 48)
            #expect(config.textConfig.hcMult == 2 && config.textConfig.numNextNPredictLayers == 2)
            #expect(config.textConfig.layerTypes.count == 45)
        }

        @Test(arguments: [
            "text_config.head_dim", "text_config.qk_head_dim", "text_config.qk_rope_head_dim",
            "text_config.mla_use_nope", "text_config.num_key_value_heads",
            "text_config.q_lora_rank",
            "text_config.index_kpool", "text_config.index_topk", "text_config.hc_mult",
            "text_config.hc_sinkhorn_iters", "text_config.hc_eps", "text_config.n_group",
            "text_config.topk_group", "text_config.num_experts_per_tok",
            "text_config.n_shared_experts",
            "text_config.first_k_dense_replace", "text_config.num_nextn_predict_layers",
            "text_config.scoring_func", "text_config.moe_router_dtype", "text_config.hidden_size",
            "text_config.linear_attn_config.gate_lower_bound",
            "text_config.linear_attn_config.head_dim",
            "text_config.linear_attn_config.short_conv_kernel_size",
            "text_config.attention_dropout", "text_config.rms_norm_eps", "text_config.swiglu_limit",
            "vision_config.num_heads", "vision_config.image_size", "vision_config.out_hidden_size",
            "vision_config.temporal_patch_size", "vision_config.spatial_merge_size",
            "text_config.eos_token_id", "image_token_id", "tie_word_embeddings",
        ])
        func malformedGeometryIsRejected(_ path: String) throws {
            let invalid: [String: Any] = [
                "text_config.head_dim": 64, "text_config.qk_head_dim": 255,
                "text_config.qk_rope_head_dim": 32,
                "text_config.mla_use_nope": false, "text_config.num_key_value_heads": 8,
                "text_config.q_lora_rank": 0, "text_config.index_kpool": 0,
                "text_config.index_topk": 2049,
                "text_config.hc_mult": Int.max, "text_config.hc_sinkhorn_iters": 0,
                "text_config.hc_eps": -1,
                "text_config.n_group": 5, "text_config.topk_group": 2,
                "text_config.num_experts_per_tok": 289,
                "text_config.n_shared_experts": 0, "text_config.first_k_dense_replace": 4,
                "text_config.num_nextn_predict_layers": -1, "text_config.scoring_func": "softmax",
                "text_config.moe_router_dtype": "bfloat16", "text_config.hidden_size": Int.max,
                "text_config.linear_attn_config.gate_lower_bound": 0,
                "text_config.linear_attn_config.head_dim": Int.max,
                "text_config.linear_attn_config.short_conv_kernel_size": 0,
                "text_config.attention_dropout": 1, "text_config.rms_norm_eps": 0,
                "text_config.swiglu_limit": -10,
                "vision_config.num_heads": 15, "vision_config.image_size": 449,
                "vision_config.out_hidden_size": 2048,
                "vision_config.temporal_patch_size": 0, "vision_config.spatial_merge_size": 3,
                "text_config.eos_token_id": [154880], "image_token_id": 154855,
                "tie_word_embeddings": true,
            ]
            let root = try setting(fixture(), path, to: #require(invalid[path]))
            #expect(throws: GLM5NextConfigurationError.self) { try decode(root) }
        }

        @Test(arguments: [
            "text_config.swiglu_limit", "text_config.routed_scaling_factor",
            "text_config.hc_eps", "text_config.rms_norm_eps",
            "vision_config.swiglu_limit", "vision_config.rms_norm_eps",
            "text_config.linear_attn_config.gate_lower_bound",
        ])
        func nativeFloatParametersRejectOverflowAndUnderflow(_ path: String) throws {
            for magnitude in [1e300, 1e-300] {
                let value = path.hasSuffix("gate_lower_bound") ? -magnitude : magnitude
                let root = try setting(fixture(), path, to: value)
                #expect(throws: GLM5NextConfigurationError.self) { try decode(root) }
            }
        }

        @Test(arguments: [
            "missing", "duplicate", "outOfRange", "overlap", "reordered", "wrongLayer", "mtpLayer",
        ])
        func divergentLinearSchedulesAreRejected(_ mutation: String) throws {
            let root = try fixture()
            let text = try #require(root["text_config"] as? [String: Any])
            let linear = try #require(text["linear_attn_config"] as? [String: Any])
            var kda = try #require(linear["kda_layers"] as? [Int])
            switch mutation {
            case "missing": kda.removeLast()
            case "duplicate": kda.append(44)
            case "outOfRange": kda.append(-1)
            case "overlap": kda.append(3)
            case "reordered": kda.swapAt(0, 1)
            case "wrongLayer": kda[0] = 3
            case "mtpLayer": kda.append(45)
            default:
                Issue.record("Unknown mutation")
                return
            }
            let changed = try setting(root, "text_config.linear_attn_config.kda_layers", to: kda)
            #expect(throws: GLM5NextConfigurationError.self) { try decode(changed) }
        }

        @Test(arguments: ["layer_types", "mlp_layer_types", "indexer_types"])
        func backboneScheduleLengthIsNotExtendedForMTP(_ key: String) throws {
            let root = try fixture()
            let text = try #require(root["text_config"] as? [String: Any])
            var schedule = try #require(text[key] as? [String])
            schedule.append(try #require(schedule.last))
            let changed = try setting(root, "text_config.\(key)", to: schedule)
            #expect(throws: GLM5NextConfigurationError.self) { try decode(changed) }
        }

        @Test func scheduleMustBeExplicitButNeedNotRepeatThePinnedCadence() throws {
            var root = try fixture()
            let text = try #require(root["text_config"] as? [String: Any])
            var layers = try #require(text["layer_types"] as? [String])
            layers[0] = "deepseek_sparse_attention"
            layers[3] = "linear_attention"
            root = try setting(root, "text_config.layer_types", to: layers)
            let linear = layers.indices.filter { layers[$0] == "linear_attention" }
            let sparse = layers.indices.filter { layers[$0] == "deepseek_sparse_attention" }
            root = try setting(root, "text_config.linear_attn_config.kda_layers", to: linear)
            root = try setting(root, "text_config.linear_attn_config.full_attn_layers", to: sparse)
            #expect(try decode(root).textConfig.layerTypes[0] == .deepseekSparseAttention)
        }

        @Test(arguments: ["4096", "4096.5", "true", "null", "\"4096\""])
        func integerFieldsDoNotCoerceStringsFractionsBooleansOrNull(_ literal: String) throws {
            let changed = Self.pinnedArchitecture.replacingOccurrences(
                of: "\"hidden_size\": 4096", with: "\"hidden_size\": \(literal)")
            if literal == "4096" {
                #expect(
                    try JSONDecoder().decode(GLM5NextConfiguration.self, from: Data(changed.utf8))
                        .textConfig.hiddenSize == 4096)
            } else {
                #expect(throws: (any Error).self) {
                    try JSONDecoder().decode(GLM5NextConfiguration.self, from: Data(changed.utf8))
                }
            }
        }

        @Test func numericJSONSpellingAndEOSScalarArePreservedSemantically() throws {
            var root = try setting(fixture(), "text_config.attention_dropout", to: 0)
            root = try setting(root, "text_config.linear_attn_config.gate_lower_bound", to: -5)
            root = try setting(root, "text_config.eos_token_id", to: 154820)
            let config = try decode(root)
            #expect(config.textConfig.attentionDropout == 0)
            #expect(config.textConfig.linearAttention.gateLowerBound == -5)
            #expect(config.textConfig.eosTokenIds == .single(154820))
            let encoded = try #require(
                JSONSerialization.jsonObject(with: JSONEncoder().encode(config)) as? [String: Any])
            let encodedText = try #require(encoded["text_config"] as? [String: Any])
            #expect(encodedText["eos_token_id"] as? Int == 154820)
            let stringNumber = try setting(
                root, "text_config.linear_attn_config.gate_lower_bound", to: "-5.0")
            #expect(throws: (any Error).self) { try decode(stringNumber) }
            let stringEOS = try setting(root, "text_config.eos_token_id", to: ["154820"])
            #expect(throws: (any Error).self) { try decode(stringEOS) }
        }

        @Test(arguments: ["NaN", "Infinity", "-Infinity"])
        func nonfiniteValuesAreRejectedEvenWithPermissiveJSONDecoder(_ value: String) throws {
            let root = try setting(fixture(), "text_config.hc_eps", to: value)
            let decoder = JSONDecoder()
            decoder.nonConformingFloatDecodingStrategy = .convertFromString(
                positiveInfinity: "Infinity", negativeInfinity: "-Infinity", nan: "NaN")
            let data = try JSONSerialization.data(withJSONObject: root)
            #expect(throws: GLM5NextConfigurationError.self) {
                try decoder.decode(GLM5NextConfiguration.self, from: data)
            }
        }

        @Test(arguments: ["shortBlock", "negativeBlock", "stringBlock", "foreignMethod"])
        func malformedSourceQuantizationIsRejected(_ mutation: String) throws {
            var metadata: [String: Any] = [
                "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                "weight_block_size": [128, 128], "modules_to_not_convert": ["lm_head"],
            ]
            switch mutation {
            case "shortBlock": metadata["weight_block_size"] = [128]
            case "negativeBlock": metadata["weight_block_size"] = [128, -128]
            case "stringBlock": metadata["weight_block_size"] = ["128", "128"]
            case "foreignMethod": metadata["quant_method"] = "affine"
            default:
                Issue.record("Unknown mutation")
                return
            }
            let root = try setting(fixture(), "quantization_config", to: metadata)
            #expect(throws: (any Error).self) { try decode(root) }
        }

        @Test(arguments: ["text_config", "vision_config"])
        func missingArchitectureFieldsDoNotReceiveGuessedDefaults(_ section: String) throws {
            var root = try fixture()
            var child = try #require(root[section] as? [String: Any])
            child.removeValue(forKey: "hidden_size")
            root[section] = child
            #expect(throws: (any Error).self) { try decode(root) }
        }

        @Test func unknownGeometryKeysFailClosedWithDeterministicField() throws {
            let changed = try setting(fixture(), "text_config.linear_num_heads", to: 8)
            #expect(
                throws: GLM5NextConfigurationError.invalid(
                    field: "text_config.linear_num_heads", reason: "unrecognized field")
            ) {
                try decode(changed)
            }
        }

        @Test func foreignLayerAndUnqualifiedIndexerModesFailClosed() throws {
            let root = try fixture()
            let unknownLayer = try setting(
                root, "text_config.layer_types", to: Array(repeating: "full_attention", count: 45))
            #expect(throws: (any Error).self) { try decode(unknownLayer) }
            let shared = try setting(
                root, "text_config.indexer_types", to: Array(repeating: "shared", count: 45))
            #expect(throws: GLM5NextConfigurationError.self) { try decode(shared) }
        }
    }
}
