import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {

    /// Forward-pass tests of more text models with tiny random models:
    /// hidden size 32, 2 or 3 layers, 4 query heads over 2 key heads,
    /// vocabulary 64, and 4 experts with top-2 routing for the MoE models.
    ///
    /// Tolerance 1e-4 for the float32 comparisons: the cached path and the
    /// full path differ in the order of the attention sums, and the
    /// recurrent layers run chunk by chunk. The differences are near 1e-6
    /// for logits of size 1 to 5.
    ///
    /// A check that fails because of a known production defect runs inside
    /// `withKnownIssue` (see `ModelCase.knownIssues`).
    @Suite
    struct TextModelForwardPassTests {

        static var jamba: [String: Any] {
            [
                "model_type": "jamba", "hidden_size": 32, "intermediate_size": 48,
                "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                "attn_layer_offset": 1, "attn_layer_period": 2, "expert_layer_offset": 1,
                "expert_layer_period": 2, "mamba_d_conv": 4, "mamba_d_state": 8,
                "mamba_expand": 2, "num_experts": 4, "num_experts_per_tok": 2,
                "rms_norm_eps": 1e-6, "max_position_embeddings": 256, "vocab_size": 64,
                "tie_word_embeddings": false,
            ]
        }

        static var baichuan: [String: Any] {
            [
                "vocab_size": 64, "hidden_size": 32, "intermediate_size": 48,
                "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                "rope_theta": 10000, "sliding_window": 4, "sliding_window_layers": [0],
                "conv_window": 2, "rms_norm_eps": 1e-6, "tie_word_embeddings": false,
            ]
        }

        static let cases: [ModelCase] = [
            ModelCase(
                "Apertus",
                expectedShapes: [
                    "model.layers.0.attention_layernorm.weight": [32],
                    "model.layers.0.self_attn.q_norm.weight": [8],
                    "model.layers.1.mlp.up_proj.weight": [48, 32],
                    "model.layers.1.mlp.act_fn.alpha_p": [1],
                    "lm_head.weight": [64, 32],
                ],
                droppedKeys: ["model.layers.0.self_attn.rotary_emb.inv_freq": [4]]
            ) { seed in
                try ModelCase.build(
                    ApertusConfiguration.self,
                    [
                        "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2,
                        "num_attention_heads": 4, "num_key_value_heads": 2,
                        "rms_norm_eps": 1e-6, "vocab_size": 64, "tie_word_embeddings": false,
                    ], seed: seed
                ) { ApertusModel($0) }
            },
            ModelCase(
                "InternLM2",
                expectedShapes: [
                    "model.tok_embeddings.weight": [64, 32],
                    "model.layers.0.attention.wqkv.weight": [64, 32],
                    "model.layers.0.attention.wo.weight": [32, 32],
                    "model.layers.1.feed_forward.w2.weight": [32, 48],
                    "output.weight": [64, 32],
                ],
                droppedKeys: ["model.layers.0.attention.rope.inv_freq": [4]]
            ) { seed in
                try ModelCase.build(
                    InternLM2Configuration.self,
                    [
                        "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                        "num_attention_heads": 4, "num_key_value_heads": 2,
                        "rms_norm_eps": 1e-6, "vocab_size": 64, "max_position_embeddings": 256,
                    ], seed: seed
                ) { InternLM2Model($0) }
            },
            ModelCase(
                "Mistral3Text",
                expectedShapes: [
                    "model.layers.0.self_attn.q_proj.weight": [32, 32],
                    "model.layers.1.mlp.gate_proj.weight": [48, 32],
                    "lm_head.weight": [64, 32],
                ],
                droppedKeys: ["model.layers.0.self_attn.rotary_emb.inv_freq": [4]]
            ) { seed in
                try ModelCase.build(
                    Mistral3TextConfiguration.self,
                    [
                        "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                        "rms_norm_eps": 1e-6, "vocab_size": 64,
                        "layer_types": ["sliding_attention", "full_attention"],
                        "sliding_window": 4,
                        "rope_parameters": [
                            "rope_type": "default", "rope_theta": 10000,
                            "llama_4_scaling_beta": 0.1, "original_max_position_embeddings": 4,
                        ],
                    ], seed: seed
                ) { Mistral3TextModel($0) }
            },
            ModelCase(
                "MiniMax",
                expectedShapes: [
                    "model.layers.0.self_attn.q_norm.weight": [32],
                    "model.layers.0.block_sparse_moe.gate.weight": [4, 32],
                    "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight": [4, 16, 32],
                ],
                checkpoint: {
                    CheckpointLayout.splitExperts(
                        $0, stacked: "switch_mlp", perExpert: "experts",
                        names: ["gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"])
                }
            ) { seed in
                try ModelCase.build(
                    MiniMaxConfiguration.self,
                    [
                        "model_type": "minimax", "hidden_size": 32, "intermediate_size": 16,
                        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                        "max_position_embeddings": 256, "num_experts_per_tok": 2,
                        "num_local_experts": 4, "shared_intermediate_size": 16,
                        "num_hidden_layers": 2, "rms_norm_eps": 1e-6, "rope_theta": 10000,
                        "rotary_dim": 4, "vocab_size": 64, "tie_word_embeddings": false,
                        "scoring_func": "sigmoid", "use_qk_norm": true,
                    ], seed: seed
                ) { MiniMaxModel($0) }
            },
            ModelCase(
                "BailingMoe",
                expectedShapes: [
                    "model.word_embeddings.weight": [64, 32],
                    "model.layers.0.attention.query_key_value.weight": [64, 32],
                    "model.layers.1.mlp.gate.gate_proj.weight": [4, 32],
                    "model.layers.1.mlp.gate.expert_bias": [4],
                    "model.layers.1.mlp.switch_mlp.up_proj.weight": [4, 16, 32],
                ]
            ) { seed in
                try ModelCase.build(
                    BailingMoeConfiguration.self,
                    [
                        "model_type": "bailing_moe", "hidden_size": 32, "intermediate_size": 48,
                        "moe_intermediate_size": 16, "num_experts": 4, "num_shared_experts": 1,
                        "norm_topk_prob": true, "num_attention_heads": 4,
                        "num_experts_per_tok": 2, "num_hidden_layers": 2,
                        "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "rope_theta": 10000,
                        "vocab_size": 64, "first_k_dense_replace": 1, "use_bias": false,
                        "use_qkv_bias": true, "use_qk_norm": true, "tie_word_embeddings": false,
                        "partial_rotary_factor": 0.5, "moe_router_enable_expert_bias": true,
                        "routed_scaling_factor": 1.0, "score_function": "sigmoid",
                        "n_group": 2, "topk_group": 1,
                    ], seed: seed
                ) { BailingMoeModel($0) }
            },
            ModelCase(
                "GLM4MoE",
                expectedShapes: [
                    "model.layers.0.mlp.gate_proj.weight": [48, 32],
                    "model.layers.1.mlp.gate.weight": [4, 32],
                    "model.layers.1.mlp.switch_mlp.down_proj.weight": [4, 32, 16],
                ],
                droppedKeys: ["model.layers.2.eh_proj.weight": [32, 64]],
                checkpoint: {
                    CheckpointLayout.splitExperts($0, stacked: "switch_mlp", perExpert: "experts")
                }
            ) { seed in
                try ModelCase.build(
                    GLM4MoEConfiguration.self,
                    [
                        "model_type": "glm4_moe", "vocab_size": 64, "hidden_size": 32,
                        "intermediate_size": 48, "max_position_embeddings": 256,
                        "moe_intermediate_size": 16, "norm_topk_prob": true,
                        "num_attention_heads": 4, "n_group": 1, "head_dim": 8, "topk_group": 1,
                        "routed_scaling_factor": 1.0, "num_experts_per_tok": 2,
                        "first_k_dense_replace": 1, "num_hidden_layers": 2,
                        "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "rope_theta": 10000,
                        "use_qk_norm": true, "tie_word_embeddings": false,
                        "attention_bias": false, "partial_rotary_factor": 0.5,
                        "n_shared_experts": 1, "n_routed_experts": 4,
                    ], seed: seed
                ) { GLM4MoEModel($0) }
            },
            ModelCase(
                "AfMoE",
                expectedShapes: [
                    "model.layers.0.self_attn.gate_proj.weight": [32, 32],
                    "model.layers.1.mlp.router.gate.weight": [4, 32],
                    "model.layers.1.mlp.expert_bias": [4],
                    "model.layers.1.mlp.experts.gate_proj.weight": [4, 16, 32],
                ],
                droppedKeys: ["model.layers.0.self_attn.rotary_emb.inv_freq": [4]],
                checkpoint: {
                    CheckpointLayout.splitExperts($0, stacked: "experts", perExpert: "experts")
                }
            ) { seed in
                try ModelCase.build(
                    AfMoEConfiguration.self,
                    [
                        "layer_types": ["sliding_attention", "full_attention"],
                        "vocab_size": 64, "hidden_size": 32, "intermediate_size": 48,
                        "moe_intermediate_size": 16, "num_hidden_layers": 2,
                        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                        "num_experts": 4, "num_experts_per_tok": 2, "num_shared_experts": 1,
                        "num_dense_layers": 1, "sliding_window": 4,
                    ], seed: seed
                ) { AfMoEModel($0) }
            },
            ModelCase(
                "LFM2MoE",
                expectedShapes: [
                    "model.layers.0.conv.conv.weight": [32, 3, 1],
                    "model.layers.1.self_attn.q_proj.weight": [32, 32],
                    "model.layers.2.feed_forward.gate.weight": [4, 32],
                    "model.layers.2.feed_forward.switch_mlp.gate_proj.weight": [4, 16, 32],
                ],
                checkpoint: { weights in
                    var result: [String: MLXArray] = [:]
                    for (key, value) in CheckpointLayout.splitExperts(
                        weights, stacked: "switch_mlp", perExpert: "experts",
                        names: ["gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"])
                    {
                        result[key] =
                            key.hasSuffix("conv.conv.weight")
                            ? value.transposed(0, 2, 1) : value
                    }
                    return result
                }
            ) { seed in
                try ModelCase.build(
                    LFM2MoEConfiguration.self,
                    [
                        "model_type": "lfm2_moe", "vocab_size": 64, "hidden_size": 32,
                        "intermediate_size": 48, "moe_intermediate_size": 16,
                        "num_hidden_layers": 3, "num_experts": 4, "num_experts_per_tok": 2,
                        "norm_topk_prob": true, "num_attention_heads": 4,
                        "num_key_value_heads": 2, "max_position_embeddings": 256,
                        "use_expert_bias": true, "num_dense_layers": 2, "norm_eps": 1e-5,
                        "conv_bias": false, "conv_L_cache": 3,
                        "layer_types": ["conv", "full_attention", "conv"],
                    ], seed: seed
                ) { LFM2MoEModel($0) }
            },
            ModelCase(
                "LFM2",
                expectedShapes: [
                    "model.layers.0.conv.in_proj.weight": [96, 32],
                    "model.layers.1.self_attn.k_proj.weight": [16, 32],
                    "model.layers.2.feed_forward.w1.weight": [48, 32],
                ],
                checkpoint: { weights in
                    weights.mapValues { $0 }.reduce(into: [:]) { result, item in
                        result[item.key] =
                            item.key.hasSuffix("conv.conv.weight")
                            ? item.value.transposed(0, 2, 1) : item.value
                    }
                }
            ) { seed in
                try ModelCase.build(
                    LFM2Configuration.self,
                    [
                        "hidden_size": 32, "num_hidden_layers": 3, "num_attention_heads": 4,
                        "num_key_value_heads": 2, "norm_eps": 1e-5, "vocab_size": 64,
                        "block_ff_dim": 48, "block_auto_adjust_ff_dim": false,
                        "conv_L_cache": 3, "layer_types": ["conv", "full_attention", "conv"],
                    ], seed: seed
                ) { LFM2Model($0) }
            },
            ModelCase(
                "MiMoV2Flash",
                expectedShapes: [
                    "model.layers.0.self_attn.attention_sink_bias": [4],
                    "model.layers.1.mlp.switch_mlp.gate_proj.weight": [4, 16, 32],
                ],
                droppedKeys: ["model.mtp.layers.0.weight": [4]],
                checkpoint: {
                    CheckpointLayout.splitExperts($0, stacked: "switch_mlp", perExpert: "experts")
                }
            ) { seed in
                try ModelCase.build(
                    MiMoV2FlashConfiguration.self,
                    [
                        "model_type": "mimo_v2_flash", "num_experts_per_tok": 2,
                        "hybrid_layer_pattern": [1, 0], "moe_layer_freq": [0, 1],
                        "add_swa_attention_sink_bias": true,
                        "add_full_attention_sink_bias": false, "sliding_window_size": 4,
                        "vocab_size": 64, "hidden_size": 32, "intermediate_size": 48,
                        "moe_intermediate_size": 16, "num_hidden_layers": 2,
                        "num_attention_heads": 4, "num_key_value_heads": 2,
                        "topk_method": "noaux_tc", "scoring_func": "sigmoid",
                        "norm_topk_prob": true, "n_group": 1, "topk_group": 1,
                        "max_position_embeddings": 256, "layernorm_epsilon": 1e-6,
                        "rope_theta": 10000, "swa_rope_theta": 10000,
                        "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2,
                        "head_dim": 8, "v_head_dim": 8, "swa_head_dim": 8, "swa_v_head_dim": 8,
                        "partial_rotary_factor": 0.5, "n_shared_experts": 1,
                        "n_routed_experts": 4, "routed_scaling_factor": 1.0,
                    ], seed: seed
                ) { MiMoV2FlashModel($0) }
            },
            ModelCase(
                "GLM4MoELite",
                expectedShapes: [
                    "model.layers.1.mlp.switch_mlp.up_proj.weight": [4, 16, 32]
                ],
                droppedKeys: ["model.layers.2.eh_proj.weight": [32, 64]],
                checkpoint: {
                    CheckpointLayout.splitExperts($0, stacked: "switch_mlp", perExpert: "experts")
                }
            ) { seed in
                try ModelCase.build(
                    GLM4MoELiteConfiguration.self,
                    [
                        "model_type": "glm4_moe_lite", "vocab_size": 64, "hidden_size": 32,
                        "intermediate_size": 48, "moe_intermediate_size": 16,
                        "num_hidden_layers": 2, "num_attention_heads": 4,
                        "num_key_value_heads": 4, "routed_scaling_factor": 1.0,
                        "kv_lora_rank": 16, "qk_rope_head_dim": 8, "qk_nope_head_dim": 8,
                        "v_head_dim": 8, "norm_topk_prob": true, "n_group": 1,
                        "topk_group": 1, "num_experts_per_tok": 2, "first_k_dense_replace": 1,
                        "max_position_embeddings": 256, "rms_norm_eps": 1e-6,
                        "rope_theta": 10000, "attention_bias": false,
                        "partial_rotary_factor": 1.0, "q_lora_rank": 16,
                        "n_routed_experts": 4, "n_shared_experts": 1,
                    ], seed: seed
                ) { GLM4MoELiteModel($0) }
            },
            ModelCase(
                "GraniteMoeHybrid",
                expectedShapes: [
                    "model.layers.1.self_attn.q_proj.weight": [32, 32],
                    "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight": [4, 48, 32],
                ],
                knownIssues: [
                    .causality: """
                    Without a cache, the attention mask is `.none` \
                    (GraniteMoeHybrid.swift:464-470), so the full pass is not causal.
                    """,
                    .cache: "The full pass without a cache is not causal (see causality).",
                ],

                checkpoint: { weights in
                    var result = weights
                    for layer in 0 ..< 2 {
                        let prefix = "model.layers.\(layer).block_sparse_moe"
                        let gate = result.removeValue(
                            forKey: "\(prefix).switch_mlp.gate_proj.weight")!
                        let up = result.removeValue(forKey: "\(prefix).switch_mlp.up_proj.weight")!
                        let down = result.removeValue(
                            forKey: "\(prefix).switch_mlp.down_proj.weight")!
                        result["\(prefix).input_linear.weight"] = concatenated([gate, up], axis: 1)
                        result["\(prefix).output_linear.weight"] = down
                    }
                    return result.reduce(into: [:]) { out, item in
                        out[item.key] =
                            item.key.hasSuffix("conv1d.weight")
                            ? item.value.swappedAxes(1, 2) : item.value
                    }
                }
            ) { seed in
                try ModelCase.build(
                    GraniteMoeHybridConfiguration.self,
                    [
                        "vocab_size": 64, "hidden_size": 32, "intermediate_size": 48,
                        "num_hidden_layers": 2, "max_position_embeddings": 256,
                        "num_attention_heads": 4, "num_key_value_heads": 2,
                        "attention_bias": false, "embedding_multiplier": 1.0,
                        "attention_multiplier": 0.125, "logits_scaling": 1.0,
                        "residual_multiplier": 1.0, "layer_types": ["mamba", "attention"],
                        "rms_norm_eps": 1e-6, "rope_theta": 10000, "num_local_experts": 4,
                        "num_experts_per_tok": 2, "shared_intermediate_size": 16,
                        "mamba_n_heads": 4, "mamba_d_head": 8, "mamba_proj_bias": false,
                        "mamba_d_state": 16, "mamba_d_conv": 4, "mamba_n_groups": 1,
                        "mamba_conv_bias": true, "tie_word_embeddings": false,
                    ], seed: seed
                ) { GraniteMoeHybridModel($0) }
            },
            ModelCase(
                "FalconH1",
                expectedShapes: [
                    "model.layers.0.mamba.conv1d.weight": [64, 4, 1],
                    "model.layers.1.self_attn.k_proj.weight": [16, 32],
                ]
            ) { seed in
                try ModelCase.build(
                    FalconH1Configuration.self,
                    [
                        "hidden_size": 32, "num_attention_heads": 4, "num_key_value_heads": 2,
                        "head_dim": 8, "intermediate_size": 48, "num_hidden_layers": 2,
                        "mamba_d_ssm": 32, "mamba_n_heads": 4, "mamba_d_head": 8,
                        "mamba_d_state": 16, "mamba_n_groups": 1, "mamba_d_conv": 4,
                        "vocab_size": 64,
                    ], seed: seed
                ) { FalconH1Model($0) }
            },
            ModelCase(
                "Jamba",
                expectedShapes: [
                    "model.layers.1.feed_forward.router.weight": [4, 32],
                    "model.layers.1.feed_forward.switch_mlp.gate_proj.weight": [4, 48, 32],
                ]
            ) { seed in
                try ModelCase.build(JambaConfiguration.self, Self.jamba, seed: seed) {
                    JambaModel($0)
                }
            },
            ModelCase(
                "BaichuanM1",
                expectedShapes: [
                    "model.layers.0.self_attn.W_pack.weight": [64, 32],
                    "model.layers.0.self_attn.conv_k": [1, 1, 2, 1, 2],
                    "lm_head.weight": [64, 32],
                ],
                // sanitize(weights:) divides the head rows by their length
                // plus 1e-7, which moves unit rows by about 1e-7.
                loadTolerance: 1e-5,
                knownIssues: [
                    .cache: """
                    The sliding-window layer gets a mask without the window \
                    (createAttentionMask(h:cache:) without windowSize, BaichuanM1.swift:223), \
                    so a prompt longer than the window attends to all earlier tokens, \
                    while decode with its RotatingKVCache attends to the window only.
                    """
                ]
            ) { seed in
                let model = try ModelCase.build(
                    BaichuanM1Configuration.self, baichuan, seed: seed
                ) { BaichuanM1Model($0) }
                // sanitize(weights:) scales each head row to length 1 (a
                // normalized head). Start from unit rows, so that the loaded
                // model matches.
                let head = SyntheticModel.flatParameters(model)["lm_head.weight"]!
                let unit = head / sqrt((head * head).sum(axis: -1, keepDims: true))
                model.update(parameters: ModuleParameters.unflattened(["lm_head.weight": unit]))
                eval(model)
                return model
            },
        ]

        /// A Hugging Face Jamba checkpoint stores each expert as
        /// `feed_forward.experts.N.{gate,up,down}_proj`.
        @Test func jambaLoaderStacksPerExpertWeights() throws {
            let reference = try ModelCase.build(JambaConfiguration.self, Self.jamba, seed: 5) {
                JambaModel($0)
            }
            let checkpoint = CheckpointLayout.splitExperts(
                SyntheticModel.flatParameters(reference), stacked: "switch_mlp",
                perExpert: "experts")
            #expect(checkpoint["model.layers.1.feed_forward.experts.3.down_proj.weight"] != nil)
            let loaded = try ModelCase.build(JambaConfiguration.self, Self.jamba, seed: 6) {
                JambaModel($0)
            }
            let rows = [SyntheticModel.tokens(count: 11, vocabularySize: 64, seed: 3)]
            // The load throws, so a thrown error is the known issue, as in
            // ModelCaseChecks.loading. Once it loads, the logits must match.
            try withKnownIssue(
                """
                JambaModel.sanitize(weights:) stacks experts only under \
                `block_sparse_moe.experts.N.w1` and writes `block_sparse_moe.switch_mlp` \
                (Jamba.swift:521-565), but the module path is `feed_forward.switch_mlp`, \
                so a per-expert checkpoint does not load.
                """
            ) {
                try SyntheticModel.load(checkpoint, into: loaded)
                #expect(
                    SyntheticModel.maxAbsDifference(
                        ForwardPassChecks.logits(reference, rows),
                        ForwardPassChecks.logits(loaded, rows)) == 0, "loaded logits")
            } matching: {
                $0.error != nil || $0.isFailedExpectation(["loaded logits"])
            }
        }

        /// With tied embeddings the model must still return logits over the
        /// vocabulary.
        @Test func baichuanWithTiedEmbeddingsReturnsLogits() throws {
            let model = try ModelCase.build(
                BaichuanM1Configuration.self,
                Self.baichuan.merging(["tie_word_embeddings": true]) { $1 }, seed: 1
            ) { BaichuanM1Model($0) }
            let logits = ForwardPassChecks.logits(model, [[1, 2, 3]])
            withKnownIssue(
                """
                BaichuanM1Model has no head when the embeddings are tied, and \
                callAsFunction then returns the hidden states (BaichuanM1.swift:254-261). \
                The reference mlx-lm baichuan_m1.py has no tied path either: it builds \
                lm_head only when the embeddings are not tied and always calls it.
                """
            ) {
                #expect(logits.shape == [1, 3, 64], "logits shape")
            } matching: {
                $0.isFailedExpectation(["logits shape"])
            }
        }

        /// `sanitize(weights:)` scales each row of an unquantized head to
        /// length 1.
        @Test func baichuanLoaderNormalizesTheHead() throws {
            let model = try ModelCase.build(BaichuanM1Configuration.self, Self.baichuan, seed: 1) {
                BaichuanM1Model($0)
            }
            let head = MLXRandom.normal([64, 32], key: MLXRandom.key(4)) * 3
            let sanitized = model.sanitize(weights: ["lm_head.weight": head])
            let norms = sqrt((sanitized["lm_head.weight"]! ** 2).sum(axis: -1))
            #expect(SyntheticModel.maxAbsDifference(norms, MLXArray.ones([64])) <= 1e-5)
        }

        /// Dynamic NTK scaling raises the RoPE base when the sequence is
        /// longer than `max_position_embeddings`. The layer gets queries and
        /// keys as `[B, heads, L, D]`.
        @Test func internLM2DynamicRopeScalesByTheSequenceLength() {
            let rope = Internlm2DynamicNTKScalingRoPE(
                dims: 8, maxPositionEmbeddings: 8, base: 10000, scale: 2)
            let x = MLXRandom.normal([1, 4, 12, 8], key: MLXRandom.key(2))
            // seq_len 12 > 8: base * (2 * 12 / 8 - 1) ^ (8 / 6)
            let base = 10000 * pow(Float(2 * 12) / 8 - 1, Float(8) / 6)
            let expected = MLXFast.RoPE(
                x, dimensions: 8, traditional: false, base: base, scale: 2, offset: 0)
            let output = rope(x, offset: 0)
            withKnownIssue(
                """
                Internlm2DynamicNTKScalingRoPE takes the sequence length from x.dim(1) \
                (Internlm2.swift:41), which is the head count for `[B, heads, L, D]` \
                input, so a long prompt never gets the scaled base.
                """
            ) {
                #expect(SyntheticModel.maxAbsDifference(output, expected) <= 1e-5, "scaled base")
            } matching: {
                $0.isFailedExpectation(["scaled base"])
            }
        }

        @Test(arguments: cases) func logitsHaveTheExpectedShapeAndAreFinite(_ c: ModelCase) throws {
            try ModelCaseChecks.shapeAndFinite(c)
        }

        @Test(arguments: cases) func sameSeedGivesTheSameLogits(_ c: ModelCase) throws {
            try ModelCaseChecks.determinism(c)
        }

        @Test(arguments: cases) func cachedDecodeMatchesTheFullForwardPass(_ c: ModelCase) throws {
            try ModelCaseChecks.cacheConsistency(c)
        }

        @Test(arguments: cases) func eachRowOfABatchMatchesTheRowAlone(_ c: ModelCase) throws {
            try ModelCaseChecks.batchInvariance(c)
        }

        @Test(arguments: cases) func aLaterTokenDoesNotChangeEarlierLogits(_ c: ModelCase) throws {
            try ModelCaseChecks.causality(c)
        }

        @Test(arguments: cases) func loaderAcceptsACheckpointAndRejectsAWrongShape(_ c: ModelCase)
            throws
        {
            try ModelCaseChecks.loading(c)
        }
    }
}
