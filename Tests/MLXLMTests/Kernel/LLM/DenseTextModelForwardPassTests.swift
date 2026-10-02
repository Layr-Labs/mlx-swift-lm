import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {

    /// Forward-pass tests of dense text models with tiny random models:
    /// hidden size 32 (64 for Lille), 2 layers, 4 query heads over 2 key
    /// heads, vocabulary 64.
    ///
    /// The table has one case for each configuration variant that changes
    /// the code path: tied or untied head, RoPE scaling, biases, logit
    /// scale, and the keys that `sanitize(weights:)` drops.
    ///
    /// Tolerance 1e-4 for the float32 comparisons: the cached path and the
    /// full path differ in the order of the attention sums. The differences
    /// are near 1e-6 for logits of size 1 to 5.
    ///
    /// A check that fails because of a known production defect runs inside
    /// `withKnownIssue` (see `ModelCase.knownIssues`).
    @Suite
    struct DenseTextModelForwardPassTests {

        static var miniCPM: [String: Any] {
            [
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
                "vocab_size": 64, "max_position_embeddings": 256,
            ]
        }

        static var olmo2: [String: Any] {
            [
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
                "vocab_size": 64, "max_position_embeddings": 256,
            ]
        }

        static var glm4: [String: Any] {
            [
                "model_type": "glm4", "hidden_size": 32, "num_hidden_layers": 2,
                "intermediate_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2,
                "attention_bias": true, "head_dim": 8, "rms_norm_eps": 1e-6, "vocab_size": 64,
                "partial_rotary_factor": 0.5,
            ]
        }

        static var starcoder2: [String: Any] {
            [
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                "num_attention_heads": 4, "num_key_value_heads": 2, "norm_epsilon": 1e-5,
                "vocab_size": 64, "rope_theta": 10000,
            ]
        }

        static var ernie: [String: Any] {
            [
                "hidden_size": 32, "intermediate_size": 48, "max_position_embeddings": 256,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                "num_hidden_layers": 2, "rms_norm_eps": 1e-6, "vocab_size": 64,
                "rope_theta": 10000, "use_bias": false, "tie_word_embeddings": true,
            ]
        }

        static var cohere: [String: Any] {
            [
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                "num_attention_heads": 4, "num_key_value_heads": 2, "layer_norm_eps": 1e-5,
                "vocab_size": 64, "rope_theta": 10000, "logit_scale": 0.5,
            ]
        }

        static var mimo: [String: Any] {
            [
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
                "vocab_size": 64, "num_nextn_predict_layers": 1,
            ]
        }

        /// The text configuration of the `gemma4` wrapper: 1 sliding layer
        /// and 1 full layer, no shared K/V layers and no per-layer inputs.
        /// The window of 16 is longer than the 11 tokens of the cache test.
        /// The nested `vocab_size` of 128 must be replaced by the root value.
        static var gemma4: [String: Any] {
            [
                "model_type": "gemma4", "vocab_size": 64,
                "text_config": [
                    "model_type": "gemma4_text", "hidden_size": 32, "num_hidden_layers": 2,
                    "intermediate_size": 48, "num_attention_heads": 4, "head_dim": 8,
                    "global_head_dim": 8, "num_key_value_heads": 2, "num_kv_shared_layers": 0,
                    "sliding_window": 16, "sliding_window_pattern": 2,
                    "final_logit_softcapping": 30.0, "tie_word_embeddings": true,
                    "vocab_size": 128, "vocab_size_per_layer_input": 64, "rms_norm_eps": 1e-6,
                    "hidden_size_per_layer_input": 0,
                ] as [String: Any],
            ]
        }

        static func merged(_ base: [String: Any], _ overrides: [String: Any]) -> [String: Any] {
            base.merging(overrides) { $1 }
        }

        static let cases: [ModelCase] = [
            ModelCase(
                "MiniCPM untied, linear RoPE, scaled embeddings",
                expectedShapes: [
                    "model.embed_tokens.weight": [64, 32],
                    "model.layers.0.self_attn.k_proj.weight": [16, 32],
                    "model.layers.1.mlp.down_proj.weight": [32, 48],
                    "lm_head.weight": [64, 32],
                ]
            ) { seed in
                try ModelCase.build(
                    MiniCPMConfiguration.self,
                    merged(
                        miniCPM,
                        [
                            "scale_emb": 12, "scale_depth": 1.4, "dim_model_base": 16,
                            "rope_scaling": ["rope_type": "linear", "factor": 2.0],
                        ]), seed: seed
                ) { MiniCPMModel($0) }
            },
            ModelCase(
                "MiniCPM tied",
                expectedShapes: ["model.layers.0.post_attention_layernorm.weight": [32]],
                droppedKeys: ["lm_head.weight": [64, 32]]
            ) { seed in
                try ModelCase.build(
                    MiniCPMConfiguration.self, merged(miniCPM, ["tie_word_embeddings": true]),
                    seed: seed
                ) { MiniCPMModel($0) }
            },
            ModelCase(
                "OLMo 2 tied",
                expectedShapes: [
                    "model.layers.0.self_attn.q_norm.weight": [32],
                    "model.layers.0.self_attn.k_norm.weight": [16],
                    "model.layers.1.post_feedforward_layernorm.weight": [32],
                ],
                droppedKeys: ["model.layers.0.self_attn.rotary_emb.inv_freq": [4]]
            ) { seed in
                try ModelCase.build(Olmo2Configuration.self, olmo2, seed: seed) {
                    Olmo2Model($0)
                }
            },
            ModelCase(
                "OLMo 2 untied, biases, head_dim 16, llama3 RoPE",
                expectedShapes: [
                    "model.layers.0.self_attn.q_proj.weight": [64, 32],
                    "model.layers.0.self_attn.q_proj.bias": [64],
                    "model.layers.0.self_attn.q_norm.weight": [64],
                    "model.layers.0.self_attn.k_norm.weight": [32],
                    "model.layers.0.self_attn.o_proj.weight": [32, 64],
                    "model.layers.1.mlp.gate_proj.bias": [48],
                    "lm_head.weight": [64, 32],
                ]
            ) { seed in
                try ModelCase.build(
                    Olmo2Configuration.self,
                    merged(
                        olmo2,
                        [
                            "tie_word_embeddings": false, "attention_bias": true,
                            "mlp_bias": true, "head_dim": 16,
                            "rope_scaling": [
                                "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                "high_freq_factor": 4.0, "original_max_position_embeddings": 16,
                            ],
                        ]), seed: seed
                ) { Olmo2Model($0) }
            },
            ModelCase(
                "GLM-4",
                expectedShapes: [
                    "model.layers.0.self_attn.q_proj.bias": [32],
                    "model.layers.0.self_attn.k_proj.weight": [16, 32],
                    "model.layers.0.mlp.gate_up_proj.weight": [96, 32],
                    "model.layers.1.post_self_attn_layernorm.weight": [32],
                    "model.layers.1.post_mlp_layernorm.weight": [32],
                    "lm_head.weight": [64, 32],
                ]
            ) { seed in
                try ModelCase.build(GLM4Configuration.self, glm4, seed: seed) { GLM4Model($0) }
            },
            ModelCase(
                "GLM-4 tied",
                expectedShapes: ["lm_head.weight": [64, 32]]
            ) { seed in
                try ModelCase.build(
                    GLM4Configuration.self,
                    merged(glm4, ["tie_word_embeddings": true, "attention_bias": false]),
                    seed: seed
                ) { GLM4Model($0) }
            },
            ModelCase(
                "Gemma",
                expectedShapes: [
                    "model.embed_tokens.weight": [64, 32],
                    "model.layers.0.self_attn.q_proj.weight": [32, 32],
                    "model.layers.0.self_attn.v_proj.weight": [16, 32],
                    "model.layers.1.mlp.gate_proj.weight": [48, 32],
                    "model.norm.weight": [32],
                ]
            ) { seed in
                try ModelCase.build(
                    GemmaConfiguration.self,
                    [
                        "model_type": "gemma", "hidden_size": 32, "num_hidden_layers": 2,
                        "intermediate_size": 48, "num_attention_heads": 4, "head_dim": 8,
                        "rms_norm_eps": 1e-6, "vocab_size": 64, "num_key_value_heads": 2,
                        "rope_theta": 10000, "rope_traditional": true,
                    ], seed: seed
                ) { GemmaModel($0) }
            },
            ModelCase(
                "Starcoder 2 tied",
                expectedShapes: [
                    "model.layers.0.self_attn.o_proj.bias": [32],
                    "model.layers.0.input_layernorm.bias": [32],
                    "model.layers.1.mlp.c_fc.weight": [48, 32],
                    "model.layers.1.mlp.c_proj.bias": [32],
                ]
            ) { seed in
                try ModelCase.build(Starcoder2Configuration.self, starcoder2, seed: seed) {
                    Starcoder2Model($0)
                }
            },
            ModelCase(
                "Starcoder 2 untied",
                expectedShapes: ["lm_head.weight": [64, 32]]
            ) { seed in
                try ModelCase.build(
                    Starcoder2Configuration.self,
                    merged(starcoder2, ["tie_word_embeddings": false]), seed: seed
                ) { Starcoder2Model($0) }
            },
            ModelCase(
                "Phi",
                expectedShapes: [
                    "model.layers.0.self_attn.dense.weight": [32, 32],
                    "model.layers.0.self_attn.k_proj.bias": [16],
                    "model.layers.1.mlp.fc1.weight": [48, 32],
                    "model.final_layernorm.bias": [32],
                    "lm_head.bias": [64],
                ]
            ) { seed in
                try ModelCase.build(
                    PhiConfiguration.self,
                    [
                        "max_position_embeddings": 256, "vocab_size": 64, "hidden_size": 32,
                        "num_attention_heads": 4, "num_hidden_layers": 2,
                        "num_key_value_heads": 2, "partial_rotary_factor": 0.5,
                        "intermediate_size": 48, "layer_norm_eps": 1e-5,
                    ], seed: seed
                ) { PhiModel($0) }
            },
            ModelCase(
                "ERNIE 4.5 tied, head_dim 16",
                expectedShapes: [
                    "model.layers.0.self_attn.q_proj.weight": [64, 32],
                    "model.layers.0.self_attn.o_proj.weight": [32, 64],
                    "model.layers.1.mlp.up_proj.weight": [48, 32],
                ]
            ) { seed in
                try ModelCase.build(
                    Ernie45Configuration.self, merged(ernie, ["head_dim": 16]), seed: seed
                ) { Ernie45Model($0) }
            },
            ModelCase(
                "ERNIE 4.5 untied, biases",
                expectedShapes: [
                    "model.layers.0.self_attn.q_proj.bias": [32],
                    "model.layers.0.self_attn.o_proj.bias": [32],
                    "model.layers.1.mlp.down_proj.bias": [32],
                    "lm_head.weight": [64, 32],
                ]
            ) { seed in
                try ModelCase.build(
                    Ernie45Configuration.self,
                    merged(ernie, ["use_bias": true, "tie_word_embeddings": false]), seed: seed
                ) { Ernie45Model($0) }
            },
            ModelCase(
                "Cohere",
                expectedShapes: [
                    "model.embed_tokens.weight": [64, 32],
                    "model.layers.0.input_layernorm.weight": [32],
                    "model.layers.1.self_attn.v_proj.weight": [16, 32],
                    "model.layers.1.mlp.down_proj.weight": [32, 48],
                ]
            ) { seed in
                try ModelCase.build(CohereConfiguration.self, cohere, seed: seed) {
                    CohereModel($0)
                }
            },
            ModelCase(
                "Lille 130m",
                expectedShapes: [
                    "transformer.tok_embeddings.weight": [64, 64],
                    // (4 query heads + 2 x 2 key-value heads) x head_dim 16.
                    "transformer.layers.0.attention.qkv_proj.weight": [128, 64],
                    "transformer.layers.0.attention.norm.weight": [64],
                    // 256 x round(8 x 64 / 3 / 256) = 256.
                    "transformer.layers.1.feed_forward.gate_proj.weight": [256, 64],
                    "transformer.norm.weight": [64],
                ],
                droppedKeys: ["transformer.layers.0.attention.rotary_emb.inv_freq": [8]]
            ) { seed in
                try ModelCase.build(
                    Lille130mConfiguration.self,
                    [
                        "model_type": "lille-130m", "block_size": 256, "layer_norm_eps": 1e-5,
                        "n_embd": 64, "n_head": 4, "n_kv_heads": 2, "n_layer": 2,
                        "rope_theta": 10000, "vocab_size": 64,
                    ], seed: seed
                ) { Lille130mModel($0) }
            },
            ModelCase(
                "MiMo tied",
                expectedShapes: [
                    "model.layers.0.self_attn.q_proj.bias": [32],
                    "model.layers.0.self_attn.k_proj.bias": [16],
                    "model.layers.0.self_attn.o_proj.weight": [32, 32],
                    "model.layers.1.mlp.up_proj.weight": [48, 32],
                ],
                droppedKeys: [
                    "lm_head.weight": [64, 32],
                    "model.mtp_layers.0.input_layernorm.weight": [32],
                    "model.layers.0.self_attn.rotary_emb.inv_freq": [4],
                ]
            ) { seed in
                try ModelCase.build(
                    MiMoConfiguration.self, merged(mimo, ["tie_word_embeddings": true]),
                    seed: seed
                ) { MiMoModel($0) }
            },
            ModelCase(
                "MiMo untied, traditional linear RoPE",
                expectedShapes: ["lm_head.weight": [64, 32]],
                droppedKeys: ["model.mtp_layers.0.input_layernorm.weight": [32]]
            ) { seed in
                try ModelCase.build(
                    MiMoConfiguration.self,
                    merged(
                        mimo,
                        [
                            "rope_traditional": true,
                            "rope_scaling": ["type": "linear", "factor": 2.0],
                        ]), seed: seed
                ) { MiMoModel($0) }
            },
            ModelCase(
                "Gemma 4 wrapper",
                expectedShapes: [
                    "language_model.model.embed_tokens.weight": [64, 32],
                    "language_model.model.layers.0.self_attn.q_proj.weight": [32, 32],
                    "language_model.model.layers.1.self_attn.k_proj.weight": [16, 32],
                ],
                droppedKeys: [
                    "model.vision_tower.encoder.weight": [4],
                    "model.multi_modal_projector.weight": [4],
                    "model.audio_tower.weight": [4],
                    "model.embed_audio.weight": [4],
                    "model.embed_vision.weight": [4],
                    "vision_tower.patch.weight": [4],
                    "model.language_model.layers.0.self_attn.rotary_emb.inv_freq": [4],
                    "model.language_model.layers.0.self_attn.k_proj.input_max": [1],
                ],
                checkpoint: { weights in
                    // A Hugging Face checkpoint stores `model.language_model.X`
                    // for the module path `language_model.model.X`. One key
                    // keeps the module path, which sanitize(weights:) also
                    // accepts.
                    var result: [String: MLXArray] = [:]
                    for (key, value) in weights {
                        if key == "language_model.model.norm.weight" {
                            result[key] = value
                            continue
                        }
                        let name = key.replacingOccurrences(
                            of: "language_model.model.", with: "language_model.",
                            options: .anchored)
                        result["model." + name] = value
                    }
                    return result
                }
            ) { seed in
                let model = try ModelCase.build(
                    MLXLLM.Gemma4Configuration.self, gemma4, seed: seed
                ) { Gemma4Model($0) }
                // `layer_scalar` is float16. loadWeights converts float16
                // parameters to bfloat16, so give it values that both types
                // hold exactly. The loaded logits then match the reference.
                let scalars = SyntheticModel.flatParameters(model)
                    .filter { $0.key.hasSuffix(".layer_scalar") }
                    .map { ($0.key, $0.value.asType(.bfloat16).asType(.float16)) }
                model.update(parameters: ModuleParameters.unflattened(scalars))
                eval(model)
                return model
            },
        ]

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

        /// The logits are the tied head output times `logit_scale`. Two
        /// models with the same seed and scales 0.5 and 0.25 must give
        /// logits in the ratio 2. Tolerance 1e-6: a scale by a power of 2
        /// is exact in float32.
        @Test func cohereMultipliesTheLogitsByTheLogitScale() throws {
            let half = try ModelCase.build(CohereConfiguration.self, Self.cohere, seed: 1) {
                CohereModel($0)
            }
            let quarter = try ModelCase.build(
                CohereConfiguration.self, Self.merged(Self.cohere, ["logit_scale": 0.25]),
                seed: 1
            ) { CohereModel($0) }
            let rows = [SyntheticModel.tokens(count: 6, vocabularySize: 64, seed: 4)]
            let a = ForwardPassChecks.logits(half, rows)
            let b = ForwardPassChecks.logits(quarter, rows)
            #expect(SyntheticModel.maxAbsDifference(a, 2 * b) <= 1e-6)
            #expect(SyntheticModel.maxAbs(a) > 1e-3)
        }

        /// Cohere always uses the traditional RoPE layout. The reference
        /// mlx-lm `cohere.py:67` (ml-explore/mlx-lm@53b9af37) hard-codes
        /// `nn.RoPE(head_dim, traditional=True, ...)` and its `ModelArgs`
        /// has no `rope_traditional` field. `CohereConfiguration` does not
        /// decode the key either, so both values of the key must give the
        /// same logits. Tolerance 1e-6: the two models run the same
        /// computation with the same weights.
        @Test func cohereIgnoresRopeTraditional() throws {
            let traditional = try ModelCase.build(
                CohereConfiguration.self, Self.merged(Self.cohere, ["rope_traditional": true]),
                seed: 1
            ) { CohereModel($0) }
            let split = try ModelCase.build(
                CohereConfiguration.self, Self.merged(Self.cohere, ["rope_traditional": false]),
                seed: 1
            ) { CohereModel($0) }
            let rows = [SyntheticModel.tokens(count: 8, vocabularySize: 64, seed: 4)]
            let a = ForwardPassChecks.logits(traditional, rows)
            let b = ForwardPassChecks.logits(split, rows)
            #expect(
                SyntheticModel.maxAbsDifference(a, b) <= 1e-6,
                "rope_traditional must not change the Cohere logits")
            #expect(SyntheticModel.maxAbs(a) > 1e-3)
        }

        /// `GemmaRMSNorm` scales the normalized input by `1 + weight`.
        /// Tolerance 1e-5 for the float32 norm.
        @Test func gemmaRMSNormScalesByOnePlusTheWeight() {
            let norm = GemmaRMSNorm(dimensions: 8, eps: 1e-5)
            let x = MLXRandom.normal([2, 3, 8], key: MLXRandom.key(5))
            let expected = x * rsqrt((x * x).mean(axis: -1, keepDims: true) + 1e-5) * 2
            #expect(SyntheticModel.maxAbsDifference(norm(x), expected) <= 1e-5)
        }

        /// A tied Starcoder 2 model has no head and uses the embeddings.
        @Test func starcoder2TiedModelHasNoHead() throws {
            let model = try ModelCase.build(
                Starcoder2Configuration.self, Self.starcoder2, seed: 1
            ) { Starcoder2Model($0) }
            let parameters = SyntheticModel.flatParameters(model)
            #expect(parameters["lm_head.weight"] == nil)
            #expect(parameters["model.embed_tokens.weight"]?.shape == [64, 32])
        }

        /// The `gemma4` wrapper takes the vocabulary size from the root of
        /// the configuration, forwards the text model, and gives its caches.
        /// The logits must be equal: both calls run the same operations.
        @Test func gemma4WrapperForwardsTheTextModel() throws {
            let configuration = try SyntheticModel.configuration(
                MLXLLM.Gemma4Configuration.self, Self.gemma4)
            #expect(configuration.textConfig.vocabSize == 64)
            #expect(configuration.vocabSize == 64)

            let model = Gemma4Model(configuration)
            SyntheticModel.randomize(model, seed: 2)
            #expect(model.vocabularySize == 64)
            #expect(model.kvHeads == model.textModel.kvHeads)
            #expect(model.kvHeads.count == 2)
            #expect(model.loraLayers.count == 2)
            #expect(
                model.cbv2SupportsHistoricalAttentionCheckpoint
                    == model.textModel.cbv2SupportsHistoricalAttentionCheckpoint)

            let tokens = SyntheticModel.batch([[1, 5, 9, 2]])
            let cache = model.newCache(parameters: nil)
            #expect(cache.count == 2)
            let wrapped = model(tokens, cache: cache)
            let direct = model.textModel(tokens, cache: model.textModel.newCache(parameters: nil))
            eval(wrapped, direct)
            #expect(wrapped.shape == [1, 4, 64])
            #expect(SyntheticModel.maxAbsDifference(wrapped, direct) == 0)
        }

        /// Without `text_config`, the whole configuration is the text
        /// configuration.
        @Test func gemma4WrapperDecodesAFlatConfiguration() throws {
            var flat = Self.gemma4["text_config"] as! [String: Any]
            flat["vocab_size"] = 64
            let configuration = try SyntheticModel.configuration(
                MLXLLM.Gemma4Configuration.self, flat)
            #expect(configuration.modelType == "gemma4_text")
            #expect(configuration.textConfig.vocabSize == 64)
            #expect(configuration.textConfig.numHiddenLayers == 2)
            let model = Gemma4Model(configuration)
            SyntheticModel.randomize(model, seed: 2)
            let logits = ForwardPassChecks.logits(model, [[3, 1, 4]])
            #expect(logits.shape == [1, 3, 64])
            #expect(isFinite(logits).all().item(Bool.self))
        }
    }
}
