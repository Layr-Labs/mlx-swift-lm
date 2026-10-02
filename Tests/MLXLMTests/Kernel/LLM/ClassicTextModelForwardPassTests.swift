import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {

    /// Forward-pass tests of SmolLM3, Exaone 4, OLMoE, OLMo 3, PhiMoE, Phi-3,
    /// NanoChat, Granite, OpenELM and Gemma 2 with tiny random models: hidden
    /// size 32 (48 for OpenELM), 2 layers, 4 query heads over 2 key heads,
    /// vocabulary 64, and 4 experts with top-2 routing for the MoE models.
    ///
    /// Each model has a case for each configuration that changes the code
    /// path: tied or untied embeddings, layers without RoPE, sliding-window
    /// layers, the RoPE scaling types, the expert routing and the logit soft
    /// caps.
    ///
    /// Tolerance 1e-4 for the float32 comparisons: the cached path and the
    /// full path differ in the order of the attention sums. The differences
    /// are near 1e-6 for logits of size 1 to 5.
    ///
    /// A check that fails because of a known production defect runs inside
    /// `withKnownIssue` (see `ModelCase.knownIssues`).
    @Suite
    struct ClassicTextModelForwardPassTests {

        static let tolerance: Float = 1e-4

        static var smolLM3: [String: Any] {
            [
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
                "vocab_size": 64,
            ]
        }

        static var exaone4: [String: Any] {
            [
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                "rms_norm_eps": 1e-6, "vocab_size": 64, "max_position_embeddings": 256,
                "rope_theta": 10000, "tie_word_embeddings": true,
            ]
        }

        static var olmoE: [String: Any] {
            [
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 16,
                "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
                "vocab_size": 64, "num_experts": 4, "num_experts_per_tok": 2,
            ]
        }

        static var olmo3: [String: Any] {
            [
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
                "vocab_size": 64, "max_position_embeddings": 256, "sliding_window": 4,
                "layer_types": ["sliding_attention", "full_attention"],
            ]
        }

        static var phiMoE: [String: Any] {
            [
                "model_type": "phimoe", "vocab_size": 64, "hidden_size": 32,
                "intermediate_size": 16, "num_hidden_layers": 2, "num_attention_heads": 4,
                "num_key_value_heads": 2, "max_position_embeddings": 256,
                "original_max_position_embeddings": 64, "rms_norm_eps": 1e-5,
                "num_local_experts": 4, "num_experts_per_tok": 2, "rope_theta": 10000,
            ]
        }

        static var phi3: [String: Any] {
            [
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-5,
                "vocab_size": 64, "max_position_embeddings": 256,
                "original_max_position_embeddings": 64,
            ]
        }

        static var nanoChat: [String: Any] {
            [
                "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
                "num_key_value_heads": 2, "vocab_size": 64, "max_position_embeddings": 256,
                "intermediate_size": 48,
            ]
        }

        static var granite: [String: Any] {
            [
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
                "vocab_size": 64, "logits_scaling": 4, "attention_multiplier": 0.125,
                "embedding_multiplier": 2, "residual_multiplier": 0.5,
                "max_position_embeddings": 256, "attention_bias": false, "mlp_bias": false,
                "tie_word_embeddings": true,
            ]
        }

        /// With head size 8 and 4 GQA groups, layer 0 has 4 query heads over
        /// 1 key head and layer 1 has 8 query heads over 2 key heads. The
        /// FFN widths are 24 and 192.
        static var openELM: [String: Any] {
            [
                "model_type": "openelm", "head_dim": 8, "num_transformer_layers": 2,
                "model_dim": 48, "vocab_size": 64, "ffn_dim_divisor": 8,
            ]
        }

        static var gemma2: [String: Any] {
            [
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                "num_attention_heads": 4, "head_dim": 8, "rms_norm_eps": 1e-6,
                "vocab_size": 64, "num_key_value_heads": 2, "attn_logit_softcapping": 50,
                "final_logit_softcapping": 30, "query_pre_attn_scalar": 8,
            ]
        }

        /// Builds a model from `base` plus `overrides` and randomizes it.
        static func build<C: Decodable, M: Module>(
            _ type: C.Type, _ base: [String: Any], _ overrides: [String: Any] = [:],
            seed: UInt64, _ initializer: (C) -> M
        ) throws -> M {
            let model = initializer(
                try SyntheticModel.configuration(type, base, overrides: overrides))
            SyntheticModel.randomize(model, seed: seed)
            return model
        }

        /// NanoChat keeps its RoPE frequencies in an array property, so they
        /// are parameters and `randomize` changes them. Put back the
        /// frequencies of a new model, so that the RoPE stays the real one.
        static func makeNanoChat(_ overrides: [String: Any] = [:], seed: UInt64) throws
            -> NanoChatModel
        {
            let configuration = try SyntheticModel.configuration(
                NanoChatConfiguration.self, nanoChat, overrides: overrides)
            let model = NanoChatModel(configuration)
            SyntheticModel.randomize(model, seed: seed)
            let frequencies = SyntheticModel.flatParameters(NanoChatModel(configuration))
                .filter { $0.key.hasSuffix("rope.freqs") }
            model.update(parameters: ModuleParameters.unflattened(frequencies))
            eval(model)
            return model
        }

        static func rotaryFrequencies(_ prefix: String, layers: Int = 2, count: Int = 4)
            -> [String: [Int]]
        {
            Dictionary(
                uniqueKeysWithValues: (0 ..< layers).map {
                    ("\(prefix).layers.\($0).self_attn.rotary_emb.inv_freq", [count])
                })
        }

        static let cases: [ModelCase] = [
            ModelCase(
                "SmolLM3 (untied, layer 1 without RoPE, attention bias)",
                expectedShapes: [
                    "model.embed_tokens.weight": [64, 32],
                    "model.layers.0.self_attn.q_proj.bias": [32],
                    "model.layers.1.self_attn.k_proj.weight": [16, 32],
                    "model.layers.1.mlp.gate_proj.weight": [48, 32],
                    "lm_head.weight": [64, 32],
                ],
                droppedKeys: rotaryFrequencies("model")
            ) { seed in
                try Self.build(
                    SmolLM3Configuration.self, Self.smolLM3,
                    [
                        "tie_word_embeddings": false, "no_rope_layers": [1, 0],
                        "attention_bias": true,
                    ], seed: seed
                ) { SmolLM3Model($0) }
            },
            ModelCase(
                "SmolLM3 (tied, RoPE interval 2, head size 16)",
                expectedShapes: [
                    "model.layers.0.self_attn.q_proj.weight": [64, 32],
                    "model.layers.0.self_attn.v_proj.weight": [32, 32],
                    "model.layers.1.self_attn.o_proj.weight": [32, 64],
                ],
                // sanitize(weights:) drops the head of a tied checkpoint.
                droppedKeys: rotaryFrequencies("model", count: 8).merging([
                    "lm_head.weight": [64, 32]
                ]) { $1 }
            ) { seed in
                try Self.build(
                    SmolLM3Configuration.self, Self.smolLM3,
                    ["head_dim": 16, "no_rope_layer_interval": 2], seed: seed
                ) { SmolLM3Model($0) }
            },
            ModelCase(
                "Exaone4 (global layers, tied, llama3 RoPE)",
                expectedShapes: [
                    "model.layers.0.self_attn.q_norm.weight": [8],
                    "model.layers.0.self_attn.k_proj.weight": [16, 32],
                    "model.layers.0.mlp.gate_proj.weight": [48, 32],
                    "model.layers.1.post_feedforward_layernorm.weight": [32],
                ],
                droppedKeys: ["lm_head.weight": [64, 32]]
            ) { seed in
                try Self.build(
                    Exaone4Configuration.self, Self.exaone4,
                    [
                        "rope_scaling": [
                            "rope_type": "llama3", "factor": 2, "low_freq_factor": 1,
                            "high_freq_factor": 4, "original_max_position_embeddings": 64,
                        ]
                    ], seed: seed
                ) { Exaone4Model($0) }
            },
            // The window (16) is longer than the test sequence (11), so the
            // missing window mask of the full pass does not matter here. See
            // exaone4SlidingLayersIgnoreTheWindowWithoutACache for a shorter
            // window.
            ModelCase(
                "Exaone4 (local and global layers, untied)",
                expectedShapes: [
                    "model.layers.1.self_attn.k_norm.weight": [8],
                    "lm_head.weight": [64, 32],
                ]
            ) { seed in
                try Self.build(
                    Exaone4Configuration.self, Self.exaone4,
                    [
                        "tie_word_embeddings": false, "sliding_window": 16,
                        "sliding_window_pattern": "LG",
                    ], seed: seed
                ) { Exaone4Model($0) }
            },
            ModelCase(
                "OlmoE (untied, normalized top-k)",
                expectedShapes: [
                    "model.layers.0.self_attn.q_norm.weight": [32],
                    "model.layers.0.self_attn.k_norm.weight": [16],
                    "model.layers.0.mlp.gate.weight": [4, 32],
                    "model.layers.1.mlp.switch_mlp.up_proj.weight": [4, 16, 32],
                    "lm_head.weight": [64, 32],
                ],
                checkpoint: {
                    CheckpointLayout.splitExperts($0, stacked: "switch_mlp", perExpert: "experts")
                }
            ) { seed in
                try Self.build(
                    OlmoEConfiguration.self, Self.olmoE,
                    ["tie_word_embeddings": false, "norm_topk_prob": true], seed: seed
                ) { OlmoEModel($0) }
            },
            ModelCase(
                "OlmoE (tied, top-k not normalized, attention bias)",
                expectedShapes: [
                    "model.layers.0.self_attn.q_proj.bias": [32],
                    "model.layers.1.mlp.switch_mlp.down_proj.weight": [4, 32, 16],
                ]
            ) { seed in
                try Self.build(
                    OlmoEConfiguration.self, Self.olmoE, ["attention_bias": true], seed: seed
                ) { OlmoEModel($0) }
            },
            ModelCase(
                "Olmo3 (sliding and full layers, untied, linear RoPE)",
                expectedShapes: [
                    "model.layers.0.self_attn.q_norm.weight": [32],
                    "model.layers.0.self_attn.k_norm.weight": [16],
                    "model.layers.1.post_feedforward_layernorm.weight": [32],
                    "lm_head.weight": [64, 32],
                ],
                droppedKeys: rotaryFrequencies("model"),
                knownIssues: [
                    .cache: """
                    Olmo3Model.newCache(parameters: GenerateParameters) takes a non-optional \
                    argument (Olmo3.swift:225), so it is not the LanguageModel requirement \
                    newCache(parameters: GenerateParameters?). Generation gets the default \
                    KVCacheSimple for the sliding layers, and its mask drops the window for a \
                    chunk that is not longer than the window and for decode.
                    """
                ]
            ) { seed in
                try Self.build(
                    Olmo3Configuration.self, Self.olmo3,
                    ["rope_scaling": ["rope_type": "linear", "factor": 2]], seed: seed
                ) { Olmo3Model($0) }
            },
            // Without `layer_types`, every 4th layer is a full layer, so both
            // layers here are sliding layers. The window (16) is longer than
            // the test sequence.
            ModelCase(
                "Olmo3 (default layer types, tied)",
                expectedShapes: ["model.layers.1.self_attn.o_proj.weight": [32, 32]]
            ) { seed in
                try Self.build(
                    Olmo3Configuration.self,
                    Self.olmo3.filter { $0.key != "layer_types" },
                    ["sliding_window": 16, "tie_word_embeddings": true], seed: seed
                ) { Olmo3Model($0) }
            },
            ModelCase(
                "PhiMoE (longrope with mscale)",
                expectedShapes: [
                    "model.layers.0.self_attn.q_proj.bias": [32],
                    "model.layers.0.input_layernorm.bias": [32],
                    "model.layers.1.block_sparse_moe.gate.weight": [4, 32],
                    "model.layers.1.block_sparse_moe.switch_mlp.gate_proj.weight": [4, 16, 32],
                    "lm_head.bias": [64],
                ],
                checkpoint: {
                    CheckpointLayout.splitExperts(
                        $0, stacked: "switch_mlp", perExpert: "experts",
                        names: ["gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"])
                }
            ) { seed in
                try Self.build(
                    PhiMoEConfiguration.self, Self.phiMoE,
                    [
                        "rope_scaling": [
                            "type": "longrope", "short_factor": [1, 1, 1, 1],
                            "long_factor": [1, 1.5, 2, 3], "short_mscale": 1.1,
                            "long_mscale": 1.2,
                        ]
                    ], seed: seed
                ) { PhiMoEModel($0) }
            },
            ModelCase(
                "PhiMoE (no RoPE scaling)",
                expectedShapes: [
                    "model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight": [4, 32, 16]
                ]
            ) { seed in
                try Self.build(PhiMoEConfiguration.self, Self.phiMoE, seed: seed) {
                    PhiMoEModel($0)
                }
            },
            ModelCase(
                "Phi3 (longrope, partial rotary, untied)",
                expectedShapes: [
                    "model.layers.0.self_attn.qkv_proj.weight": [64, 32],
                    "model.layers.0.mlp.gate_up_proj.weight": [96, 32],
                    "model.layers.1.mlp.down_proj.weight": [32, 48],
                    "lm_head.weight": [64, 32],
                ]
            ) { seed in
                try Self.build(
                    Phi3Configuration.self, Self.phi3,
                    [
                        "partial_rotary_factor": 0.5,
                        "rope_scaling": [
                            "type": "longrope", "short_factor": [1, 1], "long_factor": [1, 2],
                        ],
                    ], seed: seed
                ) { Phi3Model($0) }
            },
            ModelCase(
                "Phi3 (linear RoPE, traditional, tied)",
                expectedShapes: ["model.embed_tokens.weight": [64, 32]]
            ) { seed in
                try Self.build(
                    Phi3Configuration.self, Self.phi3,
                    [
                        "tie_word_embeddings": true, "rope_traditional": true,
                        "rope_scaling": ["type": "linear", "factor": 2],
                    ], seed: seed
                ) { Phi3Model($0) }
            },
            ModelCase(
                "NanoChat (soft cap 15)",
                expectedShapes: [
                    "transformer.wte.weight": [64, 32],
                    "transformer.h.0.attn.c_q.weight": [32, 32],
                    "transformer.h.0.attn.c_k.weight": [16, 32],
                    "transformer.h.1.mlp.c_fc.weight": [48, 32],
                    "lm_head.weight": [64, 32],
                ]
            ) { seed in
                try Self.makeNanoChat(seed: seed)
            },
            ModelCase(
                "NanoChat (no soft cap, no GQA)",
                expectedShapes: ["transformer.h.1.attn.c_v.weight": [32, 32]]
            ) { seed in
                try Self.makeNanoChat(
                    ["logits_softcap": 0, "num_key_value_heads": 4], seed: seed)
            },
            ModelCase(
                "Granite (tied, multipliers)",
                expectedShapes: [
                    "model.embed_tokens.weight": [64, 32],
                    "model.layers.1.self_attn.k_proj.weight": [16, 32],
                ]
            ) { seed in
                try Self.build(GraniteConfiguration.self, Self.granite, seed: seed) {
                    GraniteModel($0)
                }
            },
            ModelCase(
                "Granite (untied, biases, linear RoPE)",
                expectedShapes: [
                    "model.layers.0.self_attn.o_proj.bias": [32],
                    "model.layers.0.mlp.up_proj.bias": [48],
                    "lm_head.weight": [64, 32],
                ]
            ) { seed in
                try Self.build(
                    GraniteConfiguration.self, Self.granite,
                    [
                        "tie_word_embeddings": false, "attention_bias": true, "mlp_bias": true,
                        "rope_theta": 10000, "rope_scaling": ["type": "linear", "factor": 2],
                    ], seed: seed
                ) { GraniteModel($0) }
            },
            ModelCase(
                "OpenELM (shared embeddings, q/k norm, heads per layer)",
                expectedShapes: [
                    "transformer.token_embeddings.weight": [64, 48],
                    "transformer.layers.0.attn.qkv_proj.weight": [48, 48],
                    "transformer.layers.0.attn.q_norm.weight": [8],
                    "transformer.layers.1.attn.qkv_proj.weight": [96, 48],
                    "transformer.layers.1.attn.out_proj.weight": [48, 64],
                    "transformer.layers.0.ffn.proj_1.weight": [48, 48],
                    "transformer.layers.1.ffn.proj_2.weight": [48, 192],
                ]
            ) { seed in
                try Self.build(OpenElmConfiguration.self, Self.openELM, seed: seed) {
                    OpenELMModel($0)
                }
            },
            ModelCase(
                "OpenELM (no q/k norm)",
                expectedShapes: ["transformer.layers.1.attn_norm.weight": [48]]
            ) { seed in
                try Self.build(
                    OpenElmConfiguration.self, Self.openELM,
                    ["normalize_qk_projections": false], seed: seed
                ) { OpenELMModel($0) }
            },
            ModelCase(
                "Gemma2 (GQA)",
                expectedShapes: [
                    "model.layers.0.pre_feedforward_layernorm.weight": [32],
                    "model.layers.0.post_attention_layernorm.weight": [32],
                    "model.layers.1.self_attn.k_proj.weight": [16, 32],
                    "model.layers.0.mlp.gate_proj.weight": [48, 32],
                ]
            ) { seed in
                try Self.build(Gemma2Configuration.self, Self.gemma2, seed: seed) {
                    Gemma2Model($0)
                }
            },
            ModelCase(
                "Gemma2 (no GQA)",
                expectedShapes: ["model.layers.0.self_attn.v_proj.weight": [32, 32]]
            ) { seed in
                try Self.build(
                    Gemma2Configuration.self, Self.gemma2, ["num_key_value_heads": 4],
                    seed: seed
                ) { Gemma2Model($0) }
            },
        ]

        /// The `Linear` layers of one decoder layer that LoRA adapts by
        /// default, for each model family (the first word of the case name).
        static let loraKeys: [String: Set<String>] = [
            "SmolLM3": attentionAndMLP, "Exaone4": attentionAndMLP, "Olmo3": attentionAndMLP,
            "Granite": attentionAndMLP, "Gemma2": attentionAndMLP,
            "OlmoE": [
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate",
            ],
            "PhiMoE": [
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "block_sparse_moe.gate",
            ],
            "Phi3": [
                "self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj",
            ],
            "NanoChat": [
                "attn.c_q", "attn.c_k", "attn.c_v", "attn.c_proj", "mlp.c_fc", "mlp.c_proj",
            ],
            "OpenELM": ["attn.qkv_proj", "attn.out_proj", "ffn.proj_1", "ffn.proj_2"],
        ]

        static let attentionAndMLP: Set<String> = [
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
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

        /// LoRA adapts the decoder layers, and by default all `Linear`
        /// layers in them.
        @Test(arguments: cases) func loraAdaptsTheLinearLayersOfEachDecoderLayer(_ c: ModelCase)
            throws
        {
            let model = try #require(try c.make(1) as? any LoRAModel)
            let family = String(c.name.prefix { $0 != " " })
            let expected = try #require(Self.loraKeys[family], "\(c.name)")
            #expect(model.loraLayers.count == 2, "\(c.name)")
            #expect(Set(model.loraDefaultKeys) == expected, "\(c.name)")
            #expect(model.loraDefaultKeys.count == expected.count, "\(c.name): duplicate keys")
        }

        /// A SmolLM3 layer without RoPE sees the earlier tokens as a set: a
        /// swap of two earlier tokens does not change the logits of the last
        /// position. With RoPE, the swap changes them. Tolerance 1e-5: only
        /// the order of the attention sum changes.
        @Test func smolLM3LayerWithoutRoPEIgnoresTheTokenOrder() throws {
            let one: [String: Any] = ["num_hidden_layers": 1]
            let withoutRoPE = try Self.build(
                SmolLM3Configuration.self, Self.smolLM3,
                one.merging(["no_rope_layers": [0]]) { $1 }, seed: 2
            ) { SmolLM3Model($0) }
            let withRoPE = try Self.build(
                SmolLM3Configuration.self, Self.smolLM3,
                one.merging(["no_rope_layers": [1]]) { $1 }, seed: 2
            ) { SmolLM3Model($0) }

            let row = [3, 7, 11, 5]
            let swapped = [7, 3, 11, 5]
            func last(_ model: SmolLM3Model, _ row: [Int]) -> MLXArray {
                ForwardPassChecks.logits(model, [row])[0..., 3, 0...]
            }
            #expect(
                SyntheticModel.maxAbsDifference(
                    last(withoutRoPE, row), last(withoutRoPE, swapped)) <= 1e-5)
            #expect(
                SyntheticModel.maxAbsDifference(last(withRoPE, row), last(withRoPE, swapped))
                    > 1e-3)
        }

        /// Exaone 4 gives a RotatingKVCache to each local ("L") layer and a
        /// KVCacheSimple to each global ("G") layer. The full pass must apply
        /// the same window as the cache.
        @Test func exaone4SlidingLayersIgnoreTheWindowWithoutACache() throws {
            let model = try Self.build(
                Exaone4Configuration.self, Self.exaone4,
                ["sliding_window": 4, "sliding_window_pattern": "GL"], seed: 1
            ) { Exaone4Model($0) }
            let cache = model.newCache(parameters: nil)
            #expect(cache.count == 2)
            #expect(cache[0] is KVCacheSimple)
            #expect((cache[1] as? RotatingKVCache)?.maxSize == 4)

            let row = SyntheticModel.tokens(count: 11, vocabularySize: 64, seed: 1)
            withKnownIssue(
                """
                Exaone4ModelInner makes one mask from the first cache, without the window \
                (Exaone4.swift:168). In a pass without a cache the local layers attend to \
                all earlier tokens, while their RotatingKVCache keeps only the window.
                """
            ) {
                _ = ForwardPassChecks.checkCacheConsistency(
                    model, rows: [row], chunks: [5, 3, 1, 1, 1], tolerance: Self.tolerance,
                    cache: cache)
            } matching: {
                $0.isFailedExpectation(["cached logits differ"])
            }
        }

        /// Olmo 3 has its own cache factory with a RotatingKVCache for each
        /// sliding layer. With these caches, the chunked prompt and decode
        /// match the full pass, which applies the window through the mask.
        @Test func olmo3OwnCacheKeepsTheSlidingWindow() throws {
            let model = try Self.build(Olmo3Configuration.self, Self.olmo3, seed: 1) {
                Olmo3Model($0)
            }
            let own = model.newCache(parameters: GenerateParameters())
            #expect((own[0] as? RotatingKVCache)?.maxSize == 4)
            #expect(own[1] is KVCacheSimple)

            let rows = [1, 2].map { SyntheticModel.tokens(count: 11, vocabularySize: 64, seed: $0) }
            ForwardPassChecks.checkCacheConsistency(
                model, rows: [rows[0]], chunks: [5, 3, 1, 1, 1], tolerance: Self.tolerance,
                cache: model.newCache(parameters: GenerateParameters()))
            ForwardPassChecks.checkCacheConsistency(
                model, rows: rows, chunks: [5, 3, 1, 1, 1], tolerance: Self.tolerance,
                cache: model.newCache(parameters: GenerateParameters()))

            let generic = (model as any LanguageModel).newCache(parameters: nil)
            withKnownIssue(
                """
                Olmo3Model.newCache(parameters: GenerateParameters) is not the \
                LanguageModel requirement (Olmo3.swift:225), so generation gets a \
                KVCacheSimple for the sliding layers.
                """
            ) {
                #expect(generic[0] is RotatingKVCache, "sliding layer cache")
            } matching: {
                $0.isFailedExpectation(["sliding layer cache"])
            }
        }

        /// The final soft cap is `cap * tanh(logits / cap)`. A cap of 0
        /// turns it off. Tolerance 1e-5: the same float32 operations.
        @Test func nanoChatSoftCapsTheLogits() throws {
            let capped = try Self.makeNanoChat(["logits_softcap": 0.5], seed: 3)
            let raw = try Self.makeNanoChat(["logits_softcap": 0], seed: 3)
            let row = [SyntheticModel.tokens(count: 6, vocabularySize: 64, seed: 4)]
            let a = ForwardPassChecks.logits(capped, row)
            let b = ForwardPassChecks.logits(raw, row)
            #expect(SyntheticModel.maxAbs(b) > 0.5, "the cap must matter")
            #expect(SyntheticModel.maxAbs(a) <= 0.5)
            #expect(SyntheticModel.maxAbsDifference(a, 0.5 * tanh(b / 0.5)) <= 1e-5)
        }

        /// Gemma 2 caps the logits with `final_logit_softcapping`. A cap of
        /// 1e4 leaves logits near 1 unchanged to about 1e-8. Tolerance 1e-4:
        /// float32 rounding of the large cap.
        @Test func gemma2SoftCapsTheFinalLogits() throws {
            let capped = try Self.build(
                Gemma2Configuration.self, Self.gemma2, ["final_logit_softcapping": 0.5],
                seed: 3
            ) { Gemma2Model($0) }
            let wide = try Self.build(
                Gemma2Configuration.self, Self.gemma2, ["final_logit_softcapping": 1e4],
                seed: 3
            ) { Gemma2Model($0) }
            let row = [SyntheticModel.tokens(count: 6, vocabularySize: 64, seed: 4)]
            let a = ForwardPassChecks.logits(capped, row)
            let b = ForwardPassChecks.logits(wide, row)
            #expect(SyntheticModel.maxAbs(b) > 0.5, "the cap must matter")
            #expect(SyntheticModel.maxAbs(a) <= 0.5)
            #expect(SyntheticModel.maxAbsDifference(a, 0.5 * tanh(b / 0.5)) <= 1e-4)
        }

        /// Gemma 2 has no system role, so its message generator drops the
        /// system messages.
        @Test func gemma2MessageGeneratorDropsTheSystemMessage() throws {
            let model = Gemma2Model(
                try SyntheticModel.configuration(Gemma2Configuration.self, Self.gemma2))
            let messages = model.messageGenerator(tokenizer: TestTokenizer())
                .generate(messages: [.system("Be brief."), .user("Hello")])
            #expect(messages.count == 1)
            #expect(messages.first?["role"] as? String == "user")
            #expect(messages.first?["content"] as? String == "Hello")
        }

        /// Granite divides the logits by `logits_scaling`. A division by 4
        /// is exact in float32, so the tolerance is 1e-6.
        @Test func graniteDividesTheLogitsByTheLogitsScaling() throws {
            let scaled = try Self.build(GraniteConfiguration.self, Self.granite, seed: 3) {
                GraniteModel($0)
            }
            let unscaled = try Self.build(
                GraniteConfiguration.self, Self.granite, ["logits_scaling": 1], seed: 3
            ) { GraniteModel($0) }
            let row = [SyntheticModel.tokens(count: 6, vocabularySize: 64, seed: 4)]
            let a = ForwardPassChecks.logits(scaled, row)
            let b = ForwardPassChecks.logits(unscaled, row)
            #expect(SyntheticModel.maxAbs(b) > 1e-3)
            #expect(SyntheticModel.maxAbsDifference(a * 4, b) <= 1e-6)
        }

        /// Without q/k norm, OpenELM has no norm weights. Without shared
        /// embeddings, the head must map the model dimension to the
        /// vocabulary. The test does not run the untied model, because the
        /// wrong head shape makes the forward pass crash.
        @Test func openELMConfigurationShapesTheLayers() throws {
            let plain = try Self.build(
                OpenElmConfiguration.self, Self.openELM, ["normalize_qk_projections": false],
                seed: 1
            ) { OpenELMModel($0) }
            let plainParameters = SyntheticModel.flatParameters(plain)
            #expect(plainParameters["transformer.layers.0.attn.q_norm.weight"] == nil)
            #expect(plainParameters["transformer.layers.1.attn.k_norm.weight"] == nil)

            let untied = try Self.build(
                OpenElmConfiguration.self, Self.openELM, ["share_input_output_layers": false],
                seed: 1
            ) { OpenELMModel($0) }
            let head = SyntheticModel.flatParameters(untied)["lm_head.weight"]
            #expect(head != nil)
            withKnownIssue(
                """
                OpenELMModel builds lm_head as Linear(numTransformerLayers, vocabularySize) \
                (OpenELM.swift:194-195). The input size must be model_dim, as in the \
                reference mlx-lm openelm.py.
                """
            ) {
                #expect(head?.shape == [64, 48], "lm_head shape")
            } matching: {
                $0.isFailedExpectation(["lm_head shape"])
            }
        }
    }
}
