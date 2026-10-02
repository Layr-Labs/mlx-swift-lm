import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {

    /// Forward-pass tests of the dense Qwen 2 and Qwen 3 models with tiny
    /// random models: hidden size 32, 2 layers, 4 query heads over 2 key
    /// heads, vocabulary 64.
    ///
    /// Tolerance 1e-4 for the float32 comparisons: the cached path and the
    /// full path differ only in the order of the attention sums, which gives
    /// differences near 1e-6 for logits of size 1 to 5.
    @Suite
    struct QwenDenseForwardPassTests {

        static var common: [String: Any] {
            [
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
                "vocab_size": 64, "rope_theta": 10000,
            ]
        }

        static let cases: [ModelCase] = [
            ModelCase(
                "Qwen2",
                expectedShapes: [
                    "model.layers.0.self_attn.q_proj.weight": [32, 32],
                    "model.layers.0.self_attn.q_proj.bias": [32],
                    "model.layers.0.self_attn.k_proj.bias": [16],
                    "model.layers.1.mlp.down_proj.weight": [32, 48],
                    "lm_head.weight": [64, 32],
                ],
                droppedKeys: ["model.layers.0.self_attn.rotary_emb.inv_freq": [4]]
            ) { seed in
                try ModelCase.build(Qwen2Configuration.self, common, seed: seed) {
                    Qwen2Model($0)
                }
            },
            ModelCase(
                "Qwen2 tied, linear RoPE",
                expectedShapes: ["model.embed_tokens.weight": [64, 32]],
                droppedKeys: ["lm_head.weight": [64, 32]]
            ) { seed in
                try ModelCase.build(
                    Qwen2Configuration.self,
                    common.merging([
                        "tie_word_embeddings": true,
                        "rope_scaling": ["type": "linear", "factor": 2.0],
                    ]) { $1 }, seed: seed
                ) { Qwen2Model($0) }
            },
            ModelCase(
                "Qwen3",
                expectedShapes: [
                    "model.layers.0.self_attn.q_proj.weight": [64, 32],
                    "model.layers.0.self_attn.q_norm.weight": [16],
                    "model.layers.0.self_attn.k_norm.weight": [16],
                    "model.layers.1.mlp.gate_proj.weight": [48, 32],
                    "lm_head.weight": [64, 32],
                ]
            ) { seed in
                try ModelCase.build(
                    Qwen3Configuration.self, common.merging(["head_dim": 16]) { $1 }, seed: seed
                ) { Qwen3Model($0) }
            },
            ModelCase(
                "Qwen3 tied, linear RoPE",
                expectedShapes: ["model.embed_tokens.weight": [64, 32]],
                droppedKeys: ["lm_head.weight": [64, 32]]
            ) { seed in
                try ModelCase.build(
                    Qwen3Configuration.self,
                    common.merging([
                        "head_dim": 16, "tie_word_embeddings": true,
                        "rope_scaling": ["type": "linear", "factor": 2.0],
                    ]) { $1 }, seed: seed
                ) { Qwen3Model($0) }
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
    }
}
