import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {

    /// Forward-pass tests of `DeepseekV3Model` with a tiny random model:
    /// multi-head latent attention (LoRA ranks 16), one dense layer and one
    /// layer with 4 routed experts in 2 groups and a shared expert.
    @Suite
    struct DeepseekV3ForwardPassTests {

        static let vocabularySize = 64

        static var base: [String: Any] {
            [
                "vocab_size": vocabularySize, "hidden_size": 32, "intermediate_size": 48,
                "moe_intermediate_size": 16, "num_hidden_layers": 2, "num_attention_heads": 4,
                "num_key_value_heads": 4, "routed_scaling_factor": 1.0, "kv_lora_rank": 16,
                "q_lora_rank": 16, "qk_rope_head_dim": 8, "v_head_dim": 8,
                "qk_nope_head_dim": 8, "norm_topk_prob": true, "moe_layer_freq": 1,
                "first_k_dense_replace": 1, "max_position_embeddings": 256,
                "rms_norm_eps": 1e-6, "rope_theta": 10000, "attention_bias": false,
                "n_shared_experts": 1, "n_routed_experts": 4, "n_group": 2, "topk_group": 1,
                "num_experts_per_tok": 2,
            ]
        }

        static func makeModel(seed: UInt64 = 1) throws -> DeepseekV3Model {
            let model = DeepseekV3Model(
                try SyntheticModel.configuration(DeepseekV3Configuration.self, base))
            SyntheticModel.randomize(model, seed: seed)
            return model
        }

        static func row(_ seed: Int, count: Int = 11) -> [Int] {
            SyntheticModel.tokens(count: count, vocabularySize: vocabularySize, seed: seed)
        }

        // Tolerance of the float32 comparisons: the paths differ only in the
        // order of the attention sums; differences are near 1e-6.
        static let tolerance: Float = 1e-4

        @Test func logitsHaveTheExpectedShapeAndAreFinite() throws {
            ForwardPassChecks.checkShapeDTypeAndFinite(
                try Self.makeModel(), vocabularySize: Self.vocabularySize, length: 7)
        }

        @Test func sameSeedGivesTheSameLogits() throws {
            try ForwardPassChecks.checkDeterminism(
                make: { try Self.makeModel() }, seed: 3, vocabularySize: Self.vocabularySize)
        }

        @Test func eachRowOfABatchMatchesTheRowAlone() throws {
            ForwardPassChecks.checkBatchInvariance(
                try Self.makeModel(), rowA: Self.row(1), rowB: Self.row(2),
                tolerance: Self.tolerance)
        }

        @Test func aLaterTokenDoesNotChangeEarlierLogits() throws {
            ForwardPassChecks.checkCausality(
                try Self.makeModel(), row: Self.row(1), position: 6,
                vocabularySize: Self.vocabularySize, tolerance: Self.tolerance)
        }

        /// `newCache(parameters:)` must give one cache for each layer. The
        /// test does not run the model with that cache, because an empty
        /// cache list stops the process with an index error.
        @Test func newCacheGivesOneCachePerLayer() throws {
            let cache = try Self.makeModel().newCache(parameters: nil)
            #expect(cache.count == 2, "cache count")
        }

        /// With one KV cache for each layer, a prompt in chunks and decode
        /// steps must give the logits of the full pass.
        @Test func cachedDecodeMatchesTheFullForwardPass() throws {
            let model = try Self.makeModel()
            ForwardPassChecks.checkCacheConsistency(
                model, rows: [Self.row(1)], chunks: [5, 3, 1, 1, 1],
                tolerance: Self.tolerance, cache: [KVCacheSimple(), KVCacheSimple()])
        }

        @Test func parameterTreeHasTheCheckpointKeysAndShapes() throws {
            let parameters = SyntheticModel.flatParameters(try Self.makeModel())
            let expected: [String: [Int]] = [
                "model.layers.0.self_attn.q_a_proj.weight": [16, 32],
                "model.layers.0.self_attn.q_b_proj.weight": [64, 16],
                "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": [24, 32],
                "model.layers.0.self_attn.kv_b_proj.weight": [64, 16],
                "model.layers.1.mlp.gate.weight": [4, 32],
                "model.layers.1.mlp.gate.e_score_correction_bias": [4],
                "model.layers.1.mlp.switch_mlp.gate_proj.weight": [4, 16, 32],
                "lm_head.weight": [64, 32],
            ]
            for (key, shape) in expected {
                #expect(parameters[key]?.shape == shape, "\(key)")
            }
        }

        @Test func loaderAcceptsTheStackedLayout() throws {
            let reference = try Self.makeModel(seed: 5)
            var checkpoint = SyntheticModel.flatParameters(reference)
            checkpoint["model.layers.0.self_attn.rotary_emb.inv_freq"] = MLXArray.ones([4])
            checkpoint["model.layers.61.embed_tokens.weight"] = MLXArray.ones([4])
            let loaded = try Self.makeModel(seed: 6)
            try SyntheticModel.load(checkpoint, into: loaded)
            let rows = [Self.row(3)]
            #expect(
                SyntheticModel.maxAbsDifference(
                    ForwardPassChecks.logits(reference, rows),
                    ForwardPassChecks.logits(loaded, rows)) == 0)

            checkpoint["model.layers.1.mlp.gate.weight"] = MLXArray.zeros([5, 32])
            #expect(throws: (any Error).self) {
                try SyntheticModel.load(checkpoint, into: try Self.makeModel(seed: 6))
            }
        }

        /// The original checkpoint stores one tensor per expert.
        @Test func loaderStacksPerExpertWeights() throws {
            let reference = try Self.makeModel(seed: 5)
            let checkpoint = CheckpointLayout.splitExperts(
                SyntheticModel.flatParameters(reference), stacked: "switch_mlp",
                perExpert: "experts")
            let loaded = try Self.makeModel(seed: 6)
            let sanitized = loaded.sanitize(weights: checkpoint)
            #expect(
                sanitized["model.layers.1.mlp.switch_mlp.up_proj.weight"]?.shape == [4, 16, 32])
            #expect(
                !sanitized.keys.contains { $0.contains(".experts.") }, "per-expert keys kept")
            try SyntheticModel.load(checkpoint, into: loaded)
        }
    }
}
