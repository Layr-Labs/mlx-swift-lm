import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {

    /// Forward-pass tests of `Qwen3NextModel` with a tiny random model.
    ///
    /// The Metal kernel of the gated delta net needs a key head dimension
    /// that is a multiple of 32, so the linear layer uses 32.
    ///
    /// Layer 0 is a gated delta net (linear attention with a convolution
    /// state and a recurrent state in a `MambaCache`). Layer 1 is full
    /// attention with partial RoPE and an output gate. Both layers have a
    /// mixture of 4 experts and a gated shared expert.
    @Suite
    struct Qwen3NextForwardPassTests {

        static let vocabularySize = 64

        static var base: [String: Any] {
            [
                "vocab_size": vocabularySize,
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 48,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 16,
                "moe_intermediate_size": 16,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000,
                "partial_rotary_factor": 0.5,
                "full_attention_interval": 2,
                "norm_topk_prob": true,
            ]
        }

        static func makeModel(_ overrides: [String: Any] = [:], seed: UInt64 = 1) throws
            -> Qwen3NextModel
        {
            let configuration = try SyntheticModel.configuration(
                Qwen3NextConfiguration.self, base, overrides: overrides)
            let model = Qwen3NextModel(configuration)
            SyntheticModel.randomize(model, seed: seed)
            return model
        }

        static func row(_ seed: Int, count: Int = 11) -> [Int] {
            SyntheticModel.tokens(count: count, vocabularySize: vocabularySize, seed: seed)
        }

        // Tolerance of the float32 comparisons: the cached path runs the
        // recurrence and the convolution chunk by chunk, and attention with
        // a single query, so the sums run in another order than in the full
        // pass. The differences are near 1e-6 for logits of size 1 to 5. A
        // fault in the carried convolution or recurrent state gives
        // differences above 1e-2.
        static let tolerance: Float = 1e-4

        @Test func logitsHaveTheExpectedShapeAndAreFinite() throws {
            let model = try Self.makeModel()
            ForwardPassChecks.checkShapeDTypeAndFinite(
                model, vocabularySize: Self.vocabularySize, length: 7)
        }

        @Test func sameSeedGivesTheSameLogits() throws {
            try ForwardPassChecks.checkDeterminism(
                make: { try Self.makeModel() }, seed: 3, vocabularySize: Self.vocabularySize)
        }

        @Test func cacheHasOneStateCachePerLinearLayer() throws {
            let cache = try Self.makeModel().newCache(parameters: nil)
            #expect(cache.count == 2)
            #expect(cache[0] is MambaCache)
            #expect(cache[1] is KVCacheSimple)
        }

        @Test func cachedDecodeMatchesTheFullForwardPass() throws {
            let model = try Self.makeModel()
            ForwardPassChecks.checkCacheConsistency(
                model, rows: [Self.row(1)], chunks: [5, 3, 1, 1, 1], tolerance: Self.tolerance)
        }

        @Test func cachedDecodeMatchesTheFullForwardPassForTwoRows() throws {
            let model = try Self.makeModel()
            ForwardPassChecks.checkCacheConsistency(
                model, rows: [Self.row(1), Self.row(2)], chunks: [4, 4, 1, 1, 1],
                tolerance: Self.tolerance)
        }

        @Test func eachRowOfABatchMatchesTheRowAlone() throws {
            let model = try Self.makeModel()
            ForwardPassChecks.checkBatchInvariance(
                model, rowA: Self.row(1), rowB: Self.row(2), tolerance: Self.tolerance)
        }

        @Test func aLaterTokenDoesNotChangeEarlierLogits() throws {
            let model = try Self.makeModel()
            ForwardPassChecks.checkCausality(
                model, row: Self.row(1), position: 6, vocabularySize: Self.vocabularySize,
                tolerance: Self.tolerance)
        }

        @Test func parameterTreeHasTheCheckpointKeysAndShapes() throws {
            let parameters = SyntheticModel.flatParameters(try Self.makeModel())
            // Key dimension 2 x 32 = 64, value dimension 4 x 16 = 64, so the
            // convolution runs over 64 + 64 + 64 = 192 channels.
            let expected: [String: [Int]] = [
                "model.embed_tokens.weight": [64, 32],
                "model.layers.0.linear_attn.in_proj_qkvz.weight": [256, 32],
                "model.layers.0.linear_attn.in_proj_ba.weight": [8, 32],
                "model.layers.0.linear_attn.conv1d.weight": [192, 4, 1],
                "model.layers.0.linear_attn.dt_bias": [4],
                "model.layers.0.linear_attn.A_log": [4],
                "model.layers.0.linear_attn.norm.weight": [16],
                "model.layers.0.linear_attn.out_proj.weight": [32, 64],
                "model.layers.1.self_attn.q_proj.weight": [64, 32],
                "model.layers.1.self_attn.k_proj.weight": [16, 32],
                "model.layers.1.self_attn.o_proj.weight": [32, 32],
                "model.layers.1.mlp.gate.weight": [4, 32],
                "model.layers.1.mlp.switch_mlp.up_proj.weight": [4, 16, 32],
                "model.layers.1.mlp.shared_expert.gate_proj.weight": [16, 32],
                "model.layers.1.mlp.shared_expert_gate.weight": [1, 32],
                "model.norm.weight": [32],
                "lm_head.weight": [64, 32],
            ]
            for (key, shape) in expected {
                #expect(parameters[key]?.shape == shape, "\(key)")
            }
            #expect(parameters["model.layers.0.self_attn.q_proj.weight"] == nil)
            #expect(parameters["model.layers.1.linear_attn.A_log"] == nil)
        }

        @Test func loaderAcceptsAnMLXCheckpointAndGivesTheSameLogits() throws {
            let reference = try Self.makeModel(seed: 5)
            var checkpoint = SyntheticModel.flatParameters(reference)
            // sanitize(weights:) must drop the MTP head.
            checkpoint["mtp.fc.weight"] = MLXArray.zeros([32, 64])

            let loaded = try Self.makeModel(seed: 6)
            try SyntheticModel.load(checkpoint, into: loaded)

            let rows = [Self.row(3)]
            #expect(
                SyntheticModel.maxAbsDifference(
                    ForwardPassChecks.logits(reference, rows),
                    ForwardPassChecks.logits(loaded, rows)) == 0)
        }

        /// A Hugging Face checkpoint stores each expert on its own, the
        /// convolution as `[channels, 1, kernel]`, and the RMS norm scales
        /// as an offset from 1. `sanitize(weights:)` converts all three.
        @Test func loaderConvertsAHuggingFaceCheckpoint() throws {
            let reference = try Self.makeModel(seed: 5)
            var checkpoint: [String: MLXArray] = [:]
            let offsetNorms = [
                ".input_layernorm.weight", ".post_attention_layernorm.weight", "model.norm.weight",
                ".q_norm.weight", ".k_norm.weight",
            ]
            for (key, value) in SyntheticModel.flatParameters(reference) {
                if key.contains(".switch_mlp.") {
                    for expert in 0 ..< 4 {
                        let expertKey = key.replacingOccurrences(
                            of: ".switch_mlp.", with: ".experts.\(expert).")
                        checkpoint[expertKey] = value[expert]
                    }
                } else if key.hasSuffix("conv1d.weight") {
                    checkpoint[key] = value.movedAxis(source: 1, destination: 2)
                } else if offsetNorms.contains(where: { key.hasSuffix($0) }) {
                    checkpoint[key] = value - 1
                } else {
                    checkpoint[key] = value
                }
            }
            #expect(checkpoint["model.layers.0.mlp.experts.3.down_proj.weight"]?.shape == [32, 16])
            #expect(checkpoint["model.layers.0.linear_attn.conv1d.weight"]?.shape == [192, 1, 4])

            let loaded = try Self.makeModel(seed: 6)
            try SyntheticModel.load(checkpoint, into: loaded)

            // The norm scales go through w - 1 + 1 in float32, which can
            // round the last bit.
            let rows = [Self.row(3)]
            let difference = SyntheticModel.maxAbsDifference(
                ForwardPassChecks.logits(reference, rows), ForwardPassChecks.logits(loaded, rows))
            #expect(difference <= 1e-5)
        }

        @Test func loaderRejectsAWrongShape() throws {
            let model = try Self.makeModel()
            var checkpoint = SyntheticModel.flatParameters(model)
            checkpoint["model.layers.0.linear_attn.in_proj_ba.weight"] = MLXArray.zeros([4, 32])
            #expect(throws: (any Error).self) {
                try SyntheticModel.load(checkpoint, into: try Self.makeModel())
            }
        }
    }
}
