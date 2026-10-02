import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {

    /// Forward-pass tests of `LagunaModel` with a tiny random model.
    ///
    /// The model has one sliding-window layer (window 4) with a dense MLP and
    /// one full-attention layer with YaRN RoPE, per-head output gates and a
    /// mixture of 4 experts. The test sequences are longer than the window,
    /// so the cache tests also check the rotating cache of the sliding layer.
    @Suite
    struct LagunaForwardPassTests {

        static let vocabularySize = 64

        static var base: [String: Any] {
            [
                "vocab_size": vocabularySize,
                "hidden_size": 32,
                "intermediate_size": 48,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "max_position_embeddings": 256,
                "rms_norm_eps": 1e-6,
                "sliding_window": 4,
                "layer_types": ["sliding_attention", "full_attention"],
                "mlp_only_layers": [0],
                "gating": "per-head",
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 32,
                "shared_expert_intermediate_size": 32,
                "moe_routed_scaling_factor": 2.5,
                "rope_theta": 10000,
                "rope_parameters": [
                    "full_attention": [
                        "rope_type": "yarn", "rope_theta": 10000, "factor": 4.0,
                        "original_max_position_embeddings": 64, "partial_rotary_factor": 0.5,
                    ],
                    "sliding_attention": ["rope_type": "default", "rope_theta": 10000],
                ],
            ]
        }

        static func makeModel(_ overrides: [String: Any] = [:], seed: UInt64 = 1) throws
            -> LagunaModel
        {
            let configuration = try SyntheticModel.configuration(
                LagunaConfiguration.self, base, overrides: overrides)
            let model = LagunaModel(configuration)
            SyntheticModel.randomize(model, seed: seed)
            return model
        }

        static func row(_ seed: Int, count: Int = 11) -> [Int] {
            SyntheticModel.tokens(count: count, vocabularySize: vocabularySize, seed: seed)
        }

        // Tolerance of the float32 comparisons: the cached path and the full
        // path run the same float32 math, but attention reduces in another
        // order (a single-query kernel against a full-sequence kernel). This
        // gives differences near 1e-6 for logits of size 1 to 5. A cache
        // fault, for example a RoPE position off by one, gives differences
        // above 1e-2.
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

        @Test func cachedDecodeMatchesTheFullForwardPass() throws {
            let model = try Self.makeModel()
            // 5 + 3 + 3 steps: the second chunk and the decode steps go past
            // the sliding window of 4.
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
            let model = try Self.makeModel()
            let parameters = SyntheticModel.flatParameters(model)
            let expected: [String: [Int]] = [
                "model.embed_tokens.weight": [64, 32],
                "model.layers.0.self_attn.q_proj.weight": [32, 32],
                "model.layers.0.self_attn.k_proj.weight": [16, 32],
                "model.layers.0.self_attn.g_proj.weight": [4, 32],
                "model.layers.0.self_attn.q_norm.weight": [8],
                "model.layers.0.mlp.gate_proj.weight": [48, 32],
                "model.layers.1.mlp.gate.weight": [4, 32],
                "model.layers.1.mlp.gate.e_score_correction_bias": [4],
                "model.layers.1.mlp.switch_mlp.gate_proj.weight": [4, 32, 32],
                "model.layers.1.mlp.switch_mlp.down_proj.weight": [4, 32, 32],
                "model.layers.1.mlp.shared_expert.up_proj.weight": [32, 32],
                "model.norm.weight": [32],
                "lm_head.weight": [64, 32],
            ]
            for (key, shape) in expected {
                #expect(parameters[key]?.shape == shape, "\(key)")
            }
            // Layer 0 is in mlp_only_layers, so it has no router.
            #expect(parameters["model.layers.0.mlp.gate.weight"] == nil)
        }

        @Test func loaderAcceptsACheckpointAndGivesTheSameLogits() throws {
            let reference = try Self.makeModel(seed: 5)
            var checkpoint = SyntheticModel.flatParameters(reference)
            // sanitize(weights:) must drop precomputed rotary tables.
            checkpoint["model.layers.1.self_attn.rotary_emb.inv_freq"] = MLXArray.ones([2])

            let loaded = try Self.makeModel(seed: 6)
            try SyntheticModel.load(checkpoint, into: loaded)

            let rows = [Self.row(3)]
            let difference = SyntheticModel.maxAbsDifference(
                ForwardPassChecks.logits(reference, rows), ForwardPassChecks.logits(loaded, rows))
            #expect(difference == 0)
        }

        @Test func loaderRejectsAWrongShape() throws {
            let model = try Self.makeModel()
            var checkpoint = SyntheticModel.flatParameters(model)
            checkpoint["model.layers.0.self_attn.k_proj.weight"] = MLXArray.zeros([8, 32])
            #expect(throws: (any Error).self) {
                try SyntheticModel.load(checkpoint, into: try Self.makeModel())
            }
        }

        @Test func tiedEmbeddingsDropTheCheckpointHead() throws {
            let reference = try Self.makeModel(["tie_word_embeddings": true], seed: 5)
            #expect(SyntheticModel.flatParameters(reference)["lm_head.weight"] == nil)

            var checkpoint = SyntheticModel.flatParameters(reference)
            checkpoint["lm_head.weight"] = MLXArray.zeros([64, 32])
            let loaded = try Self.makeModel(["tie_word_embeddings": true], seed: 6)
            try SyntheticModel.load(checkpoint, into: loaded)

            let rows = [Self.row(3)]
            let logits = ForwardPassChecks.logits(loaded, rows)
            #expect(
                SyntheticModel.maxAbsDifference(logits, ForwardPassChecks.logits(reference, rows))
                    == 0)
            // A zero head would give zero logits. The tied head uses the
            // embedding table instead.
            #expect(SyntheticModel.maxAbs(logits) > 0.1)
        }

        @Test(arguments: [
            ("per-head", [4, 32] as [Int]?),
            ("per-element", [32, 32]),
            ("false", nil),
        ])
        func gatingModeSetsTheGateProjection(mode: String, gateShape: [Int]?) throws {
            let gating: Any = mode == "false" ? false : mode
            let model = try Self.makeModel(["gating": gating])
            let parameters = SyntheticModel.flatParameters(model)
            #expect(parameters["model.layers.1.self_attn.g_proj.weight"]?.shape == gateShape)
            ForwardPassChecks.checkCacheConsistency(
                model, rows: [Self.row(4)], chunks: [6, 2, 1, 1, 1], tolerance: Self.tolerance)
        }

        @Test func quantizationConfigQuantizesOnlyTheExperts() throws {
            let model = try Self.makeModel([
                "quantization": ["group_size": 32, "bits": 4]
            ])
            let parameters = SyntheticModel.flatParameters(model)
            #expect(parameters["model.layers.1.mlp.switch_mlp.gate_proj.scales"] != nil)
            #expect(parameters["model.layers.1.mlp.shared_expert.gate_proj.scales"] != nil)
            #expect(parameters["model.layers.1.self_attn.q_proj.scales"] == nil)
            #expect(parameters["model.layers.1.mlp.gate.scales"] == nil)
            #expect(parameters["model.layers.0.mlp.gate_proj.scales"] == nil)

            ForwardPassChecks.checkShapeDTypeAndFinite(
                model, vocabularySize: Self.vocabularySize, length: 5)
            ForwardPassChecks.checkCacheConsistency(
                model, rows: [Self.row(5)], chunks: [5, 3, 1, 1, 1], tolerance: Self.tolerance)
        }
    }
}
