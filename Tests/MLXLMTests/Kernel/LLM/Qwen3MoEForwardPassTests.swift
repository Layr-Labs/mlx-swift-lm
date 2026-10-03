import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {

    /// Forward-pass tests of `Qwen3MoEModel` with a tiny random model.
    ///
    /// Layer 0 has a dense MLP. Layer 1 has a mixture of 4 experts with top-2
    /// routing (`decoder_sparse_step` 2). The head dimension is 32, so that
    /// the KV cache can be quantized with a group size of 32.
    @Suite
    struct Qwen3MoEForwardPassTests {

        static let vocabularySize = 64

        static var base: [String: Any] {
            [
                "vocab_size": vocabularySize,
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 48,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 2,
                "mlp_only_layers": [Int](),
                "moe_intermediate_size": 16,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000,
                "norm_topk_prob": true,
            ]
        }

        static func makeModel(_ overrides: [String: Any] = [:], seed: UInt64 = 1) throws
            -> Qwen3MoEModel
        {
            let configuration = try SyntheticModel.configuration(
                Qwen3MoEConfiguration.self, base, overrides: overrides)
            let model = Qwen3MoEModel(configuration)
            SyntheticModel.randomize(model, seed: seed)
            return model
        }

        static func row(_ seed: Int, count: Int = 11) -> [Int] {
            SyntheticModel.tokens(count: count, vocabularySize: vocabularySize, seed: seed)
        }

        // Tolerance of the float32 comparisons: the cached path and the full
        // path differ only in the order of the attention sums. The
        // differences are near 1e-6 for logits of size 1 to 5.
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

        @Test(arguments: ["default", "linear"])
        func cachedDecodeMatchesTheFullForwardPass(rope: String) throws {
            let overrides: [String: Any] =
                rope == "linear" ? ["rope_scaling": ["type": "linear", "factor": 2.0]] : [:]
            let model = try Self.makeModel(overrides)
            ForwardPassChecks.checkCacheConsistency(
                model, rows: [Self.row(1)], chunks: [5, 3, 1, 1, 1], tolerance: Self.tolerance)
            ForwardPassChecks.checkCacheConsistency(
                model, rows: [Self.row(1), Self.row(2)], chunks: [4, 4, 1, 1, 1],
                tolerance: Self.tolerance)
        }

        @Test func linearRopeScalingChangesTheLogits() throws {
            let plain = try Self.makeModel()
            let scaled = try Self.makeModel(["rope_scaling": ["type": "linear", "factor": 2.0]])
            let rows = [Self.row(1)]
            let plainLogits = ForwardPassChecks.logits(plain, rows)
            let scaledLogits = ForwardPassChecks.logits(scaled, rows)
            // Position 0 has no rotation, so it does not depend on the scale.
            #expect(
                SyntheticModel.maxAbsDifference(plainLogits[0..., 0], scaledLogits[0..., 0])
                    <= Self.tolerance)
            #expect(
                SyntheticModel.maxAbsDifference(plainLogits[0..., 1...], scaledLogits[0..., 1...])
                    > 1e-3)
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

        /// `maybeQuantizeKVCache` turns the caches into 8-bit caches after
        /// the prompt. The decode steps then run
        /// `quantizedScaledDotProductAttention`.
        @Test func quantizedKVCacheDecodeStaysCloseToTheFullPass() throws {
            let model = try Self.makeModel()
            let row = Self.row(1)
            let full = ForwardPassChecks.logits(model, [row])

            var cache = model.newCache(parameters: nil)
            _ = ForwardPassChecks.logits(model, [Array(row[..<8])], cache: cache)
            maybeQuantizeKVCache(cache: &cache, kvBits: 8, kvGroupSize: 32, quantizedKVStart: 0)
            #expect(cache.allSatisfy { $0 is QuantizedKVCache })
            #expect(cache.allSatisfy { $0.offset == 8 })

            var worst: Float = 0
            for position in 8 ..< row.count {
                let step = ForwardPassChecks.logits(
                    model, [[row[position]]], cache: cache)
                worst = max(
                    worst,
                    SyntheticModel.maxAbsDifference(step, full[0..., position ..< position + 1]))
            }
            // 8-bit affine quantization in groups of 32 keeps each key and
            // value within about 0.5% of its group range. For logits of size
            // 1 to 5 this gives differences below 0.05. The difference is
            // not zero, which shows that the quantized path ran.
            #expect(worst <= 0.05)
            #expect(worst > 0)
        }

        @Test func parameterTreeHasTheCheckpointKeysAndShapes() throws {
            let parameters = SyntheticModel.flatParameters(try Self.makeModel())
            let expected: [String: [Int]] = [
                "model.embed_tokens.weight": [64, 32],
                "model.layers.0.self_attn.q_proj.weight": [128, 32],
                "model.layers.0.self_attn.k_proj.weight": [64, 32],
                "model.layers.0.self_attn.o_proj.weight": [32, 128],
                "model.layers.0.self_attn.q_norm.weight": [32],
                "model.layers.0.mlp.gate_proj.weight": [48, 32],
                "model.layers.1.mlp.gate.weight": [4, 32],
                "model.layers.1.mlp.switch_mlp.gate_proj.weight": [4, 16, 32],
                "model.layers.1.mlp.switch_mlp.down_proj.weight": [4, 32, 16],
                "lm_head.weight": [64, 32],
            ]
            for (key, shape) in expected {
                #expect(parameters[key]?.shape == shape, "\(key)")
            }
            #expect(parameters["model.layers.0.mlp.gate.weight"] == nil)
        }

        /// A Hugging Face checkpoint stores each expert on its own.
        /// `sanitize(weights:)` stacks them into `switch_mlp`, and drops the
        /// head when the embeddings are tied. `sanitize(weights:)` finds the
        /// per-expert format by a key of layer 0, so every layer is sparse
        /// here.
        @Test(arguments: [false, true])
        func loaderStacksPerExpertWeights(tied: Bool) throws {
            let overrides: [String: Any] = ["tie_word_embeddings": tied, "decoder_sparse_step": 1]
            let reference = try Self.makeModel(overrides, seed: 5)
            var checkpoint: [String: MLXArray] = [:]
            for (key, value) in SyntheticModel.flatParameters(reference) {
                if key.contains(".switch_mlp.") {
                    for expert in 0 ..< 4 {
                        let expertKey = key.replacingOccurrences(
                            of: ".switch_mlp.", with: ".experts.\(expert).")
                        checkpoint[expertKey] = value[expert]
                    }
                } else {
                    checkpoint[key] = value
                }
            }
            if tied {
                checkpoint["lm_head.weight"] = MLXArray.zeros([64, 32])
            }
            #expect(checkpoint["model.layers.0.mlp.experts.0.up_proj.weight"]?.shape == [16, 32])

            let loaded = try Self.makeModel(overrides, seed: 6)
            try SyntheticModel.load(checkpoint, into: loaded)
            let rows = [Self.row(3)]
            #expect(
                SyntheticModel.maxAbsDifference(
                    ForwardPassChecks.logits(reference, rows),
                    ForwardPassChecks.logits(loaded, rows)) == 0)
        }

        @Test func loaderRejectsAWrongShape() throws {
            var checkpoint = SyntheticModel.flatParameters(try Self.makeModel())
            checkpoint["model.layers.1.mlp.switch_mlp.up_proj.weight"] = MLXArray.zeros([3, 16, 32])
            #expect(throws: (any Error).self) {
                try SyntheticModel.load(checkpoint, into: try Self.makeModel())
            }
        }
    }
}
