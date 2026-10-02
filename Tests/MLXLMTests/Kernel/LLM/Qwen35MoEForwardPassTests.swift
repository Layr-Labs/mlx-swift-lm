import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {

    /// Forward-pass and loading tests of `Qwen35MoEModel` with a tiny random
    /// model: 4 layers that alternate a gated delta net and full attention,
    /// each with 4 routed experts (fused gate and up projection) and a gated
    /// shared expert.
    @Suite
    struct Qwen35MoEForwardPassTests {

        static let vocabularySize = 64

        static var base: [String: Any] {
            [
                "model_type": "qwen3_5_moe",
                "text_config": [
                    "model_type": "qwen3_5_moe_text", "hidden_size": 64,
                    "num_hidden_layers": 4, "intermediate_size": 32,
                    "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 16,
                    "linear_num_value_heads": 1, "linear_num_key_heads": 1,
                    "linear_key_head_dim": 64, "linear_value_head_dim": 64,
                    "linear_conv_kernel_dim": 4, "full_attention_interval": 2,
                    "vocab_size": vocabularySize, "num_experts": 4, "num_experts_per_tok": 2,
                    "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
                    "norm_topk_prob": true,
                ] as [String: Any],
            ]
        }

        static func makeModel(seed: UInt64 = 1) throws -> Qwen35MoEModel {
            let model = Qwen35MoEModel(
                try SyntheticModel.configuration(Qwen35Configuration.self, base))
            SyntheticModel.randomize(model, seed: seed)
            return model
        }

        static func row(_ seed: Int, count: Int = 11) -> [Int] {
            SyntheticModel.tokens(count: count, vocabularySize: vocabularySize, seed: seed)
        }

        // Tolerance of the float32 comparisons: the cached path runs the
        // recurrence chunk by chunk and attention with one query, so sums run
        // in another order. The differences are near 1e-6.
        static let tolerance: Float = 1e-4

        @Test func logitsHaveTheExpectedShapeAndAreFinite() throws {
            ForwardPassChecks.checkShapeDTypeAndFinite(
                try Self.makeModel(), vocabularySize: Self.vocabularySize, length: 7)
        }

        @Test func cachedDecodeMatchesTheFullForwardPass() throws {
            let model = try Self.makeModel()
            ForwardPassChecks.checkCacheConsistency(
                model, rows: [Self.row(1)], chunks: [5, 3, 1, 1, 1], tolerance: Self.tolerance)
            ForwardPassChecks.checkCacheConsistency(
                model, rows: [Self.row(1), Self.row(2)], chunks: [4, 4, 1, 1, 1],
                tolerance: Self.tolerance)
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

        /// Loads `checkpoint` into a new model and compares its logits with
        /// the reference model.
        static func loadedDifference(_ checkpoint: [String: MLXArray], reference: Qwen35MoEModel)
            throws -> Float
        {
            let loaded = try makeModel(seed: 6)
            try SyntheticModel.load(checkpoint, into: loaded)
            let rows = [row(3)]
            return SyntheticModel.maxAbsDifference(
                ForwardPassChecks.logits(reference, rows), ForwardPassChecks.logits(loaded, rows))
        }

        @Test func loaderAcceptsTheFusedLayout() throws {
            let reference = try Self.makeModel(seed: 5)
            let checkpoint = SyntheticModel.flatParameters(reference)
            #expect(
                checkpoint["language_model.model.layers.0.mlp.switch_mlp.gate_up_proj.weight"]?
                    .shape == [4, 64, 64])
            #expect(try Self.loadedDifference(checkpoint, reference: reference) == 0)
        }

        /// An MLX conversion stores `switch_mlp.gate_proj` and `up_proj`
        /// apart. The sanitizer joins them, gate rows first.
        @Test func loaderFusesSplitGateAndUpProjections() throws {
            let reference = try Self.makeModel(seed: 5)
            var checkpoint: [String: MLXArray] = [:]
            for (key, value) in SyntheticModel.flatParameters(reference) {
                if key.hasSuffix(".switch_mlp.gate_up_proj.weight") {
                    let halves = split(value, parts: 2, axis: 1)
                    checkpoint[key.replacingOccurrences(of: "gate_up_proj", with: "gate_proj")] =
                        halves[0]
                    checkpoint[key.replacingOccurrences(of: "gate_up_proj", with: "up_proj")] =
                        halves[1]
                } else {
                    checkpoint[key] = value
                }
            }
            #expect(try Self.loadedDifference(checkpoint, reference: reference) == 0)
        }

        /// A raw Hugging Face export: `model.language_model.*` keys, the
        /// head at the top level, stacked `experts.gate_up_proj` and
        /// `experts.down_proj` tensors without a suffix, the convolution as
        /// `[channels, 1, kernel]` and the RMS norm scales as an offset
        /// from 1.
        @Test func loaderConvertsARawHuggingFaceExport() throws {
            let reference = try Self.makeModel(seed: 5)
            let offsetNorms = [
                ".input_layernorm.weight", ".post_attention_layernorm.weight", "model.norm.weight",
                ".q_norm.weight", ".k_norm.weight",
            ]
            var checkpoint: [String: MLXArray] = [:]
            for (key, var value) in SyntheticModel.flatParameters(reference) {
                var name = key.replacingOccurrences(
                    of: "language_model.model.", with: "model.language_model.")
                name = name.replacingOccurrences(of: "language_model.lm_head.", with: "lm_head.")
                name = name.replacingOccurrences(
                    of: ".switch_mlp.gate_up_proj.weight", with: ".experts.gate_up_proj")
                name = name.replacingOccurrences(
                    of: ".switch_mlp.down_proj.weight", with: ".experts.down_proj")
                if name.hasSuffix("conv1d.weight") {
                    value = value.movedAxis(source: 1, destination: 2)
                } else if offsetNorms.contains(where: { name.hasSuffix($0) }) {
                    value = value - 1
                }
                checkpoint[name] = value
            }
            #expect(checkpoint["model.language_model.layers.2.mlp.experts.down_proj"] != nil)
            #expect(
                checkpoint["model.language_model.layers.0.linear_attn.conv1d.weight"]?.shape
                    == [192, 1, 4])
            // The norm scales go through w - 1 + 1 in float32, which can
            // round the last bit.
            #expect(try Self.loadedDifference(checkpoint, reference: reference) <= 1e-5)
        }

        @Test func loaderRejectsAMissingHalfOfASplitPair() throws {
            let reference = try Self.makeModel(seed: 5)
            var checkpoint = SyntheticModel.flatParameters(reference)
            let key = "language_model.model.layers.1.mlp.switch_mlp.gate_up_proj.weight"
            let value = checkpoint.removeValue(forKey: key)!
            checkpoint[key.replacingOccurrences(of: "gate_up_proj", with: "gate_proj")] =
                split(value, parts: 2, axis: 1)[0]
            #expect(throws: (any Error).self) {
                _ = try Self.loadedDifference(checkpoint, reference: reference)
            }
        }
    }
}
