import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {

    /// Forward-pass tests of `Gemma3nTextModel` with a tiny random model.
    ///
    /// The model has 4 layers: sliding, full, sliding, full. The last 2
    /// layers share the KV caches of the first 2. It has 4 AltUp streams,
    /// the LAuReL block, per-layer inputs, activation sparsity in layer 0
    /// and a final logit soft cap.
    @Suite
    struct Gemma3nTextForwardPassTests {

        static let vocabularySize = 64

        static var base: [String: Any] {
            [
                "model_type": "gemma3n_text", "hidden_size": 32, "num_hidden_layers": 4,
                "intermediate_size": 64, "num_attention_heads": 2, "head_dim": 16,
                "rms_norm_eps": 1e-6, "vocab_size": vocabularySize, "num_key_value_heads": 1,
                "num_kv_shared_layers": 2, "vocab_size_per_layer_input": vocabularySize,
                "sliding_window": 16, "max_position_embeddings": 256,
                "rope_local_base_freq": 10000, "rope_theta": 1_000_000,
                "final_logit_softcapping": 30,
                "layer_types": [
                    "sliding_attention", "full_attention", "sliding_attention", "full_attention",
                ],
                "activation_sparsity_pattern": [0.95, 0, 0, 0],
                "hidden_size_per_layer_input": 8, "altup_num_inputs": 4, "altup_coef_clip": 120,
                "altup_correct_scale": true, "altup_active_idx": 0, "laurel_rank": 4,
            ]
        }

        static func makeModel(_ overrides: [String: Any] = [:], seed: UInt64 = 1) throws
            -> Gemma3nTextModel
        {
            let configuration = try SyntheticModel.configuration(
                Gemma3nTextConfiguration.self, base, overrides: overrides)
            let model = Gemma3nTextModel(config: configuration)
            SyntheticModel.randomize(model, seed: seed)
            return model
        }

        /// The sliding window is 4, shorter than the test sequences.
        static var shortWindow: [String: Any] { ["sliding_window": 4] }

        static func row(_ seed: Int, count: Int = 11) -> [Int] {
            SyntheticModel.tokens(count: count, vocabularySize: vocabularySize, seed: seed)
        }

        // Tolerance of the float32 comparisons: the cached path and the full
        // path differ in the order of the attention sums. The differences
        // are near 1e-6 for logits of size 1 to 5.
        static let tolerance: Float = 1e-4

        /// Batch size 1 only. With 2 rows, `Gemma3nAltUp.correct` stops the
        /// process with a broadcast error (Gemma3nText.swift:479: the
        /// coefficients are transposed to `[streams, L, B]` and then
        /// broadcast against `[1, B, L, D]`). A crash cannot be a known
        /// issue, so no test here uses 2 rows.
        @Test func logitsHaveTheExpectedShapeAndAreFinite() throws {
            let model = try Self.makeModel()
            let logits = ForwardPassChecks.logits(model, [Self.row(1, count: 7)])
            #expect(logits.shape == [1, 7, Self.vocabularySize])
            #expect(logits.dtype == .float32)
            #expect(isFinite(logits).all().item(Bool.self))
            // The soft cap keeps every logit inside (-30, 30).
            #expect(SyntheticModel.maxAbs(logits) < 30)
        }

        @Test func sameSeedGivesTheSameLogits() throws {
            try ForwardPassChecks.checkDeterminism(
                make: { try Self.makeModel() }, seed: 3, vocabularySize: Self.vocabularySize)
        }

        @Test func cacheHasOneCachePerLayerThatOwnsItsKV() throws {
            let cache = try Self.makeModel().newCache(parameters: nil)
            #expect(cache.count == 2)
            #expect((cache[0] as? RotatingKVCache)?.maxSize == 16)
            #expect(type(of: cache[1]) == KVCacheSimple.self)
        }

        /// Without KV sharing, and with a sequence that fits in the sliding
        /// window of 16.
        @Test func cachedDecodeMatchesTheFullForwardPass() throws {
            let model = try Self.makeModel(["num_kv_shared_layers": 0])
            ForwardPassChecks.checkCacheConsistency(
                model, rows: [Self.row(1)], chunks: [5, 3, 1, 1, 1],
                tolerance: Self.tolerance)
        }

        /// With KV sharing, the last 2 layers read the caches of layers 0
        /// and 1. A prompt in chunks and decode steps must give the same
        /// logits as the same prompt in one chunk.
        @Test func kvSharedLayersGiveTheSameLogitsInChunks() throws {
            let model = try Self.makeModel()
            let row = Self.row(1)
            func run(_ chunks: [Int]) -> MLXArray {
                let cache = model.newCache(parameters: nil)
                var start = 0
                var parts: [MLXArray] = []
                for chunk in chunks {
                    parts.append(
                        ForwardPassChecks.logits(
                            model, [Array(row[start ..< start + chunk])], cache: cache))
                    start += chunk
                }
                return concatenated(parts, axis: 1)
            }
            let difference = SyntheticModel.maxAbsDifference(run([5, 3, 1, 1, 1]), run([11]))
            #expect(difference <= Self.tolerance, "differs by \(difference)")
        }

        @Test func aLaterTokenDoesNotChangeEarlierLogits() throws {
            ForwardPassChecks.checkCausality(
                try Self.makeModel(), row: Self.row(1), position: 6,
                vocabularySize: Self.vocabularySize, tolerance: Self.tolerance)
        }

        /// A prompt longer than the sliding window gets an array mask. The
        /// sliding layers must still mask later tokens.
        @Test func promptLongerThanTheWindowStaysCausal() throws {
            let model = try Self.makeModel(Self.shortWindow)
            ForwardPassChecks.checkCausality(
                model, row: Self.row(1), position: 6, vocabularySize: Self.vocabularySize,
                tolerance: Self.tolerance)
        }

        @Test func parameterTreeHasTheCheckpointKeysAndShapes() throws {
            let parameters = SyntheticModel.flatParameters(try Self.makeModel())
            let expected: [String: [Int]] = [
                "language_model.embed_tokens.weight": [64, 32],
                "language_model.embed_tokens_per_layer.weight": [64, 32],
                "language_model.per_layer_model_projection.weight": [32, 32],
                "language_model.per_layer_projection_norm.weight": [8],
                "language_model.altup_projections.2.weight": [32, 32],
                "language_model.altup_unembed_projections.0.weight": [32, 32],
                "language_model.layers.0.self_attn.q_proj.weight": [32, 32],
                "language_model.layers.0.self_attn.k_proj.weight": [16, 32],
                "language_model.layers.0.altup.prediction_coefs.weight": [16, 4],
                "language_model.layers.0.altup.correction_coefs.weight": [4, 4],
                "language_model.layers.0.altup.correct_output_scale": [32],
                "language_model.layers.0.laurel.linear_left.weight": [4, 32],
                "language_model.layers.3.per_layer_input_gate.weight": [8, 32],
                "language_model.layers.3.per_layer_projection.weight": [32, 8],
                "language_model.norm.weight": [32],
            ]
            for (key, shape) in expected {
                #expect(parameters[key]?.shape == shape, "\(key)")
            }
        }

        /// A Hugging Face checkpoint keys the text model as
        /// `model.language_model.*`.
        @Test func loaderRemapsTheLanguageModelPrefix() throws {
            let reference = try Self.makeModel(seed: 5)
            var checkpoint: [String: MLXArray] = [:]
            for (key, value) in SyntheticModel.flatParameters(reference) {
                checkpoint["model." + key] = value
            }
            let loaded = try Self.makeModel(seed: 6)
            try SyntheticModel.load(checkpoint, into: loaded)
            let rows = [Self.row(3)]
            #expect(
                SyntheticModel.maxAbsDifference(
                    ForwardPassChecks.logits(reference, rows),
                    ForwardPassChecks.logits(loaded, rows)) == 0)

            checkpoint["model.language_model.norm.weight"] = MLXArray.zeros([16])
            #expect(throws: (any Error).self) {
                try SyntheticModel.load(checkpoint, into: try Self.makeModel(seed: 6))
            }
        }

        /// `sanitize(weights:)` cuts an embedding table with more rows than
        /// the vocabulary down to the vocabulary.
        @Test func sanitizeCutsAPaddedEmbeddingTable() throws {
            let model = try Self.makeModel()
            let padded = MLXArray.zeros([70, 32])
            let sanitized = model.sanitize(weights: [
                "model.language_model.embed_tokens.weight": padded
            ])
            withKnownIssue(
                """
                sanitize(weights:) checks `language_model.model.embed_tokens.weight`, but \
                it maps the checkpoint key to `language_model.embed_tokens.weight` \
                (Gemma3nText.swift:1010-1011), so it never cuts the table.
                """
            ) {
                #expect(
                    sanitized["language_model.embed_tokens.weight"]?.dim(0) == 64,
                    "embedding rows")
            } matching: {
                $0.isFailedExpectation(["embedding rows"])
            }
        }
    }
}
