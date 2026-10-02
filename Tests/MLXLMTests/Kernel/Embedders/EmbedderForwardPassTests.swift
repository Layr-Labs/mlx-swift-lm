import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXEmbedders

extension KernelTests {

    /// Forward-pass tests of the embedding models with tiny random models:
    /// hidden size 32, 2 layers, vocabulary 64.
    ///
    /// The padding tests compare a padded row with the same row without
    /// padding. BERT and Nomic BERT mask padded keys, so the kept positions
    /// must not change. Qwen 3 and Gemma 3 use causal attention, so right
    /// padding cannot change the kept positions.
    @Suite
    struct EmbedderForwardPassTests {

        // Tolerance of the float32 comparisons: a padded row runs the same
        // math with masked keys, and the sums run in another order. The
        // error is near 1e-6 for values of size 1.
        static let tolerance: Float = 1e-4

        static let row = [3, 14, 15, 9, 26, 5]
        static let padding = 2

        /// The row with `padding` zero tokens on the right, and its mask.
        static var padded: (tokens: MLXArray, mask: MLXArray) {
            (
                SyntheticModel.batch([row + Array(repeating: 0, count: padding)]),
                MLXArray(
                    Array(repeating: Int32(1), count: row.count)
                        + Array(repeating: Int32(0), count: padding)
                ).reshaped(1, -1)
            )
        }

        static func load(_ weights: [String: MLXArray], into model: BaseLanguageModel) throws {
            let folder = FileManager.default.temporaryDirectory
                .appendingPathComponent("embedder-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: folder) }
            try MLX.save(arrays: weights, url: folder.appendingPathComponent("model.safetensors"))
            try loadWeights(modelDirectory: folder, model: model)
        }

        /// Renames every key with the first matching pair of `renames`.
        static func renamed(_ weights: [String: MLXArray], _ renames: [(String, String)])
            -> [String: MLXArray]
        {
            var result: [String: MLXArray] = [:]
            for (key, value) in weights {
                var name = key
                for (from, to) in renames {
                    name = name.replacingOccurrences(of: from, with: to)
                }
                result[name] = value
            }
            return result
        }

        // MARK: - BERT

        static var bert: [String: Any] {
            [
                "model_type": "bert", "hidden_size": 32, "num_attention_heads": 4,
                "intermediate_size": 64, "num_hidden_layers": 2, "vocab_size": 64,
                "max_position_embeddings": 32, "type_vocab_size": 2,
            ]
        }

        static func bertModel(
            _ overrides: [String: Any] = [:], lmHead: Bool = false, seed: UInt64 = 1
        ) throws -> BertModel {
            let configuration = try SyntheticModel.configuration(
                BertConfiguration.self, bert, overrides: overrides)
            let model =
                (overrides["model_type"] as? String) == "distilbert"
                ? DistilBertModel(configuration, lmHead: lmHead)
                : BertModel(configuration, lmHead: lmHead)
            SyntheticModel.randomize(model, seed: seed)
            return model
        }

        @Test func bertOutputsHaveTheExpectedShapes() throws {
            let model = try Self.bertModel()
            let output = model(SyntheticModel.batch([Self.row, Self.row.reversed()]))
            #expect(output.hiddenStates?.shape == [2, 6, 32])
            #expect(output.pooledOutput?.shape == [2, 32])
            #expect(isFinite(output.hiddenStates!).all().item(Bool.self))
            // The pooler ends with tanh.
            #expect(SyntheticModel.maxAbs(output.pooledOutput!) <= 1)

            let head = try Self.bertModel(lmHead: true)
            let logits = head(SyntheticModel.batch([Self.row]))
            #expect(logits.hiddenStates?.shape == [1, 6, 64])
            #expect(logits.pooledOutput == nil)
        }

        @Test func bertPaddedKeysDoNotChangeTheKeptPositions() throws {
            let model = try Self.bertModel()
            let plain = model(SyntheticModel.batch([Self.row])).hiddenStates!
            let (tokens, mask) = Self.padded
            let padded = model(tokens, attentionMask: mask).hiddenStates!
            #expect(
                SyntheticModel.maxAbsDifference(padded[0..., ..<Self.row.count], plain)
                    <= Self.tolerance)
            // Without the mask the padding changes the result.
            let unmasked = model(tokens).hiddenStates!
            #expect(
                SyntheticModel.maxAbsDifference(unmasked[0..., ..<Self.row.count], plain) > 1e-3)
        }

        @Test func bertTokenTypesChangeTheOutput() throws {
            let model = try Self.bertModel()
            let tokens = SyntheticModel.batch([Self.row])
            let zeros = model(tokens, tokenTypeIds: MLXArray.zeros(tokens.shape, dtype: .int32))
            let ones = model(tokens, tokenTypeIds: MLXArray.ones(tokens.shape, dtype: .int32))
            #expect(
                SyntheticModel.maxAbsDifference(zeros.hiddenStates!, ones.hiddenStates!) > 1e-3)
        }

        /// A prompt longer than `max_position_embeddings` is cut to it.
        @Test func bertCutsALongPrompt() throws {
            let model = try Self.bertModel(["max_position_embeddings": 4])
            let output = model(SyntheticModel.batch([Self.row]))
            #expect(output.hiddenStates?.shape == [1, 4, 32])
        }

        /// A Hugging Face BERT checkpoint: `bert.` prefix, `layer.N`,
        /// `attention.self.query`, `LayerNorm` and more names that
        /// `sanitize(weights:)` maps to the module names.
        @Test func bertLoaderConvertsTheHuggingFaceNames() throws {
            let reference = try Self.bertModel(seed: 5)
            var checkpoint = Self.renamed(
                SyntheticModel.flatParameters(reference),
                [
                    (".layers.", ".layer."), (".attention.query_proj.", ".attention.self.query."),
                    (".attention.key_proj.", ".attention.self.key."),
                    (".attention.value_proj.", ".attention.self.value."),
                    (".attention.out_proj.", ".attention.output.dense."),
                    (".ln1.", ".attention.output.LayerNorm."), (".ln2.", ".output.LayerNorm."),
                    (".linear1.", ".intermediate.dense."), (".linear2.", ".output.dense."),
                    ("embeddings.norm.", "embeddings.LayerNorm."), ("pooler.", "pooler.dense."),
                ])
            checkpoint = checkpoint.reduce(into: [:]) { $0["bert." + $1.key] = $1.value }
            checkpoint["bert.embeddings.position_ids"] = MLXArray(0 ..< 32)
            #expect(checkpoint["bert.encoder.layer.1.attention.self.query.weight"] != nil)
            #expect(checkpoint["bert.encoder.layer.0.attention.output.LayerNorm.bias"] != nil)

            let loaded = try Self.bertModel(seed: 6)
            try Self.load(checkpoint, into: loaded)
            let tokens = SyntheticModel.batch([Self.row])
            #expect(
                SyntheticModel.maxAbsDifference(
                    reference(tokens).hiddenStates!, loaded(tokens).hiddenStates!) == 0)
        }

        @Test func distilBertLoaderConvertsTheHuggingFaceNames() throws {
            let overrides: [String: Any] = [
                "model_type": "distilbert", "dim": 32, "n_heads": 4, "hidden_dim": 64,
                "n_layers": 2,
            ]
            let reference = try Self.bertModel(overrides, seed: 5)
            var checkpoint = Self.renamed(
                SyntheticModel.flatParameters(reference),
                [
                    ("encoder.layers.", "transformer.layer."), (".query_proj.", ".q_lin."),
                    (".key_proj.", ".k_lin."), (".value_proj.", ".v_lin."),
                    (".out_proj.", ".out_lin."), (".ln1.", ".sa_layer_norm."),
                    (".linear1.", ".ffn.lin1."), (".linear2.", ".ffn.lin2."),
                    (".ln2.", ".output_layer_norm."), ("embeddings.norm.", "embeddings.LayerNorm."),
                ])
            checkpoint = checkpoint.reduce(into: [:]) { $0["distilbert." + $1.key] = $1.value }
            #expect(checkpoint["distilbert.transformer.layer.1.attention.q_lin.weight"] != nil)

            let loaded = try Self.bertModel(overrides, seed: 6)
            try Self.load(checkpoint, into: loaded)
            let tokens = SyntheticModel.batch([Self.row])
            #expect(
                SyntheticModel.maxAbsDifference(
                    reference(tokens).hiddenStates!, loaded(tokens).hiddenStates!) == 0)
        }

        // MARK: - Nomic BERT

        static var nomic: [String: Any] {
            [
                "n_embd": 32, "n_head": 4, "n_inner": 64, "n_layer": 2, "vocab_size": 64,
                "rotary_emb_fraction": 1.0,
            ]
        }

        static func nomicModel(_ overrides: [String: Any] = [:], seed: UInt64 = 1) throws
            -> NomicBertModel
        {
            let model = NomicBertModel(
                try SyntheticModel.configuration(
                    NomicBertConfiguration.self, nomic, overrides: overrides))
            SyntheticModel.randomize(model, seed: seed)
            return model
        }

        @Test func nomicPaddedKeysDoNotChangeTheKeptPositions() throws {
            let model = try Self.nomicModel()
            let plain = model(SyntheticModel.batch([Self.row]))
            #expect(plain.hiddenStates?.shape == [1, 6, 32])
            #expect(plain.pooledOutput?.shape == [1, 32])
            let (tokens, mask) = Self.padded
            let padded = model(tokens, attentionMask: mask).hiddenStates!
            #expect(
                SyntheticModel.maxAbsDifference(
                    padded[0..., ..<Self.row.count], plain.hiddenStates!) <= Self.tolerance)
        }

        /// The rotary position embedding makes the output depend on the
        /// token order.
        @Test func nomicOutputDependsOnTheTokenOrder() throws {
            let model = try Self.nomicModel()
            let forward = model(SyntheticModel.batch([Self.row])).hiddenStates!
            let backward = model(SyntheticModel.batch([Self.row.reversed()])).hiddenStates!
            // Position 0 of the reversed row holds the last token.
            #expect(
                SyntheticModel.maxAbsDifference(forward[0..., -1], backward[0..., 0]) > 1e-3)
        }

        @Test func nomicLoaderConvertsTheCheckpointNames() throws {
            let reference = try Self.nomicModel(seed: 5)
            var checkpoint = Self.renamed(
                SyntheticModel.flatParameters(reference),
                [("embeddings.norm.", "emb_ln."), ("pooler.", "pooler.dense.")])
            checkpoint = checkpoint.reduce(into: [:]) { $0["bert." + $1.key] = $1.value }
            let loaded = try Self.nomicModel(seed: 6)
            try Self.load(checkpoint, into: loaded)
            let tokens = SyntheticModel.batch([Self.row])
            #expect(
                SyntheticModel.maxAbsDifference(
                    reference(tokens).pooledOutput!, loaded(tokens).pooledOutput!) == 0)
        }

        // MARK: - Qwen 3

        static var qwen3: [String: Any] {
            [
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                "num_attention_heads": 4, "rms_norm_eps": 1e-6, "vocab_size": 64,
                "num_key_value_heads": 2, "head_dim": 8,
            ]
        }

        static func qwen3Model(seed: UInt64 = 1) throws -> MLXEmbedders.Qwen3Model {
            let model = MLXEmbedders.Qwen3Model(
                try SyntheticModel.configuration(MLXEmbedders.Qwen3Configuration.self, qwen3))
            SyntheticModel.randomize(model, seed: seed)
            return model
        }

        /// Qwen 3 embeddings take the last kept token. With right padding and
        /// causal attention, the last kept hidden state of the padded row is
        /// the last hidden state of the plain row.
        @Test func qwen3LastTokenPoolingIgnoresRightPadding() throws {
            let model = try Self.qwen3Model()
            #expect(model.poolingStrategy == .last)
            let pooling = Pooling(strategy: .last)
            let plain = pooling(model(SyntheticModel.batch([Self.row])))
            let (tokens, mask) = Self.padded
            let padded = pooling(model(tokens, attentionMask: mask), mask: mask)
            #expect(plain.shape == [1, 32])
            #expect(SyntheticModel.maxAbsDifference(padded, plain) <= Self.tolerance)
        }

        @Test func qwen3LoaderAddsTheModelPrefixAndDropsTheHead() throws {
            let reference = try Self.qwen3Model(seed: 5)
            var checkpoint = SyntheticModel.flatParameters(reference).reduce(into: [:]) {
                $0[String($1.key.dropFirst("model.".count))] = $1.value
            }
            checkpoint["lm_head.weight"] = MLXArray.zeros([64, 32])
            let loaded = try Self.qwen3Model(seed: 6)
            try Self.load(checkpoint, into: loaded)
            let tokens = SyntheticModel.batch([Self.row])
            #expect(
                SyntheticModel.maxAbsDifference(
                    reference(tokens).hiddenStates!, loaded(tokens).hiddenStates!) == 0)
        }

        // MARK: - Gemma 3

        static var gemma3: [String: Any] {
            [
                "model_type": "gemma3_text", "hidden_size": 32, "num_hidden_layers": 2,
                "intermediate_size": 48, "num_attention_heads": 2, "head_dim": 16,
                "num_key_value_heads": 1, "vocab_size": 64, "sliding_window": 4,
                "sliding_window_pattern": 2, "query_pre_attn_scalar": 16,
            ]
        }

        static func gemma3Model(seed: UInt64 = 1) throws -> EmbeddingGemma {
            let model = EmbeddingGemma(
                try SyntheticModel.configuration(MLXEmbedders.Gemma3Configuration.self, gemma3))
            SyntheticModel.randomize(model, seed: seed)
            return model
        }

        @Test func gemma3PoolsTheKeptTokensToAUnitVector() throws {
            let model = try Self.gemma3Model()
            let plain = model(
                SyntheticModel.batch([Self.row]), positionIds: nil, tokenTypeIds: nil,
                attentionMask: nil)
            let embedding = plain.pooledOutput!
            #expect(embedding.shape == [1, 32])
            let norm = sqrt((embedding * embedding).sum()).item(Float.self)
            #expect(abs(norm - 1) <= 1e-5)

            // Token 0 counts as padding when no mask is given.
            let (tokens, mask) = Self.padded
            let byMask = model(tokens, positionIds: nil, tokenTypeIds: nil, attentionMask: mask)
            let byToken = model(tokens, positionIds: nil, tokenTypeIds: nil, attentionMask: nil)
            #expect(
                SyntheticModel.maxAbsDifference(byMask.pooledOutput!, embedding) <= Self.tolerance)
            #expect(
                SyntheticModel.maxAbsDifference(byToken.pooledOutput!, embedding) <= Self.tolerance)
        }

        /// A checkpoint with the Sentence Transformers `dense` layers makes
        /// `sanitize(weights:)` add them to the model; it drops the head and
        /// cuts a padded embedding table.
        @Test func gemma3LoaderAddsTheDenseLayers() throws {
            let reference = try Self.gemma3Model(seed: 5)
            var checkpoint = SyntheticModel.flatParameters(reference)
            let table = checkpoint["model.embed_tokens.weight"]!
            checkpoint["model.embed_tokens.weight"] = concatenated(
                [table, MLXArray.zeros([6, 32])], axis: 0)
            checkpoint["lm_head.weight"] = MLXArray.zeros([64, 32])
            checkpoint["dense.0.weight"] = MLXRandom.normal([48, 32], key: MLXRandom.key(8))
            checkpoint["dense.1.weight"] = MLXRandom.normal([32, 48], key: MLXRandom.key(9))

            let loaded = try Self.gemma3Model(seed: 6)
            try Self.load(checkpoint, into: loaded)
            let parameters = SyntheticModel.flatParameters(loaded)
            #expect(parameters["dense.0.weight"]?.shape == [48, 32])
            #expect(parameters["model.embed_tokens.weight"]?.shape == [64, 32])

            let tokens = SyntheticModel.batch([Self.row])
            let a = reference(tokens, positionIds: nil, tokenTypeIds: nil, attentionMask: nil)
            let b = loaded(tokens, positionIds: nil, tokenTypeIds: nil, attentionMask: nil)
            #expect(SyntheticModel.maxAbsDifference(a.hiddenStates!, b.hiddenStates!) == 0)
            // The dense layers change the pooled embedding.
            #expect(SyntheticModel.maxAbsDifference(a.pooledOutput!, b.pooledOutput!) > 1e-3)
        }

        // MARK: - Pooling

        @Test func poolingStrategiesUseTheMask() {
            // Batch 1, 3 positions, 2 features; the last position is padding.
            let hidden = MLXArray([1, 2, 3, 4, 50, 60] as [Float]).reshaped(1, 3, 2)
            let mask = MLXArray([1, 1, 0] as [Int32]).reshaped(1, 3)
            let output = EmbeddingModelOutput(hiddenStates: hidden, pooledOutput: nil)
            func pool(_ strategy: Pooling.Strategy) -> [Float] {
                Pooling(strategy: strategy)(output, mask: mask).asArray(Float.self)
            }
            #expect(pool(.mean) == [2, 3])
            #expect(pool(.max) == [3, 4])
            #expect(pool(.first) == [1, 2])
            #expect(pool(.last) == [3, 4])
            #expect(pool(.cls) == [1, 2])

            let normalized = Pooling(strategy: .first)(output, mask: mask, normalize: true)
            #expect(abs(sqrt((normalized * normalized).sum()).item(Float.self) - 1) <= 1e-5)
            let sliced = Pooling(strategy: .mean, dimension: 1)(output, mask: mask)
            #expect(sliced.asArray(Float.self) == [2])
        }
    }
}
