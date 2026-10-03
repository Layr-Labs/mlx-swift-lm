import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXEmbedders

/// A downloader that the tests must not call: local folders need no
/// download.
private struct NoDownloader: Downloader {
    struct Called: Error {}

    func download(
        id: String, revision: String?, matching patterns: [String], useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        throw Called()
    }
}

extension KernelTests {

    /// Tests of `EmbedderModelFactory` with a tiny BERT checkpoint that the
    /// test writes to a temporary folder: hidden size 32, 2 layers,
    /// vocabulary 64.
    @Suite
    struct EmbedderModelFactoryLoadTests {

        // The loaded model has the same weights as the reference model and
        // runs the same kernels on the same input, so the hidden states
        // must match. The margin allows only for float32 rounding.
        static let tolerance: Float = 1e-5

        static var configuration: [String: Any] {
            [
                "model_type": "bert", "hidden_size": 32, "num_attention_heads": 4,
                "intermediate_size": 64, "num_hidden_layers": 2, "vocab_size": 64,
                "max_position_embeddings": 32, "type_vocab_size": 2,
            ]
        }

        /// The values that a test reads inside the container.
        struct LoadedOutput: Sendable {
            let hidden: [Float]
            let embedding: [Float]
            let dimension: Int?
        }

        static let row = [3, 14, 15, 9, 26, 5]

        /// Writes `config.json`, `model.safetensors` and, when given,
        /// `1_Pooling/config.json` to a new folder. Returns the folder and
        /// the hidden states of the reference model for `row`.
        static func writeCheckpoint(pooling: String?) throws -> (URL, [Float]) {
            let reference = BertModel(
                try SyntheticModel.configuration(BertConfiguration.self, configuration))
            SyntheticModel.randomize(reference, seed: 11)
            let hidden = reference(SyntheticModel.batch([row])).hiddenStates!
            eval(hidden)

            let folder = try UnitTests.temporaryFolder("embedder-load")
            try JSONSerialization.data(withJSONObject: configuration)
                .write(to: folder.appendingPathComponent("config.json"))
            try MLX.save(
                arrays: SyntheticModel.flatParameters(reference),
                url: folder.appendingPathComponent("model.safetensors"))
            if let pooling {
                let poolingFolder = folder.appendingPathComponent("1_Pooling")
                try FileManager.default.createDirectory(
                    at: poolingFolder, withIntermediateDirectories: true)
                try Data(pooling.utf8).write(
                    to: poolingFolder.appendingPathComponent("config.json"))
            }
            return (folder, hidden.asArray(Float.self))
        }

        static func maxDifference(_ a: [Float], _ b: [Float]) -> Float {
            precondition(a.count == b.count)
            return zip(a, b).map { abs($0 - $1) }.max() ?? 0
        }

        @Test func loadContainerFromALocalFolder() async throws {
            let pooling = """
                {"word_embedding_dimension": 16, "pooling_mode_cls_token": false,
                 "pooling_mode_mean_tokens": true, "pooling_mode_max_tokens": false,
                 "pooling_mode_lasttoken": false}
                """
            let (folder, expected) = try Self.writeCheckpoint(pooling: pooling)
            defer { try? FileManager.default.removeItem(at: folder) }

            let loader = UnitTests.RecordingTokenizerLoader()
            let container = try await EmbedderModelFactory.shared.loadContainer(
                from: folder, using: loader)

            #expect(await loader.folders == [folder])
            #expect(try await container.modelDirectory == folder)
            #expect(try await container.tokenizerDirectory == folder)
            #expect(await container.configuration.tokenizerSource == nil)
            #expect(await container.poolingStrategy == .mean)

            let loaded = await container.perform { context in
                let tokens = SyntheticModel.batch([Self.row])
                let output = context.model(
                    tokens, positionIds: nil, tokenTypeIds: nil, attentionMask: nil)
                let pooled = context.pooling(output, normalize: true)
                eval(output.hiddenStates!, pooled)
                return LoadedOutput(
                    hidden: output.hiddenStates!.asArray(Float.self),
                    embedding: pooled.asArray(Float.self), dimension: context.pooling.dimension)
            }
            let (hidden, embedding, dimension) = (loaded.hidden, loaded.embedding, loaded.dimension)
            #expect(hidden.count == Self.row.count * 32)
            #expect(Self.maxDifference(hidden, expected) <= Self.tolerance)

            // The pooling file cuts the embedding to 16 values and the
            // embedding is normalized to length 1.
            #expect(dimension == 16)
            #expect(embedding.count == 16)
            let length = embedding.map { $0 * $0 }.reduce(0, +).squareRoot()
            #expect(abs(length - 1) <= 1e-5)
        }

        /// A tokenizer folder that is not the model folder becomes the
        /// tokenizer source of the loaded configuration.
        @Test func loadWithASeparateTokenizerFolder() async throws {
            let (folder, expected) = try Self.writeCheckpoint(pooling: nil)
            defer { try? FileManager.default.removeItem(at: folder) }
            let tokenizerFolder = try UnitTests.temporaryFolder("embedder-tokenizer")
            defer { try? FileManager.default.removeItem(at: tokenizerFolder) }

            let loader = UnitTests.RecordingTokenizerLoader()
            let context = try await EmbedderModelFactory.shared.load(
                from: NoDownloader(), using: loader,
                configuration: ModelConfiguration(
                    directory: folder, tokenizerSource: .directory(tokenizerFolder)))

            #expect(await loader.folders == [tokenizerFolder])
            #expect(context.configuration.tokenizerSource == .directory(tokenizerFolder))
            #expect(context.configuration.id == .directory(folder))
            // BERT has no pooling strategy of its own and there is no
            // pooling file.
            #expect(context.pooling.strategy == .none)

            let output = context.model(
                SyntheticModel.batch([Self.row]), positionIds: nil, tokenTypeIds: nil,
                attentionMask: nil)
            let hidden = output.hiddenStates!.asArray(Float.self)
            #expect(Self.maxDifference(hidden, expected) <= Self.tolerance)
        }

        /// The factory wraps a loaded context in a container.
        @Test func wrapMakesAContainer() async throws {
            let (folder, _) = try Self.writeCheckpoint(pooling: nil)
            defer { try? FileManager.default.removeItem(at: folder) }

            let context = try await EmbedderModelFactory.shared.load(
                from: folder, using: UnitTests.RecordingTokenizerLoader())
            let container = EmbedderModelFactory.shared._wrap(context)
            #expect(await container.poolingStrategy == .none)
            #expect(await container.tokenizer.eosTokenId == 2)
        }
    }
}
