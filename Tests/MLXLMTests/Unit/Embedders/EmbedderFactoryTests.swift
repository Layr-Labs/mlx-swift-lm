import Foundation
import MLXLMCommon
import Testing

@testable import MLXEmbedders

extension UnitTests {

    /// Tests of the embedder registries, the factory errors that come
    /// before a model exists, the pooling loader and `EmbedderModelContainer`
    /// with a stub model. A load that builds a model makes MLX arrays and is
    /// in `Kernel/Common/EmbedderModelFactoryLoadTests.swift`.
    @Suite
    struct EmbedderFactoryTests {

        // MARK: - Registries

        @Test func typeRegistryKnowsEachEmbedderType() async {
            for type in [
                "bert", "roberta", "xlm-roberta", "distilbert", "nomic_bert", "qwen3", "gemma3",
                "gemma3_text", "gemma3n",
            ] {
                #expect(await EmbedderTypeRegistry.shared.contains(type), "\(type)")
            }
            #expect(await !EmbedderTypeRegistry.shared.contains("llama"))
        }

        @Test func modelRegistryKnowsTheDefaultModels() {
            let configurations = [
                EmbedderRegistry.bge_micro, EmbedderRegistry.gte_tiny, EmbedderRegistry.minilm_l6,
                EmbedderRegistry.snowflake_xs, EmbedderRegistry.minilm_l12,
                EmbedderRegistry.bge_small, EmbedderRegistry.multilingual_e5_small,
                EmbedderRegistry.bge_base, EmbedderRegistry.nomic_text_v1,
                EmbedderRegistry.nomic_text_v1_5, EmbedderRegistry.bge_large,
                EmbedderRegistry.snowflake_lg, EmbedderRegistry.bge_m3,
                EmbedderRegistry.mixedbread_large, EmbedderRegistry.qwen3_embedding,
            ]
            #expect(Set(configurations.map(\.name)).count == 15)
            for configuration in configurations {
                #expect(EmbedderModelFactory.shared.contains(id: configuration.name))
                #expect(
                    EmbedderModelFactory.shared.configuration(id: configuration.name)
                        == configuration)
            }
            #expect(EmbedderRegistry.minilm_l6.name == "sentence-transformers/all-MiniLM-L6-v2")
            #expect(!EmbedderModelFactory.shared.contains(id: "org/unknown-embedder"))
        }

        @Test func customFactoryUsesItsRegistries() async {
            let registry = AbstractModelRegistry(modelConfigurations: [
                ModelConfiguration(id: "org/custom")
            ])
            let factory = EmbedderModelFactory(
                typeRegistry: ModelTypeRegistry(), modelRegistry: registry)
            #expect(factory.contains(id: "org/custom"))
            #expect(!factory.contains(id: EmbedderRegistry.bge_m3.name))
            #expect(await !factory.typeRegistry.contains("bert"))
        }

        // MARK: - Load errors before a model exists

        /// Writes `config` as `config.json` in a new folder (or no file when
        /// `config` is nil), loads it with the shared factory and returns
        /// the error.
        private func loadError(config: String?) async throws -> ModelFactoryError? {
            let folder = try UnitTests.temporaryFolder("embedder-error")
            defer { try? FileManager.default.removeItem(at: folder) }
            if let config {
                try Data(config.utf8).write(to: folder.appendingPathComponent("config.json"))
            }
            let loader = RecordingTokenizerLoader()
            do {
                _ = try await EmbedderModelFactory.shared.loadContainer(
                    from: folder, using: loader)
                Issue.record("expected a load error")
                return nil
            } catch let error as ModelFactoryError {
                // The errors come before the tokenizer load starts.
                #expect(await loader.folders.isEmpty)
                return error
            }
        }

        @Test func missingConfigurationFile() async throws {
            let error = try await loadError(config: nil)
            guard case .configurationFileError(let file, _, _) = error else {
                Issue.record("unexpected error \(String(describing: error))")
                return
            }
            #expect(file == "config.json")
        }

        @Test func configurationThatIsNotJSON() async throws {
            let error = try await loadError(config: "{ not json")
            guard case .configurationDecodingError(let file, _, let decodingError) = error,
                case .dataCorrupted = decodingError
            else {
                Issue.record("unexpected error \(String(describing: error))")
                return
            }
            #expect(file == "config.json")
        }

        @Test func configurationWithoutModelType() async throws {
            let error = try await loadError(config: #"{"hidden_size": 8}"#)
            guard case .configurationDecodingError(_, _, let decodingError) = error,
                case .keyNotFound(let key, _) = decodingError
            else {
                Issue.record("unexpected error \(String(describing: error))")
                return
            }
            #expect(key.stringValue == "model_type")
        }

        @Test func unsupportedModelType() async throws {
            let error = try await loadError(config: #"{"model_type": "llama"}"#)
            guard case .unsupportedModelType(let type) = error else {
                Issue.record("unexpected error \(String(describing: error))")
                return
            }
            #expect(type == "llama")
        }

        /// A wrong value type in the model configuration fails in the type
        /// registry. The factory reports it as a decoding error of
        /// `config.json`.
        @Test func modelConfigurationWithAWrongValueType() async throws {
            let error = try await loadError(
                config: #"{"model_type": "bert", "hidden_size": "large"}"#)
            guard case .configurationDecodingError(let file, _, let decodingError) = error,
                case .typeMismatch = decodingError
            else {
                Issue.record("unexpected error \(String(describing: error))")
                return
            }
            #expect(file == "config.json")
        }

        // MARK: - Pooling loader

        private func writePoolingConfiguration(_ json: String, in folder: URL) throws {
            let poolingFolder = folder.appendingPathComponent("1_Pooling", isDirectory: true)
            try FileManager.default.createDirectory(
                at: poolingFolder, withIntermediateDirectories: true)
            try Data(json.utf8).write(to: poolingFolder.appendingPathComponent("config.json"))
        }

        private func poolingConfiguration(
            dimension: Int = 16, cls: Bool = false, mean: Bool = false, max: Bool = false,
            last: Bool = false
        ) -> String {
            """
            {"word_embedding_dimension": \(dimension), "pooling_mode_cls_token": \(cls),
             "pooling_mode_mean_tokens": \(mean), "pooling_mode_max_tokens": \(max),
             "pooling_mode_lasttoken": \(last)}
            """
        }

        @Test func poolingConfigurationWinsOverTheModelStrategy() throws {
            let folder = try UnitTests.temporaryFolder("pooling")
            defer { try? FileManager.default.removeItem(at: folder) }
            try writePoolingConfiguration(
                poolingConfiguration(dimension: 24, mean: true), in: folder)

            let pooling = loadPooling(
                modelDirectory: folder, model: StubEmbeddingModel(strategy: .last))
            #expect(pooling.strategy == .mean)
            #expect(pooling.dimension == 24)
        }

        @Test func modelStrategyIsTheFallback() throws {
            let folder = try UnitTests.temporaryFolder("pooling")
            defer { try? FileManager.default.removeItem(at: folder) }

            let fromModel = loadPooling(
                modelDirectory: folder, model: StubEmbeddingModel(strategy: .last))
            #expect(fromModel.strategy == .last)
            #expect(fromModel.dimension == nil)

            let none = loadPooling(modelDirectory: folder, model: StubEmbeddingModel())
            #expect(none.strategy == .none)

            // A pooling file that does not decode is ignored.
            try writePoolingConfiguration(#"{"word_embedding_dimension": "x"}"#, in: folder)
            let invalid = loadPooling(
                modelDirectory: folder, model: StubEmbeddingModel(strategy: .cls))
            #expect(invalid.strategy == .cls)
        }

        /// `Pooling(config:)` takes the first set flag in the order CLS,
        /// mean, max, last. With no flag it uses the first token.
        @Test func poolingConfigurationPriority() throws {
            func strategy(_ json: String) throws -> Pooling.Strategy {
                let configuration = try JSONDecoder().decode(
                    PoolingConfiguration.self, from: Data(json.utf8))
                return Pooling(config: configuration).strategy
            }
            #expect(
                try strategy(poolingConfiguration(cls: true, mean: true, max: true, last: true))
                    == .cls)
            #expect(try strategy(poolingConfiguration(mean: true, max: true, last: true)) == .mean)
            #expect(try strategy(poolingConfiguration(max: true, last: true)) == .max)
            #expect(try strategy(poolingConfiguration(last: true)) == .last)
            #expect(try strategy(poolingConfiguration()) == .first)
        }

        // MARK: - Container

        private func makeContainer(
            configuration: ModelConfiguration = ModelConfiguration(id: "test/embedder"),
            strategy: Pooling.Strategy = .mean
        ) -> EmbedderModelContainer {
            EmbedderModelContainer(
                context: EmbedderModelContext(
                    configuration: configuration, model: StubEmbeddingModel(),
                    tokenizer: ScalarTokenizer(), pooling: Pooling(strategy: strategy)))
        }

        @Test func containerAccessorsReturnTheContextValues() async throws {
            let container = makeContainer(strategy: .max)
            #expect(await container.configuration.name == "test/embedder")
            #expect(await container.tokenizer.encode(text: "a") == [1, 97])
            #expect(await container.poolingStrategy == .max)

            let vocabulary = await container.perform { context in context.model.vocabularySize }
            #expect(vocabulary == 128)

            final class NonSendableBox {
                let text = "bc"
            }
            let tokens = await container.perform(nonSendable: NonSendableBox()) { context, box in
                context.tokenizer.encode(text: box.text, addSpecialTokens: false)
            }
            #expect(tokens == [98, 99])
        }

        /// The model, tokenizer and pooling form of `perform` is deprecated.
        /// The test keeps it covered until it is removed.
        @Test func deprecatedPerformFormStillWorks() async {
            let container = makeContainer(strategy: .first)
            let result = await container.perform {
                (model: any EmbeddingModel, tokenizer: any MLXLMCommon.Tokenizer, pooling: Pooling)
                in
                "\(model.vocabularySize) \(tokenizer.eosTokenId ?? -1) \(pooling.strategy)"
            }
            #expect(result == "128 2 first")
        }

        @Test func containerUpdateChangesTheContext() async {
            let container = makeContainer()
            await container.update { context in
                context.configuration.defaultPrompt = "query: "
            }
            #expect(await container.configuration.defaultPrompt == "query: ")
        }

        @Test func containerDirectories() async throws {
            let folder = URL(fileURLWithPath: "/models/embedder")
            let tokenizer = URL(fileURLWithPath: "/tokenizers/embedder")
            let local = makeContainer(
                configuration: ModelConfiguration(
                    directory: folder, tokenizerSource: .directory(tokenizer)))
            #expect(try await local.modelDirectory == folder)
            #expect(try await local.tokenizerDirectory == tokenizer)

            let remote = makeContainer()
            await #expect(
                throws: ModelConfiguration.DirectoryError.unresolvedModelDirectory("test/embedder")
            ) {
                _ = try await remote.modelDirectory
            }
            await #expect(
                throws: ModelConfiguration.DirectoryError.unresolvedModelDirectory("test/embedder")
            ) {
                _ = try await remote.tokenizerDirectory
            }
        }
    }
}
