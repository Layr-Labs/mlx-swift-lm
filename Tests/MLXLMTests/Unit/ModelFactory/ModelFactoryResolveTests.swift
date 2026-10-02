import Foundation
import MLXLMCommon
import Testing

extension UnitTests {
    @Suite
    struct ModelFactoryResolveTests {

        @Test
        func resolveDownloadsTheModelAndUsesItForTheTokenizer() async throws {
            let downloader = RecordingDownloader()
            let configuration = ModelConfiguration(
                id: "org/model", revision: "v2", defaultPrompt: "hi",
                extraEOSTokens: ["<end>"], toolCallFormat: .json)

            let resolved = try await resolve(
                configuration: configuration, from: downloader, useLatest: true,
                progressHandler: { _ in })

            #expect(resolved.modelDirectory == URL(filePath: "/unit-test-downloads/org/model"))
            #expect(resolved.tokenizerDirectory == resolved.modelDirectory)
            #expect(resolved.name == "org/model")
            #expect(resolved.defaultPrompt == "hi")
            #expect(resolved.extraEOSTokens == ["<end>"])
            #expect(resolved.toolCallFormat == .json)
            let requests = await downloader.requests
            #expect(
                requests == [
                    .init(
                        id: "org/model", revision: "v2",
                        patterns: ["*.safetensors", "*.json", "*.jinja"], useLatest: true)
                ])
        }

        @Test
        func resolveDownloadsOnlyTokenizerFilesForATokenizerID() async throws {
            let downloader = RecordingDownloader()
            let configuration = ModelConfiguration(
                id: "org/model", tokenizerSource: .id("org/tokenizer", revision: "r1"))

            let resolved = try await resolve(
                configuration: configuration, from: downloader, useLatest: false,
                progressHandler: { _ in })

            #expect(
                resolved.tokenizerDirectory == URL(filePath: "/unit-test-downloads/org/tokenizer"))
            let requests = await downloader.requests
            #expect(requests.count == 2)
            #expect(
                requests.last
                    == .init(
                        id: "org/tokenizer", revision: "r1", patterns: ["*.json", "*.jinja"],
                        useLatest: false))
        }

        @Test
        func resolveDoesNotDownloadLocalDirectories() async throws {
            let downloader = RecordingDownloader()
            let model = URL(filePath: "/models/family/local-model")
            let tokenizer = URL(filePath: "/tokenizers/local")
            let configuration = ModelConfiguration(
                directory: model, tokenizerSource: .directory(tokenizer), eosTokenIds: [2, 7])

            let resolved = try await resolve(
                configuration: configuration, from: downloader, useLatest: false,
                progressHandler: { _ in })

            #expect(resolved.modelDirectory == model)
            #expect(resolved.tokenizerDirectory == tokenizer)
            #expect(resolved.name == "family/local-model")
            #expect(resolved.eosTokenIds == [2, 7])
            #expect(await downloader.requests.isEmpty)
        }

        @Test
        func factoryLoadResolvesBeforeItLoads() async throws {
            let factory = EchoModelFactory()
            let downloader = RecordingDownloader()

            let context = try await factory.load(
                from: downloader, using: UnusedTokenizerLoader(),
                configuration: ModelConfiguration(id: "org/echo"))
            let container = try await factory.loadContainer(
                from: downloader, using: UnusedTokenizerLoader(),
                configuration: ModelConfiguration(id: "org/echo"))

            #expect(context.modelDirectory == URL(filePath: "/unit-test-downloads/org/echo"))
            #expect(container == "wrapped:org/echo")
            #expect(await downloader.requests.count == 2)
        }

        @Test
        func factoryLoadFromADirectoryUsesItForModelAndTokenizer() async throws {
            let factory = EchoModelFactory()
            let directory = URL(filePath: "/models/family/dir-model")

            let context = try await factory.load(from: directory, using: UnusedTokenizerLoader())
            let container = try await factory.loadContainer(
                from: directory, using: UnusedTokenizerLoader())

            #expect(context.modelDirectory == directory)
            #expect(context.tokenizerDirectory == directory)
            #expect(context.name == "family/dir-model")
            #expect(context.defaultPrompt.isEmpty)
            #expect(container == "wrapped:family/dir-model")
        }

        @Test
        func factoryConfigurationLooksUpTheRegistry() {
            let registered = ModelConfiguration(id: "org/known", defaultPrompt: "registered")
            let factory = EchoModelFactory(
                modelRegistry: AbstractModelRegistry(modelConfigurations: [registered]))

            #expect(factory.contains(id: "org/known"))
            #expect(factory.configuration(id: "org/known").defaultPrompt == "registered")
            #expect(!factory.contains(id: "org/unknown"))
            let unknown = factory.configuration(id: "org/unknown")
            #expect(unknown.id == .id("org/unknown", revision: "main"))
            #expect(unknown.defaultPrompt.isEmpty)
        }
    }
}
