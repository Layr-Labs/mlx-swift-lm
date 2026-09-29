import Foundation
import MLXLMCommon
import Testing

extension UnitTests {
    @Suite
    struct ModelAdapterFactoryTests {

        /// An adapter that only remembers the folder that it was made from.
        private struct MarkerAdapter: ModelAdapter {
            let directory: URL
            func load(into model: LanguageModel) throws {}
            func fuse(with model: LanguageModel) throws {}
            func unload(from model: LanguageModel) {}
        }

        private struct CreatorFailure: Error, Equatable {}

        /// Makes a new folder with `adapter_config.json`, or with no file when
        /// `configuration` is nil.
        private func adapterDirectory(configuration: String?) throws -> URL {
            let directory = FileManager.default.temporaryDirectory
                .appending(path: "adapter-factory-\(UUID().uuidString)")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            if let configuration {
                try Data(configuration.utf8).write(
                    to: directory.appending(component: "adapter_config.json"))
            }
            return directory
        }

        private func registryWithMarker() -> ModelAdapterTypeRegistry {
            ModelAdapterTypeRegistry(creators: ["marker": { MarkerAdapter(directory: $0) }])
        }

        @Test
        func registryCreatesARegisteredType() throws {
            let registry = ModelAdapterTypeRegistry()
            registry.registerAdapterType("marker") { MarkerAdapter(directory: $0) }
            let directory = URL(filePath: "/adapters/a")

            let adapter = try registry.createAdapter(directory: directory, adapterType: "marker")

            #expect((adapter as? MarkerAdapter)?.directory == directory)
        }

        @Test
        func registryRejectsAnUnknownType() {
            let registry = registryWithMarker()
            #expect {
                try registry.createAdapter(directory: URL(filePath: "/a"), adapterType: "qlora")
            } throws: { error in
                guard case ModelAdapterError.unsupportedAdapterType(let type) = error else {
                    return false
                }
                return type == "qlora"
            }
        }

        @Test
        func registryPassesOnTheCreatorError() {
            let registry = ModelAdapterTypeRegistry(creators: [
                "broken": { _ in
                    throw CreatorFailure()
                }
            ])
            #expect(throws: CreatorFailure()) {
                try registry.createAdapter(directory: URL(filePath: "/a"), adapterType: "broken")
            }
        }

        @Test
        func registerReplacesAnEarlierCreator() throws {
            let registry = ModelAdapterTypeRegistry(creators: [
                "marker": { _ in
                    throw CreatorFailure()
                }
            ])
            registry.registerAdapterType("marker") { MarkerAdapter(directory: $0) }
            let adapter = try registry.createAdapter(
                directory: URL(filePath: "/b"), adapterType: "marker")
            #expect(adapter is MarkerAdapter)
        }

        @Test
        func factoryLoadsTheTypeNamedInTheConfiguration() async throws {
            let directory = try adapterDirectory(
                configuration: #"{"fine_tune_type": "marker", "ignored": true}"#)
            defer { try? FileManager.default.removeItem(at: directory) }
            let factory = ModelAdapterFactory(registry: registryWithMarker())
            let downloader = RecordingDownloader()

            let adapter = try await factory.load(
                from: downloader, configuration: ModelConfiguration(directory: directory))

            #expect((adapter as? MarkerAdapter)?.directory == directory)
            #expect(await downloader.requests.isEmpty)
        }

        @Test
        func factoryDownloadsAnAdapterGivenByID() async throws {
            let root = try adapterDirectory(configuration: nil)
            defer { try? FileManager.default.removeItem(at: root) }
            let directory = root.appending(path: "org/adapter")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            try Data(#"{"fine_tune_type": "marker"}"#.utf8).write(
                to: directory.appending(component: "adapter_config.json"))
            let factory = ModelAdapterFactory(registry: registryWithMarker())
            let downloader = RecordingDownloader(root: root)

            let adapter = try await factory.load(
                from: downloader,
                configuration: ModelConfiguration(id: "org/adapter", revision: "r3"),
                useLatest: true)

            #expect((adapter as? MarkerAdapter)?.directory == directory)
            #expect(
                await downloader.requests == [
                    .init(
                        id: "org/adapter", revision: "r3",
                        patterns: ["*.safetensors", "*.json", "*.jinja"], useLatest: true)
                ])
        }

        @Test
        func factoryFailsWithoutAConfigurationFile() async throws {
            let directory = try adapterDirectory(configuration: nil)
            defer { try? FileManager.default.removeItem(at: directory) }
            let factory = ModelAdapterFactory(registry: registryWithMarker())

            await #expect(throws: CocoaError.self) {
                try await factory.load(
                    from: RecordingDownloader(),
                    configuration: ModelConfiguration(directory: directory))
            }
        }

        @Test
        func factoryFailsWithoutAFineTuneType() async throws {
            let directory = try adapterDirectory(configuration: #"{"rank": 8}"#)
            defer { try? FileManager.default.removeItem(at: directory) }
            let factory = ModelAdapterFactory(registry: registryWithMarker())

            await #expect(throws: DecodingError.self) {
                try await factory.load(
                    from: RecordingDownloader(),
                    configuration: ModelConfiguration(directory: directory))
            }
        }

        @Test
        func factoryRejectsAnUnregisteredType() async throws {
            let directory = try adapterDirectory(configuration: #"{"fine_tune_type": "ia3"}"#)
            defer { try? FileManager.default.removeItem(at: directory) }
            let factory = ModelAdapterFactory(registry: registryWithMarker())

            await #expect {
                try await factory.load(
                    from: RecordingDownloader(),
                    configuration: ModelConfiguration(directory: directory))
            } throws: { error in
                guard case ModelAdapterError.unsupportedAdapterType(let type) = error else {
                    return false
                }
                return type == "ia3"
            }
        }
    }
}
