import Foundation
import MLXLMCommon
import Testing

extension UnitTests {
    @Suite
    struct ModelFactoryRegistryTests {

        private func emptyDirectory() throws -> URL {
            let directory = FileManager.default.temporaryDirectory
                .appending(path: "model-factory-\(UUID().uuidString)")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            return directory
        }

        private func isMissingConfiguration(_ error: any Error, directory: URL) -> Bool {
            guard case ModelFactoryError.configurationFileError(let file, let name, _) = error
            else { return false }
            return file == "config.json"
                && name == ModelConfiguration(directory: directory).name
        }

        @Test
        func registryFindsTheLinkedFactories() {
            // The test target links MLXLLM and MLXVLM, so both trampolines
            // give a factory.
            #expect(ModelFactoryRegistry.shared.modelFactories().count >= 2)
        }

        @Test
        func loadModelFromADirectoryWithoutConfigurationFails() async throws {
            let directory = try emptyDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }

            await #expect {
                _ = try await loadModel(from: directory, using: UnusedTokenizerLoader())
            } throws: { error in
                isMissingConfiguration(error, directory: directory)
            }
        }

        @Test
        func loadModelContainerFromADirectoryWithoutConfigurationFails() async throws {
            let directory = try emptyDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }

            await #expect {
                _ = try await loadModelContainer(from: directory, using: UnusedTokenizerLoader())
            } throws: { error in
                isMissingConfiguration(error, directory: directory)
            }
        }

        @Test
        func loadModelWithADownloaderFailsOnTheMissingConfiguration() async throws {
            let root = try emptyDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let downloader = RecordingDownloader(root: root)

            await #expect(throws: ModelFactoryError.self) {
                _ = try await loadModel(
                    from: downloader, using: UnusedTokenizerLoader(), id: "org/absent")
            }
            await #expect(throws: ModelFactoryError.self) {
                _ = try await loadModelContainer(
                    from: downloader, using: UnusedTokenizerLoader(),
                    configuration: ModelConfiguration(id: "org/absent"))
            }
            let requests = await downloader.requests
            #expect(!requests.isEmpty)
            #expect(requests.allSatisfy { $0.id == "org/absent" && $0.revision == "main" })
        }
    }
}
