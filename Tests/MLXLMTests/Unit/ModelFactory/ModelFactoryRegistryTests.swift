import Foundation
import MLXLLM
import MLXLMCommon
import MLXVLM
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

        private func isMissingConfiguration(_ error: any Error, name expectedName: String) -> Bool {
            guard case ModelFactoryError.configurationFileError(let file, let name, _) = error
            else { return false }
            return file == "config.json" && name == expectedName
        }

        private func isMissingConfiguration(_ error: any Error, directory: URL) -> Bool {
            isMissingConfiguration(error, name: ModelConfiguration(directory: directory).name)
        }

        @Test
        func registryFindsTheLinkedFactories() {
            // The test target links MLXLLM and MLXVLM, so both trampolines
            // give a factory.
            let factories = ModelFactoryRegistry.shared.modelFactories()
            #expect(factories.contains { $0 is LLMModelFactory })
            #expect(factories.contains { $0 is VLMModelFactory })
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

            await #expect {
                _ = try await loadModel(
                    from: downloader, using: UnusedTokenizerLoader(), id: "org/absent")
            } throws: { error in
                isMissingConfiguration(error, name: "org/absent")
            }
            await #expect {
                _ = try await loadModelContainer(
                    from: downloader, using: UnusedTokenizerLoader(),
                    configuration: ModelConfiguration(id: "org/absent"))
            } throws: { error in
                isMissingConfiguration(error, name: "org/absent")
            }
            // Each call tries the two registered factories (MLXVLM, then
            // MLXLLM), and each factory downloads the model once: 2 x 2.
            let expected = RecordingDownloader.Request(
                id: "org/absent", revision: "main",
                patterns: ["*.safetensors", "*.json", "*.jinja"], useLatest: false)
            #expect(await downloader.requests == Array(repeating: expected, count: 4))
        }
    }
}
