import Foundation
import MLXLMCommon

// Test doubles for the unit tests of model and adapter loading.
extension UnitTests {
    /// A downloader that records each request and returns a folder named after
    /// the requested id. It does not use the network.
    actor RecordingDownloader: Downloader {
        struct Request: Equatable {
            let id: String
            let revision: String?
            let patterns: [String]
            let useLatest: Bool
        }

        private(set) var requests: [Request] = []
        private let root: URL

        init(root: URL = URL(filePath: "/unit-test-downloads")) {
            self.root = root
        }

        func download(
            id: String,
            revision: String?,
            matching patterns: [String],
            useLatest: Bool,
            progressHandler: @Sendable @escaping (Progress) -> Void
        ) async throws -> URL {
            requests.append(
                Request(id: id, revision: revision, patterns: patterns, useLatest: useLatest))
            return root.appending(path: id)
        }
    }

    /// A tokenizer loader that must not be called.
    struct UnusedTokenizerLoader: TokenizerLoader {
        struct Called: Error {}

        func load(from directory: URL) async throws -> any Tokenizer {
            throw Called()
        }
    }

    /// A model factory whose context is the resolved configuration that it
    /// receives. It loads no model.
    struct EchoModelFactory: GenericModelFactory {
        let modelRegistry: AbstractModelRegistry

        init(modelRegistry: AbstractModelRegistry = AbstractModelRegistry()) {
            self.modelRegistry = modelRegistry
        }

        func _load(
            configuration: ResolvedModelConfiguration,
            tokenizerLoader: any TokenizerLoader
        ) async throws -> ResolvedModelConfiguration {
            configuration
        }

        func _wrap(_ context: ResolvedModelConfiguration) -> String {
            "wrapped:\(context.name)"
        }
    }
}
