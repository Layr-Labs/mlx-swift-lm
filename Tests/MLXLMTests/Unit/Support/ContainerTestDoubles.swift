import Foundation
import MLX
import MLXLMCommon
import MLXNN

@testable import MLXEmbedders

// Test doubles for the unit tests of the model containers and the embedder
// factory. They make no MLX arrays: a unit test runs before the Metal
// library exists.
extension UnitTests {

    /// A tokenizer that maps each Unicode scalar to its value. With
    /// `addSpecialTokens` it puts the BOS token 1 first. The ids below 32
    /// are special tokens: `<s>` is 1 and `</s>` is 2.
    struct ScalarTokenizer: MLXLMCommon.Tokenizer {
        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            (addSpecialTokens ? [1] : []) + text.unicodeScalars.map { Int($0.value) }
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map { id in
                if id < 32 {
                    return skipSpecialTokens ? "" : "<\(id)>"
                }
                return UnicodeScalar(UInt32(id)).map { String(Character($0)) } ?? "?"
            }.joined()
        }

        func convertTokenToId(_ token: String) -> Int? {
            if token == "<s>" { return 1 }
            if token == "</s>" { return 2 }
            let scalars = Array(token.unicodeScalars)
            return scalars.count == 1 ? Int(scalars[0].value) : nil
        }

        func convertIdToToken(_ id: Int) -> String? {
            decode(tokenIds: [id], skipSpecialTokens: false)
        }

        var bosToken: String? { "<s>" }
        var eosToken: String? { "</s>" }
        var unknownToken: String? { nil }

        /// Renders each message as `role:content`, joins them with `|`,
        /// encodes the text without special tokens and appends the number
        /// of tools.
        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            let text = messages.map { message in
                "\(message["role"] as? String ?? "?"):\(message["content"] as? String ?? "")"
            }.joined(separator: "|")
            return encode(text: text, addSpecialTokens: false) + [tools?.count ?? 0]
        }
    }

    /// A tokenizer loader that returns a ``ScalarTokenizer`` and records the
    /// folder that it gets.
    actor RecordingTokenizerLoader: TokenizerLoader {
        private(set) var folders: [URL] = []

        func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
            folders.append(directory)
            return ScalarTokenizer()
        }
    }

    struct StubModelError: Error {}

    /// A language model with no parameters. The container tests never run
    /// it, so its methods throw or stop the test.
    final class StubLanguageModel: Module, LanguageModel {
        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
            -> PrepareResult
        {
            throw StubModelError()
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
    }

    /// A stub model that the generic token iterator must refuse.
    final class NativeOnlyStubModel: Module, LanguageModel, GenericGenerationValidating {
        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
            -> PrepareResult
        {
            throw StubModelError()
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }

        func validateGenericGeneration() throws {
            throw GenericGenerationError.nativeCBv2Required(modelType: "stub_native")
        }
    }

    /// An embedding model with no parameters and a fixed pooling strategy.
    final class StubEmbeddingModel: Module, EmbeddingModel {
        let vocabularySize = 128
        let strategy: Pooling.Strategy?

        init(strategy: Pooling.Strategy? = nil) {
            self.strategy = strategy
        }

        var poolingStrategy: Pooling.Strategy? { strategy }

        func callAsFunction(
            _ inputs: MLXArray, positionIds: MLXArray?, tokenTypeIds: MLXArray?,
            attentionMask: MLXArray?
        ) -> EmbeddingModelOutput {
            EmbeddingModelOutput(hiddenStates: nil, pooledOutput: nil)
        }
    }

    /// A new empty temporary folder. The caller deletes it.
    static func temporaryFolder(_ name: String) throws -> URL {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(name)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder
    }
}
