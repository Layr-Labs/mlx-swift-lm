import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon
import MLXNN

// Test doubles for the unit tests of the server engines. They make no MLX
// arrays: a unit test runs before the Metal library exists.
extension UnitTests {

    /// Records the values that a test double gets.
    final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private var _messages: [[String: any Sendable]] = []
        private var _tools: [[String: any Sendable]]?
        private var _input: String?
        private var _inputToolNames: [String]?

        private func locked<T>(_ body: () -> T) -> T {
            lock.lock()
            defer { lock.unlock() }
            return body()
        }

        /// The messages that the tokenizer template got.
        var messages: [[String: any Sendable]] { locked { _messages } }
        /// The tools that the tokenizer template got.
        var tools: [[String: any Sendable]]? { locked { _tools } }
        /// The prompt kind and size that the processor got, for example
        /// `chat 2` or `messages 3`.
        var input: String? { locked { _input } }
        /// The tool names of the input that the processor got.
        var inputToolNames: [String]? { locked { _inputToolNames } }

        func record(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?) {
            locked {
                _messages = messages
                _tools = tools
            }
        }

        func record(input: String, toolNames: [String]?) {
            locked {
                _input = input
                _inputToolNames = toolNames
            }
        }
    }

    /// A tokenizer that maps each Unicode scalar to its value. With
    /// `addSpecialTokens` it puts the BOS token 1 first. The ids below 32
    /// are special tokens: `<s>` is 1 and `</s>` is 2.
    struct ScalarTokenizer: MLXLMCommon.Tokenizer {
        var recorder: Recorder?

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
            recorder?.record(messages: messages, tools: tools)
            let text = messages.map { message in
                "\(message["role"] as? String ?? "?"):\(message["content"] as? String ?? "")"
            }.joined(separator: "|")
            return encode(text: text, addSpecialTokens: false) + [tools?.count ?? 0]
        }
    }

    struct ProcessorReached: Error {}

    /// A processor that records the prompt kind and then throws, so that a
    /// unit test can see that a request passed the engine checks without
    /// making MLX arrays.
    struct RecordingProcessor: UserInputProcessor {
        let recorder: Recorder

        func prepare(input: UserInput) async throws -> LMInput {
            let toolNames = input.tools?.map {
                ($0["function"] as? [String: any Sendable])?["name"] as? String ?? "?"
            }
            let kind: String
            switch input.prompt {
            case .text: kind = "text"
            case .messages(let messages): kind = "messages \(messages.count)"
            case .chat(let messages): kind = "chat \(messages.count)"
            }
            recorder.record(input: kind, toolNames: toolNames)
            throw ProcessorReached()
        }
    }

    struct StubModelError: Error {}

    /// A language model with no parameters. The engine unit tests never run
    /// it.
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

    /// An embedding model with no parameters. The unit tests never run it.
    final class StubEmbeddingModel: Module, EmbeddingModel {
        let vocabularySize = 8

        func callAsFunction(
            _ inputs: MLXArray, positionIds: MLXArray?, tokenTypeIds: MLXArray?,
            attentionMask: MLXArray?
        ) -> EmbeddingModelOutput {
            fatalError("the unit tests do not run the embedding model")
        }
    }
}
