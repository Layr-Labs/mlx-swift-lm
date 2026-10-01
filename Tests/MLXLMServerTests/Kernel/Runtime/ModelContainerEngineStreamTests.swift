import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLMServer

/// Text of each token of the scripted model. Decoding joins the fragments.
private let fragments: [Int: String] = [
    1: "Hello",
    2: " world",
    3: "<tool_call>",
    4: #"{"name": "lookup", "arguments": {"q": "x"}}"#,
    5: "</tool_call>",
]
private let script = [1, 2, 3, 4, 5]
private let vocabularySize = 8

/// A model that gives one-hot logits for the next token of `script`. With
/// temperature 0 the sampler takes the scripted token.
private final class ScriptedModel: Module, LanguageModel {
    private var index = 0

    private func nextLogits() -> MLXArray {
        var row = [Float](repeating: 0, count: vocabularySize)
        row[script[min(index, script.count - 1)]] = 10
        index += 1
        return MLXArray(row).reshaped([1, 1, vocabularySize])
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .logits(LMOutput(logits: nextLogits()))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        nextLogits()
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
}

private struct FragmentTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        Array(repeating: 6, count: text.count)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { fragments[$0] ?? "" }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { fragments[id] }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        encode(
            text: messages.compactMap { $0["content"] as? String }.joined(),
            addSpecialTokens: false)
    }
}

/// Makes one token for each character of the message content. A chat
/// prompt and a message prompt give the same tokens.
private struct FragmentProcessor: UserInputProcessor {
    func prepare(input: UserInput) throws -> LMInput {
        let messages: [[String: any Sendable]]
        switch input.prompt {
        case .messages(let list): messages = list
        default: messages = DefaultMessageGenerator().generate(from: input)
        }
        let tokens = try FragmentTokenizer().applyChatTemplate(
            messages: messages, tools: nil, additionalContext: nil)
        return LMInput(tokens: MLXArray(tokens.map(Int32.init)))
    }
}

extension KernelTests {

    /// Tests of `MLXModelContainerEngine.streamChatCompletion(request:)`
    /// with a scripted model. The model output is exact, so the tests
    /// compare text and tool calls exactly.
    @Suite
    struct ModelContainerEngineStreamTests {

        private func makeEngine() -> MLXModelContainerEngine {
            MLXModelContainerEngine(
                modelID: "scripted",
                model: ModelContainer(
                    context: ModelContext(
                        configuration: ModelConfiguration(id: "test/scripted"),
                        model: ScriptedModel(), processor: FragmentProcessor(),
                        tokenizer: FragmentTokenizer())))
        }

        private func request(
            _ messages: [OpenAIChatMessage], tools: [OpenAITool]? = nil
        ) -> OpenAIChatCompletionRequest {
            OpenAIChatCompletionRequest(
                model: "scripted", messages: messages, tools: tools, temperature: 0,
                maxTokens: script.count)
        }

        private func collect(
            _ stream: AsyncThrowingStream<MLXServerGenerationEvent, Error>
        ) async throws -> (text: String, calls: [ToolCall], infos: [ServerGenerationInfo]) {
            var text = ""
            var calls: [ToolCall] = []
            var infos: [ServerGenerationInfo] = []
            for try await event in stream {
                switch event {
                case .content(let chunk): text += chunk
                case .toolCall(let call): calls.append(call)
                case .info(let info): infos.append(info)
                case .parsed: Issue.record("this engine does not parse reasoning")
                }
            }
            return (text, calls, infos)
        }

        @Test func streamsContentToolCallAndInfo() async throws {
            let stream = try await makeEngine().streamChatCompletion(
                request: request([.init(role: .user, content: .text("abc"))]))
            let (text, calls, infos) = try await collect(stream)

            #expect(text == "Hello world")
            #expect(calls.map(\.function.name) == ["lookup"])
            #expect(calls.first?.function.arguments["q"] == .string("x"))
            #expect(infos.count == 1)
            #expect(infos.first?.promptTokens == 3)
            #expect(infos.first?.completionTokens == script.count)
            // The stream ends at `max_tokens`, so the finish reason must be
            // "length".
            withKnownIssue(
                "Evaluate.swift:1773 reads tokenCount from a copy of the iterator, so a max-tokens stop reports stop"
            ) {
                #expect(
                    infos.first?.stopReason == "length", "a max-tokens stop must report length")
            } matching: { issue in
                guard case .expectationFailed = issue.kind else { return false }
                return issue.comments.contains {
                    $0.rawValue.contains("a max-tokens stop must report length")
                }
            }
        }

        /// With declared tools, a call to another tool comes back as text.
        @Test func declaredToolsFilterTheCalls() async throws {
            let stream = try await makeEngine().streamChatCompletion(
                request: request(
                    [.init(role: .user, content: .text("abc"))],
                    tools: [OpenAITool(function: .init(name: "other"))]))
            let (text, calls, _) = try await collect(stream)

            #expect(calls.isEmpty)
            #expect(text == "Hello world" + fragments[3]! + fragments[4]! + fragments[5]!)
        }

        /// A request with tool history renders through template messages.
        /// The processor gives the same tokens for both prompt kinds, so
        /// the prompt has one token for each character of the contents.
        @Test func toolHistoryStreams() async throws {
            let stream = try await makeEngine().streamChatCompletion(
                request: request([
                    .init(role: .user, content: .text("ab")),
                    .init(
                        role: .assistant, content: .text("c"),
                        toolCalls: [
                            .init(id: "t", function: .init(name: "lookup", arguments: "{}"))
                        ]
                    ),
                    .init(role: .tool, content: .text("de"), toolCallID: "t"),
                ]))
            let (text, calls, infos) = try await collect(stream)

            #expect(text == "Hello world")
            #expect(calls.count == 1)
            #expect(infos.first?.promptTokens == 5)
        }

        /// A consumer that stops reading ends the stream early.
        @Test func consumerCanStopEarly() async throws {
            let stream = try await makeEngine().streamChatCompletion(
                request: request([.init(role: .user, content: .text("abc"))]))
            var first: MLXServerGenerationEvent?
            for try await event in stream {
                first = event
                break
            }
            #expect(first == .content("Hello"))
        }
    }
}
