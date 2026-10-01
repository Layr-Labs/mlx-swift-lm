import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

/// Text of each token of the scripted model. Decoding joins the fragments,
/// so the streaming detokenizer gives each fragment as one chunk.
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
/// temperature 0 the sampler takes the scripted token. The model has no
/// parameters and no cache.
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
    /// One token for each character of the text, with value 6.
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
        encode(text: messages.compactMap { $0["content"] as? String }.joined(), addSpecialTokens: false)
    }
}

/// Makes one token for each character of the chat content.
private struct FragmentProcessor: UserInputProcessor {
    func prepare(input: UserInput) throws -> LMInput {
        let tokens = try FragmentTokenizer().applyChatTemplate(
            messages: DefaultMessageGenerator().generate(from: input), tools: input.tools,
            additionalContext: nil)
        return LMInput(tokens: MLXArray(tokens.map(Int32.init)))
    }
}

extension KernelTests {

    /// Tests of `ModelContainer.prepare(input:)` and
    /// `ModelContainer.generate(input:parameters:tools:wiredMemoryTicket:)`
    /// with a scripted model. The model output is exact, so the tests
    /// compare text and tool calls exactly.
    @Suite
    struct ModelContainerGenerationTests {

        private func makeContainer() -> ModelContainer {
            ModelContainer(
                context: ModelContext(
                    configuration: ModelConfiguration(id: "test/scripted"),
                    model: ScriptedModel(), processor: FragmentProcessor(),
                    tokenizer: FragmentTokenizer()))
        }

        private func collect(_ stream: AsyncStream<Generation>) async -> (
            text: String, calls: [ToolCall], info: GenerateCompletionInfo?
        ) {
            var text = ""
            var calls: [ToolCall] = []
            var info: GenerateCompletionInfo?
            for await item in stream {
                switch item {
                case .chunk(let chunk): text += chunk
                case .toolCall(let call): calls.append(call)
                case .info(let value): info = value
                }
            }
            return (text, calls, info)
        }

        private let parameters = GenerateParameters(maxTokens: script.count, temperature: 0)

        @Test func prepareUsesTheProcessor() async throws {
            let container = makeContainer()
            let input = try await container.prepare(input: UserInput(prompt: "abcd"))
            #expect(input.text.tokens.shape == [4])
            #expect(input.text.tokens.asArray(Int32.self) == [6, 6, 6, 6])
        }

        @Test func generateStreamsTextToolCallsAndInfo() async throws {
            let container = makeContainer()
            let input = try await container.prepare(input: UserInput(prompt: "abc"))
            let stream = try await container.generate(input: input, parameters: parameters)
            let (text, calls, info) = await collect(stream)

            #expect(text == "Hello world")
            #expect(calls.map(\.function.name) == ["lookup"])
            #expect(calls.first?.function.arguments["q"] == .string("x"))
            #expect(info?.promptTokenCount == 3)
            #expect(info?.generationTokenCount == script.count)
            #expect(info?.stopReason == .length)
        }

        /// With declared tools, a call to another tool is not a tool call.
        /// The processor gives back its text.
        @Test func generatePassesTheDeclaredTools() async throws {
            let container = makeContainer()
            let input = try await container.prepare(input: UserInput(prompt: "abc"))
            let tools: [[String: any Sendable]] = [
                ["type": "function", "function": ["name": "other"] as [String: any Sendable]]
            ]
            let stream = try await container.generate(
                input: input, parameters: parameters, tools: tools)
            let (text, calls, _) = await collect(stream)

            #expect(calls.isEmpty)
            #expect(text == "Hello world" + fragments[3]! + fragments[4]! + fragments[5]!)
        }
    }
}
