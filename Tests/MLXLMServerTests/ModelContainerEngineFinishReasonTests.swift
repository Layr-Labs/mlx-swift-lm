import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLMServer

/// Regression test for the finish reason of a stream that stops at
/// `max_tokens`. The async generate loop in `Evaluate.swift` read
/// `tokenCount` from a copy of the iterator, so such a stream reported
/// `.cancelled` and the server sent the finish reason "stop", not "length".
///
/// The test is copied from
/// `ModelContainerEngineStreamTests.streamsContentToolCallAndInfo` of PR #237
/// (`Tests/MLXLMServerTests/Kernel/Runtime/ModelContainerEngineStreamTests.swift`),
/// without the known issue. The scripted model, tokenizer and processor are
/// private copies from that file with other names, so they do not collide
/// with that PR.
@Suite
struct ModelContainerEngineFinishReasonTests {

    private func makeEngine() -> MLXModelContainerEngine {
        MLXModelContainerEngine(
            modelID: "scripted",
            model: ModelContainer(
                context: ModelContext(
                    configuration: ModelConfiguration(id: "test/scripted"),
                    model: FinishReasonScriptedModel(), processor: FinishReasonProcessor(),
                    tokenizer: FinishReasonTokenizer())))
    }

    @Test func aMaxTokensStopReportsLength() async throws {
        let request = OpenAIChatCompletionRequest(
            model: "scripted", messages: [.init(role: .user, content: .text("abc"))],
            tools: nil, temperature: 0, maxTokens: finishReasonScript.count)
        let stream = try await makeEngine().streamChatCompletion(request: request)

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

        #expect(text == "Hello world")
        #expect(calls.map(\.function.name) == ["lookup"])
        #expect(calls.first?.function.arguments["q"] == .string("x"))
        #expect(infos.count == 1)
        #expect(infos.first?.promptTokens == 3)
        #expect(infos.first?.completionTokens == finishReasonScript.count)
        // The stream ends at `max_tokens`, so the finish reason must be
        // "length".
        #expect(infos.first?.stopReason == "length", "a max-tokens stop must report length")
    }
}

// MARK: - Fixtures

/// Text of each token of the scripted model. Decoding joins the fragments.
private let finishReasonFragments: [Int: String] = [
    1: "Hello",
    2: " world",
    3: "<tool_call>",
    4: #"{"name": "lookup", "arguments": {"q": "x"}}"#,
    5: "</tool_call>",
]
private let finishReasonScript = [1, 2, 3, 4, 5]
private let finishReasonVocabularySize = 8

/// A model that gives one-hot logits for the next token of the script. With
/// temperature 0 the sampler takes the scripted token.
private final class FinishReasonScriptedModel: Module, LanguageModel {
    private var index = 0

    private func nextLogits() -> MLXArray {
        var row = [Float](repeating: 0, count: finishReasonVocabularySize)
        row[finishReasonScript[min(index, finishReasonScript.count - 1)]] = 10
        index += 1
        return MLXArray(row).reshaped([1, 1, finishReasonVocabularySize])
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .logits(LMOutput(logits: nextLogits()))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        nextLogits()
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
}

private struct FinishReasonTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        Array(repeating: 6, count: text.count)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { finishReasonFragments[$0] ?? "" }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { finishReasonFragments[id] }
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

/// Makes one token for each character of the message content.
private struct FinishReasonProcessor: UserInputProcessor {
    func prepare(input: UserInput) throws -> LMInput {
        let messages: [[String: any Sendable]]
        switch input.prompt {
        case .messages(let list): messages = list
        default: messages = DefaultMessageGenerator().generate(from: input)
        }
        let tokens = try FinishReasonTokenizer().applyChatTemplate(
            messages: messages, tools: nil, additionalContext: nil)
        return LMInput(tokens: MLXArray(tokens.map(Int32.init)))
    }
}
