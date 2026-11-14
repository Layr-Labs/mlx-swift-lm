// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLMServer

/// Text of each token of the scripted model. Decoding joins the fragments.
private let fragments: [Int: String] = [
    1: " to=self", 2: "<|message|>", 3: "plan", 4: "<|eom|>", 5: "<|start|>",
    6: "assistant to=lookup",
    7: "<atem:function_calls>\n<atem:invoke name=\"lookup\">\n"
        + "<atem:parameter name=\"q\">x</atem:parameter>\n</atem:invoke>\n"
        + "<atem:invoke name=\"lookup\">\n<atem:parameter name=\"q\">y</atem:parameter>\n"
        + "</atem:invoke>\n</atem:function_calls>",
    8: "<|eot|>", 9: "assistant to=user", 10: "answer", 11: "<|end_of_text|>",
    12: " to=user",
    14: "<atem:function_calls>\n<atem:invoke name=\"lookup\">\n"
        + "<atem:parameter name=\"q\">x</atem:parameter>\n</atem:invoke>\n</atem:function_calls>",
]
private let vocabularySize = 16

/// A model that gives one-hot logits for the next token of its script. With
/// temperature 0 the sampler takes the scripted token.
private final class OnyxScriptedModel: Module, LanguageModel {
    private let script: [Int]
    private var index = 0

    init(script: [Int]) {
        self.script = script
        super.init()
    }

    private func nextLogits() -> MLXArray {
        var row = [Float](repeating: 0, count: vocabularySize)
        row[script[min(index, script.count - 1)]] = 10
        index += 1
        return MLXArray(row).reshaped([1, 1, vocabularySize])
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        .logits(LMOutput(logits: nextLogits()))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        nextLogits()
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
}

private struct OnyxFragmentTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        Array(repeating: 13, count: text.count)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { fragments[$0] ?? "" }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? {
        fragments.first { $0.value == token }?.key
    }
    func convertIdToToken(_ id: Int) -> String? { fragments[id] }
    var bosToken: String? { nil }
    var eosToken: String? { "<|eot|>" }
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

private struct OnyxFragmentProcessor: UserInputProcessor {
    func prepare(input: UserInput) throws -> LMInput {
        let messages: [[String: any Sendable]]
        switch input.prompt {
        case .messages(let list): messages = list
        default: messages = DefaultMessageGenerator().generate(from: input)
        }
        let tokens = try OnyxFragmentTokenizer().applyChatTemplate(
            messages: messages, tools: nil, additionalContext: nil)
        return LMInput(tokens: MLXArray(tokens.map(Int32.init)))
    }
}

extension KernelTests {

    /// `MLXModelContainerEngine` with a scripted Muse Glimmer response: the
    /// engine reads the Onyx frames from the token ids and reports reasoning
    /// and content apart.
    @Suite
    struct OnyxContainerEngineTests {

        struct Collected {
            var content = ""
            var reasoning = ""
            var unparsed = ""
            var calls: [ToolCall] = []
            var infos: [ServerGenerationInfo] = []
        }

        /// Without `onyxParsing` the engine takes its default.
        private func engine(_ script: [Int], onyxParsing: OnyxParsingMode? = nil)
            -> MLXModelContainerEngine
        {
            let model = ModelContainer(
                context: ModelContext(
                    configuration: ModelConfiguration(id: "test/onyx", toolCallFormat: .atem),
                    model: OnyxScriptedModel(script: script),
                    processor: OnyxFragmentProcessor(), tokenizer: OnyxFragmentTokenizer()))
            guard let onyxParsing else {
                return MLXModelContainerEngine(
                    modelID: "scripted", model: model, modelType: "muse_glimmer")
            }
            return MLXModelContainerEngine(
                modelID: "scripted", model: model, modelType: "muse_glimmer",
                onyxParsing: onyxParsing)
        }

        private var lookup: OpenAITool {
            OpenAITool(
                function: .init(
                    name: "lookup",
                    parameters: .object([
                        "type": .string("object"),
                        "properties": .object(["q": .object(["type": .string("string")])]),
                    ])))
        }

        private func run(
            _ script: [Int], maxTokens: Int, tools: [OpenAITool]? = nil,
            onyxParsing: OnyxParsingMode? = nil
        ) async throws -> Collected {
            let request = OpenAIChatCompletionRequest(
                model: "scripted", messages: [.init(role: .user, content: .text("abc"))],
                tools: tools, temperature: 0, maxTokens: maxTokens)
            var collected = Collected()
            let engine = engine(script, onyxParsing: onyxParsing)
            for try await event in try await engine.streamChatCompletion(request: request) {
                switch event {
                case .parsed(let parsed):
                    collected.content += parsed.content
                    collected.reasoning += parsed.reasoningContent ?? ""
                case .content(let text): collected.unparsed += text
                case .toolCall(let call): collected.calls.append(call)
                case .info(let info): collected.infos.append(info)
                case .logprobs, .toolCallDelta: break
                }
            }
            return collected
        }

        @Test func reasoningToolCallAndAnswerArriveOnTheirChannels() async throws {
            // Reasoning, a tool frame with one invoke, then the answer.
            let script = [1, 2, 3, 4, 5, 6, 2, 14, 4, 5, 9, 2, 10, 8]
            let collected = try await run(script, maxTokens: 64, tools: [lookup])
            #expect(collected.reasoning == "plan")
            #expect(collected.content == "answer")
            #expect(collected.unparsed.isEmpty)
            #expect(collected.calls.map(\.function.name) == ["lookup"])
            #expect(collected.calls.map(\.function.arguments) == [["q": .string("x")]])
            #expect(collected.infos.first?.stopReason == "stop")
        }

        /// The engine reads tool frames strictly unless the reference reading
        /// is asked for: a frame with two invokes gives no call by default and
        /// a call for each invoke on request. The refused frame never reaches
        /// content.
        @Test(arguments: [nil, OnyxParsingMode.strict, .referenceParity])
        func twoInvokesInOneFrameNeedTheReferenceReading(_ mode: OnyxParsingMode?) async throws {
            let script = [1, 2, 3, 4, 5, 6, 2, 7, 4, 5, 9, 2, 10, 8]
            let collected = try await run(
                script, maxTokens: 64, tools: [lookup], onyxParsing: mode)
            #expect(collected.reasoning == "plan")
            #expect(collected.content == "answer")
            #expect(collected.unparsed.isEmpty)
            // `<|eom|>` did not end generation; `<|eot|>` did.
            #expect(collected.infos.first?.stopReason == "stop")
            #expect(collected.infos.first?.completionTokens == script.count - 1)
            guard mode == .referenceParity else {
                #expect(collected.calls.isEmpty)
                return
            }
            #expect(collected.calls.map(\.function.name) == ["lookup", "lookup"])
            #expect(
                collected.calls.map(\.function.arguments)
                    == [["q": .string("x")], ["q": .string("y")]])
        }

        /// A rejected call and a protocol error have no OpenAI field: the
        /// handler emits nothing for them and generation goes on.
        @Test func rejectedCallsAndProtocolErrorsEmitNothing() throws {
            let strictLookup = OpenAITool(
                function: .init(
                    name: "lookup",
                    parameters: .object([
                        "type": .string("object"),
                        "properties": .object([
                            "q": .object(["type": .string("string"), "enum": .array([.string("z")])])
                        ]),
                    ])))
            struct Read {
                var content = ""
                var calls: [ToolCall] = []
                var otherEvents = 0
                var allMore = true
            }
            func read(_ tokens: [Int], validation: ToolCallValidationPolicy) throws -> Read {
                var handler = try #require(
                    ChannelTokenLoopHandler(
                        format: .atem, tokenizer: OnyxFragmentTokenizer(), stopStrings: [],
                        tools: [strictLookup.toolSpec()],
                        toolCallPolicy: ToolCallPolicy(validation: validation)))
                var read = Read()
                for token in tokens {
                    let disposition = handler.onToken(token, logProbabilities: nil) { event in
                        switch event {
                        case .parsed(let parsed) where (parsed.reasoningContent ?? "").isEmpty:
                            read.content += parsed.content
                        case .toolCall(let call): read.calls.append(call)
                        default: read.otherEvents += 1
                        }
                        return true
                    }
                    if case .more = disposition {} else { read.allMore = false }
                }
                return read
            }
            // An answer, then a tool frame whose value breaks the schema.
            let frame = [12, 2, 10, 4, 5, 6, 2, 14, 4]
            let rejected = try read(frame, validation: .strict)
            #expect(rejected.allMore)
            #expect(rejected.content == "answer")
            #expect(rejected.calls.isEmpty)
            #expect(rejected.otherEvents == 0)
            // The same frame is a call when validation does not refuse it.
            let permissive = try read(frame, validation: .permissive)
            #expect(permissive.content == "answer")
            #expect(permissive.calls.map(\.function.arguments) == [["q": .string("x")]])
            // A frame header that ends before `<|message|>` is a protocol
            // error; the next frame is still read.
            let broken = try read([12, 2, 10, 4, 5, 6, 4, 5, 9, 2, 10], validation: .strict)
            #expect(broken.allMore)
            #expect(broken.content == "answeranswer")
            #expect(broken.calls.isEmpty)
            #expect(broken.otherEvents == 0)
        }

        @Test(arguments: [OnyxParsingMode.strict, .referenceParity])
        func endOfTextEndsGeneration(_ mode: OnyxParsingMode) async throws {
            let collected = try await run(
                [12, 2, 10, 11, 10, 10], maxTokens: 64, onyxParsing: mode)
            #expect(collected.infos.first?.stopReason == "stop")
            guard mode == .strict else {
                #expect(collected.content == "answer")
                return
            }
            // The strict reading closes a frame only at `<|eom|>` or `<|eot|>`,
            // so the marker that ends generation stays in the frame's text.
            withKnownIssue("strict Onyx reading keeps <|end_of_text|> as user text") {
                #expect(collected.content == "answer")
            }
        }

        @Test func reasoningCutOffByTheTokenLimitIsStillReasoning() async throws {
            let collected = try await run([1, 2, 3], maxTokens: 5)
            #expect(collected.reasoning == "planplanplan")
            #expect(collected.content.isEmpty)
            #expect(collected.unparsed.isEmpty)
            #expect(collected.infos.first?.stopReason == "length")
        }

        @Test func toolCallCutOffByTheTokenLimitIsNotACall() async throws {
            // The frame opens and generation ends before its block.
            let collected = try await run([1, 2, 3, 4, 5, 6, 2], maxTokens: 7, tools: [lookup])
            #expect(collected.reasoning == "plan")
            #expect(collected.calls.isEmpty)
            #expect(collected.content.isEmpty)
            #expect(collected.infos.first?.stopReason == "length")
        }

        @Test func serviceReturnsReasoningContent() async throws {
            let service = MLXOpenAIService(
                engine: engine([1, 2, 3, 4, 5, 9, 2, 10, 8]),
                defaultReasoningParser: ReasoningParserFormat.inferred(modelType: "muse_glimmer"))
            let response = try await service.createChatCompletion(
                request: .init(
                    model: "scripted", messages: [.init(role: .user, content: .text("abc"))],
                    temperature: 0, maxTokens: 64))
            #expect(response.choices.first?.message.textContent == "answer")
            #expect(response.choices.first?.message.reasoningContent == "plan")
            #expect(response.choices.first?.finishReason == "stop")
        }
    }
}
