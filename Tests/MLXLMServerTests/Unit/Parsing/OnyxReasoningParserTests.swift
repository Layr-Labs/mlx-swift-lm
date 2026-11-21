// Copyright © 2026 Eigen Labs.
// The scripts are those of vLLM's tests/tool_use/test_muse_glimmer.py.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXLMServer

extension UnitTests {
    @Suite
    struct OnyxReasoningParserTests {
        static let answer =
            " to=self<|message|>Think about it.<|eom|>"
            + "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>"
        static let toolTurn =
            " to=self<|message|>I should read the hostname file to answer.<|eom|>"
            + "<|start|>assistant to=read.read<|message|>"
            + "<atem:function_calls>\n<atem:invoke name=\"read.read\">\n"
            + "<atem:parameter name=\"path\">/etc/hostname</atem:parameter>\n"
            + "</atem:invoke>\n</atem:function_calls><|eot|>"
        /// Cut off by the token limit inside the reasoning frame.
        static let truncated =
            " to=self<|message|>Maybe I should call "
            + "<atem:function_calls>\n<atem:invoke name=\"read.read\">\n"
            + "<atem:parameter name=\"path\">/etc/hostname</atem:parameter>\n"
            + "</atem:invoke>\n</atem:function_calls> but wait"
        static let framing = [
            "<|start|>", "<|message|>", "<|eom|>", "<|eot|>", "to=self", "to=user",
            "to=read.read",
        ]

        static let cases: [(script: String, content: String, reasoning: String?)] = [
            (answer, "The answer is 42.", "Think about it."),
            (toolTurn, "", "I should read the hostname file to answer."),
            (truncated, "", String(truncated.dropFirst(" to=self<|message|>".count))),
            (" to=user<|message|>Just a direct answer.<|eot|>", "Just a direct answer.", nil),
            ("Just a direct answer.", "Just a direct answer.", nil),
            ("<|message|>Hello.<|eot|>", "Hello.", nil),
            (" to=self<|message|>thinking", "", "thinking"),
            (
                " to=self<|message|>plan<|eom|><|start|>assistant to=user<|message|>The ans",
                "The ans", "plan"
            ),
            (
                " to=self<|message|>one<|eom|><|start|>assistant to=self<|message|>two<|eom|>"
                    + "<|start|>assistant to=user<|message|>done<|eot|>", "done", "onetwo"
            ),
        ]

        static func streamed(_ script: String, width: Int) -> ParsedReasoning {
            var parser = StreamingReasoningParser(format: .onyx)
            var content = ""
            var reasoning = ""
            var rest = script[...]
            var pieces: [ParsedReasoning] = []
            while !rest.isEmpty {
                let chunk = rest.prefix(width)
                pieces += parser.parse(String(chunk))
                rest = rest.dropFirst(chunk.count)
            }
            pieces += parser.finish()
            for piece in pieces {
                content += piece.content
                reasoning += piece.reasoningContent ?? ""
            }
            return .init(content: content, reasoningContent: reasoning.isEmpty ? nil : reasoning)
        }

        @Test func collectedTextSplitsIntoReasoningAndContent() {
            for (script, content, reasoning) in Self.cases {
                let parsed = ReasoningParser(format: .onyx).parse(script)
                #expect(parsed.content == content, "\(script)")
                #expect(parsed.reasoningContent == reasoning, "\(script)")
            }
        }

        @Test(arguments: [1, 3, 17])
        func streamedTextGivesTheSameSplit(width: Int) {
            for (script, content, reasoning) in Self.cases {
                let parsed = Self.streamed(script, width: width)
                #expect(parsed.content == content, "\(script)")
                #expect(parsed.reasoningContent == reasoning, "\(script)")
            }
        }

        @Test func noFramingReachesEitherChannel() {
            for script in [Self.answer, Self.toolTurn] {
                for width in [1, 3, 17] {
                    let parsed = Self.streamed(script, width: width)
                    for marker in Self.framing + ["<atem:"] {
                        #expect(!parsed.content.contains(marker))
                        #expect(parsed.reasoningContent?.contains(marker) != true)
                    }
                }
            }
        }

        @Test func formatNamesAndInference() throws {
            #expect(ReasoningParserFormat.onyx.rawValue == "onyx")
            for name in ["onyx", "muse_glimmer", "Muse-Glimmer", "atem"] {
                let decoded = try JSONDecoder().decode(
                    ReasoningParserFormat.self, from: Data("\"\(name)\"".utf8))
                #expect(decoded == .onyx)
            }
            #expect(ReasoningParserFormat.inferred(modelType: "muse_glimmer") == .onyx)
            #expect(ReasoningParserFormat.inferred(modelType: "MUSE_GLIMMER") == .onyx)
            #expect(ReasoningParserFormat.inferred(modelType: nil, toolCallFormat: .atem) == .onyx)
            #expect(ReasoningParserFormat.inferred(modelType: "gpt_oss") == nil)
            #expect(ReasoningParserFormat.inferred(modelType: "qwen3_5") == nil)
            #expect(ReasoningParserFormat.inferred(modelType: nil) == nil)
        }

        @Test func toolParserNamesAndInference() throws {
            #expect(
                try ServerToolParser.resolve(requested: nil, modelType: "muse_glimmer") == .atem)
            #expect(
                try ServerToolParser.resolve(requested: "auto", modelType: "muse_glimmer") == .atem)
            for name in ["atem", "onyx", "muse_glimmer", "Muse-Glimmer"] {
                #expect(try ServerToolParser.resolve(requested: name, modelType: nil) == .atem)
            }
        }

        // MARK: - Service

        static func chunks(_ script: String, width: Int = 5) -> [MLXServerGenerationEvent] {
            var events: [MLXServerGenerationEvent] = []
            var rest = script[...]
            while !rest.isEmpty {
                let chunk = rest.prefix(width)
                events.append(.content(String(chunk)))
                rest = rest.dropFirst(chunk.count)
            }
            return events
        }

        static func info(_ stopReason: String) -> MLXServerGenerationEvent {
            .info(
                .init(
                    promptTokens: 10, completionTokens: 8, promptTime: 0, generationTime: 0,
                    stopReason: stopReason))
        }

        /// The collected and the streamed chat completion of one engine script.
        static func completions(
            _ events: [MLXServerGenerationEvent], request parser: ReasoningParserFormat? = nil,
            server: ReasoningParserFormat? = .onyx
        ) async throws -> [(content: String, reasoning: String, finishReason: String?)] {
            let service = MLXOpenAIService(
                engine: OnyxFixtureEngine(events: events), defaultReasoningParser: server)
            var request = OpenAIChatCompletionRequest(
                model: "fixture", messages: [], reasoningParser: parser)
            let collected = try await service.createChatCompletion(request: request)
            let message = try #require(collected.choices.first)
            request.stream = true
            var content = ""
            var reasoning = ""
            var finishReason: String?
            for try await frame in try await service.streamChatCompletionFrames(request: request) {
                for line in frame.split(separator: "\n") where line.hasPrefix("data: {") {
                    let value =
                        try JSONSerialization.jsonObject(with: Data(line.dropFirst(6).utf8))
                        as? [String: Any]
                    for choice in value?["choices"] as? [[String: Any]] ?? [] {
                        let delta = choice["delta"] as? [String: Any]
                        content += delta?["content"] as? String ?? ""
                        reasoning += delta?["reasoning_content"] as? String ?? ""
                        finishReason = choice["finish_reason"] as? String ?? finishReason
                    }
                }
            }
            return [
                (
                    message.message.textContent, message.message.reasoningContent ?? "",
                    message.finishReason
                ),
                (content, reasoning, finishReason),
            ]
        }

        @Test func serviceReturnsReasoningContentFromFramedText() async throws {
            let events = Self.chunks(Self.answer) + [Self.info("stop")]
            for result in try await Self.completions(events) {
                #expect(result.content == "The answer is 42.")
                #expect(result.reasoning == "Think about it.")
                #expect(result.finishReason == "stop")
            }
            // A request can also name the parser itself.
            for result in try await Self.completions(events, request: .onyx, server: nil) {
                #expect(result.content == "The answer is 42.")
                #expect(result.reasoning == "Think about it.")
            }
        }

        @Test func serviceKeepsReasoningCutOffByTheTokenLimit() async throws {
            let events = Self.chunks(Self.truncated) + [Self.info("length")]
            for result in try await Self.completions(events) {
                #expect(result.content.isEmpty)
                #expect(
                    result.reasoning
                        == String(Self.truncated.dropFirst(" to=self<|message|>".count)))
                #expect(result.finishReason == "length")
            }
        }

        /// The container engine reports the channels itself; the service does
        /// not parse them again.
        @Test func serviceKeepsChannelsAnEngineAlreadySeparated() async throws {
            let call = ToolCall(
                function: .init(
                    name: "read.read", arguments: ["path": JSONValue.string("/etc/hostname")]))
            let events: [MLXServerGenerationEvent] = [
                .parsed(.init(content: "", reasoningContent: "plan with <|eom|> spelled out")),
                .toolCall(call),
                .parsed(.init(content: "Reading it.", reasoningContent: nil)),
                Self.info("stop"),
            ]
            for result in try await Self.completions(events) {
                #expect(result.content == "Reading it.")
                #expect(result.reasoning == "plan with <|eom|> spelled out")
                #expect(result.finishReason == "tool_calls")
            }
        }
    }
}

private struct OnyxFixtureEngine: MLXServerEngine {
    let events: [MLXServerGenerationEvent]
    func availableModels() async throws -> [MLXServerModel] { [] }
    func streamChatCompletion(request: OpenAIChatCompletionRequest) async throws
        -> AsyncThrowingStream<MLXServerGenerationEvent, Error>
    {
        AsyncThrowingStream { continuation in
            for event in events { continuation.yield(event) }
            continuation.finish()
        }
    }
    func tokenize(_ request: TokenizeRequest) async throws -> TokenizeResponse { .init(tokens: []) }
    func detokenize(_ request: DetokenizeRequest) async throws -> DetokenizeResponse {
        .init(text: "")
    }
    func applyTemplate(_ request: ApplyTemplateRequest) async throws -> TokenizeResponse {
        .init(tokens: [])
    }
}
