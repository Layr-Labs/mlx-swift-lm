// Copyright © 2026 Eigen Labs Inc.

import Foundation
import Testing

@testable import MLXLMServer

extension UnitTests {
    struct CollectedReasoningTests {
        @Test(
            "unclosed think blocks retain preceding answer content and all reasoning",
            arguments: [ReasoningParserFormat.qwen3, .deepseekR1])
        func unclosedBlocks(format: ReasoningParserFormat) {
            let parser = ReasoningParser(format: format)
            for (text, answer, reasoning) in [
                ("<think>Still thinking", "", "Still thinking"),
                ("Answer<think>Still thinking", "Answer", "Still thinking"),
                ("<think>first</think>Answer<think>second", "Answer", "first\nsecond"),
                ("<think>", "", ""),
            ] {
                let result = parser.parse(text)
                #expect(result.content == answer)
                #expect(result.reasoningContent == (reasoning.isEmpty ? nil : reasoning))
            }
        }

        @Test(
            "collected and streamed Chat and Responses keep truncated reasoning out of answers",
            arguments: [ReasoningParserFormat.qwen3, .deepseekR1, .none])
        func serviceChannels(format: ReasoningParserFormat) async throws {
            let service = MLXOpenAIService(
                engine: ReasoningFixtureEngine(events: [
                    .content("<think>"), .content("Still thinking"),
                    .info(
                        .init(
                            promptTokens: 10, completionTokens: 8,
                            promptTime: 0, generationTime: 0, stopReason: "length")),
                ]))
            let expectedContent = format == .none ? "<think>Still thinking" : ""
            let expectedReasoning: String? = format == .none ? nil : "Still thinking"
            var chat = OpenAIChatCompletionRequest(
                model: "fixture", messages: [], reasoningParser: format)
            let collected = try await service.createChatCompletion(request: chat)
            #expect(collected.choices.first?.message.textContent == expectedContent)
            #expect(collected.choices.first?.message.reasoningContent == expectedReasoning)
            #expect(collected.choices.first?.finishReason == "length")
            #expect(collected.usage.promptTokens == 10)
            #expect(collected.usage.completionTokens == 8)
            chat.stream = true
            var content = ""
            var reasoning = ""
            for try await frame in try await service.streamChatCompletionFrames(request: chat) {
                for line in frame.split(separator: "\n") where line.hasPrefix("data: {") {
                    let value =
                        try JSONSerialization.jsonObject(with: Data(line.dropFirst(6).utf8))
                        as? [String: Any]
                    for choice in value?["choices"] as? [[String: Any]] ?? [] {
                        let delta = choice["delta"] as? [String: Any]
                        content += delta?["content"] as? String ?? ""
                        reasoning += delta?["reasoning_content"] as? String ?? ""
                    }
                }
            }
            #expect(content == expectedContent)
            #expect(reasoning == (expectedReasoning ?? ""))
            var request = OpenAIResponseRequest(
                model: "fixture", input: .text("hi"), reasoning: .init(parser: format))
            let response = try await service.createResponse(request: request)
            #expect(response.status == .incomplete)
            #expect(response.usage?.outputTokens == 8)
            let retrieved = try await service.retrieveResponse(id: response.id)
            #expect(response == retrieved)
            checkOutput(response.output, content: expectedContent, reasoning: expectedReasoning)
            request.stream = true
            var terminal: OpenAIResponse?
            for try await frame in try await service.streamResponseFrames(request: request) {
                guard frame.hasPrefix("event: response.incomplete\n"),
                    let line = frame.split(separator: "\n").first(where: { $0.hasPrefix("data: ") })
                else { continue }
                struct Event: Decodable { let response: OpenAIResponse }
                terminal = try JSONDecoder().decode(Event.self, from: Data(line.dropFirst(6).utf8))
                    .response
            }
            let streamed = try #require(terminal)
            #expect(streamed.status == .incomplete)
            #expect(streamed.usage?.outputTokens == 8)
            checkOutput(streamed.output, content: expectedContent, reasoning: expectedReasoning)
        }

        @Test("native parsed channels remain authoritative with literal think markers")
        func nativeChannels() async throws {
            let service = MLXOpenAIService(
                engine: ReasoningFixtureEngine(events: [
                    .parsed(
                        .init(content: "literal <think>visible</think>", reasoningContent: "native")
                    )
                ]))
            let result = try await service.createChatCompletion(
                request: .init(
                    model: "fixture", messages: [], reasoningParser: .qwen3))
            #expect(result.choices.first?.message.textContent == "literal <think>visible</think>")
            #expect(result.choices.first?.message.reasoningContent == "native")
        }

        private func checkOutput(
            _ items: [OpenAIResponseOutputItem], content: String, reasoning: String?
        ) {
            #expect(
                items.filter { $0.type == "message" }.flatMap { $0.content ?? [] }.map(\.text)
                    .joined() == content)
            #expect(
                items.filter { $0.type == "reasoning" }.flatMap { $0.summary ?? [] }.map(\.text)
                    .joined() == (reasoning ?? ""))
        }
    }
}

private struct ReasoningFixtureEngine: MLXServerEngine {
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
