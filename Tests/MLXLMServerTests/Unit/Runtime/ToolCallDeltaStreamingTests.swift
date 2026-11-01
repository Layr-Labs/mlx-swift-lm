// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXLMServer

extension UnitTests {
    /// `.toolCallDelta` rendering: OpenAI `tool_calls` fragments on chat streams,
    /// whole calls on every other surface.
    @Suite
    struct ToolCallDeltaStreamingTests {
        private static let call = ToolCall(
            function: .init(
                name: "get_weather",
                arguments: ["location": .string("Paris"), "unit": .string("celsius")]))

        private static let streamedCall: [MLXServerGenerationEvent] = [
            .toolCallDelta(.init(index: 0, name: "get_weather")),
            .toolCallDelta(.init(index: 0, arguments: #"{"location":"Pa"#)),
            .toolCallDelta(.init(index: 0, arguments: #"ris","unit":"celsius"}"#, completed: call)),
        ]

        private static func info(_ stopReason: String = "stop") -> MLXServerGenerationEvent {
            .info(
                .init(
                    promptTokens: 9, completionTokens: 12, promptTime: 0, generationTime: 0,
                    stopReason: stopReason))
        }

        private static func request(parallelToolCalls: Bool? = nil) -> OpenAIChatCompletionRequest {
            .init(
                model: "local-model", messages: [.init(role: .user, content: .text("weather?"))],
                parallelToolCalls: parallelToolCalls, stream: true,
                streamOptions: .init(includeUsage: true))
        }

        private static func chunks(_ frames: [String]) throws -> [OpenAIChatCompletionChunk] {
            try frames.compactMap { frame in
                guard frame.hasPrefix("data: "), !frame.contains("[DONE]") else { return nil }
                let payload = frame.dropFirst("data: ".count).trimmingCharacters(
                    in: .whitespacesAndNewlines)
                return try JSONDecoder().decode(
                    OpenAIChatCompletionChunk.self, from: Data(payload.utf8))
            }
        }

        private static func frames(
            _ events: [MLXServerGenerationEvent], request: OpenAIChatCompletionRequest = request()
        ) async throws -> (frames: [String], engine: RecordingEngine) {
            let engine = RecordingEngine(events: events)
            var frames: [String] = []
            for try await frame in try await MLXOpenAIService(engine: engine)
                .streamChatCompletionFrames(request: request)
            {
                frames.append(frame)
            }
            return (frames, engine)
        }

        @Test("chat streams request deltas and render OpenAI fragments")
        func chatStreamRendersFragments() async throws {
            let (frames, engine) = try await Self.frames(Self.streamedCall + [Self.info()])
            #expect(await engine.lastRequest?.streamToolCallDeltas == true)
            let deltas = try Self.chunks(frames).flatMap {
                $0.choices.flatMap { $0.delta.toolCalls ?? [] }
            }
            #expect(deltas.count == 3)
            let first = try #require(deltas.first)
            #expect(first.index == 0)
            #expect(first.type == "function")
            #expect(first.id?.hasPrefix("call_") == true)
            #expect(first.function == .init(name: "get_weather", arguments: ""))
            // Continuations carry only the wire index and an arguments fragment.
            for delta in deltas.dropFirst() {
                #expect(
                    delta.index == 0 && delta.id == nil && delta.type == nil
                        && delta.function?.name == nil)
            }
            let arguments = deltas.compactMap { $0.function?.arguments }.joined()
            #expect(
                try JSONDecoder().decode([String: JSONValue].self, from: Data(arguments.utf8))
                    == Self.call.function.arguments)
            #expect(frames.contains { $0.contains(#""finish_reason":"tool_calls""#) })
            // The wire omits absent fields instead of sending nulls.
            let continuation = try #require(frames.first { $0.contains(#""arguments":"ris"#) })
            #expect(
                !continuation.contains("call_") && !continuation.contains("\"name\"")
                    && !continuation.contains("null"))
        }

        @Test("streamed and whole calls share one wire index sequence")
        func mixedCallsKeepDistinctIndices() async throws {
            let (frames, _) = try await Self.frames(
                Self.streamedCall + [.toolCall(Self.call)]
                    + Self.streamedCall.map { event in
                        guard case .toolCallDelta(var delta) = event else { return event }
                        delta.index = 1
                        return .toolCallDelta(delta)
                    } + [Self.info()])
            let deltas = try Self.chunks(frames).flatMap {
                $0.choices.flatMap { $0.delta.toolCalls ?? [] }
            }
            #expect(deltas.filter { $0.id != nil }.map(\.index) == [0, 1, 2])
            #expect(Set(deltas.compactMap(\.id)).count == 3)
        }

        @Test("a call cut off by the token limit finishes with length")
        func truncatedStreamedCallKeepsLength() async throws {
            let (frames, _) = try await Self.frames(
                Array(Self.streamedCall.prefix(2)) + [Self.info("length")])
            #expect(frames.contains { $0.contains(#""finish_reason":"length""#) })
        }

        @Test("parallel_tool_calls false rejects a second streamed call")
        func parallelToolCallsFalseRejectsSecondCall() async throws {
            let second = MLXServerGenerationEvent.toolCallDelta(
                .init(index: 1, name: "get_weather"))
            await #expect(throws: MLXOpenAIServiceError.multipleToolCallsNotAllowed) {
                _ = try await Self.frames(
                    Self.streamedCall + [second, Self.info()],
                    request: Self.request(parallelToolCalls: false))
            }
        }

        @Test("collected and Responses surfaces keep whole calls")
        func nonChatSurfacesUseCompletedCalls() async throws {
            let events = Self.streamedCall + [Self.info()]
            let response = try await MLXOpenAIService(engine: RecordingEngine(events: events))
                .createChatCompletion(request: Self.request())
            let calls = try #require(response.choices.first?.message.toolCalls)
            #expect(calls.map(\.function.name) == ["get_weather"])
            #expect(
                try JSONDecoder().decode(
                    [String: JSONValue].self, from: Data(calls[0].function.arguments.utf8))
                    == Self.call.function.arguments)
            #expect(response.choices.first?.finishReason == "tool_calls")

            let engine = RecordingEngine(events: events)
            let responseRequest = try JSONDecoder().decode(
                OpenAIResponseRequest.self,
                from: Data(#"{"model":"local-model","input":"hi","stream":true}"#.utf8))
            var frames: [String] = []
            for try await frame in try await MLXOpenAIService(engine: engine).streamResponseFrames(
                request: responseRequest)
            {
                frames.append(frame)
            }
            #expect(await engine.lastRequest?.streamToolCallDeltas == false)
            #expect(frames.contains { $0.contains("response.function_call_arguments.delta") })
            #expect(frames.contains { $0.contains(#""name":"get_weather""#) })
        }

        @Test("a whole call encodes exactly as before as a delta entry")
        func wholeCallEncodingIsUnchanged() throws {
            let call = OpenAIToolCall(
                id: "call_1", function: .init(name: "f", arguments: #"{"a":1}"#), index: 0)
            let encoder = JSONEncoder.openAIServer
            #expect(try encoder.encode(OpenAIToolCallDelta(call)) == encoder.encode(call))
            let partial = try JSONDecoder().decode(
                OpenAIToolCallDelta.self,
                from: Data(#"{"index":0,"function":{"arguments":"{\"a\""}}"#.utf8))
            #expect(partial == .init(index: 0, function: .init(arguments: #"{"a""#)))
        }
    }
}

private actor RecordingEngine: MLXServerEngine {
    private let events: [MLXServerGenerationEvent]
    private(set) var lastRequest: OpenAIChatCompletionRequest?

    init(events: [MLXServerGenerationEvent]) { self.events = events }

    func availableModels() async throws -> [MLXServerModel] { [.init(id: "local-model")] }

    func streamChatCompletion(
        request: OpenAIChatCompletionRequest
    ) async throws -> AsyncThrowingStream<MLXServerGenerationEvent, Error> {
        lastRequest = request
        let events = self.events
        return AsyncThrowingStream { continuation in
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
