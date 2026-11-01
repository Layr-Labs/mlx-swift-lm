// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXLMServer

extension UnitTests {
    struct OpenAIObjectCompatibilityTests {
        @Test("catalog fallback is stable across service facades and preserves known dates")
        func catalogDates() async throws {
            let first = MLXOpenAIService(engine: ObjectFixtureEngine())
            let second = MLXOpenAIService(engine: ObjectFixtureEngine())
            let a = try await first.availableModels()
            let b = try await second.availableModels()
            #expect(a == b)
            #expect(a.data.first?.created != nil)
            #expect(a.data.last?.created == 77)
        }

        @Test(
            "Responses tool configuration survives collection, retrieval and every stream snapshot",
            arguments: ["named", "auto", "tools_default", "default"])
        func responseConfiguration(mode: String) async throws {
            let raw: String
            switch mode {
            case "named":
                raw =
                    #"{"model":"fixture","input":"hi","tools":[{"type":"function","name":"report","description":"Record result","parameters":{"type":"object"},"strict":true}],"tool_choice":{"type":"function","name":"report"},"parallel_tool_calls":false}"#
            case "auto":
                raw =
                    #"{"model":"fixture","input":"hi","tools":[{"type":"function","name":"report","parameters":{"type":"object"}}],"tool_choice":"auto"}"#
            case "tools_default":
                raw =
                    #"{"model":"fixture","input":"hi","tools":[{"type":"function","name":"report","parameters":{"type":"object"}}]}"#
            default:
                raw = #"{"model":"fixture","input":"hi"}"#
            }
            let request = try JSONDecoder().decode(OpenAIResponseRequest.self, from: Data(raw.utf8))
            let requestFields = try JSONDecoder().decode(
                [String: JSONValue].self, from: Data(raw.utf8))
            let expected: [String: JSONValue] = [
                "tools": requestFields["tools"] ?? .array([]),
                "tool_choice": requestFields["tool_choice"]
                    ?? .string(mode == "tools_default" ? "auto" : "none"),
                "parallel_tool_calls": requestFields["parallel_tool_calls"] ?? .bool(true),
            ]
            let store = InMemoryResponseStore()
            let service = MLXOpenAIService(engine: ObjectFixtureEngine(), responseStore: store)
            let response = try await service.createResponse(request: request)
            assertConfiguration(response, expected: expected)
            let other = MLXOpenAIService(engine: ObjectFixtureEngine(), responseStore: store)
            let retrieved = try await other.retrieveResponse(id: response.id)
            #expect(response == retrieved)
            assertConfiguration(retrieved, expected: expected)
            for terminal in ["stop", "length", "error"] {
                let streaming = MLXOpenAIService(engine: ObjectFixtureEngine(terminal: terminal))
                var streamed = request
                streamed.stream = true
                var types: [String] = []
                for try await frame in try await streaming.streamResponseFrames(request: streamed) {
                    for line in frame.split(separator: "\n") where line.hasPrefix("data: ") {
                        struct Event: Decodable {
                            let type: String
                            let response: OpenAIResponse?
                        }
                        let event = try JSONDecoder().decode(
                            Event.self, from: Data(line.dropFirst(6).utf8))
                        if let snapshot = event.response {
                            types.append(event.type)
                            assertConfiguration(snapshot, expected: expected)
                        }
                    }
                }
                #expect(
                    types == [
                        "response.created", "response.in_progress",
                        "response.\(terminal == "error" ? "failed" : terminal == "length" ? "incomplete" : "completed")",
                    ])
            }
        }

        private func assertConfiguration(_ response: OpenAIResponse, expected: [String: JSONValue])
        {
            do {
                let object = try JSONDecoder().decode(
                    [String: JSONValue].self, from: JSONEncoder.openAIServer.encode(response))
                for key in ["tools", "tool_choice", "parallel_tool_calls"] {
                    #expect(object[key] == expected[key])
                }
            } catch { Issue.record("Response encoding failed: \(error)") }
        }
    }
}

private struct ObjectFixtureEngine: MLXServerEngine {
    var terminal = "stop"
    func availableModels() async throws -> [MLXServerModel] {
        [.init(id: "fixture"), .init(id: "known", created: 77)]
    }
    func streamChatCompletion(request: OpenAIChatCompletionRequest) async throws
        -> AsyncThrowingStream<MLXServerGenerationEvent, Error>
    {
        AsyncThrowingStream { continuation in
            continuation.yield(.content("42"))
            if terminal == "error" {
                continuation.finish(throwing: ObjectFixtureError.failed)
            } else {
                continuation.yield(
                    .info(
                        .init(
                            promptTokens: 10, completionTokens: 8,
                            promptTime: 0, generationTime: 0, stopReason: terminal)))
                continuation.finish()
            }
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
private enum ObjectFixtureError: Error { case failed }
