import Foundation
import Hummingbird
import HummingbirdTesting
import MLXLMCommon
@testable import MLXLMServer
import Testing

@Suite("Responses wire format and tool history")
struct OpenAIResponsesCompatibilityTests {
    @Test func standardToolsNamedChoiceAndNonReasoningHistoryReachChatUnchanged() throws {
        let arguments = #"{"text":"a \\\"quote\\\" and \\\\ café 雪"}"#
        let data = try JSONSerialization.data(withJSONObject: [
            "model": "qwen4-test", "reasoning": ["effort": "none"],
            "tools": [["type": "function", "name": "record", "parameters": ["type": "object"]]],
            "tool_choice": ["type": "function", "name": "record"],
            "input": [
                ["role": "user", "content": [["type": "input_text", "text": "record this"]]],
                ["type": "function_call", "call_id": "call_a", "name": "record", "arguments": arguments],
                ["type": "function_call", "call_id": "call_b", "name": "record", "arguments": "{}"],
                ["type": "function_call_output", "call_id": "call_a", "output": "first result"],
                ["type": "function_call_output", "call_id": "call_b", "output": [["type": "input_text", "text": "second result"]]],
                ["type": "message", "role": "assistant", "content": [["type": "output_text", "text": "recorded"]]],
            ],
        ])
        let request = try JSONDecoder().decode(OpenAIResponseRequest.self, from: data)
        let chat = request.chatCompletionRequest
        #expect(chat.reasoning?.effort == "none")
        #expect(chat.tools?.first?.function.name == "record")
        #expect(chat.toolChoice == .function(name: "record"))
        #expect(chat.messages.count == 5)
        #expect(chat.messages[0].textContent == "record this")
        #expect(chat.messages[1].toolCalls?.map(\.id) == ["call_a", "call_b"])
        #expect(chat.messages[1].toolCalls?.first?.function.arguments == arguments)
        #expect(chat.messages[2].role == .tool && chat.messages[2].toolCallID == "call_a")
        #expect(chat.messages[3].textContent == "second result")
        #expect(chat.messages[4].textContent == "recorded")
        let decoded = try JSONDecoder().decode(OpenAIResponseRequest.self, from: JSONEncoder().encode(request))
        #expect(decoded == request)
    }

    @Test func chatShapedHistoryStillDecodesAndUnknownItemsFail() throws {
        let legacy = #"{"model":"m","input":[{"role":"assistant","content":"","tool_calls":[{"id":"c","type":"function","function":{"name":"record","arguments":"{}"}}]},{"role":"tool","tool_call_id":"c","content":"ok"}]}"#
        let request = try JSONDecoder().decode(OpenAIResponseRequest.self, from: Data(legacy.utf8))
        #expect(request.chatCompletionRequest.messages[0].toolCalls?.first?.id == "c")
        for item in [
            #"{"type":"function_call","name":"f","arguments":"{}"}"#,
            #"{"type":"function_call","call_id":"c","name":"f","arguments":{}}"#,
            #"{"type":"function_call_output","call_id":"","output":"ok"}"#,
            #"{"type":"unknown","role":"user","content":"do not silently drop"}"#,
        ] {
            let data = Data("{\"model\":\"m\",\"input\":[\(item)]}".utf8)
            #expect(throws: DecodingError.self) { try JSONDecoder().decode(OpenAIResponseRequest.self, from: data) }
        }
    }

    @Test func nonReasoningToolStreamsResponsesEventsAndStoresSameFinalObject() async throws {
        let value = #"literal "quotes" \ and <think>data</think> 雪"#
        let service = MLXOpenAIService(engine: ResponsesScriptedEngine(events: [
            .toolCall(.init(function: .init(name: "record", arguments: ["text": .string(value)]))),
            .info(info(prompt: 41, output: 12)),
        ]))
        let request = OpenAIResponseRequest(model: "qwen4-test", input: .text("record"),
            reasoning: .init(effort: "none"), stream: true)
        let frames = try await collect(service.streamResponseFrames(request: request))
        let events = try decode(frames)
        let types = events.compactMap { $0["type"] as? String }
        #expect(types == ["response.created", "response.in_progress", "response.output_item.added",
            "response.function_call_arguments.delta", "response.function_call_arguments.done",
            "response.output_item.done", "response.completed"])
        #expect(events.compactMap { $0["sequence_number"] as? Int } == Array(events.indices))
        #expect(!frames.joined().contains("chat.completion.chunk"))
        #expect(!frames.joined().contains("[DONE]"))
        let response = try finalResponse(events)
        #expect(response.output.map(\.type) == ["function_call"])
        let call = try #require(response.output.first)
        let args = try #require(call.arguments)
        #expect(try JSONDecoder().decode([String: String].self, from: Data(args.utf8)) == ["text": value])
        #expect(response.usage?.inputTokens == 41 && response.usage?.outputTokens == 12)
        #expect(try await service.retrieveResponse(id: response.id) == response)
        #expect(events[2]["output_index"] as? Int == 0)
        #expect((events[2]["item"] as? [String: Any])?["call_id"] as? String == call.callID)
    }

    @Test func streamAndCollectionPreserveNativeChannelsWithoutReparsing() async throws {
        let source: [MLXServerGenerationEvent] = [
            .parsed(.init(content: "", reasoningContent: "private thought")),
            .parsed(.init(content: "literal <think>visible</think>", reasoningContent: nil)),
            .info(info(prompt: 10, output: 4)),
        ]
        let service = MLXOpenAIService(engine: ResponsesScriptedEngine(events: source), defaultReasoningParser: .qwen3)
        let request = OpenAIResponseRequest(model: "qwen4-test", input: .text("hello"), stream: true, store: false)
        let events = try decode(await collect(service.streamResponseFrames(request: request)))
        let streamed = try finalResponse(events)
        let collected = try await service.createResponse(request: request)
        #expect(streamed.outputText == collected.outputText)
        #expect(streamed.output.map(\.type) == collected.output.map(\.type))
        #expect(streamed.output.first?.summary == collected.output.first?.summary)
        #expect(streamed.output.last?.content == collected.output.last?.content)
        #expect(streamed.outputText == "literal <think>visible</think>")
        #expect(events.contains { $0["type"] as? String == "response.reasoning_summary_text.delta" })
        #expect(events.contains { $0["type"] as? String == "response.output_text.delta" })
        await #expect(throws: MLXOpenAIServiceError.responseNotFound(streamed.id)) {
            try await service.retrieveResponse(id: streamed.id)
        }
    }

    @Test func lengthIsIncompleteInStreamingAndCollection() async throws {
        let service = MLXOpenAIService(engine: ResponsesScriptedEngine(events: [
            .parsed(.init(content: "partial", reasoningContent: nil)),
            .info(info(prompt: 7, output: 1, stop: "length")),
        ]))
        let request = OpenAIResponseRequest(model: "qwen4-test", input: .text("hello"), stream: true)
        let events = try decode(await collect(service.streamResponseFrames(request: request)))
        let streamed = try finalResponse(events)
        let collected = try await service.createResponse(request: request)
        #expect(events.last?["type"] as? String == "response.incomplete")
        #expect(events.filter { $0["type"] as? String == "response.incomplete" }.count == 1)
        #expect(!events.contains { $0["type"] as? String == "response.completed" })
        #expect(streamed.status == .incomplete && collected.status == .incomplete)
        #expect(streamed.incompleteDetails?.reason == "max_output_tokens")
        #expect(streamed.incompleteDetails == collected.incompleteDetails)
    }

    @Test func nativeNonReasoningTextIsDeliveredBeforeGenerationFinishes() async throws {
        let (source, upstream) = AsyncThrowingStream<MLXServerGenerationEvent, Error>.makeStream()
        defer { upstream.finish() }
        let service = MLXOpenAIService(engine: ResponsesHeldEngine(source: source), defaultReasoningParser: .qwen3)
        let frames = try await service.streamResponseFrames(request: .init(model: "m", input: .text("hi"),
            reasoning: .init(effort: "none"), stream: true))
        upstream.yield(.parsed(.init(content: "hello", reasoningContent: nil)))
        let observed = await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                do {
                    for try await frame in frames {
                        if frame.hasPrefix("event: response.output_text.delta\n") { return true }
                    }
                } catch { }
                return false
            }
            group.addTask { try? await Task.sleep(for: .seconds(2)); return false }
            let observed = await group.next() ?? false
            group.cancelAll()
            return observed
        }
        #expect(observed, "non-reasoning content must stream before the engine reaches EOF")
    }

    @Test func splitReasoningThenParallelToolsHasStableOrderedItems() async throws {
        let service = MLXOpenAIService(engine: ResponsesScriptedEngine(events: [
            .content("<thi"), .content("nk>think once</think>"),
            .toolCall(.init(function: .init(name: "a", arguments: [:]))),
            .toolCall(.init(function: .init(name: "b", arguments: [:]))),
            .info(info(prompt: 5, output: 3)),
        ]), defaultReasoningParser: .qwen3)
        let events = try decode(await collect(service.streamResponseFrames(request: .init(model: "m", input: .text("hi"), stream: true))))
        let response = try finalResponse(events)
        #expect(response.output.map(\.type) == ["reasoning", "function_call", "function_call"])
        #expect(response.output.first?.summary?.first?.text == "think once")
        #expect(response.output.compactMap(\.name) == ["a", "b"])
        #expect(Set(response.output.compactMap(\.callID)).count == 2)
        let added = events.filter { $0["type"] as? String == "response.output_item.added" }
        #expect(added.compactMap { $0["output_index"] as? Int } == [0, 1, 2])
        #expect(response.outputText.isEmpty)

        // Clients replay the emitted items before appending tool results.
        // The next prompt must retain one assistant turn with its reasoning
        // and both calls, rather than inserting a reasoning-only turn.
        var replay = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(response.output))
                as? [[String: Any]])
        for callID in response.output.compactMap(\.callID) {
            replay.append(["type": "function_call_output", "call_id": callID, "output": "ok"])
        }
        let replayData = try JSONSerialization.data(withJSONObject: ["model": "m", "input": replay])
        let replayRequest = try JSONDecoder().decode(OpenAIResponseRequest.self, from: replayData)
        let messages = replayRequest.chatCompletionRequest.messages
        #expect(messages.map(\.role) == [.assistant, .tool, .tool])
        #expect(messages.first?.reasoningContent == "think once")
        #expect(messages.first?.toolCalls?.map(\.id) == response.output.compactMap(\.callID))
        #expect(messages.first?.toolCalls?.map(\.function.name) == ["a", "b"])
        #expect(messages.filter { $0.role == .tool }.map(\.toolCallID)
            == response.output.compactMap(\.callID))
    }

    @Test func cancelledResponseConsumptionTerminatesTheUpstreamStream() async throws {
        let (source, upstream) = AsyncThrowingStream<MLXServerGenerationEvent, Error>.makeStream()
        let (terminations, notify) = AsyncStream<Bool>.makeStream()
        upstream.onTermination = { reason in
            if case .cancelled = reason { notify.yield(true) }
            else { notify.yield(false) }
            notify.finish()
        }
        defer { upstream.finish(); notify.finish() }
        let service = MLXOpenAIService(engine: ResponsesHeldEngine(source: source))
        let frames = try await service.streamResponseFrames(request: .init(model: "m", input: .text("hi"), stream: true))
        let consumer = Task {
            do { for try await _ in frames { } } catch { }
        }
        upstream.yield(.parsed(.init(content: "first", reasoningContent: nil)))
        consumer.cancel()
        await consumer.value
        let cancelled = await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                for await cancelled in terminations { return cancelled }
                return false
            }
            group.addTask { try? await Task.sleep(for: .seconds(2)); return false }
            let observed = await group.next() ?? false
            group.cancelAll()
            return observed
        }
        #expect(cancelled, "consumer cancellation must reach the engine stream's termination handler")
    }

    @Test func generationFailureHasFailedTerminalAndNoSuccessOrPrivateErrorText() async throws {
        let service = MLXOpenAIService(engine: ResponsesScriptedEngine(events: [
            .parsed(.init(content: "partial", reasoningContent: nil))
        ], failure: true))
        let request = OpenAIResponseRequest(model: "qwen4-test", input: .text("hello"), stream: true)
        let frames = try await collect(service.streamResponseFrames(request: request))
        let events = try decode(frames)
        #expect(events.last?["type"] as? String == "response.failed")
        #expect(events.filter { $0["type"] as? String == "response.failed" }.count == 1)
        #expect(!frames.joined().contains("response.completed"))
        #expect(!frames.joined().contains("private-fixture-error"))
        let response = try finalResponse(events)
        #expect(response.status == .failed)
        #expect(response.output.first?.status == .incomplete)
        #expect(response.error?.code == "server_error")
    }

    @Test func parallelCallViolationDoesNotFinishSuccessfully() async throws {
        let service = MLXOpenAIService(engine: ResponsesScriptedEngine(events: [
            .toolCall(.init(function: .init(name: "a", arguments: [:]))),
            .toolCall(.init(function: .init(name: "b", arguments: [:]))),
        ]))
        let request = OpenAIResponseRequest(model: "qwen4-test", input: .text("hello"), parallelToolCalls: false, stream: true)
        let events = try decode(await collect(service.streamResponseFrames(request: request)))
        let response = try finalResponse(events)
        #expect(response.status == .failed)
        #expect(response.error?.code == "tool_noncompliance")
        #expect(response.output.compactMap(\.name) == ["a"])
    }

    @Test func responseRouteUsesResponsesEventsWhileChatRouteKeepsChatEvents() async throws {
        let service = MLXOpenAIService(engine: ResponsesScriptedEngine(events: [
            .parsed(.init(content: "hello", reasoningContent: nil)), .info(info(prompt: 3, output: 1)),
        ]))
        let app = MLXServerApplication.buildApplication(service: service, host: "127.0.0.1", port: 8080)
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/responses", method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"model":"m","input":"hi","stream":true,"reasoning":{"effort":"none"}}"#)) { response in
                    #expect(response.status == .ok)
                    let body = String(buffer: response.body)
                    #expect(body.contains("event: response.output_text.delta"))
                    #expect(body.contains("event: response.completed"))
                    #expect(!body.contains("chat.completion.chunk"))
                }
            try await client.execute(uri: "/v1/chat/completions", method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}"#)) { response in
                    #expect(response.status == .ok)
                    let body = String(buffer: response.body)
                    #expect(body.contains("chat.completion.chunk"))
                    #expect(body.contains("data: [DONE]"))
                    #expect(!body.contains("event: response."))
                }
        }
    }

    private func info(prompt: Int, output: Int, stop: String = "stop") -> ServerGenerationInfo {
        .init(promptTokens: prompt, completionTokens: output, promptTime: 0.1, generationTime: 0.1, stopReason: stop)
    }

    private func collect(_ stream: AsyncThrowingStream<String, Error>) async throws -> [String] {
        var frames: [String] = []
        for try await frame in stream { frames.append(frame) }
        return frames
    }

    private func decode(_ frames: [String]) throws -> [[String: Any]] {
        try frames.map { frame in
            let line = try #require(frame.split(separator: "\n").first { $0.hasPrefix("data: ") })
            return try #require(JSONSerialization.jsonObject(with: Data(line.dropFirst(6).utf8)) as? [String: Any])
        }
    }

    private func finalResponse(_ events: [[String: Any]]) throws -> OpenAIResponse {
        let object = try #require(events.last?["response"] as? [String: Any])
        return try JSONDecoder().decode(OpenAIResponse.self, from: JSONSerialization.data(withJSONObject: object))
    }
}

private struct ResponsesScriptedEngine: MLXServerEngine {
    var events: [MLXServerGenerationEvent]
    var failure = false
    func availableModels() async throws -> [MLXServerModel] { [.init(id: "qwen4-test")] }
    func streamChatCompletion(request: OpenAIChatCompletionRequest) async throws -> AsyncThrowingStream<MLXServerGenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            for event in events { continuation.yield(event) }
            if failure { continuation.finish(throwing: FixtureFailure()) }
            else { continuation.finish() }
        }
    }
    func tokenize(_ request: TokenizeRequest) async throws -> TokenizeResponse { .init(tokens: []) }
    func detokenize(_ request: DetokenizeRequest) async throws -> DetokenizeResponse { .init(text: "") }
    func applyTemplate(_ request: ApplyTemplateRequest) async throws -> TokenizeResponse { .init(tokens: []) }
    private struct FixtureFailure: Error, CustomStringConvertible { var description: String { "private-fixture-error" } }
}

private struct ResponsesHeldEngine: MLXServerEngine {
    let source: AsyncThrowingStream<MLXServerGenerationEvent, Error>
    func availableModels() async throws -> [MLXServerModel] { [] }
    func streamChatCompletion(request: OpenAIChatCompletionRequest) async throws -> AsyncThrowingStream<MLXServerGenerationEvent, Error> { source }
    func tokenize(_ request: TokenizeRequest) async throws -> TokenizeResponse { .init(tokens: []) }
    func detokenize(_ request: DetokenizeRequest) async throws -> DetokenizeResponse { .init(text: "") }
    func applyTemplate(_ request: ApplyTemplateRequest) async throws -> TokenizeResponse { .init(tokens: []) }
}
