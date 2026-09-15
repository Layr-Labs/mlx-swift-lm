import Foundation
import Testing

@testable import MLXLMServer

@Suite("Responses assistant-turn replay")
struct OpenAIResponsesInputReplayTests {
    private func decode(
        _ input: [[String: Any]], effort: String = "high"
    ) throws -> OpenAIResponseRequest {
        let data = try JSONSerialization.data(withJSONObject: [
            "model": "native-qwen4-test",
            "reasoning": ["effort": effort],
            "input": input,
        ])
        return try JSONDecoder().decode(OpenAIResponseRequest.self, from: data)
    }

    private func reasoning(_ text: String) -> [String: Any] {
        ["type": "reasoning", "summary": [["type": "summary_text", "text": text]]]
    }

    private func call(_ id: String, arguments: String = "{}") -> [String: Any] {
        ["type": "function_call", "call_id": id, "name": "record", "arguments": arguments]
    }

    private func output(_ id: String, text: String = "ok") -> [String: Any] {
        ["type": "function_call_output", "call_id": id, "output": text]
    }

    @Test func reasoningAndParallelCallsReplayAsOneAssistantTurn() throws {
        let thought = "Use two calls; <think>literal example</think> café e\u{301} 雪"
        let arguments = #"{"text":"quotes \"hi\", backslash \\, <think>data</think>, café e\u0301 雪"}"#
        let request = try decode([
            ["role": "user", "content": "record both"],
            reasoning(thought), call("call_a", arguments: arguments), call("call_b"),
            output("call_a", text: "first"), output("call_b", text: "second"),
        ])
        let chat = request.chatCompletionRequest
        #expect(chat.model == "native-qwen4-test")
        #expect(chat.reasoning?.effort == "high")
        #expect(chat.messages.map(\.role) == [.user, .assistant, .tool, .tool])
        let assistant = try #require(chat.messages.first { $0.toolCalls != nil })
        #expect(assistant.textContent.isEmpty)
        #expect(assistant.reasoningContent == thought)
        #expect(assistant.toolCalls?.map(\.id) == ["call_a", "call_b"])
        #expect(assistant.toolCalls?.first?.function.arguments.utf8.elementsEqual(arguments.utf8) == true)
        #expect(chat.messages.filter { $0.role == .tool }.map(\.toolCallID) == ["call_a", "call_b"])
        let roundTrip = try JSONDecoder().decode(
            OpenAIResponseRequest.self, from: JSONEncoder().encode(request))
        #expect(roundTrip == request)
    }

    @Test func emptyOrOmittedReasoningSummaryStillOwnsItsCalls() throws {
        for item: [String: Any] in [
            ["type": "reasoning"],
            ["type": "reasoning", "summary": []],
            reasoning(""),
        ] {
            let messages = try decode([item, call("c")]).chatCompletionRequest.messages
            #expect(messages.count == 1)
            #expect(messages.last?.reasoningContent == "")
            #expect(messages.last?.toolCalls?.map(\.id) == ["c"])
        }
    }

    @Test func toolResultsSeparateSuccessiveAssistantTurns() throws {
        let messages = try decode([
            reasoning("first"), call("a"), output("a"),
            reasoning("second"), call("b"), output("b"),
        ]).chatCompletionRequest.messages
        #expect(messages.map(\.role) == [.assistant, .tool, .assistant, .tool])
        let assistants = messages.filter { $0.role == .assistant }
        #expect(assistants.map(\.reasoningContent) == ["first", "second"])
        #expect(assistants.map { $0.toolCalls?.map(\.id) } == [["a"], ["b"]])
    }

    @Test func explicitChatAssistantTurnsAreNotMergedJustForHavingReasoning() throws {
        let original: [String: Any] = [
            "role": "assistant", "content": "a separate answer",
            "name": "assistant_one", "reasoning_content": "a separate thought",
        ]
        let messages = try decode([original, call("a")]).chatCompletionRequest.messages
        #expect(messages.count == 2)
        #expect(messages[0].textContent == "a separate answer")
        #expect(messages[0].name == "assistant_one")
        #expect(messages[0].reasoningContent == "a separate thought")
        #expect(messages[0].toolCalls == nil)
        #expect(messages[1].toolCalls?.map(\.id) == ["a"])
    }

    @Test func explicitMessageItemIsABoundaryAfterReasoning() throws {
        let message: [String: Any] = [
            "type": "message", "role": "assistant", "content": "",
            "tool_calls": [[
                "id": "a", "type": "function",
                "function": ["name": "record", "arguments": "{}"],
            ]],
        ]
        let messages = try decode([reasoning("prior"), message]).chatCompletionRequest.messages
        #expect(messages.count == 2)
        #expect(messages[0].reasoningContent == "prior")
        #expect(messages[0].toolCalls == nil)
        #expect(messages[1].reasoningContent == nil)
        #expect(messages[1].toolCalls?.map(\.id) == ["a"])
    }

    @Test func interveningMessagesAndToolOutputsPreventReasoningMerge() throws {
        let boundaries: [[String: Any]] = [
            ["role": "user", "content": "new user turn"],
            ["role": "system", "content": "new system turn"],
            ["role": "assistant", "content": "separate answer"],
            output("prior"),
        ]
        for boundary in boundaries {
            let messages = try decode([
                reasoning("earlier"), boundary, call("next"),
            ]).chatCompletionRequest.messages
            #expect(messages.count == 3)
            #expect(messages[0].reasoningContent == "earlier")
            #expect(messages[0].toolCalls == nil)
            #expect(messages.last?.reasoningContent == nil)
            #expect(messages.last?.toolCalls?.map(\.id) == ["next"])
        }
    }

    @Test func nonReasoningParallelCallsKeepExistingGrouping() throws {
        let request = try decode([call("a"), call("b"), output("a"), output("b")], effort: "none")
        let chat = request.chatCompletionRequest
        #expect(chat.reasoning?.effort == "none")
        #expect(chat.messages.map(\.role) == [.assistant, .tool, .tool])
        #expect(chat.messages[0].reasoningContent == nil)
        #expect(chat.messages[0].toolCalls?.map(\.id) == ["a", "b"])
    }

    @Test func mediaAndToolOutputPartsRemainUnchanged() throws {
        let image = "data:image/png;base64,fixture"
        let video = "data:video/mp4;base64,fixture"
        let media: [[String: Any]] = [
            ["type": "input_text", "text": "inspect"],
            ["type": "image_url", "image_url": ["url": image]],
            ["type": "video_url", "video_url": ["url": video]],
        ]
        let messages = try decode([
            ["role": "user", "content": media],
            reasoning("check media"), call("a"),
            ["type": "function_call_output", "call_id": "a", "output": media],
        ]).chatCompletionRequest.messages
        #expect(messages.map(\.role) == [.user, .assistant, .tool])
        let expected: OpenAIMessageContent = .parts([.text("inspect"), .imageURL(image), .videoURL(video)])
        #expect(messages.first?.content == expected)
        #expect(messages.last?.content == expected)
        #expect(messages.last?.toolCallID == "a")
    }

    @Test func malformedHistoryStillFailsClosed() throws {
        for item in [
            #"{"type":"function_call","name":"record","arguments":"{}"}"#,
            #"{"type":"function_call","call_id":"a","name":"record","arguments":{}}"#,
            #"{"type":"function_call_output","call_id":"","output":"ok"}"#,
            #"{"type":"reasoning","summary":"not-an-array"}"#,
            #"{"type":"unknown","role":"assistant","content":"keep me"}"#,
        ] {
            let data = Data("{\"model\":\"native-qwen4-test\",\"input\":[\(item)]}".utf8)
            #expect(throws: DecodingError.self) {
                try JSONDecoder().decode(OpenAIResponseRequest.self, from: data)
            }
        }
    }

    @Test func plainTextAndExplicitChatHistoryRoundTripUnchanged() throws {
        let text = OpenAIResponseRequest(model: "native-qwen4-test", input: .text("hello"))
        let decoded = try JSONDecoder().decode(OpenAIResponseRequest.self, from: JSONEncoder().encode(text))
        #expect(decoded == text)
        #expect(decoded.chatCompletionRequest.messages == [.init(role: .user, content: .text("hello"))])
        let chat = try decode([
            ["role": "user", "content": "hello"],
            ["role": "assistant", "content": "hello back", "reasoning_content": "private"],
            ["role": "user", "content": "next"],
        ])
        #expect(chat.chatCompletionRequest.messages.map(\.role) == [.user, .assistant, .user])
        #expect(chat.chatCompletionRequest.messages[1].reasoningContent == "private")
        #expect(chat.chatCompletionRequest.messages[1].toolCalls == nil)
    }
}
