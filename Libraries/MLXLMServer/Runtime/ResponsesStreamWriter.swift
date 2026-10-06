import Foundation
import MLXLMCommon

/// Responses SSE framing over already classified generation events. It does not
/// parse tool arguments or run a second reasoning parser on native channels.
struct ResponsesStreamWriter {
    private let request: OpenAIResponseRequest
    private let idProvider: @Sendable (String) -> String
    private var response: OpenAIResponse
    private var openItem: OpenAIResponseOutputItem?
    private var sequence = 0
    private var pending: [String] = []
    private var toolCallCount = 0
    private var finishReason = "stop"

    init(request: OpenAIResponseRequest, idProvider: @escaping @Sendable (String) -> String) {
        self.request = request
        self.idProvider = idProvider
        response = .init(id: idProvider("resp"), status: .inProgress,
            model: request.model, output: [], outputText: "", usage: nil, metadata: request.metadata)
    }

    mutating func start() throws {
        try emit("response.created", ["response": encoded(response)])
        try emit("response.in_progress", ["response": encoded(response)])
    }

    mutating func append(_ parsed: ParsedReasoning) throws {
        if let text = parsed.reasoningContent, !text.isEmpty {
            try appendText(text, reasoning: true)
        }
        if !parsed.content.isEmpty { try appendText(parsed.content, reasoning: false) }
    }

    mutating func append(_ call: ToolCall) throws {
        if request.parallelToolCalls == false, toolCallCount > 0 {
            throw MLXOpenAIServiceError.multipleToolCallsNotAllowed
        }
        try closeItem()
        let wire = try OpenAIToolCall(toolCall: call, id: idProvider("call"))
        var item = OpenAIResponseOutputItem.functionCall(id: idProvider("fc"), toolCall: wire)
        item.status = .inProgress
        item.arguments = ""
        openItem = item
        try emit("response.output_item.added", ["output_index": .int(response.output.count), "item": encoded(item)])
        if !wire.function.arguments.isEmpty {
            try emit("response.function_call_arguments.delta", itemFields(item, ["delta": .string(wire.function.arguments)]))
        }
        openItem?.arguments = wire.function.arguments
        toolCallCount += 1
        try closeItem()
    }

    mutating func update(_ info: ServerGenerationInfo) {
        response.usage = .init(inputTokens: info.promptTokens, outputTokens: info.completionTokens,
            cachedInputTokens: info.cachedPromptTokens)
        finishReason = info.stopReason
    }

    mutating func finish() throws -> OpenAIResponse {
        try closeItem()
        let incomplete = toolCallCount == 0 && ["length", "content_filter"].contains(finishReason)
        response.status = incomplete ? .incomplete : .completed
        if incomplete {
            response.incompleteDetails = .init(reason: finishReason == "length" ? "max_output_tokens" : finishReason)
        }
        try emit(incomplete ? "response.incomplete" : "response.completed", ["response": encoded(response)])
        return response
    }

    mutating func fail(_ error: any Error) throws -> OpenAIResponse {
        // Partial items stay incomplete when inference fails. Do not fabricate
        // successful item.done or response.completed events for that output.
        if var item = openItem {
            item.status = .incomplete
            response.output.append(item)
            openItem = nil
        }
        response.status = .failed
        response.error = .init(
            code: error as? MLXOpenAIServiceError == .multipleToolCallsNotAllowed ? "tool_noncompliance" : "server_error",
            message: "Response generation failed")
        try emit("response.failed", ["response": encoded(response)])
        return response
    }

    mutating func takeFrames() -> [String] {
        let frames = pending
        pending.removeAll(keepingCapacity: true)
        return frames
    }

    private mutating func appendText(_ text: String, reasoning: Bool) throws {
        let type = reasoning ? "reasoning" : "message"
        if openItem?.type != type {
            try closeItem()
            var item: OpenAIResponseOutputItem = reasoning
                ? .reasoning(id: idProvider("rs"), text: "")
                : .message(id: idProvider("msg"), text: "")
            item.status = .inProgress
            if reasoning { item.summary = []; item.content = nil }
            else { item.content = [] }
            openItem = item
            try emit("response.output_item.added", ["output_index": .int(response.output.count), "item": encoded(item)])
            let part = OpenAIResponseOutputContent(type: reasoning ? "summary_text" : "output_text",
                text: "", annotations: reasoning ? nil : [])
            var fields = try itemFields(item, ["part": encoded(part)])
            fields[reasoning ? "summary_index" : "content_index"] = .int(0)
            try emit(reasoning ? "response.reasoning_summary_part.added" : "response.content_part.added", fields)
            if reasoning { openItem?.summary = [part] }
            else { openItem?.content = [part] }
        }
        guard let item = openItem else { return }
        if reasoning { openItem?.summary?[0].text += text }
        else { openItem?.content?[0].text += text; response.outputText += text }
        var fields = itemFields(item, ["delta": .string(text)])
        fields[reasoning ? "summary_index" : "content_index"] = .int(0)
        if !reasoning { fields["logprobs"] = .array([]) }
        try emit(reasoning ? "response.reasoning_summary_text.delta" : "response.output_text.delta", fields)
    }

    private mutating func closeItem() throws {
        guard var item = openItem else { return }
        if item.type == "function_call" {
            try emit("response.function_call_arguments.done", itemFields(item, ["arguments": .string(item.arguments ?? "")]))
        } else {
            let reasoning = item.type == "reasoning"
            let part = reasoning ? item.summary![0] : item.content![0]
            var fields = itemFields(item, ["text": .string(part.text)])
            fields[reasoning ? "summary_index" : "content_index"] = .int(0)
            if !reasoning { fields["logprobs"] = .array([]) }
            try emit(reasoning ? "response.reasoning_summary_text.done" : "response.output_text.done", fields)
            fields.removeValue(forKey: "text")
            fields.removeValue(forKey: "logprobs")
            fields["part"] = try encoded(part)
            try emit(reasoning ? "response.reasoning_summary_part.done" : "response.content_part.done", fields)
            // Match the existing non-streaming reasoning representation.
            if reasoning { item.content = [.init(type: "reasoning_text", text: part.text)] }
        }
        item.status = .completed
        try emit("response.output_item.done", ["output_index": .int(response.output.count), "item": encoded(item)])
        response.output.append(item)
        openItem = nil
    }

    private func itemFields(_ item: OpenAIResponseOutputItem, _ extra: [String: JSONValue]) -> [String: JSONValue] {
        ["item_id": .string(item.id), "output_index": .int(response.output.count)].merging(extra) { _, new in new }
    }

    private mutating func emit(_ type: String, _ fields: [String: JSONValue]) throws {
        var event = fields
        event["type"] = .string(type)
        event["sequence_number"] = .int(sequence)
        sequence += 1
        pending.append("event: \(type)\n" + (try ServerSentEventEncoder.encode(event)))
    }

    private func encoded<T: Encodable>(_ value: T) throws -> JSONValue {
        try JSONDecoder().decode(JSONValue.self, from: JSONEncoder.openAIServer.encode(value))
    }
}
