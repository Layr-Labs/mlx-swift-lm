import Foundation

/// HTTP-only terminal failure framing. Never serialize an arbitrary error's
/// description: it may contain request data, paths, credentials or backend text.
enum ChatStreamFailure {
    private struct Frame: Encodable {
        let id: String
        let object = "chat.completion.chunk"
        let created: Int
        let model: String
        let choices: [OpenAIChatCompletionChunk.Choice]
        let usage: OpenAIUsage?
        let error: OpenAIErrorResponse.Payload
    }

    static func encode(
        _ error: any Error, id: String, model: String, created: Int, usage: OpenAIUsage?
    ) throws -> String {
        let code = error as? MLXOpenAIServiceError == .multipleToolCallsNotAllowed
            ? "tool_noncompliance" : "server_error"
        return try ServerSentEventEncoder.encode(Frame(
            id: id, created: created, model: model,
            choices: [.init(index: 0,
                delta: .init(role: nil, content: "", reasoningContent: nil, toolCalls: nil),
                finishReason: "error")],
            usage: usage,
            error: OpenAIErrorResponse(message: "Response generation failed", code: code).error))
    }
}
