import Foundation
import Hummingbird
import HummingbirdTesting
import MLXLMCommon
import Testing
@testable import MLXLMServer

struct ChatStreamingFailureHTTPTests {
    @Test(arguments: ["/v1/chat/completions", "/v1/completions"])
    func committedFailureIsACompleteSanitizedSSEBody(path: String) async throws {
        for phase in 0..<4 {
            let service = MLXOpenAIService(engine: FailingChatEngine(phase: phase))
            let app = MLXServerApplication.buildApplication(service: service,
                host: "127.0.0.1", port: 0)
            try await app.test(.router) { client in
                let input = path == "/v1/completions" ? #""prompt":"hi""#
                    : #""messages":[{"role":"user","content":"hi"}]"#
                do {
                    try await client.execute(uri: path, method: .post,
                        headers: [.contentType: "application/json"], body: ByteBuffer(string:
                            #"{"model":"fixture","stream":true,\#(input)}"#)) { response in
                        #expect(response.status == .ok)
                        let text = String(buffer: response.body)
                        #expect(!text.contains("sensitive_failure_detail"))
                        #expect(!text.contains("[DONE]"), "failed generation is not a successful completion")
                        let frames = try text.components(separatedBy: "\n")
                            .filter { $0.hasPrefix("data: ") }
                            .map { try JSONSerialization.jsonObject(with: Data($0.dropFirst(6).utf8)) as! [String: Any] }
                        let failure = try #require(frames.last)
                        #expect(failure["object"] as? String == "chat.completion.chunk")
                        #expect(failure["model"] as? String == "fixture")
                        #expect(!(failure["id"] as? String ?? "").isEmpty)
                        let detail = try #require(failure["error"] as? [String: Any])
                        #expect(detail["message"] as? String == "Response generation failed")
                        #expect(detail["code"] as? String == "server_error")
                        let choices = try #require(failure["choices"] as? [[String: Any]])
                        #expect(choices.count == 1)
                        #expect(choices.first?["finish_reason"] as? String == "error")
                        #expect(frames.filter { $0["error"] != nil }.count == 1)
                    }
                } catch {
                    Issue.record("committed HTTP body must terminate with an error frame, not throw: \(type(of: error))")
                }
            }
        }
    }

    @Test func directServiceAndNonstreamingRetainThrowingContract() async throws {
        let service = MLXOpenAIService(engine: FailingChatEngine(phase: 1))
        let request = OpenAIChatCompletionRequest(model: "fixture",
            messages: [.init(role: .user, content: .text("hi"))])
        await #expect(throws: ChatFailureProbe.self) {
            _ = try await service.createChatCompletion(request: request)
        }
        // Preserve the original method's function type as well as ordinary
        // call syntax; a new defaulted parameter alone would break this use.
        let originalEntry: (OpenAIChatCompletionRequest) async throws
            -> AsyncThrowingStream<String, Error> = service.streamChatCompletionFrames(request:)
        let stream = try await originalEntry(request)
        await #expect(throws: ChatFailureProbe.self) {
            for try await _ in stream {}
        }
    }

    @Test func cancellationNeverBecomesASuccessOrErrorFrame() async throws {
        let service = MLXOpenAIService(engine: FailingChatEngine(phase: 1, cancels: true))
        let request = OpenAIChatCompletionRequest(model: "fixture",
            messages: [.init(role: .user, content: .text("hi"))])
        let stream = try await service.streamChatCompletionFrames(
            request: request, frameGenerationErrors: true)
        var frames: [String] = []
        await #expect(throws: CancellationError.self) {
            for try await frame in stream { frames.append(frame) }
        }
        #expect(!frames.contains { $0.contains("\"error\"") || $0.contains("[DONE]") })
    }

    @Test(arguments: [false, true])
    func errorUsageOnlyContainsObservedRequestedAccounting(includeUsage: Bool) async throws {
        let service = MLXOpenAIService(engine: FailingChatEngine(phase: 3))
        let request = OpenAIChatCompletionRequest(model: "fixture",
            messages: [.init(role: .user, content: .text("hi"))],
            streamOptions: .init(includeUsage: includeUsage, continuousUsageStats: nil))
        let stream = try await service.streamChatCompletionFrames(
            request: request, frameGenerationErrors: true)
        var frames: [String] = []
        for try await frame in stream { frames.append(frame) }
        let last = try #require(frames.last)
        let event = try #require(JSONSerialization.jsonObject(
            with: Data(last.dropFirst(6).utf8)) as? [String: Any])
        let usage = event["usage"] as? [String: Any]
        if includeUsage {
            #expect(usage?["prompt_tokens"] as? Int == 2)
            #expect(usage?["completion_tokens"] as? Int == 3)
        } else {
            #expect(usage == nil)
        }
    }
}

private struct ChatFailureProbe: Error, LocalizedError {
    var errorDescription: String? { "sensitive_failure_detail" }
}

private struct FailingChatEngine: MLXServerEngine {
    let phase: Int
    var cancels = false
    func availableModels() async throws -> [MLXServerModel] { [.init(id: "fixture")] }
    func streamChatCompletion(request: OpenAIChatCompletionRequest) async throws
        -> AsyncThrowingStream<MLXServerGenerationEvent, Error>
    {
        AsyncThrowingStream { continuation in
            if phase >= 1 { continuation.yield(.parsed(.init(content: "", reasoningContent: "checking"))) }
            if phase >= 2 { continuation.yield(.content("partial")) }
            if phase >= 3 {
                continuation.yield(.toolCall(.init(function: .init(name: "lookup", arguments: ["value": .string("opaque")]))))
                continuation.yield(.info(.init(promptTokens: 2, completionTokens: 3,
                    promptTime: 0.01, generationTime: 0.01, stopReason: "length")))
            }
            if cancels { continuation.finish(throwing: CancellationError()) }
            else { continuation.finish(throwing: ChatFailureProbe()) }
        }
    }
    func tokenize(_ request: TokenizeRequest) async throws -> TokenizeResponse { .init(tokens: [1]) }
    func detokenize(_ request: DetokenizeRequest) async throws -> DetokenizeResponse { .init(text: "ok") }
    func applyTemplate(_ request: ApplyTemplateRequest) async throws -> TokenizeResponse { .init(tokens: [1]) }
}
