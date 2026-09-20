import Foundation
import Hummingbird
import HummingbirdTesting
import MLXLMCommon
import Testing
@testable import MLXLMServer

struct OutputTokenLimitHTTPTests {
    @Test(arguments: [false, true])
    func negativeLimitsFailBeforeEngineOrStreamingHeaders(streaming: Bool) async throws {
        let engine = TokenLimitFixtureEngine()
        let app = MLXServerApplication.buildApplication(
            service: MLXOpenAIService(engine: engine), host: "127.0.0.1", port: 0)
        try await app.test(.router) { client in
            for value in [-1, Int.min] {
                for (path, fields) in [
                    ("/v1/chat/completions", #""messages":[{"role":"user","content":"hi"}],"max_tokens":\#(value)"#),
                    ("/v1/completions", #""prompt":"hi","max_tokens":\#(value)"#),
                    ("/v1/responses", #""input":"hi","max_output_tokens":\#(value)"#),
                ] {
                    let body = #"{"model":"fixture","stream":\#(streaming),\#(fields)}"#
                    try await client.execute(uri: path, method: .post,
                        headers: [.contentType: "application/json"], body: ByteBuffer(string: body)) { response in
                        #expect(response.status == .badRequest)
                        let text = String(buffer: response.body)
                        let error = try JSONDecoder().decode(OpenAIErrorResponse.self, from: Data(text.utf8))
                        #expect(error.error.type == "invalid_request_error")
                        #expect(error.error.message == "Output token limits must not be negative.")
                        #expect(!text.contains("data:"))
                    }
                }
            }
        }
        #expect(await engine.seen.isEmpty, "invalid budgets must not load or submit a model")
    }

    @Test func batchRejectsNegativeBeforeEngine() async throws {
        let engine = TokenLimitFixtureEngine()
        let app = MLXServerApplication.buildApplication(
            service: MLXOpenAIService(engine: engine), host: "127.0.0.1", port: 0)
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/chat/completions/batch", method: .post,
                headers: [.contentType: "application/json"], body: ByteBuffer(string:
                    #"[{"model":"fixture","messages":[{"role":"user","content":"hi"}],"max_tokens":-1}]"#)) { response in
                #expect(response.status == .badRequest)
            }
        }
        #expect(await engine.seen.isEmpty)
    }

    @Test func zeroPositiveAndOmittedLimitsRemainUnchanged() async throws {
        let engine = TokenLimitFixtureEngine()
        let service = MLXOpenAIService(engine: engine)
        for limit: Int? in [nil, 0, 1] {
            var request = OpenAIChatCompletionRequest(model: "fixture",
                messages: [.init(role: .user, content: .text("hi"))], maxTokens: limit)
            _ = try await service.createChatCompletion(request: request)
            request.stream = true
            for try await _ in try await service.streamChatCompletionFrames(request: request) {}
        }
        #expect(await engine.seen == [nil, nil, 0, 0, 1, 1])
    }
}

private actor TokenLimitFixtureEngine: MLXServerEngine {
    var seen: [Int?] = []
    func availableModels() async throws -> [MLXServerModel] { [.init(id: "fixture")] }
    func streamChatCompletion(request: OpenAIChatCompletionRequest)
        async throws -> AsyncThrowingStream<MLXServerGenerationEvent, Error>
    {
        seen.append(request.maxTokens)
        return AsyncThrowingStream { continuation in
            if request.maxTokens != 0 { continuation.yield(.content("ok")) }
            continuation.yield(.info(.init(promptTokens: 1,
                completionTokens: request.maxTokens == 0 ? 0 : 1,
                promptTime: 0.01, generationTime: 0.01,
                stopReason: request.maxTokens == 0 ? "length" : "stop")))
            continuation.finish()
        }
    }
    func tokenize(_ request: TokenizeRequest) async throws -> TokenizeResponse { .init(tokens: [1]) }
    func detokenize(_ request: DetokenizeRequest) async throws -> DetokenizeResponse { .init(text: "ok") }
    func applyTemplate(_ request: ApplyTemplateRequest) async throws -> TokenizeResponse { .init(tokens: [1]) }
}
