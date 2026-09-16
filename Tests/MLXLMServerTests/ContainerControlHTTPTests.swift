import Foundation
import Hummingbird
import HummingbirdTesting
import Testing

@testable import MLXLMServer

/// Actual HTTP/service translation with a scripted engine using the container
/// guard. This does not claim a loaded-model or native sampler execution pass.
struct ContainerControlHTTPTests {
    @Test func unsupportedControlsFailBeforeStreamingHeaders() async throws {
        let app = MLXServerApplication.buildApplication(
            service: MLXOpenAIService(engine: ControlFixtureEngine()), host: "127.0.0.1", port: 8080)
        try await app.test(.router) { client in
            for streaming in [false, true] {
                for control in [#""seed":0"#, #""logit_bias":{"0":100}"#] {
                    let body = #"{"model":"fixture","messages":[{"role":"user","content":"hi"}],"stream":\#(streaming),\#(control)}"#
                    try await client.execute(uri: "/v1/chat/completions", method: .post,
                        headers: [.contentType: "application/json"], body: ByteBuffer(string: body)) { response in
                        #expect(response.status == .badRequest)
                        let data = Data(String(buffer: response.body).utf8)
                        let error = try JSONDecoder().decode(OpenAIErrorResponse.self, from: data)
                        #expect(error.error.type == "invalid_request_error")
                        #expect(error.error.message.contains("generic model-container"))
                        #expect(!String(buffer: response.body).contains("data:"))
                    }
                }
            }
        }
    }

    @Test func sharedHTTPStackDoesNotRejectCapableEngines() async throws {
        let engine = ControlFixtureEngine(acceptsControls: true)
        let app = MLXServerApplication.buildApplication(
            service: MLXOpenAIService(engine: engine), host: "127.0.0.1", port: 8080)
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/chat/completions", method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"model":"fixture","messages":[{"role":"user","content":"hi"}],"seed":17,"logit_bias":{"42":-12.5}}"#)) { response in
                    #expect(response.status == .ok)
                }
        }
        let seen = await engine.lastRequest
        #expect(seen?.seed == 17)
        #expect(seen?.logitBias == ["42": -12.5])
    }
}

private actor ControlFixtureEngine: MLXServerEngine {
    let acceptsControls: Bool
    var lastRequest: OpenAIChatCompletionRequest?
    init(acceptsControls: Bool = false) { self.acceptsControls = acceptsControls }
    func availableModels() async throws -> [MLXServerModel] { [.init(id: "fixture")] }
    func streamChatCompletion(request: OpenAIChatCompletionRequest)
        async throws -> AsyncThrowingStream<MLXServerGenerationEvent, Error>
    {
        if !acceptsControls { try MLXModelContainerEngine.validateSamplingControls(request) }
        lastRequest = request
        return AsyncThrowingStream { continuation in
            continuation.yield(.content("ok"))
            continuation.yield(.info(.init(promptTokens: 1, completionTokens: 1,
                promptTime: 0.01, generationTime: 0.01, stopReason: "stop")))
            continuation.finish()
        }
    }
    func tokenize(_ request: TokenizeRequest) async throws -> TokenizeResponse { .init(tokens: [1]) }
    func detokenize(_ request: DetokenizeRequest) async throws -> DetokenizeResponse { .init(text: "ok") }
    func applyTemplate(_ request: ApplyTemplateRequest) async throws -> TokenizeResponse { .init(tokens: [1]) }
}
