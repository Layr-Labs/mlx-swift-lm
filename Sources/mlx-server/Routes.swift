// HTTP route handlers for the OpenAI-compatible API.

import Foundation
import Hummingbird
import MLX
import MLXLLM
import MLXLMCommon

// MARK: - JSON helpers

private let jsonEncoder: JSONEncoder = {
    let e = JSONEncoder()
    e.outputFormatting = .sortedKeys
    return e
}()

private func jsonBody<T: Encodable>(_ value: T) throws -> ResponseBody {
    let data = try jsonEncoder.encode(value)
    return .init(byteBuffer: .init(data: data))
}

// MARK: - Sampling params

private func samplingParams(from req: ChatCompletionRequest, defaultMaxTokens: Int) -> SamplingParams {
    SamplingParams(
        maxTokens: req.max_tokens ?? defaultMaxTokens,
        temperature: req.temperature ?? 0.7,
        topP: req.top_p ?? 0.9,
        topK: req.top_k ?? 0,
        minP: req.min_p ?? 0.0,
        repetitionPenalty: req.repetition_penalty ?? 1.0,
        presencePenalty: req.presence_penalty ?? 0.0,
        frequencyPenalty: req.frequency_penalty ?? 0.0,
        stop: req.stop?.strings ?? [],
        seed: req.seed.map { UInt64(bitPattern: Int64($0)) }
    )
}

private func samplingParams(from req: CompletionRequest, defaultMaxTokens: Int) -> SamplingParams {
    SamplingParams(
        maxTokens: req.max_tokens ?? defaultMaxTokens,
        temperature: req.temperature ?? 0.7,
        topP: req.top_p ?? 0.9,
        stop: req.stop?.strings ?? [],
        seed: req.seed.map { UInt64(bitPattern: Int64($0)) }
    )
}

// MARK: - Gemma 4 MTP generation helper

/// Run Gemma 4 MTP generation for a plain-text prompt, returning (outputText, promptTokens, completionTokens, finishReason).
private func generateWithGemma4MTP(
    prompt: String,
    params: SamplingParams,
    g4: Gemma4ServerContext
) async throws -> (String, Int, Int, String) {
    let tokenIds = g4.target.tokenizer.encode(text: prompt, addSpecialTokens: true)
    let input = LMInput(text: .init(tokens: MLXArray(tokenIds)))
    let genParams = GenerateParameters(
        maxTokens: params.maxTokens,
        temperature: params.temperature,
        topP: params.topP,
        topK: params.topK,
        minP: params.minP
    )
    var outputText = ""
    var promptToks = tokenIds.count
    var completionToks = 0
    var finishReason = "stop"
    for await gen in try generateGemma4MTP(
        input: input, parameters: genParams, target: g4.target, drafter: g4.drafter
    ) {
        switch gen {
        case .chunk(let text):
            outputText += text
        case .info(let info):
            promptToks = info.promptTokenCount
            completionToks = info.generationTokenCount
            finishReason = info.stopReason == .stop ? "stop" : "length"
        case .toolCall:
            break
        }
    }
    return (outputText, promptToks, completionToks, finishReason)
}

// MARK: - Router builder

func buildRouter(
    engine: BatchedEngine,
    gemma4Context: Gemma4ServerContext? = nil,
    modelName: String,
    defaultMaxTokens: Int
) -> Router<BasicRequestContext> {
    let router = Router()

    // Health check
    router.get("/health") { _, _ in
        let body = try jsonBody(HealthResponse(status: "ok", model: modelName))
        return Response(status: .ok, headers: [.contentType: "application/json"], body: body)
    }

    // Model list
    router.get("/v1/models") { _, _ in
        let created = Int(Date().timeIntervalSince1970) - 86400
        let resp = ModelsResponse(
            object: "list",
            data: [ModelData(id: modelName, object: "model", created: created, owned_by: "mlx")]
        )
        let body = try jsonBody(resp)
        return Response(status: .ok, headers: [.contentType: "application/json"], body: body)
    }

    // Text completions
    router.post("/v1/completions") { request, context in
        let req = try await request.decode(as: CompletionRequest.self, context: context)
        let params = samplingParams(from: req, defaultMaxTokens: defaultMaxTokens)

        if req.stream == true {
            return sseTextResponse(engine: engine, modelName: modelName, prompt: req.prompt, params: params)
        } else if let g4 = gemma4Context {
            let (text, pp, tg, finish) = try await generateWithGemma4MTP(
                prompt: req.prompt, params: params, g4: g4)
            let resp = CompletionResponse(
                id: "cmpl-\(UUID().uuidString)",
                object: "text_completion",
                created: Int(Date().timeIntervalSince1970),
                model: modelName,
                choices: [CompletionChoice(index: 0, text: text, finish_reason: finish)],
                usage: Usage(prompt_tokens: pp, completion_tokens: tg, total_tokens: pp + tg)
            )
            let body = try jsonBody(resp)
            return Response(status: .ok, headers: [.contentType: "application/json"], body: body)
        } else {
            let output = try await engine.generateWithResult(prompt: req.prompt, samplingParams: params)
            let resp = CompletionResponse(
                id: "cmpl-\(UUID().uuidString)",
                object: "text_completion",
                created: Int(Date().timeIntervalSince1970),
                model: modelName,
                choices: [CompletionChoice(
                    index: 0, text: output.outputText, finish_reason: output.finishReason
                )],
                usage: Usage(
                    prompt_tokens: output.promptTokens,
                    completion_tokens: output.completionTokens,
                    total_tokens: output.promptTokens + output.completionTokens
                )
            )
            let body = try jsonBody(resp)
            return Response(status: .ok, headers: [.contentType: "application/json"], body: body)
        }
    }

    // Chat completions
    router.post("/v1/chat/completions") { request, context in
        let req = try await request.decode(as: ChatCompletionRequest.self, context: context)
        let params = samplingParams(from: req, defaultMaxTokens: defaultMaxTokens)
        let prompt = engine.buildPrompt(
            messages: req.messages.map { ["role": $0.role, "content": $0.content] }
        )

        if req.stream == true {
            return sseChatResponse(engine: engine, modelName: modelName, prompt: prompt, params: params)
        } else if let g4 = gemma4Context {
            let (text, pp, tg, finish) = try await generateWithGemma4MTP(
                prompt: prompt, params: params, g4: g4)
            let resp = ChatCompletionResponse(
                id: "chatcmpl-\(UUID().uuidString)",
                object: "chat.completion",
                created: Int(Date().timeIntervalSince1970),
                model: modelName,
                choices: [ChatChoice(
                    index: 0,
                    message: ChatMessage(role: "assistant", content: text),
                    finish_reason: finish
                )],
                usage: Usage(prompt_tokens: pp, completion_tokens: tg, total_tokens: pp + tg)
            )
            let body = try jsonBody(resp)
            return Response(status: .ok, headers: [.contentType: "application/json"], body: body)
        } else {
            let out = try await engine.generateWithResult(prompt: prompt, samplingParams: params)
            let resp = ChatCompletionResponse(
                id: "chatcmpl-\(UUID().uuidString)",
                object: "chat.completion",
                created: Int(Date().timeIntervalSince1970),
                model: modelName,
                choices: [ChatChoice(
                    index: 0,
                    message: ChatMessage(role: "assistant", content: out.outputText),
                    finish_reason: out.finishReason
                )],
                usage: Usage(
                    prompt_tokens: out.promptTokens,
                    completion_tokens: out.completionTokens,
                    total_tokens: out.promptTokens + out.completionTokens
                )
            )
            let body = try jsonBody(resp)
            return Response(status: .ok, headers: [.contentType: "application/json"], body: body)
        }
    }

    return router
}

// MARK: - SSE streaming helpers

private func sseData<T: Encodable>(_ value: T) throws -> ByteBuffer {
    let json = String(data: try jsonEncoder.encode(value), encoding: .utf8) ?? "{}"
    return ByteBuffer(string: "data: \(json)\n\n")
}

private func sseChatResponse(
    engine: BatchedEngine,
    modelName: String,
    prompt: String,
    params: SamplingParams
) -> Response {
    let (stream, continuation) = AsyncStream<ByteBuffer>.makeStream()

    Task {
        let requestId = "chatcmpl-\(UUID().uuidString)"
        let created = Int(Date().timeIntervalSince1970)

        // First chunk carries the assistant role.
        let firstChunk = ChatCompletionChunk(
            id: requestId, object: "chat.completion.chunk", created: created, model: modelName,
            choices: [ChunkChoice(
                index: 0,
                delta: ChunkDelta(role: "assistant", content: ""),
                finish_reason: nil
            )]
        )
        if let buf = try? sseData(firstChunk) { continuation.yield(buf) }

        for await output in engine.streamOutputs(prompt: prompt, samplingParams: params) {
            if !output.newText.isEmpty {
                let chunk = ChatCompletionChunk(
                    id: requestId, object: "chat.completion.chunk", created: created,
                    model: modelName,
                    choices: [ChunkChoice(
                        index: 0,
                        delta: ChunkDelta(role: nil, content: output.newText),
                        finish_reason: nil
                    )]
                )
                if let buf = try? sseData(chunk) { continuation.yield(buf) }
            }
            if output.finished || output.error != nil {
                let finalChunk = ChatCompletionChunk(
                    id: requestId, object: "chat.completion.chunk", created: created,
                    model: modelName,
                    choices: [ChunkChoice(
                        index: 0,
                        delta: ChunkDelta(role: nil, content: nil),
                        finish_reason: output.finishReason ?? "stop"
                    )]
                )
                if let buf = try? sseData(finalChunk) { continuation.yield(buf) }
                break
            }
        }
        continuation.yield(ByteBuffer(string: "data: [DONE]\n\n"))
        continuation.finish()
    }

    return Response(
        status: .ok,
        headers: [.contentType: "text/event-stream", .cacheControl: "no-cache"],
        body: .init(asyncSequence: stream)
    )
}

private func sseTextResponse(
    engine: BatchedEngine,
    modelName: String,
    prompt: String,
    params: SamplingParams
) -> Response {
    let (stream, continuation) = AsyncStream<ByteBuffer>.makeStream()

    Task {
        let completionId = "cmpl-\(UUID().uuidString)"
        let created = Int(Date().timeIntervalSince1970)

        for await output in engine.streamOutputs(prompt: prompt, samplingParams: params) {
            let chunk = CompletionChunk(
                id: completionId, object: "text_completion", created: created, model: modelName,
                choices: [CompletionChunkChoice(
                    index: 0,
                    text: output.newText,
                    finish_reason: output.finished ? (output.finishReason ?? "stop") : nil
                )]
            )
            if let buf = try? sseData(chunk) { continuation.yield(buf) }
            if output.finished || output.error != nil { break }
        }
        continuation.yield(ByteBuffer(string: "data: [DONE]\n\n"))
        continuation.finish()
    }

    return Response(
        status: .ok,
        headers: [.contentType: "text/event-stream", .cacheControl: "no-cache"],
        body: .init(asyncSequence: stream)
    )
}
