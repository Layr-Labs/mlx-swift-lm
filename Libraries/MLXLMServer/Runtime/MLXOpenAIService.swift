// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon

public enum MLXOpenAIServiceError: Error, LocalizedError, Equatable {
    case responseNotFound(String)
    case embeddingsNotConfigured
    case invalidResponseFormatOutput(String)
    case multipleToolCallsNotAllowed

    public var errorDescription: String? {
        switch self {
        case .responseNotFound(let id):
            return "Response '\(id)' was not found"
        case .embeddingsNotConfigured:
            return
                "Embeddings require an embedding model engine; this server instance was started without one."
        case .invalidResponseFormatOutput(let message):
            return "Generated output did not satisfy response_format: \(message)"
        case .multipleToolCallsNotAllowed:
            return
                "Generated output contained multiple tool calls while parallel_tool_calls was false"
        }
    }
}

public struct MLXOpenAIService: Sendable {
    private struct CollectedChatOutput: Sendable {
        var content: String = ""
        var nativeReasoning: String = ""
        var hasNativeChannels = false
        var toolCalls: [OpenAIToolCall] = []
        var usage: OpenAIUsage?
        var finishReason = "stop"
    }

    private let engine: any MLXServerEngine
    private let embeddingEngine: (any MLXEmbeddingServerEngine)?
    private let responseStore: InMemoryResponseStore
    private let metrics: ServerMetrics
    private let defaultReasoningParser: ReasoningParserFormat?
    private let idProvider: @Sendable (String) -> String

    public init(
        engine: any MLXServerEngine,
        embeddingEngine: (any MLXEmbeddingServerEngine)? = nil,
        responseStore: InMemoryResponseStore = .init(),
        metrics: ServerMetrics = .init(),
        defaultReasoningParser: ReasoningParserFormat? = nil,
        idProvider: @escaping @Sendable (String) -> String = { prefix in
            "\(prefix)_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))"
        }
    ) {
        self.engine = engine
        self.embeddingEngine = embeddingEngine
        self.responseStore = responseStore
        self.metrics = metrics
        self.defaultReasoningParser = defaultReasoningParser
        self.idProvider = idProvider
    }

    public func availableModels() async throws -> OpenAIModelListResponse {
        await metrics.recordOtherRequest()
        var models = try await engine.availableModels()
        if let embeddingEngine {
            models.append(contentsOf: try await embeddingEngine.availableModels())
        }
        return .init(data: models)
    }

    public func createChatCompletion(
        request: OpenAIChatCompletionRequest
    ) async throws -> OpenAIChatCompletionResponse {
        await metrics.recordChatRequest()
        do {
            let output = try await collectChatOutput(request: request)
            let response = try chatCompletionResponse(request: request, output: output)
            await metrics.recordUsage(response.usage)
            return response
        } catch {
            await metrics.recordError()
            throw error
        }
    }

    public func streamChatCompletionFrames(
        request: OpenAIChatCompletionRequest
    ) async throws -> AsyncThrowingStream<String, Error> {
        try await streamChatCompletionFrames(request: request, frameGenerationErrors: false)
    }

    public func streamChatCompletionFrames(
        request: OpenAIChatCompletionRequest,
        frameGenerationErrors: Bool
    ) async throws -> AsyncThrowingStream<String, Error> {
        await metrics.recordChatRequest()
        let generationRequest = try OpenAIRequestValidation.preparedRequest(request)
        let stream = try await engine.streamChatCompletion(request: generationRequest)
        let id = idProvider("chatcmpl")
        let created = Int(Date().timeIntervalSince1970)
        let includeUsage = request.streamOptions?.includeUsage == true

        return AsyncThrowingStream { continuation in
            let task = Task {
                var usage: OpenAIUsage?
                var finishReason = "stop"
                var nextToolCallIndex = 0
                var reasoningParser = StreamingReasoningParser(
                    format: request.reasoningParser ?? defaultReasoningParser ?? .none
                )
                do {
                    continuation.yield(
                        try ServerSentEventEncoder.encode(
                            OpenAIChatCompletionChunk(
                                id: id,
                                model: request.model,
                                choices: [
                                    .init(
                                        index: 0,
                                        delta: .init(
                                            role: "assistant",
                                            content: nil,
                                            reasoningContent: nil,
                                            toolCalls: nil
                                        ),
                                        finishReason: nil
                                    )
                                ],
                                usage: nil,
                                created: created
                            )
                        )
                    )

                    func emitParsed(_ parsed: ParsedReasoning) throws {
                        let content = parsed.content.isEmpty ? nil : parsed.content
                        guard content != nil || parsed.reasoningContent != nil else { return }
                        continuation.yield(
                            try ServerSentEventEncoder.encode(
                                OpenAIChatCompletionChunk(
                                    id: id,
                                    model: request.model,
                                    choices: [
                                        .init(
                                            index: 0,
                                            delta: .init(
                                                role: nil,
                                                content: content,
                                                reasoningContent: parsed.reasoningContent,
                                                toolCalls: nil
                                            ),
                                            finishReason: nil
                                        )
                                    ],
                                    usage: nil,
                                    created: created
                                )
                            )
                        )
                    }
                    for try await event in stream {
                        switch event {
                        case .content(let text):
                            for parsed in reasoningParser.parse(text) {
                                try emitParsed(parsed)
                            }
                        case .parsed(let parsed):
                            try emitParsed(parsed)
                        case .toolCall(let toolCall):
                            if request.parallelToolCalls == false,
                                nextToolCallIndex > 0
                            {
                                throw MLXOpenAIServiceError.multipleToolCallsNotAllowed
                            }
                            finishReason = "tool_calls"
                            let openAIToolCall = try OpenAIToolCall(
                                toolCall: toolCall,
                                id: idProvider("call"),
                                index: nextToolCallIndex
                            )
                            nextToolCallIndex += 1
                            continuation.yield(
                                try ServerSentEventEncoder.encode(
                                    OpenAIChatCompletionChunk(
                                        id: id,
                                        model: request.model,
                                        choices: [
                                            .init(
                                                index: 0,
                                                delta: .init(
                                                    role: nil,
                                                    content: nil,
                                                    reasoningContent: nil,
                                                    toolCalls: [openAIToolCall]
                                                ),
                                                finishReason: nil
                                            )
                                        ],
                                        usage: nil,
                                        created: created
                                    )
                                )
                            )
                        case .info(let info):
                            usage = .init(
                                promptTokens: info.promptTokens,
                                completionTokens: info.completionTokens,
                                cachedPromptTokens: info.cachedPromptTokens
                            )
                            if finishReason != "tool_calls" {
                                finishReason = info.stopReason
                            }
                        }
                    }

                    for parsed in reasoningParser.finish() {
                        let content = parsed.content.isEmpty ? nil : parsed.content
                        guard content != nil || parsed.reasoningContent != nil else { continue }
                        continuation.yield(
                            try ServerSentEventEncoder.encode(
                                OpenAIChatCompletionChunk(
                                    id: id,
                                    model: request.model,
                                    choices: [
                                        .init(
                                            index: 0,
                                            delta: .init(
                                                role: nil,
                                                content: content,
                                                reasoningContent: parsed.reasoningContent,
                                                toolCalls: nil
                                            ),
                                            finishReason: nil
                                        )
                                    ],
                                    usage: nil,
                                    created: created
                                )
                            )
                        )
                    }

                    continuation.yield(
                        try ServerSentEventEncoder.encode(
                            OpenAIChatCompletionChunk(
                                id: id,
                                model: request.model,
                                choices: [
                                    .init(
                                        index: 0,
                                        delta: .init(
                                            role: nil,
                                            content: nil,
                                            reasoningContent: nil,
                                            toolCalls: nil
                                        ),
                                        finishReason: finishReason
                                    )
                                ],
                                usage: includeUsage ? usage : nil,
                                created: created
                            )
                        )
                    )
                    await metrics.recordUsage(usage)
                    continuation.yield(ServerSentEventEncoder.done)
                    continuation.finish()
                } catch {
                    await metrics.recordError()
                    // Direct service clients retain the throwing contract.
                    // HTTP callers have committed headers and need a complete
                    // error event, not an abruptly truncated transfer. Consumer
                    // cancellation still propagates without a terminal frame.
                    guard frameGenerationErrors, !Task.isCancelled,
                        !(error is CancellationError)
                    else {
                        continuation.finish(throwing: error)
                        return
                    }
                    do {
                        continuation.yield(try ChatStreamFailure.encode(
                            error, id: id, model: request.model, created: created,
                            usage: includeUsage ? usage : nil))
                        await metrics.recordUsage(usage)
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    public func streamResponseFrames(
        request: OpenAIResponseRequest
    ) async throws -> AsyncThrowingStream<String, Error> {
        await metrics.recordResponseRequest()
        let generationRequest = try OpenAIRequestValidation.preparedRequest(request.chatCompletionRequest)
        let stream = try await engine.streamChatCompletion(request: generationRequest)
        return AsyncThrowingStream { continuation in
            let task = Task {
                var writer = ResponsesStreamWriter(request: request, idProvider: idProvider)
                var parser = StreamingReasoningParser(format: request.reasoning?.parser ?? defaultReasoningParser ?? .none)
                do {
                    try writer.start()
                    for frame in writer.takeFrames() { continuation.yield(frame) }
                    for try await event in stream {
                        try Task.checkCancellation()
                        switch event {
                        case .content(let text):
                            for parsed in parser.parse(text) { try writer.append(parsed) }
                        case .parsed(let parsed):
                            try writer.append(parsed)
                        case .toolCall(let call):
                            for parsed in parser.finish() { try writer.append(parsed) }
                            try writer.append(call)
                        case .info(let info):
                            writer.update(info)
                        }
                        for frame in writer.takeFrames() { continuation.yield(frame) }
                    }
                    try Task.checkCancellation()
                    for parsed in parser.finish() { try writer.append(parsed) }
                    let response = try writer.finish()
                    if request.store != false { await responseStore.save(response) }
                    if let usage = response.usage { await metrics.recordUsage(usage.chatUsage) }
                    for frame in writer.takeFrames() { continuation.yield(frame) }
                    continuation.finish()
                } catch {
                    await metrics.recordError()
                    if Task.isCancelled {
                        continuation.finish(throwing: CancellationError())
                        return
                    }
                    do {
                        let response = try writer.fail(error)
                        if request.store != false { await responseStore.save(response) }
                        for frame in writer.takeFrames() { continuation.yield(frame) }
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public func createResponse(request: OpenAIResponseRequest) async throws -> OpenAIResponse {
        await metrics.recordResponseRequest()
        do {
            let output = try await collectChatOutput(request: request.chatCompletionRequest)
            let response = responseObject(request: request, output: output)
            if request.store != false {
                await responseStore.save(response)
            }
            await metrics.recordUsage(output.usage)
            return response
        } catch {
            await metrics.recordError()
            throw error
        }
    }

    public func retrieveResponse(id: String) async throws -> OpenAIResponse {
        guard let response = await responseStore.get(id: id) else {
            throw MLXOpenAIServiceError.responseNotFound(id)
        }
        return response
    }

    public func cancelResponse(id: String) async throws -> OpenAIResponse {
        guard let response = await responseStore.cancel(id: id) else {
            throw MLXOpenAIServiceError.responseNotFound(id)
        }
        return response
    }

    public func tokenize(_ request: TokenizeRequest) async throws -> TokenizeResponse {
        await metrics.recordOtherRequest()
        return try await engine.tokenize(request)
    }

    public func detokenize(_ request: DetokenizeRequest) async throws -> DetokenizeResponse {
        await metrics.recordOtherRequest()
        return try await engine.detokenize(request)
    }

    public func applyTemplate(_ request: ApplyTemplateRequest) async throws -> TokenizeResponse {
        await metrics.recordOtherRequest()
        return try await engine.applyTemplate(request)
    }

    public func createEmbedding(
        request: OpenAIEmbeddingRequest
    ) async throws -> OpenAIEmbeddingResponse {
        await metrics.recordOtherRequest()
        guard let embeddingEngine else {
            await metrics.recordError()
            throw MLXOpenAIServiceError.embeddingsNotConfigured
        }
        do {
            let response = try await embeddingEngine.createEmbedding(request: request)
            await metrics.recordUsage(response.usage)
            return response
        } catch {
            await metrics.recordError()
            throw error
        }
    }

    public func metricsSnapshot() async -> ServerMetricsSnapshot {
        await metrics.snapshot()
    }

    public func prometheusMetrics() async -> String {
        await metrics.prometheusText()
    }

    private func collectChatOutput(
        request: OpenAIChatCompletionRequest
    ) async throws -> CollectedChatOutput {
        let generationRequest = try OpenAIRequestValidation.preparedRequest(request)
        let stream = try await engine.streamChatCompletion(request: generationRequest)
        var output = CollectedChatOutput()

        for try await event in stream {
            switch event {
            case .content(let text):
                output.content += text
            case .parsed(let parsed):
                output.hasNativeChannels = true
                output.content += parsed.content
                output.nativeReasoning += parsed.reasoningContent ?? ""
            case .toolCall(let toolCall):
                if request.parallelToolCalls == false,
                    !output.toolCalls.isEmpty
                {
                    throw MLXOpenAIServiceError.multipleToolCallsNotAllowed
                }
                output.toolCalls.append(
                    try OpenAIToolCall(toolCall: toolCall, id: idProvider("call"))
                )
                output.finishReason = "tool_calls"
            case .info(let info):
                output.usage = .init(
                    promptTokens: info.promptTokens,
                    completionTokens: info.completionTokens,
                    cachedPromptTokens: info.cachedPromptTokens
                )
                if output.finishReason != "tool_calls" {
                    output.finishReason = info.stopReason
                }
            }
        }

        return output
    }

    private func chatCompletionResponse(
        request: OpenAIChatCompletionRequest,
        output: CollectedChatOutput
    ) throws -> OpenAIChatCompletionResponse {
        let parsed =
            output.hasNativeChannels
            ? ParsedReasoning(
                content: output.content,
                reasoningContent: output.nativeReasoning.isEmpty ? nil : output.nativeReasoning)
            : ReasoningParser(format: request.reasoningParser ?? defaultReasoningParser ?? .none)
                .parse(output.content)
        let content =
            output.toolCalls.isEmpty
            ? try OpenAIResponseFormatSupport.normalizedContent(
                parsed.content,
                for: request.responseFormat
            )
            : parsed.content
        let message = OpenAIChatMessage(
            role: .assistant,
            content: .text(content),
            toolCalls: output.toolCalls.isEmpty ? nil : output.toolCalls,
            reasoningContent: parsed.reasoningContent
        )

        return .init(
            id: idProvider("chatcmpl"),
            model: request.model,
            choices: [
                .init(index: 0, message: message, finishReason: output.finishReason)
            ],
            usage: output.usage ?? .init(promptTokens: 0, completionTokens: 0)
        )
    }

    private func responseObject(
        request: OpenAIResponseRequest,
        output: CollectedChatOutput
    ) -> OpenAIResponse {
        let parsed =
            output.hasNativeChannels
            ? ParsedReasoning(
                content: output.content,
                reasoningContent: output.nativeReasoning.isEmpty ? nil : output.nativeReasoning)
            : ReasoningParser(format: request.reasoning?.parser ?? defaultReasoningParser ?? .none)
                .parse(output.content)
        var outputItems: [OpenAIResponseOutputItem] = []
        if let reasoningContent = parsed.reasoningContent, !reasoningContent.isEmpty {
            outputItems.append(.reasoning(id: idProvider("rs"), text: reasoningContent))
        }
        if !parsed.content.isEmpty {
            outputItems.append(.message(id: idProvider("msg"), text: parsed.content))
        }
        for toolCall in output.toolCalls {
            outputItems.append(.functionCall(id: idProvider("fc"), toolCall: toolCall))
        }

        let incomplete = output.toolCalls.isEmpty && ["length", "content_filter"].contains(output.finishReason)
        return .init(
            id: idProvider("resp"),
            status: incomplete ? .incomplete : .completed,
            model: request.model,
            output: outputItems,
            outputText: parsed.content,
            usage: output.usage.map(OpenAIResponseUsage.init(chatUsage:)),
            metadata: request.metadata,
            incompleteDetails: incomplete ? .init(reason: output.finishReason == "length" ? "max_output_tokens" : output.finishReason) : nil
        )
    }
}
