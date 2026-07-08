// Copyright © 2026 Eigen Labs Inc.

import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

/// Single-request server engine backed by ``ModelContainer``'s serial actor.
/// Concurrent ``streamChatCompletion(request:)`` calls serialise; batched
/// concurrent serving lives downstream in the Darkbloom provider's
/// ContinuousBatchingV2 bridge, not in this CLI engine.
public struct MLXModelContainerEngine: MLXServerEngine {
    private let modelID: String
    private let model: ModelContainer
    private let modelType: String?
    private let defaultToolCallParser: String?

    public init(
        modelID: String,
        model: ModelContainer,
        modelType: String? = nil,
        defaultToolCallParser: String? = nil
    ) {
        self.modelID = modelID
        self.model = model
        self.modelType = modelType
        self.defaultToolCallParser = defaultToolCallParser
    }

    public func availableModels() async throws -> [MLXServerModel] {
        [.init(id: modelID)]
    }

    public func streamChatCompletion(
        request: OpenAIChatCompletionRequest
    ) async throws -> AsyncThrowingStream<MLXServerGenerationEvent, Error> {
        // This engine builds its prompt by flattening each message to text via
        // `chatMessage()`, which discards `image_url`/`video_url` parts. Rather
        // than accept media and silently ignore it, fail loud — real media
        // serving lives in the downstream VLM path, not this text-only engine.
        // Checked before any model state is mutated so a rejected request is a
        // no-op.
        if request.messages.contains(where: { $0.content.hasMedia }) {
            throw MLXModelContainerEngineError.mediaUnsupported
        }

        try await configureToolParser(for: request)

        // Multi-turn tool-use history (assistant `tool_calls`,
        // `tool_call_id`, `reasoning_content`) is not representable in
        // `Chat.Message` (role + text content only) — flattening through
        // `chatMessage()` silently drops those fields before the chat
        // template renders, corrupting tool-call conversations. When any
        // message carries them, feed the template the full
        // `templateMessage()` dictionaries instead (`UserInput(messages:)`
        // hands them to `applyChatTemplate` verbatim). Plain chats keep
        // the structured `.chat` path unchanged.
        let userInput: UserInput
        if Self.requiresTemplateMessages(request.messages) {
            userInput = UserInput(
                messages: request.messages.map { $0.templateMessage() },
                tools: request.tools?.map { $0.toolSpec() }
            )
        } else {
            userInput = UserInput(
                chat: request.messages.map { $0.chatMessage() },
                tools: request.tools?.map { $0.toolSpec() }
            )
        }
        let input = try await model.prepare(input: userInput)
        let stream = try await model.generate(
            input: input, parameters: request.generationParameters)

        return AsyncThrowingStream { continuation in
            let task = Task {
                for await item in stream {
                    switch item {
                    case .chunk(let text):
                        continuation.yield(.content(text))
                    case .toolCall(let toolCall):
                        continuation.yield(.toolCall(toolCall))
                    case .info(let info):
                        continuation.yield(.info(.init(info)))
                    }
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    /// True when any message carries tool-use history fields that
    /// `Chat.Message` cannot represent — the signal to render through
    /// `templateMessage()` dictionaries instead of the structured chat.
    /// Internal (not private) so the predicate is directly testable.
    static func requiresTemplateMessages(_ messages: [OpenAIChatMessage]) -> Bool {
        messages.contains {
            ($0.toolCalls?.isEmpty == false) || $0.toolCallID != nil
                || $0.reasoningContent != nil
        }
    }

    public func tokenize(_ request: TokenizeRequest) async throws -> TokenizeResponse {
        let tokenizer = await model.tokenizer
        return .init(
            tokens: tokenizer.encode(
                text: request.prompt,
                addSpecialTokens: request.addSpecialTokens ?? true
            )
        )
    }

    public func detokenize(_ request: DetokenizeRequest) async throws -> DetokenizeResponse {
        let tokenizer = await model.tokenizer
        return .init(
            text: tokenizer.decode(
                tokenIds: request.tokens,
                skipSpecialTokens: request.skipSpecialTokens ?? false
            )
        )
    }

    public func applyTemplate(_ request: ApplyTemplateRequest) async throws -> TokenizeResponse {
        // Full wire-type translation (role/content plus name,
        // tool_call_id, tool_calls with decoded arguments, and
        // reasoning_content) so `/apply-template` renders tool-use history
        // exactly like the serving path — not a role+content flattening.
        let messages = request.messages.map { $0.templateMessage() }
        let tools = request.tools?.map { $0.toolSpec() }
        let tokens = try await model.perform { context in
            try context.tokenizer.applyChatTemplate(
                messages: messages,
                tools: tools,
                additionalContext: nil
            )
        }
        return .init(tokens: tokens)
    }

    private func configureToolParser(for request: OpenAIChatCompletionRequest) async throws {
        let configuration = await model.configuration
        let format: ToolCallFormat
        let requested = request.toolCallParser ?? defaultToolCallParser
        if requested == nil, let existing = configuration.toolCallFormat {
            format = existing
        } else {
            format = try ServerToolParser.resolve(
                requested: requested,
                modelType: modelType
            )
        }

        await model.update { context in
            context.configuration.toolCallFormat = format
        }
    }
}

public enum MLXServerModelLoader {
    public static func load(
        configuration: ModelConfiguration
    ) async throws -> ModelContainer {
        try await #huggingFaceLoadModelContainer(configuration: configuration)
    }

    /// Load a model and return the raw ``ModelContext`` (bypasses the
    /// serial-access container so ``MLXBatchedEngineServerEngine`` can drive
    /// ``BatchedEngine`` directly).
    public static func loadContext(
        configuration: ModelConfiguration
    ) async throws -> sending ModelContext {
        try await #huggingFaceLoadModel(configuration: configuration)
    }
}
