// Copyright © 2026 Eigen Labs.
//
// Port of omlx/omlx/engine/batched.py — High-level batched engine.
// https://github.com/jundot/omlx/blob/main/omlx/engine/batched.py
//
// Provides the user-facing generate/stream API.
// The EngineCore runs the scheduler loop; BatchedEngine handles
// prompt tokenization, chat templates, and output decoding.

import Foundation
import MLX

/// High-level continuous-batching engine.
///
/// Wraps EngineCore + Scheduler with the tokenization, chat template,
/// and streaming APIs that the server layer needs.
///
/// Usage:
/// ```swift
/// let engine = try BatchedEngine(model: model, tokenizer: tokenizer)
/// await engine.start()
///
/// // Non-streaming
/// let result = await engine.generate(prompt: "Hello", maxTokens: 100)
///
/// // Streaming
/// for await output in engine.streamGenerate(prompt: "Hi", maxTokens: 50) {
///     print(output.text, terminator: "")
/// }
/// ```
public final class BatchedEngine: @unchecked Sendable {
    public let core: EngineCore

    private let tokenizer: any Tokenizer
    public let modelName: String
    /// Chat template loaded from `chat_template.jinja` in the model directory,
    /// used as fallback when the tokenizer config lacks a `chat_template` field.
    private let externalChatTemplate: String?

    /// Create a batched engine from a loaded model context.
    public convenience init(
        context: ModelContext,
        config: ContinuousBatchingConfig = ContinuousBatchingConfig()
    ) {
        let prefixCache = config.prefixCacheConfig.map {
            PrefixCache(config: $0, modelName: context.configuration.name)
        }
        let scheduler = Scheduler(
            model: context.model,
            tokenizer: context.tokenizer,
            config: config.schedulerConfig,
            eosTokenIds: context.configuration.eosTokenIds,
            prefixCache: prefixCache
        )
        // Load chat_template.jinja if the tokenizer lacks an embedded chat template.
        let externalTemplate: String? = (try? context.configuration.modelDirectory).flatMap {
            try? String(contentsOf: $0.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        }
        self.init(
            scheduler: scheduler,
            tokenizer: context.tokenizer,
            modelName: context.configuration.name,
            config: config,
            externalChatTemplate: externalTemplate
        )
    }

    /// Create a batched engine with explicit scheduler and tokenizer.
    public init(
        scheduler: Scheduler,
        tokenizer: any Tokenizer,
        modelName: String = "",
        config: ContinuousBatchingConfig = ContinuousBatchingConfig(),
        externalChatTemplate: String? = nil
    ) {
        self.core = EngineCore(scheduler: scheduler, config: config)
        self.tokenizer = tokenizer
        self.modelName = modelName
        self.externalChatTemplate = externalChatTemplate
    }

    // MARK: - Lifecycle

    /// Start the engine (loads nothing — model is already loaded).
    public func start() async {
        core.start()
    }

    /// Stop the engine and cleanup resources.
    public func stop() async {
        core.stop()
    }

    // MARK: - Non-streaming Generation

    /// Generate a complete response (non-streaming).
    public func generate(prompt: String, samplingParams: SamplingParams = SamplingParams()) async throws -> String {
        try await generateWithResult(prompt: prompt, samplingParams: samplingParams).outputText
    }

    /// Generate a complete response using the legacy convenience parameters.
    public func generate(
        prompt: String,
        maxTokens: Int,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        topK: Int = 0,
        minP: Float = 0.0
    ) async throws -> String {
        try await generate(
            prompt: prompt,
            samplingParams: SamplingParams(
                maxTokens: maxTokens,
                temperature: temperature,
                topP: topP,
                topK: topK,
                minP: minP))
    }

    /// Generate with structured result (includes token counts).
    public func generateWithResult(prompt: String, samplingParams: SamplingParams = SamplingParams()) async throws -> RequestOutput {
        try await core.generate(prompt: prompt, samplingParams: samplingParams)
    }

    // MARK: - Streaming Generation

    /// Stream outputs (RequestOutput per step) with full SamplingParams control.
    public func streamOutputs(prompt: String, samplingParams: SamplingParams = SamplingParams()) -> AsyncStream<RequestOutput> {
        let rid = UUID().uuidString
        let request = Request(requestId: rid, prompt: prompt, samplingParams: samplingParams)
        return AsyncStream { continuation in
            Task {
                let _ = await core.addRequest(request)
                for await output in core.streamOutputs(requestId: rid) {
                    continuation.yield(output)
                    if output.finished || output.error != nil { break }
                }
                continuation.finish()
            }
        }
    }

    /// Stream generation token by token (returns text chunks).
    public func streamGenerate(prompt: String, samplingParams: SamplingParams = SamplingParams()) -> AsyncStream<String> {
        AsyncStream { continuation in
            Task {
                for await output in streamOutputs(prompt: prompt, samplingParams: samplingParams) {
                    if !output.newText.isEmpty { continuation.yield(output.newText) }
                    if output.finished || output.error != nil { break }
                }
                continuation.finish()
            }
        }
    }

    /// Stream generation using the legacy convenience parameters.
    public func streamGenerate(
        prompt: String,
        maxTokens: Int,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        topK: Int = 0,
        minP: Float = 0.0
    ) -> AsyncStream<String> {
        streamGenerate(
            prompt: prompt,
            samplingParams: SamplingParams(
                maxTokens: maxTokens,
                temperature: temperature,
                topP: topP,
                topK: topK,
                minP: minP))
    }

    // MARK: - Chat Completion

    /// Apply chat template to messages and return the prompt string.
    /// If the tokenizer config lacks a chat_template, falls back to `chat_template.jinja`
    /// loaded from the model directory.
    public func buildPrompt(messages: [[String: String]]) -> String {
        do {
            let tokenIds = try tokenizer.applyChatTemplate(messages: messages)
            return tokenizer.decode(tokenIds: tokenIds)
        } catch {
            if let template = externalChatTemplate {
                do {
                    let tokenIds = try tokenizer.applyChatTemplate(
                        messages: messages, chatTemplate: template)
                    return tokenizer.decode(tokenIds: tokenIds)
                } catch {}
            }
            return messages.map { "\($0["role"] ?? "user"): \($0["content"] ?? "")" }
                .joined(separator: "\n") + "\nassistant:"
        }
    }

    /// Chat completion (non-streaming). Applies chat template.
    public func chat(messages: [[String: String]], samplingParams: SamplingParams = SamplingParams()) async throws -> String {
        let prompt = buildPrompt(messages: messages)
        return try await generate(prompt: prompt, samplingParams: samplingParams)
    }

    /// Chat completion using the legacy convenience parameters.
    public func chat(
        messages: [[String: String]],
        maxTokens: Int,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        topK: Int = 0,
        minP: Float = 0.0
    ) async throws -> String {
        try await chat(
            messages: messages,
            samplingParams: SamplingParams(
                maxTokens: maxTokens,
                temperature: temperature,
                topP: topP,
                topK: topK,
                minP: minP))
    }

    /// Stream chat completion.
    public func streamChat(messages: [[String: String]], samplingParams: SamplingParams = SamplingParams()) -> AsyncStream<String> {
        streamGenerate(prompt: buildPrompt(messages: messages), samplingParams: samplingParams)
    }

    // MARK: - Status

    public var hasActiveRequests: Bool {
        core.scheduler.hasRequests()
    }

    public func getStats() -> [String: Any] {
        var stats = core.getStats()
        stats["engine_type"] = "batched"
        stats["model_name"] = modelName
        return stats
    }
}
