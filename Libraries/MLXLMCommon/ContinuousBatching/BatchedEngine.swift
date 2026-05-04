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

    /// Create a batched engine from a loaded model context.
    public convenience init(
        context: ModelContext,
        config: ContinuousBatchingConfig = ContinuousBatchingConfig()
    ) {
        let ssdCache = config.ssdCacheConfig.map { SSDCacheManager(config: $0) }
        let prefixCache = config.prefixCacheConfig.map {
            PrefixCache(config: $0, ssdCache: ssdCache, modelName: context.configuration.name)
        }
        let scheduler = Scheduler(
            model: context.model,
            tokenizer: context.tokenizer,
            config: config.schedulerConfig,
            eosTokenIds: context.configuration.eosTokenIds,
            prefixCache: prefixCache
        )
        self.init(
            scheduler: scheduler,
            tokenizer: context.tokenizer,
            modelName: context.configuration.name,
            config: config
        )
    }

    /// Create a batched engine with explicit scheduler and tokenizer.
    public init(
        scheduler: Scheduler,
        tokenizer: any Tokenizer,
        modelName: String = "",
        config: ContinuousBatchingConfig = ContinuousBatchingConfig()
    ) {
        self.core = EngineCore(scheduler: scheduler, config: config)
        self.tokenizer = tokenizer
        self.modelName = modelName
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
    public func generate(
        prompt: String,
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        topK: Int = 0,
        minP: Float = 0.0
    ) async throws -> String {
        let result = try await core.generate(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP
        )
        return result.outputText
    }

    /// Generate with structured result (includes token counts).
    public func generateWithResult(
        prompt: String,
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topP: Float = 0.9
    ) async throws -> RequestOutput {
        try await core.generate(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP
        )
    }

    // MARK: - Streaming Generation

    /// Stream generation token by token.
    public func streamGenerate(
        prompt: String,
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topP: Float = 0.9
    ) -> AsyncStream<String> {
        let rid = UUID().uuidString
        let request = Request(
            requestId: rid,
            prompt: prompt,
            samplingParams: SamplingParams(
                maxTokens: maxTokens,
                temperature: temperature,
                topP: topP
            )
        )

        return AsyncStream { continuation in
            Task {
                let _ = await core.addRequest(request)

                for await output in core.streamOutputs(requestId: rid) {
                    if !output.newText.isEmpty {
                        continuation.yield(output.newText)
                    }

                    if output.finished {
                        continuation.finish()
                        return
                    }
                }

                continuation.finish()
            }
        }
    }

    // MARK: - Chat Completion

    /// Chat completion (non-streaming). Applies chat template.
    public func chat(
        messages: [[String: String]],
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topP: Float = 0.9
    ) async throws -> String {
        let prompt: String
        do {
            let tokenIds = try tokenizer.applyChatTemplate(messages: messages)
            prompt = tokenizer.decode(tokenIds: tokenIds)
        } catch {
            // Fallback: simple formatting
            prompt = messages.map { "\($0["role"] ?? "user"): \($0["content"] ?? "")" }
                .joined(separator: "\n") + "\nassistant:"
        }

        return try await generate(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP
        )
    }

    /// Stream chat completion.
    public func streamChat(
        messages: [[String: String]],
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topP: Float = 0.9
    ) -> AsyncStream<String> {
        let prompt: String
        do {
            let tokenIds = try tokenizer.applyChatTemplate(messages: messages)
            prompt = tokenizer.decode(tokenIds: tokenIds)
        } catch {
            prompt = messages.map { "\($0["role"] ?? "user"): \($0["content"] ?? "")" }
                .joined(separator: "\n") + "\nassistant:"
        }

        return streamGenerate(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP
        )
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
