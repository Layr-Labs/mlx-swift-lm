// Copyright © 2025 Apple Inc.

import CoreGraphics
import Foundation
import MLX

/// Simplified API for multi-turn conversations with LLMs and VLMs.
///
/// For example:
///
/// ```swift
/// let modelContainer = try await loadModelContainer(id: "mlx-community/Qwen3-4B-4bit")
/// let session = ChatSession(modelContainer)
/// print(try await session.respond(to: "What are two things to see in San Francisco?"))
/// print(try await session.respond(to: "How about a great place to eat?"))
/// ```
///
/// - Note: `ChatSession` is not thread-safe. Each session should be used from a single
///   task/thread at a time. The underlying `ModelContainer` handles thread safety for
///   model operations.
public final class ChatSession {

    enum Cache {
        case empty
        case kvcache([KVCache], LMOutput.State?)
        case history([Chat.Message])
    }

    private final class ContinuationStateBox: @unchecked Sendable {
        var value: LMOutput.State?

        init(_ value: LMOutput.State?) {
            self.value = value
        }
    }

    private let model: ModelContainer
    public var instructions: String?
    private let cache: SerialAccessContainer<Cache>
    public var processing: UserInput.Processing
    public var generateParameters: GenerateParameters
    public var additionalContext: [String: any Sendable]?
    public var tools: [ToolSpec]?
    public var toolDispatch: (@Sendable (ToolCall) async throws -> String)?

    /// Initialize the `ChatSession`.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContainer``
    ///   - instructions: optional system instructions for the session
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContainer,
        instructions: String? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = model
        self.instructions = instructions
        self.cache = .init(.empty)
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
    }

    /// Initialize the `ChatSession`.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContext``
    ///   - instructions: optional system instructions for the session
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContext,
        instructions: String? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = ModelContainer(context: model)
        self.instructions = instructions
        self.cache = .init(.empty)
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
    }

    /// Initialize the `ChatSession` with an existing message history.
    ///
    /// This enables "Prompt Re-hydration" for persistent chat applications.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContainer``
    ///   - instructions: optional system instructions for the session
    ///   - history: The full array of messages to restore (including system prompt)
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContainer,
        instructions: String? = nil,
        history: consuming [Chat.Message],
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = model
        self.instructions = instructions
        self.cache = .init(.history(history))
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
    }

    /// Initialize the `ChatSession` with an existing message history.
    ///
    /// This enables "Prompt Re-hydration" for persistent chat applications.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContext``
    ///   - instructions: optional system instructions for the session
    ///   - history: The full array of messages to restore (including system prompt)
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContext,
        instructions: String? = nil,
        history: [Chat.Message],
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = ModelContainer(context: model)
        self.instructions = instructions
        self.cache = .init(.history(history))
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
    }

    /// Initialize the `ChatSession` with a pre-built KV cache.
    ///
    /// This enables prefix caching: build a KV cache from a long shared context (e.g. a
    /// system prompt and document) once, save it via ``saveCache(to:)``, and restore it
    /// across multiple sessions to avoid re-prefilling the same tokens each time.
    ///
    /// > Important: If the cache was built from a session that already included system
    /// > instructions, do not pass the same `instructions` here — they would be
    /// > re-tokenized on each call to ``respond(to:role:images:videos:)`` without matching
    /// > KV state, producing incoherent output.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContainer``
    ///   - instructions: optional system instructions for the session — leave `nil` if the
    ///     cache already encodes a system prompt
    ///   - cache: a non-empty `[KVCache]` previously loaded with ``loadPromptCache(url:)``,
    ///     matching the given model
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContainer,
        instructions: String? = nil,
        cache: consuming [KVCache],
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = model
        self.instructions = instructions
        self.cache = .init(.kvcache(cache, nil))
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
    }

    /// Initialize the `ChatSession` with a pre-built KV cache.
    ///
    /// This enables prefix caching: build a KV cache from a long shared context (e.g. a
    /// system prompt and document) once, save it via ``saveCache(to:)``, and restore it
    /// across multiple sessions to avoid re-prefilling the same tokens each time.
    ///
    /// > Important: If the cache was built from a session that already included system
    /// > instructions, do not pass the same `instructions` here — they would be
    /// > re-tokenized on each call to ``respond(to:role:images:videos:)`` without matching
    /// > KV state, producing incoherent output.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContext``
    ///   - instructions: optional system instructions for the session — leave `nil` if the
    ///     cache already encodes a system prompt
    ///   - cache: a non-empty `[KVCache]` previously loaded with ``loadPromptCache(url:)``,
    ///     matching the given model
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContext,
        instructions: String? = nil,
        cache: consuming [KVCache],
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = ModelContainer(context: model)
        self.instructions = instructions
        self.cache = .init(.kvcache(cache, nil))
        self.processing = processing
        self.generateParameters = generateParameters
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
    }

    /// Produces a response to a prompt.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - role: the message role (defaults to `.user`)
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    /// - Returns: the model's response
    public func respond(
        to prompt: String,
        role: Chat.Message.Role = .user,
        images: consuming [UserInput.Image],
        videos: consuming [UserInput.Video]
    ) async throws -> String {
        var output = ""
        for try await chunk in streamResponse(
            to: prompt, role: role, images: images, videos: videos
        ) {
            output += chunk
        }
        return output
    }

    /// Produces a response to a prompt.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - role: the message role (defaults to `.user`)
    ///   - image: optional image (for use with VLMs)
    ///   - video: optional video (for use with VLMs)
    /// - Returns: the model's response
    public func respond(
        to prompt: String,
        role: Chat.Message.Role = .user,
        image: UserInput.Image? = nil,
        video: UserInput.Video? = nil
    ) async throws -> String {
        try await respond(
            to: prompt,
            role: role,
            images: image.map { [$0] } ?? [],
            videos: video.map { [$0] } ?? []
        )
    }

    /// Produces a streaming response to a prompt as Strings.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - role: the message role (defaults to `.user`)
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    /// - Returns: a stream of string chunks from the model
    public func streamResponse(
        to prompt: String,
        role: Chat.Message.Role = .user,
        images: consuming [UserInput.Image],
        videos: consuming [UserInput.Video]
    ) -> AsyncThrowingStream<String, Error> {
        streamMap(to: prompt, role: role, images: images, videos: videos) {
            $0.chunk
        }
    }

    /// Produces a streaming response to a prompt as `Generation`.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - role: the message role (defaults to `.user`)
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    /// - Returns: a stream of `Generation` from the model
    public func streamDetails(
        to prompt: String,
        role: Chat.Message.Role = .user,
        images: consuming [UserInput.Image],
        videos: consuming [UserInput.Video]
    ) -> AsyncThrowingStream<Generation, Error> {
        streamMap(to: prompt, role: role, images: images, videos: videos) {
            $0
        }
    }

    /// Produces a streaming response after appending structured chat messages.
    public func streamResponse(
        to messages: consuming [Chat.Message]
    ) -> AsyncThrowingStream<String, Error> {
        streamMap(messages: messages) { $0.chunk }
    }

    /// Produces a detailed streaming response after appending structured chat messages.
    public func streamDetails(
        to messages: consuming [Chat.Message]
    ) -> AsyncThrowingStream<Generation, Error> {
        streamMap(messages: messages) { $0 }
    }

    /// Produces a streaming response to a prompt by transforming the
    /// raw `Generation` values.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    /// - Returns: a stream of transformed values from the model
    private func streamMap<R: Sendable>(
        to prompt: String,
        role: Chat.Message.Role,
        images: consuming [UserInput.Image],
        videos: consuming [UserInput.Video],
        transform: @Sendable @escaping (Generation) -> R?
    ) -> AsyncThrowingStream<R, Error> {
        streamMap(
            messages: [
                .init(role: role, content: prompt, images: images, videos: videos)
            ],
            transform: transform
        )
    }

    private func streamMap<R: Sendable>(
        messages: consuming [Chat.Message],
        transform: @Sendable @escaping (Generation) -> R?
    ) -> AsyncThrowingStream<R, Error> {
        let (stream, continuation) = AsyncThrowingStream<R, Error>.makeStream()

        // images and videos are not Sendable (MLXArray) but they are consumed
        // and are only being sent to the inner async
        let initialMessages = SendableBox<[Chat.Message]>(messages)

        let task = Task {
            [
                model,
                instructions, processing, tools, toolDispatch,
                additionalContext, cache, generateParameters
            ] in
            do {
                try await cache.update { cache in

                    // these are all Sendable
                    let processor = await model.processor
                    let tokenizer = await model.tokenizer
                    let modelConfiguration = await model.configuration

                    var messages: [Chat.Message] = []
                    if let instructions {
                        messages.append(.system(instructions))
                    }

                    // prepare the cache, if needed.  note:
                    // this is using the LanguageModel (not Sendable) outside
                    // the protective lock.  Assuming the weights are not
                    // being mutated behind the scenes, this will obey the MLXArray
                    // contract that they be evaluated if used across threads.
                    // This is internal to the implementation and this technique
                    // should not be used in calling code.
                    //
                    // The benefit is that callers can be running multiple
                    // ChatSessions in parallel, as long as the instances
                    // are distinct.  In particular the KVCache cannot
                    // be shared and that is the lock that is held here.

                    let model = await model.perform { context in
                        SendableBox(context.model)
                    }.consume()

                    let replaysTranscript = modelConfiguration.toolCallFormat != .gptOSS
                    var kvCache: [KVCache]
                    var continuationState: LMOutput.State?
                    // A cache created by this session has a recoverable prompt
                    // transcript; a caller-provided raw cache does not.  Tool
                    // restarts must preserve that distinction so templates see
                    // earlier user turns without replaying raw-cache tokens.
                    var replayHistory: [Chat.Message]?
                    switch cache {
                    case .empty:
                        kvCache = try model.newCache(parameters: generateParameters)
                        continuationState = nil
                        replayHistory = replaysTranscript ? [] : nil

                    case .kvcache(let array, let state):
                        kvCache = array
                        continuationState = state
                        replayHistory = nil

                    case .history(let history):
                        // the KVCache is represented by a chat history
                        kvCache = try model.newCache(parameters: generateParameters)
                        continuationState = nil
                        messages.append(contentsOf: history)
                        replayHistory = replaysTranscript ? history : nil
                    }
                    let stateBox = ContinuationStateBox(continuationState)
                    defer { cache = .kvcache(kvCache, stateBox.value) }

                    // prepare the input
                    let initial = initialMessages.consume()
                    messages.append(contentsOf: initial)
                    replayHistory?.append(contentsOf: initial)

                    // A Harmony tool call is surfaced before its final control
                    // token is fed back into the cache.  Its restart must carry
                    // that one token with the tool-result suffix.
                    var harmonyCallNeedsPrefill = false

                    // loop can restart on tool calls
                    restart: while !messages.isEmpty {
                        let userInput = UserInput(
                            chat: messages, processing: processing,
                            tools: tools, additionalContext: additionalContext)
                        var input = try await processor.prepare(input: userInput)
                        if modelConfiguration.toolCallFormat == .gptOSS,
                            messages.contains(where: {
                                $0.role == .assistant && $0.tool?.calls?.isEmpty == false
                            }),
                            let callToken = tokenizer.convertTokenToId("<|call|>")
                        {
                            // Harmony's rendered tool-result transcript starts
                            // with a cold `[start, call]` prefix, but the live
                            // cache already contains the generated call (and its
                            // private analysis).  Refeeding that prefix would
                            // diverge from the cache and expose the protocol's
                            // control frames as user-visible output.  Append
                            // only the result suffix after the last call commit.
                            let tokens = input.text.tokens.asArray(Int.self)
                            if let callIndex = tokens.lastIndex(of: callToken)
                            {
                                let suffixStart = harmonyCallNeedsPrefill ? callIndex : callIndex + 1
                                if suffixStart < tokens.count {
                                    let suffix = MLXArray(Array(tokens[suffixStart...]))
                                        .expandedDimensions(axis: 0)
                                    input = LMInput(text: .init(tokens: suffix))
                                }
                                harmonyCallNeedsPrefill = false
                            }
                        }
                        messages.removeAll()

                        // generate output
                        let iterator = try TokenIterator(
                            input: input, model: model, cache: kvCache,
                            parameters: generateParameters,
                            initialState: stateBox.value,
                            onStateChange: { stateBox.value = $0 })

                        // Declared tools flow into the streaming parser too
                        // (not just the prompt template): with schemas
                        // present the parsers reject undeclared function
                        // names instead of surfacing/dispatching them.
                        let (stream, task) = MLXLMCommon.generateTask(
                            promptTokenCount: input.text.tokens.size,
                            modelConfiguration: modelConfiguration,
                            tokenizer: tokenizer,
                            iterator: iterator,
                            tools: tools
                        )

                        var pendingToolCalls: [ToolCall] = []

                        for await item in stream {
                            // collect tool calls for dispatch; if no
                            // toolDispatch the caller handles them via
                            // the transform (streamDetails path)
                            if let toolCall = item.toolCall, toolDispatch != nil {
                                pendingToolCalls.append(toolCall)
                                if modelConfiguration.toolCallFormat == .gptOSS {
                                    // Harmony commits one tool call per turn.
                                    // Stop before the next iterator step feeds
                                    // `<|call|>` into the cache; the restart
                                    // splices that token with the result below.
                                    harmonyCallNeedsPrefill = true
                                    task.cancel()
                                    break
                                }
                            } else if let value = transform(item) {
                                if case .terminated = continuation.yield(value) {
                                    break
                                }
                            }
                        }

                        // wait for the task to complete -- this is important in
                        // the case where we broke the loop early as the generation
                        // work may continue (briefly) and use the KVCache
                        await task.value

                        // dispatch all tool calls from this generation pass
                        if let toolDispatch, !pendingToolCalls.isEmpty,
                            !Task.isCancelled
                        {
                            // The generated call is already resident in the live
                            // cache, but a restarted chat template still needs a
                            // structured assistant turn to associate each result
                            // with its call.  Do not replay generated text here:
                            // that would duplicate tokens in a raw KV cache and
                            // leaks Harmony control frames into cold transcripts.
                            let assistant = Chat.Message.assistant("", toolCalls: pendingToolCalls)
                            if replayHistory != nil {
                                replayHistory!.append(assistant)
                            } else {
                                messages.append(assistant)
                            }
                            for toolCall in pendingToolCalls {
                                let toolResult = try await toolDispatch(toolCall)
                                let result = Chat.Message.tool(
                                    toolResult, id: toolCall.id, name: toolCall.function.name)
                                if replayHistory != nil {
                                    replayHistory!.append(result)
                                } else {
                                    messages.append(result)
                                }
                            }
                            if let replayHistory {
                                // This template must be re-prefilled from a cold
                                // cache. The live cache already includes the
                                // original prompt and generated tool call, while
                                // `replayHistory` is the complete transcript.
                                // Replaying both would duplicate that prefix.
                                kvCache = try model.newCache(parameters: generateParameters)
                                stateBox.value = nil
                                messages = replayHistory
                            }
                            continue restart
                        }
                    }

                    continuation.finish()
                }
            } catch {
                continuation.finish(throwing: error)
            }
        }

        continuation.onTermination = { _ in
            task.cancel()
        }

        return stream
    }

    /// Produces a streaming response to a prompt.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - image: optional image (for use with VLMs)
    ///   - video: optional video (for use with VLMs)
    /// - Returns: a stream of string chunks from the model
    public func streamResponse(
        to prompt: String,
        image: UserInput.Image? = nil,
        video: UserInput.Video? = nil
    ) -> AsyncThrowingStream<String, Error> {
        streamResponse(
            to: prompt,
            images: image.map { [$0] } ?? [],
            videos: video.map { [$0] } ?? []
        )
    }

    /// Clear the session history and cache, preserving system instructions.
    public func clear() async {
        await cache.update { cache in
            cache = .empty
        }
    }

    /// Wait for exclusive access to the KVCache.
    ///
    /// This is useful for cases where a program is terminating and wants to ensure that any
    /// async operations are complete.
    public func synchronize() async {
        await cache.read { _ in }
    }

    /// Visit the current cache value, if realized as a `[KVCache]`.
    ///
    /// This method is meant for test support.
    func withCache<R: Sendable>(_ body: @Sendable ([KVCache]?) async throws -> R) async rethrows
        -> R?
    {
        try await cache.read { cache in
            switch cache {
            case .kvcache(let cache, _):
                return try await body(cache)
            default:
                return try await body(nil)
            }
        }
    }

    /// Saves the current KV cache to disk.
    ///
    /// Use one of the initializers that accept a `cache` parameter together with
    /// ``loadPromptCache(url:)`` to restore the saved cache in a future session.
    ///
    /// - Parameter url: the file URL to write the cache to
    /// - Throws: ``ChatSessionError/noCacheAvailable`` if no generation has occurred yet,
    ///   or any error thrown by the underlying file write
    public func saveCache(to url: URL) async throws {
        try await cache.read { cache in
            switch cache {
            case .kvcache(let cache, let state):
                try savePromptCache(url: url, cache: cache, state: state)
            default:
                throw ChatSessionError.noCacheAvailable
            }
        }
    }
}

/// Errors thrown by ``ChatSession``.
public enum ChatSessionError: LocalizedError {
    /// ``ChatSession/saveCache(to:)`` was called before any generation occurred.
    case noCacheAvailable

    public var errorDescription: String? {
        "No KV cache is available. Call respond() or streamResponse() before saveCache(to:)."
    }
}
