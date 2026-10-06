import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// One request's native block state, owned by a single engine execution lane.
/// Each advance performs at most one encoder forward or one denoising step.
/// No provisional canvas is exposed to streaming, billing or prefix consumers.
public final class DiffusionGemmaGenerationSession {
    public enum Phase: Sendable, Equatable {
        case prefill, denoising, encodeCommitted, finished, cancelled, failed
    }
    public enum Step: Sendable {
        case prefill(computedTokens: Int, complete: Bool)
        case refined
        case committed(tokens: [Int32], terminal: Bool)
    }

    public private(set) var phase: Phase = .prefill
    public private(set) var result: DiffusionGemmaGenerationResult?
    public private(set) var generatedTokenCount = 0
    public private(set) var denoisingSteps = 0
    public private(set) var reusedPromptTokenCount = 0
    public private(set) var terminalStopToken: Int32?
    public var committedCachePosition: Int { cache?.position ?? 0 }
    public var retainedCacheBytes: Int { cache?.retainedBytes ?? 0 }
    public var sharedPageBytes: Int { pagedBackend == nil ? 0 : retainedCacheBytes }
    public var retainedStateBytes: Int {
        retainedCacheBytes + (state?.retainedPayloadBytes ?? 0)
            + (prompt?.nbytes ?? 0) + (pixels?.nbytes ?? 0) + (visualBlockIds?.nbytes ?? 0)
            + (preparedMedia?.retainedBytes ?? 0)
            + (prefix?.retainedBytes ?? 0)
    }
    public var computedTokenCount: Int {
        (result == nil ? min(promptCount, cache?.position ?? 0) : promptCount) + generatedTokenCount
    }

    private let model: DiffusionGemma
    private let generation: DiffusionGemmaGenerationConfiguration
    private let temperature: Float
    private let outputLimit: Int
    private let eos: Set<Int32>
    private let promptCount: Int
    private let prefillChunkSize: Int?
    private let mediaPrefillGeometry: DiffusionGemmaPrefillGeometry?
    private let pagedBackend: PagedKVBackend?
    private let started = ProcessInfo.processInfo.systemUptime
    private let isCancelled: () -> Bool
    private var prompt: MLXArray?
    private var pixels: MLXArray?
    private var visualOutputLengths: [Int]?
    private var visualBlockIds: MLXArray?
    private var preparedMedia: DiffusionGemmaVisualEmbeddings?
    private var prefix: DiffusionGemmaPrefixCheckpoint?
    private let prefixIdentity: DiffusionGemmaPrefixIdentity?
    private var onPromptCheckpoint: ((DiffusionGemmaPrefixCheckpoint) throws -> Void)?
    private var onEncodedBoundary: ((DiffusionGemmaRequestCache, Bool) throws -> Void)?
    private var cache: DiffusionGemmaRequestCache?
    private var state: DiffusionGemmaDenoisingState?
    private var key: MLXArray
    private var output = [Int32]()
    private var committedCanvas: [Int32]?
    private var canvasCount = 0
    private var workTokens = 0
    private var prefillSeconds: Double = 0
    private var firstOutputSeconds: Double?

    public init(
        model: DiffusionGemma, promptTokenIds: MLXArray,
        generation: DiffusionGemmaGenerationConfiguration, seed: UInt64,
        samplingTemperature: Float = 1, prefillChunkSize: Int? = nil,
        pagedBackend: PagedKVBackend? = nil,
        pixelValues: MLXArray? = nil, visualOutputLengths: [Int]? = nil,
        visualBlockIds: MLXArray? = nil,
        multimodal: CBv2MultimodalInput? = nil,
        prefixCheckpoint: DiffusionGemmaPrefixCheckpoint? = nil,
        prefixIdentity: DiffusionGemmaPrefixIdentity? = nil,
        onPromptCheckpoint: ((DiffusionGemmaPrefixCheckpoint) throws -> Void)? = nil,
        onEncodedBoundary: ((DiffusionGemmaRequestCache, Bool) throws -> Void)? = nil,
        isCancelled: @escaping () -> Bool = { Task.isCancelled }
    ) throws {
        guard promptTokenIds.ndim == 2, promptTokenIds.dim(0) == 1, promptTokenIds.dim(1) > 0,
            samplingTemperature.isFinite, samplingTemperature >= 0,
            prefillChunkSize == nil || prefillChunkSize! > 0
        else { throw DiffusionGemmaModelError.invalidInput("generation inputs") }
        let count = promptTokenIds.dim(1)
        let limit = try generation.outputTokenLimit(promptTokenCount: count)
        guard count <= model.configuration.textConfig.maxPositionEmbeddings - limit else {
            throw DiffusionGemmaModelError.invalidInput("logical context capacity")
        }
        if let pagedBackend {
            let largestChunk = max(prefillChunkSize ?? count, model.configuration.canvasLength,
                multimodal.map { DiffusionGemmaVisualEmbeddings.coalesced($0.spans).map(\.length).max() ?? 0 } ?? 0)
            guard largestChunk <= pagedBackend.pool.config.maxPrefillChunk else {
                throw DiffusionGemmaModelError.invalidInput("native paged write quantum exceeds pool bound")
            }
        }
        let eosValues =
            generation.eosTokenIds ?? model.configuration.base.eosTokenIds?.values
            ?? model.configuration.textConfig.eosTokenIds ?? []
        guard
            eosValues.allSatisfy({ $0 >= 0 && $0 < model.configuration.textConfig.vocabularySize })
        else {
            throw DiffusionGemmaModelError.invalidInput("EOS token range")
        }
        if prefixCheckpoint != nil || onPromptCheckpoint != nil || onEncodedBoundary != nil {
            guard let prefixIdentity, pixelValues == nil,
                visualOutputLengths == nil, visualBlockIds == nil
            else { throw DiffusionGemmaModelError.invalidInput("unbound prefix identity/media") }
            if multimodal != nil {
                guard prefixIdentity.mediaIdentity != nil, prefixIdentity.media != "text-only" else {
                    throw DiffusionGemmaModelError.invalidInput("unbound prefix identity/media")
                }
            } else {
                guard prefixIdentity.mediaIdentity == nil, prefixIdentity.media == "text-only",
                    !(promptTokenIds .== model.configuration.imageTokenId).any().item(Bool.self),
                    model.configuration.videoTokenId == nil
                        || !(promptTokenIds .== model.configuration.videoTokenId!).any().item(Bool.self)
                else { throw DiffusionGemmaModelError.invalidInput("unbound prefix identity/media") }
            }
        }
        guard
            multimodal == nil
                || (pixelValues == nil && visualBlockIds == nil && visualOutputLengths == nil)
        else {
            throw DiffusionGemmaModelError.invalidInput("ambiguous native media source")
        }
        guard multimodal == nil || model.configuration.visionConfig != nil else {
            throw DiffusionGemmaModelError.invalidInput("native media requires a vision model")
        }
        // Chunked media needs processor-owned spans and slicing. Until that
        // integration exists, refuse instead of silently splitting a visual block.
        if let prefillChunkSize, count > prefillChunkSize,
            pixelValues != nil || visualBlockIds != nil || visualOutputLengths != nil
        {
            throw DiffusionGemmaModelError.invalidInput("unprepared chunked media")
        }
        self.model = model
        self.generation = generation
        self.temperature = samplingTemperature
        self.outputLimit = limit
        self.eos = Set(eosValues.map(Int32.init))
        self.promptCount = count
        self.prefillChunkSize = prefillChunkSize
        self.mediaPrefillGeometry = try multimodal.map {
            try DiffusionGemmaPrefillGeometry(promptCount: count,
                chunkSize: prefillChunkSize ?? count, spans: $0.spans)
        }
        self.pagedBackend = pagedBackend
        self.prompt = promptTokenIds[0..., 0...]
        self.pixels = pixelValues
        self.visualOutputLengths = visualOutputLengths
        self.visualBlockIds = visualBlockIds
        self.preparedMedia = try multimodal.map {
            try DiffusionGemmaVisualEmbeddings(
                input: $0, promptCount: count,
                hiddenSize: model.configuration.textConfig.hiddenSize)
        }
        self.prefix = prefixCheckpoint
        self.prefixIdentity = prefixIdentity
        self.onPromptCheckpoint = onPromptCheckpoint
        self.onEncodedBoundary = onEncodedBoundary
        self.isCancelled = isCancelled
        self.key = MLXRandom.key(seed)
    }

    /// Cancellation is terminal: uncommitted state is dropped and cannot be
    /// resumed or donated. The engine serializes this call with advance().
    public func cancel() {
        guard phase != .finished, phase != .failed, phase != .cancelled else { return }
        phase = .cancelled
        releaseBuffers()
    }

    public func advance() throws -> Step {
        if phase == .cancelled { throw CancellationError() }
        guard phase != .finished, phase != .failed else {
            throw DiffusionGemmaModelError.invalidInput("terminal generation session")
        }
        if isCancelled() {
            cancel()
            throw CancellationError()
        }
        do {
            return try MLX.withError { errors in
                switch phase {
                case .prefill: return try prefill(errors: errors)
                case .encodeCommitted:
                    guard let cache, let committedCanvas else {
                        throw DiffusionGemmaModelError.invalidInput(
                            "missing finalized encoder state")
                    }
                    _ = try model.encode(
                        tokenIds: MLXArray(committedCanvas).reshaped(1, committedCanvas.count),
                        cache: cache)
                    eval(cache.stateArrays())
                    try errors.check()
                    self.committedCanvas = nil
                    phase = .denoising
                    return .refined
                case .denoising: return try denoise(errors: errors)
                default: throw DiffusionGemmaModelError.invalidInput("terminal generation session")
                }
            }
        } catch {
            if error is CancellationError { phase = .cancelled } else { phase = .failed }
            releaseBuffers()
            throw error
        }
    }

    private func prefill(errors: MLX.ErrorBox) throws -> Step {
        guard let prompt else { throw DiffusionGemmaModelError.invalidInput("missing prompt") }
        if cache == nil {
            if let prefix, let prefixIdentity {
                guard mediaPrefillGeometry?.permitsRestore(position: prefix.tokenCount) ?? true else {
                    throw DiffusionGemmaModelError.invalidInput("prefix splits native media quantum")
                }
                cache = try model.model.decoder.restorePrefix(
                    prefix, identity: prefixIdentity, promptTokenIds: prompt,
                    maximumSequenceLength: pagedBackend == nil ? nil : promptCount + outputLimit,
                    pagedBackend: pagedBackend)
            } else {
                cache = try model.makeCache(expectedPromptLength: promptCount,
                    maximumSequenceLength: pagedBackend == nil ? nil : promptCount + outputLimit,
                    pagedBackend: pagedBackend)
            }
            reusedPromptTokenCount = cache!.position
            prefix = nil
        }
        let cache = cache!
        let start = cache.position
        let count =
            try mediaPrefillGeometry?.chunkLength(start: start)
            ?? min(promptCount - start, prefillChunkSize ?? promptCount)
        if count > 0 {
            let chunk = prompt[0..., start ..< (start + count)]
            if let preparedMedia {
                try preparedMedia.encode(model: model, tokens: chunk, cache: cache, start: start)
            } else {
                _ = try model.encode(
                    tokenIds: chunk, cache: cache,
                    pixelValues: pixels, visualOutputLengths: visualOutputLengths,
                    visualBlockIds: visualBlockIds)
            }
        }
        eval(cache.stateArrays())
        try errors.check()
        if isCancelled() { throw CancellationError() }
        let complete = cache.position == promptCount
        if count > 0 { try onEncodedBoundary?(cache, complete) }
        if complete {
            if let onPromptCheckpoint, let prefixIdentity {
                try onPromptCheckpoint(
                    model.model.decoder.checkpoint(cache: cache, identity: prefixIdentity))
            }
            onPromptCheckpoint = nil
            onEncodedBoundary = nil
            promptCleanup()
            prefillSeconds = ProcessInfo.processInfo.systemUptime - started
            phase = .denoising
        }
        return .prefill(computedTokens: count, complete: complete)
    }

    private func nextKey() -> MLXArray {
        let parts = MLXRandom.split(key: key)
        key = parts.0
        return parts.1
    }
    private func noise() -> MLXArray {
        MLXRandom.randInt(
            Int32(0) ..< Int32(model.configuration.textConfig.vocabularySize),
            [1, model.configuration.canvasLength], key: nextKey())
    }
    private func denoise(errors: MLX.ErrorBox) throws -> Step {
        guard let cache else {
            throw DiffusionGemmaModelError.invalidInput("missing encoder state")
        }
        if state == nil {
            state = try DiffusionGemmaDenoisingState(
                initialCanvas: noise(),
                vocabularySize: model.configuration.textConfig.vocabularySize,
                embeddingDType: model.model.decoder.selfConditioningLogitsDType,
                configuration: generation)
        }
        let state = state!
        let logits = try model.denoise(
            canvasIds: state.currentCanvas, cache: cache,
            selfConditioningLogits: state.selfConditioningLogits)
        try state.stepNative(
            rawLogits: logits, samplingTemperature: temperature, nextKey: nextKey)
        try errors.check()
        eval(
            state.currentCanvas, state.argmaxCanvas, state.finishedRows,
            state.selfConditioningLogits!)
        try errors.check()
        denoisingSteps += 1
        workTokens += model.configuration.canvasLength
        guard state.remainingSteps == 0 || state.finishedRows.all().item(Bool.self) else {
            return .refined
        }
        if isCancelled() { throw CancellationError() }
        let canvas = try state.finalizedCanvas().asArray(Int32.self)
        try errors.check()
        canvasCount += 1
        var block = [Int32]()
        var finish = "length"
        for token in canvas.prefix(outputLimit - generatedTokenCount) {
            generatedTokenCount += 1
            if eos.contains(token) {
                terminalStopToken = token
                finish = "stop"
                break
            }
            block.append(token)
        }
        if !block.isEmpty {
            if firstOutputSeconds == nil {
                firstOutputSeconds = ProcessInfo.processInfo.systemUptime - started
            }
            output.append(contentsOf: block)
        }
        self.state = nil
        let terminal = finish == "stop" || generatedTokenCount == outputLimit
        if terminal {
            result = DiffusionGemmaGenerationResult(
                tokenIds: output, generatedTokenCount: generatedTokenCount, finishReason: finish,
                promptTokenCount: promptCount, reusedPromptTokenCount: reusedPromptTokenCount,
                committedCanvasCount: canvasCount, denoisingSteps: denoisingSteps,
                workTokenCount: workTokens,
                prefillSeconds: prefillSeconds, firstCommittedOutputSeconds: firstOutputSeconds,
                totalSeconds: ProcessInfo.processInfo.systemUptime - started)
            phase = .finished
            releaseBuffers()
        } else {
            committedCanvas = canvas
            phase = .encodeCommitted
        }
        return .committed(tokens: block, terminal: terminal)
    }

    private func promptCleanup() {
        prompt = nil
        pixels = nil
        visualBlockIds = nil
        visualOutputLengths = nil
        preparedMedia = nil
    }
    private func releaseBuffers() {
        promptCleanup()
        cache = nil
        state = nil
        prefix = nil
        committedCanvas = nil
        onPromptCheckpoint = nil
        onEncodedBoundary = nil
    }
}
