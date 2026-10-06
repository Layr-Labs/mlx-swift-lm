import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Adapter only: all sampling, commit and state ownership remain in the native
/// session. It never advances an AR sampler or constraint on a draft canvas.
private final class DiffusionGemmaEngineSession: CBv2NativeBlockSession {
    let native: DiffusionGemmaGenerationSession
    private let initialCacheOutcome: CBv2PrefixCacheOutcome
    private let matchedPrefix: Int
    private let cacheTier: CBv2PrefixCacheTier
    private var onFinish: ((Bool) -> Void)?
    init(
        _ native: DiffusionGemmaGenerationSession, cacheOutcome: CBv2PrefixCacheOutcome = .disabled,
        matchedPrefix: Int = 0, cacheTier: CBv2PrefixCacheTier = .memorySnapshot,
        onFinish: ((Bool) -> Void)? = nil
    ) {
        self.native = native
        self.initialCacheOutcome = cacheOutcome
        self.matchedPrefix = matchedPrefix
        self.cacheTier = cacheTier
        self.onFinish = onFinish
    }
    var retainedBytes: Int { native.retainedStateBytes }
    var sharedStorageBytes: Int { native.sharedPageBytes }
    var activeTokenCount: Int { native.computedTokenCount }
    var generatedTokenCount: Int { native.generatedTokenCount }
    var prefixUsage: CBv2Usage {
        let saved = native.reusedPromptTokenCount
        let outcome: CBv2PrefixCacheOutcome =
            saved > 0
            ? .hit
            : matchedPrefix > 0
                ? (native.phase == .failed ? .adoptionFailed : .skippedPolicy)
                : initialCacheOutcome
        return .init(
            promptTokens: 0, completionTokens: 0, prefixCacheHitTokens: saved,
            prefixCacheOutcome: outcome, prefixCacheTier: saved > 0 ? cacheTier : nil,
            prefixCacheMatchedTokens: saved, prefixCachePrefillTokensSaved: saved,
            prefixCacheStrategy: saved > 0 ? .direct : nil)
    }
    func finish(reason: CBv2FinishReason) {
        let success: Bool
        switch reason {
        case .stop, .length: success = true
        default: success = false
        }
        onFinish?(success)
        onFinish = nil
    }
    func cancel() {
        onFinish?(false)
        onFinish = nil
        native.cancel()
    }
    func advanceNative() throws -> CBv2NativeBlockStep {
        switch try native.advance() {
        case .prefill(let computed, let complete):
            return .prefill(computedTokens: computed, complete: complete)
        case .refined: return .progress
        case .committed(let tokens, let terminal):
            let reason: CBv2FinishReason? =
                terminal
                ? (native.result?.finishReason == "stop" ? .stop : .length) : nil
            return .committed(
                tokens: tokens.map(Int.init), stopToken: native.terminalStopToken.map(Int.init),
                finishReason: reason)
        }
    }
}

/// Frozen evaluated weights shared only through the engine's single execution
/// queue. No request state belongs to this owner and no GPU arrays cross queues.
private final class DiffusionGemmaEngineOwner: @unchecked Sendable {
    let model: DiffusionGemma
    init(_ model: DiffusionGemma) { self.model = model }
}

private struct DiffusionGemmaEnginePolicy: Sendable {
    let configuration: DiffusionGemmaTextConfiguration
    let generation: DiffusionGemmaGenerationConfiguration
    let eos: [Int]
    let itemBytes: Int
    let canvasLength: Int
    let prefillChunkSize: Int
    let supportsMedia: Bool
    let prefixRestoreBudget: Int
    let pagedMemory: CBv2NativeBlockPagedMemory?

    func reservation(_ request: CBv2Request) throws -> Int {
        let count = request.promptTokens.count
        guard count > 0, request.maxTokens > 0,
            count <= configuration.maxPositionEmbeddings - request.maxTokens,
            request.promptTokens.allSatisfy({ $0 >= 0 && $0 < configuration.vocabularySize }),
            request.stopTokens.allSatisfy({ $0 >= 0 && $0 < configuration.vocabularySize })
        else { throw CBv2NativeBlockError.unsupportedRequest("token range/context capacity") }
        let p = request.sampling
        guard p.temperature.isFinite, p.temperature >= 0,
            p.topP == 1, p.topK == 0, p.minP == 0,
            p.repetitionPenalty == 1, p.frequencyPenalty == 0, p.presencePenalty == 0,
            p.logitBias.isEmpty, p.topLogprobs == 0
        else {
            throw CBv2NativeBlockError.unsupportedRequest("non-native sampling/logprobs control")
        }
        guard request.positionState == nil,
            request.hybridPrefixIdentity == nil || request.multimodal != nil,
            request.tokenConstraint == nil
        else {
            throw CBv2NativeBlockError.unsupportedRequest(
                "unprepared media/positions/diffusion constraint")
        }
        var mediaTokens = 0
        var mediaChunk = 0
        if let media = request.multimodal {
            guard supportsMedia else {
                throw CBv2NativeBlockError.unsupportedRequest("model has no native vision tower")
            }
            guard media.attention == .bidirectionalSpans, media.positionState == nil,
                media.deepstackEmbeddings == nil
            else {
                throw CBv2NativeBlockError.unsupportedRequest("non-native visual control")
            }
            try DiffusionGemmaVisualEmbeddings.validate(spans: media.spans, promptCount: count)
            mediaTokens = media.spans.reduce(0) { $0 + $1.length }
            mediaChunk =
                DiffusionGemmaVisualEmbeddings.coalesced(media.spans).map(\.length).max() ?? 0
        }
        let mediaGeometry = try request.multimodal.map {
            try DiffusionGemmaPrefillGeometry(promptCount: count,
                chunkSize: prefillChunkSize, spans: $0.spans)
        }
        func product(_ factors: Int...) throws -> Int {
            var result = 1
            for factor in factors {
                let next = result.multipliedReportingOverflow(by: factor)
                guard factor >= 0, !next.overflow else {
                    throw CBv2NativeBlockError.invalidConfiguration
                }
                result = next.partialValue
            }
            return result
        }
        func sum(_ parts: [Int]) throws -> Int {
            try parts.reduce(0) { value, part in
                let next = value.addingReportingOverflow(part)
                guard !next.overflow else { throw CBv2NativeBlockError.invalidConfiguration }
                return next.partialValue
            }
        }
        let total = count + request.maxTokens
        let initial = min(
            configuration.maxPositionEmbeddings,
            count + min(256, configuration.maxPositionEmbeddings - count))
        var capacity = initial
        while capacity < total {
            capacity = min(
                configuration.maxPositionEmbeddings,
                capacity + min(capacity, configuration.maxPositionEmbeddings - capacity))
        }
        var parts = [Int]()
        if let pagedMemory {
            parts.append(pagedMemory.nominalBytes(tokens: total))
            // Exact page storage does not remove the native attention graph's
            // temporary gathered K/V plus prefix+canvas concatenation. Charge
            // those bounded views separately; only target pages overlap the
            // backend physical floor. No activation/OS reserve is reduced.
            for kind in configuration.layerTypes {
                let windowed = kind == "sliding_attention"
                let history = windowed ? min(total, configuration.slidingWindow) : total
                let visible = try sum([history, max(prefillChunkSize, canvasLength, mediaChunk)])
                let logical = try product(
                    visible,
                    windowed
                        ? configuration.keyValueHeads
                        : (configuration.globalKeyValueHeads ?? configuration.keyValueHeads),
                    windowed ? configuration.headDimension : configuration.globalHeadDimension,
                    itemBytes)
                parts.append(
                    try product(4, Memory.allocationFootprintUpperBound(byteCount: logical)))
            }
        } else {
            for kind in configuration.layerTypes {
                if kind == "sliding_attention" {
                    // Ring + pre-eviction chunk views retained by native storage.
                    let tokens = try sum([
                        configuration.slidingWindow, configuration.slidingWindow - 1,
                        max(prefillChunkSize, canvasLength, mediaChunk),
                    ])
                    parts.append(
                        try product(
                            2, tokens, configuration.keyValueHeads, configuration.headDimension,
                            itemBytes))
                } else {
                    parts.append(
                        try product(
                            2, capacity,
                            configuration.globalKeyValueHeads ?? configuration.keyValueHeads,
                            configuration.globalHeadDimension, itemBytes))
                }
            }
        }
        // Conservative request-local logits/conditioning payload allowance;
        // device workspace/activation and OS reserves stay with provider admission.
        parts.append(try product(3, canvasLength, configuration.vocabularySize, 4))
        parts.append(try product(canvasLength, sum([generation.stabilityThreshold, 4]), 4))
        parts.append(try product(total, 16))
        // Request-retained visual values plus possible native-dtype cast. The
        // original callback owner can coexist until retirement; count both.
        parts.append(try product(mediaTokens, configuration.hiddenSize, 8))
        if let mediaGeometry {
            // Engine capture policy and generation session can overlap with
            // separate host boundary arrays. Fund their capacity/bookkeeping;
            // this is not a second device KV charge. Text admission is unchanged.
            parts.append(try sum([try product(mediaGeometry.boundaries.count, 32), 1024]))
        }
        if prefixRestoreBudget > 0, request.prefixCacheEnabled, request.cacheSalt != nil,
            request.multimodal == nil || request.hybridPrefixIdentity != nil
        {
            // Cover both a pinned restore source and every staged compact
            // endpoint, including if the slot is resliced during this request.
            let positions = mediaGeometry?.capturePositions ?? Set([
                min(count, prefillChunkSize), count / prefillChunkSize * prefillChunkSize, count,
            ]).filter { $0 > 0 }
            var snapshots = [Int]()
            for position in positions {
                var source = [
                    try product(position, 8),
                    try sum([try product(configuration.layerCount, 512), 64 * 1024]),
                ]
                for kind in configuration.layerTypes {
                    let windowed = kind == "sliding_attention"
                    let logical = try product(
                        windowed ? min(position, configuration.slidingWindow) : position,
                        windowed
                            ? configuration.keyValueHeads
                            : (configuration.globalKeyValueHeads ?? configuration.keyValueHeads),
                        windowed ? configuration.headDimension : configuration.globalHeadDimension,
                        itemBytes)
                    source.append(
                        try product(2, Memory.allocationFootprintUpperBound(byteCount: logical)))
                }
                snapshots.append(try sum(source))
            }
            parts.append(min(prefixRestoreBudget, try sum(snapshots)))
        }
        return try sum(parts)
    }

    func recipe(for request: CBv2Request) throws -> DiffusionGemmaGenerationConfiguration {
        try DiffusionGemmaGenerationConfiguration(
            maxNewTokens: request.maxTokens, maxDenoisingSteps: generation.maxDenoisingSteps,
            sampler: generation.sampler, minimumTemperature: generation.minimumTemperature,
            maximumTemperature: generation.maximumTemperature,
            stabilityThreshold: generation.stabilityThreshold,
            confidenceThreshold: generation.confidenceThreshold, bosTokenId: generation.bosTokenId,
            padTokenId: generation.padTokenId,
            eosTokenIds: Array(Set(eos).union(request.stopTokens)).sorted())
    }
}

extension DiffusionGemmaContext {
    /// Construct the native engine that can be consumed by DarkBloom's existing
    /// EngineV2Bridge. This does not itself register/advertise a provider slot.
    /// Media, constraints and persistent/paged caches must be integrated before
    /// declaring complete serving support; unsupported controls fail explicitly.
    public func makeNativeEngine(
        kvBytesCapacity: Int, maxConcurrentRequests: Int = 4,
        maxWaiting: Int = 64, prefillChunkSize: Int = 512,
        prefixCache: DiffusionGemmaResidentPrefixConfiguration? = nil,
        completePrefixCache: (any CBv2NativeBlockPrefixCache)? = nil,
        retainMemoryPrefixes: Bool = true,
        pagedConfiguration: PagedKVPoolConfig? = nil,
        processMemoryOwner: (any CBv2ProcessMemoryOwner)? = nil,
        loopConfig: CBv2EngineLoopConfig = .init()
    ) throws -> CBv2NativeBlockEngine {
        guard prefillChunkSize > 0 else { throw CBv2NativeBlockError.invalidConfiguration }
        let embedding = model.model.decoder.embedTokens
        let dtype = (embedding as? QuantizedEmbedding)?.scales.dtype ?? embedding.weight.dtype
        guard dtype.isFloatingPoint else { throw CBv2NativeBlockError.invalidConfiguration }
        let pageMemory: CBv2NativeBlockPagedMemory?
        if var pagedConfiguration {
            let types = Array(repeating: dtype, count: model.configuration.textConfig.layerCount)
            guard pagedConfiguration.capacityBytes == kvBytesCapacity,
                pagedConfiguration.dtype == dtype,
                pagedConfiguration.layerDTypes == nil || pagedConfiguration.layerDTypes == types,
                pagedConfiguration.maxPrefillChunk
                    >= max(prefillChunkSize, model.configuration.canvasLength)
            else { throw CBv2NativeBlockError.invalidConfiguration }
            pagedConfiguration.layerDTypes = types
            pageMemory = try .init(
                layerKinds: model.configuration.textConfig.diffusionPagedLayerKinds,
                configuration: pagedConfiguration, processMemoryOwner: processMemoryOwner)
        } else {
            guard processMemoryOwner == nil else { throw CBv2NativeBlockError.invalidConfiguration }
            pageMemory = nil
        }
        let persistence: DiffusionGemmaPrefixPersistence?
        if let completePrefixCache {
            guard let prefixCache,
                prefixCache.artifactIdentity == completePrefixCache.identity.modelAggregateHash,
                prefixCache.templateIdentity == completePrefixCache.identity.promptContractID,
                prefixCache.numericalProfile == completePrefixCache.identity.numericsFingerprint
            else {
                throw CBv2NativeBlockError.invalidConfiguration
            }
            persistence = try .init(
                store: completePrefixCache,
                codec: model.model.decoder.makePersistentPrefixCodec(
                    verifiedIdentity: completePrefixCache.identity,
                    kvDType: dtype, prefillChunkSize: prefillChunkSize), chunkSize: prefillChunkSize
            )
        } else {
            persistence = nil
        }
        let policy = DiffusionGemmaEnginePolicy(
            configuration: model.configuration.textConfig, generation: generationConfiguration,
            eos: generationConfiguration.eosTokenIds ?? model.configuration.base.eosTokenIds?.values
                ?? model.configuration.textConfig.eosTokenIds ?? [],
            itemBytes: dtype.size, canvasLength: model.configuration.canvasLength,
            prefillChunkSize: prefillChunkSize,
            supportsMedia: model.configuration.visionConfig != nil,
            prefixRestoreBudget: prefixCache?.maximumBytes ?? 0, pagedMemory: pageMemory)
        let owner = DiffusionGemmaEngineOwner(model)
        let cache = prefixCache.map {
            DiffusionGemmaResidentPrefixCache(
                configuration: $0, chunkSize: prefillChunkSize,
                retainsPublishedEntries: retainMemoryPrefixes, persistence: persistence)
        }
        let shared = cache.map { cache in
            CBv2NativeBlockSharedResources(
                maximumBytes: cache.configuration.maximumBytes,
                retainedBytes: { cache.retainedBytes }, trim: { cache.trim(to: $0) },
                clear: { cache.clear() })
        }
        let planner: CBv2NativeBlockCheckpointPlanner?
        if let cache, let persistence {
            planner = { manifest, request, engine in
                guard let identity = try cache.identity(for: request) else {
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                }
                return try persistence.codec.importPlan(
                    manifest, prefixIdentity: identity,
                    promptTokens: request.promptTokens, chunkSize: prefillChunkSize,
                    maximumNewTokens: request.maxTokens,
                    prefillGeometry: try request.multimodal.map {
                        try DiffusionGemmaPrefillGeometry(promptCount: request.promptTokens.count,
                            chunkSize: prefillChunkSize, spans: $0.spans)
                    }, engine: engine)
            }
        } else {
            planner = nil
        }
        let engine = try CBv2NativeBlockEngine(
            tokenizer: tokenizer, kvBytesCapacity: kvBytesCapacity,
            maxConcurrentRequests: maxConcurrentRequests, maxWaiting: maxWaiting,
            loopConfig: loopConfig,
            sharedResources: shared,
            pagedMemory: pageMemory,
            completeNativePrefixCache: completePrefixCache, checkpointPlanner: planner,
            reservationForRequest: { try policy.reservation($0) },
            makeSession: { request, cancellation in
                let requestKey = UUID()
                let tokens = request.promptTokens.map(Int32.init)
                let geometry = try request.multimodal.map {
                    try DiffusionGemmaPrefillGeometry(promptCount: tokens.count,
                        chunkSize: prefillChunkSize, spans: $0.spans)
                }
                let hasUnboundMedia =
                    request.multimodal == nil && (tokens.contains(Int32(owner.model.configuration.imageTokenId))
                    || owner.model.configuration.videoTokenId.map { tokens.contains(Int32($0)) }
                        == true)
                let identity = try !hasUnboundMedia ? cache?.identity(for: request) : nil
                let durable = identity.flatMap {
                    persistence?.take(request: request, identity: $0, geometry: geometry)
                }
                let diskHit = durable?.checkpoint
                let hit =
                    diskHit ?? identity.flatMap { cache?.lookup(tokens: tokens, identity: $0, geometry: geometry) }
                let outcome: CBv2PrefixCacheOutcome =
                    cache == nil
                    ? .disabled
                    : cache?.enabled == false
                        ? .skippedCapacity
                        : identity == nil
                            ? .skippedPolicy : durable?.failed == true ? .adoptionFailed : .miss
                let capture: ((DiffusionGemmaRequestCache, Bool) throws -> Void)?
                if let cache, let identity, hit?.tokenCount != tokens.count {
                    capture = { state, _ in
                        try cache.capture(
                            model: owner.model, cache: state, promptTokens: tokens,
                            request: requestKey, identity: identity, geometry: geometry)
                    }
                } else {
                    capture = nil
                }
                let native = try DiffusionGemmaGenerationSession(
                    model: owner.model,
                    promptTokenIds: MLXArray(tokens).reshaped(
                        1, request.promptTokens.count),
                    generation: policy.recipe(for: request),
                    seed: request.sampling.seed ?? UInt64.random(in: .min ... .max),
                    samplingTemperature: request.sampling.temperature,
                    prefillChunkSize: prefillChunkSize, pagedBackend: pageMemory?.backend,
                    multimodal: request.multimodal,
                    prefixCheckpoint: hit, prefixIdentity: identity, onEncodedBoundary: capture,
                    isCancelled: { cancellation.isCancelled })
                return DiffusionGemmaEngineSession(
                    native, cacheOutcome: outcome, matchedPrefix: hit?.tokenCount ?? 0,
                    cacheTier: diskHit == nil ? .memorySnapshot : .snapshot,
                    onFinish: cache.map { cache in
                        {
                            cache.finish(
                                request: requestKey, successful: $0, input: request, reused: hit)
                        }
                    })
            })
        persistence?.engine = engine
        return engine
    }
}
