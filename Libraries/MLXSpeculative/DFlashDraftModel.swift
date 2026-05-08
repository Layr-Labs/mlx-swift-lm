// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

private final class DFlashMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        _up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

private final class DFlashAttention: Module {
    let config: DFlashConfiguration
    let layerType: DFlashLayerType
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: DFlashConfiguration, layerIdx: Int) {
        self.config = config
        self.layerType = config.layerTypes[layerIdx]
        self.scale = pow(Float(config.headDim), -0.5)

        _qProj.wrappedValue = Linear(
            config.hiddenSize, config.attentionHeads * config.headDim, bias: false)
        _kProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        _vProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        _oProj.wrappedValue = Linear(
            config.attentionHeads * config.headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        context: MLXArray,
        rope: RoPELayer,
        cache: KVCache
    ) throws -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var context = context
        var contextLength = context.dim(1)

        if layerType == .slidingAttention {
            guard let slidingWindow = config.slidingWindow else {
                throw DFlashError.missingSlidingWindow
            }
            let keepContext = slidingWindow - 1
            if contextLength > keepContext {
                let skip = contextLength - keepContext
                context = context[0..., skip..., 0...]
                contextLength = context.dim(1)
                if let cache = cache as? BaseKVCache {
                    cache.offset += skip
                }
            }
        }

        let baseOffset = cache.offset

        var queries = qProj(x)
        var contextKeys = kProj(context)
        var contextValues = vProj(context)
        var proposalKeys = kProj(x)
        var proposalValues = vProj(x)

        queries =
            qNorm(queries.reshaped(B, L, config.attentionHeads, -1))
            .transposed(0, 2, 1, 3)
        contextKeys =
            kNorm(contextKeys.reshaped(B, contextLength, config.kvHeads, -1))
            .transposed(0, 2, 1, 3)
        contextValues =
            contextValues.reshaped(B, contextLength, config.kvHeads, -1)
            .transposed(0, 2, 1, 3)
        proposalKeys =
            kNorm(proposalKeys.reshaped(B, L, config.kvHeads, -1))
            .transposed(0, 2, 1, 3)
        proposalValues =
            proposalValues.reshaped(B, L, config.kvHeads, -1)
            .transposed(0, 2, 1, 3)

        queries = rope(queries, offset: baseOffset + contextLength)
        contextKeys = rope(contextKeys, offset: baseOffset)
        proposalKeys = rope(proposalKeys, offset: baseOffset + contextLength)

        let (cachedKeys, cachedValues) = cache.update(
            keys: contextKeys, values: contextValues)
        let cachedLength = cachedKeys.dim(2)
        let keys = concatenated([cachedKeys, proposalKeys], axis: 2)
        let values = concatenated([cachedValues, proposalValues], axis: 2)

        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if layerType == .slidingAttention {
            let slidingWindow = config.slidingWindow!
            mask =
                cachedLength + L <= slidingWindow
                ? .causal
                : .array(createCausalMask(n: L, offset: cachedLength, windowSize: slidingWindow))
        } else {
            mask = .none
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )

        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

private final class DFlashDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DFlashAttention
    @ModuleInfo var mlp: DFlashMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: DFlashConfiguration, layerIdx: Int) {
        _selfAttn.wrappedValue = DFlashAttention(config, layerIdx: layerIdx)
        _mlp.wrappedValue = DFlashMLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize
        )
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        context: MLXArray,
        rope: RoPELayer,
        cache: KVCache
    ) throws -> MLXArray {
        let h = try x + selfAttn(inputLayerNorm(x), context: context, rope: rope, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public final class DFlashDraftModel: Module, @unchecked Sendable {
    public let config: DFlashConfiguration

    @ModuleInfo(key: "fc") public var contextProjection: Linear
    @ModuleInfo(key: "hidden_norm") public var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") private var layers: [DFlashDecoderLayer]
    @ModuleInfo public var norm: RMSNorm

    private let rope: RoPELayer
    private var targetEmbed: ((MLXArray) -> MLXArray)?
    private var targetLMHead: ((MLXArray) -> MLXArray)?
    private var boundTargetID: ObjectIdentifier?

    public init(config: DFlashConfiguration) {
        self.config = config
        _contextProjection.wrappedValue = Linear(
            config.targetHiddenSize, config.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            DFlashDecoderLayer(config, layerIdx: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.rope = initializeRope(
            dims: config.headDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
        super.init()
    }

    public func bind(target: any DFlashTargetModel) throws {
        let newID = ObjectIdentifier(target as AnyObject)
        if let existing = boundTargetID {
            if existing == newID { return }
            throw DFlashError.rebindForbidden
        }

        try validateCompatibility(with: target)
        self.targetEmbed = { [target] tokens in
            target.embedTokensForDFlash(tokens)
        }
        self.targetLMHead = { [target] hidden in
            target.logitsForDFlashHidden(hidden)
        }
        self.boundTargetID = newID
    }

    public func unbind() {
        self.targetEmbed = nil
        self.targetLMHead = nil
        self.boundTargetID = nil
    }

    public func makeCache() throws -> [KVCache] {
        var caches = [KVCache]()
        caches.reserveCapacity(config.layerTypes.count)
        for layerType in config.layerTypes {
            switch layerType {
            case .fullAttention:
                caches.append(StandardKVCache())
            case .slidingAttention:
                guard let slidingWindow = config.slidingWindow else {
                    throw DFlashError.missingSlidingWindow
                }
                caches.append(RotatingKVCache(maxSize: slidingWindow - 1, keep: 0))
            }
        }
        return caches
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        targetHidden: MLXArray,
        cache: [KVCache],
        logitsStart: Int = 0
    ) throws -> MLXArray {
        guard let targetEmbed, let targetLMHead else {
            throw DFlashError.drafterNotBound
        }
        guard cache.count == layers.count else {
            throw DFlashError.invalidCacheCount(expected: layers.count, actual: cache.count)
        }
        guard logitsStart >= 0 else {
            throw DFlashError.invalidLogitsStart(logitsStart)
        }
        guard targetHidden.dim(-1) == config.targetHiddenSize else {
            throw DFlashError.targetHiddenSizeMismatch(
                expected: config.targetHiddenSize,
                actual: targetHidden.dim(-1)
            )
        }

        var h = targetEmbed(inputs)
        let context = hiddenNorm(contextProjection(targetHidden))

        for (i, layer) in layers.enumerated() {
            h = try layer(h, context: context, rope: rope, cache: cache[i])
        }

        if logitsStart > 0 {
            h = h[0..., logitsStart..., 0...]
        }

        var logits = targetLMHead(norm(h))
        if let cap = config.finalLogitSoftcapping {
            logits = tanh(logits / cap) * cap
        }
        return logits
    }

    public func draftBlock(
        bonus: Int,
        targetHidden: MLXArray,
        cache: [KVCache],
        blockSize: Int
    ) throws -> MLXArray {
        guard blockSize >= 2 else {
            throw DFlashError.invalidBlockSize(blockSize)
        }
        let maskValues = Array(repeating: Int32(config.maskTokenId), count: blockSize - 1)
        let block = MLXArray([Int32(bonus)] + maskValues)[.newAxis, .ellipsis]
        let logits = try self(
            block,
            targetHidden: targetHidden,
            cache: cache,
            logitsStart: 1
        )
        return logits.asType(.float32).argMax(axis: -1)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { key, _ in
            !key.hasPrefix("embed_tokens.") && !key.hasPrefix("lm_head.")
        }
    }

    public static func loadConfiguration(from directory: URL) throws -> DFlashConfiguration {
        let configURL = directory.appending(component: "config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw DFlashError.missingConfig(directory.path)
        }
        let configData = try Data(contentsOf: configURL)
        return try JSONDecoder.json5().decode(DFlashConfiguration.self, from: configData)
    }

    public static func load(from directory: URL) async throws -> DFlashDraftModel {
        try await load(from: directory, bindTo: nil)
    }

    public static func load(
        from directory: URL,
        bindTo target: (any DFlashTargetModel)?
    ) async throws -> DFlashDraftModel {
        let config = try loadConfiguration(from: directory)
        let drafter = DFlashDraftModel(config: config)
        if let target {
            try drafter.bind(target: target)
        }

        let weights = try loadWeights(from: directory)
        let params = ModuleParameters.unflattened(drafter.sanitize(weights: weights))
        try drafter.update(parameters: params, verify: [.all])
        eval(drafter)
        return drafter
    }

    public static func load(
        from downloader: any Downloader,
        id: String,
        revision: String? = nil,
        useLatest: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> DFlashDraftModel {
        let directory = try await downloader.download(
            id: id,
            revision: revision,
            matching: ["*.safetensors", "config.json"],
            useLatest: useLatest,
            progressHandler: progressHandler
        )
        return try await load(from: directory)
    }

    public static func load(
        from downloader: any Downloader,
        id: String,
        bindTo target: any DFlashTargetModel,
        revision: String? = nil,
        useLatest: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> DFlashDraftModel {
        let directory = try await downloader.download(
            id: id,
            revision: revision,
            matching: ["*.safetensors", "config.json"],
            useLatest: useLatest,
            progressHandler: progressHandler
        )
        return try await load(from: directory, bindTo: target)
    }

    private static func loadWeights(from directory: URL) throws -> [String: MLXArray] {
        let configURL = directory.appending(component: "config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw DFlashError.missingConfig(directory.path)
        }

        var weights = [String: MLXArray]()
        guard let enumerator = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: nil)
        else {
            throw DFlashError.unreadableDirectory(directory.path)
        }

        let urls = enumerator.allObjects
            .compactMap { $0 as? URL }
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.path < $1.path }
        guard !urls.isEmpty else {
            throw DFlashError.noSafetensorsFound(directory.path)
        }

        for url in urls where url.pathExtension == "safetensors" {
            let (shardWeights, _) = try loadArraysAndMetadata(url: url)
            for (key, value) in shardWeights {
                guard weights[key] == nil else {
                    throw DFlashError.duplicateWeightKey(key)
                }
                weights[key] = value
            }
        }

        return weights
    }

    private func validateCompatibility(with target: any DFlashTargetModel) throws {
        if config.hiddenSize != target.dFlashHiddenSize {
            throw DFlashError.incompatibleDrafter(
                field: "hiddenSize",
                drafter: "\(config.hiddenSize)",
                target: "\(target.dFlashHiddenSize)"
            )
        }
        if config.vocabularySize != target.dFlashVocabularySize {
            throw DFlashError.incompatibleDrafter(
                field: "vocabSize",
                drafter: "\(config.vocabularySize)",
                target: "\(target.dFlashVocabularySize)"
            )
        }
        if config.numTargetLayers != target.dFlashLayerCount {
            throw DFlashError.incompatibleDrafter(
                field: "numTargetLayers",
                drafter: "\(config.numTargetLayers)",
                target: "\(target.dFlashLayerCount)"
            )
        }
        try DFlashTargetValidation.validateTargetLayerIds(
            config.targetLayerIds, layerCount: target.dFlashLayerCount)
    }
}
