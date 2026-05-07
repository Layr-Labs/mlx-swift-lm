//
//  Gemma4MTP.swift
//  mlx-swift-lm
//
//  Gemma 4 Multi-Token Prediction (MTP) speculative decoding.
//  Single-file consolidation of drafter + round-loop + token iterator.
//
//  Port of https://github.com/Layr-Labs/mlx-swift-lm/pull/9

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom

// MARK: - Errors

/// Errors thrown by the Gemma 4 MTP drafter pipeline.
public enum Gemma4MTPError: LocalizedError, Sendable, Equatable {
    case unsupportedTarget(String)
    case rebindForbidden
    case incompatibleDrafter(field: String, drafter: String, target: String)
    case invalidBlockSize(Int)
    case drafterNotBound

    public var errorDescription: String? {
        switch self {
        case .unsupportedTarget(let typeName):
            return "Gemma4 MTP requires a Gemma4TextModel target; got \(typeName)."
        case .rebindForbidden:
            return
                "Gemma4AssistantDraftModel cannot be rebound to a different target. "
                + "Construct a new drafter instead."
        case .incompatibleDrafter(let field, let drafter, let target):
            return "Drafter/target mismatch on \(field): drafter=\(drafter), target=\(target)."
        case .invalidBlockSize(let n):
            return "Invalid blockSize \(n): must be between 2 and 16 inclusive."
        case .drafterNotBound:
            return
                "Gemma4AssistantDraftModel.bind(target:) must be called before any forward pass."
        }
    }
}

// MARK: - Shared KV types

/// Immutable snapshot of the target's K/V at the last non-shared full-attention
/// and sliding-attention layers. Every drafter layer reads from the appropriate slot.
public struct Gemma4SharedKV: @unchecked Sendable {
    /// K/V from the target's last non-shared full-attention layer.
    public let fullAttention: (MLXArray, MLXArray)
    /// K/V from the target's last non-shared sliding-attention layer.
    public let slidingAttention: (MLXArray, MLXArray)

    public init(
        fullAttention: (MLXArray, MLXArray),
        slidingAttention: (MLXArray, MLXArray)
    ) {
        self.fullAttention = fullAttention
        self.slidingAttention = slidingAttention
    }

    /// Trim `rejected` positions from the tail of each K/V tensor (axis=2).
    /// Ensures there is always at least one K/V position to attend to.
    public static func sliceTail(from shared: Gemma4SharedKV, rejected: Int) -> Gemma4SharedKV {
        func slice(_ kv: (MLXArray, MLXArray)) -> (MLXArray, MLXArray) {
            let T = kv.0.dim(2)
            let keep = max(1, T - rejected)
            return (kv.0[.ellipsis, ..<keep, 0...], kv.1[.ellipsis, ..<keep, 0...])
        }
        return Gemma4SharedKV(
            fullAttention: slice(shared.fullAttention),
            slidingAttention: slice(shared.slidingAttention))
    }

    /// Zero positions `[keepLengths[b], T)` for each row `b` in a batched KV pair.
    public static func zeroTailPerRow(
        from shared: Gemma4SharedKV, keepLengths: MLXArray
    ) -> Gemma4SharedKV {
        func zero(_ kv: (MLXArray, MLXArray)) -> (MLXArray, MLXArray) {
            let T = kv.0.dim(2)
            let positions = MLXArray(Int32(0) ..< Int32(T)).reshaped([1, 1, T, 1])
            let keep = keepLengths.asType(.int32).reshaped([-1, 1, 1, 1])
            let mask = (positions .< keep).asType(kv.0.dtype)
            return (kv.0 * mask, kv.1 * mask)
        }
        return Gemma4SharedKV(
            fullAttention: zero(shared.fullAttention),
            slidingAttention: zero(shared.slidingAttention))
    }
}

/// Mutable capture object. The `forwardForMTP` closure writes into it;
/// `snapshot()` freezes it into a `Gemma4SharedKV`.
final class Gemma4SharedKVCapture {
    var fullAttention: (MLXArray, MLXArray)? = nil
    var slidingAttention: (MLXArray, MLXArray)? = nil

    func snapshot() -> Gemma4SharedKV {
        guard let full = fullAttention else {
            fatalError("Gemma4SharedKVCapture: fullAttention was not populated")
        }
        guard let sliding = slidingAttention else {
            fatalError("Gemma4SharedKVCapture: slidingAttention was not populated")
        }
        return Gemma4SharedKV(fullAttention: full, slidingAttention: sliding)
    }
}

/// Output of `Gemma4TextModel.forwardForMTP`.
public struct Gemma4MTPForward: @unchecked Sendable {
    /// `[B, L, vocab]` — softcap applied.
    public let logits: MLXArray
    /// `[B, L, hiddenSize]` — pre-norm hidden, used as drafter input.
    public let lastHidden: MLXArray
    /// Per-layer-type K/V snapshot for the drafter.
    public let capturedSharedKV: Gemma4SharedKV

    public init(logits: MLXArray, lastHidden: MLXArray, capturedSharedKV: Gemma4SharedKV) {
        self.logits = logits
        self.lastHidden = lastHidden
        self.capturedSharedKV = capturedSharedKV
    }
}

/// Count of accepted speculative tokens per round.
public enum Gemma4AcceptCount: @unchecked Sendable {
    case scalar(Int)
    case perRow(MLXArray)

    func maxAccepted() -> Int {
        switch self {
        case .scalar(let v): return v
        case .perRow(let arr): return Int(arr.max().item(Int32.self))
        }
    }
}

// MARK: - Gemma4TextModel MTP extensions

extension Gemma4TextModel {

    /// Forward pass tailored for MTP speculative decoding.
    ///
    /// Returns logits, pre-norm trunk hidden, and a shared-KV snapshot for
    /// the drafter's next round. The capture hook fires on the last non-shared
    /// layer of each attention type.
    public func forwardForMTP(_ tokens: MLXArray, cache: [KVCache]) -> Gemma4MTPForward {
        let capture = Gemma4SharedKVCapture()
        let fullIdx = model.lastFullAttentionNonSharedIdx
        let slidingIdx = model.lastSlidingAttentionNonSharedIdx

        let (postNorm, preNorm) = model.callCapturingPreNorm(
            tokens, cache: cache
        ) { idx, kv in
            if idx == fullIdx { capture.fullAttention = kv }
            else if idx == slidingIdx { capture.slidingAttention = kv }
        }
        let logits = applyLMHead(postNorm)
        return Gemma4MTPForward(
            logits: logits,
            lastHidden: preNorm,
            capturedSharedKV: capture.snapshot())
    }

    /// Rewind the target KV caches after a speculative-decoding round.
    ///
    /// Uniformly trims by `blockSize - max(accepted) - 1`, then calls
    /// `BatchKVCache.zeroTailPerRow` for the `.perRow` case to handle
    /// per-row divergence.
    public func rollbackSpeculativeCache(
        _ caches: [KVCache],
        accepted: Gemma4AcceptCount,
        blockSize: Int
    ) {
        let maxAccepted = accepted.maxAccepted()
        let trim = Swift.max(0, blockSize - maxAccepted - 1)

        for cache in caches {
            guard cache.isTrimmable else { continue }
            if trim > 0 { _ = cache.trim(trim) }
        }

        if case .perRow(let perRowAccepted) = accepted, maxAccepted > 0 {
            for cache in caches {
                guard let batched = cache as? BatchKVCache else { continue }
                let postTrimLen = batched.offset
                let keepLengths =
                    perRowAccepted.asType(.int32)
                    + Int32(postTrimLen - maxAccepted)
                batched.zeroTailPerRow(keepLengths: keepLengths)
            }
        }
    }
}

// MARK: - DrafterMasks

/// Bidirectional attention-mask helpers for the Gemma 4 MTP drafter.
enum DrafterMasks {

    /// Full-attention mask: always `.none`. SDPA handles bidirectional
    /// attention without an explicit mask.
    static func bidirectionalFull(
        queryLen: Int, kvLen: Int, dtype: DType
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        _ = (queryLen, kvLen, dtype)
        return .none
    }

    /// Bidirectional sliding-window mask.
    ///
    /// Returns `.none` when the whole KV fits inside every query's window.
    /// Otherwise returns a materialized `.array(...)` additive mask with
    /// `-inf` outside the window and `0` inside.
    static func bidirectionalSWA(
        queryLen: Int, queryOffset: Int, kvLen: Int,
        window: Int, dtype: DType
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if kvLen <= window && queryOffset + queryLen <= kvLen + window {
            return .none
        }

        let qIdx = MLXArray(Int32(queryOffset) ..< Int32(queryOffset + queryLen))
            .reshaped([queryLen, 1])
        let kIdx = MLXArray(Int32(0) ..< Int32(kvLen)).reshaped([1, kvLen])
        let dist = qIdx - kIdx
        let inside = MLX.logicalAnd(dist .> Int32(-window), dist .< Int32(window))
        let zero = MLXArray(0.0).asType(dtype)
        let negInf = MLXArray(-Float.infinity).asType(dtype)
        let bias = MLX.where(inside, zero, negInf)
        return .array(bias.reshaped([1, 1, queryLen, kvLen]))
    }
}

// MARK: - MaskedEmbedder

/// Centroid-routed sparse LM head for Gemma 4 E2B / E4B drafters.
public final class MaskedEmbedder: Module, @unchecked Sendable {
    @ModuleInfo(key: "centroids") public var centroids: Linear
    @ParameterInfo(key: "token_ordering") public var tokenOrdering: MLXArray

    public let hiddenSize: Int
    public let numCentroids: Int
    public let topK: Int
    public let vocabSize: Int
    public let vocabSizePerCentroid: Int

    public init(hiddenSize: Int, numCentroids: Int, topK: Int, vocabSize: Int) {
        precondition(
            vocabSize % numCentroids == 0,
            "vocabSize must be divisible by numCentroids")
        self.hiddenSize = hiddenSize
        self.numCentroids = numCentroids
        self.topK = topK
        self.vocabSize = vocabSize
        self.vocabSizePerCentroid = vocabSize / numCentroids

        self._centroids.wrappedValue = Linear(hiddenSize, numCentroids, bias: false)
        self._tokenOrdering.wrappedValue = MLXArray.zeros([vocabSize], type: Int32.self)
        super.init()
    }

    /// Compute sparse logits over the full vocab.
    ///
    /// - Parameters:
    ///   - hiddenStates: `[B, L, hidden]`
    ///   - lmHeadWeight: `[vocabSize, hidden]`
    /// - Returns: `[B, L, vocabSize]` with non-selected positions set to `min(selected) - 1`.
    public func callAsFunction(hiddenStates: MLXArray, lmHeadWeight: MLXArray) -> MLXArray {
        let B = hiddenStates.dim(0)
        let L = hiddenStates.dim(1)

        let centroidLogits = centroids(hiddenStates)  // [B, L, numCentroids]
        let topKIndices = argPartition(-centroidLogits, kth: topK - 1, axis: -1)[
            .ellipsis, ..<topK]  // [B, L, topK]

        let ordering = tokenOrdering.reshaped([numCentroids, vocabSizePerCentroid])
        let selectedCanonical = ordering[topKIndices]  // [B, L, topK, vocabSizePerCentroid]

        let flatIdx = selectedCanonical.reshaped([-1])
        let selectedEmb = lmHeadWeight[flatIdx].reshaped([
            B, L, topK * vocabSizePerCentroid, hiddenSize,
        ])

        let selectedLogits = matmul(
            hiddenStates.expandedDimensions(axis: -2),
            selectedEmb.swappedAxes(-1, -2)
        ).squeezed(axis: -2)  // [B, L, topK * vsc]

        let sentinelValue = selectedLogits.min().item(Float.self) - 1.0
        let out = MLX.full(
            [B, L, vocabSize],
            values: MLXArray(sentinelValue),
            dtype: hiddenStates.dtype)

        let scatterIdx = selectedCanonical.reshaped([B, L, topK * vocabSizePerCentroid])
        return putAlong(out, scatterIdx, values: selectedLogits, axis: -1)
    }
}

// MARK: - SpeculativeWalk

/// Accept/reject walker for speculative decoding. Pure Swift — no MLX dependency.
public enum SpeculativeWalk {

    /// Single-row greedy accept-prefix walker.
    ///
    /// - Returns: `(acceptedCount, emittedTokens)` where `emittedTokens.count == acceptedCount + 1`.
    public static func single(draft: [Int], main: [Int]) -> (Int, [Int]) {
        guard !draft.isEmpty else {
            precondition(!main.isEmpty, "main must contain at least the bonus")
            return (0, [main[0]])
        }
        precondition(
            main.count >= draft.count + 1,
            "main must have at least draft.count + 1 tokens (drafts + bonus)"
        )
        var accepted = 0
        for i in 0 ..< draft.count {
            if main[i] != draft[i] { break }
            accepted += 1
        }
        return (accepted, Array(main[0 ... accepted]))
    }

    /// Multi-row greedy accept-prefix walker with per-row emit budgets.
    public static func batched(
        draft: [[Int]], main: [[Int]], budgets: [Int]
    ) -> ([Int], [[Int]]) {
        precondition(
            draft.count == main.count && draft.count == budgets.count,
            "batched: all inputs must have the same outer length B"
        )
        var acceptedOut: [Int] = []
        var emittedOut: [[Int]] = []
        acceptedOut.reserveCapacity(draft.count)
        emittedOut.reserveCapacity(draft.count)
        for i in 0 ..< draft.count {
            var (a, e) = single(draft: draft[i], main: main[i])
            let budget = Swift.max(0, budgets[i])
            if e.count > budget {
                e = Array(e.prefix(budget))
                a = Swift.max(0, e.count - 1)
            }
            acceptedOut.append(a)
            emittedOut.append(e)
        }
        return (acceptedOut, emittedOut)
    }
}

// MARK: - SpeculativeSample

/// Result of one round of rejection-based speculative sampling.
public struct SpeculativeSampleResult: Sendable {
    public let accepted: Int
    public let emitted: [Int]

    public init(accepted: Int, emitted: [Int]) {
        self.accepted = accepted
        self.emitted = emitted
    }
}

/// Apply temperature to logits, optionally masking to topP / topK / minP nucleus.
private func softmaxWithFilters(
    _ logits: MLXArray, temperature: Float, topP: Float, topK: Int, minP: Float
) -> MLXArray {
    var scaled = (temperature > 0 ? logits / temperature : logits).asType(.float32)
    if topP > 0 && topP < 1 { scaled = applyTopP(scaled, p: topP) }
    if minP > 0 { scaled = applyMinP(scaled, minP: minP) }
    if topK > 0 { scaled = applyTopK(scaled, k: topK) }
    return softmax(scaled, axis: -1)
}

/// Sample one token from `logits` with temperature + optional filters.
public func sampleLogitsWithFilters(
    _ logits: MLXArray, temperature: Float, topP: Float, topK: Int, minP: Float,
    rngKey: MLXArray
) -> MLXArray {
    let probs = softmaxWithFilters(logits, temperature: temperature, topP: topP, topK: topK, minP: minP)
    return sampleFromDistribution(probs, key: rngKey)
}

private func applyTopK(_ logits: MLXArray, k: Int) -> MLXArray {
    let vocab = logits.shape.last ?? 0
    if k <= 0 || k >= vocab { return logits }
    let sortedIdx = argSort(-logits, axis: -1)
    let topIdx = sortedIdx[.ellipsis, 0 ..< k]
    let topVals = takeAlong(logits, topIdx, axis: -1)
    let threshold = topVals.min(axes: [-1], keepDims: true)
    let negInf = MLXArray(-Float.infinity)
    return MLX.where(logits .>= threshold, logits, negInf)
}

private func applyTopP(_ logits: MLXArray, p: Float) -> MLXArray {
    let sortedIdxAsc = argSort(logits, axis: -1)
    let sortedLogits = takeAlong(logits, sortedIdxAsc, axis: -1)
    let sortedProbs = softmax(sortedLogits, axis: -1)
    let cumProbs = sortedProbs.cumsum(axis: -1)
    let keepMask = cumProbs .> MLXArray(1.0 - p)
    let negInf = MLXArray(-Float.infinity)
    let maskedSorted = MLX.where(keepMask, sortedLogits, negInf)
    let inverseIdx = argSort(sortedIdxAsc, axis: -1)
    return takeAlong(maskedSorted, inverseIdx, axis: -1)
}

private func applyMinP(_ logits: MLXArray, minP: Float) -> MLXArray {
    let maxLogit = logits.max(axis: -1, keepDims: true)
    let threshold = maxLogit + MLXArray(log(minP))
    let negInf = MLXArray(-Float.infinity)
    return MLX.where(logits .>= threshold, logits, negInf)
}

private func sampleFromDistribution(_ probs: MLXArray, key: MLXArray) -> MLXArray {
    let u = MLXRandom.uniform(low: Float(0), high: Float(1), key: key)
    let cdf = probs.cumsum(axis: -1)
    let lt = cdf .< u
    return lt.sum(axis: -1).asType(.int32)
}

/// Run rejection-based speculative sampling for one round (Leviathan et al. 2023).
public func speculativeSampleRound(
    draftLogits: MLXArray,
    draftTokens: [Int],
    verifyLogits: MLXArray,
    temperature: Float,
    topP: Float = 1.0,
    topK: Int = 0,
    minP: Float = 0.0,
    rngKey: MLXArray
) -> SpeculativeSampleResult {
    precondition(temperature > 0, "stochastic path requires temperature > 0")
    let K = draftTokens.count
    precondition(verifyLogits.dim(0) == K + 1, "verifyLogits must have K+1 positions")
    precondition(draftLogits.dim(0) == K, "draftLogits must have K positions")

    let q = softmaxWithFilters(
        draftLogits, temperature: temperature, topP: topP, topK: topK, minP: minP)
    let p = softmaxWithFilters(
        verifyLogits, temperature: temperature, topP: topP, topK: topK, minP: minP)

    let keys = MLXRandom.split(key: rngKey, into: 2 * (K + 1))

    var accepted = 0
    var emitted: [Int] = []
    for i in 0 ..< K {
        let c = draftTokens[i]
        let pi = p[i]
        let qi = q[i]
        let pc = pi[c].item(Float.self)
        let qc = qi[c].item(Float.self)
        let alpha = min(Float(1), qc > 0 ? pc / qc : 0)
        let r = MLXRandom.uniform(low: Float(0), high: Float(1), key: keys[i]).item(Float.self)
        if r < alpha {
            emitted.append(c)
            accepted += 1
        } else {
            let diff = MLX.maximum(pi - qi, MLXArray(Float(0)))
            let s = diff.sum().item(Float.self)
            let resampleDist: MLXArray = s > 0 ? diff / s : pi
            let nextTok = sampleFromDistribution(resampleDist, key: keys[K + i])
            emitted.append(Int(nextTok.item(Int32.self)))
            return SpeculativeSampleResult(accepted: accepted, emitted: emitted)
        }
    }
    let bonus = sampleFromDistribution(p[K], key: keys[2 * K])
    emitted.append(Int(bonus.item(Int32.self)))
    return SpeculativeSampleResult(accepted: accepted, emitted: emitted)
}

// MARK: - Gemma4AssistantConfiguration

/// Configuration for the Gemma 4 Multi-Token Prediction "assistant" drafter.
public struct Gemma4AssistantConfiguration: Codable, Sendable {
    public var modelType: String
    public var backboneHiddenSize: Int
    public var useOrderedEmbeddings: Bool
    public var numCentroids: Int
    public var centroidIntermediateTopK: Int
    public var blockSize: Int
    public var textConfig: Gemma4TextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case backboneHiddenSize = "backbone_hidden_size"
        case useOrderedEmbeddings = "use_ordered_embeddings"
        case numCentroids = "num_centroids"
        case centroidIntermediateTopK = "centroid_intermediate_top_k"
        case blockSize = "block_size"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_assistant"
        self.backboneHiddenSize = try container.decode(Int.self, forKey: .backboneHiddenSize)
        self.useOrderedEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .useOrderedEmbeddings) ?? false
        self.numCentroids =
            try container.decodeIfPresent(Int.self, forKey: .numCentroids) ?? 2048
        self.centroidIntermediateTopK =
            try container.decodeIfPresent(Int.self, forKey: .centroidIntermediateTopK) ?? 32
        self.blockSize =
            try container.decodeIfPresent(Int.self, forKey: .blockSize) ?? 4

        var textConfig = try container.decode(Gemma4TextConfiguration.self, forKey: .textConfig)
        // HF post-init: clamp numKvSharedLayers so every drafter layer is KV-shared.
        if textConfig.numKvSharedLayers == 0
            || textConfig.numKvSharedLayers > textConfig.numHiddenLayers
        {
            textConfig.numKvSharedLayers = textConfig.numHiddenLayers
        }
        self.textConfig = textConfig
    }
}

// MARK: - Gemma4AssistantDraftModel

/// Gemma 4 MTP "assistant" drafter model.
///
/// 4-layer trunk reusing `Gemma4TextModelInner` with every layer `forceSharedKV: true`.
/// Requires `bind(target:)` before any forward pass.
public final class Gemma4AssistantDraftModel: Module, @unchecked Sendable {
    public let config: Gemma4AssistantConfiguration

    @ModuleInfo public var model: Gemma4TextModelInner
    @ModuleInfo(key: "pre_projection") public var preProjection: Linear
    @ModuleInfo(key: "post_projection") public var postProjection: Linear
    @ModuleInfo(key: "lm_head") public var lmHead: Linear?
    @ModuleInfo(key: "masked_embedding") public var maskedEmbedder: MaskedEmbedder?

    internal var targetEmbed: ((MLXArray) -> MLXArray)?
    internal var boundTargetID: ObjectIdentifier?

    public init(config: Gemma4AssistantConfiguration) {
        self.config = config
        let textCfg = config.textConfig

        self._model.wrappedValue = Gemma4TextModelInner(textCfg, forceSharedKV: true)
        self._preProjection.wrappedValue = Linear(
            2 * config.backboneHiddenSize, textCfg.hiddenSize, bias: false)
        self._postProjection.wrappedValue = Linear(
            textCfg.hiddenSize, config.backboneHiddenSize, bias: false)

        if !textCfg.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(textCfg.hiddenSize, textCfg.vocabSize, bias: false)
        }
        if config.useOrderedEmbeddings {
            self._maskedEmbedder.wrappedValue = MaskedEmbedder(
                hiddenSize: textCfg.hiddenSize,
                numCentroids: config.numCentroids,
                topK: config.centroidIntermediateTopK,
                vocabSize: textCfg.vocabSize)
        }
        super.init()
    }

    /// Bind the drafter to a Gemma 4 target. Idempotent on the same target.
    public func bind(target: Gemma4TextModel) throws {
        let newID = ObjectIdentifier(target)
        if let existing = boundTargetID {
            if existing == newID { return }
            throw Gemma4MTPError.rebindForbidden
        }
        try validateCompatibility(with: target)
        self.targetEmbed = { [target] tokens in target.embedTokensForDrafter(tokens) }
        self.boundTargetID = newID
    }

    /// Clear the binding.
    public func unbind() {
        self.targetEmbed = nil
        self.boundTargetID = nil
    }

    // MARK: - Forward

    /// Drafter forward pass.
    ///
    /// - Parameters:
    ///   - inputsEmbeds: `[B, 1, 2 * backboneHiddenSize]`
    ///   - sharedKV: K/V snapshot from the target's last non-shared layers.
    ///   - positionOffset: absolute position of the bonus token.
    /// - Returns: `(lastHidden: [B, 1, backboneHiddenSize], logits: [B, 1, vocabSize])`.
    public func callAsFunction(
        inputsEmbeds: MLXArray,
        sharedKV: Gemma4SharedKV,
        positionOffset: Gemma4.PositionOffset
    ) -> (lastHidden: MLXArray, logits: MLXArray) {
        let textCfg = config.textConfig

        var h = preProjection(inputsEmbeds)

        let queryLen = h.dim(1)
        let queryOffset: Int
        switch positionOffset {
        case .scalar(let v): queryOffset = v
        case .batch(let arr): queryOffset = Int(arr[0].item(Int32.self))
        }

        let fullKVLen = sharedKV.fullAttention.0.dim(2)
        let slidingKVLen = sharedKV.slidingAttention.0.dim(2)
        let dtype = h.dtype

        let fullMask = DrafterMasks.bidirectionalFull(
            queryLen: queryLen, kvLen: fullKVLen, dtype: dtype)
        let slidingMask = DrafterMasks.bidirectionalSWA(
            queryLen: queryLen, queryOffset: queryOffset,
            kvLen: slidingKVLen, window: textCfg.slidingWindow, dtype: dtype)

        for (i, layer) in model.layers.enumerated() {
            let layerType = textCfg.layerTypes[i]
            let kv: (MLXArray, MLXArray)
            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            switch layerType {
            case "full_attention":
                kv = sharedKV.fullAttention
                mask = fullMask
            case "sliding_attention":
                kv = sharedKV.slidingAttention
                mask = slidingMask
            default:
                preconditionFailure(
                    "Gemma4AssistantDraftModel: unexpected layerType '\(layerType)' at layer \(i)")
            }
            let (out, _, _) = layer(
                h,
                mask: mask,
                cache: nil,
                perLayerInput: nil,
                sharedKV: kv,
                positionOffset: positionOffset
            )
            h = out
        }

        h = model.norm(h)
        let lastHidden = postProjection(h)
        let logits = applyLMHead(h)
        return (lastHidden, logits)
    }

    private func applyLMHead(_ hidden: MLXArray) -> MLXArray {
        if let maskedEmbedder {
            return maskedEmbedder(hiddenStates: hidden, lmHeadWeight: model.embedTokens.weight)
        }
        if let lmHead { return lmHead(hidden) }
        return model.embedTokens.asLinear(hidden)
    }

    // MARK: - Weight sanitization

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)

        let forbiddenSubstrings = [
            ".self_attn.k_proj", ".self_attn.v_proj",
            ".self_attn.k_norm", ".self_attn.v_norm",
        ]

        for (key, value) in weights {
            for substring in forbiddenSubstrings {
                if key.contains(substring) {
                    throw Gemma4MTPError.incompatibleDrafter(
                        field: key,
                        drafter: "(unexpected — drafter layers are kv-shared)",
                        target: "(should not be present)")
                }
            }
            if key == "lm_head.weight" && config.textConfig.tieWordEmbeddings { continue }
            if key == "masked_embedding.token_ordering" {
                out[key] = value.asType(.int32)
                continue
            }
            out[key] = value
        }
        return out
    }

    // MARK: - Compatibility validation

    private func validateCompatibility(with target: Gemma4TextModel) throws {
        let drafterT = config.textConfig
        let targetCfg = target.configuration

        guard config.backboneHiddenSize == targetCfg.hiddenSize else {
            throw Gemma4MTPError.incompatibleDrafter(
                field: "backboneHiddenSize",
                drafter: "\(config.backboneHiddenSize)",
                target: "\(targetCfg.hiddenSize)")
        }
        guard drafterT.vocabSize == targetCfg.vocabSize else {
            throw Gemma4MTPError.incompatibleDrafter(
                field: "vocabSize",
                drafter: "\(drafterT.vocabSize)",
                target: "\(targetCfg.vocabSize)")
        }
        for (i, lt) in drafterT.layerTypes.enumerated() {
            guard lt == "full_attention" || lt == "sliding_attention" else {
                throw Gemma4MTPError.incompatibleDrafter(
                    field: "layerTypes[\(i)]", drafter: lt,
                    target: "full_attention|sliding_attention")
            }
        }
        let drafterHasFullAttn = drafterT.layerTypes.contains("full_attention")
        if drafterHasFullAttn && drafterT.attentionKeqV {
            guard targetCfg.attentionKeqV else {
                throw Gemma4MTPError.incompatibleDrafter(
                    field: "attentionKeqV", drafter: "true", target: "false")
            }
            guard drafterT.numGlobalKeyValueHeads == targetCfg.numGlobalKeyValueHeads else {
                throw Gemma4MTPError.incompatibleDrafter(
                    field: "numGlobalKeyValueHeads",
                    drafter: "\(drafterT.numGlobalKeyValueHeads.map(String.init) ?? "nil")",
                    target: "\(targetCfg.numGlobalKeyValueHeads.map(String.init) ?? "nil")")
            }
        }
        guard drafterT.numKvSharedLayers == drafterT.numHiddenLayers else {
            throw Gemma4MTPError.incompatibleDrafter(
                field: "numKvSharedLayers",
                drafter: "\(drafterT.numKvSharedLayers)",
                target: "\(drafterT.numHiddenLayers)")
        }
    }

    // MARK: - Loading

    /// Load a drafter from a local directory containing `config.json` and `*.safetensors` files.
    public static func load(from directory: URL) async throws -> Gemma4AssistantDraftModel {
        let configURL = directory.appending(component: "config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: configData)

        let drafter = Gemma4AssistantDraftModel(config: config)

        var weights = [String: MLXArray]()
        guard let enumerator = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: nil)
        else {
            throw NSError(
                domain: "Gemma4AssistantDraftModel", code: 1,
                userInfo: [NSLocalizedDescriptionKey:
                    "Could not enumerate drafter directory: \(directory.path)"])
        }
        let urls = enumerator.allObjects.compactMap { $0 as? URL }
        for url in urls where url.pathExtension == "safetensors" {
            let (shardWeights, _) = try loadArraysAndMetadata(url: url)
            for (k, v) in shardWeights { weights[k] = v }
        }

        let sanitized = try drafter.sanitize(weights: weights)
        let params = ModuleParameters.unflattened(sanitized)
        try drafter.update(parameters: params, verify: [.all])
        eval(drafter)
        return drafter
    }

    /// Load a drafter from a remote model ID via a `Downloader`.
    public static func load(
        from downloader: any Downloader,
        id: String,
        revision: String? = nil,
        useLatest: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> Gemma4AssistantDraftModel {
        let directory = try await downloader.download(
            id: id,
            revision: revision,
            matching: ["*.safetensors", "config.json"],
            useLatest: useLatest,
            progressHandler: progressHandler)
        return try await load(from: directory)
    }
}

// MARK: - Round loop (B=1)

/// Run the Gemma 4 MTP round loop for a single request (B=1).
///
/// The caller must have already run a prefill forward (via `forwardForMTP`) and
/// provided the resulting `firstBonus`, `firstHidden`, and `firstSharedKV`.
///
/// Tokens are yielded as `.chunk("<int>")` strings; wrap with a tokenizer to
/// decode to text.
public func runGemma4MTPRounds(
    target: Gemma4TextModel,
    drafter: Gemma4AssistantDraftModel,
    targetCache: [KVCache],
    firstBonus: Int,
    firstHidden: MLXArray,
    firstSharedKV: Gemma4SharedKV,
    maxTokens: Int,
    blockSize: Int
) throws -> AsyncStream<Generation> {
    guard blockSize >= 2 && blockSize <= 16 else {
        throw Gemma4MTPError.invalidBlockSize(blockSize)
    }
    try drafter.bind(target: target)

    return AsyncStream<Generation> { continuation in
        continuation.yield(.chunk("\(firstBonus)"))
        var emitted = 1

        var bonus = firstBonus
        var hidden = firstHidden
        var sharedKV = firstSharedKV

        while emitted < maxTokens {
            let remaining = maxTokens - emitted
            let bs = min(blockSize, remaining + 1)
            if bs <= 1 { break }
            let k = bs - 1

            let driveOffset = targetCache[0].offset
            var tok = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]
            var h = hidden
            var draftPerStep: [MLXArray] = []
            draftPerStep.reserveCapacity(k)
            for _ in 0 ..< k {
                let tokEmbed = target.embedTokensForDrafter(tok)
                let inputsEmbeds = concatenated([tokEmbed, h], axis: -1)
                let (newH, logits) = drafter(
                    inputsEmbeds: inputsEmbeds,
                    sharedKV: sharedKV,
                    positionOffset: .scalar(driveOffset))
                let sampled = logits.squeezed(axis: 1).argMax(axis: -1)
                let sampled2d = sampled[.newAxis, .ellipsis]
                draftPerStep.append(sampled2d)
                tok = sampled2d
                h = newH
            }

            let bonusCol = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]
            let verifyInput: MLXArray =
                draftPerStep.isEmpty
                ? bonusCol
                : concatenated([bonusCol] + draftPerStep, axis: 1)
            let verifyOut = target.forwardForMTP(verifyInput, cache: targetCache)
            let mainTokens = verifyOut.logits.asType(.float32).argMax(axis: -1)
            let draftConcat: MLXArray =
                draftPerStep.isEmpty
                ? MLXArray.zeros([1, 0], dtype: .int32)
                : concatenated(draftPerStep, axis: 1)
            eval(mainTokens, draftConcat)
            let mainInts = mainTokens.squeezed(axis: 0).asArray(Int.self)
            let draftTokens = draftConcat.squeezed(axis: 0).asArray(Int32.self).map { Int($0) }

            let (accepted, newTokens) = SpeculativeWalk.single(draft: draftTokens, main: mainInts)

            var stopEarly = false
            for t in newTokens {
                continuation.yield(.chunk("\(t)"))
                emitted += 1
                if emitted >= maxTokens { stopEarly = true; break }
            }
            if stopEarly { continuation.finish(); return }

            if accepted < k {
                target.rollbackSpeculativeCache(
                    targetCache, accepted: .scalar(accepted), blockSize: bs)
            }

            hidden = verifyOut.lastHidden[0..., accepted ..< accepted + 1, 0...]
            bonus = newTokens.last!
            let rejected = k - accepted
            sharedKV = Gemma4SharedKV.sliceTail(
                from: verifyOut.capturedSharedKV, rejected: rejected)

            if emitted % 256 == 0 { MLX.Memory.clearCache() }
        }

        continuation.finish()
    }
}

// MARK: - Batched (B>1) output type

/// Per-step output of the B>1 MTP round loop.
public struct BatchedGeneration: Sendable {
    public let slots: [Slot]

    public struct Slot: Sendable {
        public let row: Int
        public let token: Int?
        public let finishReason: FinishReason?

        public init(row: Int, token: Int?, finishReason: FinishReason?) {
            self.row = row
            self.token = token
            self.finishReason = finishReason
        }
    }

    public enum FinishReason: Sendable {
        case stop
        case eos
        case length
    }

    public init(slots: [Slot]) { self.slots = slots }
}

// MARK: - Round loop (B>1)

/// Run the Gemma 4 MTP round loop for a batch of requests (B>1).
///
/// Mirrors `runGemma4MTPRounds` but tracks per-row acceptance counts and
/// per-row finishedness. Supports continuous batching: rows that finish are
/// compacted out so subsequent rounds only pay compute for active rows.
public func runGemma4MTPRoundsBatched(
    target: Gemma4TextModel,
    drafter: Gemma4AssistantDraftModel,
    targetCache: [KVCache],
    firstBonus: [Int],
    firstHidden: MLXArray,
    firstSharedKV: Gemma4SharedKV,
    maxTokens: Int,
    blockSize: Int,
    eosTokenIds: Set<Int>?,
    maxTokensPerRow: [Int]? = nil
) throws -> AsyncStream<BatchedGeneration> {
    guard blockSize >= 2 && blockSize <= 16 else {
        throw Gemma4MTPError.invalidBlockSize(blockSize)
    }
    try drafter.bind(target: target)
    let originalB = firstBonus.count
    let perRowBudget: [Int] = maxTokensPerRow ?? Array(repeating: maxTokens, count: originalB)
    precondition(perRowBudget.count == originalB, "maxTokensPerRow.count must equal firstBonus.count")

    return AsyncStream<BatchedGeneration> { continuation in
        var firstSlots: [BatchedGeneration.Slot] = []
        firstSlots.reserveCapacity(originalB)
        var emitted = Array(repeating: 1, count: originalB)
        var finished = Array(repeating: false, count: originalB)
        for (i, b) in firstBonus.enumerated() {
            if let eosTokenIds, eosTokenIds.contains(b) {
                finished[i] = true
                firstSlots.append(.init(row: i, token: b, finishReason: .eos))
            } else if emitted[i] >= perRowBudget[i] {
                finished[i] = true
                firstSlots.append(.init(row: i, token: b, finishReason: .length))
            } else {
                firstSlots.append(.init(row: i, token: b, finishReason: nil))
            }
        }
        continuation.yield(BatchedGeneration(slots: firstSlots))
        if finished.allSatisfy({ $0 }) { continuation.finish(); return }

        var activeIndices: [Int] = (0 ..< originalB).filter { !finished[$0] }
        var bonus: [Int] = activeIndices.map { firstBonus[$0] }
        var hidden: MLXArray = {
            if activeIndices.count == originalB { return firstHidden }
            let idx = MLXArray(activeIndices.map { Int32($0) }, [activeIndices.count])
            return MLX.take(firstHidden, idx, axis: 0)
        }()
        var sharedKV: Gemma4SharedKV = {
            if activeIndices.count == originalB { return firstSharedKV }
            let idx = MLXArray(activeIndices.map { Int32($0) }, [activeIndices.count])
            return Gemma4SharedKV(
                fullAttention: (
                    MLX.take(firstSharedKV.fullAttention.0, idx, axis: 0),
                    MLX.take(firstSharedKV.fullAttention.1, idx, axis: 0)),
                slidingAttention: (
                    MLX.take(firstSharedKV.slidingAttention.0, idx, axis: 0),
                    MLX.take(firstSharedKV.slidingAttention.1, idx, axis: 0)))
        }()
        if activeIndices.count < originalB {
            let idx = MLXArray(activeIndices.map { Int32($0) }, [activeIndices.count])
            for cache in targetCache {
                if let bc = cache as? BatchedCache { bc.filterBatched(batchIndices: idx) }
            }
        }

        while !finished.allSatisfy({ $0 }) {
            let B = activeIndices.count
            let activeRemaining = activeIndices.map { perRowBudget[$0] - emitted[$0] }
            guard let minRemaining = activeRemaining.min(), minRemaining > 0 else { break }
            let bs = min(blockSize, minRemaining + 1)
            if bs <= 1 { break }
            let k = bs - 1

            let driveOffset = targetCache[0].offset
            var tok = MLXArray(bonus.map { Int32($0) }, [B, 1])
            var h = hidden
            var draftPerStep: [MLXArray] = []
            draftPerStep.reserveCapacity(k)
            for _ in 0 ..< k {
                let tokEmbed = target.embedTokensForDrafter(tok)
                let inputsEmbeds = concatenated([tokEmbed, h], axis: -1)
                let (newH, logits) = drafter(
                    inputsEmbeds: inputsEmbeds,
                    sharedKV: sharedKV,
                    positionOffset: .scalar(driveOffset))
                let sampled = logits.squeezed(axis: 1).argMax(axis: -1)
                let sampled2d = sampled.reshaped([B, 1])
                draftPerStep.append(sampled2d)
                tok = sampled2d
                h = newH
            }

            let bonusCol = MLXArray(bonus.map { Int32($0) }, [B, 1])
            let verifyInput: MLXArray =
                draftPerStep.isEmpty
                ? bonusCol
                : concatenated([bonusCol] + draftPerStep, axis: 1)
            let verifyOut = target.forwardForMTP(verifyInput, cache: targetCache)
            let mainTokens = verifyOut.logits.asType(.float32).argMax(axis: -1)
            let draftConcat: MLXArray =
                draftPerStep.isEmpty
                ? MLXArray.zeros([B, 0], dtype: .int32)
                : concatenated(draftPerStep, axis: 1)
            eval(mainTokens, draftConcat)
            let mainFlat = mainTokens.asArray(Int32.self)
            let draftFlat = draftConcat.asArray(Int32.self)
            var mainPerRow: [[Int]] = []
            var draftTokensPerRow: [[Int]] = []
            mainPerRow.reserveCapacity(B)
            draftTokensPerRow.reserveCapacity(B)
            for bi in 0 ..< B {
                let mStart = bi * bs
                mainPerRow.append(mainFlat[mStart ..< mStart + bs].map { Int($0) })
                let dStart = bi * k
                draftTokensPerRow.append(draftFlat[dStart ..< dStart + k].map { Int($0) })
            }

            let budgets = activeIndices.map { perRowBudget[$0] - emitted[$0] }
            let (accepted, newTokensPerRow) = SpeculativeWalk.batched(
                draft: draftTokensPerRow, main: mainPerRow, budgets: budgets)
            let maxAcceptedInt = accepted.max() ?? 0

            var finishedThisRound = Array(repeating: false, count: B)
            let maxNew = newTokensPerRow.map(\.count).max() ?? 0
            for pos in 0 ..< maxNew {
                var slots: [BatchedGeneration.Slot] = []
                slots.reserveCapacity(B)
                for bi in 0 ..< B {
                    let origRow = activeIndices[bi]
                    if finishedThisRound[bi] {
                        slots.append(.init(row: origRow, token: nil, finishReason: nil))
                        continue
                    }
                    let row = newTokensPerRow[bi]
                    if pos < row.count {
                        let t = row[pos]
                        emitted[origRow] += 1
                        var finishReason: BatchedGeneration.FinishReason? = nil
                        if let eosTokenIds, eosTokenIds.contains(t) {
                            finished[origRow] = true
                            finishedThisRound[bi] = true
                            finishReason = .eos
                        } else if emitted[origRow] >= perRowBudget[origRow] {
                            finished[origRow] = true
                            finishedThisRound[bi] = true
                            finishReason = .length
                        }
                        slots.append(.init(row: origRow, token: t, finishReason: finishReason))
                    } else {
                        slots.append(.init(row: origRow, token: nil, finishReason: nil))
                    }
                }
                continuation.yield(BatchedGeneration(slots: slots))
            }

            if finished.allSatisfy({ $0 }) { continuation.finish(); return }

            let uniformAccepted = accepted.allSatisfy { $0 == accepted[0] }
            if maxAcceptedInt < k {
                if uniformAccepted {
                    target.rollbackSpeculativeCache(
                        targetCache, accepted: .scalar(accepted[0]), blockSize: bs)
                } else {
                    let acceptedArr = MLXArray(accepted.map { Int32($0) })
                    target.rollbackSpeculativeCache(
                        targetCache, accepted: .perRow(acceptedArr), blockSize: bs)
                }
            }

            let hiddenDim = verifyOut.lastHidden.dim(2)
            let acceptedIdx = MLX.broadcast(
                MLXArray(accepted.map { Int32($0) }, [B, 1, 1]),
                to: [B, 1, hiddenDim])
            hidden = MLX.takeAlong(verifyOut.lastHidden, acceptedIdx, axis: 1)

            for bi in 0 ..< B {
                if let last = newTokensPerRow[bi].last { bonus[bi] = last }
            }

            if uniformAccepted {
                let rejected = k - accepted[0]
                sharedKV = Gemma4SharedKV.sliceTail(
                    from: verifyOut.capturedSharedKV, rejected: rejected)
            } else {
                let capturedT = verifyOut.capturedSharedKV.fullAttention.0.dim(2)
                let rejectedPerRow = accepted.map { Int32(k - $0) }
                let keepLengths = MLXArray(rejectedPerRow.map { Int32(capturedT) - $0 })
                sharedKV = Gemma4SharedKV.zeroTailPerRow(
                    from: verifyOut.capturedSharedKV, keepLengths: keepLengths)
            }

            if finishedThisRound.contains(true) {
                let keepLocal: [Int] = finishedThisRound.enumerated()
                    .compactMap { $0.element ? nil : $0.offset }
                if keepLocal.isEmpty { break }
                activeIndices = keepLocal.map { activeIndices[$0] }
                bonus = keepLocal.map { bonus[$0] }
                let keepIdx = MLXArray(keepLocal.map { Int32($0) }, [keepLocal.count])
                hidden = MLX.take(hidden, keepIdx, axis: 0)
                sharedKV = Gemma4SharedKV(
                    fullAttention: (
                        MLX.take(sharedKV.fullAttention.0, keepIdx, axis: 0),
                        MLX.take(sharedKV.fullAttention.1, keepIdx, axis: 0)),
                    slidingAttention: (
                        MLX.take(sharedKV.slidingAttention.0, keepIdx, axis: 0),
                        MLX.take(sharedKV.slidingAttention.1, keepIdx, axis: 0)))
                for cache in targetCache {
                    if let bc = cache as? BatchedCache { bc.filterBatched(batchIndices: keepIdx) }
                }
            }

            if (emitted.max() ?? 0) % 256 == 0 { MLX.Memory.clearCache() }
        }

        continuation.finish()
    }
}

// MARK: - Gemma4MTPTokenIterator

/// Single-batch (B=1) MTP token iterator. Conforms to ``TokenIteratorProtocol``
/// so callers can use the standard ``generateTask`` entry point.
///
/// Supports both greedy (`temperature=0`) and stochastic (`temperature>0`)
/// speculative decoding.
public struct Gemma4MTPTokenIterator: TokenIteratorProtocol {

    private let target: Gemma4TextModel
    private let drafter: Gemma4AssistantDraftModel
    private var cache: [KVCache]
    private let blockSize: Int
    private let temperature: Float
    private let topP: Float
    private let topK: Int
    private let minP: Float
    private var rngKey: MLXArray

    private var bonus: Int
    private var hidden: MLXArray
    private var sharedKV: Gemma4SharedKV

    private var pendingTokens: [Int] = []
    private var pendingIndex: Int = 0

    public var tokenCount: Int = 0
    public let maxTokens: Int?
    public var promptPrefillTime: TimeInterval = 0.0

    /// Initialize and run the prompt prefill. The first token emitted is the
    /// bonus sampled from the last prefill position.
    public init(
        input: LMInput,
        target: Gemma4TextModel,
        drafter: Gemma4AssistantDraftModel,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters,
        blockSize: Int = 4,
        rngSeed: UInt64 = 0
    ) throws {
        guard blockSize >= 2 && blockSize <= 16 else {
            throw Gemma4MTPError.invalidBlockSize(blockSize)
        }
        try drafter.bind(target: target)

        self.target = target
        self.drafter = drafter
        self.cache = cache ?? target.newCache(parameters: parameters)
        self.blockSize = blockSize
        self.maxTokens = parameters.maxTokens
        self.temperature = parameters.temperature
        self.topP = parameters.topP
        self.topK = parameters.topK
        self.minP = parameters.minP
        self.rngKey = rngSeed == 0
            ? MLXRandom.key(UInt64(Date().timeIntervalSince1970 * 1e6))
            : MLXRandom.key(rngSeed)

        let prefillStart = Date()
        var promptTokens = input.text.tokens
        if promptTokens.ndim == 1 { promptTokens = promptTokens[.newAxis, .ellipsis] }

        let prefillOut = target.forwardForMTP(promptTokens, cache: self.cache)
        let lastLogits = prefillOut.logits[0..., -1, 0...]
        let firstBonus: Int
        if parameters.temperature == 0 {
            let arr = lastLogits.asType(.float32).argMax(axis: -1)
            eval(arr)
            firstBonus = Int(arr.item(Int32.self))
        } else {
            let keys = MLXRandom.split(key: self.rngKey, into: 2)
            let idx = sampleLogitsWithFilters(
                lastLogits, temperature: parameters.temperature,
                topP: parameters.topP, topK: parameters.topK, minP: parameters.minP,
                rngKey: keys[0])
            eval(idx)
            firstBonus = Int(idx.item(Int32.self))
            self.rngKey = keys[1]
        }
        self.bonus = firstBonus
        self.hidden = prefillOut.lastHidden[
            0..., -1 ..< prefillOut.lastHidden.dim(1), 0...]
        self.sharedKV = prefillOut.capturedSharedKV
        self.promptPrefillTime = -prefillStart.timeIntervalSinceNow

        self.pendingTokens.append(firstBonus)
    }

    public mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens { return nil }

        if pendingIndex < pendingTokens.count {
            let t = pendingTokens[pendingIndex]
            pendingIndex += 1
            tokenCount += 1
            return t
        }

        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        let remaining = (maxTokens.map { $0 - tokenCount }) ?? blockSize
        guard remaining > 0 else { return nil }
        let bs = Swift.min(blockSize, remaining + 1)
        if bs <= 1 { return nil }
        let k = bs - 1

        let driveOffset = cache[0].offset
        var tok = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]
        var h = hidden
        var draftPerStep: [MLXArray] = []
        var draftLogitsPerStep: [MLXArray] = []
        draftPerStep.reserveCapacity(k)
        if temperature > 0 { draftLogitsPerStep.reserveCapacity(k) }

        for _ in 0 ..< k {
            let tokEmbed = target.embedTokensForDrafter(tok)
            let inputsEmbeds = concatenated([tokEmbed, h], axis: -1)
            let (newH, logits) = drafter(
                inputsEmbeds: inputsEmbeds,
                sharedKV: sharedKV,
                positionOffset: .scalar(driveOffset))
            let stepLogits = logits.squeezed(axis: 1)
            let sampled: MLXArray
            if temperature > 0 {
                draftLogitsPerStep.append(stepLogits)
                let keys = MLXRandom.split(key: rngKey, into: 2)
                rngKey = keys[0]
                sampled = sampleLogitsWithFilters(
                    stepLogits, temperature: temperature,
                    topP: topP, topK: topK, minP: minP, rngKey: keys[1])
            } else {
                sampled = stepLogits.argMax(axis: -1)
            }
            let sampled2d = sampled[.newAxis, .ellipsis]
            draftPerStep.append(sampled2d)
            tok = sampled2d
            h = newH
        }

        let bonusCol = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]
        let verifyInput: MLXArray =
            draftPerStep.isEmpty
            ? bonusCol
            : concatenated([bonusCol] + draftPerStep, axis: 1)
        let verifyOut = target.forwardForMTP(verifyInput, cache: cache)
        let mainTokens = verifyOut.logits.asType(.float32).argMax(axis: -1)
        let draftConcat: MLXArray =
            draftPerStep.isEmpty
            ? MLXArray.zeros([1, 0], dtype: .int32)
            : concatenated(draftPerStep, axis: 1)
        eval(mainTokens, draftConcat)
        let mainInts = mainTokens.squeezed(axis: 0).asArray(Int.self)
        let draftTokens = draftConcat.squeezed(axis: 0).asArray(Int32.self).map { Int($0) }

        let accepted: Int
        let newTokens: [Int]
        if temperature > 0 && k > 0 {
            let verifyLogits = verifyOut.logits.squeezed(axis: 0)
            let draftLogits = concatenated(draftLogitsPerStep, axis: 0)
            let keys = MLXRandom.split(key: rngKey, into: 2)
            rngKey = keys[0]
            let result = speculativeSampleRound(
                draftLogits: draftLogits,
                draftTokens: draftTokens,
                verifyLogits: verifyLogits,
                temperature: temperature,
                topP: topP, topK: topK, minP: minP,
                rngKey: keys[1])
            accepted = result.accepted
            newTokens = result.emitted
        } else {
            var a = 0
            while a < k && draftTokens[a] == mainInts[a] { a += 1 }
            accepted = a
            newTokens = Array(mainInts[0 ..< a + 1])
        }

        if accepted < k {
            target.rollbackSpeculativeCache(
                cache, accepted: .scalar(accepted), blockSize: bs)
        }

        hidden = verifyOut.lastHidden[0..., accepted ..< accepted + 1, 0...]
        bonus = newTokens.last!
        let rejected = k - accepted
        sharedKV = Gemma4SharedKV.sliceTail(
            from: verifyOut.capturedSharedKV, rejected: rejected)

        pendingTokens.append(contentsOf: newTokens)
        if pendingTokens.isEmpty { return nil }
        let t = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return t
    }
}

// MARK: - generateGemma4MTP

/// Generate text from a Gemma 4 target using a Gemma 4 MTP drafter.
///
/// Wraps `Gemma4MTPTokenIterator` and a tokenizer-decoding loop, producing
/// an `AsyncStream<Generation>` that yields decoded text chunks and a
/// terminal `.info(...)` with counts.
///
/// - Parameters:
///   - input: prepared language-model input.
///   - parameters: generation parameters. `temperature > 0` enables stochastic
///     rejection-based speculative sampling.
///   - target: `ModelContext` whose model is `Gemma4TextModel` or `Gemma4Model`.
///   - drafter: the loaded Gemma 4 MTP drafter.
///   - blockSize: speculative block size (2–16). Default 4.
///   - rngSeed: seed for stochastic sampling. Default 0 → seeds from system clock.
public func generateGemma4MTP(
    input: LMInput,
    parameters: GenerateParameters,
    target: ModelContext,
    drafter: Gemma4AssistantDraftModel,
    blockSize: Int = 4,
    rngSeed: UInt64 = 0
) throws -> AsyncStream<Generation> {
    let gemma4: Gemma4TextModel
    if let t = target.model as? Gemma4TextModel {
        gemma4 = t
    } else if let wrapper = target.model as? Gemma4Model {
        gemma4 = wrapper.textModel
    } else {
        throw Gemma4MTPError.unsupportedTarget(String(describing: type(of: target.model)))
    }

    let tokenizer = target.tokenizer
    let eosIds = target.configuration.eosTokenIds
    var params = parameters
    if params.maxTokens == nil { params.maxTokens = 1024 }
    let promptTokenCount = input.text.tokens.size

    var iter = try Gemma4MTPTokenIterator(
        input: input,
        target: gemma4,
        drafter: drafter,
        parameters: params,
        blockSize: blockSize,
        rngSeed: rngSeed)
    let prefillElapsed = iter.promptPrefillTime

    let boxed = IteratorBox(iter: iter)

    return AsyncStream<Generation> { continuation in
        let task = Task {
            let generateStart = Date()
            var tokenCount = 0
            var stopReason: GenerateStopReason = .length
            while let tok = boxed.next() {
                tokenCount += 1
                continuation.yield(.chunk(tokenizer.decode(tokenIds: [tok])))
                if eosIds.contains(tok) { stopReason = .stop; break }
                if Task.isCancelled { stopReason = .cancelled; break }
            }
            let elapsed = Date().timeIntervalSince(generateStart)
            let info = GenerateCompletionInfo(
                promptTokenCount: promptTokenCount,
                generationTokenCount: tokenCount,
                promptTime: prefillElapsed,
                generationTime: elapsed,
                stopReason: stopReason)
            continuation.yield(.info(info))
            continuation.finish()
        }
        continuation.onTermination = { _ in task.cancel() }
    }
}

private final class IteratorBox: @unchecked Sendable {
    private var iter: Gemma4MTPTokenIterator
    init(iter: Gemma4MTPTokenIterator) { self.iter = iter }
    func next() -> Int? { iter.next() }
}
