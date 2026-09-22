// Adapted from MLX-VLM e79b0e041677ec4ca5333ba750376bb4e8c434cb.
// Copyright © 2025 Prince Canuma. MIT: docs/diffusiongemma/LICENSE-MLX-VLM.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public final class DiffusionGemmaEncoderScalar: Module {
    @ModuleInfo(key: "layer_scalar") public var value: MLXArray
    public override init() {
        _value.wrappedValue = .ones([1])
        super.init()
    }
}

/// Only the encoder-specific learned scalars are registered here. The decoder
/// owns all shared text weights exactly once, avoiding duplicate quantization,
/// parameter enumeration and memory admission charges.
public final class DiffusionGemmaEncoderTextParameters: Module {
    @ModuleInfo public var layers: [DiffusionGemmaEncoderScalar]
    public init(layerCount: Int) {
        _layers.wrappedValue = (0 ..< layerCount).map { _ in DiffusionGemmaEncoderScalar() }
    }
}

public final class DiffusionGemmaTextDecoder: Module {
    public let configuration: DiffusionGemmaTextConfiguration
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo public var layers: [DiffusionGemmaTextBlock]
    @ModuleInfo public var norm: RMSNorm
    @ModuleInfo(key: "self_conditioning") public var selfConditioning:
        DiffusionGemmaSelfConditioning
    private let embeddingScale: Float
    private let cacheOwner = UUID()

    public init(_ configuration: DiffusionGemmaTextConfiguration) {
        self.configuration = configuration
        embeddingScale = sqrt(Float(configuration.hiddenSize))
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize,
            dimensions: configuration.hiddenSize)
        _layers.wrappedValue = (0 ..< configuration.layerCount).map {
            DiffusionGemmaTextBlock(configuration, layer: $0)
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEpsilon)
        _selfConditioning.wrappedValue = DiffusionGemmaSelfConditioning(configuration)
    }

    /// The pinned MLX quant reference retains FP32 conditioning logits, while
    /// the original dense checkpoint rounds them to its embedding dtype.
    /// Record this artifact-specific contract; a different policy needs new
    /// numerical/quality qualification, not a hidden speed toggle.
    public var selfConditioningLogitsDType: DType {
        embedTokens is QuantizedEmbedding ? .float32 : embedTokens.weight.dtype
    }

    func validateCache(_ cache: DiffusionGemmaRequestCache) throws {
        try cache.validate(
            for: configuration, owner: cacheOwner, appendCount: 0, kvDType: nativeKVDType)
    }

    private var nativeKVDType: DType {
        (embedTokens as? QuantizedEmbedding)?.scales.dtype ?? embedTokens.weight.dtype
    }

    public func scaledEmbeddings(_ tokenIds: MLXArray) throws -> MLXArray {
        try validateTokenIds(tokenIds)
        return embedTokens(tokenIds) * embeddingScale
    }

    private func validateTokenIds(_ ids: MLXArray) throws {
        guard ids.ndim == 2, ids.dim(0) == 1, ids.dim(1) > 0,
            ids.dtype == .int32 || ids.dtype == .uint32
        else { throw DiffusionGemmaModelError.invalidInput("token shape/type") }
        guard logicalAnd(ids .>= 0, ids .< configuration.vocabularySize).all().item(Bool.self)
        else {
            throw DiffusionGemmaModelError.invalidInput("token range")
        }
    }

    /// Encode prompt or finalized-canvas tokens into request-owned KV. The
    /// returned hidden state is optional work; callers needing only the cache
    /// evaluate stateArrays() rather than materializing unused final-layer FFNs.
    public func encode(
        tokenIds: MLXArray, cache: DiffusionGemmaRequestCache,
        encoderParameters: DiffusionGemmaEncoderTextParameters,
        preparedEmbeddings: MLXArray? = nil,
        visualBlockIds: MLXArray? = nil
    ) throws -> MLXArray {
        try cache.performStep {
            try encodeStep(
                tokenIds: tokenIds, cache: cache, encoderParameters: encoderParameters,
                preparedEmbeddings: preparedEmbeddings, visualBlockIds: visualBlockIds)
        }
    }

    private func encodeStep(
        tokenIds: MLXArray, cache: DiffusionGemmaRequestCache,
        encoderParameters: DiffusionGemmaEncoderTextParameters,
        preparedEmbeddings: MLXArray?, visualBlockIds: MLXArray?
    ) throws -> MLXArray {
        try validateTokenIds(tokenIds)
        let length = tokenIds.dim(1)
        try cache.validate(
            for: configuration, owner: cacheOwner, appendCount: length, kvDType: nativeKVDType)
        guard encoderParameters.layers.count == layers.count else {
            throw DiffusionGemmaModelError.invalidInput("encoder scalars")
        }
        if let preparedEmbeddings,
            preparedEmbeddings.shape != [1, length, configuration.hiddenSize]
        {
            throw DiffusionGemmaModelError.invalidInput("prepared embeddings")
        }
        if let visualBlockIds, visualBlockIds.shape != [1, length] {
            throw DiffusionGemmaModelError.invalidInput("visual block shape")
        }
        var hidden = preparedEmbeddings ?? (embedTokens(tokenIds) * embeddingScale)
        let position = cache.position
        let nextWindowOrder = cache.windowOrder.appending(
            length, priorPosition: position, window: configuration.slidingWindow)
        for (index, block) in layers.enumerated() {
            let row = cache.rows[index]
            let sliding = configuration.layerTypes[index] == "sliding_attention"
            let history =
                sliding
                ? min(row.retainedCount, configuration.slidingWindow - 1) : row.retainedCount
            let mask = encoderMask(
                length: length, history: history,
                window: sliding ? configuration.slidingWindow : nil,
                visualBlockIds: visualBlockIds)
            hidden = block(
                hidden, position: position, mask: mask,
                encoderScalar: encoderParameters.layers[index].value
            ) { keys, values in
                let updated = row.update(keys: keys, values: values)
                guard sliding, length == 1, position + 1 >= configuration.slidingWindow,
                    nextWindowOrder.cursor < configuration.slidingWindow
                else { return updated }
                // Match the reference's singleton reduction order. Snapshots
                // and denoising remain canonical oldest-to-newest views.
                let split = configuration.slidingWindow - nextWindowOrder.cursor
                func ringOrder(_ value: MLXArray) -> MLXArray {
                    concatenated(
                        [value[.ellipsis, split..., 0...], value[.ellipsis, ..<split, 0...]],
                        axis: 2)
                }
                return (ringOrder(updated.0), ringOrder(updated.1))
            }
        }
        cache.recordCommittedTokens(tokenIds)
        let result = norm(hidden)
        try cache.finishPagedStep([result])
        return result
    }

    private func encoderMask(
        length: Int, history: Int, window: Int?, visualBlockIds: MLXArray?
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if length == 1 { return .none }
        if history == 0, visualBlockIds == nil, window == nil || length <= window! {
            return .causal
        }
        let keyPositions = MLXArray(0 ..< (history + length))
        let queries = MLXArray(history ..< (history + length)).expandedDimensions(axis: -1)
        var mask = queries .>= keyPositions
        if let window { mask = logicalAnd(mask, queries .< keyPositions + window) }
        mask = mask[.newAxis, .newAxis, 0..., 0...]
        if let visualBlockIds, configuration.bidirectionalAttention == "vision" {
            let q = visualBlockIds[0..., 0..., .newAxis]
            let k = visualBlockIds[0..., .newAxis, 0...]
            var overlay = logicalAnd(q .>= 0, q .== k)[0..., .newAxis, 0..., 0...]
            if history > 0 {
                overlay = concatenated(
                    [.zeros([1, 1, length, history], dtype: .bool), overlay], axis: -1)
            }
            mask = logicalOr(mask, overlay)
        }
        return .array(mask)
    }

    public func denoise(
        canvasIds: MLXArray, cache: DiffusionGemmaRequestCache,
        selfConditioningLogits: MLXArray? = nil
    ) throws -> MLXArray {
        try cache.performStep {
            try denoiseStep(
                canvasIds: canvasIds, cache: cache, selfConditioningLogits: selfConditioningLogits)
        }
    }

    private func denoiseStep(
        canvasIds: MLXArray, cache: DiffusionGemmaRequestCache,
        selfConditioningLogits: MLXArray?
    ) throws -> MLXArray {
        try validateTokenIds(canvasIds)
        try cache.validate(
            for: configuration, owner: cacheOwner, appendCount: 0, kvDType: nativeKVDType)
        let embeddings = embedTokens(canvasIds) * embeddingScale
        let soft: MLXArray
        if let logits = selfConditioningLogits {
            guard logits.shape == canvasIds.shape + [configuration.vocabularySize] else {
                throw DiffusionGemmaModelError.invalidInput("conditioning shape")
            }
            let probabilities = softmax(logits.asType(.float32), axis: -1, precise: true).asType(
                embeddings.dtype)
            if let packed = embedTokens as? QuantizedEmbedding {
                soft =
                    DiffusionGemmaSoftEmbedding.project(
                        probabilities, weight: packed.weight, scales: packed.scales,
                        biases: packed.biases, groupSize: packed.groupSize, bits: packed.bits,
                        mode: packed.mode, inference: !training,
                        enabled: DiffusionGemmaSoftEmbedding.enabled
                    )
                    .asType(embeddings.dtype) * embeddingScale
            } else {
                soft =
                    matmul(probabilities, embedTokens.weight).asType(embeddings.dtype)
                    * embeddingScale
            }
        } else {
            soft = .zeros(like: embeddings)
        }
        var hidden = selfConditioning(embeddings, softEmbeddings: soft)
        for (index, block) in layers.enumerated() {
            let row = cache.rows[index]
            var prefix: (MLXArray, MLXArray)?
            if row.retainedCount > 0 {
                let snapshot = row.snapshot()
                let count =
                    configuration.layerTypes[index] == "sliding_attention"
                    ? min(row.retainedCount, configuration.slidingWindow - 1) : row.retainedCount
                let start = snapshot.keys.dim(2) - count
                prefix = (
                    snapshot.keys[.ellipsis, start..., 0...],
                    snapshot.values[.ellipsis, start..., 0...]
                )
            }
            hidden = block(hidden, position: cache.position, mask: .none) { keys, values in
                guard let prefix else { return (keys, values) }
                return (
                    concatenated([prefix.0, keys], axis: 2),
                    concatenated([prefix.1, values], axis: 2)
                )
            }
        }
        let raw = embedTokens.asLinear(norm(hidden)).asType(.float32)
        let result = tanh(raw / configuration.finalLogitSoftcap) * configuration.finalLogitSoftcap
        try cache.finishPagedStep([result])
        return result
    }
}

public enum DiffusionGemmaModelError: Error, Sendable, Equatable {
    case invalidInput(String)
}
