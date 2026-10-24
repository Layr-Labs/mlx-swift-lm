// Copyright © 2026 Eigen Labs.
import MLX
import MLXLMCommon
import MLXNN

public enum Qwen35MediaForwardError: Error {
    case invalidInputs, incompatibleCache
}

/// Explicit request-owned media forward seams. Ordinary calls do not enter
/// these methods and retain their existing scalar-offset RoPE/embedding path.
extension Qwen35TextModel {
    private func validateMediaInputs(_ tokens: MLXArray, embeddings: MLXArray?, positions: MLXArray)
        throws
    {
        guard tokens.ndim == 2, tokens.dim(0) > 0, tokens.dim(1) > 0, tokens.dtype == .int32,
            positions.shape == [3, tokens.dim(0), tokens.dim(1)], positions.dtype == .int32
        else {
            throw Qwen35MediaForwardError.invalidInputs
        }
        if let embeddings {
            guard embeddings.shape == [tokens.dim(0), tokens.dim(1), configuration.hiddenSize],
                [.float16, .bfloat16, .float32].contains(embeddings.dtype)
            else {
                throw Qwen35MediaForwardError.invalidInputs
            }
        }
    }

    /// Legacy cache ownership used by the native TP model: the original
    /// projection, recurrent update, mask, final norm and head with supplied
    /// text-space image features and explicit three-axis positions.
    public func forwardWithMedia(
        _ tokens: MLXArray, inputEmbeddings: MLXArray?,
        positionIds: MLXArray, cache: [KVCache]
    ) throws -> MLXArray {
        try validateMediaInputs(tokens, embeddings: inputEmbeddings, positions: positionIds)
        guard cache.count == configuration.hiddenLayers,
            cache.enumerated().allSatisfy({ index, cache in
                (index + 1) % configuration.fullAttentionInterval != 0
                    ? cache is MambaCache : !(cache is ArraysCache)
            })
        else { throw Qwen35MediaForwardError.incompatibleCache }
        let hidden = model(
            tokens, cache: cache, inputEmbeddings: inputEmbeddings, positionIds: positionIds)
        let normalized = model.norm(hidden)
        return lmHead.map { $0(normalized) } ?? model.embedTokens.asLinear(normalized)
    }

    /// The producer stage's true pre-norm residual, without building a head
    /// graph. CBv2 keeps recurrent rows and page transactions request-owned.
    public func cbv2MediaHidden(
        _ tokens: MLXArray, inputEmbeddings: MLXArray?,
        positionIds: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) throws -> MLXArray {
        try validateMediaInputs(tokens, embeddings: inputEmbeddings, positions: positionIds)
        guard recurrentState.count == tokens.dim(0),
            caches.count == configuration.cbv2LayerKinds.count
        else {
            throw Qwen35MediaForwardError.incompatibleCache
        }
        let attending = try caches.map { cache -> any CBv2AttendingLayerCache in
            guard let value = cache as? any CBv2AttendingLayerCache else {
                throw Qwen35MediaForwardError.incompatibleCache
            }
            return value
        }
        return Qwen35BatchDecodeProjection.withModel(tokens) {
            model.cbv2Forward(
                tokens, inputEmbeddings: inputEmbeddings,
                caches: attending, recurrentState: recurrentState, positionIds: positionIds)
        }
    }
}

extension Qwen35Model {
    public func forwardWithMedia(
        _ tokens: MLXArray, inputEmbeddings: MLXArray?,
        positionIds: MLXArray, cache: [KVCache]
    ) throws -> MLXArray {
        try languageModel.forwardWithMedia(
            tokens, inputEmbeddings: inputEmbeddings,
            positionIds: positionIds, cache: cache)
    }
    public func cbv2MediaHidden(
        _ tokens: MLXArray, inputEmbeddings: MLXArray?,
        positionIds: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) throws -> MLXArray {
        try languageModel.cbv2MediaHidden(
            tokens, inputEmbeddings: inputEmbeddings,
            positionIds: positionIds, caches: caches, recurrentState: recurrentState)
    }
}
