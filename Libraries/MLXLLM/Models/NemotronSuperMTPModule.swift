import MLX
import MLXLMCommon
import MLXNN

/// Unregistered Super VL prediction head. Takes POST-final-norm target hidden
/// and the next token's embedding, and returns post-norm draft hidden. Sharing
/// the target embedding/head and speculative acceptance remain host obligations.
/// Equation: NVIDIA TensorRT-LLM 26927494aeded524677d61bba0915372c78354ff,
/// modeling_nemotron_h.py NemotronHMTPDecoderLayer. No Lightning target cast.
public final class NemotronSuperMTPModule: Module {
    public enum Failure: Error {
        case configuration
        case inventory(String)
        case shape(String)
        case budget, cachePolicy, construction
    }
    public enum CachePolicy { case unquantized, fp8, paged }
    public struct KVCalibration {
        public let keyScale: Float
        public let valueScale: Float
    }
    @ModuleInfo(key: "layers") var layers: [Module]
    public let configuration: NemotronHConfiguration
    public let calibration: KVCalibration
    let weightDType: DType

    init(configuration: NemotronHConfiguration, calibration: KVCalibration, dtype: DType) throws {
        guard configuration.numNextnPredictLayers == 1,
            configuration.mtpLayersBlockType == ["attention", "moe"],
            configuration.hiddenSize > 0, configuration.hiddenSize % 16 == 0,
            let latent = configuration.moeLatentSize, latent > 0, latent % 16 == 0,
            configuration.moeIntermediateSize > 0, configuration.moeIntermediateSize % 16 == 0,
            configuration.moeSharedExpertIntermediateSize > 0,
            configuration.moeSharedExpertIntermediateSize % 16 == 0,
            configuration.numAttentionHeads > 0, configuration.numKeyValueHeads > 0,
            configuration.numAttentionHeads % configuration.numKeyValueHeads == 0,
            (configuration.headDim ?? (configuration.hiddenSize / configuration.numAttentionHeads))
                > 0,
            configuration.nGroup > 0, configuration.nRoutedExperts % configuration.nGroup == 0,
            configuration.topkGroup > 0, configuration.topkGroup <= configuration.nGroup,
            configuration.layerNormEpsilon.isFinite, configuration.layerNormEpsilon > 0,
            configuration.routedScalingFactor.isFinite,
            configuration.nSharedExperts == 1,
            !configuration.attentionBias, !configuration.mlpBias,
            configuration.nRoutedExperts > 0, configuration.nRoutedExperts <= 4096,
            configuration.numExpertsPerTok > 0,
            configuration.numExpertsPerTok <= configuration.nRoutedExperts
        else { throw Failure.configuration }
        self.configuration = configuration
        self.calibration = calibration
        self.weightDType = dtype
        // These two native blocks implement the exact same fusion/attention/
        // latent-MoE equations. None of the Lightning assistant is inherited.
        _layers.wrappedValue = [
            NemotronH35MTPAttentionBlock(configuration), NemotronH35MTPExpertBlock(configuration),
        ]
        super.init()
    }

    /// Bounded single-row correctness path. KV metadata is retained, never
    /// multiplied into weights or unquantized keys/values. This is an explicit
    /// unquantized numerical variant when selected. The FP8 path uses the bounded
    /// unit-scale cache codec, matching the pinned TRTLLM scale policy but not
    /// claiming fused-attention accumulation parity. This eager head is not a
    /// graph-only CBv2 drafter; no speculative serving is activated here.
    public func forward(
        postNormHidden: MLXArray, nextEmbedding: MLXArray,
        cache: KVCache, policy: CachePolicy, maximumCacheTokens: Int
    ) throws -> MLXArray {
        guard maximumCacheTokens > 0, maximumCacheTokens <= 16384 else { throw Failure.budget }
        switch policy {
        case .fp8:
            guard let fp8 = cache as? NemotronSuperFP8KVCache,
                fp8.heads == configuration.numKeyValueHeads,
                fp8.headDimension
                    == (configuration.headDim
                        ?? (configuration.hiddenSize / configuration.numAttentionHeads)),
                fp8.dtype == weightDType, fp8.capacityTokens <= maximumCacheTokens,
                fp8.offset < fp8.capacityTokens
            else { throw Failure.cachePolicy }
        case .paged: throw Failure.cachePolicy
        case .unquantized:
            guard type(of: cache) == KVCacheSimple.self,
                let simple = cache as? KVCacheSimple
            else { throw Failure.cachePolicy }
            // Inspect unsliced backing state first: KVCacheSimple.state itself slices
            // using offset and can fail before we can reject malformed restored KV.
            let state = simple.innerState()
            let headDim =
                configuration.headDim
                ?? (configuration.hiddenSize / configuration.numAttentionHeads)
            guard maximumCacheTokens > 0, maximumCacheTokens <= 16384,
                simple.offset >= 0, simple.offset < maximumCacheTokens,
                simple.step > 0, simple.step <= maximumCacheTokens,
                state.isEmpty ? simple.offset == 0 : state.count == 2
            else { throw Failure.shape("MTP cache") }
            guard
                state.allSatisfy({ value in
                    value.ndim == 4 && value.dim(0) == 1
                        && value.dim(1) == configuration.numKeyValueHeads
                        && value.dim(2) >= simple.offset && value.dim(2) <= maximumCacheTokens
                        && value.dim(3) == headDim && value.dtype == weightDType
                }), state.count != 2 || state[0].shape == state[1].shape
            else { throw Failure.shape("MTP cache") }
            let capacity = state.first?.dim(2) ?? 0
            if simple.offset + 1 > capacity {
                // KVCacheSimple appends a full step, retaining existing backing
                // capacity when offset is step-aligned, otherwise trimming to offset.
                let retained = simple.offset % simple.step == 0 ? capacity : simple.offset
                guard simple.step <= maximumCacheTokens - retained else { throw Failure.budget }
            }
        }
        guard postNormHidden.shape == [1, 1, configuration.hiddenSize],
            nextEmbedding.shape == postNormHidden.shape,
            postNormHidden.dtype == weightDType, nextEmbedding.dtype == weightDType
        else { throw Failure.shape("MTP input/cache") }
        let attention = layers[0] as! NemotronH35MTPAttentionBlock
        let experts = layers[1] as! NemotronH35MTPExpertBlock
        return experts(attention(hidden: postNormHidden, embedding: nextEmbedding, cache: cache))
    }
}
