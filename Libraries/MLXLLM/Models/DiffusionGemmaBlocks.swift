// Adapted from MLX-VLM e79b0e041677ec4ca5333ba750376bb4e8c434cb,
// mlx_vlm/models/diffusion_gemma/language.py.
// Copyright © 2025 Prince Canuma. MIT license: docs/diffusiongemma/LICENSE-MLX-VLM.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private let diffusionGemmaGEGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { gate, up in
        MLXNN.geluApproximate(gate) * up
    }
    return MLXHardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
}()

final class DiffusionGemmaRMSNormNoScale: Module {
    let epsilon: Float
    init(_ epsilon: Float) {
        self.epsilon = epsilon
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: .mlxNone, eps: epsilon)
    }
}

final class DiffusionGemmaDenseMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ config: DiffusionGemmaTextConfiguration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(diffusionGemmaGEGLU(gate(x), up(x))) }
}

final class DiffusionGemmaRouter: Module {
    @ModuleInfo var proj: Linear
    @ModuleInfo var scale: MLXArray
    @ModuleInfo(key: "per_expert_scale") var perExpertScale: MLXArray
    let epsilon: Float
    let rootSize: Float
    let topK: Int
    init(_ config: DiffusionGemmaTextConfiguration) {
        _proj.wrappedValue = Linear(config.hiddenSize, config.expertCount, bias: false)
        _scale.wrappedValue = .ones([config.hiddenSize])
        _perExpertScale.wrappedValue = .ones([config.expertCount])
        epsilon = config.rmsNormEpsilon
        rootSize = pow(Float(config.hiddenSize), -0.5)
        topK = config.topKExperts
    }
    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let normalized = MLXFast.rmsNorm(x, weight: .mlxNone, eps: epsilon)
        let scores = proj(normalized * scale * rootSize)
        let indices = argPartition(scores, kth: -topK, axis: -1)[.ellipsis, (-topK)...]
        let selected = takeAlong(scores, indices, axis: -1)
        let weights = softmax(selected, axis: -1, precise: true) * perExpertScale[indices]
        return (indices, weights)
    }
}

final class DiffusionGemmaExperts: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUp: SwitchLinear
    @ModuleInfo(key: "down_proj") var down: SwitchLinear
    let intermediateSize: Int
    init(_ config: DiffusionGemmaTextConfiguration) {
        intermediateSize = config.moeIntermediateSize
        _gateUp.wrappedValue = SwitchLinear(
            inputDims: config.hiddenSize,
            outputDims: 2 * config.moeIntermediateSize, numExperts: config.expertCount, bias: false)
        _down.wrappedValue = SwitchLinear(
            inputDims: config.moeIntermediateSize,
            outputDims: config.hiddenSize, numExperts: config.expertCount, bias: false)
    }
    func callAsFunction(_ inputs: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray {
        var x = expandedDimensions(inputs, axes: [-2, -3])
        let sorted = indices.size >= 64
        var expertIndices = indices
        var inverse: MLXArray?
        if sorted {
            let gathered = gatherSort(x: x, indices: indices)
            (x, expertIndices, inverse) = (gathered.0, gathered.1, gathered.2)
        }
        let projected = gateUp(x, expertIndices, sortedIndices: sorted)
        let gate = projected[.ellipsis, ..<intermediateSize]
        let up = projected[.ellipsis, intermediateSize...]
        let result = down(diffusionGemmaGEGLU(gate, up), expertIndices, sortedIndices: sorted)
        return DiffusionGemmaExpertReduction.reduce(
            result, inverse: inverse, indicesShape: indices.shape, weights: weights,
            enabled: !training && DiffusionGemmaExpertReduction.enabled)
    }
}

/// Shared projections. Encoder callers append committed tokens; denoiser callers
/// concatenate a read-only prefix with ephemeral canvas K/V. No model-owned cache.
final class DiffusionGemmaAttention: Module {
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear?
    @ModuleInfo(key: "o_proj") var output: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    @ModuleInfo(key: "v_norm") var valueNorm: DiffusionGemmaRMSNormNoScale
    @ModuleInfo var rope: RoPELayer
    let headDimension: Int
    let heads: Int
    let kvHeads: Int

    init(_ config: DiffusionGemmaTextConfiguration, layer: Int) {
        let kind = config.layerTypes[layer]
        let sliding = kind == "sliding_attention"
        headDimension = sliding ? config.headDimension : config.globalHeadDimension
        heads = config.attentionHeads
        kvHeads =
            sliding ? config.keyValueHeads : (config.globalKeyValueHeads ?? config.keyValueHeads)
        _query.wrappedValue = Linear(
            config.hiddenSize, heads * headDimension, bias: config.attentionBias)
        _key.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDimension, bias: config.attentionBias)
        _value.wrappedValue =
            sliding
            ? Linear(config.hiddenSize, kvHeads * headDimension, bias: config.attentionBias) : nil
        _output.wrappedValue = Linear(
            heads * headDimension, config.hiddenSize, bias: config.attentionBias)
        _queryNorm.wrappedValue = RMSNorm(dimensions: headDimension, eps: config.rmsNormEpsilon)
        _keyNorm.wrappedValue = RMSNorm(dimensions: headDimension, eps: config.rmsNormEpsilon)
        _valueNorm.wrappedValue = DiffusionGemmaRMSNormNoScale(config.rmsNormEpsilon)
        let parameters = config.ropeParameters[kind]!
        let fraction = parameters.partialRotaryFactor ?? 1
        _rope.wrappedValue = initializeRope(
            dims: sliding ? Int(Float(headDimension) * fraction) : headDimension,
            base: parameters.theta, traditional: false,
            scalingConfig: sliding
                ? nil
                : [
                    "type": .string("proportional"), "partial_rotary_factor": .float(fraction),
                ], maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, position: Int, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        keyValues: (MLXArray, MLXArray) -> (MLXArray, MLXArray)
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        var q = queryNorm(query(x).reshaped(batch, length, heads, headDimension)).transposed(
            0, 2, 1, 3)
        q = rope(q, offset: position)
        let rawKeys = key(x).reshaped(batch, length, kvHeads, headDimension)
        let rawValues =
            value.map { $0(x).reshaped(batch, length, kvHeads, headDimension) } ?? rawKeys
        let k = rope(keyNorm(rawKeys).transposed(0, 2, 1, 3), offset: position)
        let v = valueNorm(rawValues).transposed(0, 2, 1, 3)
        let (allKeys, allValues) = keyValues(k, v)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: allKeys, values: allValues, scale: 1, mask: mask)
        return output(attended.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

/// Shared-weight text block with distinct encoder and denoiser learned scalars.
/// There is no autoregressive PLE or cross-layer KV sharing.
public final class DiffusionGemmaTextBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: DiffusionGemmaAttention
    @ModuleInfo var mlp: DiffusionGemmaDenseMLP
    @ModuleInfo var router: DiffusionGemmaRouter
    @ModuleInfo var experts: DiffusionGemmaExperts
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFFN: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFFN: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm_1") var denseNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preExpertNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm_2") var expertNorm: RMSNorm
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    public init(_ config: DiffusionGemmaTextConfiguration, layer: Int) {
        precondition(config.layerTypes.indices.contains(layer))
        _attention.wrappedValue = DiffusionGemmaAttention(config, layer: layer)
        _mlp.wrappedValue = DiffusionGemmaDenseMLP(config)
        _router.wrappedValue = DiffusionGemmaRouter(config)
        _experts.wrappedValue = DiffusionGemmaExperts(config)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEpsilon)
        _postAttentionNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEpsilon)
        _preFFN.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEpsilon)
        _postFFN.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEpsilon)
        _denseNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEpsilon)
        _preExpertNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEpsilon)
        _expertNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEpsilon)
        _layerScalar.wrappedValue = .ones([1])
    }

    public func callAsFunction(
        _ x: MLXArray, position: Int,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        encoderScalar: MLXArray? = nil,
        keyValues: (MLXArray, MLXArray) -> (MLXArray, MLXArray)
    ) -> MLXArray {
        let attended = attention(inputNorm(x), position: position, mask: mask, keyValues: keyValues)
        let residual = x + postAttentionNorm(attended)
        let dense = denseNorm(mlp(preFFN(residual)))
        let flat = residual.reshaped(-1, residual.dim(-1))
        let routed = router(flat)
        let expert = experts(preExpertNorm(flat), indices: routed.indices, weights: routed.weights)
            .reshaped(residual.shape)
        let combined = postFFN(dense + expertNorm(expert))
        return (residual + combined) * (encoderScalar ?? layerScalar)
    }
}

public final class DiffusionGemmaSelfConditioning: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: RMSNorm
    @ModuleInfo(key: "post_norm") var postNorm: DiffusionGemmaRMSNormNoScale
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    public init(_ config: DiffusionGemmaTextConfiguration) {
        _preNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEpsilon)
        _postNorm.wrappedValue = DiffusionGemmaRMSNormNoScale(config.rmsNormEpsilon)
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }
    public func callAsFunction(_ embeddings: MLXArray, softEmbeddings: MLXArray) -> MLXArray {
        let normalized = preNorm(softEmbeddings)
        return postNorm(embeddings + down(diffusionGemmaGEGLU(gate(normalized), up(normalized))))
    }
}
