// DeepSeek-V4 model port.
//
// Based on SharpAI/mlx-swift-lm (MIT) with corrections and MTP hooks.
// Reference:
//   ml-explore/mlx-lm#1192 (base model)
//   Blaizzy/mlx-lm#15 (MTP)
//   omlx/patches/mlx_lm_mtp/deepseek_v4_model.py
//   https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct DeepseekV4Configuration: Codable, Sendable {
    var vocabSize: Int = 129280
    var hiddenSize: Int = 4096
    var moeIntermediateSize: Int = 2048
    var numHiddenLayers: Int = 43
    var numAttentionHeads: Int = 64
    var headDim: Int = 512
    var qLoraRank: Int = 1024
    var qkRopeHeadDim: Int = 64
    var rmsNormEps: Float = 1e-6
    var oGroups: Int = 8
    var oLoraRank: Int = 1024
    var slidingWindow: Int = 128
    var compressRatios: [Int] = []
    var compressRopeTheta: Float = 160000.0
    var nRoutedExperts: Int = 256
    var nSharedExperts: Int = 1
    var numExpertsPerTok: Int = 6
    var scoringFunc: String = "sqrtsoftplus"
    var routedScalingFactor: Float = 1.5
    var swiguLimit: Float = 10.0
    var numHashLayers: Int = 3
    var numNextnPredictLayers: Int = 1
    var normTopkProb: Bool = true
    var hcMult: Int = 4
    var hcSinkhornIters: Int = 20
    var hcEps: Float = 1e-6
    var ropeTheta: Float = 10000.0
    var ropeScaling: [String: StringOrNumber]?
    var maxPositionEmbeddings: Int = 1_048_576

    var nopeHeadDim: Int { headDim - qkRopeHeadDim }

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case oGroups = "o_groups"
        case oLoraRank = "o_lora_rank"
        case slidingWindow = "sliding_window"
        case compressRatios = "compress_ratios"
        case compressRopeTheta = "compress_rope_theta"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case scoringFunc = "scoring_func"
        case routedScalingFactor = "routed_scaling_factor"
        case swiguLimit = "swiglu_limit"
        case numHashLayers = "num_hash_layers"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case normTopkProb = "norm_topk_prob"
        case hcMult = "hc_mult"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case hcEps = "hc_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
    }
}

// MARK: - Helper Functions

private func sqrtSoftplus(_ x: MLXArray) -> MLXArray {
    let sp = MLX.maximum(x, MLXArray(0)) + MLX.log1p(MLX.exp(-MLX.abs(x)))
    return MLX.sqrt(sp)
}

private func headRmsNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    x * rsqrt(x.square().mean(axis: -1, keepDims: true) + eps)
}

// MARK: - Hyper-Connection helpers
//
// HyperConnection replaces the standard residual stream with a learned hc_mult-way
// parallel stream. Each block does:
//   1. hcPre:  [B,S,hc,D] → [B,S,D] using Sinkhorn-normalized weights
//   2. sublayer (attn or ffn) on the 3-D reduced tensor
//   3. hcPost: expand output back to [B,S,hc,D] and blend with the residual
// Reference: ml-explore/mlx-lm#1192 hyper_connection.py

private func hcSplitSinkhorn(
    _ mixes: MLXArray,
    hcScale: MLXArray,
    hcBase: MLXArray,
    hcMult: Int,
    sinkhornIters: Int,
    eps: Float
) -> (MLXArray, MLXArray, MLXArray) {
    let hc = hcMult
    let B = mixes.dim(0), S = mixes.dim(1)

    let preMix = mixes[.ellipsis, ..<hc]
    let postMix = mixes[.ellipsis, hc ..< 2 * hc]
    let combMix = mixes[.ellipsis, (2 * hc)...]

    let preBase = hcBase[..<hc]
    let postBase = hcBase[hc ..< 2 * hc]
    let combBase = hcBase[(2 * hc)...]

    var pre = sigmoid(preMix * hcScale[0] + preBase) + eps
    let post = sigmoid(postMix * hcScale[1] + postBase) + eps
    var comb = (sigmoid(combMix * hcScale[2] + combBase) + eps)
        .reshaped(B, S, hc, hc)

    pre = pre / pre.sum(axis: -1, keepDims: true)

    for _ in 0 ..< sinkhornIters {
        comb = comb / comb.sum(axis: -2, keepDims: true)
        comb = comb / comb.sum(axis: -1, keepDims: true)
    }

    return (pre, post, comb)
}

func hcPre(
    x: MLXArray,
    hcFn: MLXArray,
    hcScale: MLXArray,
    hcBase: MLXArray,
    hcMult: Int,
    sinkhornIters: Int,
    eps: Float
) -> (MLXArray, MLXArray, MLXArray) {
    let dtype = x.dtype
    let B = x.dim(0), S = x.dim(1), hc = x.dim(2), D = x.dim(3)

    let xFlat = x.reshaped(B, S, hc * D).asType(.float32)
    let normScale = rsqrt(xFlat.square().mean(axis: -1, keepDims: true) + eps)
    let mixes = matmul(xFlat, hcFn.T) * normScale

    let (pre, post, comb) = hcSplitSinkhorn(
        mixes, hcScale: hcScale, hcBase: hcBase,
        hcMult: hcMult, sinkhornIters: sinkhornIters, eps: eps)

    let y = (pre.expandedDimensions(axis: -1).asType(dtype) * x).sum(axis: -2)
    return (y, post, comb)
}

func hcPost(
    x: MLXArray,
    residual: MLXArray,
    post: MLXArray,
    comb: MLXArray
) -> MLXArray {
    let term1 = post.expandedDimensions(axis: -1) * x.expandedDimensions(axis: -2)
    let combExp = comb.expandedDimensions(axis: -1)
    let residualExp = residual.expandedDimensions(axis: -2)
    let term2 = (combExp * residualExp).sum(axis: 2)
    return (term1 + term2).asType(x.dtype)
}

func hcHeadReduce(
    x: MLXArray,
    hcFn: MLXArray,
    hcScale: MLXArray,
    hcBase: MLXArray,
    eps: Float
) -> MLXArray {
    let dtype = x.dtype
    let B = x.dim(0), S = x.dim(1), hc = x.dim(2), D = x.dim(3)

    let xFlat = x.reshaped(B, S, hc * D).asType(.float32)
    let normScale = rsqrt(xFlat.square().mean(axis: -1, keepDims: true) + eps)
    let mixes = matmul(xFlat, hcFn.T) * normScale
    let pre = sigmoid(mixes * hcScale + hcBase) + eps

    return (pre.expandedDimensions(axis: -1).asType(dtype) * x).sum(axis: -2).asType(dtype)
}

// MARK: - HCParams

/// Holds the three learnable tensors for one HyperConnection block.
/// Property names (fn, base, scale) match the checkpoint key suffixes.
class HCParams: Module {
    var fn: MLXArray
    var base: MLXArray
    var scale: MLXArray

    init(fn: MLXArray, base: MLXArray, scale: MLXArray) {
        self.fn = fn
        self.base = base
        self.scale = scale
    }
}

// MARK: - Attention

private func deepseekAttentionWithSinks(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?
) -> MLXArray {
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask, sinks: sinks)
    }
    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        precondition(sinks == nil, "Quantized SDPA does not support attention sinks.")
        let (qk, qv) = quantizedKVCache.updateQuantized(keys: keys, values: values)
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: qk, quantizedValues: qv,
            scale: scale, mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode)
    }
    let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
    return MLXFast.scaledDotProductAttention(
        queries: queries, keys: cachedKeys, values: cachedValues,
        scale: scale, mask: mask, sinks: sinks)
}

class DeepseekV4Attention: Module {
    let numHeads: Int
    let headDim: Int
    let nopeHeadDim: Int
    let ropeHeadDim: Int
    let oGroups: Int
    let oLoraRank: Int
    let nHeadsPerGroup: Int
    let scale: Float
    let eps: Float

    let rope: RoPELayer

    @ModuleInfo(key: "wq_a") var wqA: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm
    @ModuleInfo(key: "wo_a") var woA: Linear
    @ModuleInfo(key: "wo_b") var woB: Linear

    var attn_sink: MLXArray

    init(config: DeepseekV4Configuration) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.nopeHeadDim = config.nopeHeadDim
        self.ropeHeadDim = config.qkRopeHeadDim
        self.oGroups = config.oGroups
        self.oLoraRank = config.oLoraRank
        self.nHeadsPerGroup = config.numAttentionHeads / config.oGroups
        self.scale = pow(Float(config.headDim), -0.5)
        self.eps = config.rmsNormEps

        self._wqA.wrappedValue = Linear(config.hiddenSize, config.qLoraRank, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: config.rmsNormEps)
        self._wqB.wrappedValue = Linear(
            config.qLoraRank, config.numAttentionHeads * config.headDim, bias: false)

        self._wkv.wrappedValue = Linear(config.hiddenSize, config.headDim, bias: false)
        self._kvNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)

        self._woA.wrappedValue = Linear(
            nHeadsPerGroup * config.headDim, config.oGroups * config.oLoraRank, bias: false)
        self._woB.wrappedValue = Linear(
            config.oGroups * config.oLoraRank, config.hiddenSize, bias: false)

        self.attn_sink = zeros([config.numAttentionHeads])

        self.rope = initializeRope(
            dims: config.qkRopeHeadDim,
            base: config.compressRopeTheta,
            traditional: true,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    private func groupedOutputProjection(_ out: MLXArray) -> MLXArray {
        let B = out.dim(0), L = out.dim(1)
        let groupFeat = numHeads * headDim / oGroups
        let outFlat = out.reshaped(B, L, numHeads * headDim)

        if let qLinear = woA as? QuantizedLinear {
            var pieces: [MLXArray] = []
            for g in 0 ..< oGroups {
                let gStart = g * groupFeat, gEnd = (g + 1) * groupFeat
                let rStart = g * oLoraRank, rEnd = (g + 1) * oLoraRank
                let groupInput = outFlat[0..., 0..., gStart ..< gEnd]
                let wRows = qLinear.weight[rStart ..< rEnd]
                let sRows = qLinear.scales[rStart ..< rEnd]
                let bRows = qLinear.biases.map { $0[rStart ..< rEnd] }
                pieces.append(quantizedMM(
                    groupInput, wRows, scales: sRows, biases: bRows,
                    transpose: true, groupSize: qLinear.groupSize,
                    bits: qLinear.bits, mode: qLinear.mode))
            }
            return concatenated(pieces, axis: -1)
        } else {
            var pieces: [MLXArray] = []
            for g in 0 ..< oGroups {
                let gStart = g * groupFeat, gEnd = (g + 1) * groupFeat
                let rStart = g * oLoraRank, rEnd = (g + 1) * oLoraRank
                let groupInput = outFlat[0..., 0..., gStart ..< gEnd]
                let wg = woA.weight[rStart ..< rEnd]
                pieces.append(matmul(groupInput, wg.T))
            }
            return concatenated(pieces, axis: -1)
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = wqB(qNorm(wqA(x)))
        q = q.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        q = headRmsNorm(q, eps: eps)

        let qNope = q[.ellipsis, ..<nopeHeadDim]
        var qRope = q[.ellipsis, nopeHeadDim...]
        qRope = applyRotaryPosition(rope, to: qRope, cache: cache)
        let queries = concatenated([qNope, qRope], axis: -1)

        let kv = kvNorm(wkv(x))
        let kvNope = kv[.ellipsis, ..<nopeHeadDim].reshaped(B, L, 1, nopeHeadDim).transposed(0, 2, 1, 3)
        var kvRope = kv[.ellipsis, nopeHeadDim...].reshaped(B, L, 1, ropeHeadDim).transposed(0, 2, 1, 3)
        kvRope = applyRotaryPosition(rope, to: kvRope, cache: cache)
        let kFull = concatenated([kvNope, kvRope], axis: -1)

        let sinksToUse: MLXArray? = attn_sink.sum().item(Float.self) != 0
            ? attn_sink.asType(queries.dtype)
            : nil
        let output = deepseekAttentionWithSinks(
            queries: queries, keys: kFull, values: kFull,
            cache: cache, scale: scale, mask: mask, sinks: sinksToUse)
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, numHeads, headDim)

        return woB(groupedOutputProjection(output))
    }
}

// MARK: - MoE

class DeepseekV4Gate: Module {
    let topK: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool
    let scoringFunc: String

    var weight: MLXArray
    var e_score_correction_bias: MLXArray

    init(config: DeepseekV4Configuration) {
        self.topK = config.numExpertsPerTok
        self.routedScalingFactor = config.routedScalingFactor
        self.normTopkProb = config.normTopkProb
        self.scoringFunc = config.scoringFunc
        self.weight = zeros([config.nRoutedExperts, config.hiddenSize])
        self.e_score_correction_bias = zeros([config.nRoutedExperts])
    }

    override func updateMissing(
        parameter: String, verify: VerifyUpdate, path: [String], modulePath: [String]
    ) throws {
        if parameter == "e_score_correction_bias" { return }
        try super.updateMissing(parameter: parameter, verify: verify, path: path, modulePath: modulePath)
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let logits = x.matmul(weight.T)
        var scores: MLXArray
        switch scoringFunc {
        case "softmax": scores = softmax(logits, axis: -1)
        case "sigmoid": scores = sigmoid(logits)
        default: scores = sqrtSoftplus(logits)
        }

        let scoresForChoice = scores + e_score_correction_bias
        let inds = argPartition(-scoresForChoice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var selectedScores = takeAlong(scores, inds, axis: -1)

        if topK > 1 && normTopkProb {
            selectedScores = selectedScores / (selectedScores.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (inds, selectedScores * routedScalingFactor)
    }
}

class DeepseekV4MoE: Module, UnaryLayer {
    let numExpertsPerTok: Int
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    var gate: DeepseekV4Gate
    @ModuleInfo(key: "shared_experts") var sharedExperts: Linear

    init(config: DeepseekV4Configuration) {
        self.numExpertsPerTok = config.numExpertsPerTok
        let limit = config.swiguLimit
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts,
            activation: { x in limit > 0 ? silu(clip(x, min: -limit, max: limit)) : silu(x) })
        self.gate = DeepseekV4Gate(config: config)
        self._sharedExperts.wrappedValue = Linear(
            config.hiddenSize, config.moeIntermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var y = switchMLP(x, indices)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
        return y + sharedExperts(x)
    }
}

// MARK: - Decoder Block

class DeepseekV4Block: Module {
    let config: DeepseekV4Configuration

    @ModuleInfo(key: "attn") var selfAttn: DeepseekV4Attention
    var ffn: DeepseekV4MoE
    @ModuleInfo(key: "attn_norm") var attnNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    var hc_attn: HCParams
    var hc_ffn: HCParams

    init(config: DeepseekV4Configuration) {
        self.config = config
        self._selfAttn.wrappedValue = DeepseekV4Attention(config: config)
        self.ffn = DeepseekV4MoE(config: config)
        self._attnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        let hc = config.hcMult
        let mixHc = (2 + hc) * hc
        let hcDim = hc * config.hiddenSize
        self.hc_attn = HCParams(fn: zeros([mixHc, hcDim]), base: zeros([mixHc]), scale: ones([3]))
        self.hc_ffn  = HCParams(fn: zeros([mixHc, hcDim]), base: zeros([mixHc]), scale: ones([3]))
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let residualAttn = x
        let (xAttn, postAttn, combAttn) = hcPre(
            x: residualAttn, hcFn: hc_attn.fn, hcScale: hc_attn.scale, hcBase: hc_attn.base,
            hcMult: config.hcMult, sinkhornIters: config.hcSinkhornIters, eps: config.hcEps)
        let attnOut = selfAttn(attnNorm(xAttn), mask: mask, cache: cache)
        let residualFfn = hcPost(x: attnOut, residual: residualAttn, post: postAttn, comb: combAttn)

        let (xFfn, postFfn, combFfn) = hcPre(
            x: residualFfn, hcFn: hc_ffn.fn, hcScale: hc_ffn.scale, hcBase: hc_ffn.base,
            hcMult: config.hcMult, sinkhornIters: config.hcSinkhornIters, eps: config.hcEps)
        let ffnOut = ffn(ffnNorm(xFfn))
        return hcPost(x: ffnOut, residual: residualFfn, post: postFfn, comb: combFfn)
    }
}

// MARK: - Inner Model

public class DeepseekV4ModelInner: Module {
    var config: DeepseekV4Configuration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DeepseekV4Block]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    var hc_head: HCParams

    init(config: DeepseekV4Configuration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        // numHiddenLayers refers to main transformer layers only.
        // MTP prediction layers are separate (mtp.*) and NOT included here.
        self.layers = (0 ..< config.numHiddenLayers).map { _ in DeepseekV4Block(config: config) }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        let hc = config.hcMult
        self.hc_head = HCParams(
            fn: zeros([hc, hc * config.hiddenSize]),
            base: zeros([hc]),
            scale: ones([1]))
    }

    func callAsFunction(_ x: MLXArray, cache: [KVCache]?) -> MLXArray {
        let (h, _) = forward(x, cache: cache, returnRawHidden: false)
        return h
    }

    /// Forward pass that optionally returns the 4D pre-hcHead hidden state.
    /// Used by MTP to produce the draft proposal from the raw hidden state.
    func forward(_ x: MLXArray, cache: [KVCache]?, returnRawHidden: Bool) -> (MLXArray, MLXArray?) {
        let B = x.dim(0), S = x.dim(1)
        let hc = config.hcMult

        var h = embedTokens(x)
        h = h.expandedDimensions(axis: 2)
        h = repeated(h, count: hc, axis: 2)

        let hForMask = h.reshaped([B, S, hc * config.hiddenSize])
        let attentionMask = createAttentionMask(h: hForMask, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: attentionMask, cache: cache?[i])
        }

        let rawHidden = returnRawHidden ? h : nil
        h = hcHeadReduce(
            x: h, hcFn: hc_head.fn, hcScale: hc_head.scale, hcBase: hc_head.base, eps: config.hcEps)
        return (norm(h), rawHidden)
    }
}

// MARK: - Top-level Model

public class DeepseekV4Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public var kvHeads: [Int]

    var args: DeepseekV4Configuration
    public var model: DeepseekV4ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    /// MTP prediction blocks. Non-nil only when `_deepseekV4MTPEnabled == true` at init time.
    /// Defined here (not in DeepseekV4MTP.swift) so weight loading works via Module introspection.
    var mtp: [DeepseekV4MTPBlock]?

    init(_ args: DeepseekV4Configuration) {
        self.args = args
        self.kvHeads = Array(repeating: 1, count: args.numHiddenLayers)
        self.model = DeepseekV4ModelInner(config: args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)

        if args.numNextnPredictLayers > 0 && _deepseekV4MTPEnabled {
            let n = args.numHiddenLayers
            self.mtp = (0 ..< args.numNextnPredictLayers).map {
                DeepseekV4MTPBlock(config: args, layerIdx: n + $0)
            }
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var w = weights
        let hasMTP = mtp != nil
        let hasMTPWeights = weights.keys.contains { $0.hasPrefix("mtp.") }

        // If MTP is enabled but weights are absent, disable the module.
        if hasMTP && !hasMTPWeights {
            self.mtp = nil
        }

        // FP8 dequant (same as V3/SharpAI pattern).
        func dequant(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
            let bs = 128
            let (m, n) = (weight.dim(0), weight.dim(1))
            let padBottom = (bs - m % bs) % bs
            let padSide = (bs - n % bs) % bs
            var p = MLX.padded(weight, widths: [.init((0, padBottom)), .init((0, padSide))])
            p = p.reshaped([(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
            let scaled = p * scaleInv[0..., .newAxis, 0..., .newAxis]
            return scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
        }
        for (key, value) in weights {
            if key.contains("weight_scale_inv") {
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let wt = weights[weightKey] {
                    w[weightKey] = dequant(weight: wt, scaleInv: value)
                }
            } else if w[key] == nil {
                w[key] = value
            }
        }

        // Stack per-expert weights → switch_mlp format, for both main and MTP layers.
        for l in 0 ..< args.numHiddenLayers {
            let prefix = "model.layers.\(l)"
            for projName in ["gate_proj", "down_proj", "up_proj"] {
                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).ffn.experts.0.\(projName).\(key)"
                    if weights[firstKey] != nil {
                        let stacked = (0 ..< args.nRoutedExperts).map {
                            w["\(prefix).ffn.experts.\($0).\(projName).\(key)"]
                                ?? weights["\(prefix).ffn.experts.\($0).\(projName).\(key)"]!
                        }
                        w["\(prefix).ffn.switch_mlp.\(projName).\(key)"] = MLX.stacked(stacked)
                        for j in 0 ..< args.nRoutedExperts {
                            w.removeValue(forKey: "\(prefix).ffn.experts.\(j).\(projName).\(key)")
                        }
                    }
                }
            }
        }

        // Stack per-expert weights for MTP layers.
        if hasMTP && hasMTPWeights {
            for i in 0 ..< args.numNextnPredictLayers {
                let prefix = "mtp.\(i).block.ffn.experts"
                for projName in ["gate_proj", "down_proj", "up_proj"] {
                    for key in ["weight", "scales"] {
                        let firstKey = "\(prefix).0.\(projName).\(key)"
                        if weights[firstKey] != nil {
                            let stacked = (0 ..< args.nRoutedExperts).map {
                                w["\(prefix).\($0).\(projName).\(key)"]
                                    ?? weights["\(prefix).\($0).\(projName).\(key)"]!
                            }
                            w["mtp.\(i).block.ffn.switch_mlp.\(projName).\(key)"] = MLX.stacked(stacked)
                            for j in 0 ..< args.nRoutedExperts {
                                w.removeValue(forKey: "\(prefix).\(j).\(projName).\(key)")
                            }
                        }
                    }
                }
            }
        }

        return w.filter { key, _ in
            // Drop MTP weights when MTP is not enabled.
            if key.hasPrefix("mtp.") && !hasMTP { return false }
            // Drop compressor/indexer (not yet implemented in this port).
            if key.contains(".attn.compressor.") || key.contains(".attn.indexer.") { return false }
            // Drop hash-layer tid2eid (hash routing not yet implemented; fall back to learned gate).
            if key.contains(".ffn.gate.tid2eid") { return false }
            // Drop precomputed rotary frequencies.
            if key.contains("rotary_emb.inv_freq") { return false }
            return true
        }
    }

    public var loraLayers: [Module] { model.layers }
}
