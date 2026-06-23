// Copyright © 2025 Apple Inc.

// Port of DeepSeek-V4 inference code
// Reference: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct DeepseekV4Configuration: Codable, Sendable {
    // Core architecture
    var vocabSize: Int
    var hiddenSize: Int
    var moeIntermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var headDim: Int
    var qLoraRank: Int
    var qkRopeHeadDim: Int
    var rmsNormEps: Float

    // Output projection grouping
    var oGroups: Int
    var oLoraRank: Int

    // Attention / compression (per layer)
    var slidingWindow: Int
    var compressRatios: [Int]
    var compressRopeTheta: Float

    // MoE
    var nRoutedExperts: Int
    var nSharedExperts: Int
    var numExpertsPerTok: Int
    var scoringFunc: String
    var routedScalingFactor: Float
    var swiguLimit: Float
    var numHashLayers: Int
    var numNextnPredictLayers: Int
    var normTopkProb: Bool

    // Hyper-Connections (mHC)
    var hcMult: Int
    var hcSinkhornIters: Int
    var hcEps: Float

    // RoPE
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var maxPositionEmbeddings: Int

    // Nope head dim (derived)
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

/// sqrtsoftplus activation: sqrt(softplus(x)) = sqrt(log(1 + e^x))
/// Uses numerically stable form to avoid exp overflow for large positive x.
private func sqrtSoftplus(_ x: MLXArray) -> MLXArray {
    let sp = MLX.maximum(x, MLXArray(0)) + MLX.log1p(MLX.exp(-MLX.abs(x)))
    return MLX.sqrt(sp)
}

/// Apply per-head RMS normalization (without learnable scale)
private func headRmsNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    x * rsqrt(x.square().mean(axis: -1, keepDims: true) + eps)
}

// MARK: - Sinkhorn-based Hyper-Connection helpers

/// Cache of compiled Sinkhorn graphs keyed by (hcMult, iters, eps).
/// The raw graph is ~80 tiny elementwise/reduce ops; at 2 calls/layer × 43 layers that is
/// thousands of kernel dispatches per decoded token. `MLX.compile` fuses them.
private final class HCSinkhornCompileCache: @unchecked Sendable {
    static let shared = HCSinkhornCompileCache()
    private var cache: [String: @Sendable ([MLXArray]) -> [MLXArray]] = [:]
    private let lock = NSLock()

    func fn(hcMult: Int, sinkhornIters: Int, eps: Float)
        -> @Sendable ([MLXArray]) -> [MLXArray]
    {
        let key = "\(hcMult)-\(sinkhornIters)-\(eps)"
        lock.lock()
        defer { lock.unlock() }
        if let f = cache[key] { return f }
        let hc = hcMult
        let f = compile { (args: [MLXArray]) -> [MLXArray] in
            let mixes = args[0], hcScale = args[1], hcBase = args[2]
            let B = mixes.dim(0), S = mixes.dim(1)

            // Split mixes / base into (pre, post, comb) parts.
            let preMix = mixes[.ellipsis, ..<hc]
            let postMix = mixes[.ellipsis, hc ..< 2 * hc]
            let combMix = mixes[.ellipsis, (2 * hc)...]
            let preBase = hcBase[..<hc]
            let postBase = hcBase[hc ..< 2 * hc]
            let combBase = hcBase[(2 * hc)...]

            // Match the official DeepSeek-V4 hc_split_sinkhorn kernel EXACTLY (kernel.py:372).
            // pre: sigmoid + eps, NOT normalized.  post: 2*sigmoid, NO eps (range (0,2),
            // identity at zero logits).
            let pre = sigmoid(preMix * hcScale[0] + preBase) + eps     // [B, S, hc]
            let post = 2 * sigmoid(postMix * hcScale[1] + postBase)   // [B, S, hc]

            // comb: RAW affine logits → softmax(-1)+eps → column → (iters-1)×(row, column).
            var comb = (combMix * hcScale[2] + combBase).reshaped(B, S, hc, hc)
            comb = softmax(comb, axis: -1) + eps
            comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
            for _ in 0 ..< (sinkhornIters - 1) {
                comb = comb / (comb.sum(axis: -1, keepDims: true) + eps)
                comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
            }
            return [pre, post, comb]
        }
        cache[key] = f
        return f
    }
}

/// Split mixes into (pre, post, comb) with Sinkhorn normalization (compiled).
/// mixes: [B, S, mix_hc] where mix_hc = (2+hc)*hc
/// Returns pre [B,S,hc], post [B,S,hc], comb [B,S,hc,hc]
private func hcSplitSinkhorn(
    _ mixes: MLXArray,
    hcScale: MLXArray,    // [3]
    hcBase: MLXArray,     // [mix_hc]
    hcMult: Int,
    sinkhornIters: Int,
    eps: Float
) -> (MLXArray, MLXArray, MLXArray) {
    let f = HCSinkhornCompileCache.shared.fn(
        hcMult: hcMult, sinkhornIters: sinkhornIters, eps: eps)
    let out = f([mixes, hcScale, hcBase])
    return (out[0], out[1], out[2])
}

/// Hyper-Connection pre-step: reduce [B,S,hc,D] → [B,S,D] with Sinkhorn weights.
/// Returns (reduced_x, post_weights, comb_matrix).
private func hcPre(
    x: MLXArray,           // [B, S, hc, D]
    hcFn: MLXArray,        // [mix_hc, hc*D]
    hcScale: MLXArray,     // [3]
    hcBase: MLXArray,      // [mix_hc]
    hcMult: Int,
    sinkhornIters: Int,
    eps: Float
) -> (MLXArray, MLXArray, MLXArray) {
    let dtype = x.dtype
    let B = x.dim(0), S = x.dim(1), hc = x.dim(2), D = x.dim(3)

    // Flatten: [B, S, hc*D]
    let xFlat = x.reshaped(B, S, hc * D).asType(.float32)

    // RMS-style normalization scale
    let normScale = rsqrt(xFlat.square().mean(axis: -1, keepDims: true) + eps)

    // Linear projection: [B, S, mix_hc]
    let mixes = matmul(xFlat, hcFn.T) * normScale

    let (pre, post, comb) = hcSplitSinkhorn(
        mixes, hcScale: hcScale, hcBase: hcBase,
        hcMult: hcMult, sinkhornIters: sinkhornIters, eps: eps)

    // Weighted sum of hc copies: [B, S, D] — reduce in fp32 (pre is fp32, xFlat is fp32),
    // matching the official kernel, then cast back.
    let y = (pre.expandedDimensions(axis: -1) * xFlat.reshaped(B, S, hc, D)).sum(axis: -2).asType(dtype)

    return (y, post, comb)
}

/// Hyper-Connection post-step: expand sublayer output back to [B,S,hc,D].
/// y[b,s,j,:] = post[b,s,j]*x[b,s,:] + sum_i(comb[b,s,i,j]*residual[b,s,i,:])
private func hcPost(
    x: MLXArray,        // [B, S, D] - sublayer output
    residual: MLXArray, // [B, S, hc, D] - input to this block
    post: MLXArray,     // [B, S, hc]
    comb: MLXArray      // [B, S, hc, hc]
) -> MLXArray {
    // term1: post[b,s,j] * x[b,s,:] → broadcast to [B,S,hc,D]
    let term1 = post.expandedDimensions(axis: -1) * x.expandedDimensions(axis: -2)

    // term2: sum_i(comb[b,s,i,j] * residual[b,s,i,:])
    // comb.unsqueeze(-1): [B,S,hc_i,hc_j,1]
    // residual.unsqueeze(-2): [B,S,hc_i,1,D]
    // product: [B,S,hc_i,hc_j,D] → sum over dim 2 → [B,S,hc_j,D]
    let combExp = comb.expandedDimensions(axis: -1)         // [B,S,hc,hc,1]
    let residualExp = residual.expandedDimensions(axis: -2) // [B,S,hc,1,D]
    let term2 = (combExp * residualExp).sum(axis: 2)        // [B,S,hc,D]

    return (term1 + term2).asType(x.dtype)
}

// MARK: - HCParams Module

/// Lightweight Module to hold the three Hyper-Connection tensors loaded from checkpoint.
/// Key names (fn, base, scale) match the `hc_attn.*` / `hc_ffn.*` / `hc_head.*` paths.
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

/// Final HC head: reduce [B,S,hc,D] → [B,S,D] for lm_head.
/// No Sinkhorn – just sigmoid + eps weighted sum.
private func hcHead(
    x: MLXArray,        // [B, S, hc, D]
    hcFn: MLXArray,     // [hc, hc*D]
    hcScale: MLXArray,  // [1]
    hcBase: MLXArray,   // [hc]
    eps: Float
) -> MLXArray {
    let dtype = x.dtype
    let B = x.dim(0), S = x.dim(1), hc = x.dim(2), D = x.dim(3)

    let xFlat = x.reshaped(B, S, hc * D).asType(.float32)
    let normScale = rsqrt(xFlat.square().mean(axis: -1, keepDims: true) + eps)
    let mixes = matmul(xFlat, hcFn.T) * normScale           // [B, S, hc]
    let pre = sigmoid(mixes * hcScale + hcBase) + eps        // [B, S, hc]

    // Weighted sum in fp32 (matches official hc_head), then cast back: [B, S, D]
    let y = (pre.expandedDimensions(axis: -1) * xFlat.reshaped(B, S, hc, D)).sum(axis: -2)
    return y.asType(dtype)
}

// MARK: - Inverse RoPE (de-rotation)

/// Negate the second element of each traditional (interleaved) RoPE pair.
/// For x[..., rd] viewed as rd/2 pairs (x0,x1): returns (x0, -x1).
private func ropeFlipPairSign(_ x: MLXArray) -> MLXArray {
    let lead = Array(x.shape.dropLast())
    let rd = x.dim(-1)
    let signs = MLXArray([Float(1), Float(-1)])               // [2]
    return (x.reshaped(lead + [rd / 2, 2]) * signs).reshaped(x.shape)
}

/// Apply the INVERSE rotary embedding R(-θ) to the last `rope` dims of `x`, using the
/// conjugate identity  R(-θ) = C ∘ R(θ) ∘ C  where C flips the sign of the 2nd element of
/// each interleaved pair. Matches the official `apply_rotary_emb(o, freqs_cis, inverse=True)`
/// (which conjugates freqs_cis). `x` must have its sequence axis at -2 (e.g. [B, H, L, rd]).
private func inverseRotary<R: RoPELayer>(_ rope: R, _ x: MLXArray, offset: Int) -> MLXArray {
    ropeFlipPairSign(rope(ropeFlipPairSign(x), offset: offset))
}

// MARK: - DSA (DeepSeek Sparse Attention) — window + compressed-KV structure

/// YaRN-interpolated rope wavelengths, replicating YarnRoPE's internal freq
/// computation (RoPEUtils.swift) so DSA compressed slots can be roped at
/// block-start positions with stride `ratio` (see DeepseekV4Compressor).
private func dsv4YarnWavelengths(config: DeepseekV4Configuration) -> MLXArray {
    let dims = config.qkRopeHeadDim
    let base = config.compressRopeTheta
    let s = config.ropeScaling
    let factor = s?["factor"]?.asFloat() ?? 16.0
    let origMax = s?["original_max_position_embeddings"]?.asInt() ?? 65536
    let betaFast = s?["beta_fast"]?.asFloat() ?? 32.0
    let betaSlow = s?["beta_slow"]?.asFloat() ?? 1.0

    func correctionDim(_ numRotations: Float) -> Float {
        Float(dims) * log(Float(origMax) / (numRotations * 2 * Float.pi)) / (2 * log(base))
    }
    let low = max(Int(floor(correctionDim(betaFast))), 0)
    let high = min(Int(ceil(correctionDim(betaSlow))), dims - 1)
    var maxV = Float(high)
    let minV = Float(low)
    if minV == maxV { maxV += 0.001 }
    let linear = (MLXArray(0 ..< (dims / 2)).asType(.float32) - minV) / (maxV - minV)
    let freqMask = 1.0 - clip(linear, min: 0, max: 1)

    let freqExtra = pow(base, MLXArray(stride(from: 0, to: dims, by: 2)).asType(.float32) / Float(dims))
    let freqInter = factor * freqExtra
    return (freqInter * freqExtra) / (freqInter * freqMask + freqExtra * (1 - freqMask))
}

/// Learned KV compressor (official model.py:279-377): gated softmax pooling of
/// `ratio` consecutive tokens (overlapping the previous block when ratio==4)
/// into one 512-dim slot, RMS-normed and roped at the block-start position.
class DeepseekV4Compressor: Module {
    let ratio: Int
    let headDim: Int
    let ropeDim: Int
    let overlap: Bool
    let coff: Int

    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "wgate") var wgate: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm
    var ape: MLXArray  // [ratio, coff*headDim] (bf16 in checkpoint)

    // Slot rope: slot k of a run starting at block-aligned position s sits at
    // absolute position s + k*ratio. theta = (s+k·r)/λ = (s/r + k)/(λ/r), so
    // rope with freqs λ/r at integer offset s/r yields stride-r positions.
    private let slotFreqs: MLXArray

    init(config: DeepseekV4Configuration, ratio: Int) {
        self.ratio = ratio
        self.headDim = config.headDim
        self.ropeDim = config.qkRopeHeadDim
        self.overlap = (ratio == 4)
        self.coff = overlap ? 2 : 1

        self._wkv.wrappedValue = Linear(config.hiddenSize, coff * config.headDim, bias: false)
        self._wgate.wrappedValue = Linear(config.hiddenSize, coff * config.headDim, bias: false)
        self._norm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        self.ape = zeros([ratio, coff * config.headDim])
        self.slotFreqs = dsv4YarnWavelengths(config: config) / Float(ratio)
        super.init()
    }

    /// Consume `x` ([B, L, hidden], the attention sublayer input) and emit any
    /// completed compressed slots into the cache's slot buffer.
    func process(x: MLXArray, cache: DSAKVCache) {
        let dtype = x.dtype
        let B = x.dim(0)
        let d = headDim

        // fp32 projections (official compresses in fp32).
        var kvP = wkv(x).asType(.float32)      // [B, L, coff*d]
        var scP = wgate(x).asType(.float32)
        let L = kvP.dim(1)

        if cache.pendKV == nil {
            cache.pendKV = MLXArray.zeros([B, ratio, coff * d], dtype: .float32)
            cache.pendScore = MLXArray.zeros([B, ratio, coff * d], dtype: .float32)
            cache.pendLen = 0
        }

        // ── Decode fast path: block not yet complete → one in-place pending
        // write per projection, no concat/alloc churn.
        if cache.pendLen + L < ratio {
            cache.pendKV?[0..., cache.pendLen ..< (cache.pendLen + L), 0...] = kvP
            cache.pendScore?[0..., cache.pendLen ..< (cache.pendLen + L), 0...] = scP
            cache.pendLen += L
            return
        }

        // ── Block-completing path (every `ratio`-th token at decode; any
        // prefill chunk): assemble pending + new, run the batch pooling.
        if cache.pendLen > 0 {
            kvP = concatenated([cache.pendKV![0..., ..<cache.pendLen, 0...], kvP], axis: 1)
            scP = concatenated([cache.pendScore![0..., ..<cache.pendLen, 0...], scP], axis: 1)
        }

        let P = kvP.dim(1)
        let nBlocks = P / ratio
        if nBlocks > 0 {
            let cutoff = nBlocks * ratio
            var blocksKV = kvP[0..., ..<cutoff, 0...].reshaped(B, nBlocks, ratio, coff * d)
            var blocksSc = scP[0..., ..<cutoff, 0...].reshaped(B, nBlocks, ratio, coff * d)
                + ape.asType(.float32)  // ape per in-block position

            var poolK: MLXArray
            var poolS: MLXArray
            if overlap {
                // rows ratio..<2r: current block second-half channels;
                // rows 0..<ratio: PREVIOUS block first-half channels
                // (block 0's previous comes from carried state, else pad 0/-inf).
                let prevK0 = cache.prevBlockKV
                    ?? MLXArray.zeros([B, 1, ratio, d]).asType(.float32)
                let prevS0 = cache.prevBlockScore
                    ?? MLXArray.full([B, 1, ratio, d], values: MLXArray(-Float.infinity)).asType(.float32)
                let curFirstHalfK = blocksKV[.ellipsis, ..<d]    // [B, n, r, d]
                let curFirstHalfS = blocksSc[.ellipsis, ..<d]
                let prevKs = nBlocks > 1
                    ? concatenated([prevK0, curFirstHalfK[0..., ..<(nBlocks - 1), 0..., 0...]], axis: 1)
                    : prevK0
                let prevSs = nBlocks > 1
                    ? concatenated([prevS0, curFirstHalfS[0..., ..<(nBlocks - 1), 0..., 0...]], axis: 1)
                    : prevS0
                poolK = concatenated([prevKs, blocksKV[.ellipsis, d...]], axis: 2)  // [B,n,2r,d]
                poolS = concatenated([prevSs, blocksSc[.ellipsis, d...]], axis: 2)
                // Carry the last complete block's first-half channels forward.
                cache.prevBlockKV = curFirstHalfK[0..., (nBlocks - 1)..., 0..., 0...]
                cache.prevBlockScore = curFirstHalfS[0..., (nBlocks - 1)..., 0..., 0...]
            } else {
                poolK = blocksKV  // [B, n, r, d]
                poolS = blocksSc
            }

            // Gated softmax pooling over the block positions → [B, n, d].
            var slots = (poolK * softmax(poolS, axis: 2)).sum(axis: 2)
            slots = norm(slots.asType(dtype))

            // Rope the tail dims at block-start positions (stride ratio).
            let slotIdx0 = cache.pendStart / ratio
            let nope = slots[.ellipsis, ..<(d - ropeDim)]
            let roped = MLXFast.RoPE(
                slots[.ellipsis, (d - ropeDim)...],
                dimensions: ropeDim, traditional: true, base: nil,
                scale: 1.0, offset: slotIdx0, freqs: slotFreqs)
            slots = concatenated([nope, roped], axis: -1)

            let slotsHeaded = slots.expandedDimensions(axis: 1)  // [B, 1, n, d]
            cache.appendSlots(slotsHeaded)
            cache.pendStart += cutoff

            // Remainder becomes the new pending (raw scores, no ape) — written
            // in-place into the preallocated pending buffers.
            let rem = P - cutoff
            if rem > 0 {
                cache.pendKV?[0..., ..<rem, 0...] = kvP[0..., cutoff..., 0...]
                cache.pendScore?[0..., ..<rem, 0...] = scP[0..., cutoff..., 0...]
            }
            cache.pendLen = rem
        } else {
            // L ≥ 1 but pending+L < ratio handled above; here only when a
            // multi-token chunk still doesn't complete a block (pendLen+L < ratio
            // was false only if == ratio-boundary edge) — store everything.
            cache.pendKV?[0..., ..<P, 0...] = kvP
            cache.pendScore?[0..., ..<P, 0...] = scP
            cache.pendLen = P
        }
    }
}

// MARK: - Attention

/// Attention with cache update that optionally applies per-head sink bias.
/// Mirrors `attentionWithCacheUpdateAndSinks` from MiMoV2Flash but uses the public API.
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
    let config: DeepseekV4Configuration
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

    // Q low-rank projections
    @ModuleInfo(key: "wq_a") var wqA: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "wq_b") var wqB: Linear

    // Unified KV projection (K and V share the same projection)
    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm

    // Grouped output projection
    @ModuleInfo(key: "wo_a") var woA: Linear
    @ModuleInfo(key: "wo_b") var woB: Linear

    // Attention sink bias (per head, no .weight suffix)
    // Stored via update(parameters:) using the key "attn_sink"
    var attn_sink: MLXArray
    // Cached "is the sink non-zero" check. The naive per-call
    // `attn_sink.sum().item()` forces a blocking GPU sync EVERY layer EVERY
    // token (43 syncs/token on DSV4) — evaluate once on first use instead.
    private var _sinkActive: Bool? = nil

    // DSA: learned KV compressor on compress layers (ratio 4 or 128); nil on
    // pure sliding-window layers (0, 1). Checkpoint keys: attn.compressor.*
    @ModuleInfo(key: "compressor") var compressor: DeepseekV4Compressor?
    let compressRatio: Int
    let windowSize: Int

    init(config: DeepseekV4Configuration, layerIdx: Int = 0) {
        self.config = config
        self.compressRatio = layerIdx < config.compressRatios.count
            ? config.compressRatios[layerIdx] : 0
        self.windowSize = config.slidingWindow
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.nopeHeadDim = config.nopeHeadDim
        self.ropeHeadDim = config.qkRopeHeadDim
        self.oGroups = config.oGroups
        self.oLoraRank = config.oLoraRank
        self.nHeadsPerGroup = config.numAttentionHeads / config.oGroups
        self.scale = pow(Float(config.headDim), -0.5)
        self.eps = config.rmsNormEps

        // Q projections
        self._wqA.wrappedValue = Linear(config.hiddenSize, config.qLoraRank, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: config.rmsNormEps)
        self._wqB.wrappedValue = Linear(config.qLoraRank, config.numAttentionHeads * config.headDim, bias: false)

        // Unified KV: single head, headDim dimensional
        self._wkv.wrappedValue = Linear(config.hiddenSize, config.headDim, bias: false)
        self._kvNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)

        // Grouped output projection
        // wo_a: Linear(nHeadsPerGroup * headDim, oGroups * oLoraRank) per group → stored as [oGroups*oLoraRank, nHeadsPerGroup*headDim]
        self._woA.wrappedValue = Linear(nHeadsPerGroup * config.headDim, config.oGroups * config.oLoraRank, bias: false)
        self._woB.wrappedValue = Linear(config.oGroups * config.oLoraRank, config.hiddenSize, bias: false)

        // Attention sink: per-head bias [numAttentionHeads], applied to attention logits before softmax.
        // Shape matches numAttentionHeads (== qkRopeHeadDim in this architecture).
        self.attn_sink = zeros([config.numAttentionHeads])

        // Per-layer RoPE, matching the official DeepSeek-V4 reference (model.py:475-482):
        //  - compress layers (compress_ratios[l] != 0): base = compress_rope_theta (160000)
        //    with YaRN *frequency* interpolation (factor/origMax/beta from rope_scaling).
        //  - pure sliding-window layers (compress_ratios[l] == 0, i.e. layers 0,1):
        //    base = rope_theta (10000), YaRN disabled (plain RoPE).
        // CRITICAL: the official applies NO YaRN magnitude correction anywhere
        //   (softmax_scale = head_dim**-0.5 flat; freqs_cis = polar(ones, .)). The generic
        //   YarnRoPE here would otherwise bake an mscale of ~1.277 into q AND k, scaling every
        //   attention logit by ~1.63x. We neutralize it by forcing _mscale == 1.
        let isCompressLayer = layerIdx < config.compressRatios.count
            && config.compressRatios[layerIdx] != 0
        if isCompressLayer {
            let s = config.ropeScaling
            self.rope = YarnRoPE(
                dimensions: config.qkRopeHeadDim,
                traditional: true,
                maxPositionEmbeddings: config.maxPositionEmbeddings,
                base: config.compressRopeTheta,
                scalingFactor: s?["factor"]?.asFloat() ?? 16.0,
                originalMaxPositionEmbeddings: s?["original_max_position_embeddings"]?.asInt() ?? 65536,
                betaFast: s?["beta_fast"]?.asFloat() ?? 32.0,
                betaSlow: s?["beta_slow"]?.asFloat() ?? 1.0,
                mscale: 0,        // mscale==mscaleAllDim → _mscale == 1 (no magnitude scaling)
                mscaleAllDim: 0
            )
        } else {
            self.rope = RoPE(
                dimensions: config.qkRopeHeadDim,
                traditional: true,
                base: config.ropeTheta,
                scale: 1.0
            )
        }

        // DSA compressor exists only on compress layers (checkpoint has
        // attn.compressor.* for those layers only).
        if compressRatio > 0 && dsaEnabled {
            self._compressor.wrappedValue =
                DeepseekV4Compressor(config: config, ratio: compressRatio)
        }
    }

    /// Grouped output projection matching the reference Python implementation.
    /// For QuantizedLinear wo_a: slices weight rows per group, calls quantizedMM.
    /// For plain Linear wo_a: uses batched matmul after weight reshape.
    /// Input:  [B, L, n_heads, head_dim]
    /// Output: [B, L, oGroups * oLoraRank]
    private func groupedOutputProjection(_ out: MLXArray) -> MLXArray {
        let B = out.dim(0), L = out.dim(1)
        let groupFeat = numHeads * headDim / oGroups  // = nHeadsPerGroup * headDim

        // Flatten to [B, L, n_heads * head_dim] for easy group slicing
        let outFlat = out.reshaped(B, L, numHeads * headDim)

        if let qLinear = woA as? QuantizedLinear {
            var pieces: [MLXArray] = []
            for g in 0 ..< oGroups {
                let gStart = g * groupFeat
                let gEnd   = (g + 1) * groupFeat
                let rStart = g * oLoraRank
                let rEnd   = (g + 1) * oLoraRank

                // Per-group input: [B, L, groupFeat]
                let groupInput = outFlat[0..., 0..., gStart ..< gEnd]
                // Slice weight rows for this group
                let wRows = qLinear.weight[rStart ..< rEnd]
                let sRows = qLinear.scales[rStart ..< rEnd]
                let bRows = qLinear.biases.map { $0[rStart ..< rEnd] }

                // quantizedMM: [B, L, groupFeat] @ dequant(wRows)^T → [B, L, oLoraRank]
                let y = quantizedMM(
                    groupInput,
                    wRows,
                    scales: sRows,
                    biases: bRows,
                    transpose: true,
                    groupSize: qLinear.groupSize,
                    bits: qLinear.bits,
                    mode: qLinear.mode
                )
                pieces.append(y)
            }
            return concatenated(pieces, axis: -1)  // [B, L, oGroups * oLoraRank]
        } else {
            // Non-quantized fallback: per-group matmul (same structure as quantized path).
            // A single batched matmul would broadcast batch dims [B,L] against [oGroups],
            // which fails when L != oGroups, so we loop instead.
            var pieces: [MLXArray] = []
            for g in 0 ..< oGroups {
                let gStart = g * groupFeat
                let gEnd   = (g + 1) * groupFeat
                let rStart = g * oLoraRank
                let rEnd   = (g + 1) * oLoraRank
                let groupInput = outFlat[0..., 0..., gStart ..< gEnd]  // [B, L, groupFeat]
                let wa_g = woA.weight[rStart ..< rEnd]                 // [oLoraRank, groupFeat]
                pieces.append(matmul(groupInput, wa_g.T))              // [B, L, oLoraRank]
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
        // Capture the RoPE offset BEFORE the cache update (the attention call advances
        // cache.offset). Needed to de-rotate the attention output at the query positions.
        let ropeOffset = cache?.offset ?? 0

        // --- Query ---
        // Low-rank Q: wq_a → q_norm → wq_b
        var q = wqB(qNorm(wqA(x)))                       // [B, L, n_heads * head_dim]
        q = q.reshaped(B, L, numHeads, headDim)
            .transposed(0, 2, 1, 3)                       // [B, n_heads, L, head_dim]
        // Per-head RMS normalization (no learnable scale)
        q = headRmsNorm(q, eps: eps)

        // Split Q into nope and rope parts
        let qNope = q[.ellipsis, ..<nopeHeadDim]          // [B, n_heads, L, nope_head_dim]
        var qRope = q[.ellipsis, nopeHeadDim...]           // [B, n_heads, L, rope_head_dim]
        qRope = applyRotaryPosition(rope, to: qRope, cache: cache)
        let queries = concatenated([qNope, qRope], axis: -1) // [B, n_heads, L, head_dim]

        // --- KV: k = v (reference: k = v = concat([k_nope, k_pe_roped])) ---
        let kv = kvNorm(wkv(x))                           // [B, L, head_dim]
        let kvNope = kv[.ellipsis, ..<nopeHeadDim]
            .reshaped(B, L, 1, nopeHeadDim)
            .transposed(0, 2, 1, 3)                       // [B, 1, L, nope_head_dim]
        var kvRope = kv[.ellipsis, nopeHeadDim...]
            .reshaped(B, L, 1, ropeHeadDim)
            .transposed(0, 2, 1, 3)                       // [B, 1, L, rope_head_dim]
        kvRope = applyRotaryPosition(rope, to: kvRope, cache: cache)
        let kFull = concatenated([kvNope, kvRope], axis: -1) // [B, 1, L, head_dim]
        // In reference k = v = kFull: both K and V have rope applied to their rope dims.
        // attentionWithCacheUpdate handles the KV cache update internally.

        // --- Attention ---
        // Pass kFull as both keys and values; cache update happens inside.
        // Apply attn_sink (per-head bias) to attention logits when non-zero.
        // Cast to queries.dtype (bfloat16) — attn_sink may be loaded as float32 from the
        // checkpoint, but MLXFast.scaledDotProductAttention requires sinks to promote to
        // the output dtype (bfloat16); float32 does not satisfy this constraint.
        if _sinkActive == nil {
            _sinkActive = attn_sink.sum().item(Float.self) != 0  // one-time sync per layer
        }
        let sinksToUse: MLXArray? = (_sinkActive == true)
            ? attn_sink.asType(queries.dtype)
            : nil
        var attnOut: MLXArray
        if let dsa = cache as? DSAKVCache {
            // ── DSA path: sliding-window raw KV (ring buffer) + compressed
            // slots. Matches the official structure (model.py:484-534): queries
            // see at most the last `windowSize` raw tokens plus all completed
            // compressed slots (slot j visible iff (j+1)*ratio ≤ pos+1).
            // Ring rows are in (pos % windowSize) order — SDPA over a key SET is
            // permutation-invariant (rope is baked into keys), so decode needs
            // no mask at all; prefill masks use the rows' absolute positions.
            let offset = dsa.offset
            let winK = dsa.appendWindow(kFull)        // in-place ring write, view out
            let nWin = winK.dim(2)

            // Emit compressed slots BEFORE building masks so slots completed by
            // this chunk are visible to its later queries.
            if let comp = compressor { comp.process(x: x, cache: dsa) }
            let nSlots = dsa.slotCount

            let keysAll = dsa.slotsView.map { concatenated([winK, $0], axis: 2) } ?? winK
            let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
            if L == 1 {
                // Decode fast path: all ≤ windowSize ring rows are valid past
                // tokens and every completed slot is visible — no mask needed.
                maskMode = .none
            } else {
                // Additive masks in fp32 → cast to query dtype. Window rows are
                // in ring order; use their stored absolute positions.
                let qPos = (MLXArray(0 ..< L) + offset).reshaped(L, 1).asType(.float32)
                let kPos = MLXArray(dsa.windowPositions(count: nWin).map { Float($0) })
                    .reshaped(1, nWin)
                let diff = qPos - kPos
                let winVisible = (diff .>= 0) .&& (diff .< Float(windowSize))
                var maskArr = which(
                    winVisible, MLXArray(Float(0)), MLXArray(-Float.infinity))
                if nSlots > 0 {
                    // Slot j (j from 0) visible iff (j+1)*ratio ≤ qPos+1.
                    let slotEnd = (MLXArray(0 ..< nSlots) + 1).reshaped(1, nSlots)
                        .asType(.float32) * Float(compressRatio)
                    let compVisible = slotEnd .<= (qPos + 1)
                    let compMask = which(
                        compVisible, MLXArray(Float(0)), MLXArray(-Float.infinity))
                    maskArr = concatenated([maskArr, compMask], axis: -1)
                }
                maskMode = .array(maskArr.asType(queries.dtype))
            }
            attnOut = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keysAll, values: keysAll,
                scale: scale, mask: maskMode,
                sinks: sinksToUse)

            dsa.offset += L
        } else {
            attnOut = deepseekAttentionWithSinks(
                queries: queries,
                keys: kFull,
                values: kFull,
                cache: cache,
                scale: scale,
                mask: mask,
                sinks: sinksToUse
            )                                              // [B, n_heads, L, head_dim]
        }

        // --- Inverse-RoPE the output (official: apply_rotary_emb(o, freqs, inverse=True)) ---
        // In this MLA variant k == v == kv, so the value carries position-rotated content in
        // its last rope_head_dim dims; the attention output must be de-rotated at the query
        // positions before the output projection. Done here while the sequence axis is at -2.
        let oNope = attnOut[.ellipsis, ..<nopeHeadDim]
        let oRope = attnOut[.ellipsis, nopeHeadDim...]
        attnOut = concatenated(
            [oNope, inverseRotary(rope, oRope, offset: ropeOffset)], axis: -1)

        let output = attnOut
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, numHeads, headDim)               // [B, L, n_heads, head_dim]

        // --- Grouped output projection ---
        let oLora = groupedOutputProjection(output)        // [B, L, oGroups * oLoraRank]
        return woB(oLora)
    }
}

// MARK: - MoE Components

/// Single FFN expert: SwiGLU with optional activation clamping.
class DeepseekV4Expert: Module, UnaryLayer {
    let swiguLimit: Float

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int, swiguLimit: Float) {
        self.swiguLimit = swiguLimit
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gate = gateProj(x)
        var up = upProj(x)
        if swiguLimit > 0 {
            gate = clip(gate, min: -swiguLimit, max: swiguLimit)
            up = clip(up, min: -swiguLimit, max: swiguLimit)
        }
        return downProj(silu(gate) * up)
    }
}

/// MoE routing gate with sqrtsoftplus scoring. The first `numHashLayers` layers route by a
/// fixed token-ID → expert hash table (`tid2eid`); the rest route by top-k of the scores.
/// Matches the official DeepSeek-V4 Gate (model.py:546-584).
class DeepseekV4Gate: Module {
    let topK: Int
    let nRoutedExperts: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool
    let scoringFunc: String
    let isHash: Bool

    var weight: MLXArray              // [n_routed_experts, hidden_size]
    var e_score_correction_bias: MLXArray  // [n_routed_experts]
    // Token-ID → expert hash table [vocab, topk] (hash layers only). Tiny placeholder
    // on non-hash layers (kept missing, never indexed).
    var tid2eid: MLXArray

    init(config: DeepseekV4Configuration, layerIdx: Int = 999) {
        self.topK = config.numExpertsPerTok
        self.nRoutedExperts = config.nRoutedExperts
        self.routedScalingFactor = config.routedScalingFactor
        self.normTopkProb = config.normTopkProb
        self.scoringFunc = config.scoringFunc
        self.isHash = layerIdx < config.numHashLayers
        self.weight = zeros([config.nRoutedExperts, config.hiddenSize])
        self.e_score_correction_bias = zeros([config.nRoutedExperts])
        // Real [vocab, topk] shape on hash layers so a non-streaming load (verify: .all)
        // doesn't shape-mismatch; tiny placeholder elsewhere (never indexed). Overwritten on load.
        self.tid2eid = (layerIdx < config.numHashLayers)
            ? zeros([config.vocabSize, config.numExpertsPerTok]).asType(.int64)
            : zeros([1])
    }

    /// Hash layers have no `e_score_correction_bias`; non-hash layers have no `tid2eid`.
    /// Allow either to be absent from the checkpoint without failing the load.
    override func updateMissing(
        parameter: String,
        verify: VerifyUpdate,
        path: [String],
        modulePath: [String]
    ) throws {
        if parameter == "e_score_correction_bias" || parameter == "tid2eid" {
            return  // keep zero-initialized default
        }
        try super.updateMissing(
            parameter: parameter, verify: verify, path: path, modulePath: modulePath)
    }

    /// - Parameter inputIds: original token IDs `[B, S]` (required for hash layers).
    func callAsFunction(_ x: MLXArray, inputIds: MLXArray? = nil) -> (MLXArray, MLXArray) {
        // Scores in fp32 (official computes the gate in float32 for stable routing).
        let logits = x.asType(.float32).matmul(weight.asType(.float32).T)  // [B, S, n_experts]
        let scores: MLXArray
        switch scoringFunc {
        case "softmax": scores = softmax(logits, axis: -1)
        case "sigmoid": scores = sigmoid(logits)
        default:        scores = sqrtSoftplus(logits)   // sqrt(softplus(x))
        }

        let inds: MLXArray
        if isHash, let ids = inputIds {
            // Fixed per-token-ID expert selection: tid2eid[input_ids] → [B, S, topk].
            inds = take(tid2eid, ids, axis: 0).asType(.int32)
        } else {
            // Bias-shifted scores for top-k selection (bias not applied to routing weights).
            let scoresForChoice = scores + e_score_correction_bias
            inds = argPartition(-scoresForChoice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        }

        // Routing weights from the original (non-biased) scores at the selected experts.
        var selectedScores = takeAlong(scores, inds, axis: -1)
        // Official normalizes whenever score_func != softmax (config has norm_topk_prob=true).
        if topK > 1 && (normTopkProb || scoringFunc != "softmax") {
            selectedScores = selectedScores / (selectedScores.sum(axis: -1, keepDims: true) + 1e-20)
        }
        selectedScores = (selectedScores * routedScalingFactor).asType(x.dtype)

        return (inds, selectedScores)
    }
}

/// Mixture-of-Experts layer with shared expert.
class DeepseekV4MoE: Module, UnaryLayer {
    let numExpertsPerTok: Int

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    var gate: DeepseekV4Gate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV4Expert

    init(config: DeepseekV4Configuration, layerIdx: Int = 999) {
        self.numExpertsPerTok = config.numExpertsPerTok

        // Routed experts (stacked via SwitchGLU, same as V3)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts,
            activation: { x in
                // SwiGLU with limit
                if config.swiguLimit > 0 {
                    let g = clip(x, min: -config.swiguLimit, max: config.swiguLimit)
                    return silu(g)
                }
                return silu(x)
            }
        )
        self.gate = DeepseekV4Gate(config: config, layerIdx: layerIdx)

        // Shared expert (1 expert, same intermediate size)
        self._sharedExperts.wrappedValue = DeepseekV4Expert(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.moeIntermediateSize,
            swiguLimit: config.swiguLimit
        )
    }

    // UnaryLayer conformance — score-routed path (non-hash layers / callers without token IDs).
    func callAsFunction(_ x: MLXArray) -> MLXArray { routed(x, inputIds: nil) }

    /// MoE forward. `inputIds` (`[B, S]`) is required for hash-routing layers.
    func routed(_ x: MLXArray, inputIds: MLXArray?) -> MLXArray {
        let (indices, scores) = gate(x, inputIds: inputIds)
        var y = switchMLP(x, indices)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)

        // Add shared expert output
        y = y + sharedExperts(x)
        return y
    }
}

// MARK: - Decoder Block (with mHC Hyper-Connections)

public class DeepseekV4Block: Module {
    let config: DeepseekV4Configuration

    // Key "attn" matches checkpoint path `layers.{l}.attn.*`
    @ModuleInfo(key: "attn") var selfAttn: DeepseekV4Attention
    // Plain var: property name "ffn" matches checkpoint path `layers.{l}.ffn.*`
    var ffn: DeepseekV4MoE
    // Key names match checkpoint: `attn_norm`, `ffn_norm`
    @ModuleInfo(key: "attn_norm") var attnNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    // Hyper-Connection parameter bundles.
    // Underscore names match checkpoint paths: `hc_attn.fn/base/scale`, `hc_ffn.fn/base/scale`
    var hc_attn: HCParams
    var hc_ffn: HCParams

    init(config: DeepseekV4Configuration, layerIdx: Int = 0) {
        self.config = config

        self._selfAttn.wrappedValue = DeepseekV4Attention(config: config, layerIdx: layerIdx)
        self.ffn = DeepseekV4MoE(config: config, layerIdx: layerIdx)

        self._attnNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._ffnNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Initialize HC parameters (will be overwritten by weight loading)
        let hc = config.hcMult
        let mixHc = (2 + hc) * hc
        let hcDim = hc * config.hiddenSize
        self.hc_attn = HCParams(
            fn: zeros([mixHc, hcDim]),
            base: zeros([mixHc]),
            scale: ones([3]))
        self.hc_ffn = HCParams(
            fn: zeros([mixHc, hcDim]),
            base: zeros([mixHc]),
            scale: ones([3]))
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        inputIds: MLXArray? = nil
    ) -> MLXArray {
        // x: [B, S, hc, D]
        let residualAttn = x

        // HC pre for attention: [B,S,hc,D] → [B,S,D]
        let (xAttn, postAttn, combAttn) = hcPre(
            x: residualAttn,
            hcFn: hc_attn.fn,
            hcScale: hc_attn.scale,
            hcBase: hc_attn.base,
            hcMult: config.hcMult,
            sinkhornIters: config.hcSinkhornIters,
            eps: config.hcEps
        )

        // Attention sublayer: [B,S,D] → [B,S,D]
        let attnOut = selfAttn(attnNorm(xAttn), mask: mask, cache: cache)

        // HC post for attention: [B,S,D] → [B,S,hc,D]
        let residualFfn = hcPost(x: attnOut, residual: residualAttn, post: postAttn, comb: combAttn)

        // HC pre for FFN: [B,S,hc,D] → [B,S,D]
        let (xFfn, postFfn, combFfn) = hcPre(
            x: residualFfn,
            hcFn: hc_ffn.fn,
            hcScale: hc_ffn.scale,
            hcBase: hc_ffn.base,
            hcMult: config.hcMult,
            sinkhornIters: config.hcSinkhornIters,
            eps: config.hcEps
        )

        // FFN sublayer: [B,S,D] → [B,S,D] (inputIds drives hash routing on layers 0..<numHashLayers)
        let ffnOut = ffn.routed(ffnNorm(xFfn), inputIds: inputIds)

        // HC post for FFN: [B,S,D] → [B,S,hc,D]
        return hcPost(x: ffnOut, residual: residualFfn, post: postFfn, comb: combFfn)
    }
}

// MARK: - Inner Model

public class DeepseekV4ModelInner: Module, LayerPartitionable, StreamableMoE {
    var config: DeepseekV4Configuration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DeepseekV4Block]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    // HC head parameter bundle for final reduction [B,S,hc,D] → [B,S,D]
    // Underscore name matches checkpoint path `model.hc_head.fn/base/scale`
    var hc_head: HCParams

    public var gpuLayerCount: Int? = nil
    public var streamExperts: Bool = false
    // Main forward runs all `num_hidden_layers` real layers. In DeepSeek-V4 `num_hidden_layers`
    // counts ONLY the main transformer layers (43 here, indices 0..42); MTP layers, when present,
    // are ADDITIONAL at indices >= num_hidden_layers. The mlx-community checkpoint ships no MTP
    // weights, so we must NOT subtract num_nextn_predict_layers (that dropped real layer 42).
    public var totalLayerCount: Int { config.numHiddenLayers }

    init(config: DeepseekV4Configuration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        let retainMTP = MTPConfig.retainMTPWeights && config.numNextnPredictLayers > 0
        // All real layers, plus optional trailing MTP blocks only when explicitly retained.
        let totalCount = config.numHiddenLayers + (retainMTP ? config.numNextnPredictLayers : 0)
        self.layers = (0 ..< totalCount).map {
            DeepseekV4Block(config: config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // HC head parameters (will be overwritten by weight loading)
        let hc = config.hcMult
        self.hc_head = HCParams(
            fn: zeros([hc, hc * config.hiddenSize]),
            base: zeros([hc]),
            scale: ones([1]))
    }

    func callAsFunction(_ x: MLXArray, cache: [KVCache]?) -> MLXArray {
        // x: [B, S] token IDs
        let B = x.dim(0), S = x.dim(1)
        let hc = config.hcMult

        // Embed tokens: [B, S, D]
        var h = embedTokens(x)

        // Expand to hc copies: [B, S, hc, D]
        // Repeat along new hc dimension
        h = h.expandedDimensions(axis: 2)                  // [B, S, 1, D]
        h = repeated(h, count: hc, axis: 2)                // [B, S, hc, D]

        // Create causal attention mask; reshape to 3D so dim(1)==S
        let hForMask = h.reshaped([B, S, hc * config.hiddenSize])  // [B, S, hc*D]
        let attentionMask = createAttentionMask(h: hForMask, cache: cache?.first)

        for (i, layer) in layers.prefix(totalLayerCount).enumerated() {
            h = partitionedLayerCall(
                index: i, gpuLayerCount: gpuLayerCount, stream: streamExperts
            ) {
                // Pass token IDs so hash-routing layers (0..<numHashLayers) can index tid2eid.
                layer(h, mask: attentionMask, cache: cache?[i], inputIds: x)
            }
        }

        // HC head: [B, S, hc, D] → [B, S, D]
        h = hcHead(
            x: h, hcFn: hc_head.fn, hcScale: hc_head.scale,
            hcBase: hc_head.base, eps: config.hcEps)

        return norm(h)
    }
}

// MARK: - Top-level Model

public class DeepseekV4Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    /// One KV head per layer (unified KV, single head)
    public var kvHeads: [Int]

    var args: DeepseekV4Configuration
    public var model: DeepseekV4ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: DeepseekV4Configuration) {
        self.args = args
        self.kvHeads = Array(repeating: 1, count: args.numHiddenLayers)
        self.model = DeepseekV4ModelInner(config: args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead(out)
    }

    /// Per-layer caches.
    /// Default (DSA on): DSAKVCache — sliding window of `sliding_window` raw
    /// tokens + learned compressed slots, the structure DSV4 was trained with.
    /// K==V is stored once by construction. `--ctx-size` (maxKVSize) is ignored
    /// for these layers: the window is architectural and compressed slots grow
    /// at tokens/ratio (tiny).
    /// Fallback (MLX_DSA=0): dense caches, SharedKVCache-wrapped for k==v dedup.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        if dsaEnabled {
            return (0 ..< kvHeads.count).map { i in
                let ratio = i < args.compressRatios.count ? args.compressRatios[i] : 0
                return DSAKVCache(windowSize: args.slidingWindow, ratio: ratio)
            }
        }
        let base: [KVCache]
        if let maxKVSize = parameters?.maxKVSize {
            base = (0 ..< kvHeads.count).map { _ in RotatingKVCache(maxSize: maxKVSize, keep: 4) }
        } else {
            base = (0 ..< kvHeads.count).map { _ in KVCacheSimple() }
        }
        guard sharedKVEnabled else { return base }
        return base.map { SharedKVCache(wrapping: $0) }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = weights

        // 1. Dequantize FP8 weights (weight_scale_inv pattern, same as V3)
        func dequant(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
            let bs = 128
            let (m, n) = (weight.dim(0), weight.dim(1))
            let padBottom = (bs - m % bs) % bs
            let padSide = (bs - n % bs) % bs
            var padded = MLX.padded(weight, widths: [.init((0, padBottom)), .init((0, padSide))])
            padded = padded.reshaped([(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
            let scaled = padded * scaleInv[0..., .newAxis, 0..., .newAxis]
            return scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
        }

        for (key, value) in weights {
            if key.contains("weight_scale_inv") {
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = weights[weightKey] {
                    newWeights[weightKey] = dequant(weight: weight, scaleInv: value)
                }
            } else if newWeights[key] == nil {
                newWeights[key] = value
            }
        }

        // 2. Stack per-expert weights into SwitchGLU format (for non-pre-stacked checkpoints)
        // MLX quantized checkpoints already have stacked weights; this is a no-op for them.
        let mainLayerCount = args.numHiddenLayers
        for l in 0 ..< mainLayerCount {
            let prefix = "model.layers.\(l)"
            for projName in ["gate_proj", "down_proj", "up_proj"] {
                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).ffn.experts.0.\(projName).\(key)"
                    if weights[firstKey] != nil {
                        let stacked = (0 ..< args.nRoutedExperts).map {
                            // Prefer dequantized value from newWeights (FP8 dequant), fall back to original
                            newWeights["\(prefix).ffn.experts.\($0).\(projName).\(key)"]
                                ?? weights["\(prefix).ffn.experts.\($0).\(projName).\(key)"]!
                        }
                        newWeights["\(prefix).ffn.switch_mlp.\(projName).\(key)"] = MLX.stacked(stacked)
                        for j in 0 ..< args.nRoutedExperts {
                            newWeights.removeValue(forKey: "\(prefix).ffn.experts.\(j).\(projName).\(key)")
                        }
                    }
                }
            }
        }

        // 3. Filter out MTP (multi-token prediction) layers and rotary_emb keys
        // Also drop compressor/indexer sub-module keys (not yet implemented).
        // MTP layers (if any) live at indices >= num_hidden_layers; real layers are 0..<num_hidden_layers.
        let numMainLayers = args.numHiddenLayers
        var finalWeights = [String: MLXArray]()
        for (key, value) in newWeights {
            // Drop rotary embedding precomputed frequencies
            if key.contains("rotary_emb.inv_freq") { continue }
            // Compressor weights are consumed by DeepseekV4Compressor (DSA path);
            // drop them only when DSA is disabled. The indexer (top-k slot
            // selection) remains unimplemented — attending ALL compressed slots
            // is exact, just unpruned, so dropping indexer weights is lossless.
            if !dsaEnabled && key.contains(".attn.compressor.") { continue }
            if key.contains(".attn.indexer.") { continue }
            // Keep ffn.gate.tid2eid — consumed by hash-routing layers (0..<numHashLayers).

            if key.starts(with: "model.layers.") {
                let parts = key.split(separator: ".")
                if parts.count >= 3, let layerIdx = Int(parts[2]) {
                    if layerIdx >= numMainLayers && !MTPConfig.retainMTPWeights {
                        continue
                    }
                }
            }
            // 4. Key-name compat: official DeepSeek-V4 checkpoints (mlx-community)
            // name the per-layer Hyper-Connection bundles `attn_hc` / `ffn_hc`;
            // our block properties are `hc_attn` / `hc_ffn`. Remap so both naming
            // conventions load. (`model.hc_head.*` already matches.)
            var outKey = key
            if outKey.contains(".attn_hc.") {
                outKey = outKey.replacingOccurrences(of: ".attn_hc.", with: ".hc_attn.")
            } else if outKey.contains(".ffn_hc.") {
                outKey = outKey.replacingOccurrences(of: ".ffn_hc.", with: ".hc_ffn.")
            }
            finalWeights[outKey] = value
        }
        return finalWeights
    }

    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - MTPLanguageModel Conformance for DeepseekV4Model

/// DeepSeek V4 uses a different MTP scheme: the MTP layers are the last
/// `numNextnPredictLayers` standard transformer blocks (`model.layers[numMainLayers...]`).
/// They share the same architecture as the main blocks but operate on the final hidden state.
/// The main `lm_head` is reused for all MTP depth projections.
extension DeepseekV4Model: MTPLanguageModel {
    public func callMTP(_ inputs: MLXArray, cache: [KVCache]?, mtpCaches: [[KVCache]]?) -> [MLXArray] {
        let mtpLayers = model.layers.suffix(args.numNextnPredictLayers)
        guard MTPConfig.retainMTPWeights, !mtpLayers.isEmpty else {
            return [callAsFunction(inputs, cache: cache)]
        }

        // Run the main model body (excludes MTP layers \u2014 DeepseekV4ModelInner only
        // instantiates `numMain` blocks, so this is the standard forward pass)
        let mainHidden = model(inputs, cache: cache)
        let mainLogits = lmHead(mainHidden)
        var result = [mainLogits]

        // Chain MTP blocks stored in `model.mtpLayers`
        var prevHidden = mainHidden
        let B = prevHidden.dim(0), S = prevHidden.dim(1)
        let hc = args.hcMult
        for (i, mtpLayer) in mtpLayers.enumerated() {
            let mtpCache = mtpCaches?[i]
            // Expand [B, S, D] -> [B, S, hc, D]
            var h = prevHidden.expandedDimensions(axis: 2)
            h = repeated(h, count: hc, axis: 2)

            let hForMask = h.reshaped([B, S, hc * args.hiddenSize])
            let attentionMask = createAttentionMask(h: hForMask, cache: mtpCache?.first)
            
            h = mtpLayer(h, mask: attentionMask, cache: mtpCache?.first)
            
            // Reduce back to [B, S, D]
            prevHidden = hcHead(
                x: h, hcFn: model.hc_head.fn, hcScale: model.hc_head.scale,
                hcBase: model.hc_head.base, eps: args.hcEps)
                
            let mtpLogits = lmHead(model.norm(prevHidden))
            result.append(mtpLogits)
        }

        return result
    }

    public func makeMTPCaches(parameters: GenerateParameters?) -> [[KVCache]] {
        return (0 ..< args.numNextnPredictLayers).map { _ in
            [KVCacheSimple()]
        }
    }
}
