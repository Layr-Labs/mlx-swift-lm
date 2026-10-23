// Copyright © 2024-2026 Jinho Jang (eric@jangq.ai)
//
// Gemma 4 VLM — vision-language model with:
//   - Linear patch embedding with 2D position embeddings
//   - 2D multidimensional RoPE for vision encoder
//   - VisionPooler for downsampling patches
//   - MultimodalEmbedder projecting vision features into text space
//   - Full Gemma4 text decoder (MoE 26B or Dense 31B)
//
// Python reference: mlx_vlm/models/gemma4/

import CoreImage
import CoreMedia
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// Darkbloom note: vision and projector components are ported from vMLX's
// Gemma4 VLM, with RuntimeMoETopKOverride, audio, and cache-scope integration
// omitted because those surfaces do not exist in this LMInput integration.
// Language work delegates to MLXLLM's canonical Gemma4TextModel, preserving
// its CBv2 and MTP/capture surfaces directly.

// Local replacement for vMLX's `QwenVL.intExtent` (absent in our tree). Rejects
// zero-area / non-finite extents so the scale-factor math below cannot divide by
// zero or trap inside `Int(floor(.nan))` / `Int(.infinity)`.
private func gemma4IntExtent(_ size: CGSize) throws -> (Int, Int) {
    let w = size.width
    let h = size.height
    guard w.isFinite, h.isFinite, w > 0, h > 0 else {
        throw VLMError.imageProcessingFailure(
            "Gemma4: image has a zero-area or non-finite extent (w=\(w), h=\(h)).")
    }
    return (Int(h), Int(w))
}

/// Vision RMSNorm — full float32 computation for precision
private class VisionRMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float
    let fused: Bool
    let usePower: Bool
    init(dimensions: Int, eps: Float = 1e-6, fused: Bool = false, usePower: Bool = false) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        self.fused = fused
        self.usePower = usePower
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if fused { return MLXFast.rmsNorm(x, weight: weight, eps: eps) }
        let xf = x.asType(.float32)
        let v = (usePower ? DiffusionGemmaVisionMath.square(xf) : xf * xf).mean(
            axis: -1, keepDims: true)
        return ((xf * rsqrt(v + eps)) * weight.asType(.float32)).asType(x.dtype)
    }
}

/// Parameterless RMS normalization
func rmsNormNoScale(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
}

private func visionRmsNormNoScale(_ x: MLXArray, eps: Float = 1e-6, usePower: Bool = false)
    -> MLXArray
{
    let xf = x.asType(.float32)
    let v = (usePower ? DiffusionGemmaVisionMath.square(xf) : xf * xf).mean(
        axis: -1, keepDims: true)
    return (xf * rsqrt(v + eps)).asType(x.dtype)
}

// MARK: - Configurations

/// Keep the existing Gemma4 path unchanged. DiffusionGemma's pinned vision
/// reference has different normalization/SDPA/grid contracts despite sharing
/// the same checkpoint parameter names.
enum Gemma4VisionContract {
    case gemma4
    case diffusionGemma
}

public struct Gemma4VisionConfig: Codable, Sendable {
    let hiddenSize: Int
    let intermediateSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let rmsNormEps: Float
    let patchSize: Int
    let positionEmbeddingSize: Int
    let defaultOutputLength: Int
    let poolingKernelSize: Int
    let standardize: Bool
    let useClippedLinears: Bool
    let ropeTheta: Float

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case patchSize = "patch_size"
        case positionEmbeddingSize = "position_embedding_size"
        case defaultOutputLength = "default_output_length"
        case poolingKernelSize = "pooling_kernel_size"
        case standardize
        case useClippedLinears = "use_clipped_linears"
    }

    enum TopKeys: String, CodingKey {
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 16
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 12
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 12
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        positionEmbeddingSize =
            try c.decodeIfPresent(Int.self, forKey: .positionEmbeddingSize) ?? 10240
        defaultOutputLength = try c.decodeIfPresent(Int.self, forKey: .defaultOutputLength) ?? 280
        poolingKernelSize = try c.decodeIfPresent(Int.self, forKey: .poolingKernelSize) ?? 3
        standardize = try c.decodeIfPresent(Bool.self, forKey: .standardize) ?? false
        useClippedLinears = try c.decodeIfPresent(Bool.self, forKey: .useClippedLinears) ?? false

        if let rc = try? decoder.container(keyedBy: TopKeys.self),
            let rp = try? rc.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeParameters),
            let t = rp["rope_theta"]?.asFloat()
        {
            ropeTheta = t
        } else {
            ropeTheta = 100.0
        }
    }
}

public struct Gemma4Configuration: Codable, Sendable {
    let textConfig: MLXLLM.Gemma4TextConfiguration
    let visionConfig: Gemma4VisionConfig
    let modelType: String
    let imageTokenId: Int
    let videoTokenId: Int?
    let visionSoftTokensPerImage: Int
    let quantization: BaseConfiguration.Quantization?

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case modelType = "model_type"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case quantization
        case quantizationConfig = "quantization_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let rootBaseConfiguration = try? BaseConfiguration(from: decoder)
        let rootPerLayerQuantization = rootBaseConfiguration?.perLayerQuantization
        let decodedRootQuantization =
            try c.decodeIfPresent(
                BaseConfiguration.Quantization.self, forKey: .quantization)
            ?? c.decodeIfPresent(
                BaseConfiguration.Quantization.self, forKey: .quantizationConfig)
        let rootQuantization =
            rootPerLayerQuantization?.quantization
            ?? decodedRootQuantization
        var textConfig = try MLXLLM.Gemma4TextConfiguration(
            from: c.superDecoder(forKey: .textConfig),
            defaults: .visionLanguageModel)
        textConfig.mergeQuantization(rootQuantization)
        textConfig.mergeQuantization(rootPerLayerQuantization)
        self.textConfig = textConfig
        visionConfig = try c.decode(Gemma4VisionConfig.self, forKey: .visionConfig)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4"
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 258_880
        // Default to the Gemma4 video token id so the model stays in sync with the
        // processor (which emits video placeholders) even when config.json omits it.
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 258_884
        visionSoftTokensPerImage =
            try c.decodeIfPresent(Int.self, forKey: .visionSoftTokensPerImage)
            ?? visionConfig.defaultOutputLength
        quantization = rootQuantization
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(textConfig, forKey: .textConfig)
        try c.encode(visionConfig, forKey: .visionConfig)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(imageTokenId, forKey: .imageTokenId)
        try c.encodeIfPresent(videoTokenId, forKey: .videoTokenId)
        try c.encode(visionSoftTokensPerImage, forKey: .visionSoftTokensPerImage)
        if let quantization {
            let overrides = textConfig.perLayerQuantization?.perLayerQuantization ?? [:]
            if overrides.isEmpty {
                try c.encode(quantization, forKey: .quantization)
            } else {
                try c.encode(
                    Gemma4RootQuantizationConfiguration(
                        fallback: quantization, overrides: overrides),
                    forKey: .quantization)
            }
        }
    }
}

private struct Gemma4RootQuantizationConfiguration: Encodable {
    let fallback: BaseConfiguration.Quantization
    let overrides: [String: BaseConfiguration.QuantizationOption]

    private struct DynamicKey: CodingKey {
        let stringValue: String
        let intValue: Int? = nil
        init(stringValue: String) { self.stringValue = stringValue }
        init(intValue: Int) { self.stringValue = "\(intValue)" }
    }

    func encode(to encoder: Encoder) throws {
        try fallback.encode(to: encoder)
        var container = encoder.container(keyedBy: DynamicKey.self)
        for (path, option) in overrides {
            switch option {
            case .skip:
                try container.encode(false, forKey: DynamicKey(stringValue: path))
            case .quantize(let quantization):
                try container.encode(quantization, forKey: DynamicKey(stringValue: path))
            }
        }
    }
}

// MARK: - Vision Components

private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
}

private func applyMultidimensionalRope(
    _ inputs: MLXArray, positions: MLXArray, base: Float,
    contract: Gemma4VisionContract = .gemma4
) -> MLXArray {
    let headDim = inputs.dim(-1)
    let ndim = positions.dim(-1)
    let chPerDim = 2 * (headDim / (2 * ndim))
    let halfPerDim = chPerDim / 2

    var parts: [MLXArray] = []
    for d in 0 ..< ndim {
        let xPart = inputs[.ellipsis, (d * chPerDim) ..< ((d + 1) * chPerDim)]
        let freqExp = (2.0 / Float(chPerDim)) * MLXArray(0 ..< halfPerDim).asType(.float32)
        let timescale =
            contract == .diffusionGemma
            ? DiffusionGemmaVisionMath.positivePower(base: base, exponents: freqExp)
            : pow(base, freqExp)
        let sinInp = positions[.ellipsis, d ..< (d + 1)].asType(.float32) / timescale
        var cosD = cos(sinInp)
        var sinD = sin(sinInp)
        cosD = concatenated([cosD, cosD], axis: -1).asType(inputs.dtype)
        sinD = concatenated([sinD, sinD], axis: -1).asType(inputs.dtype)
        cosD = expandedDimensions(cosD, axis: 2)
        sinD = expandedDimensions(sinD, axis: 2)
        parts.append(xPart * cosD + rotateHalf(xPart) * sinD)
    }
    return concatenated(parts, axis: -1)
}

private func oneHot(_ indices: MLXArray, numClasses: Int) -> MLXArray {
    (expandedDimensions(indices, axis: -1) .== MLXArray(0 ..< Int32(numClasses))).asType(.float32)
}

// Vision Attention
private class VisionAttn: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let ropeBase: Float
    let contract: Gemma4VisionContract

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: VisionRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: VisionRMSNorm

    init(_ cfg: Gemma4VisionConfig, contract: Gemma4VisionContract = .gemma4) {
        self.contract = contract
        numHeads = cfg.numAttentionHeads
        numKVHeads = cfg.numKeyValueHeads
        headDim = cfg.headDim
        ropeBase = cfg.ropeTheta
        _qProj.wrappedValue = Linear(cfg.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(cfg.hiddenSize, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(cfg.hiddenSize, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, cfg.hiddenSize, bias: false)
        _qNorm.wrappedValue = VisionRMSNorm(
            dimensions: headDim, usePower: contract == .diffusionGemma)
        _kNorm.wrappedValue = VisionRMSNorm(
            dimensions: headDim, usePower: contract == .diffusionGemma)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, positions: MLXArray, mask: MLXArray?,
        trace: ((String, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var q = qProj(x).reshaped(B, L, numHeads, headDim)
        var k = kProj(x).reshaped(B, L, numKVHeads, headDim)
        var v = vProj(x).reshaped(B, L, numKVHeads, headDim)
        trace?("qRaw", q)
        trace?("kRaw", k)
        trace?("vRaw", v)
        if let trace {
            let squared = DiffusionGemmaVisionMath.square(q)
            let mean = squared.mean(axis: -1, keepDims: true)
            trace("qSquared", squared)
            trace("qMean", mean)
            trace("qInv", rsqrt(mean + Float(1e-6)))
        }
        q = qNorm(q)
        k = kNorm(k)
        v = visionRmsNormNoScale(v, usePower: contract == .diffusionGemma)
        trace?("qNorm", q)
        trace?("kNorm", k)
        trace?("vNorm", v)
        q = applyMultidimensionalRope(q, positions: positions, base: ropeBase, contract: contract)
        k = applyMultidimensionalRope(k, positions: positions, base: ropeBase, contract: contract)
        trace?("qRope", q)
        trace?("kRope", k)
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)
        // vmlx #52: Gemma 4 vision tower weights are float16 and attention
        // scores can exceed ±65504, producing -inf → NaN propagation through
        // embed_vision → model emits only <pad> tokens. Promote Q/K/V to
        // float32 for the SDPA, then cast back. Mirrors the Python
        // v1.3.29 patch.
        let origDType = q.dtype
        if origDType == .float16 && contract == .gemma4 {
            q = q.asType(.float32)
            k = k.asType(.float32)
            v = v.asType(.float32)
        }
        if contract == .diffusionGemma {
            // Same zero-padding as MLX-VLM ensure_fused_sdpa; the released
            // 72-wide heads use the 80-wide fused reduction, then slice back.
            let target = [64, 80, 128].first { headDim <= $0 } ?? headDim
            if target != headDim {
                let widths: [IntOrPair] = [0, 0, 0, [0, target - headDim]]
                q = padded(q, widths: widths)
                k = padded(k, widths: widths)
                v = padded(v, widths: widths)
            }
        }
        var out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0,
            mask: mask != nil ? .array(mask!) : .none)
        if contract == .diffusionGemma { out = out[.ellipsis, ..<headDim] }
        if origDType == .float16 && contract == .gemma4 {
            out = out.asType(.float16)
        }
        trace?("attended", out)
        let projected = oProj(out.transposed(0, 2, 1, 3).reshaped(B, L, -1))
        trace?("attention", projected)
        return projected
    }
}

private class VisionMLP: Module {
    let contract: Gemma4VisionContract
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    init(_ cfg: Gemma4VisionConfig, contract: Gemma4VisionContract = .gemma4) {
        self.contract = contract
        _gateProj.wrappedValue = Linear(cfg.hiddenSize, cfg.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(cfg.hiddenSize, cfg.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(cfg.intermediateSize, cfg.hiddenSize, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, trace: ((String, MLXArray) -> Void)? = nil) -> MLXArray {
        let gate = gateProj(x)
        let activated =
            contract == .diffusionGemma ? geluApproximate(gate) : safeGeluApproximate(gate)
        let up = upProj(x)
        trace?("gate", gate)
        trace?("up", up)
        trace?("activation", activated)
        let down = downProj(activated * up)
        trace?("down", down)
        return down
    }
}

private class VisionBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: VisionAttn
    @ModuleInfo var mlp: VisionMLP
    @ModuleInfo(key: "input_layernorm") var inputLN: VisionRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttnLN: VisionRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFFLN: VisionRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFFLN: VisionRMSNorm

    init(_ cfg: Gemma4VisionConfig, contract: Gemma4VisionContract = .gemma4) {
        _selfAttn.wrappedValue = VisionAttn(cfg, contract: contract)
        self.mlp = VisionMLP(cfg, contract: contract)
        let fused = contract == .diffusionGemma
        _inputLN.wrappedValue = VisionRMSNorm(
            dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps, fused: fused)
        _postAttnLN.wrappedValue = VisionRMSNorm(
            dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps, fused: fused)
        _preFFLN.wrappedValue = VisionRMSNorm(
            dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps, fused: fused)
        _postFFLN.wrappedValue = VisionRMSNorm(
            dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps, fused: fused)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, positions: MLXArray, mask: MLXArray?,
        trace: ((String, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        let normalized = inputLN(x)
        trace?("inputNorm", normalized)
        let post = postAttnLN(selfAttn(normalized, positions: positions, mask: mask, trace: trace))
        trace?("postAttention", post)
        var h = x + post
        trace?("residual", h)
        let pre = preFFLN(h)
        trace?("preFFN", pre)
        h = h + postFFLN(mlp(pre, trace: trace))
        trace?("output", h)
        return h
    }
}

private class VisionPatchEmbedder: Module {
    let patchSize: Int
    let posEmbSize: Int
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ModuleInfo(key: "position_embedding_table") var posTable: MLXArray

    init(_ cfg: Gemma4VisionConfig) {
        patchSize = cfg.patchSize
        posEmbSize = cfg.positionEmbeddingSize
        _inputProj.wrappedValue = Linear(
            3 * cfg.patchSize * cfg.patchSize, cfg.hiddenSize, bias: false)
        _posTable.wrappedValue = MLXArray.ones([2, cfg.positionEmbeddingSize, cfg.hiddenSize])
        super.init()
    }

    func callAsFunction(pixels: MLXArray, patchPos: MLXArray, padPos: MLXArray) -> MLXArray {
        let (B, C, H, W) = (pixels.dim(0), pixels.dim(1), pixels.dim(2), pixels.dim(3))
        let p = patchSize
        let patches = pixels.reshaped(B, C, H / p, p, W / p, p)
            .transposed(0, 2, 4, 3, 5, 1).reshaped(B, (H / p) * (W / p), C * p * p)
        let normalized = 2 * (patches - 0.5)
        let embedded = inputProj(normalized.asType(inputProj.weight.dtype))

        let oh = oneHot(patchPos, numClasses: posEmbSize)
            .transposed(0, 2, 1, 3).asType(posTable.dtype)
        var posEmb = matmul(oh, posTable).sum(axis: 1)
        posEmb = MLX.where(
            expandedDimensions(padPos, axis: -1), MLXArray(Float(0), dtype: posEmb.dtype), posEmb)
        return embedded + posEmb
    }
}

private class VisionPooler: Module {
    let defaultLen: Int
    let rootH: Float
    init(_ cfg: Gemma4VisionConfig) {
        defaultLen = cfg.defaultOutputLength
        rootH = sqrt(Float(cfg.hiddenSize))
        super.init()
    }
    /// Pool `h` down to `outputLen` soft tokens. Images use the default (image)
    /// budget; video frames pass a smaller per-frame budget (e.g. 70 vs 280) so a
    /// frame resized to the video patch budget pools to its own token count.
    func callAsFunction(
        _ h: MLXArray, patchPos: MLXArray, padPos: MLXArray, outputLen: Int? = nil
    ) -> (MLXArray, MLXArray) {
        let outLen = outputLen ?? defaultLen
        let L = h.dim(1)
        if L == outLen { return (h * rootH, logicalNot(padPos)) }
        let k = Int(sqrt(Float(L / outLen)))
        let kSq = Float(k * k)
        let clamped = maximum(patchPos, MLXArray(Int32(0)))
        let maxX = clamped[.ellipsis, 0].max(axis: -1, keepDims: true) + 1
        let ki = floor(clamped.asType(.float32) / Float(k)).asType(.int32)
        let linearIdx = ki[.ellipsis, 0] + (maxX / MLXArray(Int32(k))) * ki[.ellipsis, 1]
        let w = oneHot(linearIdx, numClasses: outLen) / kSq
        let out = matmul(w.transposed(0, 2, 1), h)
        let mask = logicalNot(all(w .== Float(0), axis: 1))
        return (out.asType(h.dtype) * rootH, mask)
    }
}

private class VisionEncoder: Module {
    @ModuleInfo var layers: [VisionBlock]
    init(_ cfg: Gemma4VisionConfig, contract: Gemma4VisionContract = .gemma4) {
        _layers.wrappedValue = (0 ..< cfg.numHiddenLayers).map { _ in
            VisionBlock(cfg, contract: contract)
        }
        super.init()
    }
    func callAsFunction(
        _ x: MLXArray, pos: MLXArray, mask: MLXArray?,
        trace: ((String, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        var h = x
        for (index, layer) in layers.enumerated() {
            let observe: ((String, MLXArray) -> Void)? = trace.map { callback in
                { name, array in callback("block\(index)." + name, array) }
            }
            h = layer(h, positions: pos, mask: mask, trace: observe)
        }
        return h
    }
}

// Shared tower implementation; architecture-specific projectors remain separate.
class Gemma4VisionTower: Module {
    let cfg: Gemma4VisionConfig
    let contract: Gemma4VisionContract
    @ModuleInfo(key: "patch_embedder") private var patchEmb: VisionPatchEmbedder
    @ModuleInfo private var encoder: VisionEncoder
    @ModuleInfo private var pooler: VisionPooler
    @ModuleInfo(key: "std_bias") var stdBias: MLXArray?
    @ModuleInfo(key: "std_scale") var stdScale: MLXArray?

    init(_ cfg: Gemma4VisionConfig, contract: Gemma4VisionContract = .gemma4) {
        self.cfg = cfg
        self.contract = contract
        _patchEmb.wrappedValue = VisionPatchEmbedder(cfg)
        self.encoder = VisionEncoder(cfg, contract: contract)
        self.pooler = VisionPooler(cfg)
        if cfg.standardize {
            _stdBias.wrappedValue = MLXArray.zeros([cfg.hiddenSize])
            _stdScale.wrappedValue = MLXArray.ones([cfg.hiddenSize])
        }
        super.init()
    }

    /// Encode one image / video frame to `outputLength` soft tokens. `outputLength`
    /// defaults to the image budget (`defaultOutputLength`); video frames pass a
    /// smaller per-frame budget so the local patch budget (`outputLength * pool^2`)
    /// and the pooler output both shrink to the trained video-frame representation.
    func callAsFunction(
        _ pixels: MLXArray, outputLength: Int? = nil,
        trace: ((String, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        if contract == .diffusionGemma {
            return diffusionFeatures(pixels, outputLength: outputLength, trace: trace)
        }
        let (B, _, H, W) = (pixels.dim(0), pixels.dim(1), pixels.dim(2), pixels.dim(3))
        let outLen = outputLength ?? cfg.defaultOutputLength
        let localMaxPatches = max(1, outLen) * cfg.poolingKernelSize * cfg.poolingKernelSize
        let p = cfg.patchSize
        var pH = H / p
        let pW = W / p

        // Truncate oversized grids CONSISTENTLY: drop whole trailing rows so the
        // patch embedding (built from the same cropped pixels), the position grid,
        // and the pooler all agree on the patch count. Previously `nReal` was
        // clamped to `maxPatches` but `posFlat` was still built for every `pH * pW`
        // patch, so the `reshaped(1, nReal, 2)` below trapped before the clamp could
        // help — and the pixels stayed untruncated, mismatching the clamped
        // positions. Cropping rows keeps `pW` (and the per-row x positions) intact.
        var croppedPixels = pixels
        if pH * pW > localMaxPatches {
            pH = max(1, localMaxPatches / max(1, pW))
            croppedPixels = pixels[0..., 0..., ..<(pH * p), 0...]
        }
        let nReal = pH * pW
        let nPad = localMaxPatches - nReal

        // Build position grid [nReal, 2] then expand to [B, nReal, 2]
        var posFlat = [Int32]()
        posFlat.reserveCapacity(nReal * 2)
        for y in 0 ..< pH {
            for x in 0 ..< pW {
                posFlat.append(Int32(x))
                posFlat.append(Int32(y))
            }
        }
        var patchPos = MLXArray(posFlat).reshaped(1, nReal, 2)
        patchPos = repeated(patchPos, count: B, axis: 0)
        var padPos = MLXArray.zeros([B, localMaxPatches]).asType(.bool)

        if nPad > 0 {
            let padFlat = [Int32](repeating: -1, count: nPad * 2)
            let pp = MLXArray(padFlat).reshaped(1, nPad, 2)
            patchPos = concatenated([patchPos, repeated(pp, count: B, axis: 0)], axis: 1)
            padPos = concatenated(
                [
                    MLXArray.zeros([B, nReal]).asType(.bool),
                    MLXArray.ones([B, nPad]).asType(.bool),
                ], axis: 1)
        }

        var emb = patchEmb(
            pixels: croppedPixels, patchPos: patchPos[0..., ..<nReal],
            padPos: padPos[0..., ..<nReal])
        if nPad > 0 {
            emb = concatenated(
                [emb, MLXArray.zeros([B, nPad, cfg.hiddenSize]).asType(emb.dtype)], axis: 1)
        }

        let valid = logicalNot(padPos).asType(.float32)
        var mask = expandedDimensions(valid, axis: 1) * expandedDimensions(valid, axis: 2)
        let zeroVal = MLXArray(Float(0), dtype: emb.dtype)
        let negInfVal = MLXArray(Float(-1e9), dtype: emb.dtype)
        mask = MLX.where(mask .> MLXArray(Float(0), dtype: mask.dtype), zeroVal, negInfVal)
        mask = expandedDimensions(mask, axis: 1)

        var h = encoder(emb, pos: patchPos, mask: mask)
        let (pooled, _) = pooler(h, patchPos: patchPos, padPos: padPos, outputLen: outLen)
        // Return all `outLen` features — the processor inserts exactly that many
        // image/video soft tokens, so maskedScatter needs them all to match.
        h = pooled
        if cfg.standardize, let sb = stdBias, let ss = stdScale { h = (h - sb) * ss }
        return h
    }

    /// Raw images are already resized to a patch/pool-aligned grid by the
    /// processor. The reference uses the actual grid, not a padded maximum
    /// budget; changing that length also changes attention and pooled features.
    private func diffusionFeatures(
        _ pixels: MLXArray, outputLength: Int?,
        trace: ((String, MLXArray) -> Void)?
    ) -> MLXArray {
        let batch = pixels.dim(0)
        let height = pixels.dim(2) / cfg.patchSize
        let width = pixels.dim(3) / cfg.patchSize
        let count = height * width
        let length = count / (cfg.poolingKernelSize * cfg.poolingKernelSize)
        precondition(
            height.isMultiple(of: cfg.poolingKernelSize)
                && width.isMultiple(of: cfg.poolingKernelSize))
        precondition(length > 0 && (outputLength == nil || outputLength == length))
        var positions = [Int32]()
        for y in 0 ..< height { for x in 0 ..< width { positions += [Int32(x), Int32(y)] } }
        let patchPos = repeated(MLXArray(positions).reshaped(1, count, 2), count: batch, axis: 0)
        let padding = MLXArray.zeros([batch, count], dtype: .bool)
        let embeddings = patchEmb(pixels: pixels, patchPos: patchPos, padPos: padding)
        trace?("embedding", embeddings)
        let mask = MLXArray.zeros([batch, 1, count, count], dtype: embeddings.dtype)
        let hidden = encoder(embeddings, pos: patchPos, mask: mask, trace: trace)
        let (pooled, _) = pooler(hidden, patchPos: patchPos, padPos: padding, outputLen: length)
        var result = pooled.reshaped(1, batch * length, cfg.hiddenSize)
        if cfg.standardize, let bias = stdBias, let scale = stdScale {
            result = (result - bias) * scale
        }
        trace?("features", result)
        return result
    }
}

// MARK: - Multimodal Embedder

private class MultimodalEmbedder: Module {
    @ModuleInfo(key: "embedding_projection") var proj: Linear
    init(embDim: Int, textDim: Int) {
        _proj.wrappedValue = Linear(embDim, textDim, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { rmsNormNoScale(proj(x)) }
}

private func maskedScatter(input: MLXArray, mask: MLXArray, source: MLXArray) throws -> MLXArray {
    let inputShape = input.shape
    let inputFlat = input.flattened()
    let maskFlat = mask.flattened()
    let sourceFlat = source.flattened()

    let maskValues = maskFlat.asArray(Bool.self)
    let positions = maskValues.enumerated().compactMap { i, v in v ? UInt32(i) : nil }

    guard !positions.isEmpty else { return input }

    let posArray = MLXArray(positions)
    // Surface the bundle/processor-config mismatch as a recoverable
    // VLMError instead of an abort. Per `docs/GEMMA4-DEEP-TRACE-2026-05-10.md`
    // §7.3, a `fatalError` here was never reachable cleanly — a
    // mis-stamped `imageSeqLength` would crash the whole process on
    // first image. Throw so the caller (osaurus, JANG Studio, etc.)
    // can surface the diagnostic without process abort.
    guard sourceFlat.shape[0] == posArray.shape[0] else {
        throw VLMError.processing(
            """
            Gemma4 maskedScatter: size mismatch between vision features and image token positions. \
            Vision features: \(sourceFlat.shape[0]), image positions: \(posArray.shape[0]). \
            Check that imageSeqLength in preprocessor_config matches vision tower output (defaultOutputLength).
            """)
    }
    inputFlat[posArray] = sourceFlat
    return inputFlat.reshaped(inputShape)
}

// MARK: - Gemma4 VLM

public class Gemma4: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: Gemma4VisionTower
    @ModuleInfo(key: "language_model") private var languageModel: Gemma4TextModel
    @ModuleInfo(key: "embed_vision") private var embedVision: MultimodalEmbedder

    public let config: Gemma4Configuration
    /// The exact text tower owned by this VLM. Continuous Batching V2 uses
    /// this instance directly so multimodal serving never constructs a second
    /// language module or duplicates the checkpoint's resident weights.
    public var textModel: Gemma4TextModel { languageModel }
    public var vocabularySize: Int { config.textConfig.vocabSize }
    public var kvHeads: [Int] {
        let tc = config.textConfig
        return (0 ..< tc.numHiddenLayers).map { i in
            let lt = i < tc.layerTypes.count ? tc.layerTypes[i] : "sliding_attention"
            return lt == "full_attention"
                ? (tc.numGlobalKeyValueHeads ?? tc.numKeyValueHeads) : tc.numKeyValueHeads
        }
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    public init(_ config: Gemma4Configuration) {
        self.config = config
        _visionTower.wrappedValue = Gemma4VisionTower(config.visionConfig)
        _languageModel.wrappedValue = Gemma4TextModel(config.textConfig)
        _embedVision.wrappedValue = MultimodalEmbedder(
            embDim: config.visionConfig.hiddenSize, textDim: config.textConfig.hiddenSize)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        // Gemma4 VLM does not implement audio. Our `LMInput` carries no audio
        // field, and the `sanitize` path drops `audio_tower.*` / `embed_audio.*`
        // weights, so there is nothing to guard here.
        var emb = languageModel.scaledInputEmbeddings(input.text.tokens)

        // Accumulate the image+video soft-token positions so the text tower can
        // apply Gemma4's blockwise bidirectional attention over those spans. nil
        // keeps the text-only hot path on the symbolic causal mask.
        var visualTokenMask: MLXArray? = nil

        if let pixels = input.image?.pixels {
            let imgFeatures = encodeVisionFeatures(
                pixels: pixels, frames: input.image?.frames, dtype: emb.dtype)
            let imgMask = MLX.equal(input.text.tokens, MLXArray(Int32(config.imageTokenId)))
            let imgMaskExp = MLX.broadcast(expandedDimensions(imgMask, axis: -1), to: emb.shape)
            emb = try maskedScatter(input: emb, mask: imgMaskExp, source: imgFeatures)
            visualTokenMask = imgMask
        }

        // Video frames run through the same vision tower as images; only the
        // placeholder token they scatter into differs (image_token vs video_token).
        if let videoPixels = input.video?.pixels, let videoTokenId = config.videoTokenId {
            let vidFeatures = encodeVisionFeatures(
                pixels: videoPixels, frames: input.video?.frames, dtype: emb.dtype, isVideo: true)
            let vidMask = MLX.equal(input.text.tokens, MLXArray(Int32(videoTokenId)))
            let vidMaskExp = MLX.broadcast(expandedDimensions(vidMask, axis: -1), to: emb.shape)
            emb = try maskedScatter(input: emb, mask: vidMaskExp, source: vidFeatures)
            visualTokenMask = visualTokenMask.map { logicalOr($0, vidMask) } ?? vidMask
        }

        let out = languageModel(
            input.text.tokens, inputEmbedding: emb, cache: castCache(cache),
            imageTokenMask: visualTokenMask)
        // The whole prompt runs in one forward, so the progress sequence is
        // only the terminal report.
        let totalPositions = input.text.tokens.dim(-1)
        prefill.progress?(totalPositions, totalPositions)
        return .logits(.init(logits: out))
    }

    /// Encode one batch of images / video frames through the vision tower and
    /// project into the text embedding space. Each frame is sliced to its real
    /// (un-padded) dimensions stored in `frames`. Images yield `defaultOutputLength`
    /// soft tokens; video frames (`isVideo`) yield a per-frame, data-driven count
    /// (`(h/patch)*(w/patch)/pool^2`) from their video-budget-resized grid, matching
    /// the processor's per-frame placeholder expansion (HF/mlx-vlm Gemma4 use a
    /// smaller per-frame video soft-token budget, ~70, vs 280 for images).
    private func encodeVisionFeatures(
        pixels: MLXArray, frames: [THW]?, dtype: DType, isVideo: Bool = false
    ) -> MLXArray {
        let featuresList = visionFeatureList(pixels: pixels, frames: frames, isVideo: isVideo)
        return (featuresList.count == 1 ? featuresList[0] : concatenated(featuresList))
            .asType(dtype)
    }

    /// Shared tower + projector loop: one `[1, softTokens, textHidden]` array
    /// per image / video frame, in the projector's native dtype (callers cast).
    /// Split out of `encodeVisionFeatures` so the external per-image seam
    /// (`perImageVisionFeatures`) reuses exactly the arrays `prepare` scatters.
    private func visionFeatureList(
        pixels: MLXArray, frames: [THW]?, isVideo: Bool
    ) -> [MLXArray] {
        let B = pixels.dim(0)
        var featuresList = [MLXArray]()
        featuresList.reserveCapacity(B)
        let p = config.visionConfig.patchSize
        let poolSq = config.visionConfig.poolingKernelSize * config.visionConfig.poolingKernelSize
        // All frames of one video share the same resized size, so this per-frame
        // count is uniform within a video and matches the processor's placeholder
        // expansion (both derive it from the same video-budget-resized grid).
        func videoOutputLength(h: Int, w: Int) -> Int {
            max(1, ((h / p) * (w / p)) / max(1, poolSq))
        }
        for i in 0 ..< B {
            // Extract each frame at its original dimensions (stored in frames)
            // to avoid processing zero-padded regions through the vision tower.
            if let frames, i < frames.count {
                let h = frames[i].h
                let w = frames[i].w
                let single = pixels[i, 0..., ..<h, ..<w].expandedDimensions(axis: 0)
                let outLen = isVideo ? videoOutputLength(h: h, w: w) : nil
                featuresList.append(embedVision(visionTower(single, outputLength: outLen)))
            } else {
                let single = pixels[i].expandedDimensions(axis: 0)
                let outLen = isVideo ? videoOutputLength(h: pixels.dim(2), w: pixels.dim(3)) : nil
                featuresList.append(embedVision(visionTower(single, outputLength: outLen)))
            }
        }
        return featuresList
    }

    private func castCache(_ cache: [any KVCache]) -> [KVCache]? {
        guard !cache.isEmpty else { return nil }
        return cache.map { $0 }
    }

    private func castCache(_ cache: [any KVCache]?) -> [KVCache]? {
        guard let cache else { return nil }
        return castCache(cache)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: castCache(cache))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var normalized = [String: MLXArray]()
        for (k, v) in weights {
            var nk = k
            if nk.hasPrefix("model.") { nk = String(nk.dropFirst("model.".count)) }
            // Skip audio — Gemma4 VLM doesn't implement audio, these weights have no module
            if nk.hasPrefix("audio_tower.") || nk.hasPrefix("embed_audio.") { continue }
            // Skip clipped linear params — training artifacts, not used in inference (we use plain Linear)
            if nk.contains("input_min") || nk.contains("input_max") || nk.contains("output_min")
                || nk.contains("output_max")
            {
                continue
            }
            if nk.contains("rotary_emb") { continue }
            // Remap language_model keys to include model. prefix — EXCEPT
            // `language_model.lm_head.*`. The untied head lives directly
            // under `Gemma4TextModel` at `language_model.lm_head`, not under
            // its inner `model` module. The vocab-trim loop below keeps the
            // same boundary.
            if nk.hasPrefix("language_model.") && !nk.hasPrefix("language_model.model.")
                && !nk.hasPrefix("language_model.lm_head")
            {
                nk = "language_model.model." + String(nk.dropFirst("language_model.".count))
            }
            if nk.contains(".switch_mlp.") {
                nk = nk.replacingOccurrences(of: ".switch_mlp.", with: ".experts.switch_glu.")
            }
            // Vision tower uses ClippableLinear wrappers — checkpoint has .linear. segment
            // that doesn't exist in our module tree (we use plain Linear)
            if nk.hasPrefix("vision_tower.") && nk.contains(".linear.") {
                nk = nk.replacingOccurrences(of: ".linear.", with: ".")
            }
            normalized[nk] = v
        }

        // Keep vision tensors outside the text sanitizer: vision layer names
        // also contain `layers.N.self_attn` and must never be interpreted as
        // Gemma 4 language-layer indices. Language tensors retain their
        // `language_model.` ownership prefix in the VLM module tree.
        let textWeights = normalized.filter { $0.key.hasPrefix("language_model.") }
        var p = normalized.filter { !$0.key.hasPrefix("language_model.") }
        for (k, v) in languageModel.sanitize(weights: textWeights) {
            p[k] = v
        }
        let ev = config.textConfig.vocabSize
        for k in [
            "language_model.model.embed_tokens.weight", "language_model.model.embed_tokens.scales",
            "language_model.model.embed_tokens.biases", "language_model.lm_head.weight",
            "language_model.lm_head.scales", "language_model.lm_head.biases",
        ] {
            if let w = p[k], w.dim(0) != ev { p[k] = w[0 ..< ev] }
        }
        return p
    }
}

extension Gemma4: LoRAModel {
    public var loraLayers: [Module] { languageModel.decoderLayers }
}

// MARK: - External vision-embedding seam (CBv2 multimodal prefill)

extension Gemma4 {
    /// The token id every image soft-token position carries in the tokenized
    /// prompt (`image_token_id`; the processor writes `imageSeqLength` of
    /// these per image between the ordinary `boi`/`eoi` delimiter tokens).
    /// External engines locate the per-image placeholder runs with it.
    public var imagePlaceholderTokenId: Int { config.imageTokenId }

    /// Per-image soft-token embeddings — the EXACT arrays `prepare` scatters
    /// over the image placeholder positions (vision tower + multimodal
    /// projector, `embedVision(visionTower(frame))`): one
    /// `[1, softTokens, textHidden]` array per image, in the text embedding
    /// space and the language model's token-embedding dtype (matching
    /// `prepare`'s `emb.dtype`).
    ///
    /// This is the seam external continuous-batching engines (CBv2
    /// multimodal prefill) use to precompute the embeddings they splice at
    /// the placeholder spans; `prepare` itself is unchanged and shares the
    /// same private loop, so the two paths cannot drift.
    public func perImageVisionFeatures(pixels: MLXArray, frames: [THW]?) -> [MLXArray] {
        // Same dtype resolution as `prepare`: the token-embedding output
        // dtype (dtype/shape are lazy graph metadata — nothing evaluates).
        let dtype = languageModel.scaledInputEmbeddings(
            MLXArray([Int32(0)]).reshaped([1, 1])
        ).dtype
        return visionFeatureList(pixels: pixels, frames: frames, isVideo: false)
            .map { $0.asType(dtype) }
    }

    /// The token id every VIDEO soft-token position carries in the tokenized
    /// prompt (`video_token_id`, default 258_884 — the processor emits one
    /// delimited `timestamp + boi + <placeholder>×count + eoi` block of these
    /// per sampled frame). External engines locate the per-frame placeholder
    /// runs with it. nil mirrors a config that explicitly disables video.
    public var videoPlaceholderTokenId: Int? { config.videoTokenId }

    /// Per-FRAME soft-token embeddings for a sampled video — the EXACT
    /// arrays `prepare` scatters over the video placeholder positions: the
    /// mirror of `perImageVisionFeatures` with the VIDEO patch budget
    /// (`isVideo` selects the per-frame, data-driven output length derived
    /// from the video-budget-resized grid — ~70 soft tokens per frame vs 280
    /// per image). One `[1, softTokens, textHidden]` array per frame, in the
    /// language model's token-embedding dtype (matching `prepare`'s
    /// `emb.dtype`). Shares the same private tower loop as `prepare`, so the
    /// two paths cannot drift.
    public func perVideoFrameVisionFeatures(pixels: MLXArray, frames: [THW]?) -> [MLXArray] {
        let dtype = languageModel.scaledInputEmbeddings(
            MLXArray([Int32(0)]).reshaped([1, 1])
        ).dtype
        return visionFeatureList(pixels: pixels, frames: frames, isVideo: true)
            .map { $0.asType(dtype) }
    }
}

// MARK: - Processor

/// Nested `video_processor` sub-config from `preprocessor_config.json`. Gemma4
/// ships a separate, lower per-frame soft-token budget for video than for images
/// (`video_processor.max_soft_tokens` ~70 vs the image `max_soft_tokens` 280).
private struct Gemma4VideoProcessorConfiguration: Codable {
    let maxSoftTokens: Int?
    enum CodingKeys: String, CodingKey {
        case maxSoftTokens = "max_soft_tokens"
    }
}

public struct Gemma4ProcessorConfiguration: Codable, Sendable {
    public let processorClass: String
    public let patchSize: Int
    public let maxSoftTokens: Int
    public let videoMaxSoftTokens: Int
    public let poolingKernelSize: Int
    public let imageSeqLength: Int
    public let audioSeqLength: Int
    // Gemma4 wraps every image / video-frame soft-token block with begin-of-image
    // (boi) and end-of-image (eoi) delimiter tokens, matching the Python
    // processor. These are ordinary vocab tokens (embedded normally, attended
    // causally) — they are NOT image/video soft tokens, so they do not affect the
    // maskedScatter count or the bidirectional visual-span mask.
    public let boiTokenId: Int
    public let eoiTokenId: Int?
    /// Read by upstream's `EmbeddingGemma2`, which refuses checkpoints that normalize.
    public let doNormalize: Bool

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case patchSize = "patch_size"
        case maxSoftTokens = "max_soft_tokens"
        case poolingKernelSize = "pooling_kernel_size"
        case imageSeqLength = "image_seq_length"
        case audioSeqLength = "audio_seq_length"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case doNormalize = "do_normalize"
    }

    /// `do_normalize` nested under `image_processor` in processor_config.json.
    private enum ImageProcessorTopKeys: String, CodingKey {
        case imageProcessor = "image_processor"
    }

    private struct ImageProcessorNormalization: Decodable {
        let doNormalize: Bool?

        enum CodingKeys: String, CodingKey {
            case doNormalize = "do_normalize"
        }
    }

    // `video_processor` is a nested block, so it is read via its own keyed
    // container below rather than the main `CodingKeys`. `videoMaxSoftTokens` is
    // therefore decoded separately and intentionally omitted from `CodingKeys`
    // (the synthesized Encodable simply skips it; this config is decode-only in
    // practice). Mirrors the `TopKeys`/`rope_parameters` pattern in
    // `Gemma4VisionConfig`.
    enum VideoProcessorTopKeys: String, CodingKey {
        case videoProcessor = "video_processor"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass =
            try c.decodeIfPresent(String.self, forKey: .processorClass) ?? "Gemma4Processor"
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        maxSoftTokens = try c.decodeIfPresent(Int.self, forKey: .maxSoftTokens) ?? 280
        // Video frames use the nested `video_processor.max_soft_tokens` budget (~70),
        // not the image `max_soft_tokens` (280). Tolerate the key being absent or a
        // non-dict so text/image-only configs keep loading.
        var decodedVideoSoftTokens: Int? = nil
        if let vc = try? decoder.container(keyedBy: VideoProcessorTopKeys.self),
            let videoSub = try? vc.decodeIfPresent(
                Gemma4VideoProcessorConfiguration.self, forKey: .videoProcessor)
        {
            decodedVideoSoftTokens = videoSub.maxSoftTokens
        }
        videoMaxSoftTokens = decodedVideoSoftTokens ?? 70
        poolingKernelSize = try c.decodeIfPresent(Int.self, forKey: .poolingKernelSize) ?? 3
        imageSeqLength = try c.decodeIfPresent(Int.self, forKey: .imageSeqLength) ?? 280
        audioSeqLength = try c.decodeIfPresent(Int.self, forKey: .audioSeqLength) ?? 750
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: .boiTokenId) ?? 255_999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: .eoiTokenId) ?? 258_882
        let nested = try? decoder.container(keyedBy: ImageProcessorTopKeys.self)
            .decodeIfPresent(ImageProcessorNormalization.self, forKey: .imageProcessor)
        doNormalize =
            try c.decodeIfPresent(Bool.self, forKey: .doNormalize)
            ?? nested?.doNormalize ?? false
    }

    /// Soft tokens the vision tower produces for an image of the given
    /// (already resized) pixel dimensions. From upstream; `EmbeddingGemma2` uses it.
    public func softTokenCount(height: Int, width: Int) -> Int {
        ((height / patchSize) * (width / patchSize)) / (poolingKernelSize * poolingKernelSize)
    }

    /// Upstream's port of the Python Gemma4ImageProcessor aspect-ratio preserving
    /// resize. `EmbeddingGemma2` uses it. The fork's Gemma 4 processor keeps its own
    /// resize path.
    public func aspectPreservingTargetSize(for imageSize: CGSize, budget: Int? = nil) -> CGSize {
        let maxSoftTokens = budget ?? self.maxSoftTokens
        let kernelArea = poolingKernelSize * poolingKernelSize
        let maxPatches = maxSoftTokens * kernelArea
        let sideMultiple = poolingKernelSize * patchSize
        let height = Double(imageSize.height)
        let width = Double(imageSize.width)

        let targetPixels = Double(maxPatches * patchSize * patchSize)
        let factor = (targetPixels / max(height * width, 1)).squareRoot()
        var targetHeight = Int((factor * height / Double(sideMultiple)).rounded(.down))
        var targetWidth = Int((factor * width / Double(sideMultiple)).rounded(.down))

        let maxSideLength = maxSoftTokens
        if targetHeight == 0 {
            targetHeight = 1
            targetWidth = min(Int((width / max(height, 1)).rounded(.down)), maxSideLength)
            targetWidth = max(targetWidth, 1)
        } else if targetWidth == 0 {
            targetWidth = 1
            targetHeight = min(Int((height / max(width, 1)).rounded(.down)), maxSideLength)
            targetHeight = max(targetHeight, 1)
        }

        return CGSize(
            width: targetWidth * sideMultiple, height: targetHeight * sideMultiple)
    }
}

/// Gemma4-specific chat message generator. Unlike `Qwen2VLMessageGenerator`,
/// Gemma4's chat template expects SYSTEM messages as a plain string `content`
/// (the template folds the system text into the first user turn); only the
/// non-system turns carry typed multimodal content arrays. Reusing Qwen2VL's
/// generator emits a typed array for the system role too, which renders malformed
/// Gemma4 prompts. Modality order is images, then videos, then text — matching the
/// HF Gemma4 chat template / processor. (PR #56's vMLX decode port dropped this;
/// `UserInputTests.testGemma4Conversion*` still pin the plain-string-system
/// contract, so they reference this type.)
public struct Gemma4MessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        var dictionary: MLXLMCommon.Message
        if message.role == .system {
            dictionary = [
                "role": message.role.rawValue,
                "content": message.content,
            ]
        } else {
            dictionary = [
                "role": message.role.rawValue,
                "content": contentParts(for: message, layout: .imagesThenVideosThenText),
            ]
        }
        addToolMetadata(to: &dictionary, for: message)
        return dictionary
    }
}

public struct Gemma4Processor: UserInputProcessor {
    private let config: Gemma4ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Gemma4ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    /// Resize a single image/frame to Gemma4's soft-token budget and convert it
    /// to the sRGB tone curve the vision tower was trained on (PIL/Python
    /// default). Shared by the image and video paths so frames go through the
    /// exact same pipeline.
    private func resizedSRGB(from ci: CIImage, maxSoftTokens: Int? = nil) throws -> CIImage {
        let ps = config.patchSize
        let budget = maxSoftTokens ?? config.maxSoftTokens
        let maxP = budget * config.poolingKernelSize * config.poolingKernelSize
        // Reject zero-area, infinite, and NaN extents explicitly. The scale-factor
        // math below divides by `w * h`; a CIImage with a zero extent produces an
        // infinite `f` and a NaN trap inside `Int(floor(.nan))`. A non-finite
        // extent (e.g. `CIImage(color:)` returns `(.infinity, .infinity)`) traps
        // even earlier inside `Int(.infinity)`. Both surface as
        // VLMError.imageProcessingFailure now.
        let (h, w) = try gemma4IntExtent(ci.extent.size)
        let f = sqrt(Float(maxP * ps * ps) / Float(w * h))
        let sm = config.poolingKernelSize * ps
        var tH = Int(floor(f * Float(h) / Float(sm))) * sm
        var tW = Int(floor(f * Float(w) / Float(sm))) * sm
        if tH == 0 { tH = sm }
        if tW == 0 { tW = sm }
        let resized = MediaProcessing.resampleBicubic(ci, to: CGSize(width: tW, height: tH))
        return MediaProcessing.inSRGBToneCurveSpace(resized)
    }

    /// Resize + convert one image/frame to a `[1, C, H, W]` (NCHW) pixel tensor
    /// with float values in `[0, 1]`.
    private func resizedPixels(from ci: CIImage) throws -> MLXArray {
        MediaProcessing.asMLXArray(try resizedSRGB(from: ci))
    }

    /// Pack per-image/per-frame `[1, C, H, W]` tensors into one flat batch and
    /// record each one's real (un-padded) size in `frames`, so the vision tower
    /// processes each at its original size. Shorter tensors are zero-padded up to
    /// the batch max so they can share a single storage tensor.
    private func packFrames(_ arrays: [MLXArray]) -> (pixels: MLXArray, frames: [THW]) {
        let sizes = arrays.map { THW(1, $0.dim(2), $0.dim(3)) }
        if arrays.count == 1 { return (arrays[0], sizes) }
        let maxH = arrays.map { $0.dim(2) }.max()!
        let maxW = arrays.map { $0.dim(3) }.max()!
        let stored = arrays.map { arr -> MLXArray in
            let h = arr.dim(2)
            let w = arr.dim(3)
            if h == maxH && w == maxW { return arr }
            return MLX.padded(arr, widths: [[0, 0], [0, 0], [0, maxH - h], [0, maxW - w]])
        }
        return (concatenated(stored), sizes)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Gemma4MessageGenerator().generate(from: input)
        var tokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)

        // ── Images ── each image is resized independently (aspect-ratio
        // preserving) and stored as a flat `[N, C, H, W]` batch indexed by
        // `frames`; the vision tower processes each at its real, un-padded size.
        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let arrays = try input.images.map { try resizedPixels(from: $0.asCIImage()) }
            let packed = packFrames(arrays)
            processedImage = LMInput.ProcessedImage(pixels: packed.pixels, frames: packed.frames)
        }

        // ── Videos ── frames are sampled (~32 spread uniformly across the whole
        // clip, matching the Python Gemma4VideoProcessor) and resized with the
        // lower per-frame VIDEO soft-token budget (`video_processor.max_soft_tokens`,
        // ~70) rather than the image budget (280), then stored the same flat way.
        // Per-frame timestamps (seconds) are captured for the `mm:ss` prompt prefix
        // (HF `Gemma4Processor.replace_video_token`); the per-video soft-token count
        // is data-driven from the resized frame grid so it matches the vision tower's
        // pooled output exactly.
        var processedVideo: LMInput.ProcessedVideo?
        var videoTimestamps: [[Double]] = []
        var videoSoftTokenCounts: [Int] = []
        if !input.videos.isEmpty {
            var frames: [MLXArray] = []
            let poolSq = config.poolingKernelSize * config.poolingKernelSize
            for video in input.videos {
                let sequence = try await MediaProcessing.asProcessedSequence(
                    video, targetFPS: { duration in 32.0 / max(duration.seconds, 1.0) },
                    maxFrames: 32
                ) { frame in
                    VideoFrame(
                        frame: try resizedSRGB(
                            from: frame.frame, maxSoftTokens: config.videoMaxSoftTokens),
                        timeStamp: frame.timeStamp)
                }
                videoTimestamps.append(
                    sequence.timestamps.map { $0.seconds.isFinite ? $0.seconds : 0 })
                // All frames of one video share the same resized size, so the
                // per-frame soft-token count is uniform; derive it from the first
                // frame's grid (`(h/patch)*(w/patch)/pool^2`), matching the model's
                // `videoOutputLength` and the placeholder expansion below.
                if let first = sequence.frames.first {
                    let h = first.dim(2)
                    let w = first.dim(3)
                    videoSoftTokenCounts.append(
                        max(1, ((h / config.patchSize) * (w / config.patchSize)) / max(1, poolSq)))
                } else {
                    videoSoftTokenCounts.append(config.videoMaxSoftTokens)
                }
                frames.append(contentsOf: sequence.frames)
            }
            if !frames.isEmpty {
                let packed = packFrames(frames)
                processedVideo = LMInput.ProcessedVideo(
                    pixels: packed.pixels, frames: packed.frames)
            }
        }

        // ── Expand placeholders into delimited soft-token blocks ──
        // Gemma4 represents each image / video frame as `boi + soft_token*count +
        // eoi`. For images `count` is `imageSeqLength` (== the model's
        // `defaultOutputLength`); for video frames it is the per-frame video budget
        // (`video_processor.max_soft_tokens`, derived from the resized grid), and
        // each block is additionally prefixed with the frame's `mm:ss` timestamp.
        // Either way the expanded soft-token count matches the vision features
        // `maskedScatter` writes. The boi/eoi delimiters are ordinary tokens (not
        // soft tokens) so they do not change that count or the visual-span mask.
        //
        // `convertTokenToId` is used rather than `encode("<|image|>").last` so the
        // lookup goes straight through the tokenizer's special-token map and never
        // picks up an appended BOS/EOS — `encode(text:)` defaults to
        // `addSpecialTokens: true`, which on some tokenizers prepends BOS. The
        // numeric fallbacks cover tokenizers that don't expose the placeholders as
        // addable special tokens.
        if processedImage != nil || processedVideo != nil {
            let imgId = tokenizer.convertTokenToId("<|image|>") ?? 258_880
            let vidId = tokenizer.convertTokenToId("<|video|>") ?? 258_884
            func appendBlock(_ acc: inout [Int], _ tokenId: Int, _ count: Int) {
                acc.append(config.boiTokenId)
                acc.append(contentsOf: Array(repeating: tokenId, count: count))
                if let eoi = config.eoiTokenId { acc.append(eoi) }
            }
            var expanded = [Int]()
            expanded.reserveCapacity(tokens.count)
            var videoIndex = 0
            for t in tokens {
                if t == imgId, processedImage != nil {
                    appendBlock(&expanded, imgId, config.imageSeqLength)
                } else if t == vidId, processedVideo != nil {
                    // One `mm:ss <boi> <|video|>*count <eoi>` block per sampled frame,
                    // joined by a space — mirrors HF Gemma4Processor.replace_video_token
                    // `" ".join("{ts} {boi}{video*n}{eoi}")`. The leading space on
                    // frames after the first reproduces that join separator. The mm:ss
                    // prefix is encoded as ordinary text tokens (addSpecialTokens:false
                    // so no BOS leaks in). One block per frame keeps the total video
                    // soft-token count equal to the vision features (frames == timestamps
                    // count); `count` is the per-frame video budget.
                    let timestamps =
                        videoIndex < videoTimestamps.count ? videoTimestamps[videoIndex] : []
                    let count =
                        videoIndex < videoSoftTokenCounts.count
                        ? videoSoftTokenCounts[videoIndex] : config.videoMaxSoftTokens
                    for (frameIdx, seconds) in timestamps.enumerated() {
                        let totalSeconds = max(0, Int(seconds.rounded(.towardZero)))
                        let tsString = String(
                            format: "%02d:%02d", totalSeconds / 60, totalSeconds % 60)
                        let prefix = (frameIdx == 0 ? "" : " ") + tsString + " "
                        expanded.append(
                            contentsOf: tokenizer.encode(text: prefix, addSpecialTokens: false))
                        appendBlock(&expanded, vidId, count)
                    }
                    videoIndex += 1
                } else {
                    expanded.append(t)
                }
            }
            tokens = expanded
        }

        let pa = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: pa, mask: ones(like: pa).asType(.int8)),
            image: processedImage,
            video: processedVideo)
    }
}

// MARK: - Upstream Gemma 4 text stack and Gemma 4 Unified (12B)

// The sections below come from upstream ml-explore/mlx-swift-lm. The fork's
// `Gemma4` VLM above keeps the MLXLLM text tower (`Gemma4TextModel`). The
// upstream text stack serves `Gemma4Unified` and the upstream assistant
// drafter (`Gemma4Assistant.swift`).

private enum Gemma4Error: LocalizedError {
    case multimodalTokenCountMismatch(kind: String, featureTokens: Int, promptTokens: Int)
    case imagePlaceholderMismatch(images: Int, placeholders: Int)

    var errorDescription: String? {
        switch self {
        case .multimodalTokenCountMismatch(let kind, let featureTokens, let promptTokens):
            return
                "Gemma4 \(kind) token count mismatch: encoder produced \(featureTokens) soft tokens, but the prompt contains \(promptTokens) \(kind) tokens."
        case .imagePlaceholderMismatch(let images, let placeholders):
            return
                "Gemma4 image placeholder mismatch: the request has \(images) images but the prompt contains at least \(placeholders) image placeholders."
        }
    }
}

private func gemma4BuildLayerTypes(hiddenLayers: Int, slidingWindowPattern: Int) -> [String] {
    let pattern =
        Array(repeating: "sliding_attention", count: max(slidingWindowPattern - 1, 0))
        + ["full_attention"]
    guard !pattern.isEmpty else { return Array(repeating: "full_attention", count: hiddenLayers) }
    var result: [String] = []
    result.reserveCapacity(hiddenLayers)
    while result.count < hiddenLayers {
        result.append(contentsOf: pattern)
    }
    return Array(result.prefix(hiddenLayers))
}

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
func gemma4DefaultTextRopeParameters() -> [String: [String: StringOrNumber]] {
    [
        "full_attention": [
            "partial_rotary_factor": .float(1.0),
            "rope_theta": .float(1_000_000.0),
            "rope_type": .string("proportional"),
        ],
        "sliding_attention": [
            "partial_rotary_factor": .float(1.0),
            "rope_theta": .float(10_000.0),
            "rope_type": .string("default"),
        ],
    ]
}

private func gemma4MaskedScatter(
    inputTensor: MLXArray, mask: MLXArray, source: MLXArray
) -> MLXArray {
    let flattenedInput = inputTensor.flattened()
    let flattenedMask = mask.flattened().asArray(Bool.self)
    let flattenedSource = source.flattened()

    let targetIndices = flattenedMask.enumerated().compactMap { idx, value in
        value ? Int32(idx) : nil
    }
    guard !targetIndices.isEmpty else {
        return inputTensor
    }

    guard flattenedSource.dim(0) == targetIndices.count else {
        fatalError(
            "Masked scatter shape mismatch. source=\(flattenedSource.dim(0)) mask=\(targetIndices.count)"
        )
    }

    let result = flattenedInput
    result[MLXArray(targetIndices, [targetIndices.count])] = flattenedSource
    return result.reshaped(inputTensor.shape)
}

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
enum Gemma4SharedKVState {
    case regular(keys: MLXArray, values: MLXArray)
    case quantized(
        keys: (MLXArray, MLXArray, MLXArray?),
        values: (MLXArray, MLXArray, MLXArray?),
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    )

    var sequenceLength: Int {
        switch self {
        case .regular(let keys, _):
            return keys.dim(2)
        case .quantized(let keys, _, _, _, _):
            return keys.0.dim(-2)
        }
    }
}

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
func gemma4AdjustAttentionMask(
    _ mask: MLXFast.ScaledDotProductAttentionMaskMode,
    keyLength: Int
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    switch mask {
    case .array(let maskArray):
        let maskLength = maskArray.dim(-1)
        guard maskLength > keyLength else {
            return mask
        }
        let start = maskLength - keyLength
        return .array(maskArray[.ellipsis, start...])
    case .arrays, .causal, .none:
        return mask
    }
}

private func gemma4TokenTypeIds(
    inputIds: MLXArray,
    imageTokenId: Int?,
    videoTokenId: Int?,
    audioTokenId: Int?
) -> MLXArray {
    var tokenTypeIds = MLXArray.zeros(like: inputIds).asType(.int32)
    if let imageTokenId {
        tokenTypeIds = MLX.where(inputIds .== imageTokenId, MLXArray(1), tokenTypeIds)
    }
    if let videoTokenId {
        tokenTypeIds = MLX.where(inputIds .== videoTokenId, MLXArray(2), tokenTypeIds)
    }
    if let audioTokenId {
        tokenTypeIds = MLX.where(inputIds .== audioTokenId, MLXArray(3), tokenTypeIds)
    }
    return tokenTypeIds
}

private func gemma4TextOnlyPromptTokens(_ input: LMInput) -> MLXArray {
    let tokens = input.text.tokens
    if tokens.ndim == 2, tokens.dim(0) == 1 {
        return tokens[0]
    }
    if tokens.ndim == 1 {
        return tokens
    }
    return tokens.flattened()
}

private func gemma4PrepareTextOnly(
    _ input: LMInput,
    cache: [any KVCache],
    prefill: PrefillParameters,
    languageModel: Gemma4TextLanguageModel
) throws -> PrepareResult {
    let y = gemma4TextOnlyPromptTokens(input).expandedDimensions(axis: 0)
    let convertedCache = cache.map { $0 }
    let totalPositions = y.dim(1)

    let processed = try prefill.forEachChunk(total: totalPositions) { range in
        _ = languageModel(y[0..., range], cache: convertedCache)
        asyncEval(cache)
    }
    if processed > 0 { eval(cache) }
    let result = languageModel(y[0..., processed...], cache: convertedCache)
    prefill.progress?(totalPositions, totalPositions)
    return .logits(result)
}

private func gemma4BlockSequenceIdsForMask(_ tokenTypeIds: MLXArray) -> MLXArray {
    let isVision = (tokenTypeIds .== 1) | (tokenTypeIds .== 2)
    let sequenceLength = isVision.dim(1)
    guard sequenceLength > 0 else {
        return MLXArray.zeros(like: tokenTypeIds).asType(.int32) - 1
    }

    let previous =
        if sequenceLength == 1 {
            MLXArray.zeros(like: isVision)
        } else {
            concatenated(
                [
                    MLXArray.zeros(like: isVision[0..., ..<1]),
                    isVision[0..., ..<(sequenceLength - 1)],
                ],
                axis: 1
            )
        }
    let starts = isVision & logicalNot(previous)
    let groupIds = cumsum(starts.asType(.int32), axis: 1) - 1
    return MLX.where(isVision, groupIds, MLXArray.zeros(like: groupIds) - 1)
}

private func gemma4ApplyBlockwiseBidirectionalOverlay(
    _ mask: MLXFast.ScaledDotProductAttentionMaskMode,
    tokenTypeIds: MLXArray,
    sequenceLength: Int,
    windowSize: Int?
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let baseMask: MLXArray
    switch mask {
    case .array(let array):
        baseMask = array
    case .arrays(let arrays):
        guard let first = arrays.first else {
            return mask
        }
        baseMask = first
    case .causal:
        baseMask = createCausalMask(n: sequenceLength, offset: 0, windowSize: windowSize)
    case .none:
        baseMask = MLXArray.ones([sequenceLength, sequenceLength], dtype: .bool)
    }

    guard tokenTypeIds.dim(1) == baseMask.dim(-1) else {
        return mask
    }

    let blockSequenceIds = gemma4BlockSequenceIdsForMask(tokenTypeIds)
    let queryBlocks = expandedDimensions(blockSequenceIds, axis: -1)
    let keyBlocks = expandedDimensions(blockSequenceIds, axis: -2)
    let sameBlock = (queryBlocks .!= -1) & (queryBlocks .== keyBlocks)
    return .array(baseMask | expandedDimensions(sameBlock, axis: 1))
}

private func gemma4CompactPrefixRows(features: MLXArray, validMask: MLXArray) -> MLXArray {
    let maskRows = validMask.asArray(Bool.self)
    let batch = validMask.dim(0)
    let length = validMask.dim(1)
    var rows: [MLXArray] = []
    rows.reserveCapacity(batch)

    for batchIdx in 0 ..< batch {
        let start = batchIdx * length
        let count = maskRows[start ..< start + length].reduce(0) { $0 + ($1 ? 1 : 0) }
        if count > 0 {
            rows.append(features[batchIdx, ..<count, 0...])
        }
    }

    guard !rows.isEmpty else {
        return features.reshaped(-1, features.dim(-1))[..<0, 0...]
    }
    return concatenated(rows, axis: 0)
}
// MARK: - Configuration

public struct Gemma4TextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let globalKVHeads: Int?
    public let headDim: Int
    public let globalHeadDim: Int
    public let vocabularySize: Int
    public let vocabularySizePerLayerInput: Int
    public let numKVSharedLayers: Int
    public let hiddenSizePerLayerInput: Int
    public let slidingWindow: Int
    public let slidingWindowPattern: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTraditional: Bool
    public let finalLogitSoftcapping: Float?
    public let useDoubleWideMLP: Bool
    public let enableMoEBlock: Bool
    public let numExperts: Int?
    public let topKExperts: Int?
    public let moeIntermediateSize: Int?
    public let attentionKEqV: Bool
    public let useBidirectionalAttention: String?
    public let layerTypes: [String]
    public let ropeParameters: [String: [String: StringOrNumber]]
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case globalKVHeads = "num_global_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case vocabularySize = "vocab_size"
        case vocabularySizePerLayerInput = "vocab_size_per_layer_input"
        case numKVSharedLayers = "num_kv_shared_layers"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTraditional = "rope_traditional"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useDoubleWideMLP = "use_double_wide_mlp"
        case enableMoEBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case attentionKEqV = "attention_k_eq_v"
        case useBidirectionalAttention = "use_bidirectional_attention"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType) ?? "gemma4_text"
        let isUnified = modelType == "gemma4_unified_text" || modelType == "gemma4_unified"
        hiddenSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenSize)
            ?? (isUnified ? 3840 : 1536)
        hiddenLayers =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenLayers)
            ?? (isUnified ? 48 : 35)
        intermediateSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.intermediateSize)
            ?? (isUnified ? 15_360 : 6144)
        attentionHeads =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.attentionHeads)
            ?? (isUnified ? 16 : 8)
        kvHeads =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.kvHeads) ?? (isUnified ? 8 : 1)
        globalKVHeads =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.globalKVHeads)
            ?? (isUnified ? 1 : nil)
        headDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.headDim) ?? 256
        globalHeadDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.globalHeadDim) ?? 512
        vocabularySize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.vocabularySize) ?? 262_144
        vocabularySizePerLayerInput =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.vocabularySizePerLayerInput)
            ?? vocabularySize
        numKVSharedLayers =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.numKVSharedLayers)
            ?? (isUnified ? 0 : 20)
        hiddenSizePerLayerInput =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenSizePerLayerInput)
            ?? (isUnified ? 0 : 256)
        slidingWindow =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.slidingWindow)
            ?? (isUnified ? 1024 : 512)
        slidingWindowPattern =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.slidingWindowPattern)
            ?? (isUnified ? 6 : 5)
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.maxPositionEmbeddings) ?? 131_072
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: CodingKeys.rmsNormEps) ?? 1e-6
        ropeTraditional =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.ropeTraditional) ?? false
        finalLogitSoftcapping =
            try c.decodeIfPresent(Float.self, forKey: CodingKeys.finalLogitSoftcapping) ?? 30.0
        useDoubleWideMLP =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.useDoubleWideMLP) ?? true
        enableMoEBlock =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.enableMoEBlock) ?? false
        numExperts = try c.decodeIfPresent(Int.self, forKey: CodingKeys.numExperts)
        topKExperts = try c.decodeIfPresent(Int.self, forKey: CodingKeys.topKExperts)
        moeIntermediateSize = try c.decodeIfPresent(
            Int.self, forKey: CodingKeys.moeIntermediateSize)
        attentionKEqV =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.attentionKEqV) ?? isUnified
        useBidirectionalAttention =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.useBidirectionalAttention)
            ?? (isUnified ? "vision" : nil)
        ropeParameters =
            try c.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: CodingKeys.ropeParameters)
            ?? (isUnified
                ? [
                    "full_attention": [
                        "partial_rotary_factor": .float(0.25),
                        "rope_theta": .float(1_000_000.0),
                        "rope_type": .string("proportional"),
                    ],
                    "sliding_attention": [
                        "rope_theta": .float(10_000.0),
                        "rope_type": .string("default"),
                    ],
                ] : gemma4DefaultTextRopeParameters())
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: CodingKeys.layerTypes)
            ?? gemma4BuildLayerTypes(
                hiddenLayers: hiddenLayers, slidingWindowPattern: slidingWindowPattern)
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.tieWordEmbeddings) ?? true
    }
}

// MARK: - Text

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
typealias Gemma4RMSNormNoScale = MLXLMCommon.Gemma4RMSNormNoScale

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
typealias Gemma4RMSNormZeroShift = MLXLMCommon.Gemma4RMSNormZeroShift

final class Gemma4TextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(config: Gemma4TextConfiguration, layerIdx: Int) {
        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        let isKVSharedLayer = layerIdx >= firstKVSharedLayer && firstKVSharedLayer > 0
        let useDoubleWide = config.useDoubleWideMLP && isKVSharedLayer
        let hiddenDimensions = config.intermediateSize * (useDoubleWide ? 2 : 1)

        self._gateProj.wrappedValue = Linear(config.hiddenSize, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, config.hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, hiddenDimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

final class Gemma4TextRouter: Module {
    let topKExperts: Int
    let config: Gemma4TextConfiguration
    private let rootSize: Float

    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "scale") var scale: MLXArray
    @ParameterInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    init(config: Gemma4TextConfiguration) {
        guard let numExperts = config.numExperts, let topKExperts = config.topKExperts else {
            fatalError("Gemma4 MoE router requires numExperts and topKExperts")
        }

        self.topKExperts = topKExperts
        self.config = config
        self.rootSize = pow(Float(config.hiddenSize), -0.5)

        self._proj.wrappedValue = Linear(config.hiddenSize, numExperts, bias: false)
        self._scale.wrappedValue = MLXArray.ones([config.hiddenSize])
        self._perExpertScale.wrappedValue = MLXArray.ones([numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let normed = MLXFast.rmsNorm(
            x, weight: (scale * rootSize).asType(x.dtype), eps: config.rmsNormEps)

        let scores = proj(normed)

        let topKIndices = MLX.argPartition(scores, kth: -topKExperts, axis: -1)[
            .ellipsis, (-topKExperts)...,
        ]
        var topKWeights = MLX.takeAlong(scores, topKIndices, axis: -1)
        topKWeights = MLX.softmax(topKWeights, axis: -1)
        topKWeights = topKWeights * perExpertScale[topKIndices].asType(topKWeights.dtype)
        return (topKIndices, topKWeights)
    }
}

final class Gemma4TextExperts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU

    init(config: Gemma4TextConfiguration) {
        guard let numExperts = config.numExperts,
            let moeIntermediateSize = config.moeIntermediateSize
        else {
            fatalError("Gemma4 MoE experts require numExperts and moeIntermediateSize")
        }

        self._switchGLU.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: moeIntermediateSize,
            numExperts: numExperts,
            activation: geluApproximate,
            bias: false
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, topKIndices: MLXArray, topKWeights: MLXArray
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let hidden = x.dim(2)
        let topK = topKIndices.dim(-1)

        let expertOutput = switchGLU(
            x.reshaped(batch * length, hidden),
            topKIndices.reshaped(batch * length, topK)
        )
        let weights = topKWeights.reshaped(batch * length, topK).asType(expertOutput.dtype)
        return weightedExpertSum(expertOutput, weights).reshaped(batch, length, hidden)
    }
}

final class Gemma4ScaledLinear: Module, UnaryLayer {
    @ModuleInfo(key: "weight") var weight: MLXArray
    let scalar: Float

    init(inFeatures: Int, outFeatures: Int, scalar: Float) {
        self.scalar = scalar
        self._weight.wrappedValue = MLXArray.zeros([outFeatures, inFeatures])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (x.matmul(weight.transposed())) * scalar
    }
}

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
/// Use `kvSharedOnly: true` in the constructor to skip building local K/V
/// projections (the drafter consumes the target's K/V via `sharedKV` instead).
final class Gemma4TextAttention: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String
    let isSliding: Bool
    let headDim: Int
    let numHeads: Int
    let numKVHeads: Int
    let scale: Float
    let isKVSharedLayer: Bool
    let useKEqV: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear?
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "k_norm") var kNorm: Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "v_norm") var vNorm: Gemma4RMSNormNoScale?
    @ModuleInfo var rope: OffsetLayer

    init(config: Gemma4TextConfiguration, layerIdx: Int, kvSharedOnly: Bool = false) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.layerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"
        self.headDim =
            layerType == "full_attention" && config.globalHeadDim > 0
            ? config.globalHeadDim : config.headDim
        self.numHeads = config.attentionHeads
        self.useKEqV = config.attentionKEqV && !isSliding
        self.numKVHeads =
            useKEqV ? (config.globalKVHeads ?? config.kvHeads) : config.kvHeads
        self.scale = 1.0

        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        self.isKVSharedLayer = layerIdx >= firstKVSharedLayer && firstKVSharedLayer > 0

        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        // KV-shared layers (the last `num_kv_shared_layers`) reuse an earlier layer's
        // K/V and own no k_proj/v_proj/k_norm/v_norm — the checkpoint ships none, so the
        // module tree must not declare them (else loadWeights fails keyNotFound on the
        // shared layers, e.g. E2B layer 15 / E4B layer 24). `kvSharedOnly` is the drafter's
        // always-shared variant; `isKVSharedLayer` covers the target's own shared tail.
        // Mirrors the text backbone (MLXLLM/Models/Gemma4Text.swift).
        if !kvSharedOnly && !isKVSharedLayer {
            self._kProj.wrappedValue = Linear(
                config.hiddenSize, numKVHeads * headDim, bias: false)
            if !useKEqV {
                self._vProj.wrappedValue = Linear(
                    config.hiddenSize, numKVHeads * headDim, bias: false)
            }
            self._kNorm.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: headDim, eps: config.rmsNormEps)
            self._vNorm.wrappedValue = Gemma4RMSNormNoScale(eps: config.rmsNormEps)
        }
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: headDim, eps: config.rmsNormEps)

        let ropeKey = isSliding ? "sliding_attention" : "full_attention"
        let ropeConfig = config.ropeParameters[ropeKey]
        let ropeTheta = ropeConfig?["rope_theta"]?.asFloat() ?? (isSliding ? 10_000 : 1_000_000)
        self._rope.wrappedValue = initializeRope(
            dims: headDim,
            base: ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: ropeConfig,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil,
        sharedKV: Gemma4SharedKVState? = nil,
        offset: Int? = nil
    ) -> (MLXArray, Gemma4SharedKVState?, Int) {
        let (batch, length, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(batch, length, numHeads, headDim)
        queries = qNorm(queries)

        let currentOffset: Int
        let kvState: Gemma4SharedKVState?

        if let sharedKV {
            currentOffset = offset ?? 0
            kvState = sharedKV
        } else {
            // KV-owning path: K/V projections must be present. If they are nil
            // here the layer is KV-shared (drafter `kvSharedOnly`, or the target's
            // shared tail `isKVSharedLayer`) and the caller forgot to pass
            // `sharedKV` — a configuration bug.
            guard let kProj, let kNorm, let vNorm else {
                fatalError(
                    "Gemma4 attention called without sharedKV on a KV-shared layer")
            }
            currentOffset = cache?.offset ?? 0
            var keys = kProj(x).reshaped(batch, length, numKVHeads, headDim)
            var values =
                if useKEqV {
                    keys
                } else {
                    vProj!(x).reshaped(batch, length, numKVHeads, headDim)
                }
            keys = kNorm(keys).transposed(0, 2, 1, 3)
            values = vNorm(values).transposed(0, 2, 1, 3)
            keys = rope(keys, offset: currentOffset)
            if let quantizedCache = cache as? QuantizedKVCacheProtocol {
                let (quantizedKeys, quantizedValues) = quantizedCache.updateQuantized(
                    keys: keys, values: values)
                kvState = .quantized(
                    keys: quantizedKeys,
                    values: quantizedValues,
                    groupSize: quantizedCache.groupSize,
                    bits: quantizedCache.bits,
                    mode: quantizedCache.mode
                )
            } else {
                if let cache {
                    (keys, values) = cache.update(keys: keys, values: values)
                }
                kvState = .regular(keys: keys, values: values)
            }
        }

        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: currentOffset)

        guard let kvState else {
            fatalError("Gemma4 attention expected a KV state")
        }
        let localMask = gemma4AdjustAttentionMask(mask, keyLength: kvState.sequenceLength)

        let output: MLXArray =
            switch kvState {
            case .regular(let keys, let values):
                MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: keys,
                    values: values,
                    scale: scale,
                    mask: localMask
                )
            case .quantized(let keys, let values, let groupSize, let bits, let mode):
                quantizedScaledDotProductAttention(
                    queries: queries,
                    quantizedKeys: keys,
                    quantizedValues: values,
                    scale: scale,
                    mask: localMask,
                    groupSize: groupSize,
                    bits: bits,
                    mode: mode
                )
            }

        return (
            oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)),
            kvState,
            currentOffset
        )
    }
}

/// Module-internal — also consumed by `Gemma4Assistant.swift`.
final class Gemma4TextDecoderLayer: Module {
    let layerType: String
    let enableMoE: Bool

    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4TextAttention
    @ModuleInfo var mlp: Gemma4TextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm:
        Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm:
        Gemma4RMSNormZeroShift
    @ModuleInfo(key: "router") var router: Gemma4TextRouter?
    @ModuleInfo(key: "experts") var experts: Gemma4TextExperts?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1:
        Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2:
        Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2:
        Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(config: Gemma4TextConfiguration, layerIdx: Int, kvSharedOnly: Bool = false) {
        self.layerType = config.layerTypes[layerIdx]
        self.enableMoE = config.enableMoEBlock
        self._selfAttention.wrappedValue = Gemma4TextAttention(
            config: config, layerIdx: layerIdx, kvSharedOnly: kvSharedOnly)
        self._mlp.wrappedValue = Gemma4TextMLP(config: config, layerIdx: layerIdx)
        self._inputLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if config.enableMoEBlock {
            self._router.wrappedValue = Gemma4TextRouter(config: config)
            self._experts.wrappedValue = Gemma4TextExperts(config: config)
            self._postFeedforwardLayerNorm1.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm2.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._preFeedforwardLayerNorm2.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
        if config.hiddenSizePerLayerInput > 0 {
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSizePerLayerInput, bias: false)
            self._perLayerProjection.wrappedValue = Linear(
                config.hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
        self._layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil,
        sharedKV: Gemma4SharedKVState? = nil,
        offset: Int? = nil
    ) -> (MLXArray, Gemma4SharedKVState?, Int) {
        var residual = x
        var h = inputLayerNorm(x)
        let (attentionOutput, kvState, attentionOffset) = selfAttention(
            h, mask: mask, cache: cache, sharedKV: sharedKV, offset: offset)
        h = attentionOutput
        h = postAttentionLayerNorm(h)
        h = residual + h

        residual = h
        if enableMoE,
            let router,
            let experts,
            let postFeedforwardLayerNorm1,
            let postFeedforwardLayerNorm2,
            let preFeedforwardLayerNorm2
        {
            var dense = preFeedforwardLayerNorm(h)
            dense = mlp(dense)
            dense = postFeedforwardLayerNorm1(dense)

            let (topKIndices, topKWeights) = router(h)
            var sparse = preFeedforwardLayerNorm2(h)
            sparse = experts(sparse, topKIndices: topKIndices, topKWeights: topKWeights)
            sparse = postFeedforwardLayerNorm2(sparse)

            h = dense + sparse
        } else {
            h = preFeedforwardLayerNorm(h)
            h = mlp(h)
        }
        h = postFeedforwardLayerNorm(h)
        h = residual + h

        if let perLayerInputGate, let perLayerProjection, let postPerLayerInputNorm,
            let perLayerInput
        {
            residual = h
            var gated = perLayerInputGate(h)
            gated = geluApproximate(gated)
            gated = gated * perLayerInput
            gated = perLayerProjection(gated)
            gated = postPerLayerInputNorm(gated)
            h = residual + gated
        }

        return (h * layerScalar, kvState, attentionOffset)
    }
}

/// Module-internal — also consumed by `Gemma4Assistant.swift` (for target-side
/// `embed_tokens` / `embed_scale` / `layer_types` access during drafter bind).
final class Gemma4TextBackbone: Module {
    let config: Gemma4TextConfiguration
    let firstKVSharedLayerIdx: Int
    let layerIdxToCacheIdx: [Int]
    let firstFullCacheIdx: Int
    let firstSlidingCacheIdx: Int
    let embedScale: Float
    let embedTokensPerLayerScale: Float
    let perLayerProjectionScale: Float
    private let _perLayerInputScale: MLXArray

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4TextDecoderLayer]
    @ModuleInfo(key: "norm") var norm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm:
        Gemma4RMSNormZeroShift?

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.firstKVSharedLayerIdx = config.hiddenLayers - config.numKVSharedLayers
        self.embedScale = pow(Float(config.hiddenSize), 0.5)
        self.embedTokensPerLayerScale = pow(Float(max(config.hiddenSizePerLayerInput, 1)), 0.5)
        self._perLayerInputScale = rsqrt(MLXArray(2.0))

        let concreteLayers = Array(config.layerTypes.prefix(firstKVSharedLayerIdx))
        let sharedFullIdx = concreteLayers.lastIndex(of: "full_attention") ?? 0
        let sharedSlidingIdx = concreteLayers.lastIndex(of: "sliding_attention") ?? 0

        var cacheMap: [Int] = []
        cacheMap.reserveCapacity(config.hiddenLayers)
        for (idx, layerType) in config.layerTypes.enumerated() {
            if idx < firstKVSharedLayerIdx {
                cacheMap.append(idx)
            } else {
                cacheMap.append(layerType == "full_attention" ? sharedFullIdx : sharedSlidingIdx)
            }
        }
        layerIdxToCacheIdx = cacheMap
        firstFullCacheIdx = concreteLayers.firstIndex(of: "full_attention") ?? 0
        firstSlidingCacheIdx = concreteLayers.firstIndex(of: "sliding_attention") ?? 0

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        // KV-shared tail layers (`layer_idx >= hidden_layers - num_kv_shared_layers`)
        // reuse an earlier same-type layer's K/V and own no local K/V projection or
        // K norm. Build them with `kvSharedOnly: true` so the module tree omits those
        // tensors — matching what `sanitize` drops from the checkpoint, and mirroring
        // both `Gemma4AssistantDraftInner` and upstream mlx-lm (`Attention.has_kv`).
        // Without this, shared layers declare a `v_proj`/`k_proj`/`k_norm` that no
        // weight fills, so loading a Gemma 4 checkpoint fails under the strict loader
        // verify with e.g. `keyNotFound(… layers.24.self_attn.v_proj.weight …)`.
        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { idx in
            Gemma4TextDecoderLayer(
                config: config, layerIdx: idx,
                kvSharedOnly: firstKVSharedLayer > 0 && idx >= firstKVSharedLayer)
        }
        self._norm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if config.hiddenSizePerLayerInput > 0 {
            self.perLayerProjectionScale = pow(Float(config.hiddenSize), -0.5)
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabularySizePerLayerInput,
                dimensions: config.hiddenLayers * config.hiddenSizePerLayerInput
            )
            self._perLayerModelProjection.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenLayers * config.hiddenSizePerLayerInput,
                bias: false
            )
            self._perLayerProjectionNorm.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
        } else {
            self.perLayerProjectionScale = 1.0
        }

        super.init()
    }

    func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray {
        guard let embedTokensPerLayer else {
            fatalError("Per-layer inputs requested for a model without embed_tokens_per_layer")
        }
        let validMask =
            logicalAnd(
                inputIds .>= 0, inputIds .< config.vocabularySizePerLayerInput)
        let tokens = MLX.where(validMask, inputIds, MLXArray.zeros(like: inputIds))
        var result = embedTokensPerLayer(tokens)
        result = (result * MLXArray(embedTokensPerLayerScale, dtype: .float32)).asType(result.dtype)
        return result.reshaped(
            Array(inputIds.shape) + [config.hiddenLayers, config.hiddenSizePerLayerInput]
        )
    }

    func projectPerLayerInputs(
        _ inputsEmbeds: MLXArray, perLayerInputs: MLXArray?
    ) -> MLXArray? {
        guard let perLayerModelProjection, let perLayerProjectionNorm else {
            return nil
        }

        var perLayerProjection = perLayerModelProjection(inputsEmbeds) * perLayerProjectionScale
        perLayerProjection = perLayerProjection.reshaped(
            Array(inputsEmbeds.shape.dropLast()) + [
                config.hiddenLayers, config.hiddenSizePerLayerInput,
            ]
        )
        perLayerProjection = perLayerProjectionNorm(perLayerProjection)

        guard let perLayerInputs else {
            return perLayerProjection
        }

        return (perLayerProjection + perLayerInputs)
            * _perLayerInputScale.asType(inputsEmbeds.dtype)
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil,
        perLayerInputs: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil,
        emitDrafterState: Bool = false
    ) -> (
        hidden: MLXArray, sharedKV: [String: (MLXArray, MLXArray)]?,
        sharedKVSources: [String: Int]
    ) {
        // Tolerate callers that hand us a 1D `(L,)` token array instead
        // of the canonical 2D `(B, L)` produced by `Gemma4Processor.prepare`.
        // The downstream `perLayerInputs` indexing path (`finalPerLayerInputs[
        // 0..., 0..., idx, 0...]`) requires 4D shapes; with 1D inputs the
        // model otherwise crashes inside `MLXArray.subscript.getter`
        // → `mlx_array_dim` → `_mlx_error`. This expansion is zero-copy
        // and behaves identically when the caller already passed 2D.
        let inputs = inputs.map { $0.ndim == 1 ? $0.expandedDimensions(axis: 0) : $0 }
        let inputsEmbeds = inputsEmbeds.map {
            $0.ndim == 2 ? $0.expandedDimensions(axis: 0) : $0
        }

        let h0: MLXArray
        if let inputsEmbeds {
            h0 = inputsEmbeds
        } else if let inputs {
            let embeddings = embedTokens(inputs)
            h0 = (embeddings * MLXArray(embedScale, dtype: .float32)).asType(embeddings.dtype)
        } else {
            fatalError("Either inputs or inputsEmbeds must be provided")
        }

        let processedPerLayerInputs: MLXArray?
        if config.hiddenSizePerLayerInput > 0 {
            if let perLayerInputs {
                processedPerLayerInputs = perLayerInputs
            } else if let inputs {
                processedPerLayerInputs = getPerLayerInputs(inputs)
            } else {
                processedPerLayerInputs = nil
            }
        } else {
            processedPerLayerInputs = nil
        }
        let finalPerLayerInputs = projectPerLayerInputs(h0, perLayerInputs: processedPerLayerInputs)

        let localCache =
            cache ?? Array(repeating: nil as KVCache?, count: max(firstKVSharedLayerIdx, 1))
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode
        var slidingMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let mask {
            fullMask = mask
            slidingMask = mask
        } else {
            let tokenTypeIds = tokenTypeIds.map {
                $0.ndim == 1 ? $0.expandedDimensions(axis: 0) : $0
            }
            let hasAudioTokens =
                if let tokenTypeIds {
                    (tokenTypeIds .== 3).asType(.int32).sum().item(Int.self) > 0
                } else {
                    false
                }
            let hasVisualTokens =
                if let tokenTypeIds {
                    ((tokenTypeIds .== 1) | (tokenTypeIds .== 2))
                        .asType(.int32).sum().item(Int.self) > 0
                } else {
                    false
                }
            let useBidirectionalVision =
                config.useBidirectionalAttention == "vision"
                && tokenTypeIds != nil
                && hasVisualTokens
                && !hasAudioTokens
                && h0.dim(1) > 1

            fullMask = createAttentionMask(
                h: h0,
                cache: firstFullCacheIdx < localCache.count ? localCache[firstFullCacheIdx] : nil,
                returnArray: useBidirectionalVision)
            slidingMask = createAttentionMask(
                h: h0,
                cache: firstSlidingCacheIdx < localCache.count
                    ? localCache[firstSlidingCacheIdx] : nil,
                windowSize: config.slidingWindow,
                returnArray: useBidirectionalVision
            )
            if useBidirectionalVision, let tokenTypeIds {
                fullMask = gemma4ApplyBlockwiseBidirectionalOverlay(
                    fullMask,
                    tokenTypeIds: tokenTypeIds,
                    sequenceLength: h0.dim(1),
                    windowSize: nil
                )
                slidingMask = gemma4ApplyBlockwiseBidirectionalOverlay(
                    slidingMask,
                    tokenTypeIds: tokenTypeIds,
                    sequenceLength: h0.dim(1),
                    windowSize: config.slidingWindow
                )
            }
        }

        var h = h0
        var intermediates = [(kv: Gemma4SharedKVState?, offset: Int?)](
            repeating: (nil, nil), count: config.hiddenLayers)
        for (idx, layer) in layers.enumerated() {
            let sourceIdx = layerIdxToCacheIdx[idx]
            let layerCache: KVCache? =
                if idx < firstKVSharedLayerIdx, sourceIdx < localCache.count {
                    localCache[sourceIdx]
                } else {
                    nil
                }
            let layerMask =
                if layer.layerType == "full_attention" {
                    fullMask
                } else {
                    slidingMask
                }
            let layerInput: MLXArray? =
                if let finalPerLayerInputs {
                    finalPerLayerInputs[0..., 0..., idx, 0...]
                } else {
                    nil
                }
            let (output, kvState, attentionOffset) = layer(
                h,
                mask: layerMask,
                cache: layerCache,
                perLayerInput: layerInput,
                sharedKV: idx >= firstKVSharedLayerIdx
                    ? intermediates[sourceIdx].kv : nil,
                offset: idx >= firstKVSharedLayerIdx
                    ? intermediates[sourceIdx].offset : nil
            )
            h = output
            intermediates[idx] = (kvState, attentionOffset)
        }
        let finalHidden = norm(h)

        guard emitDrafterState else {
            return (finalHidden, nil, [:])
        }

        // Walk intermediates from the last layer backward; for each unique
        // `layer_type`, take the first `.regular` K/V encountered. Quantized
        // cases are skipped — the iterator treats absent `sharedKV` as a
        // signal to fall back to single-token generation (R8/R13 limitation,
        // documented).
        var sharedKV: [String: (MLXArray, MLXArray)] = [:]
        // Which cache entry each emitted tuple came from. The consumer reconciles the emitted
        // snapshot against the cache after a speculative commit, and it can only do that exactly
        // if it knows the entry -- a sliding layer's snapshot is bounded by its ring, a global
        // layer's is not, and the two are indistinguishable by length at the crossing.
        var sharedKVSources: [String: Int] = [:]
        var seenTypes = Set<String>()
        let targetTypes: Set<String> = ["full_attention", "sliding_attention"]
        for idx in stride(from: layers.count - 1, through: 0, by: -1) {
            let layerType = layers[idx].layerType
            guard targetTypes.contains(layerType), !seenTypes.contains(layerType) else {
                continue
            }
            if case .regular(let keys, let values) = intermediates[idx].kv {
                sharedKV[layerType] = (keys, values)
                // Recorded here rather than derived from `config.layerTypes`: the walk keeps
                // descending past a quantized entry, so which layer supplies a type is a runtime
                // fact.
                sharedKVSources[layerType] = layerIdxToCacheIdx[idx]
                seenTypes.insert(layerType)
            }
            if seenTypes == targetTypes { break }
        }
        // Treat partial coverage (e.g. only one layer_type populated, or
        // quantized cache for the other) as no-emit — iterator falls back.
        let complete = seenTypes == targetTypes
        return (finalHidden, complete ? sharedKV : nil, complete ? sharedKVSources : [:])
    }
}

/// Module-internal — also consumed by `Gemma4Assistant.swift` (the MTP drafter
/// reaches `embed_tokens` / `embed_scale` / `config.layer_types` through this).
final class Gemma4TextLanguageModel: Module, KVCacheDimensionProvider {
    let config: Gemma4TextConfiguration
    let finalLogitSoftcapping: Float?

    @ModuleInfo(key: "model") var model: Gemma4TextBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    var kvHeads: [Int] {
        (0 ..< config.hiddenLayers).map { idx in
            let layerType = config.layerTypes[idx]
            if config.attentionKEqV && layerType == "full_attention" {
                return config.globalKVHeads ?? config.kvHeads
            } else {
                return config.kvHeads
            }
        }
    }

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.finalLogitSoftcapping = config.finalLogitSoftcapping
        self._model.wrappedValue = Gemma4TextBackbone(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
        }
        super.init()
    }

    func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        let slidingWindow = config.slidingWindow > 0 ? config.slidingWindow : 4096
        return try config.layerTypes.prefix(config.hiddenLayers - config.numKVSharedLayers).map {
            layerType in
            try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: slidingWindow,
                usesSlidingWindow: layerType != "full_attention")
        }
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        cache: [KVCache]? = nil,
        inputsEmbeds: MLXArray? = nil,
        perLayerInputs: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        emitDrafterState: Bool = false
    ) -> LMOutput {
        let (hidden, sharedKV, sharedKVSources) = model(
            inputs, inputsEmbeds: inputsEmbeds, mask: mask, cache: cache?.map { $0 as KVCache? },
            perLayerInputs: perLayerInputs,
            tokenTypeIds: tokenTypeIds,
            emitDrafterState: emitDrafterState
        )
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(hidden)
        } else {
            logits = model.embedTokens.asLinear(hidden)
        }
        let softcappedLogits: MLXArray
        if let finalLogitSoftcapping, finalLogitSoftcapping > 0 {
            let scale = MLXArray(finalLogitSoftcapping)
            softcappedLogits = gemma4LogitSoftcap(logits, scale)
        } else {
            softcappedLogits = logits
        }

        guard emitDrafterState, let sharedKV else {
            return LMOutput(logits: softcappedLogits)
        }
        var state = LMOutput.State()
        state[mtpLastHiddenStatesKey] = hidden
        state[mtpSharedKVStatesKey] = sharedKV
        state[mtpSharedKVSourceIndicesKey] = sharedKVSources
        return LMOutput(logits: softcappedLogits, state: state)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count + 1)

        for (key, value) in weights {
            if key.contains("rotary_emb") {
                continue
            }
            // Drop redundant k_proj/v_proj/k_norm for KV-shared layers: they reuse an
            // earlier layer's K/V and own no K projection or K norm, so the module tree
            // has none. QAT checkpoints already omit these; some (PTQ) checkpoints still
            // ship them, and keeping them would be an unexpected weight.
            // Scope: text backbone only — the vision/audio towers share the
            // `layers.N.self_attn.{k,v}_proj` naming, so without these guards the drop
            // would amputate tower layers >= firstKVSharedLayer.
            if firstKVSharedLayer > 0,
                !key.contains("vision_tower"),
                !key.contains("audio_tower"),
                key.contains("self_attn.k_proj")
                    || key.contains("self_attn.v_proj")
                    || key.contains("self_attn.k_norm"),
                let layerIdx = Self.decoderLayerIndex(in: key),
                layerIdx >= firstKVSharedLayer
            {
                continue
            }

            var newKey = key
            if newKey.hasPrefix("model.") {
                newKey.removeFirst("model.".count)
            }
            if newKey.hasPrefix("language_model."),
                !newKey.hasPrefix("language_model.model."),
                !newKey.hasPrefix("language_model.lm_head.")
            {
                let rest = String(newKey.dropFirst("language_model.".count))
                newKey = "language_model.model.\(rest)"
            }

            if newKey.hasSuffix(".experts.down_proj") {
                newKey = newKey.replacingOccurrences(
                    of: ".experts.down_proj",
                    with: ".experts.switch_glu.down_proj.weight"
                )
            }

            if newKey.hasSuffix(".experts.gate_up_proj") {
                let mid = value.dim(-2) / 2
                sanitized[
                    newKey.replacingOccurrences(
                        of: ".experts.gate_up_proj",
                        with: ".experts.switch_glu.gate_proj.weight"
                    )
                ] = value[.ellipsis, ..<mid, 0...]
                sanitized[
                    newKey.replacingOccurrences(
                        of: ".experts.gate_up_proj",
                        with: ".experts.switch_glu.up_proj.weight"
                    )
                ] = value[.ellipsis, mid..., 0...]
                continue
            }

            sanitized[newKey] = value
        }

        if config.tieWordEmbeddings {
            sanitized = sanitized.filter { key, _ in
                !key.hasPrefix("language_model.lm_head.")
            }
        } else if sanitized["language_model.lm_head.weight"] == nil,
            let embedWeight = sanitized["language_model.model.embed_tokens.weight"]
        {
            sanitized["language_model.lm_head.weight"] = embedWeight
        }

        return sanitized
    }

    /// Extract `N` from a weight key shaped like `…layers.N.…`, else nil.
    private static func decoderLayerIndex(in key: String) -> Int? {
        guard let range = key.range(of: "layers.") else { return nil }
        let digits = key[range.upperBound...].prefix { $0.isNumber }
        return Int(digits)
    }
}

// Shared encoders keep the existing MLXVLM names available.
public typealias Gemma4VisionConfiguration = MLXLMCommon.Gemma4VisionConfiguration
package typealias Gemma4AudioConfiguration = MLXLMCommon.Gemma4AudioConfiguration
package typealias Gemma4VisionModel = MLXLMCommon.Gemma4VisionModel
package typealias Gemma4AudioModel = MLXLMCommon.Gemma4AudioModel
package typealias Gemma4MultimodalEmbedder = MLXLMCommon.Gemma4MultimodalEmbedder

// MARK: - Gemma 4 Unified

public struct Gemma4UnifiedAudioConfiguration: Codable, Sendable {
    public let modelType: String
    public let audioSamplesPerToken: Int
    public let audioEmbedDim: Int
    public let hiddenSize: Int
    public let outputProjectionDimensions: Int
    public let rmsNormEps: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case audioSamplesPerToken = "audio_samples_per_token"
        case audioEmbedDim = "audio_embed_dim"
        case hiddenSize = "hidden_size"
        case outputProjectionDimensions = "output_proj_dims"
        case rmsNormEps = "rms_norm_eps"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType)
            ?? "gemma4_unified_audio"
        audioSamplesPerToken =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioSamplesPerToken) ?? 640
        audioEmbedDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioEmbedDim) ?? 640
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenSize) ?? 640
        outputProjectionDimensions =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.outputProjectionDimensions) ?? 640
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: CodingKeys.rmsNormEps) ?? 1e-6
    }
}

public struct Gemma4UnifiedVisionConfiguration: Codable, Sendable {
    public let modelType: String
    public let patchSize: Int
    public let poolingKernelSize: Int
    public let modelPatchSize: Int
    public let mmEmbedDim: Int
    public let mmPositionEmbeddingSize: Int
    public let numSoftTokens: Int
    public let rmsNormEps: Float
    public let outputProjectionDimensions: Int

    public var hiddenSize: Int { outputProjectionDimensions }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case patchSize = "patch_size"
        case poolingKernelSize = "pooling_kernel_size"
        case modelPatchSize = "model_patch_size"
        case mmEmbedDim = "mm_embed_dim"
        case mmPositionEmbeddingSize = "mm_posemb_size"
        case numSoftTokens = "num_soft_tokens"
        case rmsNormEps = "rms_norm_eps"
        case outputProjectionDimensions = "output_proj_dims"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType)
            ?? "gemma4_unified_vision"
        patchSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys.patchSize) ?? 16
        poolingKernelSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.poolingKernelSize) ?? 3
        modelPatchSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.modelPatchSize)
            ?? patchSize * poolingKernelSize
        mmEmbedDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.mmEmbedDim) ?? 3840
        mmPositionEmbeddingSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.mmPositionEmbeddingSize) ?? 1120
        numSoftTokens =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.numSoftTokens) ?? 280
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: CodingKeys.rmsNormEps) ?? 1e-6
        outputProjectionDimensions =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.outputProjectionDimensions) ?? 3840
    }
}

public struct Gemma4UnifiedConfiguration: Decodable, Sendable {
    public let textConfiguration: Gemma4TextConfiguration
    public let visionConfiguration: Gemma4UnifiedVisionConfiguration?
    public let audioConfiguration: Gemma4UnifiedAudioConfiguration?
    public let modelType: String
    public let quantization: BaseConfiguration.Quantization?
    public let imageTokenId: Int
    public let audioTokenId: Int?
    public let videoTokenId: Int?
    public let boiTokenId: Int
    public let eoiTokenId: Int?
    public let boaTokenId: Int
    public let eoaTokenId: Int?
    public let visionSoftTokensPerImage: Int
    public let visionSoftTokensPerVideoFrame: Int
    public let audioSoftTokensPerImage: Int
    public let audioMsPerToken: Int
    public let tieWordEmbeddings: Bool

    private let _vocabularySize: Int?
    private let _hiddenSize: Int?
    private let _padTokenId: Int?

    public var vocabularySize: Int { _vocabularySize ?? textConfiguration.vocabularySize }
    public var hiddenSize: Int { _hiddenSize ?? textConfiguration.hiddenSize }
    public var padTokenId: Int { _padTokenId ?? 0 }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case audioConfiguration = "audio_config"
        case modelType = "model_type"
        case quantization
        case imageTokenId = "image_token_id"
        case audioTokenId = "audio_token_id"
        case videoTokenId = "video_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case boaTokenId = "boa_token_id"
        case eoaTokenId = "eoa_token_id"
        case eoaTokenIndex = "eoa_token_index"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case visionSoftTokensPerVideoFrame = "vision_soft_tokens_per_video_frame"
        case audioSoftTokensPerImage = "audio_soft_tokens_per_image"
        case audioMsPerToken = "audio_ms_per_token"
        case tieWordEmbeddings = "tie_word_embeddings"
        case _vocabularySize = "vocab_size"
        case _hiddenSize = "hidden_size"
        case _padTokenId = "pad_token_id"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration =
            try c.decodeIfPresent(
                Gemma4TextConfiguration.self, forKey: CodingKeys.textConfiguration)
            ?? Gemma4TextConfiguration(from: decoder)
        visionConfiguration = try c.decodeIfPresent(
            Gemma4UnifiedVisionConfiguration.self, forKey: CodingKeys.visionConfiguration)
        audioConfiguration = try c.decodeIfPresent(
            Gemma4UnifiedAudioConfiguration.self, forKey: CodingKeys.audioConfiguration)
        modelType =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType) ?? "gemma4_unified"
        quantization = try c.decodeIfPresent(
            BaseConfiguration.Quantization.self, forKey: CodingKeys.quantization)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageTokenId) ?? 258_880
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioTokenId) ?? 258_881
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.videoTokenId) ?? 258_884
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.boiTokenId) ?? 255_999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.eoiTokenId) ?? 258_882
        boaTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.boaTokenId) ?? 256_000
        eoaTokenId =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.eoaTokenId)
            ?? c.decodeIfPresent(Int.self, forKey: CodingKeys.eoaTokenIndex)
        visionSoftTokensPerImage =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.visionSoftTokensPerImage) ?? 280
        visionSoftTokensPerVideoFrame =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.visionSoftTokensPerVideoFrame) ?? 70
        audioSoftTokensPerImage =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioSoftTokensPerImage) ?? 750
        audioMsPerToken =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioMsPerToken) ?? 40
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.tieWordEmbeddings) ?? true
        _vocabularySize = try c.decodeIfPresent(Int.self, forKey: CodingKeys._vocabularySize)
        _hiddenSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys._hiddenSize)
        _padTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys._padTokenId)
    }
}

private final class Gemma4UnifiedVisionEmbedder: Module {
    let patchDim: Int

    @ModuleInfo(key: "patch_ln1") var patchLayerNorm1: LayerNorm
    @ModuleInfo(key: "patch_dense") var patchDense: Linear
    @ModuleInfo(key: "patch_ln2") var patchLayerNorm2: LayerNorm
    @ParameterInfo(key: "pos_embedding") var positionEmbedding: MLXArray
    @ModuleInfo(key: "pos_norm") var positionNorm: LayerNorm

    init(config: Gemma4UnifiedVisionConfiguration) {
        self.patchDim = config.modelPatchSize * config.modelPatchSize * 3
        self._patchLayerNorm1.wrappedValue = LayerNorm(dimensions: patchDim)
        self._patchDense.wrappedValue = Linear(patchDim, config.mmEmbedDim)
        self._patchLayerNorm2.wrappedValue = LayerNorm(dimensions: config.mmEmbedDim)
        self._positionEmbedding.wrappedValue = MLXArray.zeros([
            config.mmPositionEmbeddingSize, 2, config.mmEmbedDim,
        ])
        self._positionNorm.wrappedValue = LayerNorm(dimensions: config.mmEmbedDim)
        super.init()
    }

    func callAsFunction(_ pixelValues: MLXArray, imagePositionIds: MLXArray? = nil) -> MLXArray {
        var pixels = pixelValues
        if pixels.ndim == 4 && pixels.dim(-1) == patchDim {
            pixels = pixels.reshaped(pixels.dim(0), -1, patchDim)
        }

        var hiddenStates = patchLayerNorm1(pixels)
        hiddenStates = patchDense(hiddenStates)
        hiddenStates = patchLayerNorm2(hiddenStates)

        if let imagePositionIds {
            let clamped = maximum(imagePositionIds, MLXArray(0)).asType(.int32)
            let batch = clamped.dim(0)
            let sequenceLength = clamped.dim(1)
            let xIndices = clamped[0..., 0..., 0].flattened()
            let yIndices = clamped[0..., 0..., 1].flattened()
            let xEmbeddings = take(positionEmbedding[0..., 0], xIndices, axis: 0)
                .reshaped(batch, sequenceLength, -1)
            let yEmbeddings = take(positionEmbedding[0..., 1], yIndices, axis: 0)
                .reshaped(batch, sequenceLength, -1)
            let valid = (imagePositionIds .!= -1).asType(hiddenStates.dtype)
            hiddenStates =
                hiddenStates
                + xEmbeddings * expandedDimensions(valid[0..., 0..., 0], axis: -1)
                + yEmbeddings * expandedDimensions(valid[0..., 0..., 1], axis: -1)
        }

        return positionNorm(hiddenStates)
    }
}

public final class Gemma4Unified: Module, VLMModel, KVCacheDimensionProvider,
    SuppressedTokensProviding
{
    @ModuleInfo(key: "language_model") private var languageModel: Gemma4TextLanguageModel
    @ModuleInfo(key: "vision_embedder") private var visionEmbedder: Gemma4UnifiedVisionEmbedder?
    @ModuleInfo(key: "embed_vision") private var embedVision: Gemma4MultimodalEmbedder?
    @ModuleInfo(key: "embed_audio") private var embedAudio: Gemma4MultimodalEmbedder?

    public let config: Gemma4UnifiedConfiguration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.model.layers }

    /// Multimodal placeholder token IDs that must never be sampled as text.
    /// Seeded from the model configuration; the model factory merges
    /// `suppress_tokens` from `generation_config.json` on top.
    /// See `SuppressedTokensProviding` in MLXLMCommon.
    public var suppressedTokenIds: Set<Int>

    public init(_ config: Gemma4UnifiedConfiguration) {
        self.config = config
        self.suppressedTokenIds = Set(
            [
                config.imageTokenId, config.audioTokenId, config.videoTokenId,
                config.boiTokenId, config.eoiTokenId, config.boaTokenId, config.eoaTokenId,
            ].compactMap { $0 })
        self._languageModel.wrappedValue = Gemma4TextLanguageModel(config.textConfiguration)
        if let visionConfiguration = config.visionConfiguration {
            self._visionEmbedder.wrappedValue = Gemma4UnifiedVisionEmbedder(
                config: visionConfiguration)
            self._embedVision.wrappedValue = Gemma4MultimodalEmbedder(
                embeddingDim: visionConfiguration.outputProjectionDimensions,
                textHiddenSize: config.textConfiguration.hiddenSize,
                eps: visionConfiguration.rmsNormEps
            )
        }
        if let audioConfiguration = config.audioConfiguration {
            self._embedAudio.wrappedValue = Gemma4MultimodalEmbedder(
                embeddingDim: audioConfiguration.outputProjectionDimensions,
                textHiddenSize: config.textConfiguration.hiddenSize,
                eps: audioConfiguration.rmsNormEps
            )
        }
        super.init()
    }

    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        try languageModel.newCache(parameters: parameters)
    }

    private func getImageFeatures(
        pixelValues: MLXArray,
        imagePositionIds: MLXArray?
    ) throws -> MLXArray {
        guard let visionEmbedder, let embedVision else {
            throw VLMError.processing("Vision inputs were provided, but vision_config is missing.")
        }
        var projected = embedVision(visionEmbedder(pixelValues, imagePositionIds: imagePositionIds))
        if let imagePositionIds {
            let validMask =
                (imagePositionIds[0..., 0..., 0] .!= -1)
                | (imagePositionIds[0..., 0..., 1] .!= -1)
            projected = gemma4CompactPrefixRows(features: projected, validMask: validMask)
        } else {
            projected = projected.reshaped(-1, projected.dim(-1))
        }
        return projected
    }

    private func getAudioFeatures(features: MLXArray, mask: MLXArray?) throws -> MLXArray {
        guard let embedAudio else {
            throw VLMError.processing("Audio inputs were provided, but audio_config is missing.")
        }
        var projected = embedAudio(features)
        if let mask {
            projected = gemma4CompactPrefixRows(features: projected, validMask: mask)
        } else {
            projected = projected.reshaped(-1, projected.dim(-1))
        }
        return projected
    }

    private func scatterFeatures(
        inputEmbeds: MLXArray,
        inputIds: MLXArray,
        tokenId: Int?,
        features: MLXArray?,
        kind: String
    ) throws -> MLXArray {
        guard let tokenId, let features else {
            return inputEmbeds
        }
        let tokenMask = inputIds .== tokenId
        let expectedTokens = tokenMask.asType(.int32).sum().item(Int.self)
        guard expectedTokens == features.dim(0) else {
            throw Gemma4Error.multimodalTokenCountMismatch(
                kind: kind, featureTokens: features.dim(0), promptTokens: expectedTokens)
        }
        var expandedMask = expandedDimensions(tokenMask, axis: -1)
        expandedMask = broadcast(expandedMask, to: inputEmbeds.shape)
        return gemma4MaskedScatter(
            inputTensor: inputEmbeds,
            mask: expandedMask,
            source: features.asType(inputEmbeds.dtype)
        )
    }

    private func getInputEmbeddings(
        inputIds: MLXArray,
        pixelValues: MLXArray? = nil,
        imagePositionIds: MLXArray? = nil,
        videoPixelValues: MLXArray? = nil,
        videoPositionIds: MLXArray? = nil,
        audioFeatures: MLXArray? = nil,
        audioMask: MLXArray? = nil
    ) throws -> (MLXArray, MLXArray?) {
        var inputsEmbeds = languageModel.model.embedTokens(inputIds)
        inputsEmbeds =
            (inputsEmbeds
            * MLXArray(pow(Float(config.textConfiguration.hiddenSize), 0.5), dtype: .float32))
            .asType(inputsEmbeds.dtype)

        var perLayerInputs: MLXArray? = nil
        if config.textConfiguration.hiddenSizePerLayerInput > 0 {
            var multimodalMask = inputIds .== config.imageTokenId
            if let audioTokenId = config.audioTokenId {
                multimodalMask = multimodalMask | (inputIds .== audioTokenId)
            }
            if let videoTokenId = config.videoTokenId {
                multimodalMask = multimodalMask | (inputIds .== videoTokenId)
            }
            let perLayerTokens = MLX.where(
                logicalNot(multimodalMask), inputIds, MLXArray.zeros(like: inputIds))
            perLayerInputs = languageModel.model.getPerLayerInputs(perLayerTokens)
        }

        let imageFeatures =
            if let pixelValues {
                try getImageFeatures(pixelValues: pixelValues, imagePositionIds: imagePositionIds)
            } else {
                nil as MLXArray?
            }
        inputsEmbeds = try scatterFeatures(
            inputEmbeds: inputsEmbeds,
            inputIds: inputIds,
            tokenId: config.imageTokenId,
            features: imageFeatures,
            kind: "image"
        )

        let videoFeatures =
            if let videoPixelValues {
                try getImageFeatures(
                    pixelValues: videoPixelValues, imagePositionIds: videoPositionIds)
            } else {
                nil as MLXArray?
            }
        inputsEmbeds = try scatterFeatures(
            inputEmbeds: inputsEmbeds,
            inputIds: inputIds,
            tokenId: config.videoTokenId,
            features: videoFeatures,
            kind: "video"
        )

        let audioFeatures =
            if let audioFeatures {
                try getAudioFeatures(features: audioFeatures, mask: audioMask)
            } else {
                nil as MLXArray?
            }
        inputsEmbeds = try scatterFeatures(
            inputEmbeds: inputsEmbeds,
            inputIds: inputIds,
            tokenId: config.audioTokenId,
            features: audioFeatures,
            kind: "audio"
        )

        return (inputsEmbeds, perLayerInputs)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?, prefill: PrefillParameters
    ) throws
        -> PrepareResult
    {
        if input.image == nil, input.video == nil, input.audio == nil {
            return try gemma4PrepareTextOnly(
                input, cache: cache, prefill: prefill, languageModel: languageModel)
        }

        let (inputsEmbeds, perLayerInputs) = try getInputEmbeddings(
            inputIds: input.text.tokens,
            pixelValues: input.image?.pixels,
            imagePositionIds: input.image?.positionIds,
            videoPixelValues: input.video?.pixels,
            videoPositionIds: input.video?.positionIds,
            audioFeatures: input.audio?.features,
            audioMask: input.audio?.mask
        )
        let tokenTypeIds = gemma4TokenTypeIds(
            inputIds: input.text.tokens,
            imageTokenId: config.imageTokenId,
            videoTokenId: config.videoTokenId,
            audioTokenId: config.audioTokenId
        )
        let result = languageModel(
            nil,
            cache: cache.map { $0 },
            inputsEmbeds: inputsEmbeds,
            perLayerInputs: perLayerInputs,
            tokenTypeIds: tokenTypeIds
        )
        let total = inputsEmbeds.dim(1)
        prefill.progress?(total, total)
        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let logits = languageModel(inputs, cache: cache?.map { $0 })
        return logits.logits
    }

    /// MTP-aware `LanguageModel` entry point. Reads `mtpEmitFlagKey` from
    /// the incoming `state` and threads it through to `Gemma4TextLanguageModel`;
    /// the returned `LMOutput` carries `mtpLastHiddenStatesKey` and
    /// `mtpSharedKVStatesKey` populated when the flag is set, empty otherwise.
    /// Overrides the protocol-extension default at `LanguageModel` which
    /// would discard `state`. Mirrors `Gemma4.callAsFunction(_:cache:state:)`.
    public func callAsFunction(
        _ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let emit = state?[mtpEmitFlagKey] ?? false
        return languageModel(
            input.tokens, cache: cache?.map { $0 },
            emitDrafterState: emit
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var filtered: [String: MLXArray] = [:]
        filtered.reserveCapacity(weights.count)
        for (key, value) in weights {
            if key == "lm_head.weight" {
                continue
            }
            if embedAudio == nil && key.contains("embed_audio") {
                continue
            }
            if visionEmbedder == nil
                && (key.contains("vision_embedder") || key.contains("embed_vision"))
            {
                continue
            }
            filtered[key] = value
        }
        return languageModel.sanitize(weights: filtered)
    }
}

// MARK: - MTP drafter target access

/// A Gemma 4 - family VLM whose text stack is the shared
/// `Gemma4TextBackbone`. The MTP drafter
/// (`Gemma4AssistantDraftModel.draftBlock`) casts its `target` to this
/// protocol instead of dispatching on concrete classes, so it stays agnostic
/// of which Gemma 4 variant wraps the backbone.
///
/// The fork's `Gemma4` uses the MLXLLM text tower and does not conform. Gemma 4
/// MTP for that model runs through the MLXLLM drafter path.
protocol Gemma4BackboneProviding {
    /// The text backbone whose `embedTokens` / `embedScale` the drafter
    /// shares with the target.
    var textBackbone: Gemma4TextBackbone { get }
}

extension Gemma4Unified: Gemma4BackboneProviding {
    var textBackbone: Gemma4TextBackbone { languageModel.model }
}

// MARK: - Gemma 4 Unified processor

public struct Gemma4UnifiedProcessorConfiguration: Decodable, Sendable {
    public let processorClass: String
    public let doResize: Bool
    public let doRescale: Bool
    public let rescaleFactor: CGFloat
    public let doNormalize: Bool
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let imageSeqLength: Int
    public let audioSeqLength: Int
    public let audioMsPerToken: Int
    public let patchSize: Int
    public let poolingKernelSize: Int
    public let modelPatchSize: Int
    public let maxSoftTokens: Int
    public let size: Gemma3ProcessorConfiguration.ImageSize?
    public let imageTokenId: Int
    public let audioTokenId: Int
    public let videoTokenId: Int?
    public let boiTokenId: Int
    public let eoiTokenId: Int?

    private struct ImageProcessorConfiguration: Decodable, Sendable {
        let doResize: Bool?
        let doRescale: Bool?
        let rescaleFactor: CGFloat?
        let doNormalize: Bool?
        let imageMean: [CGFloat]?
        let imageStd: [CGFloat]?
        let patchSize: Int?
        let poolingKernelSize: Int?
        let modelPatchSize: Int?
        let maxSoftTokens: Int?
        let numSoftTokens: Int?
        let size: Gemma3ProcessorConfiguration.ImageSize?

        enum CodingKeys: String, CodingKey {
            case doResize = "do_resize"
            case doRescale = "do_rescale"
            case rescaleFactor = "rescale_factor"
            case doNormalize = "do_normalize"
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case patchSize = "patch_size"
            case poolingKernelSize = "pooling_kernel_size"
            case modelPatchSize = "model_patch_size"
            case maxSoftTokens = "max_soft_tokens"
            case numSoftTokens = "num_soft_tokens"
            case size
        }
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageProcessor = "image_processor"
        case doResize = "do_resize"
        case doRescale = "do_rescale"
        case rescaleFactor = "rescale_factor"
        case doNormalize = "do_normalize"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case imageSeqLength = "image_seq_length"
        case audioSeqLength = "audio_seq_length"
        case audioMsPerToken = "audio_ms_per_token"
        case patchSize = "patch_size"
        case poolingKernelSize = "pooling_kernel_size"
        case modelPatchSize = "model_patch_size"
        case maxSoftTokens = "max_soft_tokens"
        case numSoftTokens = "num_soft_tokens"
        case size
        case imageTokenId = "image_token_id"
        case audioTokenId = "audio_token_id"
        case videoTokenId = "video_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let imageProcessor = try c.decodeIfPresent(
            ImageProcessorConfiguration.self, forKey: CodingKeys.imageProcessor)

        processorClass =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.processorClass)
            ?? "Gemma4UnifiedProcessor"
        doResize =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.doResize)
            ?? imageProcessor?.doResize
            ?? true
        doRescale =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.doRescale)
            ?? imageProcessor?.doRescale
            ?? true
        rescaleFactor =
            try c.decodeIfPresent(CGFloat.self, forKey: CodingKeys.rescaleFactor)
            ?? imageProcessor?.rescaleFactor
            ?? (1.0 / 255.0)
        doNormalize =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.doNormalize)
            ?? imageProcessor?.doNormalize
            ?? false
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: CodingKeys.imageMean)
            ?? imageProcessor?.imageMean
            ?? [0.5, 0.5, 0.5]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: CodingKeys.imageStd)
            ?? imageProcessor?.imageStd
            ?? [0.5, 0.5, 0.5]
        patchSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.patchSize)
            ?? imageProcessor?.patchSize
            ?? 16
        poolingKernelSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.poolingKernelSize)
            ?? imageProcessor?.poolingKernelSize
            ?? 3
        modelPatchSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.modelPatchSize)
            ?? imageProcessor?.modelPatchSize
            ?? patchSize * poolingKernelSize
        maxSoftTokens =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.maxSoftTokens)
            ?? c.decodeIfPresent(Int.self, forKey: CodingKeys.numSoftTokens)
            ?? imageProcessor?.maxSoftTokens
            ?? imageProcessor?.numSoftTokens
            ?? 280
        imageSeqLength =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageSeqLength) ?? maxSoftTokens
        audioSeqLength =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioSeqLength) ?? 750
        audioMsPerToken =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioMsPerToken) ?? 40
        size =
            try c.decodeIfPresent(
                Gemma3ProcessorConfiguration.ImageSize.self, forKey: CodingKeys.size)
            ?? imageProcessor?.size
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageTokenId) ?? 258_880
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioTokenId) ?? 258_881
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.videoTokenId) ?? 258_884
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.boiTokenId) ?? 255_999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.eoiTokenId) ?? 258_882
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    public var fixedSize: CGSize {
        let patchesPerSide = max(1, Int(floor(sqrt(Double(maxSoftTokens)))))
        let side = patchesPerSide * patchSize * poolingKernelSize
        return CGSize(width: side, height: side)
    }

    public func aspectRatioPreservingSize(for imageSize: CGSize) throws -> CGSize {
        let width = max(1, Int(ceil(imageSize.width)))
        let height = max(1, Int(ceil(imageSize.height)))
        let sideMultiple = max(1, patchSize * poolingKernelSize)
        let maxTokens = max(1, maxSoftTokens)

        let targetPixels = Double(maxTokens * sideMultiple * sideMultiple)
        let resizeFactor = sqrt(targetPixels / Double(width * height))

        var targetWidth =
            Int(floor(Double(width) * resizeFactor / Double(sideMultiple))) * sideMultiple
        var targetHeight =
            Int(floor(Double(height) * resizeFactor / Double(sideMultiple))) * sideMultiple

        if targetWidth == 0 && targetHeight == 0 {
            throw VLMError.processing("Image is too small to resize for Gemma4 unified vision.")
        } else if targetHeight == 0 {
            targetHeight = sideMultiple
            targetWidth = max(
                sideMultiple,
                min(
                    maxTokens * sideMultiple,
                    Int(floor(Double(width) / Double(height))) * sideMultiple))
        } else if targetWidth == 0 {
            targetWidth = sideMultiple
            targetHeight = max(
                sideMultiple,
                min(
                    maxTokens * sideMultiple,
                    Int(floor(Double(height) / Double(width))) * sideMultiple))
        }

        return CGSize(width: targetWidth, height: targetHeight)
    }
}

public struct Gemma4UnifiedProcessor: UserInputProcessor {
    private let config: Gemma4UnifiedProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Gemma4UnifiedProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    private func patchify(_ pixelValues: MLXArray) -> (MLXArray, MLXArray, Int, THW) {
        let channels = pixelValues.dim(1)
        let height = pixelValues.dim(2)
        let width = pixelValues.dim(3)
        let patchHeight = height / config.modelPatchSize
        let patchWidth = width / config.modelPatchSize
        let realCount = min(patchHeight * patchWidth, config.maxSoftTokens)
        let patchDim = config.modelPatchSize * config.modelPatchSize * channels

        var patches = pixelValues.reshaped(
            1, channels, patchHeight, config.modelPatchSize, patchWidth, config.modelPatchSize)
        patches = patches.transposed(0, 2, 4, 3, 5, 1)
        patches = patches.reshaped(patchHeight * patchWidth, patchDim)
        if realCount < patches.dim(0) {
            patches = patches[..<realCount, 0...]
        }
        if realCount < config.maxSoftTokens {
            patches = padded(patches, widths: [.init((0, config.maxSoftTokens - realCount)), 0])
        }

        var positionValues: [Int32] = []
        positionValues.reserveCapacity(config.maxSoftTokens * 2)
        var emitted = 0
        for y in 0 ..< patchHeight {
            for x in 0 ..< patchWidth where emitted < realCount {
                positionValues.append(Int32(x))
                positionValues.append(Int32(y))
                emitted += 1
            }
        }
        while emitted < config.maxSoftTokens {
            positionValues.append(-1)
            positionValues.append(-1)
            emitted += 1
        }
        let positions = MLXArray(positionValues, [config.maxSoftTokens, 2])
        return (patches, positions, realCount, THW(1, height, width))
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        pixels: MLXArray, positionIds: MLXArray, tokenCounts: [Int], frames: [THW]
    ) {
        var patchRows: [MLXArray] = []
        var positionRows: [MLXArray] = []
        var tokenCounts: [Int] = []
        var frames: [THW] = []

        for image in images {
            let processedImage = MediaProcessing.apply(image, processing: processing)
            let srgbImage = MediaProcessing.inSRGBToneCurveSpace(processedImage)
            let resizedImage =
                if config.doResize {
                    try MediaProcessing.resampleBicubic(
                        srgbImage,
                        to: config.aspectRatioPreservingSize(for: srgbImage.extent.size))
                } else {
                    srgbImage
                }

            var pixelValues = MediaProcessing.asMLXArray(resizedImage)
            let rescaleMultiplier = Float(config.doRescale ? config.rescaleFactor * 255 : 255)
            if rescaleMultiplier != 1 {
                pixelValues = pixelValues * MLXArray(rescaleMultiplier, dtype: pixelValues.dtype)
            }
            if config.doNormalize {
                let mean = MLXArray(
                    config.imageMean.map { Float($0) }, [1, config.imageMean.count, 1, 1]
                )
                .asType(pixelValues.dtype)
                let std = MLXArray(
                    config.imageStd.map { Float($0) }, [1, config.imageStd.count, 1, 1]
                )
                .asType(pixelValues.dtype)
                pixelValues = (pixelValues - mean) / std
            }
            let (patches, positions, tokenCount, frame) = patchify(pixelValues)
            patchRows.append(patches)
            positionRows.append(positions)
            tokenCounts.append(tokenCount)
            frames.append(frame)
        }

        return (
            pixels: stacked(patchRows, axis: 0),
            positionIds: stacked(positionRows, axis: 0),
            tokenCounts: tokenCounts,
            frames: frames
        )
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Gemma4MessageGenerator().generate(
            from: input.removingSpecialTokenLabels(using: tokenizer))

        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)

        if input.images.isEmpty {
            // No mask: one unpadded sequence needs none, and a mask stops prompt-cache reuse.
            return LMInput(tokens: MLXArray(promptTokens).expandedDimensions(axis: 0))
        }

        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imageData = try preprocess(
                images: input.images.map { try $0.asCIImage() },
                processing: input.processing
            )
            processedImage = LMInput.ProcessedImage(
                pixels: imageData.pixels,
                positionIds: imageData.positionIds,
                frames: imageData.frames
            )

            var imageIndex = 0
            var expandedTokens: [Int] = []
            expandedTokens.reserveCapacity(
                promptTokens.count + imageData.tokenCounts.reduce(0, +))
            for token in promptTokens {
                if token == config.imageTokenId {
                    let count =
                        imageIndex < imageData.tokenCounts.count
                        ? imageData.tokenCounts[imageIndex]
                        : config.imageSeqLength
                    expandedTokens.append(config.boiTokenId)
                    expandedTokens.append(
                        contentsOf: Array(repeating: config.imageTokenId, count: count))
                    if let eoiTokenId = config.eoiTokenId {
                        expandedTokens.append(eoiTokenId)
                    }
                    imageIndex += 1
                } else {
                    expandedTokens.append(token)
                }
            }
            promptTokens = expandedTokens
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(text: .init(tokens: promptArray, mask: mask), image: processedImage)
    }
}

// MARK: - Chat conventions

extension Gemma4Unified {
    public var toolCallFormat: ToolCallFormat? { .gemma4 }
}
