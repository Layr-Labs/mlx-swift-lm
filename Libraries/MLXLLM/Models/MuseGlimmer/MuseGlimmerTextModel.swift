// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Native Muse Glimmer text decoder. The arithmetic is the MLXVLM `MuseGlimmer`
// text tower's, operation for operation, so BF16 weights give the same bits in
// both. The differences are structural: projection slots hold any `UnaryLayer`
// (ModelOpt FP8 and weight-only NVFP4 layers install there), attention has a
// CBv2 branch, and the forward can return the residual stream after chosen layers.

// MARK: - Elementwise fusions

/// The elementwise regions of the block norms and the attention gate, compiled
/// where the hardware allows. Compilation keeps every operation, its order and
/// its dtype; it only stops materializing the intermediates.
enum MuseGlimmerFusion {
    private static func fused(
        _ body: @escaping @Sendable ([MLXArray]) -> [MLXArray]
    ) -> @Sendable ([MLXArray]) -> [MLXArray] {
        MLXHardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
    }

    /// `[x, weight, eps]` -> centered RMS norm of `x`, scale `1 + weight`, in FP32.
    static let centeredRMSNorm = fused { args in
        let (x, weight, eps) = (args[0], args[1], args[2])
        let dtype = x.dtype
        let f = x.asType(.float32)
        let variance = mean(f * f, axis: -1, keepDims: true)
        var out = f * rsqrt(variance + eps)
        out = out * (1.0 + weight.asType(.float32))
        return [out.asType(dtype)]
    }

    /// `[residual, x, weight, eps]` -> `residual + centeredRMSNorm(x)`.
    static let addCenteredRMSNorm = fused { args in
        let (residual, x, weight, eps) = (args[0], args[1], args[2], args[3])
        let dtype = x.dtype
        let f = x.asType(.float32)
        let variance = mean(f * f, axis: -1, keepDims: true)
        var out = f * rsqrt(variance + eps)
        out = out * (1.0 + weight.asType(.float32))
        return [residual + out.asType(dtype)]
    }

    /// `[output, gate]` -> `output * sigmoid(gate)`.
    static let sigmoidGate = fused { args in [args[0] * sigmoid(args[1])] }
}

/// Keeps a constant array out of `Module` parameter reflection.
final class MuseGlimmerConstant {
    let value: MLXArray
    init(_ value: MLXArray) { self.value = value }
}

// MARK: - Norms

/// RMS norm without a learned scale (`embed_norm`, `qk_norm`). No parameters.
final class MuseGlimmerUnitRMSNorm: Module, UnaryLayer {
    let eps: Float

    init(eps: Float) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

/// RMS norm whose stored scale is centered at zero: the effective scale is
/// `1 + weight`, applied in FP32. No mean is subtracted, despite the name.
final class MuseGlimmerCenteredRMSNorm: Module, UnaryLayer {
    @ModuleInfo var weight: MLXArray
    let eps: Float
    private let epsArray: MuseGlimmerConstant
    private let numerics: MuseGlimmerNumerics

    init(dimensions: Int, eps: Float, numerics: MuseGlimmerNumerics) {
        _weight.wrappedValue = MLXArray.zeros([dimensions])
        self.eps = eps
        epsArray = MuseGlimmerConstant(MLXArray(eps))
        self.numerics = numerics
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        numerics.reduceRows([x]) {
            MuseGlimmerFusion.centeredRMSNorm([$0[0], weight, epsArray.value])[0]
        }
    }

    /// `residual + self(x)` as one fused region.
    func addedTo(_ residual: MLXArray, _ x: MLXArray) -> MLXArray {
        numerics.reduceRows([x, residual]) {
            MuseGlimmerFusion.addCenteredRMSNorm([$0[1], $0[0], weight, epsArray.value])[0]
        }
    }
}

// MARK: - Blocks

final class MuseGlimmerTextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: any UnaryLayer
    @ModuleInfo(key: "up_proj") var upProj: any UnaryLayer
    @ModuleInfo(key: "down_proj") var downProj: any UnaryLayer
    private let numerics: MuseGlimmerNumerics

    init(_ c: MuseGlimmerTextModelConfiguration, numerics: MuseGlimmerNumerics) {
        _gateProj.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        self.numerics = numerics
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let n = numerics
        return n.project(
            downProj, compiledSiluProduct(n.project(gateProj, x), n.project(upProj, x)))
    }
}

final class MuseGlimmerTextAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProj: any UnaryLayer
    @ModuleInfo(key: "k_proj") var keyProj: any UnaryLayer
    @ModuleInfo(key: "v_proj") var valueProj: any UnaryLayer
    @ModuleInfo(key: "o_proj") var outputProj: any UnaryLayer
    /// Gates the attention output; not the MLP's `gate_proj`.
    @ModuleInfo(key: "gate_proj") var gateProj: any UnaryLayer

    let qkNorm: MuseGlimmerUnitRMSNorm
    /// Nil on layers whose `layer_rope_theta` is 0.
    let rope: RoPE?
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    /// The ordinary softmax scale; `qkScaleFactor` multiplies the queries as well.
    let scale: Float
    let qkScaleFactor: Float
    private let numerics: MuseGlimmerNumerics

    init(
        _ c: MuseGlimmerTextModelConfiguration, layerIndex: Int, numerics: MuseGlimmerNumerics
    ) {
        self.numerics = numerics
        heads = c.attentionHeads
        kvHeads = c.kvHeads
        headDimensions = c.headDimensions
        scale = pow(Float(c.headDimensions), -0.5)
        qkScaleFactor = c.qkScaleFactor
        let theta = c.layerRopeTheta[layerIndex]
        rope =
            theta != 0
            ? RoPE(dimensions: c.headDimensions, traditional: false, base: theta, scale: 1.0) : nil
        qkNorm = MuseGlimmerUnitRMSNorm(eps: c.rmsNormEps)
        let queryDim = c.attentionHeads * c.headDimensions
        let kvDim = c.kvHeads * c.headDimensions
        _queryProj.wrappedValue = Linear(c.hiddenSize, queryDim, bias: false)
        _keyProj.wrappedValue = Linear(c.hiddenSize, kvDim, bias: false)
        _valueProj.wrappedValue = Linear(c.hiddenSize, kvDim, bias: false)
        _gateProj.wrappedValue = Linear(c.hiddenSize, queryDim, bias: false)
        _outputProj.wrappedValue = Linear(queryDim, c.hiddenSize, bias: false)
        super.init()
    }

    /// `ropeOffsets` is a snapshot of this layer's per-row positions taken
    /// before any CBv2 cache of the step advanced; nil takes this cache's own.
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        ropeOffsets: MLXArray? = nil
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let n = numerics
        var queries = n.project(queryProj, x).reshaped(batch, length, heads, headDimensions)
        var keys = n.project(keyProj, x).reshaped(batch, length, kvHeads, headDimensions)
        var values = n.project(valueProj, x).reshaped(batch, length, kvHeads, headDimensions)

        // The query scale is applied in FP32 before RoPE. Folding it into the
        // softmax scale is the same expression but rounds differently in BF16.
        queries = qkNorm(queries)
        queries = (queries.asType(.float32) * qkScaleFactor)
            .asType(queries.dtype).transposed(0, 2, 1, 3)
        keys = qkNorm(keys).transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        var output: MLXArray
        if let attending = cache as? any CBv2AttendingLayerCache {
            // The layer cache owns storage, the causal or window visibility
            // and the attention itself; rows carry their own positions.
            if let rope {
                let offsets = ropeOffsets ?? (attending.positionOffsets + 0)
                queries = rope(queries, offset: offsets)
                keys = rope(keys, offset: offsets)
            }
            output = attending.updateAndAttend(
                queries: queries, keys: keys, values: values, scale: scale, sinks: nil)
        } else {
            if let rope {
                let offset = cache?.offset ?? 0
                queries = rope(queries, offset: offset)
                keys = rope(keys, offset: offset)
            }
            output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values, cache: cache, scale: scale,
                mask: mask)
        }
        output = output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        // The gate comes from the layer input, not from the attention output.
        output = MuseGlimmerFusion.sigmoidGate([output, n.project(gateProj, x)])[0]
        return n.project(outputProj, output)
    }
}

final class MuseGlimmerDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: MuseGlimmerTextAttention
    @ModuleInfo var mlp: MuseGlimmerTextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: MuseGlimmerCenteredRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm:
        MuseGlimmerCenteredRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm:
        MuseGlimmerCenteredRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm:
        MuseGlimmerCenteredRMSNorm
    let isSliding: Bool

    init(
        _ c: MuseGlimmerTextModelConfiguration, layerIndex: Int, numerics: MuseGlimmerNumerics
    ) {
        _selfAttention.wrappedValue = MuseGlimmerTextAttention(
            c, layerIndex: layerIndex, numerics: numerics)
        _mlp.wrappedValue = MuseGlimmerTextMLP(c, numerics: numerics)
        // The norms before a sublayer use rms_norm_eps, the ones after it post_norm_eps.
        _inputLayerNorm.wrappedValue = MuseGlimmerCenteredRMSNorm(
            dimensions: c.hiddenSize, eps: c.rmsNormEps, numerics: numerics)
        _postAttentionLayerNorm.wrappedValue = MuseGlimmerCenteredRMSNorm(
            dimensions: c.hiddenSize, eps: c.postNormEps, numerics: numerics)
        _preFeedforwardLayerNorm.wrappedValue = MuseGlimmerCenteredRMSNorm(
            dimensions: c.hiddenSize, eps: c.rmsNormEps, numerics: numerics)
        _postFeedforwardLayerNorm.wrappedValue = MuseGlimmerCenteredRMSNorm(
            dimensions: c.hiddenSize, eps: c.postNormEps, numerics: numerics)
        isSliding = c.isSliding(layerIndex)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        ropeOffsets: MLXArray? = nil
    ) -> MLXArray {
        let h = postAttentionLayerNorm.addedTo(
            x, selfAttention(inputLayerNorm(x), mask: mask, cache: cache, ropeOffsets: ropeOffsets))
        return postFeedforwardLayerNorm.addedTo(h, mlp(preFeedforwardLayerNorm(h)))
    }
}

final class MuseGlimmerTextTrunk: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [MuseGlimmerDecoderLayer]
    /// A plain RMS norm: its scale is used directly, unlike the block norms.
    @ModuleInfo var norm: RMSNorm
    let embedNorm: MuseGlimmerUnitRMSNorm
    let configuration: MuseGlimmerTextModelConfiguration
    let numerics: MuseGlimmerNumerics
    private let fullAttentionIndex: Int
    private let slidingAttentionIndex: Int?

    init(_ c: MuseGlimmerTextModelConfiguration, numerics: MuseGlimmerNumerics) {
        configuration = c
        self.numerics = numerics
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        embedNorm = MuseGlimmerUnitRMSNorm(eps: c.rmsNormEps)
        _layers.wrappedValue = (0 ..< c.hiddenLayers).map {
            MuseGlimmerDecoderLayer(c, layerIndex: $0, numerics: numerics)
        }
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        fullAttentionIndex =
            c.layerTypes.firstIndex(of: MuseGlimmerTextModelConfiguration.fullAttention) ?? 0
        slidingAttentionIndex = c.layerTypes.firstIndex(
            of: MuseGlimmerTextModelConfiguration.slidingAttention)
        super.init()
    }

    /// Token embeddings as the first layer takes them (after `embed_norm`).
    func inputEmbeddings(_ tokens: MLXArray) -> MLXArray { embedNorm(embedTokens(tokens)) }

    /// Runs every layer on `h` and returns the residual stream before the
    /// final norm, with the outputs of the layers in `capture` in that order.
    func residual(
        _ h: MLXArray, cache: [KVCache]?, capture: [Int] = []
    ) -> (hidden: MLXArray, captured: [MLXArray]) {
        var h = h
        let usesCBv2 = cache?.first is any CBv2AttendingLayerCache
        var fullMask = MLXFast.ScaledDotProductAttentionMaskMode.none
        var slidingMask = MLXFast.ScaledDotProductAttentionMaskMode.none
        if !usesCBv2 {
            // One mask per attention kind, each from the first cache of its
            // kind, so the offsets are right. CBv2 caches own their visibility.
            fullMask = createAttentionMask(h: h, cache: cache?[fullAttentionIndex])
            if let slidingAttentionIndex {
                slidingMask = createAttentionMask(
                    h: h, cache: cache?[slidingAttentionIndex],
                    windowSize: configuration.slidingWindow)
            }
        }
        var captured = [Int: MLXArray]()
        // Rows of one sequence sit at the same position in every layer, so one
        // pre-step snapshot of the positions serves every rotary layer whose
        // rows report the same host positions.
        var shared: (positions: [Int], offsets: MLXArray)?
        for (index, layer) in layers.enumerated() {
            var ropeOffsets: MLXArray?
            if layer.selfAttention.rope != nil,
                let attending = cache?[index] as? any CBv2AttendingLayerCache
            {
                let positions = attending.rows.map(\.absoluteOffset)
                if let shared, shared.positions == positions {
                    ropeOffsets = shared.offsets
                } else {
                    let offsets = attending.positionOffsets + 0
                    shared = (positions, offsets)
                    ropeOffsets = offsets
                }
            }
            h = layer(
                h, mask: layer.isSliding ? slidingMask : fullMask, cache: cache?[index],
                ropeOffsets: ropeOffsets)
            if capture.contains(index) { captured[index] = h }
        }
        return (h, capture.map { captured[$0]! })
    }

    func callAsFunction(_ h: MLXArray, cache: [KVCache]?) -> MLXArray {
        norm(residual(h, cache: cache).hidden)
    }
}

// MARK: - Model

public final class MuseGlimmerTextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let configuration: MuseGlimmerTextModelConfiguration
    @ModuleInfo var model: MuseGlimmerTextTrunk
    @ModuleInfo(key: "lm_head") var lmHead: any UnaryLayer

    /// 0-based layers whose output residual a block drafter consumes
    /// (`hidden_states[i + 1]` in the reference). Set before building an engine.
    public var mtpHiddenLayerIDs: [Int] {
        didSet { precondition(Self.validHiddenLayers(mtpHiddenLayerIDs, configuration)) }
    }

    /// A prompt chunk's bytes depend on its rows, so the engine cuts every
    /// prompt on this grid, whatever the load: the solo stripe and the
    /// checkpoint capture stride (`CBv2ModelCapabilities.prefillChunkAlignment`).
    public static let defaultPrefillChunkAlignment = 1024
    /// The grid the tower reports; tests with small chunks lower it.
    var prefillChunkAlignment = MuseGlimmerTextModel.defaultPrefillChunkAlignment

    public var vocabularySize: Int { configuration.vocabularySize }
    public var kvHeads: [Int] {
        Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
    }
    public var loraLayers: [Module] { model.layers }
    public var toolCallFormat: ToolCallFormat? { .atem }
    public var reasoningConfig: ReasoningConfig? {
        ReasoningConfig(
            startDelimiter: "to=self<|message|>", endDelimiter: "<|eom|>",
            promptStrategy: .none, isSpecialToken: true)
    }

    public init(_ configuration: MuseGlimmerTextModelConfiguration) {
        self.configuration = configuration
        _model.wrappedValue = MuseGlimmerTextTrunk(configuration, numerics: MuseGlimmerNumerics())
        _lmHead.wrappedValue = Linear(
            configuration.hiddenSize, configuration.vocabularySize, bias: false)
        mtpHiddenLayerIDs = Self.defaultHiddenLayers(configuration)
        super.init()
    }

    /// The published drafter's five layers when the model is deep enough.
    static func defaultHiddenLayers(_ c: MuseGlimmerTextModelConfiguration) -> [Int] {
        [1, 13, 25, 37, 49].filter { $0 < c.hiddenLayers }
    }

    static func validHiddenLayers(_ ids: [Int], _ c: MuseGlimmerTextModelConfiguration) -> Bool {
        ids.allSatisfy { $0 >= 0 && $0 < c.hiddenLayers } && Set(ids).count == ids.count
    }

    /// Sliding layers get a rotating cache bounded by the window; full layers
    /// honor the requested capacity.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try (0 ..< configuration.hiddenLayers).map {
            try makeHybridAttentionKVCache(
                parameters: parameters, slidingWindow: configuration.slidingWindow,
                usesSlidingWindow: configuration.isSliding($0))
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        outputLogits(model(model.inputEmbeddings(inputs), cache: cache))
    }

    /// Forward from first-layer embeddings (`embed_norm` already applied, or
    /// media features spliced in): `[B, L, hidden]` to logits.
    public func callAsFunction(inputEmbeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        outputLogits(model(inputEmbeddings, cache: cache))
    }

    // MARK: Readout

    /// The switch shared by every block (`setRowExactWindows`).
    var numerics: MuseGlimmerNumerics { model.numerics }

    /// `lm_head(hidden)` without the output multiplier or the softcap.
    public func headProduct(_ hidden: MLXArray) -> MLXArray { numerics.project(lmHead, hidden) }

    /// The readout of a final-normed hidden: `lm_head`, then
    /// `output_multiplier`, then the tanh softcap.
    public func outputLogits(_ normedHidden: MLXArray) -> MLXArray {
        var logits = headProduct(normedHidden) * configuration.outputMultiplier
        let cap = configuration.finalLogitSoftcapping
        if cap > 0 { logits = tanh(logits / cap) * cap }
        return logits
    }

    // MARK: Embeddings

    /// `embed_tokens(tokens)` without `embed_norm`, as a block drafter embeds.
    public func rawTokenEmbeddings(_ tokens: MLXArray) -> MLXArray { model.embedTokens(tokens) }

    /// `embed_norm(embed_tokens(tokens))`, the first layer's input.
    public func scaledTokenEmbeddings(_ tokens: MLXArray) -> MLXArray {
        model.inputEmbeddings(tokens)
    }

    public var tokenEmbeddingTable: MLXArray { model.embedTokens.weight }

    /// The dtype activations and attention K/V run in.
    public var activationDType: DType { model.embedTokens.weight.dtype }

    // MARK: Weights

    /// Key renaming only. Accepts the MLX layout (`language_model.model.*`,
    /// `language_model.lm_head.*`) and the transformers layout
    /// (`model.language_model.*`). Vision tensors are left in place, so a
    /// strict update refuses them instead of dropping them.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray](minimumCapacity: weights.count)
        for (key, value) in weights {
            if key.contains("rotary_emb.inv_freq") { continue }
            var key = key
            for (prefix, replacement) in [
                ("language_model.model.", "model."), ("language_model.lm_head.", "lm_head."),
                ("model.language_model.", "model."),
            ] where key.hasPrefix(prefix) {
                key = replacement + key.dropFirst(prefix.count)
                break
            }
            result[key] = value
        }
        return result
    }
}
