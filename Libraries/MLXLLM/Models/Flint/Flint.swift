// Adapted from Prince Canuma's mlx-vlm Flint implementation (MIT) and
// Recursive's rec reference (Apache-2.0). See ThirdPartyNotices/Flint.md.
import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class FlintAttention: Module {
    let configuration: FlintConfiguration
    let window: Int?
    let rope: RoPE?
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ParameterInfo(key: "softmax_offset") var sinks: MLXArray

    init(_ c: FlintConfiguration, index: Int) {
        configuration = c
        window = c.layerTypes[index] == "sliding_attention" ? c.slidingWindow : nil
        rope =
            c.noRopeLayerIndices.contains(index)
            ? nil
            : RoPE(
                dimensions: c.headDim, traditional: false, base: c.ropeTheta)
        _qProj.wrappedValue = Linear(c.hiddenSize, c.attentionHeads * c.headDim, bias: false)
        _kProj.wrappedValue = Linear(c.hiddenSize, c.kvHeads * c.headDim, bias: false)
        _vProj.wrappedValue = Linear(c.hiddenSize, c.kvHeads * c.headDim, bias: false)
        _oProj.wrappedValue = Linear(c.attentionHeads * c.headDim, c.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
        _sinks.wrappedValue = MLXArray.zeros([c.attentionHeads])
    }

    /// The learned sinks narrowed to the query dtype, built once per
    /// (parameter, dtype) instead of once per call. A stable array identity
    /// also lets the paged decode kernel reuse its prepared FP32 sinks
    /// (`PagedLayerCache` keys that cache on object identity). The values
    /// are exactly `sinks.asType(dtype)`: the BF16 rounding is preserved.
    private let sinkCache = FlintSinkCache()

    func querySinks(_ dtype: DType) -> MLXArray {
        let source = sinks
        if let cached = sinkCache.value, cached.source === source, cached.dtype == dtype {
            return cached.sinks
        }
        let narrowed = source.asType(dtype)
        sinkCache.value = (source, dtype, narrowed)
        return narrowed
    }

    /// `ropeOffsets` is a pre-step snapshot of this layer's per-row RoPE
    /// offsets shared across layers by `FlintTrunk`; nil takes this cache's own.
    func callAsFunction(_ x: MLXArray, cache: KVCache?, ropeOffsets: MLXArray? = nil) -> MLXArray {
        let c = configuration
        let (batch, length) = (x.dim(0), x.dim(1))
        var q = qNorm(qProj(x).reshaped(batch, length, c.attentionHeads, c.headDim)).transposed(
            0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(batch, length, c.kvHeads, c.headDim)).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(batch, length, c.kvHeads, c.headDim).transposed(0, 2, 1, 3)
        let scale = 1 / sqrt(Float(c.headDim))
        let output: MLXArray
        if let attending = cache as? any CBv2AttendingLayerCache {
            if let rope {
                let offsets = ropeOffsets ?? (attending.positionOffsets + 0)
                q = rope(q, offset: offsets)
                k = rope(k, offset: offsets)
            }
            // Even zero-valued learned sinks contribute exp(0) to the denominator.
            output = attending.updateAndAttend(
                queries: q, keys: k, values: v, scale: scale, sinks: querySinks(q.dtype))
        } else {
            let mask = createAttentionMask(h: x, cache: cache, windowSize: window)
            if let rope {
                q = applyRotaryPosition(rope, to: q, cache: cache)
                k = applyRotaryPosition(rope, to: k, cache: cache)
            }
            if let cache { (k, v) = cache.update(keys: k, values: v) }
            output = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: mask,
                sinks: querySinks(q.dtype))
        }
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }

    /// Commit this layer's prompt-chunk K/V without computing its attention
    /// output. K and V are the exact tensors `callAsFunction` would hand the
    /// cache (same projection, norm and RoPE offsets). Returns `false`, having
    /// built nothing, unless `cache` is a paged layer that qualifies.
    func commitPromptKVWithoutAttention(_ x: MLXArray, cache: KVCache?) -> Bool {
        guard x.ndim == 3, x.dim(1) > 1, let paged = cache as? PagedLayerCache,
            paged.acceptsPromptKVWithoutAttention
        else { return false }
        let c = configuration
        let (batch, length) = (x.dim(0), x.dim(1))
        var k = kNorm(kProj(x).reshaped(batch, length, c.kvHeads, c.headDim)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(batch, length, c.kvHeads, c.headDim).transposed(0, 2, 1, 3)
        if let rope { k = rope(k, offset: paged.positionOffsets + 0) }
        paged.writePromptKVWithoutAttention(keys: k, values: v)
        return true
    }
}

/// Reference box, not a stored tuple: `Module` reflection would register an
/// `MLXArray` inside a tuple as a parameter; a plain class is ignored.
final class FlintSinkCache {
    var value: (source: MLXArray, dtype: DType, sinks: MLXArray)?
}

final class FlintMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(hidden: Int, intermediate: Int) {
        _gate.wrappedValue = Linear(hidden, intermediate, bias: false)
        _up.wrappedValue = Linear(hidden, intermediate, bias: false)
        _down.wrappedValue = Linear(intermediate, hidden, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class FlintMoE: Module, UnaryLayer {
    let topK: Int
    @ModuleInfo var gate: Linear
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray
    @ModuleInfo(key: "switch_mlp") var experts: SwitchGLU
    @ModuleInfo(key: "shared_expert") var shared: FlintMLP
    init(_ c: FlintConfiguration) {
        topK = c.expertsPerToken
        _gate.wrappedValue = Linear(c.hiddenSize, c.numExperts, bias: false)
        _expertBias.wrappedValue = MLXArray.zeros([c.numExperts])
        _experts.wrappedValue = SwitchGLU(
            inputDims: c.hiddenSize, hiddenDims: c.moeIntermediateSize, numExperts: c.numExperts)
        _shared.wrappedValue = FlintMLP(
            hidden: c.hiddenSize, intermediate: c.sharedExpertIntermediateSize)
    }
    func route(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let scores = sigmoid(gate(x).asType(.float32))
        let choice = scores + expertBias.asType(.float32)
        let indices = stopGradient(
            argPartition(choice, kth: -topK, axis: -1)[.ellipsis, (-topK)...])
        let selected = takeAlong(scores, indices, axis: -1)
        return (indices, selected / (selected.sum(axis: -1, keepDims: true) + 1e-20))
    }
    /// Test hook: `false` evaluates the original unfused combine below.
    var usesFusedExpertCombine = true
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = route(x)
        guard usesFusedExpertCombine else {
            let routed = experts(x, indices).asType(.float32)
            // Preserve the BF16 reference's FP32 weighting and reduction.
            return (routed * weights[.ellipsis, .newAxis]).sum(axis: -2).asType(x.dtype)
                + shared(x)
        }
        // Byte-identical to the expression above; sorted BF16 prompt chunks
        // skip its unsort gather and FP32 intermediates (one fused pass).
        return experts.callAndFlintFP32WeightedSum(x, indices, weights: weights, adding: shared(x))
    }
}

final class FlintDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: FlintAttention
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    let mlp: UnaryLayer
    init(_ c: FlintConfiguration, index: Int) {
        _attention.wrappedValue = FlintAttention(c, index: index)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        mlp =
            c.mlpOnlyLayers.contains(index)
            ? FlintMLP(hidden: c.hiddenSize, intermediate: c.intermediateSize) : FlintMoE(c)
    }
    func callAsFunction(_ x: MLXArray, cache: KVCache?, ropeOffsets: MLXArray? = nil) -> MLXArray {
        let h = x + attention(inputNorm(x), cache: cache, ropeOffsets: ropeOffsets)
        return h + mlp(postNorm(h))
    }
}

final class FlintTrunk: Module {
    @ModuleInfo(key: "embed_tokens") var embedding: Embedding
    let layers: [FlintDecoderLayer]
    let norm: RMSNorm
    init(_ c: FlintConfiguration) {
        _embedding.wrappedValue = Embedding(
            embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        layers = (0 ..< c.hiddenLayers).map { FlintDecoderLayer(c, index: $0) }
        norm = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }
    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedding(tokens)
        // Every rotary CBv2 layer snapshots its pre-update offsets (`+ 0`, a
        // graph-safe copy). Rows of one sequence sit at the same position in
        // every layer, so one snapshot serves each later rotary layer whose
        // rows report identical host positions; any mismatch takes its own.
        var shared: (positions: [Int], offsets: MLXArray)?
        for (i, layer) in layers.enumerated() {
            var ropeOffsets: MLXArray?
            if layer.attention.rope != nil, let attending = cache?[i] as? any CBv2AttendingLayerCache
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
            h = layer(h, cache: cache?[i], ropeOffsets: ropeOffsets)
        }
        return norm(h)
    }

    /// An intermediate prompt chunk whose output is never read: every layer
    /// but the last runs unchanged, and the last only commits its K/V when its
    /// paged cache qualifies (`FlintAttention.commitPromptKVWithoutAttention`),
    /// skipping its attention, o_proj and MLP. Returns a tensor that depends
    /// on every layer that ran; its VALUES are unspecified.
    func committingPromptKV(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedding(tokens)
        let last = layers.count - 1
        for i in 0 ..< last { h = layers[i](h, cache: cache?[i]) }
        let final = layers[last]
        if final.attention.commitPromptKVWithoutAttention(final.inputNorm(h), cache: cache?[last]) {
            return h
        }
        return norm(final(h, cache: cache?[last]))
    }
}

public final class FlintModel: Module, LLMModel, KVCacheDimensionProvider {
    public let configuration: FlintConfiguration
    let model: FlintTrunk
    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    public var vocabularySize: Int { configuration.vocabularySize }
    public var kvHeads: [Int] {
        Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
    }
    public var loraLayers: [Module] { model.layers }
    /// Test hook: `false` runs the final layer in full on `.evaluationOnly`
    /// prompt chunks (see `FlintTrunk.committingPromptKV`).
    var commitsFinalPromptKVOnly = true

    public init(_ configuration: FlintConfiguration) {
        self.configuration = configuration
        model = FlintTrunk(configuration)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabularySize, bias: false)
        }
    }
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let h = model(inputs, cache: cache).asType(.float32)
        return lmHead?(h) ?? model.embedding.asLinear(h)
    }
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        configuration.layerTypes.map {
            $0 == "sliding_attention"
                ? RotatingKVCache(maxSize: configuration.slidingWindow, keep: 0) : KVCacheSimple()
        }
    }
    public var cbv2LayerKinds: [CBv2LayerKind] { configuration.cbv2LayerKinds }
    public func newCacheV2(
        makeLayerCache: (Int, CBv2LayerKind) throws -> any CBv2AttendingLayerCache
    ) rethrows -> [any CBv2AttendingLayerCache] {
        try cbv2LayerKinds.enumerated().map { try makeLayerCache($0.offset, $0.element) }
    }
}

// All persistent target state is attention KV. The engine still requires its
// observed per-layer dtype table and a supported backend/store before issuance.
extension FlintModel: CBv2HistoricalAttentionCheckpointProviding {
    public var cbv2SupportsHistoricalAttentionCheckpoint: Bool { true }
}
