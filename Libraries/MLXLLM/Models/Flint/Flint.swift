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
        let (qx, kx, vx) = projectQKV(x)
        var q = qNorm(qx.reshaped(batch, length, c.attentionHeads, c.headDim)).transposed(
            0, 2, 1, 3)
        var k = kNorm(kx.reshaped(batch, length, c.kvHeads, c.headDim)).transposed(0, 2, 1, 3)
        var v = vx.reshaped(batch, length, c.kvHeads, c.headDim).transposed(0, 2, 1, 3)
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

    /// Load-time q/k/v concatenation (see `FlintFusedProjection`); nil until
    /// `fuseProjections()` and after any parameter update.
    let fusedQKV = FlintFusedProjectionSlot()
    /// Test hook: `false` always runs the three separate projections.
    var usesFusedProjection = true

    /// `(qProj(x), kProj(x), vProj(x))`. A single-row call (one decode token)
    /// on a fused layer runs one quantized matmul over the concatenated rows
    /// and splits its output: every row view stays row-contiguous, so the
    /// norms and the cache read it without a copy. More rows take the three
    /// matmuls (their q/k views would be strided and cost copies).
    func projectQKV(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        if usesFusedProjection, let fused = fusedQKV.value,
            let parts = fused(x, members: [qProj, kProj, vProj], maxRows: 1)
        {
            return (parts[0], parts[1], parts[2])
        }
        return (qProj(x), kProj(x), vProj(x))
    }

    /// Re-points q/k/v at row views of one concatenated quantized matrix
    /// (memory unchanged) and records it for single-row calls. A no-op unless
    /// all three are plain affine `QuantizedLinear`s that can share a matmul.
    func fuseProjections() throws {
        fusedQKV.value = try FlintFusedProjection.fuse(
            [("q_proj", qProj), ("k_proj", kProj), ("v_proj", vProj)], in: self)
    }

    @discardableResult
    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        fusedQKV.value = nil
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
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
    /// Load-time gate/up concatenation (see `FlintFusedProjection`).
    let fusedGateUp = FlintFusedProjectionSlot()
    /// Test hook: `false` always runs the original expression.
    var usesFusedProjection = true

    /// Fewer rows than MLX's smallest qmv-to-qmm crossover for the fused
    /// width take one matmul over the concatenated rows and the compiled
    /// `silu(gate) * up` (same BF16 roundings as `silu` then `*`). Prompt
    /// chunks keep the original expression on the row views.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if usesFusedProjection, let fused = fusedGateUp.value,
            let parts = fused(x, members: [gate, up], maxRows: fused.qmvRows - 1)
        {
            return down(compiledSiluProduct(parts[0], parts[1]))
        }
        return down(silu(gate(x)) * up(x))
    }

    func fuseProjections() throws {
        fusedGateUp.value = try FlintFusedProjection.fuse(
            [("gate_proj", gate), ("up_proj", up)], in: self)
    }

    @discardableResult
    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        fusedGateUp.value = nil
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }
}

/// A non-module box: `Module` reflection ignores plain class references.
final class FlintFusedProjectionSlot {
    var value: FlintFusedProjection?
}

/// Several affine-quantized projections of one input, concatenated along
/// their output rows at load time. The member modules' parameters become row
/// views of the concatenation (memory unchanged; every unfused call reads the
/// very same bytes), and calls with few rows run ONE quantized matmul instead
/// of one per member.
///
/// Exactness: below `qmvRows` rows MLX's `QuantizedMatmul` runs a qmv kernel
/// (`qmv_fast` / `qmv_wide`) whose output row `n` is a dot product of weight
/// row `n` with each input row, reduced in an order fixed by K, the group
/// size and the row count alone — never by N or the row's position. So the
/// fused output's columns are bit-identical to each member's output. The
/// row-count crossover itself depends on N (`get_qmv_batch_limit`), hence the
/// bound: the smallest crossover across every Apple GPU MLX tunes for, for
/// the fused width (members are narrower, so their crossover is no lower).
/// Prompt chunks never take this path.
final class FlintFusedProjection {
    let members: [Linear]
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let widths: [Int]
    let groupSize: Int
    let bits: Int
    /// Rows at or above which some device may leave qmv for this width.
    let qmvRows: Int

    init(
        members: [Linear], weight: MLXArray, scales: MLXArray, biases: MLXArray, widths: [Int],
        groupSize: Int, bits: Int, inputDims: Int
    ) {
        self.members = members
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.widths = widths
        self.groupSize = groupSize
        self.bits = bits
        self.qmvRows = Self.qmvRowFloor(
            inputDims: inputDims, outputDims: widths.reduce(0, +))
    }

    /// The smallest row count at which the pinned MLX's `get_qmv_batch_limit`
    /// sends a transposed quantized matmul to qmm on ANY device (M1–M5, all
    /// sizes): 13 when K and N <= 2048, 10 when <= 4096, otherwise 6.
    static func qmvRowFloor(inputDims: Int, outputDims: Int) -> Int {
        if inputDims <= 2048 && outputDims <= 2048 { return 13 }
        if inputDims <= 4096 && outputDims <= 4096 { return 10 }
        return 6
    }

    /// The members' outputs, or nil when the fused matmul is not provably the
    /// members' arithmetic for this call (or `members` were replaced).
    func callAsFunction(_ x: MLXArray, members current: [Linear], maxRows: Int) -> [MLXArray]? {
        guard current.count == members.count,
            zip(current, members).allSatisfy({ $0 === $1 }),
            x.ndim >= 2, x.dtype == scales.dtype,
            x.size / x.dim(-1) <= min(maxRows, qmvRows - 1)
        else { return nil }
        let output = quantizedMM(
            x, weight, scales: scales, biases: biases, transpose: true, groupSize: groupSize,
            bits: bits, mode: .affine)
        var start = 0
        return widths.map { width in
            defer { start += width }
            return output[.ellipsis, start ..< (start + width)]
        }
    }

    /// Concatenates `named` (keys under `owner`) and installs the row views.
    /// Returns nil, changing nothing, unless every member is a plain affine
    /// 4-bit/group-64 `QuantizedLinear` without bias over the same input, with
    /// matching dtypes, widths a multiple of 8 and K a multiple of 512 (MLX's
    /// `qmv_fast` alignment for 4 bits), so the members and the fusion select
    /// the same qmv kernel family.
    static func fuse(_ named: [(String, Linear)], in owner: Module) throws
        -> FlintFusedProjection?
    {
        let members = named.compactMap { $0.1 as? QuantizedLinear }
        guard members.count == named.count, let first = members.first,
            let firstOffsets = first.biases,
            members.allSatisfy({
                ObjectIdentifier(type(of: $0)) == ObjectIdentifier(QuantizedLinear.self)
                    && $0.mode == .affine && $0.groupSize == 64 && $0.bits == 4
                    && $0.bias == nil && $0.weight.ndim == 2 && $0.scales.ndim == 2
                    && $0.weight.dtype == .uint32
                    && $0.weight.dim(1) == first.weight.dim(1)
                    && $0.weight.dim(0).isMultiple(of: 8)
                    && $0.scales.dtype == first.scales.dtype
                    && $0.scales.shape == [$0.weight.dim(0), first.scales.dim(1)]
                    && $0.biases?.dtype == firstOffsets.dtype
                    && $0.biases?.shape == $0.scales.shape
            })
        else { return nil }
        let inputDims = first.weight.dim(1) * 32 / 4
        guard inputDims.isMultiple(of: 512), first.scales.dim(1) * 64 == inputDims else {
            return nil
        }
        let weight = concatenated(members.map(\.weight), axis: 0)
        let scales = concatenated(members.map(\.scales), axis: 0)
        let biases = concatenated(members.map { $0.biases! }, axis: 0)
        let widths = members.map { $0.weight.dim(0) }
        var views = [String: MLXArray]()
        var start = 0
        for ((key, _), width) in zip(named, widths) {
            let rows = start ..< (start + width)
            views["\(key).weight"] = weight[rows]
            views["\(key).scales"] = scales[rows]
            views["\(key).biases"] = biases[rows]
            start += width
        }
        try owner.update(
            parameters: ModuleParameters.unflattened(views), verify: [.shapeMismatch])
        return FlintFusedProjection(
            members: members, weight: weight, scales: scales, biases: biases, widths: widths,
            groupSize: 64, bits: 4, inputDims: inputDims)
    }

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
        let experts = SwitchGLU(
            inputDims: c.hiddenSize, hiddenDims: c.moeIntermediateSize, numExperts: c.numExperts)
        experts.suppliesHostRowIndices = true
        _experts.wrappedValue = experts
        _shared.wrappedValue = FlintMLP(
            hidden: c.hiddenSize, intermediate: c.sharedExpertIntermediateSize)
    }
    func route(_ x: MLXArray) -> (MLXArray, MLXArray) {
        guard usesFusedRouter else {
            let scores = sigmoid(gate(x).asType(.float32))
            let choice = scores + expertBias.asType(.float32)
            let indices = stopGradient(
                argPartition(choice, kth: -topK, axis: -1)[.ellipsis, (-topK)...])
            let selected = takeAlong(scores, indices, axis: -1)
            return (indices, selected / (selected.sum(axis: -1, keepDims: true) + 1e-20))
        }
        // The same graph with its elementwise links compiled: the FP32 widen,
        // sigmoid and bias add become one kernel (two outputs), the `+ 1e-20`
        // and divide another. The matmul, argPartition, gather and sum stay
        // MLX's own kernels, so selection order and reduction order are unchanged.
        let (scores, choice) = FlintRouter.scores(gate(x), expertBias)
        let indices = stopGradient(
            argPartition(choice, kth: -topK, axis: -1)[.ellipsis, (-topK)...])
        return (indices, FlintRouter.normalized(takeAlong(scores, indices, axis: -1)))
    }
    /// Test hook: `false` evaluates the original unfused combine below.
    var usesFusedExpertCombine = true
    /// Test hook: `false` evaluates the router's original uncompiled graph.
    var usesFusedRouter = true
    func callAsFunction(_ x: MLXArray) -> MLXArray { combine(x, residual: nil) }

    /// `residual + self(x)`; on the fused paths the residual add is folded
    /// into the combine kernel (same `bfloat16_t` add, same operand order).
    func callAsFunction(_ x: MLXArray, residual: MLXArray) -> MLXArray {
        combine(x, residual: residual)
    }

    private func combine(_ x: MLXArray, residual: MLXArray?) -> MLXArray {
        let (indices, weights) = route(x)
        guard usesFusedExpertCombine else {
            let routed = experts(x, indices).asType(.float32)
            // Preserve the BF16 reference's FP32 weighting and reduction.
            let output =
                (routed * weights[.ellipsis, .newAxis]).sum(axis: -2).asType(x.dtype) + shared(x)
            return residual.map { $0 + output } ?? output
        }
        // Byte-identical to the expression above; sorted BF16 prompt chunks
        // skip its unsort gather and FP32 intermediates (one fused pass), and
        // decode folds the widen, weighting, reduction, narrowing, shared-expert
        // and residual adds into one slot-order pass.
        return experts.callAndFlintFP32WeightedSum(
            x, indices, weights: weights, adding: shared(x), residual: residual)
    }

    /// Concatenates the routed experts' gate/up banks (one gather matmul per
    /// call instead of two) and the shared expert's gate/up projections.
    func fuseProjections() throws {
        try shared.fuseProjections()
        if let fused = experts.fusingQuantizedGateUp() {
            update(modules: ModuleChildren.unflattened([("switch_mlp", fused)]))
        }
    }
}

/// Flint's router links between MLX's non-fusable operations. Compiled where
/// the hardware allows (`MLXHardwareInfo.isCompiledDecodeSupported`), plain
/// closures of the identical expressions otherwise. Compilation keeps every
/// intermediate's dtype and op order; it only stops materializing them.
enum FlintRouter {
    /// `(sigmoid(logits.asType(.float32)), that + bias.asType(.float32))`.
    static let scores: @Sendable (MLXArray, MLXArray) -> (MLXArray, MLXArray) = {
        let body: @Sendable ([MLXArray]) -> [MLXArray] = { inputs in
            let scores = sigmoid(inputs[0].asType(.float32))
            return [scores, scores + inputs[1].asType(.float32)]
        }
        let function = MLXHardwareInfo.isCompiledDecodeSupported
            ? compile(shapeless: true, body) : body
        return { logits, bias in
            let outputs = function([logits, bias])
            return (outputs[0], outputs[1])
        }
    }()

    /// `selected / (selected.sum(axis: -1, keepDims: true) + 1e-20)`.
    static let normalized: @Sendable (MLXArray) -> MLXArray = {
        let body: @Sendable (MLXArray) -> MLXArray = { selected in
            selected / (selected.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return MLXHardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body
    }()
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
        if let moe = mlp as? FlintMoE { return moe(postNorm(h), residual: h) }
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
            if layer.attention.rope != nil,
                let attending = cache?[i] as? any CBv2AttendingLayerCache
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
    /// Load-time projection fusion for affine-quantized (exported) weights:
    /// every layer's q/k/v and dense or shared-expert gate/up become row views
    /// of one concatenated matrix used by single-token decode calls, and the
    /// routed experts' gate/up banks become one `gate_up_proj` bank. Layers
    /// whose projections do not qualify are left untouched. Outputs are
    /// byte-identical; see `FlintFusedProjection` and
    /// `SwitchGLU.fusingQuantizedGateUp`.
    func fuseProjections() throws {
        for layer in model.layers {
            try layer.attention.fuseProjections()
            if let moe = layer.mlp as? FlintMoE {
                try moe.fuseProjections()
            } else if let dense = layer.mlp as? FlintMLP {
                try dense.fuseProjections()
            }
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
