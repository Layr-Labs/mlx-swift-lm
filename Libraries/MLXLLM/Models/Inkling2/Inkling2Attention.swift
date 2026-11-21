// Inkling 2 attention: scaled-L2-clipped queries/keys, RMS-normed values, short
// convolutions on K/V, a learned relative-position bias band, per-query sinks
// and a sigmoid-gated output. No RoPE. Reference: transformers PR 51050
// `Inkling2LoopSlidingAttention` / `Inkling2LoopFullAttention` (Apache-2.0).
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// One slot's attention call. Shapes follow the frozen kernel interface.
public struct Inkling2AttentionInputs {
    /// `[B, heads, L, headDim]`, model dtype.
    public let queries: MLXArray
    /// The step's new keys and values, `[B, kvHeads, L, headDim]`.
    public let keys: MLXArray
    public let values: MLXArray
    /// Bias by backward distance `0 ..< extent`: `[B, heads, L, extent]`, model
    /// dtype (`relativeStates @ projection`, as the reference rounds it).
    public let relativeLogits: MLXArray
    /// The factors of `relativeLogits` for in-kernel use: `[B, L, heads, dRel]`
    /// and `[dRel, extent]`.
    public let relativeStates: MLXArray
    public let relativeProjection: MLXArray
    /// One extra softmax logit per query and head with a zero value: `[B, heads, L]`.
    public let sinks: MLXArray
    /// Absolute positions of the step's tokens, `[B, L]` int32.
    public let positions: MLXArray
    /// 1024 for sliding slots, nil for full slots.
    public let window: Int?
    public let scale: Float
    /// A prompt chunk (`Inkling2Model.cbv2RecurrentPrefill`), whatever its
    /// length; false for decode steps and verify windows.
    public var isPrefill: Bool = false
}

/// Set while `cbv2RecurrentPrefill` runs, so every attention call of a
/// prompt chunk reports `isPrefill` (cores pick their prefill arithmetic for
/// any chunk length).
public enum Inkling2PrefillStep {
    @TaskLocal public static var active = false
}

/// Where a step's per-slot state lives: no cache (full recompute), or the
/// paged engine. Implementations own K/V history and the carried conv states.
public protocol Inkling2StepState: AnyObject {
    /// Absolute positions of the step's tokens, `[B, L]` int32.
    var positions: MLXArray { get }
    /// `[B, kernel - 1, channels]` or nil for an empty history.
    func convState(slot: Int, _ kind: Inkling2ConvKind) -> MLXArray?
    /// `state` is the next `[B, kernel - 1, channels]` state; `input` the
    /// step's convolution input `[B, L, channels]` (per-position states for
    /// verify windows are slices of `previous ++ input`).
    func setConvState(slot: Int, _ kind: Inkling2ConvKind, _ state: MLXArray, input: MLXArray)
    /// Store the step's K/V for `slot` and attend. Returns `[B, L, heads * headDim]`.
    func attend(slot: Int, _ inputs: Inkling2AttentionInputs) -> MLXArray
}

public enum Inkling2ReferenceAttention {
    /// `eager_attention_forward` with the relative bias (distance `< extent` only),
    /// the causal/window mask and the sink column, over explicit keys. FP32
    /// scores and softmax. `keyPositions`: `[T]` or `[B, T]`; queries take
    /// `inputs.positions`. Returns `[B, L, heads * headDim]` in the query dtype.
    public static func attend(
        _ inputs: Inkling2AttentionInputs, keys: MLXArray, values: MLXArray,
        keyPositions: MLXArray
    ) -> MLXArray {
        let q = inputs.queries
        let (b, h, l, d) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
        let kvHeads = keys.dim(1)
        let t = keys.dim(2)
        let groups = h / kvHeads
        let extent = inputs.relativeLogits.dim(-1)

        let qg = q.asType(.float32).reshaped(b, kvHeads, groups, l, d)
        let k = keys.asType(.float32).expandedDimensions(axis: 2)
        var scores = matmul(qg, k.swappedAxes(-1, -2)).reshaped(b, h, l, t) * inputs.scale

        let queryPositions = inputs.positions.reshaped(-1, 1, l, 1)
        let keyPositions = keyPositions.reshaped(-1, 1, 1, t)
        let distance = queryPositions - keyPositions
        let inBand = logicalAnd(distance .>= 0, distance .< extent)
        let gatherIndex = broadcast(
            clip(distance, min: 0, max: extent - 1), to: [distance.dim(0), 1, l, t]
        ).asType(.int32)
        let bias = takeAlong(
            inputs.relativeLogits.asType(.float32), gatherIndex, axis: -1)
        var allowed = distance .>= 0
        if let window = inputs.window { allowed = logicalAnd(allowed, distance .< window) }
        scores = scores + which(inBand, bias, MLXArray(Float(0)))
        scores = which(allowed, scores, MLXArray(-Float.infinity))

        let logits = concatenated(
            [scores, inputs.sinks.asType(.float32).expandedDimensions(axis: -1)], axis: -1)
        let probabilities = softmax(logits, axis: -1)[.ellipsis, ..<t]
        let out = matmul(
            probabilities.reshaped(b, kvHeads, groups, l, t),
            values.asType(.float32).expandedDimensions(axis: 2))
        return out.reshaped(b, h, l, d).asType(q.dtype).transposed(0, 2, 1, 3)
            .reshaped(b, l, h * d)
    }
}

/// The learned bank of bias-vs-distance profiles (`rel_logits_proj.proj`).
final class Inkling2RelativeProjection: Module {
    let proj: MLXArray

    init(dimensions: Int, extent: Int) {
        proj = MLXArray.zeros([dimensions, extent])
    }
}

/// Inference-only concatenated weights of the projections that share an input
/// in row-exact steps (decode and verify windows). Built by `prepare()`; the
/// module's own projections become row views of these buffers, so nothing is
/// stored twice. Not part of the module topology (a plain object under an
/// underscore name), so checkpoint keys are unchanged.
final class Inkling2DecodeProjections: @unchecked Sendable {
    /// Sliding layers: `[wq; wr; sink; gate; wk; wv]` for steps whose query,
    /// key and value inputs are one array.
    var shared: MLXArray?
    /// `[wq; wr; sink; gate]` (a row view of `shared` on sliding layers).
    var query: MLXArray?
    /// Full layers: `[wk; k_sconv.weight_proj]` and `[wv; v_sconv.weight_proj]`.
    var key: MLXArray?
    var value: MLXArray?
    /// The projection modules the buffers back; a replaced module disables them.
    var sources: [ObjectIdentifier] = []

    func clear() {
        shared = nil
        query = nil
        key = nil
        value = nil
        sources = []
    }
}

final class Inkling2Attention: Module {
    let isFull: Bool
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let window: Int?
    let scale: Float
    let qkThreshold: Float
    let eps: Float
    let logScalingFloor: Int
    let logScalingAlpha: Float

    @ModuleInfo(key: "wq_du") var queryProjection: Linear
    @ModuleInfo(key: "wk_dv") var keyProjection: Linear
    @ModuleInfo(key: "wv_dv") var valueProjection: Linear
    @ModuleInfo(key: "wo_ud") var outputProjection: Linear
    @ModuleInfo(key: "gate_attn") var outputGate: Linear
    @ModuleInfo(key: "wr_du") var relativeLinear: Linear
    @ModuleInfo(key: "attn_sink_proj") var sinkProjection: Linear
    @ModuleInfo(key: "q_l2_clip") var queryClip: Inkling2ScaledL2Clip
    @ModuleInfo(key: "rel_l2_clip") var relativeClip: Inkling2ScaledL2Clip
    @ModuleInfo(key: "rel_logits_proj") var relativeProjection: Inkling2RelativeProjection
    @ModuleInfo(key: "k_sconv") var keyConvolution: Module
    @ModuleInfo(key: "v_sconv") var valueConvolution: Module

    private let _decodeProjections = Inkling2DecodeProjections()

    init(_ c: Inkling2TextConfiguration, layer: Int) {
        isFull = !c.isSliding(layer: layer)
        heads = c.attentionHeads
        kvHeads = c.kvHeads
        headDim = c.headDim
        window = isFull ? nil : c.slidingWindow
        scale = pow(Float(c.headDim), -0.5)
        qkThreshold = c.qkClipThreshold
        eps = c.rmsNormEps
        logScalingFloor = c.logScalingFloor
        logScalingAlpha = c.logScalingAlpha
        let hidden = c.hiddenSize
        let qWidth = heads * headDim
        let kvWidth = kvHeads * headDim
        let extent = isFull ? c.relativeExtent : c.slidingWindow
        _queryProjection.wrappedValue = Linear(hidden, qWidth, bias: false)
        _keyProjection.wrappedValue = Linear(hidden, kvWidth, bias: false)
        _valueProjection.wrappedValue = Linear(hidden, kvWidth, bias: false)
        _outputProjection.wrappedValue = Linear(qWidth, hidden, bias: false)
        _outputGate.wrappedValue = Linear(hidden, qWidth, bias: false)
        _relativeLinear.wrappedValue = Linear(hidden, heads * c.relativeDimensions, bias: false)
        _sinkProjection.wrappedValue = Linear(hidden, heads, bias: false)
        _queryClip.wrappedValue = Inkling2ScaledL2Clip(
            heads: heads, dimensions: headDim, threshold: c.qkClipThreshold)
        _relativeClip.wrappedValue = Inkling2ScaledL2Clip(
            heads: heads, dimensions: c.relativeDimensions, threshold: c.relativeClipThreshold)
        _relativeProjection.wrappedValue = Inkling2RelativeProjection(
            dimensions: c.relativeDimensions, extent: extent)
        if isFull {
            _keyConvolution.wrappedValue = Inkling2DynamicShortConvolution(
                channels: kvWidth, inputDimensions: hidden, heads: kvHeads,
                kernel: c.convKernelSize)
            _valueConvolution.wrappedValue = Inkling2DynamicShortConvolution(
                channels: kvWidth, inputDimensions: hidden, heads: kvHeads,
                kernel: c.convKernelSize)
        } else {
            _keyConvolution.wrappedValue = Inkling2ShortConvolution(
                channels: kvWidth, kernel: c.convKernelSize)
            _valueConvolution.wrappedValue = Inkling2ShortConvolution(
                channels: kvWidth, kernel: c.convKernelSize)
        }
    }

    private var projectionModules: [Module] {
        [
            queryProjection, relativeLinear, sinkProjection, outputGate, keyProjection,
            valueProjection, keyConvolution, valueConvolution,
        ]
    }

    /// Concatenate the projections that share an input and install the
    /// separate projections as row views of the result (`LanguageModel.prepare()`).
    /// A row-exact step then streams one matrix per input instead of up to six.
    /// Each output column equals its separate product: the row-exact route
    /// (`Inkling2RowExact.matmul`, at most five rows) runs MLX's multi-vector
    /// gemv, whose per-column K reduction does not depend on the matrix height
    /// (Inkling2ConcatenatedProjectionTests checks this per device).
    func prepareDecodeProjections() {
        _decodeProjections.clear()
        guard Inkling2Fused.concatenatedProjections else { return }
        let linears = [
            queryProjection, relativeLinear, sinkProjection, outputGate, keyProjection,
            valueProjection,
        ]
        guard
            linears.allSatisfy({
                ObjectIdentifier(type(of: $0)) == ObjectIdentifier(Linear.self) && $0.bias == nil
            }),
            Set(linears.map { $0.weight.dtype }).count == 1,
            // Half-precision weights take MLX's multi-vector gemv on the
            // row-exact route; FP32 small-M products are GEMM tiles whose
            // column reduction may depend on the matrix height.
            [DType.bfloat16, .float16].contains(queryProjection.weight.dtype)
        else { return }
        let dynamic = [keyConvolution, valueConvolution].compactMap {
            $0 as? Inkling2DynamicShortConvolution
        }
        guard dynamic.count == (isFull ? 2 : 0),
            dynamic.allSatisfy({ $0.weightProjection.dtype == queryProjection.weight.dtype })
        else { return }

        func rows(_ fused: MLXArray, _ sizes: [Int]) -> [MLXArray] {
            var start = 0
            return sizes.map { n in
                defer { start += n }
                return fused[start ..< start + n]
            }
        }
        let queryWeights = [
            queryProjection.weight, relativeLinear.weight, sinkProjection.weight,
            outputGate.weight,
        ]
        let queryRows = queryWeights.map { $0.dim(0) }.reduce(0, +)
        var views: [String: Linear] = [:]
        if isFull {
            let query = concatenated(queryWeights, axis: 0)
            let key = concatenated([keyProjection.weight, dynamic[0].weightProjection], axis: 0)
            let value = concatenated(
                [valueProjection.weight, dynamic[1].weightProjection], axis: 0)
            eval(query, key, value)
            let q = rows(query, queryWeights.map { $0.dim(0) })
            let k = rows(key, [keyProjection.weight.dim(0), dynamic[0].weightProjection.dim(0)])
            let v = rows(
                value, [valueProjection.weight.dim(0), dynamic[1].weightProjection.dim(0)])
            views = [
                "wq_du": Linear(weight: q[0]), "wr_du": Linear(weight: q[1]),
                "attn_sink_proj": Linear(weight: q[2]), "gate_attn": Linear(weight: q[3]),
                "wk_dv": Linear(weight: k[0]), "wv_dv": Linear(weight: v[0]),
            ]
            _ = dynamic[0].update(parameters: ModuleParameters.unflattened(["weight_proj": k[1]]))
            _ = dynamic[1].update(parameters: ModuleParameters.unflattened(["weight_proj": v[1]]))
            eval(dynamic[0].weightProjection, dynamic[1].weightProjection)
            _decodeProjections.query = query
            _decodeProjections.key = key
            _decodeProjections.value = value
        } else {
            let all = queryWeights + [keyProjection.weight, valueProjection.weight]
            let shared = concatenated(all, axis: 0)
            eval(shared)
            let parts = rows(shared, all.map { $0.dim(0) })
            views = [
                "wq_du": Linear(weight: parts[0]), "wr_du": Linear(weight: parts[1]),
                "attn_sink_proj": Linear(weight: parts[2]), "gate_attn": Linear(weight: parts[3]),
                "wk_dv": Linear(weight: parts[4]), "wv_dv": Linear(weight: parts[5]),
            ]
            _decodeProjections.shared = shared
            _decodeProjections.query = shared[0 ..< queryRows]
        }
        let buffers = _decodeProjections
        let saved = (buffers.shared, buffers.query, buffers.key, buffers.value)
        // updateModule clears the buffers; restore them once the views are in.
        try! update(
            modules: ModuleChildren(
                values: views.mapValues { NestedItem<String, Module>.value($0) }),
            verify: [])
        (buffers.shared, buffers.query, buffers.key, buffers.value) = saved
        eval([buffers.query!] + views.values.map(\.weight))
        buffers.sources = projectionModules.map { ObjectIdentifier($0) }
    }

    /// The concatenated buffers when they still back this module's projections.
    private var decodeProjections: Inkling2DecodeProjections? {
        let buffers = _decodeProjections
        guard buffers.query != nil,
            buffers.sources == projectionModules.map({ ObjectIdentifier($0) })
        else { return nil }
        return buffers
    }

    var hasDecodeProjections: Bool { decodeProjections != nil }

    /// Drops the concatenated buffers (parameters or modules replaced).
    func invalidateDecodeProjections() { _decodeProjections.clear() }

    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        invalidateDecodeProjections()
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    override func updateModule(key: String, _ value: Any) throws {
        invalidateDecodeProjections()
        try super.updateModule(key: key, value)
    }

    private func convolve(
        _ module: Module, _ x: MLXArray, input: MLXArray, slot: Int, kind: Inkling2ConvKind,
        state: Inkling2StepState, projected: MLXArray? = nil
    ) -> MLXArray {
        let previous = state.convState(slot: slot, kind)
        let (y, next): (MLXArray, MLXArray)
        if let dynamic = module as? Inkling2DynamicShortConvolution {
            (y, next) = dynamic(x, dynamicInput: input, state: previous, projected: projected)
        } else {
            (y, next) = (module as! Inkling2ShortConvolution)(x, state: previous)
        }
        state.setConvState(slot: slot, kind, next, input: x)
        return y
    }

    /// Inputs are `[B, L, hidden]` in the model dtype. Returns `[B, L, hidden]`.
    func callAsFunction(
        query x: MLXArray, key keyInput: MLXArray, value valueInput: MLXArray,
        slot: Int, state: Inkling2StepState
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let dtype = x.dtype
        let tau: MLXArray? =
            isFull
            ? Inkling2Math.logScale(
                positions: state.positions, floor: logScalingFloor, alpha: logScalingAlpha
            ).reshaped(-1, l, 1, 1)
            : nil
        func logScaled(_ v: MLXArray) -> MLXArray {
            guard let tau else { return v }
            return (v.asType(.float32) * tau).asType(v.dtype)
        }

        let project: (Linear, MLXArray) -> MLXArray = { Inkling2RowExact.linear($0, $1) }
        let products = concatenatedProducts(query: x, key: keyInput, value: valueInput)
        let queries = logScaled(
            queryClip(
                (products?.query ?? project(queryProjection, x)).reshaped(b, l, heads, headDim))
        )
        .transposed(0, 2, 1, 3)
        let relativeStates = logScaled(
            relativeClip(
                (products?.relative ?? project(relativeLinear, x)).reshaped(b, l, heads, -1)))
        let relativeLogits = Inkling2RowExact.matmul(
            relativeStates, transposedWeight: relativeProjection.proj.T
        ).transposed(0, 2, 1, 3)

        let keyStates = convolve(
            keyConvolution, products?.key ?? project(keyProjection, keyInput), input: keyInput,
            slot: slot, kind: .key, state: state, projected: products?.keyDynamic)
        let keys = Inkling2Math.l2Clip(
            keyStates.reshaped(b, l, kvHeads, headDim), threshold: qkThreshold
        ).asType(dtype).transposed(0, 2, 1, 3)
        let valueStates = convolve(
            valueConvolution, products?.value ?? project(valueProjection, valueInput),
            input: valueInput, slot: slot, kind: .value, state: state,
            projected: products?.valueDynamic)
        let values = Inkling2Math.unweightedRMSNorm(
            valueStates.reshaped(b, l, kvHeads, headDim), eps: eps
        ).asType(dtype).transposed(0, 2, 1, 3)
        let sinks = (products?.sink ?? project(sinkProjection, x)).transposed(0, 2, 1)

        let attended = state.attend(
            slot: slot,
            Inkling2AttentionInputs(
                queries: queries, keys: keys, values: values, relativeLogits: relativeLogits,
                relativeStates: relativeStates, relativeProjection: relativeProjection.proj,
                sinks: sinks, positions: state.positions, window: window, scale: scale,
                isPrefill: Inkling2PrefillStep.active))
        let gateLogits = products?.gate ?? project(outputGate, x)
        let gated =
            Inkling2Fused.active
            ? Inkling2Fused.outputGate(attended, gate: gateLogits)
            : (attended.asType(.float32) * sigmoid(gateLogits.asType(.float32)))
                .asType(attended.dtype)
        return project(outputProjection, gated)
    }

    private struct ConcatenatedProducts {
        var query: MLXArray
        var relative: MLXArray
        var sink: MLXArray
        var gate: MLXArray
        var key: MLXArray?
        var value: MLXArray?
        var keyDynamic: MLXArray?
        var valueDynamic: MLXArray?
    }

    /// The projections through the concatenated buffers, one matmul per
    /// distinct input; nil before `prepare()` or with fusion off. Row-exact
    /// steps: each column equals its separate product (multi-vector gemv).
    /// Prefill: a row's result is the same at any chunk length and row
    /// company, as with the separate products (probed per device;
    /// Inkling2ConcatenatedProjectionTests), though not their bits.
    private func concatenatedProducts(query x: MLXArray, key: MLXArray, value: MLXArray)
        -> ConcatenatedProducts?
    {
        guard Inkling2Fused.concatenationActive, let buffers = decodeProjections,
            let queryWeight = buffers.query
        else { return nil }
        func split(_ y: MLXArray, _ sizes: [Int]) -> [MLXArray] {
            var start = 0
            return sizes.map { n in
                defer { start += n }
                return y[.ellipsis, start ..< start + n]
            }
        }
        let querySizes = [
            queryProjection.weight.dim(0), relativeLinear.weight.dim(0),
            sinkProjection.weight.dim(0), outputGate.weight.dim(0),
        ]
        if let shared = buffers.shared, key === x, value === x {
            let parts = split(
                Inkling2RowExact.matmul(x, transposedWeight: shared),
                querySizes + [keyProjection.weight.dim(0), valueProjection.weight.dim(0)])
            return ConcatenatedProducts(
                query: parts[0], relative: parts[1], sink: parts[2], gate: parts[3],
                key: parts[4], value: parts[5])
        }
        let q = split(Inkling2RowExact.matmul(x, transposedWeight: queryWeight), querySizes)
        var products = ConcatenatedProducts(query: q[0], relative: q[1], sink: q[2], gate: q[3])
        if let keyWeight = buffers.key, let valueWeight = buffers.value {
            let kvRows = keyProjection.weight.dim(0)
            let k = Inkling2RowExact.matmul(key, transposedWeight: keyWeight)
            let v = Inkling2RowExact.matmul(value, transposedWeight: valueWeight)
            products.key = k[.ellipsis, ..<kvRows]
            products.keyDynamic = k[.ellipsis, kvRows...]
            products.value = v[.ellipsis, ..<valueProjection.weight.dim(0)]
            products.valueDynamic = v[.ellipsis, valueProjection.weight.dim(0)...]
        }
        return products
    }
}
