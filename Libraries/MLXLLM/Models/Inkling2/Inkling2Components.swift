// Inkling 2 building blocks. Reference: transformers PR 51050
// `modeling_inkling2_loop.py` (Apache-2.0). Dtypes follow the reference: the
// residual stream between layers is the model dtype, the mixes, gates,
// convolutions and routing run in FP32.
import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum Inkling2Math {
    /// `x * rsqrt(mean(x^2) + eps)` in FP32.
    static func unweightedRMSNorm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
        Inkling2Fused.active
            ? Inkling2Fused.unweightedRMSNorm(x, eps: eps) : plainUnweightedRMSNorm(x, eps: eps)
    }

    static func plainUnweightedRMSNorm(_ x: MLXArray, eps: Float) -> MLXArray {
        let x = x.asType(.float32)
        return x * rsqrt(mean(square(x), axis: -1, keepDims: true) + eps)
    }

    /// Scales each vector down to `threshold` when its L2 norm is larger. FP32.
    static func l2Clip(_ x: MLXArray, threshold: Float) -> MLXArray {
        Inkling2Fused.active
            ? Inkling2Fused.l2Clip(x, threshold: threshold) : plainL2Clip(x, threshold: threshold)
    }

    static func plainL2Clip(_ x: MLXArray, threshold: Float) -> MLXArray {
        let x = x.asType(.float32)
        let norm = sqrt(sum(square(x), axis: -1, keepDims: true))
        return x * minimum(threshold / (norm + 1e-6), 1)
    }

    static func softcap(_ x: MLXArray, _ cap: Float) -> MLXArray {
        tanh(x / cap) * cap
    }

    /// `1 + alpha * log(max((position + 1) / floor, 1))` in FP32, shape of `positions`.
    static func logScale(positions: MLXArray, floor: Int, alpha: Float) -> MLXArray {
        let ratio = (positions.asType(.float32) + 1) / Float(floor)
        return 1 + alpha * log(maximum(ratio, 1))
    }
}

/// RMS norm with a learned weight: `(w * normalize(x.f32))` in FP32. Callers
/// cast to the model dtype where the reference does.
final class Inkling2RMSNorm: Module {
    let weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        weight = MLXArray.ones([dimensions])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if Inkling2Fused.active {
            return Inkling2Fused.weightedRMSNorm(x, weight: weight, eps: eps)
        }
        return weight.asType(.float32) * Inkling2Math.plainUnweightedRMSNorm(x, eps: eps)
    }
}

/// `Inkling2LoopScaledL2Clip`: a learned per-head scale, then the L2 clip.
final class Inkling2ScaledL2Clip: Module {
    let scale: MLXArray
    let threshold: Float

    init(heads: Int, dimensions: Int, threshold: Float) {
        scale = MLXArray.ones([heads, dimensions])
        self.threshold = threshold
    }

    /// `x`: `[..., heads, dimensions]`. Returns `x.dtype`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if Inkling2Fused.active {
            return Inkling2Fused.scaledClip(x, scale: scale, threshold: threshold)
        }
        return Inkling2Math.plainL2Clip(
            x.asType(.float32) * scale.asType(.float32), threshold: threshold
        ).asType(x.dtype)
    }
}

/// The four short convolutions of one layer application.
public enum Inkling2ConvKind: Int, CaseIterable, Sendable {
    case key = 0
    case value = 1
    case postAttention = 2
    case postMLP = 3
}

/// Causal depthwise convolution with a carried `[B, kernel - 1, C]` state,
/// output `(conv + x) / sqrt(kernel + 1)` (`Inkling2LoopShortConvolution`).
/// `weight` is stored `[C, 1, kernel]` (index `kernel - 1` weights the current token).
final class Inkling2ShortConvolution: Module {
    let weight: MLXArray
    let kernel: Int

    init(channels: Int, kernel: Int) {
        weight = MLXArray.zeros([channels, 1, kernel])
        self.kernel = kernel
    }

    /// Returns the output in `x.dtype` and the next state in `x.dtype` (exact: the
    /// state holds input rows).
    func callAsFunction(_ x: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
        let (b, l, c) = (x.dim(0), x.dim(1), x.dim(2))
        let previous = state ?? MLXArray.zeros([b, kernel - 1, c], dtype: x.dtype)
        let taps = weight.reshaped(c, kernel).asType(.float32)
        if Inkling2Fused.active {
            return Inkling2Fused.shortConvolution(previous, x, taps: taps)
        }
        let context = concatenated([previous.asType(.float32), x.asType(.float32)], axis: 1)
        var out = context[0..., 0 ..< l, 0...] * taps[0..., 0]
        for j in 1 ..< kernel {
            out = out + context[0..., j ..< (j + l), 0...] * taps[0..., j]
        }
        let y = (out + x.asType(.float32)) / sqrt(Float(kernel + 1))
        let next = concatenated([previous, x], axis: 1)[0..., l ..< (l + kernel - 1), 0...]
        return (y.asType(x.dtype), next)
    }
}

/// Short convolution whose taps add per-token weights projected from
/// `dynamicInput`, shared within each head (`Inkling2LoopDynamicShortConvolution`).
/// `static_w` is stored `[kernel, C]` with tap index = backward distance.
final class Inkling2DynamicShortConvolution: Module {
    @ParameterInfo(key: "static_w") var staticWeight: MLXArray
    @ParameterInfo(key: "weight_proj") var weightProjection: MLXArray
    let kernel: Int
    let heads: Int

    init(channels: Int, inputDimensions: Int, heads: Int, kernel: Int) {
        _staticWeight.wrappedValue = MLXArray.zeros([kernel, channels])
        _weightProjection.wrappedValue = MLXArray.zeros([heads * kernel, inputDimensions])
        self.kernel = kernel
        self.heads = heads
    }

    /// `projected`: `dynamicInput @ weight_proj.T` when the caller already has it
    /// (the concatenated decode projections).
    func callAsFunction(
        _ x: MLXArray, dynamicInput: MLXArray, state: MLXArray?, projected: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let (b, l, c) = (x.dim(0), x.dim(1), x.dim(2))
        let dynamic =
            (projected
            ?? Inkling2RowExact.matmul(dynamicInput, transposedWeight: weightProjection))
            .asType(.float32)
            .reshaped(b, l, heads, 1, kernel)
        let taps =
            staticWeight.asType(.float32).T.reshaped(heads, c / heads, kernel) + dynamic
        let previous = state ?? MLXArray.zeros([b, kernel - 1, c], dtype: x.dtype)
        if Inkling2Fused.active {
            return Inkling2Fused.dynamicConvolution(previous, x, taps: taps)
        }
        let context = concatenated([previous.asType(.float32), x.asType(.float32)], axis: 1)
        var out: MLXArray? = nil
        for lag in 0 ..< kernel {
            let start = kernel - 1 - lag
            let window = context[0..., start ..< (start + l), 0...]
            let term = window * taps[.ellipsis, lag].reshaped(b, l, c)
            out = out.map { $0 + term } ?? term
        }
        let y = (out! + x.asType(.float32)) / sqrt(Float(2 * kernel + 1))
        let next = concatenated([previous, x], axis: 1)[0..., l ..< (l + kernel - 1), 0...]
        return (y.asType(x.dtype), next)
    }
}

/// `(1 - tanh(forget)) * h + tanh(gain) * m` with rank-`r` gates (`Inkling2LoopMessageGate`).
/// FP32 in and out; `down_weight` runs in the model dtype as in the reference.
final class Inkling2MessageGate: Module {
    @ParameterInfo(key: "down_weight") var down: MLXArray
    @ParameterInfo(key: "forget_weight") var forget: MLXArray
    @ParameterInfo(key: "gain_weight") var gain: MLXArray

    /// `[forget; gain]` after `prepare()`, with `forget` and `gain` installed
    /// as its row views (inference-only, outside the module topology).
    final class Fused: @unchecked Sendable { var weight: MLXArray? }
    private let _fused = Fused()

    /// One product for both gates. Each column equals its separate product:
    /// the per-row gemv (K = 64) and the prefill kernel reduce a column the
    /// same way whatever the matrix height (Inkling2ConcatenatedProjectionTests).
    func prepareFusedGates() {
        _fused.weight = nil
        guard Inkling2Fused.concatenatedProjections, forget.dtype == .float32,
            gain.dtype == .float32, forget.shape == gain.shape
        else { return }
        let rows = forget.dim(0)
        let fused = concatenated([forget, gain], axis: 0)
        eval(fused)
        _ = update(
            parameters: ModuleParameters.unflattened([
                "forget_weight": fused[0 ..< rows], "gain_weight": fused[rows...],
            ]))
        eval(forget, gain)
        _fused.weight = fused
    }

    func invalidateFusedGates() { _fused.weight = nil }

    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        invalidateFusedGates()
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    var hasFusedGates: Bool { _fused.weight != nil }

    init(hidden: Int, rank: Int) {
        _down.wrappedValue = MLXArray.zeros([rank, 2 * hidden])
        _forget.wrappedValue = MLXArray.zeros([hidden, rank], dtype: .float32)
        _gain.wrappedValue = MLXArray.zeros([hidden, rank], dtype: .float32)
    }

    func callAsFunction(_ h: MLXArray, message m: MLXArray) -> MLXArray {
        let h = h.asType(.float32)
        let m = m.asType(.float32)
        let mixed: MLXArray
        if !Inkling2RowExact.active, Inkling2GateInput.supported(h, m, dtype: down.dtype) {
            mixed = Inkling2GateInput.normalized(h, m, dtype: down.dtype)
        } else {
            mixed = concatenated(
                [Inkling2Math.unweightedRMSNorm(h), Inkling2Math.unweightedRMSNorm(m)], axis: -1
            ).asType(down.dtype)
        }
        let g = Inkling2RowExact.matmul(mixed, transposedWeight: down).asType(.float32)
        let fl: MLXArray
        let gl: MLXArray
        if Inkling2Fused.concatenationActive, let fused = _fused.weight {
            // Inkling2StageFusion: both products and the tail in one kernel, same bits.
            if let y = Inkling2StageFusion.messageGate(g, fused: fused, h: h, m: m) { return y }
            let both = Inkling2RowExact.fp32Rows(g, fused)
            let rows = forget.dim(0)
            (fl, gl) = (both[.ellipsis, ..<rows], both[.ellipsis, rows...])
        } else {
            fl = Inkling2RowExact.fp32Rows(g, forget)
            gl = Inkling2RowExact.fp32Rows(g, gain)
        }
        if Inkling2Fused.active {
            return Inkling2Fused.messageGateTail(h, m, f: fl, g: gl)
        }
        return (1 - tanh(fl)) * h + tanh(gl) * m
    }
}

/// Attention over the stack of completed block states plus the partial state
/// with learned queries; keys = values = the RMS-normed states
/// (`Inkling2LoopBlockAttentionResidual`). Returns `[..., queries, hidden]` FP32.
func inkling2AttentionResidual(
    queries: MLXArray, blocks: [MLXArray], partial: MLXArray
) -> MLXArray {
    if !Inkling2RowExact.active,
        Inkling2ResidualMix.supported(queries: queries, states: blocks + [partial])
    {
        return Inkling2ResidualMix.mix(queries: queries, states: blocks + [partial])
    }
    if Inkling2Fused.active {
        return Inkling2Fused.attentionResidual(queries: queries, blocks: blocks, partial: partial)
    }
    return plainAttentionResidual(queries: queries, blocks: blocks, partial: partial)
}

/// Scores and the weighted sum are per-row FP32 gemvs, the same arithmetic in
/// prefill and decode, independent of batch company and of TF32.
func plainAttentionResidual(
    queries: MLXArray, blocks: [MLXArray], partial: MLXArray
) -> MLXArray {
    let states = stacked(blocks.map { $0.asType(.float32) } + [partial.asType(.float32)], axis: -2)
    let normed = Inkling2Math.plainUnweightedRMSNorm(states, eps: 1e-6)
    let hidden = normed.dim(-1)
    let count = normed.dim(-2)
    let q = queries.asType(.float32).reshaped(-1, hidden)
    let nq = q.dim(0)
    let scores = Inkling2RowExact.fp32Rows(normed, q).swappedAxes(-1, -2) * pow(Float(hidden), -0.5)
    let weights = softmax(scores, axis: -1)
    var outer = normed.shape
    outer.removeLast(2)
    let rows = outer.reduce(1, *)
    let values = normed.reshaped(rows, count, hidden)
    let flatWeights = weights.reshaped(rows, nq, count)
    let mixed = (0 ..< nq).map { i in
        matmul(flatWeights[0..., i ..< i + 1, 0...], values)
    }
    let out = nq == 1 ? mixed[0] : concatenated(mixed, axis: 1)
    return out.reshaped(outer + [nq, hidden])
}

final class Inkling2BlockAttentionResidual: Module {
    let query: MLXArray

    init(hidden: Int, queries: Int) {
        query =
            queries == 1
            ? MLXArray.zeros([hidden], dtype: .float32)
            : MLXArray.zeros([queries, hidden], dtype: .float32)
    }

    /// `[..., queries, hidden]` FP32.
    func callAsFunction(blocks: [MLXArray], partial: MLXArray) -> MLXArray {
        inkling2AttentionResidual(queries: query, blocks: blocks, partial: partial)
    }
}

/// Learned message queries `[loop layers, 4, hidden]` (q, k, v, mlp).
final class Inkling2MessageAttentionResidual: Module {
    let query: MLXArray

    init(loopLayers: Int, hidden: Int) {
        query = MLXArray.zeros([loopLayers, 4, hidden], dtype: .float32)
    }

    /// Messages of loop layer `layer` alone, `[..., 4, hidden]` FP32: the same
    /// per-query arithmetic as `callAsFunction(blocks:partial:)[..., layer, 0..., 0...]`
    /// (the 4- and 48-query score products take the same gemv lanes; the
    /// prefill mix kernel computes each query independently), without the
    /// `[..., loop layers, 4, hidden]` tensor.
    func callAsFunction(layer: Int, blocks: [MLXArray], partial: MLXArray) -> MLXArray {
        inkling2AttentionResidual(queries: query[layer], blocks: blocks, partial: partial)
    }

    /// Messages of every loop layer at once: `[..., loop layers, 4, hidden]` FP32.
    func callAsFunction(blocks: [MLXArray], partial: MLXArray) -> MLXArray {
        let out = inkling2AttentionResidual(queries: query, blocks: blocks, partial: partial)
        var shape = out.shape
        shape.removeLast(2)
        return out.reshaped(shape + [query.dim(0), 4, query.dim(2)])
    }
}
