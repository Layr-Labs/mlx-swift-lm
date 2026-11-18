// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN

/// `encoder`: the tapped target layers, concatenated, projected to the
/// drafter width and normed. The result feeds every layer's keys and values.
final class MuseGlimmerDFlashContextProjection: Module {
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "output_norm_enc") var outputNorm: RMSNorm

    init(_ c: MuseGlimmerDFlashConfiguration, dtype: DType) {
        _fc.wrappedValue = MuseGlimmerDFlashDraftModel.placeholder(
            c.hiddenSize, c.contextWidth, dtype)
        _outputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        super.init()
    }
}

final class MuseGlimmerDFlashAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ c: MuseGlimmerDFlashConfiguration, dtype: DType) {
        let queryWidth = c.attentionHeads * c.headDimensions
        let keyWidth = c.kvHeads * c.headDimensions
        _qProj.wrappedValue = MuseGlimmerDFlashDraftModel.placeholder(
            queryWidth, c.hiddenSize, dtype)
        _kProj.wrappedValue = MuseGlimmerDFlashDraftModel.placeholder(keyWidth, c.hiddenSize, dtype)
        _vProj.wrappedValue = MuseGlimmerDFlashDraftModel.placeholder(keyWidth, c.hiddenSize, dtype)
        _oProj.wrappedValue = MuseGlimmerDFlashDraftModel.placeholder(
            c.hiddenSize, queryWidth, dtype)
        _qNorm.wrappedValue = RMSNorm(dimensions: c.headDimensions, eps: c.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: c.headDimensions, eps: c.rmsNormEps)
        super.init()
    }
}

final class MuseGlimmerDFlashMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ c: MuseGlimmerDFlashConfiguration, dtype: DType) {
        _gate.wrappedValue = MuseGlimmerDFlashDraftModel.placeholder(
            c.intermediateSize, c.hiddenSize, dtype)
        _up.wrappedValue = MuseGlimmerDFlashDraftModel.placeholder(
            c.intermediateSize, c.hiddenSize, dtype)
        _down.wrappedValue = MuseGlimmerDFlashDraftModel.placeholder(
            c.hiddenSize, c.intermediateSize, dtype)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class MuseGlimmerDFlashLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var attention: MuseGlimmerDFlashAttention
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: MuseGlimmerDFlashMLP

    init(_ c: MuseGlimmerDFlashConfiguration, dtype: DType) {
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _attention.wrappedValue = MuseGlimmerDFlashAttention(c, dtype: dtype)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _mlp.wrappedValue = MuseGlimmerDFlashMLP(c, dtype: dtype)
        super.init()
    }
}

/// The Muse Glimmer DFlash block drafter.
///
/// One forward denoises a block `[anchor, mask, ...]` into draft hidden
/// states. Block queries attend to each other in both directions and to the
/// cached keys and values of committed target positions. Parameter names
/// equal the checkpoint's tensor names.
public final class MuseGlimmerDFlashDraftModel: Module {
    public let configuration: MuseGlimmerDFlashConfiguration
    /// Dtype of dense weights, norms and activations.
    public let activationDType: DType

    @ModuleInfo(key: "encoder") var encoder: MuseGlimmerDFlashContextProjection
    @ModuleInfo(key: "layers") var layers: [MuseGlimmerDFlashLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    /// Every weight is a lazy placeholder until a loader replaces it.
    public init(_ configuration: MuseGlimmerDFlashConfiguration, dtype: DType = .bfloat16) throws {
        try configuration.validate()
        guard [.bfloat16, .float16, .float32].contains(dtype) else {
            throw MuseGlimmerDFlashError.configuration("dtype")
        }
        self.configuration = configuration
        self.activationDType = dtype
        _encoder.wrappedValue = MuseGlimmerDFlashContextProjection(configuration, dtype: dtype)
        _layers.wrappedValue = (0 ..< configuration.hiddenLayers).map { _ in
            MuseGlimmerDFlashLayer(configuration, dtype: dtype)
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        super.init()
    }

    /// A dense projection whose weight is lazy zeros of the final shape.
    static func placeholder(_ output: Int, _ input: Int, _ dtype: DType) -> Linear {
        Linear(weight: MLXArray.zeros([output, input], dtype: dtype))
    }

    private var scale: Float { 1 / Float(configuration.headDimensions).squareRoot() }

    private func rope(_ x: MLXArray, at position: Int) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: configuration.headDimensions, traditional: false,
            base: configuration.ropeTheta, scale: 1, offset: position)
    }

    private func activation(_ x: MLXArray) -> MLXArray {
        x.dtype == activationDType ? x : x.asType(activationDType)
    }

    /// `output_norm_enc(fc(features))`: `[n, contextWidth]` to `[n, hiddenSize]`.
    public func projectContext(_ features: MLXArray) -> MLXArray {
        precondition(
            features.ndim == 2 && features.dim(1) == configuration.contextWidth,
            "DFlash context features must be [rows, contextWidth]")
        return encoder.outputNorm(encoder.fc(activation(features)))
    }

    /// Keys and values of projected context rows `[n, hiddenSize]` for every
    /// layer, `[layers, kvHeads, n, headDimensions]`. Keys carry `k_norm` and
    /// RoPE at absolute positions `position ..< position + n`. Context rows
    /// skip the layer's input norm.
    public func contextKeyValues(_ context: MLXArray, at position: Int) -> (
        keys: MLXArray, values: MLXArray
    ) {
        let c = configuration
        precondition(
            context.ndim == 2 && context.dim(1) == c.hiddenSize && context.dim(0) > 0,
            "DFlash projected context must be [rows, hiddenSize]")
        let rows = context.dim(0)
        let shape = [1, rows, c.kvHeads, c.headDimensions]
        var keys: [MLXArray] = []
        var values: [MLXArray] = []
        for layer in layers {
            let attention = layer.attention
            let key = attention.kNorm(
                attention.kProj(context).reshaped(shape).transposed(0, 2, 1, 3))
            keys.append(rope(key, at: position))
            values.append(attention.vProj(context).reshaped(shape).transposed(0, 2, 1, 3))
        }
        return (concatenated(keys, axis: 0), concatenated(values, axis: 0))
    }

    /// `contextKeyValues` for several runs of projected rows at once: one
    /// key and one value projection per layer over all of `context`, then
    /// `k_norm` and RoPE per segment at its own first position. Segment `i`
    /// covers `context` rows `segments[i].rows`; the result is in segment
    /// order.
    public func contextKeyValues(
        _ context: MLXArray, segments: [(rows: Range<Int>, position: Int)]
    ) -> [(keys: MLXArray, values: MLXArray)] {
        let c = configuration
        precondition(
            context.ndim == 2 && context.dim(1) == c.hiddenSize && !segments.isEmpty
                && segments.allSatisfy { !$0.rows.isEmpty && $0.rows.upperBound <= context.dim(0) },
            "DFlash projected context segments must be rows of [rows, hiddenSize]")
        if segments.count == 1, segments[0].rows == 0 ..< context.dim(0) {
            return [contextKeyValues(context, at: segments[0].position)]
        }
        var keys = [[MLXArray]](repeating: [], count: segments.count)
        var values = [[MLXArray]](repeating: [], count: segments.count)
        for layer in layers {
            let attention = layer.attention
            let projectedKeys = attention.kProj(context)
            let projectedValues = attention.vProj(context)
            for (index, segment) in segments.enumerated() {
                let shape = [1, segment.rows.count, c.kvHeads, c.headDimensions]
                let key = attention.kNorm(
                    projectedKeys[segment.rows, 0...].reshaped(shape).transposed(0, 2, 1, 3))
                keys[index].append(rope(key, at: segment.position))
                values[index].append(
                    projectedValues[segment.rows, 0...].reshaped(shape).transposed(0, 2, 1, 3))
            }
        }
        return (0 ..< segments.count).map {
            (concatenated(keys[$0], axis: 0), concatenated(values[$0], axis: 0))
        }
    }

    /// One block forward.
    /// - Parameters:
    ///   - embeddings: `[blockSize, hiddenSize]`, the target's raw embeddings
    ///     of the anchor and the mask tokens.
    ///   - position: absolute position of the anchor.
    ///   - context: keys and values `[layers, kvHeads, n, headDimensions]`
    ///     of the visible context rows, in any row order.
    ///   - mask: `[blockSize, n + blockSize]`, true where a block query may
    ///     attend; nil when every key is visible.
    /// - Returns: final-normed block hidden states `[blockSize, hiddenSize]`.
    public func blockForward(
        embeddings: MLXArray, at position: Int,
        context: (keys: MLXArray, values: MLXArray)?, mask: MLXArray?
    ) -> MLXArray {
        let c = configuration
        precondition(
            embeddings.shape == [c.blockSize, c.hiddenSize],
            "DFlash block embeddings must be [blockSize, hiddenSize]")
        let block = c.blockSize
        var hidden = activation(embeddings).reshaped([1, block, c.hiddenSize])
        for (index, layer) in layers.enumerated() {
            let attention = layer.attention
            let x = layer.inputNorm(hidden)
            let queries = rope(
                attention.qNorm(
                    attention.qProj(x).reshaped([1, block, c.attentionHeads, c.headDimensions])
                        .transposed(0, 2, 1, 3)),
                at: position)
            var keys = rope(
                attention.kNorm(
                    attention.kProj(x).reshaped([1, block, c.kvHeads, c.headDimensions])
                        .transposed(0, 2, 1, 3)),
                at: position)
            var values = attention.vProj(x).reshaped([1, block, c.kvHeads, c.headDimensions])
                .transposed(0, 2, 1, 3)
            if let context {
                keys = concatenated([context.keys[index ..< index + 1], keys], axis: 2)
                values = concatenated([context.values[index ..< index + 1], values], axis: 2)
            }
            let attended = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask)
            hidden =
                hidden
                + attention.oProj(
                    attended.transposed(0, 2, 1, 3).reshaped(
                        [1, block, c.attentionHeads * c.headDimensions]))
            hidden = hidden + layer.mlp(layer.postAttentionNorm(hidden))
        }
        return norm(hidden).reshaped([block, c.hiddenSize])
    }

    /// `blockForward` for several blocks in one pass. Every projection, norm
    /// and MLP runs over all blocks' rows, so each weight is read once per
    /// layer; RoPE and attention stay per block, at its own anchor over its
    /// own context and mask. Block `i` of the result is
    /// `blockForward(embeddings[i], at: positions[i], context: contexts[i],
    /// mask: masks[i])` up to the projections' accumulation order. One block
    /// is exactly `blockForward`.
    public func blockForwards(
        embeddings: [MLXArray], at positions: [Int],
        contexts: [(keys: MLXArray, values: MLXArray)?], masks: [MLXArray?]
    ) -> [MLXArray] {
        let c = configuration
        let n = embeddings.count
        precondition(
            n > 0 && positions.count == n && contexts.count == n && masks.count == n
                && embeddings.allSatisfy { $0.shape == [c.blockSize, c.hiddenSize] },
            "DFlash blocks must be [blockSize, hiddenSize] with one anchor, context and mask each")
        if n == 1 {
            return [
                blockForward(
                    embeddings: embeddings[0], at: positions[0], context: contexts[0],
                    mask: masks[0])
            ]
        }
        let block = c.blockSize
        var hidden = activation(
            concatenated(embeddings.map { $0.reshaped([1, block, c.hiddenSize]) }, axis: 0))
        for (index, layer) in layers.enumerated() {
            let attention = layer.attention
            let x = layer.inputNorm(hidden)
            let queries = attention.qNorm(
                attention.qProj(x).reshaped([n, block, c.attentionHeads, c.headDimensions])
                    .transposed(0, 2, 1, 3))
            let blockKeys = attention.kNorm(
                attention.kProj(x).reshaped([n, block, c.kvHeads, c.headDimensions])
                    .transposed(0, 2, 1, 3))
            let blockValues = attention.vProj(x).reshaped([n, block, c.kvHeads, c.headDimensions])
                .transposed(0, 2, 1, 3)
            var attended: [MLXArray] = []
            attended.reserveCapacity(n)
            for row in 0 ..< n {
                let rows = row ..< (row + 1)
                var keys = rope(blockKeys[rows], at: positions[row])
                var values = blockValues[rows]
                if let context = contexts[row] {
                    keys = concatenated([context.keys[index ..< index + 1], keys], axis: 2)
                    values = concatenated([context.values[index ..< index + 1], values], axis: 2)
                }
                attended.append(
                    MLXFast.scaledDotProductAttention(
                        queries: rope(queries[rows], at: positions[row]), keys: keys,
                        values: values, scale: scale, mask: masks[row]))
            }
            hidden =
                hidden
                + attention.oProj(
                    concatenated(attended, axis: 0).transposed(0, 2, 1, 3).reshaped(
                        [n, block, c.attentionHeads * c.headDimensions]))
            hidden = hidden + layer.mlp(layer.postAttentionNorm(hidden))
        }
        let normed = norm(hidden)
        return (0 ..< n).map { normed[$0].reshaped([block, c.hiddenSize]) }
    }

    /// Block embeddings `[blockSize, hiddenSize]` from the anchor row and the
    /// mask row, each `[1, hiddenSize]` or `[hiddenSize]`.
    public func blockEmbeddings(anchor: MLXArray, mask: MLXArray) -> MLXArray {
        let c = configuration
        let anchorRow = anchor.reshaped([1, c.hiddenSize])
        let maskRows = broadcast(
            mask.reshaped([1, c.hiddenSize]), to: [c.blockSize - 1, c.hiddenSize])
        return concatenated([activation(anchorRow), activation(maskRows)], axis: 0)
    }
}
