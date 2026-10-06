// Copyright 2026 bstnxbt
// SPDX-License-Identifier: Apache-2.0
// Swift port of dflash-mlx DFlash2 at the revision recorded in NOTICE.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

func dflash2AttentionMask(
    blockLength: Int,
    queryOffset: Int,
    keyLength: Int,
    slidingWindow: Int,
    keyPositions explicitKeyPositions: MLXArray? = nil
) -> MLXArray {
    let queryPositions = MLXArray(
        Int32(queryOffset) ..< Int32(queryOffset + blockLength))[0..., .newAxis]
    let keyPositions: MLXArray
    if let explicitKeyPositions {
        keyPositions = explicitKeyPositions[.newAxis, 0...]
    } else {
        let keyStart = queryOffset + blockLength - keyLength
        keyPositions =
            MLXArray(
                Int32(keyStart) ..< Int32(keyStart + keyLength))[.newAxis, 0...]
    }
    let context =
        (keyPositions .< queryOffset)
        & ((queryPositions - keyPositions) .< slidingWindow)
    let proposal = keyPositions .>= queryOffset
    return context | proposal
}

final class DFlash2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        _up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        down(silu(gate(hidden)) * up(hidden))
    }
}

final class DFlash2Attention: Module {
    let heads: Int
    let keyValueHeads: Int
    let headDimension: Int
    let slidingWindow: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm

    private let rope: RoPELayer

    init(configuration: DFlash2Configuration) {
        heads = configuration.attentionHeads
        keyValueHeads = configuration.keyValueHeads
        headDimension = configuration.headDimension
        slidingWindow = configuration.slidingWindow
        scale = 1 / sqrt(Float(configuration.headDimension))
        _queryProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.attentionHeads * configuration.headDimension,
            bias: false)
        _keyProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.keyValueHeads * configuration.headDimension,
            bias: false)
        _valueProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.keyValueHeads * configuration.headDimension,
            bias: false)
        _outputProjection.wrappedValue = Linear(
            configuration.attentionHeads * configuration.headDimension,
            configuration.hiddenSize,
            bias: false)
        _queryNorm.wrappedValue = RMSNorm(
            dimensions: configuration.headDimension,
            eps: Float(configuration.rmsNormEpsilon))
        _keyNorm.wrappedValue = RMSNorm(
            dimensions: configuration.headDimension,
            eps: Float(configuration.rmsNormEpsilon))
        rope = initializeRope(
            dims: configuration.headDimension,
            base: Float(configuration.ropeTheta),
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: configuration.maxPositionEmbeddings)
        super.init()
    }

    func callAsFunction(
        _ hidden: MLXArray,
        targetContext: MLXArray,
        cache: DFlash2ContextKVCache
    ) -> MLXArray {
        let batch = hidden.dim(0)
        let blockLength = hidden.dim(1)
        let contextLength = targetContext.dim(1)
        let contextOffset = cache.offset
        let proposalOffset = contextOffset + contextLength

        var queries = queryNorm(
            queryProjection(hidden).reshaped([batch, blockLength, heads, headDimension])
        )
        .transposed(0, 2, 1, 3)
        var proposalKeys = keyNorm(
            keyProjection(hidden).reshaped(
                [batch, blockLength, keyValueHeads, headDimension])
        )
        .transposed(0, 2, 1, 3)
        let proposalValues = valueProjection(hidden)
            .reshaped([batch, blockLength, keyValueHeads, headDimension])
            .transposed(0, 2, 1, 3)

        queries = rope(queries, offset: proposalOffset)
        proposalKeys = rope(proposalKeys, offset: proposalOffset)

        if contextLength > 0 {
            let contextSpans = cache.contextSpans(inputLength: contextLength)
            let selectedContext =
                contextSpans.count == 1
                ? targetContext[0..., contextSpans[0], 0...]
                : concatenated(
                    contextSpans.map { targetContext[0..., $0, 0...] }, axis: 1)
            let selectedContextLength = selectedContext.dim(1)
            var contextKeys = keyNorm(
                keyProjection(selectedContext).reshaped(
                    [batch, selectedContextLength, keyValueHeads, headDimension])
            )
            .transposed(0, 2, 1, 3)
            let contextValues = valueProjection(selectedContext)
                .reshaped([batch, selectedContextLength, keyValueHeads, headDimension])
                .transposed(0, 2, 1, 3)
            var ropedContextSegments = [MLXArray]()
            var contextPositionSegments = [MLXArray]()
            var cursor = 0
            for span in contextSpans {
                let length = span.count
                ropedContextSegments.append(
                    rope(
                        contextKeys[0..., 0..., cursor ..< (cursor + length), 0...],
                        offset: contextOffset + span.lowerBound))
                contextPositionSegments.append(
                    MLXArray(
                        Int32(contextOffset + span.lowerBound)
                            ..< Int32(contextOffset + span.upperBound)))
                cursor += length
            }
            contextKeys =
                ropedContextSegments.count == 1
                ? ropedContextSegments[0]
                : concatenated(ropedContextSegments, axis: 2)
            let contextPositions =
                contextPositionSegments.count == 1
                ? contextPositionSegments[0]
                : concatenated(contextPositionSegments, axis: 0)
            cache.appendContext(
                keys: contextKeys,
                values: contextValues,
                positions: contextPositions,
                inputLength: contextLength)
        }
        let cachedKeys = cache.keys!
        let cachedValues = cache.values!
        let keys = concatenated([cachedKeys, proposalKeys], axis: 2)
        let values = concatenated([cachedValues, proposalValues], axis: 2)
        let proposalPositions = MLXArray(
            Int32(proposalOffset) ..< Int32(proposalOffset + blockLength))
        let keyPositions = concatenated(
            [cache.positions!, proposalPositions], axis: 0)
        let mask = dflash2AttentionMask(
            blockLength: blockLength,
            queryOffset: proposalOffset,
            keyLength: keys.dim(2),
            slidingWindow: slidingWindow,
            keyPositions: keyPositions)
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .array(mask))
        return outputProjection(
            attended.transposed(0, 2, 1, 3)
                .reshaped([batch, blockLength, heads * headDimension]))
    }

    @inline(__always)
    func appendProjectedContextCache(
        _ targetContext: MLXArray,
        cache: DFlash2ContextKVCache
    ) {
        let batch = targetContext.dim(0)
        let contextLength = targetContext.dim(1)
        guard contextLength > 0 else { return }
        let contextOffset = cache.offset
        let contextSpans = cache.contextSpans(inputLength: contextLength)
        let selectedContext: MLXArray
        if contextSpans.count == 1 {
            selectedContext = targetContext[0..., contextSpans[0], 0...]
        } else {
            selectedContext = concatenated(
                contextSpans.map { targetContext[0..., $0, 0...] }, axis: 1)
        }
        let selectedContextLength = selectedContext.dim(1)
        var keys = keyNorm(
            keyProjection(selectedContext).reshaped(
                [batch, selectedContextLength, keyValueHeads, headDimension])
        )
        .transposed(0, 2, 1, 3)
        let values = valueProjection(selectedContext)
            .reshaped([batch, selectedContextLength, keyValueHeads, headDimension])
            .transposed(0, 2, 1, 3)
        var ropedSegments = [MLXArray]()
        var positionSegments = [MLXArray]()
        var cursor = 0
        for span in contextSpans {
            let length = span.count
            ropedSegments.append(
                rope(
                    keys[0..., 0..., cursor ..< (cursor + length), 0...],
                    offset: contextOffset + span.lowerBound))
            positionSegments.append(
                MLXArray(
                    Int32(contextOffset + span.lowerBound)
                        ..< Int32(contextOffset + span.upperBound)))
            cursor += length
        }
        keys =
            ropedSegments.count == 1
            ? ropedSegments[0]
            : concatenated(ropedSegments, axis: 2)
        let positions =
            positionSegments.count == 1
            ? positionSegments[0]
            : concatenated(positionSegments, axis: 0)
        cache.appendContext(
            keys: keys,
            values: values,
            positions: positions,
            inputLength: contextLength)
    }
}

final class DFlash2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: DFlash2Attention
    @ModuleInfo var mlp: DFlash2MLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "attention_conv") var attentionConv: GroupedDynamicCausalConv
    @ModuleInfo(key: "mlp_conv") var mlpConv: GroupedDynamicCausalConv

    init(configuration: DFlash2Configuration) {
        _attention.wrappedValue = DFlash2Attention(configuration: configuration)
        _mlp.wrappedValue = DFlash2MLP(
            hiddenSize: configuration.hiddenSize,
            intermediateSize: configuration.intermediateSize)
        _inputNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: Float(configuration.rmsNormEpsilon))
        _postAttentionNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: Float(configuration.rmsNormEpsilon))
        _attentionConv.wrappedValue = GroupedDynamicCausalConv(
            hiddenSize: configuration.hiddenSize,
            kernelSize: configuration.convKernelSize,
            groupSize: configuration.convGroupSize)
        _mlpConv.wrappedValue = GroupedDynamicCausalConv(
            hiddenSize: configuration.hiddenSize,
            kernelSize: configuration.convKernelSize,
            groupSize: configuration.convGroupSize)
        super.init()
    }

    func callAsFunction(
        _ hidden: MLXArray,
        targetContext: MLXArray,
        cache: DFlash2ContextKVCache
    ) -> MLXArray {
        let residual = hidden
        let (attentionInput, attentionDynamic) = attentionConv.prepare(inputNorm(hidden))
        let attended = attention(
            attentionInput,
            targetContext: targetContext,
            cache: cache)
        let afterAttention = residual + attentionConv.finish(attended, dynamic: attentionDynamic)
        let (mlpInput, mlpDynamic) = mlpConv.prepare(postAttentionNorm(afterAttention))
        return afterAttention + mlpConv.finish(mlp(mlpInput), dynamic: mlpDynamic)
    }

    @inline(__always)
    func advanceProjectedContextCache(
        _ draftContext: MLXArray,
        cache: DFlash2ContextKVCache
    ) {
        attention.appendProjectedContextCache(draftContext, cache: cache)
    }
}
