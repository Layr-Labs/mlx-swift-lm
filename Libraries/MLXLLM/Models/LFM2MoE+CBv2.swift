//
//  LFM2MoE+CBv2.swift
//  mlx-swift-lm
//
//  Continuous-batching v2 target surface for LFM2-MoE (LFM2.5-24B-A2B).
//  Full-attention layers own paged/contiguous CBv2 KV; short-convolution
//  layers own request-scoped convolution history only (no SSM state).
//  Equations are identical to the serial path in LFM2MoE.swift and to
//  mlx-lm 0.32.0 `lfm2_moe.py`.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Layers

extension LFM2MoEAttention {
    /// Rows have independent histories: RoPE uses the per-row device offsets
    /// captured before the cache advances.
    func cbv2Forward(_ x: MLXArray, cache: any CBv2AttendingLayerCache) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = qLayerNorm(qProj(x).reshaped(B, L, args.attentionHeads, -1))
            .transposed(0, 2, 1, 3)
        var keys = kLayerNorm(kProj(x).reshaped(B, L, args.kvHeads, -1))
            .transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        let offsets = cache.positionOffsets + 0
        queries = rope(queries, offset: offsets)
        keys = rope(keys, offset: offsets)

        let output = cache.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: nil
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return outProj(output)
    }
}

extension LFM2MoEShortConv {
    /// Request-owned convolution history. Every active row's `[1, L-1, H]`
    /// state is gathered into the rectangle, the depthwise convolution runs
    /// once, and each row's last `L-1` gated inputs are staged back into its
    /// own recurrent transaction (committed or rolled back by the engine).
    func cbv2Forward(
        _ x: MLXArray, modelLayerIndex: Int,
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        precondition(
            recurrentState.count == batch, "LFM2MoE CBv2 recurrent row count mismatch")

        let parts = inProj(x).split(parts: 3, axis: -1)
        let gatedInput = parts[0] * parts[2]

        let history = lCache - 1
        let rows = recurrentState.map { evaluation -> MLXArray in
            evaluation.inputState(modelLayerIndex: modelLayerIndex)?.conv
                ?? MLXArray.zeros([1, history, args.hiddenSize], dtype: gatedInput.dtype)
        }
        let state = rows.count == 1 ? rows[0] : concatenated(rows, axis: 0)
        let padded = concatenated([state, gatedInput], axis: 1)

        // A prompt chunk's tail is copied out so request state does not keep
        // the whole chunk's projection buffer alive; decode views are tiny.
        let tail = padded[0..., (padded.dim(1) - history)..., 0...]
        let nextState = length > 1 ? tail.contiguous() : tail
        for (row, evaluation) in recurrentState.enumerated() {
            do {
                try evaluation.stage(
                    modelLayerIndex: modelLayerIndex,
                    conv: nextState[row ..< row + 1], ssm: nil)
            } catch {
                preconditionFailure(
                    "LFM2MoE CBv2 recurrent stage failed at layer \(modelLayerIndex): \(error)")
            }
        }

        return outProj(parts[1] * conv(padded))
    }
}

extension LFM2MoEDecoderLayer {
    func cbv2Forward(
        _ x: MLXArray, modelLayerIndex: Int,
        attentionCache: (any CBv2AttendingLayerCache)?,
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        let normed = operatorNorm(x)
        let r: MLXArray
        if isAttentionLayer {
            guard let attentionCache else {
                preconditionFailure("LFM2MoE attention layer is missing its CBv2 cache")
            }
            r = attention!.cbv2Forward(normed, cache: attentionCache)
        } else {
            precondition(attentionCache == nil, "LFM2MoE convolution layer received attention KV")
            r = conv!.cbv2Forward(
                normed, modelLayerIndex: modelLayerIndex, recurrentState: recurrentState)
        }
        let h = x + r
        return h + feedForward(ffnNorm(h))
    }
}

extension LFM2MoEModelInner {
    func cbv2Forward(
        _ inputs: MLXArray,
        caches: [any CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        let observation =
            CBv2ForwardShapeObservation.isActive
            ? CBv2ForwardShapeObservation.beginTarget(
                liveBatchRows: inputs.dim(0), sequenceWidth: inputs.dim(1))
            : nil
        defer { observation?.end() }

        var hidden = embedTokens(inputs)
        var attentionIndex = 0
        for (modelLayerIndex, layer) in layers.enumerated() {
            let cache: (any CBv2AttendingLayerCache)?
            if layer.isAttentionLayer {
                guard attentionIndex < caches.count else {
                    preconditionFailure("LFM2MoE CBv2 attention cache count is too small")
                }
                cache = caches[attentionIndex]
                precondition(
                    cache!.kind.modelLayerIndex == nil
                        || cache!.kind.modelLayerIndex == modelLayerIndex,
                    "LFM2MoE CBv2 attention cache mapped to the wrong model layer")
                attentionIndex += 1
            } else {
                cache = nil
            }
            hidden = layer.cbv2Forward(
                hidden, modelLayerIndex: modelLayerIndex,
                attentionCache: cache, recurrentState: recurrentState)
        }
        precondition(
            attentionIndex == caches.count, "LFM2MoE CBv2 attention cache count is too large")
        return embeddingNorm(hidden)
    }
}

// MARK: - Model surface

extension LFM2MoEConfiguration {
    /// Compact CBv2 attention-storage layout: one entry per full-attention
    /// layer, mapped back to its transformer layer.
    public var cbv2LayerKinds: [CBv2LayerKind] {
        fullAttnIdxs.sorted().map { modelLayerIndex in
            CBv2LayerKind(
                attention: .full, headDim: headDimensions, kvHeads: kvHeads,
                queryHeads: attentionHeads, modelLayerIndex: modelLayerIndex)
        }
    }

    /// Convolution-only recurrent state for every non-attention layer.
    public func cbv2RecurrentStateSpec(activationDType: DType = .bfloat16)
        -> CBv2RecurrentStateSpec
    {
        cbv2RecurrentStateSpec(activationDTypes: [:], fallbackActivationDType: activationDType)
    }

    func cbv2RecurrentStateSpec(
        activationDTypes: [Int: DType], fallbackActivationDType: DType
    ) -> CBv2RecurrentStateSpec {
        let attention = Set(fullAttnIdxs)
        let layers = (0 ..< hiddenLayers).compactMap {
            modelLayerIndex -> CBv2RecurrentLayerStateSpec? in
            guard !attention.contains(modelLayerIndex) else { return nil }
            return .convolutionOnly(
                modelLayerIndex: modelLayerIndex,
                convShape: [1, max(0, convLCache - 1), hiddenSize],
                convDType: activationDTypes[modelLayerIndex] ?? fallbackActivationDType)
        }
        return CBv2RecurrentStateSpec(layers: layers)
    }

    public var cbv2Capabilities: CBv2ModelCapabilities {
        // Same qualified surface as Nemotron 3.5: exact hybrid checkpoints
        // (attention KV + convolution history) and native paged KV. No MTP
        // drafter exists for LFM2-MoE; compiled decode and packed prefill
        // stay fail-closed until independently qualified.
        var capabilities = CBv2ModelCapabilities.initialRecurrentTarget
        capabilities.supportsRecurrentCheckpointReuse = true
        capabilities.supportsPagedKV = true
        capabilities.requiresNativePagedKV = true
        capabilities.supportsMTP = false
        capabilities.supportsCompiledDecode = false
        capabilities.supportsPackedPrefill = false
        return capabilities
    }
}

extension LFM2MoEModel {
    public var cbv2LayerKinds: [CBv2LayerKind] { configuration.cbv2LayerKinds }

    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec {
        let probe = nativeStateProbe()
        return configuration.cbv2RecurrentStateSpec(
            activationDTypes: probe.convolution,
            fallbackActivationDType: model.embedTokens(MLXArray([0])).dtype)
    }

    public var cbv2Capabilities: CBv2ModelCapabilities { configuration.cbv2Capabilities }

    public func newCacheV2(
        makeLayerCache: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
            any CBv2AttendingLayerCache
    ) rethrows -> [any CBv2AttendingLayerCache] {
        try cbv2LayerKinds.enumerated().map { storageIndex, kind in
            try makeLayerCache(kind.modelLayerIndex ?? storageIndex, kind)
        }
    }

    /// Observe native activation types through a lazy one-token serial graph
    /// over private caches. Nothing is evaluated and no serving state is
    /// touched; quantized storage types are never mistaken for activations.
    func nativeStateProbe() -> (convolution: [Int: DType], attentionKV: [DType]?) {
        let caches = newCache(parameters: nil)
        _ = model(MLXArray([0]).reshaped(1, 1), cache: caches)
        var convolution: [Int: DType] = [:]
        var kv: [DType] = []
        for (modelLayerIndex, cache) in caches.enumerated() {
            if let mamba = cache as? MambaCache {
                if let state = mamba.state.first { convolution[modelLayerIndex] = state.dtype }
            } else {
                let state = cache.state
                guard state.count == 2, state[0].dtype == state[1].dtype,
                    [.float16, .bfloat16, .float32].contains(state[0].dtype)
                else { return (convolution, nil) }
                kv.append(state[0].dtype)
            }
        }
        return (convolution, kv.count == cbv2LayerKinds.count ? kv : nil)
    }

    func cbv2Hidden(
        _ inputs: MLXArray, caches: [KVCache], recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        let attending = caches.map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                preconditionFailure("LFM2MoE CBv2 target received a legacy KV cache")
            }
            return attending
        }
        return model.cbv2Forward(inputs, caches: attending, recurrentState: recurrentState)
    }

    func cbv2Logits(_ hidden: MLXArray) -> MLXArray {
        model.embedTokens.asLinear(hidden)
    }
}

extension LFM2MoEModel: CBv2RecurrentLanguageModelForwardable {
    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        cbv2Logits(cbv2Hidden(tokens, caches: caches, recurrentState: recurrentState))
    }
}

extension LFM2MoEModel: CBv2RecurrentLanguageModelPrefillForwardable {
    public var cbv2SupportsPackedPrefill: Bool { false }

    public func cbv2RecurrentPrefill(
        _ inputs: MLXArray,
        inputEmbedding: MLXArray?,
        cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        precondition(inputEmbedding == nil, "LFM2MoE is text-only")
        precondition(positionIds == nil, "LFM2MoE uses per-row cache RoPE offsets")
        let hidden = cbv2Hidden(inputs, caches: cache ?? [], recurrentState: recurrentState)
        switch requirement {
        case .evaluationOnly:
            return hidden[0..., -1, 0 ..< 1]
        case .lastPositionLogits:
            return cbv2Logits(hidden[0..., -1, 0...])
        }
    }
}

extension LFM2MoEModel: CBv2CompleteCheckpointKVTypeProviding {
    public var cbv2CompleteCheckpointKVDTypes: [DType]? { nativeStateProbe().attentionKV }
}
