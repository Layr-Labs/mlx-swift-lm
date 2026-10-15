// Copyright © 2026 Eigen Labs.

import MLX
import MLXLMCommon

extension GLM5NextTextModel: CBv2RecurrentMTPForwardable,
    CBv2RecurrentPrefillHiddenForwardable, CBv2RecurrentLanguageModelPrefillForwardable,
    CBv2TargetAuxiliaryAllocationProviding, CBv2CompleteCheckpointKVTypeProviding,
    GLM5NextPagedModelProviding
{
    /// The paged branch requires the actual package-issued GLM main-pool binding.
    /// Complete target checkpoints are supported; actual reuse still requires
    /// the bound codec/store and, for MTP, a real assistant checkpoint codec.
    /// This does not activate a provider default or a protected prefix profile.
    public var cbv2Capabilities: CBv2ModelCapabilities {
        var value = CBv2ModelCapabilities.initialRecurrentTarget
        value.supportsPagedKV = true
        value.requiresNativePagedKV = true
        value.supportsPrefixReuse = true
        value.supportsRecurrentCheckpointReuse = true
        return value
    }
    public var cbv2CompleteCheckpointKVDTypes: [DType]? {
        Array(repeating: .float32, count: cbv2LayerKinds.count)
    }
    package var glm5NextPagedKinds: [CBv2LayerKind] { cbv2LayerKinds }
    package var glm5NextPagedMaximumContext: Int { configuration.maxPositionEmbeddings }
    package var glm5NextPagedIndexHeads: Int { configuration.indexNHeads }
    package var glm5NextPagedIndexDimension: Int { configuration.indexHeadDim }
    package var glm5NextPagedSelectionWidth: Int { configuration.indexTopK + 3 }

    public func makeCBv2PagedBackend(
        kvBytesCapacity: Int, maximumBatchSize: Int, maximumQueryTokens: Int,
        maximumContextTokens: Int, maximumScratchBytes: Int,
        segmentBytes: Int, maximumBufferBytes: Int
    ) throws -> PagedKVBackend {
        let limits = try CBv2PagedGatheredAttentionLimits(
            maximumBatchSize: maximumBatchSize, maximumQueryTokens: maximumQueryTokens,
            maximumContextTokens: maximumContextTokens, maximumInFlightGraphs: 2,
            maximumScratchBytes: maximumScratchBytes, admissionMode: .stepOwned(.pinnedMetal))
        return try PagedKVBackend(glm5NextTarget: self, config: .init(
            capacityBytes: kvBytesCapacity, dtype: .float32, maxPrefillChunk: maximumQueryTokens,
            nominalMaxSequenceLength: maximumContextTokens, maxBufferLength: maximumBufferBytes,
            segmentSizeBytes: segmentBytes,
            layerDTypes: Array(repeating: .float32, count: cbv2LayerKinds.count), gatheredAttention: limits))
    }

    /// All persistent sparse payloads are actual K/V columns below; KDA uses
    /// the separately priced recurrent spec. No unpriced parallel history.
    public var cbv2TargetAuxiliaryAllocationSpecs: [CBv2AuxiliaryAllocationSpec]? { [] }

    public var cbv2NativeActivationDType: DType {
        (model.embedTokens as? GLM5NextPackedEmbedding)?.scales.dtype ?? model.embedTokens.weight.dtype
    }

    /// These are real FP32 storage widths, not expanded attention Q/K/V widths.
    /// All raw index and validity state is charged in the ordinary KV rows.
    public var cbv2LayerKinds: [CBv2LayerKind] {
        configuration.layerTypes.enumerated().compactMap { index, type in
            guard type == .deepseekSparseAttention else { return nil }
            return CBv2LayerKind(
                attention: .full, headDim: configuration.kvLoraRank,
                valueHeadDim: 2 * configuration.indexHeadDim + 1,
                kvHeads: 1, queryHeads: configuration.attentionHeads, modelLayerIndex: index)
        }
    }

    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec {
        let linear = configuration.linearAttention
        return .init(layers: configuration.layerTypes.enumerated().compactMap { index, type in
            guard type == .linearAttention else { return nil }
            return .init(
                modelLayerIndex: index,
                convShape: [1, 3 * linear.numHeads * linear.headDim, linear.shortConvKernelSize],
                convDType: .float32,
                ssmShape: [1, linear.numHeads, linear.headDim, linear.headDim], ssmDType: .float32)
        })
    }

    /// Lossless FP32 storage accommodates loaded projection dtype differences;
    /// consumers restore the trained dtype before pooling/projection. EngineV2
    /// adds its existing three-generation recurrent reservation. No
    /// duplicate GLM text cache or sparse owner is allocated beside these rows.
    public var cbv2NativeAdmissionConfig: AdmissionV2.Config {
        .init(elementBytes: DType.float32.size,
              layerElementBytes: Array(repeating: DType.float32.size, count: cbv2LayerKinds.count))
    }

    public func newCacheV2(
        makeLayerCache: (Int, CBv2LayerKind) throws -> any CBv2AttendingLayerCache
    ) rethrows -> [any CBv2AttendingLayerCache] {
        try cbv2LayerKinds.map { try makeLayerCache($0.modelLayerIndex!, $0) }
    }

    /// Builds one native engine step. KV suffixes are optimistic like ordinary
    /// CBv2 rows; the engine owns evaluation, discard rollback and retirement.
    /// KDA generations remain uncommitted until its existing finalize boundary.
    public func prepareCBv2Forward(
        tokens: MLXArray, caches: [any KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation],
        validity: MLXArray? = nil, positionIds: MLXArray? = nil,
        inputEmbeddings: MLXArray? = nil, requirement: CBv2PrefillRequirement? = nil
    ) throws -> GLM5NextTextOutput {
        guard tokens.ndim == 2, tokens.dim(0) > 0, tokens.dim(1) > 0,
            [.int32, .uint32, .int64].contains(tokens.dtype),
            recurrentState.count == tokens.dim(0), positionIds == nil
        else { throw GLM5NextLoadError.invalid("native GLM token/state/position geometry") }
        let batch = tokens.dim(0), length = tokens.dim(1)
        let kinds = cbv2LayerKinds
        guard !kinds.isEmpty, caches.count == kinds.count else {
            throw GLM5NextLoadError.invalid("native GLM requires all DSA storage rows")
        }
        let validity = validity ?? MLXArray.ones([batch, length], dtype: .bool)
        guard validity.shape == [batch, length], validity.dtype == .bool else {
            throw GLM5NextLoadError.invalid("native GLM validity geometry")
        }
        let embeddings = inputEmbeddings ?? embed(tokens)
        guard embeddings.shape == [batch, length, configuration.hiddenSize],
            embeddings.dtype == cbv2NativeActivationDType else {
            throw GLM5NextLoadError.invalid("native GLM activation dtype/shape")
        }
        var attending: [any CBv2AttendingLayerCache] = []
        for (cache, kind) in zip(caches, kinds) {
            guard let cache = cache as? any CBv2AttendingLayerCache, cache.kind == kind,
                cache.rows.count == batch,
                cache.rows.allSatisfy({ row in
                    if let row = row as? CBv2FullSequenceKV { return row.absoluteOffset <= row.maxLength - length }
                    if let row = row as? PagedSequenceKV, let paged = cache as? PagedLayerCache {
                        return paged.matchesGLM5NextTarget(ObjectIdentifier(self))
                            && row.absoluteOffset <= row.maxLength - length
                    }
                    return false
                })
            else { throw GLM5NextLoadError.invalid("native GLM requires its bound payload layout") }
            attending.append(cache)
        }
        let starts = attending[0].rows.map(\.absoluteOffset)
        guard starts.allSatisfy({ $0 <= configuration.maxPositionEmbeddings - length }),
            attending.allSatisfy({ $0.rows.map(\.absoluteOffset) == starts }) else {
            throw GLM5NextLoadError.invalid("native GLM layer frontiers disagree")
        }
        var streams = broadcast(embeddings.expandedDimensions(axis: 2),
            to: [batch, length, configuration.hcMult, configuration.hiddenSize])
        var staged: [(Int, GLM5NextLinearAttention.State)] = []
        var sparseIndex = 0
        do {
            for (index, layer) in model.layers.enumerated() {
                streams = try layer.forward(streams) { hidden in
                    if let attention = layer.attention as? GLM5NextKDASublayer {
                        let states = try recurrentState.map { evaluation -> GLM5NextLinearAttention.State in
                            if let state = evaluation.inputState(modelLayerIndex: index) {
                                guard let conv = state.conv, let ssm = state.ssm else {
                                    throw GLM5NextLoadError.invalid("incomplete native KDA input state")
                                }
                                return .init(convolution: conv, recurrent: ssm)
                            }
                            let linear = configuration.linearAttention
                            return .init(
                                convolution: .zeros([1, 3 * linear.numHeads * linear.headDim, linear.shortConvKernelSize], dtype: .float32),
                                recurrent: .zeros([1, linear.numHeads, linear.headDim, linear.headDim], dtype: .float32))
                        }
                        let input = GLM5NextLinearAttention.State(
                            convolution: states.count == 1 ? states[0].convolution : concatenated(states.map(\.convolution), axis: 0),
                            recurrent: states.count == 1 ? states[0].recurrent : concatenated(states.map(\.recurrent), axis: 0))
                        let result = try attention.forward(hidden, state: input, validity: validity,
                                                           canonicalConvolution: true)
                        staged.append((index, result.state))
                        return result.values
                    }
                    guard let attention = layer.attention as? GLM5NextDSAAttention else {
                        throw GLM5NextLoadError.invalid("unknown native GLM layer")
                    }
                    let result = try attention.forwardCBv2(hidden, cache: attending[sparseIndex], validity: validity,
                                                          targetIdentity: ObjectIdentifier(self))
                    sparseIndex += 1
                    return result
                }
            }
            let hidden = MLX.where(validity.expandedDimensions(axis: -1), model.norm(streams.mean(axis: 2)), 0)
            let logits: MLXArray
            switch requirement {
            case .evaluationOnly?: logits = lmHead(hidden[0..., -1, 0...])[0..., 0..<1]
            case .lastPositionLogits?: logits = lmHead(hidden[0..., -1, 0...])
            case nil: logits = lmHead(hidden)
            }
            // No KDA generation is staged until all 45 layers and the head have
            // constructed successfully. Each engine-owned evaluation stages once.
            for (index, next) in staged {
                for row in 0..<batch {
                    try recurrentState[row].stage(modelLayerIndex: index,
                        conv: next.convolution[row..<row + 1].asType(.float32), ssm: next.recurrent[row..<row + 1])
                }
            }
            return .init(logits: logits, hidden: hidden)
        } catch {
            for cache in attending {
                if let paged = cache as? PagedLayerCache {
                    paged.recordGLM5NextFailure(error)
                    continue // engine failure retirement owns the pending page graph
                }
                for (row, start) in zip(cache.rows, starts) where row.absoluteOffset > start {
                    row.rollback(row.absoluteOffset - start)
                }
                cache.setRows(cache.rows)
            }
            throw error
        }
    }

    public func cbv2Forward(_ tokens: MLXArray, caches: [any KVCache],
                             recurrentState: [CBv2RecurrentStateEvaluation]) -> MLXArray {
        cbv2NativeOutput(tokens, caches: caches, recurrentState: recurrentState).logits
    }

    public func cbv2ForwardWithHidden(
        _ tokens: MLXArray, caches: [any KVCache], recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        let value = cbv2NativeOutput(tokens, caches: caches, recurrentState: recurrentState, positionIds: positionIds)
        return (value.logits, value.hidden)
    }

    public func cbv2ForwardWithHiddenForPrefill(
        _ tokens: MLXArray, caches: [any KVCache], recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?, requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        let value = cbv2NativeOutput(tokens, caches: caches, recurrentState: recurrentState,
                                     positionIds: positionIds, requirement: requirement)
        // The recurrent MTP consumer removes the retained position axis. The
        // ordinary prefill seam below instead consumes already-narrowed rows.
        return (value.logits.expandedDimensions(axis: 1), value.hidden)
    }

    public func cbv2RecurrentPrefill(
        _ inputs: MLXArray, inputEmbedding: MLXArray?, cache: [any KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?, requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        cbv2NativeOutput(inputs, caches: cache ?? [], recurrentState: recurrentState,
                        positionIds: positionIds, inputEmbeddings: inputEmbedding, requirement: requirement).logits
    }

    private func cbv2NativeOutput(
        _ tokens: MLXArray, caches: [any KVCache], recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray? = nil, inputEmbeddings: MLXArray? = nil,
        requirement: CBv2PrefillRequirement? = nil
    ) -> GLM5NextTextOutput {
        do {
            return try prepareCBv2Forward(tokens: tokens, caches: caches, recurrentState: recurrentState,
                positionIds: positionIds, inputEmbeddings: inputEmbeddings, requirement: requirement)
        } catch {
            let paged = caches.compactMap { $0 as? PagedLayerCache }
            if !paged.isEmpty {
                for cache in paged { cache.recordGLM5NextFailure(error) }
                // The existing checkedModelForward boundary observes this fault
                // before sampling. These are unwind shapes, never served logits.
                let shape = requirement == nil ? [tokens.dim(0), tokens.dim(1), vocabularySize]
                    : [tokens.dim(0), requirement == .evaluationOnly ? 1 : vocabularySize]
                return .init(logits: .zeros(shape), hidden: .zeros([tokens.dim(0), tokens.dim(1), configuration.hiddenSize]))
            }
            // Like the existing recurrent adapter contracts, invalid internal
            // bindings cannot return plausible logits through a nonthrowing seam.
            preconditionFailure("native GLM forward contract violated: \(error)")
        }
    }
}
