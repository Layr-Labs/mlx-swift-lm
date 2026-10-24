// Copyright © 2026 Eigen Labs.

import MLX
import MLXLMCommon
import MLXNN

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
    package var glm5NextPagedSelectionScoreBlockBytes: Int {
        GLM5NextSparseAttention.scoreBlockBytes
    }
    package var glm5NextPagedAttentionBlockBytes: Int {
        model.layers.lazy.compactMap { $0.attention as? GLM5NextDSAAttention }
            .map(\.absorbedBlockBytes).max() ?? GLM5NextDSAAttention.absorbedLatentBlockBytes
    }
    package var glm5NextPagedDenseHistoryPrefix: Bool {
        executionPolicy.sparseAttention == .absorbedDenseHistory
    }
    package var glm5NextPagedNativeSparseDecodeQueries: Int {
        executionPolicy.sparseDecode == .nativeKernels
            && executionPolicy.sparseAttention != .reference
            ? GLM5NextSparseDecodeKernel.maximumQueries : 0
    }
    /// Decode and MTP verification steps keep their per-query reads.
    package var glm5NextPagedLatentHistoryMinimumQueries: Int {
        GLM5NextSparseDecodeKernel.maximumQueries + 1
    }
    package var glm5NextPagedAbsorbedAttention: Bool {
        executionPolicy.sparseAttention != .reference
    }
    package var glm5NextPagedMetadataReads: GLM5NextPagedMetadataReads {
        executionPolicy.pagedMetadataReads
    }

    public func makeCBv2PagedBackend(
        kvBytesCapacity: Int, maximumBatchSize: Int, maximumQueryTokens: Int,
        maximumContextTokens: Int, maximumScratchBytes: Int,
        segmentBytes: Int, maximumBufferBytes: Int
    ) throws -> PagedKVBackend {
        let limits = try CBv2PagedGatheredAttentionLimits(
            maximumBatchSize: maximumBatchSize, maximumQueryTokens: maximumQueryTokens,
            maximumContextTokens: maximumContextTokens, maximumInFlightGraphs: 2,
            maximumScratchBytes: maximumScratchBytes, admissionMode: .stepOwned(.pinnedMetal))
        return try PagedKVBackend(
            glm5NextTarget: self,
            config: .init(
                capacityBytes: kvBytesCapacity, dtype: .float32,
                maxPrefillChunk: maximumQueryTokens,
                nominalMaxSequenceLength: maximumContextTokens, maxBufferLength: maximumBufferBytes,
                segmentSizeBytes: segmentBytes,
                layerDTypes: Array(repeating: .float32, count: cbv2LayerKinds.count),
                gatheredAttention: limits))
    }

    /// All persistent sparse payloads are actual K/V columns below; KDA uses
    /// the separately priced recurrent spec. No unpriced parallel history.
    public var cbv2TargetAuxiliaryAllocationSpecs: [CBv2AuxiliaryAllocationSpec]? { [] }

    public var cbv2NativeActivationDType: DType {
        (model.embedTokens as? GLM5NextPackedEmbedding)?.scales.dtype
            ?? model.embedTokens.weight.dtype
    }

    /// Dtypes of the values the FP32 DSA payload rows and KDA convolution
    /// history only widen, by the promotion each producer applies to the
    /// activation dtype: latents are kv_a then the kv_a RMSNorm weight, index
    /// keys are wk (its LayerNorm returns the input dtype), pool gates are the
    /// activations times the pool-gate weight, and the convolution history
    /// holds the raw q/k/v projections. The MTP prefix schema derives its own
    /// tensors the same way. Any non-uniform layer keeps every role FP32.
    package var glm5NextPagedPayloadStorage: CBv2GLM5NextCheckpointState.Storage? {
        let hidden = cbv2NativeActivationDType
        func promote(_ a: DType, _ b: DType) -> DType { a == b ? a : .float32 }
        func floating(_ value: MLXArray?) -> DType? {
            guard let value, [.float16, .bfloat16, .float32].contains(value.dtype) else {
                return nil
            }
            return value.dtype
        }
        func projection(_ module: Module, input: DType) -> DType? {
            // NVFP4 scales are FP8 codes; its FP32 global scale is cast back.
            if module is GLM5NextNVFP4Linear { return input }
            let parameters = Dictionary(uniqueKeysWithValues: module.parameters().flattened())
            guard let weight = parameters["weight"],
                var output =
                    (weight.dtype == .uint32
                    ? floating(parameters["scales"]) : floating(weight)).map({ promote(input, $0) })
            else { return nil }
            for name in ["biases", "bias"] where parameters[name] != nil {
                guard let dtype = floating(parameters[name]) else { return nil }
                output = promote(output, dtype)
            }
            return output
        }
        var latents = Set<DType>()
        var indexRows = Set<DType>()
        var convolutions = Set<DType>()
        for layer in model.layers {
            if let attention = layer.attention as? GLM5NextDSAAttention {
                guard let projected = projection(attention.kvA, input: hidden),
                    let norm = floating(attention.kvNorm.weight),
                    let keys = projection(attention.indexer.key, input: hidden),
                    let gate = floating(attention.indexer.poolGate)
                else { return nil }
                latents.insert(promote(projected, norm))
                let gates = promote(hidden, gate)
                indexRows.insert(keys == gates ? keys : .float32)
            } else if let attention = layer.attention as? GLM5NextKDASublayer {
                let members = [attention.q, attention.k, attention.v].map {
                    projection($0, input: hidden)
                }
                guard let first = members[0], members.allSatisfy({ $0 == first }) else {
                    return nil
                }
                convolutions.insert(first)
            }
        }
        guard latents.count == 1, indexRows.count == 1, convolutions.count == 1,
            let latent = latents.first.flatMap(CBv2CheckpointDType.init),
            let index = indexRows.first.flatMap(CBv2CheckpointDType.init),
            let convolution = convolutions.first.flatMap(CBv2CheckpointDType.init)
        else { return nil }
        let storage = CBv2GLM5NextCheckpointState.Storage(
            latent: latent, index: index, convolution: convolution)
        return storage == .float32 ? nil : storage
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
        return .init(
            layers: configuration.layerTypes.enumerated().compactMap { index, type in
                guard type == .linearAttention else { return nil }
                return .init(
                    modelLayerIndex: index,
                    convShape: [
                        1, 3 * linear.numHeads * linear.headDim, linear.shortConvKernelSize,
                    ],
                    convDType: .float32,
                    ssmShape: [1, linear.numHeads, linear.headDim, linear.headDim],
                    ssmDType: .float32)
            })
    }

    /// Lossless FP32 storage accommodates loaded projection dtype differences;
    /// consumers restore the trained dtype before pooling/projection. EngineV2
    /// adds its existing three-generation recurrent reservation. No
    /// duplicate GLM text cache or sparse owner is allocated beside these rows.
    public var cbv2NativeAdmissionConfig: AdmissionV2.Config {
        .init(
            elementBytes: DType.float32.size,
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
    ///
    /// MLX faults raised while constructing the graph are scoped to this call
    /// and rethrown as Swift errors (after the usual row rollback/failure
    /// record), never left to the process-wide handler. Checks run at each
    /// sublayer boundary so later Swift shape arithmetic never consumes a
    /// faulted (empty) array from an earlier layer.
    ///
    /// `captureRecurrent` (MTP capture-verify) runs the same window graph and
    /// stages every position's KDA state, so finalize can commit any accepted
    /// prefix without a repair forward.
    public func prepareCBv2Forward(
        tokens: MLXArray, caches: [any KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation],
        validity: MLXArray? = nil, positionIds: MLXArray? = nil,
        inputEmbeddings: MLXArray? = nil, requirement: CBv2PrefillRequirement? = nil,
        captureRecurrent: Bool = false
    ) throws -> GLM5NextTextOutput {
        try withError { fault in
            try constructCBv2Forward(
                tokens: tokens, caches: caches, recurrentState: recurrentState,
                validity: validity, positionIds: positionIds, inputEmbeddings: inputEmbeddings,
                requirement: requirement, captureRecurrent: captureRecurrent, fault: fault)
        }
    }

    private func constructCBv2Forward(
        tokens: MLXArray, caches: [any KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation],
        validity suppliedValidity: MLXArray?, positionIds: MLXArray?,
        inputEmbeddings: MLXArray?, requirement: CBv2PrefillRequirement?,
        captureRecurrent: Bool, fault: ErrorBox
    ) throws -> GLM5NextTextOutput {
        // Request-owned positions never reach here: EngineV2.validateRequest
        // refuses them at submit for native engines and for models without
        // positioned forwarding, which GLM does not declare.
        guard tokens.ndim == 2, tokens.dim(0) > 0, tokens.dim(1) > 0,
            [.int32, .uint32, .int64].contains(tokens.dtype),
            recurrentState.count == tokens.dim(0), positionIds == nil,
            !captureRecurrent || requirement == nil
        else { throw GLM5NextLoadError.invalid("native GLM token/state/position geometry") }
        let batch = tokens.dim(0)
        let length = tokens.dim(1)
        let kinds = cbv2LayerKinds
        guard !kinds.isEmpty, caches.count == kinds.count else {
            throw GLM5NextLoadError.invalid("native GLM requires all DSA storage rows")
        }
        let validity =
            suppliedValidity ?? GLM5NextKernelConstants.shared.allTrue([batch, length])
            ?? MLXArray.ones([batch, length], dtype: .bool)
        guard validity.shape == [batch, length], validity.dtype == .bool else {
            throw GLM5NextLoadError.invalid("native GLM validity geometry")
        }
        let streamShape = [batch, length, configuration.hcMult, configuration.hiddenSize]
        // Decode and verification steps materialize the token rows' streams in
        // one launch, with the bits of the lookup, dequantize and broadcast.
        let tokenStreams =
            inputEmbeddings == nil && length <= GLM5NextPackedEmbedding.maximumStreamTokens
            ? (model.embedTokens as? GLM5NextPackedEmbedding)?.streams(
                tokens, copies: configuration.hcMult)
            : nil
        let embeddings = tokenStreams == nil ? inputEmbeddings ?? embed(tokens) : nil
        try fault.check()
        if let tokenStreams {
            guard tokenStreams.shape == streamShape,
                tokenStreams.dtype == cbv2NativeActivationDType
            else {
                throw GLM5NextLoadError.invalid("native GLM activation dtype/shape")
            }
        } else if let embeddings {
            guard embeddings.shape == [batch, length, configuration.hiddenSize],
                embeddings.dtype == cbv2NativeActivationDType
            else {
                throw GLM5NextLoadError.invalid("native GLM activation dtype/shape")
            }
        }
        var attending: [any CBv2AttendingLayerCache] = []
        for (cache, kind) in zip(caches, kinds) {
            guard let cache = cache as? any CBv2AttendingLayerCache, cache.kind == kind,
                cache.rows.count == batch,
                cache.rows.allSatisfy({ row in
                    if let row = row as? CBv2FullSequenceKV {
                        return row.absoluteOffset <= row.maxLength - length
                    }
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
            attending.allSatisfy({ $0.rows.map(\.absoluteOffset) == starts })
        else {
            throw GLM5NextLoadError.invalid("native GLM layer frontiers disagree")
        }
        // `Streams` lets a norm-fused layer fold its final expand into the
        // next layer's collapse; the result is materialized once after the loop.
        var streams = GLM5NextTextLayer.Streams.materialized(
            tokenStreams
                ?? broadcast(embeddings!.expandedDimensions(axis: 2), to: streamShape))
        var staged: [(Int, GLM5NextLinearAttention.State)] = []
        var captured: [(Int, GLM5NextKDASublayer.CapturedStates)] = []
        var sparseIndex = 0
        do {
            for (index, layer) in model.layers.enumerated() {
                streams = try layer.forward(streams) { hidden in
                    try fault.check()
                    if let attention = layer.attention as? GLM5NextKDASublayer {
                        let states = try recurrentState.map {
                            evaluation -> GLM5NextLinearAttention.State in
                            if let state = evaluation.inputState(modelLayerIndex: index) {
                                guard let conv = state.conv, let ssm = state.ssm else {
                                    throw GLM5NextLoadError.invalid(
                                        "incomplete native KDA input state")
                                }
                                return .init(convolution: conv, recurrent: ssm)
                            }
                            let linear = configuration.linearAttention
                            return .init(
                                convolution: .zeros(
                                    [
                                        1, 3 * linear.numHeads * linear.headDim,
                                        linear.shortConvKernelSize,
                                    ], dtype: .float32),
                                recurrent: .zeros(
                                    [1, linear.numHeads, linear.headDim, linear.headDim],
                                    dtype: .float32))
                        }
                        let input = GLM5NextLinearAttention.State(
                            convolution: states.count == 1
                                ? states[0].convolution
                                : concatenated(states.map(\.convolution), axis: 0),
                            recurrent: states.count == 1
                                ? states[0].recurrent
                                : concatenated(states.map(\.recurrent), axis: 0))
                        let result = try attention.forwardCapturing(
                            hidden, state: input, validity: validity,
                            canonicalConvolution: true, capture: captureRecurrent)
                        try fault.check()
                        if let states = result.captured {
                            captured.append((index, states))
                        } else {
                            staged.append((index, result.state))
                        }
                        return result.values
                    }
                    guard let attention = layer.attention as? GLM5NextDSAAttention else {
                        throw GLM5NextLoadError.invalid("unknown native GLM layer")
                    }
                    let result = try attention.forwardCBv2(
                        hidden, cache: attending[sparseIndex], validity: validity,
                        targetIdentity: ObjectIdentifier(self))
                    try fault.check()
                    sparseIndex += 1
                    return result
                }
                try fault.check()
            }
            let final = try streams.materialize()
            try fault.check()
            let hidden = MLX.where(
                validity.expandedDimensions(axis: -1), model.norm(final.mean(axis: 2)), 0)
            let logits: MLXArray
            switch requirement {
            case .evaluationOnly?:
                logits = lmHead(lastValidHidden(hidden, validity: suppliedValidity))[0..., 0 ..< 1]
            case .lastPositionLogits?:
                logits = lmHead(lastValidHidden(hidden, validity: suppliedValidity))
            case nil: logits = lmHead(hidden)
            }
            try fault.check()
            // No KDA generation is staged until all 45 layers and the head have
            // constructed successfully. Each engine-owned evaluation stages once.
            for (index, next) in staged {
                for row in 0 ..< batch {
                    try recurrentState[row].stage(
                        modelLayerIndex: index,
                        conv: next.convolution[row ..< row + 1].asType(.float32),
                        ssm: next.recurrent[row ..< row + 1])
                }
            }
            // Position-major stacks: entry t is the state after window token
            // t, exactly what a committed (t + 1)-token prefix stores.
            for (index, states) in captured {
                for row in 0 ..< batch {
                    try recurrentState[row].stageCaptured(
                        modelLayerIndex: index,
                        conv: states.convolution[row].asType(.float32),
                        ssm: states.recurrent[row].asType(.float32), positions: length)
                }
            }
            return .init(logits: logits, hidden: hidden)
        } catch {
            for cache in attending {
                if let paged = cache as? PagedLayerCache {
                    paged.recordGLM5NextFailure(error)
                    continue  // engine failure retirement owns the pending page graph
                }
                for (row, start) in zip(cache.rows, starts) where row.absoluteOffset > start {
                    row.rollback(row.absoluteOffset - start)
                }
                cache.setRows(cache.rows)
            }
            throw error
        }
    }

    /// Each row's hidden state at its last valid column, `[B,H]`. Padded
    /// columns are zero, so reading column -1 of a right-padded row would
    /// produce logits of a zero vector. Rows without a valid column keep that
    /// zero vector, exactly as before; all-valid input reads column -1.
    private func lastValidHidden(_ hidden: MLXArray, validity: MLXArray?) -> MLXArray {
        guard let validity else { return hidden[0..., -1, 0...] }
        let length = hidden.dim(1)
        let columns = MLXArray.arange(length, dtype: .int32).reshaped([1, length])
        let last = MLX.where(validity, columns, Int32(-1)).argMax(axis: 1).asType(.int32)
        return takeAlong(hidden, last.reshaped([hidden.dim(0), 1, 1]), axis: 1)
            .squeezed(axis: 1)
    }

    public func cbv2Forward(
        _ tokens: MLXArray, caches: [any KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        cbv2NativeOutput(tokens, caches: caches, recurrentState: recurrentState).logits
    }

    public func cbv2ForwardWithHidden(
        _ tokens: MLXArray, caches: [any KVCache], recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        let value = cbv2NativeOutput(
            tokens, caches: caches, recurrentState: recurrentState, positionIds: positionIds)
        return (value.logits, value.hidden)
    }

    /// Capture-verify window of the native rectangular MTP profile. The generic
    /// steppable adapter deliberately does not expose GLM as a captured target.
    public func cbv2ForwardWithHiddenCaptured(
        _ tokens: MLXArray, caches: [any KVCache], recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        let value = cbv2NativeOutput(
            tokens, caches: caches, recurrentState: recurrentState, positionIds: positionIds,
            captureRecurrent: true)
        return (value.logits, value.hidden)
    }

    public func cbv2ForwardWithHiddenForPrefill(
        _ tokens: MLXArray, caches: [any KVCache], recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?, requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        let value = cbv2NativeOutput(
            tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds, requirement: requirement)
        // The recurrent MTP consumer removes the retained position axis. The
        // ordinary prefill seam below instead consumes already-narrowed rows.
        return (value.logits.expandedDimensions(axis: 1), value.hidden)
    }

    public func cbv2RecurrentPrefill(
        _ inputs: MLXArray, inputEmbedding: MLXArray?, cache: [any KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        cbv2NativeOutput(
            inputs, caches: cache ?? [], recurrentState: recurrentState,
            positionIds: positionIds, inputEmbeddings: inputEmbedding, requirement: requirement
        ).logits
    }

    private func cbv2NativeOutput(
        _ tokens: MLXArray, caches: [any KVCache], recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray? = nil, inputEmbeddings: MLXArray? = nil,
        requirement: CBv2PrefillRequirement? = nil, captureRecurrent: Bool = false
    ) -> GLM5NextTextOutput {
        do {
            return try prepareCBv2Forward(
                tokens: tokens, caches: caches, recurrentState: recurrentState,
                positionIds: positionIds, inputEmbeddings: inputEmbeddings,
                requirement: requirement, captureRecurrent: captureRecurrent)
        } catch {
            let paged = caches.compactMap { $0 as? PagedLayerCache }
            if !paged.isEmpty {
                for cache in paged { cache.recordGLM5NextFailure(error) }
                // The existing checkedModelForward boundary observes this fault
                // before sampling. These are unwind shapes, never served logits.
                let shape =
                    requirement == nil
                    ? [tokens.dim(0), tokens.dim(1), vocabularySize]
                    : [tokens.dim(0), requirement == .evaluationOnly ? 1 : vocabularySize]
                return .init(
                    logits: .zeros(shape),
                    hidden: .zeros([tokens.dim(0), tokens.dim(1), configuration.hiddenSize]))
            }
            // Like the existing recurrent adapter contracts, invalid internal
            // bindings cannot return plausible logits through a nonthrowing seam.
            preconditionFailure("native GLM forward contract violated: \(error)")
        }
    }
}
