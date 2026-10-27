// Copyright 2023-2024 SGLang Team
// Copyright contributors to the vLLM project
// Modified by Eigen Labs in 2026: native trained NextN module and request-local state.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

import MLX
import MLXLMCommon
import MLXNN

// Authoritative trained NextN sources (Transformers omits this layer):
// sgl-project/sglang@a6a8c56015b8dbca4147634fa9ef22c7c836de05
//   python/sglang/srt/models/{glm5_next_nextn,deepseek_nextn}.py
//   python/sglang/srt/layers/layernorm.py (RMSNorm.forward_native)
// vllm-project/vllm@273a04964ed9d1e389c4a7cb9d660e8294782f8d
//   vllm/models/glm5next/common/{mtp,model}.py
//   vllm/models/glm5next/nvidia/ops/fused_eh_norm.py
//   vllm/v1/spec_decode/llm_base_proposer.py (per-round index sharing)

public enum GLM5NextMTPError: Error, Equatable {
    case invalid(String)
    case incompatibleState
    case releasedState
    case activeRound
    case noActiveRound
    case unsupportedDraftDepth(Int)
}

public struct GLM5NextMTPOutput {
    /// All post-shared_head.norm rows [1,T,D]; the last row is recycled unchanged.
    public let hidden: MLXArray
    /// Last-position readout [1,1,V], including during full-prompt priming.
    public let logits: MLXArray
    public var greedyTokens: MLXArray { argMax(logits, axis: -1).asType(.int32) }
}

/// Actual paged attention reads stay leased through the complete MTP readout.
/// The caller, not this value, commits or rolls back the sparse transaction.
public struct GLM5NextMTPPagedOutput {
    public let output: GLM5NextMTPOutput
    private let attention: GLM5NextDSAAttention.StagedOutput

    fileprivate init(output: GLM5NextMTPOutput, attention: GLM5NextDSAAttention.StagedOutput) {
        self.output = output
        self.attention = attention
    }

    public func withEvaluationLease(_ fence: ([MLXArray]) throws -> Void) rethrows {
        try attention.withEvaluationLease { roots in
            try fence(roots + [output.hidden, output.logits])
        }
    }
}

private final class GLM5NextMTPBinding {}

// Do not register either target owner as a child of this Module. Only the
// shared embedding/head are children; a target-owned MTP would otherwise cycle.
private final class GLM5NextMTPTargetReference {
    private weak var model: GLM5NextTextModel?
    private weak var stage: GLM5NextTextStage?
    let identity: ObjectIdentifier
    let configuration: GLM5NextTextConfiguration
    let inventory: GLM5NextWeightInventory
    let executionPolicy: GLM5NextExecutionPolicy
    let embedding: Embedding
    let head: Linear

    init(_ model: GLM5NextTextModel) {
        self.model = model
        identity = ObjectIdentifier(model)
        configuration = model.configuration
        inventory = model.inventory
        executionPolicy = model.executionPolicy
        embedding = model.model.embedTokens
        head = model.lmHead
    }

    init(_ stage: GLM5NextTextStage) throws {
        guard stage.plan.isFinal, let embedding = stage.embedding, let head = stage.head else {
            throw GLM5NextMTPError.invalid(
                "MTP requires a final text stage with embedding and head")
        }
        self.stage = stage
        identity = ObjectIdentifier(stage)
        configuration = stage.plan.configuration
        inventory = stage.inventory
        executionPolicy = stage.executionPolicy
        self.embedding = embedding
        self.head = head
    }

    func matches(embedding: Embedding, head: Linear) -> Bool {
        if let model {
            return model.model.embedTokens === embedding && model.lmHead === head
        }
        if let stage {
            return stage.embedding === embedding && stage.head === head
        }
        return false
    }
}

/// In-process immutable prefix, not a durable codec or an admitted CBv2 lease.
public final class GLM5NextMTPCheckpoint {
    fileprivate let binding: GLM5NextMTPBinding
    fileprivate let cache: GLM5NextDSACache
    fileprivate let hidden: MLXArray?
    public var inputCount: Int { cache.offset }
    /// The retained history (and recycled hidden), for a caller's evaluation
    /// fence and byte accounting; a paged history's roots are its page
    /// launches' (`GLM5NextDSACache.innerState`), its bytes `pagedBytes`.
    public var arrays: [MLXArray] { cache.innerState() + [hidden].compactMap { $0 } }
    /// Bytes of the pages a paged history holds; nil when contiguous.
    public var pagedBytes: Int? { cache.pagedBytes }

    fileprivate init(binding: GLM5NextMTPBinding, cache: GLM5NextDSACache, hidden: MLXArray?) {
        self.binding = binding
        self.cache = cache.copy() as! GLM5NextDSACache
        self.hidden = hidden.map { $0.sharingArray() }
    }
}

/// Graph-only reference cache; never aliases the target's DSA/KDA state.
/// Its construction frontier is not an evaluated serving receipt. Active CBv2
/// uses a private execution-owner candidate and publishes only after its fence.
public final class GLM5NextMTPState {
    fileprivate let binding: GLM5NextMTPBinding
    fileprivate var cache = GLM5NextDSACache() {
        didSet { cache.capacityLimit = historyCapacityLimit }
    }
    fileprivate var hidden: MLXArray?
    fileprivate var round: GLM5NextMTPCheckpoint?
    fileprivate var released = false
    public let maximumInputs: Int
    /// The history's `GLM5NextDSACache.capacityLimit`, kept across every
    /// replacement of the history (trim, restore, reset). Nil is unbounded.
    public var historyCapacityLimit: Int? {
        didSet { cache.capacityLimit = historyCapacityLimit }
    }
    public var committedInputCount: Int { round?.inputCount ?? cache.offset }
    public var stagedInputCount: Int { cache.offset - committedInputCount }
    public var recycledHidden: MLXArray? { hidden.map { $0.sharingArray() } }
    /// Paged NextN history (`GLM5NextDSAPagePool`); nil keeps it contiguous.
    public let pagePool: GLM5NextDSAPagePool?
    /// The reservation the history's new pages count against.
    public var pageAccount: GLM5NextDSAPageBook.Account? {
        didSet { if pagePool != nil { cache.pageAccount = pageAccount } }
    }
    /// The NextN history's arrays (`GLM5NextDSACache.state`: gathered from
    /// its pages when paged), for comparisons and diagnostics.
    public var history: [MLXArray] { cache.state }

    fileprivate init(
        binding: GLM5NextMTPBinding, maximumInputs: Int, pagePool: GLM5NextDSAPagePool? = nil,
        pageAccount: GLM5NextDSAPageBook.Account? = nil
    ) {
        self.binding = binding
        self.maximumInputs = maximumInputs
        self.pagePool = pagePool
        self.pageAccount = pageAccount
        if let pagePool { cache = GLM5NextDSACache(pagePool: pagePool, account: pageAccount) }
    }

    /// An empty history in this state's storage.
    fileprivate func emptyHistory() -> GLM5NextDSACache {
        pagePool.map { GLM5NextDSACache(pagePool: $0, account: pageAccount) }
            ?? GLM5NextDSACache()
    }
}

/// SGLang/vLLM normalize the FP32 residual sum before casting it back, rather
/// than normalizing a prematurely rounded BF16 sum or applying a second head norm.
private final class GLM5NextMTPNorm: Module {
    var weight: MLXArray
    let epsilon: Float

    init(weight: MLXArray, epsilon: Float) {
        self.weight = weight
        self.epsilon = epsilon
    }

    func apply(_ input: MLXArray, residual: MLXArray? = nil)
        -> (normalized: MLXArray, residual: MLXArray)
    {
        let sum =
            residual.map { input.asType(.float32) + $0.asType(.float32) }
            ?? input.asType(.float32)
        let normalized = sum * rsqrt(sum.square().mean(axis: -1, keepDims: true) + epsilon)
        return ((normalized * weight.asType(.float32)).asType(input.dtype), sum.asType(input.dtype))
    }
}

private final class GLM5NextMTPSharedHead: Module {
    let norm: GLM5NextMTPNorm
    init(norm: GLM5NextMTPNorm) { self.norm = norm }
}

/// Actual checkpoint NextN layer (`layers.45` in the official model), not a
/// substitute drafter. It is deliberately not registered or enabled in CBv2.
public final class GLM5NextMTP: Module {
    public let configuration: GLM5NextTextConfiguration
    public let maximumDraftTokens = 1
    /// Identity of the full model or final stage that owns the shared readout.
    public let targetIdentity: ObjectIdentifier
    /// Always the target's load-time policy; this layer's attention must match it.
    public let executionPolicy: GLM5NextExecutionPolicy
    private let targetReference: GLM5NextMTPTargetReference
    private let binding = GLM5NextMTPBinding()
    private let embedding: Embedding
    private let head: Linear
    @ModuleInfo(key: "enorm") private var embeddingNorm: GLM5NextMTPNorm
    @ModuleInfo(key: "hnorm") private var hiddenNorm: GLM5NextMTPNorm
    @ModuleInfo(key: "eh_proj") private var projection: Linear
    @ModuleInfo(key: "self_attn") private var attention: GLM5NextDSAAttention
    @ModuleInfo(key: "input_layernorm") private var inputNorm: GLM5NextMTPNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postNorm: GLM5NextMTPNorm
    @ModuleInfo(key: "mlp") private var mlp: GLM5NextMoE
    @ModuleInfo(key: "shared_head") private var sharedHead: GLM5NextMTPSharedHead

    /// Every module/vector is supplied from trained weights. No default head,
    /// random replacement module or missing-tensor fallback is constructed here.
    public convenience init(
        target: GLM5NextTextModel,
        embeddingNorm: MLXArray, hiddenNorm: MLXArray, projection: Linear,
        attention: GLM5NextDSAAttention, inputNorm: MLXArray, postNorm: MLXArray,
        mlp: Module & UnaryLayer, outputNorm: MLXArray
    ) throws {
        try self.init(
            target: GLM5NextMTPTargetReference(target),
            embeddingNorm: embeddingNorm, hiddenNorm: hiddenNorm, projection: projection,
            attention: attention, inputNorm: inputNorm, postNorm: postNorm,
            mlp: mlp, outputNorm: outputNorm)
    }

    /// Binds directly to the final stage's trained embedding/head. The caller
    /// must retain the stage; no complete text model or extra target layers are built.
    public convenience init(
        stage: GLM5NextTextStage,
        embeddingNorm: MLXArray, hiddenNorm: MLXArray, projection: Linear,
        attention: GLM5NextDSAAttention, inputNorm: MLXArray, postNorm: MLXArray,
        mlp: Module & UnaryLayer, outputNorm: MLXArray
    ) throws {
        try self.init(
            target: GLM5NextMTPTargetReference(stage),
            embeddingNorm: embeddingNorm, hiddenNorm: hiddenNorm, projection: projection,
            attention: attention, inputNorm: inputNorm, postNorm: postNorm,
            mlp: mlp, outputNorm: outputNorm)
    }

    private init(
        target: GLM5NextMTPTargetReference,
        embeddingNorm: MLXArray, hiddenNorm: MLXArray, projection: Linear,
        attention: GLM5NextDSAAttention, inputNorm: MLXArray, postNorm: MLXArray,
        mlp: Module & UnaryLayer, outputNorm: MLXArray
    ) throws {
        let c = target.configuration
        guard c.numNextNPredictLayers == 1, let moe = mlp as? GLM5NextMoE,
            attention.configuration == c,
            projection.shape.0 == c.hiddenSize, projection.shape.1 == 2 * c.hiddenSize,
            target.embedding.shape.0 == c.vocabularySize,
            target.embedding.shape.1 == c.hiddenSize,
            target.head.shape.0 == c.vocabularySize, target.head.shape.1 == c.hiddenSize,
            moe.experts.hiddenSize == c.hiddenSize, moe.experts.numExperts == c.nRoutedExperts,
            moe.experts.intermediateSize == c.moeIntermediateSize,
            moe.experts.swigluLimit == Float(c.swigluLimit),
            moe.sharedExperts.intermediateSize == c.moeIntermediateSize * c.nSharedExperts,
            moe.sharedExperts.swigluLimit == Float(c.swigluLimit),
            moe.gate.topK == c.numExpertsPerTok, moe.gate.nGroup == c.nGroup,
            moe.gate.topkGroup == c.topkGroup, moe.gate.normTopkProb == c.normTopkProb,
            moe.gate.routedScalingFactor == Float(c.routedScalingFactor)
        else { throw GLM5NextMTPError.invalid("trained MTP component geometry") }
        guard attention.executionPolicy == target.executionPolicy else {
            throw GLM5NextMTPError.invalid("MTP execution policy must match the target")
        }
        let vectors = [embeddingNorm, hiddenNorm, inputNorm, postNorm, outputNorm]
        guard vectors.allSatisfy({ $0.shape == [c.hiddenSize] && $0.dtype.isFloatingPoint }) else {
            throw GLM5NextMTPError.invalid("trained MTP normalization weights")
        }
        configuration = c
        targetIdentity = target.identity
        executionPolicy = target.executionPolicy
        targetReference = target
        embedding = target.embedding
        head = target.head
        let eps = Float(c.rmsNormEps)
        _embeddingNorm.wrappedValue = GLM5NextMTPNorm(weight: embeddingNorm, epsilon: eps)
        _hiddenNorm.wrappedValue = GLM5NextMTPNorm(weight: hiddenNorm, epsilon: eps)
        _projection.wrappedValue = projection
        _attention.wrappedValue = attention
        _inputNorm.wrappedValue = GLM5NextMTPNorm(weight: inputNorm, epsilon: eps)
        _postNorm.wrappedValue = GLM5NextMTPNorm(weight: postNorm, epsilon: eps)
        _mlp.wrappedValue = moe
        _sharedHead.wrappedValue = GLM5NextMTPSharedHead(
            norm: GLM5NextMTPNorm(weight: outputNorm, epsilon: eps))
        super.init()
    }

    /// Reads ONLY the target inventory's MTP partition, reusing its bounded tensor
    /// resolver and expert-stack materialization contract. Embedding/head are shared.
    public convenience init(
        target: GLM5NextTextModel, tensor: @escaping (String) throws -> MLXArray,
        materializeExpertStack: ((MLXArray) throws -> Void)? = nil,
        releaseSourceTensors: (([String]) -> Void)? = nil,
        materializeFusedProjections: (([MLXArray]) throws -> Void)? = nil
    ) throws {
        try self.init(
            target: GLM5NextMTPTargetReference(target), tensor: tensor,
            materializeExpertStack: materializeExpertStack,
            releaseSourceTensors: releaseSourceTensors,
            materializeFusedProjections: materializeFusedProjections)
    }

    /// Loads only the MTP partition after validating final-stage ownership.
    /// The embedding/head are already owned by the stage and are never reloaded.
    public convenience init(
        stage: GLM5NextTextStage, tensor: @escaping (String) throws -> MLXArray,
        materializeExpertStack: ((MLXArray) throws -> Void)? = nil,
        releaseSourceTensors: (([String]) -> Void)? = nil,
        materializeFusedProjections: (([MLXArray]) throws -> Void)? = nil,
        prepared: GLM5NextPreparedParts? = nil
    ) throws {
        try self.init(
            target: GLM5NextMTPTargetReference(stage), tensor: tensor,
            materializeExpertStack: materializeExpertStack,
            releaseSourceTensors: releaseSourceTensors,
            materializeFusedProjections: materializeFusedProjections, prepared: prepared)
    }

    private convenience init(
        target: GLM5NextMTPTargetReference, tensor: @escaping (String) throws -> MLXArray,
        materializeExpertStack: ((MLXArray) throws -> Void)?,
        releaseSourceTensors: (([String]) -> Void)?,
        materializeFusedProjections: (([MLXArray]) throws -> Void)?,
        prepared: GLM5NextPreparedParts? = nil
    ) throws {
        let c = target.configuration
        guard c.numNextNPredictLayers == 1 else {
            throw GLM5NextMTPError.invalid("exactly one trained NextN layer is required")
        }
        guard
            !target.inventory.storage.isPacked
                || (materializeExpertStack != nil && releaseSourceTensors != nil)
        else {
            throw GLM5NextMTPError.invalid(
                "Q4 MTP loading requires materialization and source-cache eviction")
        }
        let loader = GLM5NextTensorLoader(
            target.inventory, tensor, releaseSourceTensors: releaseSourceTensors,
            materializeFusedProjections: materializeFusedProjections)
        loader.prepared = prepared
        let p = "model.language_model.layers.\(c.hiddenLayers)"
        try self.init(
            target: target,
            embeddingNorm: loader.array(p + ".enorm.weight"),
            hiddenNorm: loader.array(p + ".hnorm.weight"),
            projection: loader.linear(p + ".eh_proj"),
            attention: GLM5NextDSAAttention(
                c, prefix: p + ".self_attn", loader: loader,
                executionPolicy: target.executionPolicy),
            inputNorm: loader.array(p + ".input_layernorm.weight"),
            postNorm: loader.array(p + ".post_attention_layernorm.weight"),
            mlp: loader.moe(c, prefix: p + ".mlp", materialize: materializeExpertStack),
            outputNorm: loader.array(p + ".shared_head.norm.weight"))
        if let missing = target.inventory.entries(in: .mtp).first(where: {
            !loader.consumed.contains($0.tensor.name)
        }) {
            throw GLM5NextLoadError.missing("unconsumed MTP tensor \(missing.tensor.name)")
        }
    }

    public func newRequestState(maximumInputs: Int? = nil) throws -> GLM5NextMTPState {
        try newRequestState(maximumInputs: maximumInputs, pagePool: nil, pageAccount: nil)
    }

    /// A request state whose NextN history lives in `pagePool`'s pages (its
    /// new pages counting against `pageAccount`); nil keeps it contiguous.
    public func newRequestState(
        maximumInputs: Int? = nil, pagePool: GLM5NextDSAPagePool?,
        pageAccount: GLM5NextDSAPageBook.Account?
    ) throws -> GLM5NextMTPState {
        try validateTarget()
        let maximum = maximumInputs ?? configuration.maxPositionEmbeddings
        guard maximum > 0 && maximum <= configuration.maxPositionEmbeddings else {
            throw GLM5NextMTPError.invalid("MTP request context bound")
        }
        if let pagePool {
            let types = try historyTypes()
            guard pagePool.latentWidth == configuration.kvLoraRank,
                pagePool.keyWidth == configuration.indexHeadDim,
                pagePool.latentType == types.latents, pagePool.keyType == types.keys,
                pagePool.gateType == types.gates
            else {
                throw GLM5NextMTPError.invalid("NextN history types differ from the page pool's")
            }
        }
        return GLM5NextMTPState(
            binding: binding, maximumInputs: maximum, pagePool: pagePool, pageAccount: pageAccount)
    }

    /// Element types of the NextN history rows (`prepareHistory`): a lazy
    /// graph over a zero target hidden row, never evaluated.
    public func historyTypes() throws -> (latents: DType, keys: DType, gates: DType) {
        let hidden = MLXArray.zeros(
            [1, 1, configuration.hiddenSize], dtype: embedding(MLXArray([Int32(0)], [1, 1])).dtype)
        let projected = try projectInput(
            shiftedEmbeddings: embedding(MLXArray([Int32(0)], [1, 1])), targetHidden: hidden,
            firstPosition: 1)
        let normalized = inputNorm.apply(projected).normalized
        let indexer = attention.indexer
        return (
            attention.kvNorm(attention.kvA(normalized)).dtype,
            indexer.keyNorm(indexer.key(normalized)).dtype,
            matmul(normalized, indexer.poolGate.T).dtype
        )
    }

    /// Exact MTP math on the supplied real admitted paged owner. This does not
    /// publish its transaction. The current owner performs synchronous page work;
    /// do not call this from CBv2's graph-only drafter methods until that seam is asynchronous.
    public func stagePaged(
        shiftedTokens: MLXArray, targetHidden: MLXArray,
        state: GLM5NextPagedSparseState,
        transaction: GLM5NextPagedSparseState.Transaction,
        binding: GLM5NextSparseStateBinding
    ) throws -> GLM5NextMTPPagedOutput {
        guard shiftedTokens.ndim == 2, shiftedTokens.dim(0) == 1, shiftedTokens.dim(1) > 0,
            [.int32, .uint32, .int64].contains(shiftedTokens.dtype),
            binding.modelIdentity == targetIdentity,
            binding.modelLayerIndex == configuration.hiddenLayers
        else { throw GLM5NextMTPError.invalid("paged MTP input or loaded-layer binding") }
        let projected = try projectInput(
            shiftedEmbeddings: embedding(shiftedTokens), targetHidden: targetHidden,
            firstPosition: state.committedTokenCount + state.stagedTokenCount)
        let valid = MLXArray.ones(shiftedTokens.shape, dtype: .bool)
        let staged = try attention.stageForward(
            inputNorm.apply(projected).normalized, state: state, transaction: transaction,
            binding: binding, validity: valid)
        return GLM5NextMTPPagedOutput(
            output: try readout(projected: projected, attended: staged.values, validity: valid),
            attention: staged)
    }

    /// `targetHidden` is already post-final-RMS(meanHC). At physical MTP position p,
    /// pair it with the embedding of token p+1. Only position 0's embedding is zeroed.
    public func projectInput(
        shiftedEmbeddings: MLXArray, targetHidden: MLXArray, firstPosition: Int
    ) throws -> MLXArray {
        try validateTarget()
        guard shiftedEmbeddings.ndim == 3, shiftedEmbeddings.dim(0) == 1,
            shiftedEmbeddings.dim(1) > 0, shiftedEmbeddings.dim(2) == configuration.hiddenSize,
            shiftedEmbeddings.dtype.isFloatingPoint, targetHidden.shape == shiftedEmbeddings.shape,
            targetHidden.dtype == shiftedEmbeddings.dtype,
            firstPosition >= 0 && firstPosition <= configuration.maxPositionEmbeddings,
            shiftedEmbeddings.dim(1) <= configuration.maxPositionEmbeddings - firstPosition
        else { throw GLM5NextMTPError.invalid("aligned MTP hidden/embedding/position geometry") }
        let positions = MLXArray(firstPosition ..< firstPosition + shiftedEmbeddings.dim(1))
        let masked = MLX.where(
            (positions .== 0).reshaped([1, -1, 1]), 0, shiftedEmbeddings)
        let input = concatenated(
            [
                embeddingNorm.apply(masked).normalized,
                hiddenNorm.apply(targetHidden).normalized,
            ], axis: -1)
        return projection(input)
    }

    /// Teacher-force ALL supplied shifted pairs, not only the last prompt token.
    /// Advances a private/reference graph cache, not evaluated active serving state.
    @discardableResult
    public func prime(
        shiftedTokens: MLXArray, targetHidden: MLXArray, at position: Int,
        state: GLM5NextMTPState, validity: MLXArray? = nil
    ) throws -> GLM5NextMTPOutput {
        try validate(state)
        guard state.round == nil else { throw GLM5NextMTPError.activeRound }
        guard position == state.cache.offset else {
            throw GLM5NextMTPError.invalid("noncontiguous MTP priming")
        }
        return try append(
            shiftedTokens: shiftedTokens, hidden: targetHidden, state: state, validity: validity
        ).output
    }

    /// `prime` without its query: appends only the NextN history of these
    /// shifted pairs (`prepareHistory`), which the attention output, MoE and
    /// readout cannot change. For prompt rows whose proposal is not needed.
    /// Graph-only, like `prime`; the recycled hidden is cleared.
    public func primeHistory(
        shiftedTokens: MLXArray, targetHidden: MLXArray, at position: Int,
        state: GLM5NextMTPState
    ) throws {
        try validate(state)
        guard state.round == nil else { throw GLM5NextMTPError.activeRound }
        guard position == state.cache.offset else {
            throw GLM5NextMTPError.invalid("noncontiguous MTP priming")
        }
        guard shiftedTokens.ndim == 2, shiftedTokens.dim(0) == 1, shiftedTokens.dim(1) > 0,
            [.int32, .uint32, .int64].contains(shiftedTokens.dtype),
            shiftedTokens.dim(1) <= state.maximumInputs - state.cache.offset
        else { throw GLM5NextMTPError.invalid("MTP tokens or request context limit") }
        state.cache = try prepareHistory(
            shiftedTokens: shiftedTokens, hidden: targetHidden, cache: state.cache)
        state.hidden = nil
    }

    /// `primeHistory` for rows whose next input is not a token's embedding
    /// row: the target's own input embeddings at those positions (an image's
    /// or video's feature rows spliced over its marker tokens), as a
    /// multimodal target's drafter is fed. The same history arithmetic as the
    /// token form, which computes `embedding(shiftedTokens)` for its input.
    public func primeHistory(
        shiftedEmbeddings: MLXArray, targetHidden: MLXArray, at position: Int,
        state: GLM5NextMTPState
    ) throws {
        try validate(state)
        guard state.round == nil else { throw GLM5NextMTPError.activeRound }
        guard position == state.cache.offset else {
            throw GLM5NextMTPError.invalid("noncontiguous MTP priming")
        }
        guard shiftedEmbeddings.ndim == 3, shiftedEmbeddings.dim(0) == 1,
            shiftedEmbeddings.dim(1) > 0,
            shiftedEmbeddings.dim(1) <= state.maximumInputs - state.cache.offset
        else { throw GLM5NextMTPError.invalid("MTP embeddings or request context limit") }
        state.cache = try prepareHistory(
            shiftedEmbeddings: shiftedEmbeddings, hidden: targetHidden, cache: state.cache)
        state.hidden = nil
    }

    /// The deepest `primeAndPropose`: step 0 plus index-shared continuations,
    /// the native rectangular profile's envelope.
    public static let maximumProposalDepth = GLM5NextMTPVerification.maximumRectangularDepth

    /// Picks one proposal from a proposal step's readout logits `[1, 1, V]`;
    /// `step` is the proposal's place in the round (`0 ..< depth`). Returns
    /// one token id (one element, any integer dtype). Graph-only.
    public typealias ProposalSelection = (_ logits: MLXArray, _ step: Int) -> MLXArray

    /// One speculative round's proposal: teacher-forces the verified shifted
    /// pairs exactly as `prime` (the state advances by them and never by a
    /// proposal), then proposes `depth` tokens: the greedy token of the last
    /// primed row, then `depth - 1` recursive steps that reuse that row's
    /// sparse selection and append nothing (`index_share_for_mtp_iteration`),
    /// each fed the previous proposal and its own output hidden. Returns the
    /// lazy `[1, depth]` int32 proposals. Graph-only, like `prime`.
    ///
    /// `select` replaces the greedy token at every step (nil keeps it, bit
    /// for bit). A caller that verifies against keyed target samples can draw
    /// each proposal with the target's own sampler and the noise of the output
    /// index it proposes for (coupled drafting): the target keeps more of
    /// them, and since a verifier commits only its own samples, a committed
    /// token depends on the selection only through where rounds end. The
    /// state never does: only verified pairs are ever primed.
    public func primeAndPropose(
        shiftedTokens: MLXArray, targetHidden: MLXArray, at position: Int,
        state: GLM5NextMTPState, depth: Int, select: ProposalSelection? = nil
    ) throws -> MLXArray {
        try validate(state)
        guard state.round == nil else { throw GLM5NextMTPError.activeRound }
        guard position == state.cache.offset else {
            throw GLM5NextMTPError.invalid("noncontiguous MTP priming")
        }
        guard (1 ... Self.maximumProposalDepth).contains(depth),
            depth == 1 || configuration.indexShareForMTPIteration
        else { throw GLM5NextMTPError.unsupportedDraftDepth(depth) }
        let primed = try append(
            shiftedTokens: shiftedTokens, hidden: targetHidden, state: state, validity: nil)
        func choose(_ output: GLM5NextMTPOutput, step: Int) -> MLXArray {
            guard let select else {
                return argMax(output.logits[0..., -1, 0...], axis: -1).asType(.int32).reshaped(1, 1)
            }
            return select(output.logits[0..., (-1)..., 0...], step).asType(.int32).reshaped(1, 1)
        }
        var token = choose(primed.output, step: 0)
        var hidden = primed.output.hidden[0..., (-1)..., 0...]
        var proposals = [token]
        for step in 1 ..< depth {
            let output = try prepareContinuation(
                shiftedToken: token, draftHidden: hidden, step: step, history: primed.history,
                selection: primed.selection)
            token = choose(output, step: step)
            hidden = output.hidden
            proposals.append(token)
        }
        return proposals.count == 1 ? proposals[0] : concatenated(proposals, axis: 1)
    }

    /// One request of `primeAndProposeLanes`: `primeAndPropose`'s arguments.
    public struct ProposalLane {
        public let shiftedTokens: MLXArray
        public let targetHidden: MLXArray
        public let position: Int
        public let state: GLM5NextMTPState
        public let depth: Int
        public let select: ProposalSelection?

        public init(
            shiftedTokens: MLXArray, targetHidden: MLXArray, position: Int, state: GLM5NextMTPState,
            depth: Int, select: ProposalSelection? = nil
        ) {
            self.shiftedTokens = shiftedTokens
            self.targetHidden = targetHidden
            self.position = position
            self.state = state
            self.depth = depth
            self.select = select
        }
    }

    /// `primeAndPropose` for several requests, step by step: each step of
    /// every lane runs over that lane's own state and history, then ONE launch
    /// applies the head to every lane's readout row (`GLM5NextMultiVectorQMV`:
    /// each row's logits are bitwise its own one-row head's), then each lane
    /// chooses its proposal. Each lane's proposals and state are therefore
    /// bitwise those of its own `primeAndPropose`. A lane's own head row stays
    /// in the graph unevaluated; a head the multi-vector launch cannot read
    /// keeps every lane's own row. Graph-only.
    public func primeAndProposeLanes(_ lanes: [ProposalLane]) throws -> [MLXArray] {
        for lane in lanes {
            try validate(lane.state)
            guard lane.state.round == nil else { throw GLM5NextMTPError.activeRound }
            guard lane.position == lane.state.cache.offset else {
                throw GLM5NextMTPError.invalid("noncontiguous MTP priming")
            }
            guard (1 ... Self.maximumProposalDepth).contains(lane.depth),
                lane.depth == 1 || configuration.indexShareForMTPIteration
            else { throw GLM5NextMTPError.unsupportedDraftDepth(lane.depth) }
        }
        guard Set(lanes.map { ObjectIdentifier($0.state) }).count == lanes.count else {
            throw GLM5NextMTPError.invalid("MTP lanes need distinct states")
        }
        // Each listed lane's logits [1, 1, V] for its output's last row.
        func logits(_ outputs: [GLM5NextMTPOutput]) -> [MLXArray] {
            var result: [MLXArray] = []
            var start = 0
            while start < outputs.count {
                let chunk = Array(outputs[start ..< min(outputs.count, start + Self.headLaneChunk)])
                let rows = concatenated(
                    chunk.map { $0.hidden[0..., (-1)..., 0...].reshaped(1, -1) }, axis: 0)
                if chunk.count > 1, let batched = GLM5NextMultiVectorQMV.apply(head, rows) {
                    result += (0 ..< chunk.count).map { batched[$0 ..< $0 + 1].reshaped(1, 1, -1) }
                } else {
                    result += chunk.map(\.logits)
                }
                start += chunk.count
            }
            return result
        }
        func choose(_ logits: MLXArray, _ lane: ProposalLane, step: Int) -> MLXArray {
            guard let select = lane.select else {
                return argMax(logits[0..., -1, 0...], axis: -1).asType(.int32).reshaped(1, 1)
            }
            return select(logits, step).asType(.int32).reshaped(1, 1)
        }
        let primed = try lanes.map {
            try append(
                shiftedTokens: $0.shiftedTokens, hidden: $0.targetHidden, state: $0.state,
                validity: nil)
        }
        var tokens = zip(logits(primed.map(\.output)), lanes).map { choose($0, $1, step: 0) }
        var hidden = primed.map { $0.output.hidden[0..., (-1)..., 0...] }
        var proposals = tokens.map { [$0] }
        for step in 1 ..< (lanes.map(\.depth).max() ?? 1) {
            let active = lanes.indices.filter { lanes[$0].depth > step }
            let outputs = try active.map { i in
                try prepareContinuation(
                    shiftedToken: tokens[i], draftHidden: hidden[i], step: step,
                    history: primed[i].history,
                    selection: primed[i].selection)
            }
            for (i, (output, rowLogits)) in zip(active, zip(outputs, logits(outputs))) {
                tokens[i] = choose(rowLogits, lanes[i], step: step)
                hidden[i] = output.hidden
                proposals[i].append(tokens[i])
            }
        }
        return proposals.map { $0.count == 1 ? $0[0] : concatenated($0, axis: 1) }
    }

    /// Lanes one multi-vector head launch takes: past four vectors the launch
    /// is no cheaper than separate rows (it is ALU-bound), so larger groups
    /// split into launches of four.
    static let headLaneChunk = 4

    /// Full shifted prompt includes the confirmed carry token following its last
    /// target hidden row: [prompt[1], ..., prompt.last, carry].
    @discardableResult
    public func primePrompt(
        tokens: MLXArray, targetHidden: MLXArray, carryToken: MLXArray,
        state: GLM5NextMTPState
    ) throws -> GLM5NextMTPOutput {
        try validate(state)
        guard state.cache.offset == 0, tokens.ndim == 2, tokens.dim(0) == 1,
            tokens.dim(1) > 0, carryToken.shape == [1, 1], tokens.dtype == carryToken.dtype
        else { throw GLM5NextMTPError.invalid("MTP prompt/carry alignment") }
        let shifted = concatenated([tokens[0..., 1...], carryToken], axis: 1)
        return try prime(shiftedTokens: shifted, targetHidden: targetHidden, at: 0, state: state)
    }

    public func beginRound(state: GLM5NextMTPState, candidates: Int = 1) throws {
        try validate(state)
        // The current DSA seam cannot carry/reuse step-0 top-k selections.
        // Never silently recompute them for index_share_for_mtp_iteration at depth > 1.
        guard candidates == maximumDraftTokens else {
            throw GLM5NextMTPError.unsupportedDraftDepth(candidates)
        }
        guard state.round == nil else { throw GLM5NextMTPError.activeRound }
        state.round = snapshot(state)
    }

    public func draftStep(
        shiftedToken: MLXArray, targetHidden: MLXArray, state: GLM5NextMTPState
    ) throws -> GLM5NextMTPOutput {
        try validate(state)
        guard state.round != nil else { throw GLM5NextMTPError.noActiveRound }
        guard state.stagedInputCount == 0 else { throw GLM5NextMTPError.unsupportedDraftDepth(2) }
        guard shiftedToken.shape == [1, 1] else {
            throw GLM5NextMTPError.invalid("one draft input required")
        }
        return try append(
            shiftedTokens: shiftedToken, hidden: targetHidden, state: state, validity: nil
        ).output
    }

    /// Acceptance counts INPUTS consumed by this round, never its unconsumed output token.
    public func commitRound(state: GLM5NextMTPState, acceptingInputs: Int) throws {
        try validate(state)
        guard let round = state.round else { throw GLM5NextMTPError.noActiveRound }
        guard acceptingInputs >= 0 && acceptingInputs <= state.stagedInputCount else {
            throw GLM5NextMTPError.invalid("MTP acceptance outside staged prefix")
        }
        if acceptingInputs == 0 { adopt(round, into: state) }
        state.round = nil
    }

    public func rollbackRound(state: GLM5NextMTPState) throws {
        try validate(state)
        guard let round = state.round else { throw GLM5NextMTPError.noActiveRound }
        adopt(round, into: state)
        state.round = nil
    }

    public func capture(state: GLM5NextMTPState) throws -> GLM5NextMTPCheckpoint {
        try validate(state)
        guard state.round == nil else { throw GLM5NextMTPError.activeRound }
        return snapshot(state)
    }

    public func restore(_ checkpoint: GLM5NextMTPCheckpoint, into state: GLM5NextMTPState) throws {
        try validate(state)
        guard state.round == nil else { throw GLM5NextMTPError.activeRound }
        guard checkpoint.binding === binding, checkpoint.inputCount <= state.maximumInputs else {
            throw GLM5NextMTPError.incompatibleState
        }
        adopt(checkpoint, into: state)
    }

    public func trim(state: GLM5NextMTPState, to inputCount: Int) throws {
        try validate(state)
        guard state.round == nil else { throw GLM5NextMTPError.activeRound }
        let working = state.cache.copy() as! GLM5NextDSACache
        try working.truncate(to: inputCount)
        if inputCount != state.cache.offset { state.hidden = nil }
        state.cache = working
    }

    public func reset(state: GLM5NextMTPState) throws {
        try validate(state)
        state.cache = state.emptyHistory()
        state.hidden = nil
        state.round = nil
    }

    public func release(state: GLM5NextMTPState) throws {
        // Cleanup must remain possible after the target has been unloaded.
        guard state.binding === binding else { throw GLM5NextMTPError.incompatibleState }
        state.cache = state.emptyHistory()
        state.hidden = nil
        state.round = nil
        state.released = true
    }

    /// Keep these lazy roots alive through the caller's MLX fault/evaluation fence.
    /// (A contiguous history's roots are its arrays; a paged one's, its last
    /// page launch's fence: `GLM5NextDSACache.innerState`.)
    public func evaluationTargets(state: GLM5NextMTPState) throws -> [MLXArray] {
        try validate(state)
        return state.cache.innerState() + [state.hidden].compactMap { $0 }
    }

    private func validateTarget() throws {
        guard targetReference.matches(embedding: embedding, head: head) else {
            throw GLM5NextMTPError.invalid("MTP target released or shared components replaced")
        }
    }

    func validateExecutionTarget() throws { try validateTarget() }

    private func validate(_ state: GLM5NextMTPState) throws {
        try validateTarget()
        guard state.binding === binding else { throw GLM5NextMTPError.incompatibleState }
        guard !state.released else { throw GLM5NextMTPError.releasedState }
    }

    private func snapshot(_ state: GLM5NextMTPState) -> GLM5NextMTPCheckpoint {
        GLM5NextMTPCheckpoint(binding: binding, cache: state.cache, hidden: state.hidden)
    }

    private func adopt(_ checkpoint: GLM5NextMTPCheckpoint, into state: GLM5NextMTPState) {
        state.cache = checkpoint.cache.copy() as! GLM5NextDSACache
        // A restored prefix's later pages are this request's.
        if state.cache.pagePool != nil { state.cache.pageAccount = state.pageAccount }
        state.hidden = checkpoint.hidden.map { $0.sharingArray() }
    }

    /// Also returns the appended history and its final query's selection, for
    /// index-shared continuations (`primeAndPropose`).
    private func append(
        shiftedTokens: MLXArray, hidden: MLXArray, state: GLM5NextMTPState, validity: MLXArray?
    ) throws -> (
        output: GLM5NextMTPOutput, history: GLM5NextDSACache,
        selection: GLM5NextDSAAttention.SharedSelection
    ) {
        guard shiftedTokens.ndim == 2, shiftedTokens.dim(0) == 1, shiftedTokens.dim(1) > 0,
            [.int32, .uint32, .int64].contains(shiftedTokens.dtype),
            shiftedTokens.dim(1) <= state.maximumInputs - state.cache.offset
        else { throw GLM5NextMTPError.invalid("MTP tokens or request context limit") }
        let valid = validity ?? MLXArray.ones(shiftedTokens.shape, dtype: .bool)
        guard valid.shape == shiftedTokens.shape && valid.dtype == .bool else {
            throw GLM5NextMTPError.invalid("MTP validity geometry")
        }
        let result = try prepareExecution(
            shiftedTokens: shiftedTokens, hidden: hidden,
            cache: state.cache, validity: valid)
        let output = result.output
        let last = output.hidden[0..., (-1)..., 0...]
        state.cache = result.cache
        state.hidden = MLX.where(
            MLXArray.ones(last.shape, dtype: .bool), last,
            MLXArray.zeros(last.shape, dtype: last.dtype))
        return (output, result.cache, result.selection)
    }

    /// Private-cache construction shared with the admitted contiguous owner.
    /// It never changes the source cache or evaluates any array.
    func prepareExecution(
        shiftedTokens: MLXArray, hidden: MLXArray,
        cache: GLM5NextDSACache, validity: MLXArray? = nil
    ) throws
        -> (
            output: GLM5NextMTPOutput, cache: GLM5NextDSACache,
            selection: GLM5NextDSAAttention.SharedSelection
        )
    {
        let valid = validity ?? MLXArray.ones(shiftedTokens.shape, dtype: .bool)
        let projected = try projectInput(
            shiftedEmbeddings: embedding(shiftedTokens), targetHidden: hidden,
            firstPosition: cache.offset)
        let working = cache.copy() as! GLM5NextDSACache
        let attended = try attention.forwardRetainingSelection(
            inputNorm.apply(projected).normalized, cache: working, validity: valid)
        let output = try readout(projected: projected, attended: attended.values, validity: valid)
        return (output, working, attended.selection)
    }

    /// Recursive proposal step `step` (>= 1) of the single trained layer, after
    /// `history` was extended by step 0 (SGLang/vLLM index_share_for_mtp_iteration).
    /// The token is the previous proposal; `draftHidden` is that step's own
    /// post-shared-head output. Step 0's final-query selection is reused, so the
    /// indexer is skipped and nothing is appended: a later round re-primes these
    /// positions from verified target hidden rows instead.
    func prepareContinuation(
        shiftedToken: MLXArray, draftHidden: MLXArray, step: Int,
        history: GLM5NextDSACache, selection: GLM5NextDSAAttention.SharedSelection
    ) throws -> GLM5NextMTPOutput {
        guard configuration.indexShareForMTPIteration, step >= 1,
            shiftedToken.shape == [1, 1], [.int32, .uint32, .int64].contains(shiftedToken.dtype),
            history.offset > 0
        else { throw GLM5NextMTPError.unsupportedDraftDepth(step + 1) }
        let valid = MLXArray.ones(shiftedToken.shape, dtype: .bool)
        let projected = try projectInput(
            shiftedEmbeddings: embedding(shiftedToken), targetHidden: draftHidden,
            firstPosition: history.offset - 1 + step)
        let normalized = inputNorm.apply(projected).normalized
        let attended: MLXArray
        if history.pagePool != nil {
            // Only the rows the shared selection reads leave the pages.
            attended = try attention.attendSharedSelection(
                normalized, pagedHistory: history, selection: selection)
        } else {
            guard let latents = history.latents, let keyValidity = history.validity else {
                throw GLM5NextMTPError.unsupportedDraftDepth(step + 1)
            }
            attended = try attention.attendSharedSelection(
                normalized, latents: latents, keyValidity: keyValidity, selection: selection)
        }
        return try readout(projected: projected, attended: attended, validity: valid)
    }

    /// Single-layer NextN history depends only on the projected, normalized
    /// input. Attention output, MoE and readout cannot change these stored rows.
    /// Use the same trained projection modules as forward, without dead queries.
    func prepareHistory(
        shiftedTokens: MLXArray, hidden: MLXArray, cache: GLM5NextDSACache
    ) throws -> GLM5NextDSACache {
        let projected = try projectInput(
            shiftedEmbeddings: embedding(shiftedTokens), targetHidden: hidden,
            firstPosition: cache.offset)
        let normalized = inputNorm.apply(projected).normalized
        let indexer = attention.indexer
        let keys = indexer.keyNorm(indexer.key(normalized))
        let gates = matmul(normalized, indexer.poolGate.T)
        let latents = attention.kvNorm(attention.kvA(normalized))
        let working = cache.copy() as! GLM5NextDSACache
        try working.append(
            latents: latents, keys: keys, gates: gates,
            validity: MLXArray.ones(shiftedTokens.shape, dtype: .bool))
        return working
    }

    /// `prepareHistory` fed input embeddings instead of tokens.
    func prepareHistory(
        shiftedEmbeddings: MLXArray, hidden: MLXArray, cache: GLM5NextDSACache
    ) throws -> GLM5NextDSACache {
        let projected = try projectInput(
            shiftedEmbeddings: shiftedEmbeddings, targetHidden: hidden, firstPosition: cache.offset)
        let normalized = inputNorm.apply(projected).normalized
        let indexer = attention.indexer
        let keys = indexer.keyNorm(indexer.key(normalized))
        let gates = matmul(normalized, indexer.poolGate.T)
        let latents = attention.kvNorm(attention.kvA(normalized))
        let working = cache.copy() as! GLM5NextDSACache
        try working.append(
            latents: latents, keys: keys, gates: gates,
            validity: MLXArray.ones(Array(shiftedEmbeddings.shape.prefix(2)), dtype: .bool))
        return working
    }

    private func readout(projected: MLXArray, attended: MLXArray, validity: MLXArray) throws
        -> GLM5NextMTPOutput
    {
        let post = postNorm.apply(attended, residual: projected)
        let result = sharedHead.norm.apply(
            try mlp(post.normalized, routedExperts: executionPolicy.routedExperts),
            residual: post.residual
        ).normalized
        let final = MLX.where(validity.expandedDimensions(axis: -1), result, 0)
        // Priming needs every DSA row, not a prompt-length vocabulary allocation.
        return GLM5NextMTPOutput(hidden: final, logits: head(final[0..., (-1)..., 0...]))
    }
}
