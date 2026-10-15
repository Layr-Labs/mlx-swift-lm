// Copyright 2026 the HuggingFace Team. All rights reserved.
// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0
// Source: Transformers f0705f991a71f05dad9569336b181119f5546f93, Glm5NextTextModel/DecoderLayer.
import MLX
import MLXLMCommon
import MLXNN

public struct GLM5NextTextOutput {
    public let logits: MLXArray
    /// Unweighted stream mean, then final RMSNorm. This is the NextN boundary.
    public let hidden: MLXArray
}

/// Engine-thread-confined candidate. Graph construction never changes the
/// destination. The required caller fence must evaluate every supplied root,
/// join device work, and throw on MLX faults before this owner can publish.
public final class GLM5NextTextCandidate {
    public let output: GLM5NextTextOutput
    private let destination: GLM5NextTextCache
    private let original: GLM5NextTextCache.Revision
    private var working: GLM5NextTextCache?
    private enum State: Equatable { case pending, evaluating, committed, discarded }
    private var state = State.pending

    fileprivate init(
        output: GLM5NextTextOutput, destination: GLM5NextTextCache,
        original: GLM5NextTextCache.Revision, working: GLM5NextTextCache
    ) {
        self.output = output
        self.destination = destination
        self.original = original
        self.working = working
    }

    @discardableResult
    public func commit(afterEvaluating fence: ([MLXArray]) throws -> Void) throws
        -> GLM5NextTextOutput
    {
        guard state == .pending, let working else {
            throw GLM5NextLoadError.invalid("candidate is no longer pending")
        }
        guard destination.revision() == original else {
            discard()
            throw GLM5NextLoadError.invalid("candidate destination changed before evaluation")
        }
        state = .evaluating
        do {
            try fence([output.logits, output.hidden] + working.layers.flatMap { $0.state })
            guard destination.revision() == original else {
                throw GLM5NextLoadError.invalid("candidate destination changed during evaluation")
            }
            destination.adopt(working)
            self.working = nil
            state = .committed
            return output
        } catch {
            self.working = nil
            state = .discarded
            throw error
        }
    }

    public func discard() {
        guard state == .pending else { return }
        working = nil
        state = .discarded
    }
}

public final class GLM5NextTextLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Module
    public var sparseAttention: GLM5NextDSAAttention? { attention as? GLM5NextDSAAttention }
    @ModuleInfo(key: "input_layernorm") var inputNorm: GLM5NextRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: GLM5NextRMSNorm
    let mlp: Module
    @ParameterInfo(key: "hc_attn_fn") var attentionFunction: MLXArray
    @ParameterInfo(key: "hc_attn_base") var attentionBase: MLXArray
    @ParameterInfo(key: "hc_attn_scale") var attentionScale: MLXArray
    @ParameterInfo(key: "hc_ffn_fn") var ffnFunction: MLXArray
    @ParameterInfo(key: "hc_ffn_base") var ffnBase: MLXArray
    @ParameterInfo(key: "hc_ffn_scale") var ffnScale: MLXArray
    let configuration: GLM5NextTextConfiguration
    init(
        _ c: GLM5NextTextConfiguration, index: Int, loader: GLM5NextTensorLoader,
        materialize: ((MLXArray) throws -> Void)?
    ) throws {
        configuration = c
        let p = "model.language_model.layers.\(index)"
        if c.layerTypes[index] == .linearAttention {
            _attention.wrappedValue = try GLM5NextKDASublayer(
                c, prefix: p + ".self_attn", loader: loader)
        } else {
            _attention.wrappedValue = try GLM5NextDSAAttention(
                c, prefix: p + ".self_attn", loader: loader)
        }
        _inputNorm.wrappedValue = try loader.norm(p + ".input_layernorm", epsilon: c.rmsNormEps)
        _postNorm.wrappedValue = try loader.norm(
            p + ".post_attention_layernorm", epsilon: c.rmsNormEps)
        mlp =
            c.mlpLayerTypes[index] == .dense
            ? try loader.mlp(c, prefix: p + ".mlp", intermediate: c.intermediateSize)
            : try loader.moe(c, prefix: p + ".mlp", materialize: materialize)
        _attentionFunction.wrappedValue = try loader.array(p + ".hc_attn_fn")
        _attentionBase.wrappedValue = try loader.array(p + ".hc_attn_base")
        _attentionScale.wrappedValue = try loader.array(p + ".hc_attn_scale")
        _ffnFunction.wrappedValue = try loader.array(p + ".hc_ffn_fn")
        _ffnBase.wrappedValue = try loader.array(p + ".hc_ffn_base")
        _ffnScale.wrappedValue = try loader.array(p + ".hc_ffn_scale")
    }
    func forward(
        _ streams: MLXArray, cache: any KVCache, validity: MLXArray,
        recurrenceExecution: GLM5NextLinearAttention.ExecutionPolicy = .reference
    ) throws -> MLXArray {
        try forward(streams) { normalized in
            if let attention = attention as? GLM5NextKDASublayer,
                let cache = cache as? GLM5NextKDACache
            {
                return try attention.forward(
                    normalized, cache: cache, validity: validity,
                    recurrenceExecution: recurrenceExecution)
            }
            if let attention = attention as? GLM5NextDSAAttention,
                let cache = cache as? GLM5NextDSACache
            {
                return try attention.forward(normalized, cache: cache, validity: validity)
            }
            throw GLM5NextLoadError.invalid("attention/cache kind mismatch")
        }
    }

    /// Shared mHC/MLP path; the caller supplies its request-owned attention state.
    func forward(_ streams: MLXArray, attend: (MLXArray) throws -> MLXArray) throws -> MLXArray {
        let c = configuration
        let attn = GLM5NextHyperConnection.collapse(
            streams, fn: attentionFunction, base: attentionBase,
            scale: attentionScale, rmsNormEpsilon: Float(c.rmsNormEps), epsilon: Float(c.hcEps),
            sinkhornIterations: c.hcSinkhornIters)
        let normalized = inputNorm(attn.collapsed)
        let attentionOutput = try attend(normalized)
        let residual = GLM5NextHyperConnection.expand(
            attentionOutput, residual: streams, post: attn.post, comb: attn.comb)
        let ffn = GLM5NextHyperConnection.collapse(
            residual, fn: ffnFunction, base: ffnBase,
            scale: ffnScale, rmsNormEpsilon: Float(c.rmsNormEps), epsilon: Float(c.hcEps),
            sinkhornIterations: c.hcSinkhornIters)
        let input = postNorm(ffn.collapsed)
        let output: MLXArray
        if let dense = mlp as? GLM5NextMLP {
            output = dense(input)
        } else if let moe = mlp as? GLM5NextMoE {
            output = moe(input)
        } else {
            throw GLM5NextLoadError.invalid("unsupported GLM feed-forward module")
        }
        return GLM5NextHyperConnection.expand(
            output, residual: residual, post: ffn.post, comb: ffn.comb)
    }
}

public final class GLM5NextTextBackbone: Module {
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    public let layers: [GLM5NextTextLayer]
    public let norm: GLM5NextRMSNorm
    let configuration: GLM5NextTextConfiguration
    init(
        _ c: GLM5NextTextConfiguration, loader: GLM5NextTensorLoader,
        materialize: ((MLXArray) throws -> Void)?
    ) throws {
        configuration = c
        _embedTokens.wrappedValue = try loader.embedding("model.language_model.embed_tokens")
        layers = try (0 ..< c.hiddenLayers).map {
            try GLM5NextTextLayer(c, index: $0, loader: loader, materialize: materialize)
        }
        norm = try loader.norm("model.language_model.norm", epsilon: c.rmsNormEps)
    }
    func forward(
        _ embeddings: MLXArray, cache: GLM5NextTextCache, validity: MLXArray,
        recurrenceExecution: GLM5NextLinearAttention.ExecutionPolicy = .reference
    ) throws
        -> MLXArray
    {
        var streams = broadcast(
            embeddings.expandedDimensions(axis: 2),
            to: [
                embeddings.dim(0), embeddings.dim(1), configuration.hcMult,
                configuration.hiddenSize,
            ])
        for (layer, state) in zip(layers, cache.layers) {
            streams = try layer.forward(
                streams, cache: state, validity: validity, recurrenceExecution: recurrenceExecution)
        }
        return norm(streams.mean(axis: 2))
    }
}

/// Architectural target with a separate conservative native CBv2 adapter.
/// Not registered; paged/MTP/prefix capabilities require their own integration.
/// The validated inventory keeps MTP and vision separate; this owner loads only
/// the text partition. The caller must retain and load the remaining partitions.
public final class GLM5NextTextModel: Module, LLMModel {
    public let configuration: GLM5NextTextConfiguration
    public let inventory: GLM5NextWeightInventory
    public let model: GLM5NextTextBackbone
    @ModuleInfo(key: "lm_head") public var lmHead: Linear
    public var vocabularySize: Int { configuration.vocabularySize }
    public var loraLayers: [Module] { model.layers }

    /// Resolver must be bounded by shard/projection and retain no duplicate whole
    /// model payload. For Q4, materializeExpertStack MUST evaluate/check each
    /// packed stack before returning so its unstacked inputs can be released.
    /// This callback belongs under the caller's MLX error/fault boundary.
    /// releaseSourceTensors must evict the named lookup aliases (and unused shard
    /// dictionary aliases), without invalidating independently owning MLXArrays.
    public init(
        configuration: GLM5NextTextConfiguration, inventory: GLM5NextWeightInventory,
        tensor: @escaping (String) throws -> MLXArray,
        materializeExpertStack: ((MLXArray) throws -> Void)? = nil,
        releaseSourceTensors: (([String]) -> Void)? = nil
    ) throws {
        guard configuration == inventory.configuration.textConfig else {
            throw GLM5NextLoadError.invalid("target configuration/inventory mismatch")
        }
        if inventory.storage == .vontraQ4
            && (materializeExpertStack == nil || releaseSourceTensors == nil)
        {
            throw GLM5NextLoadError.invalid(
                "Q4 target construction requires bounded stack materialization and source-cache eviction"
            )
        }
        self.configuration = configuration
        self.inventory = inventory
        let loader = GLM5NextTensorLoader(
            inventory, tensor, releaseSourceTensors: releaseSourceTensors)
        model = try GLM5NextTextBackbone(
            configuration, loader: loader, materialize: materializeExpertStack)
        _lmHead.wrappedValue = try loader.linear("lm_head")
        try loader.finishText()
    }
    public func embed(_ tokens: MLXArray) -> MLXArray { model.embedTokens(tokens) }
    public func newTextCache() -> GLM5NextTextCache { .init(configuration: configuration) }
    public func newCache(parameters: GenerateParameters?) -> [any KVCache] { newTextCache().layers }
    public func prepareForward(
        tokens: MLXArray, cache: GLM5NextTextCache, validity: MLXArray? = nil
    )
        throws -> GLM5NextTextCandidate
    {
        try prepareForward(
            tokens: tokens, cache: cache, validity: validity, recurrenceExecution: .reference)
    }
    /// Internal qualification seam. Public/LLM callers retain reference execution.
    func prepareForward(
        tokens: MLXArray, cache: GLM5NextTextCache, validity: MLXArray? = nil,
        recurrenceExecution: GLM5NextLinearAttention.ExecutionPolicy
    ) throws -> GLM5NextTextCandidate {
        guard tokens.ndim == 2, [.int32, .uint32, .int64].contains(tokens.dtype) else {
            throw GLM5NextLoadError.invalid("tokens must be integer [B,T]")
        }
        return try prepareForward(
            embeddings: embed(tokens), cache: cache, validity: validity,
            recurrenceExecution: recurrenceExecution)
    }
    public func prepareForward(
        embeddings: MLXArray, cache: GLM5NextTextCache, validity: MLXArray? = nil
    )
        throws -> GLM5NextTextCandidate
    {
        try prepareForward(
            embeddings: embeddings, cache: cache, validity: validity,
            recurrenceExecution: .reference)
    }
    func prepareForward(
        embeddings: MLXArray, cache: GLM5NextTextCache, validity: MLXArray? = nil,
        recurrenceExecution: GLM5NextLinearAttention.ExecutionPolicy
    ) throws -> GLM5NextTextCandidate {
        try cache.validate(configuration)
        let original = cache.revision()
        let working = cache.copy()
        let output = try forward(
            embeddings: embeddings, cache: working, validity: validity,
            recurrenceExecution: recurrenceExecution)
        guard cache.revision() == original else {
            throw GLM5NextLoadError.invalid("cache changed while preparing candidate")
        }
        return GLM5NextTextCandidate(
            output: output, destination: cache, original: original, working: working)
    }
    /// Unfenced graph/cache compatibility API for ordinary LanguageModel use.
    /// CBv2 and speculative callers MUST use prepareForward + candidate.commit,
    /// not pass their active cache here. No device-success claim is made here.
    public func forward(tokens: MLXArray, cache: GLM5NextTextCache?, validity: MLXArray? = nil)
        throws -> GLM5NextTextOutput
    {
        try forward(
            tokens: tokens, cache: cache, validity: validity, recurrenceExecution: .reference)
    }
    func forward(
        tokens: MLXArray, cache: GLM5NextTextCache?, validity: MLXArray? = nil,
        recurrenceExecution: GLM5NextLinearAttention.ExecutionPolicy
    ) throws -> GLM5NextTextOutput {
        guard tokens.ndim == 2, [.int32, .uint32, .int64].contains(tokens.dtype) else {
            throw GLM5NextLoadError.invalid("tokens must be integer [B,T]")
        }
        return try forward(
            embeddings: embed(tokens), cache: cache, validity: validity,
            recurrenceExecution: recurrenceExecution)
    }
    public func forward(embeddings: MLXArray, cache: GLM5NextTextCache?, validity: MLXArray? = nil)
        throws -> GLM5NextTextOutput
    {
        try forward(
            embeddings: embeddings, cache: cache, validity: validity,
            recurrenceExecution: .reference)
    }
    func forward(
        embeddings: MLXArray, cache: GLM5NextTextCache?, validity: MLXArray? = nil,
        recurrenceExecution: GLM5NextLinearAttention.ExecutionPolicy
    ) throws -> GLM5NextTextOutput {
        guard embeddings.ndim == 3, embeddings.dim(0) > 0, embeddings.dim(1) > 0,
            embeddings.dim(2) == configuration.hiddenSize,
            [.float32, .bfloat16, .float16].contains(embeddings.dtype)
        else {
            throw GLM5NextLoadError.invalid("embeddings must be floating [B,T,hidden]")
        }
        let validity = validity ?? MLXArray.ones(Array(embeddings.shape.prefix(2)), dtype: .bool)
        guard validity.shape == Array(embeddings.shape.prefix(2)), validity.dtype == .bool else {
            throw GLM5NextLoadError.invalid("validity must be boolean [B,T]")
        }
        let destination = cache ?? newTextCache()
        try destination.validate(configuration)
        guard embeddings.dim(1) <= configuration.maxPositionEmbeddings - destination.offset else {
            throw GLM5NextLoadError.invalid("context length exceeds configuration")
        }
        let working = destination.copy()
        let hidden = try model.forward(
            embeddings, cache: working, validity: validity, recurrenceExecution: recurrenceExecution
        )
        let maskedHidden = MLX.where(validity.expandedDimensions(axis: -1), hidden, 0)
        let logits = lmHead(maskedHidden)
        // Compatibility-only mutation of a caller's private graph cache. This
        // is NOT evaluated publication; active serving uses prepareForward and
        // the candidate's required evaluate/join/fault-check commit fence.
        destination.adopt(working)
        return GLM5NextTextOutput(logits: logits, hidden: maskedHidden)
    }
    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        do {
            return try forward(tokens: inputs, cache: cache.map { GLM5NextTextCache(layers: $0) })
                .logits
        } catch { preconditionFailure("GLM5Next forward failed: \(error)") }
    }
    public func callAsFunction(_ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        precondition(state == nil, "GLM5Next has no LMOutput cross-attention state")
        do {
            return LMOutput(
                logits: try forward(
                    tokens: input.tokens,
                    cache: cache.map { GLM5NextTextCache(layers: $0) }, validity: input.mask
                ).logits)
        } catch { preconditionFailure("GLM5Next forward failed: \(error)") }
    }
}
