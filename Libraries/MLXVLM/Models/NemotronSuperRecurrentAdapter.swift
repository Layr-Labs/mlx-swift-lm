import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Super-only serial causal adapter. It owns the composite, not a copied text
/// tower. No MTP, positional remapping, packed rows or vision prefix reuse.
/// Reference: Transformers 049d2bf1220747b6d39e2a978b9f5fe0defa1dca
/// NemotronHModel/Attention (v5.8.0); Omni checkpoint
/// 7d4bc747b6bdc2cff64c58d65da0d08dd9caa25c substitutes image embeddings.
public final class NemotronSuperRecurrentAdapter: Module, LanguageModel,
    CBv2RecurrentLanguageModelForwardable, CBv2RecurrentLanguageModelPrefillForwardable,
    CBv2PositionedRecurrentEmbeddingForwardable, CBv2CausalPositionRequirementProviding {
    @ModuleInfo public var composite: NemotronSuperModel
    public enum Failure: Error { case position, batch, recurrentOwnership, caches }

    public init(_ composite: NemotronSuperModel) {
        self._composite.wrappedValue = composite
    }

    /// Pure admission validation, before cache/state binding or graph creation.
    /// Explicit positions might encode packing or reset masks; never drop them.
    public static func validateLayout(batchSize: Int, sequenceLength: Int,
        hasExplicitPositions: Bool, recurrentOwnerCount: Int,
        attentionCacheCount: Int, expectedAttentionCacheCount: Int) throws {
        guard !hasExplicitPositions else { throw Failure.position }
        guard batchSize == 1, sequenceLength > 0 else { throw Failure.batch }
        guard recurrentOwnerCount == 1 else { throw Failure.recurrentOwnership }
        guard expectedAttentionCacheCount > 0,
            attentionCacheCount == expectedAttentionCacheCount else { throw Failure.caches }
    }

    public var cbv2CausalPositionRequirement: CBv2CausalPositionRequirement { .scalarCacheOffset }
    public var cbv2Capabilities: CBv2ModelCapabilities { composite.languageModel.cbv2Capabilities }
    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec { composite.cbv2RecurrentStateSpec }
    public var cbv2LayerKinds: [CBv2LayerKind] { composite.cbv2LayerKinds }
    public var cbv2SupportsPackedPrefill: Bool { false }
    public var supportsVisionSpanPrefill: Bool { false }
    public var supportsCausalVisionPrefill: Bool { true }
    public var calibratedPrefillTokenLimit: Int? { composite.languageModel.calibratedPrefillTokenLimit }

    public func newCacheV2(makeLayerCache: (Int, CBv2LayerKind) throws -> any CBv2AttendingLayerCache)
        rethrows -> [any CBv2AttendingLayerCache] {
        try composite.languageModel.newCacheV2(makeLayerCache: makeLayerCache)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        composite.languageModel.newCache(parameters: parameters)
    }
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        try composite.languageModel.prepare(input, cache: cache, windowSize: windowSize)
    }
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        composite.languageModel(inputs, cache: cache)
    }
    public func scaledInputEmbeddings(_ inputs: MLXArray) -> MLXArray {
        composite.languageModel.inputEmbeddings(inputs)
    }
    public func embeddingForward(_ inputs: MLXArray, inputEmbedding: MLXArray,
        cache: [KVCache]?) -> MLXArray {
        preconditionFailure("Super image prefill requires request-owned recurrent state")
    }
    public func cbv2Forward(_ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]) -> MLXArray {
        validateInvariant(tokens, cache: caches, state: recurrentState, positions: nil)
        return composite.languageModel.cbv2Forward(tokens, caches: caches, recurrentState: recurrentState)
    }
    public func embeddingForward(_ inputs: MLXArray, inputEmbedding: MLXArray,
        cache: [KVCache]?, recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?) -> MLXArray {
        validateInvariant(inputs, cache: cache ?? [], state: recurrentState, positions: positionIds)
        // Engine resolution validates dimensions/dtype before this nonthrowing
        // protocol entry. Preserve full-position logits for its forward contract.
        do {
            return try composite.languageModel.forwardWithEmbeddings(inputs, embeddings: inputEmbedding,
                cache: cache ?? [], recurrentState: recurrentState)
        } catch { preconditionFailure("Super embedding validation was bypassed: \(error)") }
    }
    public func cbv2RecurrentPrefill(_ inputs: MLXArray, inputEmbedding: MLXArray?,
        cache: [KVCache]?, recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?, requirement: CBv2PrefillRequirement) -> MLXArray {
        validateInvariant(inputs, cache: cache ?? [], state: recurrentState, positions: positionIds)
        return composite.languageModel.cbv2RecurrentPrefill(inputs,
            inputEmbedding: inputEmbedding, cache: cache, recurrentState: recurrentState,
            positionIds: nil, requirement: requirement)
    }

    private func validateInvariant(_ tokens: MLXArray, cache: [KVCache],
        state: [CBv2RecurrentStateEvaluation], positions: MLXArray?) {
        precondition(tokens.ndim == 2, "Super expects rectangular serial tokens")
        precondition((try? Self.validateLayout(batchSize: tokens.dim(0), sequenceLength: tokens.dim(1),
            hasExplicitPositions: positions != nil, recurrentOwnerCount: state.count,
            attentionCacheCount: cache.count, expectedAttentionCacheCount: cbv2LayerKinds.count)) != nil,
            "Super recurrent layout must be admitted before forward")
    }
}
