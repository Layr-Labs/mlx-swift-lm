import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Explicit, unregistered text-only target wrapper for serial MTP qualification.
/// It does not construct an assistant or enable engine speculation. Ordinary
/// SuperRecurrentAdapter remains unchanged; image/embedding MTP is not exposed.
public final class NemotronSuperMTPTextTarget: Module, LanguageModel,
    CBv2RecurrentLanguageModelForwardable, CBv2RecurrentLanguageModelPrefillForwardable,
    CBv2RecurrentMTPForwardable, CBv2RecurrentPrefillHiddenForwardable,
    CBv2CausalPositionRequirementProviding, CBv2MTPPolicyTopTwoProviding
{
    @ModuleInfo public var base: NemotronSuperRecurrentAdapter
    public init(_ composite: NemotronSuperModel) {
        _base.wrappedValue = NemotronSuperRecurrentAdapter(composite)
    }
    public var cbv2MTPTargetIdentity: ObjectIdentifier {
        ObjectIdentifier(base.composite.languageModel)
    }
    public var cbv2CausalPositionRequirement: CBv2CausalPositionRequirement { .scalarCacheOffset }
    public var cbv2Capabilities: CBv2ModelCapabilities { base.cbv2Capabilities }
    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec { base.cbv2RecurrentStateSpec }
    public var cbv2LayerKinds: [CBv2LayerKind] { base.cbv2LayerKinds }
    public var cbv2SupportsPackedPrefill: Bool { false }
    public var calibratedPrefillTokenLimit: Int? { base.calibratedPrefillTokenLimit }

    /// Host admission calls this throwing validator before the protocol's
    /// nonthrowing entries. This wrapper must never receive image embeddings.
    public static func validateTextPrefill(hasEmbeddings: Bool) throws {
        guard !hasEmbeddings else { throw NemotronHModel.SuperMTPTextError.media }
    }
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        base.newCache(parameters: parameters)
    }
    public func newCacheV2(
        makeLayerCache: (Int, CBv2LayerKind) throws -> any CBv2AttendingLayerCache
    )
        rethrows -> [any CBv2AttendingLayerCache]
    { try base.newCacheV2(makeLayerCache: makeLayerCache) }
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        guard input.image == nil, input.video == nil else {
            throw NemotronHModel.SuperMTPTextError.media
        }
        guard input.text.mask == nil else { throw NemotronHModel.SuperMTPTextError.layout }
        return try base.prepare(input, cache: cache, windowSize: windowSize)
    }
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        base(inputs, cache: cache)
    }
    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        base.cbv2Forward(tokens, caches: caches, recurrentState: recurrentState)
    }
    public func cbv2RecurrentPrefill(
        _ inputs: MLXArray, inputEmbedding: MLXArray?,
        cache: [KVCache]?, recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?, requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        precondition(inputEmbedding == nil, "Super MTP target is text-only")
        return base.cbv2RecurrentPrefill(
            inputs, inputEmbedding: nil, cache: cache,
            recurrentState: recurrentState, positionIds: positionIds, requirement: requirement)
    }
    public func cbv2MTPTopTwo(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
        cbv2TopTwoRows(logits)
    }
    public func cbv2ForwardWithHidden(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        do {
            return try base.composite.languageModel.superMTPTextForward(
                tokens, caches: caches,
                recurrentState: recurrentState, positionIds: positionIds)
        } catch { preconditionFailure("Super MTP text layout was not admitted") }
    }
    public func cbv2ForwardWithHiddenForPrefill(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        do {
            return try base.composite.languageModel.superMTPTextPrefill(
                tokens, caches: caches,
                recurrentState: recurrentState, positionIds: positionIds, requirement: requirement)
        } catch { preconditionFailure("Super MTP text prefill was not admitted") }
    }
}
