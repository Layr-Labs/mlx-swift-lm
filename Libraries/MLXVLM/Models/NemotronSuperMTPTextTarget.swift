import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Explicit, unregistered target wrapper for MTP (serial verification; the
/// captured subclass verifies rectangular windows). Only this wrapper
/// advertises MTP; the engine speculates only with an assistant bound to its
/// exact language model. Ordinary SuperRecurrentAdapter remains unchanged and
/// fail-closed for MTP. Speculation is text-only: an image request is
/// request-scoped target-only work (`CBv2RecurrentMTPRequestScopedMediaFallback`)
/// whose causal embedding prefill and decode run `base`, the ordinary adapter,
/// and never reach the hidden-returning MTP seams below.
public class NemotronSuperMTPTextTarget: Module, LanguageModel,
    CBv2RecurrentLanguageModelForwardable, CBv2RecurrentLanguageModelPrefillForwardable,
    CBv2RecurrentMTPForwardable, CBv2RecurrentPrefillHiddenForwardable,
    CBv2CausalPositionRequirementProviding, CBv2MTPPolicyTopTwoProviding,
    CBv2RecurrentMTPRequestScopedMediaFallback
{
    @ModuleInfo public var base: NemotronSuperRecurrentAdapter

    public init(_ composite: NemotronSuperModel) {
        _base.wrappedValue = NemotronSuperRecurrentAdapter(composite)
    }

    /// Text target for up to `maximumRequests` concurrent requests. Decode
    /// steps of several requests run the row-exact batched trunk, so every
    /// row's logits and hidden equal that request's one-row decode; prompts
    /// stay one request per forward. One request is the serial target.
    /// `exactPrefixCheckpoints` opts into the adapter's chunk-aligned prefix
    /// checkpoints; a text checkpoint also holds the assistant's prompt history.
    public init(
        _ composite: NemotronSuperModel, maximumRequests: Int, exactPrefixCheckpoints: Bool = false
    ) throws {
        _base.wrappedValue = try NemotronSuperRecurrentAdapter(
            composite, maximumDecodeRows: maximumRequests, decodeProducts: .rowExact,
            exactPrefixCheckpoints: exactPrefixCheckpoints)
    }

    /// Requests this target can decode in one step.
    public var maximumRequests: Int { base.maximumDecodeRows }
    /// Requests one speculative round can verify together: one for serial
    /// verification; the captured target verifies up to `maximumRequests`.
    public var maximumSpeculativeRows: Int { 1 }
    public var cbv2MTPTargetIdentity: ObjectIdentifier {
        ObjectIdentifier(base.composite.languageModel)
    }
    public var cbv2CausalPositionRequirement: CBv2CausalPositionRequirement { .scalarCacheOffset }
    public var cbv2Capabilities: CBv2ModelCapabilities {
        var capabilities = base.cbv2Capabilities
        capabilities.supportsMTP = true
        return capabilities
    }
    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec { base.cbv2RecurrentStateSpec }
    public var cbv2LayerKinds: [CBv2LayerKind] { base.cbv2LayerKinds }
    public var cbv2SupportsPackedPrefill: Bool { false }
    public var calibratedPrefillTokenLimit: Int? { base.calibratedPrefillTokenLimit }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        base.newCache(parameters: parameters)
    }
    public func newCacheV2(
        makeLayerCache: (Int, CBv2LayerKind) throws -> any CBv2AttendingLayerCache
    )
        rethrows -> [any CBv2AttendingLayerCache]
    { try base.newCacheV2(makeLayerCache: makeLayerCache) }
    /// The single-request `LMInput` path stays text-only; images are served
    /// through the CBv2 engine.
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
        base.cbv2RecurrentPrefill(
            inputs, inputEmbedding: inputEmbedding, cache: cache,
            recurrentState: recurrentState, positionIds: positionIds, requirement: requirement)
    }
    public var supportsVisionSpanPrefill: Bool { base.supportsVisionSpanPrefill }
    public var supportsCausalVisionPrefill: Bool { base.supportsCausalVisionPrefill }
    public func scaledInputEmbeddings(_ inputs: MLXArray) -> MLXArray {
        base.scaledInputEmbeddings(inputs)
    }
    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        base.embeddingForward(inputs, inputEmbedding: inputEmbedding, cache: cache)
    }
    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray,
        cache: [KVCache]?, recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?
    ) -> MLXArray {
        base.embeddingForward(
            inputs, inputEmbedding: inputEmbedding, cache: cache,
            recurrentState: recurrentState, positionIds: positionIds)
    }
    public func cbv2MTPTopTwo(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
        cbv2TopTwoRows(logits)
    }
    public func cbv2ForwardWithHidden(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        do {
            guard tokens.ndim == 2, tokens.dim(0) > 1 else {
                return try base.composite.languageModel.superMTPTextForward(
                    tokens, caches: caches,
                    recurrentState: recurrentState, positionIds: positionIds)
            }
            guard let plan = base.rowExactPlan, tokens.dim(0) <= maximumRequests else {
                throw NemotronHModel.SuperMTPTextError.layout
            }
            return try base.composite.languageModel.superMTPBatchedTextForward(
                tokens, caches: caches, recurrentState: recurrentState,
                positionIds: positionIds, plan: plan)
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

/// Text target that also verifies a drafted window in one captured forward
/// whose every position is bit-identical to one-token decode. Opt-in: only an
/// assistant requiring `.rectangularExact` uses the captured seam. One window
/// may carry up to `maximumRequests` requests (at most
/// `NemotronHModel.superMTPMaximumWindowRequests`).
public final class NemotronSuperMTPCaptureTarget: NemotronSuperMTPTextTarget,
    CBv2RecurrentCaptureMTPForwardable
{
    private let plan: NemotronSuperMTPCapturePlan

    public init(capturing composite: NemotronSuperModel) throws {
        plan = try composite.languageModel.superMTPCapturePlan()
        super.init(composite)
    }

    public init(
        capturing composite: NemotronSuperModel, maximumRequests: Int,
        exactPrefixCheckpoints: Bool = false
    ) throws {
        guard maximumRequests <= NemotronHModel.superMTPMaximumWindowRequests else {
            throw NemotronHModel.SuperMTPCaptureError.window
        }
        plan = try composite.languageModel.superMTPCapturePlan()
        try super.init(
            composite, maximumRequests: maximumRequests,
            exactPrefixCheckpoints: exactPrefixCheckpoints)
    }

    public override var maximumSpeculativeRows: Int { maximumRequests }

    public func cbv2ForwardWithHiddenCaptured(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        do {
            guard tokens.ndim == 2, tokens.dim(0) <= maximumRequests else {
                throw NemotronHModel.SuperMTPCaptureError.window
            }
            return try base.composite.languageModel.superMTPCapturedTextForward(
                tokens, caches: caches, recurrentState: recurrentState,
                positionIds: positionIds, plan: plan)
        } catch { preconditionFailure("Super MTP captured window was not admitted: \(error)") }
    }
}
