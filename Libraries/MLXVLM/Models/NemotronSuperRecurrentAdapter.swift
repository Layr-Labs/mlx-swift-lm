import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Super-only causal adapter. It owns the composite, not a copied text tower.
/// Prompt and media prefill are serial; text decode may be a rectangular
/// `[B, 1]` step of independent requests, opted into with `maximumDecodeRows`.
/// Exact prefix reuse is opted into with `exactPrefixCheckpoints`.
/// No MTP, positional remapping or packed prompt rows.
/// Reference: Transformers 049d2bf1220747b6d39e2a978b9f5fe0defa1dca
/// NemotronHModel/Attention (v5.8.0); Omni checkpoint
/// 7d4bc747b6bdc2cff64c58d65da0d08dd9caa25c substitutes image embeddings.
public final class NemotronSuperRecurrentAdapter: Module, LanguageModel,
    CBv2RecurrentLanguageModelForwardable, CBv2RecurrentLanguageModelPrefillForwardable,
    CBv2PositionedRecurrentEmbeddingForwardable, CBv2CausalPositionRequirementProviding
{
    @ModuleInfo public var composite: NemotronSuperModel
    /// Largest decode batch. One is the serial owner. A larger value is valid
    /// only when the owner admits every row's `NemotronSuperDecodeCharge` and
    /// schedules at most this many concurrent requests.
    public let maximumDecodeRows: Int
    public let decodeProducts: DecodeProducts
    /// The engine may capture and restore recurrent checkpoints (conv/SSM
    /// state with the attention KV) only at multiples of the fixed prefill
    /// chunk: tiled prefill output depends on the chunk partition, so only
    /// there does a restore continue a cold prefill bit for bit. Images are
    /// causal embeddings at scalar positions, so an image request reuses only
    /// within the scope of a host-bound identity of its evaluated embeddings
    /// and spans, and never without one.
    public let exactPrefixCheckpoints: Bool
    /// Resolved blocks of the row-exact batched trunk; nil for one row and for
    /// shared products.
    let rowExactPlan: NemotronSuperMTPCapturePlan?
    /// `cbv2RecurrentStateSpec`, resolved once: the language model derives it
    /// from a lazy probe forward of the whole trunk (no evaluation, but a full
    /// graph construction), and the engine reads it on every step.
    private let recurrentSpec = NemotronSuperRecurrentSpecCache()
    public enum Failure: Error { case position, batch, recurrentOwnership, caches }

    /// How a decode step with several rows computes its products.
    public enum DecodeProducts: Sendable, Equatable {
        /// Every row is bit-identical to its one-token decode. FP8 projections,
        /// the BF16 latent projections and the router read each weight once
        /// per dispatch of up to eight rows (`ModelOptRowExact`); NVFP4
        /// projections and experts run one single-vector product per row;
        /// recurrent rows are separate buffers.
        case rowExact
        /// One M = B product per projection reads every weight once per step.
        /// MLX's wider kernels change the float32 summation order, so rows
        /// may differ from their one-token decode; recurrent rows are views of
        /// one batch buffer (`NemotronSuperDecodeCharge.batchAliasBytes`).
        case shared
    }

    /// Rows of one decode step this adapter can be configured for.
    public static let decodeRowLimit = 64

    public init(_ composite: NemotronSuperModel) {
        maximumDecodeRows = 1
        decodeProducts = .rowExact
        exactPrefixCheckpoints = false
        rowExactPlan = nil
        self._composite.wrappedValue = composite
    }

    /// Opt-in rectangular text decode of up to `maximumDecodeRows` requests
    /// and, with `exactPrefixCheckpoints`, exact prefix reuse.
    public init(
        _ composite: NemotronSuperModel, maximumDecodeRows: Int,
        decodeProducts: DecodeProducts = .rowExact, exactPrefixCheckpoints: Bool = false
    ) throws {
        guard (1 ... Self.decodeRowLimit).contains(maximumDecodeRows) else {
            throw Failure.batch
        }
        self.maximumDecodeRows = maximumDecodeRows
        self.decodeProducts = decodeProducts
        self.exactPrefixCheckpoints = exactPrefixCheckpoints
        rowExactPlan =
            decodeProducts == .rowExact && maximumDecodeRows > 1
            ? try composite.languageModel.superMTPCapturePlan() : nil
        self._composite.wrappedValue = composite
    }

    /// Pure admission validation, before cache/state binding or graph creation.
    /// Explicit positions might encode packing or reset masks; never drop them.
    /// Several rows are accepted only as one-token decode, one recurrent owner
    /// per row; every prompt chunk and media prefill stays a single request.
    public static func validateLayout(
        batchSize: Int, sequenceLength: Int,
        hasExplicitPositions: Bool, recurrentOwnerCount: Int,
        attentionCacheCount: Int, expectedAttentionCacheCount: Int,
        maximumDecodeRows: Int = 1
    ) throws {
        guard !hasExplicitPositions else { throw Failure.position }
        guard sequenceLength > 0, batchSize >= 1,
            batchSize == 1 || (sequenceLength == 1 && batchSize <= maximumDecodeRows)
        else { throw Failure.batch }
        guard recurrentOwnerCount == batchSize else { throw Failure.recurrentOwnership }
        guard expectedAttentionCacheCount > 0,
            attentionCacheCount == expectedAttentionCacheCount
        else { throw Failure.caches }
    }

    public var cbv2CausalPositionRequirement: CBv2CausalPositionRequirement { .scalarCacheOffset }
    /// `supportsPrefixReuse` stays false: attention-only reuse would drop the
    /// Mamba state. The attention KV may live in segmented native paged
    /// storage with the observed layer dtypes; the backend is the host's choice.
    public var cbv2Capabilities: CBv2ModelCapabilities {
        var capabilities = composite.languageModel.cbv2Capabilities
        capabilities.supportsPagedKV = true
        capabilities.requiresNativePagedKV = true
        if exactPrefixCheckpoints {
            capabilities.supportsRecurrentCheckpointReuse = true
            capabilities.alignsRecurrentCheckpointsToPrefillChunks = true
            capabilities.supportsCausalMediaCheckpointReuse = true
        }
        return capabilities
    }
    /// The loaded composite's spec (layer shapes and the activation types of
    /// its loaded projections), fixed once loaded.
    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec {
        recurrentSpec.value { composite.cbv2RecurrentStateSpec }
    }
    public var cbv2LayerKinds: [CBv2LayerKind] { composite.cbv2LayerKinds }
    public var cbv2SupportsPackedPrefill: Bool { false }
    public var supportsVisionSpanPrefill: Bool { false }
    public var supportsCausalVisionPrefill: Bool { true }
    public var calibratedPrefillTokenLimit: Int? {
        composite.languageModel.calibratedPrefillTokenLimit
    }

    public func newCacheV2(
        makeLayerCache: (Int, CBv2LayerKind) throws -> any CBv2AttendingLayerCache
    )
        rethrows -> [any CBv2AttendingLayerCache]
    {
        try composite.languageModel.newCacheV2(makeLayerCache: makeLayerCache)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try composite.languageModel.newCache(parameters: parameters)
    }
    /// The single-request `LMInput` path is text-only: the language model's
    /// prepare reads only the tokens and would drop media silently. Media is
    /// served through the CBv2 engine.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws
        -> PrepareResult
    {
        guard input.image == nil, input.video == nil else {
            throw NemotronHModel.SuperMTPTextError.media
        }
        return try composite.languageModel.prepare(
            input, cache: cache, state: state, prefill: prefill)
    }
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        composite.languageModel(inputs, cache: cache)
    }
    public func scaledInputEmbeddings(_ inputs: MLXArray) -> MLXArray {
        composite.languageModel.inputEmbeddings(inputs)
    }
    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray,
        cache: [KVCache]?
    ) -> MLXArray {
        preconditionFailure("Super image prefill requires request-owned recurrent state")
    }
    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        validateInvariant(
            tokens, cache: caches, state: recurrentState, positions: nil,
            maximumDecodeRows: maximumDecodeRows)
        if tokens.dim(0) > 1, let rowExactPlan {
            do {
                return try composite.languageModel.superRowExactBatchedDecode(
                    tokens, caches: caches, recurrentState: recurrentState, plan: rowExactPlan)
            } catch { preconditionFailure("Super batched decode layout was bypassed: \(error)") }
        }
        return composite.languageModel.cbv2Forward(
            tokens, caches: caches, recurrentState: recurrentState)
    }
    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray,
        cache: [KVCache]?, recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?
    ) -> MLXArray {
        validateInvariant(inputs, cache: cache ?? [], state: recurrentState, positions: positionIds)
        // Engine resolution validates dimensions/dtype before this nonthrowing
        // protocol entry. Preserve full-position logits for its forward contract.
        do {
            return try composite.languageModel.forwardWithEmbeddings(
                inputs, embeddings: inputEmbedding,
                cache: cache ?? [], recurrentState: recurrentState)
        } catch { preconditionFailure("Super embedding validation was bypassed: \(error)") }
    }
    public func cbv2RecurrentPrefill(
        _ inputs: MLXArray, inputEmbedding: MLXArray?,
        cache: [KVCache]?, recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?, requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        validateInvariant(inputs, cache: cache ?? [], state: recurrentState, positions: positionIds)
        return composite.languageModel.cbv2RecurrentPrefill(
            inputs,
            inputEmbedding: inputEmbedding, cache: cache, recurrentState: recurrentState,
            positionIds: nil, requirement: requirement)
    }

    /// Embedding and prompt entries keep the serial default of one row.
    private func validateInvariant(
        _ tokens: MLXArray, cache: [KVCache],
        state: [CBv2RecurrentStateEvaluation], positions: MLXArray?,
        maximumDecodeRows: Int = 1
    ) {
        precondition(tokens.ndim == 2, "Super expects rectangular tokens")
        precondition(
            (try? Self.validateLayout(
                batchSize: tokens.dim(0), sequenceLength: tokens.dim(1),
                hasExplicitPositions: positions != nil, recurrentOwnerCount: state.count,
                attentionCacheCount: cache.count, expectedAttentionCacheCount: cbv2LayerKinds.count,
                maximumDecodeRows: maximumDecodeRows))
                != nil,
            "Super recurrent layout must be admitted before forward")
    }
}

/// One adapter's recurrent-state spec, computed on first use and then
/// returned as is. The engine asks for the spec on its hot paths (every
/// forward, plan and MTP eligibility check), while the language model
/// resolves it from a lazy probe forward of the whole trunk; the result
/// depends only on the loaded configuration and projection types.
final class NemotronSuperRecurrentSpecCache: @unchecked Sendable {
    private let lock = NSLock()
    private var spec: CBv2RecurrentStateSpec?

    func value(_ compute: () -> CBv2RecurrentStateSpec) -> CBv2RecurrentStateSpec {
        lock.lock()
        defer { lock.unlock() }
        if let spec { return spec }
        let resolved = compute()
        spec = resolved
        return resolved
    }
}
