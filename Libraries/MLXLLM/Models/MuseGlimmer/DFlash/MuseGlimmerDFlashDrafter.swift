// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXLMCommon

/// What the drafter reads from its target. The drafter has no embedding
/// table and no output head of its own.
public protocol MuseGlimmerDFlashTarget: AnyObject {
    /// The identity the engine's model reports as `mtpTargetIdentity`.
    var dflashTargetIdentity: ObjectIdentifier { get }
    var dflashHiddenSize: Int { get }
    var dflashVocabularySize: Int { get }
    /// Width of the hidden rows the target hands to the MTP seam: the
    /// residual stream after each tapped layer, concatenated.
    var dflashContextWidth: Int { get }
    /// The layers whose residual streams make those rows, in order. The
    /// drafter was trained on one set and refuses any other.
    var dflashTappedLayerIDs: [Int] { get }
    /// Rows of the token embedding table WITHOUT the target's embedding
    /// norm: `[..., hiddenSize]`.
    func dflashTokenEmbeddings(_ tokens: MLXArray) -> MLXArray
    /// The output head alone, with no logit multiplier and no softcap:
    /// `[..., vocabulary]`. Both are increasing maps, so the argmax is the
    /// target's.
    func dflashHeadProduct(_ hidden: MLXArray) -> MLXArray
    /// True when one forward over a verify window gives every row the bytes
    /// of the decode step it replaces (`rectangularExact`).
    var dflashVerifiesWindowsExactly: Bool { get }
}

extension MuseGlimmerDFlashTarget {
    public var dflashVerifiesWindowsExactly: Bool { false }
}

/// The Muse Glimmer DFlash block drafter as a request-stateful CBv2 drafter.
///
/// Bound to one target instance. Each request owns the keys and values of its
/// last `sliding_window` committed target positions. The first draft step of
/// a round runs one block forward and keeps its proposals; later steps of the
/// round return them. Proposals are the argmax of the target's own head, and
/// the engine keeps one only when it equals the token the target itself
/// chose, so they change speed and never the output.
public final class MuseGlimmerDFlashDrafter: CBv2MTPAdvisoryHistoryDrafter, @unchecked Sendable {
    public struct Settings: Sendable, Equatable {
        /// Draft tokens per round, at most the engine's tested window of 7
        /// and the block's `blockSize - 1`.
        public var maximumDraftTokens: Int
        /// How the target scores a round; nil keeps the engine's setting.
        public var verification: CBv2MTPVerificationMode?
        /// Requests one round may hold. More than one, with rectangular
        /// verification, needs a target whose rows do not depend on each
        /// other.
        public var maximumSpeculativeRows: Int
        /// True only for a target whose verify arithmetic for one row never
        /// depends on the other rows of the step.
        public var commitsAcceptedPrefixPerRow: Bool
        /// Separately admitted allowance for allocator and backend scratch.
        public var backendScratchBytes: Int
        /// A rectangular round attends its (row, column) pairs as rows of one
        /// paged decode dispatch per layer instead of one per column.
        public var batchesVerifyColumnsAsDecodeRows: Bool

        public init(
            maximumDraftTokens: Int = CBv2MTPConfig.testedMaxDraftTokens,
            verification: CBv2MTPVerificationMode? = nil, maximumSpeculativeRows: Int = 1,
            commitsAcceptedPrefixPerRow: Bool = false, backendScratchBytes: Int = 0,
            batchesVerifyColumnsAsDecodeRows: Bool = false
        ) {
            self.maximumDraftTokens = maximumDraftTokens
            self.verification = verification
            self.maximumSpeculativeRows = maximumSpeculativeRows
            self.commitsAcceptedPrefixPerRow = commitsAcceptedPrefixPerRow
            self.backendScratchBytes = backendScratchBytes
            self.batchesVerifyColumnsAsDecodeRows = batchesVerifyColumnsAsDecodeRows
        }

        /// One-forward exact verification on a target with exact windows,
        /// each row committing its own prefix and attending its columns as
        /// decode rows; serial verification otherwise.
        public static func verifying(
            _ target: any MuseGlimmerDFlashTarget, maximumSpeculativeRows: Int = 1,
            backendScratchBytes: Int = 0, batchesVerifyColumnsAsDecodeRows: Bool = true
        ) -> Settings {
            let exact = target.dflashVerifiesWindowsExactly
            return Settings(
                verification: exact ? .rectangularExact : .serialTarget,
                maximumSpeculativeRows: exact ? maximumSpeculativeRows : 1,
                commitsAcceptedPrefixPerRow: exact, backendScratchBytes: backendScratchBytes,
                batchesVerifyColumnsAsDecodeRows: exact && batchesVerifyColumnsAsDecodeRows)
        }
    }

    public let target: any MuseGlimmerDFlashTarget
    public let model: MuseGlimmerDFlashDraftModel
    public let settings: Settings
    /// `[1, hiddenSize]`, the target's raw embedding of the mask token.
    private let maskEmbedding: MLXArray

    public init(
        target: any MuseGlimmerDFlashTarget, model: MuseGlimmerDFlashDraftModel,
        settings: Settings = Settings()
    ) throws {
        let c = model.configuration
        guard target.dflashHiddenSize == c.hiddenSize else {
            throw MuseGlimmerDFlashError.target("hidden size")
        }
        guard target.dflashContextWidth == c.contextWidth else {
            throw MuseGlimmerDFlashError.target("context width")
        }
        guard target.dflashTappedLayerIDs == c.targetLayerIDs else {
            throw MuseGlimmerDFlashError.target("tapped layers")
        }
        guard c.maskTokenID < target.dflashVocabularySize else {
            throw MuseGlimmerDFlashError.target("mask token")
        }
        guard
            (1 ... min(CBv2MTPConfig.testedMaxDraftTokens, c.draftTokensPerBlock)).contains(
                settings.maximumDraftTokens),
            (1 ... CBv2MTPConfig.testedMaxSpeculativeBatch).contains(
                settings.maximumSpeculativeRows),
            settings.backendScratchBytes >= 0
        else { throw MuseGlimmerDFlashError.configuration("settings") }
        guard settings.verification != .rectangularExact || target.dflashVerifiesWindowsExactly
        else { throw MuseGlimmerDFlashError.target("exact verify windows") }
        self.target = target
        self.model = model
        self.settings = settings
        let mask = target.dflashTokenEmbeddings(MLXArray([Int32(c.maskTokenID)]))
        guard mask.shape == [1, c.hiddenSize] else {
            throw MuseGlimmerDFlashError.target("embedding shape")
        }
        maskEmbedding = mask.asType(model.activationDType)
        try MLX.withError { errors in
            eval(maskEmbedding)
            try errors.check()
        }
    }

    // MARK: Engine policy

    public var mtpTargetIdentity: ObjectIdentifier? { target.dflashTargetIdentity }
    public var requiredVerificationMode: CBv2MTPVerificationMode? { settings.verification }
    public var maximumDraftTokens: Int? { settings.maximumDraftTokens }
    public var maximumSpeculativeBatch: Int? { settings.maximumSpeculativeRows }
    public var commitsAcceptedPrefixPerRow: Bool { settings.commitsAcceptedPrefixPerRow }
    public var batchesVerifyColumnsAsDecodeRows: Bool {
        settings.verification == .rectangularExact && settings.batchesVerifyColumnsAsDecodeRows
    }
    /// Exact acceptance only (decision D6): typical acceptance can emit a
    /// draft, and the advisory history then loses its observation positions.
    public var requiresExactAcceptance: Bool { true }
    /// A proposal is kept only when it equals the token pre-sampled from the
    /// target's own distribution, and that token is what is emitted, so
    /// sampled rows speculate too.
    public var supportsTargetPrefixAcceptance: Bool { true }
    /// Observation builds lazy writes into the context ring; they must be
    /// evaluated with the step that produced the observed rows.
    public var requiresCommittedObservationFence: Bool { true }
    /// The ring is a fixed charge (`boundedRequestAllocation`), not a rate.
    public var requestStateBytesPerToken: Int { 0 }
    /// The context only decides what is proposed. It is not part of a prefix
    /// checkpoint; after a restore it starts at the first observed position.
    public var requestStateIsAdvisory: Bool { true }

    // MARK: Request state

    final class State: CBv2MTPRequestState {
        let owner: ObjectIdentifier
        let context: MuseGlimmerDFlashContextState
        var released = false
        /// The last committed position's row has not arrived yet: a verify
        /// round hands it over as the next carry.
        var awaitsCarry = false
        /// This round's proposals `[maximumDraftTokens]` and their hidden
        /// rows `[maximumDraftTokens, hiddenSize]`.
        var memo: (tokens: MLXArray, hidden: MLXArray)?
        var committedInputCount = 0
        var stagedInputCount = 0
        /// Why this request cannot continue; reported at the next fence.
        var refusal: String?

        init(owner: ObjectIdentifier, context: MuseGlimmerDFlashContextState) {
            self.owner = owner
            self.context = context
        }

        var arrays: [MLXArray] { context.arrays + (memo.map { [$0.tokens, $0.hidden] } ?? []) }
        var materializedBytes: Int { arrays.reduce(0) { $0 + $1.nbytes } }
    }

    private final class Prepared: CBv2MTPPreparedCapture {}

    private func state(_ value: any CBv2MTPRequestState) -> State {
        guard let state = value as? State, state.owner == ObjectIdentifier(self), !state.released
        else { preconditionFailure("Muse Glimmer DFlash: foreign or released request state") }
        return state
    }

    public func makeRequestState() -> any CBv2MTPRequestState {
        State(owner: ObjectIdentifier(self), context: model.makeContextState())
    }

    public func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture { Prepared() }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure("Muse Glimmer DFlash requires request-owned state")
    }

    /// Target hidden rows `[1, rows, contextWidth]` as feature rows; nil for
    /// any other shape.
    private func features(_ hidden: MLXArray) -> MLXArray? {
        guard hidden.ndim == 3, hidden.dim(0) == 1, hidden.dim(1) > 0,
            hidden.dim(2) == model.configuration.contextWidth
        else { return nil }
        return hidden.reshaped([hidden.dim(1), hidden.dim(2)])
    }

    /// Commit the rows of one observation as the next context positions.
    public func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState
    ) {
        let s = state(requestState)
        observe(observation, at: s.context.nextPosition, state: s)
    }

    /// The same, with the absolute position of the observation's first row.
    /// A position after the held rows restarts the context there; rows
    /// already held are skipped.
    public func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation, startingAt position: Int,
        requestState: any CBv2MTPRequestState
    ) {
        observe(observation, at: position, state: state(requestState))
    }

    private func observe(
        _ observation: CBv2MTPCommittedTargetObservation, at position: Int, state s: State
    ) {
        guard s.refusal == nil else { return }
        guard s.stagedInputCount == 0, position >= 0, let rows = features(observation.hidden)
        else {
            s.refusal = "observation"
            return
        }
        model.commit(features: rows, at: position, to: s.context)
        // Whatever arrived, the context now ends at its newest row.
        s.awaitsCarry = false
        s.committedInputCount += rows.dim(0)
    }

    /// Draft index 0 commits a carry row that is still missing, runs one
    /// block forward and keeps its proposals. Later indices return them.
    ///
    /// A call this drafter cannot serve never stops the engine thread: it
    /// returns a well-formed proposal and the request ends with an error at
    /// the round's fence (`requestStateDidFinishEvaluation`).
    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        let s = state(requestState)
        let c = model.configuration
        let index = s.stagedInputCount
        s.stagedInputCount += 1
        if s.refusal == nil, index == 0 {
            s.refusal = startRound(s, tokens: tokens, hidden: hidden)
        }
        guard s.refusal == nil, shortlist == nil, index < settings.maximumDraftTokens,
            let memo = s.memo
        else {
            s.refusal = s.refusal ?? "draft step"
            s.memo = nil
            return (
                MLXArray([Int32(0)]),
                MLXArray.zeros([1, 1, c.hiddenSize], dtype: model.activationDType)
            )
        }
        return (
            memo.tokens[index ..< (index + 1)],
            memo.hidden[index ..< (index + 1), 0...].reshaped([1, 1, c.hiddenSize])
        )
    }

    /// Builds the round's block and memo. Returns why it could not.
    private func startRound(_ s: State, tokens: MLXArray, hidden: MLXArray) -> String? {
        let c = model.configuration
        guard tokens.shape == [1, 1] else { return "seed token" }
        if s.awaitsCarry {
            guard let row = features(hidden), row.dim(0) == 1 else { return "carry" }
            model.commit(features: row, at: s.context.nextPosition, to: s.context)
            s.awaitsCarry = false
            s.committedInputCount += 1
        }
        let anchor = target.dflashTokenEmbeddings(tokens.asType(.int32))
        guard anchor.size == c.hiddenSize else { return "embedding" }
        let block = model.draftBlock(
            embeddings: model.blockEmbeddings(anchor: anchor, mask: maskEmbedding),
            state: s.context)
        let rows = block[1 ..< (1 + settings.maximumDraftTokens), 0...]
        let logits = target.dflashHeadProduct(rows)
        guard logits.shape == [settings.maximumDraftTokens, target.dflashVocabularySize] else {
            return "head"
        }
        s.memo = (argMax(logits, axis: -1).asType(.int32), rows)
        return nil
    }

    public func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] {
        guard let s = requestState as? State, s.owner == ObjectIdentifier(self), !s.released
        else { return [] }
        return s.arrays
    }

    /// Ends a request whose target rows or round calls did not fit.
    public func requestStateDidFinishEvaluation(_ requestState: any CBv2MTPRequestState) throws {
        guard let s = requestState as? State, s.owner == ObjectIdentifier(self), !s.released
        else { throw MuseGlimmerDFlashError.state("foreign or released") }
        if let refusal = s.refusal { throw MuseGlimmerDFlashError.state(refusal) }
    }

    /// The accepted drafts' trusted target rows become context. The row of
    /// the last committed position follows as the next carry.
    public func finalizeRound(
        requestState: any CBv2MTPRequestState, confirmedInputTokens: Int,
        committedDraftTokens: MLXArray, committedTargetHidden: MLXArray
    ) {
        let s = state(requestState)
        defer { clearRound(s) }
        guard s.refusal == nil else { return }
        let rows = confirmedInputTokens - 1
        guard s.stagedInputCount > 0, rows >= 0, rows <= s.stagedInputCount,
            committedTargetHidden.ndim == 3, committedTargetHidden.dim(1) == rows
        else {
            s.refusal = "finalize"
            return
        }
        if rows > 0 {
            guard let accepted = features(committedTargetHidden) else {
                s.refusal = "accepted rows"
                return
            }
            model.commit(features: accepted, at: s.context.nextPosition, to: s.context)
        }
        s.awaitsCarry = true
        s.committedInputCount += rows
    }

    /// Only the memo belongs to a round: a block never writes the context.
    public func discardRound(requestState: any CBv2MTPRequestState) {
        clearRound(state(requestState))
    }

    private func clearRound(_ s: State) {
        s.memo = nil
        s.stagedInputCount = 0
    }

    public func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        guard let s = requestState as? State, s.owner == ObjectIdentifier(self), !s.released
        else { return }
        clearRound(s)
        s.context.release()
        s.awaitsCarry = false
        s.committedInputCount = 0
        s.released = true
    }
}

// MARK: - Admission

extension MuseGlimmerDFlashDrafter: CBv2MTPBoundedAllocationProviding {
    /// Explicit arrays of one context commit of `rows` rows: the feature rows
    /// in the drafter dtype, the projected context before and after its norm,
    /// and each layer's keys (projected, normed, rotated) and values.
    func commitBytes(rows: Int) -> Int {
        let c = model.configuration
        let size = model.activationDType.size
        let keyWidth = c.kvHeads * c.headDimensions
        return rows * size * (c.contextWidth + 2 * c.hiddenSize + c.hiddenLayers * 4 * keyWidth)
    }

    /// Explicit arrays of one block forward and its head rows. Per layer: the
    /// keys and values handed to attention (held rows and the block), float32
    /// scores, queries, the attention output and the MLP's three products.
    /// Then float32 head rows and their argmax.
    var blockBytes: Int {
        let c = model.configuration
        let size = model.activationDType.size
        let keys = c.slidingWindow + c.blockSize
        let layer =
            2 * c.kvHeads * keys * c.headDimensions * size
            + c.attentionHeads * c.blockSize * keys * 4
            + c.blockSize * size
            * (3 * c.attentionHeads * c.headDimensions + 6 * c.hiddenSize + 3 * c.intermediateSize)
        let head =
            settings.maximumDraftTokens * (target.dflashVocabularySize * 4 + c.hiddenSize * 4)
        return c.hiddenLayers * layer + head + c.blockSize * (keys + c.hiddenSize * size)
    }

    /// Per request: the ring's two buffers, one copy of each for a write MLX
    /// cannot make in place, and one commit of at most `window` rows (older
    /// rows of a longer chunk are never projected). Once per engine: one
    /// block forward for each request a round may hold. The ring does not
    /// grow with the request, so nothing is charged per token.
    public func boundedRequestAllocation(limits: CBv2MTPAllocationLimits)
        -> CBv2MTPBoundedAllocationSpec?
    {
        let c = model.configuration
        guard limits.maximumPrefillTokens > 0, limits.maximumDraftTokens > 0 else { return nil }
        let buffer =
            MuseGlimmerDFlashContextState.residentBytes(
                configuration: c, dtype: model.activationDType) / 2
        let commitRows = min(max(limits.maximumPrefillTokens, c.blockSize), c.slidingWindow)
        var owner = [
            CBv2MTPFixedBufferSpec(
                logicalBytes: blockBytes, allocationCount: settings.maximumSpeculativeRows)
        ]
        if settings.backendScratchBytes > 0 {
            owner.append(.init(logicalBytes: settings.backendScratchBytes))
        }
        return CBv2MTPBoundedAllocationSpec(
            resident: [.init(logicalBytes: buffer, allocationCount: 2)],
            working: [
                .init(logicalBytes: buffer, allocationCount: 2),
                .init(logicalBytes: commitBytes(rows: commitRows)),
            ],
            hostBytes: c.blockSize * (c.slidingWindow + c.blockSize),
            ownerWorking: owner)
    }
}
