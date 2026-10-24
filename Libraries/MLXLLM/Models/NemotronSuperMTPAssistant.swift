import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Request-stateful speculative assistant over the Super checkpoint's embedded
/// prediction head. It is bound to the exact loaded language model whose
/// embeddings, output head and verification it shares, consumes only trusted
/// shifted target history, drafts greedily from each request's own state and
/// requires serial or exact captured rectangular target verification
/// (`Settings.Verification`). Rectangular rounds may verify up to
/// `Settings.maximumSpeculativeRows` requests in one window. Draft graphs never synchronize with
/// the host. A refused graph (for example, invalid registered calibration) or
/// invalid device routing ends the request at the engine's committed fence.
/// Media requests never reach it: the text target refuses embeddings.
public final class NemotronSuperMTPAssistant: CBv2MTPRequestStatefulDrafter, @unchecked Sendable {
    public enum Failure: Error, Equatable {
        case target, head, outputHead, depth, sequenceBound, workspace, state, rows
        /// The draft graph could not be built; nothing was committed.
        case graphRefused
        /// Device routing evidence was invalid after evaluation.
        case invalidRouting
    }

    public struct Settings: Sendable, Equatable {
        /// How the target scores a drafted chain. `rectangularExact` needs a
        /// target with the captured window (`NemotronSuperMTPCaptureTarget`);
        /// without it the engine clamps speculation to depth zero.
        public enum Verification: Sendable, Equatable { case serial, rectangularExact }
        /// Depth one is the qualified baseline. Deeper chains need the explicit
        /// `allowsDeepDrafts` opt-in and remain bounded by `maximumDepth`.
        public static let maximumDepth = 3
        public var maximumDraftTokens: Int
        public var allowsDeepDrafts: Bool
        /// Largest prompt plus generation one request may carry.
        public var maximumSequenceLength: Int
        /// Canonical prompt-history chunk, independent of target prefill chunks.
        public var historyChunkRows: Int
        /// Separately admitted allowance for allocator/backend scratch.
        public var backendScratchBytes: Int
        public var verification: Verification
        /// Requests one rectangular round verifies together, at most CBv2's
        /// tested batch ceiling. Above one requires `rectangularExact` and a
        /// target serving that many requests; every row commits its own
        /// accepted prefix. Serial verification stays one request.
        public var maximumSpeculativeRows: Int

        public init(
            maximumDraftTokens: Int = 1, allowsDeepDrafts: Bool = false,
            maximumSequenceLength: Int, historyChunkRows: Int = 512,
            backendScratchBytes: Int, verification: Verification = .serial,
            maximumSpeculativeRows: Int = 1
        ) {
            self.maximumDraftTokens = maximumDraftTokens
            self.allowsDeepDrafts = allowsDeepDrafts
            self.maximumSequenceLength = maximumSequenceLength
            self.historyChunkRows = historyChunkRows
            self.backendScratchBytes = backendScratchBytes
            self.verification = verification
            self.maximumSpeculativeRows = maximumSpeculativeRows
        }
    }

    enum OutputHead {
        case fp8(ModelOptCalibratedLinear)
        case dense(Linear)
    }

    let target: NemotronHModel
    let head: NemotronSuperMTPModule
    let outputHead: OutputHead
    public let settings: Settings
    public let workspace: Workspace
    /// Per-request history capacity: sequence bound plus one round's drafts.
    let historyCapacity: Int

    /// The engine admits `boundedRequestAllocation` for every request before
    /// any assistant array exists, and `boundedRequestTokenAllocation` per
    /// reserved token; there is no separate grant here.
    public init(target: NemotronHModel, head: NemotronSuperMTPModule, settings: Settings) throws {
        guard type(of: target) == NemotronHModel.self else { throw Failure.target }
        let t = target.configuration
        let h = head.configuration
        guard h.hiddenSize == t.hiddenSize, h.vocabSize == t.vocabSize,
            h.numAttentionHeads == t.numAttentionHeads,
            h.numKeyValueHeads == t.numKeyValueHeads, h.headDim == t.headDim,
            h.nRoutedExperts == t.nRoutedExperts, h.numExpertsPerTok == t.numExpertsPerTok,
            h.moeLatentSize == t.moeLatentSize, t.numNextnPredictLayers == 1,
            t.mtpLayersBlockType == ["attention", "moe"]
        else { throw Failure.head }
        guard (1 ... Settings.maximumDepth).contains(settings.maximumDraftTokens),
            settings.maximumDraftTokens == 1 || settings.allowsDeepDrafts
        else { throw Failure.depth }
        guard settings.maximumSequenceLength > 0, settings.historyChunkRows > 0,
            settings.backendScratchBytes >= 0
        else { throw Failure.sequenceBound }
        guard
            (1 ... NemotronHModel.superMTPMaximumWindowRequests).contains(
                settings.maximumSpeculativeRows),
            settings.maximumSpeculativeRows == 1 || settings.verification == .rectangularExact
        else { throw Failure.rows }
        let output: OutputHead
        if let projection = target.lmHead as? ModelOptCalibratedLinear {
            guard case .fp8 = projection.format, projection.inputDimensions == t.hiddenSize,
                projection.outputDimensions == t.vocabSize,
                projection.weightDType == head.weightDType
            else { throw Failure.outputHead }
            output = .fp8(projection)
        } else if let linear = target.lmHead as? Linear, type(of: linear) == Linear.self {
            guard linear.bias == nil, linear.shape == (t.vocabSize, t.hiddenSize) else {
                throw Failure.outputHead
            }
            output = .dense(linear)
        } else {
            throw Failure.outputHead
        }
        let (capacity, overflow) = settings.maximumSequenceLength.addingReportingOverflow(
            settings.maximumDraftTokens + 1)
        guard !overflow else { throw Failure.sequenceBound }
        let probe = try NemotronSuperFP8KVHistory(
            heads: t.numKeyValueHeads,
            headDimension: t.headDim ?? (t.hiddenSize / t.numAttentionHeads),
            capacityTokens: capacity, dtype: head.weightDType)
        self.workspace = try Self.workspace(
            head: head, outputHead: output, settings: settings, history: probe)
        self.target = target
        self.head = head
        self.outputHead = output
        self.settings = settings
        self.historyCapacity = capacity
    }

    // MARK: Engine policy

    public var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(target) }
    public var requiredVerificationMode: CBv2MTPVerificationMode? {
        switch settings.verification {
        case .serial: .serialTarget
        case .rectangularExact: .rectangularExact
        }
    }
    public var maximumDraftTokens: Int? { settings.maximumDraftTokens }
    public var maximumSpeculativeBatch: Int? { settings.maximumSpeculativeRows }
    /// Every product of the captured Super verify window is row-exact, so a
    /// row's arithmetic never depends on its neighbors or on the next step's
    /// batch shape. Serial verification serves one request.
    public var commitsAcceptedPrefixPerRow: Bool { settings.verification == .rectangularExact }
    /// Drafts are deterministic greedy proposals. Accepting one iff it equals
    /// the token pre-sampled from the target's own distribution, and otherwise
    /// emitting that sample, is exact speculative sampling, so sampled rows
    /// speculate too.
    public var supportsTargetPrefixAcceptance: Bool { true }
    public var requiresCommittedObservationFence: Bool { true }
    // History, the partial-block backlog and round workspace are the bounded
    // fixed charge plus its sequence-scaled per-token declaration; no legacy
    // rate is added.
    public var requestStateBytesPerToken: Int { 0 }

    // MARK: Request state

    final class State: CBv2MTPRequestState {
        let owner: ObjectIdentifier
        let backlogLimit: Int
        var history: NemotronSuperFP8KVHistory?
        var committedInputCount = 0
        var stagedInputCount = 0
        var released = false
        var started = false
        var baseOffset = 0
        var trustedOffset = 0
        var pendingTokens: [MLXArray] = []
        var pendingHidden: [MLXArray] = []
        var frontier: MLXArray?
        /// FP8 codes `[K, V]` of a restored prefix checkpoint, written to the
        /// history when the request's bound configures it.
        var restoredCodes: [MLXArray] = []
        var savedTokens: [MLXArray] = []
        var savedHidden: [MLXArray] = []
        var savedFrontier: MLXArray?
        var roots: [MLXArray] = []
        var validity: [MLXArray] = []
        var refused = false

        init(owner: ObjectIdentifier, backlogLimit: Int) {
            self.owner = owner
            self.backlogLimit = backlogLimit
        }

        var arrays: [MLXArray] {
            (history?.innerState() ?? []) + restoredCodes + pendingTokens + pendingHidden
                + savedTokens + savedHidden + [frontier, savedFrontier].compactMap { $0 } + roots
                + validity
        }
        var materializedBytes: Int {
            arrays.reduce(0) { total, array in
                let (sum, overflow) = total.addingReportingOverflow(array.nbytes)
                return overflow ? Int.max : sum
            }
        }
        var hasPendingPrefillForCostAccounting: Bool {
            !started || pendingTokens.reduce(0) { $0 + $1.dim(1) } > backlogLimit
        }
    }

    private final class Prepared: CBv2MTPPreparedCapture {}

    private func state(_ value: any CBv2MTPRequestState) throws -> State {
        guard let s = value as? State, s.owner == ObjectIdentifier(self), !s.released else {
            throw Failure.state
        }
        return s
    }

    public func makeRequestState() -> any CBv2MTPRequestState {
        State(owner: ObjectIdentifier(self), backlogLimit: settings.maximumDraftTokens)
    }

    public func configureRequestState(
        _ requestState: any CBv2MTPRequestState, maximumSequenceLength: Int
    ) throws {
        let s = try state(requestState)
        guard maximumSequenceLength > 0, maximumSequenceLength <= settings.maximumSequenceLength
        else { throw Failure.sequenceBound }
        let bound = maximumSequenceLength + settings.maximumDraftTokens + 1
        if let history = s.history {
            guard history.capacityTokens >= bound else { throw Failure.sequenceBound }
            return
        }
        let c = target.configuration
        let history = try NemotronSuperFP8KVHistory(
            heads: c.numKeyValueHeads,
            headDimension: c.headDim ?? (c.hiddenSize / c.numAttentionHeads),
            capacityTokens: bound, dtype: head.weightDType)
        // A restored checkpoint's drained blocks: the same codes, and the
        // elementwise mirror decoded from them, as the cold request's.
        if !s.restoredCodes.isEmpty {
            try history.restore(s.restoredCodes)
            s.restoredCodes = []
        }
        s.history = history
    }

    public func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture { Prepared() }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure("Super MTP requires request-owned state")
    }

    /// Queue (next token, previous trusted hidden) pairs without evaluation.
    /// The final row becomes the frontier paired with the next input. Queued
    /// pairs go to the FP8 history as soon as they form what the next draft
    /// step would write (`drainTrustedRows`), so only a partial prompt block
    /// stays queued.
    public func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState
    ) {
        guard let s = try? state(requestState) else {
            preconditionFailure("Super MTP foreign or released request state")
        }
        let n = observation.tokens.dim(1)
        precondition(s.stagedInputCount == 0, "Super MTP observed history during a round")
        precondition(
            observation.tokens.shape == [1, n]
                && observation.hidden.shape == [1, n, target.configuration.hiddenSize])
        guard n > 0 else { return }
        var tokens = s.pendingTokens
        var hidden = s.pendingHidden
        if let previous = s.frontier {
            tokens.append(observation.tokens[0..., 0 ..< 1])
            hidden.append(previous)
        }
        // A view of the step's hidden until written or copied below.
        let observed = n > 1 ? observation.hidden[0..., 0 ..< (n - 1), 0...] : nil
        if let observed {
            tokens.append(observation.tokens[0..., 1 ..< n])
            hidden.append(observed)
        }
        let drained = drainTrustedRows(s, tokens: tokens, hidden: hidden)
        let rows = tokens.reduce(0) { $0 + $1.dim(1) }
        if drained > 0 {
            // Only the rows still queued stay, as their own arrays.
            let rest = drained ..< rows
            s.pendingTokens = rest.isEmpty ? [] : [queuedTokens(Self.rows(tokens, rest))]
            s.pendingHidden = rest.isEmpty ? [] : [queued(Self.rows(hidden, rest))]
        } else {
            s.pendingTokens = tokens
            if let observed { hidden[hidden.count - 1] = queued(observed) }
            s.pendingHidden = hidden
        }
        s.frontier = queued(observation.hidden[0..., (n - 1) ..< n, 0...])
        s.committedInputCount += n
    }

    /// Rows `range` of queued pieces `[1, rows, ...]` along axis 1: a view when
    /// they lie in one piece, otherwise one concatenation of their slices.
    static func rows(_ pieces: [MLXArray], _ range: Range<Int>) -> MLXArray {
        var parts: [MLXArray] = []
        var start = 0
        for piece in pieces {
            let end = start + piece.dim(1)
            let lower = max(range.lowerBound, start)
            let upper = min(range.upperBound, end)
            if lower < upper { parts.append(piece[0..., (lower - start) ..< (upper - start)]) }
            start = end
        }
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
    }

    /// Write the queued pairs the next draft step would write, with its
    /// arithmetic, to the FP8 history now: before the first draft every whole
    /// `historyChunkRows` block from the history's canonical block boundary,
    /// afterwards every row one at a time. Returns the rows written. A refused
    /// write leaves its rows and every later row queued; the next draft step
    /// then refuses the request. Evaluation stays with the engine's
    /// committed-observation fence.
    private func drainTrustedRows(_ s: State, tokens: [MLXArray], hidden: [MLXArray]) -> Int {
        let chunk = s.started ? 1 : workspace.graph.historyChunkRows
        let rows = tokens.reduce(0) { $0 + $1.dim(1) }
        guard s.stagedInputCount == 0, let history = s.history,
            s.started || history.offset % chunk == 0, rows >= chunk
        else { return 0 }
        var drained = 0
        while drained + chunk <= rows {
            let block = drained ..< (drained + chunk)
            do {
                try head.appendTrustedHistory(
                    hidden: headInput(Self.rows(hidden, block)),
                    embedding: embedding(Self.rows(tokens, block)),
                    history: history, quote: workspace.graph)
            } catch { break }
            drained += chunk
        }
        return drained
    }

    /// Queued tokens as their own array, like `queued` for hidden rows.
    private func queuedTokens(_ rows: MLXArray) -> MLXArray {
        MLX.where(MLXArray(true), rows, rows)
    }

    /// Queued target hidden is held in the head dtype, the type ingestion casts
    /// it to anyway, and as its own array: a slice would keep the whole
    /// step's hidden (every row of a batched window) alive.
    private func queued(_ rows: MLXArray) -> MLXArray {
        rows.dtype == head.weightDType
            ? MLX.where(MLXArray(true), rows, rows) : rows.asType(head.weightDType)
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        do {
            return try checkedDraftStep(
                tokens: tokens, hidden: hidden, shortlist: shortlist, requestState: requestState)
        } catch {
            preconditionFailure("Super MTP draft graph was not admitted: \(error)")
        }
    }

    /// Head activations follow the head's checkpoint dtype; a promoted target
    /// hidden is rounded once at this boundary, never per layer.
    private func headInput(_ value: MLXArray) -> MLXArray {
        value.dtype == head.weightDType ? value : value.asType(head.weightDType)
    }

    private func embedding(_ tokens: MLXArray) -> MLXArray {
        headInput(target.mtpEmbedding(tokens.asType(.int32)))
    }

    func checkedDraftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) throws -> (tokens: MLXArray, hidden: MLXArray) {
        let s = try state(requestState)
        guard shortlist == nil, let history = s.history,
            s.stagedInputCount < settings.maximumDraftTokens, tokens.shape == [1, 1],
            hidden.shape == [1, 1, target.configuration.hiddenSize]
        else { throw Failure.state }
        let first = s.stagedInputCount == 0
        if first {
            s.baseOffset = history.offset
            s.savedTokens = s.pendingTokens
            s.savedHidden = s.pendingHidden
            s.savedFrontier = s.frontier
        }
        let ids: MLXArray
        let draftHidden: MLXArray
        do {
            try requireOutputHeadReady()
            try appendPendingHistory(s, history: history)
            s.pendingTokens = []
            s.pendingHidden = []
            s.frontier = nil
            let draft = try head.graphForward(
                postNormHidden: headInput(hidden), nextEmbedding: embedding(tokens),
                history: history, quote: workspace.graph)
            let logits: MLXArray
            switch outputHead {
            case .fp8(let projection): logits = try projection.graphForward(draft.hidden)
            case .dense(let linear): logits = linear(draft.hidden)
            }
            ids = argMax(logits[0..., -1, 0...], axis: -1).asType(.int32)
            draftHidden = draft.hidden
            s.validity.append(draft.valid)
        } catch {
            // Never trap the engine thread: return a well-formed proposal the
            // fence turns into a request error before the accept walk.
            s.refused = true
            ids = MLXArray([Int32(0)])
            draftHidden = MLXArray.full(
                [1, 1, target.configuration.hiddenSize], values: MLXArray(Float.nan),
                type: Float.self
            ).asType(head.weightDType)
        }
        s.stagedInputCount += 1
        if first { s.trustedOffset = history.offset }
        s.roots.append(contentsOf: [draftHidden, ids])
        return (ids, draftHidden)
    }

    /// The fastpath projection refuses invalid registered calibration when its
    /// graph is built. Check its host validation result first, so refusal
    /// precedes every history mutation of this step.
    private func requireOutputHeadReady() throws {
        guard case .fp8(let projection) = outputHead else { return }
        guard projection.graphCalibrationValid else {
            throw ModelOptCalibratedLinear.Failure.invalidMetadata
        }
    }

    /// Prompt history uses canonical fixed-width chunks, so assistant state is
    /// independent of the scheduler's prefill boundaries: observation already
    /// wrote every whole block, and the first draft step writes the queued
    /// partial block. Accepted rows after the first round keep the one-row
    /// arithmetic of a draft step.
    private func appendPendingHistory(_ s: State, history: NemotronSuperFP8KVHistory) throws {
        guard !s.pendingTokens.isEmpty else { return }
        if !s.started {
            let ids = concatenated(s.pendingTokens, axis: 1)
            let rows = concatenated(s.pendingHidden, axis: 1)
            let chunk = workspace.graph.historyChunkRows
            for start in stride(from: 0, to: ids.dim(1), by: chunk) {
                let end = min(start + chunk, ids.dim(1))
                try head.appendTrustedHistory(
                    hidden: headInput(rows[0..., start ..< end, 0...]),
                    embedding: embedding(ids[0..., start ..< end]),
                    history: history, quote: workspace.graph)
                // Bound the live priming graph to one chunk; nonblocking.
                asyncEval(history.innerState())
            }
            return
        }
        for (ids, rows) in zip(s.pendingTokens, s.pendingHidden) {
            for i in 0 ..< ids.dim(1) {
                try head.appendTrustedHistory(
                    hidden: headInput(rows[0..., i ..< (i + 1), 0...]),
                    embedding: embedding(ids[0..., i ..< (i + 1)]),
                    history: history, quote: workspace.graph)
            }
        }
    }

    public func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] {
        (try? state(requestState))?.arrays ?? []
    }

    /// Runs after this round's evaluation targets completed and before the
    /// accept walk commits anything. A refused graph or invalid routing ends
    /// the request.
    public func requestStateDidFinishEvaluation(_ requestState: any CBv2MTPRequestState) throws {
        let s = try state(requestState)
        guard !s.refused else { throw Failure.graphRefused }
        for flag in s.validity {
            guard flag.item(Bool.self) else { throw Failure.invalidRouting }
        }
        s.validity = []
    }

    public func finalizeRound(
        requestState: any CBv2MTPRequestState, confirmedInputTokens: Int,
        committedDraftTokens: MLXArray, committedTargetHidden: MLXArray
    ) {
        guard let s = try? state(requestState), let history = s.history else {
            preconditionFailure("Super MTP finalized a foreign or unconfigured state")
        }
        let n = committedDraftTokens.dim(1)
        precondition(
            s.stagedInputCount > 0
                && (1 ... (s.stagedInputCount + 1)).contains(confirmedInputTokens)
                && n <= s.stagedInputCount && n <= confirmedInputTokens - 1
                && committedTargetHidden.shape == [1, n, target.configuration.hiddenSize],
            "Super MTP finalize does not match the staged round")
        // Only the seed row used authoritative hidden. Discard every chained
        // row even on full acceptance; accepted inputs replay with the
        // verifier's hidden before the next draft.
        do { try history.rewind(to: s.trustedOffset) } catch {
            preconditionFailure("Super MTP rewind outside the staged round: \(error)")
        }
        if n > 0 {
            s.pendingTokens = [committedDraftTokens]
            s.pendingHidden = [committedTargetHidden]
        }
        s.committedInputCount += confirmedInputTokens
        s.started = true
        clearRound(s)
    }

    public func discardRound(requestState: any CBv2MTPRequestState) {
        guard let s = try? state(requestState), s.stagedInputCount > 0 else { return }
        do { try s.history?.rewind(to: s.baseOffset) } catch {
            preconditionFailure("Super MTP discard outside the staged round: \(error)")
        }
        s.pendingTokens = s.savedTokens
        s.pendingHidden = s.savedHidden
        s.frontier = s.savedFrontier
        clearRound(s)
    }

    private func clearRound(_ s: State) {
        s.stagedInputCount = 0
        s.savedTokens = []
        s.savedHidden = []
        s.savedFrontier = nil
        s.roots = []
        s.validity = []
        s.refused = false
    }

    public func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        guard let s = try? state(requestState) else { return }
        clearRound(s)
        s.history = nil
        s.pendingTokens = []
        s.pendingHidden = []
        s.frontier = nil
        s.restoredCodes = []
        s.committedInputCount = 0
        s.released = true
    }
}
