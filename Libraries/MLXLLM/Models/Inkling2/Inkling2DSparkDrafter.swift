import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// CBv2 request-stateful adapter for the Inkling 2 DSpark drafter.
///
/// The target hands every committed position's four captured features
/// (`target_layer_ids`, concatenated: [1, n, 4 * hidden]) through
/// observations, the round seed, and `finalizeRound`. They become the
/// drafter's context K/V in a private paged cache (layers 0 and 2 slide over
/// 1024 positions, layer 1 is full). The first `draftStep` of a round runs the
/// whole block forward and the base logits; each step then advances the
/// Markov head by one token. Context is append-only: drafts never enter the
/// cache, so nothing is rolled back.
public final class Inkling2DSparkDrafter: CBv2MTPRequestStatefulDrafter, @unchecked Sendable {
    public let model: Inkling2DSparkModel
    public let target: any Inkling2DSparkTarget
    public let head: Inkling2DSparkFoldedHead
    private let identity: ObjectIdentifier
    private let draftLimit: Int
    private let verificationMode: CBv2MTPVerificationMode?
    private let perRowCommit: Bool
    private let speculativeBatch: Int?
    /// Positions written per context chunk (the paged windowed write bound).
    static let contextChunk = 512
    /// Test seam: every prefix capture is refused (a lost checkpoint, never a
    /// changed request). Off in production.
    var refusesPrefixCaptureForTesting = false
    /// Certified draft selection over the target's certified head
    /// (`Inkling2DSparkCertifiedUnembed`): the full product's drafts from the
    /// int6 proxy and the rows in reach. Off unless
    /// `DARKBLOOM_INKLING2_DSPARK_CERTIFIED=1`: it saves about 0.6 ms of GPU
    /// per round but adds more host build than that to the exposed draft
    /// build, so rounds were not faster (M5 Max, 2026-11-21). Tests set
    /// `certifiedOverride`.
    @TaskLocal static var certifiedOverride: Bool? = nil
    static let certifiedFromEnvironment = environmentSwitch("DARKBLOOM_INKLING2_DSPARK_CERTIFIED")
    static var certifiedUnembedEnabled: Bool { certifiedOverride ?? certifiedFromEnvironment }

    /// Opt-in switches: "1", "true", "on" or "yes" turn one on; unset is off.
    static func environmentSwitch(_ name: String) -> Bool {
        let raw = ProcessInfo.processInfo.environment[name]?.lowercased()
        return raw.map { ["1", "true", "on", "yes"].contains($0) } ?? false
    }

    /// Submit the round's context writes, block forward and base-logit bounds
    /// before building the Markov chain, so the GPU starts them while the host
    /// builds the rest. Values are unchanged; only submission order moves.
    /// Off unless `DARKBLOOM_INKLING2_DSPARK_PIPELINED=1`: rounds were not
    /// faster (M5 Max and M3 Ultra, 2026-11-21; slower at depth 7).
    static let pipelinedSubmission = environmentSwitch("DARKBLOOM_INKLING2_DSPARK_PIPELINED")

    /// The full path's Markov chain (`markovStep` then `draftTokens`, per
    /// step) as one compiled function: the same ops at their traced dtypes
    /// (fused kernels keep every op's dtype), so the same drafts, built with
    /// one call per round instead of about 29 ops per step and run in fewer
    /// kernels. MLX traces it once per (rows, depth); a new pair pays one
    /// trace on its first round. Off unless
    /// `DARKBLOOM_INKLING2_DSPARK_COMPILED_CHAIN=1`: the draft build fell by
    /// only 0.05-0.15 ms and rounds stayed within noise (M5 Max and M3 Ultra,
    /// 2026-11-21). Tests set `compiledChainOverride`.
    @TaskLocal static var compiledChainOverride: Bool? = nil
    static let compiledChainFromEnvironment = environmentSwitch(
        "DARKBLOOM_INKLING2_DSPARK_COMPILED_CHAIN")
    static var compiledChainEnabled: Bool { compiledChainOverride ?? compiledChainFromEnvironment }

    /// [context [B, depth, 3r] fp32, seeds [B] int32, base logits [B, depth, padded] bf16]
    /// -> [drafts [B, depth] int32]. Built in `init`, not lazily: a drafter
    /// may serve several engines, and Swift's lazy initialisation is not
    /// thread-safe (MLX guards the compiled function's own trace cache).
    private let compiledChain: ([MLXArray]) -> [MLXArray]

    /// The full path's chain, op by op (the compiled function's body).
    static func markovChain(
        model: Inkling2DSparkModel, head: Inkling2DSparkFoldedHead, context: MLXArray,
        seeds: MLXArray, base: MLXArray
    ) -> MLXArray {
        var markov = model.initialMarkovState(batch: context.dim(0))
        var previous = seeds
        var drafts: [MLXArray] = []
        for step in 0 ..< context.dim(1) {
            let next = model.markovStep(
                state: markov, previous: previous, context: context[0..., step, 0...])
            markov = next.state
            previous = model.draftTokens(
                base: base[0..., step, 0...], latent: next.latent, head: head)
            drafts.append(previous)
        }
        return stacked(drafts, axis: 1)
    }

    /// Benchmark seam for round anatomy: drafts are constant zeros with no
    /// block forward or Markov chain (the round's own bookkeeping still runs),
    /// so a round costs the target's verify window alone. Off in production.
    public var benchmarkNullDrafts = false

    /// - Parameters:
    ///   - targetIdentity: must equal the bound steppable model's
    ///     `mtpTargetIdentity`; defaults to the target object.
    ///   - requiredVerificationMode / commitsAcceptedPrefixPerRow: the
    ///     target's verification guarantees, passed through to the engine.
    public init(
        model: Inkling2DSparkModel, target: any Inkling2DSparkTarget,
        head: Inkling2DSparkFoldedHead? = nil, targetIdentity: ObjectIdentifier? = nil,
        maximumDraftTokens: Int? = nil,
        requiredVerificationMode: CBv2MTPVerificationMode? = nil,
        commitsAcceptedPrefixPerRow: Bool = false, maximumSpeculativeBatch: Int? = nil
    ) {
        self.model = model
        self.target = target
        var folded = head ?? model.foldHead(unembedding: target.dsparkUnembedWeight)
        if folded.certified == nil, Self.certifiedUnembedEnabled,
            let certified = target.dsparkCertifiedHead(),
            certified.rows == folded.unembedding.dim(0),
            certified.columns == folded.unembedding.dim(1)
        {
            folded.certified = .init(
                head: certified, cap: model.configuration.finalLogitSoftcapping,
                vocabularySize: model.configuration.vocabularySize, dtype: model.dtype)
        }
        self.head = folded
        self.compiledChain = compile(inputs: [], outputs: [], shapeless: false) {
            [model, folded] args in
            [
                Self.markovChain(
                    model: model, head: folded, context: args[0], seeds: args[1], base: args[2])
            ]
        }
        self.identity = targetIdentity ?? ObjectIdentifier(target)
        let blockDrafts = model.configuration.draftTokens
        self.draftLimit = min(max(maximumDraftTokens ?? blockDrafts, 1), blockDrafts)
        self.verificationMode = requiredVerificationMode
        self.perRowCommit = commitsAcceptedPrefixPerRow
        self.speculativeBatch = maximumSpeculativeBatch
    }

    var configuration: Inkling2DSparkConfiguration { model.configuration }

    // MARK: Policy

    public var mtpTargetIdentity: ObjectIdentifier? { identity }
    public var requiredVerificationMode: CBv2MTPVerificationMode? { verificationMode }
    public var maximumDraftTokens: Int? { draftLimit }
    public var maximumSpeculativeBatch: Int? { speculativeBatch }
    /// Acceptance compares drafts with the target's own (pre-sampled) tokens,
    /// so it is exact at any temperature; drafts stay greedy.
    public var supportsTargetPrefixAcceptance: Bool { true }
    public var commitsAcceptedPrefixPerRow: Bool { perRowCommit }
    public var supportsOrdinaryDecodeHistory: Bool { true }
    public var supportsDepthZeroChaining: Bool { true }

    /// Context K/V of all three layers (bf16), plus queued feature rows held
    /// until the next round writes them.
    public var requestStateBytesPerToken: Int {
        let c = configuration
        let kv = 2 * c.numKeyValueHeads * c.headDim * 2 * c.numHiddenLayers
        return kv + c.featureWidth * 2
    }
    public var requestStateTokenGranularity: Int { CBv2PagedDefaults.pageSize }
    public var requestStateTokenAllocationPadding: Int { configuration.blockSize }

    // MARK: State

    final class PagedState {
        let backend: PagedKVBackend
        let rows: [PagedSequenceKV]
        let caches: [PagedLayerCache]
        let maximumSequenceLength: Int
        private var released = false

        init(configuration c: Inkling2DSparkConfiguration, maximumSequenceLength: Int) throws {
            self.maximumSequenceLength = maximumSequenceLength
            let kinds = (0 ..< c.numHiddenLayers).map { index in
                CBv2LayerKind(
                    attention: c.isLocal(layer: index)
                        ? .slidingWindow(c.slidingWindowSize) : .full,
                    headDim: c.headDim, kvHeads: c.numKeyValueHeads,
                    queryHeads: c.numAttentionHeads, modelLayerIndex: index)
            }
            let pageTokens = CBv2PagedDefaults.pageSize
            let pageBytes = 2 * c.numKeyValueHeads * c.headDim * pageTokens * 2
            let fullPages = (maximumSequenceLength + pageTokens - 1) / pageTokens
            let ringPages =
                (c.slidingWindowSize + CBv2PagedSpeculation.maxSpeculativeSpan
                    + Inkling2DSparkDrafter.contextChunk + pageTokens - 1) / pageTokens
            let pages = kinds.reduce(0) { total, kind in
                if case .full = kind.attention { return total + fullPages + 2 }
                return total + min(fullPages, ringPages) + 2
            }
            let capacity = pageBytes * pages + 4 * 1024 * 1024
            backend = try PagedKVBackend(
                layerKinds: kinds,
                config: .init(
                    capacityBytes: capacity, dtype: .bfloat16,
                    maxPrefillChunk: Inkling2DSparkDrafter.contextChunk,
                    nominalMaxSequenceLength: maximumSequenceLength,
                    segmentSizeBytes: min(capacity, 4 * 1024 * 1024),
                    layerDTypes: kinds.map { _ in .bfloat16 }))
            let state = try backend.makeSequenceState(
                layerKinds: kinds, promptLength: 0, maxLength: maximumSequenceLength)
            rows = state.map { $0 as! PagedSequenceKV }
            caches = backend.makeLayerCaches()
            for (cache, row) in zip(caches, rows) { cache.setRows([row]) }
        }

        var written: Int { rows[0].absoluteOffset }

        func release() {
            guard !released else { return }
            for cache in caches { cache.setRows([]) }
            backend.release(rows.map { $0 as CBv2SequenceKV? })
            released = true
        }

        deinit { release() }
    }

    final class State: CBv2MTPRequestState {
        let owner: ObjectIdentifier
        var paged: PagedState?
        var committedInputCount = 0
        var stagedInputCount = 0
        var released = false
        /// Feature rows [1, n, featureWidth] queued for the context, in order.
        var pending: [MLXArray] = []
        /// The newest observed row; a round seed carries the same position.
        var frontier: MLXArray?
        var plan: Inkling2DSparkMarkovPlan?
        var markovState: MLXArray?
        var blockHidden: MLXArray?
        var roots: [MLXArray] = []
        /// Set by `discardRound`: the cache already holds the seed's position.
        var seedWritten = false
        /// A prefix checkpoint's context, written at the first flush.
        var restored: Inkling2DSparkPrefixCheckpoint?
        /// Prefix-capture context: the request's prompt, installed once before
        /// observation; observed prompt inputs are its identity evidence.
        var prefixContext: [Int32]?
        var prefixMayInstall = true
        /// True until the first round: only prompt history has been observed.
        var promptOnly = true
        var observedTokens: [MLXArray] = []

        init(owner: ObjectIdentifier) { self.owner = owner }

        var pendingRows: Int { pending.reduce(0) { $0 + $1.dim(1) } }
        var hasPendingPrefillForCostAccounting: Bool {
            paged == nil || pendingRows > 8
        }
        var arrays: [MLXArray] {
            (paged?.caches.flatMap { $0.innerState() } ?? []) + pending
                + [frontier, markovState, blockHidden].compactMap { $0 } + roots
                + (restored?.evaluationTargets ?? [])
        }
    }

    private final class Prepared: CBv2MTPPreparedCapture {}

    private func state(_ value: any CBv2MTPRequestState) -> State {
        guard let s = value as? State, s.owner == ObjectIdentifier(self), !s.released else {
            preconditionFailure("Inkling2 DSpark: foreign or released request state")
        }
        return s
    }

    public func makeRequestState() -> any CBv2MTPRequestState {
        State(owner: ObjectIdentifier(self))
    }

    public func configureRequestState(
        _ requestState: any CBv2MTPRequestState, maximumSequenceLength: Int
    ) throws {
        let s = state(requestState)
        guard maximumSequenceLength > 0 else {
            throw CBv2KVError.backendIneligible(
                reason: "Inkling2 DSpark requires a positive sequence bound")
        }
        let bound = maximumSequenceLength + requestStateTokenAllocationPadding
        if let paged = s.paged {
            guard paged.maximumSequenceLength >= bound else {
                throw CBv2KVError.backendIneligible(
                    reason: "Inkling2 DSpark state bound \(paged.maximumSequenceLength) < \(bound)")
            }
            return
        }
        s.paged = try PagedState(configuration: configuration, maximumSequenceLength: bound)
    }

    public func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture { Prepared() }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure("Inkling2 DSpark requires request-owned state")
    }

    public func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState
    ) {
        let s = state(requestState)
        precondition(s.stagedInputCount == 0)
        let n = observation.tokens.dim(1)
        precondition(
            observation.tokens.shape == [1, n]
                && observation.hidden.shape == [1, n, configuration.featureWidth],
            "Inkling2 DSpark observation \(observation.hidden.shape) is not [1, n, features]")
        guard n > 0 else { return }
        s.committedInputCount += n
        if s.promptOnly {
            // Identity evidence matters only while a capture strictly inside
            // the installed prompt is still possible (capturePrefixCheckpoint);
            // past it, or with no capture context, the list would only grow
            // (an ordinary-decode stretch before the first round).
            if let context = s.prefixContext, s.committedInputCount < context.count {
                s.observedTokens.append(observation.tokens)
            } else {
                s.observedTokens = []
            }
        }
        var rows = observation.hidden
        if s.seedWritten && s.frontier == nil {
            // A discarded round already wrote this carry row.
            s.seedWritten = false
            guard n > 1 else { return }
            rows = rows[0..., 1..., 0...]
        }
        let count = rows.dim(1)
        if let frontier = s.frontier { s.pending.append(frontier) }
        if count > 1 { s.pending.append(rows[0..., 0 ..< (count - 1), 0...]) }
        s.frontier = rows[0..., (count - 1) ..< count, 0...]
    }

    /// Writes every queued feature row into the context cache.
    private func flushContext(_ s: State) {
        guard let paged = s.paged else { return }
        if let restored = s.restored {
            writeRestored(restored, into: s)
            s.restored = nil
        }
        guard !s.pending.isEmpty else { return }
        let rows = concatenated(s.pending, axis: 1)
        s.pending = []
        let total = rows.dim(1)
        for start in stride(from: 0, to: total, by: Self.contextChunk) {
            let end = min(start + Self.contextChunk, total)
            let features = rows[0, start ..< end, 0...]
            let position = paged.written
            for (index, cache) in paged.caches.enumerated() {
                let (keys, values) = model.contextKV(
                    layer: index, features: features, start: position)
                s.roots += cache.appendTrusted(keys: keys, values: values)
            }
        }
    }

    /// Writes the queued context and the seed row (the position before the
    /// anchor), and returns the anchor position and each layer's attendable
    /// context K/V (sliding layers hold only their window).
    private func beginRound(_ s: State, seed hidden: MLXArray) -> (
        position: Int, contexts: [(keys: MLXArray, values: MLXArray)]
    ) {
        guard let paged = s.paged else {
            preconditionFailure("Inkling2 DSpark state requires an admitted sequence bound")
        }
        precondition(
            hidden.shape == [1, 1, configuration.featureWidth],
            "Inkling2 DSpark seed hidden \(hidden.shape) is not [1, 1, features]")
        s.promptOnly = false
        s.prefixMayInstall = false
        s.observedTokens = []
        // The seed row replaces the observed frontier (same position).
        s.frontier = nil
        if s.seedWritten {
            s.seedWritten = false
        } else {
            s.pending.append(hidden)
        }
        flushContext(s)
        let contexts = paged.rows.map { row -> (keys: MLXArray, values: MLXArray) in
            let snapshot = row.snapshot()
            return (snapshot.keys, snapshot.values)
        }
        return (paged.written, contexts)
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        let s = state(requestState)
        let c = configuration
        precondition(s.stagedInputCount < draftLimit && tokens.shape == [1, 1])
        if s.stagedInputCount == 0 {
            let (position, contexts) = beginRound(s, seed: hidden)
            let anchors = target.dsparkNormedEmbedding(tokens).reshaped(1, c.hiddenSize)
            let block = model.blockHidden(
                anchors: anchors, positions: [position],
                contexts: contexts.map { (keys: $0.keys, values: $0.values, mask: nil) })
            s.blockHidden = block
            s.plan = model.markovPlan(
                rows: block[0..., 1..., 0...],
                markovFeatures: model.markovFeatures(hidden.reshaped(1, c.featureWidth)),
                head: head)
            s.markovState = model.initialMarkovState(batch: 1)
        }
        let step = s.stagedInputCount
        let plan = s.plan!
        let next = model.markovStep(
            state: s.markovState!, previous: tokens.reshaped(1).asType(.int32),
            context: plan.context[0..., step, 0...])
        s.markovState = next.state
        let ids = model.draftTokens(plan: plan, step: step, latent: next.latent, head: head)
        s.stagedInputCount += 1
        s.roots += [ids, next.state]
        return (ids, s.blockHidden![0..., (step + 1) ..< (step + 2), 0...])
    }

    /// Drafts `depth` tokens for every row at once: one block forward over
    /// all rows (contexts padded and masked) and one Markov chain whose base
    /// logits read the unembedding once. Each state stages `depth` inputs, as
    /// `depth` calls of `draftStep` would. Returns [rows, depth] int32.
    public func draftBlock(
        tokens: [MLXArray], hidden: [MLXArray], requestStates: [any CBv2MTPRequestState],
        depth: Int
    ) -> MLXArray {
        let c = configuration
        let batch = requestStates.count
        precondition(
            batch > 0 && tokens.count == batch && hidden.count == batch
                && (1 ... draftLimit).contains(depth))
        let states = requestStates.map { state($0) }
        if benchmarkNullDrafts {
            // No context writes, block forward or Markov chain: drafter cost zero.
            let result = MLXArray.zeros([batch, depth], dtype: .int32)
            for s in states {
                precondition(s.stagedInputCount == 0)
                s.stagedInputCount = depth
                s.roots.append(result)
            }
            return result
        }
        var positions: [Int] = []
        var rowContexts: [[(keys: MLXArray, values: MLXArray)]] = []
        for (index, s) in states.enumerated() {
            precondition(s.stagedInputCount == 0 && tokens[index].shape == [1, 1])
            let round = beginRound(s, seed: hidden[index])
            positions.append(round.position)
            rowContexts.append(round.contexts)
        }
        let contexts = (0 ..< c.numHiddenLayers).map { layer in
            Self.padded(rowContexts.map { $0[layer] }, block: c.blockSize)
        }
        let seeds = concatenated(tokens, axis: 0).asType(.int32)
        let anchors = target.dsparkNormedEmbedding(seeds).reshaped(batch, c.hiddenSize)
        let block = model.blockHidden(anchors: anchors, positions: positions, contexts: contexts)
        let plan = model.markovPlan(
            rows: block[0..., 1 ... depth, 0...],
            markovFeatures: model.markovFeatures(
                concatenated(hidden, axis: 0).reshaped(batch, c.featureWidth)),
            head: head)
        if Self.pipelinedSubmission {
            // The engine's round submission joins these; nothing waits here.
            asyncEval(
                [block, plan.context] + states.flatMap(\.arrays)
                    + [plan.baseLogits, plan.lower, plan.upper].compactMap { $0 })
        }
        let result: MLXArray
        if Self.compiledChainEnabled, let base = plan.baseLogits {
            result = compiledChain([plan.context, seeds.reshaped(batch), base])[0]
        } else {
            var markov = model.initialMarkovState(batch: batch)
            var previous = seeds.reshaped(batch)
            var drafts: [MLXArray] = []
            for step in 0 ..< depth {
                let next = model.markovStep(
                    state: markov, previous: previous, context: plan.context[0..., step, 0...])
                markov = next.state
                previous = model.draftTokens(
                    plan: plan, step: step, latent: next.latent, head: head)
                drafts.append(previous)
            }
            result = stacked(drafts, axis: 1)
        }
        for s in states {
            s.stagedInputCount = depth
            s.roots.append(result)
        }
        return result
    }

    /// Right-pads per-row context K/V to one length and masks the padding.
    static func padded(_ rows: [(keys: MLXArray, values: MLXArray)], block: Int) -> (
        keys: MLXArray, values: MLXArray, mask: MLXArray?
    ) {
        let lengths = rows.map { $0.keys.dim(2) }
        let longest = lengths.max() ?? 0
        guard rows.count > 1 else {
            return (rows[0].keys, rows[0].values, nil)
        }
        func pad(_ x: MLXArray, _ length: Int) -> MLXArray {
            guard length < longest else { return x }
            var shape = x.shape
            shape[2] = longest - length
            return concatenated([x, MLXArray.zeros(shape, dtype: x.dtype)], axis: 2)
        }
        let keys = concatenated(
            zip(rows, lengths).map { pad($0.0.keys, $0.1) }, axis: 0)
        let values = concatenated(
            zip(rows, lengths).map { pad($0.0.values, $0.1) }, axis: 0)
        guard lengths.contains(where: { $0 != longest }) else { return (keys, values, nil) }
        let columns = MLXArray(Int32(0) ..< Int32(longest + block)).reshaped(1, -1)
        let limits = MLXArray(lengths.map { Int32($0) }).reshaped(-1, 1)
        let visible = (columns .< limits) .|| (columns .>= Int32(longest))
        return (keys, values, visible.reshaped(rows.count, 1, 1, longest + block))
    }

    public func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] {
        state(requestState).arrays
    }

    public func finalizeRound(
        requestState: any CBv2MTPRequestState, confirmedInputTokens: Int,
        committedDraftTokens: MLXArray, committedTargetHidden: MLXArray
    ) {
        let s = state(requestState)
        precondition(
            s.stagedInputCount > 0
                && (1 ... (s.stagedInputCount + 1)).contains(confirmedInputTokens))
        let n = committedDraftTokens.dim(1)
        precondition(
            n <= s.stagedInputCount && n <= confirmedInputTokens - 1
                && committedTargetHidden.shape == [1, n, configuration.featureWidth])
        // Rows 0 ..< n are the anchor and accepted drafts; the next seed
        // carries row n.
        if n > 0 { s.pending.append(committedTargetHidden) }
        s.committedInputCount += confirmedInputTokens
        clearRound(s)
    }

    /// Context writes are committed target truth and stay; the seed row they
    /// include must not be written twice if the carry is replayed.
    public func discardRound(requestState: any CBv2MTPRequestState) {
        let s = state(requestState)
        guard s.stagedInputCount > 0 else { return }
        s.seedWritten = true
        clearRound(s)
    }

    private func clearRound(_ s: State) {
        s.stagedInputCount = 0
        s.plan = nil
        s.markovState = nil
        s.blockHidden = nil
        s.roots = []
    }

    public func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        guard let s = requestState as? State, s.owner == ObjectIdentifier(self), !s.released
        else { return }
        clearRound(s)
        s.paged?.release()
        s.paged = nil
        s.pending = []
        s.frontier = nil
        s.restored = nil
        s.observedTokens = []
        s.prefixContext = nil
        s.committedInputCount = 0
        s.released = true
    }
}

extension Inkling2DSparkDrafter: CBv2MTPBatchedBlockDrafter {
    /// The engine configuration the drafter ships with: on, up to the
    /// checkpoint's full block (7 drafts), up to eight speculating rows, one
    /// verify window per round. `adaptive` lets the engine's marginal policy
    /// pick each round's depth from the request's measured acceptance and the
    /// measured cost of each window depth (down to plain decode where drafting
    /// does not pay); otherwise every round drafts `maxDraftTokens`. The exactness of
    /// the window is the target's claim (`requiredVerificationMode`).
    public static func defaultMTPConfig(
        maxDraftTokens: Int = 7, adaptive: Bool = true, maxSpeculativeRows: Int = 8
    ) -> CBv2MTPConfig {
        CBv2MTPConfig(
            enabled: true, maxDraftTokens: maxDraftTokens,
            maxSpeculativeBatch: maxSpeculativeRows,
            fixedDraftTokens: adaptive ? nil : maxDraftTokens, verificationMode: .rectangular,
            maxAutomaticRectangularTokens: maxSpeculativeRows * (maxDraftTokens + 1),
            costCurveDepthSelection: true,
            speculativeBatchLimit: maxSpeculativeRows)
    }
}
