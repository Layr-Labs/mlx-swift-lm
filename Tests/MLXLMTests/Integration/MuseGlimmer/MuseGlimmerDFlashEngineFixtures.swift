// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN

@testable import MLXLLM
@testable import MLXLMCommon

/// A tiny attention-only target shaped like the Muse Glimmer text tower for
/// the drafter: sliding and full layers, the residual stream after chosen
/// layers as the MTP hidden, raw embeddings and a bare head. Random weights.
final class MuseGlimmerDFlashTinyTarget: Module, CBv2MTPPrefillSteppableModel,
    CBv2MTPPolicyTopTwoProviding, CBv2HistoricalAttentionCheckpointProviding,
    CBv2MultimodalSteppableModel, MuseGlimmerDFlashTarget
{
    static let window = 17
    static let tapLayers = [0, 2]

    let config: TinyTestModelConfig
    let embed: Embedding
    let blocks: [TinyBlock]
    let finalNorm: RMSNorm
    let lmHead: Linear
    /// Rows of every forward `[batch, length]`, in call order.
    private(set) var forwardShapes: [[Int]] = []
    /// Hands the MTP seam hidden rows one tap wider than declared.
    var widensHidden = false

    let layerKinds: [CBv2LayerKind]
    let cbv2SupportsHistoricalAttentionCheckpoint = true

    init(seed: UInt64 = 0xD1F1) {
        var config = TinyTestModelConfig()
        config.hiddenSize = 64
        config.queryHeads = 2
        config.kvHeads = 1
        config.headDim = 64
        config.vocabSize = 128
        config.windowSize = Self.window
        self.config = config
        MLXRandom.seed(seed)
        embed = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        let sliding = [true, false, true, false]
        blocks = sliding.map { TinyBlock(config: config, isSliding: $0) }
        finalNorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        lmHead = Linear(config.hiddenSize, config.vocabSize, bias: false)
        layerKinds = sliding.map {
            CBv2LayerKind(
                attention: $0 ? .slidingWindow(Self.window) : .full,
                headDim: config.headDim, kvHeads: config.kvHeads, queryHeads: config.queryHeads)
        }
        super.init()
        eval(self)
    }

    /// Final-normed hidden and the tapped residual streams, concatenated.
    private func trunk(
        _ tokens: MLXArray, _ caches: [CBv2AttendingLayerCache], embeddings: MLXArray? = nil
    ) -> (normed: MLXArray, taps: MLXArray) {
        precondition(caches.count == blocks.count)
        forwardShapes.append(tokens.shape)
        var hidden = embeddings ?? embed(tokens)
        var taps: [MLXArray] = []
        for (index, block) in blocks.enumerated() {
            hidden = block.forwardV2(hidden, cache: caches[index])
            if Self.tapLayers.contains(index) { taps.append(hidden) }
        }
        if widensHidden { taps.append(hidden) }
        return (finalNorm(hidden), concatenated(taps, axis: -1))
    }

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        lmHead(trunk(tokens, caches).normed)
    }

    // MARK: Causal image spans

    var causalPositionRequirement: CBv2CausalPositionRequirement { .scalarCacheOffset }
    func supportsMultimodalPrefill(attention: CBv2MultimodalAttention) -> Bool {
        attention == .causal
    }
    func embedPromptTokens(_ tokens: MLXArray) -> MLXArray { embed(tokens) }
    func forward(
        tokens: MLXArray, inputEmbeddings: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> MLXArray {
        lmHead(trunk(tokens, caches, embeddings: inputEmbeddings).normed)
    }

    // MARK: MTP target

    var mtpCaptureLayers: CBv2MTPCaptureLayers? { nil }
    var supportsRequestStatefulMTP: Bool { true }
    var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(self) }

    func forwardWithHidden(tokens: MLXArray, caches: [CBv2AttendingLayerCache])
        -> (logits: MLXArray, lastHidden: MLXArray)
    {
        let output = trunk(tokens, caches)
        return (lmHead(output.normed), output.taps)
    }

    func forwardWithHiddenForPrefill(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache], requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        let output = trunk(tokens, caches)
        let last = lmHead(output.normed[0..., -1, 0...])
        switch requirement {
        case .evaluationOnly: return (last[0..., 0 ..< 1], output.taps)
        case .lastPositionLogits: return (last, output.taps)
        }
    }

    func cbv2MTPTopTwo(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
        cbv2TopTwoRows(logits)
    }

    // MARK: Drafter target

    var dflashTargetIdentity: ObjectIdentifier { ObjectIdentifier(self) }
    var dflashHiddenSize: Int { config.hiddenSize }
    var dflashVocabularySize: Int { config.vocabSize }
    var dflashContextWidth: Int { Self.tapLayers.count * config.hiddenSize }
    var dflashTappedLayerIDs: [Int] { Self.tapLayers }
    func dflashTokenEmbeddings(_ tokens: MLXArray) -> MLXArray { embed(tokens) }
    func dflashHeadProduct(_ hidden: MLXArray) -> MLXArray { lmHead(hidden) }
}

/// Proposal override for engine tests. The real drafter runs every step and
/// keeps its own state; only the proposed token is replaced, from the known
/// continuation of the request, so a test decides where the target accepts.
final class MuseGlimmerDFlashScriptedDrafter: CBv2MTPAdvisoryHistoryDrafter,
    CBv2MTPCoupledProposalDrafter, CBv2MTPBoundedAllocationProviding
{
    enum Mode: Equatable, Sendable {
        /// The drafter's own proposals.
        case native
        /// The target's own continuation: every draft is accepted.
        case right
        /// Right, except draft index `position` of every round.
        case wrong(at: Int)
    }

    let native: MuseGlimmerDFlashDrafter
    let mode: Mode
    let vocabulary: Int
    /// Full token sequences (prompt and continuation) of the expected requests.
    let scripts: [[Int]]
    private var bound: [ObjectIdentifier: [Int]] = [:]
    private var anchors: [ObjectIdentifier: Int] = [:]
    private(set) var states: [MuseGlimmerDFlashDrafter.State] = []
    /// Anchor position of every block drafted.
    private(set) var blockAnchors: [Int] = []
    /// Context rows held when each block was drafted.
    private(set) var blockContextRows: [Int] = []
    /// First absolute position of every observation, when the engine gave one.
    private(set) var observedPositions: [Int] = []
    /// Called on the engine thread when a block is drafted, with the count so far.
    var onBlock: ((Int) -> Void)?
    /// Every proposal selection the engine installed, in order.
    private(set) var selections: [CBv2MTPProposalSelection?] = []
    /// Every proposal returned, per drafted block (lazy until read).
    private(set) var proposals: [[MLXArray]] = []
    /// Replaces a non-nil selection before the drafter sees it.
    var selectionOverride: ((CBv2MTPProposalSelection) -> CBv2MTPProposalSelection)?

    init(
        _ native: MuseGlimmerDFlashDrafter, mode: Mode, scripts: [[Int]] = [], vocabulary: Int
    ) {
        self.native = native
        self.mode = mode
        self.scripts = scripts
        self.vocabulary = vocabulary
    }

    var mtpTargetIdentity: ObjectIdentifier? { native.mtpTargetIdentity }
    var requiredVerificationMode: CBv2MTPVerificationMode? { native.requiredVerificationMode }
    var maximumDraftTokens: Int? { native.maximumDraftTokens }
    var maximumSpeculativeBatch: Int? { native.maximumSpeculativeBatch }
    var supportsTargetPrefixAcceptance: Bool { native.supportsTargetPrefixAcceptance }
    var couplesSampledProposals: Bool { native.couplesSampledProposals }
    func setProposalSelection(
        _ selection: CBv2MTPProposalSelection?, requestState: any CBv2MTPRequestState
    ) {
        selections.append(selection)
        native.setProposalSelection(
            selection.map { selectionOverride?($0) ?? $0 }, requestState: requestState)
    }
    var commitsAcceptedPrefixPerRow: Bool { native.commitsAcceptedPrefixPerRow }
    var batchesVerifyColumnsAsDecodeRows: Bool { native.batchesVerifyColumnsAsDecodeRows }
    /// Lets a test drive the driver's switches under typical acceptance.
    var waivesExactAcceptance = false
    var requiresExactAcceptance: Bool {
        native.requiresExactAcceptance && !waivesExactAcceptance
    }
    var requiresCommittedObservationFence: Bool { native.requiresCommittedObservationFence }
    var requestStateBytesPerToken: Int { native.requestStateBytesPerToken }
    /// False makes the engine treat this drafter like any other
    /// request-stateful one.
    var requestStateIsAdvisory = true
    func boundedRequestAllocation(limits: CBv2MTPAllocationLimits)
        -> CBv2MTPBoundedAllocationSpec?
    {
        native.boundedRequestAllocation(limits: limits)
    }

    func makeRequestState() -> any CBv2MTPRequestState {
        let state = native.makeRequestState() as! MuseGlimmerDFlashDrafter.State
        states.append(state)
        return state
    }

    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture { native.prepare(rows: rows) }
    func draftStep(tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture)
        -> (tokens: MLXArray, hidden: MLXArray)
    {
        native.draftStep(tokens: tokens, hidden: hidden, prepared: prepared)
    }

    /// Bind a new state to the script whose tokens at `position` are these.
    private func bind(_ state: any CBv2MTPRequestState, tokens: MLXArray, at position: Int) {
        let id = ObjectIdentifier(state)
        guard bound[id] == nil else { return }
        let observed = tokens.reshaped([-1]).asArray(Int32.self).map(Int.init)
        bound[id] = scripts.first {
            $0.count >= position + observed.count
                && Array($0[position ..< (position + observed.count)]) == observed
        }
    }

    func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation, requestState: any CBv2MTPRequestState
    ) {
        let state = requestState as! MuseGlimmerDFlashDrafter.State
        bind(state, tokens: observation.tokens, at: state.context.nextPosition)
        native.observeCommittedTarget(observation, requestState: requestState)
    }

    func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation, startingAt position: Int,
        requestState: any CBv2MTPRequestState
    ) {
        observedPositions.append(position)
        // A carry is paired with the token after it; only real rows bind.
        if observation.tokens.dim(1) > 1 || bound[ObjectIdentifier(requestState)] == nil {
            bind(requestState, tokens: observation.tokens, at: position)
        }
        native.observeCommittedTarget(
            observation, startingAt: position, requestState: requestState)
    }

    func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        let state = requestState as! MuseGlimmerDFlashDrafter.State
        let id = ObjectIdentifier(state)
        let index = state.stagedInputCount
        let result = native.draftStep(
            tokens: tokens, hidden: hidden, shortlist: shortlist, requestState: requestState)
        if index == 0 {
            anchors[id] = state.context.nextPosition
            blockAnchors.append(state.context.nextPosition)
            blockContextRows.append(state.context.count)
            proposals.append([])
            onBlock?(blockAnchors.count)
        }
        if !proposals.isEmpty { proposals[proposals.count - 1].append(result.tokens) }
        guard mode != .native, let script = bound[id], let anchor = anchors[id] else {
            return result
        }
        let position = anchor + 1 + index
        var token = position < script.count ? script[position] : 0
        if case .wrong(let at) = mode, at == index { token = (token + 1) % vocabulary }
        return (MLXArray([Int32(token)]), result.hidden)
    }

    func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] {
        native.evaluationTargets(for: requestState)
    }

    func requestStateDidFinishEvaluation(_ requestState: any CBv2MTPRequestState) throws {
        try native.requestStateDidFinishEvaluation(requestState)
    }

    func finalizeRound(
        requestState: any CBv2MTPRequestState, confirmedInputTokens: Int,
        committedDraftTokens: MLXArray, committedTargetHidden: MLXArray
    ) {
        native.finalizeRound(
            requestState: requestState, confirmedInputTokens: confirmedInputTokens,
            committedDraftTokens: committedDraftTokens,
            committedTargetHidden: committedTargetHidden)
    }

    func discardRound(requestState: any CBv2MTPRequestState) {
        native.discardRound(requestState: requestState)
    }

    func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        native.releaseRequestState(requestState)
    }
}

enum MuseGlimmerDFlashEngineFixture {
    static let chunk = max(32, CBv2AttentionV1.queryBlockSize)

    static func drafterConfiguration() throws -> MuseGlimmerDFlashConfiguration {
        try MuseGlimmerDFlashConfiguration(
            hiddenSize: 64, intermediateSize: 128, hiddenLayers: 2, attentionHeads: 4,
            kvHeads: 2, headDimensions: 16, blockSize: 16, maskTokenID: 127,
            targetLayerIDs: MuseGlimmerDFlashTinyTarget.tapLayers, slidingWindow: 32,
            rmsNormEps: 1e-5, ropeTheta: 500_000)
    }

    /// The real drafter over a random tiny draft model.
    static func drafter(
        target: MuseGlimmerDFlashTinyTarget,
        settings: MuseGlimmerDFlashDrafter.Settings = .init(maximumSpeculativeRows: 2)
    ) throws -> MuseGlimmerDFlashDrafter {
        let model = try MuseGlimmerDFlashFixture.model(
            try drafterConfiguration(), seed: 21, dtype: .float32)
        return try MuseGlimmerDFlashDrafter(target: target, model: model, settings: settings)
    }

    /// A paged engine over the tiny target. `store` turns complete prefix
    /// checkpoints on.
    static func engine(
        target: MuseGlimmerDFlashTinyTarget, drafter: (any CBv2MTPDrafter)?, depth: Int = 7,
        verification: CBv2MTPVerificationMode = .serialTarget,
        store: CompleteCheckpointFixtureStore? = nil, acceptance: CBv2MTPAcceptance = .exact,
        sampler: CBv2StepSampler = CBv2DefaultSampler()
    ) throws -> (EngineV2, PagedKVBackend) {
        let kinds = target.layerKinds
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 256 << 20, maxPrefillChunk: chunk, segmentSizeBytes: 64 << 10,
                layerDTypes: Array(repeating: .float32, count: kinds.count)))
        let engine = EngineV2(
            model: target, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            sampler: sampler,
            schedulerConfig: .init(
                maxConcurrentRequests: 2, maxBatchedTokensPerStep: 2 * chunk,
                prefillChunkSize: chunk, maxWaiting: 4, enablePrefixCache: store != nil),
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store,
            mtpDrafter: drafter,
            mtpConfig: CBv2MTPConfig(
                enabled: drafter != nil, maxDraftTokens: depth, maxSpeculativeBatch: 2,
                fixedDraftTokens: depth, verificationMode: verification,
                maxAutomaticRectangularTokens: 16, acceptance: acceptance))
        if store != nil {
            engine.loopForTesting.onEngineQueueSync {
                engine.completeCheckpointCapture?.historicalCheckpointStrideTokens = chunk
            }
        }
        return (engine, backend)
    }

    static func prompt(_ length: Int, salt: Int) -> [Int] {
        (0 ..< length).map { 1 + ($0 * 7 + salt * 13 + ($0 * $0) % 5) % 120 }
    }

    static func request(
        _ id: UInt64, prompt: [Int], maxTokens: Int, sampling: CBv2SamplingParams
    ) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt, sampling: sampling, maxTokens: maxTokens)
    }

    static let greedy = CBv2SamplingParams(temperature: 0)
    static let sampled = CBv2SamplingParams(temperature: 1, topP: 0.95, topK: 64, seed: 20_261_113)
}
