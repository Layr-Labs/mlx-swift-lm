// Copyright © 2026 Eigen Labs.

import MLX
import MLXLMCommon

/// Retained, fully charged input storage. A lazy copy must retain its source
/// allocation through didFinishEvaluation; nbytes is not a residency receipt.
public protocol GLM5NextMTPArrayLease: AnyObject {
    var arrays: [MLXArray] { get }
    var materializedBytes: Int { get }
    var hasUnmeasuredResidency: Bool { get }
}

public protocol GLM5NextMTPExecutionCheckpoint: AnyObject {
    var inputCount: Int { get }
    var materializedBytes: Int { get }
    var evaluationTargets: [MLXArray] { get }
}

/// Private candidate, not published storage. prepare/commit/discard MUST NOT
/// evaluate or synchronize. didFinishEvaluation runs only after Common's fault-
/// checked fence and must validate every retained native reader/ownership lease.
/// commit must reject stale generations, including trim/reset-to-zero ABA.
public protocol GLM5NextMTPPreparedExecution: AnyObject {
    var output: GLM5NextMTPOutput { get }
    var inputCount: Int { get }
    var evaluationTargets: [MLXArray] { get }
    func didFinishEvaluation() throws
    func commit() throws
    func discard() throws
}

/// One real request allocation owner. All methods except didFinishEvaluation
/// and close are graph/metadata-only. No array-cache or unpriced fallback exists.
/// The current synchronous GLM5NextPagedSparseState is NOT an implementation of
/// this contract: stage/indexView/gatherLatents must first gain a graph-only seam.
public protocol GLM5NextMTPExecutionOwner: AnyObject {
    var targetIdentity: ObjectIdentifier { get }
    var predictorIdentity: ObjectIdentifier { get }
    var maximumInputs: Int { get }
    var committedInputCount: Int { get }
    var materializedBytes: Int { get }
    /// True when this exact lease is already included in materializedBytes.
    /// This is accounting membership, not proof that its storage is measured.
    /// Borrowed prefix leases may belong to a different execution owner.
    func accountsForResidency(of lease: any GLM5NextMTPArrayLease) -> Bool
    var hasUnmeasuredResidency: Bool { get }
    var evaluationTargets: [MLXArray] { get }
    func hold(_ arrays: [MLXArray]) throws -> any GLM5NextMTPArrayLease
    func prepare(shiftedTokens: MLXArray, targetHidden: MLXArray) throws
        -> any GLM5NextMTPPreparedExecution
    func didFinishEvaluation() throws
    func checkpoint() throws -> any GLM5NextMTPExecutionCheckpoint
    func restore(_ checkpoint: any GLM5NextMTPExecutionCheckpoint) throws
    func close() throws
}

extension GLM5NextMTPExecutionOwner {
    public var evaluationTargets: [MLXArray] { [] }
    /// Existing owners report their own execution buffers; the adapter adds
    /// their input leases unless the owner explicitly accounts for them too.
    public func accountsForResidency(of lease: any GLM5NextMTPArrayLease) -> Bool { false }
}

public struct GLM5NextMTPHistoryChunk {
    public let shiftedTokens: MLXArray
    public let targetHidden: MLXArray
    public init(shiftedTokens: MLXArray, targetHidden: MLXArray) {
        self.shiftedTokens = shiftedTokens
        self.targetHidden = targetHidden
    }
}

/// Trusted history only. Stage bounded chunks on observation; publish them in
/// didFinishEvaluation after Common's existing fault-checked observation fence.
public protocol GLM5NextMTPIncrementalExecutionOwner: GLM5NextMTPExecutionOwner {
    var maximumObservationTokens: Int { get }
    func stageHistory(_ chunks: [GLM5NextMTPHistoryChunk]) throws
}

public protocol GLM5NextMTPNativeCompletionOwner: GLM5NextMTPExecutionOwner {
    func fenceNativeCompletion() throws
    func commitNativeCompletion() throws
}

/// Recursive proposal steps after the active step-0 candidate. Graph-only and
/// never canonical: the candidate's commit or discard retires every step.
public protocol GLM5NextMTPContinuationOwner: GLM5NextMTPExecutionOwner {
    func prepareContinuation(shiftedToken: MLXArray, draftHidden: MLXArray) throws
        -> GLM5NextMTPOutput
}

/// Target verification of the trained proposal, fixed per execution factory.
/// Serial depth one is the protected default profile. Rectangular verification
/// is a separate native profile: one captured `[1, 1+depth]` target forward per
/// round, whose arithmetic is the target policy's `rectangularVerification`.
public enum GLM5NextMTPVerification: Sendable, Equatable {
    case serialTarget
    case rectangular(depth: Int)

    /// Matches the native paged MTP work envelope.
    public static let maximumRectangularDepth = CBv2NativeMTPDepth.supported.upperBound

    public var depth: Int {
        switch self {
        case .serialTarget: 1
        case .rectangular(let depth): depth
        }
    }

    public var engineMode: CBv2MTPVerificationMode {
        switch self {
        case .serialTarget: .serialTarget
        case .rectangular: .rectangular
        }
    }

    var isSupported: Bool {
        switch self {
        case .serialTarget: true
        case .rectangular(let depth): (1 ... Self.maximumRectangularDepth).contains(depth)
        }
    }
}

/// The loaded native owner supplies the actual slot admission, bound layer salt
/// and complete allocation declaration. Nil allocation refuses engine admission.
/// Creation must return a fresh owner; neither the adapter nor factory may share
/// mutable request history. There is deliberately no default owner implementation.
public protocol GLM5NextMTPExecutionFactory: AnyObject {
    var targetIdentity: ObjectIdentifier { get }
    var predictorIdentity: ObjectIdentifier { get }
    var verification: GLM5NextMTPVerification { get }
    var maximumSpeculativeBatch: Int { get }
    func makeOwner(maximumInputs: Int) throws -> any GLM5NextMTPExecutionOwner
    func allocation(limits: CBv2MTPAllocationLimits) -> CBv2MTPBoundedAllocationSpec?
}

extension GLM5NextMTPExecutionFactory {
    public var verification: GLM5NextMTPVerification { .serialTarget }
    public var maximumSpeculativeBatch: Int { 1 }
}

public protocol GLM5NextMTPAdmittedExecutionFactory: GLM5NextMTPExecutionFactory {
    func bind(drafter: AnyObject) throws
    func makeOwner(admission: CBv2AdmittedMTPContext) throws -> any GLM5NextMTPExecutionOwner
}

public final class GLM5NextMTPAdapter: CBv2MTPRequestStatefulDrafter,
    CBv2MTPExactHistoryPrefixCheckpointCoding, CBv2MTPBoundedAllocationProviding,
    CBv2AdmittedMTPRequestStatefulDrafter, CBv2NativeMTPCompletionSplitting,
    GLM5NextNativeAssistantValidating
{
    private let predictor: GLM5NextMTP
    private let factory: any GLM5NextMTPExecutionFactory
    public let verification: GLM5NextMTPVerification
    private weak var target: GLM5NextTextModel?
    private let binding = Binding()
    fileprivate final class Binding {}

    fileprivate struct Pair {
        let lease: any GLM5NextMTPArrayLease
        var tokens: MLXArray { lease.arrays[0] }
        var hidden: MLXArray { lease.arrays[1] }
        var count: Int { tokens.dim(1) }
    }

    fileprivate final class Round {
        let candidate: any GLM5NextMTPPreparedExecution
        let backlog: [Pair]
        let frontier: (any GLM5NextMTPArrayLease)?
        let seed: Pair
        let awaitingCarry: Bool
        let targetCount: Int
        var evaluated = false
        /// Draft steps built this round: step 0 plus index-shared continuations.
        var proposals = 1
        init(candidate: any GLM5NextMTPPreparedExecution, state: State, seed: Pair) {
            self.candidate = candidate
            backlog = state.backlog
            frontier = state.frontier
            self.seed = seed
            awaitingCarry = state.awaitingCarry
            targetCount = state.targetCount
        }
    }

    public final class State: CBv2MTPRequestResidencyReporting {
        fileprivate let binding: Binding
        fileprivate var owner: (any GLM5NextMTPExecutionOwner)?
        fileprivate var admission: CBv2AdmittedMTPContext?
        fileprivate var pendingCheckpoint: (any GLM5NextMTPExecutionCheckpoint)?
        fileprivate var pendingDecodedPrefix: DecodedPrefix?
        fileprivate var backlog: [Pair] = []
        fileprivate var frontier: (any GLM5NextMTPArrayLease)?
        fileprivate var prefixFrontiers: (any GLM5NextMTPArrayLease)?
        fileprivate var retired: [any GLM5NextMTPArrayLease] = []
        fileprivate var round: Round?
        fileprivate var targetCount = 0
        fileprivate var awaitingCarry = false
        fileprivate var expectingOrdinaryObservation = false
        fileprivate var observationPending = false
        fileprivate var released = false
        fileprivate var maximumInputs: Int?
        fileprivate var publishedInputCount = 0
        fileprivate init(binding: Binding) { self.binding = binding }

        public var committedInputCount: Int {
            // A queued observation is not published before its required fence.
            publishedInputCount
        }
        public var stagedInputCount: Int { round?.candidate.inputCount ?? 0 }
        public var hasPendingPrefillForCostAccounting: Bool { !backlog.isEmpty }
        public var hasUnmeasuredResidency: Bool {
            observationPending || pendingDecodedPrefix != nil || round != nil
                || owner?.hasUnmeasuredResidency == true
                || leases.contains { $0.hasUnmeasuredResidency }
        }
        public var materializedBytes: Int {
            leases.reduce(owner?.materializedBytes ?? 0) { total, lease in
                guard owner?.accountsForResidency(of: lease) != true else { return total }
                let sum = total.addingReportingOverflow(lease.materializedBytes)
                return sum.overflow ? Int.max : sum.partialValue
            }
        }
        fileprivate var leases: [any GLM5NextMTPArrayLease] {
            var all = backlog.map(\.lease) + retired
            if let frontier { all.append(frontier) }
            if let prefixFrontiers { all.append(prefixFrontiers) }
            if let round {
                all += round.backlog.map(\.lease) + [round.seed.lease]
                if let frontier = round.frontier { all.append(frontier) }
            }
            var seen = Set<ObjectIdentifier>()
            return all.filter { seen.insert(ObjectIdentifier($0)).inserted }
        }
    }

    private final class Prefix: CBv2MTPPrefixCheckpoint {
        let binding: Binding
        let cache: any GLM5NextMTPExecutionCheckpoint
        let backlog: [Pair]
        let frontier: any GLM5NextMTPArrayLease
        let frontiers: any GLM5NextMTPArrayLease
        let maximumInputs: Int
        let targetInputCount: Int
        var materializedBytes: Int {
            (backlog.map(\.lease) + [frontier, frontiers]).reduce(cache.materializedBytes) {
                let sum = $0.addingReportingOverflow($1.materializedBytes)
                return sum.overflow ? Int.max : sum.partialValue
            }
        }
        var evaluationTargets: [MLXArray] {
            cache.evaluationTargets + backlog.flatMap { $0.lease.arrays }
                + frontier.arrays + frontiers.arrays
        }
        init(
            state: State, cache: any GLM5NextMTPExecutionCheckpoint,
            frontier: any GLM5NextMTPArrayLease, frontiers: any GLM5NextMTPArrayLease
        ) {
            binding = state.binding
            self.cache = cache
            backlog = state.backlog
            self.frontier = frontier
            self.frontiers = frontiers
            maximumInputs = state.maximumInputs!
            targetInputCount = state.targetCount
        }
    }

    fileprivate final class DecodedPrefix: CBv2MTPPrefixCheckpoint {
        let binding: Binding
        let value: GLM5NextMTPExecution.DecodedPrefix
        var targetInputCount: Int { value.targetInputCount }
        var materializedBytes: Int { value.materializedBytes }
        var evaluationTargets: [MLXArray] { value.tensors }
        init(binding: Binding, value: GLM5NextMTPExecution.DecodedPrefix) {
            self.binding = binding
            self.value = value
        }
    }

    /// Every draft step is submitted as soon as it is built (the engine's
    /// `submitsEveryDraftStepEarly`), so continuations draft on the GPU while
    /// the verification forward is still being built. `false` submits only
    /// step 0 early, as before; both evaluate the same graphs.
    public let submitsEveryDraftStepEarly: Bool

    public init(
        target: GLM5NextTextModel, predictor: GLM5NextMTP, factory: any GLM5NextMTPExecutionFactory,
        submitsEveryDraftStepEarly: Bool = true
    ) throws {
        guard predictor.targetIdentity == ObjectIdentifier(target),
            factory.targetIdentity == predictor.targetIdentity,
            factory.predictorIdentity == ObjectIdentifier(predictor),
            factory.verification.isSupported
        else { throw GLM5NextMTPError.incompatibleState }
        self.target = target
        self.predictor = predictor
        self.factory = factory
        verification = factory.verification
        self.submitsEveryDraftStepEarly = submitsEveryDraftStepEarly
        try (factory as? any GLM5NextMTPAdmittedExecutionFactory)?.bind(drafter: self)
    }

    public var mtpTargetIdentity: ObjectIdentifier? { target.map(ObjectIdentifier.init) }
    public var maximumDraftTokens: Int? { verification.depth }
    /// Request states are independent, so rectangular rounds draft each
    /// speculating row separately and verify all windows in one forward.
    public var maximumSpeculativeBatch: Int? {
        verification == .serialTarget ? 1 : factory.maximumSpeculativeBatch
    }
    public var requiredVerificationMode: CBv2MTPVerificationMode? { verification.engineMode }
    public var requiresCommittedObservationFence: Bool { true }
    /// Rectangular rounds verify a deterministic greedy proposal: accepting it
    /// iff it equals the token pre-sampled from the target's own distribution,
    /// and otherwise emitting that sample, is exact speculative sampling.
    public var supportsTargetPrefixAcceptance: Bool { verification != .serialTarget }
    public func boundedRequestAllocation(limits: CBv2MTPAllocationLimits)
        -> CBv2MTPBoundedAllocationSpec?
    {
        guard (1 ... verification.depth).contains(limits.maximumDraftTokens) else { return nil }
        return factory.allocation(limits: limits)
    }

    public func makeRequestState() -> any CBv2MTPRequestState { State(binding: binding) }
    public func configureRequestState(
        _ requestState: any CBv2MTPRequestState, maximumSequenceLength: Int
    ) throws {
        guard !(factory is any GLM5NextMTPAdmittedExecutionFactory) else {
            throw CBv2MTPBindingError.missingAdmission
        }
        let state = try checked(requestState, configured: false)
        guard maximumSequenceLength > 0,
            maximumSequenceLength <= predictor.configuration.maxPositionEmbeddings,
            state.maximumInputs == nil || state.maximumInputs == maximumSequenceLength
        else {
            throw GLM5NextMTPError.invalid("MTP request bound changed or exceeds model context")
        }
        if state.owner == nil {
            let owner = try factory.makeOwner(maximumInputs: maximumSequenceLength)
            guard owner.targetIdentity == predictor.targetIdentity,
                owner.predictorIdentity == ObjectIdentifier(predictor),
                owner.maximumInputs == maximumSequenceLength, owner.committedInputCount == 0
            else {
                try owner.close()
                throw GLM5NextMTPError.incompatibleState
            }
            state.owner = owner
            state.maximumInputs = maximumSequenceLength
        }
    }

    public func configureRequestState(
        _ requestState: any CBv2MTPRequestState, admission: CBv2AdmittedMTPContext
    ) throws {
        let state = try checked(requestState, configured: false)
        try admission.validate(
            target: predictor.targetIdentity,
            drafter: ObjectIdentifier(self), maximumInputs: admission.maximumInputs)
        guard let factory = factory as? any GLM5NextMTPAdmittedExecutionFactory else {
            throw CBv2MTPBindingError.missingAdmission
        }
        if let previous = state.admission {
            guard previous === admission else { throw CBv2MTPBindingError.staleRequest }
            return
        }
        let owner = try factory.makeOwner(admission: admission)
        guard owner.targetIdentity == predictor.targetIdentity,
            owner.predictorIdentity == ObjectIdentifier(predictor),
            owner.maximumInputs == admission.maximumInputs, owner.committedInputCount == 0
        else {
            try owner.close()
            throw GLM5NextMTPError.incompatibleState
        }
        if let decoded = state.pendingDecodedPrefix {
            guard let concrete = factory as? GLM5NextMTPExecution,
                state.targetCount <= admission.maximumInputs
            else {
                try owner.close()
                throw GLM5NextMTPError.incompatibleState
            }
            do {
                let copied = try concrete.restoreDecodedPrefix(decoded.value, owner: owner)
                state.frontier = copied.hidden
                state.prefixFrontiers = copied.frontiers
                state.observationPending = true
                state.publishedInputCount = 0
            } catch {
                try owner.close()
                throw error
            }
        } else if let checkpoint = state.pendingCheckpoint {
            guard state.targetCount <= admission.maximumInputs else {
                try owner.close()
                throw CBv2MTPBindingError.changedLimit
            }
            do { try owner.restore(checkpoint) } catch {
                try owner.close()
                throw error
            }
        }
        state.owner = owner
        state.pendingCheckpoint = nil
        state.pendingDecodedPrefix = nil
        state.maximumInputs = admission.maximumInputs
        state.admission = admission
    }

    /// Typed throwing entry point for native callers. Common's nonthrowing twin
    /// treats a violated shape/ownership invariant as an explicit contract failure.
    public func observe(_ observation: CBv2MTPCommittedTargetObservation, state: State) throws {
        _ = try checked(state)
        guard state.round == nil else { throw GLM5NextMTPError.activeRound }
        try validate(tokens: observation.tokens, hidden: observation.hidden)
        let count = observation.tokens.dim(1)
        let owner = state.owner!
        if let incremental = owner as? any GLM5NextMTPIncrementalExecutionOwner {
            guard count <= incremental.maximumObservationTokens else {
                throw GLM5NextMTPError.invalid(
                    "target observation exceeds admitted MTP prefill bound")
            }
        }
        var backlog = state.backlog
        var frontier = state.frontier
        var retired = state.retired
        var targetCount = state.targetCount
        var awaitingCarry = state.awaitingCarry
        var expectingOrdinary = state.expectingOrdinaryObservation
        if state.awaitingCarry {
            // Common sends (new carry, PREVIOUS target hidden) before the ordinary
            // observation of that same carry. This is already a shifted pair.
            guard count == 1 else { throw GLM5NextMTPError.invalid("expected one history carry") }
            guard
                state.owner!.committedInputCount + backlog.reduce(0, { $0 + $1.count })
                    < state.maximumInputs!
            else { throw GLM5NextMTPError.invalid("history carry exceeds request context") }
            backlog.append(try pair(observation.tokens, observation.hidden, owner: owner))
            awaitingCarry = false
            expectingOrdinary = true
        } else {
            guard count <= state.maximumInputs! - state.targetCount else {
                throw GLM5NextMTPError.invalid("MTP observed context exceeds request bound")
            }
            if let frontier = state.frontier {
                backlog.append(
                    try pair(observation.tokens[0..., ..<1], frontier.arrays[0], owner: owner))
                retired.append(frontier)
            } else if state.targetCount > 0 && !state.expectingOrdinaryObservation {
                throw GLM5NextMTPError.invalid("missing preceding target hidden frontier")
            }
            if count > 1 {
                backlog.append(
                    try pair(
                        observation.tokens[0..., 1...],
                        observation.hidden[0..., ..<(count - 1), 0...], owner: owner))
            }
            let nextFrontier = try owner.hold([observation.hidden[0..., (-1)..., 0...]])
            guard nextFrontier.arrays.count == 1,
                nextFrontier.arrays[0].shape == [1, 1, predictor.configuration.hiddenSize]
            else { throw GLM5NextMTPError.incompatibleState }
            frontier = nextFrontier
            targetCount += count
            expectingOrdinary = false
        }
        if let incremental = owner as? any GLM5NextMTPIncrementalExecutionOwner, !backlog.isEmpty {
            try incremental.stageHistory(
                backlog.map {
                    GLM5NextMTPHistoryChunk(shiftedTokens: $0.tokens, targetHidden: $0.hidden)
                })
            // Copies and their source leases survive until the observation fence.
            retired += backlog.map(\.lease)
            backlog.removeAll()
        }
        if let concrete = factory as? GLM5NextMTPExecution {
            let positions = try concrete.holdPrefixFrontiers(
                owner: owner, targetInputCount: targetCount)
            if let previous = state.prefixFrontiers { retired.append(previous) }
            state.prefixFrontiers = positions
        }
        state.backlog = backlog
        state.frontier = frontier
        state.retired = retired
        state.targetCount = targetCount
        state.awaitingCarry = awaitingCarry
        state.expectingOrdinaryObservation = expectingOrdinary
        state.observationPending = true
    }

    public func draft(tokens: MLXArray, hidden: MLXArray, state: State, shortlist: MLXArray? = nil)
        throws
        -> (tokens: MLXArray, hidden: MLXArray)
    {
        _ = try checked(state)
        if let round = state.round {
            return try draftContinuation(
                tokens: tokens, hidden: hidden, state: state, round: round, shortlist: shortlist)
        }
        guard !state.observationPending, state.targetCount > 0, shortlist == nil,
            tokens.shape == [1, 1]
        else {
            throw GLM5NextMTPError.invalid(
                "unfenced/unprimed MTP history or unsupported shortlist/shape")
        }
        try validate(tokens: tokens, hidden: hidden)
        let owner = state.owner!
        if let incremental = owner as? any GLM5NextMTPIncrementalExecutionOwner,
            !state.backlog.isEmpty
        {
            // Accepted inputs queued by finalize are canonical target data.
            // Prime them separately; the new proposal remains exactly one query.
            try incremental.stageHistory(
                state.backlog.map {
                    GLM5NextMTPHistoryChunk(shiftedTokens: $0.tokens, targetHidden: $0.hidden)
                })
            state.retired += state.backlog.map(\.lease)
            state.backlog.removeAll()
        }
        let seed = try pair(tokens, hidden, owner: owner)
        state.retired.append(seed.lease)
        let pairs = state.backlog + [seed]
        let count = pairs.reduce(0) { $0 + $1.count }
        guard count <= state.maximumInputs! - owner.committedInputCount else {
            throw GLM5NextMTPError.invalid("MTP draft context exceeds request bound")
        }
        let prepared = try owner.prepare(
            shiftedTokens: concatenated(pairs.map(\.tokens), axis: 1),
            targetHidden: concatenated(pairs.map(\.hidden), axis: 1))
        guard prepared.inputCount == count,
            prepared.output.hidden.shape == [1, count, predictor.configuration.hiddenSize],
            prepared.output.logits.shape == [1, 1, predictor.configuration.vocabularySize]
        else {
            try prepared.discard()
            throw GLM5NextMTPError.incompatibleState
        }
        let round = Round(candidate: prepared, state: state, seed: seed)
        state.round = round
        state.backlog = []
        state.frontier = nil
        state.awaitingCarry = false
        let last = prepared.output.hidden[0..., (-1)..., 0...]
        return (argMax(prepared.output.logits[0..., -1, 0...], axis: -1).asType(.int32), last)
    }

    /// The engine chains `(previous proposal, its returned hidden)`. Neither
    /// is trusted history; only the round's step-0 candidate can commit.
    private func draftContinuation(
        tokens: MLXArray, hidden: MLXArray, state: State, round: Round, shortlist: MLXArray?
    ) throws -> (tokens: MLXArray, hidden: MLXArray) {
        guard round.proposals < verification.depth, !round.evaluated,
            let owner = state.owner as? any GLM5NextMTPContinuationOwner
        else { throw GLM5NextMTPError.unsupportedDraftDepth(round.proposals + 1) }
        guard shortlist == nil, tokens.shape == [1, 1] else {
            throw GLM5NextMTPError.invalid("unsupported continuation shortlist/shape")
        }
        try validate(tokens: tokens, hidden: hidden)
        let output = try owner.prepareContinuation(shiftedToken: tokens, draftHidden: hidden)
        guard output.hidden.shape == [1, 1, predictor.configuration.hiddenSize],
            output.logits.shape == [1, 1, predictor.configuration.vocabularySize]
        else { throw GLM5NextMTPError.incompatibleState }
        round.proposals += 1
        return (argMax(output.logits[0..., -1, 0...], axis: -1).asType(.int32), output.hidden)
    }

    public func finalize(
        state: State, confirmedInputs: Int, acceptedTokens: MLXArray, targetHidden: MLXArray
    ) throws {
        _ = try checked(state)
        guard let round = state.round else { throw GLM5NextMTPError.noActiveRound }
        guard acceptedTokens.ndim == 2 else {
            throw GLM5NextMTPError.invalid("accepted token rank")
        }
        let count = acceptedTokens.dim(1)
        guard round.evaluated, acceptedTokens.dim(0) == 1,
            count <= round.proposals, confirmedInputs == count + 1,
            targetHidden.shape == [1, count, predictor.configuration.hiddenSize],
            confirmedInputs <= state.maximumInputs! - round.targetCount
        else {
            throw GLM5NextMTPError.invalid("unfenced MTP finalization or invalid accepted prefix")
        }
        if count > 0 { try validate(tokens: acceptedTokens, hidden: targetHidden) }
        // All consumed inputs are trusted backlog + confirmed seed. The proposed
        // token has NOT been consumed; acceptance never commits a recycled draft hidden.
        let accepted =
            count == 0 ? nil : try pair(acceptedTokens, targetHidden, owner: state.owner!)
        try round.candidate.commit()
        state.backlog = accepted.map { [$0] } ?? []
        state.targetCount = round.targetCount + confirmedInputs
        state.awaitingCarry = true
        state.frontier = nil
        state.round = nil
        state.prefixFrontiers = nil
        state.retired.removeAll()
        state.publishedInputCount =
            state.owner!.committedInputCount
            + (state.owner is any GLM5NextMTPIncrementalExecutionOwner ? 0 : count)
    }

    public func discard(state: State) throws {
        _ = try checked(state)
        guard let round = state.round else { return }
        try round.candidate.discard()
        state.backlog = round.backlog
        state.frontier = round.frontier
        state.awaitingCarry = round.awaitingCarry
        state.targetCount = round.targetCount
        state.round = nil
        state.publishedInputCount =
            state.owner!.committedInputCount
            + (state.owner is any GLM5NextMTPIncrementalExecutionOwner
                ? 0 : state.backlog.reduce(0) { $0 + $1.count })
        // Inputs may still be referenced by in-flight work. The owner must retain
        // discarded candidates until its required completion/retirement fence.
    }

    public func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation, requestState: any CBv2MTPRequestState
    ) {
        do { try observe(observation, state: checked(requestState)) } catch {
            preconditionFailure("GLM5Next MTP observation contract: \(error)")
        }
    }
    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    )
        -> (tokens: MLXArray, hidden: MLXArray)
    {
        do {
            return try draft(
                tokens: tokens, hidden: hidden, state: checked(requestState), shortlist: shortlist)
        } catch { preconditionFailure("GLM5Next MTP draft contract: \(error)") }
    }
    public func finalizeRound(
        requestState: any CBv2MTPRequestState, confirmedInputTokens: Int,
        committedDraftTokens: MLXArray, committedTargetHidden: MLXArray
    ) {
        do {
            try finalize(
                state: checked(requestState), confirmedInputs: confirmedInputTokens,
                acceptedTokens: committedDraftTokens, targetHidden: committedTargetHidden)
        } catch { preconditionFailure("GLM5Next MTP finalization contract: \(error)") }
    }
    public func discardRound(requestState: any CBv2MTPRequestState) {
        do { try discard(state: checked(requestState)) } catch {
            preconditionFailure("GLM5Next MTP discard contract: \(error)")
        }
    }
    private final class UnusedCapture: CBv2MTPPreparedCapture {}
    public func prepare(rows: [CBv2MTPRowCapture]) -> any CBv2MTPPreparedCapture { UnusedCapture() }
    public func draftStep(tokens: MLXArray, hidden: MLXArray, prepared: any CBv2MTPPreparedCapture)
        -> (tokens: MLXArray, hidden: MLXArray)
    { preconditionFailure("GLM5Next MTP requires request-owned state") }

    public func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] {
        guard let state = requestState as? State, state.binding === binding, !state.released else {
            return []
        }
        let output =
            state.round.map { [$0.candidate.output.hidden, $0.candidate.output.logits] } ?? []
        return state.leases.flatMap(\.arrays) + (state.owner?.evaluationTargets ?? [])
            + (state.pendingDecodedPrefix?.evaluationTargets ?? [])
            + (state.round?.candidate.evaluationTargets ?? [])
            + output
    }
    public func requestStateDidFinishEvaluation(_ requestState: any CBv2MTPRequestState) throws {
        let state = try checked(requestState, draining: true)
        try state.owner!.didFinishEvaluation()
        guard state.admission?.isActive != false else { return }
        try publishCompletedState(state)
    }

    package func fenceRequestStateForNativeCompletion(_ requestState: any CBv2MTPRequestState)
        throws
    {
        let state = try checked(requestState, draining: true)
        guard let owner = state.owner as? any GLM5NextMTPNativeCompletionOwner else {
            throw GLM5NextMTPError.invalid(
                "native GLM completion requires its concrete admitted owner")
        }
        try owner.fenceNativeCompletion()
    }

    package func commitRequestStateNativeCompletion(_ requestState: any CBv2MTPRequestState) throws
    {
        let state = try checked(requestState, draining: true)
        guard let owner = state.owner as? any GLM5NextMTPNativeCompletionOwner else {
            throw GLM5NextMTPError.invalid("native GLM completion owner changed")
        }
        try owner.commitNativeCompletion()
        guard state.admission?.isActive != false else { return }
        try publishCompletedState(state)
    }

    package func validateNativeGLMAssistant(source: any GLM5NextNativeProfileSource) throws {
        guard let factory = factory as? GLM5NextMTPExecution,
            factory === source, predictorIdentity == source.nativeGLMPredictorIdentity,
            predictor.targetIdentity == source.nativeGLMTargetIdentity, target != nil
        else {
            throw GLM5NextMTPError.incompatibleState
        }
    }

    private var predictorIdentity: ObjectIdentifier { ObjectIdentifier(predictor) }

    private func publishCompletedState(_ state: State) throws {
        if let round = state.round {
            try round.candidate.didFinishEvaluation()
            round.evaluated = true
        }
        state.observationPending = false
        state.retired.removeAll()
        if state.round == nil {
            state.publishedInputCount =
                state.owner!.committedInputCount
                + (state.owner is any GLM5NextMTPIncrementalExecutionOwner
                    ? 0 : state.backlog.reduce(0) { $0 + $1.count })
        }
    }
    public func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        guard let state = requestState as? State, state.binding === binding, !state.released else {
            return
        }
        // Common calls this only after its cancellation/teardown fence.
        do { try state.owner?.close() } catch {
            preconditionFailure("GLM5Next MTP retirement failed: \(error)")
        }
        state.owner = nil
        state.admission = nil
        state.pendingCheckpoint = nil
        state.pendingDecodedPrefix = nil
        state.backlog = []
        state.frontier = nil
        state.prefixFrontiers = nil
        state.round = nil
        state.retired = []
        state.released = true
        state.publishedInputCount = 0
    }

    public func capturePrefixCheckpoint(
        requestState: any CBv2MTPRequestState, targetInputCount: Int
    )
        -> (any CBv2MTPPrefixCheckpoint)?
    {
        guard targetInputCount > 1, let state = try? checked(requestState),
            state.round == nil, !state.observationPending,
            !state.awaitingCarry, !state.expectingOrdinaryObservation,
            state.backlog.isEmpty, state.pendingCheckpoint == nil,
            state.pendingDecodedPrefix == nil,
            state.targetCount == targetInputCount, let frontier = state.frontier,
            let frontiers = state.prefixFrontiers,
            state.committedInputCount == targetInputCount - 1,
            !state.hasUnmeasuredResidency, let checkpoint = try? state.owner!.checkpoint()
        else { return nil }
        return Prefix(state: state, cache: checkpoint, frontier: frontier, frontiers: frontiers)
    }
    public func restorePrefixCheckpoint(_ checkpoint: any CBv2MTPPrefixCheckpoint) -> (
        any CBv2MTPRequestState
    )? {
        if let decoded = checkpoint as? DecodedPrefix {
            guard decoded.binding === binding, factory is GLM5NextMTPExecution else { return nil }
            let state = State(binding: binding)
            state.pendingDecodedPrefix = decoded
            state.targetCount = decoded.targetInputCount
            state.observationPending = true
            return state
        }
        guard let prefix = checkpoint as? Prefix, prefix.binding === binding else { return nil }
        let state = State(binding: binding)
        if factory is any GLM5NextMTPAdmittedExecutionFactory {
            // The saved leases remain charged to their source request. A NEW
            // Common-issued request context must arrive before allocating an owner.
            state.pendingCheckpoint = prefix.cache
            state.backlog = prefix.backlog
            state.frontier = prefix.frontier
            state.prefixFrontiers = prefix.frontiers
            state.targetCount = prefix.targetInputCount
            state.publishedInputCount = prefix.targetInputCount - 1
            return state
        }
        do {
            try configureRequestState(state, maximumSequenceLength: prefix.maximumInputs)
            try state.owner!.restore(prefix.cache)
            state.backlog = prefix.backlog
            state.frontier = prefix.frontier
            state.prefixFrontiers = prefix.frontiers
            state.targetCount = prefix.targetInputCount
            state.publishedInputCount = prefix.targetInputCount - 1
            return state
        } catch {
            releaseRequestState(state)
            return nil
        }
    }

    /// The prefix checkpoint is the predictor's own cache, frontier hidden and
    /// frontiers, so it stays exact beside lossy latent rows
    /// (`CBv2MTPExactHistoryPrefixCheckpointCoding`).
    public var prefixCheckpointCodecID: String { GLM5NextMTPExecution.prefixCodecID }

    public func prefixCheckpointTensorDescriptors(targetInputCount: Int)
        -> [CBv2CheckpointTensorDescriptor]?
    {
        (factory as? GLM5NextMTPExecution)?.prefixDescriptors(targetInputCount: targetInputCount)
    }

    public func encodePrefixCheckpoint(_ checkpoint: any CBv2MTPPrefixCheckpoint) -> [MLXArray]? {
        guard let prefix = checkpoint as? Prefix, prefix.binding === binding,
            prefix.backlog.isEmpty, let concrete = factory as? GLM5NextMTPExecution
        else { return nil }
        return try? concrete.encodePrefix(
            checkpoint: prefix.cache, hidden: prefix.frontier, frontiers: prefix.frontiers,
            targetInputCount: prefix.targetInputCount)
    }

    public func decodePrefixCheckpoint(tensors: [MLXArray], prefixTokens: [Int])
        -> (any CBv2MTPPrefixCheckpoint)?
    {
        guard let concrete = factory as? GLM5NextMTPExecution,
            prefixTokens.allSatisfy({ $0 >= 0 && $0 < predictor.configuration.vocabularySize }),
            let value = try? concrete.decodePrefix(
                tensors: tensors, targetInputCount: prefixTokens.count)
        else { return nil }
        // Authentication/model/prompt identity and stage ownership belong to
        // Common's import transaction. This value owns no mutable request state.
        return DecodedPrefix(binding: binding, value: value)
    }

    private func checked(
        _ requestState: any CBv2MTPRequestState, configured: Bool = true, draining: Bool = false
    ) throws
        -> State
    {
        guard target != nil, let state = requestState as? State, state.binding === binding else {
            throw GLM5NextMTPError.incompatibleState
        }
        guard !state.released else { throw GLM5NextMTPError.releasedState }
        if let admission = state.admission, !draining {
            try admission.validate(
                target: predictor.targetIdentity,
                drafter: ObjectIdentifier(self), maximumInputs: state.maximumInputs!)
        }
        guard !configured || state.owner != nil else {
            throw GLM5NextMTPError.invalid("unconfigured MTP request owner")
        }
        return state
    }
    private func validate(tokens: MLXArray, hidden: MLXArray) throws {
        guard tokens.ndim == 2, tokens.dim(0) == 1, tokens.dim(1) > 0,
            [.int32, .uint32, .int64].contains(tokens.dtype),
            hidden.shape == [1, tokens.dim(1), predictor.configuration.hiddenSize],
            hidden.dtype.isFloatingPoint
        else {
            throw GLM5NextMTPError.invalid(
                "MTP expects aligned [1,T] tokens and final-normalized target hidden")
        }
    }
    private func pair(_ tokens: MLXArray, _ hidden: MLXArray, owner: any GLM5NextMTPExecutionOwner)
        throws -> Pair
    {
        let lease = try owner.hold([tokens, hidden])
        guard lease.arrays.count == 2, lease.arrays[0].shape == tokens.shape,
            lease.arrays[1].shape == hidden.shape
        else { throw GLM5NextMTPError.incompatibleState }
        return Pair(lease: lease)
    }
}
