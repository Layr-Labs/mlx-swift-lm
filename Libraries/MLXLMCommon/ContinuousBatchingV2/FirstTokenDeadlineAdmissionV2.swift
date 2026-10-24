// Copyright © 2026 Eigen Labs.
//
// First-token deadline admission for the serialized-prefill scheduler.
//
// The scheduler owns only a deterministic work projection. Deadline policy
// (remaining duration and the caller's conservative service rate) stays in
// EngineV2, where enqueue, prefix adoption, the verdict, and activation are
// serialized against engine steps.

/// Caller-owned policy inputs for first-token deadline admission.
///
/// `deadline` is an absolute `ContinuousClock` instant. The engine reads its
/// own monotonic clock only after the serialized engine-queue closure starts;
/// time spent waiting for that closure therefore consumes the original budget
/// instead of receiving a fresh relative allowance.
///
/// Phase rates are optional conservative lower bounds. A projection that
/// contains work for a phase whose rate is nil fails closed. In particular, a
/// mixed prefill/decode posture needs both rates: the engine never prices
/// decode assignments at prefill throughput.
///
/// `calibration` is optional reviewed caller evidence. The engine matches its
/// cells only after actual queue projection and prefix adoption. Unsupported,
/// expired or invalidated evidence uses these original conservative rates;
/// calibration never restarts or extends `deadline`.
///
/// `decodeRateShape` selects per-batch decode charging. A per-request decode
/// rate prices one row's cadence; all decode rows of one step advance
/// together, so summing every row's tokens overcharges a B-row step about B
/// times. When set, each projected step costs one measured step of the
/// requests that produced the rate, scaled linearly when the step has more
/// decode rows or a wider assignment than they did. nil keeps the legacy
/// per-row charge.
public struct CBv2FirstTokenDeadlineAdmission: Sendable, Equatable {
    public let deadline: ContinuousClock.Instant
    public let conservativePrefillTokensPerSecond: Double?
    public let conservativeDecodeTokensPerSecond: Double?
    public let calibration: CBv2FirstContentCalibration?
    public let nativeTargetPrefill: CBv2NativeTargetPrefillPolicy?
    public let decodeRateShape: CBv2DecodeRateShape?

    public init(
        deadline: ContinuousClock.Instant,
        conservativePrefillTokensPerSecond: Double?,
        conservativeDecodeTokensPerSecond: Double?,
        calibration: CBv2FirstContentCalibration? = nil,
        nativeTargetPrefill: CBv2NativeTargetPrefillPolicy? = nil,
        decodeRateShape: CBv2DecodeRateShape? = nil
    ) {
        self.deadline = deadline
        self.conservativePrefillTokensPerSecond = conservativePrefillTokensPerSecond
        self.conservativeDecodeTokensPerSecond = conservativeDecodeTokensPerSecond
        self.calibration = calibration
        self.nativeTargetPrefill = nativeTargetPrefill
        self.decodeRateShape = decodeRateShape
    }

    /// Per-batch charging is requested with a usable measured shape.
    var chargesDecodePerBatch: Bool {
        decodeRateShape?.isValid == true
    }

    /// Whether the projection may apply the live MTP depth gates. A reviewed
    /// calibration's error envelope was measured against the configured
    /// `1+k` work vector, so supplying one keeps that vector unchanged.
    var mirrorsSpeculationDepth: Bool {
        chargesDecodePerBatch && calibration == nil
    }
}

/// The decode steps behind a per-request decode rate, averaged over the
/// requests that measured it: token-producing rows per step, confirmed tokens
/// per decode step, and assigned width per decode step (1 for plain decode,
/// `1+k` for an MTP verify round). One measured step therefore takes
/// `tokensPerStep / rate` seconds.
public struct CBv2DecodeRateShape: Sendable, Equatable {
    public let batchRows: Double
    public let tokensPerStep: Double
    public let stepWidth: Double

    public init(batchRows: Double, tokensPerStep: Double = 1, stepWidth: Double = 1) {
        self.batchRows = batchRows
        self.tokensPerStep = tokensPerStep
        self.stepWidth = stepWidth
    }

    /// Every component is finite and at least one.
    public var isValid: Bool {
        [batchRows, tokensPerStep, stepWidth].allSatisfy { $0.isFinite && $0 >= 1 }
    }
}

/// One projected step's decode rows and its widest decode assignment.
public struct CBv2DecodeStepShape: Sendable, Hashable {
    public let rows: Int
    public let width: Int

    public init(rows: Int, width: Int) {
        self.rows = rows
        self.width = width
    }
}

/// Scheduler work through the step that can produce the target's first token.
///
/// `mixedSteps` counts steps containing both phases. The service bound charges
/// phase envelopes serially:
///
///     T = prefillTokens / prefillRate + decodeTokens / decodeRate
///
/// This is deliberately conservative for one `asyncEval`: if phase execution
/// overlaps perfectly, the excess is at most the smaller phase envelope for
/// that step. Summed over the projection, false rejection from serialization
/// is therefore bounded by the sum of those smaller envelopes. If either
/// lower-bound phase rate is unavailable, mixed work is unbounded and
/// enforcement fails closed.
///
/// Per-batch charging (`decodeRateShape`) replaces `decodeTokens` in `T` with
/// `batchDecodeTokens(rate:)`.
public struct CBv2FirstTokenScheduledWork: Sendable, Equatable {
    public let prefillTokens: Int
    public let decodeTokens: Int
    public let scheduledSteps: Int
    public let mixedSteps: Int
    /// Projected decode steps by shape, rather than per-row tokens. A plain
    /// decode step has width 1 whatever its row count. nil when the work did
    /// not come from a scheduler projection (callers then keep the per-row
    /// `decodeTokens` charge).
    public let decodeSteps: [CBv2DecodeStepShape: Int]?

    public init(
        prefillTokens: Int,
        decodeTokens: Int,
        scheduledSteps: Int,
        mixedSteps: Int,
        decodeSteps: [CBv2DecodeStepShape: Int]? = nil
    ) {
        self.prefillTokens = prefillTokens
        self.decodeTokens = decodeTokens
        self.scheduledSteps = scheduledSteps
        self.mixedSteps = mixedSteps
        self.decodeSteps = decodeSteps
    }

    /// Per-batch decode charge in decode-rate tokens: each step costs the
    /// measured `tokensPerStep`, scaled by `max(1, width / stepWidth)` and
    /// `max(1, rows / batchRows)`. That is a bound when step time never falls
    /// and grows at most linearly as rows or width grow. nil without a
    /// per-step projection or with an invalid shape.
    func batchDecodeTokens(rate shape: CBv2DecodeRateShape) -> Double? {
        guard let decodeSteps, shape.isValid else { return nil }
        var total = 0.0
        for (step, count) in decodeSteps {
            guard step.rows > 0, step.width > 0, count >= 0 else { return nil }
            total +=
                Double(count) * shape.tokensPerStep
                * max(1, Double(step.width) / shape.stepWidth)
                * max(1, Double(step.rows) / shape.batchRows)
        }
        return total.isFinite ? total : nil
    }

    public var scheduledTokens: Int {
        let (total, overflow) = prefillTokens.addingReportingOverflow(decodeTokens)
        return overflow ? .max : total
    }
}

/// Conservative work through the step that can produce the new request's
/// first token.
public enum CBv2FirstTokenProjectedWork: Sendable, Equatable {
    /// `work` includes the whole target step, not just assignments preceding
    /// the target in plan order: all rows in one asyncEval complete at the
    /// same readback boundary.
    case bounded(
        work: CBv2FirstTokenScheduledWork,
        serviceDuration: Duration
    )
    /// One idle native-media observation under the original absolute deadline.
    /// Work is bounded but time is unknown: do not record zero predicted time.
    case unmeasuredNativeMedia(work: CBv2FirstTokenScheduledWork)
    /// The current scheduler state cannot produce a finite safe projection
    /// (for example, serialized prefill is not configured, all running slots
    /// are paused, the request is multimodal, or the projection complexity
    /// guard is reached). Deadline admission fails closed. The engine supplies
    /// the first failed guard; nil represents unavailable legacy/test evidence.
    case unbounded(reason: CBv2FirstTokenUnboundedReason? = nil)
}

/// Generation-bound acknowledgement that an admitted request no longer owns
/// scheduler, KV-ledger, or backend state. This is intentionally separate
/// from stream termination: a watchdog can deliver a terminal event while a
/// wedged engine step still references the row.
public struct CBv2RequestRetirement: Sendable {
    private let waitUntilRetired: @Sendable () async -> Void

    public init(waitUntilRetired: @escaping @Sendable () async -> Void) {
        self.waitUntilRetired = waitUntilRetired
    }

    init(stream: CBv2OutputStream) {
        self.init {
            await stream.waitUntilEngineOwnershipReleased()
        }
    }

    /// For implementations whose returned stream has no separately retained
    /// engine ownership (immediate results and protocol test doubles).
    public static var acknowledged: Self {
        Self(waitUntilRetired: {})
    }

    public func wait() async {
        await waitUntilRetired()
    }
}

/// Cancellation won after queue-side admission committed. The caller may
/// return promptly only by transferring both values to a cleanup owner that
/// cancels and retains `stream`, and retains external resources until
/// `retirement`.
public struct CBv2FirstTokenAdmissionCancellation: Error, Sendable {
    public let stream: AsyncStream<CBv2Event>
    public let retirement: CBv2RequestRetirement

    public init(
        stream: AsyncStream<CBv2Event>,
        retirement: CBv2RequestRetirement
    ) {
        self.stream = stream
        self.retirement = retirement
    }
}

/// Atomic submit result. A deadline rejection never enters a GPU prompt step.
public enum CBv2FirstTokenDeadlineResult: Sendable {
    case admitted(
        stream: AsyncStream<CBv2Event>,
        projectedWork: CBv2FirstTokenProjectedWork,
        admittedAt: ContinuousClock.Instant,
        retirement: CBv2RequestRetirement
    )
    case deadlineUnreachable(projectedWork: CBv2FirstTokenProjectedWork)
}

/// Scheduler-only projection before conversion through a caller-owned rate.
enum CBv2FirstTokenWorkProjection: Equatable {
    case bounded(
        work: CBv2FirstTokenScheduledWork,
        capacityOperations: [CBv2ProjectedCapacityOperation]
    )
    case unbounded(reason: CBv2FirstTokenUnboundedReason)
}

struct CBv2ProjectedCapacityReservation: Sendable, Equatable {
    let id: CBv2RequestID
    let additionalTokens: Int
    let additionalBytes: Int
}

enum CBv2ProjectedCapacityOperation: Sendable, Equatable {
    case reserve(CBv2ProjectedCapacityReservation)
    /// Runtime's chained-decode fast path executes only when its conservative
    /// headroom probe succeeds. A failed optional reservation skips that
    /// speculative chain; it does not make the target unreachable.
    case reserveIfAvailable(CBv2ProjectedCapacityReservation)
    case unreserve(CBv2ProjectedCapacityReservation)
    case release(CBv2RequestID)
}

private struct CBv2ProjectionRow {
    let id: CBv2RequestID
    let promptTokens: Int
    let maxTokens: Int
    let isMultimodal: Bool
    let requestAllowsCheckpoint: Bool
    let checkpointTargetTokens: Int?
    let isPaused: Bool
    let cancelRequested: Bool
    let prefixReusePlan: CBv2PrefixReusePlan?
    let fullSequenceCapacityTokens: Int?
    var knownTokens: Int
    var computedTokens: Int
    var generatedTokens: Int
    var demandedCheckpointContinuation: CBv2DemandedCheckpointContinuation?

    var remainingKnownTokens: Int {
        knownTokens - computedTokens
    }
}

private struct CBv2ProjectionAssignment {
    let id: CBv2RequestID
    let count: Int
    let startComputedTokens: Int
    let knownTokensBeforeStep: Int
    let prefillCount: Int

    var decodeCount: Int {
        count - prefillCount
    }
}

extension SchedulerV2 {
    /// Project scheduler work through `id`'s first sampled token.
    ///
    /// This method is pure: it does not call `plan()`, reserve capacity, read a
    /// clock, or mutate records. `inFlightAssignments` are work already
    /// launched but not finalized. Their optimistic advances are subtracted
    /// to recover confirmed cursors, then the full assignments are charged
    /// once from the admission instant; no elapsed fraction is credited.
    ///
    /// The finite model assumes successful future execution and the current
    /// queue/pause/cancel state. Future arrivals, pause transitions, and
    /// capacity failures are external state changes, not facts available to
    /// an atomic snapshot.
    /// Already-launched packing is reflected in the record's capture veto.
    /// Future ordinary packing is not predicted without model/cache claims;
    /// a later disarm can conservatively remove one priced short boundary.
    ///
    /// `mirrorsSpeculationDepth` applies the live MTP plan's pure depth gates
    /// (`speculationDepthCeiling`): no decode row is charged more than
    /// `1 + ceiling`, so beyond the batch gate, or where the automatic
    /// verification envelope leaves no depth, every decode row is charged its
    /// one target token instead of the configured `1+k`.
    ///
    /// The decode-only interleave (`decodeStepsBetweenPrefillChunks`) is
    /// replayed from the live owed count: every held step is charged as a
    /// decode-only step of the rows decoding then, so prompt work behind the
    /// held steps is priced at their measured decode-step cost.
    func firstTokenWorkProjection(
        for id: CBv2RequestID,
        inFlightAssignments: [(id: CBv2RequestID, numTokens: Int)] = [],
        inFlightAllowsChainedSuccessor: Bool = true,
        unmaterializedPrefixAdoption: Bool = false,
        mirrorsSpeculationDepth: Bool = false
    ) -> CBv2FirstTokenWorkProjection {
        // The projection does not model the opt-in prompt-row ceiling, so it
        // refuses to forecast under it.
        guard config.maxConcurrentPartialPrefills == 1,
            config.maxConcurrentRequests > 0,
            config.maxBatchedTokensPerStep > 0,
            config.prefillChunkSize > 0,
            (config.maxPrefillRowsPerStep ?? 0) <= 0
        else {
            return .unbounded(reason: .unsupportedScheduler)
        }
        guard record(for: id) != nil else {
            return .unbounded(reason: .targetMissing)
        }

        var inFlightByID: [CBv2RequestID: Int] = [:]
        for assignment in inFlightAssignments {
            guard assignment.numTokens > 0,
                let nextForID = Self.projectionAdd(
                    inFlightByID[assignment.id, default: 0], assignment.numTokens)
            else {
                return .unbounded(reason: .invalidInFlightAssignment)
            }
            inFlightByID[assignment.id] = nextForID
        }
        var projectedPrefillTokens = 0
        var projectedDecodeTokens = 0
        var projectedSteps = 0
        var projectedMixedSteps = 0
        var projectedDecodeSteps: [CBv2DecodeStepShape: Int] = [:]
        var projectedCapacityOperations: [CBv2ProjectedCapacityOperation] = []
        // The live count already settles the launched step's plan.
        var projectedDecodeOnlyStepsOwed = decodeOnlyStepsOwed
        let decodeStepsBetweenChunks = max(0, config.decodeStepsBetweenPrefillChunks)

        /// The interleave bookkeeping at the end of `plan()`, for `steps`
        /// identical steps; more than one only for decode-only stretches.
        func settleInterleave(hasDecodeRows: Bool, promptRows: Int, steps: Int = 1) {
            if !hasDecodeRows {
                projectedDecodeOnlyStepsOwed = 0
            } else if promptRows > 0 {
                projectedDecodeOnlyStepsOwed = decodeStepsBetweenChunks
            } else {
                projectedDecodeOnlyStepsOwed = max(0, projectedDecodeOnlyStepsOwed - steps)
            }
        }

        /// One step's decode rows advance together; record its shape once.
        func chargeDecodeSteps(rows: Int, width: Int, steps: Int) -> Bool {
            let shape = CBv2DecodeStepShape(rows: rows, width: width)
            guard rows > 0, width > 0, steps > 0,
                let next = Self.projectionAdd(projectedDecodeSteps[shape, default: 0], steps)
            else {
                return false
            }
            projectedDecodeSteps[shape] = next
            return true
        }

        func chargeStep(_ assignments: [CBv2ProjectionAssignment]) -> Bool {
            var prefillTokens = 0
            var decodeTokens = 0
            var decodeRows = 0
            var widestDecode = 0
            for assignment in assignments {
                guard assignment.count > 0,
                    assignment.prefillCount >= 0,
                    assignment.prefillCount <= assignment.count,
                    let nextPrefill = Self.projectionAdd(
                        prefillTokens, assignment.prefillCount),
                    let nextDecode = Self.projectionAdd(
                        decodeTokens, assignment.decodeCount)
                else {
                    return false
                }
                prefillTokens = nextPrefill
                decodeTokens = nextDecode
                if assignment.decodeCount > 0 {
                    decodeRows += 1
                    widestDecode = max(widestDecode, assignment.decodeCount)
                }
            }
            guard !assignments.isEmpty,
                let nextPrefill = Self.projectionAdd(
                    projectedPrefillTokens, prefillTokens),
                let nextDecode = Self.projectionAdd(
                    projectedDecodeTokens, decodeTokens),
                let nextSteps = Self.projectionAdd(projectedSteps, 1)
            else {
                return false
            }
            if decodeRows > 0 {
                guard chargeDecodeSteps(rows: decodeRows, width: widestDecode, steps: 1)
                else { return false }
            }
            projectedPrefillTokens = nextPrefill
            projectedDecodeTokens = nextDecode
            projectedSteps = nextSteps
            if prefillTokens > 0, decodeTokens > 0 {
                guard let nextMixed = Self.projectionAdd(projectedMixedSteps, 1) else {
                    return false
                }
                projectedMixedSteps = nextMixed
            }
            return true
        }

        /// `steps` plain-decode steps, each advancing every one of `rows`.
        func chargeDecodeStretch(rows: Int, steps: Int) -> Bool {
            guard rows > 0, steps > 0,
                let tokens = Self.projectionMultiply(rows, steps),
                let nextDecode = Self.projectionAdd(projectedDecodeTokens, tokens),
                let nextSteps = Self.projectionAdd(projectedSteps, steps),
                chargeDecodeSteps(rows: rows, width: 1, steps: steps)
            else {
                return false
            }
            projectedDecodeTokens = nextDecode
            projectedSteps = nextSteps
            return true
        }

        func boundedProjection() -> CBv2FirstTokenWorkProjection {
            .bounded(
                work: CBv2FirstTokenScheduledWork(
                    prefillTokens: projectedPrefillTokens,
                    decodeTokens: projectedDecodeTokens,
                    scheduledSteps: projectedSteps,
                    mixedSteps: projectedMixedSteps,
                    decodeSteps: projectedDecodeSteps),
                capacityOperations: projectedCapacityOperations)
        }

        var rows: [CBv2RequestID: CBv2ProjectionRow] = [:]
        for rec in running + waiting {
            let assigned = inFlightByID[rec.id, default: 0]
            guard rec.numComputedTokens >= assigned else {
                return .unbounded(reason: .inconsistentTokenCursor)
            }
            let confirmedComputed = rec.numComputedTokens - assigned
            guard confirmedComputed >= 0, confirmedComputed <= rec.tokens.count else {
                return .unbounded(reason: .inconsistentTokenCursor)
            }
            // A pending sample without the launched assignment that owns it
            // cannot be placed on this projection's timeline.
            guard rec.pendingSamples == 0 || assigned > 0 else {
                return .unbounded(reason: .unownedPendingSample)
            }
            // Multimodal token counts are not a safe service-work proxy: image
            // spans can carry model-specific superlinear work.
            guard rec.cancelRequested || rec.multimodalBlocks.isEmpty else {
                return .unbounded(reason: .multimodalWork)
            }
            rows[rec.id] = CBv2ProjectionRow(
                id: rec.id,
                promptTokens: rec.request.promptTokens.count,
                maxTokens: max(0, rec.request.maxTokens),
                isMultimodal: rec.request.multimodal != nil,
                requestAllowsCheckpoint: rec.canScheduleDemandedShortCheckpoint,
                checkpointTargetTokens: rec.request.prefixCheckpointTargetTokens,
                isPaused: rec.isPaused,
                cancelRequested: rec.cancelRequested,
                prefixReusePlan: rec.prefixReusePlan,
                fullSequenceCapacityTokens: rec.fullSequenceCapacityTokens,
                knownTokens: rec.tokens.count,
                computedTokens: confirmedComputed,
                generatedTokens: rec.generatedTokenCount,
                demandedCheckpointContinuation: rec.demandedCheckpointContinuation)
        }

        /// The live MTP plan's depth ceiling for a step with `decodeRows`
        /// decode-ready rows. Media rows can leave the planner's batch count,
        /// so any media row keeps the conservative configured `1+k` charge.
        func speculationDepthLimit(decodeRows: Int) -> Int? {
            guard mirrorsSpeculationDepth, speculationPlanner != nil,
                let ceiling = speculationDepthCeiling,
                !rows.values.contains(where: \.isMultimodal)
            else {
                return nil
            }
            return max(0, ceiling(decodeRows))
        }

        if unmaterializedPrefixAdoption {
            guard let plan = rows[id]?.prefixReusePlan,
                plan.capacityReservationTokens >= 0,
                plan.initialAdditionalCapacityBytes >= 0
            else {
                return .unbounded(reason: .invalidPrefixReservation)
            }
            projectedCapacityOperations.append(
                .reserve(
                    CBv2ProjectedCapacityReservation(
                        id: id,
                        additionalTokens: plan.capacityReservationTokens,
                        additionalBytes: plan.initialAdditionalCapacityBytes)))
        }

        var runningIDs = running.map(\.id)
        var waitingIDs = waiting.map(\.id)

        func capacityReservation(
            id reservationID: CBv2RequestID,
            start: Int,
            count: Int
        ) -> CBv2ProjectedCapacityReservation? {
            guard let row = rows[reservationID], start >= 0, count > 0 else {
                return nil
            }
            let tokens = CBv2ScheduledRequest.capacityTokensForChunk(
                start: start, count: count, plan: row.prefixReusePlan,
                fullSequenceTokens: row.fullSequenceCapacityTokens)
            let bytes =
                row.prefixReusePlan?.capacityBytesForChunk(start: start, count: count)
                ?? 0
            guard tokens >= 0, bytes >= 0 else { return nil }
            return CBv2ProjectedCapacityReservation(
                id: reservationID,
                additionalTokens: tokens,
                additionalBytes: bytes)
        }

        func appendCapacityReservation(
            id reservationID: CBv2RequestID,
            start: Int,
            count: Int
        ) -> Bool {
            guard
                let reservation = capacityReservation(
                    id: reservationID, start: start, count: count)
            else {
                return false
            }
            if reservation.additionalTokens > 0 || reservation.additionalBytes > 0 {
                projectedCapacityOperations.append(.reserve(reservation))
            }
            return true
        }

        func appendOptionalCapacityReservation(
            id reservationID: CBv2RequestID,
            start: Int,
            count: Int
        ) -> Bool {
            guard
                let reservation = capacityReservation(
                    id: reservationID, start: start, count: count)
            else {
                return false
            }
            if reservation.additionalTokens > 0 || reservation.additionalBytes > 0 {
                projectedCapacityOperations.append(.reserveIfAvailable(reservation))
            }
            return true
        }

        func appendCapacityUnreservation(
            id reservationID: CBv2RequestID,
            start: Int,
            count: Int
        ) -> Bool {
            guard
                let reservation = capacityReservation(
                    id: reservationID, start: start, count: count)
            else {
                return false
            }
            if reservation.additionalTokens > 0 || reservation.additionalBytes > 0 {
                projectedCapacityOperations.append(.unreserve(reservation))
            }
            return true
        }

        func projectionAssignment(
            id assignmentID: CBv2RequestID,
            count: Int,
            row: CBv2ProjectionRow
        ) -> CBv2ProjectionAssignment? {
            guard count > 0,
                row.computedTokens >= 0,
                row.promptTokens >= 0
            else {
                return nil
            }
            // A replay assignment can cross the prompt/decode boundary.
            // Split its token charge at that exact cursor even though it is
            // one rectangular engine assignment and one scheduler step.
            let remainingPrompt = max(
                0, row.promptTokens - row.computedTokens)
            return CBv2ProjectionAssignment(
                id: assignmentID,
                count: count,
                startComputedTokens: row.computedTokens,
                knownTokensBeforeStep: row.knownTokens,
                prefillCount: min(count, remainingPrompt))
        }

        func removeTerminatedRows() {
            let terminated = Set(
                rows.values.compactMap { row -> CBv2RequestID? in
                    row.cancelRequested || row.generatedTokens >= row.maxTokens ? row.id : nil
                })
            guard !terminated.isEmpty else { return }
            runningIDs.removeAll { terminated.contains($0) }
            waitingIDs.removeAll { terminated.contains($0) }
            for terminatedID in terminated {
                rows.removeValue(forKey: terminatedID)
                projectedCapacityOperations.append(.release(terminatedID))
            }
        }

        /// The engine may launch one pure-decode successor before finalizing
        /// the current samples. When a full running set keeps the target in
        /// waiting, that successor cannot include the target and still runs
        /// even if current finalization retires one of its rows. Charge the
        /// whole one-step-late batch; leaving surviving rows unadvanced is
        /// conservative and avoids depending on speculative acceptance.
        func chargeTerminalChainedStepIfNeeded(
            sampledIDs: [CBv2RequestID],
            allowsChainedSuccessor: Bool
        ) -> Bool {
            guard waitingIDs.contains(id),
                runningIDs.count >= config.maxConcurrentRequests,
                allowsChainedSuccessor
            else {
                return true
            }
            let activeDecodeIDs = runningIDs.compactMap { runningID -> CBv2RequestID? in
                guard let row = rows[runningID],
                    !row.isPaused,
                    !row.cancelRequested,
                    row.remainingKnownTokens == 1
                else {
                    return nil
                }
                return runningID
            }
            guard !activeDecodeIDs.isEmpty,
                runningIDs.allSatisfy({ runningID in
                    guard let row = rows[runningID] else { return false }
                    return row.isPaused
                        || (!row.cancelRequested && row.remainingKnownTokens == 1)
                }),
                activeDecodeIDs == sampledIDs,
                activeDecodeIDs.contains(where: { runningID in
                    guard let row = rows[runningID] else { return false }
                    return row.generatedTokens >= row.maxTokens
                })
            else {
                return true
            }
            for runningID in activeDecodeIDs {
                guard let row = rows[runningID],
                    appendOptionalCapacityReservation(
                        id: runningID,
                        start: row.computedTokens,
                        count: 1)
                else {
                    return false
                }
            }
            guard chargeDecodeStretch(rows: activeDecodeIDs.count, steps: 1) else {
                return false
            }
            settleInterleave(hasDecodeRows: true, promptRows: 0)
            return true
        }

        /// Apply one executed assignment. Speculative decode may assign more
        /// slots than known inputs; only one output is credited, the minimum
        /// target-authoritative progress, while all assigned work is charged.
        func applyExecutedAssignment(
            id assignedID: CBv2RequestID,
            count: Int,
            startComputedTokens: Int,
            knownTokensBeforeStep: Int
        ) -> Bool? {
            guard var row = rows[assignedID] else { return false }
            guard startComputedTokens >= 0,
                knownTokensBeforeStep >= startComputedTokens,
                count > 0
            else {
                return nil
            }
            let remaining = knownTokensBeforeStep - startComputedTokens
            if count < remaining {
                guard let computed = Self.projectionAdd(startComputedTokens, count) else {
                    return nil
                }
                row.computedTokens = computed
                rows[assignedID] = row
                return false
            }
            guard remaining > 0 else { return nil }
            // A 1+k speculative assignment reserves every target-write slot,
            // but this conservative service path credits only the guaranteed
            // first target token. Mirror finalize-time rollback for the
            // rejected suffix so repeated rounds do not accumulate impossible
            // live KV. The full width remains charged at the step peak because
            // its reserve operation precedes this temporal unreserve.
            let rejectedSpeculativeSuffix = count - remaining
            if rejectedSpeculativeSuffix > 0 {
                guard
                    appendCapacityUnreservation(
                        id: assignedID,
                        start: startComputedTokens + remaining,
                        count: rejectedSpeculativeSuffix)
                else {
                    return nil
                }
            }
            row.computedTokens = knownTokensBeforeStep
            if !row.cancelRequested {
                guard let generated = Self.projectionAdd(row.generatedTokens, 1),
                    let known = Self.projectionAdd(row.knownTokens, 1)
                else {
                    return nil
                }
                row.generatedTokens = generated
                row.knownTokens = known
            }
            rows[assignedID] = row
            return assignedID == id && !row.cancelRequested
        }

        // Place the currently launched step first. Scheduler records already
        // contain its optimistic advances; reconstructing from the confirmed
        // cursor prevents those assignments from being treated as free.
        if !inFlightAssignments.isEmpty {
            var targetSampled = false
            var sampledIDs: [CBv2RequestID] = []
            var projectedAssignments: [CBv2ProjectionAssignment] = []
            projectedAssignments.reserveCapacity(inFlightAssignments.count)
            for assignment in inFlightAssignments {
                guard let row = rows[assignment.id], row.remainingKnownTokens > 0 else {
                    return .unbounded(reason: .invalidInFlightAssignment)
                }
                guard
                    let projectedAssignment = projectionAssignment(
                        id: assignment.id,
                        count: assignment.numTokens,
                        row: row)
                else {
                    return .unbounded(reason: .invalidProjectionAssignment)
                }
                projectedAssignments.append(projectedAssignment)
                if assignment.numTokens >= row.remainingKnownTokens {
                    sampledIDs.append(assignment.id)
                }
                guard
                    let sampled = applyExecutedAssignment(
                        id: assignment.id,
                        count: assignment.numTokens,
                        startComputedTokens: row.computedTokens,
                        knownTokensBeforeStep: row.knownTokens)
                else {
                    return .unbounded(reason: .invalidProjectionTransition)
                }
                targetSampled = targetSampled || sampled
            }
            guard chargeStep(projectedAssignments) else {
                return .unbounded(reason: .projectionArithmetic)
            }
            if targetSampled {
                return boundedProjection()
            }
            guard
                chargeTerminalChainedStepIfNeeded(
                    sampledIDs: sampledIDs,
                    allowsChainedSuccessor: inFlightAllowsChainedSuccessor)
            else {
                return .unbounded(reason: .chainedStepUnprojectable)
            }
            removeTerminatedRows()
        } else {
            removeTerminatedRows()
        }

        // A request can carry very large output budgets. Decode-only full-slot
        // stretches are jumped exactly below; this guard bounds adversarial
        // prompt/chunk shapes so projection itself cannot monopolize the
        // engine queue. Failing closed is safer than returning a partial sum.
        let maxProjectionIterations = 32_768
        var iterations = 0

        while rows[id] != nil {
            iterations += 1
            guard iterations <= maxProjectionIterations else {
                return .unbounded(reason: .iterationLimit)
            }

            // Exact acceleration for the common "all slots are decode/paused,
            // target waits" stretch. Every active plain-decode row gets one
            // token per step when the step budget can seat all of them. An MTP
            // stretch whose mirrored depth is zero is plain decode too: its row
            // count only falls when the first row finishes, where the jump ends.
            if waitingIDs.contains(id),
                runningIDs.count >= config.maxConcurrentRequests
            {
                let activeDecodeIDs = runningIDs.filter { runningID in
                    guard let row = rows[runningID], !row.isPaused else { return false }
                    return row.remainingKnownTokens == 1
                }
                let allRunningDecodeOrPaused = runningIDs.allSatisfy { runningID in
                    guard let row = rows[runningID] else { return false }
                    return row.isPaused || row.remainingKnownTokens == 1
                }
                if speculationPlanner == nil
                    || speculationDepthLimit(decodeRows: activeDecodeIDs.count) == 0,
                    allRunningDecodeOrPaused,
                    !activeDecodeIDs.isEmpty,
                    activeDecodeIDs.count <= config.maxBatchedTokensPerStep
                {
                    let jump =
                        activeDecodeIDs.compactMap { runningID -> Int? in
                            guard let row = rows[runningID] else { return nil }
                            return row.maxTokens - row.generatedTokens
                        }.min() ?? 0
                    guard jump > 0 else {
                        return .unbounded(reason: .projectionArithmetic)
                    }
                    for runningID in activeDecodeIDs {
                        guard var row = rows[runningID],
                            appendCapacityReservation(
                                id: runningID,
                                start: row.computedTokens,
                                count: jump),
                            let generated = Self.projectionAdd(row.generatedTokens, jump),
                            let known = Self.projectionAdd(row.knownTokens, jump),
                            let computed = Self.projectionAdd(row.computedTokens, jump)
                        else {
                            return .unbounded(reason: .invalidProjectionTransition)
                        }
                        row.generatedTokens = generated
                        row.knownTokens = known
                        row.computedTokens = computed
                        rows[runningID] = row
                    }
                    guard chargeDecodeStretch(rows: activeDecodeIDs.count, steps: jump) else {
                        return .unbounded(reason: .projectionArithmetic)
                    }
                    settleInterleave(hasDecodeRows: true, promptRows: 0, steps: jump)
                    guard
                        chargeTerminalChainedStepIfNeeded(
                            sampledIDs: activeDecodeIDs,
                            allowsChainedSuccessor: true)
                    else {
                        return .unbounded(reason: .chainedStepUnprojectable)
                    }
                    removeTerminatedRows()
                    continue
                }
            }

            let baseBudget = config.maxBatchedTokensPerStep
            let liveRunningIDs = runningIDs.filter { runningID in
                guard let row = rows[runningID] else { return false }
                return !row.cancelRequested && row.remainingKnownTokens > 0
            }
            let soloStripe: (id: CBv2RequestID, tokens: Int)? = {
                guard let stripe = config.soloPrefillStripeTokens,
                    stripe > config.prefillChunkSize
                else {
                    return nil
                }
                let candidateID: CBv2RequestID?
                if liveRunningIDs.count + waitingIDs.count == 1 {
                    candidateID = liveRunningIDs.first ?? waitingIDs.first
                } else if liveRunningIDs.count <= 1,
                    !liveRunningIDs.contains(where: { runningID in
                        guard let row = rows[runningID] else { return false }
                        return !row.isPaused && row.remainingKnownTokens == 1
                    })
                {
                    candidateID = liveRunningIDs.first ?? waitingIDs.first
                } else {
                    candidateID = nil
                }
                guard let candidateID, let candidate = rows[candidateID],
                    !candidate.isPaused,
                    !candidate.cancelRequested,
                    candidate.remainingKnownTokens > 1
                else {
                    return nil
                }
                guard
                    let selectedStripe = config.resolvedSoloPrefillStripeTokens(
                        isMultimodal: candidate.isMultimodal)
                else { return nil }
                return (candidateID, selectedStripe)
            }()

            var budget = max(baseBudget, soloStripe?.tokens ?? 0)
            var totalAssignedTokens = 0
            var prefillTokensAssigned = 0
            var midPrefillAssigned = 0
            var assignments: [CBv2ProjectionAssignment] = []

            let decodeReadyRows = runningIDs.filter { runningID in
                guard let row = rows[runningID] else { return false }
                return !row.isPaused && !row.cancelRequested
                    && row.remainingKnownTokens == 1
            }.count
            let hasDecodeWork = decodeReadyRows > 0
            // A held plan runs its decode rows and no prompt work.
            let promptWorkHeld = hasDecodeWork && projectedDecodeOnlyStepsOwed > 0
            var promptRowsAssigned = 0
            let speculationLimit = speculationDepthLimit(decodeRows: decodeReadyRows)
            let prefillCap = hasDecodeWork ? mixedStepPrefillTokenCap.map { max(0, $0) } : nil

            func prefillHeadroom() -> Int {
                guard let prefillCap else { return .max }
                return max(0, prefillCap - prefillTokensAssigned)
            }

            func chunkCap(for rowID: CBv2RequestID) -> Int {
                rows[rowID]?.prefixReusePlan?.recurrentChunkSize
                    ?? (soloStripe?.id == rowID ? soloStripe!.tokens : config.prefillChunkSize)
            }

            func prefixClamp(
                row: inout CBv2ProjectionRow,
                proposed: Int
            ) -> Int {
                let range = config.demandedCheckpointRange(
                    promptTokens: row.promptTokens, hintTokens: row.checkpointTargetTokens,
                    computedTokens: row.computedTokens, proposed: proposed,
                    armedSoloStripeTokens: soloStripe?.id == row.id ? soloStripe?.tokens : nil,
                    hasPrefixReuse: row.prefixReusePlan != nil,
                    requestAllowsCheckpoint: row.requestAllowsCheckpoint,
                    continuation: row.demandedCheckpointContinuation)
                let count =
                    row.prefixReusePlan?.clampedChunk(
                        start: row.computedTokens,
                        proposed: range.count) ?? range.count
                row.demandedCheckpointContinuation = count == range.count ? range.continuation : nil
                return count
            }

            // RUNNING first, preserving authoritative scheduler order.
            for runningID in runningIDs where budget > 0 {
                guard var row = rows[runningID],
                    !row.isPaused,
                    !row.cancelRequested
                else {
                    continue
                }
                let remaining = row.remainingKnownTokens
                guard remaining > 0 else { continue }
                let isPrefill = remaining > 1
                if isPrefill, midPrefillAssigned >= 1 || promptWorkHeld {
                    continue
                }

                var count: Int
                if isPrefill {
                    count = min(remaining, chunkCap(for: runningID), budget)
                    if prefillCap != nil {
                        let headroom = prefillHeadroom()
                        guard headroom > 0 else { continue }
                        count = min(count, headroom)
                    }
                    count = prefixClamp(row: &row, proposed: count)
                    if count == 0, row.prefixReusePlan?.recurrentChunkSize != nil {
                        // The live scheduler may cold-restart after bounded
                        // geometry waits. A prefix-only bound would underprice
                        // that fallback, so this projection must fail closed.
                        return .unbounded(reason: .prefixGeometryBlocked)
                    }
                } else {
                    // The live MTP planner mutates controller/round state and
                    // cannot be called from this pure projection. Charge its
                    // configured 1+k ceiling (at most the tested width), with
                    // the scheduler's exact plain-decode fallback when that
                    // width does not fit the remaining step budget. A mirrored
                    // depth gate can only lower the ceiling, to zero at most.
                    if speculationPlanner != nil, speculationLimit != 0 {
                        guard let upperBound = speculationDraftTokenUpperBound else {
                            return .unbounded(reason: .speculationBoundMissing)
                        }
                        guard
                            let speculativeWidth = Self.projectionAdd(
                                1, min(max(0, upperBound), speculationLimit ?? .max))
                        else {
                            return .unbounded(reason: .projectionArithmetic)
                        }
                        count = speculativeWidth <= budget ? speculativeWidth : 1
                    } else {
                        count = 1
                    }
                }
                guard count > 0 else { continue }

                guard
                    let assignment = projectionAssignment(
                        id: runningID,
                        count: count,
                        row: row)
                else {
                    return .unbounded(reason: .invalidProjectionAssignment)
                }
                assignments.append(assignment)
                guard let computed = Self.projectionAdd(row.computedTokens, count),
                    let assigned = Self.projectionAdd(totalAssignedTokens, count)
                else {
                    return .unbounded(reason: .projectionArithmetic)
                }
                row.computedTokens = computed
                rows[runningID] = row
                budget -= count
                totalAssignedTokens = assigned
                if isPrefill {
                    guard
                        let prefillAssigned = Self.projectionAdd(
                            prefillTokensAssigned, count)
                    else {
                        return .unbounded(reason: .projectionArithmetic)
                    }
                    prefillTokensAssigned = prefillAssigned
                    promptRowsAssigned += 1
                    if row.knownTokens - row.computedTokens > 1 {
                        midPrefillAssigned += 1
                    }
                }
            }

            // WAITING admission. Cancel-pending rows are absent from this
            // projection because the engine processes cancellation before its
            // next plan. Paused waiters are skipped but retain queue position.
            var waitingIndex = 0
            while budget > 0,
                runningIDs.count < config.maxConcurrentRequests,
                waitingIndex < waitingIDs.count
            {
                let waitingID = waitingIDs[waitingIndex]
                guard var row = rows[waitingID] else {
                    waitingIDs.remove(at: waitingIndex)
                    continue
                }
                if row.isPaused {
                    waitingIndex += 1
                    continue
                }
                guard midPrefillAssigned < 1, !promptWorkHeld else { break }
                let admissionHeadroom = prefillHeadroom()
                guard admissionHeadroom > 0 else { break }

                let normalHeadroom =
                    soloStripe?.id == waitingID
                    ? budget
                    : max(0, baseBudget - totalAssignedTokens)
                var count = min(
                    row.remainingKnownTokens,
                    chunkCap(for: waitingID),
                    budget,
                    normalHeadroom,
                    admissionHeadroom)
                count = prefixClamp(row: &row, proposed: count)
                if count == 0, row.prefixReusePlan?.recurrentChunkSize != nil {
                    return .unbounded(reason: .prefixGeometryBlocked)
                }
                guard count > 0 else { break }

                guard
                    let assignment = projectionAssignment(
                        id: waitingID,
                        count: count,
                        row: row)
                else {
                    return .unbounded(reason: .invalidProjectionAssignment)
                }
                assignments.append(assignment)
                guard let computed = Self.projectionAdd(row.computedTokens, count),
                    let assigned = Self.projectionAdd(totalAssignedTokens, count),
                    let prefillAssigned = Self.projectionAdd(prefillTokensAssigned, count)
                else {
                    return .unbounded(reason: .projectionArithmetic)
                }
                row.computedTokens = computed
                rows[waitingID] = row
                budget -= count
                totalAssignedTokens = assigned
                prefillTokensAssigned = prefillAssigned
                promptRowsAssigned += 1
                waitingIDs.remove(at: waitingIndex)
                runningIDs.append(waitingID)
                if row.knownTokens - row.computedTokens > 1 {
                    midPrefillAssigned += 1
                }
            }

            guard !assignments.isEmpty else {
                return .unbounded(reason: .noSchedulingProgress)
            }
            for assignment in assignments {
                guard
                    appendCapacityReservation(
                        id: assignment.id,
                        start: assignment.startComputedTokens,
                        count: assignment.count)
                else {
                    return .unbounded(reason: .invalidPrefixReservation)
                }
            }
            guard chargeStep(assignments) else {
                return .unbounded(reason: .projectionArithmetic)
            }
            settleInterleave(hasDecodeRows: hasDecodeWork, promptRows: promptRowsAssigned)

            // Finalize the whole projected step before testing the target.
            // Later queue rows admitted beside the target's final chunk are
            // already included in `totalAssignedTokens` above.
            var targetSampled = false
            var sampledIDs: [CBv2RequestID] = []
            for assignment in assignments {
                if assignment.count
                    >= assignment.knownTokensBeforeStep - assignment.startComputedTokens
                {
                    sampledIDs.append(assignment.id)
                }
                guard
                    let sampled = applyExecutedAssignment(
                        id: assignment.id,
                        count: assignment.count,
                        startComputedTokens: assignment.startComputedTokens,
                        knownTokensBeforeStep: assignment.knownTokensBeforeStep)
                else {
                    return .unbounded(reason: .invalidProjectionTransition)
                }
                targetSampled = targetSampled || sampled
            }
            if targetSampled {
                return boundedProjection()
            }
            // Future MTP decisions are intentionally not invoked from this
            // pure projection. A projected terminal step may therefore be a
            // plain fallback that chains; charging one successor is safe.
            // When live MTP suppresses it, false rejection is bounded by one
            // decode batch at the slot-release boundary.
            guard
                chargeTerminalChainedStepIfNeeded(
                    sampledIDs: sampledIDs,
                    allowsChainedSuccessor: true)
            else {
                return .unbounded(reason: .chainedStepUnprojectable)
            }
            removeTerminatedRows()
        }

        return .unbounded(reason: .targetNotSampled)
    }

    private static func projectionAdd(_ lhs: Int, _ rhs: Int) -> Int? {
        let (value, overflow) = lhs.addingReportingOverflow(rhs)
        return overflow ? nil : value
    }

    private static func projectionMultiply(_ lhs: Int, _ rhs: Int) -> Int? {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        return overflow ? nil : value
    }
}
