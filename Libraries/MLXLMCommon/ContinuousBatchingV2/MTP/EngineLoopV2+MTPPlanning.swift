// EngineLoopV2+MTPPlanning.swift
//
// MTP row eligibility and step-global scheduler planning.

extension EngineLoopV2 {

    /// Per-row hard gates. Ineligible rows remain ordinary target rows and do
    /// not contribute MTP skip metrics.
    ///
    /// The historical `temperature == 0` gate is lifted when the drafter
    /// opted into target-prefix acceptance AND the installed sampler can
    /// pre-sample verify positions with the request's real sampler semantics
    /// (`CBv2StepSampler.mtpVerifySample`): every committed token is then a
    /// genuine target sample, which is exact for the output distribution at
    /// any temperature/top-p/top-k/min-p. Opted-in constrained rows instead
    /// use the ordinary sampler on their committed prefix at finalization
    /// (grammar verification), and opted-in free spans accept only pure
    /// same-state previews. Presence/frequency penalties pass only under the
    /// penalty opt-in (`mtpPenaltiesEligible`), stop strings only under the
    /// stop-string opt-in (`effectiveStopStringSpeculation`; finalize confirms
    /// token by token through the detokenizer). Logprob capture, bias and
    /// repetition penalty keep their gates.
    func mtpBasicEligible(_ rec: CBv2ScheduledRequest) -> Bool {
        mtpHistoryEligible(rec)
            && (rec.request.tokenConstraint == nil || grammarMTPEnabled
                || mtpFreeSpan(rec) != nil)
    }

    /// Protected spans still own trusted history for later free-span rounds.
    /// Whole-request exclusions apply to both history and drafting.
    func mtpHistoryEligible(_ rec: CBv2ScheduledRequest) -> Bool {
        // Opted-in native media is target-only for its complete immutable
        // request lifetime, including text tails, preemption and recompute.
        mtp?.registerTargetOnlyMedia(rec.request)
        if mtp?.requiresTargetOnlyMedia(rec.request) == true { return false }
        let sampling = rec.request.sampling
        let samplingEligible =
            sampling.temperature == 0
            || (mtp?.targetPrefixAcceptance == true && sampler.supportsMTPTargetPrefix)
        let constraintEligible =
            rec.request.tokenConstraint == nil || grammarMTPEnabled
            || mtpSpanConstraint(rec) != nil
        // A reasoning budget keeps the whole request target-only: no drafts
        // and no assistant history.
        guard rec.request.reasoningBudget == nil,
            constraintEligible,
            samplingEligible,
            sampling.topLogprobs == 0,
            sampling.logitBias.isEmpty,
            sampling.repetitionPenalty == 1,
            mtpPenaltiesEligible(rec),
            rec.request.stopStrings.isEmpty
                || mtp?.config.effectiveStopStringSpeculation == true
        else { return false }
        // The current multimodal seam does not return trusted pre-norm
        // hidden rows from embedding-spliced forwards. Only persistent-history
        // drafters require those rows; stateless Gemma keeps its established
        // frozen-KV multimodal path.
        let hasMultimodalSpans =
            multimodalByID[rec.id]?.spans.isEmpty == false
        if !Self.mtpMultimodalHistoryEligible(
            hasSpans: hasMultimodalSpans,
            tracksPersistentHistory: mtp?.tracksPersistentHistory == true)
        {
            return false
        }
        return true
    }

    /// Presence/frequency penalties live in the sampler's incremental row
    /// state. A penalty row may draft only when the engine opted in and the
    /// kill switch is not set (`CBv2MTPConfig.effectivePenaltySpeculation`), the drafter opted into
    /// target-prefix acceptance and the sampler re-derives the penalties per
    /// verify position (`supportsMTPPenaltyVerification`; such rows always
    /// accept exactly). Constrained rows keep the historical exclusion.
    func mtpPenaltiesEligible(_ rec: CBv2ScheduledRequest) -> Bool {
        let sampling = rec.request.sampling
        guard CBv2MTPVerifyPenalties.applies(sampling) else { return true }
        return Self.mtpPenaltyRowEligible(
            switchOn: mtp?.config.effectivePenaltySpeculation == true,
            targetPrefix: mtp?.targetPrefixAcceptance == true && sampler.supportsMTPTargetPrefix,
            penaltyVerification: sampler.supportsMTPPenaltyVerification,
            constrained: rec.request.tokenConstraint != nil,
            finite: sampling.presencePenalty.isFinite && sampling.frequencyPenalty.isFinite)
    }

    static func mtpPenaltyRowEligible(
        switchOn: Bool, targetPrefix: Bool, penaltyVerification: Bool,
        constrained: Bool, finite: Bool
    ) -> Bool {
        switchOn && targetPrefix && penaltyVerification && !constrained && finite
    }

    static func mtpMultimodalHistoryEligible(
        hasSpans: Bool, tracksPersistentHistory: Bool
    ) -> Bool {
        !hasSpans || !tracksPersistentHistory
    }

    /// Only the explicit native fallback may remove media/ineligible rows from
    /// the head batch count, and only in a cohort actually containing media, a
    /// logprob row or, on a row-exact recurrent fallback target, any other
    /// ineligible row (stop strings or penalties without their opt-ins, logit bias). Such a row never
    /// drafts; its one-token decode beside verifying rows is exact (row-exact
    /// decode, ordinary sampler), so the other rows keep speculating.
    /// All-row target costs, token/KV headroom and reservations stay unchanged.
    /// With no opt-in or such row, return the historical cohort verbatim.
    private func mtpDraftBatchRows(_ rows: [CBv2ScheduledRequest]) -> [CBv2ScheduledRequest] {
        guard let mtp,
            rows.contains(where: {
                mtp.requiresTargetOnlyMedia($0.request)
                    || (mtp.supportsRequestScopedMediaFallback
                        && $0.request.sampling.topLogprobs != 0)
                    || (mtp.ineligibleRowsDecodeBesideSpeculation && !mtpBasicEligible($0))
            })
        else {
            return rows
        }
        return rows.filter { rec in
            guard mtpBasicEligible(rec), let state = kvStates[rec.id] else { return false }
            return Self.mtpStorageEligible(state)
        }
    }

    /// Every storage-owning row must support value-exact multi-token writes
    /// and rollback. Kept internal for backend contract tests.
    static func mtpStorageEligible(_ state: [CBv2SequenceKV?]) -> Bool {
        state.allSatisfy { $0?.supportsSpeculativeWrites ?? true }
    }

    private enum CBv2MTPPlanAction {
        case round(k: Int)
        case seed
        case none
    }

    private func mtpPlanAction(
        for rec: CBv2ScheduledRequest, recordSkips: Bool
    ) -> CBv2MTPPlanAction {
        guard let mtp else { return .none }
        guard mtpHistoryEligible(rec) else { return .none }
        let depth = mtp.planDepth
        guard let state = kvStates[rec.id] else { return .none }
        guard Self.mtpStorageEligible(state) else {
            if recordSkips { mtp.recordSkip("kv_unsupported") }
            return .none
        }

        let remainingToLength = rec.request.maxTokens - rec.generatedTokenCount
        if !mtpBasicEligible(rec) {
            return mtp.tracksPersistentHistory && remainingToLength > 0 ? .seed : .none
        }
        if mtp.forceSeedPlan {
            return remainingToLength > 0 ? .seed : .none
        }
        if depth == 0 {
            return mtp.tracksPersistentHistory && remainingToLength > 0 ? .seed : .none
        }

        // Verify writes 1+k entries at [C-1, C-1+k]. A seed only pays off
        // when a full round can follow after its one generated token.
        switch mtp.validatedCarry(for: rec) {
        case .valid:
            guard remainingToLength >= depth else {
                if recordSkips { mtp.recordSkip("max_tokens") }
                return .none
            }
            return .round(k: depth)
        case .stale:
            if recordSkips { mtp.recordSkip("carry_invalid") }
            fallthrough
        case .none:
            guard remainingToLength >= depth + 1 else { return .none }
            return .seed
        }
    }

    /// Scheduler speculation hook for one decode-ready running row.
    func mtpPlanSpeculation(for rec: CBv2ScheduledRequest) -> Int {
        guard let mtp else { return 0 }
        switch mtpPlanAction(for: rec, recordSkips: true) {
        case .round(let k):
            mtp.markRound(rec.id, k: k)
            return k
        case .seed:
            mtp.markSeed(rec.id)
            return 0
        case .none:
            return 0
        }
    }

    /// Break the chained fast path when the next step must seed or verify.
    func mtpWantsStep(ids: [CBv2RequestID]) -> Bool {
        guard let mtp else { return false }
        let rows = ids.compactMap { scheduler.record(for: $0) }
        if mtp.config.fixedDraftTokens == 0, mtp.usesRequestStatefulDrafter {
            return false
        }
        let batchRows = mtpDraftBatchRows(rows)
        // Preserve the old ids.count guard for unresolved/stale ids too.
        let withinBatchGate =
            batchRows.count + (ids.count - rows.count) <= mtp.config.maxSpeculativeBatch
        let canSpeculate =
            withinBatchGate && rows.count == ids.count
            && mtpRowsCanSpeculate(batchRows)
        let decision = mtp.previewDecision(
            plannedDecodeRows: ids.count, canSpeculate: canSpeculate)
        let eligible = rows.filter { rec in
            guard mtpHistoryEligible(rec), let state = kvStates[rec.id] else { return false }
            return Self.mtpStorageEligible(state)
        }
        guard !eligible.isEmpty else { return false }
        // Every positive-cap stateful Qwen request owns trusted target
        // history, including fixed-depth requests temporarily forced to k=0
        // by batch/headroom pressure. Route those target-only steps through
        // the hidden-returning MTP path. Explicit fixed k=0 has no history
        // ownership and remains on ordinary chained decode. A drafter that
        // takes ordinary-decode history keeps chaining while no row can
        // speculate and, when it opts in to depth-zero chaining, while the
        // next plan would run at depth zero (`mtpPreviewRunsAsOrdinaryDecode`);
        // the chained launch captures hidden for finalize.
        if mtp.tracksPersistentHistory {
            guard mtpFeedsHistoryFromOrdinaryDecode else { return true }
            guard canSpeculate, mtp.depthZeroChaining else { return canSpeculate }
            return !mtpPreviewRunsAsOrdinaryDecode()
        }
        if decision.depth == 0 {
            if mtp.config.verificationMode == .automatic,
                mtp.maximumAutomaticDepth(plannedDecodeRows: ids.count) == 0
            {
                return false
            }
            return canSpeculate && mtp.requiresNonChainedDepthZeroProbe(decision)
        }
        guard canSpeculate else { return false }
        return eligible.allSatisfy { rec in
            let remaining = rec.request.maxTokens - rec.generatedTokenCount
            return mtp.hasValidCarry(for: rec)
                ? remaining >= 1
                : remaining >= 2
        }
    }

    /// A depth-k round verifies and can emit k+1 target tokens. Reserve one
    /// output slot for the mandatory target token, even when a carry already
    /// exists. A draft with only one output slot left cannot avoid any target
    /// work. Apply this bound to every offer, including fixed and exploration
    /// depths, using the shortest row's budget for the common rectangle.
    static func mtpDepthWithinOutputBudget(
        offeredDepth: Int, remainingTokens: [Int]
    ) -> Int {
        guard offeredDepth > 0, !remainingTokens.isEmpty else { return 0 }
        return remainingTokens.reduce(offeredDepth) { depth, remaining in
            min(depth, remaining > 1 ? remaining - 1 : 0)
        }
    }

    /// Decode rows of the next scheduler plan, as `beginMTPPlan` counts them.
    private func mtpPlanDecodeRows() -> [CBv2ScheduledRequest] {
        scheduler.running.filter { !$0.isPaused && !$0.cancelRequested && $0.isDecodeReady }
    }

    private func mtpPlanEligibleRows(
        _ rows: [CBv2ScheduledRequest]
    ) -> [CBv2ScheduledRequest] {
        rows.filter { rec in
            guard mtpBasicEligible(rec), let state = kvStates[rec.id] else { return false }
            return Self.mtpStorageEligible(state)
        }
    }

    /// The per-row clamps `beginMTPPlan` applies to an offered decision, in
    /// order. Each clamp lowers the depth and, like `clampPlanDepth`, makes
    /// the decision a non-exploration one. `clamp` receives every applied
    /// clamp (nil for a preview, which leaves the driver untouched). The one
    /// definition keeps the chain predicate and the plan in step.
    private func mtpClampedPlanDecision(
        _ decision: CBv2MTPDepthDecision, rows: [CBv2ScheduledRequest],
        eligibleRows: [CBv2ScheduledRequest], driver mtp: CBv2MTPRoundDriver,
        clamp: ((Int, String) -> Void)?
    ) -> (depth: Int, isExploration: Bool) {
        var depth = decision.depth
        var isExploration = decision.isExploration
        func lower(to requested: Int, reason: String) {
            let newDepth = min(max(requested, 0), depth)
            guard newDepth != depth else { return }
            depth = newDepth
            isExploration = false
            clamp?(newDepth, reason)
        }
        if mtp.usesMarginalPolicy, !decision.isExploration, depth > 0, !eligibleRows.isEmpty {
            let verificationLimit = mtp.maximumAutomaticDepth(plannedDecodeRows: rows.count)
            let offered = depth
            let marginalDepth =
                eligibleRows.map { rec in
                    mtp.marginalDepth(
                        for: rec.id,
                        offeredDepth: offered,
                        remainingTokens: rec.request.maxTokens - rec.generatedTokenCount,
                        verificationLimit: verificationLimit,
                        decodeRowBucket: decision.decodeRowBucket)
                }.min() ?? 0
            lower(to: marginalDepth, reason: "marginal_policy")
        }
        if depth > 0 {
            lower(
                to: Self.mtpDepthWithinOutputBudget(
                    offeredDepth: depth,
                    remainingTokens: eligibleRows.map {
                        $0.request.maxTokens - $0.generatedTokenCount
                    }),
                reason: "tail_depth")
        }
        if depth > 0 {
            let eligibleIDs = Set(eligibleRows.map(\.id))
            let stepTokens = rows.count + eligibleRows.count * depth
            let capacityTokens = rows.reduce(0) { total, rec in
                let count = 1 + (eligibleIDs.contains(rec.id) ? depth : 0)
                return total
                    + rec.capacityTokensForChunk(start: rec.numComputedTokens, count: count)
            }
            if stepTokens > scheduler.config.maxBatchedTokensPerStep {
                lower(to: 0, reason: "step_token_budget")
            } else if !(capacity?.hasHeadroom(additionalTokens: capacityTokens) ?? true) {
                lower(to: 0, reason: "step_kv_headroom")
            }
        }
        if depth > 0, let limit = mtp.planDepthLimitForTesting,
            let cap = limit(rows.map { $0.generatedTokenCount + $0.pendingSamples })
        {
            lower(to: cap, reason: "test_schedule")
        }
        return (depth, isExploration)
    }

    /// Whether the next plan would run as ordinary decode feeding history
    /// (`CBv2MTPRoundDriver.planRunsAsOrdinaryDecode`), computed without
    /// touching the driver: the same rows, gates, offered decision and clamps
    /// as `beginMTPPlan`, which runs right after on the chained path.
    private func mtpPreviewRunsAsOrdinaryDecode() -> Bool {
        guard let mtp else { return false }
        let rows = mtpPlanDecodeRows()
        let batchRows = mtpDraftBatchRows(rows)
        let canSpeculate =
            batchRows.count <= mtp.config.maxSpeculativeBatch && mtpRowsCanSpeculate(batchRows)
        guard canSpeculate else { return true }
        let decision = mtp.previewPlanDecision(
            plannedDecodeRows: rows.count, canSpeculate: canSpeculate)
        let planned = mtpClampedPlanDecision(
            decision, rows: rows, eligibleRows: mtpPlanEligibleRows(rows), driver: mtp,
            clamp: nil)
        return planned.depth == 0 && !planned.isExploration
    }

    /// Select one controller depth for all decode rows in the scheduler plan.
    /// Chunked-prefill neighbors do not change the controller batch bucket.
    func beginMTPPlan() {
        guard let mtp else { return }
        let rows = mtpPlanDecodeRows()
        let batchRows = CBv2StepProfiler.time("v2.plan.mtp.batchRows") { mtpDraftBatchRows(rows) }
        let withinBatchGate = batchRows.count <= mtp.config.maxSpeculativeBatch
        let canSpeculate = CBv2StepProfiler.time("v2.plan.mtp.canSpeculate") {
            withinBatchGate && mtpRowsCanSpeculate(batchRows)
        }
        CBv2StepProfiler.time("v2.plan.mtp.begin") {
            mtp.beginPlan(
                plannedDecodeRows: rows.count, canSpeculate: canSpeculate,
                rowIDs: rows.map(\.id))
        }
        let eligibleRows = mtpPlanEligibleRows(rows)
        _ = mtpClampedPlanDecision(
            mtp.planDecision, rows: rows, eligibleRows: eligibleRows, driver: mtp
        ) { depth, reason in
            mtp.clampPlanDepth(to: depth, reason: reason)
        }
        if mtp.planDepth > 0,
            eligibleRows.contains(where: { !mtp.hasValidCarry(for: $0) })
        {
            // Seeding is step-global. Keep existing persistent histories
            // resident and perform one ordinary hidden-returning target step
            // for every eligible row rather than destroying their carries.
            mtp.forceSynchronizedSeed()
            mtp.recordControllerFallback("synchronized_seed")
        }

        for rec in rows {
            let storageEligible = kvStates[rec.id].map(Self.mtpStorageEligible) ?? false
            if mtpBasicEligible(rec), kvStates[rec.id] != nil, !storageEligible {
                mtp.recordSkip("kv_unsupported")
            }
            let keepsHistory = mtp.tracksPersistentHistory && mtpHistoryEligible(rec)
            if (!mtpBasicEligible(rec) && !keepsHistory) || !storageEligible
                || (mtp.planDepth == 0 && !mtp.tracksPersistentHistory)
            {
                mtp.invalidateCarry(rec.id)
            }
        }
        if !rows.isEmpty, !withinBatchGate {
            for _ in batchRows { mtp.recordSkip("batch_gate") }
        }
    }

    private func mtpRowsCanSpeculate(_ rows: [CBv2ScheduledRequest]) -> Bool {
        !rows.isEmpty
            && rows.allSatisfy { rec in
                guard mtpBasicEligible(rec), let state = kvStates[rec.id] else { return false }
                return Self.mtpStorageEligible(state)
            }
    }

    /// True when this scheduler plan carries seed or verify work.
    func mtpRoundNeeded(_ plan: CBv2StepPlan) -> Bool {
        guard let mtp else { return false }
        if mtpPlanFeedsHistoryFromOrdinaryDecode(plan) { return false }
        if mtp.planHasMTPWork {
            return plan.assignments.contains {
                mtp.roundMark(for: $0.id) != nil || mtp.isSeedMarked($0.id)
            }
        }
        guard mtp.tracksPersistentHistory else { return false }
        return plan.assignments.contains { assignment in
            guard let rec = scheduler.record(for: assignment.id),
                mtpHistoryEligible(rec)
                    || mtpRecordsPromptHistory(rec, range: Self.mtpPlannedRange(rec, assignment))
            else { return false }
            // Prompt rows acquire their KV state inside execution. Absence
            // here is not ineligibility; route them through the hidden-returning
            // MTP prefill path so persistent history starts at prompt position 0.
            guard let state = kvStates[assignment.id] else { return true }
            return Self.mtpStorageEligible(state)
        }
    }

    /// A plan that cannot speculate or runs at depth zero
    /// (`planRunsAsOrdinaryDecode`), whose history-owning rows are all plain
    /// decode rows, needs no MTP step: `executeMixed` captures their hidden
    /// and the step may anchor a chain. Prompt rows keep the MTP prefill path
    /// so persistent history starts at prompt position 0.
    private func mtpPlanFeedsHistoryFromOrdinaryDecode(_ plan: CBv2StepPlan) -> Bool {
        guard mtpFeedsHistoryFromOrdinaryDecode, let mtp,
            !mtp.planCanSpeculate || mtp.planRunsAsOrdinaryDecode
        else {
            return false
        }
        return plan.assignments.allSatisfy { assignment in
            guard let rec = scheduler.record(for: assignment.id) else { return true }
            guard mtpHistoryEligible(rec) else {
                return !mtpRecordsPromptHistory(
                    rec, range: Self.mtpPlannedRange(rec, assignment))
            }
            guard let state = kvStates[assignment.id] else { return false }
            return !Self.mtpStorageEligible(state)
                || (assignment.numTokens == 1 && rec.generatedTokenCount > 0
                    && rec.numComputedTokens == rec.tokens.count)
        }
    }

    /// The token range a plan assigned (the scheduler already advanced it).
    private static func mtpPlannedRange(
        _ rec: CBv2ScheduledRequest, _ assignment: (id: CBv2RequestID, numTokens: Int)
    ) -> Range<Int> {
        rec.numComputedTokens - assignment.numTokens ..< rec.numComputedTokens
    }

    /// A request that cannot speculate still records the assistant's prompt
    /// history over `range` when that prompt can leave a prefix checkpoint.
    /// The history is a function of the prompt alone, and a checkpoint
    /// without it could not be restored by a request that speculates. Only a
    /// first-pass text prompt of a drafter that opted in
    /// (`checkpointsNonSpeculatingRows`); such a row never drafts or stores a
    /// carry, and `beginMTPPlan` releases its state once it is decode-ready.
    func mtpRecordsPromptHistory(_ rec: CBv2ScheduledRequest, range: Range<Int>) -> Bool {
        guard let mtp, mtp.tracksPersistentHistory,
            hybridPrefixCache != nil || completeCheckpointCapture?.codec.assistant != nil,
            (mtp.drafter as? any CBv2MTPPrefixCheckpointDrafter)?.checkpointsNonSpeculatingRows
                == true,
            rec.request.prefixCacheEnabled, !rec.request.hasOutOfBandCheckpointInput,
            rec.preemptionCount == 0
        else { return false }
        // The final prompt token alone is a decode row whose state is released.
        let promptCount = rec.request.promptTokens.count
        return range.upperBound < promptCount
            || (range.upperBound == promptCount && range.count > 1)
    }
}
