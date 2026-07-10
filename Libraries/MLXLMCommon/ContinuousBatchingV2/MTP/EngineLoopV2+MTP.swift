// EngineLoopV2+MTP.swift
//
// ContinuousBatchingV2 — the MTP (speculative decoding) round driver: the
// ONE new step kind, branched BEFORE `executeMixed` whenever the plan holds
// a 1+k decode assignment or a seed-marked row. `executeMixed` is never
// extended for speculation — its `rec.tokens` slicing and `samples`
// predicate are structurally wrong for speculative tokens — and nothing
// here goes anywhere near `CBv2CompiledDecodeV2.decodeStep` (it
// precondition-traps on L > 1); plain-decode steps keep the compiled path
// untouched.
//
// Round shape (engine thread, one asyncEval, one finalize host sync):
//   (a) carry-less eligible rows take a SEED step: their [B, 1] decode runs
//       through the model seam's `forwardWithHidden`, capturing the bonus
//       token + pre-norm hidden that establish the drafter carry;
//   (b) rows with a valid carry draft k tokens via `CBv2MTPDrafter`
//       (`prepare` over frozen pre-round KV snapshots, then k chained
//       `draftStep` [B, 1] forwards at frozen positions — zero engine-KV
//       writes);
//   (c) ONE rectangular [B, 1+k] verify through the eager
//       `forwardWithHidden` path, with `beginSpeculativeWrite()` armed on
//       every storage-owning row BEFORE the verify update (windowed rings
//       stage; full/quantized/paged-full are contract no-ops);
//   (d) the greedy accept-walk is DEFERRED to finalize: draft ids, verify
//       argmaxes and the verify hidden ride the step's asyncEval and
//       materialize at the same boundary `finalize()` already uses —
//       invariant 7 (one host sync per step) holds.
//
// MTP rounds NEVER chain (the chained path's finalize loop and
// deferredReleases assume exactly one sample per row); `engineStep` guards
// on `CBv2InFlightStep.mtpRound` and on `mtpWantsStep` so a chain breaks
// exactly when seeding/speculation should start.

import Foundation
import MLX

extension EngineLoopV2 {

    // MARK: - Eligibility

    /// Per-row hard gates that make a row round-eligible at all. Rows
    /// failing these are ordinary rows, not "skipped" (no metrics):
    ///  - pure-greedy sampling: temperature 0 AND no transform that could
    ///    move the argmax (logit bias, penalties). The accept-walk emits
    ///    RAW target argmaxes, so any transform-bearing row would lose
    ///    token-exactness vs plain decode;
    ///  - no logprobs (verify rows never run the sampler, so per-token
    ///    logprob capture cannot ride a round);
    ///  - not the final-prompt-token image-span decode shape (that row must
    ///    take the embedding-splice prefill path — see `executeMixed`).
    private func mtpBasicEligible(_ rec: CBv2ScheduledRequest) -> Bool {
        let sampling = rec.request.sampling
        guard sampling.temperature == 0,
            sampling.topLogprobs == 0,
            sampling.logitBias.isEmpty,
            sampling.repetitionPenalty == 1,
            sampling.frequencyPenalty == 0,
            sampling.presencePenalty == 0
        else { return false }
        if multimodalByID[rec.id]?.containsSpan(at: rec.tokens.count - 1) ?? false {
            return false
        }
        return true
    }

    /// Every storage-owning row must guarantee value-exact multi-token
    /// write + rollback (`supportsSpeculativeWrites`) — paged windowed
    /// rings do not, so such rows fall back to plain decode.
    private func mtpStorageEligible(_ state: [CBv2SequenceKV?]) -> Bool {
        state.allSatisfy { $0?.supportsSpeculativeWrites ?? true }
    }

    /// The action the planner would take for a decode-ready running row.
    /// `recordSkips: true` only from the scheduler's planner hook (metrics
    /// count row-steps clamped to plain decode); the chained-path pre-check
    /// passes false so it never double-counts.
    private enum CBv2MTPPlanAction {
        case round(k: Int)
        case seed
        case none
    }

    private func mtpPlanAction(
        for rec: CBv2ScheduledRequest, recordSkips: Bool
    ) -> CBv2MTPPlanAction {
        guard let mtp else { return .none }
        guard mtpBasicEligible(rec) else { return .none }
        // Static batch gate: above it the batched forward already amortizes
        // weight streaming and speculation stops paying for itself.
        guard scheduler.runningCount <= mtp.config.maxSpeculativeBatch else {
            if recordSkips { mtp.recordSkip("batch_gate") }
            return .none
        }
        guard let state = kvStates[rec.id] else { return .none }
        guard mtpStorageEligible(state) else {
            if recordSkips { mtp.recordSkip("kv_unsupported") }
            return .none
        }
        // A verify writes 1+k KV entries at positions [C-1, C-1+k]; the
        // sequence cap is maxLength = prompt + maxTokens, so a round needs
        // k ≤ maxTokens - generated. A seed only pays off if a round can
        // follow it (one more generated token first).
        let remainingToLength = rec.request.maxTokens - rec.generatedTokenCount
        switch mtp.validatedCarry(for: rec) {
        case .valid:
            guard remainingToLength >= mtp.k else {
                if recordSkips { mtp.recordSkip("max_tokens") }
                return .none
            }
            return .round(k: mtp.k)
        case .stale:
            if recordSkips { mtp.recordSkip("carry_invalid") }
            fallthrough
        case .none:
            guard remainingToLength >= mtp.k + 1 else { return .none }
            return .seed
        }
    }

    // MARK: - Scheduler hook (SchedulerV2.speculationPlanner)

    /// Consulted by `SchedulerV2.plan()` for each decode-ready RUNNING row.
    /// Returns the number of EXTRA draft slots to plan (0 = plain decode);
    /// side effect: plan-scoped round/seed marks that `executeMTPRound`
    /// classifies against. A reserve(1+k) failure is retried by the
    /// scheduler seam as plain decode (never a preemption) — the clamp is
    /// detected at execution (`roundMark` present, assignment == 1) and
    /// counted as "kv_headroom".
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

    /// Chained-path pre-check: true when the next step should be an MTP
    /// step (round or seed) for any of the chain-candidate rows. The
    /// chained fast path must then be skipped — a chained launch would
    /// bypass seeding forever and a 1+k plan can never chain.
    func mtpWantsStep(ids: [CBv2RequestID]) -> Bool {
        guard mtp != nil else { return false }
        return ids.contains { id in
            guard let rec = scheduler.record(for: id) else { return false }
            switch mtpPlanAction(for: rec, recordSkips: false) {
            case .round, .seed: return true
            case .none: return false
            }
        }
    }

    /// Branch predicate for `engineStep`: this plan carries MTP work.
    func mtpRoundNeeded(_ plan: CBv2StepPlan) -> Bool {
        guard let mtp, mtp.planHasMTPWork else { return false }
        return plan.assignments.contains {
            mtp.roundMark(for: $0.id) != nil || mtp.isSeedMarked($0.id)
        }
    }

    // MARK: - Round execution (the new step kind)

    /// Execute a plan containing MTP work: verify rounds for carry-bearing
    /// rows, seed decodes for carry-less eligible rows, and — mirroring
    /// `executeMixed` — plain [B, 1] decodes and per-request [1, chunk]
    /// prefills for every neighbor row in the same plan. All decode-shaped
    /// work in an MTP step runs EAGERLY (compiled decode is untouched and
    /// untouchable here).
    func executeMTPRound(_ plan: CBv2StepPlan) -> CBv2InFlightStep? {
        guard let mtp else { return executeMixed(plan) }  // defensive; unreachable

        struct RowWork {
            let rec: CBv2ScheduledRequest
            let start: Int
            let count: Int
            let samples: Bool
            let isDecode: Bool
            let isSeed: Bool
            /// Non-nil for verify rows: the consumed carry.
            let carry: CBv2MTPCarry?
        }

        let buildStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        var work: [RowWork] = []
        work.reserveCapacity(plan.assignments.count)
        for (id, assignedTokens) in plan.assignments {
            guard let rec = scheduler.record(for: id) else { continue }
            guard ensureKVState(rec) != nil else { continue }  // requeued / error-finished
            var n = assignedTokens
            if let k = mtp.roundMark(for: id) {
                if n == 1 + k, let carry = mtp.consumeCarry(for: id) {
                    work.append(
                        RowWork(
                            rec: rec, start: rec.numComputedTokens - n, count: n,
                            samples: true, isDecode: false, isSeed: false, carry: carry))
                    continue
                }
                if n == 1 + k {
                    // Defensive (a round mark implies a valid carry on this
                    // same thread): a 1+k assignment without a consumable
                    // carry demotes to plain decode — return the k
                    // speculative slots exactly like the scheduler's own
                    // headroom retry would have.
                    scheduler.rollbackComputed(id: id, tokens: k)
                    n = 1
                } else {
                    // The planner offered k but the scheduler clamped the
                    // reservation back to plain decode (KV headroom);
                    // n == 1 by construction.
                    mtp.recordSkip("kv_headroom")
                }
            }
            // Mirror of executeMixed's row classification (see its comments
            // for the samples predicate and the final-token image-span
            // shape) — pinned, not shared, so executeMixed stays untouched.
            let start = rec.numComputedTokens - n
            let samples = rec.numComputedTokens == rec.effectiveTokenCount
            let finalTokenIsImageSpan =
                multimodalByID[id]?.containsSpan(at: rec.tokens.count - 1) ?? false
            let isDecode =
                n == 1 && samples && start == rec.tokens.count - 1 && !finalTokenIsImageSpan
            work.append(
                RowWork(
                    rec: rec, start: start, count: n, samples: samples,
                    isDecode: isDecode, isSeed: isDecode && mtp.isSeedMarked(id), carry: nil))
        }
        guard !work.isEmpty else {
            // Launch abort after plan: undo the optimistic advance BEFORE
            // any pendingSamples could be marked (a leaked pendingSamples
            // blocks the whole waiting-admission loop). No-op for rows
            // whose records were already removed above.
            scheduler.rollback(plan)
            return nil
        }

        var cacheInnerState: [MLXArray] = []
        var logprobSegments: [CBv2StepLogprobs] = []

        // Rectangular [B, 1] decode batch (plain + seed rows), EAGER with
        // hidden capture: seed rows need the pre-norm hidden, and the
        // logits side of forwardWithHidden is contractually identical to
        // the plain forward, so plain neighbors ride the same batch.
        let decodeRows = work.filter(\.isDecode)
        var decodeSampled: MLXArray?
        var seedRows: [(id: CBv2RequestID, decodeIndex: Int)] = []
        var seedHidden: MLXArray?
        if !decodeRows.isEmpty {
            let inputs = MLXArray(decodeRows.map { Int32($0.rec.tokens[$0.start]) })
                .reshaped([decodeRows.count, 1])
            let caches = eagerCaches(rowStates: decodeRows.map { kvStates[$0.rec.id]! })
            let (logits, hidden) = mtp.model.forwardWithHidden(tokens: inputs, caches: caches)
            cacheInnerState.append(contentsOf: eagerCacheInnerState(caches))
            decodeSampled = sampler.sample(
                logits: logits[0..., -1, 0...],
                params: decodeRows.map(\.rec.request.sampling),
                requestIDs: decodeRows.map(\.rec.id),
                stepIndex: stepCount,
                pendingSampledTokens: nil,  // finalize preceded: all confirmed
                rowContext: { decodeRows.map { Self.samplerRow($0.rec) } })
            if let stepLogprobs = sampler.takeStepLogprobs() {
                logprobSegments.append(stepLogprobs)
            }
            for (index, row) in decodeRows.enumerated() where row.isSeed {
                seedRows.append((id: row.rec.id, decodeIndex: index))
            }
            if !seedRows.isEmpty { seedHidden = hidden }
        }

        // Per-request prefill chunks [1, chunk] — mirror of executeMixed.
        var prefillSampled: [CBv2RequestID: MLXArray] = [:]
        var evalTargets: [MLXArray] = []
        for row in work where !row.isDecode && row.carry == nil {
            let rec = row.rec
            let slice = rec.tokens[row.start ..< row.start + row.count]
            let inputs = MLXArray(slice.map(Int32.init)).reshaped([1, row.count])
            let caches = eagerCaches(rowStates: [kvStates[rec.id]!])
            let logits: MLXArray
            if let multimodal = multimodalByID[rec.id],
                let spanContext = multimodal.chunkContext(start: row.start, count: row.count)
            {
                logits = multimodalChunkForward(
                    tokens: inputs, start: row.start, count: row.count,
                    multimodal: multimodal, spanContext: spanContext, caches: caches)
            } else {
                logits = model.forward(tokens: inputs, caches: caches)
            }
            cacheInnerState.append(contentsOf: eagerCacheInnerState(caches))
            if row.samples {
                prefillSampled[rec.id] = sampler.sample(
                    logits: logits[0..., -1, 0...],
                    params: [rec.request.sampling],
                    requestIDs: [rec.id],
                    stepIndex: stepCount,
                    pendingSampledTokens: nil,
                    rowContext: { [Self.samplerRow(rec)] })
                if let stepLogprobs = sampler.takeStepLogprobs() {
                    logprobSegments.append(stepLogprobs)
                }
            } else {
                evalTargets.append(logits[0, row.count - 1, 0 ..< 1])
            }
        }

        // Verify rounds: draft k×[B, 1] against frozen pre-round captures,
        // then ONE rectangular [B, 1+k] eager verify.
        let verifyRows = work.filter { $0.carry != nil }
        var verify: CBv2MTPRoundInFlight.Verify?
        if !verifyRows.isEmpty {
            let draftStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
            let k = mtp.k
            let batch = verifyRows.count
            var captures: [CBv2MTPRowCapture] = []
            var rowMeta: [CBv2MTPRoundInFlight.VerifyRow] = []
            var seedTokens: [Int32] = []
            var carryHiddens: [MLXArray] = []
            captures.reserveCapacity(batch)
            for row in verifyRows {
                let state = kvStates[row.rec.id]!
                let carry = row.carry!
                // Frozen capture: snapshot views are graph-built BEFORE the
                // verify update, so MLX array versioning pins the drafter's
                // reads to the pre-round KV.
                let fullRow = state[mtp.captureLayers.full]!
                let slidingRow = state[mtp.captureLayers.sliding]!
                assert(
                    fullRow.absoluteOffset == carry.kvOffset,
                    "CBv2 MTP: verify row anchor \(fullRow.absoluteOffset) != carry \(carry.kvOffset)"
                )
                let fullSnap = fullRow.snapshot()
                let slidingSnap = slidingRow.snapshot()
                captures.append(
                    CBv2MTPRowCapture(
                        fullKeys: fullSnap.keys, fullValues: fullSnap.values,
                        slidingKeys: slidingSnap.keys, slidingValues: slidingSnap.values,
                        slidingStart: slidingRow.absoluteOffset - slidingRow.retainedCount,
                        anchor: fullRow.absoluteOffset))
                rowMeta.append(
                    CBv2MTPRoundInFlight.VerifyRow(
                        id: row.rec.id, storageRows: state.compactMap { $0 }))
                seedTokens.append(Int32(carry.token))
                carryHiddens.append(carry.hidden)
            }

            // (b) draft chain — frozen positions, zero engine-KV writes.
            let prepared = mtp.drafter.prepare(rows: captures)
            let seedColumn = MLXArray(seedTokens).reshaped([batch, 1])
            var draftInput = seedColumn
            var draftHidden = concatenated(carryHiddens, axis: 0)  // [B, 1, H]
            var draftSteps: [MLXArray] = []
            draftSteps.reserveCapacity(k)
            for _ in 0 ..< k {
                let (next, nextHidden) = mtp.drafter.draftStep(
                    tokens: draftInput, hidden: draftHidden, prepared: prepared)
                draftSteps.append(next)
                draftInput = next.reshaped([batch, 1])
                draftHidden = nextHidden
            }
            let draftIds = stacked(draftSteps, axis: 1)  // [B, k]
            if CBv2StepProfiler.enabled {
                CBv2StepProfiler.record(
                    "v2.mtp.draft.build", seconds: CFAbsoluteTimeGetCurrent() - draftStart)
            }

            // (c) arm staged writes BEFORE the verify update (windowed rings
            // stage; the rest are contract no-ops), then one rectangular
            // [B, 1+k] verify through the eager forwardWithHidden path.
            let verifyStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
            for meta in rowMeta {
                for sequence in meta.storageRows { sequence.beginSpeculativeWrite() }
            }
            let verifyTokens = concatenated(
                [seedColumn] + draftSteps.map { $0.reshaped([batch, 1]) }, axis: 1)
            let caches = eagerCaches(rowStates: verifyRows.map { kvStates[$0.rec.id]! })
            let (verifyLogits, verifyHidden) = mtp.model.forwardWithHidden(
                tokens: verifyTokens, caches: caches)
            cacheInnerState.append(contentsOf: eagerCacheInnerState(caches))
            // Emitted tokens come ONLY from these argmaxes (greedy
            // losslessness) — the sampler never sees verify rows.
            let targetArgmax = argMax(verifyLogits, axis: -1).asType(.int32)  // [B, 1+k]
            if CBv2StepProfiler.enabled {
                CBv2StepProfiler.record(
                    "v2.mtp.verify.build", seconds: CFAbsoluteTimeGetCurrent() - verifyStart)
            }
            verify = CBv2MTPRoundInFlight.Verify(
                k: k, rows: rowMeta, draftIds: draftIds,
                targetArgmax: targetArgmax, lastHidden: verifyHidden)
        }

        // Assemble the plain sampled tokens in plan order (verify rows are
        // excluded — their 1+k samples are confirmed by the MTP finalize).
        var pieces: [MLXArray] = []
        var sampledRows: [CBv2RequestID] = []
        var decodeIdx = 0
        for row in work {
            if row.isDecode {
                pieces.append(decodeSampled![decodeIdx ..< decodeIdx + 1])
                decodeIdx += 1
                sampledRows.append(row.rec.id)
            } else if let sampled = prefillSampled[row.rec.id] {
                pieces.append(sampled)
                sampledRows.append(row.rec.id)
            }
        }
        let sampledTokens: MLXArray? =
            pieces.isEmpty ? nil : (pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 0))

        scheduler.markPendingSamples(ids: sampledRows)
        if let verify {
            scheduler.markPendingSamples(
                counts: verify.rows.map { (id: $0.id, count: 1 + verify.k) })
        }
        mtp.recordSeedSteps(seedRows.count)

        var toEval = evalTargets
        if let sampledTokens { toEval.append(sampledTokens) }
        for segment in logprobSegments { toEval.append(contentsOf: segment.evalTargets) }
        if let verify {
            // The finalize sync materializes draft ids + argmaxes and reads
            // the (already-evaluated) hidden lazily — one boundary, no
            // second sync (invariant 7).
            toEval.append(verify.draftIds)
            toEval.append(verify.targetArgmax)
            toEval.append(verify.lastHidden)
        }
        if let seedHidden { toEval.append(seedHidden) }
        if !cacheInnerState.isEmpty {
            toEval.append(contentsOf: cacheInnerState)
            offsetChainEvalSteps += 1
        }
        asyncEval(toEval)
        if CBv2StepProfiler.enabled {
            CBv2StepProfiler.record(
                "v2.mtp.launch.total", seconds: CFAbsoluteTimeGetCurrent() - buildStart)
        }

        let step = CBv2InFlightStep(
            participants: Set(work.map(\.rec.id)),
            sampledRows: sampledRows,
            sampledTokens: sampledTokens,
            evalTargets: evalTargets)
        step.logprobSegments = logprobSegments
        step.mtpRound = CBv2MTPRoundInFlight(
            verify: verify, seedRows: seedRows, seedHidden: seedHidden)
        return step
    }

    // MARK: - Round finalize (accept-walk at the step's one host sync)

    /// Runs inside `finalize(_:)` AFTER the plain per-row loop (seed rows'
    /// bonus tokens are confirmed there) and BEFORE the deferred releases.
    /// The draft/argmax readbacks hit buffers the step's asyncEval already
    /// computed — same pattern as the logprob segments, no extra GPU work.
    func finalizeMTPRound(_ step: CBv2InFlightStep) {
        guard let mtp, let round = step.mtpRound else { return }

        // Seed carries: bonus token (confirmed by the plain loop just now) +
        // pre-norm hidden at the position before it.
        if let seedHidden = round.seedHidden {
            for (id, decodeIndex) in round.seedRows {
                guard !step.discard.contains(id),
                    let rec = scheduler.record(for: id)
                else { continue }
                mtp.storeCarry(
                    id: id, token: rec.tokens.last!,
                    hidden: seedHidden[decodeIndex ..< (decodeIndex + 1), 0..., 0...],
                    tokensCount: rec.tokens.count,
                    kvOffset: rec.numComputedTokens)
            }
        }

        guard let verify = round.verify else { return }
        let k = verify.k
        let draftHost = verify.draftIds.asArray(Int32.self)  // [B * k]
        let argmaxHost = verify.targetArgmax.asArray(Int32.self)  // [B * (1+k)]
        var anyRejected = false

        for (batchIndex, meta) in verify.rows.enumerated() {
            let id = meta.id
            // Mid-round cancel/deadline: the record's removal zeroed its
            // pendingSamples and the KV is fenced via deferredReleases —
            // nothing to correct here. The device offsets are stale for the
            // departed row, but its departure changes membership, forcing
            // an eager rebind anyway; flag it for uniformity.
            if step.discard.contains(id) || scheduler.record(for: id) == nil {
                anyRejected = true
                continue
            }
            let rec = scheduler.record(for: id)!
            let drafts = (0 ..< k).map { Int(draftHost[batchIndex * k + $0]) }
            let targets = (0 ..< (1 + k)).map { Int(argmaxHost[batchIndex * (1 + k) + $0]) }

            // Greedy accept-walk (Gemma4SpeculativeWalk semantics): accept
            // drafts while the target's argmax agrees; the emitted tokens
            // are target argmaxes ONLY (bonus/correction included).
            var accepted = 0
            while accepted < k, targets[accepted] == drafts[accepted] { accepted += 1 }
            let emitted = Array(targets[0 ... accepted])  // a+1 tokens

            // In-order confirm with mid-scan stop-token / stop-string /
            // maxTokens truncation — per-token semantics identical to the
            // plain finalize loop, so MTP-on output is token-exact.
            let detokenizer = detokenizers[id]
            let hasStopStrings = !rec.request.stopStrings.isEmpty
            var kept: [Int] = []
            var textPieces: [String] = []
            var finishReason: CBv2FinishReason?
            for token in emitted {
                scheduler.recordSampled(id: id, token: token)
                kept.append(token)
                if rec.request.stopTokens.contains(token) {
                    finishReason = .stop  // stop token: never detokenized
                    break
                }
                if hasStopStrings {
                    textPieces.append(detokenizer?.push([token]) ?? "")
                    if detokenizer?.matchedStopString == true {
                        finishReason = .stop
                        break
                    }
                }
                if rec.generatedTokenCount >= rec.request.maxTokens {
                    finishReason = .length
                    break
                }
            }

            // KV + scheduler corrections BEFORE any finishRequest (exit
            // path c): rollback the rejected/post-stop KV tail, commit the
            // staged windowed writes, drop the unconfirmed samples and
            // return their reservation.
            let confirmed = kept.count
            let rejected = (1 + k) - confirmed
            if rejected > 0 {
                for sequence in meta.storageRows { sequence.rollback(rejected) }
                anyRejected = true
            }
            for sequence in meta.storageRows { sequence.commitSpeculativeWrite() }
            if rejected > 0 {
                scheduler.discardPendingSamples(id: id, count: rejected)
                scheduler.rollbackComputed(id: id, tokens: rejected)
            }

            // ONE delta for the whole round (delta.tokens carries the raw
            // ids, stop token included; a stop token's text is never
            // rendered — same contract as the plain path).
            if hasStopStrings {
                stream(for: id)?.emit(
                    .delta(text: textPieces.joined(), tokens: kept, logprobs: nil))
            } else {
                let stream = stream(for: id)
                stream?.reserveEmission()
                let endsWithStopToken = finishReason == .stop
                let pushTokens = endsWithStopToken ? Array(kept.dropLast()) : kept
                let allTokens = kept
                detokQueue.async {
                    let text = pushTokens.isEmpty ? "" : (detokenizer?.push(pushTokens) ?? "")
                    stream?.emit(
                        .delta(text: text, tokens: allTokens, logprobs: nil),
                        consumingReservation: true)
                }
            }

            mtp.recordRound(drafted: k, accepted: accepted, emitted: confirmed)

            if let finishReason {
                finishRequest(id, reason: finishReason)
            } else if let deadline = rec.deadline, Date() > deadline {
                finishRequest(
                    id,
                    reason: .error("request exceeded \(Int(config.requestTimeout))s deadline"))
            } else {
                // Next carry: the new bonus is the last emitted token; the
                // hidden at the position before it is the verify hidden at
                // the accepted index — a lazy slice of an array that rode
                // this step's asyncEval (no extra sync).
                mtp.storeCarry(
                    id: id, token: kept[confirmed - 1],
                    hidden: verify.lastHidden[
                        batchIndex ..< (batchIndex + 1), (confirmed - 1) ..< confirmed, 0...],
                    tokensCount: rec.tokens.count,
                    kvOffset: rec.numComputedTokens)
            }
        }

        // Rejecting rounds leave the eager caches' ON-DEVICE positionOffsets
        // advanced by the full 1+k while the rows' host truth rolled back —
        // force the next eager bind to rebuild from host truth (the same
        // mechanism compiled steps use).
        if anyRejected {
            eagerCompositionStale = true
        }
    }
}
