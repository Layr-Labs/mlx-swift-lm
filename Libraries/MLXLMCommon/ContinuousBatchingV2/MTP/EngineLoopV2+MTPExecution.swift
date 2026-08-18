// EngineLoopV2+MTPExecution.swift
//
// MTP row classification and lazy MLX graph construction.

import Foundation
import MLX

struct CBv2MTPRowWork {
    let rec: CBv2ScheduledRequest
    let start: Int
    let count: Int
    let samples: Bool
    let isDecode: Bool
    let isSeed: Bool
    /// Non-nil for verify rows: the consumed carry.
    let carry: CBv2MTPCarry?
    /// Verify-produced carry transition that must prime assistant history
    /// before this target-only row processes the carry token.
    let historyCarry: CBv2MTPCarry?
}

struct CBv2MTPGraphBuild {
    let sampledRows: [CBv2RequestID]
    let sampledTokens: MLXArray?
    /// Non-sampling prefill handles retained by the in-flight step.
    let prefillEvalTargets: [MLXArray]
    let asyncEvalTargets: [MLXArray]
    let logprobSegments: [CBv2StepLogprobs]
    let verify: CBv2MTPRoundInFlight.Verify?
    let seedRows: [(id: CBv2RequestID, decodeIndex: Int)]
    let seedHidden: MLXArray?
    let seedPolicyTopTwoValues: MLXArray?
    let recurrentEvaluations: [CBv2RequestID: CBv2RecurrentStateEvaluation]
    let committedObservationRows: [CBv2MTPRoundInFlight.CommittedObservationRow]
}

extension EngineLoopV2 {

    /// Record only the assignments the scheduler demoted. Known capacity
    /// reasons retain their provenance; an unclassified width mismatch means
    /// the reservation changed after MTP planning and is one step-level race.
    func mtpRecordSchedulerDemotions(_ plan: CBv2StepPlan) {
        guard let mtp else { return }
        var sawReservationRace = false
        for assignment in plan.assignments {
            guard let k = mtp.roundMark(for: assignment.id),
                assignment.numTokens != 1 + k
            else { continue }
            switch plan.speculationFallbacks[assignment.id] {
            case .tokenBudget: mtp.recordSkip("token_budget")
            case .kvHeadroom: mtp.recordSkip("kv_headroom")
            case nil: sawReservationRace = true
            }
        }
        if sawReservationRace {
            mtp.recordControllerFallback("step_reservation_race")
        }
    }

    func mtpPrepareRoundWork(
        _ plan: CBv2StepPlan,
        driver mtp: CBv2MTPRoundDriver,
        demoteAllRounds: Bool
    ) -> [CBv2MTPRowWork] {
        var work: [CBv2MTPRowWork] = []
        work.reserveCapacity(plan.assignments.count)

        for (id, assignedTokens) in plan.assignments {
            guard let rec = scheduler.record(for: id) else { continue }
            guard ensureKVState(rec) != nil else { continue }
            var count = assignedTokens
            var preserveHistorySeed = false

            if let k = mtp.roundMark(for: id) {
                if demoteAllRounds {
                    if count == 1 + k { scheduler.rollbackComputed(id: id, tokens: k) }
                    count = 1
                    if mtp.tracksPersistentHistory {
                        preserveHistorySeed = true
                    } else {
                        mtp.invalidateCarry(id)
                    }
                } else if count == 1 + k, let carry = mtp.consumeCarry(for: id) {
                    work.append(
                        CBv2MTPRowWork(
                            rec: rec, start: rec.numComputedTokens - count, count: count,
                            samples: true, isDecode: false, isSeed: false,
                            carry: carry, historyCarry: nil))
                    continue
                } else if count == 1 + k {
                    // A marked assignment without a consumable carry demotes
                    // exactly like the scheduler's headroom retry.
                    scheduler.rollbackComputed(id: id, tokens: k)
                    count = 1
                    preserveHistorySeed = mtp.tracksPersistentHistory
                }
            }

            // Mirror executeMixed's row classification. MTP decode-shaped
            // work stays eager; final-token image spans remain prefill work.
            let start = rec.numComputedTokens - count
            let samples = rec.numComputedTokens == rec.effectiveTokenCount
            let finalTokenIsImageSpan =
                multimodalByID[id]?.containsSpan(at: rec.tokens.count - 1) ?? false
            let isDecode =
                count == 1 && samples && start == rec.tokens.count - 1 && !finalTokenIsImageSpan
            work.append(
                CBv2MTPRowWork(
                    rec: rec, start: start, count: count, samples: samples,
                    isDecode: isDecode,
                    isSeed:
                        isDecode && (mtp.isSeedMarked(id) || preserveHistorySeed),
                    carry: nil,
                    historyCarry: mtp.pendingHistoryCarry(for: id)))
        }
        return work
    }

    func mtpBuildRoundGraph(
        _ work: [CBv2MTPRowWork], driver mtp: CBv2MTPRoundDriver
    ) -> CBv2MTPGraphBuild {
        var cacheInnerState: [MLXArray] = []
        var logprobSegments: [CBv2StepLogprobs] = []

        // Plain and seed rows share one eager [B, 1] target batch. Seed rows
        // retain the pre-norm hidden; logits remain identical to plain eager.
        let decodeRows = work.filter(\.isDecode)
        var decodeSampled: MLXArray?
        var seedRows: [(id: CBv2RequestID, decodeIndex: Int)] = []
        var seedHidden: MLXArray?
        var seedPolicyTopTwoValues: MLXArray?
        var recurrentEvaluations: [CBv2RequestID: CBv2RecurrentStateEvaluation] = [:]
        var committedObservationRows: [CBv2MTPRoundInFlight.CommittedObservationRow] = []
        var committedObservationEvalTargets: [MLXArray] = []

        func observeCommittedTarget(
            row: CBv2MTPRowWork, tokens: MLXArray, hidden: MLXArray
        ) {
            guard mtp.tracksPersistentHistory, mtpBasicEligible(row.rec),
                let state = mtp.takeOrMakeAssistantState(for: row.rec.id)
            else { return }
            if let carry = row.historyCarry {
                mtp.observeCommittedTarget(
                    id: row.rec.id,
                    observation: CBv2MTPCommittedTargetObservation(
                        tokens: MLXArray([Int32(carry.token)]).reshaped([1, 1]),
                        hidden: carry.hidden),
                    detachedState: state)
                committedObservationEvalTargets.append(carry.hidden)
            }
            mtp.observeCommittedTarget(
                id: row.rec.id,
                observation: CBv2MTPCommittedTargetObservation(
                    tokens: tokens, hidden: hidden),
                detachedState: state)
            committedObservationRows.append(
                .init(id: row.rec.id, assistantState: state))
            committedObservationEvalTargets.append(hidden)
        }
        if !decodeRows.isEmpty {
            let inputs = MLXArray(decodeRows.map { Int32($0.rec.tokens[$0.start]) })
                .reshaped([decodeRows.count, 1])
            let caches = eagerCaches(rowStates: decodeRows.map { kvStates[$0.rec.id]! })
            let logits: MLXArray
            let hidden: MLXArray
            if let recurrentModel = mtp.model as? any CBv2RecurrentMTPSteppableModel,
                recurrentModel.recurrentStateSpec != nil
            {
                let evaluations = decodeRows.map { row -> CBv2RecurrentStateEvaluation in
                    guard let state = recurrentStates[row.rec.id] else {
                        preconditionFailure("CBv2 recurrent MTP seed state missing")
                    }
                    do { return try state.bind() } catch {
                        preconditionFailure("CBv2 recurrent MTP seed bind failed: \(error)")
                    }
                }
                let positionIds = CBv2PositionState.decodePositionIds(
                    states: decodeRows.map(\.rec.request.positionState),
                    cacheOffsets: decodeRows.map { Self.positionOffset(kvStates[$0.rec.id]!) })
                let output = recurrentModel.forwardWithHidden(
                    tokens: inputs, caches: caches, recurrentState: evaluations,
                    positionIds: positionIds)
                logits = output.logits
                hidden = output.lastHidden
                for (row, evaluation) in zip(decodeRows, evaluations) {
                    do {
                        cacheInnerState.append(contentsOf: try evaluation.evaluate())
                    } catch {
                        preconditionFailure("CBv2 recurrent MTP seed evaluation failed: \(error)")
                    }
                    recurrentEvaluations[row.rec.id] = evaluation
                }
            } else {
                let output = mtp.model.forwardWithHidden(tokens: inputs, caches: caches)
                logits = output.logits
                hidden = output.lastHidden
            }
            cacheInnerState.append(contentsOf: eagerCacheInnerState(caches))
            decodeSampled = sampler.sample(
                logits: logits[0..., -1, 0...],
                params: decodeRows.map(\.rec.request.sampling),
                requestIDs: decodeRows.map(\.rec.id),
                stepIndex: stepCount,
                pendingSampledTokens: nil,
                rowContext: { decodeRows.map { Self.samplerRow($0.rec) } })
            if let stepLogprobs = sampler.takeStepLogprobs() {
                logprobSegments.append(stepLogprobs)
            }
            for (index, row) in decodeRows.enumerated() where row.isSeed {
                seedRows.append((id: row.rec.id, decodeIndex: index))
            }
            if !seedRows.isEmpty { seedHidden = hidden }
            if mtp.usesMarginalPolicy, !seedRows.isEmpty {
                guard let provider = mtp.model as? any CBv2MTPPolicyTopTwoProviding else {
                    preconditionFailure("CBv2 adaptive seed target lacks top-two provider")
                }
                let vocabulary = logits.dim(-1)
                let topTwo = provider.cbv2MTPTopTwo(
                    logits.reshaped([1, decodeRows.count, vocabulary]))
                seedPolicyTopTwoValues = topTwo.values
                    .reshaped([decodeRows.count, 1, 2])
                    .asType(.float32)
            }
            for (index, row) in decodeRows.enumerated() {
                observeCommittedTarget(
                    row: row,
                    tokens: inputs[index ..< index + 1, 0...],
                    hidden: hidden[index ..< index + 1, 0..., 0...])
            }
        }

        // Chunked prefills remain per-request [1, chunk], matching executeMixed.
        var prefillSampled: [CBv2RequestID: MLXArray] = [:]
        var prefillEvalTargets: [MLXArray] = []
        for row in work where !row.isDecode && row.carry == nil {
            let rec = row.rec
            let slice = rec.tokens[row.start ..< row.start + row.count]
            let inputs = MLXArray(slice.map(Int32.init)).reshaped([1, row.count])
            let caches = eagerCaches(rowStates: [kvStates[rec.id]!])
            let requirement: CBv2PrefillRequirement =
                row.samples ? .lastPositionLogits : .evaluationOnly
            let output: MLXArray
            var observedHidden: MLXArray?
            if let multimodal = multimodalByID[rec.id],
                !multimodal.spansInChunk(start: row.start, count: row.count).isEmpty
            {
                let forward = multimodalChunkForward(
                    tokens: inputs, start: row.start, count: row.count,
                    id: rec.id, multimodal: multimodal, caches: caches,
                    requirement: requirement)
                output = forward.output
                cacheInnerState.append(contentsOf: forward.innerState)
                recurrentEvaluations.merge(forward.recurrent) { _, _ in
                    preconditionFailure("duplicate recurrent evaluation")
                }
            } else if mtp.tracksPersistentHistory,
                let recurrentModel = mtp.model as? any CBv2RecurrentMTPSteppableModel,
                recurrentModel.recurrentStateSpec != nil
            {
                guard let recurrentState = recurrentStates[rec.id] else {
                    preconditionFailure("CBv2 recurrent MTP prefill state missing")
                }
                let evaluation: CBv2RecurrentStateEvaluation
                do { evaluation = try recurrentState.bind() } catch {
                    preconditionFailure("CBv2 recurrent MTP prefill bind failed: \(error)")
                }
                let positions = rec.request.positionState?.promptSlice(
                    row.start ..< row.start + row.count)
                let forward = recurrentModel.forwardWithHidden(
                    tokens: inputs, caches: caches, recurrentState: [evaluation],
                    positionIds: positions)
                output = narrowPrefillOutput(forward.logits, requirement: requirement)
                observedHidden = forward.lastHidden
                do {
                    cacheInnerState.append(contentsOf: try evaluation.evaluate())
                } catch {
                    preconditionFailure(
                        "CBv2 recurrent MTP prefill evaluation failed: \(error)")
                }
                recurrentEvaluations[rec.id] = evaluation
            } else if let recurrentModel = model as? any CBv2RecurrentSteppableModel,
                recurrentModel.recurrentStateSpec != nil
            {
                let positions = rec.request.positionState?.promptSlice(
                    row.start ..< row.start + row.count)
                let forward = targetForward(
                    tokens: inputs, caches: caches, ids: [rec.id],
                    positionIds: positions)
                output = narrowPrefillOutput(forward.logits, requirement: requirement)
                cacheInnerState.append(contentsOf: forward.innerState)
                recurrentEvaluations.merge(forward.recurrent) { _, _ in
                    preconditionFailure("duplicate recurrent evaluation")
                }
            } else if mtp.tracksPersistentHistory {
                let forward = mtp.model.forwardWithHidden(tokens: inputs, caches: caches)
                output = narrowPrefillOutput(forward.logits, requirement: requirement)
                observedHidden = forward.lastHidden
            } else {
                output = prefillOutput(
                    tokens: inputs, inputEmbeddings: nil, caches: caches,
                    requirement: requirement)
            }
            if let observedHidden {
                observeCommittedTarget(row: row, tokens: inputs, hidden: observedHidden)
            }
            cacheInnerState.append(contentsOf: eagerCacheInnerState(caches))
            if row.samples {
                prefillSampled[rec.id] = sampler.sample(
                    logits: output,
                    params: [rec.request.sampling],
                    requestIDs: [rec.id],
                    stepIndex: stepCount,
                    pendingSampledTokens: nil,
                    rowContext: { [Self.samplerRow(rec)] })
                if let stepLogprobs = sampler.takeStepLogprobs() {
                    logprobSegments.append(stepLogprobs)
                }
            } else {
                prefillEvalTargets.append(output)
            }
        }

        let verifyRows = work.filter { $0.carry != nil }
        let verify = mtpBuildVerifyGraph(
            verifyRows, driver: mtp, cacheInnerState: &cacheInnerState)

        // Plain sampled tokens stay in plan order. Verify rows are finalized
        // from the target-authoritative acceptance packet instead.
        var pieces: [MLXArray] = []
        var sampledRows: [CBv2RequestID] = []
        var decodeIndex = 0
        for row in work {
            if row.isDecode {
                pieces.append(decodeSampled![decodeIndex ..< decodeIndex + 1])
                decodeIndex += 1
                sampledRows.append(row.rec.id)
            } else if let sampled = prefillSampled[row.rec.id] {
                pieces.append(sampled)
                sampledRows.append(row.rec.id)
            }
        }
        let sampledTokens: MLXArray? =
            pieces.isEmpty ? nil : (pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 0))

        var asyncEvalTargets = prefillEvalTargets
        if let sampledTokens { asyncEvalTargets.append(sampledTokens) }
        for segment in logprobSegments {
            asyncEvalTargets.append(contentsOf: segment.evalTargets)
        }
        if let verify {
            asyncEvalTargets.append(verify.acceptancePacket)
            asyncEvalTargets.append(verify.lastHidden)
            if let shortlistIDs = verify.shortlistIDs {
                asyncEvalTargets.append(shortlistIDs)
            }
            if let policyTopTwoValues = verify.policyTopTwoValues {
                asyncEvalTargets.append(policyTopTwoValues)
            }
        }
        if let seedHidden { asyncEvalTargets.append(seedHidden) }
        if let seedPolicyTopTwoValues {
            asyncEvalTargets.append(seedPolicyTopTwoValues)
        }
        asyncEvalTargets.append(contentsOf: committedObservationEvalTargets)
        if !cacheInnerState.isEmpty {
            asyncEvalTargets.append(contentsOf: cacheInnerState)
            offsetChainEvalSteps += 1
        }

        return CBv2MTPGraphBuild(
            sampledRows: sampledRows,
            sampledTokens: sampledTokens,
            prefillEvalTargets: prefillEvalTargets,
            asyncEvalTargets: asyncEvalTargets,
            logprobSegments: logprobSegments,
            verify: verify,
            seedRows: seedRows,
            seedHidden: seedHidden,
            seedPolicyTopTwoValues: seedPolicyTopTwoValues,
            recurrentEvaluations: recurrentEvaluations,
            committedObservationRows: committedObservationRows)
    }

    private func mtpBuildVerifyGraph(
        _ verifyRows: [CBv2MTPRowWork],
        driver mtp: CBv2MTPRoundDriver,
        cacheInnerState: inout [MLXArray]
    ) -> CBv2MTPRoundInFlight.Verify? {
        guard !verifyRows.isEmpty else { return nil }
        let draftStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        let depths = Set(verifyRows.compactMap { mtp.roundMark(for: $0.rec.id) })
        precondition(depths.count == 1, "CBv2 MTP: one plan must use one uniform depth")
        let k = depths.first!
        let batch = verifyRows.count
        var captures: [CBv2MTPRowCapture] = []
        var rowMetadata: [CBv2MTPRoundInFlight.VerifyRow] = []
        var seedTokens: [Int32] = []
        var carryHiddens: [MLXArray] = []
        // Each capture paired with the row it was gathered from, so
        // `mtpFreezeCaptures` can fence it against that row's own storage.
        var captured: [(row: CBv2SequenceKV, keys: MLXArray, values: MLXArray)] = []
        captures.reserveCapacity(batch)
        captured.reserveCapacity(2 * batch)

        for row in verifyRows {
            let state = kvStates[row.rec.id]!
            let carry = row.carry!
            if let captureLayers = mtp.captureLayers, !mtp.usesRequestStatefulDrafter {
                // Capture before target verification writes the speculative
                // columns; paged storage is fenced below before those writes.
                let fullRow = state[captureLayers.full]!
                let slidingRow = state[captureLayers.sliding]!
                precondition(
                    fullRow.absoluteOffset == carry.kvOffset,
                    "CBv2 MTP: verify row anchor \(fullRow.absoluteOffset) != carry \(carry.kvOffset)"
                )
                let fullSnapshot = fullRow.snapshot()
                let slidingSnapshot = slidingRow.snapshot()
                captures.append(
                    CBv2MTPRowCapture(
                        fullKeys: fullSnapshot.keys,
                        fullValues: fullSnapshot.values,
                        slidingKeys: slidingSnapshot.keys,
                        slidingValues: slidingSnapshot.values,
                        slidingStart: slidingRow.absoluteOffset - slidingRow.retainedCount,
                        anchor: fullRow.absoluteOffset))
                captured.append((fullRow, fullSnapshot.keys, fullSnapshot.values))
                captured.append((slidingRow, slidingSnapshot.keys, slidingSnapshot.values))
            } else {
                precondition(
                    state.compactMap { $0 }.allSatisfy { $0.absoluteOffset == carry.kvOffset },
                    "CBv2 request-stateful MTP target KV is not aligned with its carry")
            }
            rowMetadata.append(
                CBv2MTPRoundInFlight.VerifyRow(
                    id: row.rec.id, storageRows: state.compactMap { $0 },
                    assistantState:
                        mtp.usesRequestStatefulDrafter
                        ? mtp.takeAssistantState(for: row.rec.id) : nil))
            seedTokens.append(Int32(carry.token))
            carryHiddens.append(carry.hidden)
        }

        mtpFreezeCaptures(captured)
        let seedColumn = MLXArray(seedTokens).reshaped([batch, 1])
        var draftSteps: [MLXArray] = []
        draftSteps.reserveCapacity(k)
        var assistantEvalTargets: [MLXArray] = []
        if let stateful = mtp.drafter as? any CBv2MTPRequestStatefulDrafter {
            var currentTokens = (0 ..< batch).map {
                seedColumn[$0 ..< $0 + 1, 0...]
            }
            var currentHidden = carryHiddens
            for draftIndex in 0 ..< k {
                var nextRows: [MLXArray] = []
                var nextHiddens: [MLXArray] = []
                var stepEvalTargets: [MLXArray] = []
                nextRows.reserveCapacity(batch)
                nextHiddens.reserveCapacity(batch)
                for (index, row) in verifyRows.enumerated() {
                    guard let requestState = rowMetadata[index].assistantState else {
                        preconditionFailure(
                            "CBv2 request-stateful MTP assistant state missing for \(row.rec.id)")
                    }
                    let result = stateful.draftStep(
                        tokens: currentTokens[index],
                        hidden: currentHidden[index],
                        shortlist: draftIndex == 0 ? row.carry!.shortlist : nil,
                        requestState: requestState)
                    let next = result.tokens.reshaped([1])
                    nextRows.append(next)
                    nextHiddens.append(result.hidden)
                    stepEvalTargets.append(next)
                    stepEvalTargets.append(result.hidden)
                    stepEvalTargets.append(
                        contentsOf: stateful.evaluationTargets(for: requestState))
                }
                // Publish the first mutable head-cache generation before
                // constructing a deeper generation. This is nonblocking and
                // joins the round's sole finalize fence.
                if draftIndex == 0 { asyncEval(stepEvalTargets) }
                assistantEvalTargets.append(contentsOf: stepEvalTargets)
                let stepTokens = concatenated(nextRows, axis: 0)
                draftSteps.append(stepTokens)
                currentTokens = nextRows.map { $0.reshaped([1, 1]) }
                currentHidden = nextHiddens
            }
        } else {
            let prepared = mtp.drafter.prepare(rows: captures)
            var draftInput = seedColumn
            var draftHidden = concatenated(carryHiddens, axis: 0)
            for _ in 0 ..< k {
                let (next, nextHidden) = mtp.drafter.draftStep(
                    tokens: draftInput, hidden: draftHidden, prepared: prepared)
                draftSteps.append(next)
                draftInput = next.reshaped([batch, 1])
                draftHidden = nextHidden
            }
        }
        let draftIDs = stacked(draftSteps, axis: 1)
        if CBv2StepProfiler.enabled {
            CBv2StepProfiler.record(
                "v2.mtp.draft.build", seconds: CFAbsoluteTimeGetCurrent() - draftStart)
        }

        // Windowed rows stage provisional writes; other supported storage
        // backends implement the transaction hooks as exact no-ops/rollback.
        let verifyStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        for metadata in rowMetadata {
            for sequence in metadata.storageRows { sequence.beginSpeculativeWrite() }
        }
        let targetColumns = [seedColumn] + draftSteps.map { $0.reshaped([batch, 1]) }
        let target = mtpBuildTargetVerification(
            columns: targetColumns, rows: verifyRows, driver: mtp)
        cacheInnerState.append(contentsOf: target.cacheInnerState)
        cacheInnerState.append(contentsOf: assistantEvalTargets)
        if CBv2StepProfiler.enabled {
            CBv2StepProfiler.record(
                "v2.mtp.verify.build", seconds: CFAbsoluteTimeGetCurrent() - verifyStart)
        }
        var packetParts = [draftIDs.reshaped([-1]), target.scores.reshaped([-1])]
        if let shortlist = target.shortlist {
            packetParts.append(shortlist.massScaled.reshaped([-1]))
        }
        let acceptancePacket = concatenated(packetParts, axis: 0)
        return CBv2MTPRoundInFlight.Verify(
            k: k,
            rows: rowMetadata,
            acceptancePacket: acceptancePacket,
            draftIDs: draftIDs,
            lastHidden: target.hidden,
            shortlistIDs: target.shortlist?.ids,
            recurrentEvaluations: target.recurrent,
            policyTopTwoValues: target.policyTopTwo?.values)
    }

    /// Freeze the round's pre-write KV captures against the in-place writes
    /// the very same graph is about to perform. See `CBv2MTPCaptureFence`
    /// for the hazard and the mechanism.
    private func mtpFreezeCaptures(
        _ captured: [(row: CBv2SequenceKV, keys: MLXArray, values: MLXArray)]
    ) {
        // Contiguous rows are ARC-owned by their views and need nothing.
        // `requiresMaterializedSnapshots` is the bit that already documents
        // exactly this recyclable-storage hazard: true for `PagedKVBackend`,
        // false everywhere else, so contiguous stays byte-identical.
        guard backend.requiresMaterializedSnapshots else { return }
        let unfenceable = CBv2MTPCaptureFence.publish(captured)
        if !unfenceable.isEmpty { eval(unfenceable) }
    }

}
