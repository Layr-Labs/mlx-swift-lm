// Copyright © 2026 Eigen Labs.

import MLX

struct CBv2MTPGrammarRow {
    let batchIndex: Int
    var context: CBv2SamplerRow
    let stopTokens: Set<Int>
    let nonFinite: [Bool]
}

struct CBv2MTPGrammarPrefix {
    var tokens: [Int] = []
    var accepted = 0
    var failure: String?
    var nonFinite = false
    var finished = false
}

/// One grammar column on a native-tracked engine.
enum CBv2MTPNativeGrammarColumn {
    case tokens([Int])
    /// The column's work completed but was refused; its rows end with this error.
    case failed(String)
    /// The engine faulted: nothing more may run or commit.
    case halted
}

extension EngineLoopV2 {
    /// The same opt-in governs verification and ordinary seed/fallback state.
    /// A native-tracked engine joins only when its contract qualifies it.
    var grammarMTPEnabled: Bool {
        mtp?.config.constrainedSpeculation == true
            && (nativeShutdownState == nil || nativeConstrainedMTPVerification)
    }

    /// Mask each committed position with the ordinary sampler. `nextState`
    /// may intern states, so discarded drafts must never advance it. Returned
    /// tokens have already been confirmed and must commit even before an error.
    /// nil: a native engine faulted mid-walk, so nothing more may commit.
    func mtpSampleAndConfirmConstrainedPrefixes(
        logits: MLXArray, rows: [CBv2MTPGrammarRow], draftPacket: [Int32], depth: Int,
        commonLimit: Int, independentWidths: Bool, step: CBv2InFlightStep,
        nativeCommitHeld: inout Bool
    ) -> [CBv2RequestID: CBv2MTPGrammarPrefix]? {
        precondition(nativeShutdownState == nil || nativeConstrainedMTPVerification)
        var rows = rows
        var prefixes = rows.map { _ in CBv2MTPGrammarPrefix() }
        let limit = independentWidths ? depth + 1 : commonLimit
        columns: for column in 0 ..< limit {
            var active: [Int] = []
            var failedAtColumn = false
            for index in rows.indices where !prefixes[index].finished {
                if rows[index].nonFinite[column] {
                    prefixes[index].nonFinite = true
                    prefixes[index].finished = true
                    failedAtColumn = true
                } else {
                    active.append(index)
                }
            }
            // Earlier columns have already advanced the grammar. Commit that
            // prefix even on failure, and stop common-width peers BEFORE they
            // confirm a token that reconciliation would later discard. At
            // column zero the failed row has no prefix and can simply retire.
            if failedAtColumn && column > 0 && !independentWidths { break }
            guard !active.isEmpty else { break }
            let contexts = active.map { rows[$0].context }
            let ids = contexts.map(\.id)
            let positionLogits = concatenated(
                active.map {
                    let batch = rows[$0].batchIndex
                    return logits[batch ..< batch + 1, column, 0...]
                }, axis: 0)
            let sampled = sampler.sample(
                logits: positionLogits, params: contexts.map(\.params), requestIDs: ids,
                stepIndex: stepCount, pendingSampledTokens: nil, rowContext: { contexts })
            let tokens: [Int]
            if nativeShutdownState != nil {
                switch mtpNativeGrammarColumn(
                    sampled, column: column, step: step, commitHeld: &nativeCommitHeld)
                {
                case .tokens(let values):
                    tokens = values
                case .failed(let reason):
                    for index in active {
                        prefixes[index].failure = reason
                        prefixes[index].finished = true
                    }
                    break columns
                case .halted:
                    return nil
                }
            } else {
                tokens = sampled.asArray(Int32.self).map(Int.init)
            }
            CBv2CoreInstrumentation.recordHostSync()
            sampler.confirmSampledTokens(tokens, requestIDs: ids)

            var commonFinished = false
            for (index, token) in zip(active, tokens) {
                let row = rows[index]
                prefixes[index].tokens.append(token)
                rows[index].context.outputTokens.append(token)
                let failure = sampler.tokenConstraintFailure(for: row.context.id)
                prefixes[index].failure = failure
                let matches =
                    column < depth
                    && token == Int(draftPacket[row.batchIndex * depth + column])
                if matches && failure == nil { prefixes[index].accepted += 1 }
                let finished =
                    !matches || failure != nil || row.stopTokens.contains(token)
                    || rows[index].context.outputTokens.count >= row.context.maxTokens
                prefixes[index].finished = finished
                commonFinished = commonFinished || finished
            }
            if !independentWidths && commonFinished { break }
        }
        return Dictionary(uniqueKeysWithValues: zip(rows.map(\.context.id), prefixes))
    }

    /// Submit and read back one column under the native work protocol, never
    /// under finalize's commit lock, which is held again on return unless halted.
    func mtpNativeGrammarColumn(
        _ sampled: MLXArray, column: Int, step: CBv2InFlightStep, commitHeld: inout Bool
    ) -> CBv2MTPNativeGrammarColumn {
        guard let tracking = nativeShutdownState else {
            preconditionFailure("CBv2 MTP: native grammar column without native tracking")
        }
        if commitHeld {
            tracking.endCommit()
            commitHeld = false
        }
        do {
            if let handle = retainNativeWork([sampled]) { step.nativeRootIDs.append(handle) }
            try requireNativeWork()
            try withError { error in
                eval(sampled)
                try error.check()
            }
        } catch {
            // Completion is unknown, so the engine cannot vouch for this work.
            if tracking.mayExecute { failNativeCompletion(.nativeWorkFailed) }
            return .halted
        }
        var refusal: Error?
        do {
            try nativeWorkSubmitted()
            try mtpGrammarColumnSubmittedForTesting?(column)
        } catch {
            refusal = error
        }
        var tokens: [Int] = []
        if refusal == nil {
            guard completedNativeReadback({ tokens = sampled.asArray(Int32.self).map(Int.init) })
            else { return .halted }
        }
        guard tracking.beginCommit() else { return .halted }
        commitHeld = true
        if let refusal { return .failed("grammar verification: \(refusal)") }
        return .tokens(tokens)
    }
}
