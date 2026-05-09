// Port of mlx_lm.generate.GenerationBatch.
// https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py
//
// MTP (Multi-Token Prediction) dispatch is added at the bottom of this file
// as a private extension. It intercepts init/next/filter/extend when the
// model is MTPCapable and batchSize == 1.
// Port of omlx commit 696d90a: patches/mlx_lm_mtp/batch_generator.py

import Foundation
import MLX
import MLXRandom
import MLXNN

/// Picks one token per row from a `[B, vocab]` logits tensor.
public typealias RowSampler = @Sendable (MLXArray) -> MLXArray

/// Deterministic greedy sampler.
@Sendable public func greedySampler(_ logprobs: MLXArray) -> MLXArray {
    argMax(logprobs, axis: -1)
}

/// Per-row response from a single decode step.
///
/// Marked `@unchecked Sendable` because `promptCache` carries non-Sendable
/// `KVCacheSimple` references for cross-request prefix caching; cross-actor
/// transfer is the caller's responsibility.
public struct GenerationBatchResponse: @unchecked Sendable {
    public let uid: Int
    public let token: Int

    /// `"length"`, `"stop"`, or nil if the row is still generating.
    public let finishReason: String?

    /// The matched stop sequence if a multi-token stop completed on this token.
    public let matchedSequence: [Int]?

    /// State machine state name after this token's transition (nil = terminated).
    public let currentState: String?

    /// All produced tokens for this row. Set only on the final response.
    public let allTokens: [Int]?

    /// Single-row prompt cache for prefix caching across requests.
    /// Set only on the final response.
    public let promptCache: [any KVCache]?
}

/// Decode-phase batch over a shared `[any BatchedCache]` (one per layer).
/// Each layer's cache is the appropriate batched type for that layer:
/// `BatchKVCache` for full attention, `ArraysCache`/`MambaCache` for SSM.
/// Construct after prefill has populated the caches; call `next()` to
/// drive generation one step at a time.
public final class GenerationBatch: @unchecked Sendable {

    public let model: any LanguageModel
    public private(set) var uids: [Int]
    public private(set) var promptCache: [any BatchedCache]
    public private(set) var tokens: [[Int]]
    public private(set) var maxTokens: [Int]

    public private(set) var samplers: [RowSampler?]
    public let fallbackSampler: RowSampler
    public private(set) var stateMachines: [SequenceStateMachine]

    /// Tokens queued for the next model call. At construction this is the
    /// final prompt token for each row. After priming, and after every
    /// decode step, it holds the sampled token that should be returned on
    /// the next `next()` call. `[B]`.
    private var nextTokens: MLXArray
    private var numTokens: [Int]
    private var matcherStates: [SequenceStateMachineState]

    /// MTP state. Non-nil when the batch is running the MTP draft+verify cycle.
    /// Set by `postInitMTP` immediately after `init`; cleared on finish or fallback.
    /// Port of omlx commit 696d90a: batch_generator.py _omlx_mtp_state
    internal var _omlxMtpState: MTPState?

    public init(
        model: any LanguageModel,
        uids: [Int],
        seedTokens: MLXArray,
        promptCache: [any BatchedCache],
        tokens: [[Int]],
        maxTokens: [Int],
        samplers: [RowSampler?]? = nil,
        fallbackSampler: @escaping RowSampler = greedySampler,
        stateMachines: [SequenceStateMachine]? = nil
    ) {
        precondition(uids.count == tokens.count, "uids/tokens count mismatch")
        precondition(uids.count == maxTokens.count, "uids/max_tokens count mismatch")
        self.model = model
        self.uids = uids
        self.promptCache = promptCache
        self.tokens = tokens
        self.maxTokens = maxTokens
        self.samplers = samplers ?? Array(repeating: nil, count: uids.count)
        self.fallbackSampler = fallbackSampler
        let machines = stateMachines ?? Array(repeating: SequenceStateMachine(), count: uids.count)
        self.stateMachines = machines
        self.matcherStates = machines.map { $0.makeState() }
        self.numTokens = Array(repeating: 0, count: uids.count)
        self.nextTokens = seedTokens

        // Match upstream mlx_lm.GenerationBatch: immediately run one
        // decode step in the constructor so the first call to `next()`
        // returns an already-computed token while scheduling the following
        // token. This double-buffer keeps the GPU queue ahead of the CPU
        // token extraction path.
        if !uids.isEmpty {
            _ = step()
            // Attempt MTP post-init for eligible single-sequence batches.
            // omlx: batch_generator.py patched_init → _post_init_mtp
            if batchSize == 1, let mtpModel = model as? (any MTPCapable), mtpModel.hasMTPHead {
                postInitMTP(model: mtpModel)
            }
        }
    }

    /// Run one decode step. Finished rows (length / stop) are filtered out
    /// of the active set after this call; their final responses appear with
    /// non-nil `finishReason`.
    public func next() -> [GenerationBatchResponse] {
        if uids.isEmpty { return [] }

        // MTP dispatch: for eligible single-sequence batches, emit from queue or
        // run the 2-token verify cycle.
        // omlx: batch_generator.py patched_next → _mtp_next
        if batchSize == 1,
           let state = _omlxMtpState,
           let mtpModel = model as? (any MTPCapable)
        {
            do {
                return try mtpNext(state: state, model: mtpModel)
            } catch {
                // Fall back to standard step; drop state to prevent half-built cycles.
                _omlxMtpState = nil
            }
        }

        let stepTokens = step()

        var keep: [Int] = []
        var responses: [GenerationBatchResponse] = []
        responses.reserveCapacity(uids.count)

        for i in 0 ..< uids.count {
            numTokens[i] += 1

            var finishReason: String? = nil
            if numTokens[i] >= maxTokens[i] {
                finishReason = "length"
            }

            let machine = stateMachines[i]
            let (nextState, matchedSequence, currentState) =
                machine.match(matcherStates[i], stepTokens[i])
            matcherStates[i] = nextState
            if matchedSequence != nil, currentState == nil {
                finishReason = "stop"
            }

            if finishReason != nil {
                let extracted: [any KVCache] = promptCache.map { $0.extractBatched(i) }
                responses.append(
                    GenerationBatchResponse(
                        uid: uids[i],
                        token: stepTokens[i],
                        finishReason: finishReason,
                        matchedSequence: matchedSequence,
                        currentState: currentState,
                        allTokens: tokens[i],
                        promptCache: extracted
                    ))
            } else {
                keep.append(i)
                responses.append(
                    GenerationBatchResponse(
                        uid: uids[i],
                        token: stepTokens[i],
                        finishReason: nil,
                        matchedSequence: matchedSequence,
                        currentState: currentState,
                        allTokens: nil,
                        promptCache: nil
                    ))
            }
        }

        if keep.count < uids.count {
            filter(keep: keep)
        }

        return responses
    }

    /// In-place keep only the rows at the given indices.
    public func filter(keep: [Int]) {
        let keepArr = MLXArray(keep.map { Int32($0) })

        if keep.isEmpty {
            promptCache.removeAll()
        } else {
            for cache in promptCache {
                cache.filterBatched(batchIndices: keepArr)
            }
        }

        uids = keep.map { uids[$0] }
        tokens = keep.map { tokens[$0] }
        samplers = keep.map { samplers[$0] }
        maxTokens = keep.map { maxTokens[$0] }
        stateMachines = keep.map { stateMachines[$0] }
        matcherStates = keep.map { matcherStates[$0] }
        numTokens = keep.map { numTokens[$0] }
        if !keep.isEmpty {
            nextTokens = take(nextTokens, keepArr, axis: 0)
        }
        // MTP: drop state when batch is emptied externally (e.g. scheduler abort).
        // omlx: batch_generator.py patched_filter
        if keep.isEmpty { _omlxMtpState = nil }
    }

    /// In-place: append `other`'s rows to this batch. Per-layer caches are
    /// concatenated via `BatchedCache.extendBatched`.
    public func extend(_ other: GenerationBatch) {
        precondition(
            promptCache.count == other.promptCache.count,
            "Cannot extend with a batch that has a different layer count"
        )
        for (a, b) in zip(promptCache, other.promptCache) {
            a.extendBatched(b)
        }
        uids.append(contentsOf: other.uids)
        tokens.append(contentsOf: other.tokens)
        samplers.append(contentsOf: other.samplers)
        maxTokens.append(contentsOf: other.maxTokens)
        stateMachines.append(contentsOf: other.stateMachines)
        matcherStates.append(contentsOf: other.matcherStates)
        numTokens.append(contentsOf: other.numTokens)
        nextTokens = concatenated([nextTokens, other.nextTokens], axis: 0)
        // MTP: transfer state from donor batch when BatchGenerator merges a fresh
        // single-sequence batch into self via extend(). The MTP post-init fires on
        // the donor (whose __init__ ran with uids=[1]); without this transfer the
        // state would be lost when the donor is released.
        // omlx: batch_generator.py patched_extend
        if let donorState = other._omlxMtpState, _omlxMtpState == nil {
            _omlxMtpState = donorState
            other._omlxMtpState = nil
        }
    }

    public var isEmpty: Bool { uids.isEmpty }
    public var batchSize: Int { uids.count }

    // MARK: - MTP post-init

    /// Called from `init` immediately after the standard `step()` for eligible
    /// single-sequence batches. Sets up `_omlxMtpState` and seeds the emit queue
    /// with two confirmed tokens so the first two `next()` calls bypass `step()`.
    /// omlx: batch_generator.py patched_init → _post_init_mtp
    internal func postInitMTP(model: any MTPCapable) {
        let sampler = samplers[0] ?? fallbackSampler

        // nextTokens = main_tok: sampled from prompt[-1]'s logits in init's step().
        let mainTok = nextTokens  // shape (1,) — asyncEval'd in step(); will force on use

        // 1. Backbone forward at main_tok → (logits [1,1,vocab], preNormHidden [1,1,H]).
        // omlx: _post_init_mtp — "1-token backbone forward at main_tok with hidden state"
        let mainInput = mainTok.reshaped(1, 1)
        let backboneCache = promptCache.map { $0 as any KVCache }
        let (logits, hidden) = model.callWithHidden(
            input: LMInput.Text(tokens: mainInput),
            cache: backboneCache,
            nConfirmed: 0
        )

        // 2. Sample next_main_tok from logits[:, -1, :].
        let nextMainLogits = logits[0..., -1, 0...]  // [1, vocab]
        let nextMainLp = nextMainLogits - logSumExp(nextMainLogits, axis: -1, keepDims: true)
        let nextMainTok = sampler(nextMainLp)  // (1,)

        // 3. Run MTP head: (hidden_at_main [1,1,H], next_main_tok [1,1]) → draft [1,1,vocab].
        // omlx: _post_init_mtp — "MTP head sees (hidden_at_main, next_main_tok)"
        let mtpCache = model.makeMTPCache()
        let T = hidden.dim(1)
        let hiddenAtMain = hidden[0..., (T - 1) ..< T, 0...]  // [1, 1, H]
        let nextIds = nextMainTok.reshaped(1, 1)               // [1, 1]
        let mtpLogits = model.mtpForward(
            hidden: hiddenAtMain, nextTokenIds: nextIds, cache: mtpCache)
        let draftLogits2d = mtpLogits[0..., -1, 0...]  // [1, vocab]
        let draftLp2d = draftLogits2d - logSumExp(draftLogits2d, axis: -1, keepDims: true)
        let draftTok = sampler(draftLp2d)  // (1,)

        // 4. Single eval for all sampled tokens. Cache draft_id as Int to avoid a
        //    GPU→CPU sync in the first verify cycle's accept/reject check.
        // omlx: _post_init_mtp — "mx.eval(main_tok, next_main_tok, draft_tok)"
        eval(mainTok, nextMainTok, draftTok)
        let mainId = Int(mainTok.asArray(UInt32.self)[0])
        let nextMainId = Int(nextMainTok.asArray(UInt32.self)[0])
        let draftId = Int(draftTok.asArray(UInt32.self)[0])

        let state = MTPState()
        state.mtpCache = mtpCache
        state.nextMain = nextMainTok
        state.draftTok = draftTok
        state.draftLp = draftLp2d[0]  // [vocab]
        state.draftId = draftId
        // Queue the two confirmed tokens. The first two next() calls emit from
        // this queue without any model forward — "two confirmed tokens" from PR 990.
        state.queue.append(MTPQueueItem(tokenId: mainId, source: "init"))
        state.queue.append(MTPQueueItem(tokenId: nextMainId, source: "init"))

        _omlxMtpState = state
    }

    // MARK: - Standard decode step (unchanged)

    /// One forward pass + per-row sample, double-buffered like upstream
    /// `mlx_lm.generate.GenerationBatch._step`.
    ///
    /// `nextTokens` is treated as the *current* token batch to return from
    /// this call. We immediately feed it back through the model, sample the
    /// following token batch, and `asyncEval` that future batch before
    /// synchronously materializing the current tokens for CPU-side stop
    /// detection / response dispatch.
    private func step() -> [Int] {
        let currentTokens = nextTokens
        let inputs = currentTokens.reshaped(uids.count, 1)

        let logits = model.callAsFunction(inputs, cache: promptCache.map { $0 as any KVCache })

        // [B, 1, vocab] -> [B, vocab]
        let stepLogits = logits[0..., -1, 0...]

        let sampledTokens: MLXArray
        if samplers.contains(where: { $0 != nil }) {
            let logprobs = stepLogits - logSumExp(stepLogits, axis: -1, keepDims: true)
            var samples: [MLXArray] = []
            samples.reserveCapacity(uids.count)
            for i in 0 ..< uids.count {
                let rowLogprobs = logprobs[i ..< (i + 1), 0...]
                let sampler = samplers[i] ?? fallbackSampler
                samples.append(sampler(rowLogprobs))
            }
            sampledTokens = concatenated(samples, axis: 0)
        } else {
            // Greedy fast path. Avoid the full-vocabulary logSumExp when
            // all rows are greedy: argMax(logits) == argMax(logprobs),
            // and Swift does not currently expose logprobs downstream.
            // This removes one expensive reduction kernel per decode step.
            sampledTokens = argMax(stepLogits, axis: -1)
        }

        // Start computing the next token before forcing the current token
        // values back to the CPU. This overlaps GPU work with the CPU
        // extraction / response-building path.
        nextTokens = sampledTokens
        asyncEval(sampledTokens)

        eval(currentTokens)
        let stepTokens = currentTokens.asArray(UInt32.self).map { Int($0) }

        for (i, t) in stepTokens.enumerated() {
            tokens[i].append(t)
        }

        return stepTokens
    }
}

// MARK: - MTP fallback error

/// Signals a clean fallback to the standard step() inside the MTP dispatch path.
/// omlx: batch_generator.py _MtpStepFallback
private enum MTPFallback: Error {
    case missingVerifyInputs
    case verifyProducedNoTokens
    case cacheRollbackFailed
}

// MARK: - MTP dispatch (private extension)
//
// Port of omlx commit 696d90a:
//   patches/mlx_lm_mtp/batch_generator.py
//   _mtp_next, _run_verify_cycle, _step_mtp, _residual_sample, _emit_response,
//   _restore_or_trim_caches, _clear_rollback, _bump_emit_stat
//
// This extension intercepts next() for eligible single-sequence batches.
// The queue emitted by postInitMTP (two confirmed tokens) is drained first;
// subsequent calls run the 2-token verify cycle.

private extension GenerationBatch {

    // MARK: next() dispatch

    /// Emit one token from the queue; run a verify cycle if the queue is empty.
    /// omlx: batch_generator.py _mtp_next
    func mtpNext(state: MTPState, model: any MTPCapable) throws -> [GenerationBatchResponse] {
        if !state.queue.isEmpty {
            let item = state.queue.removeFirst()
            bumpEmitStat(&state.stats, source: item.source)
            return emitMTPToken(item, state: state)
        }

        try runVerifyCycle(state: state, model: model)
        guard !state.queue.isEmpty else {
            // Verify cycle must always produce at least the rejected-verify token.
            // omlx: _mtp_next — "verify cycle produced no emit tokens"
            throw MTPFallback.verifyProducedNoTokens
        }

        let item = state.queue.removeFirst()
        bumpEmitStat(&state.stats, source: item.source)
        return emitMTPToken(item, state: state)
    }

    // MARK: Verify cycle

    /// 2-token backbone forward [next_main, draft] with nConfirmed=1.
    /// Populates state.queue with 1 (reject) or 2 (accept) tokens for the
    /// upcoming emit calls. Updates state.nextMain / draftTok / draftLp for
    /// the following cycle.
    /// omlx: batch_generator.py _run_verify_cycle
    func runVerifyCycle(state: MTPState, model: any MTPCapable) throws {
        guard let nextMain = state.nextMain, let draftTok = state.draftTok else {
            throw MTPFallback.missingVerifyInputs
        }

        let sampler = samplers[0] ?? fallbackSampler
        let isGreedy = samplers[0] == nil

        // Concatenate [next_main (1,), draft_tok (1,)] → (2,) → [1, 2] input.
        let inputs = concatenated([nextMain, draftTok], axis: 0).reshaped(1, 2)
        let backboneCache = promptCache.map { $0 as any KVCache }

        // --- backbone forward (2-token, nConfirmed=1) ---
        // omlx: _run_verify_cycle — "logits, hidden = gen_batch.model(..., n_confirmed=1)"
        let t0 = CFAbsoluteTimeGetCurrent()
        let (logits, hidden) = model.callWithHidden(
            input: LMInput.Text(tokens: inputs),
            cache: backboneCache,
            nConfirmed: 1
        )
        state.stats.backboneMs += (CFAbsoluteTimeGetCurrent() - t0) * 1000

        // verify position = [:, 0, :]; bonus position = [:, 1, :]
        let verifyLogits = logits[0..., 0, 0...]  // [1, vocab]
        let bonusLogits  = logits[0..., 1, 0...]  // [1, vocab]

        // Batched logprob: one logsumexp over (2, vocab) vs two over (1, vocab).
        // omlx: _run_verify_cycle — "combined_logits = mx.concatenate([verify_logits, bonus_logits])"
        let tSample = CFAbsoluteTimeGetCurrent()
        let combinedLogits = concatenated([verifyLogits, bonusLogits], axis: 0)  // [2, vocab]
        let combinedLp = combinedLogits - logSumExp(combinedLogits, axis: -1, keepDims: true)
        let verifyLp2d = combinedLp[0 ..< 1, 0...]  // [1, vocab]
        let bonusLp2d  = combinedLp[1 ..< 2, 0...]  // [1, vocab]
        let verifyTok = sampler(verifyLp2d)  // (1,)
        let bonusTok  = sampler(bonusLp2d)   // (1,)
        eval(verifyTok, bonusTok)

        let draftId  = state.draftId
        let verifyId = Int(verifyTok.asArray(UInt32.self)[0])
        let bonusId  = Int(bonusTok.asArray(UInt32.self)[0])

        // Accept/reject check.
        // Greedy: accept iff verify argmax == draft.
        // Stochastic: accept with prob min(1, P_target(draft) / P_draft(draft)).
        // omlx: _run_verify_cycle — "if is_greedy: accept = verify_id == draft_id"
        let accept: Bool
        if isGreedy {
            accept = verifyId == draftId
        } else {
            let logTargetAtDraft = Double(verifyLp2d[0, draftId].asArray(Float.self)[0])
            let logDraftAtDraft  = Double(state.draftLp![draftId].asArray(Float.self)[0])
            let logAccept = logTargetAtDraft - logDraftAtDraft
            accept = logAccept >= 0 || Double.random(in: 0 ..< 1) < exp(logAccept)
        }
        state.stats.sampleMs += (CFAbsoluteTimeGetCurrent() - tSample) * 1000

        // Hidden at each verify-forward position.
        let hiddenAtConfirmed = hidden[0..., 0 ..< 1, 0...]  // [1, 1, H]
        let hiddenAtDraft     = hidden[0..., 1 ..< 2, 0...]  // [1, 1, H]

        state.stats.cycles += 1

        if accept {
            state.stats.accepts += 1

            // Accept path: clear rollback snapshots, run MTP head at draft position.
            // omlx: _run_verify_cycle accept branch
            let tCache = CFAbsoluteTimeGetCurrent()
            clearRollback()
            state.stats.cacheOpsMs += (CFAbsoluteTimeGetCurrent() - tCache) * 1000

            let (newDraft, newDraftLp) = stepMTP(
                state: state, model: model,
                hiddenAtPosition: hiddenAtDraft,
                nextMainTok: bonusTok
            )
            // Queue: accepted draft uses MTP head's original draft distribution;
            // bonus uses the verify forward's bonus distribution.
            // omlx: _run_verify_cycle — "state.queue.append((draft_id, state.draft_lp, 'draft'))"
            state.queue.append(MTPQueueItem(tokenId: draftId, source: "draft"))
            state.queue.append(MTPQueueItem(tokenId: bonusId, source: "bonus"))
            state.nextMain = bonusTok
            state.draftTok = newDraft
            state.draftLp  = newDraftLp
            return
        }

        // Reject path: restore / trim caches, residual sample, run MTP at confirmed.
        // omlx: _run_verify_cycle reject branch
        state.stats.rejects += 1
        let tCache = CFAbsoluteTimeGetCurrent()
        guard restoreOrTrimCaches() else {
            throw MTPFallback.cacheRollbackFailed
        }
        state.stats.cacheOpsMs += (CFAbsoluteTimeGetCurrent() - tCache) * 1000

        let emitId: Int
        if isGreedy {
            emitId = verifyId
        } else {
            let (rid, _) = residualSample(
                verifyLp2d: verifyLp2d, draftLp1d: state.draftLp!)
            emitId = rid ?? verifyId
        }

        let emitTok = MLXArray([UInt32(emitId)])  // (1,) uint32
        let (newDraft, newDraftLp) = stepMTP(
            state: state, model: model,
            hiddenAtPosition: hiddenAtConfirmed,
            nextMainTok: emitTok
        )
        state.queue.append(MTPQueueItem(tokenId: emitId, source: "verify"))
        state.nextMain = emitTok
        state.draftTok = newDraft
        state.draftLp  = newDraftLp
    }

    // MARK: MTP head forward

    /// One MTP-head forward + sample. Returns (draft_tok [1], draft_lp [vocab]).
    /// Side effect: stores the host-side int in `state.draftId` to avoid a
    /// GPU→CPU sync on the next verify cycle's accept/reject check.
    /// omlx: batch_generator.py _step_mtp
    func stepMTP(
        state: MTPState,
        model: any MTPCapable,
        hiddenAtPosition: MLXArray,
        nextMainTok: MLXArray
    ) -> (MLXArray, MLXArray) {
        let t0 = CFAbsoluteTimeGetCurrent()
        let sampler = samplers[0] ?? fallbackSampler

        let nextIds = nextMainTok.reshaped(1, 1)  // [1, 1]
        let mtpLogits = model.mtpForward(
            hidden: hiddenAtPosition, nextTokenIds: nextIds, cache: state.mtpCache)
        let mtpLogits2d = mtpLogits[0..., -1, 0...]  // [1, vocab]
        let newLp2d = mtpLogits2d - logSumExp(mtpLogits2d, axis: -1, keepDims: true)
        let newTok = sampler(newLp2d)  // (1,)

        // Force eval + cache int: avoids re-sync on next cycle's accept check.
        // omlx: _step_mtp — "draft_id_int = int(new_tok.tolist()[0])"
        eval(newTok)
        state.draftId = Int(newTok.asArray(UInt32.self)[0])
        state.stats.mtpHeadMs += (CFAbsoluteTimeGetCurrent() - t0) * 1000

        return (newTok, newLp2d[0])  // ([1], [vocab])
    }

    // MARK: Residual sampling

    /// Sample from max(P_target - P_draft, 0) / Z  (Leviathan et al. 2023).
    /// Returns (tokenId or nil, verify_lp_1d). Caller falls back to verify_id when nil.
    /// omlx: batch_generator.py _residual_sample
    func residualSample(
        verifyLp2d: MLXArray,
        draftLp1d: MLXArray
    ) -> (Int?, MLXArray) {
        let pTarget  = exp(verifyLp2d[0])   // [vocab]
        let pDraft   = exp(draftLp1d)        // [vocab]
        let residual = maximum(pTarget - pDraft, MLXArray(Float(0)))
        let zArr     = residual.sum()
        eval(zArr)
        let z = Double(zArr.asArray(Float.self)[0])
        if z <= 1e-8 {
            return (nil, verifyLp2d[0])
        }
        // Sample from the normalised residual distribution.
        // omlx: _residual_sample — "mx.random.categorical(mx.log(residual / z + 1e-10)...)"
        let logResidNorm = log(residual / Float(z) + Float(1e-10)).reshaped(1, -1)  // [1, vocab]
        let sample = MLXRandom.categorical(logResidNorm, axis: -1)  // (1,)
        eval(sample)
        return (Int(sample.asArray(Int32.self)[0]), verifyLp2d[0])
    }

    // MARK: Response builder

    /// Build a single-element response, applying the standard per-token epilogue
    /// (token append, maxTokens / matcher checks). Calls filter([]) and clears
    /// MTP state on the final token.
    /// omlx: batch_generator.py _emit_response
    func emitMTPToken(_ item: MTPQueueItem, state: MTPState) -> [GenerationBatchResponse] {
        let tokenId = item.tokenId

        tokens[0].append(tokenId)
        numTokens[0] += 1

        var finishReason: String? = nil
        if numTokens[0] >= maxTokens[0] {
            finishReason = "length"
        }

        let machine = stateMachines[0]
        let (nextMatchState, matchedSequence, currentState) =
            machine.match(matcherStates[0], tokenId)
        matcherStates[0] = nextMatchState
        if matchedSequence != nil, currentState == nil {
            finishReason = "stop"
        }

        if let finishReason {
            let extracted: [any KVCache] = promptCache.map { $0.extractBatched(0) }
            let response = GenerationBatchResponse(
                uid: uids[0],
                token: tokenId,
                finishReason: finishReason,
                matchedSequence: matchedSequence,
                currentState: currentState,
                allTokens: tokens[0],
                promptCache: extracted
            )
            // Drop MTP state before filter([]) so patched_filter doesn't double-clear.
            // omlx: _emit_response finish path — "delattr(gen_batch, '_omlx_mtp_state')"
            _omlxMtpState = nil
            filter(keep: [])
            return [response]
        }

        return [GenerationBatchResponse(
            uid: uids[0],
            token: tokenId,
            finishReason: nil,
            matchedSequence: matchedSequence,
            currentState: currentState,
            allTokens: nil,
            promptCache: nil
        )]
    }

    // MARK: Cache rollback helpers

    /// Roll back one token from each layer cache after a draft rejection.
    /// SSM/linear-attention layers restore their `rollbackState` snapshot;
    /// full-attention layers trim by 1. Returns false if any layer supports neither.
    /// omlx: batch_generator.py _restore_or_trim_caches
    @discardableResult
    func restoreOrTrimCaches() -> Bool {
        for cache in promptCache {
            if let ac = cache as? ArraysCache, let snap = ac.rollbackState {
                let (convSnap, ssmSnap) = snap
                ac[0] = convSnap
                ac[1] = ssmSnap
                ac.rollbackState = nil
                continue
            }
            if cache.isTrimmable {
                cache.trim(1)
                continue
            }
            return false
        }
        return true
    }

    /// Drop rollback snapshots after a draft is accepted.
    /// omlx: batch_generator.py _clear_rollback
    func clearRollback() {
        for cache in promptCache {
            if let ac = cache as? ArraysCache {
                ac.rollbackState = nil
            }
        }
    }

    // MARK: Stat helpers

    /// Increment the emit-source stat counter.
    /// omlx: batch_generator.py _bump_emit_stat
    func bumpEmitStat(_ stats: inout MTPStats, source: String) {
        switch source {
        case "init":   stats.initEmits += 1
        case "draft":  stats.draftEmits += 1
        case "bonus":  stats.bonusEmits += 1
        case "verify": stats.verifyEmits += 1
        default:       break
        }
    }
}
