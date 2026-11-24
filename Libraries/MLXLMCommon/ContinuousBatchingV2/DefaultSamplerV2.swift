// DefaultSamplerV2.swift
//
// The production `CBv2StepSampler`: WS-E's `LogitsPipelineV2` (bias →
// penalties → temperature → top-k/top-p/min-p) composed with `SamplerV2`
// (all-greedy argmax fast path; keyed Gumbel-max otherwise). This is
// EngineV2's default sampler; `CBv2GreedySampler` remains the deterministic
// stub for scheduler tests.
//
// Statefulness & exactness:
//  - Reconfiguration happens ONLY when the row-ID order changes between
//    `sample` calls. `rowContext()` (confirmed history) rebuilds the
//    per-row tensors; when `pendingSampledTokens` is present (chained
//    decode reconfiguring mid-chain), the pending [B] tokens are folded in
//    ON-DEVICE via the pipelines' own `commit`, so penalty counts and RNG
//    step indices are exact — a pure function of each request's history,
//    never of batch composition or host visibility timing.
//  - Between reconfigurations, per-step `commit(sampledTokens:)` maintains
//    the state incrementally (device scatter-adds; no host syncs anywhere
//    on this path — `sample` builds graph nodes only).
//
// Batch-composition invariance: every pipeline transform is row-independent
// and the RNG is keyed (seed, requestID, per-request step), so a row's
// tokens cannot depend on its batchmates (research report 12 item 5).

import Foundation
import MLX

public final class CBv2DefaultSampler: CBv2StepSampler, CBv2FreeSpanSampler {

    private var pipeline: LogitsPipelineV2?
    private let sampler: SamplerV2
    private let constraintSampler = CBv2TokenConstraintSampler()
    private let budgetSampler = CBv2ReasoningBudgetSampler()
    private var configuredIDs: [CBv2RequestID] = []
    /// The lazy logprob gather built by the most recent `sample` call, until
    /// the loop consumes it via `takeStepLogprobs` (take semantics).
    private var pendingStepLogprobs: CBv2StepLogprobs?
    /// Number of `sample` calls that built logprob gather nodes
    /// (telemetry/test hook — must stay 0 when no row asks for logprobs).
    public private(set) var logprobGatherCount = 0
    /// Test hook: the composed pipeline's logprob-capture counter.
    var pipelineLogprobBuildCount: Int { pipeline?.logprobBuildCount ?? 0 }
    /// Metadata-only test observation of the REAL accepted sampler. Install
    /// before engine use; never replace during execution. Invoked once per
    /// sample on the engine queue. No callback result changes sampling.
    var sampleObserverForTesting: (([CBv2RequestID], Int) -> Void)?
    /// Test-only: the presence/frequency counts each draw or verify call
    /// applies to a penalty row, read back to the host. No effect on sampling.
    var penaltyCountObserverForTesting: ((CBv2PenaltyCountObservation) -> Void)?
    public var supportsTokenConstraints: Bool { true }
    public var supportsReasoningBudgets: Bool { true }
    public var supportsConstrainedChainedDecode: Bool { true }

    /// Sampler state from just before the most recent speculative sample (a
    /// chained successor over constrained rows): the penalty arrays and RNG
    /// step indices that sample started from. A redraw restores them, so it
    /// is the draw the unchained path makes at that output index.
    private struct SpeculativeSample {
        let requestIDs: [CBv2RequestID]
        let params: [CBv2SamplingParams]
        let penalties: LogitsPipelineV2.PenaltyState
        let steps: [UInt64]
    }
    private var speculation: SpeculativeSample?
    /// The offer of the current step (`planCertifiedTokens`), consumed by `certifiedSample`.
    private var certifiedPlan: CBv2CertifiedTokenRequest?
    /// Confirmed output-token counts `[vocab]` per penalty row for MTP
    /// verification, folded incrementally from the row's confirmed output
    /// (`folded` tokens so far; `last` = the last folded token, a cheap
    /// divergence check). Engine queue only; dropped on `requestDidFinish`.
    private var mtpPenaltyCounts: [CBv2RequestID: (folded: Int, last: Int, counts: MLXArray)] =
        [:]
    /// `arange(vocab)` `[1, 1, vocab]` int32 for the draft one-hot, per vocab.
    private var mtpVocabIDs: MLXArray?
    /// Test hooks for the constrained-chaining device-mask cache.
    var constraintMaskBuildCount: Int { constraintSampler.maskBuildCount }
    var constraintMaskCacheHitCount: Int { constraintSampler.maskCacheHitCount }

    /// - Parameter fallbackSeed: engine-level seed for rows without a
    ///   per-request seed (fixed at init so nil-seed rows stay
    ///   batch-invariant within a process; random by default).
    public init(fallbackSeed: UInt64? = nil) {
        self.sampler = SamplerV2(fallbackSeed: fallbackSeed)
    }

    public func sample(
        logits: MLXArray, params: [CBv2SamplingParams], requestIDs: [CBv2RequestID],
        stepIndex: Int, pendingSampledTokens: MLXArray?,
        rowContext: () -> [CBv2SamplerRow]
    ) -> MLXArray {
        sampleObserverForTesting?(requestIDs, stepIndex)
        certifiedPlan = nil
        let pipeline = configure(
            vocab: logits.dim(-1), requestIDs: requestIDs,
            pendingSampledTokens: pendingSampledTokens, rowContext: rowContext)

        // A chained successor over constrained rows samples before its
        // predecessor's token is confirmed: snapshot the pre-sample state so
        // `redrawChainedSuccessor` can replay this draw exactly.
        let speculative =
            pendingSampledTokens != nil && constraintSampler.hasRows(in: requestIDs)
        if speculative {
            constraintSampler.beginSpeculation()
            speculation = SpeculativeSample(
                requestIDs: requestIDs, params: params,
                penalties: pipeline.penaltyState, steps: sampler.stepIndices)
        } else {
            speculation = nil
        }
        return draw(
            logits: logits, params: params, requestIDs: requestIDs,
            pipeline: pipeline, speculative: speculative)
    }

    /// The pipeline for `vocab`, (re)configured for `requestIDs`.
    private func configure(
        vocab: Int, requestIDs: [CBv2RequestID], pendingSampledTokens: MLXArray?,
        rowContext: () -> [CBv2SamplerRow]
    ) -> LogitsPipelineV2 {
        if pipeline?.vocabSize != vocab {
            pipeline = LogitsPipelineV2(vocabSize: vocab)
            configuredIDs = []
        }
        let pipeline = self.pipeline!

        if requestIDs != configuredIDs {
            let rows = rowContext()
            pipeline.setRows(rows)
            sampler.setRows(rows)
            constraintSampler.configure(rows)
            budgetSampler.configure(rows)
            configuredIDs = requestIDs
            if let pendingSampledTokens {
                // Fold the chained in-flight tokens into the fresh state so
                // penalty counts and per-row RNG step indices include them.
                pipeline.commit(sampledTokens: pendingSampledTokens)
                sampler.commit()
            }
        }
        // Same IDs ⇒ the incremental commits below already covered every
        // token sampled since configuration (including any pending ones).
        return pipeline
    }

    /// True when the configured rows' draw passes through a hard mask (a token constraint or
    /// grammar, or a reasoning budget): `draw` builds the mask exactly when this holds, and
    /// `planCertifiedTokens` never offers then.
    private var masksLogits: Bool { constraintSampler.hasRows || budgetSampler.hasRows }

    /// Configures for `requestIDs` exactly as `sample` would and, when this step's draw needs no
    /// transform beyond temperature (`LogitsPipelineV2.appliesOnlyTemperature`) and no hard mask
    /// (`masksLogits`), returns an offer carrying the row kinds, the
    /// pipeline's own temperature divisor and the noise `sample` would draw. The same decision
    /// guards `certifiedSample`, so an offer can never reach a row that transforms its logits.
    func planCertifiedTokens(
        vocab: Int, requestIDs: [CBv2RequestID], pendingSampledTokens: MLXArray?,
        rowContext: () -> [CBv2SamplerRow]
    ) -> CBv2CertifiedTokenRequest? {
        let pipeline = configure(
            vocab: vocab, requestIDs: requestIDs, pendingSampledTokens: pendingSampledTokens,
            rowContext: rowContext)
        certifiedPlan = nil
        guard pipeline.appliesOnlyTemperature, !masksLogits else { return nil }
        let request = CBv2CertifiedTokenRequest(
            vocabularySize: vocab, greedy: sampler.greedyRows,
            temperature: pipeline.temperatureColumn, noise: sampler.pendingNoise(vocab: vocab))
        certifiedPlan = request
        return request
    }

    /// `sample` for a step whose tokens a model chose through the offer from
    /// `planCertifiedTokens`: the same commits and bookkeeping, with `tokens` as the draw.
    func certifiedSample(
        _ request: CBv2CertifiedTokenRequest, tokens: MLXArray, requestIDs: [CBv2RequestID],
        stepIndex: Int
    ) -> MLXArray {
        sampleObserverForTesting?(requestIDs, stepIndex)
        precondition(
            certifiedPlan === request && configuredIDs == requestIDs,
            "certified tokens without this step's offer")
        certifiedPlan = nil
        speculation = nil
        pipeline!.commit(sampledTokens: tokens)
        sampler.commit()
        pendingStepLogprobs = nil
        return tokens
    }

    /// One configured draw: transforms, grammar mask, selection, state
    /// commit, and the lazy logprob gather.
    private func draw(
        logits: MLXArray, params: [CBv2SamplingParams], requestIDs: [CBv2RequestID],
        pipeline: LogitsPipelineV2, speculative: Bool
    ) -> MLXArray {
        if let observe = penaltyCountObserverForTesting,
            let counts = pipeline.outputCountsForTesting
        {
            for (row, id) in requestIDs.enumerated()
            where CBv2MTPVerifyPenalties.applies(params[row]) {
                observe(.init(kind: .sample, id: id, counts: [counts[row].asArray(Float.self)]))
            }
        }
        // Apply the grammar exactly once, after arithmetic transforms and
        // before top-k/top-p/min-p. This preserves the valid language even
        // for malformed-but-decodable penalties that can resurrect
        // -infinity, without rebuilding the dense mask twice. Raw logprobs
        // still come from the original unmasked distribution by contract.
        let output = pipeline.process(
            logits,
            rawLogprobsFrom: logits,
            hardMask: masksLogits
                ? { [constraintSampler, budgetSampler] transformed in
                    budgetSampler.mask(
                        constraintSampler.mask(
                            transformed, requestIDs: requestIDs, speculative: speculative),
                        requestIDs: requestIDs)
                } : nil)
        let tokens = sampler.sample(from: output.sampling)
        pipeline.commit(sampledTokens: tokens)
        sampler.commit()

        // Lazy logprob gather from the RAW (pre-transform) logprobs — graph
        // nodes only, no host sync; the loop materializes at finalization.
        // Rows with topLogprobs == 0 pay nothing beyond riding the batch's
        // shared capture (and the whole branch is skipped when NO row asks).
        if let rawLogprobs = output.rawLogprobs {
            let k = params.reduce(0) { max($0, $1.topLogprobs) }
            pendingStepLogprobs = CBv2StepLogprobs(
                rows: requestIDs,
                topLogprobsPerRow: params.map(\.topLogprobs),
                gathered: CBv2Logprobs.gather(
                    rawLogprobs: rawLogprobs, sampledTokens: tokens, k: k))
            logprobGatherCount += 1
        } else {
            pendingStepLogprobs = nil
        }
        return tokens
    }

    public func takeInvalidatedChainedRows() -> [CBv2RequestID] {
        constraintSampler.takeInvalidated()
    }

    /// Exact redraw of the most recent speculative `sample`: restore the
    /// penalty state and per-row RNG step indices it started from, then run
    /// the same transforms with masks built from the now-confirmed
    /// constraint state. RNG keys are a pure function of (seed, request,
    /// per-request step), so the redraw uses exactly the keys of the
    /// discarded draw: the stream never shifts. Rows whose speculative mask
    /// was valid get the same mask, logits and key, hence the same token.
    public func redrawChainedSuccessor(
        logits: MLXArray, requestIDs: [CBv2RequestID]
    ) -> MLXArray? {
        guard let speculation, speculation.requestIDs == requestIDs, let pipeline,
            pipeline.vocabSize == logits.dim(-1)
        else { return nil }
        self.speculation = nil
        pipeline.restore(speculation.penalties)
        sampler.restoreStepIndices(speculation.steps)
        return draw(
            logits: logits, params: speculation.params, requestIDs: requestIDs,
            pipeline: pipeline, speculative: false)
    }

    public func takeStepLogprobs() -> CBv2StepLogprobs? {
        defer { pendingStepLogprobs = nil }
        return pendingStepLogprobs
    }

    /// Invalidate the configured fingerprint when a finished request's id
    /// was part of it: a FUTURE request may legally reuse that id, and an
    /// identical `requestIDs` array must then reconfigure (fresh penalties,
    /// RNG step 0) instead of inheriting the retired request's state
    /// (PR#62 review). Forcing a full `setRows` on the next `sample` is
    /// exact — reconfiguration is a pure function of `rowContext()`.
    public func requestDidFinish(_ id: CBv2RequestID) {
        constraintSampler.requestDidFinish(id)
        budgetSampler.requestDidFinish(id)
        mtpPenaltyCounts[id] = nil
        if configuredIDs.contains(id) {
            configuredIDs = []
        }
    }

    // MARK: - MTP target-prefix verify sampling

    public var supportsMTPTargetPrefix: Bool { true }
    public var supportsMTPPenaltyVerification: Bool { true }
    public var supportsMTPRepetitionPenaltyVerification: Bool { true }

    /// Pure re-derivation of the ordinary per-token draw for every verify
    /// window position. Eligibility gating guarantees the admitted rows use
    /// only the STATELESS transforms (temperature → top-k/top-p/min-p; no
    /// bias, constraints, or logprob capture; presence/frequency penalties
    /// only through the `penalties:` overload), so this mirrors
    /// `LogitsPipelineV2.process` + `SamplerV2.sample` per (row, position)
    /// without touching the incremental pipeline state. RNG keys use the
    /// per-request output index (`stepBases[r] + j`), i.e. exactly the key
    /// the ordinary path would use for that output token.
    public func mtpVerifySample(
        logits: MLXArray, params: [CBv2SamplingParams],
        requestIDs: [CBv2RequestID], stepBases: [Int]
    ) -> MLXArray? {
        verifyWindow(
            logits: logits, params: params, requestIDs: requestIDs,
            stepBases: stepBases, typical: nil, penalties: nil
        ).tokens
    }

    /// Penalty-aware `mtpVerifySample`: window position `j` of row `r` is
    /// penalized with the row's confirmed output counts plus the drafts
    /// before it (the repetition window: the last `repetitionContextSize`
    /// tokens of prompt, confirmed output and those drafts), using `LogitsPipelineV2.process`'s ops in its order and
    /// dtype (bias → penalties → temperature → top-k/top-p/min-p), so on
    /// the accepted prefix every draw is the one sequential sampling makes
    /// from the same logits.
    public func mtpVerifySample(
        logits: MLXArray, params: [CBv2SamplingParams],
        requestIDs: [CBv2RequestID], stepBases: [Int], penalties: CBv2MTPVerifyPenalties?
    ) -> MLXArray? {
        verifyWindow(
            logits: logits, params: params, requestIDs: requestIDs,
            stepBases: stepBases, typical: nil, penalties: penalties
        ).tokens
    }

    /// Penalty-aware `mtpVerifyTypical`; penalty rows (presence, frequency or
    /// repetition) accept exactly.
    public func mtpVerifyTypical(
        logits: MLXArray, draftIDs: MLXArray, delta: Float,
        params: [CBv2SamplingParams], requestIDs: [CBv2RequestID], stepBases: [Int],
        penalties: CBv2MTPVerifyPenalties?
    ) -> (tokens: MLXArray, accept: MLXArray)? {
        let window = verifyWindow(
            logits: logits, params: params, requestIDs: requestIDs,
            stepBases: stepBases, typical: (draftIDs: draftIDs, delta: delta),
            penalties: penalties)
        return (window.tokens, window.accept!)
    }

    /// Typical acceptance on the same filtered rows `mtpVerifySample` draws
    /// from: `H = -sum p ln p` over the filtered, normalized row, `floor =
    /// min(1, delta * exp(-H))`, keep iff `p(draft) > floor`. The keyed draw
    /// is unchanged, so the token committed at the first rejection or bonus
    /// position is the same target sample exact mode would commit.
    public func mtpVerifyTypical(
        logits: MLXArray, draftIDs: MLXArray, delta: Float,
        params: [CBv2SamplingParams], requestIDs: [CBv2RequestID], stepBases: [Int]
    ) -> (tokens: MLXArray, accept: MLXArray)? {
        let window = verifyWindow(
            logits: logits, params: params, requestIDs: requestIDs,
            stepBases: stepBases, typical: (draftIDs: draftIDs, delta: delta), penalties: nil)
        return (window.tokens, window.accept!)
    }

    /// A coupled proposal per row of draft logits `[B, 1, vocab]`, row r for
    /// output index `stepBases[r]`: the keyed exponential race
    /// `mtpVerifySample` runs at that index (same temperature, same noise),
    /// over the draft's whole row. When the draft's distribution is the
    /// target's, the target's draw at that index is the same token unless the
    /// race's winner falls outside the row's top-k / top-p / min-p set; such
    /// a winner (at most the mass those filters remove) is replaced by the
    /// row's mode, which every filter keeps. Sort-free, so a few passes over
    /// the row instead of `mtpVerifySample`'s sort and scatter. Greedy rows
    /// take the argmax. A verifier that commits only its own draws stays
    /// exact whatever it is proposed; this only raises how often it agrees.
    /// Stateless transforms only: it ignores presence/frequency penalties (a
    /// penalty row's proposal is then less likely to be accepted, never
    /// inexact). `[B, 1]`.
    public func mtpCoupledProposal(
        logits: MLXArray, params: [CBv2SamplingParams],
        requestIDs: [CBv2RequestID], stepBases: [Int]
    ) -> MLXArray {
        precondition(
            logits.ndim == 3 && logits.dim(1) == 1, "MTP proposal logits must be [B, 1, vocab]")
        let b = logits.dim(0)
        let vocab = logits.dim(2)
        precondition(
            params.count == b && requestIDs.count == b && stepBases.count == b,
            "MTP proposal row metadata mismatch")
        let x = logits.reshaped([b, vocab]).asType(.float32)
        let mode = argMax(x, axis: -1).asType(.int32)
        guard params.contains(where: { $0.temperature >= LogitsPipelineV2.greedyEpsilon }) else {
            return mode.reshaped([b, 1])
        }
        var temps = [Float](repeating: 1, count: b)
        var topKs = [Int32](repeating: Int32(vocab), count: b)
        var topPs = [Float](repeating: 2, count: b)
        var minPs = [Float](repeating: 0, count: b)
        var greedyFlags = [Bool](repeating: true, count: b)
        var noiseRows: [(seed: UInt64?, id: UInt64, step: UInt64, greedy: Bool)] = []
        var anyFilter = false
        for r in 0 ..< b {
            let p = params[r]
            let greedy = p.temperature < LogitsPipelineV2.greedyEpsilon
            greedyFlags[r] = greedy
            noiseRows.append(
                (seed: p.seed, id: requestIDs[r].raw, step: UInt64(stepBases[r]), greedy: greedy))
            guard !greedy else { continue }
            temps[r] = p.temperature
            if p.topK > 0, p.topK < vocab {
                topKs[r] = Int32(p.topK)
                anyFilter = true
            }
            if p.topP > 0, p.topP < 1 {
                topPs[r] = p.topP
                anyFilter = true
            } else if p.topP <= 0 {
                topPs[r] = 0
                anyFilter = true
            }
            if p.minP > 0 {
                minPs[r] = min(p.minP, 1)
                anyFilter = true
            }
        }
        let scaled = temps.allSatisfy { $0 == 1 } ? x : x / MLXArray(temps).reshaped([b, 1])
        let probs = softmax(scaled, axis: -1)
        let noise = sampler.verifyExponentialNoise(rows: noiseRows, vocab: vocab)
        var proposals = argMax(probs / noise, axis: -1).asType(.int32)
        if anyFilter {
            // The winner's place in the descending order: the tokens strictly
            // more probable than it, and their mass.
            let winning = takeAlong(probs, proposals.reshaped([b, 1]), axis: -1)
            let above = probs .> winning
            let rank = sum(above.asType(.int32), axis: -1)
            let mass = sum(which(above, probs, MLXArray(Float(0))), axis: -1)
            let kept =
                (rank .< MLXArray(topKs)) .&& (mass .<= MLXArray(topPs))
                .&& (winning.reshaped([b]) .>= MLXArray(minPs) * probs.max(axis: -1))
            proposals = which(kept, proposals, mode)
        }
        return which(MLXArray(greedyFlags), mode, proposals).reshaped([b, 1])
    }

    /// Exact acceptance per draft position: keep iff the target token equals
    /// the draft. `tokens` is `[B, W]`, `draftIDs` is `[B, D]`, `D <= W`.
    private static func exactAccept(tokens: MLXArray, draftIDs: MLXArray) -> MLXArray {
        tokens[0..., ..<draftIDs.dim(1)] .== draftIDs
    }

    private func verifyWindow(
        logits: MLXArray, params: [CBv2SamplingParams],
        requestIDs: [CBv2RequestID], stepBases: [Int],
        typical: (draftIDs: MLXArray, delta: Float)?,
        penalties: CBv2MTPVerifyPenalties?
    ) -> (tokens: MLXArray, accept: MLXArray?) {
        precondition(logits.ndim == 3, "MTP verify logits must be [B, W, vocab]")
        let b = logits.dim(0)
        let w = logits.dim(1)
        let vocab = logits.dim(2)
        precondition(
            params.count == b && requestIDs.count == b && stepBases.count == b,
            "MTP verify sampling row metadata mismatch")
        if let typical {
            precondition(
                typical.draftIDs.ndim == 2 && typical.draftIDs.dim(0) == b
                    && typical.draftIDs.dim(1) >= 1 && typical.draftIDs.dim(1) <= w,
                "MTP typical draft ids must be [B, D] with 1 <= D <= W")
            precondition(typical.delta > 0, "MTP typical delta must be positive")
        }
        var flat = logits.reshaped([b * w, vocab]).asType(.float32)
        // Penalty rows: stage (2) of `LogitsPipelineV2.process` per window
        // position, repetition first, then presence/frequency. Rows without a
        // penalty are left untouched (the ordinary path's `x - 0 * counts`,
        // and its repetition divide by 1, are the identity on them as well).
        let repetitionRows = params.map(CBv2MTPVerifyPenalties.appliesRepetition)
        let countRows = params.map(CBv2MTPVerifyPenalties.applies)
        if let penalties, countRows.contains(true) {
            flat = applyWindowPenalties(
                flat, penalties: penalties, params: params, requestIDs: requestIDs,
                penaltyRows: countRows, b: b, w: w, vocab: vocab)
        }
        let penaltyRows = params.map(CBv2MTPVerifyPenalties.appliesAny)
        let greedyTokens = argMax(flat, axis: -1).asType(.int32)
        let anyStochastic = params.contains {
            $0.temperature >= LogitsPipelineV2.greedyEpsilon
        }
        if !anyStochastic {
            // Bit-identical to the historical argmax acceptance walk.
            let tokens = greedyTokens.reshaped([b, w])
            let accept = typical.map { Self.exactAccept(tokens: tokens, draftIDs: $0.draftIDs) }
            return (tokens, accept)
        }

        // Mirror LogitsPipelineV2.setRows parameter resolution, expanded to
        // one row per (request, position). Greedy rows keep the identity
        // sentinels; their pick is the argmax merge below.
        var temps = [Float](repeating: 1, count: b * w)
        var topKs = [Int32](repeating: Int32(vocab), count: b * w)
        var topPs = [Float](repeating: 2, count: b * w)
        var minPs = [Float](repeating: 0, count: b * w)
        var greedyFlags = [Bool](repeating: true, count: b * w)
        var noiseRows: [(seed: UInt64?, id: UInt64, step: UInt64, greedy: Bool)] = []
        noiseRows.reserveCapacity(b * w)
        var anyTemperature = false
        var anyTopKPMinP = false
        for r in 0 ..< b {
            let p = params[r]
            let greedy = p.temperature < LogitsPipelineV2.greedyEpsilon
            for j in 0 ..< w {
                let i = r * w + j
                greedyFlags[i] = greedy
                noiseRows.append(
                    (
                        seed: p.seed, id: requestIDs[r].raw,
                        step: UInt64(stepBases[r] + j), greedy: greedy
                    ))
                guard !greedy else { continue }
                temps[i] = p.temperature
                if p.temperature != 1 { anyTemperature = true }
                if p.topK > 0, p.topK < vocab {
                    topKs[i] = Int32(p.topK)
                    anyTopKPMinP = true
                }
                if p.topP > 0, p.topP < 1 {
                    topPs[i] = p.topP
                    anyTopKPMinP = true
                } else if p.topP <= 0 {
                    topPs[i] = 0
                    anyTopKPMinP = true
                }
                if p.minP > 0 {
                    minPs[i] = min(p.minP, 1)
                    anyTopKPMinP = true
                }
            }
        }

        var x = flat
        if anyTemperature {
            x = x / MLXArray(temps).reshaped([b * w, 1])
        }
        if anyTopKPMinP {
            x = LogitsPipelineV2.applyTopKTopPMinP(
                x,
                topK: MLXArray(topKs).reshaped([b * w, 1]),
                topP: MLXArray(topPs).reshaped([b * w, 1]),
                minP: MLXArray(minPs).reshaped([b * w, 1]))
        }
        let probs = softmax(x, axis: -1)
        let noise = sampler.verifyExponentialNoise(rows: noiseRows, vocab: vocab)
        let sampledTokens = argMax(probs / noise, axis: -1).asType(.int32)
        let greedyRows = MLXArray(greedyFlags)
        let tokens = which(greedyRows, greedyTokens, sampledTokens).reshaped([b, w])
        guard let typical else { return (tokens, nil) }

        // Typical floor per draft position, on the same filtered rows the
        // draw above used. Zero-mass entries add nothing to the entropy: the
        // clamp guards only the log operand.
        let d = typical.draftIDs.dim(1)
        let rows = probs.reshaped([b, w, vocab])[0..., ..<d, 0...]
        let pDraft = takeAlong(rows, typical.draftIDs.reshaped([b, d, 1]), axis: -1)
            .squeezed(axis: -1)
        let entropy = -sum(rows * log(maximum(rows, MLXArray(Float(1e-30)))), axis: -1)
        let floor = minimum(exp(-entropy) * typical.delta, MLXArray(Float(1)))
        let typicalAccept = pDraft .> floor
        let exact = Self.exactAccept(tokens: tokens, draftIDs: typical.draftIDs)
        // Greedy rows, and penalty rows when penalties are applied, accept
        // exactly (target prefix).
        let exactRows = (0 ..< b * w).map { i in
            greedyFlags[i] || (penalties != nil && penaltyRows[i / w])
        }
        let exactPerPosition = MLXArray(exactRows).reshaped([b, w])[0..., ..<d]
        return (tokens, which(exactPerPosition, exact, typicalAccept))
    }

    /// The repetition penalty per window position, as stage (2) of
    /// `LogitsPipelineV2.process` applies it: every token present in the
    /// position's window has its logit divided by the penalty when positive
    /// and multiplied by it otherwise. The window of global position `g` is the
    /// last `repetitionContextSize` tokens of prompt + confirmed output +
    /// `drafts[0 ..< g]`, the ordinary ring buffer's contents after those
    /// drafts were sampled (token ids outside `0 ..< vocab` count for nothing,
    /// as in `buildCounts`).
    private func applyWindowRepetition(
        _ flat: MLXArray, penalties: CBv2MTPVerifyPenalties, params: [CBv2SamplingParams],
        repetitionRows: [Bool], b: Int, w: Int, vocab: Int
    ) -> MLXArray {
        precondition(
            penalties.outputTokens.count == b && penalties.promptTokens.count == b,
            "MTP verify repetition row metadata mismatch")
        let draftsNeeded = penalties.columnOffset + w - 1
        // A window never holds more than the whole history, so slots past the
        // longest history plus the drafts would only ever be empty.
        let longest =
            (0 ..< b).map {
                repetitionRows[$0]
                    ? penalties.promptTokens[$0].count + penalties.outputTokens[$0].count : 0
            }.max() ?? 0
        let sizes = (0 ..< b).map {
            repetitionRows[$0]
                ? min(params[$0].repetitionContextSize, longest + max(draftsNeeded, 0)) : 0
        }
        let span = sizes.max() ?? 0
        guard span > 0 else { return flat }
        // History tail per row: its last `span` tokens, right-aligned, -1 padded.
        var tail = [Int32](repeating: -1, count: b * span)
        for r in 0 ..< b where repetitionRows[r] {
            let output = penalties.outputTokens[r].suffix(span)
            let prompt = penalties.promptTokens[r].suffix(span - output.count)
            var slot = r * span + span - prompt.count - output.count
            for token in [prompt, output].joined() {
                tail[slot] = token >= 0 && token < vocab ? Int32(token) : -1
                slot += 1
            }
        }
        var history = MLXArray(tail, [b, span])
        if draftsNeeded > 0 {
            guard let drafts = penalties.drafts else {
                preconditionFailure("CBv2 MTP: penalty verification needs the round's drafts")
            }
            precondition(
                drafts.ndim == 2 && drafts.dim(0) == b && drafts.dim(1) >= draftsNeeded,
                "CBv2 MTP: penalty drafts must be [B, >= \(draftsNeeded)]")
            history = concatenated(
                [history, drafts[0..., ..<draftsNeeded].asType(.int32)], axis: 1)
        }
        // Window slot k of position j reads history[g + k], g = columnOffset + j;
        // row r uses the last sizes[r] slots.
        var gather = [Int32]()
        gather.reserveCapacity(w * span)
        for j in 0 ..< w {
            for k in 0 ..< span { gather.append(Int32(penalties.columnOffset + j + k)) }
        }
        var inWindow = [Bool](repeating: false, count: b * span)
        for r in 0 ..< b where repetitionRows[r] {
            for k in (span - sizes[r]) ..< span { inWindow[r * span + k] = true }
        }
        let tokens = take(history, MLXArray(gather), axis: 1).reshaped([b * w, span])
        let slotInWindow = broadcast(
            MLXArray(inWindow).reshaped([b, 1, span]), to: [b, w, span]
        ).reshaped([b * w, span])
        let inVocab = (tokens .>= MLXArray(Int32(0))) .&& (tokens .< MLXArray(Int32(vocab)))
        let valid = (slotInWindow .&& inVocab).asType(.float32)
        let rowIndex = broadcast(
            MLXArray(Array(0 ..< Int32(b * w))).reshaped([b * w, 1]), to: [b * w, span])
        let columnIndex = maximum(tokens, MLXArray(Int32(0)))
        let windowCounts = MLXArray.zeros([b * w, vocab], type: Float.self)
            .at[rowIndex.reshaped([-1]), columnIndex.reshaped([-1])].add(valid.reshaped([-1]))
        let present = windowCounts .> MLXArray(Float(0))
        var factors = [Float](repeating: 1, count: b * w)
        for r in 0 ..< b where repetitionRows[r] {
            for j in 0 ..< w { factors[r * w + j] = params[r].repetitionPenalty }
        }
        let penalty = MLXArray(factors).reshaped([b * w, 1])
        let penalized = which(flat .> 0, flat / penalty, flat * penalty)
        return which(present, penalized, flat)
    }

    /// `x - α·counts - β·(counts > 0)` per window position, in
    /// `LogitsPipelineV2.process` order and float32, with counts = confirmed
    /// output counts + one-hot of the drafts preceding the position.
    private func applyWindowPenalties(
        _ flat: MLXArray, penalties: CBv2MTPVerifyPenalties, params: [CBv2SamplingParams],
        requestIDs: [CBv2RequestID], penaltyRows: [Bool], b: Int, w: Int, vocab: Int
    ) -> MLXArray {
        precondition(
            penalties.outputTokens.count == b, "MTP verify penalty row metadata mismatch")
        let zeroRow = MLXArray.zeros([vocab], type: Float.self)
        // Bound the count cache to the rows of this call plus a small slack:
        // entries of requests that left without `requestDidFinish` (none
        // today) cannot accumulate past it.
        if mtpPenaltyCounts.count > 64 {
            let live = Set(requestIDs)
            mtpPenaltyCounts = mtpPenaltyCounts.filter { live.contains($0.key) }
        }
        let confirmed = stacked(
            (0 ..< b).map { r in
                penaltyRows[r]
                    ? confirmedOutputCounts(
                        id: requestIDs[r], tokens: penalties.outputTokens[r], vocab: vocab)
                    : zeroRow
            }, axis: 0)  // [B, vocab]
        var counts = broadcast(
            confirmed.expandedDimensions(axis: 1), to: [b, w, vocab])
        let draftsNeeded = penalties.columnOffset + w - 1
        if draftsNeeded > 0 {
            guard let drafts = penalties.drafts else {
                preconditionFailure("CBv2 MTP: penalty verification needs the round's drafts")
            }
            precondition(
                drafts.ndim == 2 && drafts.dim(0) == b && drafts.dim(1) >= draftsNeeded,
                "CBv2 MTP: penalty drafts must be [B, >= \(draftsNeeded)]")
            let used = drafts[0..., ..<draftsNeeded].asType(.int32)
            if mtpVocabIDs?.dim(2) != vocab {
                mtpVocabIDs = MLXArray(Array(0 ..< Int32(vocab))).reshaped([1, 1, vocab])
            }
            let vocabIDs = mtpVocabIDs!
            let oneHot = (used.expandedDimensions(axis: 2) .== vocabIDs).asType(.float32)
            // prefix[g] = drafts before global position g, g = 0 ... draftsNeeded.
            let prefix = concatenated(
                [
                    MLXArray.zeros([b, 1, vocab], type: Float.self),
                    cumsum(oneHot, axis: 1),
                ], axis: 1)
            let window = prefix[0..., penalties.columnOffset ..< (penalties.columnOffset + w), 0...]
            counts = counts + window
        }
        if let observe = penaltyCountObserverForTesting {
            let used =
                draftsNeeded > 0 ? penalties.drafts?[0..., ..<draftsNeeded].asType(.int32) : nil
            for r in 0 ..< b where penaltyRows[r] {
                observe(
                    .init(
                        kind: .verify, id: requestIDs[r],
                        counts: (0 ..< w).map { counts[r, $0].asArray(Float.self) },
                        outputTokens: Array(penalties.outputTokens[r]),
                        drafts: used.map { $0[r].asArray(Int32.self).map(Int.init) } ?? [],
                        columnOffset: penalties.columnOffset))
            }
        }
        let flatCounts = counts.reshaped([b * w, vocab])
        var freqs = [Float](repeating: 0, count: b * w)
        var press = [Float](repeating: 0, count: b * w)
        for r in 0 ..< b where penaltyRows[r] {
            for j in 0 ..< w {
                freqs[r * w + j] = params[r].frequencyPenalty
                press[r * w + j] = params[r].presencePenalty
            }
        }
        var x = flat
        x = x - MLXArray(freqs).reshaped([b * w, 1]) * flatCounts
        x = x - MLXArray(press).reshaped([b * w, 1]) * (flatCounts .> 0).asType(.float32)
        return x
    }

    /// Confirmed output counts `[vocab]` float32 for one row, folded
    /// incrementally (scatter-add of the tokens confirmed since the last
    /// call; integer-valued, so exact in any order). Rebuilt from scratch
    /// when the confirmed output no longer extends what was folded. The
    /// last-token check suffices because confirmed output only grows while a
    /// request lives and `requestDidFinish` drops an id before any reuse.
    private func confirmedOutputCounts(
        id: CBv2RequestID, tokens: ArraySlice<Int>, vocab: Int
    ) -> MLXArray {
        var entry = mtpPenaltyCounts[id]
        if let current = entry,
            current.counts.dim(0) != vocab || current.folded > tokens.count
                || (current.folded > 0
                    && tokens[tokens.startIndex + current.folded - 1] != current.last)
        {
            entry = nil
        }
        var counts = entry?.counts ?? MLXArray.zeros([vocab], type: Float.self)
        let folded = entry?.folded ?? 0
        if folded < tokens.count {
            let fresh = tokens[(tokens.startIndex + folded)...].compactMap { token in
                token >= 0 && token < vocab ? Int32(token) : nil
            }
            if !fresh.isEmpty {
                counts = counts.at[MLXArray(fresh)].add(Float(1))
            }
        }
        mtpPenaltyCounts[id] = (tokens.count, tokens.last ?? -1, counts)
        return counts
    }

    /// Verify rows never pass through `sample`, so their pipeline/RNG row
    /// state goes stale the moment a round confirms tokens. Dropping the
    /// fingerprint forces the next `sample` to reconfigure from confirmed
    /// history — exact by the same argument as `requestDidFinish`.
    public func mtpRoundDidCommit(requestIDs: [CBv2RequestID]) {
        guard !configuredIDs.isEmpty else { return }
        if requestIDs.contains(where: { configuredIDs.contains($0) }) {
            configuredIDs = []
        }
    }

    public func confirmSampledTokens(
        _ tokens: [Int], requestIDs: [CBv2RequestID]
    ) {
        constraintSampler.confirm(tokens: tokens, requestIDs: requestIDs)
        budgetSampler.confirm(tokens: tokens, requestIDs: requestIDs)
    }

    public func reasoningBudgetFailure(for id: CBv2RequestID) -> String? {
        budgetSampler.failure(for: id)
    }

    public func tokenConstraintFailure(for id: CBv2RequestID) -> String? {
        constraintSampler.failure(for: id)
    }

    func confirmedConstraintState(for id: CBv2RequestID, outputTokenCount: Int) -> Int? {
        constraintSampler.confirmedState(for: id, outputTokenCount: outputTokenCount)
    }

    func maskFreeSpanVerification(_ logits: MLXArray, requestIDs: [CBv2RequestID]) -> MLXArray {
        constraintSampler.mask(logits, requestIDs: requestIDs)
    }
}

/// Test-only observation of the presence/frequency counts a penalty row was
/// scored with (`CBv2DefaultSampler.penaltyCountObserverForTesting`).
struct CBv2PenaltyCountObservation {
    enum Kind { case sample, verify }
    let kind: Kind
    let id: CBv2RequestID
    /// `.sample`: one row of counts; `.verify`: one per window position.
    let counts: [[Float]]
    /// `.verify` only: the confirmed output the engine passed, the round's
    /// drafts used, and the window's first global position.
    var outputTokens: [Int] = []
    var drafts: [Int] = []
    var columnOffset = 0
}
