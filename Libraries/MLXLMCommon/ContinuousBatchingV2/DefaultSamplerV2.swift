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

public final class CBv2DefaultSampler: CBv2StepSampler {

    private var pipeline: LogitsPipelineV2?
    private let sampler: SamplerV2
    private let constraintSampler = CBv2TokenConstraintSampler()
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
    public var supportsTokenConstraints: Bool { true }

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
        let vocab = logits.dim(-1)
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

        // Apply the grammar exactly once, after arithmetic transforms and
        // before top-k/top-p/min-p. This preserves the valid language even
        // for malformed-but-decodable penalties that can resurrect
        // -infinity, without rebuilding the dense mask twice. Raw logprobs
        // still come from the original unmasked distribution by contract.
        let output = pipeline.process(
            logits,
            rawLogprobsFrom: logits,
            hardMask: constraintSampler.hasRows
                ? { [constraintSampler] transformed in
                    constraintSampler.mask(
                        transformed, requestIDs: requestIDs)
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
        if configuredIDs.contains(id) {
            configuredIDs = []
        }
    }

    // MARK: - MTP target-prefix verify sampling

    public var supportsMTPTargetPrefix: Bool { true }

    /// Pure re-derivation of the ordinary per-token draw for every verify
    /// window position. Eligibility gating guarantees the admitted rows use
    /// only the STATELESS transforms (temperature → top-k/top-p/min-p; no
    /// bias, penalties, constraints, or logprob capture), so this mirrors
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
            stepBases: stepBases, typical: nil
        ).tokens
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
            stepBases: stepBases, typical: (draftIDs: draftIDs, delta: delta))
        return (window.tokens, window.accept!)
    }

    /// Exact acceptance per draft position: keep iff the target token equals
    /// the draft. `tokens` is `[B, W]`, `draftIDs` is `[B, D]`, `D <= W`.
    static func exactAccept(tokens: MLXArray, draftIDs: MLXArray) -> MLXArray {
        tokens[0..., ..<draftIDs.dim(1)] .== draftIDs
    }

    private func verifyWindow(
        logits: MLXArray, params: [CBv2SamplingParams],
        requestIDs: [CBv2RequestID], stepBases: [Int],
        typical: (draftIDs: MLXArray, delta: Float)?
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
        let flat = logits.reshaped([b * w, vocab]).asType(.float32)
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
        let greedyPerPosition = greedyRows.reshaped([b, w])[0..., ..<d]
        return (tokens, which(greedyPerPosition, exact, typicalAccept))
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
    }

    public func tokenConstraintFailure(for id: CBv2RequestID) -> String? {
        constraintSampler.failure(for: id)
    }
}
