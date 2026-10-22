// LogitsPipelineV2.swift
//
// Ordered, vectorized logits-transform pipeline for the ContinuousBatchingV2
// engine (workstream E). Operates on [B, vocab] logits, one row per running
// request, in the vLLM sampler order (research report 08 §6):
//
//   (0) capture RAW logprobs (log_softmax) when any row requests them —
//       BEFORE any transform, so reported logprobs reflect the untampered
//       distribution; top-k gather is lazy (`CBv2Logprobs`).
//   (1) logit-bias scatter-add.
//   (2) penalties — repetition (multiply/divide on presence over the
//       prompt ∪ output window), frequency (α·count over output), presence
//       (β·mask over output). Counts are per-row [B, vocab] tensors
//       maintained INCREMENTALLY via scatter-add each step; they are never
//       re-binned from scratch in the step path.
//   (3) temperature divide (greedy rows use 1.0 — argmax-invariant).
//   (4) top-k / top-p / min-p via a single descending vocab sort + cumsum
//       threshold, per-row parameters broadcast as [B, 1] tensors. When
//       every row samples with a top-k, only the first k sorted columns are
//       thresholded and scattered back (`FilterPlan`); without top-p/min-p
//       the full-vocab sort itself is replaced by an exact stable top-k
//       selection. Every plan yields the full-sort result bit for bit.
//
// Batch-composition invariance: every transform is row-independent (per-row
// reductions along the vocab axis only), so a row's transformed logits
// depend only on that row's raw logits, parameters, and history — never on
// batchmates.
//
// Threading: `process`/`commit` are engine-step-thread ops that only build
// graph nodes (no eval, no `.item()`, no host syncs). `setRows` runs at
// batch-membership change (join/leave) and may loop over rows on the host —
// that is the ONLY place per-row Swift loops are allowed.

import Foundation
import MLX

// MARK: - Row description (membership-change input)

/// Everything the sampling stage needs to know about one batch row.
/// Handed to `LogitsPipelineV2.setRows` / `SamplerV2.setRows` whenever batch
/// membership changes (a request joins, leaves, or is preempted/resumed).
/// Row order MUST match the batch row order of the logits tensors.
public struct CBv2SamplerRow {
    public var id: CBv2RequestID
    public var params: CBv2SamplingParams
    /// Prompt tokens (used to seed the repetition-penalty window).
    public var promptTokens: [Int]
    /// Tokens generated so far (non-empty when a mid-flight request is
    /// re-vectorized after a membership change or preemption resume).
    public var outputTokens: [Int]
    /// Optional row-local inference constraint. The ordinary path is nil.
    public var tokenConstraint: (any CBv2TokenConstraint)?
    /// Request output ceiling, used by the constraint runtime to reserve
    /// enough tokens to close the grammar before the engine length stop.
    public var maxTokens: Int

    public init(
        id: CBv2RequestID, params: CBv2SamplingParams,
        promptTokens: [Int], outputTokens: [Int] = [],
        tokenConstraint: (any CBv2TokenConstraint)? = nil,
        maxTokens: Int = .max
    ) {
        self.id = id
        self.params = params
        self.promptTokens = promptTokens
        self.outputTokens = outputTokens
        self.tokenConstraint = tokenConstraint
        self.maxTokens = maxTokens
    }
}

// MARK: - Pipeline

/// Vectorized [B, vocab] logits pipeline. One instance per engine; rebuild
/// per-row tensors with `setRows` at every batch-membership change, then per
/// step call `process` (pure) and, after sampling, `commit` (state update).
public final class LogitsPipelineV2 {

    /// Temperatures below this are treated as greedy (matches vLLM).
    public static let greedyEpsilon: Float = 1e-5

    public struct Output {
        /// Fully transformed logits [B, vocab] (float32), ready for
        /// greedy argmax or Gumbel-max sampling. Masked tokens are -inf.
        public let sampling: MLXArray
        /// log_softmax of the RAW logits [B, vocab], present only when at
        /// least one row requested logprobs. Consumers gather top-k lazily
        /// via `CBv2Logprobs`.
        public let rawLogprobs: MLXArray?
    }

    public let vocabSize: Int

    /// True when every row is greedy (temperature < `greedyEpsilon`).
    /// `SamplerV2` uses this for the single-argmax fast path.
    public private(set) var allGreedy = true
    /// True when at least one row requested logprobs.
    public private(set) var wantsLogprobs = false
    /// Count of `process` calls that built logprob graph nodes (the raw
    /// log_softmax capture). Telemetry/test hook: batches where no row
    /// requests logprobs must keep this at zero — logprob capture is the
    /// ONLY extra tensor work the logprobs feature adds to the step path.
    public private(set) var logprobBuildCount = 0

    // Per-batch flags, fixed between membership changes so the graph shape
    // is stable step to step (numerically pinned path per composition).
    private var rowCount = 0
    private var anyBias = false
    private var anyRepetition = false
    private var anyFrequencyPresence = false
    private var anyTemperature = false
    private var anyTopKPMinP = false

    /// How stage (4) runs for the configured rows. Every plan produces the
    /// full-sort reference's filtered logits bit for bit.
    enum FilterPlan: Equatable {
        /// One descending full-vocab sort, cumsum and thresholds over every
        /// column (some row keeps more than `maxHeadTopK` tokens, or a
        /// greedy row rides the batch).
        case fullSort
        /// Every row samples with top-k <= `k` and some row also uses top-p
        /// or min-p. The full sort and its softmax stay (the top-p/min-p
        /// thresholds divide by the full-vocab normalizer, summed in sorted
        /// order), but cumsum, thresholds and the scatter back to token
        /// order touch only the first `k` sorted columns: every later rank
        /// is dropped by top-k anyway.
        case sortedHead(k: Int)
        /// Every row samples with top-k <= `k` and no row uses top-p or
        /// min-p. Only the ranks matter, so an exact stable top-k selection
        /// (blockwise sorts) replaces the full-vocab sort.
        case selectedHead(k: Int)
    }
    /// Largest top-k the head plans serve; larger values keep `.fullSort`.
    static let maxHeadTopK = 1024
    private(set) var filterPlan: FilterPlan = .fullSort
    /// Test hook: column limit of one single-threadgroup sort, so tests can
    /// drive the multi-level selection with small vocabularies.
    var sortBlockLimitForTesting: Int?
    /// [1, vocab] int32 ranks for `.fullSort` (built once per vocabulary).
    private var fullRanks: MLXArray?
    /// [1, k] int32 ranks for the head plans (rebuilt when k changes).
    private var headRanks: MLXArray?

    // Per-row parameter tensors, [B, 1] float32 unless noted. Built once in
    // `setRows`; never touched in the step path.
    private var temperature: MLXArray?  // greedy rows pinned to 1.0
    private var repetitionPenalty: MLXArray?
    private var frequencyPenalty: MLXArray?
    private var presencePenalty: MLXArray?
    private var topK: MLXArray?  // [B, 1] int32; disabled rows = vocabSize
    private var topP: MLXArray?  // [B, 1]; disabled rows = 2.0 sentinel
    private var minP: MLXArray?  // [B, 1]; disabled rows = 0.0
    private var biasMatrix: MLXArray?  // [B, vocab] float32

    // Penalty state (incrementally maintained; see `commit`).
    /// Counts of every generated token per row [B, vocab] float32
    /// (frequency/presence penalties — output only, unwindowed).
    private var outputCounts: MLXArray?
    /// Counts of tokens inside the repetition window per row [B, vocab].
    private var repWindowCounts: MLXArray?
    /// Ring buffer of the window's token ids [B, W] int32; -1 = empty slot.
    /// W = max over rows of repetitionContextSize. Rows with a smaller
    /// window only ever use their first `windowSize` slots.
    private var repWindow: MLXArray?
    /// Next ring slot to overwrite per row [B] int32.
    private var ringPos: MLXArray?
    /// Per-row window capacity [B] int32 (≥ 1).
    private var windowSizes: MLXArray?
    /// Cached arange(B) int32 for scatter-add row indices.
    private var rowIndices: MLXArray?

    public init(vocabSize: Int) {
        precondition(vocabSize > 0, "vocabSize must be positive")
        self.vocabSize = vocabSize
    }

    // MARK: Membership change (host loops allowed here, and only here)

    /// Rebuild all per-row tensors for a new batch composition. Row order
    /// must match the logits row order used in `process`/`commit`.
    public func setRows(_ rows: [CBv2SamplerRow]) {
        rowCount = rows.count
        guard !rows.isEmpty else {
            clearState()
            return
        }

        let b = rows.count
        var temps = [Float](repeating: 1, count: b)
        var repPs = [Float](repeating: 1, count: b)
        var freqPs = [Float](repeating: 0, count: b)
        var presPs = [Float](repeating: 0, count: b)
        var topKs = [Int32](repeating: Int32(vocabSize), count: b)
        var topPs = [Float](repeating: 2, count: b)
        var minPs = [Float](repeating: 0, count: b)
        var windows = [Int32](repeating: 1, count: b)

        allGreedy = true
        wantsLogprobs = false
        anyBias = false
        anyRepetition = false
        anyFrequencyPresence = false
        anyTemperature = false
        anyTopKPMinP = false

        for (i, row) in rows.enumerated() {
            let p = row.params
            let greedy = p.temperature < Self.greedyEpsilon
            if !greedy {
                allGreedy = false
                temps[i] = p.temperature
                if p.temperature != 1 { anyTemperature = true }
                // Filtering transforms are argmax-invariant, so they only
                // matter for non-greedy rows.
                if p.topK > 0, p.topK < vocabSize {
                    topKs[i] = Int32(p.topK)
                    anyTopKPMinP = true
                }
                if p.topP > 0, p.topP < 1 {
                    topPs[i] = p.topP
                    anyTopKPMinP = true
                } else if p.topP <= 0 {
                    // top_p == 0 keeps only the most probable token.
                    topPs[i] = 0
                    anyTopKPMinP = true
                }
                if p.minP > 0 {
                    minPs[i] = min(p.minP, 1)
                    anyTopKPMinP = true
                }
            }
            if p.topLogprobs > 0 { wantsLogprobs = true }
            if !p.logitBias.isEmpty { anyBias = true }
            let windowedRepetition = p.repetitionPenalty != 1 && p.repetitionContextSize > 0
            if windowedRepetition {
                repPs[i] = p.repetitionPenalty
                anyRepetition = true
                windows[i] = Int32(max(1, p.repetitionContextSize))
            }
            if p.frequencyPenalty != 0 || p.presencePenalty != 0 {
                freqPs[i] = p.frequencyPenalty
                presPs[i] = p.presencePenalty
                anyFrequencyPresence = true
            }
        }

        filterPlan = Self.filterPlan(
            rows: rows, topKs: topKs, topPs: topPs, minPs: minPs, vocabSize: vocabSize)
        switch filterPlan {
        case .fullSort:
            break
        case .sortedHead(let k), .selectedHead(let k):
            if headRanks?.dim(1) != k { headRanks = Self.ranks(k) }
        }

        temperature = anyTemperature ? MLXArray(temps).reshaped([b, 1]) : nil
        repetitionPenalty = anyRepetition ? MLXArray(repPs).reshaped([b, 1]) : nil
        frequencyPenalty = anyFrequencyPresence ? MLXArray(freqPs).reshaped([b, 1]) : nil
        presencePenalty = anyFrequencyPresence ? MLXArray(presPs).reshaped([b, 1]) : nil
        topK = anyTopKPMinP ? MLXArray(topKs).reshaped([b, 1]) : nil
        topP = anyTopKPMinP ? MLXArray(topPs).reshaped([b, 1]) : nil
        minP = anyTopKPMinP ? MLXArray(minPs).reshaped([b, 1]) : nil

        biasMatrix = anyBias ? Self.buildBiasMatrix(rows: rows, vocabSize: vocabSize) : nil

        rowIndices = MLXArray(Array(0 ..< Int32(b)))
        if anyFrequencyPresence {
            outputCounts = Self.buildCounts(
                tokenLists: rows.map(\.outputTokens), vocabSize: vocabSize)
        } else {
            outputCounts = nil
        }
        if anyRepetition {
            windowSizes = MLXArray(windows)
            let (window, positions) = Self.buildRepetitionWindow(rows: rows, windows: windows)
            repWindow = window
            ringPos = positions
            let windowTokens = rows.enumerated().map { i, row in
                Array((row.promptTokens + row.outputTokens).suffix(Int(windows[i])))
            }
            repWindowCounts = Self.buildCounts(
                tokenLists: windowTokens, vocabSize: vocabSize)
        } else {
            windowSizes = nil
            repWindow = nil
            ringPos = nil
            repWindowCounts = nil
        }
    }

    private func clearState() {
        allGreedy = true
        wantsLogprobs = false
        anyBias = false
        anyRepetition = false
        anyFrequencyPresence = false
        anyTemperature = false
        anyTopKPMinP = false
        filterPlan = .fullSort
        temperature = nil
        repetitionPenalty = nil
        frequencyPenalty = nil
        presencePenalty = nil
        topK = nil
        topP = nil
        minP = nil
        biasMatrix = nil
        outputCounts = nil
        repWindowCounts = nil
        repWindow = nil
        ringPos = nil
        windowSizes = nil
        rowIndices = nil
    }

    // MARK: Step path — graph-build only, no host syncs, no per-row loops

    /// Apply the ordered transform pipeline to raw logits [B, vocab].
    /// Pure: does not mutate pipeline state.
    public func process(
        _ logits: MLXArray,
        rawLogprobsFrom rawLogits: MLXArray? = nil,
        hardMask: ((MLXArray) -> MLXArray)? = nil
    ) -> Output {
        precondition(
            logits.dim(0) == rowCount,
            "logits rows (\(logits.dim(0))) != configured rows (\(rowCount)) — call setRows")

        // Work in float32 for numerically stable softmax/cumsum (vLLM does
        // the same). f16→f32 is exact, so greedy argmax is unaffected.
        var x = logits.asType(.float32)

        // (0) Raw logprobs BEFORE any transform. (The counter is a telemetry
        // side effect only; the transform pipeline itself stays pure.)
        let rawLogprobs: MLXArray?
        if wantsLogprobs {
            logprobBuildCount += 1
            let raw = (rawLogits ?? logits).asType(.float32)
            rawLogprobs = raw - logSumExp(raw, axis: -1, keepDims: true)
        } else {
            rawLogprobs = nil
        }

        // (1) Logit bias.
        if let biasMatrix {
            x = x + biasMatrix
        }

        // (2) Penalties.
        if let repetitionPenalty, let repWindowCounts {
            let present = repWindowCounts .> 0
            let penalized = which(x .> 0, x / repetitionPenalty, x * repetitionPenalty)
            x = which(present, penalized, x)
        }
        if let frequencyPenalty, let presencePenalty, let outputCounts {
            x = x - frequencyPenalty * outputCounts
            x = x - presencePenalty * (outputCounts .> 0).asType(.float32)
        }

        // (3) Temperature (greedy rows pinned to 1.0 in the tensor).
        if let temperature {
            x = x / temperature
        }

        // Arithmetic transforms above can turn a forbidden -infinity into a
        // finite/+infinity value for malformed-but-decodable parameters
        // (for example a negative repetition penalty or temperature). Restore
        // hard language constraints before probability filtering so top-k/p
        // never discard every legal token in favor of a resurrected one.
        if let hardMask {
            x = hardMask(x)
        }

        // (4) top-k / top-p / min-p via one descending sort + cumsum.
        if anyTopKPMinP, let topK, let topP, let minP {
            switch filterPlan {
            case .fullSort:
                if fullRanks == nil {
                    fullRanks = Self.ranks(vocabSize)
                }
                x = Self.applyTopKTopPMinP(
                    x, topK: topK, topP: topP, minP: minP, ranks: fullRanks)
            case .sortedHead:
                x = Self.applyTopKTopPMinPSortedHead(
                    x, topK: topK, topP: topP, minP: minP, headRanks: headRanks!)
            case .selectedHead:
                x = Self.applyTopKSelectedHead(
                    x, topK: topK, headRanks: headRanks!,
                    sortBlockLimit: sortBlockLimitForTesting ?? Self.sortBlockLimit)
            }
        }

        return Output(sampling: x, rawLogprobs: rawLogprobs)
    }

    /// Fold this step's sampled tokens [B] into the incremental penalty
    /// state (scatter-add; never re-binned). Call exactly once per sampled
    /// step, after `process`. Device-only: `sampledTokens` stays on device.
    public func commit(sampledTokens: MLXArray) {
        guard rowCount > 0, anyRepetition || anyFrequencyPresence else { return }
        precondition(
            sampledTokens.dim(0) == rowCount,
            "sampledTokens rows (\(sampledTokens.dim(0))) != configured rows (\(rowCount))")
        guard let rowIndices else { return }
        let tokens = sampledTokens.asType(.int32)

        if let counts = outputCounts {
            outputCounts = counts.at[rowIndices, tokens].add(Float(1))
        }

        if let window = repWindow, let positions = ringPos, let windowSizes,
            let counts = repWindowCounts
        {
            let slot = positions.expandedDimensions(axis: 1)  // [B, 1]
            // Read the token this write evicts BEFORE overwriting it.
            let evicted = takeAlong(window, slot, axis: 1).squeezed(axis: 1)  // [B]
            let valid = (evicted .>= 0).asType(.float32)  // [B]
            let evictedIndex = maximum(evicted, MLXArray(Int32(0)))
            var updated = counts.at[rowIndices, evictedIndex].add(-valid)
            updated = updated.at[rowIndices, tokens].add(Float(1))
            repWindowCounts = updated
            repWindow = putAlong(
                window, slot, values: tokens.expandedDimensions(axis: 1), axis: 1)
            ringPos = remainder(positions + 1, windowSizes)
        }
    }

    /// The incrementally maintained penalty state (immutable device arrays,
    /// so a snapshot is a set of references). Constrained chained decode
    /// restores the state from just before a speculative step's `commit` and
    /// redraws that step exactly.
    struct PenaltyState {
        fileprivate let outputCounts: MLXArray?
        fileprivate let repWindowCounts: MLXArray?
        fileprivate let repWindow: MLXArray?
        fileprivate let ringPos: MLXArray?
    }

    var penaltyState: PenaltyState {
        PenaltyState(
            outputCounts: outputCounts, repWindowCounts: repWindowCounts,
            repWindow: repWindow, ringPos: ringPos)
    }

    /// Restore a snapshot taken under the SAME `setRows` configuration.
    func restore(_ state: PenaltyState) {
        outputCounts = state.outputCounts
        repWindowCounts = state.repWindowCounts
        repWindow = state.repWindow
        ringPos = state.ringPos
    }

    // MARK: - Vectorized top-k/top-p/min-p

    /// [1, n] int32 ranks 0 ..< n.
    static func ranks(_ n: Int) -> MLXArray {
        MLXArray(Array(0 ..< Int32(n))).reshaped([1, n])
    }

    /// Stage (4) plan for one batch composition. Head plans need every row
    /// to sample (greedy rows keep the whole vocabulary) with an active
    /// top-k no larger than `maxHeadTopK`.
    static func filterPlan(
        rows: [CBv2SamplerRow], topKs: [Int32], topPs: [Float], minPs: [Float], vocabSize: Int
    ) -> FilterPlan {
        guard !rows.isEmpty else { return .fullSort }
        var k = 0
        for (i, row) in rows.enumerated() {
            guard row.params.temperature >= greedyEpsilon, Int(topKs[i]) < vocabSize else {
                return .fullSort
            }
            k = max(k, Int(topKs[i]))
        }
        guard k > 0, k <= maxHeadTopK else { return .fullSort }
        let usesProbabilityFilters = zip(topPs, minPs).contains { $0 != 2 || $1 != 0 }
        return usesProbabilityFilters ? .sortedHead(k: k) : .selectedHead(k: k)
    }

    /// Single-sort implementation of stage (4). All three filters share one
    /// descending sort and one cumsum (vLLM's native path). The most
    /// probable token of each row is kept by construction, so a fully
    /// masked row cannot exist.
    ///
    /// The masked sorted row returns to token order by scattering it through
    /// the sort permutation (`out[sortedIndices[j]] = masked[j]`), which is
    /// the inverse-permutation gather without materializing the inverse.
    static func applyTopKTopPMinP(
        _ logits: MLXArray, topK: MLXArray, topP: MLXArray, minP: MLXArray,
        ranks cachedRanks: MLXArray? = nil
    ) -> MLXArray {
        let vocab = logits.dim(-1)
        // Descending sort (stable: ties keep ascending token order).
        let sortedIndices = argSort(-logits, axis: -1)
        let sortedLogits = takeAlong(logits, sortedIndices, axis: -1)
        let probs = softmax(sortedLogits, axis: -1)
        let cumulative = cumsum(probs, axis: -1)

        // top-k: keep ranks < k (disabled rows carry k == vocab).
        let ranks = cachedRanks ?? Self.ranks(vocab)
        var keep = ranks .< topK
        // top-p: keep tokens whose preceding cumulative mass is within p
        // (the first token crossing the threshold stays in). Disabled rows
        // carry the sentinel 2.0, which keeps everything.
        keep = keep .&& ((cumulative - probs) .<= topP)
        // min-p: drop tokens below minP × max prob (column 0 after the
        // descending sort). Disabled rows carry 0.
        let maxProb = probs[0..., ..<1]
        keep = keep .&& (probs .>= (minP * maxProb))

        let masked = which(keep, sortedLogits, MLXArray(-Float.infinity))
        // Every column of `sortedIndices` is a permutation of the row, so
        // the scatter overwrites every element of the base.
        return putAlong(sortedLogits, sortedIndices, values: masked, axis: -1)
    }

    /// `.sortedHead`: the full-sort result when every row's top-k is at most
    /// `headRanks.dim(1)`. Ranks at or past the head are masked by top-k in
    /// the reference, so they are -inf here without being thresholded. The
    /// softmax stays full-vocab: its normalizer is summed over the whole
    /// sorted row and feeds both probability filters. An inclusive cumsum's
    /// first `k` outputs depend only on the first `k` inputs, combined in
    /// the same order by MLX's scan kernel (one block, k <= 4096), so the
    /// head cumsum equals the reference's first `k` columns bit for bit.
    static func applyTopKTopPMinPSortedHead(
        _ logits: MLXArray, topK: MLXArray, topP: MLXArray, minP: MLXArray,
        headRanks: MLXArray
    ) -> MLXArray {
        let k = headRanks.dim(1)
        let sortedIndices = argSort(-logits, axis: -1)
        let sortedLogits = takeAlong(logits, sortedIndices, axis: -1)
        let probs = softmax(sortedLogits, axis: -1)
        let headProbs = probs[0..., ..<k]
        let cumulative = cumsum(headProbs, axis: -1)

        var keep = headRanks .< topK
        keep = keep .&& ((cumulative - headProbs) .<= topP)
        let maxProb = probs[0..., ..<1]
        keep = keep .&& (headProbs .>= (minP * maxProb))

        let masked = which(keep, sortedLogits[0..., ..<k], MLXArray(-Float.infinity))
        return putAlong(
            MLXArray.full(logits.shape, values: MLXArray(-Float.infinity)),
            sortedIndices[0..., ..<k], values: masked, axis: -1)
    }

    /// Column capacity of MLX's single-threadgroup sort (512 threads x 4).
    static let sortBlockLimit = 2048

    /// `.selectedHead`: the full-sort result when every row samples with
    /// top-k <= `headRanks.dim(1)` and neither top-p nor min-p is set. The
    /// reference then keeps exactly ranks < k of its stable descending sort
    /// (ties in ascending token order) — unless its softmax is NaN, which
    /// happens exactly when the row holds a NaN or +inf or is entirely
    /// -inf; such a row comes back entirely -inf.
    static func applyTopKSelectedHead(
        _ logits: MLXArray, topK: MLXArray, headRanks: MLXArray, sortBlockLimit: Int
    ) -> MLXArray {
        let k = headRanks.dim(1)
        let head = stableHead(keys: -logits, k: k, sortBlockLimit: sortBlockLimit)
        let headLogits = takeAlong(logits, head.indices, axis: -1)
        // Rank 0 is the row maximum: +inf, or -inf for an all -inf row.
        let invalid = head.anyNaN .|| isInf(headLogits[0..., ..<1])
        let keep = (headRanks .< topK) .&& logicalNot(invalid)
        let masked = which(keep, headLogits, MLXArray(-Float.infinity))
        return putAlong(
            MLXArray.full(logits.shape, values: MLXArray(-Float.infinity)),
            head.indices, values: masked, axis: -1)
    }

    /// The first `k` columns of `argSort(keys, axis: -1)` for [B, n] keys,
    /// exactly, without sorting all `n` columns: sort blocks of at most
    /// `sortBlockLimit` columns (stable, one threadgroup each), keep each
    /// block's first `k`, repeat on the concatenated candidates, then sort
    /// the final candidates once. MLX's sorts are stable merge sorts with
    /// NaN last, i.e. one total order (key, then column); every element of
    /// the global head is in its block's head under that order, and
    /// candidates keep column order among equal keys (block-major, stable
    /// within a block), so the final ranks are the global ranks.
    ///
    /// A level that would not shrink the candidates (k near the block size)
    /// ends the loop; the final sort then covers every remaining candidate.
    ///
    /// Padding columns carry +inf keys past every real column: they rank
    /// after every real non-NaN key, so they can displace only NaN keys —
    /// whose rows `anyNaN` reports ([B, 1] bool, from each first-level
    /// block's last sorted key).
    static func stableHead(
        keys: MLXArray, k: Int, sortBlockLimit: Int
    ) -> (indices: MLXArray, anyNaN: MLXArray) {
        let b = keys.dim(0)
        var candidates = keys
        var columns: MLXArray?  // nil: candidate column == global column
        var anyNaN: MLXArray?
        while candidates.dim(1) > sortBlockLimit {
            let n = candidates.dim(1)
            let (blocks, size) = blockShape(n, limit: sortBlockLimit)
            let width = min(k, size)
            // No progress (k close to the block size): sort what is left.
            guard blocks * width < n else { break }
            let pad = blocks * size - n
            var padded = candidates
            if pad > 0 {
                padded = concatenated(
                    [padded, MLXArray.full([b, pad], values: MLXArray(Float.infinity))], axis: 1)
            }
            let grouped = padded.reshaped([b, blocks, size])
            let order = argSort(grouped, axis: -1)
            if anyNaN == nil {
                let last = takeAlong(grouped, order[0..., 0..., (size - 1)...], axis: -1)
                anyNaN = any(isNaN(last), axes: [1, 2]).reshaped([b, 1])
            }
            let top = order[0..., 0..., ..<width]
            candidates = takeAlong(grouped, top, axis: -1).reshaped([b, blocks * width])
            if let previous = columns {
                var paddedColumns = previous
                if pad > 0 {
                    paddedColumns = concatenated(
                        [paddedColumns, MLXArray.zeros([b, pad], type: Int32.self)], axis: 1)
                }
                columns = takeAlong(
                    paddedColumns.reshaped([b, blocks, size]), top, axis: -1
                ).reshaped([b, blocks * width])
            } else {
                let base = MLXArray((0 ..< Int32(blocks)).map { $0 * Int32(size) })
                columns = (top.asType(.int32) + base.reshaped([1, blocks, 1]))
                    .reshaped([b, blocks * width])
            }
        }
        let order = argSort(candidates, axis: -1)
        if anyNaN == nil {
            let last = candidates.dim(1) - 1
            anyNaN = isNaN(takeAlong(candidates, order[0..., last...], axis: -1))
        }
        let top = order[0..., ..<k]
        let indices = columns.map { takeAlong($0, top, axis: -1) } ?? top.asType(.int32)
        return (indices, anyNaN!)
    }

    /// `(blocks, size)` with `size <= limit` and `blocks * size >= n`,
    /// preferring an exact split (no padding) within 25% more blocks.
    static func blockShape(_ n: Int, limit: Int) -> (blocks: Int, size: Int) {
        let minimum = (n + limit - 1) / limit
        for blocks in minimum ... max(minimum, minimum + minimum / 4) where n % blocks == 0 {
            return (blocks, n / blocks)
        }
        return (minimum, (n + minimum - 1) / minimum)
    }

    // MARK: - Membership-change tensor builders

    private static func buildBiasMatrix(rows: [CBv2SamplerRow], vocabSize: Int) -> MLXArray {
        var rowIdx = [Int32]()
        var colIdx = [Int32]()
        var values = [Float]()
        for (i, row) in rows.enumerated() {
            for (token, bias) in row.params.logitBias where token >= 0 && token < vocabSize {
                rowIdx.append(Int32(i))
                colIdx.append(Int32(token))
                values.append(bias)
            }
        }
        let zeros = MLXArray.zeros([rows.count, vocabSize], type: Float.self)
        guard !rowIdx.isEmpty else { return zeros }
        return zeros.at[MLXArray(rowIdx), MLXArray(colIdx)].add(MLXArray(values))
    }

    /// Bin an arbitrary token list per row into [B, vocab] float32 counts.
    /// Membership-change only (one scatter-add over all rows' tokens).
    private static func buildCounts(tokenLists: [[Int]], vocabSize: Int) -> MLXArray {
        var rowIdx = [Int32]()
        var colIdx = [Int32]()
        for (i, tokens) in tokenLists.enumerated() {
            for token in tokens where token >= 0 && token < vocabSize {
                rowIdx.append(Int32(i))
                colIdx.append(Int32(token))
            }
        }
        let zeros = MLXArray.zeros([tokenLists.count, vocabSize], type: Float.self)
        guard !rowIdx.isEmpty else { return zeros }
        return zeros.at[MLXArray(rowIdx), MLXArray(colIdx)].add(Float(1))
    }

    /// Seed the repetition ring buffer with the most recent
    /// min(windowSize, |history|) tokens of prompt ∪ output. When the
    /// window is full the oldest token sits at the row's ring position, so
    /// the next write evicts it; when partially filled, writes land on -1
    /// (empty) slots first.
    private static func buildRepetitionWindow(
        rows: [CBv2SamplerRow], windows: [Int32]
    ) -> (window: MLXArray, ringPos: MLXArray) {
        let b = rows.count
        let maxWindow = Int(windows.max() ?? 1)
        var flat = [Int32](repeating: -1, count: b * maxWindow)
        var positions = [Int32](repeating: 0, count: b)
        for (i, row) in rows.enumerated() {
            let w = Int(windows[i])
            let history = row.promptTokens + row.outputTokens
            let recent = history.suffix(w)
            for (j, token) in recent.enumerated() {
                flat[i * maxWindow + j] = token >= 0 ? Int32(token) : -1
            }
            positions[i] = recent.count == w ? 0 : Int32(recent.count)
        }
        return (MLXArray(flat, [b, maxWindow]), MLXArray(positions))
    }
}

// MARK: - Lazy logprob gather

/// Lazy top-k gather over the raw logprobs captured by the pipeline.
/// `gather` builds graph nodes only (safe on the engine thread);
/// `assemble` materializes host values and MUST run off the engine step
/// thread (it synchronizes).
public enum CBv2Logprobs {

    public struct Gathered {
        /// Logprob of the sampled token per row [B].
        public let chosen: MLXArray
        /// Top-k logprob values per row [B, k], descending.
        public let topValues: MLXArray
        /// Token ids matching `topValues` [B, k].
        public let topIndices: MLXArray
    }

    /// Gather the sampled token's logprob and the top-k alternatives from
    /// `rawLogprobs` [B, vocab]. `k` should be the max `topLogprobs` across
    /// rows; per-row truncation happens in `assemble`.
    public static func gather(
        rawLogprobs: MLXArray, sampledTokens: MLXArray, k: Int
    ) -> Gathered {
        let tokens = sampledTokens.asType(.int32).expandedDimensions(axis: 1)
        let chosen = takeAlong(rawLogprobs, tokens, axis: -1).squeezed(axis: 1)
        let vocab = rawLogprobs.dim(-1)
        let kEff = max(0, min(k, vocab))
        guard kEff > 0 else {
            let empty = MLXArray.zeros([rawLogprobs.dim(0), 0], type: Float.self)
            return Gathered(chosen: chosen, topValues: empty, topIndices: empty.asType(.int32))
        }
        let sortedIndices = argSort(-rawLogprobs, axis: -1)[0..., ..<kEff]
        let sortedValues = takeAlong(rawLogprobs, sortedIndices, axis: -1)
        return Gathered(
            chosen: chosen, topValues: sortedValues, topIndices: sortedIndices.asType(.int32))
    }

    /// Turn a materialized `Gathered` plus host token ids into per-row
    /// `CBv2TokenLogprob`s. Rows whose `topLogprobs` is 0 get nil.
    /// Synchronizes on the arrays — only call at a sanctioned readback
    /// boundary: the engine calls this at step FINALIZATION, right where
    /// the sampled tokens are read back (the arrays rode the step's
    /// `asyncEval`, so no extra GPU work is forced), never mid graph-build.
    public static func assemble(
        _ gathered: Gathered, sampledTokens: [Int], topLogprobsPerRow: [Int]
    ) -> [CBv2TokenLogprob?] {
        // Three host readbacks per logprob segment, each counted (gated
        // like every CBv2 sync point; the arrays are already evaluated).
        let chosen = gathered.chosen.asArray(Float.self)
        CBv2CoreInstrumentation.recordHostSync()
        let k = gathered.topValues.dim(-1)
        let values = gathered.topValues.asArray(Float.self)
        CBv2CoreInstrumentation.recordHostSync()
        let indices = gathered.topIndices.asArray(Int32.self)
        CBv2CoreInstrumentation.recordHostSync()
        var out = [CBv2TokenLogprob?]()
        out.reserveCapacity(sampledTokens.count)
        for row in 0 ..< sampledTokens.count {
            let want = min(topLogprobsPerRow[row], k)
            guard want > 0 else {
                out.append(nil)
                continue
            }
            var top = [(token: Int, logprob: Float)]()
            top.reserveCapacity(want)
            for j in 0 ..< want {
                top.append((token: Int(indices[row * k + j]), logprob: values[row * k + j]))
            }
            out.append(
                CBv2TokenLogprob(
                    token: sampledTokens[row], logprob: chosen[row], topLogprobs: top))
        }
        return out
    }
}

// MARK: - Per-step logprob handoff (sampler → engine loop)

/// One `sample` call's LAZY logprob gather, handed from the sampler to the
/// engine loop (`CBv2StepSampler.takeStepLogprobs`). `gathered` holds
/// graph-only handles over the RAW (pre-transform) logprobs; the loop adds
/// `evalTargets` to the step's `asyncEval` set and materializes the values
/// at the finalize boundary — one step late, alongside the sampled tokens,
/// matching the engine's readback discipline.
public struct CBv2StepLogprobs {
    /// Row ids in the gather's row order (== the `sample` call's row order).
    public let rows: [CBv2RequestID]
    /// Each row's requested `topLogprobs` (0 ⇒ that row reports nothing).
    public let topLogprobsPerRow: [Int]
    public let gathered: CBv2Logprobs.Gathered

    public init(
        rows: [CBv2RequestID], topLogprobsPerRow: [Int], gathered: CBv2Logprobs.Gathered
    ) {
        self.rows = rows
        self.topLogprobsPerRow = topLogprobsPerRow
        self.gathered = gathered
    }

    /// Arrays the engine adds to the step's `asyncEval` so finalization's
    /// readback finds them already computed.
    public var evalTargets: [MLXArray] {
        [gathered.chosen, gathered.topValues, gathered.topIndices]
    }
}
