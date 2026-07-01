// Copyright © 2026 Eigen Labs.

import Foundation
import MLX

/// Chronological (oldest-to-newest) window of recent token ids, trimmed to a
/// fixed capacity.
///
/// Unlike `TokenRing` (used by the penalty processors in `Evaluate.swift`,
/// where match order doesn't matter), tail-pattern loop detection is
/// order-sensitive, so this keeps a strictly time-ordered view via
/// concatenate-and-trim instead of positional overwrite.
struct ChronologicalTokenWindow {
    private(set) var tokens: MLXArray
    private(set) var count = 0
    let capacity: Int

    init(capacity: Int) {
        precondition(capacity > 0)
        self.capacity = capacity
        self.tokens = MLXArray.zeros([0], type: Int32.self)
    }

    /// Bulk-load from a prompt, keeping at most the last `capacity` tokens.
    mutating func loadPrompt(_ prompt: MLXArray) {
        let promptTokens = prompt.reshaped(-1).asType(.int32)
        let n = promptTokens.dim(0)
        if n <= capacity {
            tokens = promptTokens
            count = n
        } else {
            tokens = promptTokens[(n - capacity)...]
            count = capacity
        }
    }

    /// Append a single token, trimming the oldest entry once over capacity.
    mutating func append(_ token: MLXArray) {
        tokens = concatenated([tokens, token.reshaped(1).asType(.int32)])
        count += 1
        if count > capacity {
            tokens = tokens[(count - capacity)...]
            count = capacity
        }
    }
}

/// Detects and breaks a degenerate repeating-cycle loop during generation
/// (e.g. `"a b c a b c a b c ..."`), independent of `repetitionPenalty`/
/// presence/frequency penalties — those only discourage individual tokens
/// and don't reliably stop a model from cycling through a short phrase
/// forever.
///
/// Once the most recent `period * minCount` tokens decompose into `minCount`
/// consecutive identical repeats of some `period` in `1...maxPatternSize`,
/// the token that would continue that cycle is banned (logit set to `-inf`)
/// before the next token is sampled. Periods are checked smallest-first: a
/// genuine short cycle (e.g. period 2) also trivially satisfies the repeat
/// test at its multiples (4, 6, ...), but the smallest matching period is the
/// real fundamental cycle and the one worth breaking.
///
/// `nil`/disabled by default in `GenerateParameters` — this changes sampled
/// output (a hard ban, not a soft penalty), so it's opt-in rather than a
/// silent default for every model family.
public struct TailLoopDetector: LogitProcessor {
    private var window: ChronologicalTokenWindow
    let maxPatternSize: Int
    let minCount: Int

    public init(maxPatternSize: Int = 64, minCount: Int = 3) {
        precondition(maxPatternSize > 0)
        precondition(minCount >= 2)
        self.maxPatternSize = maxPatternSize
        self.minCount = minCount
        self.window = ChronologicalTokenWindow(capacity: maxPatternSize * minCount)
    }

    mutating public func prompt(_ prompt: MLXArray) {
        window.loadPrompt(prompt)
    }

    public func process(logits: MLXArray) -> MLXArray {
        let historyCount = window.count
        guard historyCount >= minCount else { return logits }

        let vocabSize = logits.dim(-1)
        let negInf = MLXArray(-Float.infinity)
        let zero = MLXArray(Float(0))
        var penalty: MLXArray?
        // Tracks (lazily, on-device) whether any smaller period already
        // matched. A genuine period-2 cycle also trivially satisfies the
        // repeat test at its multiples (4, 6, ...), so without this guard
        // every matching multiple would pile on an extra ban — only the
        // smallest (fundamental) matching period should ever ban a token,
        // matching `detectLoopContinuation`'s smallest-period-first + return.
        var alreadyMatched = MLXArray(false)

        for period in 1 ... maxPatternSize {
            let needed = period * minCount
            guard historyCount >= needed else { break }

            // Tail decomposes into `minCount` blocks of length `period`; it's
            // a loop iff every block equals the one after it, i.e. the first
            // (minCount-1) blocks equal the same blocks shifted by `period`.
            let tail = window.tokens[(historyCount - needed)...]
            let earlier = tail[..<(needed - period)]
            let later = tail[period...]
            let matched = MLX.all(earlier .== later)
            let effectiveMatch = logicalAnd(matched, logicalNot(alreadyMatched))

            // `bannedToken` comes straight from generated/prompt history, which
            // for multimodal/direct-token inputs can carry placeholder ids
            // (e.g. an image span) outside `[0, vocabSize)` before the model
            // swaps them for embeddings. Scattering an out-of-range index is
            // unsafe, so gate the actual ban on it being in range and clip the
            // scatter index itself (the continuous-batching sampler already
            // does the equivalent range check before writing -inf).
            let bannedToken = window.tokens[(historyCount - period) ..< (historyCount - period + 1)]
            let inRange = logicalAnd(
                bannedToken .>= MLXArray(Int32(0)), bannedToken .< MLXArray(Int32(vocabSize)))
            let safeToken = clip(bannedToken, min: MLXArray(Int32(0)), max: MLXArray(Int32(vocabSize - 1)))
            let weight = MLX.where(logicalAnd(effectiveMatch, inRange), negInf, zero)
            let contribution = MLXArray.zeros([vocabSize], type: Float32.self)
                .at[safeToken].add(weight)

            penalty = (penalty ?? MLXArray.zeros([vocabSize], type: Float32.self)) + contribution
            alreadyMatched = logicalOr(alreadyMatched, matched)
        }

        guard let penalty else { return logits }
        return logits + penalty.reshaped(1, -1)
    }

    mutating public func didSample(token: MLXArray) {
        window.append(token)
    }
}
