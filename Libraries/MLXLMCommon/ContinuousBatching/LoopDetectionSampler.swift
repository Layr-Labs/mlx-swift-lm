// Copyright © 2026 Eigen Labs.
//
// Tail-pattern loop detection for the continuous-batching sampler path.
// Mirrors the host-history + on-device single-token mask shape of
// `makeRepetitionSampler` (Scheduler.swift): the pattern scan runs over the
// plain `[Int]` history already maintained per row by `TokenHistoryHolder`
// (cheap — bounded by `maxPatternSize`, not history length), and only one
// logit is ever touched on device.

import Foundation
import MLX

/// Returns the token id that would continue the most recently completed
/// repeating cycle in `tokens`, if its tail decomposes into `>= minCount`
/// consecutive identical repeats of some period in `1...maxPatternSize`;
/// `nil` otherwise.
///
/// Periods are checked smallest-first: a genuine short cycle (e.g. period 2)
/// also trivially satisfies the repeat test at its multiples (4, 6, ...), but
/// the smallest matching period is the real fundamental cycle and the most
/// useful one to break.
func detectLoopContinuation(tokens: [Int], maxPatternSize: Int, minCount: Int) -> Int? {
    let n = tokens.count
    guard minCount >= 2, n >= minCount else { return nil }
    let upperBound = min(maxPatternSize, n / minCount)
    guard upperBound >= 1 else { return nil }

    for period in 1 ... upperBound {
        let needed = period * minCount
        var isLoop = true
        outer: for repeatIndex in 0 ..< (minCount - 1) {
            let aStart = n - needed + repeatIndex * period
            let bStart = aStart + period
            for offset in 0 ..< period where tokens[aStart + offset] != tokens[bStart + offset] {
                isLoop = false
                break outer
            }
        }
        if isLoop {
            return tokens[n - period]
        }
    }
    return nil
}

/// Wraps `base` with a loop breaker: once `history.tokens` decomposes into a
/// detectable repeating cycle (see ``detectLoopContinuation``), the token
/// that would continue it is banned (logit set to `-inf`) before `base`
/// samples, forcing generation off the loop.
func makeLoopDetectionSampler(
    base: @escaping RowSampler,
    history: TokenHistoryHolder,
    maxPatternSize: Int,
    minCount: Int
) -> RowSampler {
    return { @Sendable logits in
        guard
            let bannedToken = detectLoopContinuation(
                tokens: history.tokens, maxPatternSize: maxPatternSize, minCount: minCount),
            bannedToken >= 0, bannedToken < logits.dim(-1)
        else {
            return base(logits)
        }

        let flat = logits.reshaped(-1).asType(.float32)
        flat[MLXArray([Int32(bannedToken)])] = MLXArray([-Float.infinity])
        return base(flat.reshaped(logits.shape))
    }
}
