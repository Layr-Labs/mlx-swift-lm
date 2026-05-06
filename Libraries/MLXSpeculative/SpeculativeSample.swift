// Copyright © 2026 Apple Inc.
//
// Rejection-based speculative sampling for stochastic MTP (temperature > 0).
//
// Algorithm (Leviathan et al. 2023 / Chen et al. 2023):
//
// Per round, the drafter proposes K candidate tokens `c_1..c_K` sampled
// from its own distributions `q_1..q_K`. The target verifies by running
// one forward over `[bonus, c_1, ..., c_K]`, producing distributions
// `p_0..p_K` at each position.
//
// Walk: for each i in 0..K-1
//   - r ~ Uniform(0, 1)
//   - α = min(1, p_i(c_i) / q_i(c_i))
//   - if r < α: accept c_i
//   - else: reject. Resample from normalize(max(p_i - q_i, 0)). Stop.
// If all K accepted: sample extra token from p_K.
//
// At temperature=0, p and q are point masses. p_i(c_i)/q_i(c_i) is 1 if
// they agree (always accept), 0 if they disagree (always reject, resample
// from p_i = target's argmax). Degenerates to greedy walker.

import Foundation
import MLX
import MLXRandom

/// Result of one round of rejection-based speculative sampling.
public struct SpeculativeSampleResult: Sendable {
    /// Number of drafter tokens accepted (0...K).
    public let accepted: Int
    /// Tokens emitted by this round: first `accepted` drafter tokens,
    /// followed by one corrective / bonus token. Always `accepted + 1`
    /// entries.
    public let emitted: [Int]

    public init(accepted: Int, emitted: [Int]) {
        self.accepted = accepted
        self.emitted = emitted
    }
}

/// Apply temperature to logits. Returns softmax probabilities along the
/// last axis.
private func softmaxWithTemperature(_ logits: MLXArray, temperature: Float) -> MLXArray {
    let scaled = temperature > 0 ? logits / temperature : logits
    return softmax(scaled.asType(.float32), axis: -1)
}

/// Sample one token from a distribution `[vocab]` (or `[..., vocab]`)
/// along the last axis. Returns an `[...]` int32 array.
private func sampleFromDistribution(_ probs: MLXArray, key: MLXArray) -> MLXArray {
    // MLX has `MLXRandom.categorical(logits)` but we have probabilities.
    // Use cumulative sum + uniform random number comparison.
    let u = MLXRandom.uniform(low: Float(0), high: Float(1), key: key)
    let cdf = probs.cumsum(axis: -1)
    // count positions where cdf < u; that's the sampled index.
    let lt = cdf .< u
    return lt.sum(axis: -1).asType(.int32)
}

/// Run rejection-based speculative sampling for one round.
///
/// - Parameters:
///   - draftLogits: [K, vocab] — drafter logits at each of the K proposed
///     positions, *pre-temperature* (raw drafter output).
///   - draftTokens: [K] — tokens the drafter sampled.
///   - verifyLogits: [bs=K+1, vocab] — target logits at each verify
///     position, *pre-temperature* (raw target output).
///   - temperature: sampling temperature (> 0).
///   - rngKey: MLX random key for the uniform draws.
/// - Returns: `(accepted, emitted)` — number of drafter tokens accepted
///   and the list of emitted tokens for this round.
public func speculativeSampleRound(
    draftLogits: MLXArray,
    draftTokens: [Int],
    verifyLogits: MLXArray,
    temperature: Float,
    rngKey: MLXArray
) -> SpeculativeSampleResult {
    precondition(temperature > 0, "stochastic path requires temperature > 0")
    let K = draftTokens.count
    precondition(verifyLogits.dim(0) == K + 1,
        "verifyLogits must have K+1 positions")
    precondition(draftLogits.dim(0) == K,
        "draftLogits must have K positions")

    // Apply temperature + softmax, once.
    let q = softmaxWithTemperature(draftLogits, temperature: temperature)  // [K, vocab]
    let p = softmaxWithTemperature(verifyLogits, temperature: temperature)  // [K+1, vocab]

    // Split the rng key into 2K subkeys (K for accept r_i, K for resample
    // + final sample).
    let keys = MLXRandom.split(key: rngKey, into: 2 * (K + 1))

    var accepted = 0
    var emitted: [Int] = []
    for i in 0 ..< K {
        let c = draftTokens[i]
        let pi = p[i]  // [vocab]
        let qi = q[i]  // [vocab]
        let pc = pi[c].item(Float.self)
        let qc = qi[c].item(Float.self)
        let alpha = min(Float(1), qc > 0 ? pc / qc : 0)
        let r = MLXRandom.uniform(
            low: Float(0), high: Float(1), key: keys[i]).item(Float.self)
        if r < alpha {
            emitted.append(c)
            accepted += 1
        } else {
            // Reject. Resample from normalize(max(pi - qi, 0)).
            let diff = MLX.maximum(pi - qi, MLXArray(Float(0)))
            let s = diff.sum().item(Float.self)
            let resampleDist: MLXArray
            if s > 0 {
                resampleDist = diff / s
            } else {
                // Numerical edge case: p == q exactly. Fall back to p
                // which in theory should sum to 1 already.
                resampleDist = pi
            }
            let nextTok = sampleFromDistribution(
                resampleDist, key: keys[K + i])
            emitted.append(Int(nextTok.item(Int32.self)))
            return SpeculativeSampleResult(accepted: accepted, emitted: emitted)
        }
    }
    // All K accepted — sample one bonus from p_K.
    let bonus = sampleFromDistribution(p[K], key: keys[2 * K])
    emitted.append(Int(bonus.item(Int32.self)))
    return SpeculativeSampleResult(accepted: accepted, emitted: emitted)
}
