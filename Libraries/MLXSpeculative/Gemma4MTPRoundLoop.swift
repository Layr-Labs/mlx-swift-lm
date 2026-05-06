// Copyright © 2026 Apple Inc.
//
// Gemma 4 Multi-Token Prediction (MTP) round loop, B=1.
//
// Algorithm (one round):
//   1. Draft:  k = blockSize - 1 autoregressive steps through the drafter.
//              Input each step: [target_embed(last_token), last_hidden]
//              concatenated on axis -1. Position offset held constant at
//              the bonus token's absolute position.
//   2. Verify: one target forward over [bonus, ...draftTokens]. Produces
//              bs = k + 1 logits + bs hiddens + shared-KV capture.
//   3. Walk:   SpeculativeWalk.single(draft, main). Returns accepted prefix
//              length + emitted tokens (accepted + 1 tokens).
//   4. Emit:   yield each emitted token as .chunk("<int>") via the
//              AsyncStream continuation.
//   5. Rewind: if accepted < k, trim the target cache by k - accepted.
//   6. Update: next-round hidden = verifyOut.lastHidden[:, accepted:accepted+1, :]
//              next-round bonus = emitted tokens' last entry
//              next-round sharedKV = sliceTail(captured, rejected: k - accepted)
//
// Reference: mlx_vlm/generate.py _mtp_rounds() in Blaizzy/mlx-vlm#1112.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Run the Gemma 4 MTP round loop for a single request (B=1).
///
/// - Parameters:
///   - target: the Gemma 4 target model. Must have been prefilled (i.e.
///     `forwardForMTP(promptTokens, cache:)` called once by the caller
///     before invoking this function) so `firstBonus`, `firstHidden`, and
///     `firstSharedKV` are available.
///   - drafter: the bound Gemma 4 MTP drafter. `bind(target:)` will be
///     called internally; if already bound to `target`, it's idempotent.
///   - targetCache: the KV caches from the prefill forward. Will be
///     advanced and rewound by this function.
///   - firstBonus: the token sampled from the last prefill position
///     (greedy argmax). This is the initial bonus — we yield it
///     immediately as the first stream element.
///   - firstHidden: the last-position hidden from the prefill forward,
///     shape `[B=1, L=1, hiddenSize]`.
///   - firstSharedKV: the shared-KV snapshot from the prefill forward.
///   - maxTokens: maximum total tokens to emit (including firstBonus).
///   - blockSize: speculative block size. Typical range 2-8.
/// - Returns: an `AsyncStream<Generation>` emitting each generated token
///   as `.chunk("<int>")`. The stream finishes when `maxTokens` is
///   reached or the loop terminates for any other reason.
/// - Throws: `Gemma4MTPError.invalidBlockSize` for blockSize < 2 or > 16.
///
/// Tokens are yielded as `.chunk("<int>")` strings; a real tokenizer
/// integration layer (see `generateGemma4MTP` in Task 22) wraps this with
/// `tokenizer.decode(tokenIds:)`.
public func runGemma4MTPRounds(
    target: Gemma4TextModel,
    drafter: Gemma4AssistantDraftModel,
    targetCache: [KVCache],
    firstBonus: Int,
    firstHidden: MLXArray,
    firstSharedKV: Gemma4SharedKV,
    maxTokens: Int,
    blockSize: Int
) throws -> AsyncStream<Generation> {
    guard blockSize >= 2 && blockSize <= 16 else {
        throw Gemma4MTPError.invalidBlockSize(blockSize)
    }
    try drafter.bind(target: target)

    return AsyncStream<Generation> { continuation in
        // First yield: the prefill bonus.
        continuation.yield(.chunk("\(firstBonus)"))
        var emitted = 1

        // Mutable state.
        var bonus = firstBonus
        var hidden = firstHidden
        var sharedKV = firstSharedKV

        while emitted < maxTokens {
            let remaining = maxTokens - emitted
            let bs = min(blockSize, remaining + 1)
            if bs <= 1 { break }
            let k = bs - 1

            // --- Draft (k autoregressive steps) ---
            let driveOffset = targetCache[0].offset
            var draftTokens: [Int] = []
            draftTokens.reserveCapacity(k)
            var tok = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]  // [1, 1]
            var h = hidden
            for _ in 0 ..< k {
                let tokEmbed = target.embedTokensForDrafter(tok)
                let inputsEmbeds = concatenated([tokEmbed, h], axis: -1)
                let (newH, logits) = drafter(
                    inputsEmbeds: inputsEmbeds,
                    sharedKV: sharedKV,
                    positionOffset: .scalar(driveOffset)
                )
                // Greedy sample: logits is [1, 1, vocab] → squeeze to [1, vocab],
                // then argMax over last axis → [1] int32.
                let sampled = logits.squeezed(axis: 1).argMax(axis: -1)
                asyncEval(sampled)
                draftTokens.append(Int(sampled.item(Int32.self)))
                tok = sampled[.newAxis, .ellipsis]  // [1, 1]
                h = newH
            }

            // --- Verify ---
            let verifyIds: [Int32] = [Int32(bonus)] + draftTokens.map { Int32($0) }
            let verifyInput = MLXArray(verifyIds)[.newAxis, .ellipsis]  // [1, bs]
            let verifyOut = target.forwardForMTP(verifyInput, cache: targetCache)
            // Greedy sample all bs positions: [1, bs, vocab] → [1, bs] int32.
            let mainTokens = verifyOut.logits.argMax(axis: -1)
            eval(mainTokens)
            let mainInts = mainTokens.squeezed(axis: 0).asArray(Int.self)  // [bs]

            // --- Walk ---
            let (accepted, newTokens) = SpeculativeWalk.single(
                draft: draftTokens, main: mainInts)

            // --- Emit ---
            var stopEarly = false
            for t in newTokens {
                continuation.yield(.chunk("\(t)"))
                emitted += 1
                if emitted >= maxTokens {
                    stopEarly = true
                    break
                }
            }
            if stopEarly {
                continuation.finish()
                return
            }

            // --- Rewind ---
            if accepted < k {
                target.rollbackSpeculativeCache(
                    targetCache, accepted: .scalar(accepted), blockSize: bs)
            }

            // --- Update state ---
            hidden = verifyOut.lastHidden[
                0..., accepted ..< accepted + 1, 0...]
            bonus = newTokens.last!  // always non-empty (accepted + 1 >= 1)
            let rejected = k - accepted
            sharedKV = Gemma4SharedKV.sliceTail(
                from: verifyOut.capturedSharedKV, rejected: rejected)

            if emitted % 256 == 0 {
                MLX.Memory.clearCache()
            }
        }

        continuation.finish()
    }
}
