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
            //
            // Keep sampled drafts on GPU; concat with bonus for the verify
            // input in a single eval — avoids K separate CPU syncs per round.
            let driveOffset = targetCache[0].offset
            var tok = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]  // [1, 1]
            var h = hidden
            var draftPerStep: [MLXArray] = []  // each [1, 1]
            draftPerStep.reserveCapacity(k)
            for _ in 0 ..< k {
                let tokEmbed = target.embedTokensForDrafter(tok)
                let inputsEmbeds = concatenated([tokEmbed, h], axis: -1)
                let (newH, logits) = drafter(
                    inputsEmbeds: inputsEmbeds,
                    sharedKV: sharedKV,
                    positionOffset: .scalar(driveOffset)
                )
                // Greedy sample: logits [1, 1, vocab] → [1, 1] int32.
                let sampled = logits.squeezed(axis: 1).argMax(axis: -1)  // [1]
                let sampled2d = sampled[.newAxis, .ellipsis]  // [1, 1]
                draftPerStep.append(sampled2d)
                tok = sampled2d
                h = newH
            }

            // --- Verify ---
            // Concat drafts [1, k] + bonus [1, 1] → [1, bs].
            let bonusCol = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]  // [1, 1]
            let verifyInput: MLXArray =
                draftPerStep.isEmpty
                ? bonusCol
                : concatenated([bonusCol] + draftPerStep, axis: 1)
            let verifyOut = target.forwardForMTP(verifyInput, cache: targetCache)
            // Greedy sample all bs positions: [1, bs, vocab] → [1, bs] int32.
            // Cast to fp32 before argmax to eliminate bf16 near-uniform-tail
            // argmax flips that can otherwise diverge MTP from baseline at
            // post-EOS repetition tails.
            let mainTokens = verifyOut.logits.asType(.float32).argMax(axis: -1)
            // Materialise drafts + main tokens in a single sync.
            let draftConcat: MLXArray =
                draftPerStep.isEmpty
                ? MLXArray.zeros([1, 0], dtype: .int32)
                : concatenated(draftPerStep, axis: 1)  // [1, k]
            eval(mainTokens, draftConcat)
            let mainInts = mainTokens.squeezed(axis: 0).asArray(Int.self)  // [bs]
            let draftTokens = draftConcat.squeezed(axis: 0).asArray(Int32.self)
                                         .map { Int($0) }  // [k]

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

// MARK: - Batched (B > 1) output type

/// Per-step output of the B>1 MTP round loop. Each `slots[i]` maps to
/// the original batch row `i`.
public struct BatchedGeneration: Sendable {
    public let slots: [Slot]

    public struct Slot: Sendable {
        /// Original index in the input batch.
        public let row: Int
        /// Token emitted this step; nil when the row is finished or the
        /// step didn't produce output for this row.
        public let token: Int?
        /// Non-nil on the step where the row finishes.
        public let finishReason: FinishReason?

        public init(row: Int, token: Int?, finishReason: FinishReason?) {
            self.row = row
            self.token = token
            self.finishReason = finishReason
        }
    }

    public enum FinishReason: Sendable {
        case stop
        case eos
        case length
    }

    public init(slots: [Slot]) {
        self.slots = slots
    }
}

// MARK: - B > 1 round loop

/// Run the Gemma 4 MTP round loop for a batch of requests (B > 1).
///
/// Mirrors `runGemma4MTPRounds` but tracks per-row acceptance counts and
/// per-row finishedness. Rows that finish (reach maxTokens or emit an
/// EOS token) stay in the batch but stop emitting — their
/// `BatchedGeneration.Slot.token` becomes nil. The round-loop terminates
/// when every row is finished.
///
/// - Note: Continuous batching (removing finished rows via
///   `BatchedCache.filterBatched` to shrink the active batch) is a
///   follow-up optimization and not implemented here. All B rows run
///   every round regardless of finishedness. This is a correctness-first
///   v1; throughput optimization is Task 26's remit.
public func runGemma4MTPRoundsBatched(
    target: Gemma4TextModel,
    drafter: Gemma4AssistantDraftModel,
    targetCache: [KVCache],
    firstBonus: [Int],
    firstHidden: MLXArray,
    firstSharedKV: Gemma4SharedKV,
    maxTokens: Int,
    blockSize: Int,
    eosTokenIds: Set<Int>?
) throws -> AsyncStream<BatchedGeneration> {
    guard blockSize >= 2 && blockSize <= 16 else {
        throw Gemma4MTPError.invalidBlockSize(blockSize)
    }
    try drafter.bind(target: target)
    let B = firstBonus.count

    return AsyncStream<BatchedGeneration> { continuation in
        // First yield: prefill bonus per row.
        var firstSlots: [BatchedGeneration.Slot] = []
        firstSlots.reserveCapacity(B)
        var emitted = Array(repeating: 1, count: B)
        var finished = Array(repeating: false, count: B)
        for (i, b) in firstBonus.enumerated() {
            if let eosTokenIds, eosTokenIds.contains(b) {
                finished[i] = true
                firstSlots.append(.init(row: i, token: b, finishReason: .eos))
            } else {
                firstSlots.append(.init(row: i, token: b, finishReason: nil))
            }
        }
        continuation.yield(BatchedGeneration(slots: firstSlots))
        if finished.allSatisfy({ $0 }) {
            continuation.finish()
            return
        }

        // Mutable state.
        var bonus: [Int] = firstBonus
        var hidden = firstHidden
        var sharedKV = firstSharedKV

        while !finished.allSatisfy({ $0 }) {
            // Determine this round's block size. Use the min `remaining`
            // across active (non-finished) rows.
            let activeRemaining = emitted.enumerated()
                .filter { !finished[$0.offset] }
                .map { maxTokens - $0.element }
            guard let minRemaining = activeRemaining.min(), minRemaining > 0 else {
                break
            }
            let bs = min(blockSize, minRemaining + 1)
            if bs <= 1 { break }
            let k = bs - 1

            // --- Draft (k autoregressive steps) ---
            //
            // Keep sampled draft tokens on the GPU instead of materialising
            // each step as a Swift array. Each step's sampled [B] token
            // tensor is collected; at the end we stack them [B, k] and
            // prepend the bonus to form the verify input in one eval.
            let driveOffset = targetCache[0].offset
            // Seed token: per-row bonus as [B, 1] int32.
            var tok = MLXArray(bonus.map { Int32($0) }, [B, 1])
            var h = hidden
            var draftPerStep: [MLXArray] = []  // each is shape [B, 1]
            draftPerStep.reserveCapacity(k)
            for _ in 0 ..< k {
                let tokEmbed = target.embedTokensForDrafter(tok)
                let inputsEmbeds = concatenated([tokEmbed, h], axis: -1)
                let (newH, logits) = drafter(
                    inputsEmbeds: inputsEmbeds,
                    sharedKV: sharedKV,
                    positionOffset: .scalar(driveOffset)
                )
                // Greedy sample per row: logits [B, 1, vocab] → [B, 1]
                let sampled = logits.squeezed(axis: 1).argMax(axis: -1)  // [B]
                let sampled2d = sampled.reshaped([B, 1])  // [B, 1]
                draftPerStep.append(sampled2d)
                tok = sampled2d
                h = newH
            }

            // --- Verify ---
            // Concat drafts along axis=1 to get [B, k], prepend bonus to
            // get [B, bs=k+1]. Single eval across the whole verify step.
            let bonusCol = MLXArray(bonus.map { Int32($0) }, [B, 1])
            let verifyInput: MLXArray =
                draftPerStep.isEmpty
                ? bonusCol
                : concatenated([bonusCol] + draftPerStep, axis: 1)
            let verifyOut = target.forwardForMTP(verifyInput, cache: targetCache)
            // fp32 argmax at verify: see comment in B=1 path above.
            let mainTokens = verifyOut.logits.asType(.float32).argMax(axis: -1)  // [B, bs]
            // Materialise drafts + main tokens in a single sync.
            let draftConcat: MLXArray =
                draftPerStep.isEmpty
                ? MLXArray.zeros([B, 0], dtype: .int32)
                : concatenated(draftPerStep, axis: 1)  // [B, k]
            eval(mainTokens, draftConcat)
            let mainFlat = mainTokens.asArray(Int32.self)
            let draftFlat = draftConcat.asArray(Int32.self)
            var mainPerRow: [[Int]] = []
            var draftTokensPerRow: [[Int]] = []
            mainPerRow.reserveCapacity(B)
            draftTokensPerRow.reserveCapacity(B)
            for bi in 0 ..< B {
                let mStart = bi * bs
                mainPerRow.append(mainFlat[mStart ..< mStart + bs].map { Int($0) })
                let dStart = bi * k
                draftTokensPerRow.append(draftFlat[dStart ..< dStart + k].map { Int($0) })
            }

            // --- Walk (per-row, with remaining-budget truncation) ---
            let budgets = emitted.map { maxTokens - $0 }
            let (accepted, newTokensPerRow) = SpeculativeWalk.batched(
                draft: draftTokensPerRow,
                main: mainPerRow,
                budgets: budgets
            )
            let maxAcceptedInt = accepted.max() ?? 0

            // --- Emit one BatchedGeneration per token-position within this round ---
            let maxNew = newTokensPerRow.map(\.count).max() ?? 0
            for pos in 0 ..< maxNew {
                var slots: [BatchedGeneration.Slot] = []
                slots.reserveCapacity(B)
                for bi in 0 ..< B {
                    if finished[bi] {
                        slots.append(.init(row: bi, token: nil, finishReason: nil))
                        continue
                    }
                    let row = newTokensPerRow[bi]
                    if pos < row.count {
                        let t = row[pos]
                        emitted[bi] += 1
                        var finishReason: BatchedGeneration.FinishReason? = nil
                        if let eosTokenIds, eosTokenIds.contains(t) {
                            finished[bi] = true
                            finishReason = .eos
                        } else if emitted[bi] >= maxTokens {
                            finished[bi] = true
                            finishReason = .length
                        }
                        slots.append(.init(row: bi, token: t, finishReason: finishReason))
                    } else {
                        slots.append(.init(row: bi, token: nil, finishReason: nil))
                    }
                }
                continuation.yield(BatchedGeneration(slots: slots))
            }

            if finished.allSatisfy({ $0 }) {
                continuation.finish()
                return
            }

            // --- Rewind target cache ---
            // Fast path: if every row accepted the same count, use the
            // scalar rewind (uniform cache trim, no per-row zero-tail).
            // This is common when prompts are identical or strongly
            // aligned — typical for synthetic benchmarks and real serving
            // with prompt-prefix caching.
            let uniformAccepted = accepted.allSatisfy { $0 == accepted[0] }
            if maxAcceptedInt < k {
                if uniformAccepted {
                    target.rollbackSpeculativeCache(
                        targetCache,
                        accepted: .scalar(accepted[0]),
                        blockSize: bs
                    )
                } else {
                    let acceptedArr = MLXArray(accepted.map { Int32($0) })
                    target.rollbackSpeculativeCache(
                        targetCache,
                        accepted: .perRow(acceptedArr),
                        blockSize: bs
                    )
                }
            }

            // --- Update state ---
            // Per-row hidden: gather verifyOut.lastHidden[bi, accepted[bi], :]
            // into a [B, 1, hidden] tensor via takeAlong on axis=1. The
            // old B-way slice + concat forced B intermediate small ops;
            // takeAlong runs in one kernel.
            let hiddenDim = verifyOut.lastHidden.dim(2)
            let acceptedIdx = MLX.broadcast(
                MLXArray(accepted.map { Int32($0) }, [B, 1, 1]),
                to: [B, 1, hiddenDim])
            hidden = MLX.takeAlong(
                verifyOut.lastHidden, acceptedIdx, axis: 1)  // [B, 1, hidden]

            // Update per-row bonus: last emitted token (or carry forward if
            // the row finished without emitting).
            for bi in 0 ..< B {
                if let last = newTokensPerRow[bi].last {
                    bonus[bi] = last
                }
            }

            // Shared-KV update to match the target cache's rewind.
            // Fast path (uniform accept): all rows reject the same count
            // → use `sliceTail` which simply crops the T axis, avoiding
            // the per-row mask + multiply that `zeroTailPerRow` performs.
            if uniformAccepted {
                let rejected = k - accepted[0]
                sharedKV = Gemma4SharedKV.sliceTail(
                    from: verifyOut.capturedSharedKV, rejected: rejected)
            } else {
                // Per-row: for row i, keep the first `prev + accepted[i] + 1`
                // positions relative to the captured K/V's T axis
                // (prev + bs). That equals capturedLen - rejected[i].
                let capturedT = verifyOut.capturedSharedKV.fullAttention.0.dim(2)
                let rejectedPerRow = accepted.map { Int32(k - $0) }
                let keepLengths = MLXArray(
                    rejectedPerRow.map { Int32(capturedT) - $0 }
                )
                sharedKV = Gemma4SharedKV.zeroTailPerRow(
                    from: verifyOut.capturedSharedKV, keepLengths: keepLengths)
            }

            if (emitted.max() ?? 0) % 256 == 0 {
                MLX.Memory.clearCache()
            }

            // TODO: Continuous batching — when finished rows accumulate,
            // call `BatchedCache.filterBatched(batchIndices:)` on every
            // cache in `targetCache` to shrink B, and compact the bonus/
            // hidden/sharedKV state accordingly. Deferred to a follow-up.
        }

        continuation.finish()
    }
}
