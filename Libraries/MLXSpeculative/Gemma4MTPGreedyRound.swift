// Copyright © 2026 Apple Inc.

import MLX
import MLXLLM
import MLXLMCommon

internal struct Gemma4MTPGreedyRoundResult {
    let accepted: Int
    let tokens: [Int]
    let bonus: Int
    let hidden: MLXArray
    let sharedKV: Gemma4SharedKV
}

/// Run one single-batch greedy MTP round.
///
/// The caller owns remaining-token truncation. `blockSize` is the actual
/// per-round size (`bonus + drafts`), not necessarily the user's maximum
/// configured block size.
internal func runGemma4MTPGreedyRound(
    target: Gemma4TextModel,
    drafter: Gemma4AssistantDraftModel,
    cache: [KVCache],
    bonus: Int,
    hidden: MLXArray,
    sharedKV: Gemma4SharedKV,
    blockSize: Int
) -> Gemma4MTPGreedyRoundResult {
    let k = blockSize - 1
    let driveOffset = cache[0].offset
    let positionOffset = Gemma4.PositionOffset.scalar(driveOffset)
    let masks = drafter.makeMasks(
        queryLen: 1,
        sharedKV: sharedKV,
        positionOffset: positionOffset,
        dtype: hidden.dtype)

    var tok = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]
    var h = hidden
    var draftPerStep: [MLXArray] = []
    draftPerStep.reserveCapacity(k)

    for _ in 0 ..< k {
        let tokEmbed = target.embedTokensForDrafter(tok)
        let inputsEmbeds = concatenated([tokEmbed, h], axis: -1)
        let (newH, logits) = drafter(
            inputsEmbeds: inputsEmbeds,
            sharedKV: sharedKV,
            positionOffset: positionOffset,
            masks: masks)
        let sampled = logits.squeezed(axis: 1).argMax(axis: -1)
        let sampled2d = sampled[.newAxis, .ellipsis]
        draftPerStep.append(sampled2d)
        tok = sampled2d
        h = newH
    }

    let bonusCol = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]
    let verifyInput: MLXArray =
        draftPerStep.isEmpty
        ? bonusCol
        : concatenated([bonusCol] + draftPerStep, axis: 1)
    let verifyOut = target.forwardForMTP(verifyInput, cache: cache)
    let mainTokens = verifyOut.logits.asType(.float32).argMax(axis: -1)
    let draftConcat: MLXArray =
        draftPerStep.isEmpty
        ? MLXArray.zeros([1, 0], dtype: .int32)
        : concatenated(draftPerStep, axis: 1)

    eval(mainTokens, draftConcat)
    let mainInts = mainTokens.squeezed(axis: 0).asArray(Int.self)
    let draftTokens = draftConcat.squeezed(axis: 0).asArray(Int32.self)
                                 .map { Int($0) }
    let (accepted, newTokens) = SpeculativeWalk.single(
        draft: draftTokens, main: mainInts)

    if accepted < k {
        target.rollbackSpeculativeCache(
            cache, accepted: .scalar(accepted), blockSize: blockSize)
    }

    let rejected = k - accepted
    return Gemma4MTPGreedyRoundResult(
        accepted: accepted,
        tokens: newTokens,
        bonus: newTokens.last!,
        hidden: verifyOut.lastHidden[0..., accepted ..< accepted + 1, 0...],
        sharedKV: Gemma4SharedKV.sliceTail(
            from: verifyOut.capturedSharedKV, rejected: rejected)
    )
}
