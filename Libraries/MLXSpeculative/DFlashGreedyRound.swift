// Copyright © 2026 Apple Inc.

import MLX
import MLXLLM
import MLXLMCommon

internal struct DFlashGreedyRoundResult {
    let accepted: Int
    let tokens: [Int]
    let bonus: Int
    let targetHidden: MLXArray
}

internal func runDFlashGreedyRound(
    target: any DFlashTargetModel,
    drafter: DFlashDraftModel,
    targetCache: inout [KVCache],
    draftCache: [KVCache],
    bonus: Int,
    targetHidden: MLXArray,
    promptTokenCount: Int,
    generatedTokenCount: Int,
    blockSize: Int,
    maxEmitCount: Int
) throws -> DFlashGreedyRoundResult {
    guard blockSize >= 2 else {
        throw DFlashError.invalidBlockSize(blockSize)
    }
    let rollbackProvider = target as? any DFlashTargetCacheRollbackProvider
    let targetRollbackState =
        rollbackProvider?.makeDFlashCacheRollbackState(cache: targetCache)
        ?? target.makeDefaultDFlashCacheRollbackState(cache: targetCache)

    let draftTokens = try drafter.draftBlock(
        bonus: bonus,
        targetHidden: targetHidden,
        cache: draftCache,
        blockSize: blockSize
    )

    eval(draftTokens)
    let committedDraftOffset = Swift.max(0, promptTokenCount + generatedTokenCount - 1)
    if let draftOffset = draftCache.first?.offset {
        let extraDraftContext = draftOffset - committedDraftOffset
        if extraDraftContext > 0 {
            let trimmed = trimPromptCache(draftCache, numTokens: extraDraftContext)
            if trimmed != extraDraftContext {
                throw DFlashError.untrimmableCache
            }
        }
    }
    let draftTokenIds = draftTokens.squeezed(axis: 0).asArray(Int32.self).map { Int($0) }

    let bonusColumn = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]
    let verifyInput = concatenated([bonusColumn, draftTokens], axis: 1)
    let verifyOut = try target.forwardForDFlash(
        verifyInput,
        cache: targetCache,
        targetLayerIds: drafter.config.targetLayerIds
    )
    let targetTokens = verifyOut.logits.asType(.float32).argMax(axis: -1)
    eval(targetTokens, verifyOut.targetHidden)

    let targetTokenIds = targetTokens.squeezed(axis: 0).asArray(Int32.self).map { Int($0) }
    let (walkedAccepted, walkedTokens) = SpeculativeWalk.single(
        draft: draftTokenIds,
        main: targetTokenIds
    )
    let emitted = Array(walkedTokens.prefix(maxEmitCount))
    let accepted = emitted.count < walkedTokens.count
        ? Swift.max(0, emitted.count - 1)
        : walkedAccepted

    let trim = blockSize - accepted - 1
    let nextTargetHidden: MLXArray
    if let rollbackProvider {
        nextTargetHidden = try rollbackProvider.rollbackDFlashCache(
            &targetCache,
            state: targetRollbackState,
            verifyInput: verifyInput,
            acceptedTokenCount: accepted,
            rejectedTokenCount: trim,
            targetLayerIds: drafter.config.targetLayerIds,
            verifiedTargetHidden: verifyOut.targetHidden
        )
    } else {
        nextTargetHidden = try target.rollbackDFlashCacheUsingDefault(
            &targetCache,
            state: targetRollbackState,
            verifyInput: verifyInput,
            acceptedTokenCount: accepted,
            rejectedTokenCount: trim,
            targetLayerIds: drafter.config.targetLayerIds,
            verifiedTargetHidden: verifyOut.targetHidden
        )
    }

    return DFlashGreedyRoundResult(
        accepted: accepted,
        tokens: emitted,
        bonus: emitted.last ?? bonus,
        targetHidden: nextTargetHidden
    )
}
