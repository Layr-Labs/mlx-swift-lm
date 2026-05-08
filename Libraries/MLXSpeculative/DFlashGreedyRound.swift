// Copyright © 2026 Apple Inc.

import Foundation
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
    maxEmitCount: Int,
    phaseAccumulator: DFlashPhaseAccumulator? = nil
) throws -> DFlashGreedyRoundResult {
    guard blockSize >= 2 else {
        throw DFlashError.invalidBlockSize(blockSize)
    }
    let roundStart = dflashTimingStart(phaseAccumulator)

    let snapshotStart = dflashTimingStart(phaseAccumulator)
    let rollbackProvider = target as? any DFlashTargetCacheRollbackProvider
    let targetRollbackState =
        rollbackProvider?.makeDFlashCacheRollbackState(cache: targetCache)
        ?? target.makeDefaultDFlashCacheRollbackState(cache: targetCache)
    dflashRecord(snapshotStart, into: phaseAccumulator) {
        $0.cacheSnapshotSeconds += $1
    }

    let draftStart = dflashTimingStart(phaseAccumulator)
    let draftTokens = try drafter.draftBlock(
        bonus: bonus,
        targetHidden: targetHidden,
        cache: draftCache,
        blockSize: blockSize
    )

    asyncEval(draftTokens)
    dflashRecord(draftStart, into: phaseAccumulator) {
        $0.draftLaunchSeconds += $1
    }

    let draftTrimStart = dflashTimingStart(phaseAccumulator)
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
    dflashRecord(draftTrimStart, into: phaseAccumulator) {
        $0.draftCacheTrimSeconds += $1
    }

    let verifyStart = dflashTimingStart(phaseAccumulator)
    let bonusColumn = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]
    let verifyInput = concatenated([bonusColumn, draftTokens], axis: 1)
    let verifyOut = try target.forwardGreedyTokensForDFlash(
        verifyInput,
        cache: targetCache,
        targetLayerIds: drafter.config.targetLayerIds
    )
    let targetTokens = verifyOut.tokens
    eval(targetTokens, verifyOut.targetHidden, draftTokens)
    dflashRecord(verifyStart, into: phaseAccumulator) {
        $0.verifyAndWaitSeconds += $1
    }

    let acceptStart = dflashTimingStart(phaseAccumulator)
    let draftTokenIds = draftTokens.squeezed(axis: 0).asArray(Int32.self).map { Int($0) }
    let targetTokenIds = targetTokens.squeezed(axis: 0).asArray(Int32.self).map { Int($0) }
    let (walkedAccepted, walkedTokens) = SpeculativeWalk.single(
        draft: draftTokenIds,
        main: targetTokenIds
    )
    let emitted = Array(walkedTokens.prefix(maxEmitCount))
    let accepted = emitted.count < walkedTokens.count
        ? Swift.max(0, emitted.count - 1)
        : walkedAccepted
    dflashRecord(acceptStart, into: phaseAccumulator) {
        $0.acceptWalkSeconds += $1
    }

    let trim = blockSize - accepted - 1
    let nextTargetHidden: MLXArray
    let rollbackStart = dflashTimingStart(phaseAccumulator)
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
    dflashRecord(rollbackStart, into: phaseAccumulator) {
        $0.cacheRollbackSeconds += $1
    }
    if let phaseAccumulator {
        phaseAccumulator.rounds += 1
    }
    dflashRecord(roundStart, into: phaseAccumulator) {
        $0.roundSeconds += $1
    }

    return DFlashGreedyRoundResult(
        accepted: accepted,
        tokens: emitted,
        bonus: emitted.last ?? bonus,
        targetHidden: nextTargetHidden
    )
}

@inline(__always)
private func dflashTimingStart(_ accumulator: DFlashPhaseAccumulator?) -> Date? {
    accumulator == nil ? nil : Date()
}

@inline(__always)
private func dflashRecord(
    _ start: Date?,
    into accumulator: DFlashPhaseAccumulator?,
    _ update: (DFlashPhaseAccumulator, Double) -> Void
) {
    guard let start, let accumulator else { return }
    update(accumulator, Date().timeIntervalSince(start))
}
