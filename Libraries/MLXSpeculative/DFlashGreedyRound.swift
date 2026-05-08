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
    let verifyOut: DFlashGreedyTargetForward
    if phaseAccumulator?.collectTargetSubphaseTimings == true,
        let diagnosticTarget = target as? any DFlashTargetDiagnosticForwardProvider
    {
        verifyOut = try diagnosticTarget.forwardGreedyTokensForDFlash(
            verifyInput,
            cache: targetCache,
            targetLayerIds: drafter.config.targetLayerIds,
            collectVerifyTimings: true
        )
    } else {
        verifyOut = try target.forwardGreedyTokensForDFlash(
            verifyInput,
            cache: targetCache,
            targetLayerIds: drafter.config.targetLayerIds
        )
    }
    if let timings = verifyOut.verifyTimings, let phaseAccumulator {
        phaseAccumulator.targetTrunkSeconds += timings.trunkSeconds
        phaseAccumulator.targetHiddenConcatSeconds += timings.hiddenConcatSeconds
        phaseAccumulator.targetLMHeadSeconds += timings.lmHeadSeconds
        phaseAccumulator.targetSoftcapArgmaxSeconds += timings.softcapArgmaxSeconds
    }
    let targetTokens = verifyOut.tokens
    let draftTokenIds = draftTokens.squeezed(axis: 0)
    let targetTokenIds = targetTokens.squeezed(axis: 0)
    let proposedCount = Swift.max(0, blockSize - 1)
    let acceptedArray: MLXArray?
    if proposedCount == 0 {
        acceptedArray = nil
    } else {
        let targetPrefix = targetTokenIds[0 ..< proposedCount]
        let matches = (draftTokenIds .== targetPrefix).asType(.int32)
        let prefixMatches = matches.cumprod(axis: 0)
        acceptedArray = prefixMatches.sum()
    }
    if let acceptedArray {
        eval(targetTokens, verifyOut.targetHidden, draftTokens, acceptedArray)
    } else {
        eval(targetTokens, verifyOut.targetHidden, draftTokens)
    }
    dflashRecord(verifyStart, into: phaseAccumulator) {
        $0.verifyAndWaitSeconds += $1
    }

    let acceptStart = dflashTimingStart(phaseAccumulator)
    let walkedAccepted = acceptedArray.map { Int($0.item(Int32.self)) } ?? 0
    let walkedTokenCount = walkedAccepted + 1
    let emittedCount = Swift.min(maxEmitCount, walkedTokenCount)
    let emitted = targetTokenIds[0 ..< emittedCount]
        .asArray(Int32.self)
        .map { Int($0) }
    let accepted = emittedCount < walkedTokenCount
        ? Swift.max(0, emittedCount - 1)
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
