// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

private let dFlashCPUAcceptWalk: Bool = {
    switch ProcessInfo.processInfo.environment["MLX_DFLASH_CPU_ACCEPT_WALK"]?.lowercased() {
    case "0", "false", "no", "off":
        return false
    default:
        return true
    }
}()

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
    let verifyOut: DFlashGreedyTargetForward =
        try DFlashTargetRuntimeOptions.withSmallRowVerifyFusionsEnabled {
            if phaseAccumulator?.collectTargetSubphaseTimings == true,
                let diagnosticTarget = target as? any DFlashTargetDiagnosticForwardProvider
            {
                return try diagnosticTarget.forwardGreedyTokensForDFlash(
                    verifyInput,
                    cache: targetCache,
                    targetLayerIds: drafter.config.targetLayerIds,
                    collectVerifyTimings: true
                )
            }
            return try target.forwardGreedyTokensForDFlash(
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
        phaseAccumulator.targetTrunkEmbeddingSeconds += timings.trunkEmbeddingSeconds
        phaseAccumulator.targetTrunkPLESeconds += timings.trunkPLESeconds
        phaseAccumulator.targetTrunkMaskSeconds += timings.trunkMaskSeconds
        phaseAccumulator.targetTrunkAttentionSeconds += timings.trunkAttentionSeconds
        phaseAccumulator.targetTrunkDenseMLPSeconds += timings.trunkDenseMLPSeconds
        phaseAccumulator.targetTrunkRouterSeconds += timings.trunkRouterSeconds
        phaseAccumulator.targetTrunkExpertsSeconds += timings.trunkExpertsSeconds
        phaseAccumulator.targetTrunkPLEGateSeconds += timings.trunkPLEGateSeconds
        phaseAccumulator.targetTrunkFinalNormSeconds += timings.trunkFinalNormSeconds
    }
    let targetTokens = verifyOut.tokens
    let verifiedTokenCount = targetTokens.dim(1)
    let draftTokenIds = draftTokens.squeezed(axis: 0)
    let targetTokenIds = targetTokens.squeezed(axis: 0)
    let proposedCount = Swift.max(0, blockSize - 1)
    let comparableCount = Swift.min(proposedCount, verifiedTokenCount)
    let acceptStart: Date?
    let walkedAccepted: Int
    let emitted: [Int]
    let accepted: Int
    if dFlashCPUAcceptWalk {
        let targetReadCount = Swift.min(
            verifiedTokenCount,
            Swift.max(comparableCount, Swift.min(maxEmitCount, comparableCount + 1)))
        let targetIds = targetReadCount > 0
            ? targetTokenIds[0 ..< targetReadCount].asArray(Int32.self)
            : []
        let draftIds = comparableCount > 0
            ? draftTokenIds[0 ..< comparableCount].asArray(Int32.self)
            : []
        dflashRecord(verifyStart, into: phaseAccumulator) {
            $0.verifyAndWaitSeconds += $1
        }

        acceptStart = dflashTimingStart(phaseAccumulator)
        var prefix = 0
        while prefix < comparableCount, draftIds[prefix] == targetIds[prefix] {
            prefix += 1
        }
        walkedAccepted = prefix
        let walkedTokenCount = walkedAccepted + 1
        let emittedCount = Swift.min(maxEmitCount, walkedTokenCount, verifiedTokenCount)
        emitted = targetIds.prefix(emittedCount).map { Int($0) }
        accepted = emittedCount < walkedTokenCount
            ? Swift.max(0, emittedCount - 1)
            : walkedAccepted
    } else {
        let acceptedArray: MLXArray?
        if comparableCount == 0 {
            acceptedArray = nil
        } else {
            let targetPrefix = targetTokenIds[0 ..< comparableCount]
            let draftPrefix = draftTokenIds[0 ..< comparableCount]
            let matches = (draftPrefix .== targetPrefix).asType(.int32)
            let prefixMatches = matches.cumprod(axis: 0)
            acceptedArray = prefixMatches.sum()
        }
        if let acceptedArray {
            eval(targetTokens, draftTokens, acceptedArray)
        } else {
            eval(targetTokens, draftTokens)
        }
        dflashRecord(verifyStart, into: phaseAccumulator) {
            $0.verifyAndWaitSeconds += $1
        }

        acceptStart = dflashTimingStart(phaseAccumulator)
        walkedAccepted = acceptedArray.map { Int($0.item(Int32.self)) } ?? 0
        let walkedTokenCount = walkedAccepted + 1
        let emittedCount = Swift.min(maxEmitCount, walkedTokenCount, verifiedTokenCount)
        emitted = targetTokenIds[0 ..< emittedCount]
            .asArray(Int32.self)
            .map { Int($0) }
        accepted = emittedCount < walkedTokenCount
            ? Swift.max(0, emittedCount - 1)
            : walkedAccepted
    }
    dflashRecord(acceptStart, into: phaseAccumulator) {
        $0.acceptWalkSeconds += $1
    }

    let trim = Swift.max(0, verifiedTokenCount - accepted - 1)
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
