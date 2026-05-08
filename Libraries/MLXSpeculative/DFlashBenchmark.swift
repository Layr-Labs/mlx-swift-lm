// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// One DFlash benchmark measurement. Prefill is reported separately so
/// speculative speedup can be computed over the generation phase only.
public struct DFlashBenchmarkResult: Sendable {
    /// Number of tokens generated, excluding the prompt.
    public let generatedTokens: Int
    /// Seconds from prompt prefill start to prompt prefill end.
    public let prefillSeconds: Double
    /// Seconds from prefill end to the last emitted token.
    public let generationSeconds: Double
    /// Per-round count of accepted drafter tokens. `nil` for the no-drafter
    /// baseline; populated with one entry per DFlash round otherwise.
    public let acceptLengths: [Int]?
    /// Optional diagnostic phase timing for DFlash rounds. This is only
    /// populated when `measureDFlashThroughput(..., collectPhaseTimings: true)`
    /// or `collectVerifySubphaseTimings: true` is used.
    public let phaseTimings: DFlashBenchmarkPhaseTimings?
    /// Generated token ids, excluding the prompt. This is populated by the
    /// benchmark helpers so CLI diagnostics can compare exact outputs.
    public let generatedTokenIds: [Int]

    /// Generated tokens / generationSeconds.
    public var tokensPerSecond: Double {
        guard generationSeconds > 0 else { return 0 }
        return Double(generatedTokens) / generationSeconds
    }

    public init(
        generatedTokens: Int,
        prefillSeconds: Double,
        generationSeconds: Double,
        acceptLengths: [Int]? = nil,
        phaseTimings: DFlashBenchmarkPhaseTimings? = nil,
        generatedTokenIds: [Int] = []
    ) {
        self.generatedTokens = generatedTokens
        self.prefillSeconds = prefillSeconds
        self.generationSeconds = generationSeconds
        self.acceptLengths = acceptLengths
        self.phaseTimings = phaseTimings
        self.generatedTokenIds = generatedTokenIds
    }
}

/// Aggregate wall-clock phase timings for DFlash greedy rounds.
///
/// These timings are diagnostic. `draftLaunchSeconds` measures graph
/// construction/launch, not full drafter execution, because the current
/// DFlash loop intentionally overlaps drafter evaluation with target verify
/// work. `verifyAndWaitSeconds` includes the explicit wait on target outputs
/// and pending draft tokens.
public struct DFlashBenchmarkPhaseTimings: Sendable {
    public let rounds: Int
    public let cacheSnapshotSeconds: Double
    public let draftLaunchSeconds: Double
    public let draftCacheTrimSeconds: Double
    public let verifyAndWaitSeconds: Double
    /// Nested inside `verifyAndWaitSeconds` when target diagnostic timings
    /// are requested and the target supports `DFlashTargetDiagnosticForwardProvider`.
    public let targetTrunkSeconds: Double
    public let targetHiddenConcatSeconds: Double
    public let targetLMHeadSeconds: Double
    public let targetSoftcapArgmaxSeconds: Double
    public let targetTrunkEmbeddingSeconds: Double
    public let targetTrunkPLESeconds: Double
    public let targetTrunkMaskSeconds: Double
    public let targetTrunkAttentionSeconds: Double
    public let targetTrunkDenseMLPSeconds: Double
    public let targetTrunkRouterSeconds: Double
    public let targetTrunkExpertsSeconds: Double
    public let targetTrunkPLEGateSeconds: Double
    public let targetTrunkFinalNormSeconds: Double
    public let acceptWalkSeconds: Double
    public let cacheRollbackSeconds: Double
    public let roundSeconds: Double

    public var accountedSeconds: Double {
        cacheSnapshotSeconds
            + draftLaunchSeconds
            + draftCacheTrimSeconds
            + verifyAndWaitSeconds
            + acceptWalkSeconds
            + cacheRollbackSeconds
    }
}

internal final class DFlashPhaseAccumulator {
    let collectTargetSubphaseTimings: Bool
    var rounds = 0
    var cacheSnapshotSeconds = 0.0
    var draftLaunchSeconds = 0.0
    var draftCacheTrimSeconds = 0.0
    var verifyAndWaitSeconds = 0.0
    var targetTrunkSeconds = 0.0
    var targetHiddenConcatSeconds = 0.0
    var targetLMHeadSeconds = 0.0
    var targetSoftcapArgmaxSeconds = 0.0
    var targetTrunkEmbeddingSeconds = 0.0
    var targetTrunkPLESeconds = 0.0
    var targetTrunkMaskSeconds = 0.0
    var targetTrunkAttentionSeconds = 0.0
    var targetTrunkDenseMLPSeconds = 0.0
    var targetTrunkRouterSeconds = 0.0
    var targetTrunkExpertsSeconds = 0.0
    var targetTrunkPLEGateSeconds = 0.0
    var targetTrunkFinalNormSeconds = 0.0
    var acceptWalkSeconds = 0.0
    var cacheRollbackSeconds = 0.0
    var roundSeconds = 0.0

    init(collectTargetSubphaseTimings: Bool = false) {
        self.collectTargetSubphaseTimings = collectTargetSubphaseTimings
    }

    func snapshot() -> DFlashBenchmarkPhaseTimings {
        DFlashBenchmarkPhaseTimings(
            rounds: rounds,
            cacheSnapshotSeconds: cacheSnapshotSeconds,
            draftLaunchSeconds: draftLaunchSeconds,
            draftCacheTrimSeconds: draftCacheTrimSeconds,
            verifyAndWaitSeconds: verifyAndWaitSeconds,
            targetTrunkSeconds: targetTrunkSeconds,
            targetHiddenConcatSeconds: targetHiddenConcatSeconds,
            targetLMHeadSeconds: targetLMHeadSeconds,
            targetSoftcapArgmaxSeconds: targetSoftcapArgmaxSeconds,
            targetTrunkEmbeddingSeconds: targetTrunkEmbeddingSeconds,
            targetTrunkPLESeconds: targetTrunkPLESeconds,
            targetTrunkMaskSeconds: targetTrunkMaskSeconds,
            targetTrunkAttentionSeconds: targetTrunkAttentionSeconds,
            targetTrunkDenseMLPSeconds: targetTrunkDenseMLPSeconds,
            targetTrunkRouterSeconds: targetTrunkRouterSeconds,
            targetTrunkExpertsSeconds: targetTrunkExpertsSeconds,
            targetTrunkPLEGateSeconds: targetTrunkPLEGateSeconds,
            targetTrunkFinalNormSeconds: targetTrunkFinalNormSeconds,
            acceptWalkSeconds: acceptWalkSeconds,
            cacheRollbackSeconds: cacheRollbackSeconds,
            roundSeconds: roundSeconds
        )
    }
}

/// Run target-only greedy generation over `promptTokens` for `maxTokens`
/// steps. This is the denominator for DFlash speedup measurements.
public func measureDFlashBaselineThroughput(
    target: any DFlashTargetModel,
    promptTokens: MLXArray,
    maxTokens: Int,
    parameters: GenerateParameters = GenerateParameters(temperature: 0)
) -> DFlashBenchmarkResult {
    guard maxTokens > 0 else {
        return DFlashBenchmarkResult(
            generatedTokens: 0,
            prefillSeconds: 0,
            generationSeconds: 0
        )
    }

    var generationParameters = parameters
    generationParameters.maxTokens = maxTokens
    generationParameters.temperature = 0

    var prompt = promptTokens
    if prompt.ndim == 1 {
        prompt = prompt[.newAxis, .ellipsis]
    }

    let cache = target.newCache(parameters: generationParameters)

    let prefillStart = Date()
    var logits = target.callAsFunction(prompt, cache: cache)
    var token = logits[0..., -1, 0...].argMax(axis: -1)
    eval(token)
    var generatedIds = [Int(token.item(Int32.self))]
    let prefillElapsed = Date().timeIntervalSince(prefillStart)

    let generationStart = Date()
    var generated = 1
    for _ in 1 ..< maxTokens {
        logits = target.callAsFunction(token[.newAxis, .ellipsis], cache: cache)
        token = logits[0..., -1, 0...].argMax(axis: -1)
        eval(token)
        generatedIds.append(Int(token.item(Int32.self)))
        generated += 1
    }
    let generationElapsed = Date().timeIntervalSince(generationStart)

    return DFlashBenchmarkResult(
        generatedTokens: generated,
        prefillSeconds: prefillElapsed,
        generationSeconds: generationElapsed,
        generatedTokenIds: generatedIds
    )
}

/// Run DFlash greedy generation over `promptTokens` for `maxTokens` steps.
///
/// The drafter must be compatible with the target. Binding is performed by
/// `DFlashTokenIterator` and is idempotent for an already-bound matching target.
public func measureDFlashThroughput(
    target: any DFlashTargetModel,
    drafter: DFlashDraftModel,
    promptTokens: MLXArray,
    maxTokens: Int,
    blockSize: Int? = nil,
    parameters: GenerateParameters = GenerateParameters(temperature: 0),
    collectPhaseTimings: Bool = false,
    collectVerifySubphaseTimings: Bool = false
) throws -> DFlashBenchmarkResult {
    guard maxTokens > 0 else {
        return DFlashBenchmarkResult(
            generatedTokens: 0,
            prefillSeconds: 0,
            generationSeconds: 0
        )
    }

    var generationParameters = parameters
    generationParameters.maxTokens = maxTokens
    generationParameters.temperature = 0

    let resolvedBlockSize = blockSize ?? drafter.config.recommendedBlockSize
    guard resolvedBlockSize >= 2 else {
        throw DFlashError.invalidBlockSize(resolvedBlockSize)
    }

    try drafter.bind(target: target)

    var prompt = promptTokens
    if prompt.ndim == 1 {
        prompt = prompt[.newAxis, .ellipsis]
    }

    var targetCache = target.newCache(parameters: generationParameters)
    let draftCache = try drafter.makeCache()
    guard canTrimPromptCache(draftCache) else {
        throw DFlashError.untrimmableCache
    }

    let prefillStart = Date()
    let prefillOut = try target.forwardGreedyTokensForDFlash(
        prompt,
        cache: targetCache,
        targetLayerIds: drafter.config.targetLayerIds
    )
    let firstBonusArray = prefillOut.tokens[0..., -1]
    eval(firstBonusArray, prefillOut.targetHidden)
    let prefillElapsed = Date().timeIntervalSince(prefillStart)

    var bonus = Int(firstBonusArray.item(Int32.self))
    var targetHidden = prefillOut.targetHidden
    var generatedIds = [bonus]

    let generationStart = Date()
    var generated = 1
    var accepts: [Int] = []
    let phases = collectPhaseTimings || collectVerifySubphaseTimings
        ? DFlashPhaseAccumulator(collectTargetSubphaseTimings: collectVerifySubphaseTimings)
        : nil

    while generated < maxTokens {
        let remaining = maxTokens - generated
        let roundBlockSize = Swift.min(resolvedBlockSize, remaining + 1)
        if roundBlockSize < 2 { break }

        let round = try runDFlashGreedyRound(
            target: target,
            drafter: drafter,
            targetCache: &targetCache,
            draftCache: draftCache,
            bonus: bonus,
            targetHidden: targetHidden,
            promptTokenCount: prompt.dim(1),
            generatedTokenCount: generated,
            blockSize: roundBlockSize,
            maxEmitCount: remaining,
            phaseAccumulator: phases
        )
        accepts.append(round.accepted)

        let emitted = round.tokens.count
        generated += emitted
        generatedIds.append(contentsOf: round.tokens)
        if emitted == 0 { break }

        bonus = round.bonus
        targetHidden = round.targetHidden
    }
    let generationElapsed = Date().timeIntervalSince(generationStart)

    return DFlashBenchmarkResult(
        generatedTokens: generated,
        prefillSeconds: prefillElapsed,
        generationSeconds: generationElapsed,
        acceptLengths: accepts,
        phaseTimings: phases?.snapshot(),
        generatedTokenIds: generatedIds
    )
}
