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

    /// Generated tokens / generationSeconds.
    public var tokensPerSecond: Double {
        guard generationSeconds > 0 else { return 0 }
        return Double(generatedTokens) / generationSeconds
    }

    public init(
        generatedTokens: Int,
        prefillSeconds: Double,
        generationSeconds: Double,
        acceptLengths: [Int]? = nil
    ) {
        self.generatedTokens = generatedTokens
        self.prefillSeconds = prefillSeconds
        self.generationSeconds = generationSeconds
        self.acceptLengths = acceptLengths
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
    let prefillElapsed = Date().timeIntervalSince(prefillStart)

    let generationStart = Date()
    var generated = 1
    for _ in 1 ..< maxTokens {
        logits = target.callAsFunction(token[.newAxis, .ellipsis], cache: cache)
        token = logits[0..., -1, 0...].argMax(axis: -1)
        eval(token)
        generated += 1
    }
    let generationElapsed = Date().timeIntervalSince(generationStart)

    return DFlashBenchmarkResult(
        generatedTokens: generated,
        prefillSeconds: prefillElapsed,
        generationSeconds: generationElapsed
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
    parameters: GenerateParameters = GenerateParameters(temperature: 0)
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

    let resolvedBlockSize = blockSize ?? drafter.config.blockSize
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
    let prefillOut = try target.forwardForDFlash(
        prompt,
        cache: targetCache,
        targetLayerIds: drafter.config.targetLayerIds
    )
    let firstBonusArray = prefillOut.logits[0..., -1, 0...].argMax(axis: -1)
    eval(firstBonusArray, prefillOut.targetHidden)
    let prefillElapsed = Date().timeIntervalSince(prefillStart)

    var bonus = Int(firstBonusArray.item(Int32.self))
    var targetHidden = prefillOut.targetHidden

    let generationStart = Date()
    var generated = 1
    var accepts: [Int] = []

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
            maxEmitCount: remaining
        )
        accepts.append(round.accepted)

        let emitted = round.tokens.count
        generated += emitted
        if emitted == 0 { break }

        bonus = round.bonus
        targetHidden = round.targetHidden
    }
    let generationElapsed = Date().timeIntervalSince(generationStart)

    return DFlashBenchmarkResult(
        generatedTokens: generated,
        prefillSeconds: prefillElapsed,
        generationSeconds: generationElapsed,
        acceptLengths: accepts
    )
}
