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

    /// Generated tokens / generationSeconds.
    public var tokensPerSecond: Double {
        guard generationSeconds > 0 else { return 0 }
        return Double(generatedTokens) / generationSeconds
    }

    public init(
        generatedTokens: Int,
        prefillSeconds: Double,
        generationSeconds: Double
    ) {
        self.generatedTokens = generatedTokens
        self.prefillSeconds = prefillSeconds
        self.generationSeconds = generationSeconds
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
    var token = logits[0..., -1, 0...].asType(.float32).argMax(axis: -1)
    eval(token)
    let prefillElapsed = Date().timeIntervalSince(prefillStart)

    let generationStart = Date()
    var generated = 1
    for _ in 1 ..< maxTokens {
        logits = target.callAsFunction(token[.newAxis, .ellipsis], cache: cache)
        token = logits[0..., -1, 0...].asType(.float32).argMax(axis: -1)
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

    var iterator = try DFlashTokenIterator(
        input: LMInput(text: .init(tokens: promptTokens)),
        target: target,
        drafter: drafter,
        parameters: generationParameters,
        blockSize: blockSize
    )

    let generationStart = Date()
    var generated = 0
    while iterator.next() != nil {
        generated += 1
    }
    let generationElapsed = Date().timeIntervalSince(generationStart)

    return DFlashBenchmarkResult(
        generatedTokens: generated,
        prefillSeconds: iterator.promptPrefillTime,
        generationSeconds: generationElapsed
    )
}
