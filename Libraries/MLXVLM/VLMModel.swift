// Copyright © 2024 Apple Inc.

import MLX
import MLXLMCommon

public protocol VLMModel: LanguageModel, LoRAModel {
    /// Legacy VLM entry point retained while models migrate to the shared
    /// stateful/prefill-aware `LanguageModel.prepare` API.
    func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult
}

public extension VLMModel {
    func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        try prepare(input, cache: cache, windowSize: prefill.stepSize)
    }

    func prepare(
        _ input: LMInput, cache: [any KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        try prepare(input, cache: cache, state: nil, prefill: .init(stepSize: windowSize))
    }
}
