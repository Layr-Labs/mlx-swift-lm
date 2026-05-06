// Copyright © 2026 Apple Inc.
//
// Public API entry points for Gemma 4 Multi-Token Prediction (MTP)
// speculative decoding.
//
// Two variants:
//   - generateGemma4MTP: single-request (B=1) path, returns AsyncStream<Generation>.
//   - generateGemma4MTPBatched: multi-request (B>1) path, returns
//     AsyncStream<BatchedGeneration>. (Separate file in later work if
//     it grows large.)
//
// Both wrap the Task 20/21 round-loops with:
//   - target extraction from ModelContext + cast to Gemma4TextModel
//     (throws Gemma4MTPError.unsupportedTarget on mismatch).
//   - prefill via Gemma4TextModel.forwardForMTP to get firstBonus /
//     firstHidden / firstSharedKV.
//   - tokenizer-backed .chunk(String) emission (replaces the raw
//     "<int>" encoding used by the round loop tests).
//   - .info(GenerateCompletionInfo) at stream end (basic counts).
//   - EOS detection from ModelConfiguration.eosTokenIds.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Generate text from a Gemma 4 target using a Gemma 4 MTP drafter.
///
/// - Parameters:
///   - input: prepared language-model input. `input.text.tokens` is used
///     for prefill; images/videos are ignored (MTP is text-only).
///   - parameters: generation parameters. `maxTokens` caps output length
///     (defaults to 1024 if nil).
///   - target: `ModelContext` whose `model` is a `Gemma4TextModel`.
///     Throws `unsupportedTarget` otherwise.
///   - drafter: the loaded Gemma 4 MTP drafter.
///   - blockSize: speculative block size (2–16). Default 4.
/// - Returns: an `AsyncStream<Generation>` yielding `.chunk(String)`
///   for each emitted token (decoded via the target's tokenizer) and
///   one terminal `.info(...)` with counts.
/// - Throws: `Gemma4MTPError.unsupportedTarget`, `.invalidBlockSize`,
///   or any error thrown by `drafter.bind(target:)`.
public func generateGemma4MTP(
    input: LMInput,
    parameters: GenerateParameters,
    target: ModelContext,
    drafter: Gemma4AssistantDraftModel,
    blockSize: Int = 4
) throws -> AsyncStream<Generation> {
    guard let gemma4 = target.model as? Gemma4TextModel else {
        throw Gemma4MTPError.unsupportedTarget(String(describing: type(of: target.model)))
    }
    guard blockSize >= 2 && blockSize <= 16 else {
        throw Gemma4MTPError.invalidBlockSize(blockSize)
    }

    let tokenizer = target.tokenizer
    let eosIds = target.configuration.eosTokenIds
    let maxTokens = parameters.maxTokens ?? 1024

    // Ensure the input is shaped [B, L] — the Gemma4 forward expects that.
    var prefillTokens = input.text.tokens
    if prefillTokens.ndim == 1 {
        prefillTokens = prefillTokens[.newAxis, .ellipsis]
    }

    // Prefill: run the target once over the prompt to get the first
    // bonus + last hidden + shared-KV.
    let prefillStart = Date()
    let prefillCache = gemma4.newCache(parameters: parameters)
    let prefillOut = gemma4.forwardForMTP(prefillTokens, cache: prefillCache)

    // Greedy bonus from the last prefill position.
    let lastLogits = prefillOut.logits[0..., -1, 0...]
    let firstBonus = Int(lastLogits.argMax(axis: -1).item(Int32.self))
    let firstHidden = prefillOut.lastHidden[
        0..., -1 ..< prefillOut.lastHidden.dim(1), 0...]
    let firstSharedKV = prefillOut.capturedSharedKV
    let prefillElapsed = Date().timeIntervalSince(prefillStart)
    let promptTokenCount = prefillTokens.dim(1)

    let generateStart = Date()
    let intStream = try runGemma4MTPRounds(
        target: gemma4,
        drafter: drafter,
        targetCache: prefillCache,
        firstBonus: firstBonus,
        firstHidden: firstHidden,
        firstSharedKV: firstSharedKV,
        maxTokens: maxTokens,
        blockSize: blockSize
    )

    return AsyncStream<Generation> { continuation in
        let task = Task {
            var tokenCount = 0
            var stopReason: GenerateStopReason = .length
            for await gen in intStream {
                guard case .chunk(let s) = gen, let tok = Int(s) else { continue }
                tokenCount += 1
                continuation.yield(.chunk(tokenizer.decode(tokenIds: [tok])))
                if eosIds.contains(tok) {
                    stopReason = .stop
                    break
                }
            }
            let elapsed = Date().timeIntervalSince(generateStart)
            let info = GenerateCompletionInfo(
                promptTokenCount: promptTokenCount,
                generationTokenCount: tokenCount,
                promptTime: prefillElapsed,
                generationTime: elapsed,
                stopReason: stopReason
            )
            continuation.yield(.info(info))
            continuation.finish()
        }
        continuation.onTermination = { _ in
            task.cancel()
        }
    }
}
