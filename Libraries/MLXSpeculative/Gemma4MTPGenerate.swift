// Copyright © 2026 Apple Inc.
//
// Public API entry points for Gemma 4 Multi-Token Prediction (MTP)
// speculative decoding.
//
//   - generateGemma4MTP: single-request (B=1) path, returns
//     AsyncStream<Generation> of decoded text chunks + terminal info.
//     Internally drives `Gemma4MTPTokenIterator` so greedy and stochastic
//     (temperature > 0) paths share a single implementation.
//   - generateGemma4MTPBatched (future): multi-request (B>1) entry point
//     wrapping `runGemma4MTPRoundsBatched`. Deferred until the batched
//     path has a clear throughput win.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Generate text from a Gemma 4 target using a Gemma 4 MTP drafter.
///
/// Internally wraps ``Gemma4MTPTokenIterator`` and a tokenizer-decoding
/// loop, producing an `AsyncStream<Generation>` that yields decoded text
/// chunks and a terminal `.info(...)` with counts.
///
/// - Parameters:
///   - input: prepared language-model input. `input.text.tokens` is used
///     for prefill; images/videos are ignored (MTP is text-only).
///   - parameters: generation parameters. `maxTokens` caps output length
///     (defaults to 1024 if nil). `temperature > 0` enables stochastic
///     rejection-based speculative sampling; `temperature == 0` is
///     greedy and produces byte-identical output to no-drafter baseline.
///   - target: `ModelContext` whose `model` is a `Gemma4TextModel` (or a
///     `Gemma4Model` VLM wrapper whose text portion is `Gemma4TextModel`).
///     Throws `unsupportedTarget` otherwise.
///   - drafter: the loaded Gemma 4 MTP drafter.
///   - blockSize: speculative block size (2–16). Pass `nil` to use the
///     model-aware default from `Gemma4MTPAutomaticPolicy`.
///   - rngSeed: seed for stochastic sampling (only consulted when
///     `parameters.temperature > 0`). Default 0 → seeds from the system
///     clock.
/// - Returns: an `AsyncStream<Generation>` yielding `.chunk(String)`
///   (decoded token text) and one terminal `.info(...)`.
/// - Throws: `Gemma4MTPError.unsupportedTarget`, `.invalidBlockSize`,
///   or any error thrown by `drafter.bind(target:)`.
public func generateGemma4MTP(
    input: LMInput,
    parameters: GenerateParameters,
    target: ModelContext,
    drafter: Gemma4AssistantDraftModel,
    blockSize: Int? = nil,
    rngSeed: UInt64 = 0
) throws -> AsyncStream<Generation> {
    let gemma4: Gemma4TextModel
    if let t = target.model as? Gemma4TextModel {
        gemma4 = t
    } else if let wrapper = target.model as? Gemma4Model {
        gemma4 = wrapper.textModel
    } else {
        throw Gemma4MTPError.unsupportedTarget(
            String(describing: type(of: target.model)))
    }

    let tokenizer = target.tokenizer
    let eosIds = target.configuration.eosTokenIds
    var params = parameters
    if params.maxTokens == nil {
        params.maxTokens = 1024
    }
    let promptTokenCount = input.text.tokens.size

    // Build the iterator outside the stream closure so init errors can
    // throw synchronously. Prefill runs inside init.
    var iter = try Gemma4MTPTokenIterator(
        input: input,
        target: gemma4,
        drafter: drafter,
        parameters: params,
        blockSize: blockSize,
        rngSeed: rngSeed
    )
    let prefillElapsed = iter.promptPrefillTime

    // Wrap the iterator in a Sendable box so we can move it into the
    // stream closure without tripping strict concurrency (the iterator
    // holds MLXArray state which is `@unchecked Sendable`-compatible but
    // isn't annotated yet).
    let boxed = IteratorBox(iter: iter)

    return AsyncStream<Generation> { continuation in
        let task = Task {
            let generateStart = Date()
            var tokenCount = 0
            var stopReason: GenerateStopReason = .length
            while let tok = boxed.next() {
                tokenCount += 1
                continuation.yield(.chunk(tokenizer.decode(tokenIds: [tok])))
                if eosIds.contains(tok) {
                    stopReason = .stop
                    break
                }
                if Task.isCancelled {
                    stopReason = .cancelled
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

/// Box wrapping a `Gemma4MTPTokenIterator` so it can be captured by a
/// `Sendable` closure. The iterator itself holds `MLXArray` state which
/// is conceptually reference-counted GPU memory — safe to move across
/// tasks, but the Swift compiler needs an explicit `@unchecked`.
private final class IteratorBox: @unchecked Sendable {
    private var iter: Gemma4MTPTokenIterator
    init(iter: Gemma4MTPTokenIterator) {
        self.iter = iter
    }
    func next() -> Int? {
        iter.next()
    }
}
