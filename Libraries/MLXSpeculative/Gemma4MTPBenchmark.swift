// Copyright © 2026 Apple Inc.
//
// Benchmark primitives for Gemma 4 Multi-Token Prediction (MTP)
// speculative decoding. Provides a pair of measurement helpers:
//
//   - measureBaselineThroughput: no-drafter target-only greedy generation
//     tokens/sec, from prefill-end through generation-end.
//   - measureMTPThroughput: drafter-driven MTP generation tokens/sec over
//     the same prompt, plus per-round accept histogram.
//
// These are intentionally minimal — no model loading, no CLI, no
// harness. Callers supply a loaded target `ModelContext` and a bound
// `Gemma4AssistantDraftModel` plus a prompt, and receive back a
// `BenchmarkResult`. A full driver (model download, result table,
// speedup calculation) lives outside this library in a tools target.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// One benchmark measurement. Wall-clock times are post-prefill; prefill
/// is measured separately so MTP speedup is computed against the generation
/// phase only (MTP doesn't change prefill).
public struct Gemma4MTPBenchmarkResult: Sendable {
    /// Number of tokens generated (excludes the prompt).
    public let generatedTokens: Int
    /// Seconds from prefill start to prefill end.
    public let prefillSeconds: Double
    /// Seconds from prefill end to last emitted token.
    public let generationSeconds: Double
    /// Generated tokens / generationSeconds.
    public var tokensPerSecond: Double {
        guard generationSeconds > 0 else { return 0 }
        return Double(generatedTokens) / generationSeconds
    }
    /// Per-round count of accepted drafter tokens. `nil` for the no-drafter
    /// baseline; populated with one entry per MTP round otherwise.
    public let acceptLengths: [Int]?

    public init(
        generatedTokens: Int,
        prefillSeconds: Double,
        generationSeconds: Double,
        acceptLengths: [Int]?
    ) {
        self.generatedTokens = generatedTokens
        self.prefillSeconds = prefillSeconds
        self.generationSeconds = generationSeconds
        self.acceptLengths = acceptLengths
    }
}

/// Run target-only greedy generation over `promptTokens` for `maxTokens`
/// steps. Returns timing + token count. This is the denominator for the
/// MTP speedup calculation — matches the semantics of the parity-test
/// `runBaselineGreedy` helper.
public func measureBaselineThroughput(
    target: Gemma4TextModel,
    promptTokens: MLXArray,
    maxTokens: Int
) -> Gemma4MTPBenchmarkResult {
    var prompt = promptTokens
    if prompt.ndim == 1 {
        prompt = prompt[.newAxis, .ellipsis]
    }
    let cache = target.newCache(parameters: nil)

    let prefillStart = Date()
    var logits = target(prompt, cache: cache)
    var tok = logits[0..., -1, 0...].argMax(axis: -1)
    eval(tok)
    let prefillElapsed = Date().timeIntervalSince(prefillStart)

    let genStart = Date()
    var generated = 1
    for _ in 1 ..< maxTokens {
        let input = tok[.newAxis, .ellipsis]
        logits = target(input, cache: cache)
        tok = logits[0..., -1, 0...].argMax(axis: -1)
        eval(tok)
        generated += 1
    }
    let genElapsed = Date().timeIntervalSince(genStart)

    return Gemma4MTPBenchmarkResult(
        generatedTokens: generated,
        prefillSeconds: prefillElapsed,
        generationSeconds: genElapsed,
        acceptLengths: nil
    )
}

/// Run MTP greedy generation over `promptTokens` for `maxTokens` steps
/// via `runGemma4MTPRounds`. Returns timing + token count + per-round
/// accept histogram.
///
/// The drafter must already be bound to the target (or will be bound on
/// first call — `bind(target:)` is idempotent when the binding matches).
public func measureMTPThroughput(
    target: Gemma4TextModel,
    drafter: Gemma4AssistantDraftModel,
    promptTokens: MLXArray,
    maxTokens: Int,
    blockSize: Int
) async throws -> Gemma4MTPBenchmarkResult {
    var prompt = promptTokens
    if prompt.ndim == 1 {
        prompt = prompt[.newAxis, .ellipsis]
    }
    let cache = target.newCache(parameters: nil)

    let prefillStart = Date()
    let prefillOut = target.forwardForMTP(prompt, cache: cache)
    let firstBonus = Int(prefillOut.logits[0..., -1, 0...]
                             .argMax(axis: -1).item(Int32.self))
    let firstHidden = prefillOut.lastHidden[
        0..., -1 ..< prefillOut.lastHidden.dim(1), 0...]
    let firstSharedKV = prefillOut.capturedSharedKV
    let prefillElapsed = Date().timeIntervalSince(prefillStart)

    let genStart = Date()
    let stream = try runGemma4MTPRounds(
        target: target,
        drafter: drafter,
        targetCache: cache,
        firstBonus: firstBonus,
        firstHidden: firstHidden,
        firstSharedKV: firstSharedKV,
        maxTokens: maxTokens,
        blockSize: blockSize
    )
    var generated = 0
    for await gen in stream {
        if case .chunk = gen {
            generated += 1
        }
    }
    let genElapsed = Date().timeIntervalSince(genStart)

    // NOTE: per-round accept histogram is not currently reconstructible
    // from the token stream alone (rounds aren't delimited in the
    // AsyncStream<Generation> surface). Callers that need this should
    // drive `runGemma4MTPRounds` directly and track it from the round
    // loop. Left nil here to avoid misleading numbers.
    return Gemma4MTPBenchmarkResult(
        generatedTokens: generated,
        prefillSeconds: prefillElapsed,
        generationSeconds: genElapsed,
        acceptLengths: nil
    )
}
