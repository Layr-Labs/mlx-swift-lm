// Copyright © 2026 Apple Inc.
//
// Benchmark primitives for Gemma 4 Multi-Token Prediction (MTP)
// speculative decoding. Provides four measurement helpers:
//
//   - measureBaselineThroughput: B=1 no-drafter target-only greedy
//     generation tokens/sec, from prefill-end through generation-end.
//   - measureMTPThroughput: B=1 drafter-driven MTP generation tokens/sec
//     + per-round accept histogram (rounds driven inline so round
//     boundaries are visible).
//   - measureBatchedBaselineThroughput: B>1 target-only greedy, using
//     the same BatchKVCache / BatchRotatingKVCache used by MTP B>1.
//   - measureBatchedMTPThroughput: B>1 MTP via runGemma4MTPRoundsBatched.
//
// These are intentionally minimal — no model loading, no CLI, no
// harness. Callers supply a loaded target + bound drafter + prompt(s)
// and receive back a `BenchmarkResult`. A full driver (model download,
// result table, speedup calculation) lives in the test target's
// `realModelThroughputBenchmark`.

import Foundation
import MLX
@testable import MLXLLM
import MLXLMCommon

/// Batched benchmark result: per-row token counts + shared timing.
public struct Gemma4MTPBatchedBenchmarkResult: Sendable {
    public let batchSize: Int
    /// Total tokens emitted across all rows (sum).
    public let totalGenerated: Int
    public let prefillSeconds: Double
    public let generationSeconds: Double
    /// Sum of per-row tokens / generationSeconds. This is the metric that
    /// matters — it's the aggregate throughput of the system, which is
    /// what MTP batching is supposed to improve vs single-stream baseline.
    public var tokensPerSecond: Double {
        guard generationSeconds > 0 else { return 0 }
        return Double(totalGenerated) / generationSeconds
    }

    public init(
        batchSize: Int, totalGenerated: Int,
        prefillSeconds: Double, generationSeconds: Double
    ) {
        self.batchSize = batchSize
        self.totalGenerated = totalGenerated
        self.prefillSeconds = prefillSeconds
        self.generationSeconds = generationSeconds
    }
}

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
    var tok = logits[0..., -1, 0...].asType(.float32).argMax(axis: -1)
    eval(tok)
    let prefillElapsed = Date().timeIntervalSince(prefillStart)

    let genStart = Date()
    var generated = 1
    for _ in 1 ..< maxTokens {
        let input = tok[.newAxis, .ellipsis]
        logits = target(input, cache: cache)
        tok = logits[0..., -1, 0...].asType(.float32).argMax(axis: -1)
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
) throws -> Gemma4MTPBenchmarkResult {
    guard blockSize >= 2 && blockSize <= 16 else {
        throw Gemma4MTPError.invalidBlockSize(blockSize)
    }
    try drafter.bind(target: target)

    var prompt = promptTokens
    if prompt.ndim == 1 {
        prompt = prompt[.newAxis, .ellipsis]
    }
    let cache = target.newCache(parameters: nil)

    let prefillStart = Date()
    let prefillOut = target.forwardForMTP(prompt, cache: cache)
    let firstBonusArr = prefillOut.logits[0..., -1, 0...].asType(.float32).argMax(axis: -1)
    eval(firstBonusArr)
    var bonus = Int(firstBonusArr.item(Int32.self))
    var hidden = prefillOut.lastHidden[
        0..., -1 ..< prefillOut.lastHidden.dim(1), 0...]
    var sharedKV = prefillOut.capturedSharedKV
    let prefillElapsed = Date().timeIntervalSince(prefillStart)

    // Drive the round loop inline so we can track per-round accept counts
    // (the AsyncStream<Generation> public API doesn't delimit rounds).
    let genStart = Date()
    var emitted = 1  // the prefill bonus counts as the first emitted token
    var accepts: [Int] = []

    while emitted < maxTokens {
        let remaining = maxTokens - emitted
        let bs = min(blockSize, remaining + 1)
        if bs <= 1 { break }
        let round = runGemma4MTPGreedyRound(
            target: target,
            drafter: drafter,
            cache: cache,
            bonus: bonus,
            hidden: hidden,
            sharedKV: sharedKV,
            blockSize: bs)
        accepts.append(round.accepted)

        let emittedThisRound = min(round.tokens.count, remaining)
        emitted += emittedThisRound
        if emittedThisRound == 0 { break }

        // Update state for next round.
        hidden = round.hidden
        bonus = round.bonus
        sharedKV = round.sharedKV
    }
    let genElapsed = Date().timeIntervalSince(genStart)

    return Gemma4MTPBenchmarkResult(
        generatedTokens: emitted,
        prefillSeconds: prefillElapsed,
        generationSeconds: genElapsed,
        acceptLengths: accepts
    )
}

// MARK: - Batched (B>1) throughput

/// Run target-only greedy generation over B padded prompts, each for
/// `maxTokens` steps. Uses `BatchKVCache` / `BatchRotatingKVCache` so the
/// cache semantics match the MTP batched path.
public func measureBatchedBaselineThroughput(
    target: Gemma4TextModel,
    promptTokens: [[Int32]],
    maxTokens: Int
) -> Gemma4MTPBatchedBenchmarkResult {
    let B = promptTokens.count
    precondition(B >= 1, "Batch size must be >= 1")
    let maxLen = promptTokens.map(\.count).max() ?? 0
    // Right-pad with 0s (token id 0). For a throughput bench this is fine;
    // we're measuring compute, not semantic output. Real applications
    // would left-pad and track leftPadding per row.
    let padded = promptTokens.map { row -> [Int32] in
        row + Array(repeating: Int32(0), count: maxLen - row.count)
    }
    let flat = padded.flatMap { $0 }
    let prompt = MLXArray(flat, [B, maxLen])

    let firstKvShared = target.configuration.numHiddenLayers
                      - target.configuration.numKvSharedLayers
    let leftPadding = Array(repeating: 0, count: B)
    var cache: [KVCache] = []
    for i in 0 ..< firstKvShared {
        if target.configuration.layerTypes[i] == "full_attention" {
            cache.append(BatchKVCache(leftPadding: leftPadding))
        } else {
            cache.append(BatchRotatingKVCache(
                maxSize: target.configuration.slidingWindow,
                leftPadding: leftPadding))
        }
    }

    let prefillStart = Date()
    var logits = target(prompt, cache: cache)
    var tok = logits[0..., -1, 0...].asType(.float32).argMax(axis: -1)  // [B]
    eval(tok)
    let prefillElapsed = Date().timeIntervalSince(prefillStart)

    let genStart = Date()
    var total = B
    for _ in 1 ..< maxTokens {
        let input = tok.reshaped([B, 1])
        logits = target(input, cache: cache)
        tok = logits[0..., -1, 0...].asType(.float32).argMax(axis: -1)
        eval(tok)
        total += B
    }
    let genElapsed = Date().timeIntervalSince(genStart)

    return Gemma4MTPBatchedBenchmarkResult(
        batchSize: B, totalGenerated: total,
        prefillSeconds: prefillElapsed, generationSeconds: genElapsed)
}

/// Run MTP over B padded prompts via `runGemma4MTPRoundsBatched` with
/// per-row `maxTokens` budgets. Returns aggregate timing — sum of
/// emitted tokens / wall seconds. This is the variant that exercises
/// continuous-batching compaction: rows with smaller budgets finish
/// early, get removed from the active batch, and the remaining rows
/// run at a smaller B until they finish.
public func measureBatchedMTPThroughputStaggered(
    target: Gemma4TextModel,
    drafter: Gemma4AssistantDraftModel,
    promptTokens: [[Int32]],
    maxTokensPerRow: [Int],
    blockSize: Int
) async throws -> Gemma4MTPBatchedBenchmarkResult {
    guard blockSize >= 2 && blockSize <= 16 else {
        throw Gemma4MTPError.invalidBlockSize(blockSize)
    }
    try drafter.bind(target: target)

    let B = promptTokens.count
    precondition(maxTokensPerRow.count == B,
        "maxTokensPerRow must match the number of prompts")
    let maxLen = promptTokens.map(\.count).max() ?? 0
    let padded = promptTokens.map { row -> [Int32] in
        row + Array(repeating: Int32(0), count: maxLen - row.count)
    }
    let flat = padded.flatMap { $0 }
    let prompt = MLXArray(flat, [B, maxLen])

    let firstKvShared = target.configuration.numHiddenLayers
                      - target.configuration.numKvSharedLayers
    let leftPadding = Array(repeating: 0, count: B)
    var cache: [KVCache] = []
    for i in 0 ..< firstKvShared {
        if target.configuration.layerTypes[i] == "full_attention" {
            cache.append(BatchKVCache(leftPadding: leftPadding))
        } else {
            cache.append(BatchRotatingKVCache(
                maxSize: target.configuration.slidingWindow,
                leftPadding: leftPadding))
        }
    }

    let prefillStart = Date()
    let prefillOut = target.forwardForMTP(prompt, cache: cache)
    let firstBonusArr = prefillOut.logits[0..., -1, 0...].asType(.float32).argMax(axis: -1)
    eval(firstBonusArr)
    let bonusPerRow = firstBonusArr.asArray(Int32.self).map { Int($0) }
    let firstHidden = prefillOut.lastHidden[
        0..., -1 ..< prefillOut.lastHidden.dim(1), 0...]
    let firstSharedKV = prefillOut.capturedSharedKV
    let prefillElapsed = Date().timeIntervalSince(prefillStart)

    let genStart = Date()
    let stream = try runGemma4MTPRoundsBatched(
        target: target, drafter: drafter,
        targetCache: cache,
        firstBonus: bonusPerRow, firstHidden: firstHidden,
        firstSharedKV: firstSharedKV,
        maxTokens: maxTokensPerRow.max() ?? 0,
        blockSize: blockSize,
        eosTokenIds: nil,
        maxTokensPerRow: maxTokensPerRow
    )
    var total = 0
    for await step in stream {
        for slot in step.slots where slot.token != nil {
            total += 1
        }
    }
    let genElapsed = Date().timeIntervalSince(genStart)

    return Gemma4MTPBatchedBenchmarkResult(
        batchSize: B, totalGenerated: total,
        prefillSeconds: prefillElapsed, generationSeconds: genElapsed)
}

/// Run MTP over B padded prompts via `runGemma4MTPRoundsBatched`. Returns
/// aggregate timing — sum of emitted tokens / wall seconds.
public func measureBatchedMTPThroughput(
    target: Gemma4TextModel,
    drafter: Gemma4AssistantDraftModel,
    promptTokens: [[Int32]],
    maxTokens: Int,
    blockSize: Int
) async throws -> Gemma4MTPBatchedBenchmarkResult {
    guard blockSize >= 2 && blockSize <= 16 else {
        throw Gemma4MTPError.invalidBlockSize(blockSize)
    }
    try drafter.bind(target: target)

    let B = promptTokens.count
    let maxLen = promptTokens.map(\.count).max() ?? 0
    let padded = promptTokens.map { row -> [Int32] in
        row + Array(repeating: Int32(0), count: maxLen - row.count)
    }
    let flat = padded.flatMap { $0 }
    let prompt = MLXArray(flat, [B, maxLen])

    let firstKvShared = target.configuration.numHiddenLayers
                      - target.configuration.numKvSharedLayers
    let leftPadding = Array(repeating: 0, count: B)
    var cache: [KVCache] = []
    for i in 0 ..< firstKvShared {
        if target.configuration.layerTypes[i] == "full_attention" {
            cache.append(BatchKVCache(leftPadding: leftPadding))
        } else {
            cache.append(BatchRotatingKVCache(
                maxSize: target.configuration.slidingWindow,
                leftPadding: leftPadding))
        }
    }

    let prefillStart = Date()
    let prefillOut = target.forwardForMTP(prompt, cache: cache)
    let firstBonusArr = prefillOut.logits[0..., -1, 0...].asType(.float32).argMax(axis: -1)
    eval(firstBonusArr)
    let bonusPerRow = firstBonusArr.asArray(Int32.self).map { Int($0) }
    let firstHidden = prefillOut.lastHidden[
        0..., -1 ..< prefillOut.lastHidden.dim(1), 0...]
    let firstSharedKV = prefillOut.capturedSharedKV
    let prefillElapsed = Date().timeIntervalSince(prefillStart)

    let genStart = Date()
    let stream = try runGemma4MTPRoundsBatched(
        target: target, drafter: drafter,
        targetCache: cache,
        firstBonus: bonusPerRow, firstHidden: firstHidden,
        firstSharedKV: firstSharedKV,
        maxTokens: maxTokens, blockSize: blockSize,
        eosTokenIds: nil
    )
    var total = 0
    for await step in stream {
        for slot in step.slots where slot.token != nil {
            total += 1
        }
    }
    let genElapsed = Date().timeIntervalSince(genStart)

    return Gemma4MTPBatchedBenchmarkResult(
        batchSize: B, totalGenerated: total,
        prefillSeconds: prefillElapsed, generationSeconds: genElapsed)
}
