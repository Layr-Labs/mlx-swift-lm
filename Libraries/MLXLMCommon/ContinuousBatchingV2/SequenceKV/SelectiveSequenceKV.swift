// Copyright © 2026 Eigen Labs.
// Experimental request-local working sets. Canonical prefix snapshots remain dense.

import Foundation
import MLX

/// Explicit experiment configuration; no model or provider enables this by default.
/// Retain half of older history, with a dense recent band and spatial coverage.
public struct CBv2SelectiveKVPolicy: Sendable, Equatable {
    public let olderHistoryFraction: Double
    public let recentTokens: Int
    public let minimumTokens: Int
    public let chunkTokens: Int
    public let pruneInterval: Int
    public let anchorTokens: Int

    public init(
        olderHistoryFraction: Double = 0.5, recentTokens: Int = 512,
        minimumTokens: Int = 4096, chunkTokens: Int = 16,
        pruneInterval: Int = 256, anchorTokens: Int = 4
    ) {
        precondition(
            olderHistoryFraction.isFinite && olderHistoryFraction > 0
                && olderHistoryFraction <= 1)
        precondition(
            recentTokens > 0 && minimumTokens > recentTokens
                && chunkTokens > 0 && pruneInterval > 0 && anchorTokens >= 0)
        self.olderHistoryFraction = olderHistoryFraction
        self.recentTokens = recentTokens
        self.minimumTokens = minimumTokens
        self.chunkTokens = chunkTokens
        self.pruneInterval = pruneInterval
        self.anchorTokens = anchorTokens
    }
}

/// Logical allocation evidence, not an allocator or process-footprint receipt.
public struct CBv2SelectiveKVStatistics: Sendable, Codable, Equatable {
    public var sequenceLayers = 0
    public var pruningEvents = 0
    public var tokenEntriesRemoved = 0
    public var lastPruneSourceStorageBytes = 0
    public var lastPruneRetainedStorageBytes = 0

    mutating func add(_ other: Self) {
        sequenceLayers += other.sequenceLayers
        pruningEvents += other.pruningEvents
        tokenEntriesRemoved += other.tokenEntriesRemoved
        lastPruneSourceStorageBytes += other.lastPruneSourceStorageBytes
        lastPruneRetainedStorageBytes += other.lastPruneRetainedStorageBytes
    }
}

/// Full-attention KV whose absolute clock is independent of retained storage.
/// Selection happens only after dense prefill and outside speculative writes.
/// This is deliberately ineligible for canonical checkpoint export/adoption.
final class CBv2SelectiveSequenceKV: CBv2SequenceKV, CBv2InnerStateProviding {
    private var storage: CBv2FullSequenceKV
    let policy: CBv2SelectiveKVPolicy
    private let promptLength: Int
    private let maxLength: Int
    private let kvHeads: Int
    private let headDim: Int
    private var speculative = false
    private var lastPruneOffset = 0
    private(set) var absoluteOffset = 0
    private(set) var statistics = CBv2SelectiveKVStatistics(sequenceLayers: 1)
    var retainedCount: Int { storage.retainedCount }
    var byteCount: Int { storage.byteCount }

    init(
        promptLength: Int, maxLength: Int, kvHeads: Int, headDim: Int,
        valueHeadDim: Int, policy: CBv2SelectiveKVPolicy
    ) {
        self.promptLength = promptLength
        self.maxLength = maxLength
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.policy = policy
        self.storage = CBv2FullSequenceKV(
            promptLength: promptLength, maxLength: maxLength,
            kvHeads: kvHeads, headDim: headDim, valueHeadDim: valueHeadDim)
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        precondition(absoluteOffset + keys.dim(2) <= maxLength)
        let result = storage.update(keys: keys, values: values)
        absoluteOffset += keys.dim(2)
        return result
    }

    func snapshot() -> (keys: MLXArray, values: MLXArray, offset: Int) {
        let value = storage.snapshot()
        return (value.keys, value.values, absoluteOffset)
    }

    var supportsSpeculativeWrites: Bool { true }
    func beginSpeculativeWrite() {
        precondition(!speculative)
        speculative = true
    }
    func commitSpeculativeWrite() { speculative = false }
    func rollback(_ n: Int) {
        precondition(n >= 0 && n <= retainedCount)
        storage.rollback(n)
        absoluteOffset -= n
    }
    func cbv2InnerState() -> [MLXArray] { storage.cbv2InnerState() }

    /// Scores at most four recent queries, with O(queryHeads * 4 * history)
    /// score elements. Non-FP32 keys also need a full FP32 cast of
    /// O(kvHeads * history * headDim) elements; compaction keeps its source
    /// alive until the gathered destination has been evaluated.
    /// K already contains its original RoPE position. Sorting the selected
    /// indices preserves causal order; the new query rectangle stays dense.
    func prepareForAttention(queries: MLXArray, scale: Float, sinks: MLXArray?, softcap: Float?) {
        guard !speculative, absoluteOffset >= promptLength,
            absoluteOffset >= policy.minimumTokens,
            lastPruneOffset == 0 || absoluteOffset - lastPruneOffset >= policy.pruneInterval
        else { return }
        let anchorCount = min(policy.anchorTokens, retainedCount)
        let olderChunkCount = max(
            0,
            (retainedCount - anchorCount - policy.recentTokens)
                / policy.chunkTokens)
        let olderTokenBudget = Int(
            Double(max(0, absoluteOffset - anchorCount - policy.recentTokens))
                * policy.olderHistoryFraction)
        let retainedChunkCount = min(olderChunkCount, olderTokenBudget / policy.chunkTokens)
        guard olderChunkCount > 0, retainedChunkCount > 0, retainedChunkCount < olderChunkCount
        else {
            return
        }

        let history = storage.snapshot()
        let importance = attentionImportance(
            queries: queries, keys: history.keys,
            scale: scale, sinks: sinks, softcap: softcap)
        let indices = retainedIndices(
            importance: importance, anchorCount: anchorCount,
            olderChunkCount: olderChunkCount,
            retainedChunkCount: retainedChunkCount)
        // take owns compact buffers: slicing alone would keep the dense source.
        // Gather spare slots too, so the immediately following append does not
        // allocate and copy a second destination. Their values are unreachable
        // until update overwrites them, exactly like a rolled-back native tail.
        let spareCount = min(CBv2FullSequenceKV.initialSlack, maxLength - indices.size)
        let storageIndices = concatenated(
            [
                indices, MLXArray.zeros([spareCount], dtype: indices.dtype),
            ], axis: 0)
        let next = CBv2FullSequenceKV(
            compactedKeys: take(history.keys, storageIndices, axis: 2),
            compactedValues: take(history.values, storageIndices, axis: 2),
            retainedCount: indices.size, maxLength: maxLength)
        statistics.pruningEvents += 1
        statistics.tokenEntriesRemoved += retainedCount - next.retainedCount
        statistics.lastPruneSourceStorageBytes = storage.byteCount
        statistics.lastPruneRetainedStorageBytes = next.byteCount
        storage = next
        lastPruneOffset = absoluteOffset
    }

    private func attentionImportance(
        queries: MLXArray, keys: MLXArray, scale: Float, sinks: MLXArray?, softcap: Float?
    ) -> MLXArray {
        let queryCount = min(4, queries.dim(2))
        let queryHeads = queries.dim(1)
        precondition(queryHeads % kvHeads == 0)
        let q =
            (queries[.ellipsis, (queries.dim(2) - queryCount)..., 0...].asType(.float32) * scale)
            .reshaped([1, kvHeads, queryHeads / kvHeads, queryCount, headDim])
        let k = keys.asType(.float32).expandedDimensions(axis: 2)
        var logits = matmul(q, k.transposed(0, 1, 2, 4, 3))
            .reshaped([1, queryHeads, queryCount, retainedCount])
        if let cap = softcap { logits = cap * tanh(logits / cap) }
        let weights: MLXArray
        if let sinks {
            let sink = sinks.asType(.float32).reshaped([1, queryHeads, 1, 1])
            let peak = maximum(logits.max(axis: -1, keepDims: true), sink)
            let numerator = exp(logits - peak)
            weights = numerator / (numerator.sum(axis: -1, keepDims: true) + exp(sink - peak))
        } else {
            weights = softmax(logits, axis: -1, precise: true)
        }
        // Max across heads protects a sparse retrieval head from being diluted
        // by the many local heads. Sum within each chunk preserves its context.
        return weights.mean(axis: 2).max(axis: 1).reshaped([retainedCount])
    }

    private func retainedIndices(
        importance: MLXArray, anchorCount: Int, olderChunkCount: Int, retainedChunkCount: Int
    ) -> MLXArray {
        let tailStart = anchorCount + olderChunkCount * policy.chunkTokens
        let chunkScores = importance[anchorCount ..< tailStart]
            .reshaped([olderChunkCount, policy.chunkTokens]).sum(axis: 1)
        // Spend one quarter of the retained budget on evenly-spaced coverage.
        // Attention scoring cannot predict a later question's evidence needs.
        let coverageCount = max(1, retainedChunkCount / 4)
        let anchors = (0 ..< coverageCount).map { Int32($0 * olderChunkCount / coverageCount) }
        let anchorIDs = MLXArray(anchors)
        let rankedScores = chunkScores
        rankedScores[anchorIDs] = MLXArray(-Float.infinity)
        let ranked = argSort(-rankedScores)[..<max(0, retainedChunkCount - coverageCount)]
        let selectedChunks = sorted(concatenated([anchorIDs, ranked], axis: 0))
        let offsets = MLXArray(0 ..< policy.chunkTokens)
        let selected =
            (selectedChunks[0..., .newAxis] * policy.chunkTokens
            + offsets[.newAxis, 0...] + anchorCount).reshaped([-1])
        return concatenated(
            [
                MLXArray(0 ..< anchorCount), selected, MLXArray(tailStart ..< retainedCount),
            ], axis: 0)
    }
}

extension CBv2SequenceKV {
    func prepareSelectiveAttention(
        queries: MLXArray, scale: Float,
        sinks: MLXArray?, softcap: Float?
    ) {
        (self as? CBv2SelectiveSequenceKV)?.prepareForAttention(
            queries: queries, scale: scale, sinks: sinks, softcap: softcap)
    }
}
