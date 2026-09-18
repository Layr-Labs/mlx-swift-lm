// Copyright © 2026 Eigen Labs / Darkbloom Assignment 1.
//
// Portable gathered QSA prefill (Fusion `qsa_fast.contiguous_causal_gathered_qsa`
// + `_gathered_text_prefill_eligible` / #3355 text positions). Native Metal
// `qwen4_qsa_sparse_gqa` is capability-gated in `Qwen4ExpQSAMetal.swift`.
// MTP verify widths (S=2..15) use mlx-serve #352 `qsaVerifyGatherAttn`:
// the union of each row's top-k blocks plus the covering tail, then masked
// SDPA over R << kv. The prefill native kernel still walks the whole cache
// and is the 80K Lightning TTFT miss.
// Compact-tile softmax, portable-above-8K, and simdgroup_half8x8 QK were
// all slower than streaming the full K cache on this stack (16K 45.6 s /
// 45.9 s vs 35.5 s streamed; sg8 8K 20.5 s vs 16.3 s scalar). Native
// indexer scores + top-k (#3244) are capability-gated in
// `Qwen4ExpQSAIndexer.swift` and fail closed to this GEMM / argPartition.
// Pooled index keys can append only the new complete-block suffix (Fusion
// `pooled_indexer_keys`). Concat of the exact-width bank was a 128K miss
// (167697 / 164087 ms). The in-tree path is Fusion's zeros capacity buffer
// + slice assign, with evaluation carried by its final consumer. Default
// **off** until that no-sync buffer beats 144993 and passes state gates.
// Lab: `DARKBLOOM_QWEN4_QSA_POOLED_INCREMENTAL=1`. Index arithmetic uses
// Fusion `mx.arange` (int32) instead of host `stride`/`map` uploads.

import Foundation
import MLX
import MLXLMCommon
import os

public enum Qwen4ExpWeightSanitizer: Sendable {
    /// Keys that must not reach `update(parameters:verify: [.all])`.
    ///
    /// `keepVisionTower` is set by the `MLXVLM.Qwen4Exp` wrapper when it
    /// instantiates the Qwen3-VL tower (Fusion `qwen4_exp.Model.vision_tower`);
    /// the text-only `Qwen4ExpModel` keeps dropping the tower tensors.
    public static func shouldDrop(
        _ key: String, mmapPLE: Bool, keepVisionTower: Bool = false
    ) -> Bool {
        if key.hasPrefix("mtp.") {
            return true
        }
        if !keepVisionTower
            && (key.hasPrefix("vision_tower") || key.hasPrefix("model.visual")
                || key.contains(".visual."))
        {
            return true
        }
        if key.contains(".ple_embedding.layer_multipliers")
            || key.contains(".ple_embedding.ngram_heads_")
        {
            return true
        }
        if mmapPLE
            && (key.contains(".ngram_embedding.shards.")
                || key.contains(".ngram_embedding.shard_"))
        {
            return true
        }
        return false
    }

    /// Official BF16/Q4 cards have no Jundot-style oQ4e `weight_scale`.
    /// `Qwen4ExpShardedEmbedding` still owns that parameter (default ones).
    /// Strict `update(verify: [.all])` requires the key, so inject ones
    /// when the checkpoint omitted it. Never invent a second prefix.
    public static func injectMissingNgramWeightScale(
        into weights: inout [String: MLXArray],
        pleLayerIds: [Int]
    ) {
        let ones = MLXArray.ones([1], dtype: .bfloat16)
        for layerId in pleLayerIds {
            let idx = layerId - 1
            let needle = ".layers.\(idx).ple."
            guard let sample = weights.keys.first(where: { $0.contains(needle) }),
                let range = sample.range(of: needle)
            else { continue }
            let prefix = String(sample[..<range.lowerBound])
            let key =
                "\(prefix).layers.\(idx).ple.ple_embedding.ngram_embedding.weight_scale"
            if weights[key] == nil {
                weights[key] = ones
            }
        }
    }
}

/// The pre-materialization twin of `Qwen4ExpWeightSanitizer`.
///
/// `loadWeights` applies this closure to each safetensor key before `eval`.
/// In SSD PLE mode this is the mechanism that prevents the learned table's
/// packed weights/scales/biases from entering unified memory at load time.
/// The VLM wrapper sets `keepVisionTower`; the text wrappers do not.
public enum Qwen4ExpCheckpointLoad {
    public static func filter(
        mmapPLE: Bool, keepVisionTower: Bool
    ) -> CheckpointWeightLoadFilter {
        { key in
            !Qwen4ExpWeightSanitizer.shouldDrop(
                key, mmapPLE: mmapPLE, keepVisionTower: keepVisionTower)
        }
    }
}

public enum Qwen4ExpTextPositions {
    /// Canonical B1 text positions: absent, `(1, T)`, or broadcast-identical
    /// three-plane mRoPE (`(3, 1, T)` / `(3, T)`). Fusion #3355 emits `(1, T)`;
    /// wolfyy #3351 also accepts equal planes.
    public static func canonicalText(
        _ positionIds: MLXArray?, length: Int
    ) -> MLXArray? {
        guard let positionIds else { return nil }
        if positionIds.ndim == 2, positionIds.shape == [1, length] {
            return positionIds
        }
        if positionIds.ndim == 3, positionIds.dim(0) == 3,
            (positionIds.dim(1) == 1 || positionIds.dim(1) == length),
            positionIds.dim(2) == length
        {
            let first = positionIds[0]
            let plane = first.ndim == 2 ? first[0, 0...] : first
            let row = plane.reshaped([1, length])
            let same1 = all(equal(positionIds[1].reshaped([1, length]), row)).item(Bool.self)
            let same2 = all(equal(positionIds[2].reshaped([1, length]), row)).item(Bool.self)
            return same1 && same2 ? row : nil
        }
        return nil
    }

    public static func isBatchOneText(_ positionIds: MLXArray?, length: Int) -> Bool {
        positionIds == nil || canonicalText(positionIds, length: length) != nil
    }

    /// Genuine B1 three-plane M-RoPE (`(3, 1, T)` / `(3, T)`) whose planes
    /// diverge: image or video tokens inside the window. Returned as `(3, 1, T)`
    /// for the indexer sidecar (Fusion `QSAKVCache.index_position_ids` keeps
    /// all three planes for multimodal history).
    public static func multimodalBatchOne(
        _ positionIds: MLXArray?, length: Int
    ) -> MLXArray? {
        guard let positionIds, positionIds.ndim == 3 || positionIds.ndim == 2,
            positionIds.dim(0) == 3, positionIds.dim(-1) == length
        else { return nil }
        if positionIds.ndim == 3 {
            guard positionIds.dim(1) == 1 else { return nil }
            return positionIds
        }
        return positionIds.reshaped([3, 1, length])
    }

    /// B1 positions the gathered QSA prefill can index: absent, canonical
    /// text, or genuine multimodal planes. Batched rows stay dense.
    public static func isBatchOnePositions(_ positionIds: MLXArray?, length: Int) -> Bool {
        isBatchOneText(positionIds, length: length)
            || multimodalBatchOne(positionIds, length: length) != nil
    }

    /// Positions stored beside the raw indexer keys: text rows stay `(1, T)`,
    /// multimodal windows keep `(3, 1, T)`, and absent positions are the
    /// synthesized text ramp from the cache offset.
    public static func indexerPositions(
        positionIds: MLXArray?, textPositions: MLXArray?, offset: Int, length: Int
    ) -> MLXArray {
        if let textPositions { return textPositions }
        if let multimodal = multimodalBatchOne(positionIds, length: length) {
            return multimodal
        }
        return synthesized(offset: offset, length: length)
    }

    public static func synthesized(offset: Int, length: Int) -> MLXArray {
        MLXArray.arange(offset, offset + length, dtype: .int32).reshaped([1, length])
    }
}

enum Qwen4ExpGatheredQSA {
    /// Fusion `mx.arange(..., dtype=mx.int32)`. Host `stride`/`map` uploaded
    /// the 32K-wide pooled-block index at 128K on every QSA re-pool.
    static func int32Range(_ start: Int, _ stop: Int, step: Int = 1) -> MLXArray {
        MLXArray.arange(start, stop, step: step, dtype: .int32)
    }

    static func int32Range(_ stop: Int) -> MLXArray {
        MLXArray.arange(stop, dtype: .int32)
    }

    static func queryChunk(
        keyTokens: Int, native: Bool = false,
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Int {
        let portable: Int
        if keyTokens <= 4096 {
            portable = 32
        } else if keyTokens <= 16_384 {
            portable = 64
        } else {
            portable = 128
        }
        if native {
            return max(portable, Qwen4ExpNativeSparseGQA.resolvedQueryTile(environment: environment))
        }
        return portable
    }

    static func portableIndexerScores(
        queries: MLXArray, pooledKeys: MLXArray, headDim: Int
    ) -> MLXArray {
        let batch = queries.dim(0)
        let queryTokens = queries.dim(1)
        let queryHeads = queries.dim(2)
        let scores = matmul(
            queries.asType(.float32).reshaped([batch, queryTokens * queryHeads, headDim]),
            pooledKeys.asType(.float32).swappedAxes(-1, -2)
        ).reshaped([batch, queryTokens, queryHeads, pooledKeys.dim(1)])
        return sum(maximum(scores, 0), axis: -2) / sqrt(Float(headDim))
    }

    static func poolCompletedIndexKeys(
        indexKeys: MLXArray,
        indexPositionIds: MLXArray,
        compressRatio: Int,
        indexKeyNorm: (MLXArray) -> MLXArray,
        applyIndexRope: (MLXArray, MLXArray) -> MLXArray,
        startBlock: Int = 0,
        stopBlock: Int? = nil
    ) -> MLXArray {
        let completeBlocks = indexKeys.dim(1) / compressRatio
        let stop = stopBlock ?? completeBlocks
        precondition(startBlock >= 0 && startBlock <= stop && stop <= completeBlocks)
        precondition(stop > startBlock, "QSA pooled keys require a complete block")
        let blockCount = stop - startBlock
        let rawStart = startBlock * compressRatio
        let rawStop = stop * compressRatio
        var pooled = indexKeys[0..., rawStart ..< rawStop, 0...].reshaped([
            indexKeys.dim(0), blockCount, compressRatio, indexKeys.dim(-1),
        ])
        pooled = mean(pooled.asType(.float32), axis: -2).asType(indexKeys.dtype)
        pooled = indexKeyNorm(pooled)
        let blockStarts = int32Range(rawStart, rawStop, step: compressRatio)
        let pooledPositions: MLXArray
        if indexPositionIds.ndim == 3 {
            pooledPositions = takeAlong(
                indexPositionIds, blockStarts.reshaped([1, 1, blockCount]), axis: -1)
        } else {
            pooledPositions = takeAlong(
                indexPositionIds, blockStarts.reshaped([1, blockCount]), axis: -1)
        }
        return applyIndexRope(pooled[0..., .newAxis, 0..., 0...], pooledPositions)[0..., 0, 0..., 0...]
    }

    static func batchGatherTokens(_ values: MLXArray, indices: MLXArray) -> MLXArray {
        let batch = values.dim(0)
        let tokens = values.dim(1)
        let trailing = Array(values.shape.dropFirst(2))
        var offsetShape = [batch]
        offsetShape.append(contentsOf: Array(repeating: 1, count: indices.ndim - 1))
        let offsets = int32Range(batch).reshaped(offsetShape) * Int32(tokens)
        let flatIndices = (indices.asType(.int32) + offsets).reshaped([-1])
        let flatValues = values.reshaped([batch * tokens] + trailing)
        return flatValues[flatIndices].reshaped(indices.shape + trailing)
    }

    static func attend(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        indexQueries: MLXArray,
        indexKeys: MLXArray,
        indexPositionIds: MLXArray,
        queryHeads: Int,
        kvHeads: Int,
        headDim: Int,
        indexerHeadDim: Int,
        compressRatio: Int,
        tokenBudget: Int,
        indexKeyNorm: (MLXArray) -> MLXArray,
        applyIndexRope: (MLXArray, MLXArray) -> MLXArray,
        pooledIndexKeys: MLXArray?
    ) -> MLXArray {
        precondition(queries.ndim == 4 && queries.dim(0) == 1 && queries.dim(2) > 1)
        precondition(queries.dim(1) == queryHeads && queries.dim(3) == headDim)
        precondition(keys.shape == values.shape)
        precondition(keys.dim(1) == kvHeads && keys.dim(3) == headDim)
        precondition(queryHeads % kvHeads == 0)
        precondition(tokenBudget > 0 && tokenBudget % compressRatio == 0)

        let batch = 1
        let queryTokens = queries.dim(2)
        let keyTokens = keys.dim(2)
        precondition(queryTokens <= keyTokens)
        let queryStart = keyTokens - queryTokens
        let ratio = compressRatio
        let maxBlocks = keyTokens / ratio
        let blockBudget = tokenBudget / ratio
        let groups = queryHeads / kvHeads
        let keyRows = keys.transposed(0, 2, 1, 3)
        let valueRows = values.transposed(0, 2, 1, 3)
        let nativeEligible = Qwen4ExpNativeSparseGQA.canAttend(
            queries: queries, keys: keys, values: values,
            selectedWidth: min(maxBlocks, blockBudget))
        let chunk = queryChunk(keyTokens: keyTokens, native: nativeEligible)
        let pooled: MLXArray?
        if maxBlocks > 0 {
            if let pooledIndexKeys {
                precondition(pooledIndexKeys.shape == [batch, maxBlocks, indexerHeadDim])
                pooled = pooledIndexKeys
            } else {
                pooled = poolCompletedIndexKeys(
                    indexKeys: indexKeys,
                    indexPositionIds: indexPositionIds,
                    compressRatio: ratio,
                    indexKeyNorm: indexKeyNorm,
                    applyIndexRope: applyIndexRope)
            }
        } else {
            pooled = nil
        }

        var outputs: [MLXArray] = []
        var start = 0
        while start < queryTokens {
            let stop = min(start + chunk, queryTokens)
            let chunkTokens = stop - start
            let absolute = int32Range(queryStart + start, queryStart + stop)
            let visible = (absolute + 1).reshaped([1, chunkTokens])
            let completeCounts = visible / ratio
            let selectedIndices: MLXArray
            let selectedValid: MLXArray
            if maxBlocks > 0, let pooled {
                let chunkIndexQueries = indexQueries[0..., start ..< stop, 0..., 0...]
                let qOffset = queryStart + start
                var blockScores: MLXArray
                if Qwen4ExpNativeIndexer.matchesScoreGeometry(
                    queryHeads: chunkIndexQueries.dim(2), headDim: indexerHeadDim),
                    let nativeScores = Qwen4ExpNativeIndexer.scores(
                        queries: chunkIndexQueries, pooledKeys: pooled, maskQOffset: qOffset)
                {
                    blockScores = nativeScores
                } else {
                    blockScores = portableIndexerScores(
                        queries: chunkIndexQueries, pooledKeys: pooled, headDim: indexerHeadDim)
                    let blockIds = int32Range(maxBlocks).reshaped([
                        1, 1, maxBlocks,
                    ])
                    let validBlocks = broadcast(
                        blockIds, to: [batch, chunkTokens, maxBlocks])
                        .< completeCounts.reshaped([batch, chunkTokens, 1])
                    blockScores = MLX.where(
                        validBlocks, blockScores, MLXArray(-Float.greatestFiniteMagnitude))
                }
                let selectedWidth = min(maxBlocks, blockBudget)
                var selectedBlockRows = broadcast(
                    int32Range(selectedWidth).reshaped([1, 1, selectedWidth]),
                    to: [batch, chunkTokens, selectedWidth])
                if maxBlocks > blockBudget {
                    let ranked: MLXArray
                    if let nativeTopK = Qwen4ExpNativeIndexer.topKIndices(blockScores) {
                        ranked = nativeTopK.asType(.int32)
                    } else {
                        let kth = maxBlocks - blockBudget
                        ranked = argPartition(blockScores, kth: kth, axis: -1)[
                            0..., 0..., kth...
                        ].asType(.int32)
                    }
                    selectedBlockRows = MLX.where(
                        (completeCounts .<= Int32(blockBudget)).reshaped([batch, chunkTokens, 1]),
                        selectedBlockRows.asType(ranked.dtype),
                        ranked)
                }
                selectedBlockRows = sorted(selectedBlockRows, axis: -1).asType(.int32)
                if nativeEligible, selectedWidth == Qwen4ExpNativeSparseGQA.blockBudget,
                    let native = Qwen4ExpNativeSparseGQA.attend(
                        queries: queries[0..., 0..., start ..< stop, 0...],
                        keys: keys,
                        values: values,
                        selectedBlocks: selectedBlockRows,
                        qOffset: queryStart + start,
                        outputPartitions: verifyOutputPartitions(queryTokens: queryTokens))
                {
                    Qwen4ExpQSAInvocation.recordNative()
                    outputs.append(native)
                    start = stop
                    continue
                }
                let selectedCount = minimum(completeCounts, MLXArray(Int32(blockBudget)))
                let tokenOffsets = int32Range(ratio).reshaped([1, 1, 1, ratio])
                let gathered = selectedBlockRows.reshaped([
                    batch, chunkTokens, selectedWidth, 1,
                ]) * Int32(ratio) + tokenOffsets
                let blockValid = broadcast(
                    int32Range(selectedWidth).reshaped([
                        1, 1, selectedWidth, 1,
                    ]),
                    to: [batch, chunkTokens, selectedWidth, ratio])
                    .< selectedCount.reshaped([batch, chunkTokens, 1, 1])
                selectedIndices = gathered.reshaped([batch, chunkTokens, selectedWidth * ratio])
                selectedValid = blockValid.reshaped([batch, chunkTokens, selectedWidth * ratio])
            } else {
                selectedIndices = MLXArray.zeros([batch, chunkTokens, 0], dtype: .int32)
                selectedValid = MLXArray.zeros([batch, chunkTokens, 0], dtype: .bool)
            }

            let tailWidth = ratio - 1
            let tail = completeCounts.reshaped([batch, chunkTokens, 1]) * Int32(ratio)
                + int32Range(tailWidth).reshaped([1, 1, tailWidth])
            let tailValid = tail .< visible.reshaped([batch, chunkTokens, 1])
            let allIndices = concatenated([selectedIndices, tail], axis: -1)
            let allValid = concatenated([selectedValid, tailValid], axis: -1)
            let safeSelected = MLX.where(allValid, allIndices, 0).asType(.int32)
            let selectedKeys = batchGatherTokens(keyRows, indices: safeSelected)
                .transposed(0, 1, 3, 2, 4)
            let selectedValues = batchGatherTokens(valueRows, indices: safeSelected)
                .transposed(0, 1, 3, 2, 4)
            let chunkQueries = queries[0..., 0..., start ..< stop, 0...].transposed(0, 2, 1, 3)
            let groupedQueries = chunkQueries.reshaped([
                batch, chunkTokens, kvHeads, groups, headDim,
            ])
            var scores = matmul(
                groupedQueries.asType(.float32),
                selectedKeys.asType(.float32).swappedAxes(-1, -2)
            ) / sqrt(Float(headDim))
            scores = MLX.where(
                allValid.reshaped([batch, chunkTokens, 1, 1, allValid.dim(-1)]),
                scores,
                MLXArray(-Float.greatestFiniteMagnitude))
            let probabilities = softmax(scores, axis: -1).asType(chunkQueries.dtype)
            let output = matmul(probabilities, selectedValues)
                .reshaped([batch, chunkTokens, queryHeads, headDim])
            Qwen4ExpQSAInvocation.recordPortable()
            outputs.append(output)
            start = stop
        }
        return concatenated(outputs, axis: 1)
    }

    /// Fusion `_gathered_text_decode_eligible` block crossover: the token
    /// being decoded sits at `offset`, so `offset + 1` tokens are visible and
    /// QSA is sparse only once the completed blocks exceed the budget.
    static func decodeCrossesBudget(offset: Int, compressRatio: Int, tokenBudget: Int) -> Bool {
        guard compressRatio > 0, tokenBudget > 0 else { return false }
        return (offset + 1) / compressRatio > tokenBudget / compressRatio
    }

    /// Canonical singleton selector shared by full and compact KV readers.
    static func decodeSelectedBlocks(
        indexQueries: MLXArray, pooledIndexKeys: MLXArray, keyTokens: Int,
        indexerHeadDim: Int, maxBlocks: Int, blockBudget: Int
    ) -> MLXArray {
        var blockScores: MLXArray
        if Qwen4ExpNativeIndexer.matchesScoreGeometry(
            queryHeads: indexQueries.dim(2), headDim: indexerHeadDim),
            let nativeScores = Qwen4ExpNativeIndexer.scores(
                queries: indexQueries, pooledKeys: pooledIndexKeys, maskQOffset: keyTokens - 1)
        {
            blockScores = nativeScores
        } else {
            blockScores = portableIndexerScores(
                queries: indexQueries, pooledKeys: pooledIndexKeys, headDim: indexerHeadDim)
        }

        var selectedBlocks: MLXArray
        if let nativeTopK = Qwen4ExpNativeIndexer.topKIndices(blockScores) {
            selectedBlocks = nativeTopK
        } else {
            let kth = maxBlocks - blockBudget
            selectedBlocks = argPartition(blockScores, kth: kth, axis: -1)[0..., 0..., kth...]
        }
        // Top-k order is not chronological. Sort before the gather so SDPA
        // sees the official key order, then cast for signed index math.
        selectedBlocks = sorted(selectedBlocks, axis: -1).asType(.int32)
        return selectedBlocks
    }

    /// Canonical rectangular selector: compact storage changes no score,
    /// top-k tie handling, causal block count, or chronological sort.
    static func verifySelectedBlocks(
        indexQueries: MLXArray, pooledIndexKeys: MLXArray, keyTokens: Int,
        indexerHeadDim: Int, maxBlocks: Int, blockBudget: Int, ratio: Int
    ) -> MLXArray {
        let queryTokens = indexQueries.dim(1)
        let queryStart = keyTokens - queryTokens
        let completeCounts = floorDivide(
            int32Range(queryStart, keyTokens) + 1, ratio
        ).reshaped([1, queryTokens])
        var blockScores: MLXArray
        if Qwen4ExpNativeIndexer.matchesScoreGeometry(
            queryHeads: indexQueries.dim(2), headDim: indexerHeadDim),
            let nativeScores = Qwen4ExpNativeIndexer.scores(
                queries: indexQueries, pooledKeys: pooledIndexKeys, maskQOffset: queryStart)
        {
            blockScores = nativeScores
        } else {
            blockScores = portableIndexerScores(
                queries: indexQueries, pooledKeys: pooledIndexKeys, headDim: indexerHeadDim)
            let blockIds = int32Range(maxBlocks).reshaped([1, 1, maxBlocks])
            let validBlocks = broadcast(blockIds, to: [1, queryTokens, maxBlocks])
                .< completeCounts.reshaped([1, queryTokens, 1])
            blockScores = MLX.where(
                validBlocks, blockScores, MLXArray(-Float.greatestFiniteMagnitude))
        }
        let selectedWidth = min(maxBlocks, blockBudget)
        var selectedBlockRows = broadcast(
            int32Range(selectedWidth).reshaped([1, 1, selectedWidth]),
            to: [1, queryTokens, selectedWidth])
        if maxBlocks > blockBudget {
            let ranked: MLXArray
            if let nativeTopK = Qwen4ExpNativeIndexer.topKIndices(blockScores) {
                ranked = nativeTopK.asType(.int32)
            } else {
                let kth = maxBlocks - blockBudget
                ranked = argPartition(blockScores, kth: kth, axis: -1)[
                    0..., 0..., kth...
                ].asType(.int32)
            }
            selectedBlockRows = MLX.where(
                (completeCounts .<= Int32(blockBudget)).reshaped([1, queryTokens, 1]),
                selectedBlockRows.asType(ranked.dtype),
                ranked)
        }
        selectedBlockRows = sorted(selectedBlockRows, axis: -1).asType(.int32)
        return selectedBlockRows
    }

    /// Fusion `contiguous_causal_gathered_qsa_decode`: singleton exact QSA
    /// over only the selected K/V rows. `keys` / `values` already hold the
    /// new token. Every completed block is causal for the final token, so
    /// the top-`budget/ratio` blocks are gathered in chronological order and
    /// the zero-to-three incomplete tail rows appended; SDPA then runs
    /// unmasked over `budget + ratio - 1` rows instead of the full KV.
    /// Returns `[1, 1, H, D]` (token-major like `attend`).
    static func attendDecode(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        indexQueries: MLXArray,
        pooledIndexKeys: MLXArray,
        queryHeads: Int,
        kvHeads: Int,
        headDim: Int,
        indexerHeadDim: Int,
        compressRatio: Int,
        tokenBudget: Int
    ) -> MLXArray {
        precondition(
            queries.ndim == 4 && queries.dim(0) == 1 && queries.dim(1) == queryHeads
                && queries.dim(2) == 1 && queries.dim(3) == headDim,
            "gathered QSA decode requires [1, H, 1, D] queries")
        precondition(keys.shape == values.shape && keys.ndim == 4)
        precondition(keys.dim(0) == 1 && keys.dim(1) == kvHeads && keys.dim(3) == headDim)
        precondition(keys.dtype == queries.dtype && values.dtype == queries.dtype)
        precondition(
            indexQueries.ndim == 4 && indexQueries.dim(0) == 1 && indexQueries.dim(1) == 1
                && indexQueries.dim(3) == indexerHeadDim,
            "QSA decode index queries must have shape [1, 1, H, D]")
        precondition(tokenBudget > 0 && tokenBudget % compressRatio == 0)
        precondition(queryHeads % kvHeads == 0)

        let keyTokens = keys.dim(2)
        let ratio = compressRatio
        let maxBlocks = keyTokens / ratio
        let blockBudget = tokenBudget / ratio
        precondition(maxBlocks > blockBudget, "gathered QSA decode requires a sparse block crossover")
        precondition(
            pooledIndexKeys.shape == [1, maxBlocks, indexerHeadDim],
            "QSA decode pooled index-key cache has the wrong shape")

        let selectedBlocks = decodeSelectedBlocks(
            indexQueries: indexQueries, pooledIndexKeys: pooledIndexKeys, keyTokens: keyTokens,
            indexerHeadDim: indexerHeadDim, maxBlocks: maxBlocks, blockBudget: blockBudget)
        // Fusion #3244 native sparse-GQA for L=1: same selector, full KV, no
        // host gather+SDPA. Geometry miss falls through to the gathered path
        // (tiny decode tests, kill-switch). Prefill already uses this kernel.
        if !decodeGatherEnabled(), Qwen4ExpNativeSparseGQA.canAttend(
            queries: queries, keys: keys, values: values, selectedWidth: blockBudget),
            let native = Qwen4ExpNativeSparseGQA.attend(
                queries: queries,
                keys: keys,
                values: values,
                selectedBlocks: selectedBlocks.reshaped([1, 1, blockBudget]),
                qOffset: keyTokens - 1,
                outputPartitions: decodeOutputPartitions())
        {
            Qwen4ExpQSAInvocation.recordDecode()
            return native
        }
        var selectedTokens = (
            selectedBlocks.reshaped([1, blockBudget, 1]) * Int32(ratio)
                + int32Range(ratio).reshaped([1, 1, ratio])
        ).reshaped([1, blockBudget * ratio])
        let completeKeyLen = maxBlocks * ratio
        if completeKeyLen < keyTokens {
            selectedTokens = concatenated(
                [selectedTokens, int32Range(completeKeyLen, keyTokens).reshaped([1, -1])],
                axis: -1)
        }

        let selectedKeys = batchGatherTokens(keys.transposed(0, 2, 1, 3), indices: selectedTokens)
            .transposed(0, 2, 1, 3)
        let selectedValues = batchGatherTokens(
            values.transposed(0, 2, 1, 3), indices: selectedTokens
        ).transposed(0, 2, 1, 3)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: selectedKeys,
            values: selectedValues,
            scale: pow(Float(headDim), -0.5),
            mask: .none)
        Qwen4ExpQSAInvocation.recordDecode()
        return output.transposed(0, 2, 1, 3)
    }

    /// Lab-only singleton dispatch. The selector, chronological key order,
    /// incomplete-block tail and activation dtype are unchanged. This uses
    /// the existing gathered native SDPA path over only selected tokens; it
    /// does not replace QSA with full-context dense attention. Never changes
    /// prefill or rectangular MTP verification dispatch.
    static let decodeGatherEnvFlag = "DARKBLOOM_QWEN4_QSA_DECODE_GATHER"

    /// Separate lab arm preserving the original Steel arithmetic per output
    /// component. Only this singleton call site consumes the setting.
    static let decodeOutputPartitionsEnvFlag = "DARKBLOOM_QWEN4_QSA_DECODE_OUTPUT_PARTITIONS"

    static func decodeOutputPartitions(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Int {
        guard Qwen4ExpNativeSparseGQA.steelEnabled(environment: environment),
            let raw = environment[decodeOutputPartitionsEnvFlag],
            let count = Int(raw), [1, 2, 4].contains(count)
        else { return 1 }
        return count
    }

    /// Separate rectangular experiment. Use the WHOLE input width, not the
    /// final prefill tile width, so long prefills always retain one partition.
    static let verifyOutputPartitionsEnvFlag = "DARKBLOOM_QWEN4_QSA_VERIFY_OUTPUT_PARTITIONS"

    static func verifyOutputPartitions(
        queryTokens: Int, environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Int {
        guard (2...optimizedVerifyMaxQueryTokens).contains(queryTokens),
              Qwen4ExpNativeSparseGQA.steelEnabled(environment: environment),
              let raw = environment[verifyOutputPartitionsEnvFlag],
              let count = Int(raw), [1, 2, 4].contains(count) else { return 1 }
        return count
    }

    static func decodeGatherEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        #if DEBUG
            let raw = environment[decodeGatherEnvFlag]?.trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
        #else
            // The forced gather+SDPA experiment failed exact output parity.
            // This does not remove legitimate geometry-based fallbacks.
            return false
        #endif
    }

    /// mlx-serve `FUSED256_MIN_Q_LEN - 1`. Prefill native GQA has no q_len
    /// floor and would walk the whole cache for an MTP verify block.
    static let verifyMaxQueryTokens = 15
    /// Widest query block covered by the exact native compact/strided,
    /// parallel-score and independent-output-partition qualification matrix.
    /// Lightning draft depth five verifies the seed plus five drafts (S=6).
    static let optimizedVerifyMaxQueryTokens = 6
    /// mlx-serve `QSA_VERIFY_GATHER_MIN_KV_DEFAULT`. Below this the union
    /// is most of the cache and the prefill/native arm wins.
    static let verifyMinKeyTokensDefault = 16_384
    static let verifyGatherEnvFlag = "DARKBLOOM_QWEN4_QSA_VERIFY_GATHER"
    static let verifyMinKeyTokensEnvFlag = "DARKBLOOM_QWEN4_QSA_VERIFY_GATHER_MIN_KV"

    static func verifyGatherEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        let raw = environment[verifyGatherEnvFlag]?.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        if raw == "0" || raw == "false" || raw == "no" || raw == "off" {
            return false
        }
        return true
    }

    static func verifyMinKeyTokens(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Int {
        let raw = environment[verifyMinKeyTokensEnvFlag]?.trimmingCharacters(
            in: .whitespacesAndNewlines)
        if let raw, let parsed = Int(raw), parsed > 0 {
            return parsed
        }
        return verifyMinKeyTokensDefault
    }

    /// mlx-serve `QsaVerifyGeom`: fixed-size union of S rows' selected
    /// blocks that sit entirely below row-0's tail start, plus the covering
    /// tail `[tailStart, keyTokens)`.
    struct VerifyGeometry: Equatable, Sendable {
        var completeBlocks: Int
        var blocksBelowTail: Int
        var tailStart: Int
        var tailLength: Int
        var unionSlots: Int
        var gatheredRows: Int
    }

    static func verifyGeometry(
        queryTokens: Int, keyTokens: Int, compressRatio: Int, blockBudget: Int
    ) -> VerifyGeometry? {
        guard compressRatio > 0, blockBudget > 0 else { return nil }
        guard queryTokens >= 2, queryTokens <= verifyMaxQueryTokens else { return nil }
        guard keyTokens > queryTokens else { return nil }
        let offset = keyTokens - queryTokens
        let completeBlocks = keyTokens / compressRatio
        let tailStart = ((offset + 1) / compressRatio) * compressRatio
        guard tailStart <= keyTokens else { return nil }
        let blocksBelowTail = tailStart / compressRatio
        let unionSlots = min(blocksBelowTail, queryTokens * blockBudget)
        return VerifyGeometry(
            completeBlocks: completeBlocks,
            blocksBelowTail: blocksBelowTail,
            tailStart: tailStart,
            tailLength: keyTokens - tailStart,
            unionSlots: unionSlots,
            gatheredRows: unionSlots * compressRatio + (keyTokens - tailStart))
    }

    /// mlx-serve `qsaVerifyPlanHost`: ascending gathered token ids and the
    /// `[S, R]` visibility mask. `blocks` is `[S * kb]` row-major, `Int32.max`
    /// past each row's visible count.
    struct VerifyHostPlan: Equatable, Sendable {
        var rows: [Int32]
        var mask: [Bool]
    }

    static func verifyHostPlan(
        blocks: [Int32], queryTokens: Int, blockBudget: Int, keyTokens: Int, compressRatio: Int
    ) -> VerifyHostPlan? {
        guard let geom = verifyGeometry(
            queryTokens: queryTokens, keyTokens: keyTokens,
            compressRatio: compressRatio, blockBudget: blockBudget)
        else { return nil }
        guard blocks.count == queryTokens * blockBudget else { return nil }
        let nbLo = geom.blocksBelowTail
        var inUnion = [Bool](repeating: false, count: max(nbLo, 0))
        for i in 0 ..< queryTokens {
            for c in 0 ..< blockBudget {
                let b = blocks[i * blockBudget + c]
                if b >= 0, b < nbLo {
                    inUnion[Int(b)] = true
                }
            }
        }
        // mlx-serve pads the union to a fixed `S * kb` with masked fillers
        // so the graph stays shapeless. Those fillers make R = S·budget
        // even when nearby MTP rows share almost the same top-k. Pack only
        // the true union: lossless (fillers were masked everywhere) and R
        // tracks overlap instead of the worst-case product.
        var ids: [Int32] = []
        ids.reserveCapacity(geom.unionSlots)
        for (b, u) in inUnion.enumerated() where u && ids.count < geom.unionSlots {
            ids.append(Int32(b))
        }
        ids.sort()
        let gatheredRows = ids.count * compressRatio + geom.tailLength
        var rows = [Int32](repeating: 0, count: gatheredRows)
        var r = 0
        for b in ids {
            for j in 0 ..< compressRatio {
                rows[r] = b * Int32(compressRatio) + Int32(j)
                r += 1
            }
        }
        var t = Int32(geom.tailStart)
        while t < Int32(keyTokens) {
            rows[r] = t
            r += 1
            t += 1
        }
        let offset = keyTokens - queryTokens
        var mask = [Bool](repeating: false, count: queryTokens * gatheredRows)
        for i in 0 ..< queryTokens {
            let p = Int32(offset + i)
            let ts = ((p + 1) / Int32(compressRatio)) * Int32(compressRatio)
            for (slot, tok) in rows.enumerated() {
                if tok > p { continue }
                var vis = tok >= ts
                if !vis {
                    let b = tok / Int32(compressRatio)
                    if b < Int32(geom.completeBlocks) {
                        for c in 0 ..< blockBudget {
                            if blocks[i * blockBudget + c] == b {
                                vis = true
                                break
                            }
                        }
                    }
                }
                mask[i * gatheredRows + slot] = vis
            }
        }
        return VerifyHostPlan(rows: rows, mask: mask)
    }

    /// mlx-serve `qsaVerifyDenseHost`: selected-block tokens ∪ the row's
    /// incomplete tail, ∧ causal.
    static func verifyDenseMask(
        blocks: [Int32], queryTokens: Int, blockBudget: Int, keyTokens: Int, compressRatio: Int
    ) -> [Bool] {
        let nb = keyTokens / compressRatio
        let offset = keyTokens - queryTokens
        var out = [Bool](repeating: false, count: queryTokens * keyTokens)
        for i in 0 ..< queryTokens {
            let p = offset + i
            let ts = ((p + 1) / compressRatio) * compressRatio
            for t in 0 ..< keyTokens {
                if t > p { continue }
                if t >= ts {
                    out[i * keyTokens + t] = true
                    continue
                }
                if t >= nb * compressRatio { continue }
                let b = Int32(t / compressRatio)
                for c in 0 ..< blockBudget {
                    if blocks[i * blockBudget + c] == b {
                        out[i * keyTokens + t] = true
                        break
                    }
                }
            }
        }
        return out
    }

    /// mlx-serve `qsaVerifyGatherAttn`: gather the fixed-size union + covering
    /// tail, then SDPA under the per-row `[S, R]` mask. Returns `[1, S, H, D]`.
    static func attendVerify(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        indexQueries: MLXArray,
        pooledIndexKeys: MLXArray,
        queryHeads: Int,
        kvHeads: Int,
        headDim: Int,
        indexerHeadDim: Int,
        compressRatio: Int,
        tokenBudget: Int,
        minKeyTokens: Int
    ) -> MLXArray {
        let queryTokens = queries.dim(2)
        let keyTokens = keys.dim(2)
        let blockBudget = tokenBudget / compressRatio
        precondition(
            queries.ndim == 4 && queries.dim(0) == 1 && queries.dim(1) == queryHeads
                && queryTokens >= 2 && queryTokens <= verifyMaxQueryTokens
                && queries.dim(3) == headDim,
            "gathered QSA verify requires [1, H, S, D] queries with S in 2...15")
        precondition(keys.shape == values.shape && keys.ndim == 4)
        precondition(keys.dim(0) == 1 && keys.dim(1) == kvHeads && keys.dim(3) == headDim)
        precondition(keys.dtype == queries.dtype && values.dtype == queries.dtype)
        precondition(
            indexQueries.ndim == 4 && indexQueries.dim(0) == 1
                && indexQueries.dim(1) == queryTokens
                && indexQueries.dim(3) == indexerHeadDim)
        precondition(tokenBudget > 0 && tokenBudget % compressRatio == 0)
        precondition(queryHeads % kvHeads == 0)
        precondition(keyTokens > minKeyTokens)
        guard let geom = verifyGeometry(
            queryTokens: queryTokens, keyTokens: keyTokens,
            compressRatio: compressRatio, blockBudget: blockBudget),
            geom.unionSlots >= 0, geom.gatheredRows > 0, geom.gatheredRows < keyTokens
        else {
            preconditionFailure("gathered QSA verify geometry is not a sparsity win")
        }

        let ratio = compressRatio
        let maxBlocks = keyTokens / ratio
        precondition(maxBlocks > blockBudget, "gathered QSA verify requires a sparse block crossover")
        precondition(
            pooledIndexKeys.shape == [1, maxBlocks, indexerHeadDim],
            "QSA verify pooled index-key cache has the wrong shape")

        let queryStart = keyTokens - queryTokens
        let completeCounts = floorDivide(
            int32Range(queryStart, keyTokens) + 1, ratio
        ).reshaped([1, queryTokens])
        let selectedWidth = min(maxBlocks, blockBudget)
        var selectedBlockRows = verifySelectedBlocks(
            indexQueries: indexQueries, pooledIndexKeys: pooledIndexKeys, keyTokens: keyTokens,
            indexerHeadDim: indexerHeadDim, maxBlocks: maxBlocks, blockBudget: blockBudget, ratio: ratio)
        // MTP verify is S=2..15. Union+masked SDPA is lossless but walks a
        // rectangular gather; native sparse-GQA is the same kernel prefill
        // already uses for qL>1. Tiny-geometry tests fall through.
        if selectedWidth == Qwen4ExpNativeSparseGQA.blockBudget,
            Qwen4ExpNativeSparseGQA.canAttend(
                queries: queries, keys: keys, values: values, selectedWidth: selectedWidth),
            let native = Qwen4ExpNativeSparseGQA.attend(
                queries: queries,
                keys: keys,
                values: values,
                selectedBlocks: selectedBlockRows,
                qOffset: queryStart,
                outputPartitions: verifyOutputPartitions(queryTokens: queryTokens))
        {
            Qwen4ExpQSAInvocation.recordVerify()
            return native
        }
        let selectedCount = minimum(completeCounts, MLXArray(Int32(blockBudget)))
            .reshaped([1, queryTokens, 1])
        let slotIds = broadcast(
            int32Range(selectedWidth).reshaped([1, 1, selectedWidth]),
            to: [1, queryTokens, selectedWidth])
        selectedBlockRows = MLX.where(
            slotIds .< selectedCount, selectedBlockRows, MLXArray(Int32.max))

        let (tokInt, vis) = verifyUnionTokensAndMask(
            selectedBlocks: selectedBlockRows, geometry: geom,
            queryTokens: queryTokens, keyTokens: keyTokens, compressRatio: ratio)
        let gatheredRows = tokInt.dim(0)
        let add = MLX.where(vis, MLXArray(Float(0)), MLXArray(-Float.infinity))
            .asType(queries.dtype)
            .reshaped([1, 1, queryTokens, gatheredRows])
        let tok2 = tokInt.reshaped([1, gatheredRows])
        let gatheredKeys = batchGatherTokens(keys.transposed(0, 2, 1, 3), indices: tok2)
            .transposed(0, 2, 1, 3)
        let gatheredValues = batchGatherTokens(values.transposed(0, 2, 1, 3), indices: tok2)
            .transposed(0, 2, 1, 3)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: gatheredKeys,
            values: gatheredValues,
            scale: pow(Float(headDim), -0.5),
            mask: .array(add))
        Qwen4ExpQSAInvocation.recordVerify()
        return output.transposed(0, 2, 1, 3)
    }

    /// mlx-serve union + `[S, R]` visibility from a `[1, S, kb]` block
    /// selection (`Int32.max` unused). Token ids are strictly ascending.
    static func verifyUnionTokensAndMask(
        selectedBlocks: MLXArray, geometry geom: VerifyGeometry,
        queryTokens: Int, keyTokens: Int, compressRatio ratio: Int
    ) -> (MLXArray, MLXArray) {
        let maxBlocks = keyTokens / ratio
        let selWidth = maxBlocks + 2
        let validB = selectedBlocks .< Int32(maxBlocks)
        let sink = MLXArray(Int32(maxBlocks + 1))
        let safeB = MLX.where(validB, selectedBlocks, sink)
        let falses = MLXArray.zeros([queryTokens, selWidth], dtype: .bool)
        let sel = putAlong(
            falses, safeB.reshaped([queryTokens, selectedBlocks.dim(-1)]),
            values: MLXArray(true), axis: -1)

        let tok: MLXArray
        if geom.unionSlots > 0 {
            let selLo = sel[0..., 0 ..< geom.blocksBelowTail]
            let unionB = any(selLo, axis: 0)
            CBv2DeferredHostFill.resolveBeforeEvaluation()
            eval(unionB)
            let nUnion = min(
                geom.unionSlots,
                Int(sum(unionB.asType(.int32)).item(Int32.self)))
            let tokBlocks: MLXArray?
            if nUnion > 0 {
                let loIdx = int32Range(geom.blocksBelowTail)
                let key = MLX.where(unionB, loIdx, loIdx + Int32(geom.blocksBelowTail))
                let order = argSort(key, axis: -1).asType(.int32)
                let ids = sorted(
                    take(order, int32Range(nUnion), axis: 0), axis: -1
                ).asType(.int32)
                tokBlocks = (
                    ids.reshaped([nUnion, 1]) * Int32(ratio)
                        + int32Range(ratio).reshaped([1, ratio])
                ).reshaped([nUnion * ratio])
            } else {
                tokBlocks = nil
            }
            if let tokBlocks, geom.tailLength > 0 {
                tok = concatenated(
                    [tokBlocks, int32Range(geom.tailStart, keyTokens)], axis: 0)
            } else if let tokBlocks {
                tok = tokBlocks
            } else if geom.tailLength > 0 {
                tok = int32Range(geom.tailStart, keyTokens)
            } else {
                preconditionFailure("gathered QSA verify produced an empty row set")
            }
        } else if geom.tailLength > 0 {
            tok = int32Range(geom.tailStart, keyTokens)
        } else {
            preconditionFailure("gathered QSA verify produced an empty row set")
        }

        let tokInt = tok.asType(.int32)
        let gatheredRows = tokInt.dim(0)
        let blkOf = floorDivide(tokInt, ratio).asType(.int32)
        let member = take(sel, blkOf, axis: 1)
        let queryStart = keyTokens - queryTokens
        let pos = int32Range(queryStart, keyTokens).reshaped([queryTokens, 1])
        let rowTailStart = floorDivide(pos + 1, ratio).asType(.int32) * Int32(ratio)
        let tokRow = tokInt.reshaped([1, gatheredRows])
        let tailvis = tokRow .>= rowTailStart
        let causal = tokRow .<= pos
        let vis = (member .|| tailvis) .&& causal
        return (tokInt, vis)
    }
}

/// Fusion `QSAKVCache.pooled_indexer_keys`: keep completed pooled blocks in
/// a stepped zeros buffer and pool only the new suffix. Concat of an
/// exact-width bank compiled a 16-deep graph into the 128K indexer GEMM
/// (167697 / 164087 ms). Default **off** while the no-sync buffer candidate
/// completes whole-model qualification.
enum Qwen4ExpPooledIndex: Sendable {
    static let envFlag = "DARKBLOOM_QWEN4_QSA_POOLED_INCREMENTAL"
    /// Fusion `QSAKVCache.index_step`. Block capacity grows by
    /// `tokenStep / compressRatio` (2048 blocks at Flash-Next r=4).
    static let tokenStep = 8192

    static func isEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        let raw = environment[envFlag]?.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
    }

    /// Fusion `_growth_capacity`: round `needed` up to `step`, at least
    /// double the current allocation when growing.
    static func growthCapacity(current: Int, needed: Int, step: Int) -> Int {
        let step = max(1, step)
        let needed = max(0, needed)
        let stepped = ((needed + step - 1) / step) * step
        let floor = current > 0 ? 2 * current : step
        return max(stepped, floor)
    }

    static func logicalKeys(_ stored: MLXArray?, blocks: Int) -> MLXArray? {
        guard let stored, blocks > 0 else { return nil }
        precondition(stored.dim(1) >= blocks, "QSA pooled buffer shorter than logical blocks")
        return stored[0..., 0 ..< blocks, 0...]
    }

    /// MTP can roll the authoritative K/V row back while the sidecar keeps
    /// its capacity bank. Raw keys are overwritten at `offset`, but pooled
    /// blocks that included rejected columns must not be reused merely
    /// because a later verify window reaches the same block count.
    static func reconcileCommittedFrontier(
        cache: CBv2Qwen4GatheredCache,
        committedTokens: Int,
        compressRatio: Int
    ) {
        let committedBlocks = max(0, committedTokens) / max(1, compressRatio)
        guard cache.qwen4PooledIndexBlocks > committedBlocks else { return }
        cache.qwen4PooledIndexKeys = logicalKeys(
            cache.qwen4PooledIndexKeys, blocks: committedBlocks)
        cache.qwen4PooledIndexBlocks = committedBlocks
    }

    /// Exact vs a full `poolCompletedIndexKeys` over the same raw keys.
    /// `keys` may be wider than `blocks` (capacity). Callers that attend
    /// must use `logicalKeys`.
    static func extend(
        stored: MLXArray?,
        storedBlocks: Int,
        indexKeys: MLXArray,
        indexPositionIds: MLXArray,
        compressRatio: Int,
        indexKeyNorm: (MLXArray) -> MLXArray,
        applyIndexRope: (MLXArray, MLXArray) -> MLXArray
    ) -> (keys: MLXArray?, blocks: Int) {
        let completeBlocks = indexKeys.dim(1) / compressRatio
        guard completeBlocks > 0 else { return (nil, 0) }
        var start = storedBlocks
        var buffer = stored
        let capacity = buffer?.dim(1) ?? 0
        if start > completeBlocks || start < 0
            || (start > 0 && (buffer == nil || capacity < start))
        {
            start = 0
            buffer = nil
        }
        if start >= completeBlocks {
            return (buffer, start)
        }
        let suffix = Qwen4ExpGatheredQSA.poolCompletedIndexKeys(
            indexKeys: indexKeys,
            indexPositionIds: indexPositionIds,
            compressRatio: compressRatio,
            indexKeyNorm: indexKeyNorm,
            applyIndexRope: applyIndexRope,
            startBlock: start,
            stopBlock: completeBlocks)
        var bank: MLXArray
        if let buffer, buffer.dim(1) >= completeBlocks {
            bank = buffer
        } else {
            let step = max(1, tokenStep / max(1, compressRatio))
            let grown = growthCapacity(
                current: buffer?.dim(1) ?? 0, needed: completeBlocks, step: step)
            bank = MLXArray.zeros(
                [suffix.dim(0), grown, suffix.dim(-1)], dtype: suffix.dtype)
            if let buffer, start > 0 {
                bank[0..., 0 ..< start, 0...] = buffer[0..., 0 ..< start, 0...]
            }
        }
        bank[0..., start ..< completeBlocks, 0...] = suffix
        return (bank, completeBlocks)
    }

    static func update(
        cache: CBv2Qwen4GatheredCache,
        compressRatio: Int,
        logicalTokens: Int,
        indexKeyNorm: (MLXArray) -> MLXArray,
        applyIndexRope: (MLXArray, MLXArray) -> MLXArray,
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> MLXArray? {
        guard isEnabled(environment: environment),
            let indexKeys = Qwen4ExpIndexCapacity.logicalTokens(
                cache.qwen4IndexKeys, length: logicalTokens),
            let positions = Qwen4ExpIndexCapacity.logicalPositions(
                cache.qwen4IndexPositionIds, length: logicalTokens)
        else { return nil }
        let (keys, blocks) = extend(
            stored: cache.qwen4PooledIndexKeys,
            storedBlocks: cache.qwen4PooledIndexBlocks,
            indexKeys: indexKeys,
            indexPositionIds: positions,
            compressRatio: compressRatio,
            indexKeyNorm: indexKeyNorm,
            applyIndexRope: applyIndexRope)
        cache.qwen4PooledIndexKeys = keys
        cache.qwen4PooledIndexBlocks = blocks
        if let keys {
            // The selected-block scores consume this logical bank before the
            // final logits evaluation. Keep that dependency in the ordinary
            // lazy graph instead of synchronizing all 12 QSA layers here.
            Qwen4ExpPooledIndexInvocation.record(
                logicalTokens: logicalTokens, blocks: blocks, capacity: keys.dim(1))
        }
        return logicalKeys(keys, blocks: blocks)
    }

    /// Prefill already pooled the completed blocks. L=1 decode (and S=2..15
    /// verify) usually does not complete a new block, so re-running
    /// `poolCompletedIndexKeys` over 80K tokens × 12 QSA layers is the
    /// first-token gap vs 50K. Reuse the stashed bank when the block count
    /// is unchanged; recompute and stash without `eval` on a miss. The
    /// opt-in incremental buffer (`update`) extends only the new suffix and
    /// leaves evaluation on the final consumer; it still needs full state and
    /// lifecycle qualification before becoming the default.
    static func reuseOrCompute(
        cache: CBv2Qwen4GatheredCache,
        compressRatio: Int,
        logicalTokens: Int,
        indexKeyNorm: (MLXArray) -> MLXArray,
        applyIndexRope: (MLXArray, MLXArray) -> MLXArray
    ) -> MLXArray {
        if let incremental = update(
            cache: cache,
            compressRatio: compressRatio,
            logicalTokens: logicalTokens,
            indexKeyNorm: indexKeyNorm,
            applyIndexRope: applyIndexRope)
        {
            return incremental
        }
        let completeBlocks = logicalTokens / max(compressRatio, 1)
        if completeBlocks > 0,
            cache.qwen4PooledIndexBlocks == completeBlocks,
            let reused = logicalKeys(cache.qwen4PooledIndexKeys, blocks: completeBlocks)
        {
            return reused
        }
        guard let indexKeys = Qwen4ExpIndexCapacity.logicalTokens(
            cache.qwen4IndexKeys, length: logicalTokens),
            let indexPositionIds = Qwen4ExpIndexCapacity.logicalPositions(
                cache.qwen4IndexPositionIds, length: logicalTokens)
        else {
            preconditionFailure(
                "Qwen4 QSA indexer side-state shorter than the gathered KV")
        }
        let pooled = Qwen4ExpGatheredQSA.poolCompletedIndexKeys(
            indexKeys: indexKeys,
            indexPositionIds: indexPositionIds,
            compressRatio: compressRatio,
            indexKeyNorm: indexKeyNorm,
            applyIndexRope: applyIndexRope)
        cache.qwen4PooledIndexKeys = pooled
        cache.qwen4PooledIndexBlocks = completeBlocks
        return pooled
    }
}

public enum Qwen4ExpPooledIndexInvocation: Sendable {
    private static let lock = NSLock()
    private static let logger = Logger(subsystem: "darkbloom", category: "Qwen4PooledIndex")
    private static let diagnoseFirstPlan = Qwen4ExpEnvironment.snapshot[
        "DARKBLOOM_QWEN4_QSA_POOLED_INCREMENTAL_DIAGNOSTICS"] == "1"
    nonisolated(unsafe) private static var didRecord = false

    static func record(logicalTokens: Int, blocks: Int, capacity: Int) {
        guard diagnoseFirstPlan else { return }
        let first = lock.withLock {
            if didRecord { return false }
            didRecord = true
            return true
        }
        if first {
            // Graph-path evidence only; completed output/state gates are separate.
            logger.info(
                "qwen4_pooled_incremental first_planned_call=1 logical_tokens=\(logicalTokens, privacy: .public) blocks=\(blocks, privacy: .public) capacity=\(capacity, privacy: .public)")
        }
    }

    public static func snapshot() -> Bool { lock.withLock { didRecord } }
}

/// Sparse, allocation-only telemetry. Recording occurs only when the raw
/// index-key bank actually grows, never on ordinary decode appends.
public enum Qwen4ExpQSAIndexGrowthMetrics: Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var growthEvents = 0
    nonisolated(unsafe) private static var largestCapacityTokens = 0
    nonisolated(unsafe) private static var neededAtLargestCapacityTokens = 0

    public struct Snapshot: Sendable, Equatable {
        public var growthEvents: Int
        public var largestCapacityTokens: Int
        public var neededAtLargestCapacityTokens: Int

        public init(
            growthEvents: Int,
            largestCapacityTokens: Int,
            neededAtLargestCapacityTokens: Int
        ) {
            self.growthEvents = growthEvents
            self.largestCapacityTokens = largestCapacityTokens
            self.neededAtLargestCapacityTokens = neededAtLargestCapacityTokens
        }
    }

    static func record(needed: Int, capacity: Int) {
        lock.lock()
        growthEvents += 1
        if capacity > largestCapacityTokens {
            largestCapacityTokens = capacity
            neededAtLargestCapacityTokens = needed
        } else if capacity == largestCapacityTokens {
            neededAtLargestCapacityTokens = max(neededAtLargestCapacityTokens, needed)
        }
        lock.unlock()
    }

    public static func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(
            growthEvents: growthEvents,
            largestCapacityTokens: largestCapacityTokens,
            neededAtLargestCapacityTokens: neededAtLargestCapacityTokens)
    }

    public static func reset() {
        lock.lock()
        growthEvents = 0
        largestCapacityTokens = 0
        neededAtLargestCapacityTokens = 0
        lock.unlock()
    }
}

/// Fusion `QSAKVCache._ensure_indexer_capacity` for raw index keys / mRoPE
/// ids. Default is the zeros-buffer + slice-assign bank (no per-layer
/// `eval` — that sync missed 80K on pooled incremental). Concat of the
/// exact-width prefix each chunk is the listing 80K prefill tax after a
/// long 50K soak. Kill: `DARKBLOOM_QWEN4_QSA_INDEX_CAPACITY=0`.
enum Qwen4ExpIndexCapacity: Sendable {
    static let envFlag = "DARKBLOOM_QWEN4_QSA_INDEX_CAPACITY"

    static func isEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        let raw = environment[envFlag]?.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        if raw == "0" || raw == "false" || raw == "no" || raw == "off" {
            return false
        }
        return true
    }

    /// Retain geometric growth while bounding long-context slack to one
    /// additional rounded step. Pure doubling
    /// jumps from 65,536 to 131,072 slots for an 80K history even though the
    /// next 8K chunk needs only 73,728. This policy grows 64K -> 80K instead:
    /// values and logical slices are unchanged, but the raw index-key and
    /// mRoPE sidecars cannot approach 2x steady-state residency.
    static func growthCapacity(current: Int, needed: Int, step: Int) -> Int {
        let step = max(1, step)
        let needed = max(0, needed)
        if current >= needed { return current }
        let stepped = needed > Int.max - (step - 1)
            ? Int.max
            : ((needed + step - 1) / step) * step
        let geometric: Int
        if current <= 0 {
            geometric = step
        } else if current > Int.max / 2 {
            geometric = Int.max
        } else {
            geometric = current * 2
        }
        let boundedHeadroom = stepped > Int.max - step ? Int.max : stepped + step
        return max(stepped, min(geometric, boundedHeadroom))
    }

    static func logicalTokens(_ stored: MLXArray?, length: Int) -> MLXArray? {
        guard let stored, length > 0, stored.ndim == 3, stored.dim(1) >= length else {
            return nil
        }
        return stored.dim(1) == length ? stored : stored[0..., 0 ..< length, 0...]
    }

    static func logicalPositions(_ stored: MLXArray?, length: Int) -> MLXArray? {
        guard let stored, length > 0, stored.dim(-1) >= length else { return nil }
        if stored.dim(-1) == length { return stored }
        if stored.ndim == 2 {
            return stored[0..., 0 ..< length]
        }
        if stored.ndim == 3 {
            return stored[0..., 0..., 0 ..< length]
        }
        return nil
    }

    static func appendTokens(buffer: MLXArray?, offset: Int, rows: MLXArray) -> MLXArray {
        precondition(rows.ndim == 3 && offset >= 0)
        let length = rows.dim(1)
        let end = offset + length
        var bank: MLXArray
        if let buffer, buffer.dim(1) >= end {
            bank = buffer
        } else {
            let grown = growthCapacity(
                current: buffer?.dim(1) ?? 0, needed: end,
                step: Qwen4ExpPooledIndex.tokenStep)
            Qwen4ExpQSAIndexGrowthMetrics.record(needed: end, capacity: grown)
            bank = MLXArray.zeros(
                [rows.dim(0), grown, rows.dim(-1)], dtype: rows.dtype)
            if let buffer, offset > 0 {
                bank[0..., 0 ..< offset, 0...] = buffer[0..., 0 ..< offset, 0...]
            }
        }
        bank[0..., offset ..< end, 0...] = rows
        return bank
    }

    /// Bring a stored position sidecar and an incoming window to one rank.
    ///
    /// Fusion `QSAKVCache.update_index_position_ids`: a text history is
    /// rank two; the first genuine image/video window promotes the whole
    /// history to three identical planes, and later text rows (decode,
    /// verify, or a text-only prefill chunk) broadcast onto that rank-3
    /// history. Values never change, only the plane count.
    static func alignPositionRanks(
        stored: MLXArray, rows: MLXArray
    ) -> (stored: MLXArray, rows: MLXArray) {
        if stored.ndim == rows.ndim { return (stored, rows) }
        if stored.ndim == 3, rows.ndim == 2 {
            let length = rows.dim(-1)
            let promoted = broadcast(rows.reshaped([1, 1, length]), to: [3, 1, length])
            return (stored, promoted)
        }
        if stored.ndim == 2, rows.ndim == 3 {
            let tokens = stored.dim(-1)
            // Materialize the promoted history: it is written into in place by
            // the capacity bank on the next append.
            let promoted = broadcast(stored.reshaped([1, 1, tokens]), to: [3, 1, tokens]) + 0
            return (promoted, rows)
        }
        return (stored, rows)
    }

    static func appendPositions(buffer: MLXArray?, offset: Int, rows: MLXArray) -> MLXArray {
        precondition(offset >= 0)
        let length = rows.dim(-1)
        let end = offset + length
        var bank: MLXArray
        if let buffer, buffer.dim(-1) >= end {
            bank = buffer
        } else {
            let grown = growthCapacity(
                current: buffer?.dim(-1) ?? 0, needed: end,
                step: Qwen4ExpPooledIndex.tokenStep)
            if rows.ndim == 2 {
                bank = MLXArray.zeros([rows.dim(0), grown], dtype: rows.dtype)
                if let buffer, offset > 0 {
                    bank[0..., 0 ..< offset] = buffer[0..., 0 ..< offset]
                }
            } else {
                precondition(rows.ndim == 3)
                bank = MLXArray.zeros(
                    [rows.dim(0), rows.dim(1), grown], dtype: rows.dtype)
                if let buffer, offset > 0 {
                    bank[0..., 0..., 0 ..< offset] = buffer[0..., 0..., 0 ..< offset]
                }
            }
        }
        if rows.ndim == 2 {
            bank[0..., offset ..< end] = rows
        } else {
            bank[0..., 0..., offset ..< end] = rows
        }
        return bank
    }
}
