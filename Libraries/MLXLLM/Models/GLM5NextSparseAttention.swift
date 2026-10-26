// Copyright © 2026 Eigen Labs.
// Copyright 2026 the HuggingFace Team. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified in 2026: native Swift/MLX stateless pool selection and latent attention.
// Equations ported from the Apache-2.0 Transformers GLM5Next implementation:
// huggingface/transformers@f0705f991a71f05dad9569336b181119f5546f93
//   src/transformers/models/glm5_next/modeling_glm5_next.py
//   Glm5NextTextIndexer and Glm5NextTextAttention.
// Absorbed MLA / selected-latent execution reference:
// Blaizzy/mlx-vlm@4623cf9d2633c5ee2b1ac125eeae51423447c41f
//   mlx_vlm/models/glm5_next/language.py

import Foundation
import MLX

public enum GLM5NextSparseAttentionError: Error, Equatable {
    case invalidInput(String)
}

/// Unoptimized, stateless GLM5Next primitives, not a registered model or engine.
/// Callers own all projected history and must pass the history at the intended
/// committed/tentative frontier. No hidden pool cache survives a trim or rollback.
public enum GLM5NextSparseAttention {
    /// FP32 `[B,Qb,H,P]` index-score bound of one `Execution.queryBlocks` block.
    public static let scoreBlockBytes = 64 << 20

    /// Pool top-k implementation. Both return the pinned CPU-oracle membership.
    public enum SelectionStrategy: Sendable, Hashable {
        /// One GPU thread per row emulating the pinned libc++ selection.
        case reference
        /// One threadgroup per row finds the K-th score in parallel; a row
        /// whose boundary tie makes membership order-dependent reruns the
        /// reference emulation in-kernel.
        case parallelThreshold
    }

    /// Execution choices of one stateless `selectTokens` call. Model layers pass
    /// their load-time `GLM5NextExecutionPolicy.sparseSelection`. `selection`
    /// and `deviceRanges` are exact by construction; `queryBlocks` is not.
    public struct Execution: Sendable, Hashable {
        public var selection: SelectionStrategy
        /// Build position ramps with device `arange` instead of host arrays of
        /// history length. The int32 values are identical.
        public var deviceRanges: Bool
        /// Score a block of queries with one GEMM and one head reduction, and
        /// select every row of the block in one launch, instead of one graph
        /// per query. Same equations and membership rules; the FP32 GEMM and
        /// reduction shapes differ from the per-query graph, so a near-tie
        /// pool can change. Blocks hold at most `scoreBlockBytes` of scores.
        public var queryBlocks: Bool

        public init(
            selection: SelectionStrategy = .reference, deviceRanges: Bool = false,
            queryBlocks: Bool = false
        ) {
            self.selection = selection
            self.deviceRanges = deviceRanges
            self.queryBlocks = queryBlocks
        }

        public static let reference = Execution()
    }

    /// A selection made against one explicit history length. Do not mutate its
    /// arrays: valid indices are causal and unique within each query row.
    public struct Selection {
        public let indices: MLXArray
        public let historyLength: Int
        /// Budget columns past the last candidate pool. They are -1 in every
        /// row by construction, so consumers may skip them without reading
        /// indices. Every other column must still be checked for -1.
        public let staticPaddingColumns: Range<Int>

        fileprivate init(indices: MLXArray, historyLength: Int, staticPaddingColumns: Range<Int>) {
            self.indices = indices
            self.historyLength = historyLength
            self.staticPaddingColumns = staticPaddingColumns
        }

        /// The last query's row, as reused by index-shared MTP proposal steps.
        var finalQuery: Selection {
            Selection(
                indices: indices[0..., (-1)...], historyLength: historyLength,
                staticPaddingColumns: staticPaddingColumns)
        }

        /// This selection over `historyLength` rows read through `indices`
        /// instead: a paged history's selected rows gathered in selection
        /// order (`GLM5NextDSACache.selectedLatents`), so a consumer reads
        /// the same rows in the same order and skips the same columns.
        func rereading(_ indices: MLXArray, historyLength: Int) -> Selection {
            Selection(
                indices: indices, historyLength: historyLength,
                staticPaddingColumns: staticPaddingColumns)
        }
    }

    /// Select complete learned pools, then the visible incomplete pool's tail.
    ///
    /// Inputs are already projected: queries `[B,Q,H,D]`, normalized index keys
    /// and pool gates `[B,N,D]`, learned position bias `[4,D]`, and unscaled head
    /// weights `[B,Q,H]`. Keys use LayerNorm (epsilon 1e-6) upstream, not RMSNorm.
    /// Validity arrays are boolean `[B,N]` / `[B,Q]`. Query i has raw cache position
    /// `queryStart+i`; padding does not move the first real token's pool origin.
    /// `tokenBudget` is an explicit multiple of four in 4...2048; output is always
    /// `[B,Q,tokenBudget+3]`, with -1 slots. No model configuration is inferred.
    ///
    /// All pools are scored, but queries are processed separately so no
    /// `[B,Q,H,N]` score tensor is constructed. Tied membership follows the
    /// pinned official CPU selector; selected pools remain chronological.
    ///
    /// The selector breaks an exact tie at the budget boundary by its walk
    /// over the whole candidate row, so a row's choice also depends on how
    /// many pools the call holds. `selectionHistories` (prompt prefill, every
    /// key valid, `Execution.queryBlocks`): rows `[previous rowEnd, rowEnd)`
    /// choose as a call whose keys end at absolute key `keys` would, so a
    /// prefill cut unlike another keeps that cut's selections.
    public static func selectTokens(
        indexQueries: MLXArray,
        indexKeys: MLXArray,
        poolGates: MLXArray,
        poolPositionBias: MLXArray,
        headWeights: MLXArray,
        keyValidity: MLXArray,
        queryValidity: MLXArray,
        queryStart: Int,
        tokenBudget: Int,
        execution: Execution = .reference,
        allowsDenseShortcut: Bool = true,
        scoreBlockBytes: Int = GLM5NextSparseAttention.scoreBlockBytes,
        nativePrefillScores: Bool = false,
        afterScores: ((MLXArray) -> MLXArray)? = nil,
        selectionHistories: [(rowEnd: Int, keys: Int)]? = nil
    ) throws -> Selection {
        try validateSelectionInputs(
            indexQueries: indexQueries, indexKeys: indexKeys, poolGates: poolGates,
            poolPositionBias: poolPositionBias, headWeights: headWeights,
            keyValidity: keyValidity, queryValidity: queryValidity, queryStart: queryStart,
            tokenBudget: tokenBudget)
        let batch = indexQueries.dim(0)
        let queries = indexQueries.dim(1)
        let heads = indexQueries.dim(2)
        let dimension = indexQueries.dim(3)
        let history = indexKeys.dim(1)
        let poolCount = (history + 3) / 4
        func ramp(_ count: Int) -> MLXArray {
            execution.deviceRanges ? MLXArray.arange(count, dtype: .int32) : MLXArray(0 ..< count)
        }
        let layout = poolLayout(keyValidity: keyValidity, batch: batch, ramp: ramp)
        let (firstKey, poolIndices, completePools) = (
            layout.firstKey, layout.poolIndices, layout.completePools
        )
        // When the budget covers every pool, the selector returns every
        // retained candidate regardless of scores, so the selected set is
        // exactly the eligible complete pools. Skip scoring and build all
        // queries at once; the indices are identical to the per-query path.
        if allowsDenseShortcut, tokenBudget / 4 >= poolCount {
            return denseCausalSelection(
                poolIndices: poolIndices, completePools: completePools,
                keyValidity: keyValidity, queryValidity: queryValidity, firstKey: firstKey,
                queryStart: queryStart, queries: queries, tokenBudget: tokenBudget,
                ramp: ramp)
        }
        let pooledKeys = Self.pooledKeys(
            layout, indexKeys: indexKeys, poolGates: poolGates, poolPositionBias: poolPositionBias,
            dimension: dimension)
        let poolEnds = poolIndices[.ellipsis, 3]
        if execution.queryBlocks {
            return try blockedSelection(
                indexQueries: indexQueries, pooledKeys: pooledKeys, headWeights: headWeights,
                poolIndices: poolIndices, completePools: completePools, keyValidity: keyValidity,
                queryValidity: queryValidity, firstKey: firstKey, queryStart: queryStart,
                tokenBudget: tokenBudget, execution: execution, scoreBlockBytes: scoreBlockBytes,
                nativeScores: nativePrefillScores, afterScores: afterScores,
                selectionHistories: selectionHistories, ramp: ramp)
        }
        let positions = ramp(history).reshaped([1, history])
        let tailOffsets = MLXArray(0 ..< 3).reshaped([1, 3])
        let selectedPoolCount = min(tokenBudget / 4, poolCount)
        let indexScale = 1 / sqrt(Float(dimension))
        let headScale = 1 / sqrt(Float(heads))
        var rows = [MLXArray]()
        rows.reserveCapacity(queries)

        for query in 0 ..< queries {
            let position = queryStart + query
            let eligible = completePools .&& (poolEnds .<= position)
            let q = indexQueries[0..., query].asType(.float32)
            let scoresPerHead = maximum(
                matmul(q, pooledKeys.asType(.float32).transposed(0, 2, 1)) * indexScale, 0)
            let weights = headWeights[0..., query].asType(.float32) * headScale
            let scores = (scoresPerHead * weights.expandedDimensions(axis: -1)).sum(axis: 1)
            // Drop only columns with no complete pool in ANY batch row. Future
            // complete pools remain in the selector's candidate axis, masked
            // with the reference's finite minimum, because they affect ties.
            // The budget gives the same layout as `selectedPoolCount` (both are
            // clamped to the pool count) and keeps each selector pipeline's
            // only count specialization fixed while history grows.
            let chosen = try GLM5NextSparseSelection.select(
                scores: MLX.where(eligible, scores, -Float.greatestFiniteMagnitude),
                completePools: completePools, requestedCount: tokenBudget / 4,
                strategy: execution.selection)
            let selected = sorted(chosen, axis: 1)
            let safeSelected = clip(selected, min: 0, max: poolCount - 1)
            let selectedValid =
                (selected .< poolCount) .&& takeAlong(eligible, safeSelected, axis: 1)
            let members = takeAlong(poolIndices, safeSelected.expandedDimensions(axis: -1), axis: 1)
            var tokens = MLX.where(selectedValid.expandedDimensions(axis: -1), members, -1)
                .reshaped([batch, selectedPoolCount * 4])
            if tokens.dim(1) < tokenBudget {
                tokens = concatenated(
                    [
                        tokens,
                        MLXArray.zeros([batch, tokenBudget - tokens.dim(1)], dtype: .int32) - 1,
                    ], axis: 1)
            }

            let visible = keyValidity .&& (positions .<= position)
            let visibleCount = visible.asType(.int32).sum(axis: 1)
            let tailCount = visibleCount % 4
            let tail =
                (firstKey + visibleCount - tailCount).expandedDimensions(axis: -1) + tailOffsets
            let tailVisible = takeAlong(visible, clip(tail, min: 0, max: history - 1), axis: 1)
            let tailValid =
                (tailOffsets .< tailCount.expandedDimensions(axis: -1))
                .&& (tail .< history) .&& tailVisible
            // Normally disjoint by construction. This also preserves HF's set
            // semantics for unusual masks with holes, whose tail can overlap a
            // previously selected complete pool. Cost is at most 3*2048/row.
            let alreadySelected =
                (tail.expandedDimensions(axis: -1)
                .== tokens.expandedDimensions(axis: 1)).any(axis: -1)
            let uniqueTailValid = tailValid .&& logicalNot(alreadySelected)
            tokens = concatenated([tokens, MLX.where(uniqueTailValid, tail, -1)], axis: 1)
            rows.append(
                MLX.where(queryValidity[0..., query].expandedDimensions(axis: -1), tokens, -1))
        }
        return Selection(
            indices: stacked(rows, axis: 1).asType(.int32), historyLength: history,
            staticPaddingColumns: selectedPoolCount * 4 ..< tokenBudget)
    }

    /// `selectTokens` for steps of at most `GLM5NextSparseDecodeKernel
    /// .maximumQueries` queries in three launches: in-kernel pooling and
    /// scoring, the `strategy` selector over the batch-retained pools, and the
    /// member/tail expansion. Same equations, eligibility and tie membership;
    /// the FP32 score reduction order differs from both score graphs, so a
    /// near-tie pool can change. `poolingType` is the dtype keys and gates
    /// are pooled in (the reference casts both to it); it defaults to the key
    /// dtype. Default GPU stream only; elsewhere it throws.
    public static func selectDecodeTokens(
        indexQueries: MLXArray,
        indexKeys: MLXArray,
        poolGates: MLXArray,
        poolPositionBias: MLXArray,
        headWeights: MLXArray,
        keyValidity: MLXArray,
        queryValidity: MLXArray,
        queryStart: Int,
        tokenBudget: Int,
        strategy: SelectionStrategy = .reference,
        poolingType: DType? = nil
    ) throws -> Selection {
        try selectDecodeTokens(
            indexQueries: indexQueries, indexKeys: indexKeys, poolGates: poolGates,
            poolPositionBias: poolPositionBias, headWeights: headWeights,
            keyValidity: keyValidity, queryValidity: queryValidity, queryStart: queryStart,
            tokenBudget: tokenBudget, strategy: strategy, poolingType: poolingType,
            layout: .serving)
    }

    /// `selectDecodeTokens` with the decode kernels' launch `layout` (every
    /// layout returns the same selection).
    static func selectDecodeTokens(
        indexQueries: MLXArray,
        indexKeys: MLXArray,
        poolGates: MLXArray,
        poolPositionBias: MLXArray,
        headWeights: MLXArray,
        keyValidity: MLXArray,
        queryValidity: MLXArray,
        queryStart: Int,
        tokenBudget: Int,
        strategy: SelectionStrategy,
        poolingType: DType?,
        layout: GLM5NextSparseDecodeKernel.Layout
    ) throws -> Selection {
        try validateSelectionInputs(
            indexQueries: indexQueries, indexKeys: indexKeys, poolGates: poolGates,
            poolPositionBias: poolPositionBias, headWeights: headWeights,
            keyValidity: keyValidity, queryValidity: queryValidity, queryStart: queryStart,
            tokenBudget: tokenBudget)
        let batch = indexQueries.dim(0)
        let queries = indexQueries.dim(1)
        let history = indexKeys.dim(1)
        let poolCount = (history + 3) / 4
        let pooled = try GLM5NextSparseDecodeKernel.poolScores(
            indexQueries: indexQueries, headWeights: headWeights, indexKeys: indexKeys,
            poolGates: poolGates, poolPositionBias: poolPositionBias, keyValidity: keyValidity,
            queryStart: queryStart, poolingType: poolingType ?? indexKeys.dtype, layout: layout)
        // Pools complete in ANY batch row stay selector candidates.
        let retained = batch == 1 ? pooled.complete : pooled.complete.any(axis: 0, keepDims: true)
        let chosen = try GLM5NextSparseSelection.select(
            scores: pooled.scores.reshaped([batch * queries, poolCount]), completePools: retained,
            requestedCount: tokenBudget / 4, strategy: strategy, threads: layout.selectionThreads)
        let indices = try GLM5NextSparseDecodeKernel.expand(
            chosen: chosen, keyValidity: keyValidity, queryValidity: queryValidity,
            queryStart: queryStart, tokenBudget: tokenBudget)
        let selectedPoolCount = min(tokenBudget / 4, poolCount)
        return Selection(
            indices: indices, historyLength: history,
            staticPaddingColumns: selectedPoolCount * 4 ..< tokenBudget)
    }

    private static func validateSelectionInputs(
        indexQueries: MLXArray, indexKeys: MLXArray, poolGates: MLXArray,
        poolPositionBias: MLXArray, headWeights: MLXArray, keyValidity: MLXArray,
        queryValidity: MLXArray, queryStart: Int, tokenBudget: Int
    ) throws {
        try require(indexQueries.ndim == 4 && indexKeys.ndim == 3, "index query/key ranks")
        let batch = indexQueries.dim(0)
        let queries = indexQueries.dim(1)
        let heads = indexQueries.dim(2)
        let dimension = indexQueries.dim(3)
        let history = indexKeys.dim(1)
        try require(batch > 0 && queries > 0 && heads > 0 && dimension > 0, "empty index geometry")
        try require(history > 0 && history <= (Int(Int32.max) - 3) / 2, "index history length")
        try require(indexKeys.shape == [batch, history, dimension], "index key shape")
        try require(poolGates.shape == indexKeys.shape, "pool gate shape")
        try require(poolPositionBias.shape == [4, dimension], "pool position bias must be [4,D]")
        try require(headWeights.shape == [batch, queries, heads], "head weight shape")
        try require(
            keyValidity.shape == [batch, history] && keyValidity.dtype == .bool, "key validity")
        try require(
            queryValidity.shape == [batch, queries] && queryValidity.dtype == .bool,
            "query validity")
        try require(
            queryStart >= 0 && queries <= history && queryStart <= history - queries,
            "query positions")
        try require((4 ... 2048).contains(tokenBudget) && tokenBudget % 4 == 0, "raw token budget")
        for array in [indexQueries, indexKeys, poolGates, poolPositionBias, headWeights] {
            try require(isFloating(array), "index inputs must be float16, bfloat16 or float32")
        }
    }

    /// Every-pool selection for all queries at once. Mirrors the per-query path:
    /// the selector returns the batch-retained pools in ascending order, padded
    /// with the pool count, and each row keeps its eligible pools plus the tail.
    private static func denseCausalSelection(
        poolIndices: MLXArray, completePools: MLXArray, keyValidity: MLXArray,
        queryValidity: MLXArray, firstKey: MLXArray, queryStart: Int, queries: Int,
        tokenBudget: Int, ramp: (Int) -> MLXArray
    ) -> Selection {
        let batch = poolIndices.dim(0)
        let poolCount = poolIndices.dim(1)
        let history = keyValidity.dim(1)
        let ids = ramp(poolCount).asType(.int32)
        let retained = completePools.any(axis: 0)
        let order = argSort(MLX.where(retained, ids, ids + Int32(poolCount)), axis: 0)
            .asType(.int32)
        let selected = MLX.where(take(retained, order, axis: 0), order, Int32(poolCount))
        let safeSelected = clip(selected, min: 0, max: poolCount - 1)
        let positions = (ramp(queries).asType(.int32) + Int32(queryStart)).reshaped([1, queries, 1])
        let poolEnds = take(poolIndices[.ellipsis, 3], safeSelected, axis: 1)
        let eligible =
            take(completePools, safeSelected, axis: 1).expandedDimensions(axis: 1)
            .&& (poolEnds.expandedDimensions(axis: 1) .<= positions)
            .&& (selected .< Int32(poolCount)).reshaped([1, 1, poolCount])
        let members = take(poolIndices, safeSelected, axis: 1)
        var tokens = MLX.where(
            eligible.expandedDimensions(axis: -1), members.expandedDimensions(axis: 1),
            Int32(-1)
        ).reshaped([batch, queries, poolCount * 4])
        if poolCount * 4 < tokenBudget {
            tokens = concatenated(
                [
                    tokens,
                    MLXArray.zeros([batch, queries, tokenBudget - poolCount * 4], dtype: .int32)
                        - 1,
                ], axis: 2)
        }
        return Selection(
            indices: appendingTail(
                to: tokens, keyValidity: keyValidity, queryValidity: queryValidity,
                firstKey: firstKey, queryStart: queryStart, ramp: ramp),
            historyLength: history, staticPaddingColumns: poolCount * 4 ..< tokenBudget)
    }

    /// A selection call's pools of four keys from the first valid key: their
    /// raw positions `[B,P,4]`, which positions exist and are valid, and the
    /// complete pools `[B,P]` (`selectTokens`).
    private struct PoolLayout {
        let firstKey: MLXArray
        let poolIndices: MLXArray
        let safe: MLXArray
        let groupedValid: MLXArray
        let completePools: MLXArray
    }

    private static func poolLayout(keyValidity: MLXArray, batch: Int, ramp: (Int) -> MLXArray)
        -> PoolLayout
    {
        let history = keyValidity.dim(1)
        let poolCount = (history + 3) / 4
        let firstKey = MLX.where(
            keyValidity.any(axis: 1),
            keyValidity.asType(.int32).argMax(axis: 1).asType(.int32), history)
        let poolOffsets = ramp(poolCount * 4).reshaped([1, poolCount, 4])
        let poolIndices = firstKey.reshaped([batch, 1, 1]) + poolOffsets
        let safe = clip(poolIndices, min: 0, max: history - 1).reshaped([batch, poolCount * 4])
        let groupedValid =
            takeAlong(keyValidity, safe, axis: 1)
            .reshaped([batch, poolCount, 4]) .&& (poolIndices .< history)
        return PoolLayout(
            firstKey: firstKey, poolIndices: poolIndices, safe: safe, groupedValid: groupedValid,
            completePools: groupedValid.all(axis: 2))
    }

    /// Each pool's key: its valid keys weighted by the softmax of their gates
    /// plus the learned position bias `[B,P,D]`.
    private static func pooledKeys(
        _ layout: PoolLayout, indexKeys: MLXArray, poolGates: MLXArray, poolPositionBias: MLXArray,
        dimension: Int
    ) -> MLXArray {
        let batch = layout.poolIndices.dim(0)
        let poolCount = layout.poolIndices.dim(1)
        let (safe, groupedValid) = (layout.safe, layout.groupedValid)
        let groupedKeys = takeAlong(indexKeys, safe.expandedDimensions(axis: -1), axis: 1)
            .reshaped([batch, poolCount, 4, dimension])
        let groupedGates = takeAlong(poolGates, safe.expandedDimensions(axis: -1), axis: 1)
            .reshaped([batch, poolCount, 4, dimension])
        let gateLogits = groupedGates.asType(.float32) + poolPositionBias.asType(.float32)
        let probabilities = softmax(
            MLX.where(groupedValid.expandedDimensions(axis: -1), gateLogits, -Float.infinity),
            axis: 2, precise: true)
        // Fully padded pools have an all--inf softmax. Select zero rather than
        // multiplying NaN by zero; partial pools are never eligible for top-k.
        let poolWeights = MLX.where(groupedValid.expandedDimensions(axis: -1), probabilities, 0)
            .asType(indexKeys.dtype)
        return
            (poolWeights
            * MLX.where(
                groupedValid.expandedDimensions(axis: -1), groupedKeys, 0)).sum(axis: 2)
    }

    /// `selectTokens` with its rows' scoring and top-k shared between ranks
    /// (`shares` of them, this one `share`): the call's rows are cut into
    /// near-equal consecutive ranges, this share scores and selects its own
    /// range, and every share expands every row's chosen pools. `partial` is
    /// `[B,Q,P']` float32 (P' the selected pool count): this share's rows'
    /// chosen pool IDs, chronological with the pool count as padding, zero in
    /// the other rows; `finish` takes the shares' sum (every row's IDs; exact
    /// in float32 for fewer than 2^24 pools) and returns `selectTokens`'
    /// selection bit for bit: each row is scored by the fused index-score
    /// kernel and selected on its own, whatever rows share its call. Nil when
    /// the call would not run blocked native scores (the dense shortcut, a
    /// per-query or unfused call): run `selectTokens` instead.
    public struct SplitSelection {
        public let partial: MLXArray
        let finishing: (MLXArray) throws -> Selection
        public func finish(_ sum: MLXArray) throws -> Selection { try finishing(sum) }
    }

    public static func selectTokensSplit(
        indexQueries: MLXArray,
        indexKeys: MLXArray,
        poolGates: MLXArray,
        poolPositionBias: MLXArray,
        headWeights: MLXArray,
        keyValidity: MLXArray,
        queryValidity: MLXArray,
        queryStart: Int,
        tokenBudget: Int,
        execution: Execution,
        share: Int,
        shares: Int,
        scoreBlockBytes: Int = GLM5NextSparseAttention.scoreBlockBytes,
        nativePrefillScores: Bool,
        afterScores: ((MLXArray) -> MLXArray)? = nil,
        selectionHistories: [(rowEnd: Int, keys: Int)]? = nil
    ) throws -> SplitSelection? {
        try validateSelectionInputs(
            indexQueries: indexQueries, indexKeys: indexKeys, poolGates: poolGates,
            poolPositionBias: poolPositionBias, headWeights: headWeights,
            keyValidity: keyValidity, queryValidity: queryValidity, queryStart: queryStart,
            tokenBudget: tokenBudget)
        guard shares > 1, (0 ..< shares).contains(share) else { return nil }
        let batch = indexQueries.dim(0)
        let queries = indexQueries.dim(1)
        let dimension = indexQueries.dim(3)
        let history = indexKeys.dim(1)
        let poolCount = (history + 3) / 4
        guard execution.queryBlocks, nativePrefillScores, tokenBudget / 4 < poolCount,
            poolCount < 1 << 24
        else {
            return nil
        }
        func ramp(_ count: Int) -> MLXArray {
            execution.deviceRanges ? MLXArray.arange(count, dtype: .int32) : MLXArray(0 ..< count)
        }
        let layout = poolLayout(keyValidity: keyValidity, batch: batch, ramp: ramp)
        let pooledKeys = Self.pooledKeys(
            layout, indexKeys: indexKeys, poolGates: poolGates, poolPositionBias: poolPositionBias,
            dimension: dimension)
        let call = BlockedCall(
            indexQueries: indexQueries, pooledKeys: pooledKeys, poolIndices: layout.poolIndices,
            completePools: layout.completePools, keyValidity: keyValidity, tokenBudget: tokenBudget,
            nativeScores: true)
        guard call.fused else { return nil }
        let own = (share * queries / shares) ..< ((share + 1) * queries / shares)
        let block = call.block(queries: own.count, scoreBlockBytes: scoreBlockBytes)
        var chosen: [MLXArray] = []
        var afterScores = afterScores
        for first in stride(from: own.lowerBound, to: own.upperBound, by: block) {
            chosen.append(
                try chooseBlock(
                    call, rows: first ..< min(first + block, own.upperBound),
                    indexQueries: indexQueries,
                    headWeights: headWeights, pooledKeys: pooledKeys,
                    completePools: layout.completePools,
                    poolIndices: layout.poolIndices, queryStart: queryStart,
                    tokenBudget: tokenBudget,
                    execution: execution, afterScores: &afterScores,
                    selectionHistories: selectionHistories,
                    ramp: ramp
                ).selected.asType(.float32))
        }
        let width = call.selectedPoolCount
        var columns: [MLXArray] = []
        if own.lowerBound > 0 {
            columns.append(MLXArray.zeros([batch, own.lowerBound, width], dtype: .float32))
        }
        columns += chosen
        if own.upperBound < queries {
            columns.append(
                MLXArray.zeros([batch, queries - own.upperBound, width], dtype: .float32))
        }
        let partial = columns.count == 1 ? columns[0] : concatenated(columns, axis: 1)
        let wholeBlock = call.block(queries: queries, scoreBlockBytes: scoreBlockBytes)
        return SplitSelection(partial: partial) { sum in
            guard sum.shape == [batch, queries, width] else {
                throw GLM5NextSparseSelection.Failure.invalidTensor
            }
            let selected = sum.asType(.int32)
            var blocks: [MLXArray] = []
            for first in stride(from: 0, to: queries, by: wholeBlock) {
                let range = first ..< min(first + wholeBlock, queries)
                blocks.append(
                    expandBlock(
                        call, selected: selected[0..., range], rows: range,
                        poolIndices: layout.poolIndices,
                        keyValidity: keyValidity, queryValidity: queryValidity,
                        firstKey: layout.firstKey,
                        queryStart: queryStart, tokenBudget: tokenBudget, ramp: ramp))
            }
            return Selection(
                indices: blocks.count == 1 ? blocks[0] : concatenated(blocks, axis: 1),
                historyLength: history,
                staticPaddingColumns: width * 4 ..< tokenBudget)
        }
    }

    /// One blocked selection call's shapes and per-call arrays.
    private struct BlockedCall {
        let batch: Int, heads: Int, dimension: Int, poolCount: Int, history: Int,
            selectedPoolCount: Int
        let fused: Bool
        let keys: MLXArray
        let retained: MLXArray
        let poolEnds: MLXArray
        let complete: MLXArray
        let indexScale: Float, headScale: Float

        init(
            indexQueries: MLXArray, pooledKeys: MLXArray, poolIndices: MLXArray,
            completePools: MLXArray,
            keyValidity: MLXArray, tokenBudget: Int, nativeScores: Bool
        ) {
            batch = indexQueries.dim(0)
            heads = indexQueries.dim(2)
            dimension = indexQueries.dim(3)
            poolCount = poolIndices.dim(1)
            history = keyValidity.dim(1)
            selectedPoolCount = min(tokenBudget / 4, poolCount)
            indexScale = 1 / sqrt(Float(dimension))
            headScale = 1 / sqrt(Float(heads))
            keys = pooledKeys.asType(.float32).transposed(0, 2, 1)
            retained = completePools.any(axis: 0, keepDims: true)
            poolEnds = poolIndices[.ellipsis, 3].expandedDimensions(axis: 1)
            complete = completePools.expandedDimensions(axis: 1)
            fused =
                nativeScores
                && GLM5NextPrefillIndexScoreKernel.supports(
                    indexQueries: indexQueries, pooledKeys: pooledKeys)
        }

        /// Rows per score block for a call of `queries` rows.
        func block(queries: Int, scoreBlockBytes: Int) -> Int {
            let blockHeads = fused ? 1 : heads
            return max(
                1, min(queries, scoreBlockBytes / max(1, batch * blockHeads * poolCount * 4)))
        }

        /// Pools rows `range` (positions from `queryStart + range.lowerBound`) may select.
        func eligible(rows range: Range<Int>, queryStart: Int, ramp: (Int) -> MLXArray) -> MLXArray
        {
            let positions =
                (ramp(range.count).asType(.int32) + Int32(queryStart + range.lowerBound))
                .reshaped([1, range.count, 1])
            return complete .&& (poolEnds .<= positions)
        }
    }

    /// One score block's chosen pools `[B,Qb,P']`, chronological, the pool
    /// count as padding, and the pools its rows may select.
    private static func chooseBlock(
        _ call: BlockedCall, rows range: Range<Int>, indexQueries: MLXArray, headWeights: MLXArray,
        pooledKeys: MLXArray, completePools: MLXArray, poolIndices: MLXArray, queryStart: Int,
        tokenBudget: Int, execution: Execution, afterScores: inout ((MLXArray) -> MLXArray)?,
        selectionHistories: [(rowEnd: Int, keys: Int)]?, ramp: (Int) -> MLXArray
    ) throws -> (selected: MLXArray, eligible: MLXArray) {
        let (batch, heads, dimension, poolCount) = (
            call.batch, call.heads, call.dimension, call.poolCount
        )
        let first = range.lowerBound
        let count = range.count
        let eligible = call.eligible(rows: range, queryStart: queryStart, ramp: ramp)
        var masked: MLXArray
        if call.fused {
            masked = try GLM5NextPrefillIndexScoreKernel.scores(
                indexQueries: indexQueries[0..., range], headWeights: headWeights[0..., range],
                pooledKeys: pooledKeys, completePools: completePools,
                poolEnds: poolIndices[.ellipsis, 3], queryStart: queryStart + first)
        } else {
            let q = indexQueries[0..., range].asType(.float32).reshaped(
                [batch, count * heads, dimension])
            let scoresPerHead = maximum(matmul(q, call.keys) * call.indexScale, 0)
                .reshaped([batch, count, heads, poolCount])
            let weights = headWeights[0..., range].asType(.float32) * call.headScale
            let scores = (scoresPerHead * weights.expandedDimensions(axis: -1)).sum(axis: 2)
            masked = MLX.where(eligible, scores, -Float.greatestFiniteMagnitude)
        }
        if let hook = afterScores {
            masked = hook(masked)
            afterScores = nil
        }
        let chosen: MLXArray
        if let histories = selectionHistories {
            chosen = try pinnedSelection(
                masked, rows: range, histories: histories, retained: call.retained,
                history: call.history, poolCount: poolCount,
                selectedPoolCount: call.selectedPoolCount,
                tokenBudget: tokenBudget, execution: execution, ramp: ramp)
        } else {
            chosen = try GLM5NextSparseSelection.select(
                scores: masked.reshaped([batch * count, poolCount]),
                completePools: call.retained, requestedCount: tokenBudget / 4,
                strategy: execution.selection)
        }
        return (sorted(chosen, axis: 1).reshaped([batch, count, call.selectedPoolCount]), eligible)
    }

    /// One block's tokens `[B,Qb,tokenBudget+3]` from its rows' chosen pools
    /// `selected` (`chooseBlock`): each valid pool's four keys in order, -1
    /// for padding and pools a row may not select, then the visible tail.
    private static func expandBlock(
        _ call: BlockedCall, selected: MLXArray, rows range: Range<Int>, poolIndices: MLXArray,
        keyValidity: MLXArray, queryValidity: MLXArray, firstKey: MLXArray, queryStart: Int,
        tokenBudget: Int, ramp: (Int) -> MLXArray, eligible: MLXArray? = nil
    ) -> MLXArray {
        let (batch, poolCount, selectedPoolCount) = (
            call.batch, call.poolCount, call.selectedPoolCount
        )
        let count = range.count
        let eligible = eligible ?? call.eligible(rows: range, queryStart: queryStart, ramp: ramp)
        let safeSelected = clip(selected, min: 0, max: poolCount - 1)
        let selectedValid =
            (selected .< poolCount) .&& takeAlong(eligible, safeSelected, axis: 2)
        let members = takeAlong(
            poolIndices, safeSelected.reshaped([batch, count * selectedPoolCount, 1]),
            axis: 1
        ).reshaped([batch, count, selectedPoolCount, 4])
        var tokens = MLX.where(selectedValid.expandedDimensions(axis: -1), members, -1)
            .reshaped([batch, count, selectedPoolCount * 4])
        if selectedPoolCount * 4 < tokenBudget {
            tokens = concatenated(
                [
                    tokens,
                    MLXArray.zeros(
                        [batch, count, tokenBudget - selectedPoolCount * 4], dtype: .int32)
                        - 1,
                ], axis: 2)
        }
        return appendingTail(
            to: tokens, keyValidity: keyValidity,
            queryValidity: queryValidity[0..., range], firstKey: firstKey,
            queryStart: queryStart + range.lowerBound, ramp: ramp)
    }

    /// Per-query `selectTokens` semantics for blocks of queries: one score GEMM,
    /// one selector launch over every row of the block (pools retained in ANY
    /// batch row stay candidates, as in the per-query call), then the same
    /// chronological members, padding and tail. `nativeScores` writes each
    /// block's masked `[B,Qb,P]` scores in one `GLM5NextPrefillIndexScoreKernel`
    /// launch where it can run (no `[B,Qb,H,P]` per-head scores; blocks are
    /// then bounded by the head-summed scores). `afterScores` receives the
    /// first block's masked scores as soon as they are built and returns the
    /// array the block's selection reads instead: the same values, or the
    /// same array with ordering edges (`depends`) added.
    private static func blockedSelection(
        indexQueries: MLXArray, pooledKeys: MLXArray, headWeights: MLXArray,
        poolIndices: MLXArray, completePools: MLXArray, keyValidity: MLXArray,
        queryValidity: MLXArray, firstKey: MLXArray, queryStart: Int, tokenBudget: Int,
        execution: Execution, scoreBlockBytes: Int, nativeScores: Bool,
        afterScores: ((MLXArray) -> MLXArray)? = nil,
        selectionHistories: [(rowEnd: Int, keys: Int)]? = nil, ramp: (Int) -> MLXArray
    ) throws -> Selection {
        let queries = indexQueries.dim(1)
        let call = BlockedCall(
            indexQueries: indexQueries, pooledKeys: pooledKeys, poolIndices: poolIndices,
            completePools: completePools, keyValidity: keyValidity, tokenBudget: tokenBudget,
            nativeScores: nativeScores)
        let block = call.block(queries: queries, scoreBlockBytes: scoreBlockBytes)
        var blocks: [MLXArray] = []
        var afterScores = afterScores
        for first in stride(from: 0, to: queries, by: block) {
            let range = first ..< min(first + block, queries)
            let chosen = try chooseBlock(
                call, rows: range, indexQueries: indexQueries, headWeights: headWeights,
                pooledKeys: pooledKeys, completePools: completePools, poolIndices: poolIndices,
                queryStart: queryStart, tokenBudget: tokenBudget, execution: execution,
                afterScores: &afterScores, selectionHistories: selectionHistories, ramp: ramp)
            blocks.append(
                expandBlock(
                    call, selected: chosen.selected, rows: range, poolIndices: poolIndices,
                    keyValidity: keyValidity, queryValidity: queryValidity, firstKey: firstKey,
                    queryStart: queryStart, tokenBudget: tokenBudget, ramp: ramp,
                    eligible: chosen.eligible))
        }
        return Selection(
            indices: blocks.count == 1 ? blocks[0] : concatenated(blocks, axis: 1),
            historyLength: call.history,
            staticPaddingColumns: call.selectedPoolCount * 4 ..< tokenBudget)
    }

    /// The selector over one score block's rows `rows` (`masked`, `[B,Qb,P]`)
    /// with each row's candidates as its selection history has them
    /// (`histories`: rows before `rowEnd` choose among the pools of a step
    /// whose keys end at absolute key `keys`). The selector resolves an exact
    /// tie at the budget boundary by its walk over the whole candidate row,
    /// so its choice depends on the row's width and candidate set, not only
    /// on the scores; a row scored in a bigger or smaller call than its
    /// history's step gets that step's width (scores past this call masked,
    /// as they are in that step for a row before them) and candidate set,
    /// and so that step's choice. Prompt prefill only: the first key is key
    /// 0 (pool `p` holds keys `4p ..< 4p+4`). `retained` is this call's
    /// `[1,P]` candidate set. Returns `[B·Qb, selectedPoolCount]` pool IDs
    /// in chronological order, `poolCount` as padding.
    private static func pinnedSelection(
        _ masked: MLXArray, rows: Range<Int>, histories: [(rowEnd: Int, keys: Int)],
        retained: MLXArray, history: Int, poolCount: Int, selectedPoolCount: Int,
        tokenBudget: Int, execution: Execution, ramp: (Int) -> MLXArray
    ) throws -> MLXArray {
        let batch = masked.dim(0)
        let ends = ramp(poolCount).asType(.int32) * 4 + 3
        // This call's candidates, its partial last pool counted as the
        // bigger step would see it (complete when its keys exist there).
        let callCandidates =
            retained.reshaped([1, poolCount])
            .|| (ends .>= Int32(history)).reshaped([1, poolCount])
        var pieces: [MLXArray] = []
        var row = rows.lowerBound
        for piece in histories where piece.rowEnd > row && row < rows.upperBound {
            let end = min(piece.rowEnd, rows.upperBound)
            let pools = (piece.keys + 3) / 4
            let width = min(pools, poolCount)
            var scores = masked[0..., (row - rows.lowerBound) ..< (end - rows.lowerBound), ..<width]
            var candidates = callCandidates[0..., ..<width]
            if pools > width {
                scores = concatenated(
                    [
                        scores,
                        MLXArray.full(
                            [batch, end - row, pools - width],
                            values: MLXArray(-Float.greatestFiniteMagnitude)),
                    ], axis: 2)
                candidates = concatenated(
                    [candidates, MLXArray.ones([1, pools - width], dtype: .bool)], axis: 1)
            }
            let own = (ramp(pools).asType(.int32) * 4 + 3) .< Int32(piece.keys)
            var picked = try GLM5NextSparseSelection.select(
                scores: scores.reshaped([batch * (end - row), pools]),
                completePools: candidates .&& own.reshaped([1, pools]),
                requestedCount: tokenBudget / 4, strategy: execution.selection)
            // That step's padding and pools past this call read as padding
            // here; every eligible pool precedes them.
            picked = sorted(MLX.where(picked .>= Int32(width), Int32(poolCount), picked), axis: 1)
            if picked.dim(1) < selectedPoolCount {
                picked = concatenated(
                    [
                        picked,
                        MLXArray.full(
                            [picked.dim(0), selectedPoolCount - picked.dim(1)],
                            values: MLXArray(Int32(poolCount))),
                    ], axis: 1)
            }
            pieces.append(
                picked[0..., ..<selectedPoolCount].reshaped([batch, end - row, selectedPoolCount]))
            row = end
        }
        guard row == rows.upperBound else { throw GLM5NextSparseSelection.Failure.invalidGeometry }
        let joined = pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 1)
        return joined.reshaped([batch * rows.count, selectedPoolCount])
    }

    /// Appends each query's visible incomplete-pool tail to its `[B,Q,W]`
    /// tokens, skipping tail tokens already selected, and clears padded
    /// queries. Query i sits at raw position `queryStart + i`.
    private static func appendingTail(
        to tokens: MLXArray, keyValidity: MLXArray, queryValidity: MLXArray, firstKey: MLXArray,
        queryStart: Int, ramp: (Int) -> MLXArray
    ) -> MLXArray {
        let batch = tokens.dim(0)
        let queries = tokens.dim(1)
        let history = keyValidity.dim(1)
        let queryPositions = ramp(queries).asType(.int32) + Int32(queryStart)
        let positions = queryPositions.reshaped([1, queries, 1])
        let visibleCount = take(
            cumsum(keyValidity.asType(.int32), axis: 1), queryPositions, axis: 1)
        let tailCount = visibleCount % 4
        let tailOffsets = MLXArray(0 ..< 3).asType(.int32).reshaped([1, 1, 3])
        let tail =
            (firstKey.reshaped([batch, 1]) + visibleCount - tailCount).expandedDimensions(
                axis: -1) + tailOffsets
        let clippedTail = clip(tail, min: 0, max: history - 1)
        let tailVisible =
            takeAlong(keyValidity, clippedTail.reshaped([batch, queries * 3]), axis: 1)
            .reshaped([batch, queries, 3]) .&& (clippedTail .<= positions)
        let tailValid =
            (tailOffsets .< tailCount.expandedDimensions(axis: -1)) .&& (tail .< history)
            .&& tailVisible
        let alreadySelected =
            (tail.expandedDimensions(axis: -1) .== tokens.expandedDimensions(axis: 2)).any(
                axis: -1)
        let result = concatenated(
            [tokens, MLX.where(tailValid .&& logicalNot(alreadySelected), tail, Int32(-1))],
            axis: 2)
        return MLX.where(queryValidity.expandedDimensions(axis: -1), result, Int32(-1))
            .asType(.int32)
    }

    /// Absorbed, NoPE MLA over only the selected latents, one query at a time.
    /// Queries are projected `[B,Q,H,Dq]`, normalized latents `[B,N,C]`, and
    /// key/value projection weights `[H,Dq,C]` / `[H,Dv,C]` (the slices of kv_b).
    /// The attention scale is derived from Dq, NEVER C: production Dq=256 gives
    /// 1/16. Output is `[B,Q,H,Dv]`, before o_proj, in the query dtype.
    ///
    /// FP32 absorbed algebra is mathematically equivalent to expanding selected
    /// K/V; it is not a claim of bit-identical BF16 projection/reduction order.
    /// No RoPE, whole-history attention scores, mutable cache or GPU-only kernel.
    public static func attend(
        queries: MLXArray,
        latents: MLXArray,
        selection: Selection,
        keyProjection: MLXArray,
        valueProjection: MLXArray
    ) throws -> MLXArray {
        try require(queries.ndim == 4 && latents.ndim == 3, "attention query/latent ranks")
        try require(keyProjection.ndim == 3 && valueProjection.ndim == 3, "projection ranks")
        let batch = queries.dim(0)
        let count = queries.dim(1)
        let heads = queries.dim(2)
        let queryDimension = queries.dim(3)
        let latentDimension = latents.dim(2)
        try require(
            batch > 0 && count > 0 && heads > 0 && queryDimension > 0 && latentDimension > 0,
            "empty attention geometry")
        try require(
            latents.shape == [batch, selection.historyLength, latentDimension],
            "latent history shape")
        try require(
            selection.indices.ndim == 3 && selection.indices.dim(0) == batch
                && selection.indices.dim(1) == count, "selection query shape")
        try require(
            keyProjection.shape == [heads, queryDimension, latentDimension], "key projection shape")
        try require(
            valueProjection.dim(0) == heads && valueProjection.dim(1) > 0
                && valueProjection.dim(2) == latentDimension, "value projection shape")
        for array in [queries, latents, keyProjection, valueProjection] {
            try require(isFloating(array), "attention inputs must be float16, bfloat16 or float32")
        }
        let keyWeight = keyProjection.asType(.float32)
        let valueWeight = valueProjection.asType(.float32).transposed(0, 2, 1)
        let scale = 1 / sqrt(Float(queryDimension))
        var outputs = [MLXArray]()
        outputs.reserveCapacity(count)
        for query in 0 ..< count {
            let indices = selection.indices[0..., query]
            let valid = (indices .>= 0) .&& (indices .< selection.historyLength)
            let gathered = takeAlong(
                latents,
                clip(indices, min: 0, max: selection.historyLength - 1)
                    .expandedDimensions(axis: -1), axis: 1
            ).asType(.float32)
            let selected = MLX.where(valid.expandedDimensions(axis: -1), gathered, 0)
            let q = queries[0..., query].asType(.float32).expandedDimensions(axis: -2)
            let absorbedQuery = matmul(q, keyWeight).squeezed(axis: -2)
            let scores = matmul(absorbedQuery, selected.transposed(0, 2, 1)) * scale
            let mask = valid.expandedDimensions(axis: 1)
            let maskedScores = MLX.where(mask, scores, -Float.infinity)
            let rowMax = MLX.where(
                mask.any(axis: -1, keepDims: true), maskedScores.max(axis: -1, keepDims: true), 0)
            let exponentials = MLX.where(mask, exp(maskedScores - rowMax), 0)
            let denominator = exponentials.sum(axis: -1, keepDims: true)
            let probabilities = exponentials / MLX.where(denominator .> 0, denominator, 1)
            let latentOutput = matmul(probabilities, selected).expandedDimensions(axis: -2)
            outputs.append(matmul(latentOutput, valueWeight).squeezed(axis: -2))
        }
        return stacked(outputs, axis: 1).asType(queries.dtype)
    }

    private static func isFloating(_ array: MLXArray) -> Bool {
        [.float16, .bfloat16, .float32].contains(array.dtype)
    }

    private static func require(_ condition: Bool, _ message: String) throws {
        guard condition else { throw GLM5NextSparseAttentionError.invalidInput(message) }
    }
}
