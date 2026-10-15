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
    /// A selection made against one explicit history length. Do not mutate its
    /// arrays: valid indices are causal and unique within each query row.
    public struct Selection {
        public let indices: MLXArray
        public let historyLength: Int

        fileprivate init(indices: MLXArray, historyLength: Int) {
            self.indices = indices
            self.historyLength = historyLength
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
    /// `[B,Q,H,N]` score tensor is constructed. Top-k ties follow MLX's sort;
    /// selected pools are returned in chronological order. HF does not specify
    /// a cross-backend tie-breaking order either.
    public static func selectTokens(
        indexQueries: MLXArray,
        indexKeys: MLXArray,
        poolGates: MLXArray,
        poolPositionBias: MLXArray,
        headWeights: MLXArray,
        keyValidity: MLXArray,
        queryValidity: MLXArray,
        queryStart: Int,
        tokenBudget: Int
    ) throws -> Selection {
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

        let poolCount = (history + 3) / 4
        let firstKey = MLX.where(
            keyValidity.any(axis: 1),
            keyValidity.asType(.int32).argMax(axis: 1).asType(.int32), history)
        let poolOffsets = MLXArray(0 ..< poolCount * 4).reshaped([1, poolCount, 4])
        let poolIndices = firstKey.reshaped([batch, 1, 1]) + poolOffsets
        let safe = clip(poolIndices, min: 0, max: history - 1).reshaped([batch, poolCount * 4])
        let groupedKeys = takeAlong(indexKeys, safe.expandedDimensions(axis: -1), axis: 1)
            .reshaped([batch, poolCount, 4, dimension])
        let groupedGates = takeAlong(poolGates, safe.expandedDimensions(axis: -1), axis: 1)
            .reshaped([batch, poolCount, 4, dimension])
        let groupedValid =
            takeAlong(keyValidity, safe, axis: 1)
            .reshaped([batch, poolCount, 4]) .&& (poolIndices .< history)
        let completePools = groupedValid.all(axis: 2)
        let gateLogits = groupedGates.asType(.float32) + poolPositionBias.asType(.float32)
        let probabilities = softmax(
            MLX.where(groupedValid.expandedDimensions(axis: -1), gateLogits, -Float.infinity),
            axis: 2, precise: true)
        // Fully padded pools have an all--inf softmax. Select zero rather than
        // multiplying NaN by zero; partial pools are never eligible for top-k.
        let poolWeights = MLX.where(groupedValid.expandedDimensions(axis: -1), probabilities, 0)
            .asType(indexKeys.dtype)
        let pooledKeys =
            (poolWeights
            * MLX.where(
                groupedValid.expandedDimensions(axis: -1), groupedKeys, 0)).sum(axis: 2)
        let poolEnds = poolIndices[.ellipsis, 3]
        let positions = MLXArray(0 ..< history).reshaped([1, history])
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
            let selected = sorted(
                argSort(-MLX.where(eligible, scores, -Float.infinity), axis: 1)[
                    0..., ..<selectedPoolCount
                ].asType(.int32), axis: 1)
            let selectedValid = takeAlong(eligible, selected, axis: 1)
            let members = takeAlong(poolIndices, selected.expandedDimensions(axis: -1), axis: 1)
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
        return Selection(indices: stacked(rows, axis: 1).asType(.int32), historyLength: history)
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
