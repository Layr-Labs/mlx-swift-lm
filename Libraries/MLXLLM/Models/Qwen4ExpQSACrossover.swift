import MLX
import MLXLMCommon

extension Qwen4ExpGatheredQSA {
    /// Ordinary prefill may batch mathematically equivalent full attention,
    /// but MTP must preserve the target's actual per-token numerics. A verify
    /// window ending past the budget can still start on dense-only columns.
    /// Keep this rare transition separate; fully sparse windows stay batched.
    static func serialCrossoverEligible(
        offset: Int, length: Int, compressRatio: Int, tokenBudget: Int
    ) -> Bool {
        guard offset >= 0, length > 1, compressRatio > 0, tokenBudget > 0,
            tokenBudget % compressRatio == 0, offset <= Int.max - length
        else { return false }
        return offset + length > tokenBudget
            && !decodeCrossesBudget(offset: offset, compressRatio: compressRatio, tokenBudget: tokenBudget)
    }

    /// Q/K/V projections and indexer state were already built for the window.
    /// Only attention crosses the exact same boundary as standalone decode.
    /// One cache object owns both attention and indexer state throughout.
    static func attendCrossover(
        cache: any CBv2AttendingLayerCache & CBv2Qwen4GatheredCache,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        indexQueries: MLXArray, offset: Int,
        queryHeads: Int, kvHeads: Int, headDim: Int, indexerHeadDim: Int,
        compressRatio: Int, tokenBudget: Int, scale: Float,
        indexKeyNorm: (MLXArray) -> MLXArray,
        applyIndexRope: (MLXArray, MLXArray) -> MLXArray
    ) -> MLXArray {
        let length = queries.dim(2)
        precondition(cache.qwen4SerializesRectangularAttention)
        precondition(queries.shape == [1, queryHeads, length, headDim])
        precondition(keys.shape == [1, kvHeads, length, headDim] && values.shape == keys.shape)
        precondition(indexQueries.dim(0) == 1 && indexQueries.dim(1) == length)
        precondition(serialCrossoverEligible(
            offset: offset, length: length, compressRatio: compressRatio, tokenBudget: tokenBudget))
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(length)
        for column in 0..<length {
            let range = column..<(column + 1)
            let q = queries[0..., 0..., range, 0...]
            let k = keys[0..., 0..., range, 0...]
            let v = values[0..., 0..., range, 0...]
            if decodeCrossesBudget(
                offset: offset + column, compressRatio: compressRatio, tokenBudget: tokenBudget) {
                let views = cache.updateKVAndAdvanceOffsets(keys: k, values: v)
                precondition(views.count == 1)
                let pooled = Qwen4ExpPooledIndex.reuseOrCompute(
                    cache: cache, compressRatio: compressRatio,
                    logicalTokens: views[0].keys.dim(2),
                    indexKeyNorm: indexKeyNorm, applyIndexRope: applyIndexRope)
                outputs.append(attendDecode(
                    queries: q, keys: views[0].keys, values: views[0].values,
                    indexQueries: indexQueries[0..., range, 0..., 0...], pooledIndexKeys: pooled,
                    queryHeads: queryHeads, kvHeads: kvHeads, headDim: headDim,
                    indexerHeadDim: indexerHeadDim, compressRatio: compressRatio, tokenBudget: tokenBudget))
            } else {
                outputs.append(cache.updateAndAttend(
                    queries: q, keys: k, values: v, scale: scale, sinks: nil).transposed(0, 2, 1, 3))
            }
        }
        return concatenated(outputs, axis: 1)
    }
}
