import MLX
import MLXLMCommon

extension Qwen4ExpGatheredQSA {
    /// The portable prefill path rounds softmax probabilities to the query
    /// dtype before its value matmul. Singleton decode uses fused SDPA, so
    /// those two paths are not a bit-exact MTP verifier for unsupported
    /// native geometries. Keep the already-projected window and written KV;
    /// only attention reads each canonical visible prefix independently.
    static func attendCanonicalSparseColumns(
        cache: any CBv2Qwen4GatheredCache,
        queries: MLXArray, keys: MLXArray, values: MLXArray, indexQueries: MLXArray,
        queryHeads: Int, kvHeads: Int, headDim: Int, indexerHeadDim: Int,
        compressRatio: Int, tokenBudget: Int,
        indexKeyNorm: (MLXArray) -> MLXArray,
        applyIndexRope: (MLXArray, MLXArray) -> MLXArray
    ) -> MLXArray {
        let length = queries.dim(2)
        let offset = keys.dim(2) - length
        precondition(cache.qwen4SerializesRectangularAttention && length > 1)
        precondition(decodeCrossesBudget(offset: offset, compressRatio: compressRatio, tokenBudget: tokenBudget))
        var columns: [MLXArray] = []
        columns.reserveCapacity(length)
        for column in 0..<length {
            let visible = offset + column + 1
            let range = column..<column + 1
            let pooled = Qwen4ExpPooledIndex.reuseOrCompute(
                cache: cache, compressRatio: compressRatio, logicalTokens: visible,
                indexKeyNorm: indexKeyNorm, applyIndexRope: applyIndexRope)
            columns.append(attendDecode(
                queries: queries[0..., 0..., range, 0...],
                keys: keys[0..., 0..., 0..<visible, 0...],
                values: values[0..., 0..., 0..<visible, 0...],
                indexQueries: indexQueries[0..., range, 0..., 0...],
                pooledIndexKeys: pooled, queryHeads: queryHeads, kvHeads: kvHeads,
                headDim: headDim, indexerHeadDim: indexerHeadDim,
                compressRatio: compressRatio, tokenBudget: tokenBudget))
        }
        return concatenated(columns, axis: 1)
    }
}
