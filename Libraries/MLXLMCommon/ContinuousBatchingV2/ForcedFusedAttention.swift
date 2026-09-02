// ForcedFusedAttention.swift
//
// Single integration seam for the pending mlx-swift forceFused API. The
// currently pinned 1d452d8 dependency does not compile this call. Exact-SHA
// integration must provide the mask-mode overload with a `forceFused: Bool`
// named argument; if its argument order differs, only this file should change.

import MLX

enum CBv2ForcedFusedAttention {
    static func call(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        sinks: MLXArray?,
        executionObserver: (() -> Void)? = nil
    ) -> MLXArray {
        executionObserver?()
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask,
            sinks: sinks,
            forceFused: true)
    }
}
