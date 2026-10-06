import MLX
import MLXNN

/// Preserve singleton router arithmetic when independent decode rows join a
/// Qwen4 cohort. Dense bf16 M>1 matmul can round differently from M1; the
/// resulting router-score difference may change expert selection or weights.
/// Only the small router projection is rowwise. Attention projections and
/// routed/shared expert execution remain batched; no precision is changed.
func qwen4CanonicalBatchedRouterProjection(_ linear: Linear, _ input: MLXArray) -> MLXArray {
    precondition(input.ndim == 3 && input.dim(1) == 1)
    return concatenated((0..<input.dim(0)).map { row in
        linear(input[row..<row + 1, 0..., 0...])
    }, axis: 0)
}
