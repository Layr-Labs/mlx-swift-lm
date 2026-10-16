import MLX
import MLXLMCommon
import MLXNN

// Multi-row operations whose every row reproduces the single-row decode
// result bit for bit at the pinned Super geometry
// (NemotronSuperMTPRowExactnessTests). Stock M>1 matrix kernels change the
// accumulation, so anything not proven row-exact runs one row at a time.

/// Bias-free BF16 dense projection as one gathered M=1 product per row in a
/// single dispatch; row-exactness is characterized for BF16 only. Other types
/// and dtypes fall back to one call per row.
func superRowExactDense(_ linear: Linear, _ x: MLXArray) -> MLXArray {
    guard type(of: linear) == Linear.self, linear.bias == nil, x.ndim == 3, x.dim(0) == 1,
        x.dtype == .bfloat16, linear.weight.dtype == .bfloat16
    else { return nemotronMTPMapRows(x) { linear($0) } }
    let (outputs, inputs) = linear.shape
    let rows = x.dim(1)
    // [1, N, K] -> [1, K, N] as a strided view; no weight copy.
    let weight = linear.weight.expandedDimensions(axis: 0).swappedAxes(1, 2)
    return gatherMM(
        x.reshaped(rows, 1, inputs), weight,
        rhsIndices: MLXArray.zeros([rows], type: UInt32.self)
    ).reshaped(1, rows, outputs)
}

/// The fastpath graph's `.rowwise` NVFP4 product is row-exact for any row
/// count, and `superRowExactFP8` shares the exact FP8 vector product's weight
/// reads across rows; any other projection runs one row at a time.
func superRowExactProjection(_ layer: any UnaryLayer, _ x: MLXArray) -> MLXArray {
    if let projection = layer as? ModelOptCalibratedLinear {
        if case .nvfp4 = projection.format, projection.execution == .graph,
            projection.accumulation == .rowwise
        {
            return projection(x)
        }
        if let rows = try? projection.superRowExactFP8(x) { return rows }
    }
    if let linear = layer as? Linear, type(of: linear) == Linear.self {
        return superRowExactDense(linear, x)
    }
    return nemotronMTPMapRows(x) { layer($0) }
}
