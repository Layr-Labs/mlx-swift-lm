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
    let indices =
        NemotronSuperMTPFusion.active
        ? NemotronSuperMTPFusion.zeroRowIndices(rows)
        : MLXArray.zeros([rows], type: UInt32.self)
    return gatherMM(x.reshaped(rows, 1, inputs), weight, rhsIndices: indices)
        .reshaped(1, rows, outputs)
}

/// Up to `ModelOptRowExact.maximumRows` rows, each bit-identical to the
/// one-row decode call. ModelOpt projections run through `rowExactForward`
/// (FP8 `ModelOptRowExact` products for K > 64, which read the weights once per
/// eight-row dispatch; NVFP4 one single-vector product per row), bias-free
/// dense projections through `nemotronRowExactLinear` (also once per eight-row
/// dispatch), and larger BF16 dense inputs through the gathered M = 1 product.
/// Anything else, including FP8 with K <= 64 (whose several-row graph call
/// takes eager tiles), runs one row at a time. Callers with more rows use
/// `superRowExactRows`. `squaredReLU` returns `relu2` of the result
/// (`maximum(y, 0)`, then `y * y`), fused into a graph ModelOpt projection's
/// call as the target's decode fuses it (`NemotronHMLP`: NVFP4 squares in its
/// final rounding kernel, FP8 applies the same two ops); other routes square
/// after.
func superRowExactProjection(
    _ layer: any UnaryLayer, _ x: MLXArray, squaredReLU: Bool = false
) -> MLXArray {
    let rows = x.dim(-1) == 0 ? 0 : x.size / x.dim(-1)
    if let projection = layer as? ModelOptCalibratedLinear, projection.execution == .graph {
        if rows == 1 { return squaredReLU ? projection.squaredReLU(x) : projection(x) }
        let qualifies: Bool
        switch projection.format {
        case .fp8: qualifies = projection.inputDimensions > 64
        case .nvfp4: qualifies = true
        }
        if qualifies, rows <= ModelOptRowExact.maximumRows,
            let value = try? projection.rowExactForward(x, squaredReLU: squaredReLU)
        {
            return value
        }
    }
    if squaredReLU {
        let r = maximum(superRowExactProjection(layer, x), MLXArray(0))
        return r * r
    }
    if let linear = layer as? Linear, type(of: linear) == Linear.self {
        if linear.bias == nil, let value = nemotronRowExactLinear(x, linear.weight) {
            return value
        }
        return superRowExactDense(linear, x)
    }
    return nemotronMTPMapRows(x) { layer($0) }
}

/// `x` (`[1, R, ...]`) in consecutive groups of at most
/// `ModelOptRowExact.maximumRows` rows; each group's rows are row-local.
func superRowExactGroups(_ x: MLXArray, _ body: (MLXArray) -> MLXArray) -> MLXArray {
    let limit = ModelOptRowExact.maximumRows
    guard x.dim(1) > limit else { return body(x) }
    return concatenated(
        stride(from: 0, to: x.dim(1), by: limit).map { start in
            body(x[0..., start ..< min(x.dim(1), start + limit)])
        }, axis: 1)
}
