// Row-exact projections for Inkling 2 decode and verify steps.
import Foundation
import MLX
import MLXNN

/// Matrix products whose per-row bits do not depend on how many rows run
/// together. MLX picks gemv for one row, its multi-vector gemv for 2-15 rows
/// (lane split by row count) and tiled GEMM beyond, and each reduces K in a
/// different order. Here rows always go through the same kernel: groups of at
/// most five rows, a lone row padded to two. Each group streams the matrix once.
enum Inkling2RowExact {
    static let groupRows = 5

    /// Set for a step whose rows must equal the same rows run in any other
    /// composition (decode steps and verify windows).
    @TaskLocal static var active = false

    /// `x @ w.T` for `x` of shape `[..., K]` and `w` `[N, K]`, in the dtype matmul gives.
    static func matmul(_ x: MLXArray, transposedWeight w: MLXArray) -> MLXArray {
        if active, let y = Inkling2Kernels.rowProduct(x, transposedWeight: w) { return y }
        guard active else { return prefillMatmul(x, transposedWeight: w) }
        let shape = x.shape
        let k = shape.last!
        let rows = shape.dropLast().reduce(1, *)
        let flat = x.reshaped(rows, k)
        var outputs: [MLXArray] = []
        var start = 0
        while start < rows {
            let count = min(groupRows, rows - start)
            var group = flat[start ..< start + count]
            if count == 1 { group = concatenated([group, group], axis: 0) }
            var y = MLX.matmul(group, w.T)
            if count == 1 { y = y[0 ..< 1] }
            outputs.append(y)
            start += count
        }
        let out = outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 0)
        return out.reshaped(Array(shape.dropLast()) + [w.dim(0)])
    }

    /// FP32 `x @ w.T` as one gemv per row: `w[1, N, K] @ x[R, K, 1]`. MLX keeps
    /// this a batched gemv (a broadcast weight on the right would fold the rows
    /// into a GEMM, which uses TF32 on M5), so it is exact FP32 on every GPU,
    /// in prefill and decode alike, and a row never depends on its company.
    ///
    /// Prefill (not row-exact steps) uses `Inkling2ExactFP32`'s tiled kernel
    /// for products with at least 16 outputs (router, message gates, loop
    /// messages): also full FP32 and independent of row company and chunking,
    /// with a different (fixed) reduction order than the gemv.
    static func fp32Rows(_ x: MLXArray, _ w: MLXArray) -> MLXArray {
        let shape = x.shape
        let k = shape.last!
        let rows = shape.dropLast().reduce(1, *)
        // Narrow products (the residual-mix scores, N <= 3) stay on the gemv,
        // which streams their rows faster than a 64-column tile.
        if !active, Inkling2ExactFP32.active, w.dim(0) >= 16 {
            return Inkling2ExactFP32.matmul(
                x.asType(.float32).reshaped(rows, k), transposedWeight: w.asType(.float32)
            ).reshaped(Array(shape.dropLast()) + [w.dim(0)])
        }
        let y = MLX.matmul(
            w.asType(.float32).expandedDimensions(axis: 0),
            x.asType(.float32).reshaped(rows, k, 1))
        return y.reshaped(Array(shape.dropLast()) + [w.dim(0)])
    }

    /// A bias-free `Linear` (Inkling 2 projections have no bias).
    static func linear(_ layer: Linear, _ x: MLXArray, prefillBlockRows: Int? = nil) -> MLXArray {
        guard type(of: layer) == Linear.self, layer.bias == nil else { return layer(x) }
        guard active else {
            return prefillMatmul(x, transposedWeight: layer.weight, blockRows: prefillBlockRows)
        }
        return matmul(x, transposedWeight: layer.weight)
    }

    /// `DARKBLOOM_INKLING2_INVARIANT_PREFILL=1` makes chunk-invariant prefill
    /// the default of new models (`Inkling2Model.invariantPrefillBlockRows`).
    /// Off by default: MLX's own kernel choice, which varies with the chunk
    /// length, under the fixed chunk grid.
    static let invariantPrefillDefault: Bool = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_INVARIANT_PREFILL"]?
            .lowercased()
        return raw.map { ["1", "true", "on", "yes"].contains($0) } ?? false
    }()

    /// The products of the current step: set from the model's
    /// `invariantPrefillBlockRows` around each engine step (`cbv2Hidden`).
    @TaskLocal static var invariantPrefillOverride: Bool? = nil
    @TaskLocal static var prefillBlockRowsOverride: Int? = nil

    static var invariantPrefill: Bool { invariantPrefillOverride ?? invariantPrefillDefault }

    /// Runs `body` with prefill products in calls of `blockRows` rows, or
    /// with MLX's own kernel choice when `blockRows` is nil.
    static func withPrefill<T>(blockRows: Int?, _ body: () throws -> T) rethrows -> T {
        try $invariantPrefillOverride.withValue(blockRows != nil) {
            try $prefillBlockRowsOverride.withValue(blockRows, operation: body)
        }
    }

    /// Rows per prefill product call: every prefill product runs as calls of
    /// exactly this many rows (the last one zero-padded), so MLX picks one
    /// kernel configuration whatever the chunk length, and a row's result
    /// never depends on the chunk it came in (its company within a call does
    /// not change it either; Inkling2InvarianceProbeTests). Separate calls,
    /// not a batched product, because MLX folds a broadcast batch into M.
    /// 64 rows: a short follow-up chunk pads to 64 rows only (a product of
    /// that size costs about what one row's weight read costs), while a
    /// 2,048-token chunk makes 32 calls per product.
    /// `DARKBLOOM_INKLING2_PREFILL_BLOCK_ROWS` overrides it (cost studies).
    static let prefillBlockRowsDefault: Int = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_PREFILL_BLOCK_ROWS"]
        return raw.flatMap { Int($0) }.map { max(1, $0) } ?? 64
    }()

    static var prefillBlockRows: Int { prefillBlockRowsOverride ?? prefillBlockRowsDefault }

    /// Prefill `x @ w.T` in fixed-size row blocks (see `prefillBlockRows`).
    static func prefillMatmul(
        _ x: MLXArray, transposedWeight w: MLXArray, blockRows: Int? = nil
    ) -> MLXArray {
        let blockRows = blockRows ?? prefillBlockRows
        guard invariantPrefill else { return MLX.matmul(x, w.T) }
        let shape = x.shape
        let k = shape.last!
        let rows = shape.dropLast().reduce(1, *)
        let flat = x.reshaped(rows, k)
        var outputs: [MLXArray] = []
        var start = 0
        while start < rows {
            let count = min(blockRows, rows - start)
            var block = flat[start ..< start + count]
            if count < blockRows {
                block = concatenated(
                    [block, MLXArray.zeros([blockRows - count, k], dtype: x.dtype)], axis: 0)
            }
            var y = MLX.matmul(block, w.T)
            if count < blockRows { y = y[0 ..< count] }
            outputs.append(y)
            start += count
        }
        let out = outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 0)
        return out.reshaped(Array(shape.dropLast()) + [w.dim(0)])
    }
}
