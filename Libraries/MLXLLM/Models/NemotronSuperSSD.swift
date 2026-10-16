import MLX
import MLXFast

/// Super-only prefill SSD scan: `ssmAttn` (the mlx-lm 256-token windowed
/// quadratic form, used unchanged by every other Nemotron-H model) with each
/// window's masked decay-weighted matrix built by one Metal kernel instead of
/// materializing the repeated segment sums, their exponential, the repeated
/// C.B products and the triangular masks as separate arrays. Segment sums come
/// from an inclusive cumulative sum, `cs[i] - cs[j]`, rather than the
/// reference's masked cumulative sum of a repeated matrix, so results differ
/// in float32 rounding only (bounded in NemotronSuperSSDTests).
enum NemotronSuperSSD {
    private static let decayKernel = MLXFast.metalKernel(
        name: "nemotron_super_ssd_window",
        inputNames: ["cb", "cs"],
        outputNames: ["weights"],
        source: """
            const uint j = thread_position_in_grid.x;
            const uint i = thread_position_in_grid.y;
            const uint bh = thread_position_in_grid.z;
            if (i >= L || j >= L) {
                return;
            }
            const uint b = bh / H;
            const uint h = bh % H;
            const uint g = h / (H / G);
            float value = 0.0f;
            if (j <= i) {
                const auto row = cs + size_t(bh) * L;
                const float decay = metal::exp(row[i] - row[j]);
                value = static_cast<float>(cb[((size_t(b) * G + g) * L + i) * L + j]) * decay;
            }
            weights[(size_t(bh) * L + i) * L + j] = value;
            """)

    /// `tril(repeat(C.B) * exp(segsum(dtA)))` for one window: `cb` is
    /// `[b, g, L, L]`, `cumulative` the inclusive cumulative `dtA` `[b, h, L]`.
    private static func windowWeights(_ cb: MLXArray, cumulative: MLXArray, heads: Int)
        -> MLXArray
    {
        let (batch, groups, length) = (cb.dim(0), cb.dim(1), cb.dim(2))
        return decayKernel(
            [cb, cumulative.asType(.float32)],
            template: [("L", length), ("H", heads), ("G", groups)],
            grid: ((length + 15) / 16 * 16, length, batch * heads), threadGroup: (16, 16, 1),
            outputShapes: [[batch, heads, length, length]], outputDTypes: [.float32])[0]
    }

    /// Same contract as `ssmAttn` without a mask.
    static func scan(
        x: MLXArray, ALog: MLXArray, B: MLXArray, C: MLXArray, D: MLXArray,
        dt: MLXArray, dtBias: MLXArray, state: MLXArray?, timeStepLimit: (Float, Float)
    ) -> (MLXArray, MLXArray) {
        let (b, l, h, dh) = x.shape4
        let (_, _, g, d) = B.shape4
        let dt = computeDt(dt, dtBias, timeStepLimit)
        let repeats = h / g
        let A = -MLX.exp(ALog).asType(dt.dtype)
        let dtA = dt * A.reshaped(1, 1, -1)
        let dtx = dt.reshaped(b, l, h, 1) * x
        let step = 256
        var currentState = state
        var outputs = [MLXArray]()
        outputs.reserveCapacity((l + step - 1) / step)
        for start in stride(from: 0, to: l, by: step) {
            let end = min(start + step, l)
            let stepDtx = dtx[0..., start ..< end, 0..., 0...]
            let stepDtA = dtA[0..., start ..< end, 0...]
            var stepB = B[0..., start ..< end, 0..., 0...]
            let stepC = C[0..., start ..< end, 0..., 0...]

            stepB = MLX.transposed(stepB, axes: [0, 2, 3, 1])
            let CB = MLX.swappedAxes(stepC, 1, 2).matmul(stepB)
            let cumulative = MLX.cumsum(stepDtA.swappedAxes(1, 2), axis: -1)
            let weights = windowWeights(CB, cumulative: cumulative, heads: h)
            var y = weights.matmul(stepDtx.swappedAxes(1, 2))
            y = MLX.swappedAxes(y, 1, 2)

            let lastDecay = MLX.exp(cumulative[0..., 0..., (-1)...] - cumulative)
            let decay = lastDecay.reshaped(b, h, end - start, 1).transposed(0, 2, 1, 3)
            stepB = MLX.repeated(stepB, count: repeats, axis: 1).swappedAxes(2, 3)
            var dtxdecay = stepDtx * decay
            dtxdecay = dtxdecay.swappedAxes(1, 2).swappedAxes(2, 3)
            var nextState = dtxdecay.matmul(stepB)

            if let previousState = currentState {
                let expDtACumsum = MLX.exp(MLX.cumsum(stepDtA, axis: -2))
                nextState =
                    nextState
                    + expDtACumsum[0..., -1, 0..., .newAxis, .newAxis] * previousState
                let reshapedState = previousState.reshaped(b, 1, g, repeats, dh, d)
                let reshapedC = stepC.reshaped(b, end - start, g, 1, d, 1)
                let yPrev = (reshapedState.matmul(reshapedC))
                    .squeezed(axis: -1)
                    .flattened(start: 2, end: 3)
                y = y + expDtACumsum[.ellipsis, .newAxis] * yPrev
            }

            outputs.append(y.asType(x.dtype))
            currentState = nextState
        }

        let y = MLX.concatenated(outputs, axis: 1) + x * D.reshaped(1, 1, h, 1)
        return (y, currentState!)
    }
}
