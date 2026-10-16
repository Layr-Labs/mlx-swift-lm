import MLX
import MLXFast

/// Super-only prefill SSD scan: `ssmAttn` (the mlx-lm 256-token windowed
/// quadratic form, used unchanged by every other Nemotron-H model) with each
/// window's masked decay-weighted matrix built by one Metal kernel instead of
/// materializing the repeated segment sums, their exponential, the repeated
/// C.B products and the triangular masks as separate arrays. The kernel
/// reproduces those ops bit for bit: each segment-sum column is MLX's strided
/// cumulative sum of the strictly-lower-masked `dtA` (32-row blocks, the same
/// simd prefix sums and carried prefix), then MLX's precise `exp`, the float32
/// product with C.B and the lower-triangular mask; the decay row the state
/// update reads is the last row of the same sums. The rest of the window is the
/// oracle's own ops, so outputs and states equal `ssmAttn` exactly on any GPU
/// where MLX's own kernels are deterministic.
enum NemotronSuperSSD {
    private static let decayKernel = MLXFast.metalKernel(
        name: "nemotron_super_ssd_window_exact",
        inputNames: ["cb", "dta"],
        outputNames: ["weights", "last_decay"],
        source: """
            // One simdgroup per (batch * head, column j); lane = row in a block.
            const uint lane = thread_index_in_simdgroup;
            const uint j = threadgroup_position_in_grid.x * (TG / 32)
                + simdgroup_index_in_threadgroup;
            const uint bh = threadgroup_position_in_grid.y;
            if (j >= L) {
                return;
            }
            const uint b = bh / H;
            const uint h = bh % H;
            const uint g = h / (H / G);
            const auto steps = dta + size_t(bh) * L;
            const auto products = cb + (size_t(b) * G + g) * L * L;
            float prefix = 0.0f;
            for (uint i0 = 0; i0 < L; i0 += 32) {
                const uint i = i0 + lane;
                const float value = (i < L && i > j) ? static_cast<float>(steps[i]) : 0.0f;
                float sum = simd_prefix_inclusive_sum(value);
                sum = sum + prefix;
                prefix = simd_shuffle(sum, 31);
                if (i < L) {
                    const float decay = metal::precise::exp(sum);
                    const float weighted = static_cast<float>(products[size_t(i) * L + j]) * decay;
                    weights[(size_t(bh) * L + i) * L + j] = i >= j ? weighted : 0.0f;
                    if (i == L - 1) {
                        last_decay[size_t(bh) * L + j] = decay;
                    }
                }
            }
            """)

    /// For one window: `tril(repeat(C.B) * exp(segsum(dtA)))` (`[b, h, L, L]`)
    /// and the last row of `exp(segsum(dtA))` (`[b, h, L]`), where `cb` is
    /// `[b, g, L, L]` and `dtA` the window's `[b, h, L]`.
    private static func windowWeights(_ cb: MLXArray, dtA: MLXArray, heads: Int)
        -> (weights: MLXArray, lastDecay: MLXArray)
    {
        let (batch, groups, length) = (cb.dim(0), cb.dim(1), cb.dim(2))
        let threads = 256
        let columnsPerGroup = threads / 32
        let outputs = decayKernel(
            [cb, dtA.asType(.float32)],
            template: [("L", length), ("H", heads), ("G", groups), ("TG", threads)],
            grid: ((length + columnsPerGroup - 1) / columnsPerGroup * threads, batch * heads, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [[batch, heads, length, length], [batch, heads, length]],
            outputDTypes: [.float32, .float32])
        return (outputs[0], outputs[1])
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
            let (weights, lastDecay) = windowWeights(
                CB, dtA: stepDtA.swappedAxes(1, 2), heads: h)
            var y = weights.matmul(stepDtx.swappedAxes(1, 2))
            y = MLX.swappedAxes(y, 1, 2)

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
