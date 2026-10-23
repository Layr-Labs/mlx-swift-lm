import MLX

/// Super MoE kernels that replace short op chains bit for bit.
///
/// `select`: expert selection (one group) as one kernel per row, identical
/// to the op-by-op path of `groupExpertSelect`: MLX's float32 `sigmoid`
/// (`1 / (1 + exp(|x|))`, mirrored below zero), the bias add, MLX's stable
/// ascending arg sort of `-(score + bias)` (NaN last, ties by index, as its
/// merge sort orders them) cut to `topK`, the gathered sigmoid scores, their
/// sequential float32 sum plus `1e-20`, the division and the scaling factor.
///
/// `combine`: the weighted sum of the routed experts' outputs,
/// `(y * scores[..., newAxis]).sum(axis: -2).asType(y.dtype)`: float32
/// products summed in the order of MLX's small strided reduction (below 32
/// experts per token: `min(8, k)` partial sums over every `min(8, k)`-th
/// expert, then added in order) and rounded once to the output dtype.
enum NemotronSuperMoEKernels {
    private static let selectKernel = MLXFast.metalKernel(
        name: "nemotron_super_expert_select",
        inputNames: ["gates", "bias", "scale_factor"],
        outputNames: ["indices", "scores"],
        source: """
            const uint row = threadgroup_position_in_grid.x;
            const uint e = thread_position_in_threadgroup.x;
            threadgroup float keys[E];
            threadgroup float chosen_scores[K];
            threadgroup float denominator[1];
            float sigmoid_value = 0.0f;
            float key = 0.0f;
            if (e < E) {
                const float g = static_cast<float>(gates[size_t(row) * E + e]);
                const float y = 1 / (1 + metal::precise::exp(metal::abs(g)));
                sigmoid_value = (g < 0) ? y : 1 - y;
                key = -(sigmoid_value + bias[e]);
                keys[e] = key;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            int rank = 0;
            if (e < E) {
                const bool key_nan = metal::isnan(key);
                for (uint j = 0; j < E; j++) {
                    const float other = keys[j];
                    const bool other_nan = metal::isnan(other);
                    // MLX LessThan: NaN after every number; then a < b.
                    const bool other_less = (key_nan | other_nan)
                        ? ((!other_nan) & key_nan) : (other < key);
                    const bool key_less = (key_nan | other_nan)
                        ? ((!key_nan) & other_nan) : (key < other);
                    rank += other_less ? 1 : 0;
                    rank += (j < e && !other_less && !key_less) ? 1 : 0;
                }
                if (rank < K) {
                    chosen_scores[rank] = sigmoid_value;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (NORMALIZE && e == 0) {
                float total = 0.0f;
                for (int k = 0; k < K; k++) {
                    total = chosen_scores[k] + total;
                }
                denominator[0] = total + 1e-20f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (e < E && rank < K) {
                indices[size_t(row) * K + rank] = e;
                float value = sigmoid_value;
                if (NORMALIZE) {
                    value = value / denominator[0];
                }
                scores[size_t(row) * K + rank] = value * scale_factor[0];
            }
            """)

    /// Whether `select` reproduces `groupExpertSelect` for this configuration.
    static func supports(experts: Int, topK: Int, groups: Int) -> Bool {
        groups == 1 && topK >= 1 && topK <= experts && experts <= 1024
    }

    /// `(indices [..., topK] uint32, scores [..., topK] float32)` for router
    /// logits `gates` (`[..., E]`) and the float32 correction bias (`[E]`).
    static func select(
        gates: MLXArray, bias: MLXArray, topK: Int, scalingFactor: Float, normalize: Bool
    ) -> (MLXArray, MLXArray) {
        let experts = gates.dim(-1)
        let rows = gates.size / experts
        let threads = (experts + 31) / 32 * 32
        let outputs = selectKernel(
            [
                gates.reshaped(rows, experts), bias.asType(.float32),
                MLXArray([scalingFactor]),
            ],
            template: [
                ("E", experts), ("K", topK), ("NORMALIZE", normalize && topK > 1),
            ],
            grid: (threads * rows, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[rows, topK], [rows, topK]], outputDTypes: [.uint32, .float32])
        let shape = Array(gates.shape.dropLast()) + [topK]
        return (outputs[0].reshaped(shape), outputs[1].reshaped(shape))
    }

    private static let combineKernel = MLXFast.metalKernel(
        name: "nemotron_super_expert_combine",
        inputNames: ["y", "scores"],
        outputNames: ["out"],
        source: """
            const uint column = thread_position_in_grid.x;
            const uint row = thread_position_in_grid.y;
            if (column >= D) {
                return;
            }
            constexpr int P = K < 8 ? K : 8;
            const auto values = y + size_t(row) * K * D + column;
            const auto weights = scores + size_t(row) * K;
            float partial[P];
            for (int p = 0; p < P; p++) {
                partial[p] = 0.0f;
            }
            for (int r = 0; r < K; r++) {
                const float product = static_cast<float>(values[size_t(r) * D]) * weights[r];
                partial[r % P] = product + partial[r % P];
            }
            float total = partial[0];
            for (int p = 1; p < P; p++) {
                total = partial[p] + total;
            }
            out[size_t(row) * D + column] = static_cast<T>(total);
            """)

    /// Whether `combine` reproduces the op chain (MLX's small strided path).
    static func supportsCombine(experts: Int) -> Bool { experts >= 1 && experts < 32 }

    /// `[..., D]` in `y.dtype` from `y` (`[..., k, D]`) and float32 `scores`
    /// (`[..., k]`).
    static func combine(_ y: MLXArray, scores: MLXArray) -> MLXArray {
        let k = y.dim(-2)
        let width = y.dim(-1)
        let rows = y.size / (k * width)
        let threads = min(256, (width + 31) / 32 * 32)
        let output = combineKernel(
            [y.reshaped(rows, k, width), scores.asType(.float32).reshaped(rows, k)],
            template: [("T", y.dtype), ("K", k), ("D", width)],
            grid: ((width + threads - 1) / threads * threads, rows, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [[rows, width]], outputDTypes: [y.dtype])[0]
        return output.reshaped(Array(y.shape.dropLast(2)) + [width])
    }

    /// `combine` where `fused` and it reproduces the op chain, else the chain
    /// `(y * scores[..., newAxis]).sum(axis: -2).asType(y.dtype)`.
    static func weightedSum(_ y: MLXArray, scores: MLXArray, fused: Bool) -> MLXArray {
        fused && supportsCombine(experts: y.dim(-2))
            ? combine(y, scores: scores)
            : (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
    }
}
