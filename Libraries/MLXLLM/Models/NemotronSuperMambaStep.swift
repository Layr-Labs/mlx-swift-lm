import MLX
import MLXLMCommon

/// Super's one-token Mamba step with three op chains of `NemotronHMamba2Mixer`
/// fused, each bit-identical to the chain it replaces:
/// - `convolve`: the conv-state concatenation, MLX's depthwise conv1d (float
///   accumulation over the taps in order), the bias add, `silu`, and the
///   retained conv state (the last K - 1 rows, `* 1` as before).
/// - `scan`: `computeDt` (the float32 bias add, `softplus` as MLX's LogAddExp
///   with 0, then MLX's Maximum and Minimum for the clip) inside a copy of the
///   SSM step kernel of `SSM.swift`.
/// - `gatedNorm`: `y * silu(gate)`, MLX's single-row RMS norm per group with
///   the unit weight, then the learned weight.
/// Every bf16 intermediate is rounded where the op chain rounds it: the
/// kernels apply MLX's own functor expressions to the same dtypes.
enum NemotronSuperMambaStep {
    static let header = """
        template <typename T>
        inline T nemotron_super_sigmoid(T x) {
            auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
            return (x < 0) ? y : 1 - y;
        }

        template <typename T>
        inline T nemotron_super_silu(T x) {
            T s = nemotron_super_sigmoid(x);
            return x * s;
        }
        """

    private static let convolveKernel = MLXFast.metalKernel(
        name: "nemotron_super_mamba_conv_step",
        inputNames: ["state", "input", "weight", "bias"],
        outputNames: ["out", "next_state"],
        source: """
            const uint c = thread_position_in_grid.x;
            const uint b = thread_position_in_grid.y;
            if (c >= C) {
                return;
            }
            const auto rows = state + size_t(b) * (KS - 1) * C + c;
            const T current = input[size_t(b) * C + c];
            float acc = 0.0;
            for (int i = 0; i < KS; ++i) {
                const T value = i < KS - 1 ? rows[size_t(i) * C] : current;
                acc += static_cast<float>(value) * weight[size_t(c) * KS + i];
            }
            const T convolved = static_cast<T>(acc);
            const T biased = convolved + bias[c];
            out[size_t(b) * C + c] = nemotron_super_silu(biased);
            const T one = static_cast<T>(1);
            device T* next = next_state + size_t(b) * (KS - 1) * C + c;
            for (int i = 0; i < KS - 1; ++i) {
                const T kept = i + 1 < KS - 1 ? rows[size_t(i + 1) * C] : current;
                next[size_t(i) * C] = kept * one;
            }
            """,
        header: header)

    /// Conv step for `input` `[B, 1, C]` after `state` `[B, K - 1, C]` with the
    /// depthwise `weight` `[C, K, 1]` and `bias` `[C]`, all one dtype:
    /// `(silu(conv1d(padded) + bias) [B, 1, C], next state [B, K - 1, C])`.
    static func convolve(
        state: MLXArray, input: MLXArray, weight: MLXArray, bias: MLXArray
    ) -> (MLXArray, MLXArray) {
        let (batch, channels) = (input.dim(0), input.dim(2))
        let taps = weight.dim(1)
        let threads = 256
        let outputs = convolveKernel(
            [state, input, weight, bias],
            template: [("T", input.dtype), ("C", channels), ("KS", taps)],
            grid: ((channels + threads - 1) / threads * threads, batch, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [[batch, 1, channels], [batch, taps - 1, channels]],
            outputDTypes: [input.dtype, input.dtype])
        return (outputs[0], outputs[1])
    }

    /// Whether `convolve` reproduces `silu(conv1d(concat(state, input)))` here.
    static func supportsConvolve(
        state: MLXArray, input: MLXArray, weight: MLXArray, bias: MLXArray?
    ) -> Bool {
        guard let bias, input.ndim == 3, input.dim(1) == 1, weight.ndim == 3,
            weight.dim(2) == 1, weight.dim(0) == input.dim(2), weight.dim(1) >= 2,
            state.shape == [input.dim(0), weight.dim(1) - 1, input.dim(2)],
            bias.shape == [input.dim(2)]
        else { return false }
        let dtype = input.dtype
        return [.bfloat16, .float16, .float32].contains(dtype) && state.dtype == dtype
            && weight.dtype == dtype && bias.dtype == dtype
    }

    private static let scanKernel = MLXFast.metalKernel(
        name: "nemotron_super_mamba_ssm_step",
        inputNames: ["X", "A_log", "B", "C", "D", "dt_raw", "dt_bias", "limits", "state_in"],
        outputNames: ["out", "state_out"],
        source: """
            auto n = thread_position_in_grid.z;
            auto h_idx = n % H;
            auto g_idx = n / G;
            constexpr int n_per_t = (Ds + 31) / 32;

            auto x = X + n * Dh;
            out += n * Dh;
            auto i_state = state_in + n * Dh * Ds;
            auto o_state = state_out + n * Dh * Ds;

            auto C_ = C + g_idx * Ds;
            auto B_ = B + g_idx * Ds;

            auto ds_idx = thread_position_in_threadgroup.x;
            auto d_idx = thread_position_in_grid.y;

            if constexpr (Dh % 8 != 0) {
                if (d_idx >= Dh) return;
            }

            // computeDt: softplus(float(dt) + float(dt_bias)) as MLX's
            // LogAddExp(x, 0), then clip as Maximum(., lo) and Minimum(., hi).
            const float shifted =
                static_cast<float>(dt_raw[n]) + static_cast<float>(dt_bias[h_idx]);
            float softplus_value;
            {
                const float zero = 0.0f;
                if (metal::isnan(shifted) || metal::isnan(zero)) {
                    softplus_value = metal::numeric_limits<float>::quiet_NaN();
                } else {
                    constexpr float inf = metal::numeric_limits<float>::infinity();
                    const float maxval = metal::max(shifted, zero);
                    const float minval = metal::min(shifted, zero);
                    softplus_value = (minval == -inf || maxval == inf)
                        ? maxval
                        : (maxval + log1p(metal::exp(minval - maxval)));
                }
            }
            const float lower = limits[0];
            const float upper = limits[1];
            const float raised = metal::isnan(softplus_value)
                ? softplus_value : (softplus_value > lower ? softplus_value : lower);
            const float clipped = metal::isnan(raised)
                ? raised : (raised < upper ? raised : upper);

            auto dt_ = static_cast<float>(clipped);
            auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
            auto dA = fast::exp(A * dt_);

            float acc = 0.0;
            auto x_ = static_cast<float>(x[d_idx]);

            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * ds_idx + i;
                if constexpr (Ds % 32 != 0) {
                    if (s_idx >= Ds) continue;
                }
                auto idx = d_idx * Ds + s_idx;
                auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
                auto state = dA * i_state[idx] + dB_by_x;
                o_state[idx] = static_cast<U>(state);
                acc += state * C_[s_idx];
            }
            acc = simd_sum(acc);
            if (thread_index_in_simdgroup == 0) {
                out[d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
            }
            """)

    /// The one-token SSM update of `ssmUpdateKernel` with `computeDt` fused:
    /// `dtRaw` is the projected `dt` (`[n, 1, h]`) before the bias.
    static func scan(
        hiddenStates: MLXArray, ALog: MLXArray, B: MLXArray, C: MLXArray, D: MLXArray,
        dtRaw: MLXArray, dtBias: MLXArray, state: MLXArray, timeStepLimit: (Float, Float)
    ) -> (MLXArray, MLXArray) {
        let (n, _, h, d) = hiddenStates.shape4
        let (groups, stateSize) = (B.dim(-2), B.dim(-1))
        let outputs = scanKernel(
            [
                hiddenStates, ALog, B, C, D, dtRaw, dtBias,
                HostConstantArrays.float32([timeStepLimit.0, timeStepLimit.1]), state,
            ],
            template: [
                ("T", hiddenStates.dtype), ("U", state.dtype), ("Dh", d), ("Ds", stateSize),
                ("H", h), ("G", h / groups),
            ],
            grid: (32, d, h * n), threadGroup: (32, 8, 1),
            outputShapes: [[n, 1, h, d], state.shape],
            outputDTypes: [hiddenStates.dtype, state.dtype])
        return (outputs[0], outputs[1])
    }

    private static let gatedNormKernel = MLXFast.metalKernel(
        name: "nemotron_super_mamba_gated_norm",
        inputNames: ["y", "gate", "weight", "eps"],
        outputNames: ["out"],
        source: """
            constexpr int SIMD_SIZE = 32;
            constexpr int N_READS = 4;
            const uint gid = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            const uint axis_size = GS;

            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[SIMD_SIZE];

            float acc = 0;
            float thread_x[N_READS];
            const size_t base = size_t(gid) * axis_size + lid * N_READS;
            if (lid * N_READS + N_READS <= axis_size) {
                for (int i = 0; i < N_READS; i++) {
                    const T states = y[base + i] * nemotron_super_silu(gate[base + i]);
                    thread_x[i] = states;
                    acc += thread_x[i] * thread_x[i];
                }
            } else {
                for (int i = 0; i < N_READS; i++) {
                    if (lid * N_READS + i < axis_size) {
                        const T states = y[base + i] * nemotron_super_silu(gate[base + i]);
                        thread_x[i] = (float)states;
                    } else {
                        thread_x[i] = 0;
                    }
                    acc += thread_x[i] * thread_x[i];
                }
            }
            acc = simd_sum(acc);
            if (simd_group_id == 0) {
                local_sums[simd_lane_id] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_lane_id == 0) {
                local_sums[simd_group_id] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group_id == 0) {
                acc = simd_sum(local_sums[simd_lane_id]);
                if (simd_lane_id == 0) {
                    local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps[0]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const T unit = static_cast<T>(1);
            const uint column = (gid % GROUPS) * axis_size + lid * N_READS;
            for (int i = 0; i < N_READS; i++) {
                if (lid * N_READS + i < axis_size) {
                    const T normed = unit * static_cast<T>(thread_x[i] * local_inv_mean[0]);
                    out[base + i] = weight[column + i] * normed;
                }
            }
            """,
        header: header)

    /// `weight * rmsNorm((y * silu(gate)) per group of `groupSize`)` for `y`
    /// and `gate` `[..., groups * groupSize]` and `weight` of one dtype.
    static func gatedNorm(
        _ y: MLXArray, gate: MLXArray, weight: MLXArray, groupSize: Int, eps: Float
    ) -> MLXArray {
        let width = y.dim(-1)
        let groups = width / groupSize
        let rows = y.size / width
        let threads = ((groupSize + 3) / 4 + 31) / 32 * 32
        let output = gatedNormKernel(
            [
                y.reshaped(rows, width), gate.reshaped(rows, width), weight,
                HostConstantArrays.float32([eps]),
            ],
            template: [("T", y.dtype), ("GS", groupSize), ("GROUPS", groups)],
            grid: (threads * rows * groups, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[rows, width]], outputDTypes: [y.dtype])[0]
        return output.reshaped(y.shape)
    }

    /// Whether `gatedNorm` reproduces `NemotronHRMSNormGated` (MLX's
    /// single-row RMS kernel serves groups up to 4096).
    static func supportsGatedNorm(_ y: MLXArray, gate: MLXArray, weight: MLXArray, groupSize: Int)
        -> Bool
    {
        let width = y.dim(-1)
        return groupSize > 0 && groupSize <= 4096 && width % groupSize == 0
            && gate.shape == y.shape && weight.shape == [width]
            && [.bfloat16, .float16, .float32].contains(y.dtype) && gate.dtype == y.dtype
            && weight.dtype == y.dtype
    }
}
