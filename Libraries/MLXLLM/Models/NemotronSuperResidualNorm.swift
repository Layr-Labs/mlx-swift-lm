import MLX
import MLXLMCommon
import MLXNN

/// A Super block boundary as one kernel: the residual add `x + y` (or, for a
/// MoE layer, `x + (routed + shared)`) and the next RMS norm of the sum
/// (MLX's single-row kernel, weight applied as it does), bit-identical to the
/// adds followed by `RMSNorm`; returns both.
enum NemotronSuperResidualNorm {
    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_super_residual_rms_norm",
        inputNames: ["x", "y", "extra", "w", "eps"],
        outputNames: ["sum", "normed"],
        source: """
            constexpr int SIMD_SIZE = 32;
            constexpr int N_READS = 4;
            const uint gid = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            const uint axis_size = D;

            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[SIMD_SIZE];

            float acc = 0;
            float thread_x[N_READS];
            const size_t base = size_t(gid) * axis_size + lid * N_READS;
            for (int i = 0; i < N_READS; i++) {
                if (lid * N_READS + i < axis_size) {
                    const T mixed = EXTRA ? static_cast<T>(y[base + i] + extra[base + i])
                                          : y[base + i];
                    const T total = x[base + i] + mixed;
                    sum[base + i] = total;
                    thread_x[i] = total;
                } else {
                    thread_x[i] = 0;
                }
                acc += thread_x[i] * thread_x[i];
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
            for (int i = 0; i < N_READS; i++) {
                if (lid * N_READS + i < axis_size) {
                    normed[base + i] =
                        w[lid * N_READS + i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
                }
            }
            """)

    /// Whether `apply` reproduces `x + (y + extra)` then `RMSNorm(weight, eps)`.
    static func supports(_ x: MLXArray, _ y: MLXArray, extra: MLXArray? = nil, weight: MLXArray)
        -> Bool
    {
        let width = x.dim(-1)
        return width > 0 && width <= 4096 && x.shape == y.shape && weight.shape == [width]
            && [.bfloat16, .float16, .float32].contains(x.dtype) && y.dtype == x.dtype
            && weight.dtype == x.dtype
            && (extra.map { $0.shape == x.shape && $0.dtype == x.dtype } ?? true)
    }

    /// `(x + (y + extra), rmsNorm of that)`; without `extra`, `x + y`.
    static func apply(
        _ x: MLXArray, _ y: MLXArray, extra: MLXArray? = nil, weight: MLXArray, eps: Float
    ) -> (sum: MLXArray, normed: MLXArray) {
        let width = x.dim(-1)
        let rows = x.size / width
        let threads = ((width + 3) / 4 + 31) / 32 * 32
        let flat = y.reshaped(rows, width)
        let outputs = kernel(
            [
                x.reshaped(rows, width), flat, extra?.reshaped(rows, width) ?? flat, weight,
                HostConstantArrays.float32([eps]),
            ],
            template: [("T", x.dtype), ("D", width), ("EXTRA", extra != nil)],
            grid: (threads * rows, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[rows, width], [rows, width]], outputDTypes: [x.dtype, x.dtype])
        return (outputs[0].reshaped(x.shape), outputs[1].reshaped(x.shape))
    }

    /// `(x + (y + extra), norm of that)` as `NemotronHBackbone` forms a block
    /// boundary: `apply` where it reproduces the chain, else the adds and `norm`.
    static func boundary(_ x: MLXArray, _ y: MLXArray, extra: MLXArray? = nil, norm: RMSNorm)
        -> (sum: MLXArray, normed: MLXArray)
    {
        if supports(x, y, extra: extra, weight: norm.weight) {
            return apply(x, y, extra: extra, weight: norm.weight, eps: norm.eps)
        }
        let sum = x + (extra.map { y + $0 } ?? y)
        return (sum, norm(sum))
    }
}
