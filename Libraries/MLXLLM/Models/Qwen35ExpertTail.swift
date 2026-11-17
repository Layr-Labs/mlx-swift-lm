// Copyright © 2026 Eigen Labs / Darkbloom.
//
// The Qwen 3.5 sparse block's end and the next residual step as one kernel on
// one-stream decode: the routed experts' weighted sum (`weightedExpertSum`:
// each product in the activation dtype, then MLX's small column reduction,
// each row plus zero added to the running sum in expert order), the shared
// expert's gate (`sigmoid(gate) * shared`, MLX's `Sigmoid` functor), their
// sum, the residual add and the next RMSNorm (MLX's `rms_single_row`). Today
// these are four launches on the critical path (the weighted product, the
// column sum, the combine and `ResidualRMSNorm`). Each value is the
// expression MLX's kernels write, so the outputs are the unfused ops' bits.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

enum Qwen35ExpertTail {
    /// Diagnostic only: `DARKBLOOM_QWEN35_FUSED_EXPERT_TAIL=0` keeps the
    /// sparse block's end and the residual norm separate.
    static let enabledByEnvironment: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_QWEN35_FUSED_EXPERT_TAIL"] != "0"
    /// A test or bench's own choice (nil: the environment's).
    @TaskLocal static var enabledOverride: Bool?
    static var isEnabled: Bool { enabledOverride ?? enabledByEnvironment }

    /// The sparse block's output before its sum: routed expert rows
    /// `[N, K, D]`, their weights `[N, K]`, the shared expert's gate `[N, 1]`
    /// and output `[N, D]`.
    struct Parts {
        let rows: MLXArray
        let weights: MLXArray
        let sharedGate: MLXArray
        let shared: MLXArray
        /// The block's output as the unfused ops write it.
        var output: MLXArray {
            let routed = weightedExpertSum(rows, weights)
            return routed.reshaped(shared.shape) + sigmoid(sharedGate) * shared
        }
    }

    /// Whether the kernel covers rows of `width` in `dtype` with `experts`
    /// routed experts on this device (one thread group per row; widths where
    /// MLX's norm divides by a power of two, so every mean form agrees; at
    /// most 8 experts, where MLX's small column reduction gives each row its
    /// own lane, so the rows are added in order).
    static func supports(width: Int, experts: Int, dtype: DType) -> Bool {
        isEnabled && Device.defaultDevice().deviceType == .gpu
            && ResidualRMSNorm.isEnabled
            && ResidualRMSNorm.verifiedFamilies.contains(AffineQuantizedVectors.family())
            && [.bfloat16, .float16].contains(dtype) && (4 ... 4096).contains(width)
            && width & (width - 1) == 0 && (1 ... 8).contains(experts)
    }

    /// `h = x + parts.output` and `rmsNorm(h, weight)`, the unfused ops' bits.
    static func apply(_ x: MLXArray, _ parts: Parts, weight: MLXArray, eps: Float) -> (
        sum: MLXArray, norm: MLXArray
    ) {
        let width = x.dim(-1)
        let rows = x.size / width
        let experts = parts.weights.dim(-1)
        let threads = 32 * (((width + 3) / 4 + 31) / 32)
        let flat = [rows, width]
        let outputs = kernel(
            [
                x.reshaped(flat), parts.rows.reshaped(rows, experts, width),
                parts.weights.reshaped(rows, experts), parts.sharedGate.reshaped(rows),
                parts.shared.reshaped(flat), weight,
                ResidualRMSNorm.Constants.shared.epsilon(eps),
            ],
            template: [("T", x.dtype), ("D", width), ("KE", experts)],
            grid: (rows * threads, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [flat, flat], outputDTypes: [x.dtype, x.dtype])
        return (outputs[0].reshaped(x.shape), outputs[1].reshaped(x.shape))
    }

    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_expert_tail", inputNames: ["x", "rows", "rw", "gate", "shared", "w", "eps"],
        outputNames: ["h", "n"], source: source, header: header)

    // MLX's `Sigmoid` functor (mlx/backend/metal/kernels/unary_ops.h).
    private static let header = """
        struct ExpertTailSigmoid {
          template <typename U>
          U operator()(U x) thread {
            auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
            return (x < 0) ? y : 1 - y;
          }
        };
        """

    // `ResidualRMSNorm`'s single-row body with the add formed in place.
    private static let source = """
        constexpr int N_READS = 4;
        constexpr int SIMD_SIZE = 32;
        const uint gid = threadgroup_position_in_grid.x;
        const uint lid = thread_position_in_threadgroup.x;
        const uint simd_lane_id = thread_index_in_simdgroup;
        const uint simd_group_id = simdgroup_index_in_threadgroup;
        const uint axis_size = D;
        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[SIMD_SIZE];
        const size_t row = size_t(gid) * axis_size;
        const T gated = ExpertTailSigmoid()(gate[gid]);
        float thread_x[N_READS];
        float acc = 0;
        const uint base = lid * N_READS;
        for (int i = 0; i < N_READS; i++) {
          if (base + i < axis_size) {
            T routed = T(0);
            for (int j = 0; j < KE; j++) {
              const size_t slot = size_t(gid) * KE + j;
              const T product = rows[slot * axis_size + base + i] * rw[slot];
              const T lane = product + T(0);
              routed = j == 0 ? lane : static_cast<T>(lane + routed);
            }
            const T shared_part = gated * shared[row + base + i];
            const T added = routed + shared_part;
            const T sum = x[row + base + i] + added;
            h[row + base + i] = sum;
            thread_x[i] = sum;
          } else {
            thread_x[i] = 0;
          }
          acc += thread_x[i] * thread_x[i];
        }
        acc = simd_sum(acc);
        if (simd_group_id == 0) { local_sums[simd_lane_id] = 0; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane_id == 0) { local_sums[simd_group_id] = acc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group_id == 0) {
          acc = simd_sum(local_sums[simd_lane_id]);
          if (simd_lane_id == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps[0]);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < N_READS; i++) {
          if (base + i < axis_size) {
            n[row + base + i] = w[base + i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
          }
        }
        """
}
