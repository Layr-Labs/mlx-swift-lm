// Copyright © 2026 Eigen Labs / Darkbloom.
//
// A pre-norm transformer's residual step as one kernel: the residual sum
// h = x + T(r) (T the activation dtype; r may be a wider sum, such as a
// tensor-parallel FP32 all-sum, rounded to T first) and the next RMSNorm of
// h, both written. It replaces three launches (the cast, the add and the
// norm) with one, and computes each value with the expressions of MLX's own
// kernels: the cast and add as MLX's copy and binary kernels write them
// (`static_cast<T>(r)`, `T + T`), and the norm as MLX's `rms_single_row`
// (rows up to 4096 wide) or `rms_looped` (wider; a 1024-thread group, the
// pipeline's maximum MLX dispatches) reduce and scale, so the outputs are the
// unfused chain's bits. Byte parity against the unfused MLX ops is checked
// per GPU family before a family is enabled (`verifiedFamilies`).

import Foundation
import MLX
import MLXFast
import MLXNN

public enum ResidualRMSNorm {
    /// GPU families measured byte-equal to MLX's unfused cast, add and
    /// RMSNorm (M3 Ultra: 15, M5 Max: 17). Others keep the unfused ops.
    public static let verifiedFamilies: Set<Int> = [15, 17]

    /// Diagnostic only: `DARKBLOOM_RESIDUAL_RMS_NORM=0` keeps the unfused ops.
    public static let enabledByEnvironment: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_RESIDUAL_RMS_NORM"] != "0"
    /// A test or bench's own choice (nil: the environment's).
    @TaskLocal public static var enabledOverride: Bool?
    public static var isEnabled: Bool { enabledOverride ?? enabledByEnvironment }

    /// Whether `apply` covers these operands on the current default device.
    public static func supports(_ x: MLXArray, _ residual: MLXArray, weight: MLXArray) -> Bool {
        guard isEnabled, Device.defaultDevice().deviceType == .gpu,
            verifiedFamilies.contains(AffineQuantizedVectors.family()),
            x.ndim >= 1, residual.shape == x.shape, weight.ndim == 1, weight.dim(0) == x.dim(-1),
            [.bfloat16, .float16, .float32].contains(x.dtype), weight.dtype == x.dtype,
            residual.dtype == x.dtype || residual.dtype == .float32
        else { return false }
        return x.dim(-1) >= 1 && x.size > 0
    }

    /// `(h, norm)` with `h = x + residual.asType(x.dtype)` and
    /// `norm = MLXFast.rmsNorm(h, weight: weight, eps: eps)`, the unfused
    /// ops' bits; nil where not covered.
    public static func apply(
        _ x: MLXArray, _ residual: MLXArray, weight: MLXArray, eps: Float
    ) -> (sum: MLXArray, norm: MLXArray)? {
        guard supports(x, residual, weight: weight) else { return nil }
        let width = x.dim(-1)
        let rows = x.size / width
        let looped = width > 4096
        let threads = looped ? 1024 : 32 * (((width + 3) / 4 + 31) / 32)
        let outputs = kernel(
            [
                x.reshaped([rows, width]), residual.reshaped([rows, width]), weight,
                Constants.shared.epsilon(eps),
            ],
            template: [("T", x.dtype), ("RT", residual.dtype), ("D", width), ("LOOPED", looped)],
            grid: (rows * threads, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[rows, width], [rows, width]], outputDTypes: [x.dtype, x.dtype])
        return (outputs[0].reshaped(x.shape), outputs[1].reshaped(x.shape))
    }

    /// Cached `[1]` float32 kernel constants (an epsilon), one per value.
    public final class Constants: @unchecked Sendable {
        public static let shared = Constants()
        private let lock = NSLock()
        private var values: [UInt32: MLXArray] = [:]

        public func epsilon(_ value: Float) -> MLXArray {
            lock.lock()
            defer { lock.unlock() }
            if let cached = values[value.bitPattern] { return cached }
            let made = MLXArray([value])
            eval(made)
            values[value.bitPattern] = made
            return made
        }
    }

    /// `res` is a fenced input (`MLXFast.metalKernel(...fencedInputs:)`): when
    /// it is a tensor-parallel all-sum made on the CPU stream in the same
    /// evaluation, this kernel waits for the CPU itself (with
    /// `MLX_METAL_FENCED_INPUTS=1` and fast fences) instead of MLX's separate
    /// fence-wait launch before it; any other residual has no fence and is
    /// read at once. Only where the GPU waits changes, never a value.
    private static let kernel = MLXFast.metalKernel(
        name: "residual_rms_norm", inputNames: ["x", "res", "w", "eps"],
        outputNames: ["h", "n"], source: source, fencedInputs: ["res"])

    // The bodies of MLX's binary Add, astype copy and rms_single_row /
    // rms_looped (mlx/backend/metal/kernels/rms_norm.metal, N_READS = 4),
    // with the sum stored once and the norm reading it as MLX's reads x.
    private static let source = """
        constexpr int N_READS = 4;
        constexpr int SIMD_SIZE = 32;
        const uint gid = threadgroup_position_in_grid.x;
        const uint lid = thread_position_in_threadgroup.x;
        const uint lsize = threads_per_threadgroup.x;
        const uint simd_lane_id = thread_index_in_simdgroup;
        const uint simd_group_id = simdgroup_index_in_threadgroup;
        const uint axis_size = D;
        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[SIMD_SIZE];
        // Every thread reaches this before reading `res` (it holds a threadgroup barrier).
        MLX_FENCED_INPUT_WAIT(res);

        auto xr = x + gid * size_t(axis_size);
        auto rr = res + gid * size_t(axis_size);
        device T* hr = h + gid * size_t(axis_size);
        device T* nr = n + gid * size_t(axis_size);

        float acc = 0;
        if (!LOOPED) {
          float thread_x[N_READS];
          const uint base = lid * N_READS;
          if (base + N_READS <= axis_size) {
            for (int i = 0; i < N_READS; i++) {
              const T a = xr[base + i];
              const T b = static_cast<T>(rr[base + i]);
              const T sum = a + b;
              hr[base + i] = sum;
              thread_x[i] = sum;
              acc += thread_x[i] * thread_x[i];
            }
          } else {
            for (int i = 0; i < N_READS; i++) {
              if (base + i < axis_size) {
                const T a = xr[base + i];
                const T b = static_cast<T>(rr[base + i]);
                const T sum = a + b;
                hr[base + i] = sum;
                thread_x[i] = sum;
              } else {
                thread_x[i] = 0;
              }
              acc += thread_x[i] * thread_x[i];
            }
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
          if (base + N_READS <= axis_size) {
            for (int i = 0; i < N_READS; i++) {
              nr[base + i] = w[base + i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
            }
          } else {
            for (int i = 0; i < N_READS; i++) {
              if (base + i < axis_size) {
                nr[base + i] = w[base + i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
              }
            }
          }
        } else {
          for (uint r = 0; r < axis_size; r += lsize * N_READS) {
            const uint base = r + lid * N_READS;
            if (base + N_READS <= axis_size) {
              for (int i = 0; i < N_READS; i++) {
                const T a = xr[base + i];
                const T b = static_cast<T>(rr[base + i]);
                const T sum = a + b;
                hr[base + i] = sum;
                float xi = sum;
                acc += xi * xi;
              }
            } else {
              for (int i = 0; i < N_READS; i++) {
                if (base + i < axis_size) {
                  const T a = xr[base + i];
                  const T b = static_cast<T>(rr[base + i]);
                  const T sum = a + b;
                  hr[base + i] = sum;
                  float xi = sum;
                  acc += xi * xi;
                }
              }
            }
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
          // Each thread reads back the sums it wrote (the same lanes and order as
          // the reduction): no other thread's writes are read.
          for (uint r = 0; r < axis_size; r += lsize * N_READS) {
            const uint base = r + lid * N_READS;
            if (base + N_READS <= axis_size) {
              for (int i = 0; i < N_READS; i++) {
                nr[base + i] = w[base + i] * static_cast<T>(hr[base + i] * local_inv_mean[0]);
              }
            } else {
              for (int i = 0; i < N_READS; i++) {
                if (base + i < axis_size) {
                  nr[base + i] = w[base + i] * static_cast<T>(hr[base + i] * local_inv_mean[0]);
                }
              }
            }
          }
        }
        """
}
