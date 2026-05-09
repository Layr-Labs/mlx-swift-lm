// Copyright © 2026 Eigen Labs.
//
// Fused gate+up+SwiGLU and small-M quantized matmul kernels for the MTP verify
// forward path. The verify forward runs with M=2 (next_main + draft token),
// which stalls MLX's stock qmv_fast_impl kernel (tuned for large M). These
// kernels are faster at M≤8 by: (a) fusing gate and up projections into a
// single pass — halving bandwidth on the projection weights — and (b) using
// simdgroup matrix-multiply accumulate for the down projection.
//
// Ported from:
//   https://github.com/youssofal/MTPLX
//   mtplx/kernels/verify_mlp_fused.py
//   — _gate_up_swiglu_qmv4_kernel (non-rowwise, MAX_M=6)
//   — _small_m_qmm4_kernel (BM=8 simdgroup-MMA)

import MLX
import MLXNN

// MARK: - Eligibility

/// Returns true when the fused gate+up+SwiGLU kernel can run.
/// Requires: 4-bit affine QuantizedLinear, M ∈ [1,6], K and N divisible by 32,
/// single batch dimension, dtype bfloat16 or float16.
func gateUpSwiGLUFusedEligible(x: MLXArray, gateProj: Linear, upProj: Linear) -> Bool {
    guard let gate = gateProj as? QuantizedLinear,
          let up   = upProj   as? QuantizedLinear else { return false }
    guard gate.bits == 4, up.bits == 4 else { return false }
    guard gate.groupSize == up.groupSize else { return false }
    guard [32, 64, 128].contains(gate.groupSize) else { return false }
    guard gate.biases != nil, up.biases != nil else { return false }
    guard x.dtype == .bfloat16 || x.dtype == .float16 else { return false }
    guard x.shape.count >= 2 else { return false }
    let M = x.shape[x.shape.count - 2]
    guard M >= 1, M <= 6 else { return false }
    let K = x.shape[x.shape.count - 1]
    let N = gate.weight.shape[0]
    guard K % 32 == 0, N % 32 == 0 else { return false }
    guard Array(x.shape.dropLast(2)).reduce(1, *) == 1 else { return false }
    return true
}

/// Returns true when the small-M simdgroup-MMA kernel can run.
/// Requires: 4-bit affine QuantizedLinear, M ∈ [1,8], K and N divisible by 32.
func smallMQuantizedLinearEligible(x: MLXArray, layer: Linear) -> Bool {
    guard let ql = layer as? QuantizedLinear else { return false }
    guard ql.bits == 4 else { return false }
    guard [32, 64, 128].contains(ql.groupSize) else { return false }
    guard ql.biases != nil else { return false }
    guard x.dtype == .bfloat16 || x.dtype == .float16 else { return false }
    guard x.shape.count >= 2 else { return false }
    let M = x.shape[x.shape.count - 2]
    guard M >= 1, M <= 8 else { return false }
    let K = x.shape[x.shape.count - 1]
    let N = ql.weight.shape[0]
    guard K % 32 == 0, N % 32 == 0 else { return false }
    guard Array(x.shape.dropLast(2)).reduce(1, *) == 1 else { return false }
    return true
}

// MARK: - Gate+Up+SwiGLU fused kernel

private let _gateUpHeader = """
    using namespace metal;

    constant constexpr int SIMD_SIZE = 32;
    constant constexpr int PACK_FACTOR = 8;
    constant constexpr int PACKS_PER_THREAD = 2;
    constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
    constant constexpr int BYTES_PER_PACK = 4;
    constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
    constant constexpr int RESULTS_PER_SIMDGROUP = 4;
    constant constexpr int NUM_SIMDGROUPS = 2;
    constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;
    constant constexpr int MAX_M = 6;

    template <typename T>
    inline T sigmoid_mlx_exact(T x) {
      auto y = 1 / (1 + metal::exp(metal::abs(x)));
      return (x < T(0)) ? y : 1 - y;
    }

    template <typename T>
    inline T swiglu_mlx_exact(T gate, T up) {
      T silu = gate * sigmoid_mlx_exact<T>(gate);
      return T(silu * up);
    }

    template <typename T>
    inline float load_vector4_exact(const device T* x, thread float* x_thread) {
      float sum = 0.0f;
      for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
        sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
        x_thread[i]     = x[i];
        x_thread[i + 1] = x[i + 1] / 16.0f;
        x_thread[i + 2] = x[i + 2] / 256.0f;
        x_thread[i + 3] = x[i + 3] / 4096.0f;
      }
      return sum;
    }

    inline float qdot4_exact(
        const device uint8_t* w,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum) {
      const device uint16_t* ws = (const device uint16_t*)w;
      float accum = 0.0f;
      for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
        uint16_t packed = ws[i];
        accum +=
          x_thread[4 * i]     * float(packed & 0x000f) +
          x_thread[4 * i + 1] * float(packed & 0x00f0) +
          x_thread[4 * i + 2] * float(packed & 0x0f00) +
          x_thread[4 * i + 3] * float(packed & 0xf000);
      }
      return scale * accum + sum * bias;
    }
    """

private let _gateUpSource = """
    uint n_tile   = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int M = int(M_size[0]);
    int K = int(K_size[0]);
    int N = int(N_size[0]);
    constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
    int out_row       = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K / GS;

    const device uint8_t* gate_w_base =
      (const device uint8_t*)gate_w + out_row * in_vec_size_w
      + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device uint8_t* up_w_base =
      (const device uint8_t*)up_w + out_row * in_vec_size_w
      + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* gate_scales_base =
      gate_scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* gate_biases_base =
      gate_biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* up_scales_base =
      up_scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* up_biases_base =
      up_biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;

    float gate_result[MAX_M][RESULTS_PER_SIMDGROUP];
    float up_result[MAX_M][RESULTS_PER_SIMDGROUP];
    float x_thread[MAX_M][VALUES_PER_THREAD];
    float x_sum[MAX_M];

    for (int m = 0; m < MAX_M; ++m) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        gate_result[m][row] = 0.0f;
        up_result[m][row]   = 0.0f;
      }
    }

    for (int k_block = 0; k_block < K; k_block += BLOCK_SIZE) {
      for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
          const device T* x_m =
            x + m * K + k_block + int(simd_lid) * VALUES_PER_THREAD;
          x_sum[m] = load_vector4_exact<T>(x_m, x_thread[m]);
        }
      }

      const device uint8_t* gate_w_block =
        gate_w_base + k_block * BYTES_PER_PACK / PACK_FACTOR;
      const device uint8_t* up_w_block =
        up_w_base   + k_block * BYTES_PER_PACK / PACK_FACTOR;
      const device T* gate_scales_block = gate_scales_base + k_block / GS;
      const device T* gate_biases_block = gate_biases_base + k_block / GS;
      const device T* up_scales_block   = up_scales_base   + k_block / GS;
      const device T* up_biases_block   = up_biases_base   + k_block / GS;

      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        int n = out_row + row;
        if (n < N) {
          const device uint8_t* gate_w_row = gate_w_block + row * in_vec_size_w;
          const device uint8_t* up_w_row   = up_w_block   + row * in_vec_size_w;
          const device T* gate_sc_row = gate_scales_block + row * in_vec_size_g;
          const device T* gate_bs_row = gate_biases_block + row * in_vec_size_g;
          const device T* up_sc_row   = up_scales_block   + row * in_vec_size_g;
          const device T* up_bs_row   = up_biases_block   + row * in_vec_size_g;
          float gate_scale = float(gate_sc_row[0]);
          float gate_bias  = float(gate_bs_row[0]);
          float up_scale   = float(up_sc_row[0]);
          float up_bias    = float(up_bs_row[0]);

          for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
              gate_result[m][row] += qdot4_exact(
                gate_w_row, x_thread[m], gate_scale, gate_bias, x_sum[m]);
              up_result[m][row] += qdot4_exact(
                up_w_row, x_thread[m], up_scale, up_bias, x_sum[m]);
            }
          }
        }
      }
    }

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      if (n < N) {
        for (int m = 0; m < MAX_M; ++m) {
          if (m < M) {
            float gate_sum = simd_sum(gate_result[m][row]);
            float up_sum   = simd_sum(up_result[m][row]);
            if (simd_lid == 0) {
              T gate_val = T(gate_sum);
              T up_val   = T(up_sum);
              y[m * N + n] = swiglu_mlx_exact<T>(gate_val, up_val);
            }
          }
        }
      }
    }
    """

/// Computes swiglu(gateProj(x), upProj(x)) in a single kernel pass.
/// Loads gate and up weights simultaneously, halving memory bandwidth vs two
/// separate matmuls. M rows (up to 6) are accumulated within each threadgroup.
func gateUpSwiGLUFused(
    x: MLXArray,
    gateProj: QuantizedLinear,
    upProj: QuantizedLinear
) -> MLXArray {
    let M = x.shape[x.shape.count - 2]
    let K = x.shape[x.shape.count - 1]
    let N = gateProj.weight.shape[0]
    let x2 = x.reshaped([M, K])

    // NUM_SIMDGROUPS=2, BN=8: grid_y = 2 * ceil(N/8)
    let gridY = 2 * ((N + 7) / 8)

    let kernel = MLXFast.metalKernel(
        name: "mlxlm_gate_up_swiglu_qmv4_gs\(gateProj.groupSize)_\(x.dtype)",
        inputNames: [
            "x", "gate_w", "gate_scales", "gate_biases",
            "up_w", "up_scales", "up_biases",
            "M_size", "K_size", "N_size",
        ],
        outputNames: ["y"],
        source: _gateUpSource,
        header: _gateUpHeader
    )

    let results = kernel(
        [
            x2,
            gateProj.weight, gateProj.scales, gateProj.biases!,
            upProj.weight, upProj.scales, upProj.biases!,
            MLXArray([Int32(M)]), MLXArray([Int32(K)]), MLXArray([Int32(N)]),
        ],
        template: [("T", x.dtype), ("GS", gateProj.groupSize)],
        grid: (32, gridY, 1),
        threadGroup: (32, 2, 1),
        outputShapes: [[M, N]],
        outputDTypes: [x.dtype]
    )
    return results[0].reshaped(Array(x.shape.dropLast()) + [N])
}

// MARK: - Small-M quantized matmul (down projection)

private let _smallMQmmSource = """
    using namespace metal;
    constexpr int BM = 8;
    constexpr int BN = 32;
    constexpr int BK = 32;
    constexpr int BK_SUB = 8;

    uint tid   = thread_position_in_threadgroup.x;
    uint sg_id = tid / 32;
    uint tg_n  = threadgroup_position_in_grid.y;

    int K = int(K_size[0]);
    int N = int(N_size[0]);
    int K_by_8  = K / 8;
    int K_by_gs = K / GS;
    int n0 = int(tg_n) * BN;

    threadgroup T B_tile[BK * BN];

    simdgroup_matrix<T, 8, 8> a, b_L, b_R;
    simdgroup_matrix<float, 8, 8> c_L = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c_R = simdgroup_matrix<float, 8, 8>(0.0f);

    int t_a = int(tid);
    int t_b = int(tid) + 64;
    int dq_k_a = t_a / BN, dq_n_a = t_a % BN;
    int dq_k_b = t_b / BN, dq_n_b = t_b % BN;
    int sg_n_off = int(sg_id) * 16;

    for (int k0 = 0; k0 < K; k0 += BK) {
        {
            int n_global = n0 + dq_n_a;
            int k_base   = k0 + dq_k_a * 8;
            uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
            float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
            float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
            for (int ki = 0; ki < 8; ++ki) {
                uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                B_tile[(dq_k_a * 8 + ki) * BN + dq_n_a] = T(float(nib) * s + b);
            }
        }
        {
            int n_global = n0 + dq_n_b;
            int k_base   = k0 + dq_k_b * 8;
            uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
            float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
            float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
            for (int ki = 0; ki < 8; ++ki) {
                uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                B_tile[(dq_k_b * 8 + ki) * BN + dq_n_b] = T(float(nib) * s + b);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int ks = 0; ks < BK / BK_SUB; ++ks) {
            simdgroup_load(a,   x      + k0 + ks * BK_SUB,                 K);
            simdgroup_load(b_L, B_tile + ks * BK_SUB * BN + sg_n_off,      BN);
            simdgroup_load(b_R, B_tile + ks * BK_SUB * BN + sg_n_off + 8,  BN);
            simdgroup_multiply_accumulate(c_L, a, b_L, c_L);
            simdgroup_multiply_accumulate(c_R, a, b_R, c_R);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_matrix<T, 8, 8> c_L_T, c_R_T;
    c_L_T.thread_elements()[0] = T(c_L.thread_elements()[0]);
    c_L_T.thread_elements()[1] = T(c_L.thread_elements()[1]);
    c_R_T.thread_elements()[0] = T(c_R.thread_elements()[0]);
    c_R_T.thread_elements()[1] = T(c_R.thread_elements()[1]);
    simdgroup_store(c_L_T, y + n0 + sg_n_off,     N);
    simdgroup_store(c_R_T, y + n0 + sg_n_off + 8, N);
    """

/// Computes layer(x) using BM=8 simdgroup matrix-multiply for M≤8.
/// Pads the input to M=8, runs the dequantize+SIMD-MMA kernel, then slices
/// the actual M rows from the output.
func smallMQuantizedLinear(x: MLXArray, layer: QuantizedLinear) -> MLXArray {
    let M = x.shape[x.shape.count - 2]
    let K = x.shape[x.shape.count - 1]
    let N = layer.weight.shape[0]

    var x2 = x.reshaped([M, K])
    if M < 8 {
        x2 = concatenated([x2, MLXArray.zeros([8 - M, K], dtype: x.dtype)], axis: 0)
    }

    let kernel = MLXFast.metalKernel(
        name: "mlxlm_small_m_qmm4_bm8_gs\(layer.groupSize)_\(x.dtype)",
        inputNames: ["x", "w_q", "scales", "biases", "M_size", "K_size", "N_size"],
        outputNames: ["y"],
        source: _smallMQmmSource
    )

    let results = kernel(
        [
            x2,
            layer.weight, layer.scales, layer.biases!,
            MLXArray([Int32(M)]), MLXArray([Int32(K)]), MLXArray([Int32(N)]),
        ],
        template: [("T", x.dtype), ("GS", layer.groupSize)],
        grid: (64, N / 32, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[8, N]],
        outputDTypes: [x.dtype]
    )
    let y8 = results[0]
    let y = M < 8 ? y8[0 ..< M, 0...] : y8
    return y.reshaped(Array(x.shape.dropLast()) + [N])
}
