// SPDX-License-Identifier: Apache-2.0 AND MIT
// oMLX PR #4033, c10e4ad57b180fa1aa88bf40532cba4362b6851a.
// Modified for the selected DarkBloom core: partial outputs remain FP32,
// matching sdpa_vector_2pass_fp32partials, rather than upstream BF16 partials.
// Apple/MIT notice is preserved in MiMoV26NAXMetalSources.swift;
// oMLX Apache notice/license is docs/mimo-v26/LICENSE-OMLX.
enum MiMoV26DecodeRowsMetalSources {
    static let pass1 = #"""

  constexpr int BD = 32;
  constexpr int QK = D / BD;
  constexpr int VP = V / BD;
  typedef float U;

  const int kv_head_idx = threadgroup_position_in_grid.x;
  const int batch_idx = threadgroup_position_in_grid.y;
  const int block_idx = threadgroup_position_in_grid.z;
  const int g = simdgroup_index_in_threadgroup;
  const int simd_lid = thread_index_in_simdgroup;
  const int N = keys_shape[2];
  const int num_q_heads = NKV * G;
  const int q_head_idx = G * kv_head_idx + g;
  const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;

  U q[ROWS][QK];
  U o[ROWS][VP];
  U max_score[ROWS];
  U sum_exp_score[ROWS];
  const U sc = scale[0];
  for (int r = 0; r < ROWS; r++) {
    const device T* qp = queries + batch_idx * queries_strides[0] +
        q_head_idx * queries_strides[1] + r * queries_strides[2] +
        simd_lid * QK * queries_strides[3];
    for (int i = 0; i < QK; i++) {
      q[r][i] = static_cast<U>(sc) * qp[i * queries_strides[3]];
    }
    for (int i = 0; i < VP; i++) {
      o[r][i] = 0;
    }
    max_score[r] = Limits<U>::finite_min;
    sum_exp_score[r] = 0;
    if (HAS_SINKS && block_idx == 0) {
      max_score[r] = static_cast<U>(sinks[q_head_idx]);
      sum_exp_score[r] = 1;
    }
  }

  const int64_t ks = keys_strides[2];
  const int64_t vs = values_strides[2];
  const int64_t k3 = keys_strides[3];
  const int64_t v3 = values_strides[3];
  const device T* kp = keys + batch_idx * keys_strides[0] +
      kv_head_idx * keys_strides[1] + block_idx * ks + simd_lid * QK * k3;
  const device T* vp = values + batch_idx * values_strides[0] +
      kv_head_idx * values_strides[1] + block_idx * vs + simd_lid * VP * v3;
  auto mp = mask + (MASK_KIND ? (size_t(batch_idx) * ROWS * N) : 0);

#define ROW_STEP(KR, VR, KEY)                                             \
    for (int r = 0; r < ROWS; r++) {                                        \
      bool use_key = true;                                                  \
      if (CAUSAL) {                                                         \
        use_key = (KEY) <= (N - ROWS + r);                                  \
      } else if (MASK_KIND == 1) {                                          \
        use_key = mp[r * N + (KEY)];                                        \
      } else if (MASK_KIND == 2) {                                          \
        use_key = (mp[r * N + (KEY)] >= Limits<T>::finite_min);             \
      }                                                                     \
      if (use_key) {                                                        \
        U score = 0;                                                        \
        for (int j = 0; j < QK; j++) {                                      \
          score += q[r][j] * KR[j];                                         \
        }                                                                   \
        score = simd_sum(score);                                            \
        if (MASK_KIND == 2) {                                               \
          score += mp[r * N + (KEY)];                                       \
        }                                                                   \
        U new_max = max(max_score[r], score);                               \
        U factor = fast::exp(max_score[r] - new_max);                       \
        U exp_score = fast::exp(score - new_max);                           \
        max_score[r] = new_max;                                             \
        sum_exp_score[r] = sum_exp_score[r] * factor + exp_score;           \
        for (int j = 0; j < VP; j++) {                                      \
          o[r][j] = o[r][j] * factor + exp_score * VR[j];                   \
        }                                                                   \
      }                                                                     \
    }

#define LOAD_KV(KR, VR, K3, V3)                                           \
    for (int j = 0; j < QK; j++) {                                          \
      KR[j] = kp[j * (K3)];                                                 \
    }                                                                       \
    for (int j = 0; j < VP; j++) {                                          \
      VR[j] = vp[j * (V3)];                                                 \
    }

// Next key of the block loaded while the current one is processed.
#define PASS1_LOOP(K3, V3)                                                \
  {                                                                         \
    T ka[QK], kb[QK];                                                       \
    T va[VP], vb[VP];                                                       \
    int key = block_idx;                                                    \
    if (n_iter > 0) {                                                       \
      LOAD_KV(ka, va, K3, V3)                                               \
    }                                                                       \
    int it = 0;                                                             \
    for (; it + 1 < n_iter; it += 2) {                                      \
      kp += BLOCKS * ks;                                                    \
      vp += BLOCKS * vs;                                                    \
      LOAD_KV(kb, vb, K3, V3)                                               \
      ROW_STEP(ka, va, key)                                                 \
      key += BLOCKS;                                                        \
      if (it + 2 < n_iter) {                                                \
        kp += BLOCKS * ks;                                                  \
        vp += BLOCKS * vs;                                                  \
        LOAD_KV(ka, va, K3, V3)                                             \
      }                                                                     \
      ROW_STEP(kb, vb, key)                                                 \
      key += BLOCKS;                                                        \
    }                                                                       \
    if (it < n_iter) {                                                      \
      ROW_STEP(ka, va, key)                                                 \
    }                                                                       \
  }

  const int n_iter = block_idx < N ? (N - 1 - block_idx) / BLOCKS + 1 : 0;
  if (k3 == 1 && v3 == 1) {
    PASS1_LOOP(1, 1)
  } else {
    PASS1_LOOP(k3, v3)
  }

  for (int r = 0; r < ROWS; r++) {
    const int o_offset = q_batch_head_idx * ROWS + r;
    if (simd_lid == 0) {
      sums[o_offset * BLOCKS + block_idx] = sum_exp_score[r];
      maxs[o_offset * BLOCKS + block_idx] = max_score[r];
    }
    device float* op = partials + (size_t(o_offset) * BLOCKS + block_idx) * V + simd_lid * VP;
    for (int j = 0; j < VP; j++) {
      op[j] = o[r][j];
    }
  }

"""#
    static let pass2 = #"""

  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = V / BD;
  typedef float U;

  thread U o[elem_per_thread] = {0};
  threadgroup U outputs[BN * BD];

  const int head_idx = threadgroup_position_in_grid.x;
  const int q_seq_idx = threadgroup_position_in_grid.y;
  const int simd_gid = simdgroup_index_in_threadgroup;
  const int simd_lid = thread_index_in_simdgroup;
  const int q_offset = head_idx * ROWS + q_seq_idx;
  const device float* pp = partials + size_t(q_offset) * BLOCKS * V + simd_gid * V +
      simd_lid * elem_per_thread;
  const device float* sp = sums + q_offset * BLOCKS;
  const device float* mp = maxs + q_offset * BLOCKS;
  const int bi = head_idx / NQ;
  const int hi = head_idx % NQ;
  device T* op = out + ((size_t(bi) * ROWS + q_seq_idx) * NQ + hi) * V +
      simd_gid * elem_per_thread;

  U sum_exp_score = 0.0;
  U max_score = Limits<U>::finite_min;

  for (int b = 0; b < BLOCKS / BN; ++b) {
    max_score = max(max_score, mp[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);

  for (int b = 0; b < BLOCKS / BN; ++b) {
    U factor = fast::exp(mp[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sp[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);

  for (int b = 0; b < BLOCKS / BN; ++b) {
    U factor = fast::exp(mp[simd_gid] - max_score);
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(pp[i]);
    }
    mp += BN;
    sp += BN;
    pp += BN * V;
  }

  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      op[i] = static_cast<T>(o[i]);
    }
  }

"""#
}
