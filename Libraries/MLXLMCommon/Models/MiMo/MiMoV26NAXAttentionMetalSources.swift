// SPDX-License-Identifier: Apache-2.0 AND MIT
// Copyright 2025 oMLX contributors; Copyright © 2024-2025 Apple Inc.
// Adapted from PR #3994 head 1c487861d1c1d4c82a2b2920dd69c064eec85e18.
// Modified: native-rounded, three-pass baseline arithmetic; runtime inner-head strides.
// Ordered phase/range FP32 save/restore adapted from #4032
// 12e5084f8cf888bb6e846147929419f044802343; no online-softmax/head split.
// Apple/MIT primitive header and license: MiMoV26NAXMetalSources.swift.
// oMLX license: docs/mimo-v26/LICENSE-OMLX.

enum MiMoV26NAXAttentionMetalSources {
    static let header = #"""
using namespace mlx::steel;
#define OMLX_NAX_UNROLL STEEL_PRAGMA_UNROLL
namespace omlx_nax {

// Scalar parameters of one attention call (MLX's AttnParams without the
// strides, which the kernel reads from the inputs' own stride vectors).
struct AttnParams {
  int B;
  int H;
  int D;
  int qL;
  int kL;
  int gqa_factor;
  float scale;
  int NQ;
  int NK;
  int NQ_aligned;
  int NK_aligned;
  int qL_rem;
  int kL_rem;
  int qL_off;
};

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp(x - y);
  }
};


// Unlike the Python wrapper, this accepts runtime inner-dimension strides.
// MLX lazy-array strides must not be read on the host before evaluation.
template <typename Tile, typename T>
METAL_FUNC void load_head_rows(
    thread Tile& tile, const device T* src, int row_stride,
    int64_t head_stride, short live_rows) {
  if (head_stride == 1) {
    if (live_rows >= Tile::kRows) {
      tile.load(src, row_stride);
    } else {
      tile.load_rows(src, row_stride, live_rows);
    }
    return;
  }
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < Tile::kTileRows; ++i) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < Tile::kTileCols; ++j) {
      using Frag = typename Tile::NAXFrag_t;
      Frag::load_rows(
          tile.frag_at(i, j), src, row_stride, head_stride,
          live_rows, i * Tile::kFragRows, j * Tile::kFragCols);
    }
  }
}

// MLX's attention_nax with a value head dim BDV <= BD. Only the parameter
// plumbing differs: strides come from the inputs, the output is written as
// [B, qL, H, BDV] rows (MLX's SDPA output layout), function constants are
// template arguments, and the mask column stride is honoured.
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int BDV,
    int WM,
    int WN,
    bool align_Q,
    bool align_K,
    bool has_mask,
    bool do_causal,
    bool has_sinks,
    bool KEY_RANGES,
    bool FIRST_RANGE,
    bool LAST_RANGE,
    short SCORE_PASS,
    typename MaskType,
    typename AccumType,
    typename StridePtr,
    typename MaskPtr,
    typename SinkPtr,
    typename StatePtr>
METAL_FUNC void attention_nax_bdv_impl(
    const device T* Q,
    const device T* K,
    const device T* V,
    device T* O,
    const device AttnParams* params,
    StridePtr q_str,
    StridePtr k_str,
    StridePtr v_str,
    StridePtr m_str,
    MaskPtr mask,
    SinkPtr sinks,
    StatePtr Sin,
    device float* Sout,
    int range_begin,
    int range_end,
    uint simd_group_id,
    uint3 tid) {
  ulong3 tidl{tid.x, tid.y, tid.z};

  const int64_t Q_strides[3] = {q_str[0], q_str[1], q_str[2]};
  const int64_t K_strides[3] = {k_str[0], k_str[1], k_str[2]};
  const int64_t V_strides[3] = {v_str[0], v_str[1], v_str[2]};
  const int64_t O_strides[3] = {
      int64_t(params->qL) * params->H * BDV, BDV, int64_t(params->H) * BDV};

  Q += tidl.z * Q_strides[0] + // Batch
      tidl.y * Q_strides[1] + // Head
      tidl.x * BQ * Q_strides[2]; // Sequence

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * K_strides[0] + // Batch
      kv_head_idx * K_strides[1]; // Head

  V += tidl.z * V_strides[0] + // Batch
      kv_head_idx * V_strides[1]; // Head

  if constexpr (!KEY_RANGES || (SCORE_PASS == 2 && LAST_RANGE)) {
    O += tidl.z * O_strides[0] + // Batch
        tidl.y * O_strides[1] + // Head
        tidl.x * BQ * O_strides[2]; // Sequence
  }

  if (has_mask) {
    mask += tidl.z * m_str[0] + // Batch
        tidl.y * m_str[1]; // Head
  }


  // Prepare MMA tiles
  constexpr short kU = 16;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kU) && BQ % (kNWarps * kU) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kU);
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kU;
  // Value head dim frags
  constexpr int TDV = BDV / kU;
  // KV seq frags per warp
  constexpr short TK = BK / kU;

  static_assert(TQ == 1, "Check TQ");
  static_assert(BDV % (2 * kU) == 0, "BDV must be a multiple of 32");
  using otile_t = NAXTile<AccumType, TQ, TDV>;
  otile_t Otile;

  Otile.clear();

  // Prepare mma tile offsets
  const short tm = kU * TQ * simd_group_id;
  Q += tm * int(Q_strides[2]);

  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  // Init row reduction variables
  constexpr short kRowsPT = otile_t::kRowsPerThread;

  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};

  // Init to -Inf
  OMLX_NAX_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  if (has_sinks) {
    OMLX_NAX_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = static_cast<AccumType>(sinks[tidl.y]);
    }
  }

  // Every padded row has its own full FP32 state. Loading/storing performs
  // no arithmetic or native-dtype conversion, including phase transitions.
  constexpr int kSW = BDV + 2;
  const int64_t srow0 =
      (int64_t(tid.z) * params->H + tid.y) * (int64_t(params->NQ) * BQ) +
      int64_t(tid.x) * BQ + tm;
  if constexpr (KEY_RANGES && !(SCORE_PASS == 0 && FIRST_RANGE)) {
    Otile.load(Sin + srow0 * kSW, kSW);
    OMLX_NAX_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      const int64_t r = srow0 + sm + i * otile_t::kFragRowsJump;
      max_score[i] = Sin[r * kSW + BDV];
      sum_score[i] = Sin[r * kSW + BDV + 1];
    }
  }

  const int kb_begin = KEY_RANGES ? range_begin : 0;
  int kb_lim = KEY_RANGES ? range_end : params->NK;
  int kb_min_causal = params->NK;

  // Keep all real keys: the baseline's boolean mask uses finite T minimum.
  kb_min_causal = 0;

  const bool is_last_bq = int(tid.x) == (params->NQ_aligned);
  const bool is_last_q = is_last_bq;

  const short lim_rows_q = params->qL_rem - tm;
  const short lim_rows_k = params->kL_rem;

  const device T* const k_start = K;
  const device T* const v_start = V;
  // Three score passes preserve global-max exp arguments and the native
  // probability rounding before P @ V without a materialized score matrix.
  for (short pass = KEY_RANGES ? SCORE_PASS : 0;
       pass < (KEY_RANGES ? SCORE_PASS + 1 : 3); ++pass) {
    K = k_start;
    V = v_start;
    if constexpr (KEY_RANGES) {
      K += int64_t(kb_begin) * BK * K_strides[2];
      V += int64_t(kb_begin) * BK * V_strides[2];
    }
    if (pass == 1 && has_sinks && (!KEY_RANGES || FIRST_RANGE)) {
      OMLX_NAX_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        sum_score[i] = fast::exp(AccumType(sinks[tidl.y]) - max_score[i]);
      }
    }
  // Loop over KV seq length
  for (int kb = kb_begin; kb < kb_lim; kb++) {
    const int is_last_k = (kb == (params->NK_aligned));

    // Do S = Q @ K.T
    using stile_t = NAXTile<AccumType, TQ, TK>;
    stile_t Stile;

    Stile.clear();

    OMLX_NAX_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      OMLX_NAX_UNROLL
      for (short ik = 0; ik < TK; ik += 2) {
#pragma clang loop unroll_count(4)
        for (short id = 0; id < TD; id++) {
          NAXTile<T, 1, 1> Qtile;
          NAXTile<T, 2, 1> Ktile;

          const int64_t Q_load_off = iq * kU * Q_strides[2] + id * kU * q_str[3];
          const int64_t K_load_off = ik * kU * K_strides[2] + id * kU * k_str[3];

          if (!align_Q && is_last_q) {
            load_head_rows(Qtile,
                Q + Q_load_off, int(Q_strides[2]), q_str[3], lim_rows_q - iq * kU);
          } else {
            load_head_rows(Qtile, Q + Q_load_off, int(Q_strides[2]), q_str[3], kU);
          }

          if (!align_K && is_last_k) {
            load_head_rows(Ktile,
                K + K_load_off, int(K_strides[2]), k_str[3], lim_rows_k - ik * kU);
          } else {
            load_head_rows(Ktile, K + K_load_off, int(K_strides[2]), k_str[3], 2 * kU);
          }

          stile_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qtile.frag_at(0, 0),
              metal::false_type{},
              Ktile.frag_at(0, 0),
              Ktile.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    // Q is pre-scaled in native dtype by the host, exactly as fast.cpp.
    // The fallback materializes QK in native dtype before softmax.
    // Round at the same boundary here.
    // Round S
    OMLX_NAX_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] = AccumType(T(Stile.elems()[ii]));
    }

    // Mask out length sequence
    if (!align_K && is_last_k) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      OMLX_NAX_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        OMLX_NAX_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short col_pos = ik * kU + sn;

          thread auto& fg = Stile.frag_at(iq, ik);

          OMLX_NAX_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            OMLX_NAX_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < params->kL_rem) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= kb_min_causal) {
      constexpr auto neg_inf = AccumType(Limits<T>::finite_min);

      const int base_row = tid.x * BQ + params->qL_off + tm;
      const int base_col = kb * BK;

      OMLX_NAX_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        OMLX_NAX_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);

          OMLX_NAX_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            OMLX_NAX_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto r =
                  base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
              const auto c = base_col + ik * kU + jj + sn;
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      constexpr auto neg_inf = AccumType(Limits<T>::finite_min);

      const int base_row = tid.x * BQ + tm;
      const int base_col = kb * BK;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, AccumType>;
      using mtile_t = NAXTile<melem_t, TQ, TK>;
      using mfrag_t = typename mtile_t::frag_type;

      if (base_row + BQ <= params->qL && base_col + BK <= params->kL) {
        for (short iq = 0; iq < TQ; iq++) {
          OMLX_NAX_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            const int row_pos = base_row + iq * kU;
            const int col_pos = base_col + ik * kU;

            mfrag_t mfrag;
            mtile_t::NAXFrag_t::load(
                mfrag,
                mask,
                int64_t(m_str[2]),
                int64_t(m_str[3]),
                row_pos,
                col_pos);

            thread auto& fg = Stile.frag_at(iq, ik);

            OMLX_NAX_UNROLL
            for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
              if constexpr (is_bool) {
                fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
              } else {
                fg[jj] += AccumType(mfrag[jj]);
              }
            }
          }
        }
      } else {
        OMLX_NAX_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          OMLX_NAX_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            const int row_pos = base_row + iq * kU;
            const int col_pos = base_col + ik * kU;

            mfrag_t mfrag;
            mtile_t::NAXFrag_t::load_safe(
                mfrag,
                mask,
                int64_t(m_str[2]),
                int64_t(m_str[3]),
                params->qL,
                params->kL,
                row_pos,
                col_pos);

            thread auto& fg = Stile.frag_at(iq, ik);

            OMLX_NAX_UNROLL
            for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
              if constexpr (is_bool) {
                fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
              } else {
                fg[jj] += AccumType(mfrag[jj]);
              }
            }
          }
        }
      }
    }

    // Exclude padded K columns after applying real-key masks.
    if (!align_K && is_last_k) {
      OMLX_NAX_UNROLL
      for (short iq = 0; iq < TQ; ++iq) {
        OMLX_NAX_UNROLL
        for (short ik = 0; ik < TK; ++ik) {
          auto& frag = Stile.frag_at(iq, ik);
          OMLX_NAX_UNROLL
          for (short r = 0; r < stile_t::kFragThrRows; ++r) {
            OMLX_NAX_UNROLL
            for (short c = 0; c < stile_t::kFragThrCols; ++c) {
              if (ik * kU + sn + c >= params->kL_rem) {
                frag[r * stile_t::kFragThrCols + c] = -INFINITY;
              }
            }
          }
        }
      }
    }
    if (pass == 0) {
      Stile.template row_reduce<MaxOp>(max_score);
    } else if (pass == 1) {
      Stile.template row_bin_op<ExpSubOp>(max_score);
      Stile.template row_reduce<SumOp>(sum_score);
    } else {
      Stile.template row_bin_op<ExpSubOp>(max_score);
      metal::vec<AccumType, kRowsPT> reciprocal_sum;
      OMLX_NAX_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        reciprocal_sum[i] = 1.f / sum_score[i];
      }
      Stile.template row_bin_op<MulOp>(reciprocal_sum);
      NAXTile<T, TQ, TK> Ptile;
      OMLX_NAX_UNROLL
      for (short i = 0; i < stile_t::kElemsPerTile; ++i) {
        Ptile.elems()[i] = T(Stile.elems()[i]);
      }
      simdgroup_barrier(mem_flags::mem_none);

    // Do O = P @ V
    OMLX_NAX_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      OMLX_NAX_UNROLL
      for (short id = 0; id < TDV; id += 2) {
        if constexpr (BDV == 128) {
          if (id == 4) {
            threadgroup_barrier(mem_flags::mem_none);
          }
        }

        OMLX_NAX_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          NAXTile<T, 1, 2> Vtile;

          const int64_t V_load_off = ik * kU * V_strides[2] + id * kU * v_str[3];

          if (!align_K && is_last_k) {
            load_head_rows(Vtile,
                V + V_load_off, int(V_strides[2]), v_str[3], lim_rows_k - ik * kU);
          } else {
            load_head_rows(Vtile, V + V_load_off, int(V_strides[2]), v_str[3], kU);
          }

          otile_t::NAXFrag_t::mma(
              Otile.frag_at(iq, id),
              Otile.frag_at(iq, id + 1),
              Ptile.frag_at(iq, ik),
              metal::false_type{},
              Vtile.frag_at(0, 0),
              Vtile.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    } // normalized native-dtype probability / value product
    // Prepare for next iteration
    K += BK * int(K_strides[2]);
    V += BK * int(V_strides[2]);
  }

  } // score passes
  threadgroup_barrier(mem_flags::mem_none);

  if constexpr (KEY_RANGES && !(SCORE_PASS == 2 && LAST_RANGE)) {
    // Unique O fragment stores; only the first column lane stores each
    // replicated row reduction. Ragged Q padding is initialized too.
    Otile.store(Sout + srow0 * kSW, kSW);
    if (sn == 0) {
      OMLX_NAX_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        const int64_t r = srow0 + sm + i * otile_t::kFragRowsJump;
        Sout[r * kSW + BDV] = max_score[i];
        Sout[r * kSW + BDV + 1] = sum_score[i];
      }
    }
    // Both outputs are bound by custom-kernel ABI; initialize the unused
    // scalar, not an out-of-range B/H/Q address.
    if (tid.x == 0 && tid.y == 0 && tid.z == 0 &&
        simd_group_id == 0 && sm == 0 && sn == 0) {
      O[0] = T(0);
    }
    return;
  }
  if constexpr (KEY_RANGES) {
    if (tid.x == 0 && tid.y == 0 && tid.z == 0 &&
        simd_group_id == 0 && sm == 0 && sn == 0) {
      Sout[0] = 0.f;
    }
  }

  // Store results
  O += tm * int(O_strides[2]);

  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0)
      return;

    Otile.store_rows(O, int(O_strides[2]), lim_rows_q);
  } else {
    Otile.store(O, int(O_strides[2]));
  }
}

// Keep the original single-dispatch entry contract for ordinary callers and
// independently owned per-block descriptors. Disabled range code adds no
// floating-point operations, allocation, key truncation or state dereference.
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int BDV,
    int WM,
    int WN,
    bool align_Q,
    bool align_K,
    bool has_mask,
    bool do_causal,
    bool has_sinks,
    typename MaskType,
    typename AccumType,
    typename StridePtr,
    typename MaskPtr,
    typename SinkPtr>
METAL_FUNC void attention_nax_bdv(
    const device T* Q,
    const device T* K,
    const device T* V,
    device T* O,
    const device AttnParams* params,
    StridePtr q_str,
    StridePtr k_str,
    StridePtr v_str,
    StridePtr m_str,
    MaskPtr mask,
    SinkPtr sinks,
    uint simd_group_id,
    uint3 tid) {
  attention_nax_bdv_impl<T, BQ, BK, BD, BDV, WM, WN,
      align_Q, align_K, has_mask, do_causal, has_sinks,
      false, true, true, 0, MaskType, AccumType>(
      Q, K, V, O, params, q_str, k_str, v_str, m_str, mask, sinks,
      Q, nullptr, 0, 0, simd_group_id, tid);
}

} // namespace omlx_nax
"""#

    static let source = #"""
  omlx_nax::attention_nax_bdv<
      T, 64, 32, 192, 128, 4, 1,
      ALIGN_Q, ALIGN_K, HAS_MASK, DO_CAUSAL, HAS_SINKS, bool, float>(
      q, k, v, out,
      reinterpret_cast<const device omlx_nax::AttnParams*>(params),
      q_strides, k_strides, v_strides, mask_strides,
      mask, sinks,
      simdgroup_index_in_threadgroup,
      threadgroup_position_in_grid);
"""#
}
