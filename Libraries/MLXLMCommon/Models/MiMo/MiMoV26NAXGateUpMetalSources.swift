// SPDX-License-Identifier: Apache-2.0 AND MIT
// Adapted from oMLX #3995, 47876fbc310fbb311cd382c4eba8572ec4368308,
// to implement #3993's joint-projection intent without concatenating weights.
// Modified: independent gate/up inputs and accumulators; one activation load.
// SwiGLU epilogue intent from oMLX #4022, d6dbc9970c6df99b6a8589746e6ff8ddc38c33ff.
// Preserve the independent GEMM K order and native rounded elementwise stages.
// Apple/MIT primitive notices: MiMoV26NAXMetalSources.swift.
// oMLX notice/license: docs/mimo-v26/LICENSE-OMLX.

enum MiMoV26NAXGateUpMetalSources {
    static let header = #"""
        namespace omlx_gqmm {

        // Exact pinned MLX unary_ops.h / binary_ops.h implementations, renamed to
        // avoid collisions with the surrounding generated-kernel header. Their MIT
        // notice is preserved in MiMoV26NAXMetalSources.swift.
        struct MiMoGateUpSigmoid {
          template <typename T>
          T operator()(T x) thread {
            auto y = 1 / (1 + metal::exp(metal::abs(x)));
            return (x < 0) ? y : 1 - y;
          }
        };
        struct MiMoGateUpMultiply {
          template <typename T>
          T operator()(T x, T y) thread { return x * y; }
        };

        // Aligned native MXFP4 gate/up. The two independent output accumulators
        // follow the same per-element K sequence as gather_seg.
        template <typename T, typename Q, bool FUSE_ACTIVATION = false, bool ROW_MAP = false>
        METAL_FUNC void gather_gate_up(
            const device T* x,
            const device uint8_t* gate_w,
            const device uint8_t* up_w,
            thread Q& gate_q,
            thread Q& up_q,
            const device uint32_t* tiles,
            const device uint32_t* row_map,
            device T* gate_y,
            device T* up_y,
            const int N,
            const int K,
            threadgroup typename Q::WT* gate_ws,
            threadgroup typename Q::WT* up_ws,
            const uint3 tid,
            const uint sgid,
            const uint lane) {
          using WT = typename Q::WT;
          constexpr int BKP = kBK + 16 / sizeof(WT);
          const uint4 desc = *((const device uint4*)tiles + tid.y);
          const int row_start = int(desc.x);
          const uint32_t expert = desc.y;
          const int rows = int(desc.z);
          const int K_w = K * Q::kBits / 8;
          const int K_g = K / Q::kGroup;
          const int y_col = int(tid.x) * kBN;
          const size_t w_row = size_t(expert) * N + y_col;

          gate_q.advance(w_row * K_g);
          up_q.advance(w_row * K_g);
          TileLoader<Q> gate_loader(gate_w + w_row * K_w, K, gate_q, sgid * 32 + lane);
          TileLoader<Q> up_loader(up_w + w_row * K_w, K, up_q, sgid * 32 + lane);

          if constexpr (!ROW_MAP) { x += size_t(row_start) * K; }
          gate_y += size_t(row_start) * N + y_col;
          up_y += size_t(row_start) * N + y_col;
          const short tm = kSM * short(sgid / kWN);
          const short tn = kSN * short(sgid % kWN);
          const short live_rows = short(min(int(kSM), max(0, rows - int(tm))));
          const device T* xn = x + (ROW_MAP ? 0 : tm * K);
          // #4029: load the same token rows through the sort map, without writing
          // their repeated copies. size_t preserves offsets beyond UInt32 elements.
          size_t mapped_offsets[kTM][2];
          const short2 sc = BaseNAXFrag::get_coord();
          if constexpr (ROW_MAP) {
            STEEL_PRAGMA_UNROLL
            for (short mm = 0; mm < kTM; ++mm) {
              STEEL_PRAGMA_UNROLL
              for (short h = 0; h < 2; ++h) {
                const int local_row = min(rows - 1, int(tm) + mm * 16 + h * 8 + sc.y);
                mapped_offsets[mm][h] = size_t(row_map[row_start + local_row]) * K + sc.x;
              }
            }
          }

          NAXTile<float, kTM, kTN> gate_d;
          NAXTile<float, kTM, kTN> up_d;
          gate_d.clear();
          up_d.clear();

          for (int kb = 0; kb < K / kBK; ++kb) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            gate_loader.fetch(kb);
            up_loader.fetch(kb);
            gate_loader.store(gate_ws);
            up_loader.store(up_ws);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            STEEL_PRAGMA_NO_UNROLL
            for (short kk1 = 0; kk1 < kBK; kk1 += kSK) {
              if (live_rows > 0) {
                NAXTile<WT, kTN, kTK> gate_b;
                NAXTile<WT, kTN, kTK> up_b;
                gate_b.template load<WT, BKP, 1>(gate_ws + tn * BKP + kk1);
                up_b.template load<WT, BKP, 1>(up_ws + tn * BKP + kk1);
                STEEL_PRAGMA_UNROLL
                for (short mm = 0; mm < kTM; ++mm) {
                  if (mm * 16 < live_rows) {
                    NAXTile<T, 1, kTK> a;
                    // This single device activation read feeds both independent GEMMs.
                    if constexpr (ROW_MAP) {
                      STEEL_PRAGMA_UNROLL
                      for (short h = 0; h < 2; ++h) {
                        STEEL_PRAGMA_UNROLL
                        for (short kk = 0; kk < kTK; ++kk) {
                          const vec<T, 4> v = *(const device vec<T, 4>*)(
                              xn + mapped_offsets[mm][h] + kk1 + kk * 16);
                          STEEL_PRAGMA_UNROLL
                          for (short c = 0; c < 4; ++c) {
                            a.frag_at(0, kk)[h * 4 + c] =
                                mm * 16 + h * 8 + sc.y < live_rows ? v[c] : T(0);
                          }
                        }
                      }
                    } else if (live_rows - mm * 16 >= 16) {
                      a.load(xn + mm * 16 * K + kk1, K);
                    } else {
                      a.load_safe(xn + mm * 16 * K + kk1, K,
                                  short2(kSK, live_rows - mm * 16));
                    }
                    STEEL_PRAGMA_UNROLL
                    for (short nn = 0; nn < kTN; nn += 2) {
                      STEEL_PRAGMA_UNROLL
                      for (short kk = 0; kk < kTK; ++kk) {
                        BaseNAXFrag::mma(
                            gate_d.frag_at(mm, nn), gate_d.frag_at(mm, nn + 1),
                            a.frag_at(0, kk), metal::bool_constant<false>{},
                            gate_b.frag_at(nn, kk), gate_b.frag_at(nn + 1, kk),
                            metal::bool_constant<true>{});
                        BaseNAXFrag::mma(
                            up_d.frag_at(mm, nn), up_d.frag_at(mm, nn + 1),
                            a.frag_at(0, kk), metal::bool_constant<false>{},
                            up_b.frag_at(nn, kk), up_b.frag_at(nn + 1, kk),
                            metal::bool_constant<true>{});
                      }
                    }
                  }
                }
              }
            }
            xn += kBK;
          }
          if constexpr (FUSE_ACTIVATION) {
            // The two projections are rounded exactly where the separate stores
            // round them. Each following functor returns T before the next op.
            // Do not contract the FP32 accumulators directly into a different SiLU.
            NAXTile<T, kTM, kTN> activated;
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < decltype(activated)::kNumFrags; ++i) {
              STEEL_PRAGMA_UNROLL
              for (short e = 0; e < decltype(activated)::kElemsPerFrag; ++e) {
                const T g = static_cast<T>(gate_d.val_frags[i][e]);
                const T u = static_cast<T>(up_d.val_frags[i][e]);
                activated.val_frags[i][e] = MiMoGateUpMultiply()(
                    MiMoGateUpMultiply()(g, MiMoGateUpSigmoid()(g)), u);
              }
            }
            if (live_rows == kSM) {
              activated.store(gate_y + tm * N + tn, N);
            } else if (live_rows > 0) {
              activated.store_safe(gate_y + tm * N + tn, N, short2(kSN, live_rows));
            }
          } else if (live_rows == kSM) {
            gate_d.store(gate_y + tm * N + tn, N);
            up_d.store(up_y + tm * N + tn, N);
          } else if (live_rows > 0) {
            gate_d.store_safe(gate_y + tm * N + tn, N, short2(kSN, live_rows));
            up_d.store_safe(up_y + tm * N + tn, N, short2(kSN, live_rows));
          }
        }
        } // namespace omlx_gqmm
        """#

    static let source = #"""
            if (threadgroup_position_in_grid.y >= tile_count[0]) {
                return;
            }
            using Q = omlx_gqmm::Mxfp4Q<32>;
            constexpr int BKP = omlx_gqmm::kBK + 16 / sizeof(bfloat);
            // Two single buffers total 18432 bytes; no dual double buffering.
            threadgroup bfloat gate_ws[omlx_gqmm::kBN * BKP];
            threadgroup bfloat up_ws[omlx_gqmm::kBN * BKP];
            Q gate_q{gate_scales, 0.0f};
            Q up_q{up_scales, 0.0f};
            omlx_gqmm::gather_gate_up<T, Q>(
                x, (const device uint8_t*)gate_w, (const device uint8_t*)up_w,
                gate_q, up_q, tiles, tiles, gate_y, up_y, params[0], params[1],
                gate_ws, up_ws, threadgroup_position_in_grid,
                simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """#
}
