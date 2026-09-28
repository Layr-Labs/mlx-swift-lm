// SPDX-License-Identifier: Apache-2.0
// Adapted from jundot/omlx #3990 moe_decode.py at
// e58cab4d5db6c3ca27742a45a50ab6287b1b1bcb.
// MXFP4 arithmetic derived from MLX, Copyright © 2024 Apple Inc.
// Swift adaptation Copyright © 2026 Eigen Labs.
// See docs/mimo-v26/DECODE-EXPERT-ATTRIBUTION.md.

enum MiMoV26DecodeExpertMetal {
    static let header = """
        // MXFP4 e2m1 nibble -> float exactly like MLX's fp4_e2m1 (via half).
        inline float omlx_fp4(uint v) {
          uint b = v & 0xF;
          half c = as_type<half>(ushort((b & 7) << 9));
          c *= 16384.0;
          return static_cast<float>((b & 8) ? -c : c);
        }

        // e8m0 scale -> float like MLX's fp8_e8m0.
        inline float omlx_e8m0(uint8_t s) {
          uint out = (s == 0 ? 0x400000u : (static_cast<uint>(s) << 23));
          return as_type<float>(out);
        }

        // MLX fp_quantized.h qdot<float, 16, 4> on 8 packed bytes (4 x uint16).
        inline float omlx_qdot16(uint2 w, thread const float* x, float scale) {
          ushort ws[4] = {ushort(w.x & 0xFFFF), ushort(w.x >> 16), ushort(w.y & 0xFFFF), ushort(w.y >> 16)};
          float accum = 0;
          for (int i = 0; i < 4; i++) {
            accum +=
                (x[4 * i] * omlx_fp4(ws[i]) + x[4 * i + 1] * omlx_fp4(ws[i] >> 4) +
                 x[4 * i + 2] * omlx_fp4(ws[i] >> 8) +
                 x[4 * i + 3] * omlx_fp4(ws[i] >> 12));
          }
          return scale * accum;
        }

        // Pinned compiled.cpp writes each primitive to its declared dtype.
        // This matches unary_ops.h Sigmoid, applied in activation precision.
        struct OmlxSigmoid {
          template <typename U>
          U operator()(U x) thread {
            auto y = 1 / (1 + metal::exp(metal::abs(x)));
            return (x < 0) ? y : 1 - y;
          }
        };
        template <typename T>
        inline T mimo_swiglu(T gate, T up) {
          T sig = OmlxSigmoid{}(gate);
          T activated = gate * sig;
          return activated * up;
        }
        """

    static let gateUp = """
        constexpr int VPT = 16;
          uint pid = threadgroup_position_in_grid.x;
          uint rb = threadgroup_position_in_grid.y;
          uint sgid = simdgroup_index_in_threadgroup;
          uint lane = thread_index_in_simdgroup;
          uint e = inds[pid];
          for (uint q = 0; q < pid; q++) {
            if (inds[q] == e) {
              return;
            }
          }
          uint members[MAXDUP];
          int nm = 0;
          // One representative per token row; duplicate slots in that row have
          // identical inputs. Thus nm <= ROWS even when all NPAIRS slots repeat.
          for (uint q = pid; q < NPAIRS; q++) {
            if (inds[q] == e) {
              bool seen_row = false;
              for (int j = 0; j < nm; j++) {
                seen_row = seen_row || members[j] / TOPK == q / TOPK;
              }
              if (!seen_row) {
                members[nm++] = q;
              }
            }
          }
          const int n0 = (rb * NSG + sgid) * RPS;
          constexpr int KW = KDIM / 8;   // uint32 per weight row
          constexpr int KS = KDIM / 32;  // scales per row
          const device uint32_t* gw = wg + (size_t(e) * GROWS + n0) * KW + lane * 2;
          const device uint8_t* gs = sg + (size_t(e) * GROWS + n0) * KS + lane / 2;
          const device uint32_t* uw = wu + (size_t(e) * UROWS + UOFF + n0) * KW + lane * 2;
          const device uint8_t* us = su + (size_t(e) * UROWS + UOFF + n0) * KS + lane / 2;

          float rg[MAXDUP][RPS];
          float ru[MAXDUP][RPS];
          for (int j = 0; j < MAXDUP; j++) {
            for (int r = 0; r < RPS; r++) {
              rg[j][r] = 0;
              ru[j][r] = 0;
            }
          }
          for (int b = 0; b < KDIM / 512; b++) {
            uint2 wgv[RPS];
            uint2 wuv[RPS];
            float sgv[RPS];
            float suv[RPS];
            for (int r = 0; r < RPS; r++) {
              wgv[r] = *(const device uint2*)(gw + r * KW + b * 64);
              wuv[r] = *(const device uint2*)(uw + r * KW + b * 64);
              sgv[r] = omlx_e8m0(gs[r * KS + b * 16]);
              suv[r] = omlx_e8m0(us[r * KS + b * 16]);
            }
            for (int j = 0; j < MAXDUP; j++) {
              if (j < nm) {
                const device T* xp = x + size_t(members[j] / TOPK) * KDIM + b * 512 + lane * VPT;
                float xt[VPT];
                for (int i = 0; i < VPT; i++) {
                  xt[i] = static_cast<float>(xp[i]);
                }
                for (int r = 0; r < RPS; r++) {
                  rg[j][r] += omlx_qdot16(wgv[r], xt, sgv[r]);
                  ru[j][r] += omlx_qdot16(wuv[r], xt, suv[r]);
                }
              }
            }
          }
          for (int j = 0; j < MAXDUP; j++) {
            if (j < nm) {
              for (int r = 0; r < RPS; r++) {
                float g = simd_sum(rg[j][r]);
                float u = simd_sum(ru[j][r]);
                if (lane == 0) {
                  T gt = static_cast<T>(g);
                  T ut = static_cast<T>(u);
                  T value = mimo_swiglu(gt, ut);
                  uint first = (members[j] / TOPK) * TOPK;
                  for (uint q = first; q < first + TOPK; q++) {
                    if (inds[q] == e) {
                      act[size_t(q) * NOUT + n0 + r] = value;
                    }
                  }
                }
              }
            }
          }
        """

    static let down = """
        constexpr int VPT = 16;
          uint pid = threadgroup_position_in_grid.x;
          uint rb = threadgroup_position_in_grid.y;
          uint sgid = simdgroup_index_in_threadgroup;
          uint lane = thread_index_in_simdgroup;
          uint e = inds[pid];
          for (uint q = 0; q < pid; q++) {
            if (inds[q] == e) {
              return;
            }
          }
          uint members[MAXDUP];
          int nm = 0;
          // One representative per token row; duplicate slots in that row have
          // identical inputs. Thus nm <= ROWS even when all NPAIRS slots repeat.
          for (uint q = pid; q < NPAIRS; q++) {
            if (inds[q] == e) {
              bool seen_row = false;
              for (int j = 0; j < nm; j++) {
                seen_row = seen_row || members[j] / TOPK == q / TOPK;
              }
              if (!seen_row) {
                members[nm++] = q;
              }
            }
          }
          const int n0 = (rb * NSG + sgid) * RPS;
          constexpr int KW = KDIM / 8;
          constexpr int KS = KDIM / 32;
          const device uint32_t* dw = w + (size_t(e) * NOUT + n0) * KW + lane * 2;
          const device uint8_t* ds = s + (size_t(e) * NOUT + n0) * KS + lane / 2;

          float acc[MAXDUP][RPS];
          for (int j = 0; j < MAXDUP; j++) {
            for (int r = 0; r < RPS; r++) {
              acc[j][r] = 0;
            }
          }
          for (int b = 0; b < KDIM / 512; b++) {
            uint2 wv[RPS];
            float sv[RPS];
            for (int r = 0; r < RPS; r++) {
              wv[r] = *(const device uint2*)(dw + r * KW + b * 64);
              sv[r] = omlx_e8m0(ds[r * KS + b * 16]);
            }
            for (int j = 0; j < MAXDUP; j++) {
              if (j < nm) {
                const device T* xp = a + size_t(members[j]) * KDIM + b * 512 + lane * VPT;
                float xt[VPT];
                for (int i = 0; i < VPT; i++) {
                  xt[i] = static_cast<float>(xp[i]);
                }
                for (int r = 0; r < RPS; r++) {
                  acc[j][r] += omlx_qdot16(wv[r], xt, sv[r]);
                }
              }
            }
          }
          for (int j = 0; j < MAXDUP; j++) {
            if (j < nm) {
              for (int r = 0; r < RPS; r++) {
                float v = simd_sum(acc[j][r]);
                if (lane == 0) {
                  T value = static_cast<T>(v);
                  uint first = (members[j] / TOPK) * TOPK;
                  for (uint q = first; q < first + TOPK; q++) {
                    if (inds[q] == e) {
                      y[size_t(q) * NOUT + n0 + r] = value;
                    }
                  }
                }
              }
            }
          }
        """
}
