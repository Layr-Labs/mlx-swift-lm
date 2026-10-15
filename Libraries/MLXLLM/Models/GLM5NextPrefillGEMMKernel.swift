// SPDX-License-Identifier: Apache-2.0 AND MIT
// Copyright 2026 Eigen Labs.
// Built on Apple MLX's NAX tile primitives (MIT; pinned and noticed in
// `MiMoV26NAXMetalSources`) and the structure of MLX `qmm_t_nax_tgp_impl`,
// after oMLX P15 (native NAX affine qmm for prefill projections).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Affine 4-bit `x @ Wᵀ` for prefill-sized projections on Neural-Accelerator
/// GPUs, for `GLM5NextExecutionPolicy.PrefillProjections.nativeGEMM`.
///
/// MLX's `qmm_t_nax` computes 64 × 64 output tiles with one weight buffer and
/// two barriers per 64-value K step. This kernel computes larger tiles with a
/// double-buffered weight tile (one barrier per K step) and a swizzled tile
/// order, so each dequantized weight tile serves more activation rows and each
/// activation block stays cache-resident across neighbouring weight tiles.
///
/// Each weight is dequantized exactly as MLX's loader does it,
/// `T(float(scale) * q + float(bias))` (the product is exact in FP32), and
/// products accumulate in FP32 tensor ops in ascending K order, rounded once to
/// `T`. Only the grouping of the FP32 accumulation can differ from MLX.
enum GLM5NextPrefillGEMMKernel {
    /// Minimum activation rows routed to this kernel (smaller steps keep
    /// MLX's qmv/qmm, which are better at decode and verification sizes).
    static let minimumRows = 128

    static let naxAvailable: Bool = GLM5NextMoEPrefillKernel.naxAvailable

    struct Config: Hashable, CustomStringConvertible {
        var bm: Int
        var bn: Int
        var bk: Int
        var wm: Int
        var wn: Int
        var swizzle: Int
        var doubleBuffer: Bool

        var description: String {
            "bm\(bm)bn\(bn)bk\(bk)w\(wm)x\(wn)s\(swizzle)\(doubleBuffer ? "db" : "sb")"
        }

        static let `default` = Config(
            bm: 128, bn: 64, bk: 64, wm: 4, wn: 2, swizzle: 2, doubleBuffer: true)
    }

    enum Failure: Error, CustomStringConvertible {
        case unsupported(String)

        var description: String {
            switch self {
            case .unsupported(let reason): "GLM5Next prefill GEMM: \(reason)"
            }
        }
    }

    /// Whether `linear` can run here for activations of `dtype`.
    static func admits(_ linear: QuantizedLinear, dtype: DType) -> Bool {
        naxAvailable && linear.mode == .affine && linear.bits == 4
            && [32, 64, 128].contains(linear.groupSize) && linear.biases != nil
            && linear.bias == nil && [.bfloat16, .float16].contains(dtype)
            && linear.scales.dtype == dtype && linear.biases?.dtype == dtype
            && linear.weight.dtype == .uint32 && linear.shape.1 % 64 == 0
    }

    /// `x` `[..., K]` times the packed `[N, K]` weight, transposed: `[..., N]`.
    static func matmul(
        _ x: MLXArray, _ linear: QuantizedLinear, config: Config = .default
    ) throws -> MLXArray {
        guard admits(linear, dtype: x.dtype) else {
            throw Failure.unsupported("weight bank or dtype")
        }
        guard glm5NextOnDefaultGPUStream() else {
            throw Failure.unsupported("not on the default GPU stream")
        }
        let (n, k) = linear.shape
        guard x.ndim >= 1, x.dim(-1) == k, k % config.bk == 0 else {
            throw Failure.unsupported("input width")
        }
        let m = x.size / k
        guard m > 0, m * max(n, k) < Int(Int32.max) else {
            throw Failure.unsupported("row count")
        }
        let tilesN = (n + config.bn - 1) / config.bn
        let tilesM = (m + config.bm - 1) / config.bm
        let band = 1 << config.swizzle
        let rows = x.reshaped(m, k)
        let output = kernel(
            [
                rows, linear.weight, linear.scales, linear.biases!,
                MLXArray([Int32(m), Int32(n), Int32(k), Int32(tilesN), Int32(tilesM)]),
            ],
            template: [
                ("T", x.dtype), ("GS", linear.groupSize), ("BM", config.bm),
                ("BN", config.bn), ("BK", config.bk), ("WM", config.wm), ("WN", config.wn),
                ("SWZ", config.swizzle), ("DB", config.doubleBuffer),
            ],
            grid: (tilesN * band * 32, ((tilesM + band - 1) / band) * config.wn, config.wm),
            threadGroup: (32, config.wn, config.wm),
            outputShapes: [[m, n]], outputDTypes: [x.dtype])[0]
        return output.reshaped(Array(x.shape.dropLast()) + [n])
    }

    private static let kernel = MLXFast.metalKernel(
        name: "glm5_next_prefill_qmm_t",
        inputNames: ["x", "w", "scales", "biases", "params"],
        outputNames: ["y"],
        source: """
            constexpr int BKP = BK + 16 / sizeof(T);
            threadgroup T Ws[(DB ? 2 : 1) * BN * BKP];
            const int tiles_n = params[3];
            const int tiles_m = params[4];
            const int tx = int(threadgroup_position_in_grid.x);
            const int tile_n = tx >> SWZ;
            const int tile_m = (int(threadgroup_position_in_grid.y) << SWZ) + (tx & ((1 << SWZ) - 1));
            if (tile_n >= tiles_n || tile_m >= tiles_m) {
                return;
            }
            glm5_qmm::qmm_t<T, GS, BM, BN, BK, WM, WN, DB>(
                x, (const device uint8_t*)w, scales, biases, y, params[0], params[1],
                params[2], Ws, tile_m, tile_n, simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            """,
        header: MiMoV26NAXMetalSources.mlxHeader + "\n" + qmmHeader + "\n",
        ensureRowContiguous: true)

    private static let qmmHeader = #"""
        using namespace metal;
        using namespace mlx::steel;

        namespace glm5_qmm {

        // y[M, N] = x[M, K] · dequant(w)[N, K]ᵀ for one BM x BN output tile.
        // Thread lid dequantizes VPT consecutive values of weight row lid / TPR
        // per K step into threadgroup memory (row stride BKP); with DB the packed
        // words of step k + 1 are read before the tensor ops of step k and stored
        // into the other buffer after them (one barrier per step). Activation
        // fragments are read from device memory; rows past M are clamped to the
        // last row and weight rows past N to row N - 1, and neither is stored.
        template <typename T, int GS, int BM, int BN, int BK, int WM, int WN, bool DB>
        METAL_FUNC void qmm_t(
            const device T* x,
            const device uint8_t* w,
            const device T* scales,
            const device T* biases,
            device T* y,
            const int M,
            const int N,
            const int K,
            threadgroup T* Ws,
            const int tile_m,
            const int tile_n,
            const uint sgid,
            const uint lane) {
          constexpr int kThreads = WM * WN * 32;
          constexpr int BKP = BK + 16 / sizeof(T);
          constexpr int kTile = BN * BKP;
          constexpr int VPT = BN * BK / kThreads;
          constexpr int TPR = BK / VPT;
          constexpr int WORDS = VPT / 8;
          constexpr short SM = BM / WM;
          constexpr short SN = BN / WN;
          constexpr short TM = SM / 16;
          constexpr short TN = SN / 16;
          static_assert(VPT % 8 == 0 && GS % VPT == 0, "whole words, one group per thread");
          static_assert(BK % 32 == 0 && SM % 16 == 0 && TN % 2 == 0, "tile shape");

          const int y_row = tile_m * BM;
          const int y_col = tile_n * BN;
          const int lid = int(sgid) * 32 + int(lane);
          const int lrow = lid / TPR;
          const int lcol = (lid % TPR) * VPT;
          const int w_row = min(y_col + lrow, N - 1);
          const device uint32_t* wsrc =
              (const device uint32_t*)(w + size_t(w_row) * (K / 2) + lcol / 2);
          const device T* sp = scales + size_t(w_row) * (K / GS);
          const device T* bp = biases + size_t(w_row) * (K / GS);
          uint32_t raw[WORDS];
          float s = 0.0f;
          float b = 0.0f;

          auto fetch = [&](const int kb) {
            const device uint32_t* p = wsrc + kb * (BK / 8);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < WORDS; i++) {
              raw[i] = p[i];
            }
            const int g = (kb * BK + lcol) / GS;
            s = float(sp[g]);
            b = float(bp[g]);
          };
          auto store = [&](threadgroup T* buf) {
            threadgroup T* dst = buf + lrow * BKP + lcol;
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < WORDS; i++) {
              vec<T, 8> v;
              STEEL_PRAGMA_UNROLL
              for (short j = 0; j < 8; j++) {
                v[j] = static_cast<T>(s * float((raw[i] >> (4 * j)) & 0xfu) + b);
              }
              *(threadgroup vec<T, 8>*)(dst + i * 8) = v;
            }
          };

          const int m0 = SM * int(sgid / WN);
          const int n0 = SN * int(sgid % WN);
          const int rows = min(int(SM), M - (y_row + m0));
          const int cols = min(int(SN), N - (y_col + n0));
          const short2 sc = BaseNAXFrag::get_coord();
          const device T* xs = x + size_t(min(y_row + m0, M - 1)) * K + sc.x;
          int x_off[TM][2];
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < TM; i++) {
            STEEL_PRAGMA_UNROLL
            for (short h = 0; h < 2; h++) {
              x_off[i][h] = min(int(i * 16 + sc.y + h * 8), max(rows, 1) - 1) * K;
            }
          }
          const short m_frags = rows > 0 ? short((rows + 15) / 16) : short(0);
          const threadgroup T* wsg = Ws + n0 * BKP;

          NAXTile<float, TM, TN> D;
          D.clear();

          auto compute = [&](const threadgroup T* wb, const int kbase) {
            STEEL_PRAGMA_UNROLL
            for (short kk1 = 0; kk1 < BK; kk1 += 32) {
              NAXTile<T, TN, 2> Btile;
              Btile.template load<T, BKP, 1>(wb + kk1);
              const int k = kbase + kk1;
              STEEL_PRAGMA_UNROLL
              for (short i = 0; i < TM; i++) {
                if (i < m_frags) {
                  NAXTile<T, 1, 2> Atile;
                  STEEL_PRAGMA_UNROLL
                  for (short h = 0; h < 2; h++) {
                    const device T* xp = xs + x_off[i][h] + k;
                    const vec<T, 4> a0 = *(const device vec<T, 4>*)(xp);
                    const vec<T, 4> a1 = *(const device vec<T, 4>*)(xp + 16);
                    STEEL_PRAGMA_UNROLL
                    for (short c = 0; c < 4; c++) {
                      Atile.frag_at(0, 0)[h * 4 + c] = a0[c];
                      Atile.frag_at(0, 1)[h * 4 + c] = a1[c];
                    }
                  }
                  STEEL_PRAGMA_UNROLL
                  for (short kk = 0; kk < 2; kk++) {
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < TN; j += 2) {
                      BaseNAXFrag::mma(
                          D.frag_at(i, j),
                          D.frag_at(i, j + 1),
                          Atile.frag_at(0, kk),
                          metal::bool_constant<false>{},
                          Btile.frag_at(j, kk),
                          Btile.frag_at(j + 1, kk),
                          metal::bool_constant<true>{});
                    }
                  }
                }
              }
            }
          };

          const int K_it = K / BK;
          if constexpr (DB) {
            fetch(0);
            store(Ws);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int kb = 0; kb < K_it; kb++) {
              const bool more = kb + 1 < K_it;
              if (more) {
                fetch(kb + 1);
              }
              compute(wsg + (kb & 1) * kTile, kb * BK);
              if (more) {
                store(Ws + ((kb + 1) & 1) * kTile);
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }
          } else {
            for (int kb = 0; kb < K_it; kb++) {
              threadgroup_barrier(mem_flags::mem_threadgroup);
              fetch(kb);
              store(Ws);
              threadgroup_barrier(mem_flags::mem_threadgroup);
              compute(wsg, kb * BK);
            }
          }

          device T* yb = y + size_t(y_row + m0) * N + y_col + n0;
          if (rows >= SM && cols >= SN) {
            D.store(yb, N);
          } else if (rows > 0 && cols > 0) {
            D.store_safe(yb, N, short2(short(cols), short(rows)));
          }
        }

        } // namespace glm5_qmm
        """#
}

extension GLM5NextExecutionPolicy.PrefillProjections {
    /// `linear(x)`, or `GLM5NextPrefillGEMMKernel` under `.nativeGEMM` for
    /// steps of at least `minimumRows` rows over a bank it reads.
    func project(_ linear: Linear, _ x: MLXArray) throws -> MLXArray {
        guard self == .nativeGEMM, let packed = linear as? QuantizedLinear, x.ndim >= 1,
            x.size / max(1, x.dim(-1)) >= GLM5NextPrefillGEMMKernel.minimumRows,
            GLM5NextPrefillGEMMKernel.admits(packed, dtype: x.dtype)
        else { return linear(x) }
        return try GLM5NextPrefillGEMMKernel.matmul(x, packed)
    }
}
