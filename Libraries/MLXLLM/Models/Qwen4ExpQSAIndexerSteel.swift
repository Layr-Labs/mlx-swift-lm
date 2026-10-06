// Copyright © 2026 Eigen Labs / Darkbloom Assignment 1.
//
// Fusion `qwen4_qsa_indexer_score` (GEMMKernel + BlockLoader, BK=16,
// fp32 accum, four-head ReLU-sum / sqrt(128), pooled causal mask).
// `MLXFast.metalKernel` cannot `#include` steel headers, so this
// uses the packaged mlx-generated gemm preamble (same bytes as affine
// QMM) and instantiates Fusion's tile. Official unique-salt 128K
// regressed (~167 s first cell / warm degraded vs 145 s JIT) — the
// JIT MMA port stays the default. Lab restore:
// DARKBLOOM_QWEN4_QSA_INDEXER_STEEL=1. Missing header / flag off
// returns nil so `Qwen4ExpNativeIndexer` keeps the first JIT port.

import Foundation
import MLX
import MLXLMCommon

enum Qwen4ExpSteelIndexer: Sendable {
    static let envFlag = "DARKBLOOM_QWEN4_QSA_INDEXER_STEEL"
    static let blockN = 64
    static let blockK = 16
    static let warpsN = 2
    static let prefillBlockM = 64
    static let prefillWarpsM = 2
    static let narrowBlockM = 8
    static let midBlockM = 16
    static let narrowWarpsM = 1

    static func isEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        let raw = environment[envFlag]?.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
    }

    /// Fusion dispatch: BM=8/WM=1 for M≤8, BM=16/WM=1 for M≤16, else BM=64/WM=2.
    static func tile(queryTokens: Int) -> (blockM: Int, warpsM: Int, threads: Int) {
        if queryTokens <= narrowBlockM {
            return (narrowBlockM, narrowWarpsM, narrowWarpsM * warpsN * 32)
        }
        if queryTokens <= midBlockM {
            return (midBlockM, narrowWarpsM, narrowWarpsM * warpsN * 32)
        }
        return (prefillBlockM, prefillWarpsM, prefillWarpsM * warpsN * 32)
    }

    /// `queries` is `[1,4,M,128]`, `pooled` is `[1,1,N,128]`. Nil if the
    /// steel preamble is missing or the opt-in flag is off.
    static func scores(
        queries: MLXArray, pooled: MLXArray, maskQOffset: MLXArray,
        queryTokens: Int, blocks: Int, dtype: DType,
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> MLXArray? {
        guard isEnabled(environment: environment), let kernel = steelIndexerScoreKernel
        else { return nil }
        guard queryTokens > 0, blocks > 0 else { return nil }
        let tile = tile(queryTokens: queryTokens)
        let tilesN = (blocks + blockN - 1) / blockN
        let tilesM = (queryTokens + tile.blockM - 1) / tile.blockM
        return kernel(
            [queries, pooled, maskQOffset],
            template: [
                ("T", dtype),
                ("BM", tile.blockM),
                ("BN", blockN),
                ("BK", blockK),
                ("WM", tile.warpsM),
                ("WN", warpsN),
            ],
            grid: (tilesN * tile.threads, tilesM, 1),
            threadGroup: (tile.threads, 1, 1),
            outputShapes: [[1, queryTokens, blocks]],
            outputDTypes: [.float32])[0]
    }
}

private let steelIndexerScoreKernel: MLXFast.MLXFastKernel? = {
    guard let header = qwen4SteelIndexerHeader() else { return nil }
    return MLXFast.metalKernel(
        name: "darkbloom_qwen4_qsa_indexer_score_steel",
        inputNames: ["queries", "pooled", "mask_q_offset"],
        outputNames: ["scores"],
        source: qwen4SteelIndexerScoreSource,
        header: header,
        ensureRowContiguous: true)
}()

private let qwen4SteelIndexerScoreSource = """
    using namespace mlx::steel;
    constexpr int H = 4;
    constexpr int D = 128;
    constexpr float kInvSqrtD = 0.08838834764831845f;
    const float kSentinel = as_type<float>(uint(0xFF7FFFFF));

    using gemm_kernel = GEMMKernel<T, T, BM, BN, BK, WM, WN, false, true, true, true, float>;
    using loader_a_t = typename gemm_kernel::loader_a_t;
    using loader_b_t = typename gemm_kernel::loader_b_t;
    using mma_t = typename gemm_kernel::mma_t;

    const uint tid = thread_position_in_threadgroup.x;
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    const uint simd_lane_id = thread_index_in_simdgroup;
    const uint tile_n = threadgroup_position_in_grid.x;
    const uint tile_m = threadgroup_position_in_grid.y;
    const int M = int(queries_shape[2]);
    const int N = int(pooled_shape[2]);
    const int tiles_n = (N + BN - 1) / BN;
    const int tiles_m = (M + BM - 1) / BM;
    constexpr int kThreads = WM * WN * 32;
    if (int(tile_n) >= tiles_n || int(tile_m) >= tiles_m || int(tid) >= kThreads) {
        return;
    }

    const int c_row = int(tile_m) * BM;
    const int c_col = int(tile_n) * BN;
    const short tgp_bm = short(metal::min(BM, M - c_row));
    const short tgp_bn = short(metal::min(BN, N - c_col));
    const int mask_off = int(mask_q_offset[0]);
    const int lda = D;
    const int ldb = D;
    const int ldd = N;
    constexpr int gemm_k_iterations = D / BK;

    threadgroup T As[gemm_kernel::tgp_mem_size_a];
    threadgroup T Bs[gemm_kernel::tgp_mem_size_b];
    thread mma_t mma_op(simd_group_id, simd_lane_id);

    float accum[decltype(mma_op.Ctile)::kElemsPerTile];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(mma_op.Ctile)::kElemsPerTile; ++i) {
        accum[i] = 0.0f;
    }

    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < H; ++h) {
        mma_op.Ctile.clear();
        const device T* A = queries + ulong(h) * ulong(M) * ulong(D) + ulong(c_row) * ulong(D);
        const device T* B = pooled + ulong(c_col) * ulong(D);
        thread loader_a_t loader_a(A, lda, As, ushort(simd_group_id), ushort(simd_lane_id));
        thread loader_b_t loader_b(B, ldb, Bs, ushort(simd_group_id), ushort(simd_lane_id));
        for (int d = 0; d < gemm_k_iterations; ++d) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tgp_bm == BM) {
                loader_a.load_unsafe();
            } else {
                loader_a.load_safe(short2(BK, tgp_bm));
            }
            if (tgp_bn == BN) {
                loader_b.load_unsafe();
            } else {
                loader_b.load_safe(short2(BK, tgp_bn));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            mma_op.mma(As, Bs);
            loader_a.next();
            loader_b.next();
        }
        threadgroup_barrier(mem_flags::mem_none);

        short ai = 0;
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
                thread const auto& frag = mma_op.Ctile.frag_at(i, j);
                STEEL_PRAGMA_UNROLL
                for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
                    accum[ai++] += metal::max(frag[e], 0.0f);
                }
            }
        }
    }

    device float* Dst = scores + size_t(c_row) * size_t(ldd) + size_t(c_col)
        + size_t(mma_op.sm) * size_t(ldd) + size_t(mma_op.sn);
    short ai = 0;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
        const int row = c_row + mma_op.sm + i * mma_t::TM_stride;
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
            const int col_base = c_col + mma_op.sn + j * mma_t::TN_stride;
            const int out_base =
                (i * decltype(mma_op.Ctile)::kFragRows) * WM * ldd +
                (j * decltype(mma_op.Ctile)::kFragCols) * WN;
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
                const int col = col_base + e;
                const bool masked = col >= (mask_off + row + 1) / 4;
                const float value = masked ? kSentinel : accum[ai] * kInvSqrtD;
                if (row < M && col < N) {
                    Dst[out_base + e] = value;
                }
                ai++;
            }
        }
    }
    """

private func qwen4SteelIndexerHeader() -> String? {
    Qwen4ExpMetalHeaders.gemm
}
