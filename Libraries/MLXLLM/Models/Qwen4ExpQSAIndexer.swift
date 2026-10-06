// Copyright © 2026 Eigen Labs / Darkbloom Assignment 1.
//
// Fusion #3244 `qwen4_qsa_indexer_scores` + `qwen4_qsa_topk_indices`.
// Default scores path is the first JIT MMA port. Fusion's Steel
// GEMMKernel + BlockLoader (`Qwen4ExpQSAIndexerSteel.swift`) is opt-in
// (`DARKBLOOM_QWEN4_QSA_INDEXER_STEEL=1`) — official 128K unique-salt
// missed vs JIT. Top-k is Fusion's FP32 radix selector with
// highest-index cutoff ties (`mx.argpartition(..., kth=-512)[..., -512:]`).
//
// Kill: DARKBLOOM_QWEN4_QSA_INDEXER=0 (scores) and/or
// DARKBLOOM_QWEN4_QSA_TOPK=0. Shape / dtype misses return nil so gathered
// QSA keeps the portable GEMM + argPartition path.

import Foundation
import MLX
import MLXLMCommon

enum Qwen4ExpIndexerScoreInvocation: Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var steel = 0
    nonisolated(unsafe) private static var jit = 0

    struct Snapshot: Sendable, Equatable {
        var steel: Int
        var jit: Int
    }

    static func reset() {
        lock.lock()
        steel = 0
        jit = 0
        lock.unlock()
    }

    static func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(steel: steel, jit: jit)
    }

    static func recordSteel() {
        lock.lock()
        steel += 1
        lock.unlock()
    }

    static func recordJIT() {
        lock.lock()
        jit += 1
        lock.unlock()
    }
}

enum Qwen4ExpNativeIndexer: Sendable {
    static let scoreEnvFlag = "DARKBLOOM_QWEN4_QSA_INDEXER"
    static let steelEnvFlag = Qwen4ExpSteelIndexer.envFlag
    static let topKEnvFlag = "DARKBLOOM_QWEN4_QSA_TOPK"
    static let indexerHeads = 4
    static let headDim = 128
    static let topK = 512
    static let maskRatio = 4
    static let tileM = 64
    static let tileN = 64
    static let scoreThreads = 128
    static let topKThreads = 256

    private static let scoreLock = NSLock()
    nonisolated(unsafe) private static var scoreDisabled = false
    nonisolated(unsafe) private static var scoreProven = false
    private static let topKLock = NSLock()
    nonisolated(unsafe) private static var topKDisabled = false
    nonisolated(unsafe) private static var topKProven = false

    static func scoresEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        enabled(environment[scoreEnvFlag])
    }

    static func topKEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        enabled(environment[topKEnvFlag])
    }

    static func matchesScoreGeometry(queryHeads: Int, headDim: Int) -> Bool {
        queryHeads == indexerHeads && headDim == Self.headDim
    }

    /// Fusion `qwen4_qsa_indexer_scores`: q `[1,M,4,128]` → transposed
    /// `[1,4,M,128]`, pooled k `[1,N,128]` → `[1,1,N,128]`, fp16/bf16,
    /// mask_ratio=4. Writes fp32 `[1,M,N]`.
    static func scores(
        queries: MLXArray, pooledKeys: MLXArray, maskQOffset: Int,
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> MLXArray? {
        guard scoresEnabled(environment: environment), !isScoreDisabled else { return nil }
        guard maskQOffset >= 0, queries.ndim == 4, queries.dim(0) == 1,
            queries.dim(2) == indexerHeads, queries.dim(3) == headDim,
            pooledKeys.ndim == 3, pooledKeys.dim(0) == 1,
            pooledKeys.dim(2) == headDim
        else { return nil }
        let queryTokens = queries.dim(1)
        let blocks = pooledKeys.dim(1)
        guard queryTokens > 0, blocks > 0 else { return nil }
        let dtype = Qwen4ExpNativeSparseGQA.activationType(queries.dtype)
        guard Qwen4ExpNativeSparseGQA.matchesDtype(dtype),
            Qwen4ExpNativeSparseGQA.matchesDtype(
                Qwen4ExpNativeSparseGQA.activationType(pooledKeys.dtype))
        else { return nil }

        let q = queries.transposed(0, 2, 1, 3).asType(dtype)
        let k = pooledKeys.reshaped([1, 1, blocks, headDim]).asType(dtype)
        let offset = MLXArray([Int32(clamping: maskQOffset)])
        if let steel = Qwen4ExpSteelIndexer.scores(
            queries: q, pooled: k, maskQOffset: offset,
            queryTokens: queryTokens, blocks: blocks, dtype: dtype,
            environment: environment)
        {
            if !isScoreProven {
                CBv2DeferredHostFill.resolveBeforeEvaluation()
                eval(steel)
                markScoreProven()
            }
            Qwen4ExpIndexerScoreInvocation.recordSteel()
            return steel
        }
        let tilesN = (blocks + tileN - 1) / tileN
        let tilesM = (queryTokens + tileM - 1) / tileM
        let output = qwen4IndexerScoreKernel(
            [q, k, offset],
            template: [("T", dtype)],
            grid: (tilesN * scoreThreads, tilesM, 1),
            threadGroup: (scoreThreads, 1, 1),
            outputShapes: [[1, queryTokens, blocks]],
            outputDTypes: [.float32])[0]
        if !isScoreProven {
            CBv2DeferredHostFill.resolveBeforeEvaluation()
            eval(output)
            markScoreProven()
        }
        Qwen4ExpIndexerScoreInvocation.recordJIT()
        return output
    }

    /// Fusion `qwen4_qsa_topk_indices`: fp32 `[1,M,N>=512]` → uint32 `[1,M,512]`.
    static func topKIndices(_ scores: MLXArray) -> MLXArray? {
        guard topKEnabled(), !isTopKDisabled else { return nil }
        guard scores.ndim == 3, scores.dim(0) == 1, scores.dtype == .float32
        else { return nil }
        let rows = scores.dim(1)
        let cols = scores.dim(2)
        guard rows >= 1, cols >= topK else { return nil }
        let output = qwen4IndexerTopKKernel(
            [scores],
            grid: (topKThreads * rows, 1, 1),
            threadGroup: (topKThreads, 1, 1),
            outputShapes: [[1, rows, topK]],
            outputDTypes: [.uint32])[0]
        if !isTopKProven {
            CBv2DeferredHostFill.resolveBeforeEvaluation()
            eval(output)
            markTopKProven()
        }
        return output
    }

    private static func enabled(_ raw: String?) -> Bool {
        let value = raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if value == "0" || value == "false" || value == "no" || value == "off" {
            return false
        }
        return true
    }

    private static var isScoreDisabled: Bool {
        scoreLock.lock()
        defer { scoreLock.unlock() }
        return scoreDisabled
    }

    private static var isScoreProven: Bool {
        scoreLock.lock()
        defer { scoreLock.unlock() }
        return scoreProven
    }

    private static func markScoreProven() {
        scoreLock.lock()
        scoreProven = true
        scoreLock.unlock()
    }

    private static var isTopKDisabled: Bool {
        topKLock.lock()
        defer { topKLock.unlock() }
        return topKDisabled
    }

    private static var isTopKProven: Bool {
        topKLock.lock()
        defer { topKLock.unlock() }
        return topKProven
    }

    private static func markTopKProven() {
        topKLock.lock()
        topKProven = true
        topKLock.unlock()
    }
}

/// MMA helpers shared with Steel s64 (load converts T → fp32 accumulate).
private let qwen4IndexerScoreHeader = """
    #include <metal_stdlib>
    #include <metal_simdgroup>
    #include <metal_simdgroup_matrix>
    using namespace metal;

    #define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
    #define STEEL_CONST static constant constexpr const

    template <typename Acc>
    struct QsaBaseMMAFrag {
        STEEL_CONST int kFragRows = 8;
        STEEL_CONST int kFragCols = 8;
        STEEL_CONST int kElemsPerFrag = 2;
        STEEL_CONST int kElemRows = 1;
        STEEL_CONST int kElemCols = 2;
        typedef metal::simdgroup_matrix<Acc, 8, 8> mat_type;
        typedef metal::vec<Acc, 2> frag_type;

        METAL_FUNC static short2 get_coord(ushort simd_lane_id) {
            const short qid = simd_lane_id / 4;
            const short fm = (qid & 4) + ((simd_lane_id / 2) % 4);
            const short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
            return short2{fn, fm};
        }

        template <typename SrcPtrType>
        METAL_FUNC static void load(
            thread frag_type& dst, SrcPtrType src, int str_x, int str_y
        ) {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kElemRows; i++) {
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < kElemCols; j++) {
                    dst[i * kElemCols + j] =
                        static_cast<Acc>(src[i * str_x + j * str_y]);
                }
            }
        }

        METAL_FUNC static void mma(
            thread frag_type& D,
            thread frag_type& A,
            thread frag_type& B,
            thread frag_type& C
        ) {
            mat_type D_mat;
            mat_type A_mat;
            mat_type B_mat;
            mat_type C_mat;
            A_mat.thread_elements()[0] = A[0];
            A_mat.thread_elements()[1] = A[1];
            B_mat.thread_elements()[0] = B[0];
            B_mat.thread_elements()[1] = B[1];
            C_mat.thread_elements()[0] = C[0];
            C_mat.thread_elements()[1] = C[1];
            simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat);
            D[0] = D_mat.thread_elements()[0];
            D[1] = D_mat.thread_elements()[1];
        }
    };

    template <typename Acc, int kTileRows_, int kTileCols_>
    struct QsaMMATile {
        using Frag = QsaBaseMMAFrag<Acc>;
        using frag_type = typename Frag::frag_type;
        STEEL_CONST int kFragRows = 8;
        STEEL_CONST int kFragCols = 8;
        STEEL_CONST int kTileRows = kTileRows_;
        STEEL_CONST int kTileCols = kTileCols_;
        STEEL_CONST int kNumFrags = kTileRows_ * kTileCols_;
        STEEL_CONST int kElemsPerTile = kNumFrags * 2;
        frag_type val_frags[kNumFrags];

        METAL_FUNC void clear() thread {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kNumFrags; ++i) {
                val_frags[i] = frag_type(0);
            }
        }

        METAL_FUNC thread frag_type& frag_at(short i, short j) thread {
            return val_frags[i * kTileCols + j];
        }

        METAL_FUNC thread Acc* elems() thread {
            return reinterpret_cast<thread Acc*>(val_frags);
        }

        template <typename U, int w_x, int w_y, int str_x, int str_y>
        METAL_FUNC void load(const threadgroup U* src) thread {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kTileRows; ++i) {
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < kTileCols; ++j) {
                    Frag::load(
                        frag_at(i, j),
                        &(src[(i * kFragRows) * w_x * str_x
                            + (j * kFragCols) * w_y * str_y]),
                        str_x,
                        str_y);
                }
            }
        }
    };

    template <typename Acc, int M, int N, int K>
    METAL_FUNC void qsa_tile_matmad(
        thread QsaMMATile<Acc, M, N>& D,
        thread QsaMMATile<Acc, M, K>& A,
        thread QsaMMATile<Acc, K, N>& B,
        thread QsaMMATile<Acc, M, N>& C
    ) {
        STEEL_PRAGMA_UNROLL
        for (short m = 0; m < M; ++m) {
            STEEL_PRAGMA_UNROLL
            for (short n = 0; n < N; ++n) {
                short n_serp = (m % 2) ? (N - 1 - n) : n;
                STEEL_PRAGMA_UNROLL
                for (short k = 0; k < K; ++k) {
                    QsaBaseMMAFrag<Acc>::mma(
                        D.frag_at(m, n_serp),
                        A.frag_at(m, k),
                        B.frag_at(k, n_serp),
                        C.frag_at(m, n_serp));
                }
            }
        }
    }

    METAL_FUNC uint qwen4_ordered_key_32(float x) {
        uint bits = as_type<uint>(x);
        if ((bits & 0x7fffffffu) == 0u) {
            bits = 0u;
        }
        return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    }
"""

private let qwen4IndexerScoreKernel = MLXFast.metalKernel(
    name: "darkbloom_qwen4_qsa_indexer_score",
    inputNames: ["queries", "pooled", "mask_q_offset"],
    outputNames: ["scores"],
    source: """
        constexpr int BM = 64;
        constexpr int BN = 64;
        constexpr int BK = 16;
        constexpr int LDA = 24;
        constexpr int H = 4;
        constexpr int D = 128;
        constexpr int WM = 2;
        constexpr int WN = 2;
        constexpr float kInvSqrtD = 0.08838834764831845f;
        const float kSentinel = as_type<float>(uint(0xFF7FFFFF));

        const uint tid = thread_position_in_threadgroup.x;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint tile_n = threadgroup_position_in_grid.x;
        const uint tile_m = threadgroup_position_in_grid.y;
        const int M = int(queries_shape[2]);
        const int N = int(pooled_shape[2]);
        const int tiles_n = (N + BN - 1) / BN;
        if (int(tile_n) >= tiles_n || int(tid) >= 128) {
            return;
        }

        const int c_row = int(tile_m) * BM;
        const int c_col = int(tile_n) * BN;
        const int sg_m = int(sg / WN) * 32;
        const int sg_n = int(sg % WN) * 32;
        const bool interior = (c_row + BM <= M) && (c_col + BN <= N);
        const int mask_off = int(mask_q_offset[0]);
        const ulong q_h = ulong(queries_strides[1]);
        const ulong q_m = ulong(queries_strides[2]);
        const ulong q_d = ulong(queries_strides[3]);
        const ulong k_n = ulong(pooled_strides[2]);
        const ulong k_d = ulong(pooled_strides[3]);

        threadgroup T As[BM * LDA];
        threadgroup T Bs[BN * LDA];

        using Frag = QsaBaseMMAFrag<float>;
        QsaMMATile<float, 4, 4> Acc;
        Acc.clear();
        const short2 coord = Frag::get_coord(ushort(lane));

        STEEL_PRAGMA_UNROLL
        for (int h = 0; h < H; ++h) {
            QsaMMATile<float, 4, 4> Ctile;
            Ctile.clear();
            const device T* q_head = queries + ulong(h) * q_h;
            for (int d0 = 0; d0 < D; d0 += BK) {
                if (interior && q_d == 1 && k_d == 1) {
                    const int r = int(tid) >> 1;
                    const int c = (int(tid) & 1) * 8;
                    const device T* q_src = q_head
                        + ulong(c_row + r) * q_m + ulong(d0 + c);
                    const device T* k_src = pooled
                        + ulong(c_col + r) * k_n + ulong(d0 + c);
                    *((threadgroup uint4*)(As + r * LDA + c)) =
                        *((const device uint4*)q_src);
                    *((threadgroup uint4*)(Bs + r * LDA + c)) =
                        *((const device uint4*)k_src);
                } else {
                    for (int i = int(tid); i < BM * BK; i += 128) {
                        const int r = i / BK;
                        const int c = i - r * BK;
                        const int gr = c_row + r;
                        const int gc = d0 + c;
                        const int gn = c_col + r;
                        As[r * LDA + c] = (gr < M && gc < D)
                            ? q_head[ulong(gr) * q_m + ulong(gc) * q_d]
                            : T(0);
                        Bs[r * LDA + c] = (gn < N && gc < D)
                            ? pooled[ulong(gn) * k_n + ulong(gc) * k_d]
                            : T(0);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                STEEL_PRAGMA_UNROLL
                for (int kk = 0; kk < BK; kk += 8) {
                    QsaMMATile<float, 4, 1> Atile;
                    QsaMMATile<float, 1, 4> Btile;
                    Atile.template load<T, 1, 1, LDA, 1>(
                        &As[(sg_m + coord.y) * LDA + kk + coord.x]);
                    Btile.template load<T, 1, 1, 1, LDA>(
                        &Bs[(sg_n + coord.x) * LDA + kk + coord.y]);
                    qsa_tile_matmad<float, 4, 4, 1>(Ctile, Atile, Btile, Ctile);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            STEEL_PRAGMA_UNROLL
            for (int t = 0; t < 32; ++t) {
                Acc.elems()[t] += metal::max(Ctile.elems()[t], 0.0f);
            }
        }

        const short sm = coord.y;
        const short sn = coord.x;
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) {
            const int row = c_row + sg_m + i * 8 + sm;
            STEEL_PRAGMA_UNROLL
            for (int j = 0; j < 4; ++j) {
                const thread auto& frag = Acc.frag_at(i, j);
                STEEL_PRAGMA_UNROLL
                for (int e = 0; e < 2; ++e) {
                    const int col = c_col + sg_n + j * 8 + sn + e;
                    const bool masked = col >= (mask_off + row + 1) / 4;
                    if (row < M && col < N) {
                        scores[size_t(row) * size_t(N) + size_t(col)] =
                            masked ? kSentinel : frag[e] * kInvSqrtD;
                    }
                }
            }
        }
    """,
    header: qwen4IndexerScoreHeader,
    ensureRowContiguous: true
)

private let qwen4IndexerTopKKernel = MLXFast.metalKernel(
    name: "darkbloom_qwen4_qsa_topk_indices",
    inputNames: ["scores"],
    outputNames: ["indices"],
    source: """
        constexpr int TOPK = 512;
        constexpr int THREADS = 256;
        constexpr uint kSimdgroups = THREADS / 32;

        const uint tid = thread_position_in_threadgroup.x;
        const uint row = threadgroup_position_in_grid.x;
        const uint rows = uint(scores_shape[1]);
        const uint K = uint(scores_shape[2]);
        if (row >= rows || tid >= uint(THREADS) || K < uint(TOPK)) {
            return;
        }

        threadgroup atomic_uint hist[256];
        threadgroup uint state[2];
        threadgroup uint partial_a[kSimdgroups];
        threadgroup uint partial_b[kSimdgroups];

        if (tid == 0) {
            state[0] = 0;
            state[1] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint logical_tid = THREADS - 1 - tid;
        const uint segment = (K + THREADS - 1) / THREADS;
        const uint start = logical_tid * segment;
        const uint stop = metal::min(start + segment, K);
        const uint count = start < K ? stop - start : 0u;
        const device float* row_scores =
            scores + size_t(row) * size_t(scores_strides[1]);

        for (int shift = 24; shift >= 0; shift -= 8) {
            if (tid < 256) {
                atomic_store_explicit(&hist[tid], 0, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint prefix = state[0];
            for (uint step = 0; step < count; ++step) {
                const uint index = stop - 1 - step;
                const uint key = qwen4_ordered_key_32(row_scores[index]);
                const bool prefix_match = shift == 24 ||
                    (key >> uint(shift + 8)) == (prefix >> uint(shift + 8));
                if (prefix_match) {
                    atomic_fetch_add_explicit(
                        &hist[(key >> uint(shift)) & 255u],
                        1, memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
                uint greater = state[1];
                uint selected_byte = 0;
                for (int byte = 255; byte >= 0; --byte) {
                    const uint bucket = atomic_load_explicit(
                        &hist[byte], memory_order_relaxed);
                    if (greater + bucket >= uint(TOPK)) {
                        selected_byte = uint(byte);
                        break;
                    }
                    greater += bucket;
                }
                state[0] |= selected_byte << uint(shift);
                state[1] = greater;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const uint threshold = state[0];
        const uint greater_total = state[1];
        uint local_greater = 0;
        uint local_ties = 0;
        for (uint step = 0; step < count; ++step) {
            const uint index = stop - 1 - step;
            const uint key = qwen4_ordered_key_32(row_scores[index]);
            local_greater += key > threshold ? 1u : 0u;
            local_ties += key == threshold ? 1u : 0u;
        }

        const uint lane = tid & 31u;
        const uint simd = tid >> 5;
        uint pre_greater = metal::simd_prefix_exclusive_sum(local_greater);
        uint pre_ties = metal::simd_prefix_exclusive_sum(local_ties);
        if (lane == 31) {
            partial_a[simd] = pre_greater + local_greater;
            partial_b[simd] = pre_ties + local_ties;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0) {
            const uint group_greater = lane < kSimdgroups ? partial_a[lane] : 0u;
            const uint group_ties = lane < kSimdgroups ? partial_b[lane] : 0u;
            const uint group_pre_greater =
                metal::simd_prefix_exclusive_sum(group_greater);
            const uint group_pre_ties =
                metal::simd_prefix_exclusive_sum(group_ties);
            if (lane < kSimdgroups) {
                partial_a[lane] = group_pre_greater;
                partial_b[lane] = group_pre_ties;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        pre_greater += partial_a[simd];
        pre_ties += partial_b[simd];

        const uint tie_budget = uint(TOPK) - greater_total;
        const uint selected_ties = pre_ties >= tie_budget
            ? 0u
            : metal::min(local_ties, tie_budget - pre_ties);
        const uint local_selected = local_greater + selected_ties;

        uint pre_selected = metal::simd_prefix_exclusive_sum(local_selected);
        if (lane == 31) {
            partial_a[simd] = pre_selected + local_selected;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0) {
            const uint group_selected = lane < kSimdgroups ? partial_a[lane] : 0u;
            const uint group_pre_selected =
                metal::simd_prefix_exclusive_sum(group_selected);
            if (lane < kSimdgroups) {
                partial_a[lane] = group_pre_selected;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint output_pos = partial_a[simd] + pre_selected;
        uint ties_seen = pre_ties;
        device uint* row_out = indices + size_t(row) * size_t(TOPK);
        for (uint step = 0; step < count; ++step) {
            const uint index = stop - 1 - step;
            const uint key = qwen4_ordered_key_32(row_scores[index]);
            bool selected = key > threshold;
            if (key == threshold) {
                selected = ties_seen < tie_budget;
                ties_seen++;
            }
            if (selected) {
                row_out[output_pos++] = index;
            }
        }
    """,
    header: qwen4IndexerScoreHeader,
    ensureRowContiguous: true
)
