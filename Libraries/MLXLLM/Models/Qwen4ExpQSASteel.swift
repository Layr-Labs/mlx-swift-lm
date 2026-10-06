// Copyright © 2026 Eigen Labs / Darkbloom Assignment 1.
//
// Fusion `qwen4_qsa_sparse_gqa` (WM=2, default BK=64 DC=64) adapted for
// `MLXFast.metalKernel`. Steel headers cannot be `#include`d from JIT, so
// this copies the 8×8 MMA fragment / tile helpers Fusion uses and the
// sparse-GQA body (uint4 T staging, float accumulate, exp2 softmax).
// BK/DC are kernel template ints. Lab 128/32:
// `DARKBLOOM_QWEN4_QSA_STEEL_BK128=1`. Kill with
// `DARKBLOOM_QWEN4_QSA_STEEL=0` to keep scalar c8s.
//
// Dispatch matches Fusion `dispatch_threadgroups((qL, kv, 1), (32, WM, 1))`
// via MLX `dispatch_threads((qL*32, kv*WM, 1), (32, WM, 1))`.

import Foundation
import MLX
import MLXLMCommon

extension Qwen4ExpNativeSparseGQA {
    static let steelEnvFlag = "DARKBLOOM_QWEN4_QSA_STEEL"
    /// Fusion `qsa_fast` production is BK=64 DC=64. The C++ ABI default is
    /// 128/32. Lab: `DARKBLOOM_QWEN4_QSA_STEEL_BK128=1`. Do not default on
    /// until official 8K+128K beat 64/64.
    static let steelBK128EnvFlag = "DARKBLOOM_QWEN4_QSA_STEEL_BK128"
    static let fusionKeyTile = 64
    static let fusionDimensionTile = 64
    static let labKeyTile = 128
    static let labDimensionTile = 32
    static let steelSimdWidth = 32
    static let steelWarps = 2
    static var steelThreads: Int { steelSimdWidth * steelWarps }

    static func steelEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        let raw = environment[steelEnvFlag]?.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        if raw == "0" || raw == "false" || raw == "no" || raw == "off" {
            return false
        }
        return true
    }

    static func steelBK128Enabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        let raw = environment[steelBK128EnvFlag]?.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
    }

    static func resolvedSteelTiles(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> (keyTile: Int, dimensionTile: Int) {
        if steelBK128Enabled(environment: environment) {
            return (labKeyTile, labDimensionTile)
        }
        return (fusionKeyTile, fusionDimensionTile)
    }
}

/// Minimal Steel 8×8 MMA (Fusion `steel/attn/mma.h` subset).
let qwen4SparseGQASteelHeader = """
    #include <metal_stdlib>
    #include <metal_simdgroup>
    #include <metal_simdgroup_matrix>
    using namespace metal;

    #define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
    #define STEEL_CONST static constant constexpr const

    struct QsaMaxOp {
        template <typename U>
        METAL_FUNC static U apply(U x, U y) { return metal::max(x, y); }
    };
    struct QsaSumOp {
        template <typename U>
        METAL_FUNC static U apply(U x, U y) { return x + y; }
    };
    struct QsaMulOp {
        template <typename U>
        METAL_FUNC static U apply(U x, U y) { return x * y; }
    };
    struct QsaExpSubOp {
        template <typename U>
        METAL_FUNC static U apply(U x, U y) { return metal::fast::exp2(x - y); }
    };
    struct QsaDivOp {
        template <typename U>
        METAL_FUNC static U apply(U x, U y) { return x / y; }
    };

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

        template <typename U>
        METAL_FUNC static void store_safe(
            const thread frag_type& src,
            device U* dst,
            int str_x,
            int str_y,
            short lim_x,
            short lim_y,
            int off_x,
            int off_y
        ) {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kElemRows; i++) {
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < kElemCols; j++) {
                    if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
                        dst[(off_x + i) * str_x + (off_y + j) * str_y] =
                            static_cast<U>(src[i * kElemCols + j]);
                    }
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

        template <typename Op>
        METAL_FUNC static void row_reduce(
            thread const frag_type& inp, thread Acc* reduced
        ) {
            Acc thr = Op::apply(inp.x, inp.y);
            Acc qgr = simd_shuffle_xor(thr, ushort(1));
            qgr = Op::apply(thr, qgr);
            Acc sgr = simd_shuffle_xor(qgr, ushort(8));
            sgr = Op::apply(qgr, sgr);
            reduced[0] = Op::apply(reduced[0], sgr);
        }

        template <typename Op>
        METAL_FUNC static void row_bin_op(
            thread frag_type& inp, thread Acc* row_vals
        ) {
            inp[0] = Op::apply(inp[0], row_vals[0]);
            inp[1] = Op::apply(inp[1], row_vals[0]);
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
        STEEL_CONST int kRowsPerThread = kTileRows_;
        using MMAFrag_t = Frag;
        using elem_type = Acc;
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

        template <typename Op>
        METAL_FUNC void row_reduce(thread Acc vals[kRowsPerThread]) const thread {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kTileRows; ++i) {
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < kTileCols; ++j) {
                    Frag::template row_reduce<Op>(
                        val_frags[i * kTileCols + j], &vals[i]);
                }
            }
        }

        template <typename Op>
        METAL_FUNC void row_bin_op(thread Acc vals[kRowsPerThread]) thread {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kTileRows; ++i) {
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < kTileCols; ++j) {
                    Frag::template row_bin_op<Op>(
                        val_frags[i * kTileCols + j], &vals[i]);
                }
            }
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

        template <typename U, int w_x, int w_y>
        METAL_FUNC void store_safe(
            device U* dst, int ld, short2 dims
        ) const thread {
            STEEL_PRAGMA_UNROLL
            for (int i = 0; i < kTileRows; ++i) {
                STEEL_PRAGMA_UNROLL
                for (int j = 0; j < kTileCols; ++j) {
                    Frag::store_safe(
                        val_frags[i * kTileCols + j],
                        dst,
                        ld,
                        1,
                        dims.y,
                        dims.x,
                        (i * kFragRows) * w_x,
                        (j * kFragCols) * w_y);
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
"""

func qwen4SparseGQASteelPrefix(qkOnly: Bool = false) -> String {
    """
        uint simd_lane_id = thread_index_in_simdgroup;
        uint simd_group_id = simdgroup_index_in_threadgroup;
        uint3 tid = threadgroup_position_in_grid;
        uint qL = uint(queries_shape[2]);
        uint kL = COMPACT_KV ? uint(q_offset[1]) : uint(keys_shape[2]);
        if (tid.x >= qL || tid.y >= 2u || tid.z >= \(qkOnly ? "uint((2051 + BK - 1) / BK)" : "uint(OPARTS)")) {
            return;
        }

        // BK and DC are MLX kernel template ints (default 64/64).
        constexpr int GQA = 12;
        constexpr int H_PAD = 16;
        constexpr int D = 256;
        constexpr int WM = 2;
        constexpr short kFragSize = 8;
        constexpr short pad = 16 / sizeof(T);
        constexpr short LDQ = DC + pad;
        constexpr short LDK = BK + pad;
        constexpr short LDV = DC + pad;
        constexpr int TQ = 1;
        constexpr int TK = BK / kFragSize;
        constexpr int TDC = DC / kFragSize;
        constexpr int D_CHUNKS = D / DC;
        constexpr int O_CHUNKS = D_CHUNKS / OPARTS;
        constexpr int tgp_size = WM * 32;

        const int lane = int(simd_group_id * 32 + simd_lane_id);
        const int q_pos = int(tid.x);
        const int kv_head = int(tid.y);

        threadgroup T Qs[H_PAD * LDQ];
        threadgroup T KVs[(BK * LDV > DC * LDK) ? BK * LDV : DC * LDK];
        threadgroup int selected_pos[BK];

        using Frag = QsaBaseMMAFrag<float>;
        QsaMMATile<float, TQ, 1> Qtile;
        QsaMMATile<float, 1, TK> Ktile;
        QsaMMATile<float, TQ, TK> Stile;
        QsaMMATile<float, 1, 1> Vtile;
        QsaMMATile<float, TQ, O_CHUNKS * TDC> Otile;
        Otile.clear();

        const short2 simd_coord = Frag::get_coord(ushort(simd_lane_id));
        const short sm = simd_coord.y;
        const short sn = simd_coord.x;
        const short tm = kFragSize * TQ * short(simd_group_id);
        const short Qs_offset = (tm + sm) * LDQ + sn;
        const short Ks_offset = sm * LDK + sn;
        const short Vs_offset = sm * LDV + sn;

        const float scale = (1.0f / 16.0f) * M_LOG2E_F;
        float max_score[1];
        float sum_score[1] = {0};
        max_score[0] = Limits<float>::finite_min;

        const int query_head_base = kv_head * GQA;
        const device T* q_base = queries
            + size_t(query_head_base) * size_t(queries_strides[1])
            + size_t(q_pos) * size_t(queries_strides[2]);
        const device T* k_base = keys
            + int64_t(kv_head) * keys_strides[1];
        const device T* v_base = values
            + int64_t(kv_head) * values_strides[1];
        const size_t topk_row = size_t(q_pos) * size_t(selected_strides[2]);

        const int q_abs = int(q_offset[0]) + q_pos;
        constexpr int kCompressRatio = 4;
        constexpr int kTail = 3;
        constexpr int kTopk = 512;
        const int selected_tokens = kTopk * kCompressRatio + kTail;
        const int complete_blocks = (q_abs + 1) / kCompressRatio;
        const int valid_blocks = complete_blocks < kTopk ? complete_blocks : kTopk;
        const int n_tiles = (selected_tokens + BK - 1) / BK;

        for (int ktile = \(qkOnly ? "int(tid.z)" : "0"); ktile < \(qkOnly ? "int(tid.z) + 1" : "n_tiles"); ++ktile) {
            const int topk_off = ktile * BK;
            for (int k = lane; k < BK; k += tgp_size) {
                const int slot = topk_off + k;
                int k_pos = -1;
                if (slot < kTopk * kCompressRatio) {
                    const int block_slot = slot / kCompressRatio;
                    if (block_slot < valid_blocks) {
                        const uint raw_block = uint(selected[topk_row + size_t(block_slot)]);
                        const ulong candidate = ulong(raw_block) * 4ul
                            + ulong(slot % kCompressRatio);
                        if (candidate < ulong(kL) && candidate <= ulong(q_abs)) {
                            k_pos = int(candidate);
                        }
                    }
                } else if (slot < selected_tokens) {
                    const int candidate = complete_blocks * 4
                        + (slot - kTopk * kCompressRatio);
                    if (candidate < int(kL) && candidate <= q_abs) {
                        k_pos = candidate;
                    }
                }
                selected_pos[k] = k_pos;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

    """
}

let qwen4SparseGQASteelDotSource = """
            Stile.clear();
            STEEL_PRAGMA_UNROLL
            for (short dchunk = 0; dchunk < D_CHUNKS; ++dchunk) {
                const int dbase = int(dchunk) * DC;
                for (int elem = lane; elem < H_PAD * (DC / 8); elem += tgp_size) {
                    const int h = elem / (DC / 8);
                    const int d8 = elem - h * (DC / 8);
                    uint4 word = uint4(0);
                    if (h < GQA) {
                        if (STRIDED_KV) {
                            // contiguous() may retain an offset view. Raw
                            // loads also make query staging alignment-safe.
                            word = qsa_load_raw_kv_bits(q_base,
                                int64_t(h) * queries_strides[1]
                                    + int64_t(dbase + d8 * 8) * queries_strides[3],
                                queries_strides[3]);
                        } else {
                            word = *((const device uint4*)(q_base
                                + size_t(h) * size_t(queries_strides[1])
                                + dbase) + d8);
                        }
                    }
                    *((threadgroup uint4*)(Qs + h * LDQ) + d8) = word;
                }
                for (int elem = lane; elem < BK * (DC / 8); elem += tgp_size) {
                    const int k = elem / (DC / 8);
                    const int d8 = elem - k * (DC / 8);
                    const int k_pos = selected_pos[k];
                    uint4 word = uint4(0);
                    if (k_pos >= 0) {
                        const size_t physical_row = COMPACT_KV
                            ? size_t(q_pos) * size_t(selected_tokens) + size_t(topk_off + k)
                            : size_t(k_pos);
                        if (STRIDED_KV) {
                            word = qsa_load_raw_kv_bits(k_base,
                                int64_t(physical_row) * keys_strides[2]
                                    + int64_t(dbase + d8 * 8) * keys_strides[3],
                                keys_strides[3]);
                        } else {
                            word = *((const device uint4*)(k_base
                                + physical_row * size_t(keys_strides[2])
                                + dbase) + d8);
                        }
                    }
                    thread T* packed = (thread T*)&word;
                    const int d = d8 * 8;
                    STEEL_PRAGMA_UNROLL
                    for (short e = 0; e < 8; ++e) {
                        KVs[k + (d + e) * LDK] = packed[e];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                STEEL_PRAGMA_UNROLL
                for (short dd = 0; dd < TDC; ++dd) {
                    simdgroup_barrier(mem_flags::mem_none);
                    Qtile.template load<T, 1, 1, LDQ, 1>(
                        &Qs[Qs_offset + dd * kFragSize]);
                    Ktile.template load<T, 1, 1, LDK, 1>(
                        &KVs[Ks_offset + dd * kFragSize * LDK]);
                    simdgroup_barrier(mem_flags::mem_none);
                    qsa_tile_matmad<float, TQ, TK, 1>(Stile, Qtile, Ktile, Stile);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

    """

let qwen4SparseGQASteelValueSource = """
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < decltype(Stile)::kElemsPerTile; ++i) {
                Stile.elems()[i] *= scale;
            }
            {
                constexpr float neg_inf = float(-INFINITY);
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < decltype(Stile)::kTileCols; ++j) {
                    const short col_pos = sn + j * 8;
                    STEEL_PRAGMA_UNROLL
                    for (short e = 0; e < 2; ++e) {
                        if (selected_pos[col_pos + e] < 0) {
                            Stile.frag_at(0, j)[e] = neg_inf;
                        }
                    }
                }
            }

            float new_max[1];
            float factor[1];
            new_max[0] = max_score[0];
            Stile.template row_reduce<QsaMaxOp>(new_max);
            Stile.template row_bin_op<QsaExpSubOp>(new_max);
            factor[0] = metal::fast::exp2(max_score[0] - new_max[0]);
            max_score[0] = new_max[0];
            float sum_score_tmp[1] = {0};
            Stile.template row_reduce<QsaSumOp>(sum_score_tmp);
            sum_score[0] = sum_score[0] * factor[0] + sum_score_tmp[0];
            Otile.template row_bin_op<QsaMulOp>(factor);

            STEEL_PRAGMA_UNROLL
            for (short vchunk = 0; vchunk < O_CHUNKS; ++vchunk) {
                const int dbase = (int(tid.z) * O_CHUNKS + int(vchunk)) * DC;
                for (int elem = lane; elem < BK * (DC / 8); elem += tgp_size) {
                    const int k = elem / (DC / 8);
                    const int d8 = elem - k * (DC / 8);
                    const int k_pos = selected_pos[k];
                    uint4 word = uint4(0);
                    if (k_pos >= 0) {
                        const size_t physical_row = COMPACT_KV
                            ? size_t(q_pos) * size_t(selected_tokens) + size_t(topk_off + k)
                            : size_t(k_pos);
                        if (STRIDED_KV) {
                            word = qsa_load_raw_kv_bits(v_base,
                                int64_t(physical_row) * values_strides[2]
                                    + int64_t(dbase + d8 * 8) * values_strides[3],
                                values_strides[3]);
                        } else {
                            word = *((const device uint4*)(v_base
                                + physical_row * size_t(values_strides[2])
                                + dbase) + d8);
                        }
                    }
                    *((threadgroup uint4*)(KVs + k * LDV) + d8) = word;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                STEEL_PRAGMA_UNROLL
                for (short id = 0; id < TDC; ++id) {
                    STEEL_PRAGMA_UNROLL
                    for (short ik = 0; ik < TK; ++ik) {
                        const short kk = ik * kFragSize;
                        const short dd = id * kFragSize;
                        Vtile.template load<T, 1, 1, LDV, 1>(
                            &KVs[Vs_offset + kk * LDV + dd]);
                        Frag::mma(
                            Otile.frag_at(0, vchunk * TDC + id),
                            Stile.frag_at(0, ik),
                            Vtile.frag_at(0, 0),
                            Otile.frag_at(0, vchunk * TDC + id));
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        Otile.template row_bin_op<QsaDivOp>(sum_score);
        const int o_ld = int(qL) * D;
        device T* out = output
            + size_t(query_head_base + tm + sm) * size_t(o_ld)
            + size_t(q_pos) * size_t(D)
            + size_t(tid.z) * size_t(O_CHUNKS * DC)
            + size_t(sn);
        const short rows_left = short(GQA - (tm + sm));
        if (rows_left > 0) {
            Otile.template store_safe<T, 1, 1>(out, o_ld, short2(O_CHUNKS * DC - sn, rows_left));
        }
    """

private let qwen4SparseGQASteelSource = qwen4SparseGQASteelPrefix()
    + qwen4SparseGQASteelDotSource + qwen4SparseGQASteelValueSource

/// Raw loads preserve FP16/BF16 payload bits at arbitrary element alignment.
/// Apple MSL specification (2026-06-04), section 2.2.3/table 2.4, p34:
/// packed_ushort4 is eight bytes with TWO-byte alignment, unlike uint4.
/// https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
let qwen4SparseGQAStridedLoadHeader = """
    static_assert(sizeof(packed_ushort4) == 8, "packed ushort4 size changed");
    static_assert(alignof(packed_ushort4) == 2, "packed ushort4 alignment changed");

    template <typename T>
    inline uint4 qsa_load_raw_kv_bits(const device T* base, int64_t start, int64_t stride) {
        if (stride == 1) {
            const device packed_ushort4* packed = (const device packed_ushort4*)(base + start);
            const packed_ushort4 lo = packed[0];
            const packed_ushort4 hi = packed[1];
            return uint4(
                uint(lo.x) | (uint(lo.y) << 16),
                uint(lo.z) | (uint(lo.w) << 16),
                uint(hi.x) | (uint(hi.y) << 16),
                uint(hi.z) | (uint(hi.w) << 16));
        }
        uint4 word = uint4(0);
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 8; ++e) {
            const device ushort* bits = (const device ushort*)(base + start + int64_t(e) * stride);
            word[e / 2] |= uint(*bits) << (16 * (e % 2));
        }
        return word;
    }
    """

private func makeQwen4SparseGQASteelKernel(preserveKVStrides: Bool) -> MLXFast.MLXFastKernel {
    MLXFast.metalKernel(
        name: preserveKVStrides ? "darkbloom_qwen4_qsa_sparse_gqa_s64_strided" : "darkbloom_qwen4_qsa_sparse_gqa_s64",
        inputNames: ["queries", "keys", "values", "selected", "q_offset"],
        outputNames: ["output"], source: qwen4SparseGQASteelSource,
        header: qwen4SparseGQASteelHeader + "\n" + qwen4SparseGQAStridedLoadHeader,
        ensureRowContiguous: !preserveKVStrides)
}

let qwen4SparseGQASteelKernel = makeQwen4SparseGQASteelKernel(preserveKVStrides: false)
let qwen4SparseGQASteelStridedKernel = makeQwen4SparseGQASteelKernel(preserveKVStrides: true)
