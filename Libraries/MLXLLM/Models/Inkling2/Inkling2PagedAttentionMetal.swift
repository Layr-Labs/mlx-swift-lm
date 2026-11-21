// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

// Metal source for `Inkling2PagedAttention`. Storage follows the engine's paged
// contract (`pagedattention.metal`): `[pages, kvHeads, 16, headDim]` slabs,
// int32 page tables padded with the poison page, ring tables resolved as
// `table[(pos / 16) % tableLength]`, and a write-fence chain for ordering.
//
// Every path runs ONE per-query arithmetic, so a query's output bits depend
// only on its own query, bias, sink and attended keys:
//   - keys are split into absolute 128-token partitions and 16-key tiles
//     (one page), scored with 8x8 simdgroup matrices whose rows are
//     (query, head) pairs;
//   - inside a partition, tiles update one online softmax in position order;
//   - finished partitions merge in increasing order (`merge_partition_scalars`);
//   - the sink joins last (`sink_factors`).
// The split kernel (decode, verification) stores partition states and merges
// them in a second pass; the sequential kernel (prefill) merges them in
// registers. Both are bit-identical for every query, whatever the batch, the
// number of query rows, or the chunking.

enum Inkling2PagedAttentionMetal {

    static let header = #"""
        #include <metal_stdlib>
        #include <metal_simdgroup>
        using namespace metal;

        namespace ink2 {

        constant constexpr float kLog2e = 1.4426950408889634f;

        typedef simdgroup_matrix<float, 8, 8> frag;

        // Row of (row b, head h, query i) in the caller's query-side tensors:
        // head-major `[B, H, L, ...]` or token-major `[B, L, H, ...]`.
        template <bool TM>
        inline uint io_row(int b, int h, int i, int H, int L) {
            return TM ? (uint)((b * L + i) * H + h) : (uint)((b * H + h) * L + i);
        }

        inline int lowest_key(int qpos, int base, int window) {
            return window > 0 ? max(base, qpos - window + 1) : base;
        }

        // Page storage: the engine's fixed slabs (`[pages, KVH, S, D]` keys and
        // values) or its segments (`[2, pages, KVH, S, D]` each: keys, then values
        // `voff` elements later). A table entry is a page id (slabs) or
        // `binding << 24 | local page` (segments). Both return kv head 0 of the page.
        template <typename T>
        struct SlabPages {
            const device T* k;
            const device T* v;
            size_t page_elements;
            bool holds(int) const { return true; }
            const device T* key_page(int entry) const { return k + (size_t)entry * page_elements; }
            const device T* value_page(int entry) const { return v + (size_t)entry * page_elements; }
        };

        template <typename T, int N>
        struct SegmentPages {
            const device T* segs[N];
            const device int64_t* voff;
            size_t page_elements;
            bool holds(int) const { return true; }
            // The binding's pointer by a chain of selects on constant indices:
            // a dynamic index into the pointer array would go through memory.
            const device T* segment(int k) const {
                const device T* p = segs[0];
                _Pragma("clang loop unroll(full)")
                for (int i = 1; i < N; i++) p = k == i ? segs[i] : p;
                return p;
            }
            const device T* key_page(int entry) const {
                return segment(min(entry >> 24, N - 1)) + (size_t)(entry & 0xFFFFFF) * page_elements;
            }
            const device T* value_page(int entry) const {
                return key_page(entry) + voff[min(entry >> 24, N - 1)];
            }
        };

        // The segments one bucket of a call binds: table entries are the
        // pool's page ids, and binding k holds ids `[first, first + count)`
        // (`ranges[2k]`, `ranges[2k + 1]`). An id outside every binding reads
        // binding 0's poison page and is never written.
        template <typename T, int N>
        struct RangedSegmentPages {
            const device T* segs[N];
            const device int64_t* voff;
            const device int32_t* ranges;
            size_t page_elements;
            int2 locate(int entry) const {
                _Pragma("clang loop unroll(full)")
                for (int k = 0; k < N; k++) {
                    const uint local = uint(entry - ranges[2 * k]);
                    if (local < uint(ranges[2 * k + 1])) return int2(k, int(local));
                }
                return int2(0, -1);
            }
            bool holds(int entry) const { return locate(entry).y >= 0; }
            // As SegmentPages::segment: selects, not a dynamic array index.
            const device T* segment(int k) const {
                const device T* p = segs[0];
                _Pragma("clang loop unroll(full)")
                for (int i = 1; i < N; i++) p = k == i ? segs[i] : p;
                return p;
            }
            const device T* key_page(int entry) const {
                const int2 at = locate(entry);
                return segment(at.x) + (size_t)max(at.y, 0) * page_elements;
            }
            const device T* value_page(int entry) const {
                const int2 at = locate(entry);
                return segment(at.x) + (size_t)max(at.y, 0) * page_elements + voff[at.x];
            }
        };

        // Table entry of the page starting at `pg0`, read only when some of
        // its positions come from the slabs ([base, qstart)): a lazily grown
        // table holds exactly the pages of written positions.
        inline int page_of(
            const device int32_t* table, int pg0, int S, int tlen, int qstart, int base)
        {
            return (pg0 < qstart && pg0 + S > base) ? table[(pg0 / S) % tlen] : 0;
        }

        // Lane layout of an 8x8 simdgroup matrix: each lane holds row `y`,
        // columns `x` and `x + 1` (the layout MLX's steel kernels use).
        inline short2 frag_coord(uint lane) {
            const short qid = lane / 4;
            const short row = (qid & 4) + ((lane / 2) % 4);
            const short col = (qid & 2) * 2 + (lane % 2) * 2;
            return short2(col, row);
        }

        inline void set_frag(thread frag& f, float a, float b) {
            f.thread_elements()[0] = a;
            f.thread_elements()[1] = b;
        }

        // The one partition merge (split merge pass and sequential registers).
        inline void merge_partition_scalars(
            thread float& m, thread float& l, float mp, float lp, thread float& a, thread float& b)
        {
            const float mn = max(m, mp);
            a = exp2(m - mn);
            b = exp2(mp - mn);
            l = fma(lp, b, l * a);
            m = mn;
        }

        inline float merge_value(float o, float op, float a, float b) {
            return fma(op, b, o * a);
        }

        // A partition's state is its two half-partitions' states merged in
        // order: (m, l, o) running = first half, (m1, l1, o1) incoming = second.
        // A half without keys for the row is the identity (m = -FLT_MAX, l = 0).
        template <int NF>
        inline void merge_half(
            thread float& m, thread float& l, thread simdgroup_matrix<float, 8, 8>* o,
            float m1, float l1, thread const float* o1)
        {
            float a;
            float w;
            merge_partition_scalars(m, l, m1, l1, a, w);
            _Pragma("clang loop unroll(full)")
            for (int c = 0; c < NF; c++) {
                o[c].thread_elements()[0] = merge_value(o[c].thread_elements()[0], o1[c * 2], a, w);
                o[c].thread_elements()[1] = merge_value(o[c].thread_elements()[1], o1[c * 2 + 1], a, w);
            }
        }

        // Sink: one extra natural-log logit with a zero value. Returns the
        // factor every accumulated value is multiplied by.
        inline float2 sink_factors(float m, float l, float sink) {
            const float s = sink * kLog2e;
            const float mn = max(m, s);
            const float a = exp2(m - mn);
            const float den = fma(l, a, exp2(s - mn));
            return float2(a, 1.0f / den);
        }

        inline float finish_value(float o, float2 f) {
            return (o * f.x) * f.y;
        }

        // One block of 8 attention rows in one simdgroup. Rows are
        // (query, head) pairs of one kv head, query-major; this lane owns
        // row `fm` and columns `fn`, `fn + 1` of every fragment.
        template <int NF>
        struct Block {
            frag q[NF];     // scale * log2(e) * queries
            frag r[2];      // log2(e) * relative states
            frag o[NF];     // value accumulator of the current partition
            float m;        // this lane's row: running max (log2 domain)
            float l;        // and running sum
            int qpos;       // this lane's row position (-1 when the row is absent)
            int lo;         // lowest key this lane's row attends
            int qpos_first; // block-uniform: first row's position
            int qpos_last;  // block-uniform: last present row's position
            int lo_first;   // block-uniform: lowest key of the first row

            void reset() thread {
                m = -FLT_MAX;
                l = 0.0f;
                _Pragma("clang loop unroll(full)")
                for (int c = 0; c < NF; c++) o[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
        };

        // Fill a block's rows. `pair0` is the block's first (query, head) pair
        // index within the query group starting at query `i0`; `pairs` is the
        // number of pairs in the group.
        template <int NF, int GQA, int DREL, int REL, int BIAS, bool TM, bool LOADQ = true,
                  typename T, typename RT>
        inline void load_block(
            thread Block<NF>& blk, const device T* q, const device RT* rel,
            int b, int kvh, int QH, int L, int i0, int pair0, int pairs,
            int qstart, int base, int window, float scale_log2e, uint lane)
        {
            constexpr int D = NF * 8;
            const short2 fc = frag_coord(lane);
            const int pair = pair0 + fc.y;
            const bool present = pair < pairs && i0 + pair / GQA < L;
            const int i = i0 + pair / GQA;
            const int h = kvh * GQA + pair % GQA;
            const uint row = present ? io_row<TM>(b, h, i, QH, L) : 0u;
            blk.qpos = present ? qstart + i : -1;
            blk.lo = present ? lowest_key(qstart + i, base, window) : 0;
            const int last_pair = min(pair0 + 7, pairs - 1);
            const int i_first = i0 + pair0 / GQA;
            const int i_last = min(L - 1, i0 + last_pair / GQA);
            blk.qpos_first = qstart + i_first;
            blk.qpos_last = qstart + i_last;
            blk.lo_first = lowest_key(qstart + i_first, base, window);
            const device T* qr = q + (size_t)row * D;
            _Pragma("clang loop unroll(full)")
            for (int c = 0; c < (LOADQ ? NF : 0); c++) {
                const int col = c * 8 + fc.x;
                set_frag(
                    blk.q[c], present ? float(qr[col]) * scale_log2e : 0.0f,
                    present ? float(qr[col + 1]) * scale_log2e : 0.0f);
            }
            _Pragma("clang loop unroll(full)")
            for (int c = 0; c < 2; c++) {
                const int t = c * 8 + fc.x;
                float a = 0.0f;
                float bb = 0.0f;
                if (BIAS == 1 && present) {
                    a = t < DREL ? float(rel[(size_t)row * DREL + t]) * kLog2e : 0.0f;
                    bb = t + 1 < DREL ? float(rel[(size_t)row * DREL + t + 1]) * kLog2e : 0.0f;
                }
                set_frag(blk.r[c], a, bb);
            }
        }

        // One page of K/V staged in threadgroup memory in the slab dtype: keys
        // transposed (`kt[dim * KS + key]`), values row-major (`v[key * RS + dim]`).
        template <typename T, int KS, int RS>
        struct StagedRows {
            threadgroup const T* kt;
            threadgroup const T* v;
            // Keys `j`, `j + 1` at dimension `dim`.
            float2 key2(int j, int dim) const {
                return float2(*reinterpret_cast<threadgroup const vec<T, 2>*>(kt + dim * KS + j));
            }
            // Dimensions `dim`, `dim + 1` of key `j`.
            float2 value2(int j, int dim) const {
                return float2(*reinterpret_cast<threadgroup const vec<T, 2>*>(v + j * RS + dim));
            }
        };

        // One page of K/V staged row-major in the slab dtype (`k[key * RS + dim]`).
        // Same values as `StagedRows`; only the read pattern differs.
        template <typename T, int RS>
        struct StagedRowMajor {
            threadgroup const T* k;
            threadgroup const T* v;
            float2 key2(int j, int dim) const {
                return float2(float(k[j * RS + dim]), float(k[(j + 1) * RS + dim]));
            }
            float2 value2(int j, int dim) const {
                return float2(*reinterpret_cast<threadgroup const vec<T, 2>*>(v + j * RS + dim));
            }
        };

        // Stage the page at `pg0` for one kv head: keys transposed into `ks`
        // (`[D][KS]`), values row-major into `vs` (`[S][RS]`), in the slab dtype.
        // Positions below `qstart` come from the slab page, positions in
        // [qstart, qstart + L) from the call's dense tiles; anything outside
        // [base, qstart + L) is staged as zero so a masked key never carries a
        // non-finite value into the products.
        template <typename T, int S, int D, int KS, int RS, int STEP, bool KT = true>
        inline void stage_page(
            int pg0, int qstart, int base, int L,
            const device T* kpage, const device T* vpage,
            const device T* kn, const device T* vn, int dense_stride,
            threadgroup T* ks, threadgroup T* vs, int first)
        {
            // Fixed trip count, fully unrolled: every load of the page is issued
            // before the first threadgroup store.
            constexpr int N = S * D / 4;
            constexpr int ITERS = (N + STEP - 1) / STEP;
            vec<T, 4> kv[ITERS];
            vec<T, 4> vv[ITERS];
            _Pragma("clang loop unroll(full)")
            for (int it = 0; it < ITERS; it++) {
                const int idx = first + it * STEP;
                const int t = idx / (D / 4);
                const int col = (idx % (D / 4)) * 4;
                const int pos = pg0 + t;
                kv[it] = vec<T, 4>(0);
                vv[it] = vec<T, 4>(0);
                if (idx >= N || pos < base || pos >= qstart + L) {
                } else if (pos >= qstart) {
                    const size_t off = (size_t)(pos - qstart) * dense_stride + col;
                    kv[it] = *reinterpret_cast<const device vec<T, 4>*>(kn + off);
                    vv[it] = *reinterpret_cast<const device vec<T, 4>*>(vn + off);
                } else {
                    kv[it] = *reinterpret_cast<const device vec<T, 4>*>(kpage + t * D + col);
                    vv[it] = *reinterpret_cast<const device vec<T, 4>*>(vpage + t * D + col);
                }
            }
            _Pragma("clang loop unroll(full)")
            for (int it = 0; it < ITERS; it++) {
                const int idx = first + it * STEP;
                if (idx >= N) continue;
                const int t = idx / (D / 4);
                const int col = (idx % (D / 4)) * 4;
                if (KT) {
                    _Pragma("clang loop unroll(full)")
                    for (int e = 0; e < 4; e++) ks[(col + e) * KS + t] = kv[it][e];
                } else {
                    *reinterpret_cast<threadgroup vec<T, 4>*>(ks + t * RS + col) = kv[it];
                }
                *reinterpret_cast<threadgroup vec<T, 4>*>(vs + t * RS + col) = vv[it];
            }
        }

        // Scaled query fragments: kept in registers (split kernel) or read from a
        // threadgroup copy in the slab dtype (sequential kernel). Both produce
        // `float(q) * scale * log2(e)` for every element.
        template <int NF>
        struct RegisterQueries {
            thread const frag* q;
            frag get(int c) const { return q[c]; }
        };

        template <typename T>
        struct StagedQueries {
            threadgroup const T* row;  // this lane's row
            short col;
            float scale_log2e;
            frag get(int c) const {
                frag f;
                set_frag(
                    f, float(row[c * 8 + col]) * scale_log2e,
                    float(row[c * 8 + col + 1]) * scale_log2e);
                return f;
            }
        };

        // One 16-key tile (keys kp0 .. kp0 + 15, one page) for one block. Every
        // row keeps its own online softmax; keys outside [lo, qpos] of a row add
        // exactly nothing to it. With BIAS, the relative bias of distances below
        // REL is computed here from the rows' relative states and `proj`; tiles
        // whose distances are all >= REL skip it, so a chunk's bias cost does not
        // grow with its offset. `scratch` is this simdgroup's 8 x 24 floats.
        template <int NF, int REL, int DREL, int BIAS, typename Rows, typename PT, typename Queries>
        inline void tile_update(
            thread Block<NF>& blk, thread const Rows& rows, int kp0,
            const device PT* proj, threadgroup float* scratch, uint lane,
            thread const Queries& queries)
        {
            const short2 fc = frag_coord(lane);
            // Scores: two 8-key halves.
            frag s[2];
            s[0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            s[1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            _Pragma("clang loop unroll(full)")
            for (int c = 0; c < NF; c++) {
                const frag qc = queries.get(c);
                _Pragma("clang loop unroll(full)")
                for (int t = 0; t < 2; t++) {
                    frag kt;
                    const float2 kk = rows.key2(t * 8 + fc.x, c * 8 + fc.y);
                    set_frag(kt, kk.x, kk.y);
                    simdgroup_multiply_accumulate(s[t], qc, kt, s[t]);
                }
            }
            // Relative bias for distances dbase .. dbase + 23 (block rows span at
            // most 8 positions, keys 16).
            const int dbase = blk.qpos_first - kp0 - 15;
            float bv[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
            if (BIAS == 1 && dbase < REL) {
                frag a[3];
                _Pragma("clang loop unroll(full)")
                for (int u = 0; u < 3; u++) a[u] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                _Pragma("clang loop unroll(full)")
                for (int kc = 0; kc < 2; kc++) {
                    // Rows past DREL (zero-padded relative states) read nothing.
                    const bool live = kc * 8 + fc.y < DREL;
                    const device PT* pr = proj + min(kc * 8 + fc.y, DREL - 1) * REL;
                    _Pragma("clang loop unroll(full)")
                    for (int u = 0; u < 3; u++) {
                        const int d = dbase + u * 8 + fc.x;
                        frag pf;
                        set_frag(
                            pf, (live && d >= 0 && d < REL) ? float(pr[d]) : 0.0f,
                            (live && d + 1 >= 0 && d + 1 < REL) ? float(pr[d + 1]) : 0.0f);
                        simdgroup_multiply_accumulate(a[u], blk.r[kc], pf, a[u]);
                    }
                }
                _Pragma("clang loop unroll(full)")
                for (int u = 0; u < 3; u++) {
                    scratch[fc.y * 24 + u * 8 + fc.x] = a[u].thread_elements()[0];
                    scratch[fc.y * 24 + u * 8 + fc.x + 1] = a[u].thread_elements()[1];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                _Pragma("clang loop unroll(full)")
                for (int t = 0; t < 2; t++) {
                    _Pragma("clang loop unroll(full)")
                    for (int e = 0; e < 2; e++) {
                        const int d = blk.qpos - (kp0 + t * 8 + fc.x + e);
                        bv[t][e] = (d >= 0 && d < REL)
                            ? scratch[fc.y * 24 + clamp(d - dbase, 0, 23)] : 0.0f;
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            float sv[2][2];
            _Pragma("clang loop unroll(full)")
            for (int t = 0; t < 2; t++) {
                _Pragma("clang loop unroll(full)")
                for (int e = 0; e < 2; e++) {
                    const int k = kp0 + t * 8 + fc.x + e;
                    const bool valid = blk.qpos >= 0 && k >= blk.lo && k <= blk.qpos;
                    sv[t][e] = valid ? s[t].thread_elements()[e] + bv[t][e] : -INFINITY;
                }
            }
            // Online softmax per row (4 lanes share a row).
            float tmax = max(max(sv[0][0], sv[0][1]), max(sv[1][0], sv[1][1]));
            tmax = max(tmax, simd_shuffle_xor(tmax, ushort(1)));
            tmax = max(tmax, simd_shuffle_xor(tmax, ushort(8)));
            const float mn = max(blk.m, tmax);
            const float corr = exp2(blk.m - mn);
            frag pf[2];
            _Pragma("clang loop unroll(full)")
            for (int t = 0; t < 2; t++) {
                set_frag(pf[t], exp2(sv[t][0] - mn), exp2(sv[t][1] - mn));
            }
            float ps = (pf[0].thread_elements()[0] + pf[0].thread_elements()[1])
                + (pf[1].thread_elements()[0] + pf[1].thread_elements()[1]);
            ps = ps + simd_shuffle_xor(ps, ushort(1));
            ps = ps + simd_shuffle_xor(ps, ushort(8));
            blk.l = fma(blk.l, corr, ps);
            blk.m = mn;
            _Pragma("clang loop unroll(full)")
            for (int c = 0; c < NF; c++) {
                blk.o[c].thread_elements()[0] *= corr;
                blk.o[c].thread_elements()[1] *= corr;
                _Pragma("clang loop unroll(full)")
                for (int t = 0; t < 2; t++) {
                    frag vf;
                    const float2 vv = rows.value2(t * 8 + fc.y, c * 8 + fc.x);
                    set_frag(vf, vv.x, vv.y);
                    simdgroup_multiply_accumulate(blk.o[c], pf[t], vf, blk.o[c]);
                }
            }
        }

        // Fused write of queries [i0, i1] of row b, kv head kvh, into their page
        // slots: position `qstart + i` goes to slot `pos % S` of the page its
        // table entry `(pos / S) % tlen` names (the host reserved exactly
        // these). Single writer per byte: only the threadgroup that owns those
        // queries in the first partition (split) or its query tile (sequential)
        // writes them.
        template <typename T, int S, int D, int TGT, bool TM, typename Pages>
        inline void write_new_tokens(
            int b, int kvh, int KVH, int L, int i0, int i1,
            const device T* knew, const device T* vnew, const device int32_t* table, int tlen,
            int qstart, thread const Pages& pages, uint tid)
        {
            for (int idx = (int)tid; idx < (i1 - i0 + 1) * D; idx += TGT) {
                const int i = i0 + idx / D;
                const int e = idx % D;
                const int pos = qstart + i;
                const int entry = table[(pos / S) % tlen];
                if (!pages.holds(entry)) continue;
                const int slot = pos % S;
                const size_t dst = ((size_t)kvh * S + slot) * D + e;
                const size_t src = (size_t)io_row<TM>(b, kvh, i, KVH, L) * D + e;
                const_cast<device T*>(pages.key_page(entry))[dst] = knew[src];
                const_cast<device T*>(pages.value_page(entry))[dst] = vnew[src];
            }
        }

        } // namespace ink2

        """#

    /// Split kernel (decode and verification). One threadgroup per
    /// (row, kv head, query group of QPT queries, partition); each simdgroup
    /// holds 8 of the group's (query, head) pairs. The split pass publishes each
    /// partition's state for the merge pass.
    static let splitBody = splitCore + "\n" + splitEpilogue

    /// The split kernel with the merge in the same dispatch: each threadgroup
    /// publishes its partition's state to the plan's workspace and counts in;
    /// the last threadgroup of a (row, kv head, query group) to arrive merges
    /// the group's partitions exactly as the merge pass does. No threadgroup
    /// waits for another. Up to `FUSED_PARTS` partitions per query.
    static let fusedBody = splitCore + "\n" + fusedEpilogue

    /// The split kernel up to its partition state.
    static let splitCore = #"""
            constexpr int NF = D / 8;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint tid = thread_position_in_threadgroup.x;
            const int L = TM ? q_shape[1] : q_shape[2];
            const int QH = TM ? q_shape[2] : q_shape[1];
            const int KVH = TM ? knew_shape[2] : knew_shape[1];
            PAGES_DECL
            const int NQG = (L + QPT - 1) / QPT;
            const int kvh = threadgroup_position_in_grid.x / NQG;
            const int qg = threadgroup_position_in_grid.x % NQG;
            ITEM_DECL
            const int maxp = tables_shape[1];
            if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0
                && thread_position_in_grid.z == 0) {
                fence[0] = wfence[0] + 1;
            }
            const device int32_t* info = rowinfo + b * 8;
            const int qstart = info[0];
            const int base = info[1];
            const int tlen = info[2];
            const int window = info[3];
            const int i0 = qg * QPT;
            const int i1 = min(L, i0 + QPT) - 1;
            if (HAS_WRITE && z == 0) {
                ink2::write_new_tokens<T, S, D, 32 * NB * PSUB, TM>(
                    b, kvh, KVH, L, i0, i1, knew, vnew, tables + (size_t)b * maxp, tlen, qstart,
                    pages, tid);
            }
            // PSUB == 2: simdgroups NB .. 2 NB - 1 take the second half of the
            // partition for blocks 0 .. NB - 1; PSUB == 1: one simdgroup per block
            // takes both halves in order. Same bits either way.
            const int bi = (int)sg % NB;
            const int half_index = PSUB == 2 ? (int)sg / NB : 0;
            const int pairs = (i1 - i0 + 1) * GQA;
            const int pair0 = bi * 8;
            // A fused launch's threadgroups all count in, so none returns early:
            // simdgroups without rows or keys in the partition stay inactive.
            bool active = pair0 < pairs;
            const int p = ink2::lowest_key(qstart, base, window) / PTOK + z;
            ink2::Block<NF> blk;
            int klo = 1;
            int khi = 0;
            if (active) {
                ink2::load_block<NF, GQA, DREL, REL, BIAS, TM>(
                    blk, q, rel, b, kvh, QH, L, i0, pair0, pairs, qstart, base, window,
                    as_type<float>(info[4]), lane);
                klo = max(p * PTOK, blk.lo_first);
                khi = min((p + 1) * PTOK - 1, blk.qpos_last);
            }
            active = active && klo <= khi;
            if (active) blk.reset();
            constexpr int RS = D + 4;
            constexpr int HALF = PTOK / 2;
            threadgroup float scratch_all[NB * PSUB * 192];
            threadgroup float* scratch = scratch_all + sg * 192;
            threadgroup T stage_all[NB * PSUB * 2 * S * RS];
            threadgroup T* ks = stage_all + sg * 2 * S * RS;
            threadgroup T* vs = ks + S * RS;
            const device int32_t* table = tables + (size_t)b * maxp;
            const device T* kn = knew + (size_t)ink2::io_row<TM>(b, kvh, 0, KVH, L) * D;
            const device T* vn = vnew + (size_t)ink2::io_row<TM>(b, kvh, 0, KVH, L) * D;
            const int dense_stride = TM ? KVH * D : D;
            // First half's state while the same simdgroup takes the second (PSUB
            // 1), held in threadgroup memory (each lane its own slots) rather than
            // registers through the second half's page loop.
            constexpr int HOLD = (NF * 2 + 2) * 32;
            threadgroup float hold_all[PSUB == 1 ? NB * HOLD : 1];
            threadgroup float* hold = hold_all + (PSUB == 1 ? (int)sg * HOLD : 0);
            for (int h = half_index; h < (PSUB == 2 ? half_index + 1 : 2); h++) {
                if (PSUB == 1 && h == 1 && active) {
                    _Pragma("clang loop unroll(full)")
                    for (int c = 0; c < NF; c++) {
                        hold[(c * 2) * 32 + lane] = blk.o[c].thread_elements()[0];
                        hold[(c * 2 + 1) * 32 + lane] = blk.o[c].thread_elements()[1];
                    }
                    hold[(NF * 2) * 32 + lane] = blk.m;
                    hold[(NF * 2 + 1) * 32 + lane] = blk.l;
                    blk.reset();
                }
                const int hlo = max(klo, p * PTOK + h * HALF);
                const int hhi = min(khi, p * PTOK + (h + 1) * HALF - 1);
                for (int kp0 = (hlo / S) * S; active && kp0 <= hhi; kp0 += S) {
                    const int page = ink2::page_of(table, kp0, S, tlen, qstart, base);
                    const size_t head = (size_t)kvh * S * D;
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    ink2::stage_page<T, S, D, 0, RS, 32, false>(
                        kp0, qstart, base, L, pages.key_page(page) + head,
                        pages.value_page(page) + head, kn, vn, dense_stride, ks, vs, (int)lane);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    const ink2::StagedRowMajor<T, RS> rows{ks, vs};
                    const ink2::RegisterQueries<NF> queries{blk.q};
                    ink2::tile_update<NF, REL, DREL, BIAS>(
                        blk, rows, kp0, proj, scratch, lane, queries);
                }
            }
            if (PSUB == 1 && active) {
                // Second half (in blk) into the first: first half runs.
                const float m1 = blk.m;
                const float l1 = blk.l;
                float o1[NF * 2];
                _Pragma("clang loop unroll(full)")
                for (int c = 0; c < NF; c++) {
                    o1[c * 2] = blk.o[c].thread_elements()[0];
                    o1[c * 2 + 1] = blk.o[c].thread_elements()[1];
                    blk.o[c].thread_elements()[0] = hold[(c * 2) * 32 + lane];
                    blk.o[c].thread_elements()[1] = hold[(c * 2 + 1) * 32 + lane];
                }
                blk.m = hold[(NF * 2) * 32 + lane];
                blk.l = hold[(NF * 2 + 1) * 32 + lane];
                ink2::merge_half<NF>(blk.m, blk.l, blk.o, m1, l1, o1);
            }
            if (PSUB == 2) {
                // The second-half simdgroup hands its state to the first through
                // its own (finished) staging area.
                threadgroup float* xfer =
                    reinterpret_cast<threadgroup float*>(stage_all + (bi + NB) * 2 * S * RS);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (half_index == 1 && active) {
                    _Pragma("clang loop unroll(full)")
                    for (int c = 0; c < NF; c++) {
                        xfer[(c * 2) * 32 + lane] = blk.o[c].thread_elements()[0];
                        xfer[(c * 2 + 1) * 32 + lane] = blk.o[c].thread_elements()[1];
                    }
                    xfer[(NF * 2) * 32 + lane] = blk.m;
                    xfer[(NF * 2 + 1) * 32 + lane] = blk.l;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (half_index == 0 && active) {
                    float o1[NF * 2];
                    _Pragma("clang loop unroll(full)")
                    for (int c = 0; c < NF * 2; c++) o1[c] = xfer[c * 32 + lane];
                    ink2::merge_half<NF>(
                        blk.m, blk.l, blk.o, xfer[(NF * 2) * 32 + lane],
                        xfer[(NF * 2 + 1) * 32 + lane], o1);
                }
                // Only the first-half simdgroup publishes.
                active = active && half_index == 0;
            }
            const short2 fc = ink2::frag_coord(lane);
            // This lane's row meets the partition: it publishes the state.
            const bool publish =
                active && blk.qpos >= 0 && blk.lo <= (p + 1) * PTOK - 1 && blk.qpos >= p * PTOK;
        """#

    static let splitEpilogue = #"""
            if (publish) {
                const int pair = pair0 + fc.y;
                const int i = i0 + pair / GQA;
                const int h = kvh * GQA + pair % GQA;
                // Every slot's D values first, then every slot's m and l.
                const size_t slot = ((size_t)(b * QH + h) * L + i) * maxparts + z;
                const size_t slots = (size_t)nrows * QH * L * maxparts;
                device float* dst = partials + slot * D;
                _Pragma("clang loop unroll(full)")
                for (int c = 0; c < NF; c++) {
                    dst[c * 8 + fc.x] = blk.o[c].thread_elements()[0];
                    dst[c * 8 + fc.x + 1] = blk.o[c].thread_elements()[1];
                }
                if (fc.x == 0) {
                    partials[slots * D + slot * 2] = blk.m;
                    partials[slots * D + slot * 2 + 1] = blk.l;
                }
            }
        """#

    static let fusedEpilogue = #"""
            // The group's workspace: per (pair, partition) D values, m and l.
            // Atomic stores and loads keep every threadgroup's view coherent.
            const int gid = b * (int)threadgroups_per_grid.x + (int)threadgroup_position_in_grid.x;
            device atomic_float* slots = reinterpret_cast<device atomic_float*>(work)
                + (size_t)gid * (QPT * GQA) * maxparts * (D + 2);
            if (publish) {
                device atomic_float* dst = slots + ((size_t)(pair0 + fc.y) * maxparts + z) * (D + 2);
                _Pragma("clang loop unroll(full)")
                for (int c = 0; c < NF; c++) {
                    atomic_store_explicit(
                        dst + c * 8 + fc.x, blk.o[c].thread_elements()[0], memory_order_relaxed);
                    atomic_store_explicit(
                        dst + c * 8 + fc.x + 1, blk.o[c].thread_elements()[1], memory_order_relaxed);
                }
                if (fc.x == 0) {
                    atomic_store_explicit(dst + D, blk.m, memory_order_relaxed);
                    atomic_store_explicit(dst + D + 1, blk.l, memory_order_relaxed);
                }
            }
            // Count in; only the group's last threadgroup to arrive continues.
            threadgroup uint arrival[1];
            device atomic_uint* counter = reinterpret_cast<device atomic_uint*>(counters) + gid;
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            if (tid == 0) {
                atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
                arrival[0] = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
                atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            if (arrival[0] != uint(maxparts - 1)) return;
            // Every count is in: clear the counter for the plan's next launch.
            if (tid == 0) atomic_store_explicit(counter, 0u, memory_order_relaxed);
            // The merge pass, per (query, head) pair of the group: partitions in
            // increasing order, then the sink.
            constexpr int EPT = D / 32;
            const int pfirst = ink2::lowest_key(qstart, base, window) / PTOK;
            for (int pair = (int)sg; pair < pairs; pair += NB * PSUB) {
                const int i = i0 + pair / GQA;
                const int h = kvh * GQA + pair % GQA;
                const int qpos = qstart + i;
                const int lo = ink2::lowest_key(qpos, base, window);
                const int pbeg = lo / PTOK - pfirst;
                const int np = qpos / PTOK - lo / PTOK + 1;
                const device atomic_float* src = slots + ((size_t)pair * maxparts + pbeg) * (D + 2);
                // Every load first: lane k holds partition k's m and l.
                float pm = -FLT_MAX;
                float pl = 0.0f;
                if ((int)lane < np) {
                    pm = atomic_load_explicit(src + (size_t)lane * (D + 2) + D, memory_order_relaxed);
                    pl = atomic_load_explicit(
                        src + (size_t)lane * (D + 2) + D + 1, memory_order_relaxed);
                }
                float pv[FUSED_PARTS][EPT];
                _Pragma("clang loop unroll(full)")
                for (int k = 0; k < FUSED_PARTS; k++) {
                    _Pragma("clang loop unroll(full)")
                    for (int e = 0; e < EPT; e++) {
                        pv[k][e] = k < np
                            ? atomic_load_explicit(
                                src + (size_t)k * (D + 2) + lane * EPT + e, memory_order_relaxed)
                            : 0.0f;
                    }
                }
                float m = -FLT_MAX;
                float l = 0.0f;
                float o[EPT];
                _Pragma("clang loop unroll(full)")
                for (int e = 0; e < EPT; e++) o[e] = 0.0f;
                _Pragma("clang loop unroll(full)")
                for (int k = 0; k < FUSED_PARTS; k++) {
                    if (k < np) {
                        float a;
                        float w;
                        ink2::merge_partition_scalars(
                            m, l, simd_shuffle(pm, ushort(k)), simd_shuffle(pl, ushort(k)), a, w);
                        _Pragma("clang loop unroll(full)")
                        for (int e = 0; e < EPT; e++) o[e] = ink2::merge_value(o[e], pv[k][e], a, w);
                    }
                }
                const float2 f =
                    ink2::sink_factors(m, l, float(sinks[ink2::io_row<TM>(b, h, i, QH, L)]));
                device T* orow = out + ((size_t)(b * L + i) * QH + h) * D + lane * EPT;
                _Pragma("clang loop unroll(full)")
                for (int e = 0; e < EPT; e++) orow[e] = T(ink2::finish_value(o[e], f));
            }
        """#

    /// Merge pass of the split kernel: one simdgroup per (query, head, row).
    static let mergeBody = #"""
            constexpr int EPT = D / 32;
            const uint lane = thread_index_in_simdgroup;
            const int QH = partials_shape[1];
            const int L = partials_shape[2];
            const int maxparts = partials_shape[3];
            const int h = threadgroup_position_in_grid.x;
            const int i = threadgroup_position_in_grid.y;
            const int b = threadgroup_position_in_grid.z;
            const device int32_t* info = rowinfo + b * 8;
            const int qstart = info[0];
            const int base = info[1];
            const int window = info[3];
            const int qpos = qstart + i;
            const int lo = ink2::lowest_key(qpos, base, window);
            const int pfirst = ink2::lowest_key(qstart, base, window) / PTOK;
            float m = -FLT_MAX;
            float l = 0.0f;
            float o[EPT];
            _Pragma("clang loop unroll(full)")
            for (int e = 0; e < EPT; e++) o[e] = 0.0f;
            // Partitions in increasing order. Loads run ahead of the merge chain:
            // lane k holds the m and l of the chunk's partition k, and each group
            // of 8 partitions' values is in registers before its merges.
            const size_t slots = (size_t)partials_shape[0] * QH * L * maxparts;
            const size_t first = ((size_t)(b * QH + h) * L + i) * maxparts + (lo / PTOK - pfirst);
            const device float* src = partials + first * D;
            const device float* meta = partials + slots * D + first * 2;
            const int np = qpos / PTOK - lo / PTOK + 1;
            for (int c0 = 0; c0 < np; c0 += 32) {
                const int n = min(32, np - c0);
                float pm = -FLT_MAX;
                float pl = 0.0f;
                if ((int)lane < n) {
                    pm = meta[(size_t)(c0 + lane) * 2];
                    pl = meta[(size_t)(c0 + lane) * 2 + 1];
                }
                for (int k0 = 0; k0 < n; k0 += 8) {
                    float pv[8][EPT];
                    _Pragma("clang loop unroll(full)")
                    for (int k = 0; k < 8; k++) {
                        _Pragma("clang loop unroll(full)")
                        for (int e = 0; e < EPT; e++) {
                            pv[k][e] = k0 + k < n ? src[(size_t)(c0 + k0 + k) * D + lane * EPT + e] : 0.0f;
                        }
                    }
                    _Pragma("clang loop unroll(full)")
                    for (int k = 0; k < 8; k++) {
                        if (k0 + k < n) {
                            float a;
                            float w;
                            ink2::merge_partition_scalars(
                                m, l, simd_shuffle(pm, ushort(k0 + k)), simd_shuffle(pl, ushort(k0 + k)),
                                a, w);
                            _Pragma("clang loop unroll(full)")
                            for (int e = 0; e < EPT; e++) o[e] = ink2::merge_value(o[e], pv[k][e], a, w);
                        }
                    }
                }
            }
            const float2 f = ink2::sink_factors(m, l, float(sinks[ink2::io_row<TM>(b, h, i, QH, L)]));
            device T* orow = out + ((size_t)(b * L + i) * QH + h) * D + lane * EPT;
            _Pragma("clang loop unroll(full)")
            for (int e = 0; e < EPT; e++) orow[e] = T(ink2::finish_value(o[e], f));
        """#

    /// Neural-accelerator helpers of the prefill kernel (MetalPerformancePrimitives
    /// `matmul2d` 16 x 32 x 16 on cooperative tensors, the fragment layout MLX's
    /// NAX kernels use).
    static let prefillHeader = #"""
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        namespace na {
        using namespace metal;
        constant constexpr float kLog2e = 1.4426950408889634f;

        // 16x16 fragment layout of matmul2d's cooperative tensors (MLX NAXFrag):
        // lane holds rows fm, fm + 8 and columns fn .. fn + 3; element i = row (i >> 2) * 8, col i % 4.
        inline short2 coord(uint lane) {
            const short qid = lane >> 2;
            const short fm = (qid & 4) | ((lane >> 1) & 3);
            const short fn = ((qid & 2) | (lane & 1)) * 4;
            return short2(fn, fm);
        }

        // C(16x32) += A(16x16) * B(16x32), B given as two 16x16 fragments (TB: B is supplied transposed).
        template <typename CT, typename AT, typename BT, bool TB>
        inline void mma(thread vec<CT, 8>& c0, thread vec<CT, 8>& c1, thread const vec<AT, 8>& a,
                        thread const vec<BT, 8>& b0, thread const vec<BT, 8>& b1) {
            constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
                16, 32, 16, false, TB, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
            mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
            auto ta = op.template get_left_input_cooperative_tensor<AT, BT, CT>();
            auto tb = op.template get_right_input_cooperative_tensor<AT, BT, CT>();
            auto tc = op.template get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(ta)>, metal::remove_addrspace_t<decltype(tb)>, CT>();
            _Pragma("clang loop unroll(full)")
            for (short i = 0; i < 8; i++) { ta[i] = a[i]; tb[i] = b0[i]; tb[8 + i] = b1[i]; tc[i] = c0[i]; tc[8 + i] = c1[i]; }
            op.run(ta, tb, tc);
            _Pragma("clang loop unroll(full)")
            for (short i = 0; i < 8; i++) { c0[i] = tc[i]; c1[i] = tc[8 + i]; }
        }
        }  // namespace na

        """#

    /// Prefill kernel on the M5 neural accelerators. One threadgroup per (row,
    /// kv head, tile of QPT queries); each simdgroup holds 16 (query, head) rows.
    /// Its own arithmetic: every row keeps one online softmax over absolute
    /// 32-key tiles (Q K^T and P V on the accelerators, FP32 accumulation, the
    /// relative bias by R x proj on the accelerators), so a query's bits depend
    /// only on its own inputs and position: chunk- and batch-invariant, not
    /// bit-equal to the decode/verify kernels.
    static let prefillBody = #"""
            constexpr int NF = D / 16;          // dim fragments
            constexpr int BK = 32;              // keys per tile
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const int L = TM ? q_shape[1] : q_shape[2];
            const int QH = TM ? q_shape[2] : q_shape[1];
            const int KVH = TM ? knew_shape[2] : knew_shape[1];
            PAGES_DECL
            const int NQT = (L + QPT - 1) / QPT;
            const int kvh = threadgroup_position_in_grid.x / NQT;
            const int qt = threadgroup_position_in_grid.x % NQT;
            const int b = threadgroup_position_in_grid.y;
            const int maxp = tables_shape[1];
            const device int32_t* info = rowinfo + b * 8;
            const int qstart = info[0];
            const int base = info[1];
            const int tlen = info[2];
            const int window = info[3];
            const float scale_log2e = as_type<float>(info[4]);
            const int i0 = qt * QPT;
            const int i1 = min(L, i0 + QPT) - 1;
            const int pairs = (i1 - i0 + 1) * GQA;
            if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0
                && thread_position_in_grid.z == 0) {
                fence[0] = wfence[0] + 1;
            }
            const int pair0 = (int)sg * 16;
            if (pair0 >= pairs) return;
            const short2 fc = na::coord(lane);
            // This lane's two rows (fm, fm + 8).
            int qpos[2], lo[2], hrow[2], irow[2];
            bool present[2];
            _Pragma("clang loop unroll(full)")
            for (int r = 0; r < 2; r++) {
                const int pair = pair0 + fc.y + r * 8;
                present[r] = pair < pairs;
                irow[r] = i0 + pair / GQA;
                hrow[r] = kvh * GQA + pair % GQA;
                qpos[r] = present[r] ? qstart + irow[r] : -1;
                lo[r] = present[r] ? ink2::lowest_key(qpos[r], base, window) : 0x7fffffff;
            }
            // Block range: the union of the rows' key ranges.
            const int last_pair = min(pair0 + 15, pairs - 1);
            const int qfirst = qstart + i0 + pair0 / GQA;
            const int qlast = qstart + i0 + last_pair / GQA;
            const int kmin = ink2::lowest_key(qfirst, base, window);
            // Keys every present row attends: tiles inside need no masking.
            const int kall_lo = ink2::lowest_key(qlast, base, window);
            const int kall_hi = qfirst;
            // Queries (bf16) and relative states (bf16), one fragment per 16 dims.
            const device T* qrow[2];
            _Pragma("clang loop unroll(full)")
            for (int r = 0; r < 2; r++) {
                qrow[r] = present[r] ? q + (size_t)ink2::io_row<TM>(b, hrow[r], irow[r], QH, L) * D : nullptr;
            }
            vec<T, 8> rf;
            _Pragma("clang loop unroll(full)")
            for (int r = 0; r < 2; r++) {
                const uint row = present[r] ? ink2::io_row<TM>(b, hrow[r], irow[r], QH, L) : 0u;
                _Pragma("clang loop unroll(full)")
                for (int e = 0; e < 4; e++) {
                    const int t = fc.x + e;
                    rf[r * 4 + e] = (BIAS == 1 && present[r] && t < DREL) ? rel[(size_t)row * DREL + t] : T(0);
                }
            }
            vec<float, 8> o[NF];
            _Pragma("clang loop unroll(full)")
            for (int c = 0; c < NF; c++) o[c] = 0.0f;
            float m[2] = {-FLT_MAX, -FLT_MAX};
            float l[2] = {0.0f, 0.0f};
            threadgroup float scratch_all[NS * 16 * 64];
            threadgroup float* scratch = scratch_all + sg * 16 * 64;
            const device int32_t* table = tables + (size_t)b * maxp;
            const size_t head = (size_t)kvh * S * D;
            const int dense_stride = TM ? KVH * D : D;
            const device T* kn = knew + (size_t)ink2::io_row<TM>(b, kvh, 0, KVH, L) * D;
            const device T* vn = vnew + (size_t)ink2::io_row<TM>(b, kvh, 0, KVH, L) * D;
            for (int k0 = (kmin / BK) * BK; k0 <= qlast; k0 += BK) {
                const bool interior = k0 >= kall_lo && k0 >= base && k0 + BK - 1 <= kall_hi;
                // Row pointers of this lane's four key rows (k0 + fm, + 8, + 16, + 24); null reads zero.
                const device T* kp[4];
                const device T* vp[4];
                // Keys below `kmin` are masked for every row of the block; they
                // read as zero (a ring slot below the window may hold a page
                // whose newer positions were never written, and a masked key's
                // value still meets P x V), as the decode kernels stage them.
                if (k0 >= kmin && k0 + BK - 1 < qstart) {
                    // Two whole pages: one table read each for the simdgroup.
                    const int ea = table[(k0 / S) % tlen];
                    const int eb = table[(k0 / S + 1) % tlen];
                    kp[0] = pages.key_page(ea) + head + fc.y * D;
                    vp[0] = pages.value_page(ea) + head + fc.y * D;
                    kp[2] = pages.key_page(eb) + head + fc.y * D;
                    vp[2] = pages.value_page(eb) + head + fc.y * D;
                    kp[1] = kp[0] + 8 * D;
                    vp[1] = vp[0] + 8 * D;
                    kp[3] = kp[2] + 8 * D;
                    vp[3] = vp[2] + 8 * D;
                } else {
                    _Pragma("clang loop unroll(full)")
                    for (int j = 0; j < 4; j++) {
                        const int pos = k0 + fc.y + j * 8;
                        kp[j] = nullptr;
                        vp[j] = nullptr;
                        if (pos >= kmin && pos < qstart + L) {
                            if (pos >= qstart) {
                                kp[j] = kn + (size_t)(pos - qstart) * dense_stride;
                                vp[j] = vn + (size_t)(pos - qstart) * dense_stride;
                            } else {
                                const int entry = table[(pos / S) % tlen];
                                kp[j] = pages.key_page(entry) + head + (pos % S) * D;
                                vp[j] = pages.value_page(entry) + head + (pos % S) * D;
                            }
                        }
                    }
                }
                // S = Q K^T over 32 keys.
                vec<float, 8> s0 = 0.0f, s1 = 0.0f;
                _Pragma("clang loop unroll(full)")
                for (int c = 0; c < NF; c++) {
                    vec<T, 8> ka, kb;
                    _Pragma("clang loop unroll(full)")
                    for (int r = 0; r < 2; r++) {
                        const vec<T, 4> x = kp[r] ? *reinterpret_cast<const device vec<T, 4>*>(kp[r] + c * 16 + fc.x) : vec<T, 4>(0);
                        const vec<T, 4> y = kp[2 + r] ? *reinterpret_cast<const device vec<T, 4>*>(kp[2 + r] + c * 16 + fc.x) : vec<T, 4>(0);
                        _Pragma("clang loop unroll(full)")
                        for (int e = 0; e < 4; e++) { ka[r * 4 + e] = x[e]; kb[r * 4 + e] = y[e]; }
                    }
                    vec<T, 8> qc;
                    _Pragma("clang loop unroll(full)")
                    for (int r = 0; r < 2; r++) {
                        const vec<T, 4> x = qrow[r] ? *reinterpret_cast<const device vec<T, 4>*>(qrow[r] + c * 16 + fc.x) : vec<T, 4>(0);
                        _Pragma("clang loop unroll(full)")
                        for (int e = 0; e < 4; e++) qc[r * 4 + e] = x[e];
                    }
                    na::mma<float, T, T, true>(s0, s1, qc, ka, kb);
                }
                // Scores in log2 units, in place.
                s0 *= scale_log2e;
                s1 *= scale_log2e;
                // Relative bias for distances dbase .. dbase + 63 (rows span <= 6 positions, keys 32).
                const int dbase = qfirst - (k0 + BK - 1);
                if (BIAS == 1 && dbase < REL) {
                    vec<float, 8> a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                    _Pragma("clang loop unroll(full)")
                    for (int hh = 0; hh < 2; hh++) {
                        vec<T, 8> p0, p1;
                        _Pragma("clang loop unroll(full)")
                        for (int r = 0; r < 2; r++) {
                            const int t = fc.y + r * 8;
                            _Pragma("clang loop unroll(full)")
                            for (int e = 0; e < 4; e++) {
                                const int d0 = dbase + hh * 32 + fc.x + e;
                                const int d1 = d0 + 16;
                                p0[r * 4 + e] = (t < DREL && d0 >= 0 && d0 < REL) ? proj[t * REL + d0] : T(0);
                                p1[r * 4 + e] = (t < DREL && d1 >= 0 && d1 < REL) ? proj[t * REL + d1] : T(0);
                            }
                        }
                        if (hh == 0) na::mma<float, T, T, false>(a0, a1, rf, p0, p1);
                        else na::mma<float, T, T, false>(a2, a3, rf, p0, p1);
                    }
                    _Pragma("clang loop unroll(full)")
                    for (int r = 0; r < 2; r++) {
                        _Pragma("clang loop unroll(full)")
                        for (int e = 0; e < 4; e++) {
                            const int row = fc.y + r * 8;
                            scratch[row * 64 + fc.x + e] = a0[r * 4 + e];
                            scratch[row * 64 + 16 + fc.x + e] = a1[r * 4 + e];
                            scratch[row * 64 + 32 + fc.x + e] = a2[r * 4 + e];
                            scratch[row * 64 + 48 + fc.x + e] = a3[r * 4 + e];
                        }
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    _Pragma("clang loop unroll(full)")
                    for (int r = 0; r < 2; r++) {
                        _Pragma("clang loop unroll(full)")
                        for (int e = 0; e < 4; e++) {
                            const int dA = qpos[r] - (k0 + fc.x + e);
                            const int dB = dA - 16;
                            if (dA >= 0 && dA < REL) s0[r * 4 + e] += scratch[(fc.y + r * 8) * 64 + clamp(dA - dbase, 0, 63)] * na::kLog2e;
                            if (dB >= 0 && dB < REL) s1[r * 4 + e] += scratch[(fc.y + r * 8) * 64 + clamp(dB - dbase, 0, 63)] * na::kLog2e;
                        }
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (!interior) {
                    _Pragma("clang loop unroll(full)")
                    for (int r = 0; r < 2; r++) {
                        _Pragma("clang loop unroll(full)")
                        for (int e = 0; e < 4; e++) {
                            const int kA = k0 + fc.x + e;
                            const int kB = kA + 16;
                            const bool okA = present[r] && kA >= lo[r] && kA <= qpos[r];
                            const bool okB = present[r] && kB >= lo[r] && kB <= qpos[r];
                            s0[r * 4 + e] = okA ? s0[r * 4 + e] : -INFINITY;
                            s1[r * 4 + e] = okB ? s1[r * 4 + e] : -INFINITY;
                        }
                    }
                }
                float corr[2];
                vec<float, 8> p0, p1;
                _Pragma("clang loop unroll(full)")
                for (int r = 0; r < 2; r++) {
                    float tmax = max(max(max(s0[r * 4], s0[r * 4 + 1]), max(s0[r * 4 + 2], s0[r * 4 + 3])),
                                     max(max(s1[r * 4], s1[r * 4 + 1]), max(s1[r * 4 + 2], s1[r * 4 + 3])));
                    tmax = max(tmax, simd_shuffle_xor(tmax, ushort(1)));
                    tmax = max(tmax, simd_shuffle_xor(tmax, ushort(8)));
                    const float mn = max(m[r], tmax);
                    corr[r] = fast::exp2(m[r] - mn);
                    float ps = 0.0f;
                    _Pragma("clang loop unroll(full)")
                    for (int e = 0; e < 4; e++) {
                        p0[r * 4 + e] = fast::exp2(s0[r * 4 + e] - mn);
                        p1[r * 4 + e] = fast::exp2(s1[r * 4 + e] - mn);
                    }
                    _Pragma("clang loop unroll(full)")
                    for (int e = 0; e < 4; e++) ps += p0[r * 4 + e];
                    _Pragma("clang loop unroll(full)")
                    for (int e = 0; e < 4; e++) ps += p1[r * 4 + e];
                    ps += simd_shuffle_xor(ps, ushort(1));
                    ps += simd_shuffle_xor(ps, ushort(8));
                    l[r] = fma(l[r], corr[r], ps);
                    m[r] = mn;
                }
                _Pragma("clang loop unroll(full)")
                for (int c = 0; c < NF; c++) {
                    _Pragma("clang loop unroll(full)")
                    for (int r = 0; r < 2; r++) {
                        _Pragma("clang loop unroll(full)")
                        for (int e = 0; e < 4; e++) o[c][r * 4 + e] *= corr[r];
                    }
                }
                // O += P V: keys 0..15 (p0) and 16..31 (p1), dims in pairs of fragments.
                _Pragma("clang loop unroll(full)")
                for (int c = 0; c < NF; c += 2) {
                    _Pragma("clang loop unroll(full)")
                    for (int kh = 0; kh < 2; kh++) {
                        vec<T, 8> va, vb;
                        _Pragma("clang loop unroll(full)")
                        for (int r = 0; r < 2; r++) {
                            const device T* src = vp[kh * 2 + r];
                            const vec<T, 4> x = src ? *reinterpret_cast<const device vec<T, 4>*>(src + c * 16 + fc.x) : vec<T, 4>(0);
                            const vec<T, 4> y = src ? *reinterpret_cast<const device vec<T, 4>*>(src + (c + 1) * 16 + fc.x) : vec<T, 4>(0);
                            _Pragma("clang loop unroll(full)")
                            for (int e = 0; e < 4; e++) { va[r * 4 + e] = x[e]; vb[r * 4 + e] = y[e]; }
                        }
                        na::mma<float, float, T, false>(o[c], o[c + 1], kh == 0 ? p0 : p1, va, vb);
                    }
                }
            }
            _Pragma("clang loop unroll(full)")
            for (int r = 0; r < 2; r++) {
                if (!present[r]) continue;
                const float2 f = ink2::sink_factors(m[r], l[r], float(sinks[ink2::io_row<TM>(b, hrow[r], irow[r], QH, L)]));
                device T* orow = out + ((size_t)(b * L + irow[r]) * QH + hrow[r]) * D;
                _Pragma("clang loop unroll(full)")
                for (int c = 0; c < NF; c++) {
                    _Pragma("clang loop unroll(full)")
                    for (int e = 0; e < 4; e++) orow[c * 16 + fc.x + e] = T(ink2::finish_value(o[c][r * 4 + e], f));
                }
            }
        """#

    /// Sequential kernel (prefill chunks). One threadgroup per (row, kv head,
    /// query tile of QPT queries); pages are staged once for all its blocks,
    /// partitions run in order and merge in registers.
    static let sequentialBody = #"""
            constexpr int NF = D / 8;
            constexpr int RS = D + 4;
            constexpr int KS = S + 2;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint tid = thread_position_in_threadgroup.x;
            const int L = TM ? q_shape[1] : q_shape[2];
            const int QH = TM ? q_shape[2] : q_shape[1];
            const int KVH = TM ? knew_shape[2] : knew_shape[1];
            PAGES_DECL
            const int NQT = (L + QPT - 1) / QPT;
            const int kvh = threadgroup_position_in_grid.x / NQT;
            const int qt = threadgroup_position_in_grid.x % NQT;
            const int b = threadgroup_position_in_grid.y;
            const int maxp = tables_shape[1];
            if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0
                && thread_position_in_grid.z == 0) {
                fence[0] = wfence[0] + 1;
            }
            const device int32_t* info = rowinfo + b * 8;
            const int qstart = info[0];
            const int base = info[1];
            const int tlen = info[2];
            const int window = info[3];
            const int i0 = qt * QPT;
            const int i1 = min(L, i0 + QPT) - 1;
            if (HAS_WRITE) {
                ink2::write_new_tokens<T, S, D, 32 * NB, TM>(
                    b, kvh, KVH, L, i0, i1, knew, vnew, tables + (size_t)b * maxp, tlen, qstart,
                    pages, tid);
            }
            const int pairs = (i1 - i0 + 1) * GQA;
            const int pair0 = (int)sg * 8;
            const bool has_block = pair0 < pairs;
            ink2::Block<NF> blk;
            ink2::load_block<NF, GQA, DREL, REL, BIAS, TM, false>(
                blk, q, rel, b, kvh, QH, L, i0, min(pair0, pairs - 1), pairs, qstart, base, window,
                as_type<float>(info[4]), lane);
            if (!has_block) blk.qpos = -1;
            const short2 fc = ink2::frag_coord(lane);
            // The tile's queries, one D-row per (query, head) pair, zero past `pairs`.
            threadgroup T q_tg[NB * 8 * D];
            for (int idx = (int)tid; idx < NB * 8 * D; idx += 32 * NB) {
                const int pair = idx / D;
                const int i = i0 + pair / GQA;
                const int h = kvh * GQA + pair % GQA;
                q_tg[idx] =
                    pair < pairs ? q[(size_t)ink2::io_row<TM>(b, h, i, QH, L) * D + idx % D] : T(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const ink2::StagedQueries<T> queries{
                q_tg + (sg * 8 + fc.y) * D, fc.x, as_type<float>(info[4])};
            // Merged partitions: this lane's row, columns c * 8 + fc.x and + 1.
            float ro[NF * 2];
            float rm = -FLT_MAX;
            float rl = 0.0f;
            _Pragma("clang loop unroll(full)")
            for (int c = 0; c < NF * 2; c++) ro[c] = 0.0f;
            threadgroup T ks[D * KS];
            threadgroup T vs[S * RS];
            threadgroup float scratch_all[NB * 192];
            threadgroup float* scratch = scratch_all + sg * 192;
            const device int32_t* table = tables + (size_t)b * maxp;
            const device T* kn = knew + (size_t)ink2::io_row<TM>(b, kvh, 0, KVH, L) * D;
            const device T* vn = vnew + (size_t)ink2::io_row<TM>(b, kvh, 0, KVH, L) * D;
            const int dense_stride = TM ? KVH * D : D;
            const int tg_lo = ink2::lowest_key(qstart + i0, base, window);
            const int tg_hi = qstart + i1;
            constexpr int HALF = PTOK / 2;
            for (int p = tg_lo / PTOK; p <= tg_hi / PTOK; p++) {
                // The partition's first half, kept while the block takes the second.
                float pm = -FLT_MAX;
                float pl = 0.0f;
                float po[NF * 2];
                for (int h = 0; h < 2; h++) {
                    if (h == 1) {
                        pm = blk.m;
                        pl = blk.l;
                        _Pragma("clang loop unroll(full)")
                        for (int c = 0; c < NF; c++) {
                            po[c * 2] = blk.o[c].thread_elements()[0];
                            po[c * 2 + 1] = blk.o[c].thread_elements()[1];
                        }
                    }
                    blk.reset();
                    const int plo = max(p * PTOK + h * HALF, tg_lo);
                    const int phi = min(p * PTOK + (h + 1) * HALF - 1, tg_hi);
                    const int blo = max(p * PTOK + h * HALF, blk.lo_first);
                    const int bhi = min(p * PTOK + (h + 1) * HALF - 1, blk.qpos_last);
                    for (int pg0 = (plo / S) * S; pg0 <= phi; pg0 += S) {
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        const int page = ink2::page_of(table, pg0, S, tlen, qstart, base);
                        const size_t head = (size_t)kvh * S * D;
                        ink2::stage_page<T, S, D, KS, RS, 32 * NB>(
                            pg0, qstart, base, L, pages.key_page(page) + head,
                            pages.value_page(page) + head, kn, vn, dense_stride, ks, vs,
                            (int)tid);
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        if (has_block && pg0 + S - 1 >= blo && pg0 <= bhi) {
                            const ink2::StagedRows<T, KS, RS> rows{ks, vs};
                            ink2::tile_update<NF, REL, DREL, BIAS>(
                                blk, rows, pg0, proj, scratch, lane, queries);
                        }
                    }
                }
                {
                    // Partition = first half, then the second merged in.
                    const float m1 = blk.m;
                    const float l1 = blk.l;
                    float o1[NF * 2];
                    _Pragma("clang loop unroll(full)")
                    for (int c = 0; c < NF; c++) {
                        o1[c * 2] = blk.o[c].thread_elements()[0];
                        o1[c * 2 + 1] = blk.o[c].thread_elements()[1];
                        blk.o[c].thread_elements()[0] = po[c * 2];
                        blk.o[c].thread_elements()[1] = po[c * 2 + 1];
                    }
                    blk.m = pm;
                    blk.l = pl;
                    ink2::merge_half<NF>(blk.m, blk.l, blk.o, m1, l1, o1);
                }
                if (blk.qpos >= 0 && blk.lo <= (p + 1) * PTOK - 1 && blk.qpos >= p * PTOK) {
                    float a;
                    float w;
                    ink2::merge_partition_scalars(rm, rl, blk.m, blk.l, a, w);
                    _Pragma("clang loop unroll(full)")
                    for (int c = 0; c < NF; c++) {
                        _Pragma("clang loop unroll(full)")
                        for (int e = 0; e < 2; e++) {
                            ro[c * 2 + e] = ink2::merge_value(
                                ro[c * 2 + e], blk.o[c].thread_elements()[e], a, w);
                        }
                    }
                }
            }
            if (blk.qpos >= 0) {
                const int pair = pair0 + fc.y;
                const int i = i0 + pair / GQA;
                const int h = kvh * GQA + pair % GQA;
                const float2 f =
                    ink2::sink_factors(rm, rl, float(sinks[ink2::io_row<TM>(b, h, i, QH, L)]));
                device T* orow = out + ((size_t)(b * L + i) * QH + h) * D;
                _Pragma("clang loop unroll(full)")
                for (int c = 0; c < NF; c++) {
                    orow[c * 8 + fc.x] = T(ink2::finish_value(ro[c * 2], f));
                    orow[c * 8 + fc.x + 1] = T(ink2::finish_value(ro[c * 2 + 1], f));
                }
            }
        """#
}
