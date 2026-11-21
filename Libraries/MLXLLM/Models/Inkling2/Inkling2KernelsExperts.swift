// Grouped routed experts for Inkling 2 decode and verify steps: each selected
// expert's weights are read once per step for every row routed to it.
import Foundation
import MLX

/// One expert bank as stored: dense `[E, out, in]`, or the checkpoint's packed
/// NVFP4 / MXFP8 bytes viewed as `uint32` with `uint8` block scales and, for
/// NVFP4, the per-expert global scale in MLX's convention (`global / 2688`).
public struct Inkling2ExpertWeights {
    public let format: Inkling2ExpertFormat
    public let weight: MLXArray
    public let scales: MLXArray?
    public let globalScale: MLXArray?

    public init(
        format: Inkling2ExpertFormat, weight: MLXArray, scales: MLXArray? = nil,
        globalScale: MLXArray? = nil
    ) {
        self.format = format
        self.weight = weight
        self.scales = scales
        self.globalScale = globalScale
    }

    var experts: Int { weight.dim(0) }
    var outputs: Int { weight.dim(1) }
    var inputs: Int {
        switch format {
        case .dense: weight.dim(2)
        case .nvfp4: weight.dim(2) * 8
        case .mxfp8: weight.dim(2) * 4
        }
    }
    var formatCode: Int {
        switch format {
        case .dense: 0
        case .nvfp4: 1
        case .mxfp8: 2
        }
    }
}

extension Inkling2Kernels {
    /// Most rows (tokens) of one step the grouped kernels take.
    static let maximumExpertRows = 1024
    /// Simdgroups along N (8 outputs each) of the expert products.
    static let expertSimdgroups = 1
    /// Expert products split K eight ways (partials added in order) in every
    /// path: one-row GEMVs give each output eight lanes with short chains, and
    /// the tiles use eight simdgroups per output tile (one arithmetic for all
    /// row counts).
    static let expertSplitK = 8
    static func expertSK(_ k: Int) -> Int { max(1, min(expertSplitK, k / 32)) }

    /// Pairs per work item: each threadgroup takes one (expert, chunk of up to
    /// 64 pairs) work item, eight MMA columns per tile.
    static let pairsPerChunk = 64
    /// NVFP4 decode through a threadgroup byte table (default) or bit
    /// placement; both give the same exact weights (tests compare them).
    /// `DARKBLOOM_INKLING2_NVFP4_TABLE=0` uses the bit placement.
    nonisolated(unsafe) static var nvfp4Table =
        ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_NVFP4_TABLE"] != "0"
    /// Steps with more (row, slot) pairs than this use the `plan` launch
    /// (`DARKBLOOM_INKLING2_PLAN_THRESHOLD` overrides; same bits either way).
    static let planThreshold =
        Int(ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_PLAN_THRESHOLD"] ?? "") ?? 64

    /// Every threadgroup counts the (row, slot) pairs per expert, takes the
    /// `y`-th work item in ascending (expert, chunk) order and gathers that
    /// chunk's pairs in ascending pair order (deterministic).
    private static let expertPreamble = """
        constexpr int EPL = (E + 31) / 32;
        threadgroup atomic_uint ink2_counts[E];
        threadgroup int ink2_work[3];
        threadgroup uint ink2_scan[32];
        threadgroup ushort ink2_pairs[64];
        const uint tid = thread_index_in_threadgroup;
        const uint nt = threads_per_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint nsg = (nt + 31) / 32;
        const int assignments = idx_shape[0] * TOPK;
        for (uint i = tid; i < uint(E); i += nt) {
            atomic_store_explicit(&ink2_counts[i], 0u, memory_order_relaxed);
        }
        if (tid == 0) {
            ink2_work[0] = -1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = int(tid); i < assignments; i += int(nt)) {
            atomic_fetch_add_explicit(&ink2_counts[idx[i]], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            int chunks[EPL];
            int mine = 0;
            for (int j = 0; j < EPL; j++) {
                const int e = int(lane) * EPL + j;
                const int c = e < E
                    ? int(atomic_load_explicit(&ink2_counts[e], memory_order_relaxed)) : 0;
                chunks[j] = (c + 63) / 64;
                mine += chunks[j];
            }
            const int before = simd_prefix_exclusive_sum(mine);
            const int item = int(threadgroup_position_in_grid.y);
            if (item >= before && item < before + mine) {
                int at = before;
                for (int j = 0; j < EPL; j++) {
                    if (item < at + chunks[j]) {
                        const int e = int(lane) * EPL + j;
                        ink2_work[0] = e;
                        ink2_work[1] = item - at;
                        ink2_work[2] = int(atomic_load_explicit(&ink2_counts[e], memory_order_relaxed));
                        break;
                    }
                    at += chunks[j];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int expert = ink2_work[0];
        if (expert < 0) {
            return;
        }
        const int first = ink2_work[1] * 64;
        const int pairs = min(64, ink2_work[2] - first);
        int base = 0;
        for (int i0 = 0; i0 < assignments; i0 += int(nt)) {
            const int i = i0 + int(tid);
            const uint hit = (i < assignments && idx[i] == uint(expert)) ? 1u : 0u;
            const uint before = simd_prefix_exclusive_sum(hit);
            if (lane == 31) {
                ink2_scan[sg] = before + hit;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            int offset = base;
            int total = 0;
            for (uint s = 0; s < nsg; s++) {
                offset += s < sg ? int(ink2_scan[s]) : 0;
                total += int(ink2_scan[s]);
            }
            const int at = offset + int(before) - first;
            if (hit != 0 && at >= 0 && at < 64) {
                ink2_pairs[at] = ushort(i);
            }
            base += total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup const ushort* ink2_list = ink2_pairs;
        """

    /// The work item of a planned launch (above `planThreshold` pairs): one
    /// (expert, first pair, count) row of `items`, pairs listed in `order`.
    private static let plannedPreamble = """
        const int item = int(threadgroup_position_in_grid.y);
        const int expert = items[item * 3];
        if (expert < 0) {
            return;
        }
        const device int* ink2_list = order + items[item * 3 + 1];
        const int pairs = items[item * 3 + 2];
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        """

    /// The MMA products of one work item and the epilogue: gate/up rounded to
    /// T as projection outputs, softcapped SwiGLU in FP32 rounded to T; or the
    /// down projection rounded to T.
    private static func expertSource(up: Bool, planned: Bool) -> String {
        let listType = planned ? "const device int*" : "threadgroup const ushort*"
        let tile = { (rb: Int) in
            "ink2_mma_tile<T, WT, FMT, K, SK, \(rb), FMT == 1 && TBLON, OPT, PF>(x, src, pairs, row0, srow0, ink2_row_stride<FMT, K>(), ink2_scale_stride<FMT, K>(), lane, ks, mine, 8, ink2_lut);"
        }
        let epilogue =
            up
            ? """
            for (int e = int(thread_index_in_threadgroup); e < NSG * cols * 4;
                 e += int(threads_per_threadgroup.x)) {
                const int i = e % 4;
                const int c = (e / 4) % cols;
                const int sn = e / (4 * cols);
                if (c >= pairs) {
                    continue;
                }
                const int firstRow = tile * 8 * NSG + sn * 8;
                const int nbs = min(firstRow, N - 8);
                if (nbs + 2 * i < firstRow) {
                    continue;
                }
                const int at = ((c / 8) * 8 + 2 * i) * 8 + (c % 8);
                float g = buf[sn * 64 * MAXRB + at];
                float u = buf[sn * 64 * MAXRB + at + 8];
                for (int s = 1; s < SK; s++) {
                    g += buf[(s * NSG + sn) * 64 * MAXRB + at];
                    u += buf[(s * NSG + sn) * 64 * MAXRB + at + 8];
                }
                g = float(static_cast<T>(g * inv));
                u = float(static_cast<T>(u * inv));
                const float gated = (metal::precise::tanh(g / caps[0]) * caps[0])
                    * (1.0f / (1.0f + metal::precise::exp(-g)));
                const float a = gated * (metal::precise::tanh(u / caps[1]) * caps[1]);
                out[size_t(ink2_list[c]) * (N / 2) + (nbs / 2 + i)] = static_cast<T>(a);
            }
            """
            : """
            for (int e = int(thread_index_in_threadgroup); e < NSG * cols * 8;
                 e += int(threads_per_threadgroup.x)) {
                const int nl = e % 8;
                const int c = (e / 8) % cols;
                const int sn = e / (8 * cols);
                if (c >= pairs) {
                    continue;
                }
                const int firstRow = tile * 8 * NSG + sn * 8;
                const int n = min(firstRow, N - 8) + nl;
                if (n < firstRow) {
                    continue;
                }
                const int at = ((c / 8) * 8 + nl) * 8 + (c % 8);
                float v = buf[sn * 64 * MAXRB + at];
                for (int s = 1; s < SK; s++) {
                    v += buf[(s * NSG + sn) * 64 * MAXRB + at];
                }
                out[size_t(ink2_list[c]) * N + n] = static_cast<T>(v * inv);
            }
            """
        return (planned ? plannedPreamble : expertPreamble) + """

            const int ks = int(sg) % SK;
            const int ns = int(sg) / SK;
            const int tile = int(threadgroup_position_in_grid.x);
            threadgroup float buf[SK * NSG * 64 * MAXRB];
            threadgroup float2 ink2_lut[FMT == 1 && TBLON ? 256 : 1];
            if (FMT == 1 && TBLON) {
                ink2_fill_e2m1_table(ink2_lut, thread_index_in_threadgroup, threads_per_threadgroup.x);
            }
            threadgroup float* mine = buf + (ks * NSG + ns) * 64 * MAXRB;
            // The last tile of N shifts back to stay inside the expert's rows.
            const int nb = min(tile * 8 * NSG + ns * 8, N - 8);
            const size_t r0 = size_t(expert) * N + size_t(nb);
            const device WT* row0 = w + r0 * ink2_row_stride<FMT, K>();
            const device uchar* srow0 = sc + (FMT == 0 ? size_t(0) : r0 * ink2_scale_stride<FMT, K>());
            const float inv = FMT == 1 ? gs[expert] / 2688.0f : 1.0f;
            const ink2_rows_pairs<\(listType), \(up ? "TOPK" : "1")> src{ink2_list};
            const int rb = pairs <= 8 ? 1 : pairs <= 16 ? 2 : pairs <= 32 ? 4 : 8;
            if (rb == 1) {
                \(tile(1))
            } else if (rb == 2) {
                \(tile(2))
            } else if (rb == 4) {
                \(tile(4))
            } else {
                \(tile(8))
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const int cols = rb * 8;
            \(epilogue)
            """
    }

    private static let expertHeader = header

    /// Groups a step's (row, slot) pairs by expert once: `order` lists the pairs
    /// by ascending expert, then ascending pair; `items` has one (expert, first,
    /// count) row per chunk of up to 64 pairs, in that order, and expert -1
    /// past the last chunk. One threadgroup; thread (e, s) owns expert e on the
    /// s-th contiguous slice of the pairs.
    private static let planKernel = MLXFast.metalKernel(
        name: "inkling2_experts_plan",
        inputNames: ["idx"],
        outputNames: ["items", "order"],
        source: """
            constexpr int S = (1024 / E) > 0 ? (1024 / E) : 1;
            constexpr int EPL = (E + 31) / 32;
            threadgroup int ink2_slice[E * S];
            threadgroup int ink2_total[E];
            threadgroup int ink2_start[E];
            threadgroup int ink2_item[E];
            threadgroup int ink2_items;
            const int tid = int(thread_index_in_threadgroup);
            const int e = tid / S;
            const int s = tid % S;
            const int assignments = idx_shape[0] * TOPK;
            const int lo = (assignments * s) / S;
            const int hi = (assignments * (s + 1)) / S;
            if (e < E) {
                int c = 0;
                for (int i = lo; i < hi; i++) {
                    c += idx[i] == uint(e) ? 1 : 0;
                }
                ink2_slice[tid] = c;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (e < E && s == 0) {
                int t = 0;
                for (int j = 0; j < S; j++) {
                    t += ink2_slice[e * S + j];
                }
                ink2_total[e] = t;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simdgroup_index_in_threadgroup == 0) {
                const int lane = int(thread_index_in_simdgroup);
                int pairs = 0;
                int chunks = 0;
                for (int j = 0; j < EPL; j++) {
                    const int x = lane * EPL + j;
                    const int t = x < E ? ink2_total[x] : 0;
                    pairs += t;
                    chunks += (t + 63) / 64;
                }
                int start = simd_prefix_exclusive_sum(pairs);
                int item = simd_prefix_exclusive_sum(chunks);
                for (int j = 0; j < EPL; j++) {
                    const int x = lane * EPL + j;
                    if (x < E) {
                        ink2_start[x] = start;
                        ink2_item[x] = item;
                        start += ink2_total[x];
                        item += (ink2_total[x] + 63) / 64;
                    }
                }
                if (lane == 31) {
                    ink2_items = item;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (e < E) {
                int at = ink2_start[e];
                for (int j = 0; j < s; j++) {
                    at += ink2_slice[e * S + j];
                }
                for (int i = lo; i < hi; i++) {
                    if (idx[i] == uint(e)) {
                        order[at++] = i;
                    }
                }
                if (s == 0) {
                    const int t = ink2_total[e];
                    for (int q = 0; q * 64 < t; q++) {
                        const int row = ink2_item[e] + q;
                        items[row * 3] = e;
                        items[row * 3 + 1] = ink2_start[e] + q * 64;
                        items[row * 3 + 2] = min(64, t - q * 64);
                    }
                }
            }
            for (int row = ink2_items + tid; row < ITEMS; row += int(threads_per_threadgroup.x)) {
                items[row * 3] = -1;
                items[row * 3 + 1] = 0;
                items[row * 3 + 2] = 0;
            }
            """)

    /// Which one-row expert GEMVs run: 2 (default) shares each input load and
    /// decode between two weight rows per lane; 1 keeps one row per chain
    /// call. Same bits either way. `DARKBLOOM_INKLING2_EXPERT_GEMV`.
    nonisolated(unsafe) static var expertGemvVersion =
        Int(ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_EXPERT_GEMV"] ?? "") ?? 2
    /// Down-projection rows per lane of the version-2 GEMV (1 or 2; same bits).
    /// `DARKBLOOM_INKLING2_EXPERT_DOWN_ROWS`.
    nonisolated(unsafe) static var expertDownRowsPerLane =
        Int(ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_EXPERT_DOWN_ROWS"] ?? "") ?? 1

    /// Helpers of the version-2 one-row kernels. Each lane runs split-K chain
    /// `ks` (superblocks ks, ks + SK, ...) of R weight rows at once: one wide
    /// load per row and superblock, the 32 input values loaded and widened
    /// once for all R rows, and the 32 FMAs per row in the tiles' k-slot
    /// order (`ink2_row_dot`). Operand casts are dropped: every weight and
    /// input value is exact in the operand type, so they are identities.
    static let expertsGemvHeader =
        header + """

            // Superblock s (32 values) of one stored row: NVFP4 16 bytes and
            // two scale bytes, MXFP8 32 bytes and one, dense 32 elements.
            template <typename WT, int FMT>
            struct ink2e_sb {
                uint4 q[FMT == 1 ? 1 : FMT == 2 ? 2 : int(sizeof(WT)) * 2];
                uint s;
            };

            template <typename WT, int FMT>
            inline ink2e_sb<WT, FMT> ink2e_load(
                const device WT* row, const device uchar* srow, int s) {
                constexpr int NQ = FMT == 1 ? 1 : FMT == 2 ? 2 : int(sizeof(WT)) * 2;
                ink2e_sb<WT, FMT> r;
                const device uint4* p = (const device uint4*)row + s * NQ;
                #pragma unroll
                for (int i = 0; i < NQ; i++) {
                    r.q[i] = p[i];
                }
                if constexpr (FMT == 1) {
                    r.s = uint(((const device ushort*)srow)[s]);
                } else if constexpr (FMT == 2) {
                    r.s = uint(srow[s]);
                } else {
                    r.s = 0u;
                }
                return r;
            }

            // Two E2M1 nibbles at bits 4p and 4p + 16 of t as FP32 times 2^-14
            // (ink2_e2m1_pair(t >> 4p) without the shift).
            template <int P>
            inline float2 ink2e_e2m1_at(uint t) {
                constexpr uint mag = 0x00070007u << (4 * P);
                constexpr uint sgn = 0x00080008u << (4 * P);
                uint h;
                if constexpr (P == 3) {
                    h = ((t & mag) >> 3) | (t & sgn);
                } else {
                    h = ((t & mag) << (9 - 4 * P)) | ((t & sgn) << (12 - 4 * P));
                }
                return float2(as_type<half2>(h));
            }

            // The superblock's 32 exact dequantized values in element order (no
            // global scale): ink2_dequant8's values.
            template <typename WT, int FMT>
            inline void ink2e_decode(thread const ink2e_sb<WT, FMT>& r, thread float* v) {
                if constexpr (FMT == 1) {
                    const float sa = ink2_e4m3_shifted(r.s & 0xFFu) * 4194304.0f;
                    const float sb = ink2_e4m3_shifted(r.s >> 8) * 4194304.0f;
                    const uint words[4] = {r.q[0].x, r.q[0].y, r.q[0].z, r.q[0].w};
                    #pragma unroll
                    for (int q = 0; q < 4; q++) {
                        const uint t = words[q];
                        const float s = q < 2 ? sa : sb;
                        const float2 p0 = ink2e_e2m1_at<0>(t);
                        const float2 p1 = ink2e_e2m1_at<1>(t);
                        const float2 p2 = ink2e_e2m1_at<2>(t);
                        const float2 p3 = ink2e_e2m1_at<3>(t);
                        v[q * 8 + 0] = p0.x * s;
                        v[q * 8 + 1] = p1.x * s;
                        v[q * 8 + 2] = p2.x * s;
                        v[q * 8 + 3] = p3.x * s;
                        v[q * 8 + 4] = p0.y * s;
                        v[q * 8 + 5] = p1.y * s;
                        v[q * 8 + 6] = p2.y * s;
                        v[q * 8 + 7] = p3.y * s;
                    }
                } else if constexpr (FMT == 2) {
                    const float s = as_type<float>((r.s + 8u) << 23);
                    const uint words[8] = {
                        r.q[0].x, r.q[0].y, r.q[0].z, r.q[0].w, r.q[1].x, r.q[1].y, r.q[1].z, r.q[1].w};
                    #pragma unroll
                    for (int q = 0; q < 8; q++) {
                        const float2 a0 = ink2_e4m3_pair(words[q]);
                        const float2 a1 = ink2_e4m3_pair(words[q] >> 8);
                        v[q * 4 + 0] = a0.x * s;
                        v[q * 4 + 1] = a1.x * s;
                        v[q * 4 + 2] = a0.y * s;
                        v[q * 4 + 3] = a1.y * s;
                    }
                } else if constexpr (metal::is_same_v<WT, float>) {
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        const float4 f = as_type<float4>(r.q[i]);
                        v[i * 4 + 0] = f.x;
                        v[i * 4 + 1] = f.y;
                        v[i * 4 + 2] = f.z;
                        v[i * 4 + 3] = f.w;
                    }
                } else if constexpr (metal::is_same_v<WT, half>) {
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        const float4 a = float4(as_type<half4>(r.q[i].xy));
                        const float4 b = float4(as_type<half4>(r.q[i].zw));
                        v[i * 8 + 0] = a.x;
                        v[i * 8 + 1] = a.y;
                        v[i * 8 + 2] = a.z;
                        v[i * 8 + 3] = a.w;
                        v[i * 8 + 4] = b.x;
                        v[i * 8 + 5] = b.y;
                        v[i * 8 + 6] = b.z;
                        v[i * 8 + 7] = b.w;
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        const uint4 u = r.q[i];
                        v[i * 8 + 0] = as_type<float>(u.x << 16);
                        v[i * 8 + 1] = as_type<float>(u.x & 0xFFFF0000u);
                        v[i * 8 + 2] = as_type<float>(u.y << 16);
                        v[i * 8 + 3] = as_type<float>(u.y & 0xFFFF0000u);
                        v[i * 8 + 4] = as_type<float>(u.z << 16);
                        v[i * 8 + 5] = as_type<float>(u.z & 0xFFFF0000u);
                        v[i * 8 + 6] = as_type<float>(u.w << 16);
                        v[i * 8 + 7] = as_type<float>(u.w & 0xFFFF0000u);
                    }
                }
            }

            // Input values s * 32 .. s * 32 + 31 of one row as FP32.
            template <typename T>
            inline void ink2e_inputs(const device T* xr, int s, thread float* v) {
                #pragma unroll
                for (int b = 0; b < 4; b++) {
                    const ink2_v8 u = ink2_load8<T>(xr + s * 32 + b * 8);
                    v[b * 8 + 0] = u.lo.x;
                    v[b * 8 + 1] = u.lo.y;
                    v[b * 8 + 2] = u.lo.z;
                    v[b * 8 + 3] = u.lo.w;
                    v[b * 8 + 4] = u.hi.x;
                    v[b * 8 + 5] = u.hi.y;
                    v[b * 8 + 6] = u.hi.z;
                    v[b * 8 + 7] = u.hi.w;
                }
            }

            // Chain `ks` of R consecutive stored rows (from `row0`) against one
            // input row: acc[r] gets ink2_row_chain's value for row r. The next
            // superblock's loads are issued before this one's FMAs.
            template <typename T, typename WT, int FMT, int K, int SK, int R>
            inline void ink2e_chains(
                const device T* xr, const device WT* row0, const device uchar* srow0, int ks,
                thread float* acc) {
                #pragma unroll
                for (int r = 0; r < R; r++) {
                    acc[r] = 0.0f;
                }
                if (ks >= K / 32) {
                    return;
                }
                const size_t rs = ink2_row_stride<FMT, K>();
                const size_t ss = FMT == 0 ? size_t(0) : ink2_scale_stride<FMT, K>();
                ink2e_sb<WT, FMT> cur[R];
                #pragma unroll
                for (int r = 0; r < R; r++) {
                    cur[r] = ink2e_load<WT, FMT>(row0 + r * rs, srow0 + r * ss, ks);
                }
                for (int s = ks; s < K / 32; s += SK) {
                    ink2e_sb<WT, FMT> nxt[R];
                    const int sn = s + SK < K / 32 ? s + SK : s;
                    #pragma unroll
                    for (int r = 0; r < R; r++) {
                        nxt[r] = ink2e_load<WT, FMT>(row0 + r * rs, srow0 + r * ss, sn);
                    }
                    float xv[32];
                    ink2e_inputs<T>(xr, s, xv);
                    float wv[R][32];
                    #pragma unroll
                    for (int r = 0; r < R; r++) {
                        ink2e_decode<WT, FMT>(cur[r], wv[r]);
                    }
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        #pragma unroll
                        for (int c = 0; c < 8; c++) {
                            const int k = (c / 2) * 8 + 2 * j + (c & 1);
                            #pragma unroll
                            for (int r = 0; r < R; r++) {
                                acc[r] = fma(wv[r][k], xv[k], acc[r]);
                            }
                        }
                    }
                    #pragma unroll
                    for (int r = 0; r < R; r++) {
                        cur[r] = nxt[r];
                    }
                }
            }

            // Eight 16-bit elements (bfloat or half) widened to FP32.
            template <typename T>
            inline void ink2e_unpack8(uint4 u, thread float* v) {
                if constexpr (metal::is_same_v<T, half>) {
                    const float4 a = float4(as_type<half4>(u.xy));
                    const float4 b = float4(as_type<half4>(u.zw));
                    v[0] = a.x;
                    v[1] = a.y;
                    v[2] = a.z;
                    v[3] = a.w;
                    v[4] = b.x;
                    v[5] = b.y;
                    v[6] = b.z;
                    v[7] = b.w;
                } else {
                    v[0] = as_type<float>(u.x << 16);
                    v[1] = as_type<float>(u.x & 0xFFFF0000u);
                    v[2] = as_type<float>(u.y << 16);
                    v[3] = as_type<float>(u.y & 0xFFFF0000u);
                    v[4] = as_type<float>(u.z << 16);
                    v[5] = as_type<float>(u.z & 0xFFFF0000u);
                    v[6] = as_type<float>(u.w << 16);
                    v[7] = as_type<float>(u.w & 0xFFFF0000u);
                }
            }

            // Lane `sub` == 0 of each group of SK adjacent lanes receives the
            // chains added in order: ((c0 + c1) + c2) + ...
            template <int SK>
            inline float ink2e_chain_total(float v) {
                float t = v;
                for (int i = 1; i < SK; i++) {
                    t += simd_shuffle_down(v, ushort(i));
                }
                return t;
            }
            """

    /// Version-2 gate/up: SK adjacent lanes per (pair, gate/up row pair), each
    /// running one chain of both rows; the tiles' SwiGLU epilogue.
    private static let gateUpGemv2Kernel = MLXFast.metalKernel(
        name: "inkling2_experts_gate_up_gemv2",
        inputNames: ["x", "idx", "w", "sc", "gs", "caps"],
        outputNames: ["out"],
        source: """
            const int g0 = int(thread_position_in_grid.x);
            const int sub = g0 % SK;
            const int gid = g0 / SK;
            const int pairs = idx_shape[0] * TOPK;
            const bool valid = gid < pairs * (N / 2);
            const int pair = valid ? gid / (N / 2) : 0;
            const int i = valid ? gid % (N / 2) : 0;
            const int expert = int(idx[pair]);
            const size_t r0 = size_t(expert) * N + size_t(2 * i);
            float acc[2];
            ink2e_chains<T, WT, FMT, K, SK, 2>(
                x + size_t(pair / TOPK) * K, w + r0 * ink2_row_stride<FMT, K>(),
                sc + (FMT == 0 ? size_t(0) : r0 * ink2_scale_stride<FMT, K>()), sub, acc);
            float g = ink2e_chain_total<SK>(acc[0]);
            float u = ink2e_chain_total<SK>(acc[1]);
            if (valid && sub == 0) {
                const float inv = FMT == 1 ? gs[expert] / 2688.0f : 1.0f;
                g = float(static_cast<T>(g * inv));
                u = float(static_cast<T>(u * inv));
                const float gated = (metal::precise::tanh(g / caps[0]) * caps[0])
                    * (1.0f / (1.0f + metal::precise::exp(-g)));
                const float a = gated * (metal::precise::tanh(u / caps[1]) * caps[1]);
                out[size_t(pair) * (N / 2) + i] = static_cast<T>(a);
            }
            """,
        header: expertsGemvHeader)

    /// Version-2 down: SK adjacent lanes per (pair, RD consecutive outputs).
    private static let downGemv2Kernel = MLXFast.metalKernel(
        name: "inkling2_experts_down_gemv2",
        inputNames: ["x", "idx", "w", "sc", "gs"],
        outputNames: ["out"],
        source: """
            const int g0 = int(thread_position_in_grid.x);
            const int sub = g0 % SK;
            const int gid = g0 / SK;
            const int pairs = idx_shape[0] * TOPK;
            const bool valid = gid < pairs * (N / RD);
            const int pair = valid ? gid / (N / RD) : 0;
            const int n = valid ? RD * (gid % (N / RD)) : 0;
            const int expert = int(idx[pair]);
            const size_t r0 = size_t(expert) * N + size_t(n);
            float acc[RD];
            ink2e_chains<T, WT, FMT, K, SK, RD>(
                x + size_t(pair) * K, w + r0 * ink2_row_stride<FMT, K>(),
                sc + (FMT == 0 ? size_t(0) : r0 * ink2_scale_stride<FMT, K>()), sub, acc);
            float v[RD];
            #pragma unroll
            for (int r = 0; r < RD; r++) {
                v[r] = ink2e_chain_total<SK>(acc[r]);
            }
            if (valid && sub == 0) {
                const float inv = FMT == 1 ? gs[expert] / 2688.0f : 1.0f;
                #pragma unroll
                for (int r = 0; r < RD; r++) {
                    out[size_t(pair) * N + n + r] = static_cast<T>(v[r] * inv);
                }
            }
            """,
        header: expertsGemvHeader)

    /// One-row steps: one thread per (pair, output) with `ink2_row_dot`, the
    /// tiles' bits without seven idle MMA columns per weight. Gate/up: one
    /// thread per (pair, gate/up row pair) with the tiles' SwiGLU epilogue.
    private static let gateUpGemvKernel = MLXFast.metalKernel(
        name: "inkling2_experts_gate_up_gemv",
        inputNames: ["x", "idx", "w", "sc", "gs", "caps"],
        outputNames: ["out"],
        source: """
            // SK adjacent lanes per (pair, gate/up row pair): one split-K chain
            // each, added in chain order (the tiles' arithmetic).
            const int g0 = int(thread_position_in_grid.x);
            const int sub = g0 % SK;
            const int gid = g0 / SK;
            const int pairs = idx_shape[0] * TOPK;
            const bool valid = gid < pairs * (N / 2);
            const int pair = valid ? gid / (N / 2) : 0;
            const int i = valid ? gid % (N / 2) : 0;
            const int expert = int(idx[pair]);
            const device T* xr = x + size_t(pair / TOPK) * K;
            const size_t r0 = size_t(expert) * N + size_t(2 * i);
            const device WT* wg = w + r0 * ink2_row_stride<FMT, K>();
            const device uchar* sg = sc + (FMT == 0 ? size_t(0) : r0 * ink2_scale_stride<FMT, K>());
            const device WT* wu = wg + ink2_row_stride<FMT, K>();
            const device uchar* su = sg + (FMT == 0 ? size_t(0) : ink2_scale_stride<FMT, K>());
            const float gp = ink2_row_chain<T, WT, FMT, K, SK, OPT>(xr, wg, sg, sub);
            const float up = ink2_row_chain<T, WT, FMT, K, SK, OPT>(xr, wu, su, sub);
            float g = gp;
            float u = up;
            for (int s = 1; s < SK; s++) {
                g += simd_shuffle_down(gp, ushort(s));
                u += simd_shuffle_down(up, ushort(s));
            }
            if (valid && sub == 0) {
                const float inv = FMT == 1 ? gs[expert] / 2688.0f : 1.0f;
                g = float(static_cast<T>(g * inv));
                u = float(static_cast<T>(u * inv));
                const float gated = (metal::precise::tanh(g / caps[0]) * caps[0])
                    * (1.0f / (1.0f + metal::precise::exp(-g)));
                const float a = gated * (metal::precise::tanh(u / caps[1]) * caps[1]);
                out[size_t(pair) * (N / 2) + i] = static_cast<T>(a);
            }
            """,
        header: header)

    private static let downGemvKernel = MLXFast.metalKernel(
        name: "inkling2_experts_down_gemv",
        inputNames: ["x", "idx", "w", "sc", "gs"],
        outputNames: ["out"],
        source: """
            const int g0 = int(thread_position_in_grid.x);
            const int sub = g0 % SK;
            const int gid = g0 / SK;
            const int pairs = idx_shape[0] * TOPK;
            const bool valid = gid < pairs * N;
            const int pair = valid ? gid / N : 0;
            const int n = valid ? gid % N : 0;
            const int expert = int(idx[pair]);
            const size_t r0 = size_t(expert) * N + size_t(n);
            const device WT* wr = w + r0 * ink2_row_stride<FMT, K>();
            const device uchar* sr = sc + (FMT == 0 ? size_t(0) : r0 * ink2_scale_stride<FMT, K>());
            const float vp = ink2_row_chain<T, WT, FMT, K, SK, OPT>(x + size_t(pair) * K, wr, sr, sub);
            float v = vp;
            for (int s = 1; s < SK; s++) {
                v += simd_shuffle_down(vp, ushort(s));
            }
            if (valid && sub == 0) {
                const float inv = FMT == 1 ? gs[expert] / 2688.0f : 1.0f;
                out[size_t(pair) * N + n] = static_cast<T>(v * inv);
            }
            """,
        header: header)

    private static let gateUpKernel = MLXFast.metalKernel(
        name: "inkling2_experts_gate_up",
        inputNames: ["x", "idx", "w", "sc", "gs", "caps"],
        outputNames: ["out"],
        source: expertSource(up: true, planned: false),
        header: expertHeader)

    private static let downKernel = MLXFast.metalKernel(
        name: "inkling2_experts_down",
        inputNames: ["x", "idx", "w", "sc", "gs"],
        outputNames: ["out"],
        source: expertSource(up: false, planned: false),
        header: expertHeader)

    private static let plannedGateUpKernel = MLXFast.metalKernel(
        name: "inkling2_experts_gate_up_planned",
        inputNames: ["x", "items", "order", "w", "sc", "gs", "caps"],
        outputNames: ["out"],
        source: expertSource(up: true, planned: true),
        header: expertHeader)

    private static let plannedDownKernel = MLXFast.metalKernel(
        name: "inkling2_experts_down_planned",
        inputNames: ["x", "items", "order", "w", "sc", "gs"],
        outputNames: ["out"],
        source: expertSource(up: false, planned: true),
        header: expertHeader)

    /// Each row's `TOPK` expert outputs times their route weights, summed in
    /// FP32 in ascending expert order from zero (the reference's `index_add_`
    /// order), rounded to T.
    private static let combineKernel = MLXFast.metalKernel(
        name: "inkling2_experts_combine",
        inputNames: ["y", "idx", "wts"],
        outputNames: ["out"],
        source: """
            const uint gid = thread_position_in_grid.x;
            const int rows = idx_shape[0];
            if (int(gid) >= rows * N) {
                return;
            }
            const int r = int(gid) / N;
            const int n = int(gid) % N;
            uint e[TOPK];
            int order[TOPK];
            for (int s = 0; s < TOPK; s++) {
                e[s] = idx[r * TOPK + s];
            }
            for (int s = 0; s < TOPK; s++) {
                int rank = 0;
                for (int t = 0; t < TOPK; t++) {
                    rank += (e[t] < e[s] || (e[t] == e[s] && t < s)) ? 1 : 0;
                }
                order[rank] = s;
            }
            float sum = 0.0f;
            for (int k = 0; k < TOPK; k++) {
                const int s = order[k];
                const float c = float(y[size_t(r * TOPK + s) * N + n]) * wts[r * TOPK + s];
                sum = sum + c;
            }
            out[gid] = static_cast<T>(sum);
            """)

    private final class Constants: @unchecked Sendable {
        private let lock = NSLock()
        private var caps: [UInt64: MLXArray] = [:]
        private var route: [UInt32: MLXArray] = [:]
        // Eight elements: smaller inputs bind as `constant`, not `device`.
        let byte = MLXArray([UInt8](repeating: 0, count: 8))
        let one = MLXArray([Float](repeating: 1, count: 8))

        /// `[routeScale, 1e-20, ...]` (eight elements).
        func routeParams(_ scale: Float) -> MLXArray {
            lock.withLock {
                if let a = route[scale.bitPattern] { return a }
                let a = MLXArray([scale, 1e-20, 0, 0, 0, 0, 0, 0] as [Float])
                eval(a)
                route[scale.bitPattern] = a
                return a
            }
        }

        func caps(_ gate: Float, _ up: Float) -> MLXArray {
            lock.withLock {
                let key = UInt64(gate.bitPattern) << 32 | UInt64(up.bitPattern)
                if let a = caps[key] { return a }
                let a = MLXArray([gate, up])
                eval(a)
                caps[key] = a
                return a
            }
        }
    }

    private static let constants = Constants()

    static func covers(_ bank: Inkling2ExpertWeights, dtype: DType, inputs k: Int) -> Bool {
        guard bank.weight.ndim == 3, bank.inputs == k, k % 32 == 0, k >= 32,
            bank.outputs >= 8
        else { return false }
        switch bank.format {
        case .dense:
            return bank.weight.dtype == dtype
        case .nvfp4:
            return bank.weight.dtype == .uint32 && bank.scales?.dtype == .uint8
                && bank.globalScale?.dtype == .float32
        case .mxfp8:
            return bank.weight.dtype == .uint32 && bank.scales?.dtype == .uint8
        }
    }

    /// Whether the grouped kernels take a step of `rows` rows.
    static func covers(
        rows: Int, topK: Int, hidden: Int, dtype: DType, gateUp: Inkling2ExpertWeights,
        down: Inkling2ExpertWeights
    ) -> Bool {
        let intermediate = gateUp.outputs / 2
        return rows > 0 && rows <= maximumExpertRows && rows * topK <= Int(UInt16.max)
            && topK > 0 && topK <= gateUp.experts && gateUp.experts <= 1024
            && supportsActivation(dtype)
            && gateUp.outputs % 2 == 0 && down.experts == gateUp.experts
            && down.outputs == hidden && covers(gateUp, dtype: dtype, inputs: hidden)
            && covers(down, dtype: dtype, inputs: intermediate)
    }

    /// The routed experts of one step: `x` `[R, H]`, `indices` `[R, topK]`
    /// (uint32, distinct per row), `weights` `[R, topK]` FP32. Returns `[R, H]`
    /// in `x.dtype`, or nil when the kernels are off or do not cover the input.
    public static func expertProduct(
        _ x: MLXArray, indices: MLXArray, weights: MLXArray, gateUp: Inkling2ExpertWeights,
        down: Inkling2ExpertWeights, gateCap: Float, upCap: Float
    ) -> MLXArray? {
        guard weights.shape == indices.shape,
            let projected = expertsUpDown(
                x, indices: indices, gateUp: gateUp, down: down, gateCap: gateCap, upCap: upCap)
        else { return nil }
        let (rows, hidden, topK) = (x.dim(0), x.dim(1), indices.dim(1))
        let idx = indices.dtype == .uint32 ? indices : indices.asType(.uint32)
        let total = rows * hidden
        let combineKey = LaunchKey(kernel: 53, dtype: x.dtype, a: hidden, b: topK, c: rows)
        let combineLaunch = launches.launch(combineKey) {
            combineKernel.prepare(
                template: [("T", x.dtype), ("N", hidden), ("TOPK", topK)],
                grid: ((total + 255) / 256 * 256, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [[rows, hidden]], outputDTypes: [x.dtype])
        }
        let w = weights.dtype == .float32 ? weights : weights.asType(.float32)
        return combineLaunch([projected, idx, w])[0]
    }

    /// Gate/up with the activation and the down projection of every (row,
    /// slot) pair: `[R * topK, H]` in `x.dtype`, pair `r * topK + s` at row
    /// `r * topK + s` (no combine).
    static func expertsUpDown(
        _ x: MLXArray, indices: MLXArray, gateUp: Inkling2ExpertWeights,
        down: Inkling2ExpertWeights, gateCap: Float, upCap: Float
    ) -> MLXArray? {
        guard active, x.ndim == 2, indices.ndim == 2, indices.dim(0) == x.dim(0),
            covers(
                rows: x.dim(0), topK: indices.dim(1), hidden: x.dim(1), dtype: x.dtype,
                gateUp: gateUp, down: down)
        else { return nil }
        let (rows, hidden) = (x.dim(0), x.dim(1))
        let topK = indices.dim(1)
        let experts = gateUp.experts
        let intermediate = gateUp.outputs / 2
        let idx = indices.dtype == .uint32 ? indices : indices.asType(.uint32)
        let slots = min(experts, rows * topK) + rows * topK / pairsPerChunk
        let nsg = expertSimdgroups
        let tm = 8
        let pairs = rows * topK
        // Launch keys carry every template flag (table decode, narrow operands).
        let flags = (nvfp4Table ? 100 : 0) + (narrowOperands ? 1000 : 0)

        func inputs(_ bank: Inkling2ExpertWeights) -> [MLXArray] {
            [bank.weight, bank.scales ?? constants.byte, bank.globalScale ?? constants.one]
        }
        func template(_ bank: Inkling2ExpertWeights, n: Int, k: Int, pf: Int) -> [(
            String, any KernelTemplateArg
        )] {
            [
                ("PF", pf), ("MAXRB", maxRB(rows)),
                ("T", x.dtype), ("WT", bank.weight.dtype), ("FMT", bank.formatCode), ("N", n),
                ("K", k), ("E", experts), ("TOPK", topK), ("NSG", nsg), ("SK", expertSK(k)),
                ("TBLON", nvfp4Table),
                (
                    "OPT",
                    operandType(x: x.dtype, weight: bank.weight.dtype, dense: bank.format == .dense)
                ),
            ]
        }

        // One-row steps: one thread per output (same bits as the tiles).
        let upOperand = operandType(
            x: x.dtype, weight: gateUp.weight.dtype, dense: gateUp.format == .dense)
        let downOperand = operandType(
            x: x.dtype, weight: down.weight.dtype, dense: down.format == .dense)
        let oneRow = rows == 1 && upOperand != .float32 && downOperand != .float32
        let gemvUp = oneRow && gemvSite("up")
        let gemvDown = oneRow && gemvSite("down")
        let rd = expertDownRowsPerLane
        let gemv2Down = expertGemvVersion == 2 && (rd == 1 || rd == 2) && hidden % rd == 0
        func gemvDownProjection(_ activated: MLXArray) -> MLXArray {
            if gemv2Down {
                let downKey = LaunchKey(
                    kernel: 51 + flags, dtype: x.dtype, weightType: down.weight.dtype,
                    a: down.formatCode * 10 + rd, b: hidden, c: intermediate, d: experts, e: topK,
                    f: rows)
                let downLaunch = launches.launch(downKey) {
                    downGemv2Kernel.prepare(
                        template: template(down, n: hidden, k: intermediate, pf: 1) + [("RD", rd)],
                        grid: (
                            (pairs * (hidden / rd) * expertSK(intermediate) + 255) / 256 * 256, 1,
                            1
                        ),
                        threadGroup: (256, 1, 1),
                        outputShapes: [[pairs, hidden]], outputDTypes: [x.dtype])
                }
                return downLaunch([activated, idx] + inputs(down))[0]
            }
            let downKey = LaunchKey(
                kernel: 41 + flags, dtype: x.dtype, weightType: down.weight.dtype,
                a: down.formatCode, b: hidden, c: intermediate, d: experts, e: topK, f: rows)
            let downLaunch = launches.launch(downKey) {
                downGemvKernel.prepare(
                    template: template(down, n: hidden, k: intermediate, pf: 1),
                    grid: ((pairs * hidden * expertSK(intermediate) + 255) / 256 * 256, 1, 1),
                    threadGroup: (256, 1, 1),
                    outputShapes: [[pairs, hidden]], outputDTypes: [x.dtype])
            }
            return downLaunch([activated, idx] + inputs(down))[0]
        }
        var gemvActivated: MLXArray?
        if gemvUp, expertGemvVersion == 2 {
            let upKey = LaunchKey(
                kernel: 50 + flags, dtype: x.dtype, weightType: gateUp.weight.dtype,
                a: gateUp.formatCode, b: gateUp.outputs, c: hidden, d: experts, e: topK, f: rows)
            let upLaunch = launches.launch(upKey) {
                gateUpGemv2Kernel.prepare(
                    template: template(gateUp, n: gateUp.outputs, k: hidden, pf: 1),
                    grid: ((pairs * intermediate * expertSK(hidden) + 255) / 256 * 256, 1, 1),
                    threadGroup: (256, 1, 1),
                    outputShapes: [[pairs, intermediate]], outputDTypes: [x.dtype])
            }
            gemvActivated =
                upLaunch(
                    [x, idx] + inputs(gateUp) + [constants.caps(gateCap, upCap)])[0]
        } else if gemvUp {
            let upKey = LaunchKey(
                kernel: 40 + flags, dtype: x.dtype, weightType: gateUp.weight.dtype,
                a: gateUp.formatCode, b: gateUp.outputs, c: hidden, d: experts, e: topK, f: rows)
            let upLaunch = launches.launch(upKey) {
                gateUpGemvKernel.prepare(
                    template: template(gateUp, n: gateUp.outputs, k: hidden, pf: 1),
                    grid: ((pairs * intermediate * expertSK(hidden) + 255) / 256 * 256, 1, 1),
                    threadGroup: (256, 1, 1),
                    outputShapes: [[pairs, intermediate]], outputDTypes: [x.dtype])
            }
            gemvActivated =
                upLaunch(
                    [x, idx] + inputs(gateUp) + [constants.caps(gateCap, upCap)])[0]
        }

        // Small steps find their work items inside each threadgroup; larger ones
        // group the pairs once (`plan`) for both projections. Same bits either way.
        let planned = pairs > planThreshold
        var routing: [MLXArray] = [idx]
        if planned {
            let planThreads = min(1024, (experts * max(1, 1024 / experts) + 31) / 32 * 32)
            let planKey = LaunchKey(kernel: 3, dtype: .uint32, a: experts, b: topK, c: rows)
            let planLaunch = launches.launch(planKey) {
                planKernel.prepare(
                    template: [("E", experts), ("TOPK", topK), ("ITEMS", slots)],
                    grid: (planThreads, 1, 1), threadGroup: (planThreads, 1, 1),
                    outputShapes: [[slots * 3], [pairs]], outputDTypes: [.int32, .int32])
            }
            routing = planLaunch([idx])
        }

        let upTiles = (gateUp.outputs + tm * nsg - 1) / (tm * nsg)
        let upPF = prefetch(threadgroups: upTiles * slots)
        let upKey = LaunchKey(
            kernel: (planned ? 5 : 4) + (nvfp4Table ? 100 : 0) + (narrowOperands ? 1000 : 0)
                + (upPF > 1 ? 10_000 : 0),
            dtype: x.dtype,
            weightType: gateUp.weight.dtype,
            a: gateUp.formatCode, b: gateUp.outputs, c: hidden, d: experts, e: topK, f: rows)
        let upLaunch = launches.launch(upKey) {
            (planned ? plannedGateUpKernel : gateUpKernel).prepare(
                template: template(gateUp, n: gateUp.outputs, k: hidden, pf: upPF),
                grid: (upTiles * 32 * nsg * expertSK(hidden), slots, 1),
                threadGroup: (32 * nsg * expertSK(hidden), 1, 1),
                outputShapes: [[pairs, intermediate]], outputDTypes: [x.dtype])
        }
        let activated =
            gemvActivated
            ?? upLaunch([x] + routing + inputs(gateUp) + [constants.caps(gateCap, upCap)])[0]

        let downTiles = (hidden + tm * nsg - 1) / (tm * nsg)
        let downPF = prefetch(threadgroups: downTiles * slots)
        let downKey = LaunchKey(
            kernel: (planned ? 7 : 6) + (nvfp4Table ? 100 : 0) + (narrowOperands ? 1000 : 0)
                + (downPF > 1 ? 10_000 : 0),
            dtype: x.dtype,
            weightType: down.weight.dtype,
            a: down.formatCode, b: hidden, c: intermediate, d: experts, e: topK, f: rows)
        let downLaunch = launches.launch(downKey) {
            (planned ? plannedDownKernel : downKernel).prepare(
                template: template(down, n: hidden, k: intermediate, pf: downPF),
                grid: (downTiles * 32 * nsg * expertSK(intermediate), slots, 1),
                threadGroup: (32 * nsg * expertSK(intermediate), 1, 1),
                outputShapes: [[pairs, hidden]], outputDTypes: [x.dtype])
        }
        if gemvDown, !planned {
            return gemvDownProjection(activated)
        }
        let projected = downLaunch([activated] + routing + inputs(down))[0]

        return projected
    }

    /// The stored arrays of an `Inkling2ExpertBank`.
    static func weights(of bank: Inkling2ExpertBank) -> Inkling2ExpertWeights? {
        if let q = bank as? Inkling2QuantizedExpertBank {
            return Inkling2ExpertWeights(
                format: q.format, weight: q.weight, scales: q.scales, globalScale: q.globalScale)
        }
        if let d = bank as? Inkling2DenseExpertBank {
            return Inkling2ExpertWeights(format: .dense, weight: d.weight)
        }
        return nil
    }

    /// The router after its FP32 logits, one simdgroup per row: sigmoid scores
    /// (MLX's formula), `gate.bias`-corrected selection, the top-k picked in
    /// descending selection order (lowest expert on a tie), scores renormalized
    /// in pick order and scaled. Outputs indices `[R, K]` (rank order), weights
    /// `[R, K]` and the selection scores `[R, E]` (diagnostics).
    /// One-row router: the split GEMV's FP32 logits (the router product's
    /// bits) in one threadgroup, then the select kernel's code on its first
    /// simdgroup. One launch instead of two.
    private static let routerSelectKernel = MLXFast.metalKernel(
        name: "inkling2_router_gemv_select",
        inputNames: ["x", "w", "bias", "params"],
        outputNames: ["indices", "weights", "selection"],
        source: """
            threadgroup float lg[E];
            const int t = int(thread_index_in_threadgroup);
            const int sub = t % SK;
            const int o = t / SK;
            const float v = o < E
                ? ink2_row_chain<T, WT, 0, K, SK, OPT>(x, w + size_t(o) * K, nullptr, sub) : 0.0f;
            float sum = v;
            for (int i = 1; i < SK; i++) {
                sum += simd_shuffle_down(v, ushort(i));
            }
            if (sub == 0 && o < E) {
                lg[o] = sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (t >= 32) {
                return;
            }
            constexpr int EPL = (E + 31) / 32;
            const int row = 0;
            const int lane = t;
            float score[EPL];
            float sel[EPL];
            for (int j = 0; j < EPL; j++) {
                const int e = lane + 32 * j;
                if (e < E) {
                    const float x = lg[e];
                    const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
                    score[j] = x < 0.0f ? y : 1.0f - y;
                    sel[j] = score[j] + bias[e];
                    selection[size_t(row) * E + e] = sel[j];
                } else {
                    score[j] = 0.0f;
                    sel[j] = -INFINITY;
                }
            }
            float picked[TOPK];
            for (int k = 0; k < TOPK; k++) {
                float best = -INFINITY;
                int bestE = E;
                for (int j = 0; j < EPL; j++) {
                    const int e = lane + 32 * j;
                    if (e < E && (sel[j] > best || (sel[j] == best && e < bestE))) {
                        best = sel[j];
                        bestE = e;
                    }
                }
                const float top = simd_max(best);
                const int winner = simd_min(best == top ? bestE : E);
                float s = 0.0f;
                for (int j = 0; j < EPL; j++) {
                    if (lane + 32 * j == winner) {
                        s = score[j];
                        sel[j] = -INFINITY;
                    }
                }
                picked[k] = simd_sum(s);
                if (lane == 0) {
                    indices[row * TOPK + k] = uint(winner);
                }
            }
            if (lane == 0) {
                float total = 0.0f;
                for (int k = 0; k < TOPK; k++) {
                    total += picked[k];
                }
                total += params[1];
                for (int k = 0; k < TOPK; k++) {
                    weights[row * TOPK + k] = picked[k] / total * params[0];
                }
            }
            """,
        header: header)

    /// The select of `selectKernel` on simdgroup 0 for one row whose FP32
    /// logits are `lg[e]` (same code, same bits).
    private static let selectBody = """

        {
            constexpr int EPL = (E + 31) / 32;
            const int row = 0;
            const int lane = int(thread_index_in_simdgroup);
            float score[EPL];
            float sel[EPL];
            for (int j = 0; j < EPL; j++) {
                const int e = lane + 32 * j;
                if (e < E) {
                    const float x = SELECT_LOGIT(e);
                    const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
                    score[j] = x < 0.0f ? y : 1.0f - y;
                    sel[j] = score[j] + bias[e];
                    selection[size_t(row) * E + e] = sel[j];
                } else {
                    score[j] = 0.0f;
                    sel[j] = -INFINITY;
                }
            }
            float picked[TOPK];
            for (int k = 0; k < TOPK; k++) {
                float best = -INFINITY;
                int bestE = E;
                for (int j = 0; j < EPL; j++) {
                    const int e = lane + 32 * j;
                    if (e < E && (sel[j] > best || (sel[j] == best && e < bestE))) {
                        best = sel[j];
                        bestE = e;
                    }
                }
                const float top = simd_max(best);
                const int winner = simd_min(best == top ? bestE : E);
                float s = 0.0f;
                for (int j = 0; j < EPL; j++) {
                    if (lane + 32 * j == winner) {
                        s = score[j];
                        sel[j] = -INFINITY;
                    }
                }
                picked[k] = simd_sum(s);
                if (lane == 0) {
                    indices[row * TOPK + k] = uint(winner);
                }
            }
            if (lane == 0) {
                float total = 0.0f;
                for (int k = 0; k < TOPK; k++) {
                    total += picked[k];
                }
                total += params[1];
                for (int k = 0; k < TOPK; k++) {
                    weights[row * TOPK + k] = picked[k] / total * params[0];
                }
            }
        }
        """

    /// One-row router in one launch over E / OPG threadgroups: each stages its
    /// OPG weight rows and the input in threadgroup memory (all loads in
    /// flight at once), runs the SK split-K chains of its outputs (the router
    /// product's bits: `ink2_row_chain`, chains added in order) and publishes
    /// the FP32 logits; the last threadgroup to arrive runs the select. The
    /// arrival counter (`counters[0]`) is a zeroed buffer of this launch alone.
    private static let routerFusedKernel = MLXFast.metalKernel(
        name: "inkling2_router_fused",
        inputNames: ["x", "w", "bias", "params", "counters"],
        outputNames: ["indices", "weights", "selection", "logits"],
        source: """
            constexpr int G = E / OPG;
            constexpr int RQ = K * int(sizeof(WT)) / 16;
            constexpr int XQ = K * int(sizeof(T)) / 16;
            threadgroup uint4 ink2_w[OPG * RQ];
            threadgroup uint4 ink2_x[XQ];
            threadgroup uint ink2_arrival;
            const uint tid = thread_index_in_threadgroup;
            const uint grp = threadgroup_position_in_grid.x;
            const device uint4* wsrc = (const device uint4*)(w + size_t(grp) * OPG * K);
            for (uint i = tid; i < uint(OPG * RQ); i += NT) {
                ink2_w[i] = wsrc[i];
            }
            for (uint i = tid; i < uint(XQ); i += NT) {
                ink2_x[i] = ((const device uint4*)x)[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            device atomic_float* lg = reinterpret_cast<device atomic_float*>(logits);
            if (tid < uint(OPG * SK)) {
                const int o = int(tid) / SK;
                const int sub = int(tid) % SK;
                const threadgroup uint4* wq = ink2_w + o * RQ;
                float a = 0.0f;
                for (int s = sub; s < K / 32; s += SK) {
                    float wv[32];
                    float xv[32];
                    #pragma unroll
                    for (int b = 0; b < 4; b++) {
                        ink2e_unpack8<WT>(wq[s * 4 + b], wv + b * 8);
                        ink2e_unpack8<T>(ink2_x[s * 4 + b], xv + b * 8);
                    }
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        #pragma unroll
                        for (int c = 0; c < 8; c++) {
                            const int k = (c / 2) * 8 + 2 * j + (c & 1);
                            a = fma(wv[k], xv[k], a);
                        }
                    }
                }
                const float total = ink2e_chain_total<SK>(a);
                if (sub == 0) {
                    atomic_store_explicit(&lg[grp * OPG + o], total, memory_order_relaxed);
                }
            }
            // Count this threadgroup in; only the last to arrive continues.
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            device atomic_uint* counter = reinterpret_cast<device atomic_uint*>(counters);
            if (tid == 0) {
                atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
                ink2_arrival = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
                atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            if (ink2_arrival != uint(G - 1)) {
                return;
            }
            if (tid >= 32) {
                return;
            }
            #define SELECT_LOGIT(e) atomic_load_explicit(&lg[e], memory_order_relaxed)
            """ + selectBody + """

                #undef SELECT_LOGIT
                """,
        header: expertsGemvHeader,
        mutableInputs: ["counters"])

    /// `DARKBLOOM_INKLING2_FUSED_ROUTER=0` keeps the router product plus the
    /// select kernel at one row.
    nonisolated(unsafe) static var fusedRouter =
        ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_FUSED_ROUTER"] != "0"
    /// Router outputs per threadgroup of the fused router.
    static let fusedRouterOutputsPerGroup = 4

    /// The fused one-launch router of one row (`routerFusedKernel`), or nil
    /// when it does not cover the shapes.
    static func routerFusedOneRow(_ router: Inkling2Router, _ x: MLXArray, weight w: MLXArray)
        -> (MLXArray, MLXArray, MLXArray)?
    {
        let (experts, k, topK) = (w.dim(0), w.dim(1), router.topK)
        let sk = segmentedSplitK(k, outputs: experts)
        let opg = fusedRouterOutputsPerGroup
        // Exact 16-bit inputs and weights; independent of `narrowOperands` (a
        // test policy for the MMA operand type, which this kernel does not use).
        let exact = x.dtype == w.dtype && (x.dtype == .bfloat16 || x.dtype == .float16)
        let bytes = (opg * w.itemSize + x.itemSize) * k
        guard exact, k % 32 == 0, 32 % sk == 0, opg * sk <= 32, experts % opg == 0,
            experts >= topK, experts >= 32, experts <= 1024, bytes <= 28 * 1024
        else { return nil }
        let threads = 128
        // A fresh zeroed counter per launch: no state shared between launches,
        // engines or streams, and nothing left behind by an abandoned step.
        let counters = MLXArray.zeros([8], dtype: .uint32)
        let key = LaunchKey(
            kernel: 52, dtype: x.dtype, weightType: w.dtype, a: experts, b: topK, c: k)
        let launch = launches.launch(key) {
            routerFusedKernel.prepare(
                template: [
                    ("T", x.dtype), ("WT", w.dtype), ("K", k), ("SK", sk), ("E", experts),
                    ("TOPK", topK), ("OPG", opg), ("NT", threads),
                ],
                grid: (experts / opg * threads, 1, 1), threadGroup: (threads, 1, 1),
                outputShapes: [[1, topK], [1, topK], [1, experts], [1, experts]],
                outputDTypes: [.uint32, .float32, .float32, .float32])
        }
        let out = launch([
            x, w, router.bias.asType(.float32), constants.routeParams(router.routeScale), counters,
        ])
        return (out[0], out[1], out[2])
    }

    /// The router of one row in one launch: the fused multi-threadgroup router
    /// (default), else the one-threadgroup GEMV + select on the `router` GEMV
    /// site; nil when neither applies.
    static func routerSelectOneRow(_ router: Inkling2Router, _ x: MLXArray)
        -> (MLXArray, MLXArray, MLXArray)?
    {
        let w = halfCopiesForRouter(router.weight)
        let (experts, k, topK) = (w.dim(0), w.dim(1), router.topK)
        guard x.ndim == 2, x.dim(0) == 1, x.dim(1) == k, x.dtype == w.dtype else { return nil }
        if fusedRouter, let fused = routerFusedOneRow(router, x, weight: w) { return fused }
        let sk = segmentedSplitK(k, outputs: experts)
        let opt = operandType(x: x.dtype, weight: w.dtype, dense: true)
        guard gemvSite("router"), x.ndim == 2, x.dim(0) == 1, x.dim(1) == k, x.dtype == w.dtype,
            opt != .float32, k % 32 == 0, experts * sk <= 1024, experts >= topK
        else { return nil }
        let key = LaunchKey(kernel: 42, dtype: x.dtype, weightType: opt, a: experts, b: topK, c: k)
        let threads = (experts * sk + 31) / 32 * 32
        let launch = launches.launch(key) {
            routerSelectKernel.prepare(
                template: [
                    ("T", x.dtype), ("WT", w.dtype), ("K", k), ("SK", sk), ("OPT", opt),
                    ("E", experts), ("TOPK", topK),
                ],
                grid: (threads, 1, 1), threadGroup: (threads, 1, 1),
                outputShapes: [[1, topK], [1, topK], [1, experts]],
                outputDTypes: [.uint32, .float32, .float32])
        }
        let out = launch([
            x, w, router.bias.asType(.float32), constants.routeParams(router.routeScale),
        ])
        return (out[0], out[1], out[2])
    }

    private static let selectKernel = MLXFast.metalKernel(
        name: "inkling2_router_select",
        inputNames: ["logits", "bias", "params"],
        outputNames: ["indices", "weights", "selection"],
        source: """
            constexpr int EPL = (E + 31) / 32;
            const int row = int(threadgroup_position_in_grid.x);
            const int lane = int(thread_index_in_simdgroup);
            float score[EPL];
            float sel[EPL];
            for (int j = 0; j < EPL; j++) {
                const int e = lane + 32 * j;
                if (e < E) {
                    const float x = logits[size_t(row) * E + e];
                    const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
                    score[j] = x < 0.0f ? y : 1.0f - y;
                    sel[j] = score[j] + bias[e];
                    selection[size_t(row) * E + e] = sel[j];
                } else {
                    score[j] = 0.0f;
                    sel[j] = -INFINITY;
                }
            }
            float picked[TOPK];
            for (int k = 0; k < TOPK; k++) {
                float best = -INFINITY;
                int bestE = E;
                for (int j = 0; j < EPL; j++) {
                    const int e = lane + 32 * j;
                    if (e < E && (sel[j] > best || (sel[j] == best && e < bestE))) {
                        best = sel[j];
                        bestE = e;
                    }
                }
                const float top = simd_max(best);
                const int winner = simd_min(best == top ? bestE : E);
                float s = 0.0f;
                for (int j = 0; j < EPL; j++) {
                    if (lane + 32 * j == winner) {
                        s = score[j];
                        sel[j] = -INFINITY;
                    }
                }
                picked[k] = simd_sum(s);
                if (lane == 0) {
                    indices[row * TOPK + k] = uint(winner);
                }
            }
            if (lane == 0) {
                float total = 0.0f;
                for (int k = 0; k < TOPK; k++) {
                    total += picked[k];
                }
                total += params[1];
                for (int k = 0; k < TOPK; k++) {
                    weights[row * TOPK + k] = picked[k] / total * params[0];
                }
            }
            """)

    /// Router of a row-exact step: the module's FP32 logits (per-row FP32
    /// products, as in prefill), then the fused select. Returns indices
    /// (rank order), weights and selection scores.
    static func route(_ router: Inkling2Router, _ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        select(router, logits: Inkling2RowExact.fp32Rows(x, router.weight))
    }

    /// The fused select over FP32 router logits `[R, E]`.
    static func select(_ router: Inkling2Router, logits: MLXArray) -> (MLXArray, MLXArray, MLXArray)
    {
        let (rows, experts, topK) = (logits.dim(0), router.weight.dim(0), router.topK)
        let key = LaunchKey(kernel: 9, dtype: .float32, a: experts, b: topK, c: rows)
        let launch = launches.launch(key) {
            selectKernel.prepare(
                template: [("E", experts), ("TOPK", topK)], grid: (32 * rows, 1, 1),
                threadGroup: (32, 1, 1),
                outputShapes: [[rows, topK], [rows, topK], [rows, experts]],
                outputDTypes: [.uint32, .float32, .float32])
        }
        let out = launch([
            logits, router.bias.asType(.float32), constants.routeParams(router.routeScale),
        ])
        return (out[0], out[1], out[2])
    }

    /// `Inkling2SparseMoE` through the grouped kernels: the module's FP32 router
    /// logits (per-row FP32 products), the fused select, then gate/up with the
    /// activation, down and the ordered combine.
    static func sparseMoE(_ moe: Inkling2SparseMoE, _ x: MLXArray) -> MLXArray? {
        guard active, let gateUp = weights(of: moe.experts.gateUp),
            let down = weights(of: moe.experts.down)
        else { return nil }
        let shape = x.shape
        let flat = x.reshaped(-1, shape.last!)
        let router = moe.router
        guard
            covers(
                rows: flat.dim(0), topK: router.topK, hidden: flat.dim(1), dtype: x.dtype,
                gateUp: gateUp, down: down)
        else { return nil }
        let (indices, routeWeights, selection) = route(router, flat)
        Inkling2Diagnostics.routerIndices?(indices)
        Inkling2Diagnostics.routerSelection?(selection)
        Inkling2Diagnostics.section?("router", [indices, routeWeights])
        return expertProduct(
            flat, indices: indices, weights: routeWeights, gateUp: gateUp, down: down,
            gateCap: moe.gateSoftcap, upCap: moe.upSoftcap
        )?.reshaped(shape)
    }
}
