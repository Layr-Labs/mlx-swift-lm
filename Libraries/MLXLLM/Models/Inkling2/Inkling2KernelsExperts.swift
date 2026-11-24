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

    /// The work item of a planned grouped GEMV: one (expert, first pair, count)
    /// row of `items`; `ink2_list` lists the expert's pairs in ascending order
    /// (`auto`: MLX binds a pair list of fewer than eight entries as `constant`).
    private static let plannedGroupPreamble = """
        const uint tid = thread_index_in_threadgroup;
        const uint nt = threads_per_threadgroup.x;
        const int expert = items[int(threadgroup_position_in_grid.y) * 3];
        if (expert < 0) {
            return;
        }
        const auto ink2_list = order + items[int(threadgroup_position_in_grid.y) * 3 + 1];
        const int pairs = items[int(threadgroup_position_in_grid.y) * 3 + 2];
        """

    /// One simdgroup's MMA product of a work item's tile with RB blocks of
    /// eight pair columns.
    private static func expertTileCall(_ rb: Int) -> String {
        "ink2_mma_tile<T, WT, FMT, K, SK, \(rb), FMT == 1 && TBLON, OPT, PF>(x, src, pairs, row0, srow0, ink2_row_stride<FMT, K>(), ink2_scale_stride<FMT, K>(), lane, ks, mine, 8, ink2_lut);"
    }

    /// The tiles' epilogue: the split-K partials of `buf` added in order, then
    /// gate/up rounded to T and the softcapped SwiGLU, or the down projection
    /// rounded to T.
    private static func expertTileEpilogue(up: Bool) -> String {
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
    }

    /// The MMA products of one work item and the epilogue: gate/up rounded to
    /// T as projection outputs, softcapped SwiGLU in FP32 rounded to T; or the
    /// down projection rounded to T.
    private static func expertSource(up: Bool, planned: Bool) -> String {
        let listType = planned ? "const device int*" : "threadgroup const ushort*"
        let tile = expertTileCall
        let epilogue = expertTileEpilogue(up: up)
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
    /// Gate/up rows per lane of the version-2 GEMV: 2 (gate and up in one lane)
    /// or 1 (adjacent lane groups); same bits. `DARKBLOOM_INKLING2_EXPERT_UP_ROWS`.
    nonisolated(unsafe) static var expertUpRowsPerLane =
        Int(ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_EXPERT_UP_ROWS"] ?? "") ?? 2
    /// Steps of 2 to 8 rows run the grouped GEMVs (each selected expert's
    /// weights read once for all its pairs; the one-row bits) instead of the
    /// tiles. `DARKBLOOM_INKLING2_EXPERT_GROUP=0` keeps the tiles.
    nonisolated(unsafe) static var expertGroupGemv =
        ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_EXPERT_GROUP"] != "0"
    static let expertGroupMaxRows = 8
    /// Down-projection rows per lane of the version-2 GEMV (1 or 2; same bits).
    /// `DARKBLOOM_INKLING2_EXPERT_DOWN_ROWS`.
    nonisolated(unsafe) static var expertDownRowsPerLane =
        Int(ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_EXPERT_DOWN_ROWS"] ?? "") ?? 1

    /// NVFP4 banks of the version-2 one-row GEMVs and of the grouped GEMVs
    /// decode each k-slot pair from its raw byte right before its two FMAs,
    /// so a lane keeps the raw words instead of 32 decoded weights and 32
    /// widened inputs per row: the same products in the same order (tests
    /// compare both forms). `DARKBLOOM_INKLING2_EXPERT_INORDER=0` decodes each
    /// superblock first.
    nonisolated(unsafe) static var expertInOrderDecode =
        ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_EXPERT_INORDER"] != "0"
    /// Grouped GEMVs of 2 to 8 rows take their work items from one `plan`
    /// launch instead of gathering the pairs in every threadgroup (same
    /// chains, same bits). `DARKBLOOM_INKLING2_EXPERT_GROUP_PLAN=0` keeps the
    /// gathering in each threadgroup.
    nonisolated(unsafe) static var expertGroupPlanned =
        ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_EXPERT_GROUP_PLAN"] != "0"
    /// Gate/up rows per lane of the grouped GEMVs: 1 (gate and up rows in
    /// adjacent lane groups) or 2 (both rows in one lane); same bits. The
    /// one-row GEMV keeps its own choice (`expertUpRowsPerLane`).
    /// `DARKBLOOM_INKLING2_EXPERT_GROUP_UP_ROWS`.
    nonisolated(unsafe) static var expertGroupUpRowsPerLane =
        Int(ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_EXPERT_GROUP_UP_ROWS"] ?? "")
        ?? 1
    /// Steps of 9 to 32 rows (a drafter verify window of several streams runs
    /// streams x window rows whose consecutive tokens share experts) run the
    /// mixed launches by default: planned grouped GEMVs that take each
    /// expert's pairs in blocks, with the work items of several pairs as MMA
    /// tiles in the same launch. Smaller steps keep the grouped GEMVs without
    /// blocks and larger ones the tiles. Every form runs the same chains per
    /// output: same bits. The five switches below choose the form.
    ///
    /// Pairs per block (1 to 4, to 8 with block form 1; 0: no blocks, the
    /// grouped GEMVs stop at 8 rows and the tiles take the rest): per block
    /// the lane reads its rows once more and runs the block's chains in the
    /// block form. `DARKBLOOM_INKLING2_EXPERT_GROUP_BLOCK`.
    nonisolated(unsafe) static var expertGroupBlock =
        Int(ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_EXPERT_GROUP_BLOCK"] ?? "")
        ?? 4
    /// How a block of the blocked grouped GEMVs runs NVFP4 weights: 1, every
    /// pair decoding its own weights in order (nothing decoded or widened
    /// kept in the lane); 2, each weight pair decoded once and the block's
    /// chains in turn (the block's input words kept in the lane); 3, a block of
    /// one pair the same way and a block of several pairs decode-first (the
    /// block's chains one after the other); 4 (default), a block of one pair
    /// the same way and a block of several pairs sharing each decoded weight
    /// pair with the inputs read from memory. `DARKBLOOM_INKLING2_EXPERT_GROUP_FORM`.
    nonisolated(unsafe) static var expertGroupBlockForm =
        Int(ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_EXPERT_GROUP_FORM"] ?? "")
        ?? 4
    /// Work items of this many pairs or more of the blocked grouped GEMVs run
    /// as MMA tiles inside the same launch (0: none): a tile costs the same
    /// for one to eight pairs, a GEMV grows with each pair.
    /// `DARKBLOOM_INKLING2_EXPERT_GROUP_TILE_AT`.
    nonisolated(unsafe) static var expertGroupTileAt =
        Int(ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_EXPERT_GROUP_TILE_AT"] ?? "")
        ?? 4
    /// Fewest rows of a step the blocked grouped GEMVs take: smaller steps
    /// keep the grouped GEMVs without blocks (measured as good or better up
    /// to 8 rows). `DARKBLOOM_INKLING2_EXPERT_GROUP_FROM`.
    nonisolated(unsafe) static var expertGroupBlockFrom =
        Int(ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_EXPERT_GROUP_FROM"] ?? "")
        ?? 9
    /// Most rows of a step the blocked grouped GEMVs take (the tiles take
    /// larger steps). `DARKBLOOM_INKLING2_EXPERT_GROUP_ROWS`.
    nonisolated(unsafe) static var expertGroupBlockRows =
        Int(ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_EXPERT_GROUP_ROWS"] ?? "")
        ?? 32

    /// Test hook of the red control: a knob parsed as `main` parses it, unchecked.
    static func expertGroupKnob(
        _ name: String, default fallback: Int,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        Int(environment[name] ?? "") ?? fallback
    }
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

            // Input loader: one superblock's 32 values of a stored row as FP32
            // (load32), or its raw words (two-byte T: eight values per uint4).
            template <typename T>
            struct ink2e_xdev {
                const device T* xr;
                inline void load32(int s, thread float* v) const { ink2e_inputs<T>(xr, s, v); }
                inline void raw32(int s, thread uint4* u) const {
                    const device uint4* p = (const device uint4*)(xr + s * 32);
                    #pragma unroll
                    for (int b = 0; b < 4; b++) {
                        u[b] = p[b];
                    }
                }
            };

            // Component Q of a uint4 (Q a constant).
            template <int Q>
            inline uint ink2e_word(uint4 v) {
                return Q == 0 ? v.x : Q == 1 ? v.y : Q == 2 ? v.z : v.w;
            }

            // Two two-byte values in one word widened to FP32, as ink2_load8 does.
            template <typename T>
            inline float2 ink2e_pair(uint w) {
                if constexpr (metal::is_same_v<T, half>) {
                    return float2(as_type<half2>(w));
                } else {
                    return float2(as_type<float>(w << 16), as_type<float>(w & 0xFFFF0000u));
                }
            }

            // The two E2M1 nibbles of byte J of t (elements 2J and 2J + 1 of the
            // word's eight) as FP32 times 2^-14: ink2e_e2m1_at's values. The two
            // multiplies place each nibble's magnitude at bits 9..11 and its sign
            // at bit 15 of its own 16-bit half (no carries: the shifted copies of
            // the byte do not overlap).
            template <int J>
            inline float2 ink2e_e2m1_byte(uint t) {
                const uint b = (t >> (8 * J)) & 0xFFu;
                const uint h = ((b * 0x00200200u) & 0x0E000E00u) | ((b * 0x01001000u) & 0x80008000u);
                return float2(as_type<half2>(h));
            }

            // K-slot pair (J, Q) of one NVFP4 superblock for R rows: elements
            // Q * 8 + 2J and Q * 8 + 2J + 1, each weight times its block scale
            // (ink2e_decode's value), the two FMAs per row in order.
            template <int J, int Q, typename T, typename WT, int R>
            inline void ink2e_inorder_pair(
                uint4 xq, thread const ink2e_sb<WT, 1>* cur, thread const float* sa,
                thread const float* sb, thread float* acc) {
                const float2 xv = ink2e_pair<T>(ink2e_word<J>(xq));
                #pragma unroll
                for (int r = 0; r < R; r++) {
                    const float2 w = ink2e_e2m1_byte<J>(ink2e_word<Q>(cur[r].q[0]))
                        * (Q < 2 ? sa[r] : sb[r]);
                    acc[r] = fma(w.x, xv.x, acc[r]);
                    acc[r] = fma(w.y, xv.y, acc[r]);
                }
            }

            // One NVFP4 superblock of R chains with the weights decoded in k-slot
            // order (j, then the four eight-element blocks, then the pair): the
            // FMAs of the decode-first form below, nothing decoded kept.
            template <typename T, typename WT, int R>
            inline void ink2e_inorder_step(
                thread const uint4* xr, thread const ink2e_sb<WT, 1>* cur, thread float* acc) {
                float sa[R];
                float sb[R];
                #pragma unroll
                for (int r = 0; r < R; r++) {
                    sa[r] = ink2_e4m3_shifted(cur[r].s & 0xFFu) * 4194304.0f;
                    sb[r] = ink2_e4m3_shifted(cur[r].s >> 8) * 4194304.0f;
                }
                ink2e_inorder_pair<0, 0, T, WT, R>(xr[0], cur, sa, sb, acc);
                ink2e_inorder_pair<0, 1, T, WT, R>(xr[1], cur, sa, sb, acc);
                ink2e_inorder_pair<0, 2, T, WT, R>(xr[2], cur, sa, sb, acc);
                ink2e_inorder_pair<0, 3, T, WT, R>(xr[3], cur, sa, sb, acc);
                ink2e_inorder_pair<1, 0, T, WT, R>(xr[0], cur, sa, sb, acc);
                ink2e_inorder_pair<1, 1, T, WT, R>(xr[1], cur, sa, sb, acc);
                ink2e_inorder_pair<1, 2, T, WT, R>(xr[2], cur, sa, sb, acc);
                ink2e_inorder_pair<1, 3, T, WT, R>(xr[3], cur, sa, sb, acc);
                ink2e_inorder_pair<2, 0, T, WT, R>(xr[0], cur, sa, sb, acc);
                ink2e_inorder_pair<2, 1, T, WT, R>(xr[1], cur, sa, sb, acc);
                ink2e_inorder_pair<2, 2, T, WT, R>(xr[2], cur, sa, sb, acc);
                ink2e_inorder_pair<2, 3, T, WT, R>(xr[3], cur, sa, sb, acc);
                ink2e_inorder_pair<3, 0, T, WT, R>(xr[0], cur, sa, sb, acc);
                ink2e_inorder_pair<3, 1, T, WT, R>(xr[1], cur, sa, sb, acc);
                ink2e_inorder_pair<3, 2, T, WT, R>(xr[2], cur, sa, sb, acc);
                ink2e_inorder_pair<3, 3, T, WT, R>(xr[3], cur, sa, sb, acc);
            }

            // K-slot pair (J, Q) of one NVFP4 superblock for R rows against NB
            // input rows (xr: four raw words per input row): each weight pair is
            // decoded once (ink2e_inorder_pair's value), then every (input row,
            // weight row) chain takes its two FMAs in order.
            template <int J, int Q, typename T, typename WT, int R, int NB>
            inline void ink2e_inorder_pairs(
                thread const uint4* xr, thread const ink2e_sb<WT, 1>* cur, thread const float* sa,
                thread const float* sb, thread float* acc) {
                float2 w[R];
                #pragma unroll
                for (int r = 0; r < R; r++) {
                    w[r] = ink2e_e2m1_byte<J>(ink2e_word<Q>(cur[r].q[0]))
                        * (Q < 2 ? sa[r] : sb[r]);
                }
                #pragma unroll
                for (int p = 0; p < NB; p++) {
                    const float2 xv = ink2e_pair<T>(ink2e_word<J>(xr[p * 4 + Q]));
                    #pragma unroll
                    for (int r = 0; r < R; r++) {
                        acc[p * R + r] = fma(w[r].x, xv.x, acc[p * R + r]);
                        acc[p * R + r] = fma(w[r].y, xv.y, acc[p * R + r]);
                    }
                }
            }

            // One NVFP4 superblock of R chains for each of NB input rows, the
            // weights decoded once in k-slot order: ink2e_inorder_step's FMAs for
            // every input row, the rows' chains taking turns.
            template <typename T, typename WT, int R, int NB>
            inline void ink2e_inorder_block(
                thread const uint4* xr, thread const ink2e_sb<WT, 1>* cur, thread float* acc) {
                float sa[R];
                float sb[R];
                #pragma unroll
                for (int r = 0; r < R; r++) {
                    sa[r] = ink2_e4m3_shifted(cur[r].s & 0xFFu) * 4194304.0f;
                    sb[r] = ink2_e4m3_shifted(cur[r].s >> 8) * 4194304.0f;
                }
                ink2e_inorder_pairs<0, 0, T, WT, R, NB>(xr, cur, sa, sb, acc);
                ink2e_inorder_pairs<0, 1, T, WT, R, NB>(xr, cur, sa, sb, acc);
                ink2e_inorder_pairs<0, 2, T, WT, R, NB>(xr, cur, sa, sb, acc);
                ink2e_inorder_pairs<0, 3, T, WT, R, NB>(xr, cur, sa, sb, acc);
                ink2e_inorder_pairs<1, 0, T, WT, R, NB>(xr, cur, sa, sb, acc);
                ink2e_inorder_pairs<1, 1, T, WT, R, NB>(xr, cur, sa, sb, acc);
                ink2e_inorder_pairs<1, 2, T, WT, R, NB>(xr, cur, sa, sb, acc);
                ink2e_inorder_pairs<1, 3, T, WT, R, NB>(xr, cur, sa, sb, acc);
                ink2e_inorder_pairs<2, 0, T, WT, R, NB>(xr, cur, sa, sb, acc);
                ink2e_inorder_pairs<2, 1, T, WT, R, NB>(xr, cur, sa, sb, acc);
                ink2e_inorder_pairs<2, 2, T, WT, R, NB>(xr, cur, sa, sb, acc);
                ink2e_inorder_pairs<2, 3, T, WT, R, NB>(xr, cur, sa, sb, acc);
                ink2e_inorder_pairs<3, 0, T, WT, R, NB>(xr, cur, sa, sb, acc);
                ink2e_inorder_pairs<3, 1, T, WT, R, NB>(xr, cur, sa, sb, acc);
                ink2e_inorder_pairs<3, 2, T, WT, R, NB>(xr, cur, sa, sb, acc);
                ink2e_inorder_pairs<3, 3, T, WT, R, NB>(xr, cur, sa, sb, acc);
            }

            // Chain `ks` of R consecutive stored rows (from `row0`) against one
            // input row: acc[r] gets ink2_row_chain's value for row r. The next
            // superblock's loads are issued before this one's FMAs. IO 1 (NVFP4
            // weights, two-byte inputs): the in-order decode; same FMAs.
            template <typename T, typename XL, typename WT, int FMT, int K, int SK, int R, int IO>
            inline void ink2e_chains(
                XL xl, const device WT* row0, const device uchar* srow0, int ks, thread float* acc) {
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
                    if constexpr (IO == 1 && FMT == 1 && sizeof(T) == 2) {
                        uint4 xr[4];
                        xl.raw32(s, xr);
                        ink2e_inorder_step<T, WT, R>(xr, cur, acc);
                    } else {
                        float xv[32];
                        xl.load32(s, xv);
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

            // K-slot pair (J, Q) of one NVFP4 superblock for R rows against NB
            // input rows read from memory (xo: each row's element offset of this
            // superblock): ink2e_inorder_pairs with no input word kept in the lane.
            template <int J, int Q, typename T, typename WT, int R, int NB>
            inline void ink2e_inorder_pairs_mem(
                const device T* x, thread const uint* xo, thread const ink2e_sb<WT, 1>* cur,
                thread const float* sa, thread const float* sb, thread float* acc) {
                float2 w[R];
                #pragma unroll
                for (int r = 0; r < R; r++) {
                    w[r] = ink2e_e2m1_byte<J>(ink2e_word<Q>(cur[r].q[0]))
                        * (Q < 2 ? sa[r] : sb[r]);
                }
                #pragma unroll
                for (int p = 0; p < NB; p++) {
                    const float2 xv =
                        ink2e_pair<T>(((const device uint*)(x + xo[p]))[Q * 4 + J]);
                    #pragma unroll
                    for (int r = 0; r < R; r++) {
                        acc[p * R + r] = fma(w[r].x, xv.x, acc[p * R + r]);
                        acc[p * R + r] = fma(w[r].y, xv.y, acc[p * R + r]);
                    }
                }
            }

            // One NVFP4 superblock of R chains for each of NB input rows read
            // from memory: ink2e_inorder_block's FMAs.
            template <typename T, typename WT, int R, int NB>
            inline void ink2e_memory_block(
                const device T* x, thread const uint* xo, thread const ink2e_sb<WT, 1>* cur,
                thread float* acc) {
                float sa[R];
                float sb[R];
                #pragma unroll
                for (int r = 0; r < R; r++) {
                    sa[r] = ink2_e4m3_shifted(cur[r].s & 0xFFu) * 4194304.0f;
                    sb[r] = ink2_e4m3_shifted(cur[r].s >> 8) * 4194304.0f;
                }
                ink2e_inorder_pairs_mem<0, 0, T, WT, R, NB>(x, xo, cur, sa, sb, acc);
                ink2e_inorder_pairs_mem<0, 1, T, WT, R, NB>(x, xo, cur, sa, sb, acc);
                ink2e_inorder_pairs_mem<0, 2, T, WT, R, NB>(x, xo, cur, sa, sb, acc);
                ink2e_inorder_pairs_mem<0, 3, T, WT, R, NB>(x, xo, cur, sa, sb, acc);
                ink2e_inorder_pairs_mem<1, 0, T, WT, R, NB>(x, xo, cur, sa, sb, acc);
                ink2e_inorder_pairs_mem<1, 1, T, WT, R, NB>(x, xo, cur, sa, sb, acc);
                ink2e_inorder_pairs_mem<1, 2, T, WT, R, NB>(x, xo, cur, sa, sb, acc);
                ink2e_inorder_pairs_mem<1, 3, T, WT, R, NB>(x, xo, cur, sa, sb, acc);
                ink2e_inorder_pairs_mem<2, 0, T, WT, R, NB>(x, xo, cur, sa, sb, acc);
                ink2e_inorder_pairs_mem<2, 1, T, WT, R, NB>(x, xo, cur, sa, sb, acc);
                ink2e_inorder_pairs_mem<2, 2, T, WT, R, NB>(x, xo, cur, sa, sb, acc);
                ink2e_inorder_pairs_mem<2, 3, T, WT, R, NB>(x, xo, cur, sa, sb, acc);
                ink2e_inorder_pairs_mem<3, 0, T, WT, R, NB>(x, xo, cur, sa, sb, acc);
                ink2e_inorder_pairs_mem<3, 1, T, WT, R, NB>(x, xo, cur, sa, sb, acc);
                ink2e_inorder_pairs_mem<3, 2, T, WT, R, NB>(x, xo, cur, sa, sb, acc);
                ink2e_inorder_pairs_mem<3, 3, T, WT, R, NB>(x, xo, cur, sa, sb, acc);
            }

            // One superblock of R chains for each of np input rows (of up to MAXP),
            // the weights decoded first: the decode-first FMAs of ink2e_group_chains.
            template <typename T, typename WT, int FMT, int K, int R, int MAXP, int DIV, typename PL>
            inline void ink2e_first_block(
                const device T* x, PL list, int np, int s, thread const ink2e_sb<WT, FMT>* cur,
                thread float* acc) {
                float wv[R][32];
                #pragma unroll
                for (int r = 0; r < R; r++) {
                    ink2e_decode<WT, FMT>(cur[r], wv[r]);
                }
                #pragma unroll
                for (int p = 0; p < MAXP; p++) {
                    if (p < np) {
                        float xv[32];
                        ink2e_inputs<T>(x + size_t(int(list[p]) / DIV) * K, s, xv);
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            #pragma unroll
                            for (int c = 0; c < 8; c++) {
                                const int k = (c / 2) * 8 + 2 * j + (c & 1);
                                #pragma unroll
                                for (int r = 0; r < R; r++) {
                                    acc[p * R + r] = fma(wv[r][k], xv[k], acc[p * R + r]);
                                }
                            }
                        }
                    }
                }
            }

            // Chain `ks` of R consecutive stored rows (from `row0`) against the
            // input rows of up to MAXP pairs (np used): every superblock's weights
            // are loaded and decoded once, then each pair runs its own chain in
            // the k-slot order, so acc[p][r] equals ink2_row_chain for that pair.
            // Pair p's input row is x + (list[p] / DIV) * K. IO 1 (NVFP4 weights,
            // two-byte inputs): each pair decodes the superblock again in k-slot
            // order instead (nothing decoded kept between pairs); same FMAs. IO 2
            // (the same banks and inputs, MAXP <= 4): each weight pair is decoded
            // once for the np pairs, whose chains take turns (np is the same in
            // every lane of a threadgroup); same FMAs per chain. IO 3: one pair
            // as IO 2, several pairs decode first. IO 4: one pair as IO 2, several
            // pairs share each decoded weight pair and read their inputs from
            // memory (no input word kept in the lane).
            template <typename T, typename WT, int FMT, int K, int SK, int R, int MAXP, int DIV, int IO, typename PL>
            inline void ink2e_group_chains(
                const device T* x, PL list, int np, const device WT* row0, const device uchar* srow0,
                int ks, thread float* acc) {
                #pragma unroll
                for (int p = 0; p < MAXP; p++) {
                    #pragma unroll
                    for (int r = 0; r < R; r++) {
                        acc[p * R + r] = 0.0f;
                    }
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
                    if constexpr (IO == 4 && FMT == 1 && sizeof(T) == 2) {
                        if (np == 1) {
                            uint4 xr[4];
                            ink2e_xdev<T>{x + size_t(int(list[0]) / DIV) * K}.raw32(s, xr);
                            ink2e_inorder_block<T, WT, R, 1>(xr, cur, acc);
                        } else {
                            // Pairs past np repeat the last one and are not run.
                            uint xo[MAXP];
                            #pragma unroll
                            for (int p = 0; p < MAXP; p++) {
                                xo[p] = uint(int(list[min(p, np - 1)]) / DIV) * uint(K)
                                    + uint(s) * 32u;
                            }
                            if constexpr (MAXP >= 2) {
                                if (np == 2) {
                                    ink2e_memory_block<T, WT, R, 2>(x, xo, cur, acc);
                                }
                            }
                            if constexpr (MAXP >= 3) {
                                if (np == 3) {
                                    ink2e_memory_block<T, WT, R, 3>(x, xo, cur, acc);
                                }
                            }
                            if constexpr (MAXP >= 4) {
                                if (np == 4) {
                                    ink2e_memory_block<T, WT, R, 4>(x, xo, cur, acc);
                                }
                            }
                        }
                    } else if constexpr (IO == 3 && FMT == 1 && sizeof(T) == 2) {
                        if (np == 1) {
                            uint4 xr[4];
                            ink2e_xdev<T>{x + size_t(int(list[0]) / DIV) * K}.raw32(s, xr);
                            ink2e_inorder_block<T, WT, R, 1>(xr, cur, acc);
                        } else {
                            ink2e_first_block<T, WT, FMT, K, R, MAXP, DIV>(x, list, np, s, cur, acc);
                        }
                    } else if constexpr (IO == 2 && FMT == 1 && sizeof(T) == 2) {
                        uint4 xr[MAXP * 4];
                        #pragma unroll
                        for (int p = 0; p < MAXP; p++) {
                            if (p < np) {
                                ink2e_xdev<T>{x + size_t(int(list[p]) / DIV) * K}.raw32(
                                    s, xr + p * 4);
                            }
                        }
                        if (np == 1) {
                            ink2e_inorder_block<T, WT, R, 1>(xr, cur, acc);
                        }
                        if constexpr (MAXP >= 2) {
                            if (np == 2) {
                                ink2e_inorder_block<T, WT, R, 2>(xr, cur, acc);
                            }
                        }
                        if constexpr (MAXP >= 3) {
                            if (np == 3) {
                                ink2e_inorder_block<T, WT, R, 3>(xr, cur, acc);
                            }
                        }
                        if constexpr (MAXP >= 4) {
                            if (np == 4) {
                                ink2e_inorder_block<T, WT, R, 4>(xr, cur, acc);
                            }
                        }
                    } else if constexpr (IO == 1 && FMT == 1 && sizeof(T) == 2) {
                        #pragma unroll
                        for (int p = 0; p < MAXP; p++) {
                            if (p < np) {
                                uint4 xr[4];
                                ink2e_xdev<T>{x + size_t(int(list[p]) / DIV) * K}.raw32(s, xr);
                                ink2e_inorder_step<T, WT, R>(xr, cur, acc + p * R);
                            }
                        }
                    } else {
                        float wv[R][32];
                        #pragma unroll
                        for (int r = 0; r < R; r++) {
                            ink2e_decode<WT, FMT>(cur[r], wv[r]);
                        }
                        #pragma unroll
                        for (int p = 0; p < MAXP; p++) {
                            if (p < np) {
                                float xv[32];
                                ink2e_inputs<T>(x + size_t(int(list[p]) / DIV) * K, s, xv);
                                #pragma unroll
                                for (int j = 0; j < 4; j++) {
                                    #pragma unroll
                                    for (int c = 0; c < 8; c++) {
                                        const int k = (c / 2) * 8 + 2 * j + (c & 1);
                                        #pragma unroll
                                        for (int r = 0; r < R; r++) {
                                            acc[p * R + r] = fma(wv[r][k], xv[k], acc[p * R + r]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int r = 0; r < R; r++) {
                        cur[r] = nxt[r];
                    }
                }
            }

            // The tiles' input-row source for a work item's pair list in any
            // address space (pair p reads row list[p] / DIV).
            template <int DIV, typename PS>
            inline ink2_rows_pairs<PS, DIV> ink2e_pairs_source(PS p) {
                return ink2_rows_pairs<PS, DIV>{p};
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
            // RU 2: one lane group per (pair, i) runs the gate and up rows; RU 1:
            // the up row's group follows the gate row's (SK lanes later).
            const int g0 = int(thread_position_in_grid.x);
            const int sub = g0 % SK;
            const int hsel = RU == 1 ? (g0 / SK) % 2 : 0;
            const int gid = g0 / (SK * (3 - RU));
            const int pairs = idx_shape[0] * TOPK;
            const bool valid = gid < pairs * (N / 2);
            const int pair = valid ? gid / (N / 2) : 0;
            const int i = valid ? gid % (N / 2) : 0;
            const int expert = int(idx[pair]);
            const size_t r0 = size_t(expert) * N + size_t(2 * i + hsel);
            const device T* xr = x + size_t(pair / TOPK) * K;
            const device WT* row0 = w + r0 * ink2_row_stride<FMT, K>();
            const device uchar* srow0 = sc + (FMT == 0 ? size_t(0) : r0 * ink2_scale_stride<FMT, K>());
            float acc[RU];
            ink2e_chains<T, ink2e_xdev<T>, WT, FMT, K, SK, RU, IO>(
                ink2e_xdev<T>{xr}, row0, srow0, sub, acc);
            float g = ink2e_chain_total<SK>(acc[0]);
            float u = RU == 2 ? ink2e_chain_total<SK>(acc[RU - 1]) : simd_shuffle_down(g, ushort(SK));
            if (valid && sub == 0 && hsel == 0) {
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
            const device WT* row0 = w + r0 * ink2_row_stride<FMT, K>();
            const device uchar* srow0 = sc + (FMT == 0 ? size_t(0) : r0 * ink2_scale_stride<FMT, K>());
            float acc[RD];
            ink2e_chains<T, ink2e_xdev<T>, WT, FMT, K, SK, RD, IO>(
                ink2e_xdev<T>{x + size_t(pair) * K}, row0, srow0, sub, acc);
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

    /// Steps of 2 to MAXP rows (grouped GEMVs): each threadgroup takes one
    /// selected expert (`expertPreamble`: the work item's expert and its pairs)
    /// and a block of outputs; each lane loads and decodes its rows' weights
    /// once per superblock and runs one chain per pair (the one-row bits).
    /// Gate/up: RU 2 runs the gate and up rows in one lane, RU 1 in adjacent
    /// lane groups; 256 / SK lanes groups per threadgroup.
    private static let gateUpGroupBody = """

        const int sub = int(tid) % SK;
        const int local = int(tid) / SK;
        const int hsel = RU == 1 ? local % 2 : 0;
        const int perTile = int(nt) / SK / (3 - RU);
        const int i = int(threadgroup_position_in_grid.x) * perTile + local / (3 - RU);
        const bool valid = i < N / 2;
        const size_t r0 = size_t(expert) * N + size_t(2 * (valid ? i : 0) + hsel);
        float acc[MAXP * RU];
        ink2e_group_chains<T, WT, FMT, K, SK, RU, MAXP, TOPK, IO>(
            x, ink2_list, pairs, w + r0 * ink2_row_stride<FMT, K>(),
            sc + (FMT == 0 ? size_t(0) : r0 * ink2_scale_stride<FMT, K>()), sub, acc);
        const float inv = FMT == 1 ? gs[expert] / 2688.0f : 1.0f;
        #pragma unroll
        for (int p = 0; p < MAXP; p++) {
            float g = ink2e_chain_total<SK>(acc[p * RU]);
            float u = RU == 2 ? ink2e_chain_total<SK>(acc[p * RU + RU - 1])
                : simd_shuffle_down(g, ushort(SK));
            if (p < pairs && valid && sub == 0 && hsel == 0) {
                g = float(static_cast<T>(g * inv));
                u = float(static_cast<T>(u * inv));
                const float gated = (metal::precise::tanh(g / caps[0]) * caps[0])
                    * (1.0f / (1.0f + metal::precise::exp(-g)));
                const float a = gated * (metal::precise::tanh(u / caps[1]) * caps[1]);
                out[size_t(ink2_list[p]) * (N / 2) + i] = static_cast<T>(a);
            }
        }
        """

    private static let gateUpGroupKernel = MLXFast.metalKernel(
        name: "inkling2_experts_gate_up_group",
        inputNames: ["x", "idx", "w", "sc", "gs", "caps"],
        outputNames: ["out"],
        source: expertPreamble + gateUpGroupBody,
        header: expertsGemvHeader)

    /// The same grouped GEMV with its work item from the `plan` launch.
    private static let plannedGateUpGroupKernel = MLXFast.metalKernel(
        name: "inkling2_experts_gate_up_group_planned",
        inputNames: ["x", "items", "order", "w", "sc", "gs", "caps"],
        outputNames: ["out"],
        source: plannedGroupPreamble + gateUpGroupBody,
        header: expertsGemvHeader)

    /// The grouped gate/up GEMV with an expert's pairs taken in blocks of up to
    /// MAXP (one to four): every block reads the lane's rows again (cache
    /// resident after the first block), so a step of any row count keeps MAXP
    /// accumulators per row, and NVFP4 weights are decoded once per block.
    private static let gateUpGroupBlocksBody = """

        const int sub = int(tid) % SK;
        const int local = int(tid) / SK;
        const int hsel = RU == 1 ? local % 2 : 0;
        const int perTile = int(nt) / SK / (3 - RU);
        const int i = int(threadgroup_position_in_grid.x) * perTile + local / (3 - RU);
        const bool valid = i < N / 2;
        const size_t r0 = size_t(expert) * N + size_t(2 * (valid ? i : 0) + hsel);
        const float inv = FMT == 1 ? gs[expert] / 2688.0f : 1.0f;
        for (int first = 0; first < pairs; first += MAXP) {
            const int nb = min(MAXP, pairs - first);
            float acc[MAXP * RU];
            ink2e_group_chains<T, WT, FMT, K, SK, RU, MAXP, TOPK, IO>(
                x, ink2_list + first, nb, w + r0 * ink2_row_stride<FMT, K>(),
                sc + (FMT == 0 ? size_t(0) : r0 * ink2_scale_stride<FMT, K>()), sub, acc);
            #pragma unroll
            for (int p = 0; p < MAXP; p++) {
                float g = ink2e_chain_total<SK>(acc[p * RU]);
                float u = RU == 2 ? ink2e_chain_total<SK>(acc[p * RU + RU - 1])
                    : simd_shuffle_down(g, ushort(SK));
                if (p < nb && valid && sub == 0 && hsel == 0) {
                    g = float(static_cast<T>(g * inv));
                    u = float(static_cast<T>(u * inv));
                    const float gated = (metal::precise::tanh(g / caps[0]) * caps[0])
                        * (1.0f / (1.0f + metal::precise::exp(-g)));
                    const float a = gated * (metal::precise::tanh(u / caps[1]) * caps[1]);
                    out[size_t(ink2_list[first + p]) * (N / 2) + i] = static_cast<T>(a);
                }
            }
        }
        """

    private static let plannedGateUpGroupBlocksKernel = MLXFast.metalKernel(
        name: "inkling2_experts_gate_up_group_blocks",
        inputNames: ["x", "items", "order", "w", "sc", "gs", "caps"],
        outputNames: ["out"],
        source: plannedGroupPreamble + gateUpGroupBlocksBody,
        header: expertsGemvHeader)

    /// The grouped down GEMV with an expert's pairs in blocks of up to MAXP.
    private static let downGroupBlocksBody = """

        const int sub = int(tid) % SK;
        const int n = int(threadgroup_position_in_grid.x) * (int(nt) / SK) + int(tid) / SK;
        const bool valid = n < N;
        const size_t r0 = size_t(expert) * N + size_t(valid ? n : 0);
        const float inv = FMT == 1 ? gs[expert] / 2688.0f : 1.0f;
        for (int first = 0; first < pairs; first += MAXP) {
            const int nb = min(MAXP, pairs - first);
            float acc[MAXP];
            ink2e_group_chains<T, WT, FMT, K, SK, 1, MAXP, 1, IO>(
                x, ink2_list + first, nb, w + r0 * ink2_row_stride<FMT, K>(),
                sc + (FMT == 0 ? size_t(0) : r0 * ink2_scale_stride<FMT, K>()), sub, acc);
            #pragma unroll
            for (int p = 0; p < MAXP; p++) {
                const float v = ink2e_chain_total<SK>(acc[p]);
                if (p < nb && valid && sub == 0) {
                    out[size_t(ink2_list[first + p]) * N + n] = static_cast<T>(v * inv);
                }
            }
        }
        """

    private static let plannedDownGroupBlocksKernel = MLXFast.metalKernel(
        name: "inkling2_experts_down_group_blocks",
        inputNames: ["x", "items", "order", "w", "sc", "gs"],
        outputNames: ["out"],
        source: plannedGroupPreamble + downGroupBlocksBody,
        header: expertsGemvHeader)

    /// Work items of TILEAT pairs or more of the blocked grouped GEMVs run as
    /// MMA tiles instead (eight columns per MMA whatever the pair count): TPT
    /// tiles of eight outputs per threadgroup, one after the other, in the
    /// GEMV's grid. The tiles' bits are the GEMVs' (same chains per output).
    /// Threadgroups of eight simdgroups (SK 8).
    private static func groupTileBranch(up: Bool) -> String {
        """

        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        threadgroup float buf[SK * NSG * 64 * MAXRB];
        threadgroup float2 ink2_lut[FMT == 1 && TBLON ? 256 : 1];
        if (pairs >= TILEAT) {
            if (FMT == 1 && TBLON) {
                ink2_fill_e2m1_table(ink2_lut, tid, nt);
            }
            const int ks = int(sg) % SK;
            threadgroup float* mine = buf + ks * NSG * 64 * MAXRB;
            const float inv = FMT == 1 ? gs[expert] / 2688.0f : 1.0f;
            const auto src = ink2e_pairs_source<\(up ? "TOPK" : "1")>(ink2_list);
            const int rb = pairs <= 8 ? 1 : pairs <= 16 ? 2 : pairs <= 32 ? 4 : 8;
            const int cols = rb * 8;
            for (int tt = 0; tt < TPT; tt++) {
                const int tile = int(threadgroup_position_in_grid.x) * TPT + tt;
                if (tile * 8 >= N) {
                    break;
                }
                const int nb = min(tile * 8, N - 8);
                const size_t r0 = size_t(expert) * N + size_t(nb);
                const device WT* row0 = w + r0 * ink2_row_stride<FMT, K>();
                const device uchar* srow0 =
                    sc + (FMT == 0 ? size_t(0) : r0 * ink2_scale_stride<FMT, K>());
                if (rb == 1) {
                    \(expertTileCall(1))
                } else if (rb == 2) {
                    \(expertTileCall(2))
                } else if (rb == 4) {
                    \(expertTileCall(4))
                } else {
                    \(expertTileCall(8))
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                \(expertTileEpilogue(up: up))
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            return;
        }
        """
    }

    private static let plannedGateUpGroupMixedKernel = MLXFast.metalKernel(
        name: "inkling2_experts_gate_up_group_mixed",
        inputNames: ["x", "items", "order", "w", "sc", "gs", "caps"],
        outputNames: ["out"],
        source: plannedGroupPreamble + groupTileBranch(up: true) + gateUpGroupBlocksBody,
        header: expertsGemvHeader)

    private static let plannedDownGroupMixedKernel = MLXFast.metalKernel(
        name: "inkling2_experts_down_group_mixed",
        inputNames: ["x", "items", "order", "w", "sc", "gs"],
        outputNames: ["out"],
        source: plannedGroupPreamble + groupTileBranch(up: false) + downGroupBlocksBody,
        header: expertsGemvHeader)

    /// Down projection of 2 to MAXP rows: one output row per lane group.
    private static let downGroupBody = """

        const int sub = int(tid) % SK;
        const int n = int(threadgroup_position_in_grid.x) * (int(nt) / SK) + int(tid) / SK;
        const bool valid = n < N;
        const size_t r0 = size_t(expert) * N + size_t(valid ? n : 0);
        float acc[MAXP];
        ink2e_group_chains<T, WT, FMT, K, SK, 1, MAXP, 1, IO>(
            x, ink2_list, pairs, w + r0 * ink2_row_stride<FMT, K>(),
            sc + (FMT == 0 ? size_t(0) : r0 * ink2_scale_stride<FMT, K>()), sub, acc);
        const float inv = FMT == 1 ? gs[expert] / 2688.0f : 1.0f;
        #pragma unroll
        for (int p = 0; p < MAXP; p++) {
            const float v = ink2e_chain_total<SK>(acc[p]);
            if (p < pairs && valid && sub == 0) {
                out[size_t(ink2_list[p]) * N + n] = static_cast<T>(v * inv);
            }
        }
        """

    private static let downGroupKernel = MLXFast.metalKernel(
        name: "inkling2_experts_down_group",
        inputNames: ["x", "idx", "w", "sc", "gs"],
        outputNames: ["out"],
        source: expertPreamble + downGroupBody,
        header: expertsGemvHeader)

    /// The same grouped GEMV with its work item from the `plan` launch.
    private static let plannedDownGroupKernel = MLXFast.metalKernel(
        name: "inkling2_experts_down_group_planned",
        inputNames: ["x", "items", "order", "w", "sc", "gs"],
        outputNames: ["out"],
        source: plannedGroupPreamble + downGroupBody,
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
        // Launch keys carry every template flag (table decode, narrow operands,
        // in-order decode).
        let flags =
            (nvfp4Table ? 100 : 0) + (narrowOperands ? 1000 : 0)
            + (expertInOrderDecode ? 100_000 : 0)
        // The in-order NVFP4 decode reads the inputs as raw words: two-byte types.
        func inOrder(_ bank: Inkling2ExpertWeights) -> Bool {
            expertInOrderDecode && bank.format == .nvfp4
                && (x.dtype == .bfloat16 || x.dtype == .float16)
        }

        let blockForm = [1, 3, 4].contains(expertGroupBlockForm) ? expertGroupBlockForm : 2

        func inputs(_ bank: Inkling2ExpertWeights) -> [MLXArray] {
            [bank.weight, bank.scales ?? constants.byte, bank.globalScale ?? constants.one]
        }
        func template(_ bank: Inkling2ExpertWeights, n: Int, k: Int, pf: Int, shared: Bool = false)
            -> [(String, any KernelTemplateArg)]
        {
            [
                ("PF", pf), ("MAXRB", maxRB(rows)),
                ("T", x.dtype), ("WT", bank.weight.dtype), ("FMT", bank.formatCode), ("N", n),
                ("K", k), ("E", experts), ("TOPK", topK), ("NSG", nsg), ("SK", expertSK(k)),
                ("TBLON", nvfp4Table), ("IO", inOrder(bank) ? (shared ? blockForm : 1) : 0),
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
        let ru = expertUpRowsPerLane == 1 && 2 * expertSK(hidden) <= 32 ? 1 : 2
        if gemvUp, expertGemvVersion == 2 {
            let upKey = LaunchKey(
                kernel: 50 + flags, dtype: x.dtype, weightType: gateUp.weight.dtype,
                a: gateUp.formatCode + 10 * ru, b: gateUp.outputs, c: hidden, d: experts,
                e: topK, f: rows)
            let upLaunch = launches.launch(upKey) {
                gateUpGemv2Kernel.prepare(
                    template: template(gateUp, n: gateUp.outputs, k: hidden, pf: 1) + [
                        ("RU", ru)
                    ],
                    grid: (
                        (pairs * intermediate * expertSK(hidden) * (3 - ru) + 255) / 256 * 256, 1, 1
                    ),
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

        // Steps of 2 to 8 rows: grouped GEMVs (same bits as the tiles). Blocks of pairs
        // (planned items only) lift their row limit: the mixed launches of 9 to 32 rows.
        let block =
            expertGroupPlanned && (1 ... (blockForm == 1 ? 8 : 4)).contains(expertGroupBlock)
                && rows >= expertGroupBlockFrom
            ? expertGroupBlock : 0
        let groupRows =
            block > 0 ? max(expertGroupMaxRows, expertGroupBlockRows) : expertGroupMaxRows
        if expertGroupGemv, rows >= 2, rows <= groupRows, block > 0 || pairs <= pairsPerChunk,
            upOperand != .float32, downOperand != .float32, gemvSite("up"), gemvSite("down")
        {
            let gru = expertGroupUpRowsPerLane == 1 && 2 * expertSK(hidden) <= 32 ? 1 : 2
            let upPerTile = 256 / expertSK(hidden) / (3 - gru)
            let downPerTile = 256 / expertSK(intermediate)
            // Work items: gathered in every threadgroup, or once by the `plan` launch.
            let groupPlanned = expertGroupPlanned
            // Items of `tileAt` pairs or more as MMA tiles (threadgroups of eight simdgroups).
            let tileAt =
                block > 0 && expertGroupTileAt >= 2 && expertSK(hidden) == 8
                    && expertSK(intermediate) == 8 ? min(expertGroupTileAt, 99) : 0
            let maxPairs =
                block > 0 ? min(block, rows, tileAt > 0 ? tileAt - 1 : rows) : rows
            var groupRouting: [MLXArray] = [idx]
            if groupPlanned {
                let planThreads = min(1024, (experts * max(1, 1024 / experts) + 31) / 32 * 32)
                let planKey = LaunchKey(kernel: 3, dtype: .uint32, a: experts, b: topK, c: rows)
                let planLaunch = launches.launch(planKey) {
                    planKernel.prepare(
                        template: [("E", experts), ("TOPK", topK), ("ITEMS", slots)],
                        grid: (planThreads, 1, 1), threadGroup: (planThreads, 1, 1),
                        outputShapes: [[slots * 3], [pairs]], outputDTypes: [.int32, .int32])
                }
                groupRouting = planLaunch([idx])
            }
            let upKey = LaunchKey(
                kernel: 54 + flags, dtype: x.dtype, weightType: gateUp.weight.dtype,
                a: gateUp.formatCode + 10 * gru + (groupPlanned ? 100 : 0) + 1000 * block
                    + 10_000 * tileAt + (block > 0 ? 1_000_000 * blockForm : 0),
                b: gateUp.outputs, c: hidden, d: experts, e: topK, f: rows)
            let upLaunch = launches.launch(upKey) {
                (tileAt > 0
                    ? plannedGateUpGroupMixedKernel
                    : block > 0
                        ? plannedGateUpGroupBlocksKernel
                        : groupPlanned ? plannedGateUpGroupKernel : gateUpGroupKernel).prepare(
                        template: template(
                            gateUp, n: gateUp.outputs, k: hidden, pf: 1, shared: block > 0)
                            + [("RU", gru), ("MAXP", maxPairs)]
                            + (tileAt > 0 ? [("TILEAT", tileAt), ("TPT", upPerTile / 4)] : []),
                        grid: ((intermediate + upPerTile - 1) / upPerTile * 256, slots, 1),
                        threadGroup: (256, 1, 1),
                        outputShapes: [[pairs, intermediate]], outputDTypes: [x.dtype])
            }
            let activated = upLaunch(
                [x] + groupRouting + inputs(gateUp) + [constants.caps(gateCap, upCap)])[0]
            let downKey = LaunchKey(
                kernel: 55 + flags, dtype: x.dtype, weightType: down.weight.dtype,
                a: down.formatCode + (groupPlanned ? 100 : 0) + 1000 * block + 10_000 * tileAt
                    + (block > 0 ? 1_000_000 * blockForm : 0),
                b: hidden, c: intermediate, d: experts, e: topK, f: rows)
            let downLaunch = launches.launch(downKey) {
                (tileAt > 0
                    ? plannedDownGroupMixedKernel
                    : block > 0
                        ? plannedDownGroupBlocksKernel
                        : groupPlanned ? plannedDownGroupKernel : downGroupKernel).prepare(
                        template: template(
                            down, n: hidden, k: intermediate, pf: 1, shared: block > 0)
                            + [("MAXP", maxPairs)]
                            + (tileAt > 0 ? [("TILEAT", tileAt), ("TPT", downPerTile / 8)] : []),
                        grid: ((hidden + downPerTile - 1) / downPerTile * 256, slots, 1),
                        threadGroup: (256, 1, 1),
                        outputShapes: [[pairs, hidden]], outputDTypes: [x.dtype])
            }
            return downLaunch([activated] + groupRouting + inputs(down))[0]
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
    /// arrival counter is slot SLOT of the router's workspace on this stream;
    /// the last threadgroup clears both slots.
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
            threadgroup float ink2_sel[RANK ? E : 1];
            threadgroup float ink2_score[RANK ? E : 1];
            threadgroup float ink2_pick[TOPK];
            threadgroup uint ink2_top[TOPK];
            const uint tid = thread_index_in_threadgroup;
            const uint grp = threadgroup_position_in_grid.x;
            const device uint4* wsrc = (const device uint4*)(w + size_t(grp) * OPG * K);
            // Every load of this thread is issued before any store.
            constexpr int WPT = (OPG * RQ + NT - 1) / NT;
            constexpr int XPT = (XQ + NT - 1) / NT;
            uint4 wl[WPT];
            uint4 xl[XPT];
            #pragma unroll
            for (int j = 0; j < WPT; j++) {
                const uint i = tid + uint(j * NT);
                wl[j] = i < uint(OPG * RQ) ? wsrc[i] : uint4(0);
            }
            #pragma unroll
            for (int j = 0; j < XPT; j++) {
                const uint i = tid + uint(j * NT);
                xl[j] = i < uint(XQ) ? ((const device uint4*)x)[i] : uint4(0);
            }
            const float myBias = tid < uint(E) ? bias[tid] : 0.0f;
            #pragma unroll
            for (int j = 0; j < WPT; j++) {
                const uint i = tid + uint(j * NT);
                if (i < uint(OPG * RQ)) {
                    ink2_w[i] = wl[j];
                }
            }
            #pragma unroll
            for (int j = 0; j < XPT; j++) {
                const uint i = tid + uint(j * NT);
                if (i < uint(XQ)) {
                    ink2_x[i] = xl[j];
                }
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
            device atomic_uint* both = reinterpret_cast<device atomic_uint*>(counters);
            if (tid == 0) {
                atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
                ink2_arrival = atomic_fetch_add_explicit(&both[SLOT], 1u, memory_order_relaxed);
                atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            if (ink2_arrival != uint(G - 1)) {
                return;
            }
            if (tid == 0) {
                // Every arrival is counted: zero for the next launch on this
                // workspace (and the other slot, after a launch that never ended).
                atomic_store_explicit(&both[0], 0u, memory_order_relaxed);
                atomic_store_explicit(&both[1], 0u, memory_order_relaxed);
            }
            if (RANK) {
                // Rank select: thread e's rank among (selection desc, expert asc)
                // is its pick position: selectKernel's picks, scores and weights
                // (one thread adds the picked scores in pick order).
                // NT >= E (host check): thread e owns expert e; one round of loads.
                if (tid < uint(E)) {
                    const float x = atomic_load_explicit(&lg[tid], memory_order_relaxed);
                    const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
                    const float score = x < 0.0f ? y : 1.0f - y;
                    const float sel = score + myBias;
                    selection[tid] = sel;
                    ink2_sel[tid] = metal::isnan(sel) ? -INFINITY : sel;
                    ink2_score[tid] = score;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < uint(E)) {
                    const float v = ink2_sel[tid];
                    int rank = 0;
                    for (int j = 0; j < E; j++) {
                        const float u = ink2_sel[j];
                        rank += (u > v || (u == v && j < int(tid))) ? 1 : 0;
                    }
                    if (rank < TOPK) {
                        ink2_top[rank] = tid;
                        ink2_pick[rank] = ink2_score[tid];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) {
                    float total = 0.0f;
                    for (int k = 0; k < TOPK; k++) {
                        total += ink2_pick[k];
                    }
                    total += params[1];
                    for (int k = 0; k < TOPK; k++) {
                        indices[k] = ink2_top[k];
                        weights[k] = ink2_pick[k] / total * params[0];
                    }
                }
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
    /// Router outputs per threadgroup of the fused router (2 or 4; same bits).
    /// `DARKBLOOM_INKLING2_ROUTER_OPG`.
    nonisolated(unsafe) static var fusedRouterOutputsPerGroup =
        Int(ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_ROUTER_OPG"] ?? "") ?? 2
    /// The fused router's select: 1 a rank select over the threadgroup, 0 the
    /// select kernel's simdgroup rounds; same results.
    /// `DARKBLOOM_INKLING2_ROUTER_RANK`.
    nonisolated(unsafe) static var fusedRouterRank =
        Int(ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_ROUTER_RANK"] ?? "") ?? 1

    /// Arrival counters of the fused router, one workspace per (router module,
    /// MLX stream), passed as a mutable input: MLX orders the launches that
    /// share it on one stream (a barrier within a command buffer, a fence
    /// across command buffers), and launches on other streams (other engines)
    /// use other workspaces, so no two launches ever count on the same words at
    /// once. Two slots alternate per launch; the last threadgroup clears both,
    /// so a launch that never finished is cleared by the next one that does.
    /// No fill dispatch per launch.
    private final class RouterWorkspaces: @unchecked Sendable {
        private struct Key: Hashable {
            let router: ObjectIdentifier
            let stream: ObjectIdentifier
        }
        private struct Entry {
            let counters: MLXArray
            var launches: UInt64
            // Held so the stream's identity is not reused while its entry lives.
            let stream: MLX.Stream
        }
        private let lock = NSLock()
        private var entries: [Key: Entry] = [:]

        func next(_ router: Inkling2Router, stream: MLX.Stream) -> (counters: MLXArray, slot: Int) {
            let key = Key(router: ObjectIdentifier(router), stream: ObjectIdentifier(stream))
            let (counters, slot, fresh): (MLXArray, Int, Bool) = lock.withLock {
                // Bounded: a workspace in use stays alive in its graph.
                if entries.count > 4096, entries[key] == nil { entries.removeAll() }
                if var entry = entries[key] {
                    entry.launches &+= 1
                    entries[key] = entry
                    return (entry.counters, Int(entry.launches & 1), false)
                }
                let c = MLXArray([UInt32](repeating: 0, count: 8))
                entries[key] = Entry(counters: c, launches: 1, stream: stream)
                return (c, 1, true)
            }
            if fresh { eval(counters) }
            return (counters, slot)
        }
    }

    private static let routerWorkspaces = RouterWorkspaces()

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
            experts >= topK, experts >= 32, experts <= 256, bytes <= 28 * 1024
        else { return nil }
        let threads = max(128, (experts + 31) / 32 * 32)
        let (counters, slot) = routerWorkspaces.next(router, stream: MLX.Stream.defaultStream)
        let rank = fusedRouterRank != 0
        var key = LaunchKey(
            kernel: 52, dtype: x.dtype, weightType: w.dtype, a: experts, b: topK, c: k)
        key.d = (rank ? 1 : 0) + 2 * opg + 100 * slot
        let launch = launches.launch(key) {
            routerFusedKernel.prepare(
                template: [
                    ("T", x.dtype), ("WT", w.dtype), ("K", k), ("SK", sk), ("E", experts),
                    ("TOPK", topK), ("OPG", opg), ("NT", threads), ("RANK", rank),
                    ("SLOT", slot),
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
