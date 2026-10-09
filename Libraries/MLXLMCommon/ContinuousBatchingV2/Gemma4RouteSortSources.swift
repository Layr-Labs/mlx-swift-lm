// Copyright © 2026 Eigen Labs.
// Final Gemma challenge27c821c4 integer route-sort kernels. Bounds are supplied
// only by the score-derived routing producer, never inferred from array shape.
enum Gemma4RouteSortSources {
    private static let routeCsortPrefillBlock = 256
    private static let routeCsortPrefillWidth = 256

    static let histogram = """
        constexpr uint BLOCK = \(routeCsortPrefillBlock);
        constexpr uint WIDTH = \(routeCsortPrefillWidth);
        uint b = threadgroup_position_in_grid.x;
        uint k = thread_position_in_threadgroup.x;
        uint n = keys_shape[0];
        threadgroup atomic_uint tg_count[WIDTH];
        atomic_store_explicit(&tg_count[k], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint idx = b * BLOCK + k;
        if (idx < n) {
            atomic_fetch_add_explicit(
                &tg_count[keys[idx]], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Integer adds commute, so the table is identical for every
        // interleaving the hardware picks.
        block_hist[b * WIDTH + k] =
            atomic_load_explicit(&tg_count[k], memory_order_relaxed);
        """

    static let scan = """
        constexpr uint WIDTH = \(routeCsortPrefillWidth);
        uint e = thread_position_in_threadgroup.x;
        uint simd_id = e / 32;
        uint lane = e % 32;
        uint nblocks = (uint)block_hist_shape[0];
        uint total = 0u;
        // Admission proves every key is below NE, so columns at or above it are
        // zero in every block and cannot contribute to the total. Skipping the
        // accumulation retires whole SIMD groups at once when the counter table
        // is wider than the model's expert count.
        if (e < (uint)NE) {
            for (uint b = 0; b < nblocks; ++b) {
                total += block_hist[b * WIDTH + e];
            }
        }
        // Global bin base: exclusive prefix over the 256 expert totals.
        uint lane_excl = simd_prefix_exclusive_sum(total);
        threadgroup uint simd_totals[8];
        if (lane == 31) {
            simd_totals[simd_id] = lane_excl + total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint running = 0u;
        for (uint s = 0; s < simd_id; ++s) {
            running += simd_totals[s];
        }
        running += lane_excl;
        // Exclusive scan over blocks for this expert, offset by the bin base.
        // Column `e` of `block_offset` is read by the scatter only as
        // `block_offset[b * WIDTH + key]` for a key that occurs in block `b`,
        // so a column whose global total is zero is never read and need not be
        // written. The counter table is 256 wide while the model routes 128
        // experts, so at minimum half the columns are unconditionally dead.
        if (total > 0u) {
            for (uint b = 0; b < nblocks; ++b) {
                block_offset[b * WIDTH + e] = running;
                running += block_hist[b * WIDTH + e];
            }
        }
        """

    static let parallelScan = """
        constexpr uint WIDTH = \(routeCsortPrefillWidth);
        constexpr uint PARTS = 8;
        constexpr uint COLS = (uint)NE;
        // PROMPT-GLUE2 (pg2): 1024 threads = PARTS block ranges x COLS expert
        // columns. Every sum below is an unsigned integer sum, so the split of
        // the block loop into PARTS ranges combined in range order yields the
        // incumbent's totals and running offsets word for word.
        const uint t = thread_position_in_threadgroup.x;
        const uint e = t % COLS;
        const uint part = t / COLS;
        const uint nblocks = (uint)block_hist_shape[0];
        const uint per = nblocks / PARTS;
        const uint b0 = part * per;
        threadgroup uint part_sum[PARTS][COLS];
        threadgroup uint simd_totals[COLS / 32];
        uint partial = 0u;
        for (uint b = b0; b < b0 + per; ++b) {
            partial += block_hist[b * WIDTH + e];
        }
        part_sum[part][e] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint total = 0u;
        for (uint p = 0; p < PARTS; ++p) {
            total += part_sum[p][e];
        }
        // Global bin base: exclusive prefix over the expert totals. Each part's
        // simdgroups hold the same totals in the same lanes, so every part
        // computes the same prefix; part 0 publishes the simdgroup totals.
        const uint lane = e % 32;
        const uint simd_id = e / 32;
        const uint lane_excl = simd_prefix_exclusive_sum(total);
        if (part == 0 && lane == 31) {
            simd_totals[simd_id] = lane_excl + total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint running = 0u;
        for (uint s = 0; s < simd_id; ++s) {
            running += simd_totals[s];
        }
        running += lane_excl;
        for (uint p = 0; p < part; ++p) {
            running += part_sum[p][e];
        }
        // Exclusive scan over this part's blocks for the column, offset by the
        // bin base plus the earlier parts' counts; columns with a zero total are
        // never read by the scatter and are left unwritten, as the incumbent
        // leaves them.
        if (total > 0u) {
            for (uint b = b0; b < b0 + per; ++b) {
                block_offset[b * WIDTH + e] = running;
                running += block_hist[b * WIDTH + e];
            }
        }
        """

    static let scatter = """
        constexpr uint BLOCK = \(routeCsortPrefillBlock);
        constexpr uint WIDTH = \(routeCsortPrefillWidth);
        uint b = threadgroup_position_in_grid.x;
        uint k = thread_position_in_threadgroup.x;
        uint n = keys_shape[0];
        uint idx = b * BLOCK + k;
        // Tail block: the sentinel is outside the proven key space (keys are
        // below the 256-wide counter table), so it can never tie a real key.
        uint key = (idx < n) ? keys[idx] : 0xffffffffu;
        threadgroup uint tg_keys[BLOCK];
        tg_keys[k] = key;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (idx < n) {
            // Stable local rank: earlier keys in this block only. Read in
            // index order from threadgroup memory, so no write position ever
            // depends on scheduling.
            uint rank = 0u;
            for (uint j = 0; j < k; ++j) {
                rank += (tg_keys[j] == key) ? 1u : 0u;
            }
            uint pos = block_offset[b * WIDTH + key] + rank;
            row_order[pos] = idx / (uint)M;
            sorted_keys[pos] = key;
            inverse_order[idx] = pos;
        }
        """

    static let bitsetScatter = """
        constexpr uint BLOCK = \(routeCsortPrefillBlock);
        constexpr uint WIDTH = \(routeCsortPrefillWidth);
        constexpr uint WORDS = BLOCK / 32;
        const uint b = threadgroup_position_in_grid.x;
        const uint k = thread_position_in_threadgroup.x;
        const uint n = keys_shape[0];
        const uint idx = b * BLOCK + k;
        // Each bit names one input position; equal keys occupy distinct bits.
        // Admission proves keys < NE. Compact only scratch; block offsets
        // retain the histogram's WIDTH stride.
        constexpr uint SCRATCH_WIDTH = (uint)NE;
        threadgroup atomic_uint bitsets[SCRATCH_WIDTH * WORDS];
        for (uint i = k; i < SCRATCH_WIDTH * WORDS; i += BLOCK) {
            atomic_store_explicit(&bitsets[i], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint key = idx < n ? keys[idx] : 0u;
        const uint word = k / 32u;
        const uint bit = k & 31u;
        if (idx < n) {
            atomic_fetch_or_explicit(&bitsets[word * SCRATCH_WIDTH + key], 1u << bit, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (idx < n) {
            // Count strictly earlier positions, preserving stable tie order.
            // The current-word mask is valid even when bit is zero or 31.
            uint rank = 0u;
            for (uint w = 0; w < word; ++w) {
                rank += popcount(atomic_load_explicit(&bitsets[w * SCRATCH_WIDTH + key], memory_order_relaxed));
            }
            rank += popcount(atomic_load_explicit(&bitsets[word * SCRATCH_WIDTH + key], memory_order_relaxed)
                & ((1u << bit) - 1u));
            const uint pos = block_offset[b * WIDTH + key] + rank;
            row_order[pos] = idx / uint(M);
            sorted_keys[pos] = key;
            inverse_order[idx] = pos;
        }
        """
}
