// Copyright © 2026 Eigen Labs.
// Portions adapted from the LLVM Project's libc++ selection algorithms.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

import Cmlx
import MLX

/// Membership semantics of the pinned PyTorch CPU oracle, not a faster top-k.
/// ATen/native/TopKImpl.h at 5c4886908584029761b579af026dcfb627c84070 uses
/// score-only libc++ nth_element / partial_sort. MLX adds an index tie-break.
/// Both native devices use that CPU tie policy, not PyTorch's GPU tie policy.
enum GLM5NextSparseSelection {
    static let maximumSelectedPools = 512
    static let maximumScratchIndices = 64 * maximumSelectedPools - 1
    /// Threadgroup width of the parallel selector: a multiple of the 32-wide
    /// SIMD-group and at least one thread per radix bin (256).
    static let parallelThreads = 256

    enum Failure: Error, Equatable {
        case invalidGeometry
        case invalidTensor
        case unsupportedStream
    }

    struct Layout {
        let batch: Int
        let pools: Int
        let count: Int
        let scratchWidth: Int
        let scratchBytes: Int
        let previousIndexBytes: Int

        init(shape: [Int], requestedCount: Int) throws {
            guard shape.count == 2,
                (1...GLM5NextSparseSelection.maximumSelectedPools).contains(requestedCount),
                shape.allSatisfy({ $0 > 0 && $0 <= Int(Int32.max) })
            else { throw Failure.invalidGeometry }
            let (elements, overflow) = shape[0].multipliedReportingOverflow(by: shape[1])
            guard !overflow, elements <= Int(Int32.max) else { throw Failure.invalidGeometry }
            batch = shape[0]
            pools = shape[1]
            count = min(requestedCount, pools)
            // Above this bound, partial_sort needs only its K-element heap.
            // Below it, nth_element needs the complete retained candidate list.
            scratchWidth = min(pools, 64 * count - 1)
            guard scratchWidth >= count,
                scratchWidth <= GLM5NextSparseSelection.maximumScratchIndices
            else { throw Failure.invalidGeometry }
            let (oldBytes, oldOverflow) = elements.multipliedReportingOverflow(by: 4)
            let (bytes, byteOverflow) = (batch * scratchWidth).multipliedReportingOverflow(by: 4)
            guard !oldOverflow, !byteOverflow, bytes <= oldBytes else {
                throw Failure.invalidGeometry
            }
            scratchBytes = bytes
            previousIndexBytes = oldBytes
        }
    }

    /// `scores` is produced on the current operation stream by selectTokens.
    /// CPU readback is only for actual CPU execution, never a GPU fallback.
    /// Return original pool IDs, padded with `pools` so chronological sorting
    /// leaves padding last. The public token selection still uses -1 padding.
    /// Both strategies return the same membership; only the GPU dispatch and
    /// the order within a row differ. CPU execution has one implementation.
    /// `completePools` is `[B,P]` (a pool stays a candidate when any row has it
    /// complete) or the already-reduced `[1,P]`.
    static func select(
        scores: MLXArray, completePools: MLXArray, requestedCount: Int,
        strategy: GLM5NextSparseAttention.SelectionStrategy = .reference
    ) throws -> MLXArray {
        let layout = try Layout(shape: scores.shape, requestedCount: requestedCount)
        guard [.float32, .float16, .bfloat16].contains(scores.dtype),
            completePools.ndim == 2, completePools.dim(1) == layout.pools,
            completePools.dim(0) == layout.batch || completePools.dim(0) == 1,
            completePools.dtype == .bool
        else { throw Failure.invalidTensor }
        let stream = StreamOrDevice.default
        switch deviceType(stream) {
        case .cpu:
            return try cpu(scores: scores, completePools: completePools, layout: layout)
        case .gpu:
            #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
                // The sole output doubles as index scratch. Its retained buffer
                // is no larger than the replaced full-axis argSort output.
                let inputs = [scores.asType(.float32, stream: stream), completePools]
                // Rows, pools and scratch width are read from the input shape,
                // so a growing history reuses each strategy's one pipeline.
                let result: MLXArray
                switch strategy {
                case .reference:
                    result =
                        kernel(
                            inputs, template: referenceTemplate(requestedCount: requestedCount),
                            grid: (1, layout.batch, 1), threadGroup: (1, 1, 1),
                            outputShapes: [[layout.batch, layout.scratchWidth]],
                            outputDTypes: [.int32], stream: stream)[0]
                case .parallelThreshold:
                    result =
                        parallelKernel(
                            inputs,
                            template: parallelTemplate(requestedCount: requestedCount),
                            grid: (parallelThreads, layout.batch, 1),
                            threadGroup: (parallelThreads, 1, 1),
                            outputShapes: [[layout.batch, layout.scratchWidth]],
                            outputDTypes: [.int32], stream: stream)[0]
                }
                return result[0..., ..<layout.count]
            #else
                throw Failure.unsupportedStream
            #endif
        default:
            throw Failure.unsupportedStream
        }
    }

    /// Cmlx stream metadata access does not evaluate arrays or synchronize.
    private static func deviceType(_ stream: StreamOrDevice) -> DeviceType? {
        var device = mlx_device_new()
        defer { mlx_device_free(device) }
        var type = MLX_CPU
        guard mlx_stream_get_device(&device, stream.ctx) == 0,
            mlx_device_get_type(&type, device) == 0 else { return nil }
        if type == MLX_CPU { return .cpu }
        if type == MLX_GPU { return .gpu }
        return nil
    }

    private static func cpu(scores: MLXArray, completePools: MLXArray, layout: Layout) throws
        -> MLXArray
    {
        try MLX.withError { error in
            let scores = scores.asType(.float32)
            eval(scores, completePools)
            try error.check()
            let values = scores.asArray(Float.self)
            let complete = completePools.asArray(Bool.self)
            try error.check()
            let maskRows = completePools.dim(0)
            func keep(_ pool: Int) -> Bool {
                (0..<maskRows).contains { complete[$0 * layout.pools + pool] }
            }
            let retained = (0..<layout.pools).reduce(0) { $0 + (keep($1) ? 1 : 0) }
            let count = min(layout.count, retained)
            var result = [Int32](repeating: Int32(layout.pools), count: layout.batch * layout.count)
            guard count > 0 else { return MLXArray(result, [layout.batch, layout.count]) }
            let partial = 64 * count <= retained
            for row in 0..<layout.batch {
                var order = CPUOrder(scores: values, offset: row * layout.pools)
                order.indices.reserveCapacity(partial ? count : retained)
                for pool in 0..<layout.pools where keep(pool) {
                    if !partial || order.indices.count < count {
                        order.indices.append(Int32(pool))
                        if partial && order.indices.count == count { order.makeHeap() }
                    } else {
                        order.consider(Int32(pool))
                    }
                }
                if !partial { order.nthElement(count - 1) }
                for slot in 0..<count { result[row * layout.count + slot] = order.indices[slot] }
            }
            return MLXArray(result, [layout.batch, layout.count])
        }
    }

    /// CPU diagnostic implementation of the same score-only swaps as Metal.
    /// Only indices mutate; neither scores nor the completeness mask changes.
    private struct CPUOrder {
        let scores: [Float]
        let offset: Int
        var indices: [Int32] = []

        func beforeID(_ a: Int32, _ b: Int32) -> Bool {
            let x = scores[offset + Int(a)], y = scores[offset + Int(b)]
            return (x.isNaN && !y.isNaN) || x > y
        }
        func before(_ a: Int, _ b: Int) -> Bool { beforeID(indices[a], indices[b]) }
        mutating func swap(_ a: Int, _ b: Int) { indices.swapAt(a, b) }

        mutating func sort3(_ x: Int, _ y: Int, _ z: Int) -> Bool {
            if !before(y, x) {
                if !before(z, y) { return false }
                swap(y, z)
                if before(y, x) { swap(x, y) }
                return true
            }
            if before(z, y) { swap(x, z); return true }
            swap(x, y)
            if before(z, y) { swap(y, z) }
            return true
        }
        mutating func selectionSort(_ lower: Int, _ upper: Int) {
            for i in lower..<(upper - 1) {
                var best = i
                for j in (i + 1)..<upper where before(j, best) { best = j }
                if best != i { swap(i, best) }
            }
        }
        func isSorted(_ lower: Int, _ upper: Int) -> Bool {
            guard upper - lower > 1 else { return true }
            return ((lower + 1)..<upper).allSatisfy { !before($0, $0 - 1) }
        }
        mutating func nthElement(_ nth: Int) {
            var lower = 0, upper = indices.count
            while true {
                let length = upper - lower
                if length <= 1 { return }
                if length == 2 {
                    if before(upper - 1, lower) { swap(lower, upper - 1) }
                    return
                }
                if length == 3 { _ = sort3(lower, lower + 1, upper - 1); return }
                if length <= 7 { selectionSort(lower, upper); return }
                var middle = lower + length / 2
                var swapped = sort3(lower, middle, upper - 1)
                var i = lower, j = upper - 1
                if !before(i, middle) {
                    var found = false
                    while j > i {
                        j -= 1
                        if j == i { break }
                        if before(j, middle) { found = true; break }
                    }
                    if found {
                        swap(i, j); swapped = true
                    } else {
                        i += 1; j = upper - 1
                        if !before(lower, j) {
                            while true {
                                if i == j { return }
                                if before(lower, i) { swap(i, j); swapped = true; i += 1; break }
                                i += 1
                            }
                        }
                        if i == j { return }
                        while true {
                            while !before(lower, i) { i += 1 }
                            repeat { j -= 1 } while before(lower, j)
                            if i >= j { break }
                            swap(i, j); swapped = true; i += 1
                        }
                        if nth < i { return }
                        lower = i
                        continue
                    }
                }
                i += 1
                if i < j {
                    while true {
                        while before(i, middle) { i += 1 }
                        repeat { j -= 1 } while !before(j, middle)
                        if i >= j { break }
                        swap(i, j); swapped = true
                        if middle == i { middle = j }
                        i += 1
                    }
                }
                if i != middle && before(middle, i) { swap(i, middle); swapped = true }
                if nth == i { return }
                if !swapped && (nth < i ? isSorted(lower, i) : isSorted(i, upper)) { return }
                if nth < i { upper = i } else { lower = i + 1 }
            }
        }
        mutating func siftDown(_ start: Int) {
            let length = indices.count
            guard length > 1, start <= (length - 2) / 2 else { return }
            var position = start, child = 2 * start + 1
            if child + 1 < length && before(child, child + 1) { child += 1 }
            if before(child, position) { return }
            let top = indices[position]
            while true {
                indices[position] = indices[child]
                position = child
                if child > (length - 2) / 2 { break }
                child = 2 * child + 1
                if child + 1 < length && before(child, child + 1) { child += 1 }
                if beforeID(indices[child], top) { break }
            }
            indices[position] = top
        }
        mutating func makeHeap() {
            guard indices.count > 1 else { return }
            for start in stride(from: (indices.count - 2) / 2, through: 0, by: -1) {
                siftDown(start)
            }
        }
        mutating func consider(_ candidate: Int32) {
            if beforeID(candidate, indices[0]) { indices[0] = candidate; siftDown(0) }
        }
    }

    /// MLX compiles one Metal library per distinct template tuple, so neither
    /// tuple may contain the history-dependent row, pool or scratch extents.
    static func referenceTemplate(requestedCount: Int) -> [(String, any KernelTemplateArg)] {
        [("Requested", requestedCount)]
    }

    static func parallelTemplate(requestedCount: Int) -> [(String, any KernelTemplateArg)] {
        [("Count", requestedCount), ("Threads", parallelThreads)]
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        private static let kernel = MLXFast.metalKernel(
            name: "glm5_next_reference_sparse_selection",
            inputNames: ["scores", "complete"], outputNames: ["selection"],
            source: referenceKernelSource, header: referenceHeader)

        /// The verbatim reference row body over runtime extents that equal
        /// `Layout`'s `batch`, `pools`, `count` and `scratchWidth`.
        static let referenceKernelSource =
            """
            const int Rows = scores_shape[0];
            const int MaskRows = complete_shape[0];
            const int Pools = scores_shape[1];
            const int Count = metal::min(Requested, Pools);
            const int Scratch = metal::min(Pools, 64 * Count - 1);

            """ + referenceSource

        /// Opt-in candidate, one threadgroup per row. A parallel radix select
        /// finds the K-th score; when no tie straddles it, membership is the
        /// unique set `score >= K-th` for every valid top-K algorithm, so it is
        /// emitted in parallel (ascending IDs). A straddling tie reruns the
        /// unchanged reference emulation for that row only, on thread zero.
        private static let parallelKernel = MLXFast.metalKernel(
            name: "glm5_next_parallel_sparse_selection",
            inputNames: ["scores", "complete"], outputNames: ["selection"],
            source: parallelSource, header: parallelHeader)

        static let referenceSource = """
                const size_t row = thread_position_in_grid.y;
                if (row >= Rows) return;
                device int* order = selection + row * Scratch;
                auto values = scores + row * Pools;
                for (int i = 0; i < Scratch; ++i) order[i] = Pools;
                int retained = 0;
                for (int p = 0; p < Pools; ++p)
                    retained += glm5_sparse_keep(complete, MaskRows, Pools, p);
                const int count = metal::min(Count, retained);
                if (count == 0) return;
                const bool partial = 64 * count <= retained;
                int loaded = 0;
                for (int p = 0; p < Pools; ++p) {
                    if (!glm5_sparse_keep(complete, MaskRows, Pools, p)) continue;
                    if (!partial || loaded < count) {
                        order[loaded++] = p;
                        if (partial && loaded == count) {
                            for (int i = (count - 2) / 2; count > 1 && i >= 0; --i)
                                glm5_sparse_sift(values, order, count, i);
                        }
                    } else if (glm5_sparse_before(values, p, order[0])) {
                        order[0] = p;
                        glm5_sparse_sift(values, order, count, 0);
                    }
                }
                if (!partial) glm5_sparse_nth(values, order, retained, count - 1);
                for (int i = count; i < Scratch; ++i) order[i] = Pools;
                """

        static let referenceHeader = """
                template <typename C>
                bool glm5_sparse_keep(C complete, int rows, int pools, int p) {
                    for (int b = 0; b < rows; ++b)
                        if (complete[size_t(b) * pools + p]) return true;
                    return false;
                }
                template <typename S>
                bool glm5_sparse_before(S scores, int a, int b) {
                    const float x = scores[a], y = scores[b];
                    return (metal::isnan(x) && !metal::isnan(y)) || x > y;
                }
                template <typename S>
                bool glm5_sparse_less(S scores, device const int* order,
                                      int a, int b) {
                    return glm5_sparse_before(scores, order[a], order[b]);
                }
                void glm5_sparse_swap(device int* order, int a, int b) {
                    const int value = order[a]; order[a] = order[b]; order[b] = value;
                }
                template <typename S>
                bool glm5_sparse_sort3(S scores, device int* order,
                                       int x, int y, int z) {
                    if (!glm5_sparse_less(scores, order, y, x)) {
                        if (!glm5_sparse_less(scores, order, z, y)) return false;
                        glm5_sparse_swap(order, y, z);
                        if (glm5_sparse_less(scores, order, y, x)) glm5_sparse_swap(order, x, y);
                        return true;
                    }
                    if (glm5_sparse_less(scores, order, z, y)) {
                        glm5_sparse_swap(order, x, z); return true;
                    }
                    glm5_sparse_swap(order, x, y);
                    if (glm5_sparse_less(scores, order, z, y)) glm5_sparse_swap(order, y, z);
                    return true;
                }
                template <typename S>
                bool glm5_sparse_sorted(S scores, device const int* order,
                                        int lower, int upper) {
                    for (int i = lower + 1; i < upper; ++i)
                        if (glm5_sparse_less(scores, order, i, i - 1)) return false;
                    return true;
                }
                template <typename S>
                void glm5_sparse_nth(S scores, device int* order,
                                     int length, int nth) {
                    int lower = 0, upper = length;
                    while (true) {
                        const int n = upper - lower;
                        if (n <= 1) return;
                        if (n == 2) {
                            if (glm5_sparse_less(scores, order, upper - 1, lower))
                                glm5_sparse_swap(order, lower, upper - 1);
                            return;
                        }
                        if (n == 3) {
                            glm5_sparse_sort3(scores, order, lower, lower + 1, upper - 1);
                            return;
                        }
                        if (n <= 7) {
                            for (int i = lower; i < upper - 1; ++i) {
                                int best = i;
                                for (int j = i + 1; j < upper; ++j)
                                    if (glm5_sparse_less(scores, order, j, best)) best = j;
                                if (best != i) glm5_sparse_swap(order, i, best);
                            }
                            return;
                        }
                        int middle = lower + n / 2;
                        bool swapped = glm5_sparse_sort3(scores, order, lower, middle, upper - 1);
                        int i = lower, j = upper - 1;
                        if (!glm5_sparse_less(scores, order, i, middle)) {
                            bool found = false;
                            while (j > i) {
                                --j;
                                if (j == i) break;
                                if (glm5_sparse_less(scores, order, j, middle)) { found = true; break; }
                            }
                            if (found) {
                                glm5_sparse_swap(order, i, j); swapped = true;
                            } else {
                                ++i; j = upper - 1;
                                if (!glm5_sparse_less(scores, order, lower, j)) {
                                    while (true) {
                                        if (i == j) return;
                                        if (glm5_sparse_less(scores, order, lower, i)) {
                                            glm5_sparse_swap(order, i, j); swapped = true; ++i; break;
                                        }
                                        ++i;
                                    }
                                }
                                if (i == j) return;
                                while (true) {
                                    while (!glm5_sparse_less(scores, order, lower, i)) ++i;
                                    do { --j; } while (glm5_sparse_less(scores, order, lower, j));
                                    if (i >= j) break;
                                    glm5_sparse_swap(order, i, j); swapped = true; ++i;
                                }
                                if (nth < i) return;
                                lower = i; continue;
                            }
                        }
                        ++i;
                        if (i < j) {
                            while (true) {
                                while (glm5_sparse_less(scores, order, i, middle)) ++i;
                                do { --j; } while (!glm5_sparse_less(scores, order, j, middle));
                                if (i >= j) break;
                                glm5_sparse_swap(order, i, j); swapped = true;
                                if (middle == i) middle = j;
                                ++i;
                            }
                        }
                        if (i != middle && glm5_sparse_less(scores, order, middle, i)) {
                            glm5_sparse_swap(order, i, middle); swapped = true;
                        }
                        if (nth == i) return;
                        if (!swapped && (nth < i
                            ? glm5_sparse_sorted(scores, order, lower, i)
                            : glm5_sparse_sorted(scores, order, i, upper))) return;
                        if (nth < i) upper = i; else lower = i + 1;
                    }
                }
                template <typename S>
                void glm5_sparse_sift(S scores, device int* order,
                                      int length, int start) {
                    if (length < 2 || start > (length - 2) / 2) return;
                    int child = 2 * start + 1;
                    if (child + 1 < length && glm5_sparse_less(scores, order, child, child + 1)) ++child;
                    if (glm5_sparse_less(scores, order, child, start)) return;
                    const int top = order[start];
                    while (true) {
                        order[start] = order[child]; start = child;
                        if (child > (length - 2) / 2) break;
                        child = 2 * child + 1;
                        if (child + 1 < length && glm5_sparse_less(scores, order, child, child + 1)) ++child;
                        if (glm5_sparse_before(scores, order[child], top)) break;
                    }
                    order[start] = top;
                }
                """

        /// Wraps the reference row body verbatim as the tie fallback. Keys map
        /// the reference order to unsigned integers: retained pools are >= 1,
        /// NaNs share the top key and -0 joins +0, like the score comparison.
        private static let parallelHeader =
            referenceHeader + """

                template <typename S, typename C>
                void glm5_sparse_reference_row(
                    S scores, C complete, device int* selection, uint3 thread_position_in_grid,
                    int Rows, int MaskRows, int Pools, int Count, int Scratch) {

                """ + referenceSource + """

                }
                template <typename S, typename C>
                uint glm5_sparse_key(S scores, C complete, int rows, int pools, int p) {
                    if (!glm5_sparse_keep(complete, rows, pools, p)) return 0u;
                    const float x = scores[p];
                    if (metal::isnan(x)) return 0xFFFFFFFFu;
                    const uint bits = x == 0.0f ? 0u : as_type<uint>(x);
                    return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
                }
                // One atomic per distinct bin in a SIMD-group: high key bytes collide.
                void glm5_sparse_count(threadgroup atomic_uint* bins, bool active, uint bin) {
                    bool pending = active;
                    while (simd_any(pending)) {
                        if (pending) {
                            const uint leader = simd_broadcast_first(bin);
                            const bool same = bin == leader;
                            const uint total = simd_sum(same ? 1u : 0u);
                            if (same) {
                                if (simd_is_first())
                                    atomic_fetch_add_explicit(bins + leader, total, memory_order_relaxed);
                                pending = false;
                            }
                        }
                    }
                }
                uint glm5_sparse_group_sum(uint value, threadgroup uint* partials,
                                           uint lane, uint warp, uint warps) {
                    const uint total = simd_sum(value);
                    if (lane == 0u) partials[warp] = total;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    uint result = 0u;
                    for (uint i = 0u; i < warps; ++i) result += partials[i];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    return result;
                }
                // Scan bins from the highest key byte down. Exactly one bin holds
                // the needed rank: chosen = {bin, rank within bin, bin count}.
                void glm5_sparse_pick(threadgroup atomic_uint* bins, threadgroup uint* partials,
                                      threadgroup uint* chosen, uint tid, uint lane, uint warp,
                                      uint needed) {
                    uint count = 0u, inclusive = 0u;
                    if (tid < 256u) {
                        count = atomic_load_explicit(bins + (255u - tid), memory_order_relaxed);
                        inclusive = simd_prefix_inclusive_sum(count);
                        if (lane == 31u) partials[warp] = inclusive;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (tid < 256u) {
                        for (uint i = 0u; i < warp; ++i) inclusive += partials[i];
                        const uint exclusive = inclusive - count;
                        if (exclusive < needed && needed <= inclusive) {
                            chosen[0] = 255u - tid;
                            chosen[1] = needed - exclusive;
                            chosen[2] = count;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                """

        private static let parallelSource = """
                static_assert(Threads % 32 == 0 && Threads >= 256, "one scan thread per radix bin");
                constexpr uint warps = uint(Threads) / 32u;
                threadgroup atomic_uint bins[256];
                threadgroup uint partials[warps];
                threadgroup uint chosen[3];
                const int Rows = scores_shape[0];
                const int MaskRows = complete_shape[0];
                const int Pools = scores_shape[1];
                const int Selected = metal::min(Count, Pools);
                const int Scratch = metal::min(Pools, 64 * Selected - 1);
                const uint tid = thread_index_in_threadgroup;
                const uint lane = thread_index_in_simdgroup;
                const uint warp = simdgroup_index_in_threadgroup;
                const size_t selectedRow = threadgroup_position_in_grid.y;
                device int* output = selection + selectedRow * Scratch;
                auto rowScores = scores + selectedRow * Pools;

                for (uint i = tid; i < 256u; i += uint(Threads))
                    atomic_store_explicit(bins + i, 0u, memory_order_relaxed);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                uint kept = 0u;
                for (int base = 0; base < Pools; base += Threads) {
                    const int p = base + int(tid);
                    const uint key = p < Pools ? glm5_sparse_key(rowScores, complete, MaskRows, Pools, p) : 0u;
                    kept += key != 0u ? 1u : 0u;
                    glm5_sparse_count(bins, p < Pools, key >> 24);
                }
                const int retained = int(glm5_sparse_group_sum(kept, partials, lane, warp, warps));
                const int count = metal::min(Count, retained);

                // Radix select of the count-th largest key. `needed` of the
                // `matching` keys equal to it are selected; any shortfall is a
                // tie straddling the boundary whose membership is order-dependent.
                uint threshold = 1u;
                uint needed = 0u, matching = 0u;
                if (count > 0 && count < retained) {
                    uint prefix = 0u, mask = 0u;
                    needed = uint(count);
                    for (int shift = 24; shift >= 0; shift -= 8) {
                        if (shift < 24) {
                            for (uint i = tid; i < 256u; i += uint(Threads))
                                atomic_store_explicit(bins + i, 0u, memory_order_relaxed);
                            threadgroup_barrier(mem_flags::mem_threadgroup);
                            for (int base = 0; base < Pools; base += Threads) {
                                const int p = base + int(tid);
                                const uint key =
                                    p < Pools ? glm5_sparse_key(rowScores, complete, MaskRows, Pools, p) : 0u;
                                glm5_sparse_count(
                                    bins, p < Pools && (key & mask) == prefix, (key >> shift) & 0xFFu);
                            }
                            threadgroup_barrier(mem_flags::mem_threadgroup);
                        }
                        glm5_sparse_pick(bins, partials, chosen, tid, lane, warp, needed);
                        prefix |= chosen[0] << shift;
                        mask |= 0xFFu << shift;
                        needed = chosen[1];
                        matching = chosen[2];
                    }
                    threshold = prefix;
                }
                if (needed != matching) {
                    if (tid == 0u)
                        glm5_sparse_reference_row(
                            scores, complete, selection, thread_position_in_grid,
                            Rows, MaskRows, Pools, Count, Scratch);
                    return;
                }

                // Unique membership: emit retained keys >= threshold in ID order.
                uint written = 0u;
                for (int base = 0; base < Pools && written < uint(count); base += Threads) {
                    const int p = base + int(tid);
                    const bool member =
                        p < Pools && glm5_sparse_key(rowScores, complete, MaskRows, Pools, p) >= threshold;
                    const uint flag = member ? 1u : 0u;
                    const uint before = simd_prefix_exclusive_sum(flag);
                    const uint total = simd_sum(flag);
                    if (lane == 0u) partials[warp] = total;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    uint offset = written, tile = 0u;
                    for (uint i = 0u; i < warps; ++i) {
                        const uint value = partials[i];
                        offset += i < warp ? value : 0u;
                        tile += value;
                    }
                    if (member) output[offset + before] = p;
                    written += tile;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                for (int i = count + int(tid); i < Selected; i += Threads) output[i] = Pools;
                """
    #endif
}
