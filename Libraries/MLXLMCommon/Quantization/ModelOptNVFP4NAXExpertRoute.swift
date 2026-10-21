// SPDX-License-Identifier: Apache-2.0 AND MIT
// Copyright 2026 Eigen Labs.
// The route kernel is GLM5NextMoEPrefillKernel's `routeHeader` (mlx-swift-lm
// work/glm53-flash-release 97b2ca6d) verbatim; see ModelOptNVFP4NAXExperts.swift
// for the provenance of the matrix kernels.

import Foundation
import MLX

/// A tiled prefill chunk's whole routed-expert block (both NVFP4 projections
/// and the weighted combine) on the Neural Accelerators in five launches:
///
/// 1. `route`: one threadgroup sorts the assignments stably by expert and
///    writes the sort order, its inverse and both projections' tile lists;
/// 2. fc1's NVFP4 activation rounding reads each sorted row's latent row
///    through the order (no sorted copy of the input);
/// 3. fc1 with the weight-scale and relu2 epilogue, which also rounds its
///    output to fc2's NVFP4 activation grid (`roundsInEpilogue`);
/// 4. fc2 with its weight-scale epilogue;
/// 5. the weighted combine reads each assignment's row through the inverse
///    (no unsorted copy).
///
/// It replaces `gatherSort`, both `graphForward` calls of the tiled route,
/// `scatterUnsort` and the combine. Every row gets the same activation, the
/// same tensor-op chain and the same epilogue arithmetic as on that route,
/// so the output is bitwise the route's; the only exception is NaN: with
/// the rounding in fc1's epilogue, a NaN in fc1's output makes its 16-column
/// block NaN rather than the whole fc2 input row, and every fc2 output of
/// that row is NaN either way.
extension ModelOptNVFP4NAXExperts {
    /// On wherever the NAX expert kernels are (`isEnabled`). On an M5 Ultra at
    /// Super's chunk the whole expert block (route, fc1, fc2, combine) takes
    /// 6.47 instead of 8.85 ms, bit for bit; clearing it keeps the op-by-op
    /// route.
    nonisolated(unsafe) public static var fusedRoute = naxAvailable

    /// With `fusedRoute`, whether fc1's epilogue does fc2's activation
    /// rounding (needs fc2's `.saturating` policy) instead of a separate
    /// pass over the sorted fc1 output.
    nonisolated(unsafe) public static var roundsInEpilogue = true

    /// The route kernel keeps per-simdgroup counters of every expert in
    /// threadgroup memory.
    public static let maximumRouteExperts = 512

    /// Whether `routedExperts` runs `tokens` x `topK` router-produced
    /// assignments of `dtype` rows through `fc1` and `fc2` exactly as their
    /// sorted tiled graph route (with this type's matrix kernels) does.
    public static func admitsRoute(
        fc1: ModelOptNVFP4SwitchLinear, fc2: ModelOptNVFP4SwitchLinear, tokens: Int, topK: Int,
        dtype: DType
    ) -> Bool {
        let rows = tokens * topK
        let experts = fc1.numExperts
        guard fusedRoute, isEnabled, tokens > 0, (1 ..< 32).contains(topK),
            fc2.numExperts == experts, (1 ... maximumRouteExperts).contains(experts),
            fc1.outputDims == fc2.inputDims,
            // graphForward's tiled threshold.
            rows >= max(16, 4 * experts)
        else { return false }
        func admits(_ bank: ModelOptNVFP4SwitchLinear) -> Bool {
            let (inputs, outputs) = (bank.inputDims, bank.outputDims)
            guard bank.execution == .graph, bank.accumulation == .tiled,
                (bank.originalWeightDType ?? dtype) == dtype,
                bank.weight.dtype == .uint32, bank.blockScales.dtype == .uint8
            else { return false }
            return bank.weight.shape == [experts, outputs, inputs / 8]
                && bank.blockScales.shape == [experts, outputs, inputs / 16]
                && supports(
                    rows: rows, inputs: inputs, outputs: outputs, experts: experts, dtype: dtype)
        }
        return admits(fc1) && admits(fc2)
    }

    /// `[T, fc2.outputDims]` in `x.dtype`: for each token, the sum over its
    /// `topK` assignments of `scores` times `fc2(relu2(fc1(x)))` for that
    /// assignment's expert (`admitsRoute`). `x` is `[T, fc1.inputDims]`,
    /// `indices` `[T, topK]` uint32 router IDs below E, `scores` `[T, topK]`
    /// float32. `roundsInEpilogue` overrides the switch (diagnostics).
    public static func routedExperts(
        _ x: MLXArray, indices: MLXArray, scores: MLXArray,
        fc1: ModelOptNVFP4SwitchLinear, fc2: ModelOptNVFP4SwitchLinear,
        roundsInEpilogue epilogue: Bool? = nil
    ) -> MLXArray {
        let (tokens, topK) = (indices.dim(0), indices.dim(1))
        let rows = tokens * topK
        let experts = fc1.numExperts
        precondition(
            indices.ndim == 2 && indices.dtype == .uint32 && scores.shape == indices.shape
                && x.shape == [tokens, fc1.inputDims] && fc2.inputDims == fc1.outputDims,
            "NVFP4 NAX expert route geometry")
        let plan1 = plan(rows: rows, experts: experts, input: fc1.inputDims)
        let plan2 = plan(rows: rows, experts: experts, input: fc2.inputDims)
        let sorted = route(
            indices, experts: experts, fc1Rows: plan1.shape.rows, fc2Rows: plan2.shape.rows)
        let ids = indices.flattened()
        let a1 = ModelOptActivationKernels.nvfp4Sorted(
            x, order: sorted.order, ids: ids, topK: topK, gather: true,
            globalScale: fc1.routeTables.globalScale,
            overflowDivisor: fc1.routeTables.overflowDivisor, activationDType: x.dtype)
        let fused = (epilogue ?? roundsInEpilogue) && fc2.activationOverflow == .saturating
        var a2 = gemm(
            a1, weight: fc1.weight, scales: fc1.blockScales,
            expertScale: fc1.routeTables.inverseWeightScale, squaredReLU: true,
            tiles: sorted.fc1.list, tileCount: sorted.fc1.count,
            maximumTiles: sorted.fc1.maximum, plan: plan1,
            roundingScale: fused ? fc2.routeTables.globalScale : nil)
        if !fused {
            a2 = ModelOptActivationKernels.nvfp4Sorted(
                a2, order: sorted.order, ids: ids, topK: topK, gather: false,
                globalScale: fc2.routeTables.globalScale,
                overflowDivisor: fc2.routeTables.overflowDivisor, activationDType: x.dtype)
        }
        let y = gemm(
            a2, weight: fc2.weight, scales: fc2.blockScales,
            expertScale: fc2.routeTables.inverseWeightScale, squaredReLU: false,
            tiles: sorted.fc2.list, tileCount: sorted.fc2.count,
            maximumTiles: sorted.fc2.maximum, plan: plan2)
        return combine(y, inverse: sorted.inverse, scores: scores)
    }

    // MARK: - Route and combine

    /// `(row_start, expert, rows, 0)` tile descriptors (`maximum` slots,
    /// `count` written) of at most `rows` rows each, expert-major.
    struct RouteTiles {
        let list: MLXArray
        let count: MLXArray
        let maximum: Int
    }

    /// `order[p]`: the assignment `token·K + slot` at sorted position p,
    /// ascending within each expert; `inverse`: each assignment's position.
    struct SortedRoute {
        let order: MLXArray
        let inverse: MLXArray
        let fc1: RouteTiles
        let fc2: RouteTiles
    }

    /// GLM's route launch with no shared rows: `indices` `[T, K]` (uint32,
    /// each below `experts`), tile lists of `fc1Rows` and `fc2Rows` rows.
    static func route(_ indices: MLXArray, experts: Int, fc1Rows: Int, fc2Rows: Int)
        -> SortedRoute
    {
        let rows = indices.size
        // Fewer than 8 indices would be passed in constant memory.
        precondition(
            indices.ndim == 2 && indices.dtype == .uint32 && rows >= 8
                && (1 ... maximumRouteExperts).contains(experts),
            "route geometry")
        func maximum(_ height: Int) -> Int { (rows + height - 1) / height + min(experts, rows) }
        let (maximumA, maximumB) = (maximum(fc1Rows), maximum(fc2Rows))
        let two = fc1Rows != fc2Rows
        // Per-simdgroup expert counters: 16 simdgroups while they fit in
        // 28 KB of threadgroup memory, else 8.
        let simdgroups = 16 * experts * 4 + experts * 4 <= 28 * 1024 ? 16 : 8
        let keyBits = max(1, Int.bitWidth - (experts - 1).leadingZeroBitCount)
        let result = routeKernel(
            [
                indices.flattened(),
                MLXArray([
                    Int32(rows), Int32(0), Int32(indices.dim(1)), Int32(maximumA),
                    Int32(maximumB),
                ]),
            ],
            template: [
                ("E", experts), ("NSG", simdgroups), ("KB", keyBits), ("BMA", fc1Rows),
                ("BMB", fc2Rows), ("TWO", two),
            ],
            grid: (simdgroups * 32, 1, 1), threadGroup: (simdgroups * 32, 1, 1),
            outputShapes: [[rows], [rows], [maximumA * 4], [1], [two ? maximumB * 4 : 4], [1]],
            outputDTypes: [.uint32, .uint32, .uint32, .uint32, .uint32, .uint32])
        let fc1 = RouteTiles(list: result[2], count: result[3], maximum: maximumA)
        let fc2 =
            two ? RouteTiles(list: result[4], count: result[5], maximum: maximumB) : fc1
        return SortedRoute(order: result[0], inverse: result[1], fc1: fc1, fc2: fc2)
    }

    /// `[T, D]` in `y.dtype` from expert-sorted rows `y` (`[T·K, D]`), the
    /// sorted position of each assignment `inverse` (`[T·K]`) and float32
    /// `scores` (`[T, K]`): NemotronSuperMoEKernels' combine (MLX's small
    /// strided reduction of `(y * scores).sum(axis: -2)`, K below 32) with
    /// each row read through `inverse`.
    static func combine(_ y: MLXArray, inverse: MLXArray, scores: MLXArray) -> MLXArray {
        let (tokens, k) = (scores.dim(0), scores.dim(1))
        let width = y.dim(-1)
        precondition(
            y.shape == [tokens * k, width] && inverse.shape == [tokens * k] && k < 32,
            "sorted combine geometry")
        let threads = min(256, (width + 31) / 32 * 32)
        return combineKernel(
            [y, inverse, scores.asType(.float32)],
            template: [("T", y.dtype), ("K", k), ("D", width)],
            grid: ((width + threads - 1) / threads * threads, tokens, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [[tokens, width]], outputDTypes: [y.dtype])[0]
    }

    private static let combineKernel = MLXFast.metalKernel(
        name: "modelopt_nvfp4_nax_sorted_combine",
        inputNames: ["y", "rowmap", "scores"],
        outputNames: ["out"],
        source: """
            const uint column = thread_position_in_grid.x;
            const uint row = thread_position_in_grid.y;
            if (column >= D) {
                return;
            }
            constexpr int P = K < 8 ? K : 8;
            const auto weights = scores + size_t(row) * K;
            float partial[P];
            for (int p = 0; p < P; p++) {
                partial[p] = 0.0f;
            }
            for (int r = 0; r < K; r++) {
                const float product =
                    static_cast<float>(y[size_t(rowmap[size_t(row) * K + r]) * D + column])
                    * weights[r];
                partial[r % P] = product + partial[r % P];
            }
            float total = partial[0];
            for (int p = 1; p < P; p++) {
                total = partial[p] + total;
            }
            out[size_t(row) * D + column] = static_cast<T>(total);
            """,
        ensureRowContiguous: true)

    private static let routeKernel = MLXFast.metalKernel(
        name: "modelopt_nvfp4_nax_route",
        inputNames: ["idx", "params"],
        outputNames: ["order", "inverse", "tiles_a", "count_a", "tiles_b", "count_b"],
        source: """
            threadgroup uint counts[NSG * E];
            threadgroup uint bases[E];
            threadgroup uint totals[3 * NSG];
            glm5r_route<E, NSG, KB, BMA, BMB, TWO>(
                idx, params, order, inverse, tiles_a, count_a, tiles_b, count_b, counts,
                bases, totals, thread_index_in_threadgroup, simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            """,
        header: routeHeader, ensureRowContiguous: true)

    /// The stable counting sort of `route` in one threadgroup of `NSG`
    /// simdgroups. Simdgroup s owns rows `[s·chunk, (s+1)·chunk)`: it
    /// counts them per expert (threadgroup atomics on its own counters),
    /// then, after the scans, places them 32 at a time in row order;
    /// lanes with the same key find each other with `KB` ballots, so the
    /// order is deterministic: row order within each expert. Keys are
    /// loaded `U` batches ahead of their use.
    private static let routeHeader = #"""
        using namespace metal;

        // Lanes of this simdgroup holding this lane's key (bit l: lane l);
        // 0 for an invalid lane. Keys are below 2^KB.
        template <int KB>
        METAL_FUNC ulong glm5r_peers(const uint key, const bool valid) {
          ulong peers = static_cast<ulong>(static_cast<simd_vote::vote_t>(simd_ballot(valid)));
          for (int b = 0; b < KB; ++b) {
            const bool bit = ((key >> b) & 1u) != 0u;
            const ulong set = static_cast<ulong>(static_cast<simd_vote::vote_t>(simd_ballot(bit)));
            peers &= bit ? set : ~set;
          }
          return valid ? peers : 0ul;
        }

        // Tiles of at most BM rows over [first, first + count) of `expert`,
        // from slot `offset` (at most `max_tiles` slots).
        template <int BM>
        METAL_FUNC void glm5r_tiles(
            device uint32_t* tiles,
            const uint offset,
            const uint first,
            const uint expert,
            const uint count,
            const uint max_tiles) {
          const uint n = (count + uint(BM) - 1u) / uint(BM);
          for (uint j = 0; j < n && offset + j < max_tiles; ++j) {
            *((device uint4*)tiles + offset + j) =
                uint4(first + j * uint(BM), expert, min(uint(BM), count - j * uint(BM)), 0u);
          }
        }

        // U batches of 32 keys of rows [r0, c1) (0xffffffff past c1).
        template <int U>
        METAL_FUNC void glm5r_keys(
            thread uint (&keys)[U],
            const device uint32_t* idx,
            const int r0,
            const int c1,
            const uint lane) {
          for (int u = 0; u < U; ++u) {
            const int r = r0 + u * 32 + int(lane);
            keys[u] = r < c1 ? idx[r] : 0xffffffffu;
          }
        }

        // params: routed rows n, shared rows S, K, gate/up and down tile slots.
        // order [n + S], inverse [n], tiles_* [slots * 4], count_* [1].
        template <int E, int NSG, int KB, int BMA, int BMB, bool TWO>
        METAL_FUNC void glm5r_route(
            const device uint32_t* idx,
            const constant int* params,
            device uint32_t* order,
            device uint32_t* inverse,
            device uint32_t* tiles_a,
            device uint32_t* count_a,
            device uint32_t* tiles_b,
            device uint32_t* count_b,
            threadgroup uint* counts,
            threadgroup uint* bases,
            threadgroup uint* totals,
            const uint lid,
            const uint sg,
            const uint lane) {
          constexpr int NT = NSG * 32;
          constexpr int U = 8;
          const int n = params[0];
          const int shared_rows = params[1];
          const uint top_k = uint(params[2]);
          const uint max_a = uint(params[3]);
          const uint max_b = uint(params[4]);
          const int chunk = (n + NSG - 1) / NSG;
          const int c0 = min(n, int(sg) * chunk);
          const int c1 = min(n, c0 + chunk);
          const ulong below = (1ul << lane) - 1ul;
          const ulong above = ~0ul << (lane + 1u);
          threadgroup uint* own = counts + sg * E;

          for (int i = int(lid); i < NSG * E; i += NT) {
            counts[i] = 0u;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          // 1. Per-simdgroup counts (order-free, so atomics).
          threadgroup atomic_uint* own_count = (threadgroup atomic_uint*)own;
          for (int r0 = c0; r0 < c1; r0 += 32 * U) {
            uint keys[U];
            glm5r_keys<U>(keys, idx, r0, c1, lane);
            for (int u = 0; u < U; ++u) {
              if (keys[u] < uint(E)) {
                atomic_fetch_add_explicit(own_count + keys[u], 1u, memory_order_relaxed);
              }
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          // 2. Per expert: chunk counts become offsets within its run.
          for (int e = int(lid); e < E; e += NT) {
            uint run = 0u;
            for (int s = 0; s < NSG; ++s) {
              const uint c = counts[s * E + e];
              counts[s * E + e] = run;
              run += c;
            }
            bases[e] = run;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          // 3. Expert-major exclusive scans of rows and of both heights'
          //    tiles; the shared rows are expert E at row n.
          const int groups = E + (shared_rows > 0 ? 1 : 0);
          uint rows_run = 0u;
          uint a_run = 0u;
          uint b_run = 0u;
          for (int e0 = 0; e0 < groups; e0 += NT) {
            const int e = e0 + int(lid);
            const uint count = e < E ? bases[e] : (e < groups ? uint(shared_rows) : 0u);
            const uint na = (count + uint(BMA) - 1u) / uint(BMA);
            const uint nb = TWO ? (count + uint(BMB) - 1u) / uint(BMB) : 0u;
            const uint lr = simd_prefix_exclusive_sum(count);
            const uint la = simd_prefix_exclusive_sum(na);
            const uint lb = simd_prefix_exclusive_sum(nb);
            const uint sr = simd_sum(count);
            const uint sa = simd_sum(na);
            const uint sb = simd_sum(nb);
            if (lane == 0) {
              totals[sg] = sr;
              totals[NSG + sg] = sa;
              totals[2 * NSG + sg] = sb;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint pr = 0u, pa = 0u, pb = 0u, tr = 0u, ta = 0u, tb = 0u;
            for (uint s = 0; s < uint(NSG); ++s) {
              const uint vr = totals[s];
              const uint va = totals[NSG + s];
              const uint vb = totals[2 * NSG + s];
              pr += s < sg ? vr : 0u;
              pa += s < sg ? va : 0u;
              pb += s < sg ? vb : 0u;
              tr += vr;
              ta += va;
              tb += vb;
            }
            if (e < groups) {
              const uint first = e < E ? rows_run + pr + lr : uint(n);
              glm5r_tiles<BMA>(tiles_a, a_run + pa + la, first, uint(e), count, max_a);
              if (TWO) {
                glm5r_tiles<BMB>(tiles_b, b_run + pb + lb, first, uint(e), count, max_b);
              }
              if (e < E) {
                bases[e] = first;
              }
            }
            rows_run += tr;
            a_run += ta;
            b_run += tb;
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }
          if (lid == 0) {
            count_a[0] = min(a_run, max_a);
            count_b[0] = TWO ? min(b_run, max_b) : 0u;
          }

          // 4. Each chunk's first sorted position per expert.
          for (int e = int(lid); e < E; e += NT) {
            const uint base = bases[e];
            for (int s = 0; s < NSG; ++s) {
              counts[s * E + e] += base;
            }
          }
          for (int t = int(lid); t < shared_rows; t += NT) {
            order[n + t] = uint(t) * top_k;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          // 5. Stable scatter in row order, writing the inverse directly.
          //    The batch loop bound is uniform per simdgroup (ballots).
          for (int r0 = c0; r0 < c1; r0 += 32 * U) {
            uint keys[U];
            glm5r_keys<U>(keys, idx, r0, c1, lane);
            for (int u = 0; u < U && r0 + u * 32 < c1; ++u) {
              const int r = r0 + u * 32 + int(lane);
              const uint key = keys[u];
              const bool valid = key < uint(E);
              const ulong peers = glm5r_peers<KB>(key, valid);
              uint pos = 0u;
              if (valid) {
                pos = own[key] + uint(popcount(peers & below));
                order[pos] = uint(r);
              }
              if (r < c1) {
                inverse[r] = pos;
              }
              simdgroup_barrier(mem_flags::mem_threadgroup);
              if (valid && (peers & above) == 0ul) {
                own[key] = pos + 1u;
              }
              simdgroup_barrier(mem_flags::mem_threadgroup);
            }
          }
        }
        """#

    // MARK: - Diagnostic

    /// The fused route against the op-by-op route it replaces.
    public struct RouteComparison: CustomStringConvertible, Sendable {
        /// Largest absolute difference to the reference (infinite when NaN
        /// positions differ), with fc2's rounding in fc1's epilogue and as
        /// its own pass.
        public let maxAbs: (epilogue: Float, separate: Float)
        /// Non-NaN elements whose bits differ from the reference.
        public let bitMismatches: (epilogue: Int, separate: Int)
        /// Median wall time per evaluated call.
        public let seconds: (reference: Double, epilogue: Double, separate: Double)
        /// Whether the reference's projections ran this type's kernels
        /// (`isEnabled`) rather than MLX's sorted gather.
        public let referenceUsesNAX: Bool

        public var description: String {
            "fused NAX route: epilogue rounding maxAbs \(maxAbs.epilogue) "
                + "(\(bitMismatches.epilogue) bit mismatches) "
                + "\(String(format: "%.3f", seconds.epilogue * 1e3)) ms, separate rounding maxAbs "
                + "\(maxAbs.separate) (\(bitMismatches.separate) bit mismatches) "
                + "\(String(format: "%.3f", seconds.separate * 1e3)) ms; reference "
                + "(\(referenceUsesNAX ? "NAX" : "MLX") gather) "
                + "\(String(format: "%.3f", seconds.reference * 1e3)) ms"
        }
    }

    /// Runs one MoE layer's routed experts at Super's shapes (by default a
    /// 2,048-token chunk, 22 of 512 experts, latent 1,024, width 2,688) on
    /// random `.saturating` BF16 banks, through `routedExperts` (both fc2
    /// rounding variants) and through the route NemotronH takes without it:
    /// `gatherSort`, fc1 and fc2 `graphForward` on the sorted rows
    /// (`squaredReLU` for fc1), `scatterUnsort` and the weighted sum
    /// `(y * scores).sum(axis: -2)`.
    ///
    /// Each token picks `topK` distinct experts with log-normal popularity of
    /// `skew` (1: about Super's). fc2's input scales sit at half to the full
    /// observed fc1 output range, so some blocks saturate. `nanToken` makes
    /// token 0's input NaN.
    public static func compareRoute(
        tokens: Int = 2048, topK: Int = 22, experts: Int = 512, hidden: Int = 1024,
        intermediate: Int = 2688, seed: UInt64 = 1, skew: Double = 1, nanToken: Bool = false,
        iterations: Int = 10
    ) throws -> RouteComparison {
        let dtype = DType.bfloat16
        let rows = tokens * topK
        guard iterations > 0, skew >= 0, topK <= experts, (1 ..< 32).contains(topK),
            (1 ... maximumRouteExperts).contains(experts), rows >= max(16, 4 * experts),
            supports(
                rows: rows, inputs: hidden, outputs: intermediate, experts: experts, dtype: dtype),
            supports(
                rows: rows, inputs: intermediate, outputs: hidden, experts: experts, dtype: dtype),
            experts * intermediate * (hidden / 2) <= Int(Int32.max)
        else {
            throw Failure.unsupported(
                "tokens \(tokens), topK \(topK), experts \(experts), hidden \(hidden), "
                    + "intermediate \(intermediate), NAX \(naxAvailable)")
        }
        return try withError {
            var random = DiagnosticRandom(seed: seed)
            func bank(outputs: Int, inputs: Int, inputScales: [Float]) throws
                -> ModelOptNVFP4SwitchLinear
            {
                let packed = randomBits(
                    count: experts * outputs * inputs / 2, dtype: .uint8, scales: false,
                    salt: random.word()
                ).reshaped(experts, outputs, inputs / 2)
                let scales = randomBits(
                    count: experts * outputs * inputs / 16, dtype: .uint8, scales: true,
                    salt: random.word()
                ).reshaped(experts, outputs, inputs / 16)
                let weights = try (0 ..< experts).map { e in
                    try ModelOptNVFP4Weights(
                        packedBytes: packed[e], e4m3ScaleBytes: scales[e],
                        tensorScale: Float(0.001 + 0.01 * random.unit()))
                }
                let layer = try ModelOptNVFP4SwitchLinear(
                    experts: weights, inputScales: inputScales, weightDType: dtype,
                    activationOverflow: .saturating)
                try checkedEval(layer.weight, layer.blockScales)
                return layer
            }

            // Router-like assignments: K distinct experts per token, by
            // Gumbel top-k over log-normal popularity.
            let popularity = (0 ..< experts).map { _ in skew * random.gaussian() }
            var ids: [UInt32] = []
            ids.reserveCapacity(rows)
            for _ in 0 ..< tokens {
                let keys = popularity.map { $0 - log(-log(max(random.unit(), 1e-300))) }
                let chosen = keys.indices.sorted { keys[$0] > keys[$1] }.prefix(topK)
                ids += chosen.map { UInt32($0) }
            }
            let indices = MLXArray(ids).reshaped(tokens, topK)
            let scores = MLXArray((0 ..< rows).map { _ in Float(0.01 + 0.5 * random.unit()) })
                .reshaped(tokens, topK)
            var x = MLXRandom.normal([tokens, hidden], dtype: dtype, key: MLXRandom.key(seed))
            if nanToken {
                let mask = MLXArray((0 ..< tokens).map { $0 == 0 }).reshaped(tokens, 1)
                x = which(mask, MLXArray(Float.nan).asType(dtype), x)
            }

            let fc1 = try bank(
                outputs: intermediate, inputs: hidden,
                inputScales: (0 ..< experts).map { _ in Float(4.5 / 2688 * (1 + random.unit())) })
            let (sx, si, inverse) = gatherSort(
                x: expandedDimensions(x, axes: [-2, -3]), indices: indices)
            let probe = try fc1.graphForward(
                sx, si, sortedIndices: true, squaredReLU: true, indicesInRange: true
            ).value
            let finite = which(isNaN(probe), MLXArray(Float(0)), abs(probe.asType(.float32)))
            let range = finite.max()
            try checkedEval(range)
            let amax = max(range.item(Float.self), 1e-3)
            let fc2 = try bank(
                outputs: hidden, inputs: intermediate,
                inputScales: (0 ..< experts).map { _ in
                    Float(Double(amax) / 2688 * (0.5 + 0.5 * random.unit()))
                })

            func reference() throws -> MLXArray {
                let h = try fc1.graphForward(
                    sx, si, sortedIndices: true, squaredReLU: true, indicesInRange: true
                ).value
                let y = try fc2.graphForward(h, si, sortedIndices: true, indicesInRange: true)
                    .value
                let unsorted = squeezed(
                    scatterUnsort(x: y, invOrder: inverse, shape: indices.shape), axis: -2)
                return (unsorted * scores[.ellipsis, .newAxis]).sum(axis: -2)
                    .asType(unsorted.dtype)
            }
            func fused(_ epilogue: Bool) -> MLXArray {
                routedExperts(
                    x, indices: indices, scores: scores, fc1: fc1, fc2: fc2,
                    roundsInEpilogue: epilogue)
            }
            func seconds(_ body: () throws -> MLXArray) throws -> Double {
                for _ in 0 ..< 2 { try checkedEval(try body()) }
                var times: [Double] = []
                for _ in 0 ..< iterations {
                    let start = DispatchTime.now().uptimeNanoseconds
                    try checkedEval(try body())
                    times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
                }
                return times.sorted()[times.count / 2]
            }
            func compare(_ actual: MLXArray, _ expected: MLXArray) throws -> (Float, Int) {
                let a = actual.asType(.float32)
                let b = expected.asType(.float32)
                let nanA = isNaN(a)
                let nanB = isNaN(b)
                let either = logicalOr(nanA, nanB)
                let maxAbs = which(either, MLXArray(Float(0)), abs(a - b)).max()
                let nanMismatch = notEqual(nanA, nanB).any()
                let bits = logicalAnd(
                    logicalNot(either),
                    notEqual(actual.view(dtype: .uint16), expected.view(dtype: .uint16))
                ).asType(.int32).sum()
                try checkedEval(maxAbs, nanMismatch, bits)
                return (
                    nanMismatch.item(Bool.self) ? .infinity : maxAbs.item(Float.self),
                    Int(bits.item(Int32.self))
                )
            }

            let expected = try reference()
            try checkedEval(expected)
            let epilogue = try compare(fused(true), expected)
            let separate = try compare(fused(false), expected)
            return RouteComparison(
                maxAbs: (epilogue.0, separate.0), bitMismatches: (epilogue.1, separate.1),
                seconds: (
                    reference: try seconds(reference), epilogue: try seconds { fused(true) },
                    separate: try seconds { fused(false) }
                ),
                referenceUsesNAX: isEnabled)
        }
    }
}
