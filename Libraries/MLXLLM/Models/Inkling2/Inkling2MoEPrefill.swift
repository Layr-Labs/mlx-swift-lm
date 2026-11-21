// Inkling 2 routed experts for prefill on the M5 neural accelerators.
// Reference: transformers PR 51050 `Inkling2LoopExperts` (Apache-2.0).
import Foundation
import MLX

/// Prefill experts as grouped GEMMs on the M5 neural accelerators.
///
/// A plan groups the (token, slot) pairs by expert into tiles of at most
/// `Tile.rows` pairs. Gate/up reads each pair's token row by index and applies
/// the softcapped SwiGLU in its epilogue; down writes each pair's row; the
/// combine sums a token's rows times their route weights in FP32 in ascending
/// expert order (the reference's `index_add_` order) and rounds to BF16.
///
/// Weights are dequantized exactly to BF16 tiles (E2M1 x E4M3 and E4M3 x E8M0
/// products are representable); the NVFP4 per-expert global scale multiplies
/// the FP32 accumulator before the BF16 rounding. A pair's result depends only
/// on its own token row and expert, so outputs do not change with chunking or
/// with the other tokens of the call. Not bit-equal to the sorted
/// `gather_qmm` route, which folds the global scale into BF16 weights.
enum Inkling2MoEPrefill {
    /// `DARKBLOOM_INKLING2_MOE_PREFILL=0` keeps the sorted `gather_qmm` route.
    static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_MOE_PREFILL"]?
            .lowercased()
        return raw.map { !["0", "false", "off", "no"].contains($0) } ?? true
    }()

    /// Test hook: overrides `enabled` for the current task.
    @TaskLocal static var override: Bool? = nil

    /// The kernels need the M5 neural accelerators.
    static var active: Bool {
        (override ?? enabled) && Inkling2PagedAttention.neuralAcceleratorsAvailable
    }

    /// Threadgroup tile of the grouped GEMMs.
    struct Tile: Hashable, Sendable {
        /// Pairs of one expert per tile.
        var rows = 128
        /// Output columns per tile.
        var columns = 64
        /// K per weight tile.
        var depth = 64
        /// Simdgroups along the rows and the columns.
        var rowGroups = 4
        var columnGroups = 2
        /// Padding of each weight-tile row in threadgroup memory (elements).
        var padding = 16

        var threads: Int { 32 * rowGroups * columnGroups }

        var valid: Bool {
            rows % (16 * rowGroups) == 0 && columns % (32 * columnGroups) == 0
                && depth % 32 == 0 && depth >= 32 && padding % 8 == 0
                && threads <= 1024 && 2 * columns * (depth + padding) * 2 <= 32 * 1024
        }
    }

    /// Bench and test hook: overrides `defaultTile` for the current task.
    @TaskLocal static var tileOverride: Tile? = nil

    /// The tuned tile, or `DARKBLOOM_INKLING2_MOE_PREFILL_TILE` =
    /// `rows,columns,depth,rowGroups,columnGroups,padding` when it is valid.
    static let defaultTile: Tile = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_MOE_PREFILL_TILE"]
        let v = raw?.split(separator: ",").compactMap { Int($0) } ?? []
        guard v.count == 6 else { return Tile() }
        let tile = Tile(
            rows: v[0], columns: v[1], depth: v[2], rowGroups: v[3], columnGroups: v[4],
            padding: v[5])
        return tile.valid ? tile : Tile()
    }()

    /// One expert bank as the kernels read it.
    struct Bank {
        let format: Inkling2ExpertFormat
        let weight: MLXArray
        let scales: MLXArray?
        let globalScale: MLXArray?
        let experts: Int
        let outputs: Int
        let inputs: Int

        init?(_ bank: Inkling2ExpertBank) {
            if let q = bank as? Inkling2QuantizedExpertBank {
                format = q.format
                weight = q.weight
                scales = q.scales
                globalScale = q.globalScale
            } else if let d = bank as? Inkling2DenseExpertBank {
                format = .dense
                weight = d.weight
                scales = nil
                globalScale = nil
            } else {
                return nil
            }
            experts = bank.experts
            outputs = bank.outputDimensions
            inputs = bank.inputDimensions
        }

        var formatCode: Int {
            switch format {
            case .dense: 0
            case .nvfp4: 1
            case .mxfp8: 2
            }
        }

        func stored(dtype: DType) -> Bool {
            guard weight.ndim == 3, weight.dim(0) == experts, weight.dim(1) == outputs else {
                return false
            }
            switch format {
            case .dense:
                return weight.dtype == dtype && weight.dim(2) == inputs
            case .nvfp4:
                return weight.dtype == .uint32 && weight.dim(2) * 8 == inputs
                    && scales?.dtype == .uint8 && scales?.shape == [experts, outputs, inputs / 16]
                    && globalScale?.dtype == .float32 && globalScale?.size == experts
            case .mxfp8:
                return weight.dtype == .uint32 && weight.dim(2) * 4 == inputs
                    && scales?.dtype == .uint8 && scales?.shape == [experts, outputs, inputs / 32]
            }
        }
    }

    /// Whether the kernels take `rows` tokens of width `hidden` in `dtype`.
    static func covers(
        rows: Int, hidden: Int, dtype: DType, topK: Int, gateUp: Bank, down: Bank, tile: Tile
    ) -> Bool {
        let intermediate = gateUp.outputs / 2
        return active && tile.valid && dtype == .bfloat16 && rows > 0 && topK > 0
            && topK <= gateUp.experts && gateUp.experts <= 1024
            && rows * topK <= Int(Int32.max) / max(gateUp.outputs, hidden)
            && gateUp.format == down.format && gateUp.experts == down.experts
            && gateUp.inputs == hidden && down.outputs == hidden && down.inputs == intermediate
            && gateUp.outputs % 2 == 0 && gateUp.outputs % tile.columns == 0
            && hidden % tile.columns == 0 && hidden % tile.depth == 0
            && intermediate % tile.depth == 0 && hidden % 8 == 0
            && gateUp.stored(dtype: dtype) && down.stored(dtype: dtype)
    }

    /// `Inkling2SparseMoE` for a prefill call: the module's router, then the
    /// grouped kernels. Nil when the kernels are off or do not cover the call.
    static func sparseMoE(_ moe: Inkling2SparseMoE, _ x: MLXArray) -> MLXArray? {
        guard active, let gateUp = Bank(moe.experts.gateUp), let down = Bank(moe.experts.down)
        else { return nil }
        let shape = x.shape
        let flat = x.reshaped(-1, shape.last!)
        let tile = tileOverride ?? defaultTile
        guard
            covers(
                rows: flat.dim(0), hidden: flat.dim(1), dtype: x.dtype, topK: moe.router.topK,
                gateUp: gateUp, down: down, tile: tile)
        else { return nil }
        let (indices, weights) = moe.router(flat)
        Inkling2Diagnostics.section?("router", [indices, weights])
        return experts(
            flat, indices: indices, weights: weights, gateUp: gateUp, down: down,
            gateCap: moe.gateSoftcap, upCap: moe.upSoftcap, tile: tile
        ).reshaped(shape)
    }

    /// The routed experts of `x` `[R, H]` for `indices` `[R, topK]` (distinct
    /// per row) and FP32 route `weights` `[R, topK]`; `[R, H]` in `x.dtype`.
    /// The caller checks `covers`.
    static func experts(
        _ x: MLXArray, indices: MLXArray, weights: MLXArray, gateUp: Bank, down: Bank,
        gateCap: Float, upCap: Float, tile: Tile
    ) -> MLXArray {
        let (rows, hidden, topK) = (x.dim(0), x.dim(1), indices.dim(1))
        let pairs = rows * topK
        let experts = gateUp.experts
        let maximumTiles = pairs / tile.rows + min(experts, pairs)
        let flatIndices = (indices.dtype == .uint32 ? indices : indices.asType(.uint32))
            .reshaped(pairs)

        let plan = launches.launch(
            Key(
                kind: 0, format: 0, n: 0, k: 0, experts: experts, topK: topK, rows: rows, tile: tile
            )
        ) {
            planKernel.prepare(
                template: [("E", experts), ("BM", tile.rows), ("MAXT", maximumTiles)],
                grid: (1024, 1, 1), threadGroup: (1024, 1, 1),
                outputShapes: [[maximumTiles * 4], [pairs]], outputDTypes: [.int32, .int32])
        }([flatIndices])
        Inkling2Diagnostics.section?("moe-sort", plan)

        func product(_ input: MLXArray, _ bank: Bank, up: Bool) -> MLXArray {
            let (n, k) = (bank.outputs, bank.inputs)
            let key = Key(
                kind: up ? 1 : 2, format: bank.formatCode, n: n, k: k, experts: experts,
                topK: topK, rows: rows, tile: tile)
            let launch = launches.launch(key) {
                gemmKernel.prepare(
                    template: [
                        ("T", input.dtype), ("FMT", bank.formatCode), ("N", n), ("K", k),
                        ("E", experts), ("TOPK", topK), ("BM", tile.rows), ("BN", tile.columns),
                        ("BK", tile.depth), ("WM", tile.rowGroups), ("WN", tile.columnGroups),
                        ("PAD", tile.padding), ("RELAX", true), ("UP", up),
                    ],
                    grid: (n / tile.columns * tile.threads, maximumTiles, 1),
                    threadGroup: (tile.threads, 1, 1),
                    outputShapes: [[pairs, up ? n / 2 : n]], outputDTypes: [input.dtype])
            }
            return launch([
                input, plan[0], plan[1], bank.weight, bank.scales ?? constants.bytes,
                bank.globalScale ?? constants.ones, constants.caps(gateCap, upCap),
            ])[0]
        }

        let activated = product(x, gateUp, up: true)
        Inkling2Diagnostics.section?("moe-gate-up+act", [activated])
        let projected = product(activated, down, up: false)
        Inkling2Diagnostics.section?("moe-down", [projected])

        let combine = launches.launch(
            Key(kind: 3, format: 0, n: hidden, k: 0, experts: 0, topK: topK, rows: rows, tile: tile)
        ) {
            combineKernel.prepare(
                template: [("T", x.dtype), ("N", hidden), ("TOPK", topK)],
                grid: ((hidden / 4 + 63) / 64 * 64, rows, 1), threadGroup: (64, 1, 1),
                outputShapes: [[rows, hidden]], outputDTypes: [x.dtype])
        }
        let w = (weights.dtype == .float32 ? weights : weights.asType(.float32)).reshaped(pairs)
        return combine([projected, flatIndices, w])[0]
    }

    // MARK: - Launches

    struct Key: Hashable {
        let kind: Int
        let format: Int
        let n: Int
        let k: Int
        let experts: Int
        let topK: Int
        let rows: Int
        let tile: Tile
    }

    final class Launches: @unchecked Sendable {
        private let lock = NSLock()
        private var launches: [Key: MLXFast.MLXFastKernel.Launch] = [:]
        /// Row counts vary with chunking; start over rather than keep every one.
        static let capacity = 1024

        func launch(_ key: Key, _ make: () -> MLXFast.MLXFastKernel.Launch)
            -> MLXFast.MLXFastKernel.Launch
        {
            lock.withLock {
                if let l = launches[key] { return l }
                if launches.count >= Self.capacity { launches.removeAll() }
                let l = make()
                launches[key] = l
                return l
            }
        }
    }

    static let launches = Launches()

    final class Constants: @unchecked Sendable {
        private let lock = NSLock()
        private var caps: [UInt64: MLXArray] = [:]
        // Eight elements: smaller inputs bind in the constant address space.
        let bytes = MLXArray([UInt8](repeating: 0, count: 8))
        let ones = MLXArray([Float](repeating: 1, count: 8))

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

    static let constants = Constants()

    // MARK: - Kernels

    /// One threadgroup of 1024 threads groups the pairs by expert: `order`
    /// lists the pairs by ascending expert, `tiles` has one (expert, first,
    /// count, 0) row per tile of at most BM pairs, expert -1 past the last.
    static let planKernel = MLXFast.metalKernel(
        name: "inkling2_moe_prefill_plan", inputNames: ["idx"], outputNames: ["tiles", "order"],
        source: #"""
            threadgroup atomic_int counts[E];
            threadgroup int first[E];
            threadgroup int tile0[E];
            threadgroup int ntiles;
            constexpr int EPL = (E + 31) / 32;
            const int tid = int(thread_index_in_threadgroup);
            const int nt = int(threads_per_threadgroup.x);
            const int P = int(idx_shape[0]);
            for (int e = tid; e < E; e += nt) atomic_store_explicit(&counts[e], 0, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = tid; i < P; i += nt) atomic_fetch_add_explicit(&counts[idx[i]], 1, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simdgroup_index_in_threadgroup == 0) {
                const int lane = int(thread_index_in_simdgroup);
                int tot[EPL];
                int pairs = 0;
                int chunks = 0;
                for (int j = 0; j < EPL; j++) {
                    const int x = lane * EPL + j;
                    tot[j] = x < E ? atomic_load_explicit(&counts[x], memory_order_relaxed) : 0;
                    pairs += tot[j];
                    chunks += (tot[j] + BM - 1) / BM;
                }
                int at = simd_prefix_exclusive_sum(pairs);
                int item = simd_prefix_exclusive_sum(chunks);
                for (int j = 0; j < EPL; j++) {
                    const int x = lane * EPL + j;
                    if (x < E) {
                        first[x] = at;
                        tile0[x] = item;
                        for (int q = 0; q * BM < tot[j]; q++) {
                            tiles[(item + q) * 4] = x;
                            tiles[(item + q) * 4 + 1] = at + q * BM;
                            tiles[(item + q) * 4 + 2] = min(BM, tot[j] - q * BM);
                            tiles[(item + q) * 4 + 3] = 0;
                        }
                        at += tot[j];
                        item += (tot[j] + BM - 1) / BM;
                    }
                }
                if (lane == 31) ntiles = item;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int e = tid; e < E; e += nt) atomic_store_explicit(&counts[e], 0, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Order inside an expert is not fixed: every pair's result depends only on its own row.
            for (int i = tid; i < P; i += nt) {
                const uint e = idx[i];
                order[first[e] + atomic_fetch_add_explicit(&counts[e], 1, memory_order_relaxed)] = i;
            }
            for (int row = ntiles + tid; row < MAXT; row += nt) {
                tiles[row * 4] = -1;
                tiles[row * 4 + 1] = 0;
                tiles[row * 4 + 2] = 0;
                tiles[row * 4 + 3] = 0;
            }
            """#)

    /// Grouped expert product on the neural accelerators, one threadgroup per
    /// (column tile, row tile), weight tiles dequantized into double-buffered
    /// threadgroup memory. UP: rows are token rows read by index, epilogue is
    /// the softcapped SwiGLU on interleaved gate/up columns, written in plan
    /// order `[P, N/2]`. Down: rows in plan order, each pair's row written at
    /// the pair's index `[P, N]`.
    static let gemmKernel = MLXFast.metalKernel(
        name: "inkling2_moe_prefill_gemm",
        inputNames: ["x", "tiles", "order", "w", "sc", "gs", "caps"], outputNames: ["out"],
        source: #"""
            constexpr int NSG = WM * WN;
            constexpr int NT = NSG * 32;
            constexpr int SM = BM / WM;
            constexpr int SN = BN / WN;
            constexpr int TM = SM / 16;
            constexpr int TN = SN / 16;
            constexpr int LDW = BK + PAD;
            constexpr int UPT = (BN * BK / moe::unit_values<FMT>() + NT - 1) / NT;
            constexpr int KB = K / BK;
            threadgroup T ws[2 * BN * LDW];
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint tid = thread_index_in_threadgroup;
            const int item = int(threadgroup_position_in_grid.y);
            const int expert = tiles[item * 4];
            if (expert < 0) return;
            const int start = tiles[item * 4 + 1];
            const int count = tiles[item * 4 + 2];
            const int n0 = int(threadgroup_position_in_grid.x) * BN;
            const int sm = int(sg) / WN;
            const int sn = int(sg) % WN;
            const int r0 = sm * SM;
            const int vf = clamp((count - r0 + 15) / 16, 0, TM);
            const short2 fc = moe::coord(lane);
            // Rows past `count` read the tile's first row; their results are never stored.
            const device T* arow[TM * 2];
            _Pragma("clang loop unroll(full)")
            for (int f = 0; f < TM; f++) {
                _Pragma("clang loop unroll(full)")
                for (int i = 0; i < 2; i++) {
                    const int r = r0 + f * 16 + fc.y + i * 8;
                    const int rr = r < count ? r : 0;
                    const int src = UP ? order[start + rr] / TOPK : start + rr;
                    arow[f * 2 + i] = x + size_t(src) * K + fc.x;
                }
            }
            vec<float, 8> acc[TM][TN];
            _Pragma("clang loop unroll(full)")
            for (int f = 0; f < TM; f++) {
                _Pragma("clang loop unroll(full)")
                for (int t = 0; t < TN; t++) acc[f][t] = 0.0f;
            }
            const device uchar* wb = (const device uchar*)w;
            moe::wunit u[UPT];
            moe::fetch<FMT, BN, BK, NT, N, K, UPT>(u, wb, sc, expert, n0, 0, tid);
            moe::store<T, FMT, BN, BK, LDW, NT, UPT>(ws, u, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int kb = 0; kb < KB; kb++) {
                const threadgroup T* cur = ws + (kb & 1) * (BN * LDW);
                threadgroup T* nxt = ws + ((kb + 1) & 1) * (BN * LDW);
                const bool more = kb + 1 < KB;
                if (more) moe::fetch<FMT, BN, BK, NT, N, K, UPT>(u, wb, sc, expert, n0, (kb + 1) * BK, tid);
                const int k0 = kb * BK;
                // Full tiles skip the per-fragment checks; partial tiles skip empty fragments.
                if (vf == TM) {
                    _Pragma("clang loop unroll(disable)")
                    for (int kk = 0; kk < BK; kk += 32) {
                        volatile int compiler_barrier;
                        vec<T, 8> a[TM][2];
                        vec<T, 8> b[TN][2];
                        _Pragma("clang loop unroll(full)")
                        for (int f = 0; f < TM; f++) {
                            {
                                _Pragma("clang loop unroll(full)")
                                for (int h = 0; h < 2; h++) {
                                    _Pragma("clang loop unroll(full)")
                                    for (int i = 0; i < 2; i++) {
                                        const vec<T, 4> v = *(const device vec<T, 4>*)(arow[f * 2 + i] + k0 + kk + h * 16);
                                        _Pragma("clang loop unroll(full)")
                                        for (int j = 0; j < 4; j++) a[f][h][i * 4 + j] = v[j];
                                    }
                                }
                            }
                        }
                        _Pragma("clang loop unroll(full)")
                        for (int t = 0; t < TN; t++) {
                            const threadgroup T* wp = cur + (sn * SN + t * 16 + fc.y) * LDW + kk + fc.x;
                            _Pragma("clang loop unroll(full)")
                            for (int h = 0; h < 2; h++) {
                                _Pragma("clang loop unroll(full)")
                                for (int i = 0; i < 2; i++) {
                                    const vec<T, 4> p = *(const threadgroup vec<T, 4>*)(wp + i * 8 * LDW + h * 16);
                                    _Pragma("clang loop unroll(full)")
                                    for (int j = 0; j < 4; j++) b[t][h][i * 4 + j] = p[j];
                                }
                            }
                        }
                        _Pragma("clang loop unroll(full)")
                        for (int f = 0; f < TM; f++) {
                            {
                                _Pragma("clang loop unroll(full)")
                                for (int t = 0; t < TN; t += 2) {
                                    _Pragma("clang loop unroll(full)")
                                    for (int h = 0; h < 2; h++) {
                                        moe::mma<T, RELAX>(acc[f][t], acc[f][t + 1], a[f][h], b[t][h], b[t + 1][h]);
                                    }
                                }
                            }
                        }
                        (void)compiler_barrier;
                    }
                } else if (vf > 0) {
                    _Pragma("clang loop unroll(disable)")
                    for (int kk = 0; kk < BK; kk += 32) {
                        volatile int compiler_barrier;
                        vec<T, 8> a[TM][2];
                        vec<T, 8> b[TN][2];
                        _Pragma("clang loop unroll(full)")
                        for (int f = 0; f < TM; f++) {
                            if (f < vf) {
                                _Pragma("clang loop unroll(full)")
                                for (int h = 0; h < 2; h++) {
                                    _Pragma("clang loop unroll(full)")
                                    for (int i = 0; i < 2; i++) {
                                        const vec<T, 4> v = *(const device vec<T, 4>*)(arow[f * 2 + i] + k0 + kk + h * 16);
                                        _Pragma("clang loop unroll(full)")
                                        for (int j = 0; j < 4; j++) a[f][h][i * 4 + j] = v[j];
                                    }
                                }
                            }
                        }
                        _Pragma("clang loop unroll(full)")
                        for (int t = 0; t < TN; t++) {
                            const threadgroup T* wp = cur + (sn * SN + t * 16 + fc.y) * LDW + kk + fc.x;
                            _Pragma("clang loop unroll(full)")
                            for (int h = 0; h < 2; h++) {
                                _Pragma("clang loop unroll(full)")
                                for (int i = 0; i < 2; i++) {
                                    const vec<T, 4> p = *(const threadgroup vec<T, 4>*)(wp + i * 8 * LDW + h * 16);
                                    _Pragma("clang loop unroll(full)")
                                    for (int j = 0; j < 4; j++) b[t][h][i * 4 + j] = p[j];
                                }
                            }
                        }
                        _Pragma("clang loop unroll(full)")
                        for (int f = 0; f < TM; f++) {
                            if (f < vf) {
                                _Pragma("clang loop unroll(full)")
                                for (int t = 0; t < TN; t += 2) {
                                    _Pragma("clang loop unroll(full)")
                                    for (int h = 0; h < 2; h++) {
                                        moe::mma<T, RELAX>(acc[f][t], acc[f][t + 1], a[f][h], b[t][h], b[t + 1][h]);
                                    }
                                }
                            }
                        }
                        (void)compiler_barrier;
                    }
                }
                if (more) moe::store<T, FMT, BN, BK, LDW, NT, UPT>(nxt, u, tid);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            const float inv = FMT == 1 ? gs[expert] / 2688.0f : 1.0f;
            _Pragma("clang loop unroll(full)")
            for (int f = 0; f < TM; f++) {
                if (f >= vf) continue;
                _Pragma("clang loop unroll(full)")
                for (int i = 0; i < 2; i++) {
                    const int r = r0 + f * 16 + fc.y + i * 8;
                    if (r >= count) continue;
                    if (UP) {
                        device T* orow = out + size_t(start + r) * (N / 2);
                        const float gc = caps[0];
                        const float uc = caps[1];
                        _Pragma("clang loop unroll(full)")
                        for (int t = 0; t < TN; t++) {
                            const int n = n0 + sn * SN + t * 16 + fc.x;
                            const float g0 = float(static_cast<T>(acc[f][t][i * 4 + 0] * inv));
                            const float u0 = float(static_cast<T>(acc[f][t][i * 4 + 1] * inv));
                            const float g1 = float(static_cast<T>(acc[f][t][i * 4 + 2] * inv));
                            const float u1 = float(static_cast<T>(acc[f][t][i * 4 + 3] * inv));
                            vec<T, 2> o;
                            o[0] = static_cast<T>(moe::act(g0, u0, gc, uc));
                            o[1] = static_cast<T>(moe::act(g1, u1, gc, uc));
                            *(device vec<T, 2>*)(orow + n / 2) = o;
                        }
                    } else {
                        device T* orow = out + size_t(order[start + r]) * N;
                        _Pragma("clang loop unroll(full)")
                        for (int t = 0; t < TN; t++) {
                            const int n = n0 + sn * SN + t * 16 + fc.x;
                            vec<T, 4> o;
                            _Pragma("clang loop unroll(full)")
                            for (int j = 0; j < 4; j++) o[j] = static_cast<T>(acc[f][t][i * 4 + j] * inv);
                            *(device vec<T, 4>*)(orow + n) = o;
                        }
                    }
                }
            }
            """#,
        header: header)

    /// Each token's rows times their route weights, summed in FP32 in ascending
    /// expert order from zero, rounded to T. Four columns per thread.
    static let combineKernel = MLXFast.metalKernel(
        name: "inkling2_moe_prefill_combine", inputNames: ["y", "idx", "wts"],
        outputNames: ["out"],
        source: #"""
            #pragma clang fp contract(off)
            const int t = int(thread_position_in_grid.y);
            const int n = int(thread_position_in_grid.x) * 4;
            if (n >= N) return;
            uint e[TOPK];
            int ord[TOPK];
            _Pragma("clang loop unroll(full)")
            for (int s = 0; s < TOPK; s++) e[s] = idx[t * TOPK + s];
            _Pragma("clang loop unroll(full)")
            for (int s = 0; s < TOPK; s++) {
                int rank = 0;
                _Pragma("clang loop unroll(full)")
                for (int q = 0; q < TOPK; q++) rank += (e[q] < e[s] || (e[q] == e[s] && q < s)) ? 1 : 0;
                ord[rank] = s;
            }
            float4 sum = 0.0f;
            for (int k = 0; k < TOPK; k++) {
                const int s = ord[k];
                const vec<T, 4> v = *(const device vec<T, 4>*)(y + size_t(t * TOPK + s) * N + n);
                const float wt = wts[t * TOPK + s];
                float4 p;
                p.x = float(v[0]) * wt;
                p.y = float(v[1]) * wt;
                p.z = float(v[2]) * wt;
                p.w = float(v[3]) * wt;
                sum = sum + p;
            }
            vec<T, 4> o;
            o[0] = static_cast<T>(sum.x);
            o[1] = static_cast<T>(sum.y);
            o[2] = static_cast<T>(sum.z);
            o[3] = static_cast<T>(sum.w);
            *(device vec<T, 4>*)(out + size_t(t) * N + n) = o;
            """#)

    /// Ends with a newline: MLX appends the kernel signature right after it.
    static let header = #"""
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        namespace moe {
        using namespace metal;

        // 16x16 fragment layout of matmul2d's cooperative tensors: lane holds rows fm, fm + 8 and
        // columns fn .. fn + 3; element i is row (i >> 2) * 8, column i % 4.
        inline short2 coord(uint lane) {
            const short qid = short(lane >> 2);
            const short fm = (qid & 4) | short((lane >> 1) & 3);
            const short fn = short(((qid & 2) | short(lane & 1)) * 4);
            return short2(fn, fm);
        }

        // C[16 x 32] += A[16 x 16] * B^T, B stored [n][k] as two 16 x 16 fragments (n 0-15, 16-31).
        template <typename AT, bool RELAXED>
        inline void mma(thread vec<float, 8>& c0, thread vec<float, 8>& c1, thread const vec<AT, 8>& a,
                        thread const vec<AT, 8>& b0, thread const vec<AT, 8>& b1) {
            constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
                16, 32, 16, false, true, RELAXED, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
            mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
            auto ta = op.template get_left_input_cooperative_tensor<AT, AT, float>();
            auto tb = op.template get_right_input_cooperative_tensor<AT, AT, float>();
            auto tc = op.template get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(ta)>, metal::remove_addrspace_t<decltype(tb)>, float>();
            _Pragma("clang loop unroll(full)")
            for (short i = 0; i < 8; i++) { ta[i] = a[i]; tb[i] = b0[i]; tb[8 + i] = b1[i]; tc[i] = c0[i]; tc[8 + i] = c1[i]; }
            op.run(ta, tb, tc);
            _Pragma("clang loop unroll(full)")
            for (short i = 0; i < 8; i++) { c0[i] = tc[i]; c1[i] = tc[8 + i]; }
        }

        // E4M3 byte -> float (exact; no NaN codes in the checkpoint).
        inline float e4m3(uint b) {
            const uint e = (b >> 3) & 15u;
            const uint m = b & 7u;
            const float v = e != 0u ? as_type<float>(((e + 120u) << 23) | (m << 20)) : float(m) * 0.001953125f;
            return (b & 0x80u) != 0u ? -v : v;
        }

        // Two bf16 values (exact products) packed little end first.
        inline uint pack2(float lo, float hi) {
            return (as_type<uint>(hi) & 0xFFFF0000u) | (as_type<uint>(lo) >> 16);
        }

        // One unit of a weight tile: NVFP4 16 values (8 bytes + E4M3 scale), MXFP8 16 values
        // (16 bytes + E8M0 scale of their 32-value group), dense 8 values (16 bytes).
        struct wunit {
            uint4 d;
            uint s;
        };

        template <int FMT>
        constexpr int unit_values() { return FMT == 0 ? 8 : 16; }

        // Device -> registers for W[expert, n0 .. n0 + BN, k0 .. k0 + BK].
        template <int FMT, int BN, int BK, int NT, int N, int K, int UPT>
        inline void fetch(thread wunit* u, const device uchar* w, const device uchar* sc, int expert, int n0,
                          int k0, uint tid) {
            constexpr int UPR = BK / unit_values<FMT>();
            _Pragma("clang loop unroll(full)")
            for (int j = 0; j < UPT; j++) {
                const int q = int(tid) + j * NT;
                if (BN * UPR % NT != 0 && q >= BN * UPR) break;
                const int row = q / UPR;
                const int c = q % UPR;
                const size_t grow = size_t(expert) * N + size_t(n0 + row);
                if (FMT == 1) {
                    const uint2 v = *(const device uint2*)(w + grow * (K / 2) + size_t(k0 / 2 + c * 8));
                    u[j].d = uint4(v.x, v.y, 0u, 0u);
                    u[j].s = sc[grow * (K / 16) + size_t(k0 / 16 + c)];
                } else if (FMT == 2) {
                    u[j].d = *(const device uint4*)(w + grow * K + size_t(k0 + c * 16));
                    u[j].s = sc[grow * (K / 32) + size_t((k0 + c * 16) / 32)];
                } else {
                    u[j].d = *(const device uint4*)(w + (grow * K + size_t(k0 + c * 8)) * 2);
                    u[j].s = 0u;
                }
            }
        }

        // Registers -> exact bf16 in ws[n * LDW + k] (block scales applied; NVFP4 global scale left
        // for the epilogue).
        template <typename T, int FMT, int BN, int BK, int LDW, int NT, int UPT>
        inline void store(threadgroup T* ws, thread const wunit* u, uint tid) {
            constexpr int UPR = BK / unit_values<FMT>();
            _Pragma("clang loop unroll(full)")
            for (int j = 0; j < UPT; j++) {
                const int q = int(tid) + j * NT;
                if (BN * UPR % NT != 0 && q >= BN * UPR) break;
                const int row = q / UPR;
                const int c = q % UPR;
                threadgroup uint4* dst = (threadgroup uint4*)(ws + row * LDW + c * unit_values<FMT>());
                if (FMT == 1) {
                    const float s = e4m3(u[j].s);
                    // E2M1 code c (3 bits) as half bits c << 9 is c's value / 2^14.
                    const float s14 = s * 16384.0f;
                    uint p[8];
                    _Pragma("clang loop unroll(full)")
                    for (int b = 0; b < 8; b++) {
                        const uint word = b < 4 ? u[j].d.x : u[j].d.y;
                        const uint sh = (b & 3) * 8;
                        const uint c0 = (word >> sh) & 15u;
                        const uint c1 = (word >> (sh + 4)) & 15u;
                        const float lo = float(as_type<half>(ushort(((c0 & 7u) << 9) | ((c0 & 8u) << 12)))) * s14;
                        const float hi = float(as_type<half>(ushort(((c1 & 7u) << 9) | ((c1 & 8u) << 12)))) * s14;
                        p[b] = pack2(lo, hi);
                    }
                    dst[0] = uint4(p[0], p[1], p[2], p[3]);
                    dst[1] = uint4(p[4], p[5], p[6], p[7]);
                } else if (FMT == 2) {
                    const float s = u[j].s != 0u ? as_type<float>(u[j].s << 23) : 5.877471754111438e-39f;
                    vec<T, 4> o[4];
                    _Pragma("clang loop unroll(full)")
                    for (int q4 = 0; q4 < 4; q4++) {
                        const uint word = q4 == 0 ? u[j].d.x : (q4 == 1 ? u[j].d.y : (q4 == 2 ? u[j].d.z : u[j].d.w));
                        _Pragma("clang loop unroll(full)")
                        for (int b = 0; b < 4; b++) o[q4][b] = static_cast<T>(e4m3((word >> (8 * b)) & 255u) * s);
                    }
                    threadgroup vec<T, 4>* dv = (threadgroup vec<T, 4>*)dst;
                    _Pragma("clang loop unroll(full)")
                    for (int q4 = 0; q4 < 4; q4++) dv[q4] = o[q4];
                } else {
                    dst[0] = u[j].d;
                }
            }
        }

        inline float act(float g, float u, float gc, float uc) {
            const float gated = (metal::precise::tanh(g / gc) * gc) * (1.0f / (1.0f + metal::precise::exp(-g)));
            return gated * (metal::precise::tanh(u / uc) * uc);
        }
        }  // namespace moe

        """#
}
