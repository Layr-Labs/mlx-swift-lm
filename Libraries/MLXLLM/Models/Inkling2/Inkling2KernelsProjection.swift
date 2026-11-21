// Multi-row dense projections for Inkling 2 decode and verify steps.
import Foundation
import MLX

extension Inkling2Kernels {
    /// Rows per threadgroup (eight MMA columns per tile, up to eight tiles).
    static let rowsPerGroup = 64
    /// Inner dimensions up to this size (or fewer than 8 outputs) use one
    /// thread per output.
    static let smallInner = 256
    /// Simdgroups splitting K: a function of K alone, so an output's reduction
    /// order depends on K only (a column of a concatenated weight equals the
    /// separate product).
    static func splitK(_ k: Int) -> Int { min(2, k / 32) }

    /// Split-K of segmented products: eight ways (partials added in order),
    /// so one-row launches have short chains and many lanes in flight. A
    /// function of K only: every row count, alone or in company, gets the same
    /// arithmetic.
    static let smallSplitK = 8
    static func segmentedSplitK(_ k: Int, outputs n: Int) -> Int {
        max(1, min(smallSplitK, k / 32))
    }

    /// Inner dimensions the MMA product takes (multiples of 32).
    static func mmaCovers(outputs n: Int, inputs k: Int) -> Bool {
        k % 32 == 0 && k >= 32 && n >= 8
    }

    /// Simdgroups along N (8 outputs each): scheduling only.
    static func outputSimdgroups(_ n: Int) -> Int {
        n >= 2048 ? 4 : n >= 64 ? 2 : 1
    }

    private static let tiledKernel = MLXFast.metalKernel(
        name: "inkling2_rows_product",
        inputNames: ["x", "w"],
        outputNames: ["out"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const int sg = int(simdgroup_index_in_threadgroup);
            const int ks = sg % SK;
            const int ns = sg / SK;
            const int rows = x_shape[0];
            const int groups = (rows + 63) / 64;
            // Row groups of one output tile are adjacent in the grid, so they run
            // together and share the tile's weights in cache.
            const int group = int(threadgroup_position_in_grid.x) % groups;
            const int tile = int(threadgroup_position_in_grid.x) / groups;
            const int start = group * 64;
            const int count = min(64, rows - start);
            threadgroup float buf[SK * NSG * 64 * MAXRB];
            threadgroup float* mine = buf + (ks * NSG + ns) * 64 * MAXRB;
            // The last tile of N shifts back to stay inside the matrix.
            const int nb = min(tile * 8 * NSG + ns * 8, N - 8);
            const device T* row0 = w + size_t(nb) * K;
            const ink2_rows_contiguous src{start};
            const int rb = count <= 8 ? 1 : count <= 16 ? 2 : count <= 32 ? 4 : 8;
            if (rb == 1) {
                ink2_mma_tile<T, T, 0, K, SK, 1, false, OPT, PF>(x, src, count, row0, nullptr, K, 0, lane, ks, mine);
            } else if (rb == 2) {
                ink2_mma_tile<T, T, 0, K, SK, 2, false, OPT, PF>(x, src, count, row0, nullptr, K, 0, lane, ks, mine);
            } else if (rb == 4) {
                ink2_mma_tile<T, T, 0, K, SK, 4, false, OPT, PF>(x, src, count, row0, nullptr, K, 0, lane, ks, mine);
            } else {
                ink2_mma_tile<T, T, 0, K, SK, 8, false, OPT, PF>(x, src, count, row0, nullptr, K, 0, lane, ks, mine);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Split-K partials added in order; outputs below a simdgroup's first
            // row belong to the previous simdgroup.
            const int cols = rb * 8;
            const int total = NSG * cols * 8;
            for (int e = int(thread_index_in_threadgroup); e < total;
                 e += int(threads_per_threadgroup.x)) {
                const int nl = e % 8;
                const int c = (e / 8) % cols;
                const int sn = e / (8 * cols);
                if (c >= count) {
                    continue;
                }
                const int first = tile * 8 * NSG + sn * 8;
                const int n = min(first, N - 8) + nl;
                if (n < first) {
                    continue;
                }
                const int at = ((c / 8) * 8 + nl) * 8 + (c % 8);
                float v = buf[(0 * NSG + sn) * 64 * MAXRB + at];
                for (int s = 1; s < SK; s++) {
                    v += buf[(s * NSG + sn) * 64 * MAXRB + at];
                }
                out[size_t(start + c) * N + n] = static_cast<T>(v);
            }
            """,
        header: header)

    /// One thread per output (`ink2_row_dot`): the tiled kernel's bits for a
    /// one-row product, without seven idle MMA columns per weight.
    private static let gemvKernel = MLXFast.metalKernel(
        name: "inkling2_rows_gemv",
        inputNames: ["x", "w"],
        outputNames: ["out"],
        source: """
            const int gid = int(thread_position_in_grid.x);
            const int rows = x_shape[0];
            if (gid >= rows * N) {
                return;
            }
            const int r = gid / N;
            const int n = gid % N;
            const float v = ink2_row_dot<T, T, 0, K, SK, OPT>(
                x + size_t(r) * K, w + size_t(n) * K, nullptr);
            out[size_t(r) * N + n] = static_cast<T>(v);
            """,
        header: header)

    /// One-row products with at least this many outputs take `gemvKernel`
    /// (exact operands only); `DARKBLOOM_INKLING2_GEMV=0` keeps the tiles.
    static let gemvEnabled = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_GEMV"] != "0"
    /// Where the one-row GEMVs run: `DARKBLOOM_INKLING2_GEMV_SITES`, comma list
    /// of router, seg, rows, split, up, down (default seg, split, up, down;
    /// with eight-way split-K the segmented and expert GEMVs use eight lanes
    /// per output; rowProduct keeps two chains per thread and loses to its
    /// tiles; the fused router GEMV + select runs in one threadgroup and loses
    /// to the router tile plus the selection kernel, 0.5-0.7 ms per step on
    /// the M3 Ultra).
    static let gemvSites: Set<String> = Set(
        (ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_GEMV_SITES"]
            ?? "seg,split,up,down").split(separator: ",").map(String.init))
    static func gemvSite(_ name: String) -> Bool { gemvEnabled && gemvSites.contains(name) }
    static let gemvMinimumOutputs = 512

    /// One-row products with fewer outputs: SK lanes per output (one split-K
    /// chain each, added in order) for more threads in flight.
    static func gemvSplitCovers(rows: Int, outputs n: Int, inputs k: Int, operand: DType) -> Bool {
        gemvSite("split") && rows == 1 && n < gemvMinimumOutputs && operand != .float32
            && mmaCovers(outputs: n, inputs: k) && 32 % segmentedSplitK(k, outputs: n) == 0
    }

    static func gemvCovers(rows: Int, outputs n: Int, inputs k: Int, operand: DType) -> Bool {
        gemvEnabled && rows == 1 && n >= gemvMinimumOutputs && operand != .float32
            && mmaCovers(outputs: n, inputs: k)
    }

    static func gemvProduct(_ x: MLXArray, _ w: MLXArray, operand opt: DType) -> MLXArray {
        let (rows, n, k) = (x.dim(0), w.dim(0), w.dim(1))
        var key = LaunchKey(kernel: 8, dtype: x.dtype, a: n, b: k, c: rows)
        key.weightType = opt
        let launch = launches.launch(key) {
            gemvKernel.prepare(
                template: [("T", x.dtype), ("N", n), ("K", k), ("SK", splitK(k)), ("OPT", opt)],
                grid: ((rows * n + 255) / 256 * 256, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [[rows, n]], outputDTypes: [x.dtype])
        }
        return launch([x, w])[0]
    }

    private static let smallKernel = MLXFast.metalKernel(
        name: "inkling2_rows_product_small",
        inputNames: ["x", "wt"],
        outputNames: ["out"],
        source: """
            const uint gid = thread_position_in_grid.x;
            const int rows = x_shape[0];
            if (int(gid) >= rows * N) {
                return;
            }
            const int r = int(gid) / N;
            const int n = int(gid) % N;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc = fma(float(x[size_t(r) * K + k]), float(wt[size_t(k) * N + n]), acc);
            }
            out[gid] = static_cast<T>(acc);
            """)

    /// `x @ w.T` for `x` `[..., K]` and `w` `[N, K]` of one float dtype, in that
    /// dtype; nil when the kernels are off or do not cover the shapes.
    public static func rowProduct(_ x: MLXArray, transposedWeight w: MLXArray) -> MLXArray? {
        guard active, w.ndim == 2, x.ndim >= 1, x.dtype == w.dtype,
            supportsActivation(x.dtype)
        else { return nil }
        let (n, k) = (w.dim(0), w.dim(1))
        guard x.dim(-1) == k, n > 0, k > 0 else { return nil }
        let shape = x.shape
        let rows = shape.dropLast().reduce(1, *)
        guard rows > 0 else { return nil }
        let flat = x.reshaped(rows, k)
        let out: MLXArray
        let operand = operandType(x: x.dtype, weight: w.dtype, dense: true)
        if let coded = Inkling2BytesProjection.product(flat, w) {
            out = coded
        } else if gemvSite("rows"), gemvCovers(rows: rows, outputs: n, inputs: k, operand: operand)
        {
            out = gemvProduct(flat, w, operand: operand)
        } else if mmaCovers(outputs: n, inputs: k) {
            out = tiledProduct(flat, w)
        } else if k <= smallInner || n < 8 {
            out = smallProduct(flat, w)
        } else {
            return nil
        }
        return out.reshaped(Array(shape.dropLast()) + [n])
    }

    static func tiledProduct(_ x: MLXArray, _ w: MLXArray) -> MLXArray {
        let (rows, n, k) = (x.dim(0), w.dim(0), w.dim(1))
        let nsg = outputSimdgroups(n)
        let tiles = (n + 8 * nsg - 1) / (8 * nsg)
        let groups = (rows + rowsPerGroup - 1) / rowsPerGroup
        let sk = splitK(k)
        let threads = 32 * nsg * sk
        let opt = operandType(x: x.dtype, weight: w.dtype, dense: true)
        let pf = prefetch(threadgroups: tiles * groups)
        var key = LaunchKey(kernel: 1, dtype: x.dtype, a: n, b: k, c: rows)
        key.weightType = opt
        key.d = pf
        let launch = launches.launch(key) {
            tiledKernel.prepare(
                template: [
                    ("T", x.dtype), ("N", n), ("K", k), ("NSG", nsg), ("SK", sk), ("OPT", opt),
                    ("PF", pf), ("MAXRB", maxRB(rows)),
                ],
                grid: (tiles * groups * threads, 1, 1), threadGroup: (threads, 1, 1),
                outputShapes: [[rows, n]], outputDTypes: [x.dtype])
        }
        return launch([x, w])[0]
    }

    /// One thread per output; reads `w` as `[K, N]` (`w.T`, the stored layout of
    /// a transposed projection bank such as `rel_logits_proj.proj`).
    static func smallProduct(_ x: MLXArray, _ w: MLXArray) -> MLXArray {
        let (rows, n, k) = (x.dim(0), w.dim(0), w.dim(1))
        let total = rows * n
        let key = LaunchKey(kernel: 2, dtype: x.dtype, a: n, b: k, c: rows)
        let launch = launches.launch(key) {
            smallKernel.prepare(
                template: [("T", x.dtype), ("N", n), ("K", k)],
                grid: ((total + 255) / 256 * 256, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [[rows, n]], outputDTypes: [x.dtype])
        }
        return launch([x, w.T])[0]
    }
}
