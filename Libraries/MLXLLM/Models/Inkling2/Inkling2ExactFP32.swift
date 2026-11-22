// Full-FP32 products for Inkling 2 prefill (router logits, message gates,
// residual-mix scores): a tiled kernel with no TF32, whose per-row result does
// not depend on how many rows run together or how a prompt is chunked.
import Foundation
import MLX

enum Inkling2ExactFP32 {
    /// `DARKBLOOM_INKLING2_PREFILL_FP32_KERNEL=0` keeps the per-row gemv in prefill.
    static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_PREFILL_FP32_KERNEL"]?
            .lowercased()
        return raw.map { !["0", "false", "off", "no"].contains($0) } ?? true
    }()

    /// Test hook: overrides `enabled` for the current task.
    @TaskLocal static var override: Bool? = nil
    static var active: Bool { override ?? enabled }

    /// `y[m, n] = sum_k x[m, k] * w[n, k]`, FP32 in and out. Each thread owns a
    /// 4 x 4 block of outputs and folds k = 0, 1, ..., K - 1 in order with one
    /// `fma` per term, so an output depends only on its own row of `x` and of
    /// `w`: tile placement, the row count and zero padding never enter it.
    private static let kernel = MLXFast.metalKernel(
        name: "inkling2_exact_fp32_matmul_t",
        inputNames: ["x", "w"],
        outputNames: ["y"],
        source: """
            constexpr int BM = 64;
            constexpr int BN = 64;
            constexpr int BK = 16;
            threadgroup float xs[BK][BM];
            threadgroup float ws[BK][BN];
            const int M = x_shape[0];
            const int K = x_shape[1];
            const int N = w_shape[0];
            const int tx = thread_position_in_threadgroup.x;
            const int ty = thread_position_in_threadgroup.y;
            const int tid = ty * 16 + tx;
            const int m0 = threadgroup_position_in_grid.y * BM;
            const int n0 = threadgroup_position_in_grid.x * BN;
            float acc[4][4];
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] = 0.0f;
                }
            }
            for (int k0 = 0; k0 < K; k0 += BK) {
                for (int e = tid; e < BM * BK; e += 256) {
                    const int r = e / BK;
                    const int c = e % BK;
                    const int gm = m0 + r;
                    const int gk = k0 + c;
                    xs[c][r] = (gm < M && gk < K) ? x[size_t(gm) * K + gk] : 0.0f;
                }
                for (int e = tid; e < BN * BK; e += 256) {
                    const int r = e / BK;
                    const int c = e % BK;
                    const int gn = n0 + r;
                    const int gk = k0 + c;
                    ws[c][r] = (gn < N && gk < K) ? w[size_t(gn) * K + gk] : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const int kmax = min(BK, K - k0);
                for (int kk = 0; kk < kmax; ++kk) {
                    float a[4];
                    float b[4];
                    for (int i = 0; i < 4; ++i) {
                        a[i] = xs[kk][ty * 4 + i];
                        b[i] = ws[kk][tx * 4 + i];
                    }
                    for (int i = 0; i < 4; ++i) {
                        for (int j = 0; j < 4; ++j) {
                            acc[i][j] = fma(a[i], b[j], acc[i][j]);
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (int i = 0; i < 4; ++i) {
                const int gm = m0 + ty * 4 + i;
                for (int j = 0; j < 4; ++j) {
                    const int gn = n0 + tx * 4 + j;
                    if (gm < M && gn < N) {
                        y[size_t(gm) * N + gn] = acc[i][j];
                    }
                }
            }
            """,
        ensureRowContiguous: true
    )

    /// `DARKBLOOM_INKLING2_PREFILL_FP32_TILES=0` keeps the 64 x 64 tile with
    /// 4 x 4 outputs a thread for every product.
    static let tilesEnabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_PREFILL_FP32_TILES"]?
            .lowercased()
        return raw.map { !["0", "false", "off", "no"].contains($0) } ?? true
    }()

    /// Test hook: overrides `tilesEnabled` for the current task.
    @TaskLocal static var tilesOverride: Bool? = nil
    static var tilesActive: Bool { tilesOverride ?? tilesEnabled }

    /// A threadgroup's share of the product: `rows` x `columns` outputs, with
    /// `depth` columns of K staged per step and `threadRows` x `threadColumns`
    /// outputs folded by each thread. The fold of an output is the same for
    /// every tile, so the tile only sets how full the GPU runs.
    struct Tile: Hashable, Sendable {
        var rows: Int
        var columns: Int
        var depth: Int
        var threadRows: Int
        var threadColumns: Int

        /// Few outputs and a long K (router logits, loop-message scores): the
        /// product is a few thousand long folds, so eight outputs a thread keep
        /// each thread's chain short and put more threads on the GPU.
        static let wide = Tile(rows: 32, columns: 64, depth: 16, threadRows: 2, threadColumns: 4)
        /// Many outputs and a short K (message gates).
        static let tall = Tile(rows: 64, columns: 64, depth: 16, threadRows: 8, threadColumns: 4)

        var threads: (Int, Int) { (columns / threadColumns, rows / threadRows) }

        /// At most 1,024 threads and 32 KB of staged operands per threadgroup,
        /// at most 32 accumulators per thread.
        var valid: Bool {
            guard rows > 0, columns > 0, depth > 0, threadRows > 0, threadColumns > 0,
                rows % threadRows == 0, columns % threadColumns == 0
            else { return false }
            return threads.0 * threads.1 <= 1024 && (rows + columns) * depth * 4 <= 32768
                && threadRows * threadColumns <= 32
        }

        /// `ROWSxCOLUMNS:DEPTH:THREADROWSxTHREADCOLUMNS`, e.g. `128x128:16:4x8`.
        init?(_ text: String) {
            let parts = text.split(separator: ":").map {
                $0.split(separator: "x").compactMap { Int($0) }
            }
            guard parts.count == 3, parts[0].count == 2, parts[1].count == 1, parts[2].count == 2
            else { return nil }
            self.init(
                rows: parts[0][0], columns: parts[0][1], depth: parts[1][0],
                threadRows: parts[2][0], threadColumns: parts[2][1])
            guard valid else { return nil }
        }

        init(rows: Int, columns: Int, depth: Int, threadRows: Int, threadColumns: Int) {
            (self.rows, self.columns, self.depth) = (rows, columns, depth)
            (self.threadRows, self.threadColumns) = (threadRows, threadColumns)
        }
    }

    /// Bench hook: `DARKBLOOM_INKLING2_PREFILL_FP32_TILE` forces one tile.
    static let environmentTile: Tile? =
        ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_PREFILL_FP32_TILE"].flatMap {
            Tile($0)
        }

    /// Test hook: one tile for every product of the current task.
    @TaskLocal static var tileOverride: Tile? = nil

    /// The tile for a product with `outputs` columns: wide for the router's
    /// 128 outputs, tall for a gate's 4,608 (measured inside a chunk, where
    /// each product waits for the one before it).
    static func tile(outputs: Int) -> Tile {
        tileOverride ?? environmentTile ?? (outputs <= 256 ? .wide : .tall)
    }

    /// The fold of `kernel` on a tile given as template values: a threadgroup
    /// of (BN / RN) x (BM / RM) threads stages BK columns of its BM rows of
    /// `x` and BN rows of `w`; each thread folds RM x RN outputs, k ascending,
    /// one `fma` per term from zero.
    private static let tiledKernel = MLXFast.metalKernel(
        name: "inkling2_exact_fp32_matmul_tiled",
        inputNames: ["x", "w"],
        outputNames: ["y"],
        source: """
            constexpr int TX = BN / RN;
            constexpr int TY = BM / RM;
            constexpr int NT = TX * TY;
            threadgroup float xs[BK][BM];
            threadgroup float ws[BK][BN];
            const int M = x_shape[0];
            const int K = x_shape[1];
            const int N = w_shape[0];
            const int tx = thread_position_in_threadgroup.x;
            const int ty = thread_position_in_threadgroup.y;
            const int tid = ty * TX + tx;
            const int m0 = threadgroup_position_in_grid.y * BM;
            const int n0 = threadgroup_position_in_grid.x * BN;
            float acc[RM][RN];
            for (int i = 0; i < RM; ++i) {
                for (int j = 0; j < RN; ++j) {
                    acc[i][j] = 0.0f;
                }
            }
            for (int k0 = 0; k0 < K; k0 += BK) {
                for (int e = tid; e < BM * BK; e += NT) {
                    const int r = e / BK;
                    const int c = e % BK;
                    const int gm = m0 + r;
                    const int gk = k0 + c;
                    xs[c][r] = (gm < M && gk < K) ? x[size_t(gm) * K + gk] : 0.0f;
                }
                for (int e = tid; e < BN * BK; e += NT) {
                    const int r = e / BK;
                    const int c = e % BK;
                    const int gn = n0 + r;
                    const int gk = k0 + c;
                    ws[c][r] = (gn < N && gk < K) ? w[size_t(gn) * K + gk] : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const int kmax = min(BK, K - k0);
                for (int kk = 0; kk < kmax; ++kk) {
                    float a[RM];
                    float b[RN];
                    for (int i = 0; i < RM; ++i) {
                        a[i] = xs[kk][ty * RM + i];
                    }
                    for (int j = 0; j < RN; ++j) {
                        b[j] = ws[kk][tx * RN + j];
                    }
                    for (int i = 0; i < RM; ++i) {
                        for (int j = 0; j < RN; ++j) {
                            acc[i][j] = fma(a[i], b[j], acc[i][j]);
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (int i = 0; i < RM; ++i) {
                const int gm = m0 + ty * RM + i;
                for (int j = 0; j < RN; ++j) {
                    const int gn = n0 + tx * RN + j;
                    if (gm < M && gn < N) {
                        y[size_t(gm) * N + gn] = acc[i][j];
                    }
                }
            }
            """,
        ensureRowContiguous: true
    )

    private struct LaunchKey: Hashable {
        let tile: Tile
        let rows: Int
        let outputs: Int
    }

    private final class Launches: @unchecked Sendable {
        private let lock = NSLock()
        private var launches: [LaunchKey: MLXFast.MLXFastKernel.Launch] = [:]
        /// Chunk lengths vary; start over rather than keep every one.
        static let capacity = 1024

        func launch(_ key: LaunchKey, _ make: () -> MLXFast.MLXFastKernel.Launch)
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

    private static let launches = Launches()

    /// `matmul` on `tile`: the same bits as the 64 x 64 kernel.
    static func matmul(_ x: MLXArray, transposedWeight w: MLXArray, tile: Tile) -> MLXArray {
        let m = x.dim(0)
        let n = w.dim(0)
        let (tx, ty) = tile.threads
        let launch = launches.launch(LaunchKey(tile: tile, rows: m, outputs: n)) {
            tiledKernel.prepare(
                template: [
                    ("BM", tile.rows), ("BN", tile.columns), ("BK", tile.depth),
                    ("RM", tile.threadRows), ("RN", tile.threadColumns),
                ],
                grid: (
                    ((n + tile.columns - 1) / tile.columns) * tx,
                    ((m + tile.rows - 1) / tile.rows) * ty, 1
                ),
                threadGroup: (tx, ty, 1), outputShapes: [[m, n]], outputDTypes: [.float32])
        }
        return launch([x, w])[0]
    }

    /// `x` `[M, K]`, `w` `[N, K]`, both FP32. Returns `[M, N]` FP32.
    static func matmul(_ x: MLXArray, transposedWeight w: MLXArray) -> MLXArray {
        let m = x.dim(0)
        let n = w.dim(0)
        if tilesActive {
            let chosen = Self.tile(outputs: n)
            if chosen.valid { return matmul(x, transposedWeight: w, tile: chosen) }
        }
        return kernel(
            [x, w],
            grid: (((n + 63) / 64) * 16, ((m + 63) / 64) * 16, 1), threadGroup: (16, 16, 1),
            outputShapes: [[m, n]], outputDTypes: [.float32]
        )[0]
    }
}
