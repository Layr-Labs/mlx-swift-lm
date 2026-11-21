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

    /// `x` `[M, K]`, `w` `[N, K]`, both FP32. Returns `[M, N]` FP32.
    static func matmul(_ x: MLXArray, transposedWeight w: MLXArray) -> MLXArray {
        let m = x.dim(0)
        let n = w.dim(0)
        return kernel(
            [x, w],
            grid: (((n + 63) / 64) * 16, ((m + 63) / 64) * 16, 1), threadGroup: (16, 16, 1),
            outputShapes: [[m, n]], outputDTypes: [.float32]
        )[0]
    }
}
