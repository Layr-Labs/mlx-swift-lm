// Inkling 2 prefill products on the M5 neural accelerators with one fixed tile.
import Foundation
import MLX

/// Prefill `x @ w.T` (BF16 or F16, FP32 accumulation) with one fixed tile for
/// every row count: 64 x 128 outputs per threadgroup, eight simdgroups of
/// 32 x 32, 32-wide K steps in a fixed order, no split-K. Each output row is
/// the same sum in the same order whatever the chunk length or the other rows
/// of the call, so prefill products are chunk-invariant without padding every
/// call to a fixed row count. Not bit-equal to MLX's own GEMM.
///
/// Used by `Inkling2RowExact.prefillMatmul` on GPUs with the neural
/// accelerators; other GPUs keep the zero-padded fixed-row MLX calls.
enum Inkling2PrefillGEMM {
    /// `DARKBLOOM_INKLING2_PREFILL_GEMM=0` keeps the padded MLX calls.
    static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_PREFILL_GEMM"]?
            .lowercased()
        return raw.map { !["0", "false", "off", "no"].contains($0) } ?? true
    }()

    /// Test hook: overrides `enabled` for the current task.
    @TaskLocal static var override: Bool? = nil

    static var active: Bool {
        (override ?? enabled) && Inkling2PagedAttention.neuralAcceleratorsAvailable
    }

    /// Fragments of 16 rows per simdgroup, pairs of 16-column fragments per
    /// simdgroup, and simdgroups along the rows and the columns.
    static let rowFragments = 2
    static let columnPairs = 1
    static let rowGroups = 2
    static let columnGroups = 4
    static var tileRows: Int { 16 * rowFragments * rowGroups }
    static var tileColumns: Int { 32 * columnPairs * columnGroups }
    static var threads: Int { 32 * rowGroups * columnGroups }

    /// Whether the kernel takes `x` `[..., K]` times `w` `[N, K]`.
    static func covers(_ x: MLXArray, transposedWeight w: MLXArray) -> Bool {
        guard active, x.ndim >= 1, w.ndim == 2, x.dtype == w.dtype,
            x.dtype == .bfloat16 || x.dtype == .float16
        else { return false }
        let k = x.dim(-1)
        let rows = x.size / max(k, 1)
        return k == w.dim(1) && k > 0 && k % 32 == 0 && rows > 0 && w.dim(0) > 0
            && rows <= Int(Int32.max) / tileRows
    }

    /// `x @ w.T` for `x` `[..., K]` and `w` `[N, K]`, or nil when not covered.
    static func matmul(_ x: MLXArray, transposedWeight w: MLXArray) -> MLXArray? {
        guard covers(x, transposedWeight: w) else { return nil }
        let shape = x.shape
        let k = shape.last!
        let rows = x.size / k
        let n = w.dim(0)
        let launch = launches.launch(Key(n: n, k: k, rows: rows, dtype: x.dtype)) {
            kernel.prepare(
                template: [
                    ("T", x.dtype), ("N", n), ("K", k), ("TM", rowFragments),
                    ("TP", columnPairs), ("WM", rowGroups), ("WN", columnGroups),
                ],
                grid: (
                    (n + tileColumns - 1) / tileColumns * threads,
                    (rows + tileRows - 1) / tileRows, 1
                ),
                threadGroup: (threads, 1, 1), outputShapes: [[rows, n]],
                outputDTypes: [x.dtype])
        }
        return launch([x.reshaped(rows, k), w])[0].reshaped(Array(shape.dropLast()) + [n])
    }

    struct Key: Hashable {
        let n: Int
        let k: Int
        let rows: Int
        let dtype: DType
    }

    final class Launches: @unchecked Sendable {
        private let lock = NSLock()
        private var launches: [Key: MLXFast.MLXFastKernel.Launch] = [:]
        /// Row counts vary with chunking; start over rather than keep every one.
        static let capacity = 4096

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

    /// Rows past M read row M - 1 and columns past N read row N - 1 of `w`;
    /// neither is stored. Empty 16-row fragments are skipped.
    static let kernel = MLXFast.metalKernel(
        name: "inkling2_prefill_gemm", inputNames: ["x", "w"], outputNames: ["y"],
        source: #"""
            constexpr int SM = 16 * TM;
            constexpr int SN = 32 * TP;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const int M = int(x_shape[0]);
            const int r0 = int(threadgroup_position_in_grid.y) * (WM * SM) + (int(sg) / WN) * SM;
            const int c0 = int(threadgroup_position_in_grid.x) * (WN * SN) + (int(sg) % WN) * SN;
            if (r0 >= M || c0 >= N) return;
            const int vf = min(TM, (M - r0 + 15) / 16);
            const short2 fc = dg::coord(lane);
            const device T* arow[TM * 2];
            _Pragma("clang loop unroll(full)")
            for (int f = 0; f < TM; f++) {
                _Pragma("clang loop unroll(full)")
                for (int i = 0; i < 2; i++) {
                    const int r = min(r0 + f * 16 + fc.y + i * 8, M - 1);
                    arow[f * 2 + i] = x + size_t(r) * K + fc.x;
                }
            }
            const device T* wrow[TP * 4];
            _Pragma("clang loop unroll(full)")
            for (int t = 0; t < TP * 2; t++) {
                _Pragma("clang loop unroll(full)")
                for (int i = 0; i < 2; i++) {
                    const int n = min(c0 + t * 16 + fc.y + i * 8, N - 1);
                    wrow[t * 2 + i] = w + size_t(n) * K + fc.x;
                }
            }
            vec<float, 8> acc[TM][TP * 2];
            _Pragma("clang loop unroll(full)")
            for (int f = 0; f < TM; f++) {
                _Pragma("clang loop unroll(full)")
                for (int t = 0; t < TP * 2; t++) acc[f][t] = 0.0f;
            }
            _Pragma("clang loop unroll(disable)")
            for (int k0 = 0; k0 < K; k0 += 32) {
                volatile int compiler_barrier;
                vec<T, 8> a[TM][2];
                vec<T, 8> b[TP * 2][2];
                _Pragma("clang loop unroll(full)")
                for (int f = 0; f < TM; f++) {
                    if (f < vf) {
                        _Pragma("clang loop unroll(full)")
                        for (int h = 0; h < 2; h++) {
                            _Pragma("clang loop unroll(full)")
                            for (int i = 0; i < 2; i++) {
                                const vec<T, 4> v = *(const device vec<T, 4>*)(arow[f * 2 + i] + k0 + h * 16);
                                _Pragma("clang loop unroll(full)")
                                for (int j = 0; j < 4; j++) a[f][h][i * 4 + j] = v[j];
                            }
                        }
                    }
                }
                _Pragma("clang loop unroll(full)")
                for (int t = 0; t < TP * 2; t++) {
                    _Pragma("clang loop unroll(full)")
                    for (int h = 0; h < 2; h++) {
                        _Pragma("clang loop unroll(full)")
                        for (int i = 0; i < 2; i++) {
                            const vec<T, 4> v = *(const device vec<T, 4>*)(wrow[t * 2 + i] + k0 + h * 16);
                            _Pragma("clang loop unroll(full)")
                            for (int j = 0; j < 4; j++) b[t][h][i * 4 + j] = v[j];
                        }
                    }
                }
                _Pragma("clang loop unroll(full)")
                for (int f = 0; f < TM; f++) {
                    if (f < vf) {
                        _Pragma("clang loop unroll(full)")
                        for (int t = 0; t < TP * 2; t += 2) {
                            _Pragma("clang loop unroll(full)")
                            for (int h = 0; h < 2; h++) dg::mma<T>(acc[f][t], acc[f][t + 1], a[f][h], b[t][h], b[t + 1][h]);
                        }
                    }
                }
                (void)compiler_barrier;
            }
            _Pragma("clang loop unroll(full)")
            for (int f = 0; f < TM; f++) {
                if (f >= vf) continue;
                _Pragma("clang loop unroll(full)")
                for (int i = 0; i < 2; i++) {
                    const int r = r0 + f * 16 + fc.y + i * 8;
                    if (r >= M) continue;
                    _Pragma("clang loop unroll(full)")
                    for (int t = 0; t < TP * 2; t++) {
                        const int n = c0 + t * 16 + fc.x;
                        if (n + 3 < N) {
                            vec<T, 4> o;
                            _Pragma("clang loop unroll(full)")
                            for (int j = 0; j < 4; j++) o[j] = static_cast<T>(acc[f][t][i * 4 + j]);
                            *(device vec<T, 4>*)(y + size_t(r) * N + n) = o;
                        } else {
                            _Pragma("clang loop unroll(full)")
                            for (int j = 0; j < 4; j++) {
                                if (n + j < N) y[size_t(r) * N + n + j] = static_cast<T>(acc[f][t][i * 4 + j]);
                            }
                        }
                    }
                }
            }
            """#,
        header: header)

    /// Ends with a newline: MLX appends the kernel signature right after it.
    static let header = #"""
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        namespace dg {
        using namespace metal;
        inline short2 coord(uint lane) {
            const short qid = short(lane >> 2);
            const short fm = (qid & 4) | short((lane >> 1) & 3);
            const short fn = short(((qid & 2) | short(lane & 1)) * 4);
            return short2(fn, fm);
        }
        template <typename AT>
        inline void mma(thread vec<float, 8>& c0, thread vec<float, 8>& c1, thread const vec<AT, 8>& a,
                        thread const vec<AT, 8>& b0, thread const vec<AT, 8>& b1) {
            constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
                16, 32, 16, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
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
        }  // namespace dg

        """#
}

extension Inkling2Diagnostics {
    /// Bench hook: runs `body` with the fixed-tile prefill GEMM on or off
    /// (when off, prefill products take the padded fixed-row MLX calls).
    public static func withPrefillGEMM<T>(_ enabled: Bool, _ body: () throws -> T) rethrows -> T {
        try Inkling2PrefillGEMM.$override.withValue(enabled, operation: body)
    }
}
