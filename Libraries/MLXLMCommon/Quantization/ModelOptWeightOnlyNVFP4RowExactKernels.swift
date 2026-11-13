// Copyright © 2026 Eigen Labs.
// The simdgroup-matrix fragment layout and the paired E2M1 decode follow MLX
// `fp_qmv_mma` (kernels/fp_quantized.h, steel/gemm/mma.h at mlx 72dffa2d, MIT,
// Copyright Apple Inc.), with another order of the inputs.

import Foundation
import MLX

/// Row-exact products of 1 to 16 activation rows with one weight-only NVFP4
/// matrix (`W = e2m1 * e4m3`, ModelOpt W4A16).
///
/// Every row of a result is bit-identical to the product of that row alone,
/// at every row count and position, because one order of operations defines
/// each output element. With `w[k] = e2m1 * e4m3` (exact) and float32 sums:
/// - the inputs split into chunks of 128; partition `s` (0 to 7) takes the
///   chunks `q` with `q % 8 == s`, in ascending order;
/// - a chunk is four blocks of 32 inputs, in ascending order; inside a block
///   starting at `b` the inputs follow `b + 4c + i`, `i` in 0...3 outside and
///   `c` in 0...7 inside, each as `acc = fma(w, x, acc)` from zero;
/// - the result is `((acc0 + acc1) + ... + acc7)`, rounded to the activation
///   dtype, then (with a tensor scale) multiplied by it in float32 and rounded
///   again, as `(product.asType(.float32) * scale).asType(dtype)` does.
///
/// This is not the arithmetic of MLX's `.nvfp4` quantized matmul at any row
/// count: a model that verifies rows with it must decode single rows with it.
///
/// Two kernels run that order. The matrix kernel gives eight output rows,
/// eight activation rows and eight inputs to one simdgroup multiply-accumulate
/// and reads each weight once for up to 16 rows. The scalar kernel runs the
/// fused multiply-adds one by one for a single row, where the weight read is
/// the cost. The two agree only where a multiply-accumulate is that ascending
/// chain of fused multiply-adds (measured on Apple GPU family 15); elsewhere
/// the matrix kernel serves every row count.
public enum ModelOptWeightOnlyNVFP4RowExact {
    public static let maximumRows = 16

    /// The GPU family whose simdgroup multiply-accumulate was measured equal
    /// to the scalar kernel's chain, bit for bit.
    public static let scalarKernelMeasured: Bool = {
        let forced = ProcessInfo.processInfo.environment["MLX_METAL_GPU_ARCH"] ?? ""
        let code = Array((forced.isEmpty ? GPU.deviceInfo().architecture : forced).utf8)
        guard code.count >= 3 else { return false }
        let tens = Int(code[code.count - 3]) - 48
        let ones = Int(code[code.count - 2]) - 48
        return tens == 1 && ones == 5
    }()

    /// Rows the scalar kernel takes (0: none). Tests and benchmarks set it.
    nonisolated(unsafe) static var scalarRows = scalarKernelMeasured ? 1 : 0

    nonisolated(unsafe) private static let unit = MLXArray([Float(1)])

    static let header = """
        #include <metal_simdgroup_matrix>

        #define MO_WO_UNROLL _Pragma("clang loop unroll(full)")

        // MLX `fp8_e4m3` as a half: codes 0x7F and 0xFF are NaN.
        inline half mo_wo_e4m3(uint bits) {
            const uint v = bits & 127;
            const uint u = (v << 7) | (((v + 1) >> 7) << 14) | ((bits & 128) << 8);
            return as_type<half>(ushort(u)) * 256.0h;
        }

        // E2M1 code `i` (0 to 3) of each 16-bit half of `q`, times `s`.
        inline float2 mo_wo_pair(uint q, int i, half s) {
            uint p;
            if (i == 0) {
                p = ((q & 0x00070007) << 9) | ((q & 0x00080008) << 12);
            } else if (i == 1) {
                p = ((q & 0x00700070) << 5) | ((q & 0x00800080) << 8);
            } else if (i == 2) {
                p = ((q & 0x07000700) << 1) | ((q & 0x08000800) << 4);
            } else {
                p = ((q & 0x70007000) >> 3) | (q & 0x80008000);
            }
            const half2 hv = (as_type<half2>(p) * half2(16384.0h)) * half2(s);
            return float2(float(hv.x), float(hv.y));
        }

        // Four activations as float: packed loads, so any element offset.
        inline float4 mo_wo_load4(const device float* p) {
            return float4(*(const device packed_float4*)p);
        }
        inline float4 mo_wo_load4(const device half* p) {
            return float4(half4(*(const device packed_half4*)p));
        }
        inline float4 mo_wo_load4(const device bfloat* p) {
            return float4(as_type<vec<bfloat, 4>>(ushort4(*(const device packed_ushort4*)p)));
        }

        typedef metal::simdgroup_matrix<float, 8, 8> mo_wo_matrix;

        // C += A x B: each thread holds two neighboring columns of a row.
        inline void mo_wo_mma(thread float2& C, float2 A, float2 B) {
            mo_wo_matrix D_mat;
            mo_wo_matrix A_mat;
            mo_wo_matrix B_mat;
            mo_wo_matrix C_mat;
            reinterpret_cast<thread float2&>(A_mat.thread_elements()) = A;
            reinterpret_cast<thread float2&>(B_mat.thread_elements()) = B;
            reinterpret_cast<thread float2&>(C_mat.thread_elements()) = C;
            simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat);
            C = reinterpret_cast<thread float2&>(D_mat.thread_elements());
        }

        template <typename T>
        inline T mo_wo_result(float sum, bool scaled, float scale) {
            const T rounded = static_cast<T>(sum);
            return scaled ? static_cast<T>(static_cast<float>(rounded) * scale) : rounded;
        }

        """

    /// A threadgroup is eight simdgroups (the partitions) over 32 output
    /// rows. In a fragment, thread (fm, fn) holds output row fm with columns
    /// fn and fn + 1 of the weights, and input column fm with activation rows
    /// fn and fn + 1.
    private static let matrixKernel = MLXFast.metalKernel(
        name: "modelopt_weight_only_nvfp4_row_exact_matrix",
        inputNames: ["w", "scales", "x", "tensor_scale"],
        outputNames: ["y"],
        source: """
            constexpr int WK = 8;
            constexpr int BK = 128;
            constexpr int BN = 32;
            constexpr short TN = BN / 8;
            constexpr short TM = (V + 7) / 8;
            constexpr int row_bytes = K / 2;
            constexpr int row_scales = K / 16;
            constexpr bool whole_blocks = (K % 32) == 0;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const short qid = simd_lid / 4;
            const short fm = (qid & 4) + ((simd_lid / 2) % 4);
            const short fn = (qid & 2) * 2 + (simd_lid % 2) * 2;
            const int y_col = threadgroup_position_in_grid.y * BN;

            // Rows past V repeat row V - 1; their columns are never stored.
            const device T* xr[TM][2];
            float2 C[TN][TM];
            MO_WO_UNROLL
            for (short mt = 0; mt < TM; mt++) {
                MO_WO_UNROLL
                for (short j = 0; j < 2; j++) {
                    xr[mt][j] = x + size_t(min(8 * mt + fn + j, V - 1)) * K + 4 * fm;
                }
                MO_WO_UNROLL
                for (short t = 0; t < TN; t++) {
                    C[t][mt] = float2(0);
                }
            }
            const device uint* wl[TN];
            const device uint8_t* sl[TN];
            MO_WO_UNROLL
            for (short t = 0; t < TN; t++) {
                const int row = min(y_col + fm + 8 * t, N - 1);
                wl[t] = (const device uint*)(
                    (const device uint8_t*)w + size_t(row) * row_bytes + 2 * fn);
                sl[t] = scales + size_t(row) * row_scales + fn / 4;
            }

            for (int k = simd_gid * BK; k < K; k += WK * BK) {
                MO_WO_UNROLL
                for (short block = 0; block < 4; block++) {
                    const int b = k + 32 * block;
                    // The last block may hold 16 inputs: columns 0 to 3. A
                    // missing column is zero on both sides, which adds nothing.
                    const bool a_valid = whole_blocks ? b < K : b + 4 * fn < K;
                    const bool b_valid = whole_blocks ? b < K : b + 4 * fm < K;
                    if (whole_blocks && !a_valid) {
                        break;
                    }
                    float4 xv[TM][2];
                    MO_WO_UNROLL
                    for (short mt = 0; mt < TM; mt++) {
                        MO_WO_UNROLL
                        for (short j = 0; j < 2; j++) {
                            xv[mt][j] = b_valid ? mo_wo_load4(xr[mt][j] + b) : float4(0);
                        }
                    }
                    uint q[TN];
                    half s[TN];
                    MO_WO_UNROLL
                    for (short t = 0; t < TN; t++) {
                        q[t] = a_valid ? wl[t][b / 8] : 0u;
                        s[t] = a_valid ? mo_wo_e4m3(sl[t][b / 16]) : 0.0h;
                    }
                    MO_WO_UNROLL
                    for (short i = 0; i < 4; i++) {
                        float2 bf[TM];
                        MO_WO_UNROLL
                        for (short mt = 0; mt < TM; mt++) {
                            bf[mt] = float2(xv[mt][0][i], xv[mt][1][i]);
                        }
                        MO_WO_UNROLL
                        for (short t = 0; t < TN; t++) {
                            const float2 a = mo_wo_pair(q[t], i, s[t]);
                            MO_WO_UNROLL
                            for (short mt = 0; mt < TM; mt++) {
                                mo_wo_mma(C[t][mt], a, bf[mt]);
                            }
                        }
                    }
                }
            }

            constexpr int tile_elems = TN * TM * 2;
            threadgroup float partials[(WK - 1) * tile_elems * 32];
            if (simd_gid > 0) {
                threadgroup float* p = partials + (simd_gid - 1) * tile_elems * 32 + simd_lid;
                MO_WO_UNROLL
                for (short t = 0; t < TN; t++) {
                    MO_WO_UNROLL
                    for (short mt = 0; mt < TM; mt++) {
                        MO_WO_UNROLL
                        for (short j = 0; j < 2; j++) {
                            p[((t * TM + mt) * 2 + j) * 32] = C[t][mt][j];
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_gid == 0) {
                for (int s = 0; s < WK - 1; s++) {
                    const threadgroup float* p = partials + s * tile_elems * 32 + simd_lid;
                    MO_WO_UNROLL
                    for (short t = 0; t < TN; t++) {
                        MO_WO_UNROLL
                        for (short mt = 0; mt < TM; mt++) {
                            MO_WO_UNROLL
                            for (short j = 0; j < 2; j++) {
                                C[t][mt][j] += p[((t * TM + mt) * 2 + j) * 32];
                            }
                        }
                    }
                }
                MO_WO_UNROLL
                for (short mt = 0; mt < TM; mt++) {
                    MO_WO_UNROLL
                    for (short j = 0; j < 2; j++) {
                        if (8 * mt + fn + j < V) {
                            device T* yr = y + size_t(8 * mt + fn + j) * N;
                            MO_WO_UNROLL
                            for (short t = 0; t < TN; t++) {
                                const int out = y_col + fm + 8 * t;
                                if (out < N) {
                                    yr[out] = mo_wo_result<T>(
                                        C[t][mt][j], SCALED, tensor_scale[0]);
                                }
                            }
                        }
                    }
                }
            }
            """,
        header: header)

    /// A simdgroup is four groups of eight lanes: lane `s` of a group runs
    /// partition `s` for the group's RS output rows.
    private static let scalarKernel = MLXFast.metalKernel(
        name: "modelopt_weight_only_nvfp4_row_exact_scalar",
        inputNames: ["w", "scales", "x", "tensor_scale"],
        outputNames: ["y"],
        source: """
            constexpr int WK = 8;
            constexpr int BK = 128;
            constexpr int RS = 2;
            constexpr int row_bytes = K / 2;
            constexpr int row_scales = K / 16;
            const uint lid = thread_index_in_simdgroup;
            const uint part = lid % WK;
            const uint group = lid / WK;
            const int out_row =
                (threadgroup_position_in_grid.y * 2 + simdgroup_index_in_threadgroup) * (4 * RS)
                + group * RS;
            float acc[V][RS];
            const device uint* wr[RS];
            const device uint8_t* sr[RS];
            MO_WO_UNROLL
            for (int r = 0; r < RS; r++) {
                const int row = min(out_row + r, N - 1);
                wr[r] = (const device uint*)((const device uint8_t*)w + size_t(row) * row_bytes);
                sr[r] = scales + size_t(row) * row_scales;
                MO_WO_UNROLL
                for (int v = 0; v < V; v++) {
                    acc[v][r] = 0;
                }
            }
            for (int k = part * BK; k < K; k += WK * BK) {
                MO_WO_UNROLL
                for (int block = 0; block < 4; block++) {
                    const int b = k + 32 * block;
                    if (b >= K) {
                        break;
                    }
                    // Words (two columns each) before K: 4, or 2 in a last half block.
                    const int words = (K - b) >= 32 ? 4 : 2;
                    float xf[V][32];
                    MO_WO_UNROLL
                    for (int v = 0; v < V; v++) {
                        MO_WO_UNROLL
                        for (int h = 0; h < 8; h++) {
                            const float4 xv = (h < 4 || words == 4)
                                ? mo_wo_load4(x + size_t(v) * K + b + 4 * h) : float4(0);
                            MO_WO_UNROLL
                            for (int e = 0; e < 4; e++) {
                                xf[v][4 * h + e] = xv[e];
                            }
                        }
                    }
                    MO_WO_UNROLL
                    for (int r = 0; r < RS; r++) {
                        uint q[4];
                        half s[2];
                        MO_WO_UNROLL
                        for (int j = 0; j < 4; j++) {
                            q[j] = j < words ? wr[r][b / 8 + j] : 0u;
                        }
                        MO_WO_UNROLL
                        for (int g = 0; g < 2; g++) {
                            s[g] = 2 * g < words ? mo_wo_e4m3(sr[r][b / 16 + g]) : 0.0h;
                        }
                        MO_WO_UNROLL
                        for (int i = 0; i < 4; i++) {
                            MO_WO_UNROLL
                            for (int j = 0; j < 4; j++) {
                                if (j < words) {
                                    // Columns 2j and 2j + 1: codes i and i + 4 of word j.
                                    const float2 wv = mo_wo_pair(q[j], i, s[j / 2]);
                                    MO_WO_UNROLL
                                    for (int v = 0; v < V; v++) {
                                        acc[v][r] = fma(wv.x, xf[v][8 * j + i], acc[v][r]);
                                        acc[v][r] = fma(wv.y, xf[v][8 * j + 4 + i], acc[v][r]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            MO_WO_UNROLL
            for (int v = 0; v < V; v++) {
                MO_WO_UNROLL
                for (int r = 0; r < RS; r++) {
                    float total = acc[v][r];
                    MO_WO_UNROLL
                    for (ushort s = 1; s < WK; s++) {
                        total += simd_shuffle_down(acc[v][r], s);
                    }
                    if (part == 0 && out_row + r < N) {
                        y[size_t(v) * N + out_row + r] =
                            mo_wo_result<T>(total, SCALED, tensor_scale[0]);
                    }
                }
            }
            """,
        header: header)

    /// Whether `product` takes `rows` rows of `inputs` activations of `dtype`
    /// on the default device: 1 to 16 rows, a multiple of 16 inputs, a float
    /// dtype, the GPU.
    public static func supports(rows: Int, inputs: Int, dtype: DType) -> Bool {
        (1 ... maximumRows).contains(rows) && inputs > 0 && inputs % 16 == 0
            && [.bfloat16, .float16, .float32].contains(dtype)
            && Device.defaultDevice().deviceType == .gpu
    }

    /// `x @ (e2m1 * e4m3).T`, times `tensorScale` when given, in the dtype of
    /// `x`, with the order in the type's description.
    ///
    /// - Parameters:
    ///   - x: `[..., K]` activations, 1 to 16 rows in all (`supports`).
    ///   - weight: uint32 `[N, K / 8]`, the packed E2M1 codes: element `2i` of
    ///     a row is the low nibble of byte `i`.
    ///   - blockScales: uint8 `[N, K / 16]`, one E4M3 code per 16 inputs.
    ///   - tensorScale: float32 with one element (`weight_scale_2`), or nil
    ///     for the unscaled product.
    /// - Returns: `[..., N]`. NaN and infinite activations stay in their rows.
    public static func product(
        _ x: MLXArray, weight: MLXArray, blockScales: MLXArray, tensorScale: MLXArray? = nil
    ) -> MLXArray {
        let inputs = x.dim(-1)
        let rows = x.size / max(1, inputs)
        let outputs = weight.dim(0)
        precondition(
            supports(rows: rows, inputs: inputs, dtype: x.dtype) && weight.dtype == .uint32
                && weight.shape == [outputs, inputs / 8] && blockScales.dtype == .uint8
                && blockScales.shape == [outputs, inputs / 16]
                && (tensorScale.map { $0.dtype == .float32 && $0.size == 1 } ?? true),
            "weight-only NVFP4 row-exact product needs 1...16 rows of [K % 16 == 0] "
                + "activations, uint32 [N, K / 8] codes and uint8 [N, K / 16] scales")
        // MLX binds arrays of fewer than eight elements as constants, which
        // the kernels do not read: a tiny matrix gets zero rows up to eight.
        if weight.size < 8 || blockScales.size < 8 {
            let extra = 8 - outputs
            let padded = product(
                x,
                weight: concatenated(
                    [weight, MLXArray.zeros([extra, inputs / 8], dtype: .uint32)], axis: 0),
                blockScales: concatenated(
                    [blockScales, MLXArray.zeros([extra, inputs / 16], dtype: .uint8)], axis: 0),
                tensorScale: tensorScale)
            return padded[.ellipsis, 0 ..< outputs]
        }
        let scalar = rows <= scalarRows
        let kernel = scalar ? scalarKernel : matrixKernel
        let (tile, simdgroups) = scalar ? (16, 2) : (32, 8)
        let tiles = (outputs + tile - 1) / tile
        let result = kernel(
            [weight, blockScales, x.reshaped(rows, inputs), tensorScale?.reshaped(1) ?? unit],
            template: [
                ("T", x.dtype), ("V", rows), ("K", inputs), ("N", outputs),
                ("SCALED", tensorScale != nil),
            ],
            grid: (32, simdgroups * tiles, 1), threadGroup: (32, simdgroups, 1),
            outputShapes: [[rows, outputs]], outputDTypes: [x.dtype])[0]
        return result.reshaped(Array(x.shape.dropLast()) + [outputs])
    }
}
