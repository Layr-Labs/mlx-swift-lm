import MLX

/// Row-exact matrix-vector products for 1 to 16 rows. Every row of the result
/// is bit-identical to MLX's single-vector product (`GEMVKernel::run`,
/// `kernels/gemv.h` at mlx cb77239b, non-transposed, SM 1, SN 32, TN 4) on
/// that row alone, while each weight is read once for all rows: the same
/// per-output order of products, simd shuffles and threadgroup sums, kept in
/// separate accumulators per row. Weights are dense (dtype T) or E4M3FN bytes
/// decoded with the eager ModelOpt rounding `T(decode(byte) * scale)`.
public enum ModelOptRowExact {
    public static let maximumRows = 16

    private static let kernel = MLXFast.metalKernel(
        name: "modelopt_row_exact_gemv",
        inputNames: ["mat", "scale", "in_vec"],
        outputNames: ["out_vec"],
        source: """
            constexpr int SM = 1;
            constexpr int SN = 32;
            constexpr int TN = 4;
            constexpr int threadsM = BM * SM;
            constexpr int threadsN = BN * SN;
            constexpr int blockM = threadsM * TM;
            constexpr int blockN = threadsN * TN;
            constexpr int tgp_size = BN > 1 ? BN * (blockM + TM) : 1;
            threadgroup float tgp_memory[V * tgp_size];
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const float weight_scale = scale[0];
            const float nan_value = metal::numeric_limits<float>::quiet_NaN();
            thread float result[V][TM];
            for (int v = 0; v < V; v++) {
                for (int tm = 0; tm < TM; tm++) {
                    result[v][tm] = 0;
                }
            }
            thread T inter[TN];
            thread float v_coeff[V][TN];
            const int thrM = 0;
            const int thrN = int(simd_lid);
            const int sgN = BN != 1 ? (simd_gid % BN) : 0;
            const int simdM = BN != 1 ? SM * (simd_gid / BN) : int(SM * simd_gid);
            const int simdN = BN != 1 ? SN * (simd_gid % BN) : 0;
            int bm = (simdM + thrM) * TM;
            int bn = (simdN + thrN) * TN;
            int out_row = threadgroup_position_in_grid.x * blockM + bm;
            if (out_row >= N) {
                return;
            }
            out_row = out_row + TM <= N ? out_row : N - TM;
            const device auto* rows = mat + size_t(out_row) * K;
            constexpr int n_iter = K / blockN;
            constexpr int leftover = K - blockN * n_iter;
            for (int i = 0; i < n_iter; ++i) {
                for (int v = 0; v < V; v++) {
                    for (int tn = 0; tn < TN; tn++) {
                        v_coeff[v][tn] = static_cast<float>(in_vec[v * K + bn + tn]);
                    }
                }
                int mat_offset = 0;
                for (int tm = 0; tm < TM; tm++) {
                    for (int tn = 0; tn < TN; tn++) {
                        if constexpr (FP8) {
                            const uint8_t code = rows[mat_offset + bn + tn];
                            const float decoded =
                                (code & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(code);
                            inter[tn] = static_cast<T>(decoded * weight_scale);
                        } else {
                            inter[tn] = static_cast<T>(rows[mat_offset + bn + tn]);
                        }
                    }
                    for (int v = 0; v < V; v++) {
                        for (int tn = 0; tn < TN; tn++) {
                            result[v][tm] += inter[tn] * v_coeff[v][tn];
                        }
                    }
                    mat_offset += K;
                }
                bn += blockN;
            }
            if (leftover > 0) {
                for (int v = 0; v < V; v++) {
                    for (int tn = 0; tn < TN; tn++) {
                        v_coeff[v][tn] =
                            bn + tn < K ? static_cast<float>(in_vec[v * K + bn + tn]) : 0.0f;
                    }
                }
                for (int tm = 0; tm < TM; tm++) {
                    for (int tn = 0; tn < TN; tn++) {
                        if (bn + tn < K) {
                            if constexpr (FP8) {
                                const uint8_t code = rows[tm * K + bn + tn];
                                const float decoded =
                                    (code & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(code);
                                inter[tn] = static_cast<T>(decoded * weight_scale);
                            } else {
                                inter[tn] = static_cast<T>(rows[tm * K + bn + tn]);
                            }
                        } else {
                            inter[tn] = T(0);
                        }
                    }
                    for (int v = 0; v < V; v++) {
                        for (int tn = 0; tn < TN; tn++) {
                            result[v][tm] += inter[tn] * v_coeff[v][tn];
                        }
                    }
                }
            }
            for (int v = 0; v < V; v++) {
                for (int tm = 0; tm < TM; tm++) {
                    for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                        result[v][tm] += simd_shuffle_down(result[v][tm], sn);
                    }
                }
            }
            if (BN > 1) {
                if (thrN == 0) {
                    for (int v = 0; v < V; v++) {
                        threadgroup float* tgp_results =
                            tgp_memory + v * tgp_size + sgN * (blockM + TM) + bm;
                        for (int tm = 0; tm < TM; tm++) {
                            tgp_results[tm] = result[v][tm];
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_none);
                    if (sgN == 0) {
                        for (int v = 0; v < V; v++) {
                            threadgroup float* tgp_results =
                                tgp_memory + v * tgp_size + sgN * (blockM + TM) + bm;
                            for (int sgn = 1; sgn < BN; sgn++) {
                                for (int tm = 0; tm < TM; tm++) {
                                    result[v][tm] += tgp_results[sgn * (blockM + TM) + tm];
                                }
                            }
                        }
                    }
                }
            }
            if (simdN == 0 && thrN == 0) {
                for (int v = 0; v < V; v++) {
                    for (int tm = 0; tm < TM; tm++) {
                        out_vec[v * N + out_row + tm] = static_cast<T>(result[v][tm]);
                    }
                }
            }
            """,
        header: ModelOptActivationKernels.header)

    /// `[V, N]` products of `vectors` (`[V, K]`, dtype T, 1 <= V <= 16, K > 64)
    /// with rows `mat` (`[N, K]`): dtype T, or uint8 E4M3FN with `scale` (`[1]`
    /// float32). `splitsK` is MLX's choice for the one-row call being
    /// reproduced (8 simdgroups across K, as when K >= 16 * outputs); the
    /// other geometry only schedules work.
    static func product(
        _ mat: MLXArray, scale: MLXArray?, vectors: MLXArray, splitsK: Bool
    ) -> MLXArray {
        let (outputs, inputs) = (mat.dim(0), mat.dim(1))
        let count = vectors.dim(0)
        let bm = splitsK ? 1 : (outputs >= 4096 ? 8 : 4)
        let bn = splitsK ? 8 : 1
        let tm = outputs < 4 ? 1 : 4
        let groups = (outputs + bm * tm - 1) / (bm * tm)
        return kernel(
            [mat, scale ?? MLXArray([Float(1)]), vectors],
            template: [
                ("T", vectors.dtype), ("BM", bm), ("BN", bn), ("TM", tm), ("K", inputs),
                ("N", outputs), ("V", count), ("FP8", mat.dtype == .uint8),
            ],
            grid: (groups * 32, bn, bm), threadGroup: (32, bn, bm),
            outputShapes: [[count, outputs]], outputDTypes: [vectors.dtype])[0]
    }

    /// `x . weight^T` for `x` `[R, K]` (R <= 16, K > 64) and `weight` `[N, K]`
    /// of the same float dtype: row r is bit-identical to
    /// `matmul(x[r ..< r + 1], weight.transposed())` (MLX's single-vector
    /// product), with each weight read once for all rows.
    public static func matmul(_ x: MLXArray, _ weight: MLXArray) -> MLXArray {
        precondition(
            x.ndim == 2 && weight.ndim == 2 && x.dim(1) == weight.dim(1) && x.dim(1) > 64
                && (1 ... maximumRows).contains(x.dim(0)) && x.dtype == weight.dtype,
            "row-exact matmul needs [1...16, K>64] x [N, K] of one dtype")
        return product(
            weight, scale: nil, vectors: x,
            splitsK: x.dim(1) >= 16 * weight.dim(0))
    }
}
