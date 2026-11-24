// Row-exact gemv over `Inkling2Bytes.Code` (research; not wired into the model).
import Foundation
import MLX

/// Matrix-vector products for 1 to 8 vectors whose rows are bit-identical to the same kernel reading
/// the raw BF16 matrix, and to MLX's single-vector gemv on that row alone (the `ModelOptRowExact`
/// geometry: SM 1, SN 32, TN 4, TM 4, BN 1, BM 4 or 8; one accumulator set per vector). Only the
/// load of each weight differs: two BF16 values are rebuilt per 32-bit operation from the
/// sign|mantissa bytes and the exponent codes, escaped exponents are patched by the lanes that have
/// one (row offsets + one simd prefix sum per iteration for the four rows). The products and their
/// order are unchanged, so the bits are too.
extension Inkling2Bytes {
    static let loopHeader = """
        constexpr int SN = 32;
        constexpr int TN = 4;
        constexpr int TM = 4;
        constexpr int blockM = BM * TM;
        constexpr int blockN = SN * TN;
        constexpr int N4 = (N + 3) / 4 * 4;
        constexpr int n_iter = K / blockN;
        constexpr uint ESC = (1u << KB) - 1u;
        constexpr uint EBASE2 = uint(EBASE) * 0x10001u;
        const uint simd_gid = simdgroup_index_in_threadgroup;
        const int thrN = int(thread_index_in_simdgroup);
        const int out_row = int(threadgroup_position_in_grid.x) * blockM + int(simd_gid) * TM;
        if (out_row >= N4) {
            return;
        }
        const size_t rgrp = size_t(out_row) >> 2;
        uint run[TM];
        for (int tm = 0; tm < TM; tm++) {
            run[tm] = row_off[out_row + tm];
        }
        int bn = thrN * TN;
        """

    /// Decodes this lane's 16 weights of iteration `i` into `pairs[tm][0/1]` (two BF16 per uint).
    static let decodeStep = """
            const size_t lane = (rgrp * (K / 128) + size_t(i)) * 32 + size_t(thrN);
            const uint4 smv = ((const device uint4*)sm)[lane];
            uint ch[4];
            if (KB == 3) {
                const device ushort* cp = ((const device ushort*)codes) + lane * 3;
                const uint lo = uint(cp[0]) | (uint(cp[1]) << 16);
                const uint hi = uint(cp[2]);
                ch[0] = lo & 0xFFFu;
                ch[1] = (lo >> 12) & 0xFFFu;
                ch[2] = ((lo >> 24) | (hi << 8)) & 0xFFFu;
                ch[3] = hi >> 4;
            } else {
                const uint2 c2 = ((const device uint2*)codes)[lane];
                ch[0] = c2.x & 0xFFFFu;
                ch[1] = c2.x >> 16;
                ch[2] = c2.y & 0xFFFFu;
                ch[3] = c2.y >> 16;
            }
            uint m[4];
            uint packed = 0;
            for (int tm = 0; tm < TM; tm++) {
                m[tm] = 0;
                for (int tn = 0; tn < TN; tn++) {
                    m[tm] |= (((ch[tm] >> (KB * tn)) & ESC) == ESC ? 1u : 0u) << tn;
                }
                packed |= popcount(m[tm]) << (8 * tm);
            }
            uint pre = 0;
            uint tot = 0;
            if (KB == 3 || simd_any(packed != 0)) {
                pre = simd_prefix_exclusive_sum(packed);
                tot = simd_sum(packed);
            }
            uint pairs[4][2];
            for (int tm = 0; tm < TM; tm++) {
                const uint smw = smv[tm];
                const uint c = ch[tm];
                const uint s01 = (smw & 0xFFu) | ((smw & 0xFF00u) << 8);
                const uint s23 = ((smw >> 16) & 0xFFu) | ((smw >> 8) & 0xFF0000u);
                const uint e01 = ((c & ESC) | (((c >> KB) & ESC) << 16)) + EBASE2;
                const uint e23 = (((c >> (2 * KB)) & ESC) | (((c >> (3 * KB)) & ESC) << 16)) + EBASE2;
                uint p01 = ((s01 & 0x00800080u) << 8) | (s01 & 0x007F007Fu) | (e01 << 7);
                uint p23 = ((s23 & 0x00800080u) << 8) | (s23 & 0x007F007Fu) | (e23 << 7);
                if (m[tm] != 0u) {
                    uint r = run[tm] + ((pre >> (8 * tm)) & 0xFFu);
                    for (int tn = 0; tn < TN; tn++) {
                        if ((m[tm] >> tn) & 1u) {
                            const uint e = uint(esc[r]);
                            r++;
                            const uint sh = 7u + 16u * uint(tn & 1);
                            if (tn < 2) {
                                p01 = (p01 & ~(0xFFu << sh)) | (e << sh);
                            } else {
                                p23 = (p23 & ~(0xFFu << sh)) | (e << sh);
                            }
                        }
                    }
                }
                pairs[tm][0] = p01;
                pairs[tm][1] = p23;
                run[tm] += (tot >> (8 * tm)) & 0xFFu;
            }
        """

    private static let productKernel = MLXFast.metalKernel(
        name: "inkling2_bytes_gemv",
        inputNames: ["sm", "codes", "esc", "row_off", "in_vec"],
        outputNames: ["out_vec"],
        source: loopHeader + """
            thread float result[V][TM];
            for (int v = 0; v < V; v++) {
                for (int tm = 0; tm < TM; tm++) {
                    result[v][tm] = 0;
                }
            }
            thread T inter[TN];
            thread float v_coeff[V][TN];
            for (int i = 0; i < n_iter; ++i) {
                for (int v = 0; v < V; v++) {
                    for (int tn = 0; tn < TN; tn++) {
                        v_coeff[v][tn] = static_cast<float>(in_vec[v * K + bn + tn]);
                    }
                }
            """ + decodeStep + """
                    for (int tm = 0; tm < TM; tm++) {
                        inter[0] = as_type<T>(ushort(pairs[tm][0] & 0xFFFFu));
                        inter[1] = as_type<T>(ushort(pairs[tm][0] >> 16));
                        inter[2] = as_type<T>(ushort(pairs[tm][1] & 0xFFFFu));
                        inter[3] = as_type<T>(ushort(pairs[tm][1] >> 16));
                        for (int v = 0; v < V; v++) {
                            for (int tn = 0; tn < TN; tn++) {
                                result[v][tm] += inter[tn] * v_coeff[v][tn];
                            }
                        }
                    }
                    bn += blockN;
                }
                for (int v = 0; v < V; v++) {
                    for (int tm = 0; tm < TM; tm++) {
                        for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                            result[v][tm] += simd_shuffle_down(result[v][tm], sn);
                        }
                    }
                }
                if (thrN == 0) {
                    for (int v = 0; v < V; v++) {
                        for (int tm = 0; tm < TM; tm++) {
                            if (out_row + tm < N) {
                                out_vec[v * N + out_row + tm] = static_cast<T>(result[v][tm]);
                            }
                        }
                    }
                }
                """)

    private static let decodeKernel = MLXFast.metalKernel(
        name: "inkling2_bytes_decode",
        inputNames: ["sm", "codes", "esc", "row_off"],
        outputNames: ["out"],
        source: loopHeader + """
            for (int i = 0; i < n_iter; ++i) {
            """ + decodeStep + """
                    for (int tm = 0; tm < TM; tm++) {
                        if (out_row + tm < N) {
                            device ushort* dst = ((device ushort*)out) + size_t(out_row + tm) * K + bn;
                            dst[0] = ushort(pairs[tm][0] & 0xFFFFu);
                            dst[1] = ushort(pairs[tm][0] >> 16);
                            dst[2] = ushort(pairs[tm][1] & 0xFFFFu);
                            dst[3] = ushort(pairs[tm][1] >> 16);
                        }
                    }
                    bn += blockN;
                }
                """)

    /// The same kernel's raw-BF16 load (the reference the coded product must equal bit for bit).
    private static let rawKernel = MLXFast.metalKernel(
        name: "inkling2_bytes_gemv_raw",
        inputNames: ["mat", "in_vec"],
        outputNames: ["out_vec"],
        source: """
            constexpr int SN = 32;
            constexpr int TN = 4;
            constexpr int TM = 4;
            constexpr int blockM = BM * TM;
            constexpr int blockN = SN * TN;
            constexpr int n_iter = K / blockN;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const int thrN = int(thread_index_in_simdgroup);
            int out_row = int(threadgroup_position_in_grid.x) * blockM + int(simd_gid) * TM;
            if (out_row >= N) {
                return;
            }
            out_row = out_row + TM <= N ? out_row : N - TM;
            const device auto* rows = mat + size_t(out_row) * K;
            thread float result[V][TM];
            for (int v = 0; v < V; v++) {
                for (int tm = 0; tm < TM; tm++) {
                    result[v][tm] = 0;
                }
            }
            thread T inter[TN];
            thread float v_coeff[V][TN];
            int bn = thrN * TN;
            for (int i = 0; i < n_iter; ++i) {
                for (int v = 0; v < V; v++) {
                    for (int tn = 0; tn < TN; tn++) {
                        v_coeff[v][tn] = static_cast<float>(in_vec[v * K + bn + tn]);
                    }
                }
                int mat_offset = 0;
                for (int tm = 0; tm < TM; tm++) {
                    for (int tn = 0; tn < TN; tn++) {
                        inter[tn] = static_cast<T>(rows[mat_offset + bn + tn]);
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
            for (int v = 0; v < V; v++) {
                for (int tm = 0; tm < TM; tm++) {
                    for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                        result[v][tm] += simd_shuffle_down(result[v][tm], sn);
                    }
                }
            }
            if (thrN == 0) {
                for (int v = 0; v < V; v++) {
                    for (int tm = 0; tm < TM; tm++) {
                        out_vec[v * N + out_row + tm] = static_cast<T>(result[v][tm]);
                    }
                }
            }
            """)

    static func simdgroups(_ n: Int) -> Int { n >= 4096 ? 8 : 4 }

    /// Decodes `code` to its BF16 `[N, K]` matrix on the GPU (the load-time proof uses this).
    public static func decode(_ code: Code) -> MLXArray {
        let bm = simdgroups(code.rows)
        let groups = (code.paddedRows + 4 * bm - 1) / (4 * bm)
        return decodeKernel(
            [code.signMantissa, code.codes, code.escapes, code.rowOffsets],
            template: [
                ("BM", bm), ("K", code.columns), ("N", code.rows), ("KB", code.exponentBits),
                ("EBASE", code.base),
            ],
            grid: (groups * 32, 1, bm), threadGroup: (32, 1, bm),
            outputShapes: [[code.rows, code.columns]], outputDTypes: [.bfloat16])[0]
    }

    /// `vectors . W^T` (`vectors` BF16 `[V, K]`, 1 ... 8 rows per dispatch, chunked above) from the
    /// code; row v equals `rawProduct(W, vectors)[v]` and MLX's gemv of that row alone, bit for bit.
    public static func product(_ code: Code, vectors: MLXArray) -> MLXArray {
        precondition(
            vectors.ndim == 2 && vectors.dim(1) == code.columns && vectors.dtype == .bfloat16,
            "Inkling2Bytes.product needs BF16 [V, K]")
        if vectors.dim(0) > 8 {
            return concatenated(
                stride(from: 0, to: vectors.dim(0), by: 8).map {
                    product(code, vectors: vectors[$0 ..< min(vectors.dim(0), $0 + 8)])
                }, axis: 0)
        }
        let bm = simdgroups(code.rows)
        let groups = (code.paddedRows + 4 * bm - 1) / (4 * bm)
        return productKernel(
            [code.signMantissa, code.codes, code.escapes, code.rowOffsets, vectors],
            template: [
                ("T", DType.bfloat16), ("BM", bm), ("K", code.columns), ("N", code.rows),
                ("V", vectors.dim(0)), ("KB", code.exponentBits), ("EBASE", code.base),
            ],
            grid: (groups * 32, 1, bm), threadGroup: (32, 1, bm),
            outputShapes: [[vectors.dim(0), code.rows]], outputDTypes: [.bfloat16])[0]
    }

    /// The reference: the same kernel reading the raw BF16 `[N, K]` matrix (N >= 4, K % 128 == 0).
    public static func rawProduct(_ weight: MLXArray, vectors: MLXArray) -> MLXArray {
        precondition(vectors.ndim == 2 && vectors.dim(0) <= 8 && weight.dim(0) >= 4)
        let (n, k) = (weight.dim(0), weight.dim(1))
        let bm = simdgroups(n)
        let groups = (n + 4 * bm - 1) / (4 * bm)
        return rawKernel(
            [weight, vectors],
            template: [
                ("T", DType.bfloat16), ("BM", bm), ("K", k), ("N", n), ("V", vectors.dim(0)),
            ],
            grid: (groups * 32, 1, bm), threadGroup: (32, 1, bm),
            outputShapes: [[vectors.dim(0), n]], outputDTypes: [.bfloat16])[0]
    }

    /// Self-check used by the research harness: for each `matrices` entry, encodes with 3- and
    /// 4-bit codes, proves the GPU decode, and compares products for 1 ... 8 vectors with the raw
    /// kernel and with MLX's single-row gemv. Returns mismatching output count per check.
    public static func selfCheck(_ matrices: [(String, MLXArray)], seed: UInt64 = 1121) throws
        -> [String: Int]
    {
        var out = [String: Int]()
        for (label, w) in matrices {
            let k = w.dim(1)
            for bits in [3, 4] {
                let code = try encodeVerified(w, exponentBits: bits)
                var coded = 0
                var mlx = 0
                for v in [1, 2, 3, 4, 5, 8, 13] {
                    let x = MLXRandom.normal(
                        [v, k], key: MLXRandom.key(seed &+ UInt64(10 * v + bits))
                    )
                    .asType(.bfloat16)
                    let y = product(code, vectors: x)
                    let rawRows = v <= 8 ? rawProduct(w, vectors: x) : nil
                    if let rawRows {
                        coded += Int(
                            sum(
                                (y.view(dtype: .uint16) .!= rawRows.view(dtype: .uint16)).asType(
                                    .int32)
                            )
                            .item(Int32.self))
                    }
                    for r in 0 ..< v {
                        let one = matmul(x[r ..< r + 1], w.T)
                        mlx += Int(
                            sum(
                                (y[r ..< r + 1].view(dtype: .uint16) .!= one.view(dtype: .uint16))
                                    .asType(.int32)
                            )
                            .item(Int32.self))
                    }
                }
                out["\(label) k\(bits) coded!=raw"] = coded
                out["\(label) k\(bits) coded!=mlx_gemv_row"] = mlx
                out["\(label) k\(bits) bytes_permille"] = code.byteCount * 1000 / w.nbytes
            }
        }
        return out
    }
}
