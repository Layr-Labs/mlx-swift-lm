import MLX

/// Tiled NVFP4 grouped GEMM for prefill: rows sorted by expert are split into
/// per-expert tiles (no tile mixes experts, so skewed or small groups waste at
/// most one partial tile each), every tile multiplies its rows by the expert's
/// packed E2M1 weights with E4M3 block scales dequantized into threadgroup
/// memory, simdgroup matrix products accumulate in float32, and the epilogue
/// applies the expert's weight scale once (optionally NemotronH's relu2).
/// Producer bits and scales are those of the reference; the summation order
/// differs, as on the tiled path.
enum ModelOptNVFP4GroupedGEMM {
    static let rowsPerTile = 32
    static let outputsPerTile = 128
    static let depthPerTile = 32
    /// Multiply activation-dtype matrices directly (float32 accumulators);
    /// measured slightly slower than float32 fragments on M3 Ultra.
    static let nativeMatrices = false
    /// Benchmarks compare against MLX's sorted gather product by clearing this.
    nonisolated(unsafe) static var isEnabled = true

    /// One threadgroup: per expert, the first row (binary search in the sorted
    /// IDs), the row count and the number of tiles; an exclusive scan places
    /// each expert's tiles. Unused tail entries get a zero row count. Sorted
    /// IDs need at most `tile_capacity` tiles; unsorted ones could claim more,
    /// and those tiles are dropped rather than written past the table.
    private static let tileKernel = MLXFast.metalKernel(
        name: "modelopt_nvfp4_expert_tiles",
        inputNames: ["ids", "row_count", "tile_capacity"],
        outputNames: ["tiles"],
        source: """
            const uint e = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint group = simdgroup_index_in_threadgroup;
            threadgroup uint group_sums[32];
            const uint rows = static_cast<uint>(row_count);
            uint lo = 0;
            uint hi = 0;
            if (e < E) {
                uint a = 0;
                uint b = rows;
                while (a < b) {
                    const uint mid = (a + b) / 2;
                    if (ids[mid] < e) { a = mid + 1; } else { b = mid; }
                }
                lo = a;
                b = rows;
                while (a < b) {
                    const uint mid = (a + b) / 2;
                    if (ids[mid] <= e) { a = mid + 1; } else { b = mid; }
                }
                hi = a;
            }
            const uint count = hi - lo;
            const uint ntiles = (count + BM - 1) / BM;
            const uint within = simd_prefix_exclusive_sum(ntiles);
            const uint total_in_group = simd_sum(ntiles);
            if (lane == 0) {
                group_sums[group] = total_in_group;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint before = 0;
            uint total = 0;
            for (uint g = 0; g < TPG / 32; g++) {
                if (g < group) { before += group_sums[g]; }
                total += group_sums[g];
            }
            const uint capacity = static_cast<uint>(tile_capacity);
            const uint start = before + within;
            for (uint t = 0; t < ntiles && start + t < capacity; t++) {
                const uint index = (start + t) * 3;
                tiles[index] = e;
                tiles[index + 1] = lo + t * BM;
                tiles[index + 2] = min(static_cast<uint>(BM), count - t * BM);
            }
            for (uint index = total + e; index < capacity; index += TPG) {
                tiles[index * 3] = 0;
                tiles[index * 3 + 1] = 0;
                tiles[index * 3 + 2] = 0;
            }
            """,
        header: ModelOptActivationKernels.header)

    private static let gemmKernel = MLXFast.metalKernel(
        name: "modelopt_nvfp4_grouped_gemm",
        inputNames: ["x", "w", "scales", "tiles", "expert_scale"],
        outputNames: ["y"],
        source: """
            constexpr int XS = BK + 8;
            constexpr int WM = 2;
            constexpr int WN = 2;
            constexpr int TM = BM / (8 * WM);
            constexpr int TN = BN / (8 * WN);
            constexpr int THREADS = 32 * WM * WN;
            constexpr int PACKS = BK / 8;
            threadgroup T Xs[BM * XS];
            threadgroup T Ws[BN * XS];
            const uint tile = threadgroup_position_in_grid.y;
            const uint expert = tiles[tile * 3];
            const int start = int(tiles[tile * 3 + 1]);
            const int count = int(tiles[tile * 3 + 2]);
            if (count == 0) {
                return;
            }
            const int n0 = int(threadgroup_position_in_grid.x) * BN;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const int lid = int(sg) * 32 + int(lane);
            const int sm = int(sg / WN) * TM * 8;
            const int sn = int(sg % WN) * TN * 8;
            const short qid = lane / 4;
            const short fm = (qid & 4) + ((lane / 2) % 4);
            const short fn = (qid & 2) * 2 + (lane % 2) * 2;
            simdgroup_matrix<float, 8, 8> acc[TM][TN];
            #pragma clang loop unroll(full)
            for (int i = 0; i < TM; i++) {
                #pragma clang loop unroll(full)
                for (int j = 0; j < TN; j++) {
                    acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                }
            }
            const device uint32_t* wexp = w + size_t(expert) * N * (K / 8);
            const device uint8_t* sexp = scales + size_t(expert) * N * (K / 16);
            for (int k = 0; k < K; k += BK) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                // X: BM rows x BK values as 4-wide vectors.
                #pragma clang loop unroll(full)
                for (int idx = lid; idx < BM * BK / 4; idx += THREADS) {
                    const int r = idx / (BK / 4);
                    const int c = (idx % (BK / 4)) * 4;
                    vec<T, 4> v = vec<T, 4>(0);
                    if (r < count) {
                        v = *reinterpret_cast<const device vec<T, 4>*>(
                            x + size_t(start + r) * K + k + c);
                    }
                    *reinterpret_cast<threadgroup vec<T, 4>*>(Xs + r * XS + c) = v;
                }
                // W: BN rows x BK/8 packs of eight E2M1 codes.
                #pragma clang loop unroll(full)
                for (int idx = lid; idx < BN * PACKS; idx += THREADS) {
                    const int j = idx / PACKS;
                    const int p = idx % PACKS;
                    const int n = n0 + j;
                    uint32_t packed = 0;
                    float s = 0.0f;
                    if (n < N) {
                        packed = wexp[size_t(n) * (K / 8) + k / 8 + p];
                        s = modelopt_e4m3_value(sexp[size_t(n) * (K / 16) + (k + p * 8) / 16]);
                    }
                    vec<T, 4> lo;
                    vec<T, 4> hi;
                    #pragma clang loop unroll(full)
                    for (int q = 0; q < 4; q++) {
                        lo[q] = static_cast<T>(modelopt_e2m1_lut[(packed >> (4 * q)) & 0xF] * s);
                        hi[q] = static_cast<T>(modelopt_e2m1_lut[(packed >> (4 * q + 16)) & 0xF] * s);
                    }
                    threadgroup vec<T, 4>* dst =
                        reinterpret_cast<threadgroup vec<T, 4>*>(Ws + j * XS + p * 8);
                    dst[0] = lo;
                    dst[1] = hi;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                // A simdgroup whose rows are all past the tile's count (a short
                // last tile) has only padding to multiply.
                if (sm >= count) {
                    continue;
                }
                #pragma clang loop unroll(full)
                for (int kk = 0; kk < BK; kk += 8) {
                    simdgroup_matrix<MT, 8, 8> a[TM];
                    simdgroup_matrix<MT, 8, 8> b[TN];
                    if (NATIVE) {
                        #pragma clang loop unroll(full)
                        for (int i = 0; i < TM; i++) {
                            simdgroup_load(a[i], (const threadgroup MT*)(Xs + (sm + i * 8) * XS + kk), XS);
                        }
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < TN; j++) {
                            simdgroup_load(
                                b[j], (const threadgroup MT*)(Ws + (sn + j * 8) * XS + kk), XS,
                                ulong2(0, 0), true);
                        }
                    } else {
                        #pragma clang loop unroll(full)
                        for (int i = 0; i < TM; i++) {
                            const threadgroup T* row = Xs + (sm + i * 8 + fm) * XS + kk + fn;
                            a[i].thread_elements()[0] = static_cast<MT>(row[0]);
                            a[i].thread_elements()[1] = static_cast<MT>(row[1]);
                        }
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < TN; j++) {
                            const threadgroup T* col = Ws + (sn + j * 8 + fn) * XS + kk + fm;
                            b[j].thread_elements()[0] = static_cast<MT>(col[0]);
                            b[j].thread_elements()[1] = static_cast<MT>(col[XS]);
                        }
                    }
                    #pragma clang loop unroll(full)
                    for (int i = 0; i < TM; i++) {
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < TN; j++) {
                            simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                        }
                    }
                }
            }
            const float weight_scale = expert_scale[expert];
            #pragma clang loop unroll(full)
            for (int i = 0; i < TM; i++) {
                const int r = sm + i * 8 + fm;
                if (r >= count) {
                    continue;
                }
                device T* out_row = y + size_t(start + r) * N;
                #pragma clang loop unroll(full)
                for (int j = 0; j < TN; j++) {
                    #pragma clang loop unroll(full)
                    for (int e = 0; e < 2; e++) {
                        const int c = n0 + sn + j * 8 + fn + e;
                        if (c >= N) {
                            continue;
                        }
                        T value = static_cast<T>(acc[i][j].thread_elements()[e] * weight_scale);
                        if (RELU2) {
                            const T zero = static_cast<T>(0);
                            const T rr =
                                metal::isnan(static_cast<float>(value))
                                ? value : (value > zero ? value : zero);
                            value = rr * rr;
                        }
                        out_row[c] = value;
                    }
                }
            }
            """,
        header: ModelOptActivationKernels.header + """

            // E2M1 code -> value, equal to modelopt_e2m1_value for every code.
            constant float modelopt_e2m1_lut[16] = {
                0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
            """)

    /// The tile scheduler is one threadgroup with a thread per expert.
    static let maximumExperts = 1024

    /// Shapes where this kernel beats MLX's sorted gather product plus the
    /// separate scale/relu2 pass (M3 Ultra, 2048-token chunks): Super's fc1
    /// (K 1024, 17.9 vs 20.2 ms); at K 2688 MLX is faster (17.1 vs 18.0 ms).
    /// Banks above ``maximumExperts`` (loaders accept up to 4096) take MLX's.
    static func supports(inputs: Int, experts: Int) -> Bool {
        inputs % 32 == 0 && inputs <= 1024 && experts <= maximumExperts
    }

    /// The tiles of `experts` (uint32 `[R]`, sorted ascending, all below
    /// `expertCount`): `[capacity * 3]` uint32, each tile's expert, first row
    /// and row count, then zero row counts up to `R / rowsPerTile` (rounded
    /// up) plus `expertCount` tiles.
    static func tileTable(
        _ experts: MLXArray, expertCount: Int, rowsPerTile: Int = rowsPerTile
    ) -> MLXArray {
        precondition(
            expertCount <= maximumExperts,
            "grouped NVFP4 GEMM supports at most \(maximumExperts) experts")
        let rows = experts.dim(0)
        let capacity = (rows + rowsPerTile - 1) / rowsPerTile + expertCount
        let threads = min(maximumExperts, (expertCount + 31) / 32 * 32)
        return tileKernel(
            [experts, MLXArray(Int32(rows)), MLXArray(Int32(capacity))],
            template: [("E", expertCount), ("BM", rowsPerTile), ("TPG", threads)],
            grid: (threads, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[capacity * 3]], outputDTypes: [.uint32])[0]
    }

    /// `[R, N]` in `x.dtype`: row r times expert `experts[r]` (uint32, sorted
    /// ascending, all < E) of `w` (`[E, N, K/8]`) and `scales`
    /// (`[E, N, K/16]`), times `expertScale[e]` (`[E]` float32), optionally
    /// squared-ReLU'd. `x` is `[R, K]` holding the quantized activations.
    /// Unsorted IDs give undefined rows but no access outside the arrays.
    static func product(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, experts: MLXArray,
        expertScale: MLXArray, squaredReLU: Bool,
        rowsPerTile: Int = rowsPerTile, outputsPerTile: Int = outputsPerTile,
        depthPerTile: Int = depthPerTile, nativeMatrices: Bool = nativeMatrices
    ) -> MLXArray {
        let rows = x.dim(0)
        let inputs = x.dim(1)
        let outputs = weight.dim(1)
        let tiles = tileTable(experts, expertCount: weight.dim(0), rowsPerTile: rowsPerTile)
        let capacity = tiles.dim(0) / 3
        let depth = inputs % depthPerTile == 0 ? depthPerTile : 32
        let blocks = (outputs + outputsPerTile - 1) / outputsPerTile
        return gemmKernel(
            [x, weight, scales, tiles, expertScale],
            template: [
                ("T", x.dtype), ("K", inputs), ("N", outputs), ("BM", rowsPerTile),
                ("BN", outputsPerTile), ("BK", depth), ("RELU2", squaredReLU),
                ("MT", nativeMatrices ? x.dtype : DType.float32), ("NATIVE", nativeMatrices),
            ],
            grid: (blocks * 32, capacity * 4, 1), threadGroup: (32, 4, 1),
            outputShapes: [[rows, outputs]], outputDTypes: [x.dtype])[0]
    }
}
