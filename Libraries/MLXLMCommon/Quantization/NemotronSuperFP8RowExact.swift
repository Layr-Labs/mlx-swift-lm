import MLX
import MLXFast

/// Several activation rows through a calibrated FP8 projection in one pass
/// over the weights, each row bit-identical to the one-row `.exactReference`
/// graph call (the fused vector kernel). Used by the Super MTP verify window;
/// `NemotronSuperMTPRowExactnessTests` checks it at every Super FP8 shape.
extension ModelOptCalibratedLinear {
    /// Rows the multi-row kernel keeps in registers.
    public static let superRowExactFP8MaximumRows = 8

    /// `[..., K]` with at most `superRowExactFP8MaximumRows` rows, or nil when
    /// this projection's one-row call is not the fused vector kernel (NVFP4,
    /// eager, `.packed`, K <= 64 or failed calibration): callers then run one
    /// row at a time. Throws only on a malformed input.
    public func superRowExactFP8(_ input: MLXArray) throws -> MLXArray? {
        guard format == .fp8, execution == .graph, fp8Product == .exactReference,
            inputDimensions > 64, let constants = graphConstants,
            case .fp8(let multipliers, _, let invalidOutputs) = constants.storage,
            weight.dtype == .uint8, weight.shape == [outputDimensions, inputDimensions]
        else { return nil }
        guard input.ndim > 0, input.dim(-1) == inputDimensions, input.dtype == weightDType else {
            throw Failure.invalidInput
        }
        let rows = input.size / inputDimensions
        guard rows >= 1, rows <= Self.superRowExactFP8MaximumRows else { return nil }
        let activation = ModelOptActivationKernels.fp8(
            input.reshaped(rows, inputDimensions), multipliers: multipliers
        ).asType(weightDType)
        let pieces = NemotronSuperFP8RowKernel.oneRowRuns(
            outputs: outputDimensions, inputs: inputDimensions
        ).map { run in
            NemotronSuperFP8RowKernel.product(
                weight[run.rows, 0...], scale: constants.outputScale, vectors: activation,
                splitsK: run.splitsK)
        }
        var value = pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: -1)
        if let invalidOutputs {
            value = which(invalidOutputs, MLXArray(Float.nan).asType(weightDType), value)
        }
        return value.reshaped(Array(input.shape.dropLast()) + [outputDimensions])
    }
}

enum NemotronSuperFP8RowKernel {
    /// The output-row runs the one-row exact product dispatches (eager tiles
    /// of a one-row call merged while they share MLX's K split); each output's
    /// reduction order depends only on its run's split.
    static func oneRowRuns(outputs: Int, inputs: Int) -> [(rows: Range<Int>, splitsK: Bool)] {
        let limit = (64 << 20) / 32
        let live = inputs + outputs
        let tileElements = live <= limit ? limit - live : 1 << 25
        let tileRows = max(1, min(outputs, tileElements / inputs))
        var runs: [(rows: Range<Int>, splitsK: Bool)] = []
        for start in stride(from: 0, to: outputs, by: tileRows) {
            let tile = start ..< min(outputs, start + tileRows)
            let splitsK = inputs >= 16 * tile.count
            if let last = runs.last, last.splitsK == splitsK {
                runs[runs.count - 1].rows = last.rows.lowerBound ..< tile.upperBound
            } else {
                runs.append((tile, splitsK))
            }
        }
        return runs
    }

    /// `T(x_r . T(decode(bytes) * scale))` for every row `x_r` of `vectors`
    /// (`[R, K]`, dtype T) against E4M3FN `bytes` (`[N, K]`, K > 64). Every
    /// row repeats the vector kernel's per-output order of products, simd
    /// shuffles and threadgroup sums; the decoded weight is shared. `[R, N]`.
    static func product(
        _ bytes: MLXArray, scale: MLXArray, vectors: MLXArray, splitsK: Bool
    ) -> MLXArray {
        let (outputs, inputs) = (bytes.dim(0), bytes.dim(1))
        let rows = vectors.dim(0)
        let bm = splitsK ? 1 : (outputs >= 4096 ? 8 : 4)
        let bn = splitsK ? 8 : 1
        let tm = outputs < 4 ? 1 : 4
        let groups = (outputs + bm * tm - 1) / (bm * tm)
        return kernel(
            [bytes, scale, vectors],
            template: [
                ("T", vectors.dtype), ("BM", bm), ("BN", bn), ("TM", tm), ("K", inputs),
                ("N", outputs), ("ROWS", rows),
            ],
            grid: (groups * 32, bn, bm), threadGroup: (32, bn, bm),
            outputShapes: [[rows, outputs]], outputDTypes: [vectors.dtype])[0]
    }

    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_super_fp8_rows",
        inputNames: ["bytes", "scale", "in_vec"],
        outputNames: ["out_vec"],
        source: """
            constexpr int SM = 1;
            constexpr int SN = 32;
            constexpr int TN = 4;
            constexpr int threadsM = BM * SM;
            constexpr int threadsN = BN * SN;
            constexpr int blockM = threadsM * TM;
            constexpr int blockN = threadsN * TN;
            constexpr int rowStride = BN * (blockM + TM);
            threadgroup float tgp_memory[BN > 1 ? ROWS * rowStride : 1];
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const float weight_scale = scale[0];
            const float nan_value = metal::numeric_limits<float>::quiet_NaN();
            thread float result[ROWS][TM];
            for (int r = 0; r < ROWS; r++) {
                for (int tm = 0; tm < TM; tm++) {
                    result[r][tm] = 0;
                }
            }
            thread T inter[TN];
            thread float v_coeff[ROWS][TN];
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
            const device uint8_t* mat = bytes + size_t(out_row) * K;
            constexpr int n_iter = K / blockN;
            constexpr int leftover = K - blockN * n_iter;
            for (int i = 0; i < n_iter; ++i) {
                for (int r = 0; r < ROWS; r++) {
                    for (int tn = 0; tn < TN; tn++) {
                        v_coeff[r][tn] = static_cast<float>(in_vec[r * K + bn + tn]);
                    }
                }
                int mat_offset = 0;
                for (int tm = 0; tm < TM; tm++) {
                    for (int tn = 0; tn < TN; tn++) {
                        const uint8_t code = mat[mat_offset + bn + tn];
                        const float decoded =
                            (code & 0x7F) == 0x7F ? nan_value : super_e4m3_value(code);
                        inter[tn] = static_cast<T>(decoded * weight_scale);
                    }
                    for (int r = 0; r < ROWS; r++) {
                        for (int tn = 0; tn < TN; tn++) {
                            result[r][tm] += inter[tn] * v_coeff[r][tn];
                        }
                    }
                    mat_offset += K;
                }
                bn += blockN;
            }
            if (leftover > 0) {
                for (int r = 0; r < ROWS; r++) {
                    for (int tn = 0; tn < TN; tn++) {
                        v_coeff[r][tn] =
                            bn + tn < K ? static_cast<float>(in_vec[r * K + bn + tn]) : 0.0f;
                    }
                }
                for (int tm = 0; tm < TM; tm++) {
                    for (int tn = 0; tn < TN; tn++) {
                        if (bn + tn < K) {
                            const uint8_t code = mat[tm * K + bn + tn];
                            const float decoded =
                                (code & 0x7F) == 0x7F ? nan_value : super_e4m3_value(code);
                            inter[tn] = static_cast<T>(decoded * weight_scale);
                        } else {
                            inter[tn] = T(0);
                        }
                    }
                    for (int r = 0; r < ROWS; r++) {
                        for (int tn = 0; tn < TN; tn++) {
                            result[r][tm] += inter[tn] * v_coeff[r][tn];
                        }
                    }
                }
            }
            for (int r = 0; r < ROWS; r++) {
                for (int tm = 0; tm < TM; tm++) {
                    for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                        result[r][tm] += simd_shuffle_down(result[r][tm], sn);
                    }
                }
            }
            if (BN > 1) {
                if (thrN == 0) {
                    for (int r = 0; r < ROWS; r++) {
                        threadgroup float* tgp_results =
                            tgp_memory + r * rowStride + sgN * (blockM + TM) + bm;
                        for (int tm = 0; tm < TM; tm++) {
                            tgp_results[tm] = result[r][tm];
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (sgN == 0) {
                        for (int r = 0; r < ROWS; r++) {
                            threadgroup float* tgp_results = tgp_memory + r * rowStride + bm;
                            for (int sgn = 1; sgn < BN; sgn++) {
                                for (int tm = 0; tm < TM; tm++) {
                                    result[r][tm] += tgp_results[sgn * (blockM + TM) + tm];
                                }
                            }
                        }
                    }
                }
            }
            if (simdN == 0 && thrN == 0) {
                for (int r = 0; r < ROWS; r++) {
                    for (int tm = 0; tm < TM; tm++) {
                        out_vec[r * N + out_row + tm] = static_cast<T>(result[r][tm]);
                    }
                }
            }
            """,
        header: """
            inline float super_e4m3_value(uint8_t bits) {
                half converted = as_type<half>(static_cast<uint16_t>((bits & 127) << 7));
                converted *= 256.0;
                return static_cast<float>((bits & 128) ? -converted : converted);
            }
            """)
}
