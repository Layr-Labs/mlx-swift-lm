import MLX

/// Graph-only activation quantizers and exact products for the ModelOpt
/// projections. Each activation kernel reproduces, bit for bit, the activation
/// that the eager reference feeds to its matrix product. Exact NVFP4 paths keep
/// activations in float32 (holding the activation-dtype value), so the packed
/// product returns unscaled float32 sums and the caller applies the producer
/// weight scale with the same single rounding as MLX's global-scale kernels;
/// tiled prefill may keep them in the activation dtype, which holds them
/// exactly, for faster matrix kernels.
///
/// Conversions copy MLX `fp8_e4m3`/`fp4_e2m1` (`kernels/fp8.h`, `fp4.h`) and
/// `fp_quantize_dequantize` (`kernels/fp_quantized.h`) at mlx cb77239b, MIT.
enum ModelOptActivationKernels {
    static let header = """
        inline uint8_t modelopt_e4m3_bits(float value) {
            uint32_t fp8_max = 543 << 21;
            uint32_t denorm_mask = 141 << 23;
            uint32_t f_bits = as_type<uint32_t>(value);
            uint32_t sign = f_bits & 0x80000000;
            f_bits ^= sign;
            uint8_t bits;
            if (f_bits >= fp8_max) {
                bits = 0x7E;
            } else if (f_bits < (121 << 23)) {
                f_bits = as_type<uint32_t>(
                    as_type<float>(f_bits) + as_type<float>(denorm_mask));
                bits = static_cast<uint8_t>(f_bits - denorm_mask);
            } else {
                uint8_t mant_odd = (f_bits >> 20) & 1;
                f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
                f_bits += mant_odd;
                bits = static_cast<uint8_t>(f_bits >> 20);
            }
            return bits | static_cast<uint8_t>(sign >> 24);
        }

        inline float modelopt_e4m3_value(uint8_t bits) {
            half converted = as_type<half>(static_cast<uint16_t>((bits & 127) << 7));
            converted *= 256.0;
            return static_cast<float>((bits & 128) ? -converted : converted);
        }

        inline uint8_t modelopt_e2m1_bits(float x) {
            if (metal::isnan(x)) {
                return 0x7;
            }
            const uint8_t sign_bit = metal::signbit(x) ? 0x8 : 0x0;
            x = metal::abs(x);
            uint8_t bits;
            if (x > 5.0f) {
                bits = 0x7;
            } else if (x >= 3.5f) {
                bits = 0x6;
            } else if (x > 2.5f) {
                bits = 0x5;
            } else if (x >= 1.75f) {
                bits = 0x4;
            } else if (x > 1.25f) {
                bits = 0x3;
            } else if (x >= 0.75f) {
                bits = 0x2;
            } else if (x > 0.25f) {
                bits = 0x1;
            } else {
                bits = 0x0;
            }
            return bits | sign_bit;
        }

        inline float modelopt_e2m1_value(uint8_t bits) {
            half converted = as_type<half>(static_cast<ushort>((bits & 7) << 9));
            converted *= 16384.0;
            return static_cast<float>((bits & 8) ? -converted : converted);
        }
        """

    /// One threadgroup per output row. The row's expert selects the producer
    /// input scale; invalid expert IDs, NaN rows and rows beyond ModelOpt's
    /// E4M3 rounding boundary (464) become NaN, exactly as the eager mask. The
    /// last rule is data: an `overflow_divisor` of +infinity never triggers it
    /// (``ModelOptActivationOverflow/saturating``), and the encoders saturate.
    private static func nvfp4Body(selection: String) -> String {
        """
        const uint row = threadgroup_position_in_grid.y;
        \(selection)
        const uint lane = thread_index_in_simdgroup;
        const uint local = thread_position_in_threadgroup.x;
        const uint group = simdgroup_index_in_threadgroup;
        constexpr uint groups = TG / 32;
        threadgroup float group_max[groups];
        threadgroup bool group_nan[groups];
        const device T* xr = x + static_cast<size_t>(source_row) * K;
        device O* yr = out + static_cast<size_t>(row) * K;
        float row_max = 0.0f;
        bool row_nan = false;
        for (uint c = local; c < K; c += TG) {
            const float v = static_cast<float>(xr[c]);
            row_nan = row_nan || metal::isnan(v);
            row_max = metal::max(row_max, metal::abs(v));
        }
        row_max = simd_max(row_max);
        row_nan = simd_any(row_nan);
        if (lane == 0) {
            group_max[group] = row_max;
            group_nan[group] = row_nan;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        row_max = group_max[0];
        row_nan = group_nan[0];
        for (uint g = 1; g < groups; ++g) {
            row_max = metal::max(row_max, group_max[g]);
            row_nan = row_nan || group_nan[g];
        }
        bool invalid = expert < 0 || expert >= E || row_nan;
        float scale_enc = 0.0f;
        if (!invalid) {
            invalid = row_max / overflow_divisor[expert] > 464.0f;
            scale_enc = (448.0f * 6.0f) / global_scale[expert];
        }
        const float nan_value = metal::numeric_limits<float>::quiet_NaN();
        for (uint base = 0; base < K; base += TG) {
            const uint c = base + local;
            const bool inside = c < K;
            const float w = inside ? static_cast<float>(xr[c]) : 0.0f;
            const float left = simd_max(lane < 16 ? metal::abs(w) : 0.0f);
            const float right = simd_max(lane >= 16 ? metal::abs(w) : 0.0f);
            float scale_dec = lane < 16 ? left : right;
            scale_dec /= 6.0f;
            scale_dec *= scale_enc;
            const float scale = modelopt_e4m3_value(modelopt_e4m3_bits(scale_dec));
            const float scale_enc_b = (scale == 0) ? 0.0f : (scale_enc / scale);
            const float q = modelopt_e2m1_value(modelopt_e2m1_bits(w * scale_enc_b));
            const float value = static_cast<float>(static_cast<T>((scale / scale_enc) * q));
            if (inside) {
                yr[c] = static_cast<O>(invalid ? nan_value : value);
            }
        }
        """
    }

    private static let routedNVFP4 = MLXFast.metalKernel(
        name: "modelopt_nvfp4_routed_activation",
        inputNames: [
            "x", "sources", "experts", "global_scale", "overflow_divisor", "inverse_weight_scale",
        ],
        outputNames: ["out", "safe", "row_scale"],
        source: nvfp4Body(
            selection: """
                const int expert = static_cast<int>(experts[row]);
                const uint source_row = static_cast<uint>(sources[row]);
                if (thread_position_in_threadgroup.x == 0) {
                    const int clamped = metal::clamp(expert, 0, E - 1);
                    safe[row] = static_cast<uint32_t>(clamped);
                    row_scale[row] = inverse_weight_scale[clamped];
                }
                """),
        header: header)

    private static let denseNVFP4 = MLXFast.metalKernel(
        name: "modelopt_nvfp4_dense_activation",
        inputNames: ["x", "global_scale", "overflow_divisor"],
        outputNames: ["out"],
        source: nvfp4Body(
            selection: """
                const int expert = 0;
                const uint source_row = row;
                """),
        header: header)

    /// ModelOpt `_fp8_eager` activation cast with a per-tensor static scale:
    /// E4M3FN NaN codes and |x*m| > `overflow_bound` (464, or +infinity to
    /// saturate as the encoder does) are NaN. Output dtype T, unlike the NVFP4
    /// kernels, because the exact FP8 product consumes T.
    private static let fp8Kernel = MLXFast.metalKernel(
        name: "modelopt_fp8_activation",
        inputNames: ["x", "multipliers", "overflow_bound"],
        outputNames: ["out"],
        source: """
            const uint i = thread_position_in_grid.x;
            const float scaled = static_cast<float>(x[i]) * multipliers[0];
            const uint8_t bits = modelopt_e4m3_bits(scaled);
            const float nan_value = metal::numeric_limits<float>::quiet_NaN();
            const float decoded = (bits & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(bits);
            const bool invalid = metal::isnan(scaled) || metal::abs(scaled) > overflow_bound[0];
            out[i] = static_cast<T>(invalid ? nan_value : decoded * multipliers[1]);
            """,
        header: header)

    /// `T(acc * scale)` with one float32 rounding, as MLX applies a global
    /// weight scale inside its kernels. `scale` is per row or a single value.
    /// With RELU2 the result is then squared-ReLU'd exactly as NemotronH's
    /// `relu2` (MLX `maximum(y, 0)` in T, then `r * r` in T).
    private static let scaledCastKernel = MLXFast.metalKernel(
        name: "modelopt_scaled_cast",
        inputNames: ["acc", "scale"],
        outputNames: ["out"],
        source: """
            const uint i = thread_position_in_grid.x;
            const float s = ROW_SCALED ? scale[i / N] : scale[0];
            T value = static_cast<T>(acc[i] * s);
            if (RELU2) {
                const T zero = static_cast<T>(0);
                const T r = metal::isnan(static_cast<float>(value)) ? value : (value > zero ? value : zero);
                value = r * r;
            }
            out[i] = value;
            """)

    /// MLX's in-kernel `global_scale / (448 * 6)`, computed once per table by
    /// the first forward that reads it.
    private static let inverseScaleKernel = MLXFast.metalKernel(
        name: "modelopt_nvfp4_inverse_scale",
        inputNames: ["global_scale"],
        outputNames: ["inverse"],
        source: """
            const uint i = thread_position_in_grid.x;
            inverse[i] = global_scale[i] / (448.0f * 6.0f);
            """)

    /// Eager `decodedRows`: `T(decode(byte) * scale)` with E4M3FN NaN codes.
    private static let fp8WeightKernel = MLXFast.metalKernel(
        name: "modelopt_fp8_dense_weight",
        inputNames: ["bytes", "scale"],
        outputNames: ["out"],
        source: """
            const uint i = thread_position_in_grid.x;
            const uint8_t code = bytes[i];
            const float nan_value = metal::numeric_limits<float>::quiet_NaN();
            const float decoded = (code & 0x7F) == 0x7F ? nan_value : modelopt_e4m3_value(code);
            out[i] = static_cast<T>(decoded * scale[0]);
            """,
        header: header)

    /// Threadgroup width for one row: whole simdgroups, at most 256 lanes.
    private static func rowWidth(_ k: Int) -> Int { min(256, (k + 31) / 32 * 32) }

    /// `x` is `[sourceRows, K]`; row `r` reads `x[sources[r]]` (int32 `[R]`)
    /// with the input scale of `experts[r]`. Tables are `[E]` float32:
    /// `globalScale` is `inputScale * 2688`, `overflowDivisor` comes from
    /// ``ModelOptActivationOverflow/nvfp4OverflowDivisor(inputScale:)``.
    /// `experts` may be int32 or uint32. Also returns each row's clamped expert
    /// ID (uint32, for the gather) and its `inverseWeightScale` entry (float32,
    /// for ``scaledCast``).
    static func nvfp4Routed(
        _ x: MLXArray, sources: MLXArray, experts: MLXArray,
        globalScale: MLXArray, overflowDivisor: MLXArray, inverseWeightScale: MLXArray,
        activationDType: DType = .float32
    ) -> (activation: MLXArray, safeExperts: MLXArray, rowScale: MLXArray) {
        let rows = experts.dim(0)
        let width = rowWidth(x.dim(1))
        let outputs = routedNVFP4(
            [x, sources, experts, globalScale, overflowDivisor, inverseWeightScale],
            template: [
                ("T", x.dtype), ("O", activationDType), ("K", x.dim(1)),
                ("E", globalScale.dim(0)), ("TG", width),
            ],
            grid: (width, rows, 1), threadGroup: (width, 1, 1),
            outputShapes: [[rows, x.dim(1)], [rows], [rows]],
            outputDTypes: [activationDType, .uint32, .float32])
        return (outputs[0], outputs[1], outputs[2])
    }

    /// Single-matrix variant of ``nvfp4Routed``: tables are `[1]`.
    static func nvfp4Dense(
        _ x: MLXArray, globalScale: MLXArray, overflowDivisor: MLXArray,
        activationDType: DType = .float32
    ) -> MLXArray {
        let rows = x.dim(0)
        let width = rowWidth(x.dim(1))
        return denseNVFP4(
            [x, globalScale, overflowDivisor],
            template: [
                ("T", x.dtype), ("O", activationDType), ("K", x.dim(1)), ("E", 1), ("TG", width),
            ],
            grid: (width, rows, 1), threadGroup: (width, 1, 1),
            outputShapes: [[rows, x.dim(1)]], outputDTypes: [activationDType])[0]
    }

    /// `multipliers` is `[2]` float32: the activation multiplier `448 / amax`
    /// and its decoded inverse, both computed on the host as the eager path does.
    /// `overflowBound` is `[1]` float32, ``ModelOptActivationOverflow/fp8NaNBound``
    /// (the eager 464 unless given). Returns dtype `x.dtype`.
    static func fp8(
        _ x: MLXArray, multipliers: MLXArray,
        overflowBound: MLXArray = MLXArray([ModelOptActivationOverflow.eagerNaN.fp8NaNBound])
    ) -> MLXArray {
        let count = x.size
        return fp8Kernel(
            [x, multipliers, overflowBound],
            template: [("T", x.dtype)],
            grid: (count, 1, 1), threadGroup: (min(256, count), 1, 1),
            outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
    }

    /// `acc` is float32 `[R, N]`; `scale` is `[R]` (per row) or `[1]`.
    /// `squaredReLU` fuses NemotronH's `relu2` on the dtype result.
    static func scaledCast(
        _ acc: MLXArray, scale: MLXArray, dtype: DType, squaredReLU: Bool = false
    ) -> MLXArray {
        let count = acc.size
        return scaledCastKernel(
            [acc, scale],
            template: [
                ("T", dtype), ("N", acc.dim(-1)), ("ROW_SCALED", scale.size > 1),
                ("RELU2", squaredReLU),
            ],
            grid: (count, 1, 1), threadGroup: (min(256, count), 1, 1),
            outputShapes: [acc.shape], outputDTypes: [dtype])[0]
    }

    static func inverseWeightScale(_ globalScale: MLXArray) -> MLXArray {
        inverseScaleKernel(
            [globalScale], grid: (globalScale.size, 1, 1),
            threadGroup: (min(256, globalScale.size), 1, 1),
            outputShapes: [globalScale.shape], outputDTypes: [.float32])[0]
    }

    /// `bytes` is uint8 E4M3FN; `scale` is `[1]` float32.
    static func fp8DenseWeight(_ bytes: MLXArray, scale: MLXArray, dtype: DType) -> MLXArray {
        let count = bytes.size
        return fp8WeightKernel(
            [bytes, scale], template: [("T", dtype)],
            grid: (count, 1, 1), threadGroup: (min(256, count), 1, 1),
            outputShapes: [bytes.shape], outputDTypes: [dtype])[0]
    }

}
