// Copyright © 2026 Eigen Labs.
// Per-row arithmetic transcribed from MLX `qmv_fast_impl`, `load_vector` and
// `qdot` (mlx/backend/metal/kernels/quantized.h), Copyright © 2023-2024 Apple
// Inc., MIT license. Pinned comparison: core 3fa8f25e6451174d7b06be372c3a24272b77d88e.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Short-row affine 8-bit projection whose rows match one-row decode.
///
/// MLX sends a 2...7-row affine matmul on gen-15+ GPUs to `qmv_wide`, whose
/// reduction order differs from the `qmv_fast` kernel a one-row decode uses, so
/// a `[1, 1+k]` rectangular verify cannot reproduce serial decode. This kernel
/// keeps `qmv_fast`'s per-row arithmetic exactly (lane mapping, factored bias,
/// FP32 accumulation, SIMD reduction and output rounding) and streams each
/// weight tile once for every row, instead of launching one matmul per row.
enum MiMoV26RowExactProjection {
    static let maximumRows = 7

    static func supports(
        shape: [Int], dtype: DType, weightShape: [Int], weightDType: DType,
        scalesDType: DType, biasesDType: DType?, groupSize: Int, bits: Int,
        mode: QuantizationMode, device: DeviceType?
    ) -> Bool {
        guard device == .gpu, mode == .affine, bits == 8, groupSize == 64,
            dtype == .bfloat16 || dtype == .float16,
            scalesDType == dtype, biasesDType == dtype, weightDType == .uint32,
            shape.count == 3, shape[0] == 1, (2 ... maximumRows).contains(shape[1]),
            weightShape.count == 2
        else { return false }
        let k = shape[2]
        let n = weightShape[0]
        // qmv_fast alignment for 8-bit: K % 256 and N % 8 (two simdgroups x four rows).
        return k > 0 && k % 256 == 0 && weightShape[1] * 4 == k && n > 0 && n % 8 == 0
    }

    /// Nil means the caller keeps its existing path. The returned array has
    /// the layer's output shape `[1, rows, N]` and includes the layer bias.
    static func apply(_ layer: QuantizedLinear, _ x: MLXArray) -> MLXArray? {
        let stream = StreamOrDevice.default
        guard ObjectIdentifier(type(of: layer)) == ObjectIdentifier(QuantizedLinear.self),
            let offsets = layer.biases,
            supports(
                shape: x.shape, dtype: x.dtype, weightShape: layer.weight.shape,
                weightDType: layer.weight.dtype, scalesDType: layer.scales.dtype,
                biasesDType: offsets.dtype, groupSize: layer.groupSize, bits: layer.bits,
                mode: layer.mode, device: MiMoV26DecodeStream.deviceType(of: stream))
        else { return nil }
        let rows = x.dim(1)
        let k = x.dim(2)
        let n = layer.weight.dim(0)
        var output = kernel(
            [x, layer.weight, layer.scales, offsets],
            template: [("T", x.dtype), ("ROWS", rows), ("KDIM", k), ("NOUT", n)],
            grid: (32, 2 * (n / 8), 1), threadGroup: (32, 2, 1),
            outputShapes: [[1, rows, n]], outputDTypes: [x.dtype], stream: stream)[0]
        if let bias = layer.bias {
            output = output + bias.asType(output.dtype)
        }
        return output
    }

    private static let kernel = MLXFast.metalKernel(
        name: "mimo_v26_row_exact_affine8_qmv",
        inputNames: ["x", "w", "scales", "biases"], outputNames: ["y"],
        source: """
            constexpr int values_per_thread = 8;
            constexpr int block_size = values_per_thread * 32;
            constexpr int group_size = 64;
            constexpr int scale_step_per_thread = group_size / values_per_thread;
            constexpr int results_per_simdgroup = 4;
            constexpr int num_simdgroups = 2;
            const int in_vec_size_g = KDIM / group_size;
            uint simd_gid = simdgroup_index_in_threadgroup;
            uint simd_lid = thread_index_in_simdgroup;
            const int out_row = threadgroup_position_in_grid.y
                    * (num_simdgroups * results_per_simdgroup)
                + simd_gid * results_per_simdgroup;

            const device uint8_t* ws = (const device uint8_t*)w + size_t(out_row) * KDIM
                + simd_lid * values_per_thread;
            const device T* sl = scales + size_t(out_row) * in_vec_size_g
                + simd_lid / scale_step_per_thread;
            const device T* bl = biases + size_t(out_row) * in_vec_size_g
                + simd_lid / scale_step_per_thread;
            const device T* xp = x + simd_lid * values_per_thread;

            float result[ROWS][results_per_simdgroup];
            for (int r = 0; r < ROWS; r++) {
                for (int row = 0; row < results_per_simdgroup; row++) { result[r][row] = 0; }
            }
            float x_thread[ROWS][values_per_thread];
            float sum[ROWS];

            for (int k = 0; k < KDIM; k += block_size) {
                for (int r = 0; r < ROWS; r++) {
                    // load_vector<T, float, 8, 8>
                    const device T* xr = xp + size_t(r) * KDIM + k;
                    float s_ = 0;
                    for (int i = 0; i < values_per_thread; i++) {
                        s_ += xr[i];
                        x_thread[r][i] = xr[i];
                    }
                    sum[r] = s_;
                }
                for (int row = 0; row < results_per_simdgroup; row++) {
                    const device uint8_t* wl = ws + size_t(row) * KDIM;
                    float scale = sl[size_t(row) * in_vec_size_g];
                    float bias = bl[size_t(row) * in_vec_size_g];
                    for (int r = 0; r < ROWS; r++) {
                        // qdot<float, 8, 8>
                        float accum = 0;
                        for (int i = 0; i < values_per_thread; i++) {
                            accum += x_thread[r][i] * wl[i];
                        }
                        result[r][row] += scale * accum + sum[r] * bias;
                    }
                }
                ws += block_size;
                sl += block_size / group_size;
                bl += block_size / group_size;
            }

            for (int r = 0; r < ROWS; r++) {
                for (int row = 0; row < results_per_simdgroup; row++) {
                    float value = simd_sum(result[r][row]);
                    if (simd_lid == 0) {
                        y[size_t(r) * NOUT + out_row + row] = static_cast<T>(value);
                    }
                }
            }
            """, ensureRowContiguous: true)
}
