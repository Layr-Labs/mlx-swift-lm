import MLX
import MLXNN

/// Exact M1 arithmetic over a short verify window without rereading each W4
/// matrix once per position. One SIMD lane owns the same quantized values and
/// reduction order as the stock one-row QMV; the verify positions share the
/// weight load but retain independent accumulators.
private let qwen35A3BExactW4G64VerifyKernel = MLXFast.metalKernel(
    name: "qwen35_a3b_exact_w4_g64_verify",
    inputNames: ["x", "w", "scales", "biases"],
    outputNames: ["y"],
    source: """
        uint n_tile = threadgroup_position_in_grid.y;
        uint batch = threadgroup_position_in_grid.z;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        constexpr int PACK_FACTOR = 8;
        constexpr int VALUES_PER_THREAD = 16;
        constexpr int BLOCK_SIZE = 512;
        constexpr int RESULTS_PER_SIMDGROUP = 4;
        constexpr int OUTPUTS_PER_THREADGROUP = 8;

        int output_row = int(n_tile) * OUTPUTS_PER_THREADGROUP
            + int(simd_group) * RESULTS_PER_SIMDGROUP;
        int weight_row_bytes = K_SIZE / 2;
        int groups_per_row = K_SIZE / 64;

        const device uint8_t* weight_base =
            (const device uint8_t*)w + output_row * weight_row_bytes
            + int(lane) * 8;
        const device T* scale_base =
            scales + output_row * groups_per_row + int(lane) / 4;
        const device T* bias_base =
            biases + output_row * groups_per_row + int(lane) / 4;
        const device T* input_base =
            x + int(batch) * VERIFY_T * K_SIZE + int(lane) * VALUES_PER_THREAD;

        float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
        float input_values[VERIFY_T][VALUES_PER_THREAD];
        for (int t = 0; t < VERIFY_T; ++t) {
            for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
                result[t][row] = 0.0f;
            }
        }

        const device uint8_t* weight_block = weight_base;
        const device T* scale_block = scale_base;
        const device T* bias_block = bias_base;
        const device T* input_block = input_base;

        for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
            float sums[VERIFY_T];
            for (int t = 0; t < VERIFY_T; ++t) {
                const device T* input = input_block + t * K_SIZE;
                float sum = 0.0f;
                for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
                    sum += input[i] + input[i + 1] + input[i + 2] + input[i + 3];
                    input_values[t][i] = input[i];
                    input_values[t][i + 1] = input[i + 1] / 16.0f;
                    input_values[t][i + 2] = input[i + 2] / 256.0f;
                    input_values[t][i + 3] = input[i + 3] / 4096.0f;
                }
                sums[t] = sum;
            }

            for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
                const device uint16_t* packed =
                    (const device uint16_t*)(weight_block + row * weight_row_bytes);
                const device T* row_scales = scale_block + row * groups_per_row;
                const device T* row_biases = bias_block + row * groups_per_row;
                float scale = float(row_scales[0]);
                float bias = float(row_biases[0]);
                for (int t = 0; t < VERIFY_T; ++t) {
                    float dot = 0.0f;
                    for (int i = 0; i < VALUES_PER_THREAD / 4; ++i) {
                        dot +=
                            input_values[t][4 * i] * (packed[i] & 0x000f)
                            + input_values[t][4 * i + 1] * (packed[i] & 0x00f0)
                            + input_values[t][4 * i + 2] * (packed[i] & 0x0f00)
                            + input_values[t][4 * i + 3] * (packed[i] & 0xf000);
                    }
                    result[t][row] += scale * dot + sums[t] * bias;
                }
            }

            weight_block += BLOCK_SIZE / 2;
            scale_block += BLOCK_SIZE / 64;
            bias_block += BLOCK_SIZE / 64;
            input_block += BLOCK_SIZE;
        }

        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            int output = output_row + row;
            for (int t = 0; t < VERIFY_T; ++t) {
                float reduced = simd_sum(result[t][row]);
                if (lane == 0) {
                    y[(int(batch) * VERIFY_T + t) * N_SIZE + output] = T(reduced);
                }
            }
        }
    """,
    header: "using namespace metal;",
    ensureRowContiguous: true)

/// Pure shape transform used by the exact fallback and its construction test.
/// The projection is built once per verify column, so every call has logical
/// M1 even though the results are returned as the original rectangle.
func qwen35A3BTimewiseProjection(
    _ input: MLXArray, projection: (MLXArray) -> MLXArray
) -> MLXArray {
    precondition(input.ndim == 3 && input.dim(1) > 1)
    return concatenated(
        (0 ..< input.dim(1)).map { position in
            projection(input[0..., position ..< (position + 1), 0...])
        }, axis: 1)
}

/// Installed W4/g64 target-verification route. Artifact inspection fixes this
/// packing before model construction, so the measured path executes directly:
/// there is no eligibility check and no stock fallback.
func qwen35A3BExactW4G64Projection(
    _ linear: Linear, _ input: MLXArray
) -> MLXArray {
    let quantized = linear as! QuantizedLinear
    let quantizationBiases = quantized.biases!
    let batch = input.dim(0)
    let width = input.dim(1)
    let inputSize = input.dim(2)
    let outputSize = quantized.weight.dim(0)
    return qwen35A3BExactW4G64VerifyKernel(
        [input, quantized.weight, quantized.scales, quantizationBiases],
        template: [
            ("T", input.dtype),
            ("VERIFY_T", width),
            ("K_SIZE", inputSize),
            ("N_SIZE", outputSize),
        ],
        grid: (32, 2 * (outputSize / 8), batch),
        threadGroup: (32, 2, 1),
        outputShapes: [[batch, width, outputSize]],
        outputDTypes: [input.dtype])[0]
}

/// Installed W8/unquantized route. Its matrices are small enough that explicit
/// M1 projection calls are faster than rereading every W4 target matrix.
func qwen35A3BExactTimewiseProjection(
    _ linear: Linear, _ input: MLXArray
) -> MLXArray {
    qwen35A3BTimewiseProjection(input) { linear($0) }
}
