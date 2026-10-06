import MLX
import MLXNN

/// Exact M1 arithmetic over a short verify window without rereading each W4
/// matrix once per position. One SIMD lane owns the same quantized values and
/// reduction order as the stock one-row QMV; the verify positions share the
/// weight load but retain independent accumulators.
private let qwen35A3BExactW4G64VerifyKernel = MLXFast.metalKernel(
    name: "qwen35_a3b_exact_w4_g64_verify_narrow2",
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
        constexpr int RESULTS_PER_SIMDGROUP = 2;
        constexpr int OUTPUTS_PER_THREADGROUP = 4;

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

/// Two same-shape exact projections sharing one launch. Matrix ownership is
/// uniform per threadgroup; lane/K ownership and every arithmetic operation
/// remain identical to the single-projection M1-ordered kernel above.
private let qwen35A3BExactW4G64PairVerifyKernel = MLXFast.metalKernel(
    name: "qwen35_a3b_exact_w4_g64_verify_pair_narrow2",
    inputNames: [
        "x", "w0", "scales0", "biases0", "w1", "scales1", "biases1",
    ],
    outputNames: ["y0", "y1"],
    source: """
        uint combined_tile = threadgroup_position_in_grid.y;
        uint matrix = combined_tile / TILES_PER_MATRIX;
        uint n_tile = combined_tile - matrix * TILES_PER_MATRIX;
        uint batch = threadgroup_position_in_grid.z;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        constexpr int VALUES_PER_THREAD = 16;
        constexpr int BLOCK_SIZE = 512;
        constexpr int RESULTS_PER_SIMDGROUP = 2;
        constexpr int OUTPUTS_PER_THREADGROUP = 4;

        int output_row = int(n_tile) * OUTPUTS_PER_THREADGROUP
            + int(simd_group) * RESULTS_PER_SIMDGROUP;
        int weight_row_bytes = K_SIZE / 2;
        int groups_per_row = K_SIZE / 64;

        const device uint8_t* selected_weights = matrix == 0
            ? (const device uint8_t*)w0 : (const device uint8_t*)w1;
        const device T* selected_scales = matrix == 0 ? scales0 : scales1;
        const device T* selected_biases = matrix == 0 ? biases0 : biases1;
        device T* selected_output = matrix == 0 ? y0 : y1;

        const device uint8_t* weight_base =
            selected_weights + output_row * weight_row_bytes + int(lane) * 8;
        const device T* scale_base =
            selected_scales + output_row * groups_per_row + int(lane) / 4;
        const device T* bias_base =
            selected_biases + output_row * groups_per_row + int(lane) / 4;
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
                    selected_output[
                        (int(batch) * VERIFY_T + t) * N_SIZE + output] = T(reduced);
                }
            }
        }
    """,
    header: "using namespace metal;",
    ensureRowContiguous: true)

/// Four exact projections sharing one launch. Each threadgroup selects one
/// matrix uniformly; the selected projection retains the single-projection
/// kernel's row ownership, K traversal, lane mapping, and reduction order.
private let qwen35A3BExactW4G64QuadVerifyKernel = MLXFast.metalKernel(
    name: "qwen35_a3b_exact_w4_g64_verify_quad_narrow2",
    inputNames: [
        "x",
        "w0", "scales0", "biases0",
        "w1", "scales1", "biases1",
        "w2", "scales2", "biases2",
        "w3", "scales3", "biases3",
    ],
    outputNames: ["y0", "y1", "y2", "y3"],
    source: """
        uint combined_tile = threadgroup_position_in_grid.y;
        uint matrix;
        uint n_tile;
        if (combined_tile < TILES0) {
            matrix = 0;
            n_tile = combined_tile;
        } else if (combined_tile < TILES01) {
            matrix = 1;
            n_tile = combined_tile - TILES0;
        } else if (combined_tile < TILES012) {
            matrix = 2;
            n_tile = combined_tile - TILES01;
        } else {
            matrix = 3;
            n_tile = combined_tile - TILES012;
        }
        uint batch = threadgroup_position_in_grid.z;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        constexpr int VALUES_PER_THREAD = 16;
        constexpr int BLOCK_SIZE = 512;
        constexpr int RESULTS_PER_SIMDGROUP = 2;
        constexpr int OUTPUTS_PER_THREADGROUP = 4;

        int output_row = int(n_tile) * OUTPUTS_PER_THREADGROUP
            + int(simd_group) * RESULTS_PER_SIMDGROUP;
        int weight_row_bytes = K_SIZE / 2;
        int groups_per_row = K_SIZE / 64;

        const device uint8_t* selected_weights = (const device uint8_t*)w0;
        const device T* selected_scales = scales0;
        const device T* selected_biases = biases0;
        device T* selected_output = y0;
        int output_size = N0_SIZE;
        if (matrix == 1) {
            selected_weights = (const device uint8_t*)w1;
            selected_scales = scales1;
            selected_biases = biases1;
            selected_output = y1;
            output_size = N1_SIZE;
        } else if (matrix == 2) {
            selected_weights = (const device uint8_t*)w2;
            selected_scales = scales2;
            selected_biases = biases2;
            selected_output = y2;
            output_size = N2_SIZE;
        } else if (matrix == 3) {
            selected_weights = (const device uint8_t*)w3;
            selected_scales = scales3;
            selected_biases = biases3;
            selected_output = y3;
            output_size = N3_SIZE;
        }

        const device uint8_t* weight_base =
            selected_weights + output_row * weight_row_bytes + int(lane) * 8;
        const device T* scale_base =
            selected_scales + output_row * groups_per_row + int(lane) / 4;
        const device T* bias_base =
            selected_biases + output_row * groups_per_row + int(lane) / 4;
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
                    selected_output[
                        (int(batch) * VERIFY_T + t) * output_size + output] = T(reduced);
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

func qwen35A3BExactW4G64Projection(
    _ linear: Linear, _ input: MLXArray
) -> MLXArray {
    // The fused verify kernel is not bit-exact to MLX's one-row QMV on the
    // current pinned runtime. Correctness support keeps the canonical M1
    // arithmetic; a fused replacement must first pass the independent oracle.
    qwen35A3BTimewiseProjection(input) { linear($0) }
}

func qwen35A3BExactW4G64ProjectionPair(
    _ first: Linear, _ second: Linear, _ input: MLXArray
) -> (MLXArray, MLXArray) {
    (
        qwen35A3BTimewiseProjection(input) { first($0) },
        qwen35A3BTimewiseProjection(input) { second($0) }
    )
}

func qwen35A3BExactW4G64ProjectionQuad(
    _ first: Linear, _ second: Linear, _ third: Linear, _ fourth: Linear,
    _ input: MLXArray
) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
    (
        qwen35A3BTimewiseProjection(input) { first($0) },
        qwen35A3BTimewiseProjection(input) { second($0) },
        qwen35A3BTimewiseProjection(input) { third($0) },
        qwen35A3BTimewiseProjection(input) { fourth($0) }
    )
}

/// Installed W8/unquantized route. Its matrices are small enough that explicit
/// M1 projection calls are faster than rereading every W4 target matrix.
func qwen35A3BExactTimewiseProjection(
    _ linear: Linear, _ input: MLXArray
) -> MLXArray {
    qwen35A3BTimewiseProjection(input) { linear($0) }
}
