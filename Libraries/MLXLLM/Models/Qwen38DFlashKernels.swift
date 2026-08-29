// Copyright 2026 bstnxbt
// SPDX-License-Identifier: Apache-2.0

import MLX
import MLXFast

private let qwen38QKRMSRoPEKernel = MLXFast.metalKernel(
    name: "qwen38_qk_rms_rope_bf16_h256_r64_v1",
    inputNames: ["q", "k", "q_weight", "k_weight", "eps", "offset", "log2_base"],
    outputNames: ["q_out", "k_out"],
    source: """
            constexpr uint n_reads = 4;
            constexpr uint simd_size = 32;
            constexpr uint rotary_dimensions = 64;
            constexpr uint rotary_pairs = rotary_dimensions / 2;

            uint row = threadgroup_position_in_grid.x;
            uint thread_id = thread_position_in_threadgroup.x;
            uint simd_thread = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            uint batch_size = uint(q_shape[0]);
            uint sequence_length = uint(q_shape[1]);
            uint query_heads = uint(q_shape[2]);
            uint key_heads = uint(k_shape[2]);
            uint axis_size = uint(q_shape[3]);
            uint query_rows = batch_size * query_heads * sequence_length;
            bool is_query = row < query_rows;
            uint local_row = is_query ? row : row - query_rows;
            uint head_count = is_query ? query_heads : key_heads;
            uint batch = local_row / (head_count * sequence_length);
            uint head_sequence = local_row % (head_count * sequence_length);
            uint head = head_sequence / sequence_length;
            uint sequence = head_sequence % sequence_length;

            ulong input_base;
            ulong input_axis_stride;
            ulong output_base = ulong(local_row) * ulong(axis_size);
            if (is_query) {
                input_base = ulong(batch) * ulong(q_strides[0])
                    + ulong(sequence) * ulong(q_strides[1])
                    + ulong(head) * ulong(q_strides[2]);
                input_axis_stride = ulong(q_strides[3]);
            } else {
                input_base = ulong(batch) * ulong(k_strides[0])
                    + ulong(sequence) * ulong(k_strides[1])
                    + ulong(head) * ulong(k_strides[2]);
                input_axis_stride = ulong(k_strides[3]);
            }

            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[simd_size];
            threadgroup bfloat normalized[256];
            float acc = 0.0f;
            uint first = thread_id * n_reads;
            for (uint i = 0; i < n_reads; ++i) {
                uint element = first + i;
                if (element < axis_size) {
                    ulong index = input_base + ulong(element) * input_axis_stride;
                    float value = is_query ? float(q[index]) : float(k[index]);
                    acc += value * value;
                }
            }
            acc = simd_sum(acc);
            if (simd_group == 0) local_sums[simd_thread] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_thread == 0) local_sums[simd_group] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                acc = simd_sum(local_sums[simd_thread]);
                if (simd_thread == 0) {
                    local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float inv_mean = local_inv_mean[0];
            for (uint i = 0; i < n_reads; ++i) {
                uint element = first + i;
                if (element < axis_size) {
                    ulong index = input_base + ulong(element) * input_axis_stride;
                    bfloat input_value = is_query ? q[index] : k[index];
                    bfloat rms_value = bfloat(float(input_value) * inv_mean);
                    bfloat weight = is_query ? q_weight[element] : k_weight[element];
                    normalized[element] = weight * rms_value;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = 0; i < n_reads; ++i) {
                uint element = first + i;
                if (element >= rotary_dimensions && element < axis_size) {
                    if (is_query) q_out[output_base + element] = normalized[element];
                    else k_out[output_base + element] = normalized[element];
                }
            }
            if (thread_id < rotary_pairs / n_reads) {
                for (uint i = 0; i < n_reads; ++i) {
                    uint pair = first + i;
                    float d = float(pair) / float(rotary_pairs);
                    float inv_freq = metal::exp2(-d * float(log2_base));
                    float position = float(int(sequence) + int(offset));
                    float theta = position * inv_freq;
                    float costheta = metal::fast::cos(theta);
                    float sintheta = metal::fast::sin(theta);
                    float x1 = float(normalized[pair]);
                    float x2 = float(normalized[pair + rotary_pairs]);
                    bfloat rx1 = bfloat(x1 * costheta - x2 * sintheta);
                    bfloat rx2 = bfloat(x1 * sintheta + x2 * costheta);
                    if (is_query) {
                        q_out[output_base + pair] = rx1;
                        q_out[output_base + pair + rotary_pairs] = rx2;
                    } else {
                        k_out[output_base + pair] = rx1;
                        k_out[output_base + pair + rotary_pairs] = rx2;
                    }
                }
            }
        """,
    ensureRowContiguous: false)

func qwen38FusedQKRMSRoPE(
    queries: MLXArray,
    keys: MLXArray,
    queryWeight: MLXArray,
    keyWeight: MLXArray,
    epsilon: Float,
    offset: Int
) -> (MLXArray, MLXArray) {
    let batch = queries.dim(0)
    let length = queries.dim(1)
    let outputs = qwen38QKRMSRoPEKernel(
        [
            queries, keys, queryWeight, keyWeight,
            MLXArray(epsilon), MLXArray(Int32(offset)), MLXArray(23.253496664211536),
        ],
        grid: (batch * length * 28 * 64, 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [
            [batch, 24, length, 256],
            [batch, 4, length, 256],
        ],
        outputDTypes: [.bfloat16, .bfloat16])
    return (outputs[0], outputs[1])
}

private let qwen38AddRMSNormKernel = MLXFast.metalKernel(
    name: "qwen38_add_rmsnorm_bf16_h5120_v1",
    inputNames: ["base", "delta", "weight", "eps"],
    outputNames: ["hidden", "normed"],
    source: """
            constexpr uint SIMD_SIZE = 32;
            constexpr uint N_READS = 4;
            constexpr uint AXIS = 5120;
            uint row = threadgroup_position_in_grid.x;
            uint lid = thread_position_in_threadgroup.x;
            uint lsize = threads_per_threadgroup.x;
            uint simd_lane = thread_index_in_simdgroup;
            uint simd_group = simdgroup_index_in_threadgroup;
            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[SIMD_SIZE];
            size_t row_offset = size_t(row) * size_t(AXIS);
            float acc = 0.0f;

            for (uint start = 0; start < AXIS; start += lsize * N_READS) {
                uint first = start + lid * N_READS;
                _Pragma("unroll")
                for (uint i = 0; i < N_READS; ++i) {
                    uint index = first + i;
                    if (index < AXIS) {
                        T value = base[row_offset + index] + delta[row_offset + index];
                        acc += float(value) * float(value);
                    }
                }
            }
            acc = simd_sum(acc);
            if (simd_group == 0) local_sums[simd_lane] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_lane == 0) local_sums[simd_group] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                acc = simd_sum(local_sums[simd_lane]);
                if (simd_lane == 0) {
                    local_inv_mean[0] = metal::precise::rsqrt(acc / float(AXIS) + eps);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float inv_mean = local_inv_mean[0];
            for (uint start = 0; start < AXIS; start += lsize * N_READS) {
                uint first = start + lid * N_READS;
                _Pragma("unroll")
                for (uint i = 0; i < N_READS; ++i) {
                    uint index = first + i;
                    if (index < AXIS) {
                        T value = base[row_offset + index] + delta[row_offset + index];
                        hidden[row_offset + index] = value;
                        normed[row_offset + index] = weight[index] * T(float(value) * inv_mean);
                    }
                }
            }
        """)

func qwen38FusedAddRMSNorm(
    base: MLXArray,
    delta: MLXArray,
    weight: MLXArray,
    epsilon: Float
) -> (MLXArray, MLXArray) {
    let leading = Array(base.shape.dropLast())
    let rows = leading.reduce(1, *)
    let outputs = qwen38AddRMSNormKernel(
        [
            base.reshaped([rows, 5_120]),
            delta.reshaped([rows, 5_120]),
            weight,
            MLXArray(epsilon),
        ],
        template: [("T", base.dtype)],
        grid: (rows * 1_024, 1, 1),
        threadGroup: (1_024, 1, 1),
        outputShapes: [[rows, 5_120], [rows, 5_120]],
        outputDTypes: [base.dtype, base.dtype])
    let shape = leading + [5_120]
    return (outputs[0].reshaped(shape), outputs[1].reshaped(shape))
}

// The DFlash2 runner installs this route only after validating the Qwen3.8
// recurrent geometry (Hk=16, Hv=48, Dk=Dv=128), bf16 activations, and fp32
// state. The measured path therefore dispatches the fixed recurrence directly.
let qwen38InnovationTapeThreadgroupY = 8

private let qwen38GatedDeltaFromConvTapeKernel = MLXFast.metalKernel(
    name: "qwen38_gated_delta_from_conv_tape_bf16_hk16_hv48_d128_v1",
    inputNames: ["conv_out", "g", "beta", "state_in", "T"],
    outputNames: ["y", "state_out", "innovation_tape"],
    source: """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto local_dv_idx = thread_position_in_threadgroup.y;
            auto dv_idx = thread_position_in_grid.y;
            float inv_scale = 1.0f / metal::sqrt(float(Dk));
            float q_scale = inv_scale * inv_scale;
            float k_scale = static_cast<float>(static_cast<InT>(inv_scale));
            threadgroup float q_shared[Dk];
            threadgroup float k_shared[Dk];

            const device StateT* state_ptr =
                state_in + (n * Dv + dv_idx) * Dk;
            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = static_cast<float>(state_ptr[s_idx]);
            }

            for (int t = 0; t < T; ++t) {
                auto conv_t = conv_out + (b_idx * T + t) * ConvDim;
                auto q_t = conv_t + hk_idx * Dk;
                auto k_t = conv_t + KeyDim + hk_idx * Dk;
                auto v_t = conv_t + 2 * KeyDim + hv_idx * Dv;
                auto g_t = g + (b_idx * T + t) * Hv;
                auto beta_t = beta + (b_idx * T + t) * Hv;

                if (local_dv_idx == 0) {
                    float q_sum = 0.0f;
                    float k_sum = 0.0f;
                    float q_raw[n_per_t];
                    float k_raw[n_per_t];
                    for (int i = 0; i < n_per_t; ++i) {
                        auto s_idx = n_per_t * dk_idx + i;
                        q_raw[i] = static_cast<float>(q_t[s_idx]);
                        k_raw[i] = static_cast<float>(k_t[s_idx]);
                        q_sum += q_raw[i] * q_raw[i];
                        k_sum += k_raw[i] * k_raw[i];
                    }
                    q_sum = simd_sum(q_sum);
                    k_sum = simd_sum(k_sum);
                    float q_inv = metal::precise::rsqrt(
                        q_sum / float(Dk) + 1.0e-6f);
                    float k_inv = metal::precise::rsqrt(
                        k_sum / float(Dk) + 1.0e-6f);

                    for (int i = 0; i < n_per_t; ++i) {
                        auto s_idx = n_per_t * dk_idx + i;
                        auto q_norm = static_cast<InT>(q_raw[i] * q_inv);
                        auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
                        q_shared[s_idx] = static_cast<float>(static_cast<InT>(
                            static_cast<float>(q_norm) * q_scale));
                        k_shared[s_idx] = static_cast<float>(static_cast<InT>(
                            static_cast<float>(k_norm) * k_scale));
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    auto k_val = k_shared[s_idx];
                    state[i] = state[i] * g_t[hv_idx];
                    kv_mem += state[i] * k_val;
                }
                kv_mem = simd_sum(kv_mem);
                auto delta =
                    (static_cast<float>(v_t[dv_idx]) - kv_mem) * beta_t[hv_idx];

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    auto k_val = k_shared[s_idx];
                    auto q_val = q_shared[s_idx];
                    state[i] = state[i] + k_val * delta;
                    out += state[i] * q_val;
                }
                out = simd_sum(out);
                auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
                if (thread_index_in_simdgroup == 0) {
                    y_t[dv_idx] = static_cast<InT>(out);
                    innovation_tape[
                        ((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx
                    ] = delta;
                }

                for (int i = 0; i < n_per_t; ++i) {
                    state[i] = static_cast<float>(static_cast<StateT>(state[i]));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            auto state_t = state_out + (n * Dv + dv_idx) * Dk;
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state_t[s_idx] = static_cast<StateT>(state[i]);
            }
        """)

func qwen38GatedDeltaFromConvWithInnovationTape(
    convOutput: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray
) -> (output: MLXArray, state: MLXArray, tape: MLXArray) {
    let batch = convOutput.dim(0)
    let steps = convOutput.dim(1)
    let outputs = qwen38GatedDeltaFromConvTapeKernel(
        [convOutput, g, beta, state, MLXArray(steps)],
        template: [
            ("InT", DType.bfloat16),
            ("StateT", DType.float32),
            ("Dk", 128),
            ("Dv", 128),
            ("Hk", 16),
            ("Hv", 48),
            ("KeyDim", 2_048),
            ("ConvDim", 10_240),
        ],
        grid: (32, 128, batch * 48),
        threadGroup: (32, qwen38InnovationTapeThreadgroupY, 1),
        outputShapes: [
            [batch, steps, 48, 128],
            state.shape,
            [batch, steps, 48, 128],
        ],
        outputDTypes: [.bfloat16, .float32, .float32])
    return (outputs[0], outputs[1], outputs[2])
}

private let qwen38InnovationReplayKernel = MLXFast.metalKernel(
    name: "qwen38_innovation_from_conv_replay_bf16_hk16_hv48_d128_v1",
    inputNames: ["tape", "conv_out", "g", "state_in", "T"],
    outputNames: ["state_out"],
    source: """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto local_dv_idx = thread_position_in_threadgroup.y;
            auto dv_idx = thread_position_in_grid.y;
            float inv_scale = 1.0f / metal::sqrt(float(Dk));
            float k_scale = static_cast<float>(static_cast<InT>(inv_scale));
            threadgroup float k_shared[Dk];

            const device StateT* state_ptr =
                state_in + (n * Dv + dv_idx) * Dk;
            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = static_cast<float>(state_ptr[s_idx]);
            }

            for (int t = 0; t < Steps; ++t) {
                auto conv_t = conv_out + (b_idx * T + t) * ConvDim;
                auto k_t = conv_t + KeyDim + hk_idx * Dk;
                auto g_t = g + (b_idx * T + t) * Hv;

                if (local_dv_idx == 0) {
                    float k_sum = 0.0f;
                    float k_raw[n_per_t];
                    for (int i = 0; i < n_per_t; ++i) {
                        auto s_idx = n_per_t * dk_idx + i;
                        k_raw[i] = static_cast<float>(k_t[s_idx]);
                        k_sum += k_raw[i] * k_raw[i];
                    }
                    k_sum = simd_sum(k_sum);
                    float k_inv = metal::precise::rsqrt(
                        k_sum / float(Dk) + 1.0e-6f);
                    for (int i = 0; i < n_per_t; ++i) {
                        auto s_idx = n_per_t * dk_idx + i;
                        auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
                        k_shared[s_idx] = static_cast<float>(static_cast<InT>(
                            static_cast<float>(k_norm) * k_scale));
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                auto delta = tape[
                    ((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx];
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] * g_t[hv_idx];
                    state[i] = state[i] + k_shared[s_idx] * delta;
                    state[i] = static_cast<float>(static_cast<StateT>(state[i]));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            auto state_t = state_out + (n * Dv + dv_idx) * Dk;
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state_t[s_idx] = static_cast<StateT>(state[i]);
            }
        """)

func qwen38ReplayInnovationTape(
    tape: MLXArray,
    convOutput: MLXArray,
    g: MLXArray,
    state: MLXArray,
    steps: Int
) -> MLXArray {
    let batch = convOutput.dim(0)
    let verifyRows = convOutput.dim(1)
    return qwen38InnovationReplayKernel(
        [tape, convOutput, g, state, MLXArray(verifyRows)],
        template: [
            ("InT", DType.bfloat16),
            ("StateT", DType.float32),
            ("Dk", 128),
            ("Dv", 128),
            ("Hk", 16),
            ("Hv", 48),
            ("KeyDim", 2_048),
            ("ConvDim", 10_240),
            ("Steps", steps),
        ],
        grid: (32, 128, batch * 48),
        threadGroup: (32, qwen38InnovationTapeThreadgroupY, 1),
        outputShapes: [state.shape],
        outputDTypes: [.float32])[0]
}

private func qwen38RepeatedTailCausalMask(
    queryLength: Int, keyLength: Int
) -> MLXArray {
    let queryPositions = arange(
        keyLength - queryLength, keyLength
    ).expandedDimensions(axis: -1)
    let keyPositions = arange(keyLength).expandedDimensions(axis: 0)
    let allowed = keyPositions .<= queryPositions
    let additive = MLX.where(
        allowed,
        MLXArray(0, dtype: .bfloat16),
        MLXArray(-Float.greatestFiniteMagnitude, dtype: .bfloat16))
    return tiled(additive, repetitions: [6, 1])
}

enum Qwen38DFlashGQARoute {
    static func usesPerHead(queryLength: Int, cachedLength: Int) -> Bool {
        (16_384 ..< 32_768).contains(cachedLength)
            && (6 ... 8).contains(queryLength)
    }
}

/// Retained 16K DFlash grouped-GQA arithmetic. Packing each KV head's six
/// query heads into the sequence dimension preserves the source layout while
/// avoiding the pinned MLX revision's invalid one-query-head long-KV result.
func qwen38DFlashGroupedGQA(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float
) -> MLXArray {
    let batch = queries.dim(0)
    let queryLength = queries.dim(2)
    let keyLength = keys.dim(2)
    let groupedQueries = queries.reshaped(batch, 4, 6 * queryLength, 256)
    let mask = qwen38RepeatedTailCausalMask(
        queryLength: queryLength, keyLength: keyLength)
    return MLXFast.scaledDotProductAttention(
        queries: groupedQueries,
        keys: keys,
        values: values,
        scale: scale,
        mask: .array(mask)
    ).reshaped(batch, 24, queryLength, 256)
}
