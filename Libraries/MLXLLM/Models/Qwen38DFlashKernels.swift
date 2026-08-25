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
