// Copyright © 2026 Eigen Labs.
// Final challenge27c821c4 QKV norm/RoPE bodies. Decode scalar twin changes
// accesses only. Prefill repairs keep inactive rows at barriers and make shared
// position writes single-writer; no Q4-KV packing or residency code is imported.
enum Gemma4QKVNormSources {
    static let decodeVector = """
        typedef vec<T, 4> T4;
        constexpr uint reads = 4;
        const uint row = threadgroup_position_in_grid.x;
        const uint lid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint simd_group = simdgroup_index_in_threadgroup;

        const bool is_query = row < Q_ROWS;
        const bool is_key = row >= Q_ROWS && row < Q_ROWS + K_ROWS;
        const bool weighted = is_query || is_key;
        const device T* input = q;
        const device T* weight = q_weight;
        device T* output_row = q_out;
        uint local_row = row;
        if (!KEY_VALUE_SHARED && row >= Q_ROWS + K_ROWS) {
            input = v;
            output_row = v_out;
            local_row = row - Q_ROWS - K_ROWS;
        } else if (is_key) {
            input = k;
            weight = k_weight;
            output_row = k_out;
            local_row = row - Q_ROWS;
        }

        input += local_row * D + lid * reads;
        output_row += local_row * D;
        device T* output = output_row + lid * reads;
        weight += lid * reads;
        // Keep the pointer inside the V allocation for Q rows even though
        // those rows never dereference it. K rows advance to their matching
        // V row only in the compile-time shared-input variant.
        device T* shared_value_output = v_out;
        if (KEY_VALUE_SHARED && is_key) {
            shared_value_output += local_row * D + lid * reads;
        }

        const T4 vin = *reinterpret_cast<const device T4*>(input);
        float sum = 0.0f;
        for (uint i = 0; i < reads; ++i) {
            const float value = float(vin[i]);
            sum += value * value;
        }
        sum = simd_sum(sum);

        threadgroup float partials[32];
        threadgroup float inverse_rms;
        threadgroup T rounded[D];
        if (lane == 0) partials[simd_group] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            sum = simd_sum(lane < (D / 128) ? partials[lane] : 0.0f);
            if (lane == 0) {
                inverse_rms = metal::precise::rsqrt(sum / float(D) + 1.0e-6f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (weighted) {
            const T4 wv = *reinterpret_cast<const device T4*>(weight);
            if (APPLY_ROPE) {
                for (uint i = 0; i < reads; ++i) {
                    const uint element = lid * reads + i;
                    const T normalized = T(float(vin[i]) * inverse_rms);
                    // Reproduce the separate norm kernel's BF16 output-store
                    // boundary before any RoPE arithmetic reads the value.
                    rounded[element] = T(wv[i] * normalized);
                }
            } else {
                T4 outv;
                for (uint i = 0; i < reads; ++i) {
                    const T normalized = T(float(vin[i]) * inverse_rms);
                    outv[i] = wv[i] * normalized;
                }
                *reinterpret_cast<device T4*>(output) = outv;
            }
            // Gemma's full-attention K-eq-V layers feed the same raw key
            // projection to K RMSNorm and V RMSNormNoScale. The reduction
            // above is therefore identical for both outputs; keep each
            // output's established final expression, but write V while the
            // exact normalizer and input value are live.
            if (KEY_VALUE_SHARED && is_key) {
                T4 sharedv;
                for (uint i = 0; i < reads; ++i) {
                    const T normalized = T(float(vin[i]) * inverse_rms);
                    sharedv[i] = T(1) * normalized;
                }
                *reinterpret_cast<device T4*>(shared_value_output) = sharedv;
            }
        } else {
            T4 outv;
            for (uint i = 0; i < reads; ++i) {
                const T normalized = T(float(vin[i]) * inverse_rms);
                outv[i] = T(1) * normalized;
            }
            *reinterpret_cast<device T4*>(output) = outv;
        }
        if (APPLY_ROPE) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (APPLY_ROPE && weighted && lid * reads < D / 2) {
            const uint heads = is_query ? Q_HEADS : K_HEADS;
            const uint batch = local_row / heads;
            const float L = static_cast<float>(position_offsets[batch]);
            for (uint i = 0; i < reads; ++i) {
                const uint pair = lid * reads + i;
                const float d = static_cast<float>(pair) / static_cast<float>(D / 2);
                const float inv_freq = USE_FREQS
                    ? 1.0f / rope_freqs[pair]
                    : metal::exp2(-d * rope_log2_base[0]);
                const float theta = L * inv_freq;
                const float costheta = metal::fast::cos(theta);
                const float sintheta = metal::fast::sin(theta);
                const float x1 = static_cast<float>(rounded[pair]);
                const float x2 = static_cast<float>(rounded[pair + D / 2]);
                const float rx1 = x1 * costheta - x2 * sintheta;
                const float rx2 = x1 * sintheta + x2 * costheta;
                output_row[pair] = static_cast<T>(rx1);
                output_row[pair + D / 2] = static_cast<T>(rx2);
            }
        }
    """

    static let fullPrefill = """
        constexpr uint reads = 4;
        constexpr uint row_threads = D / reads;
        const uint tid = thread_position_in_threadgroup.x;
        const uint slot = tid / row_threads;
        const uint lid = tid - slot * row_threads;
        const uint row = threadgroup_position_in_grid.x * RPT + slot;
        const uint lane = thread_index_in_simdgroup;
        const uint row_simd = lid / 32;

        threadgroup float partials[RPT][32];
        threadgroup float inv_rms[RPT];
        threadgroup T rounded[RPT][D];
        threadgroup uint row_position[RPT];

        const device T* input = q;
        const device T* weight = q_weight;
        device T* output = q_out;
        // Held inside the V allocation on Q rows, which never dereference it.
        device T* value_output = v_out;
        bool is_key = false;

        if (row < TOTAL_ROWS) {
            if (row < Q_ROWS) {
                const uint b = row / (LQ * HQ);
                const uint rem = row - b * (LQ * HQ);
                const uint l = rem / HQ;
                const uint h = rem - l * HQ;
                if (lid == 0) row_position[slot] = l;
                input = q + (size_t)row * D;
                output = q_out + (((size_t)b * HQ + h) * LQ + l) * D;
            } else {
                is_key = true;
                const uint krow = row - Q_ROWS;
                const uint b = krow / (LK * HK);
                const uint rem = krow - b * (LK * HK);
                const uint l = rem / HK;
                const uint h = rem - l * HK;
                if (lid == 0) row_position[slot] = l;
                const size_t off = (((size_t)b * HK + h) * LK + l) * D;
                input = k + (size_t)krow * D;
                weight = k_weight;
                output = k_out + off;
                value_output += off;
            }
        }

        input += lid * reads;
        device T* output_row = output;
        output += lid * reads;
        weight += lid * reads;
        value_output += lid * reads;

        float sum = 0.0f;
        if (row < TOTAL_ROWS) {
            for (uint i = 0; i < reads; ++i) {
                const float value = float(input[i]);
                sum += value * value;
            }
        }
        sum = simd_sum(sum);

        if (lane == 0) partials[slot][row_simd] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row_simd == 0) {
            sum = simd_sum(lane < (D / 128) ? partials[slot][lane] : 0.0f);
            if (lane == 0) {
                inv_rms[slot] = metal::precise::rsqrt(sum / float(D) + 1.0e-6f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float inverse_rms = inv_rms[slot];
        if (row < TOTAL_ROWS) {
        for (uint i = 0; i < reads; ++i) {
            const T normalized = T(float(input[i]) * inverse_rms);
            if (APPLY_ROPE) {
                // Stage the weighted norm AS T first — the BF16 memory
                // boundary the separate norm kernel's output store performed
                // before stock RoPE read it.
                rounded[slot][lid * reads + i] = T(weight[i] * normalized);
            } else {
                output[i] = weight[i] * normalized;
            }
            // K rows also carry V: same raw input, same normalizer, and
            // `RMSNormNoScale`'s own final expression.
            if (is_key) {
                value_output[i] = T(1) * normalized;
            }
        }
        }
        if (APPLY_ROPE) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (row < TOTAL_ROWS && APPLY_ROPE && lid * reads < D / 2) {
            const uint b = row < Q_ROWS
                ? row / (LQ * HQ)
                : (row - Q_ROWS) / (LK * HK);
            const float L =
                static_cast<float>(row_position[slot] + position_offsets[b]);
            for (uint i = 0; i < reads; ++i) {
                const uint pair = lid * reads + i;
                const float inv_freq = 1.0f / rope_freqs[pair];
                const float theta = L * inv_freq;
                const float costheta = metal::fast::cos(theta);
                const float sintheta = metal::fast::sin(theta);
                const float x1 = static_cast<float>(rounded[slot][pair]);
                const float x2 = static_cast<float>(rounded[slot][pair + D / 2]);
                const float rx1 = x1 * costheta - x2 * sintheta;
                const float rx2 = x1 * sintheta + x2 * costheta;
                output_row[pair] = static_cast<T>(rx1);
                output_row[pair + D / 2] = static_cast<T>(rx2);
            }
        }
    """

    static let slidingPrefill = """
        constexpr uint reads = 4;
        constexpr uint row_threads = D / reads;
        const uint tid = thread_position_in_threadgroup.x;
        const uint slot = tid / row_threads;
        const uint lid = tid - slot * row_threads;
        const uint row = threadgroup_position_in_grid.x * RPT + slot;
        const uint lane = thread_index_in_simdgroup;
        const uint row_simd = lid / 32;

        threadgroup float partials[RPT][32];
        threadgroup float inv_rms[RPT];
        threadgroup T rounded[RPT][D];
        threadgroup uint row_position[RPT];

        // Clean per-bank input row pointers: flat [B, L, H, D] rows.
        const device T* input = q;
        const device T* weight = q_weight;
        device T* output = q_out;
        uint local_row = row;
        bool weighted = true;
        if (row >= Q_ROWS + K_ROWS) {
            input = v;
            output = v_out;
            local_row = row - Q_ROWS - K_ROWS;
            weighted = false;
        } else if (row >= Q_ROWS) {
            input = k;
            weight = k_weight;
            output = k_out;
            local_row = row - Q_ROWS;
        }

        if (row < TOTAL_ROWS) {
            // Flat input rows -> head-major [B, H, L, D] output slots; each
            // bank carries its own head count and length.
            const uint h_count = row < Q_ROWS ? HQ : HK;
            const uint l_count = row < Q_ROWS ? LQ : LK;
            const uint b = local_row / (l_count * h_count);
            const uint rem = local_row - b * (l_count * h_count);
            const uint l = rem / h_count;
            const uint h = rem - l * h_count;
            if (lid == 0) row_position[slot] = l;
            output += (((size_t)b * h_count + h) * l_count + l) * D;
        }

        if (row < TOTAL_ROWS) {
        if (row < Q_ROWS) {
            input = q + (size_t)row * D + lid * reads;
        } else if (row < Q_ROWS + K_ROWS) {
            input = k + (size_t)local_row * D + lid * reads;
        } else {
            input = v + (size_t)local_row * D + lid * reads;
        }
        }
        device T* output_row = output;
        output += lid * reads;
        weight += lid * reads;

        float sum = 0.0f;
        if (row < TOTAL_ROWS) {
            for (uint i = 0; i < reads; ++i) {
                const float value = float(input[i]);
                sum += value * value;
            }
        }
        sum = simd_sum(sum);

        if (lane == 0) partials[slot][row_simd] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row_simd == 0) {
            sum = simd_sum(lane < (D / 128) ? partials[slot][lane] : 0.0f);
            if (lane == 0) {
                inv_rms[slot] = metal::precise::rsqrt(sum / float(D) + 1.0e-6f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float inverse_rms = inv_rms[slot];
        if (row < TOTAL_ROWS) {
        for (uint i = 0; i < reads; ++i) {
            const T normalized = T(float(input[i]) * inverse_rms);
            if (APPLY_ROPE && weighted) {
                // The BF16 memory boundary the separate norm kernel's
                // output store performed before stock RoPE read it.
                rounded[slot][lid * reads + i] = T(weight[i] * normalized);
            } else {
                output[i] = weighted ? weight[i] * normalized : T(1) * normalized;
            }
        }
        }
        if (APPLY_ROPE) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (row < TOTAL_ROWS && APPLY_ROPE && weighted && lid * reads < D / 2) {
            const uint h_count = row < Q_ROWS ? HQ : HK;
            const uint l_count = row < Q_ROWS ? LQ : LK;
            const uint b = local_row / (l_count * h_count);
            const float L =
                static_cast<float>(row_position[slot] + position_offsets[b]);
            for (uint i = 0; i < reads; ++i) {
                const uint pair = lid * reads + i;
                const float d = static_cast<float>(pair) / static_cast<float>(D / 2);
                const float inv_freq = metal::exp2(-d * rope_log2_base[0]);
                const float theta = L * inv_freq;
                const float costheta = metal::fast::cos(theta);
                const float sintheta = metal::fast::sin(theta);
                const float x1 = static_cast<float>(rounded[slot][pair]);
                const float x2 = static_cast<float>(rounded[slot][pair + D / 2]);
                const float rx1 = x1 * costheta - x2 * sintheta;
                const float rx2 = x1 * sintheta + x2 * costheta;
                output_row[pair] = static_cast<T>(rx1);
                output_row[pair + D / 2] = static_cast<T>(rx2);
            }
        }
    """

    static let decodeScalar = """
        typedef T T4[4];
        constexpr uint reads = 4;
        const uint row = threadgroup_position_in_grid.x;
        const uint lid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint simd_group = simdgroup_index_in_threadgroup;

        const bool is_query = row < Q_ROWS;
        const bool is_key = row >= Q_ROWS && row < Q_ROWS + K_ROWS;
        const bool weighted = is_query || is_key;
        const device T* input = q;
        const device T* weight = q_weight;
        device T* output_row = q_out;
        uint local_row = row;
        if (!KEY_VALUE_SHARED && row >= Q_ROWS + K_ROWS) {
            input = v;
            output_row = v_out;
            local_row = row - Q_ROWS - K_ROWS;
        } else if (is_key) {
            input = k;
            weight = k_weight;
            output_row = k_out;
            local_row = row - Q_ROWS;
        }

        input += local_row * D + lid * reads;
        output_row += local_row * D;
        device T* output = output_row + lid * reads;
        weight += lid * reads;
        // Keep the pointer inside the V allocation for Q rows even though
        // those rows never dereference it. K rows advance to their matching
        // V row only in the compile-time shared-input variant.
        device T* shared_value_output = v_out;
        if (KEY_VALUE_SHARED && is_key) {
            shared_value_output += local_row * D + lid * reads;
        }

        T4 vin;
        for (uint i = 0; i < reads; ++i) vin[i] = input[i];
        float sum = 0.0f;
        for (uint i = 0; i < reads; ++i) {
            const float value = float(vin[i]);
            sum += value * value;
        }
        sum = simd_sum(sum);

        threadgroup float partials[32];
        threadgroup float inverse_rms;
        threadgroup T rounded[D];
        if (lane == 0) partials[simd_group] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            sum = simd_sum(lane < (D / 128) ? partials[lane] : 0.0f);
            if (lane == 0) {
                inverse_rms = metal::precise::rsqrt(sum / float(D) + 1.0e-6f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (weighted) {
            T4 wv;
            for (uint i = 0; i < reads; ++i) wv[i] = weight[i];
            if (APPLY_ROPE) {
                for (uint i = 0; i < reads; ++i) {
                    const uint element = lid * reads + i;
                    const T normalized = T(float(vin[i]) * inverse_rms);
                    // Reproduce the separate norm kernel's BF16 output-store
                    // boundary before any RoPE arithmetic reads the value.
                    rounded[element] = T(wv[i] * normalized);
                }
            } else {
                T4 outv;
                for (uint i = 0; i < reads; ++i) {
                    const T normalized = T(float(vin[i]) * inverse_rms);
                    outv[i] = wv[i] * normalized;
                }
                for (uint i = 0; i < reads; ++i) output[i] = outv[i];
            }
            // Gemma's full-attention K-eq-V layers feed the same raw key
            // projection to K RMSNorm and V RMSNormNoScale. The reduction
            // above is therefore identical for both outputs; keep each
            // output's established final expression, but write V while the
            // exact normalizer and input value are live.
            if (KEY_VALUE_SHARED && is_key) {
                T4 sharedv;
                for (uint i = 0; i < reads; ++i) {
                    const T normalized = T(float(vin[i]) * inverse_rms);
                    sharedv[i] = T(1) * normalized;
                }
                for (uint i = 0; i < reads; ++i) shared_value_output[i] = sharedv[i];
            }
        } else {
            T4 outv;
            for (uint i = 0; i < reads; ++i) {
                const T normalized = T(float(vin[i]) * inverse_rms);
                outv[i] = T(1) * normalized;
            }
            for (uint i = 0; i < reads; ++i) output[i] = outv[i];
        }
        if (APPLY_ROPE) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (APPLY_ROPE && weighted && lid * reads < D / 2) {
            const uint heads = is_query ? Q_HEADS : K_HEADS;
            const uint batch = local_row / heads;
            const float L = static_cast<float>(position_offsets[batch]);
            for (uint i = 0; i < reads; ++i) {
                const uint pair = lid * reads + i;
                const float d = static_cast<float>(pair) / static_cast<float>(D / 2);
                const float inv_freq = USE_FREQS
                    ? 1.0f / rope_freqs[pair]
                    : metal::exp2(-d * rope_log2_base[0]);
                const float theta = L * inv_freq;
                const float costheta = metal::fast::cos(theta);
                const float sintheta = metal::fast::sin(theta);
                const float x1 = static_cast<float>(rounded[pair]);
                const float x2 = static_cast<float>(rounded[pair + D / 2]);
                const float rx1 = x1 * costheta - x2 * sintheta;
                const float rx2 = x1 * sintheta + x2 * costheta;
                output_row[pair] = static_cast<T>(rx1);
                output_row[pair + D / 2] = static_cast<T>(rx2);
            }
        }
    """

}
