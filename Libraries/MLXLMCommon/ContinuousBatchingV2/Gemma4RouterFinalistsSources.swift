// Copyright © 2026 Eigen Labs.
// Final challenge27c821c4 router finalist source. Native comparison/softmax
// parity remains a runtime gate; no global caches or diagnostics are imported.
enum Gemma4RouterFinalistsSources {
    static let selection = """
            const uint row = threadgroup_position_in_grid.x;
            const uint lane = thread_index_in_simdgroup;
            const uint group = simdgroup_index_in_threadgroup;
            const uint expert = group * 32u + lane;
            // Pack the unchanged BF16 bits and the original expert index.
            // This is a payload, NOT an unsigned floating-point ordinal:
            // comparisons below retain native BF16 LessThan semantics.
            uint item = (uint(bfloat16_to_uint16(scores[row * 128u + expert])) << 7)
                | expert;
            threadgroup uint finalists[32];

            for (uint width = 2u; width <= 32u; width <<= 1) {
                for (uint stride = width >> 1; stride > 0u; stride >>= 1) {
                    const uint other = simd_shuffle_xor(item, ushort(stride));
                    const bool otherBefore = gemma4_finalists_before(other, item);
                    const bool takeMinimum = ((lane & width) == 0u)
                        == ((lane & stride) == 0u);
                    if (takeMinimum ? otherBefore : !otherBefore) item = other;
                }
            }

            if (lane >= 24u) {
                finalists[group * 8u + lane - 24u] = item;
            }
            // All four complete SIMD groups participate in this barrier.
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (group == 0u) {
                item = finalists[lane];
                for (uint width = 2u; width <= 32u; width <<= 1) {
                    for (uint stride = width >> 1; stride > 0u; stride >>= 1) {
                        const uint other = simd_shuffle_xor(item, ushort(stride));
                        const bool otherBefore = gemma4_finalists_before(other, item);
                        const bool takeMinimum = ((lane & width) == 0u)
                            == ((lane & stride) == 0u);
                        if (takeMinimum ? otherBefore : !otherBefore) item = other;
                    }
                }
                if (lane >= 24u) indices[row * 8u + lane - 24u] = item & 127u;
            }
        """
    static let selectionHeader = """
            inline bool gemma4_finalists_before(uint a, uint b) {
                const bfloat16_t av = uint16_to_bfloat16(uint16_t(a >> 7));
                const bfloat16_t bv = uint16_to_bfloat16(uint16_t(b >> 7));
                const bool an = metal::isnan(av);
                const bool bn = metal::isnan(bv);
                bool ab;
                bool ba;
                if (an | bn) {
                    ab = (!an) & bn;
                    ba = (!bn) & an;
                } else {
                    ab = av < bv;
                    ba = bv < av;
                }
                return ab || (!ba && (a & 127u) < (b & 127u));
            }
        """
    static let nativeKeysWeights = """
            constexpr int K = 8;
            constexpr int N_READS = 4;
            const uint row = threadgroup_position_in_grid.x;
            const uint lane = thread_index_in_simdgroup;
            uint a = gemma4_finalists_pack(scores[row * 128u + lane], lane);
            uint b = gemma4_finalists_pack(scores[row * 128u + 32u + lane], 32u + lane);
            uint c = gemma4_finalists_pack(scores[row * 128u + 64u + lane], 64u + lane);
            uint d = gemma4_finalists_pack(scores[row * 128u + 96u + lane], 96u + lane);
            uint item = 0u;
            // Each key is unique and positive; zero removes exactly one winner.
            // Rank zero belongs in lane 31, preserving ascending output order.
            #pragma clang loop unroll(full)
            for (uint rank = 0u; rank < 8u; ++rank) {
                const uint winner = simd_max(max(max(a, b), max(c, d)));
                if (lane == 31u - rank) item = winner;
                if (a == winner) a = 0u;
                if (b == winner) b = 0u;
                if (c == winner) c = 0u;
                if (d == winner) d = 0u;
            }
            if (lane >= 24u) indices[row * 8u + lane - 24u] = item & 127u;
            uint chosen[N_READS];
            float ld[N_READS];
            // Every lane participates in each shuffle. Lanes 0 and 1 receive the
            // same four original values as the incumbent axis-eight softmax.
            for (int i = 0; i < N_READS; ++i) {
                chosen[i] = simd_shuffle(item,
                    ushort(24u + (lane & 1u) * 4u + uint(i))) & 127u;
                ld[i] = lane < 2u ? float(scores[row * 128u + chosen[i]]) : Limits<float>::min;
            }
            float maxval = Limits<float>::finite_min;
            for (int i = 0; i < N_READS; ++i) {
                maxval = (maxval < ld[i]) ? ld[i] : maxval;
            }
            maxval = simd_max(maxval);
            // Retain both stock reduction stages and their exact operands.
            maxval = simd_max(lane == 0u ? maxval : Limits<float>::min);
            float normalizer = 0;
            for (int i = 0; i < N_READS; ++i) {
                float exp_x = fast::exp(ld[i] - maxval);
                ld[i] = exp_x;
                normalizer += exp_x;
            }
            normalizer = simd_sum(normalizer);
            normalizer = simd_sum(lane == 0u ? normalizer : 0.0f);
            normalizer = 1 / normalizer;
            if (lane < 2u) {
                for (int i = 0; i < N_READS; ++i) {
                    const T w = T(ld[i] * normalizer);
                    weights[row * 8u + lane * 4u + uint(i)] = w * pes[chosen[i]];
                }
            }
        """
    static let bitonicWeights = """
            constexpr int SIMD_SIZE = 32;
            constexpr int N_READS = 4;
            constexpr int K = 8;

            const uint row = threadgroup_position_in_grid.x;
            const uint lane = thread_index_in_simdgroup;
            const uint group = simdgroup_index_in_threadgroup;
            const uint expert = group * 32u + lane;
            // Pack the unchanged BF16 bits and the original expert index.
            // This is a payload, NOT an unsigned floating-point ordinal:
            // comparisons below retain native BF16 LessThan semantics.
            uint item = (uint(bfloat16_to_uint16(scores[row * 128u + expert])) << 7)
                | expert;
            threadgroup uint finalists[32];
            threadgroup float topv[K];
            threadgroup uint topi[K];
            threadgroup float local_max[SIMD_SIZE];
            threadgroup float local_normalizer[SIMD_SIZE];

            for (uint width = 2u; width <= 32u; width <<= 1) {
                for (uint stride = width >> 1; stride > 0u; stride >>= 1) {
                    const uint other = simd_shuffle_xor(item, ushort(stride));
                    const bool otherBefore = gemma4_finalists_before(other, item);
                    const bool takeMinimum = ((lane & width) == 0u)
                        == ((lane & stride) == 0u);
                    if (takeMinimum ? otherBefore : !otherBefore) item = other;
                }
            }

            if (lane >= 24u) {
                finalists[group * 8u + lane - 24u] = item;
            }
            // All four complete SIMD groups participate in this barrier.
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (group == 0u) {
                item = finalists[lane];
                for (uint width = 2u; width <= 32u; width <<= 1) {
                    for (uint stride = width >> 1; stride > 0u; stride >>= 1) {
                        const uint other = simd_shuffle_xor(item, ushort(stride));
                        const bool otherBefore = gemma4_finalists_before(other, item);
                        const bool takeMinimum = ((lane & width) == 0u)
                            == ((lane & stride) == 0u);
                        if (takeMinimum ? otherBefore : !otherBefore) item = other;
                    }
                }
                if (lane >= 24u) {
                    indices[row * 8u + lane - 24u] = item & 127u;
                    // Stage the winners for the weight tail in the stock
                    // chain's ascending-rank (takeAlong) order: the
                    // unchanged BF16 score bits and the expert id.
                    topv[lane - 24u] = float(uint16_to_bfloat16(uint16_t(item >> 7)));
                    topi[lane - 24u] = item & 127u;
                }
            }
            // All four complete SIMD groups participate in this barrier.
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // softmax_single_row<T, float, N_READS=4> transcription
            // (softmax.h) at axis_size = K on group 0 — the stock axis-8
            // launch's single 32-thread simdgroup — with the stock bf16
            // per-expert-scale multiply fused into the write (the
            // verified decode tail of Gemma4FusedRouterTop8). Groups 1-3
            // only meet the barriers; every shared-slot write is gated to
            // group 0 so slots 1-31 keep their init values exactly as the
            // stock one-simdgroup launch leaves them.
            float ld[N_READS];
            const int base = int(lane) * N_READS;
            if (group == 0u) {
                if (base + N_READS <= K) {
                    for (int i = 0; i < N_READS; i++) {
                        ld[i] = topv[base + i];
                    }
                } else {
                    for (int i = 0; i < N_READS; i++) {
                        ld[i] = ((base + i) < K) ? topv[base + i] : Limits<float>::min;
                    }
                }
                local_max[lane] = Limits<float>::min;
                local_normalizer[lane] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (group == 0u) {
                float maxval = Limits<float>::finite_min;
                for (int i = 0; i < N_READS; i++) {
                    maxval = (maxval < ld[i]) ? ld[i] : maxval;
                }
                maxval = simd_max(maxval);
                if (lane == 0u) {
                    local_max[0] = maxval;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (group == 0u) {
                float maxval = simd_max(local_max[lane]);
                if (lane == 0u) {
                    local_max[0] = maxval;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (group == 0u) {
                const float maxval = local_max[0];
                float normalizer = 0;
                for (int i = 0; i < N_READS; i++) {
                    float exp_x = fast::exp(ld[i] - maxval);
                    ld[i] = exp_x;
                    normalizer += exp_x;
                }
                normalizer = simd_sum(normalizer);
                if (lane == 0u) {
                    local_normalizer[0] = normalizer;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (group == 0u) {
                float normalizer = simd_sum(local_normalizer[lane]);
                if (lane == 0u) {
                    local_normalizer[0] = normalizer;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (group == 0u) {
                const float normalizer = 1 / local_normalizer[0];
                if (base + N_READS <= K) {
                    for (int i = 0; i < N_READS; i++) {
                        const T w = T(ld[i] * normalizer);
                        weights[row * 8u + uint(base + i)] = w * pes[topi[base + i]];
                    }
                } else {
                    for (int i = 0; i < N_READS; i++) {
                        if ((base + i) < K) {
                            const T w = T(ld[i] * normalizer);
                            weights[row * 8u + uint(base + i)]
                                = w * pes[topi[base + i]];
                        }
                    }
                }
            }
        """
    static func weightsHeader(orderKeysEnabled: Bool) -> String {
        """
            constant constexpr bool gemma4_route_order_keys = \(orderKeysEnabled ? "true" : "false");

            inline uint gemma4_finalists_pack(bfloat16_t value, uint expert) {
                const uint bits = uint(bfloat16_to_uint16(value));
                if (!gemma4_route_order_keys) return (bits << 7) | expert;
                // Use the native equality operation here: under Metal's flush-
                // to-zero mode, BF16 subnormals must tie with signed zero too.
                const uint ordinal = metal::isnan(value) ? 0xffffu
                    : (value == bfloat16_t(0.0f) ? 0x8000u
                       : ((bits & 0x8000u) ? ((~bits) & 0xffffu) : (bits ^ 0x8000u)));
                return (ordinal << 7) | expert;
            }

            inline bfloat16_t gemma4_finalists_value(
                uint item, const device bfloat16_t* row_scores)
            {
                return gemma4_route_order_keys ? row_scores[item & 127u]
                    : uint16_to_bfloat16(uint16_t(item >> 7));
            }

            inline bool gemma4_finalists_before(uint a, uint b) {
                if (gemma4_route_order_keys) return a < b;
                const bfloat16_t av = uint16_to_bfloat16(uint16_t(a >> 7));
                const bfloat16_t bv = uint16_to_bfloat16(uint16_t(b >> 7));
                const bool an = metal::isnan(av);
                const bool bn = metal::isnan(bv);
                bool ab;
                bool ba;
                if (an | bn) {
                    ab = (!an) & bn;
                    ba = (!bn) & an;
                } else {
                    ab = av < bv;
                    ba = bv < av;
                }
                return ab || (!ba && (a & 127u) < (b & 127u));
            }
        """
    }
}
