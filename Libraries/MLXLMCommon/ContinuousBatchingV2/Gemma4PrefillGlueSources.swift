// Copyright © 2026 Eigen Labs.
// MSL adapted from Layr-Labs/mlxfast-gemma4-26b-a4b-engine
// 27c821c466c9799e87162e9436618863b7d0a0ba, Gemma4PrefillGlueV1.swift.
// Source strings only: this file creates no device, array, or kernel.
// Native qualification is tied to the specific source/dependency tuple.
enum Gemma4PrefillGlueSources {

    static let preNorm = """
            threadgroup float local_sums[32];
            threadgroup float local_inv[1];

            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;

            const size_t base = size_t(row) * GLUE_AXIS + lid * GLUE_NREADS;

            float xv[GLUE_NREADS];
            GLUE_LOADF(xv, x, base);

            const float inv = glue_inv_rms(
                xv, local_sums, local_inv, simd_lane_id, simd_group_id, GLUE_EPS);

            T outv[GLUE_NREADS];
            #pragma clang loop unroll(full)
            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint j = lid * GLUE_NREADS + i;
                const T scaled = static_cast<T>(xv[i] * inv);
                outv[i] = w[j] * scaled;
            }
            GLUE_STORET(out, base, outv);
            """

    // The final vector body is unchanged. The scalar twin replaces only its
    // unconditional vec4 accesses; arithmetic and reduction order are retained.
    static let expertTailChainedVector = """
            threadgroup float local_sums_a[32];
            threadgroup float local_sums_b[32];
            threadgroup float local_inv2[2];

            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            const size_t base = size_t(row) * GLUE_AXIS + lid * GLUE_NREADS;
            const uint assignment_base = row * 8;

            // The route metadata is invariant across this thread's four
            // features. Keep one copy per thread instead of reloading both
            // arrays in every feature/slot iteration.
            uint inv_orders[8];
            float route_weight_values[8];
            #pragma clang loop unroll(full)
            for (uint slot = 0; slot < 8; ++slot) {
                const uint assignment = assignment_base + slot;
                inv_orders[slot] = uint(inverse_order[assignment]);
                route_weight_values[slot] = float(route_weights[assignment]);
            }

            float av[GLUE_NREADS];
            float bv[GLUE_NREADS];
            const vec<T, 4> h1_values =
                *((const device vec<T, 4>*)(h1 + base));
            #pragma clang loop unroll(full)
            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint feature = lid * GLUE_NREADS + i;
                av[i] = static_cast<float>(h1_values[i]);
                T accumulator = (T)0;
                #pragma clang loop unroll(full)
                for (uint slot = 0; slot < 8; ++slot) {
                    const uint sorted_row = inv_orders[slot];
                    const T weighted = (T)(
                        (float)sorted[size_t(sorted_row) * GLUE_AXIS + feature]
                        * route_weight_values[slot]);
                    accumulator = accumulator + weighted;
                }
                bv[i] = static_cast<float>(accumulator);
            }

            float inv_a = 0;
            float inv_b = 0;
            glue_inv_rms2(
                av, bv, local_sums_a, local_sums_b, local_inv2,
                simd_lane_id, simd_group_id, GLUE_EPS, inv_a, inv_b);

            float tv[GLUE_NREADS];
            #pragma clang loop unroll(full)
            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint j = lid * GLUE_NREADS + i;
                const T n1 = static_cast<T>(
                    w1[j] * static_cast<T>(av[i] * inv_a));
                const T n2 = static_cast<T>(
                    w2[j] * static_cast<T>(bv[i] * inv_b));
                tv[i] = static_cast<float>(static_cast<T>(n1 + n2));
            }

            const float inv_t = glue_inv_rms(
                tv, local_sums_a, local_inv2,
                simd_lane_id, simd_group_id, GLUE_EPS);

            const T scalar = s[0];
            const vec<T, 4> residual_values =
                *((const device vec<T, 4>*)(res2 + base));
            float ov[GLUE_NREADS];
            vec<T, 4> out_values;
            #pragma clang loop unroll(full)
            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint j = lid * GLUE_NREADS + i;
                const T normed3 = static_cast<T>(
                    w3[j] * static_cast<T>(tv[i] * inv_t));
                const T summed = static_cast<T>(residual_values[i] + normed3);
                const T scaled = static_cast<T>(summed * scalar);
                out_values[i] = scaled;
                ov[i] = static_cast<float>(scaled);
            }
            *((device vec<T, 4>*)(out + base)) = out_values;

            const float inv_n = glue_inv_rms(
                ov, local_sums_a, local_inv2,
                simd_lane_id, simd_group_id, GLUE_EPS);

            vec<T, 4> normed_values;
            #pragma clang loop unroll(full)
            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint j = lid * GLUE_NREADS + i;
                normed_values[i] =
                    wn[j] * static_cast<T>(ov[i] * inv_n);
            }
            *((device vec<T, 4>*)(normed + base)) = normed_values;
        """

    static let expertTailChainedScalar = """
            threadgroup float local_sums_a[32];
            threadgroup float local_sums_b[32];
            threadgroup float local_inv2[2];

            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            const size_t base = size_t(row) * GLUE_AXIS + lid * GLUE_NREADS;
            const uint assignment_base = row * 8;

            // The route metadata is invariant across this thread's four
            // features. Keep one copy per thread instead of reloading both
            // arrays in every feature/slot iteration.
            uint inv_orders[8];
            float route_weight_values[8];
            #pragma clang loop unroll(full)
            for (uint slot = 0; slot < 8; ++slot) {
                const uint assignment = assignment_base + slot;
                inv_orders[slot] = uint(inverse_order[assignment]);
                route_weight_values[slot] = float(route_weights[assignment]);
            }

            float av[GLUE_NREADS];
            float bv[GLUE_NREADS];
            T h1_values[GLUE_NREADS];
            GLUE_LOADT(h1_values, h1, base);
            #pragma clang loop unroll(full)
            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint feature = lid * GLUE_NREADS + i;
                av[i] = static_cast<float>(h1_values[i]);
                T accumulator = (T)0;
                #pragma clang loop unroll(full)
                for (uint slot = 0; slot < 8; ++slot) {
                    const uint sorted_row = inv_orders[slot];
                    const T weighted = (T)(
                        (float)sorted[size_t(sorted_row) * GLUE_AXIS + feature]
                        * route_weight_values[slot]);
                    accumulator = accumulator + weighted;
                }
                bv[i] = static_cast<float>(accumulator);
            }

            float inv_a = 0;
            float inv_b = 0;
            glue_inv_rms2(
                av, bv, local_sums_a, local_sums_b, local_inv2,
                simd_lane_id, simd_group_id, GLUE_EPS, inv_a, inv_b);

            float tv[GLUE_NREADS];
            #pragma clang loop unroll(full)
            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint j = lid * GLUE_NREADS + i;
                const T n1 = static_cast<T>(
                    w1[j] * static_cast<T>(av[i] * inv_a));
                const T n2 = static_cast<T>(
                    w2[j] * static_cast<T>(bv[i] * inv_b));
                tv[i] = static_cast<float>(static_cast<T>(n1 + n2));
            }

            const float inv_t = glue_inv_rms(
                tv, local_sums_a, local_inv2,
                simd_lane_id, simd_group_id, GLUE_EPS);

            const T scalar = s[0];
            T residual_values[GLUE_NREADS];
            GLUE_LOADT(residual_values, res2, base);
            float ov[GLUE_NREADS];
            T out_values[GLUE_NREADS];
            #pragma clang loop unroll(full)
            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint j = lid * GLUE_NREADS + i;
                const T normed3 = static_cast<T>(
                    w3[j] * static_cast<T>(tv[i] * inv_t));
                const T summed = static_cast<T>(residual_values[i] + normed3);
                const T scaled = static_cast<T>(summed * scalar);
                out_values[i] = scaled;
                ov[i] = static_cast<float>(scaled);
            }
            GLUE_STORET(out, base, out_values);

            const float inv_n = glue_inv_rms(
                ov, local_sums_a, local_inv2,
                simd_lane_id, simd_group_id, GLUE_EPS);

            T normed_values[GLUE_NREADS];
            #pragma clang loop unroll(full)
            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint j = lid * GLUE_NREADS + i;
                normed_values[i] =
                    wn[j] * static_cast<T>(ov[i] * inv_n);
            }
            GLUE_STORET(normed, base, normed_values);
        """


    static let attentionBranchPrefix = """
                threadgroup float local_sums[32];
                threadgroup float local_inv[1];

                const uint row = threadgroup_position_in_grid.y;
                const uint lid = thread_position_in_threadgroup.x;
                const uint simd_lane_id = thread_index_in_simdgroup;
                const uint simd_group_id = simdgroup_index_in_threadgroup;

                const size_t base = size_t(row) * GLUE_AXIS + lid * GLUE_NREADS;

                float xv[GLUE_NREADS];
                GLUE_LOADF(xv, x, base);

                const float inv = glue_inv_rms(
                    xv, local_sums, local_inv, simd_lane_id, simd_group_id, GLUE_EPS);

                // `normResidualKernel`'s row, verbatim; the T values just
                // stored to `out` are kept in registers instead of re-read.
                T resv[GLUE_NREADS];
                GLUE_LOADT(resv, res, base);
                T outv[GLUE_NREADS];
                #pragma clang loop unroll(full)
                for (int i = 0; i < GLUE_NREADS; i++) {
                    const uint j = lid * GLUE_NREADS + i;
                    const T normed = static_cast<T>(w[j] * static_cast<T>(xv[i] * inv));
                    outv[i] = resv[i] + normed;
                }
                GLUE_STORET(out, base, outv);

                float ov[GLUE_NREADS];
                #pragma clang loop unroll(full)
                for (int i = 0; i < GLUE_NREADS; i++) {
                    ov[i] = static_cast<float>(outv[i]);
                }

                // One sum-of-squares over the rounded `out` row serves both
                // the dense weight and the router weight: the two stock
                // kernels reduce the identical array.
                const float inv2 = glue_inv_rms(
                    ov, local_sums, local_inv, simd_lane_id, simd_group_id, GLUE_EPS);

                T densev[GLUE_NREADS];
                T routerv[GLUE_NREADS];
                #pragma clang loop unroll(full)
                for (int i = 0; i < GLUE_NREADS; i++) {
                    const uint j = lid * GLUE_NREADS + i;
                    const T scaled = static_cast<T>(ov[i] * inv2);
                    densev[i] = wd[j] * scaled;
                    routerv[i] = wr[j] * scaled;
                }
                GLUE_STORET(dense, base, densev);
                GLUE_STORET(router, base, routerv);
                """

    static let preNormScatter = """
                threadgroup float local_sums[32];
                threadgroup float local_inv[1];
                threadgroup uint cached_positions[8];

                const uint row = threadgroup_position_in_grid.y;
                const uint lid = thread_position_in_threadgroup.x;
                const uint simd_lane_id = thread_index_in_simdgroup;
                const uint simd_group_id = simdgroup_index_in_threadgroup;
                const size_t assignment_base = size_t(row) * 8;

                // The first eight threads issue the row's complete metadata
                // load while every thread begins its independent RMS work.
                // `glue_inv_rms` reaches a threadgroup-memory barrier before
                // returning, which makes these words visible without another
                // barrier in this kernel.
                if (lid < 8) {
                    cached_positions[lid] = inverse[assignment_base + lid];
                }

                const size_t base = size_t(row) * GLUE_AXIS + lid * GLUE_NREADS;

                float xv[GLUE_NREADS];
                GLUE_LOADF(xv, x, base);

                const float inv = glue_inv_rms(
                    xv, local_sums, local_inv, simd_lane_id, simd_group_id, GLUE_EPS);

                T normed[GLUE_NREADS];
                #pragma clang loop unroll(full)
                for (int i = 0; i < GLUE_NREADS; i++) {
                    const uint j = lid * GLUE_NREADS + i;
                    const T scaled = static_cast<T>(xv[i] * inv);
                    normed[i] = w[j] * scaled;
                }

                #pragma clang loop unroll(full)
                for (uint k = 0; k < 8; ++k) {
                    const size_t pos = size_t(cached_positions[k]);
                    const size_t obase = pos * GLUE_AXIS + lid * GLUE_NREADS;
                    GLUE_STORET(out, obase, normed);
                }
                """

    static func header(vectorized: Bool) -> String {
        let vec4Macros: String = vectorized
        ? """
            #define GLUE_LOADF(dstf, src, base) \
              { const vec<T, 4> gv4_a = *((const device vec<T, 4>*)((src) + (base))); \
                (dstf)[0] = static_cast<float>(gv4_a[0]); \
                (dstf)[1] = static_cast<float>(gv4_a[1]); \
                (dstf)[2] = static_cast<float>(gv4_a[2]); \
                (dstf)[3] = static_cast<float>(gv4_a[3]); }
            #define GLUE_LOADT(dstt, src, base) \
              { const vec<T, 4> gv4_b = *((const device vec<T, 4>*)((src) + (base))); \
                (dstt)[0] = gv4_b[0]; \
                (dstt)[1] = gv4_b[1]; \
                (dstt)[2] = gv4_b[2]; \
                (dstt)[3] = gv4_b[3]; }
            #define GLUE_STORET(dst, base, srct) \
              { vec<T, 4> gv4_c; \
                gv4_c[0] = (srct)[0]; \
                gv4_c[1] = (srct)[1]; \
                gv4_c[2] = (srct)[2]; \
                gv4_c[3] = (srct)[3]; \
                *((device vec<T, 4>*)((dst) + (base))) = gv4_c; }
            """
        : """
            #define GLUE_LOADF(dstf, src, base) \
              { (dstf)[0] = static_cast<float>((src)[(base) + 0]); \
                (dstf)[1] = static_cast<float>((src)[(base) + 1]); \
                (dstf)[2] = static_cast<float>((src)[(base) + 2]); \
                (dstf)[3] = static_cast<float>((src)[(base) + 3]); }
            #define GLUE_LOADT(dstt, src, base) \
              { (dstt)[0] = (src)[(base) + 0]; \
                (dstt)[1] = (src)[(base) + 1]; \
                (dstt)[2] = (src)[(base) + 2]; \
                (dstt)[3] = (src)[(base) + 3]; }
            #define GLUE_STORET(dst, base, srct) \
              { (dst)[(base) + 0] = (srct)[0]; \
                (dst)[(base) + 1] = (srct)[1]; \
                (dst)[(base) + 2] = (srct)[2]; \
                (dst)[(base) + 3] = (srct)[3]; }
            """
        return """
        constant constexpr const int GLUE_AXIS = 2816;
        constant constexpr const int GLUE_NREADS = 4;
        constant constexpr const int GLUE_SIMDGROUPS = GLUE_AXIS / GLUE_NREADS / 32;
        // Pinned; `planeRows` refuses any other eps.
        constant constexpr const float GLUE_EPS = 1e-6f;

        // The accessor macros are written for a four-element lane window.
        static_assert(GLUE_NREADS == 4, "glue accessors assume a vec4 window");
        \(vec4Macros)

        inline float glue_inv_rms(
            thread const float* xv,
            threadgroup float* local_sums,
            threadgroup float* local_inv,
            uint simd_lane_id,
            uint simd_group_id,
            float eps) {
          float acc = 0;
          #pragma clang loop unroll(full)
          for (int i = 0; i < GLUE_NREADS; i++) {
            acc += xv[i] * xv[i];
          }
          acc = simd_sum(acc);
          if (simd_lane_id == 0) {
            local_sums[simd_group_id] = acc;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (simd_group_id == 0) {
            acc = simd_sum(
                simd_lane_id < GLUE_SIMDGROUPS ? local_sums[simd_lane_id] : 0.0f);
            if (simd_lane_id == 0) {
              // Match the pinned native FP32 reciprocal/FMA mean boundary.
              // Keep safe BF16 arithmetic everywhere else in the fused tape.
              local_inv[0] = metal::precise::rsqrt(fma(acc, (1.0f / GLUE_AXIS), eps));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          return local_inv[0];
        }

        /// Both reductions of the tail in one pass, so `h1` and `h2` are each
        /// read from device memory exactly once.
        inline void glue_inv_rms2(
            thread const float* av,
            thread const float* bv,
            threadgroup float* local_sums_a,
            threadgroup float* local_sums_b,
            threadgroup float* local_inv2,
            uint simd_lane_id,
            uint simd_group_id,
            float eps,
            thread float& inv_a,
            thread float& inv_b) {
          float acc_a = 0;
          float acc_b = 0;
          #pragma clang loop unroll(full)
          for (int i = 0; i < GLUE_NREADS; i++) {
            acc_a += av[i] * av[i];
            acc_b += bv[i] * bv[i];
          }
          acc_a = simd_sum(acc_a);
          acc_b = simd_sum(acc_b);
          if (simd_lane_id == 0) {
            local_sums_a[simd_group_id] = acc_a;
            local_sums_b[simd_group_id] = acc_b;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (simd_group_id == 0) {
            acc_a = simd_sum(
                simd_lane_id < GLUE_SIMDGROUPS ? local_sums_a[simd_lane_id] : 0.0f);
            acc_b = simd_sum(
                simd_lane_id < GLUE_SIMDGROUPS ? local_sums_b[simd_lane_id] : 0.0f);
            if (simd_lane_id == 0) {
              local_inv2[0] = metal::precise::rsqrt(fma(acc_a, (1.0f / GLUE_AXIS), eps));
              local_inv2[1] = metal::precise::rsqrt(fma(acc_b, (1.0f / GLUE_AXIS), eps));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          inv_a = local_inv2[0];
          inv_b = local_inv2[1];
        }
        """
    }

    static let normResidual = """
            threadgroup float local_sums[32];
            threadgroup float local_inv[1];

            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;

            const size_t base = size_t(row) * GLUE_AXIS + lid * GLUE_NREADS;

            float xv[GLUE_NREADS];
            GLUE_LOADF(xv, x, base);

            const float inv = glue_inv_rms(
                xv, local_sums, local_inv, simd_lane_id, simd_group_id, GLUE_EPS);

            T resv[GLUE_NREADS];
            GLUE_LOADT(resv, res, base);
            T outv[GLUE_NREADS];
            #pragma clang loop unroll(full)
            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint j = lid * GLUE_NREADS + i;
                // The stock pair stores `w * T(x*inv)` to bf16, then reads it
                // back for the add. Round in the same place.
                const T normed = static_cast<T>(w[j] * static_cast<T>(xv[i] * inv));
                outv[i] = resv[i] + normed;
            }
            GLUE_STORET(out, base, outv);
            """

    static let dualPreNorm = """
            threadgroup float local_sums[32];
            threadgroup float local_inv[1];

            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;

            const size_t base = size_t(row) * GLUE_AXIS + lid * GLUE_NREADS;

            float xv[GLUE_NREADS];
            GLUE_LOADF(xv, x, base);

            // One sum-of-squares serves both weights: the two stock kernels
            // reduce the identical input and differ only in the weight vector.
            const float inv = glue_inv_rms(
                xv, local_sums, local_inv, simd_lane_id, simd_group_id, GLUE_EPS);

            T out1v[GLUE_NREADS];
            T out2v[GLUE_NREADS];
            #pragma clang loop unroll(full)
            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint j = lid * GLUE_NREADS + i;
                const T scaled = static_cast<T>(xv[i] * inv);
                out1v[i] = w1[j] * scaled;
                out2v[i] = w2[j] * scaled;
            }
            GLUE_STORET(out1, base, out1v);
            GLUE_STORET(out2, base, out2v);
            """

    static let branchTail = """
            threadgroup float local_sums_a[32];
            threadgroup float local_sums_b[32];
            threadgroup float local_inv2[2];

            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;

            const size_t base = size_t(row) * GLUE_AXIS + lid * GLUE_NREADS;

            float av[GLUE_NREADS];
            float bv[GLUE_NREADS];
            GLUE_LOADF(av, h1, base);
            GLUE_LOADF(bv, h2, base);

            float inv_a = 0;
            float inv_b = 0;
            glue_inv_rms2(
                av, bv, local_sums_a, local_sums_b, local_inv2,
                simd_lane_id, simd_group_id, GLUE_EPS, inv_a, inv_b);

            // The branch sum stays in registers. The stock graph writes it to
            // bf16 between the norms and the final norm, so round it here.
            float tv[GLUE_NREADS];
            #pragma clang loop unroll(full)
            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint j = lid * GLUE_NREADS + i;
                const T n1 = static_cast<T>(w1[j] * static_cast<T>(av[i] * inv_a));
                const T n2 = static_cast<T>(w2[j] * static_cast<T>(bv[i] * inv_b));
                tv[i] = static_cast<float>(static_cast<T>(n1 + n2));
            }

            const float inv_t = glue_inv_rms(
                tv, local_sums_a, local_inv2, simd_lane_id, simd_group_id, GLUE_EPS);

            T res2v[GLUE_NREADS];
            GLUE_LOADT(res2v, res2, base);
            T outv[GLUE_NREADS];
            #pragma clang loop unroll(full)
            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint j = lid * GLUE_NREADS + i;
                const T normed = static_cast<T>(w3[j] * static_cast<T>(tv[i] * inv_t));
                outv[i] = res2v[i] + normed;
            }
            GLUE_STORET(out, base, outv);
            """

    static let branchTailChained = """
            threadgroup float local_sums_a[32];
            threadgroup float local_sums_b[32];
            threadgroup float local_inv2[2];

            const uint row = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;

            const size_t base = size_t(row) * GLUE_AXIS + lid * GLUE_NREADS;

            float av[GLUE_NREADS];
            float bv[GLUE_NREADS];
            GLUE_LOADF(av, h1, base);
            GLUE_LOADF(bv, h2, base);

            float inv_a = 0;
            float inv_b = 0;
            glue_inv_rms2(
                av, bv, local_sums_a, local_sums_b, local_inv2,
                simd_lane_id, simd_group_id, GLUE_EPS, inv_a, inv_b);

            float tv[GLUE_NREADS];
            #pragma clang loop unroll(full)
            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint j = lid * GLUE_NREADS + i;
                const T n1 = static_cast<T>(w1[j] * static_cast<T>(av[i] * inv_a));
                const T n2 = static_cast<T>(w2[j] * static_cast<T>(bv[i] * inv_b));
                tv[i] = static_cast<float>(static_cast<T>(n1 + n2));
            }

            const float inv_t = glue_inv_rms(
                tv, local_sums_a, local_inv2, simd_lane_id, simd_group_id, GLUE_EPS);

            // The stock graph stores the residual sum to bf16, then the scalar
            // multiply reads it back and stores again. Both roundings are
            // explicit here, so `out` is the same array either way.
            const T scalar = s[0];
            T res2v[GLUE_NREADS];
            GLUE_LOADT(res2v, res2, base);
            float ov[GLUE_NREADS];
            T outv[GLUE_NREADS];
            #pragma clang loop unroll(full)
            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint j = lid * GLUE_NREADS + i;
                const T normed3 = static_cast<T>(w3[j] * static_cast<T>(tv[i] * inv_t));
                const T summed = static_cast<T>(res2v[i] + normed3);
                const T scaled = static_cast<T>(summed * scalar);
                outv[i] = scaled;
                ov[i] = static_cast<float>(scaled);
            }
            GLUE_STORET(out, base, outv);

            // The next layer's input norm, over exactly the bf16 values just
            // stored to `out`.
            const float inv_n = glue_inv_rms(
                ov, local_sums_a, local_inv2, simd_lane_id, simd_group_id, GLUE_EPS);

            T normedv[GLUE_NREADS];
            #pragma clang loop unroll(full)
            for (int i = 0; i < GLUE_NREADS; i++) {
                const uint j = lid * GLUE_NREADS + i;
                normedv[i] = wn[j] * static_cast<T>(ov[i] * inv_n);
            }
            GLUE_STORET(normed, base, normedv);
            """
}
