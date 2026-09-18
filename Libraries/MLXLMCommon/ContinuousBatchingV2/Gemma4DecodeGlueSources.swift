// Copyright © 2026 Eigen Labs.
// David Tai SDK0cb4c68d per-row Gemma decode glue. B8 fixed-stride tables and
// their consumers are deliberately separate. No device construction here.
import Foundation

enum Gemma4DecodeGlueSources {
    static func rmsReduce(
        _ src: String, into slot: String, axis: Int = 2816, simdGroups: Int = 22
    ) -> String {
        // The packaged target RMS kernel closes its FP32 mean with one FMA.
        // Spell only that boundary explicitly; relaxing the whole fused kernel
        // would remove required BF16 boundaries in the residual/tail arithmetic.
        let mean = axis == 2816 ? "fma(acc, (1.0f / 2816.0f), 1e-06f)"
            : "acc / \(axis).0f + 1e-06f"
        return """
            {
                float acc = 0;
                for (int i = 0; i < 4; i++) {
                    float xi = (float)\(src)[base + i];
                    acc += xi * xi;
                }
                acc = simd_sum(acc);
                if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group_id == 0) {
                    acc = simd_sum(
                        simd_lane_id < \(simdGroups) ? local_sums[simd_lane_id] : 0.0f);
                    if (simd_lane_id == 0) {
                        \(slot) = metal::precise::rsqrt(\(mean));
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        """
    }

    static let pairedRmsSource = """
        float av[4];
        float bv[4];
        threadgroup float local_sums_b[32];
        {
            float acc_a = 0;
            float acc_b = 0;
            for (int i = 0; i < 4; i++) {
                av[i] = (float)a[base + i];
                bv[i] = (float)b[base + i];
                acc_a += av[i] * av[i];
                acc_b += bv[i] * bv[i];
            }
            acc_a = simd_sum(acc_a);
            acc_b = simd_sum(acc_b);
            if (simd_lane_id == 0) {
                local_sums[simd_group_id] = acc_a;
                local_sums_b[simd_group_id] = acc_b;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group_id == 0) {
                acc_a = simd_sum(
                    simd_lane_id < 22 ? local_sums[simd_lane_id] : 0.0f);
                acc_b = simd_sum(
                    simd_lane_id < 22 ? local_sums_b[simd_lane_id] : 0.0f);
                if (simd_lane_id == 0) {
                    local_inv[0] = metal::precise::rsqrt(fma(acc_a, (1.0f / 2816.0f), 1e-06f));
                    local_inv[1] = metal::precise::rsqrt(fma(acc_b, (1.0f / 2816.0f), 1e-06f));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        """

    static func pairedRmsTailSource(_ source: String) -> String {
        var result = source
        func replaceOnce(_ old: String, with new: String) {
            precondition(result.components(separatedBy: old).count == 2)
            result = result.replacingOccurrences(of: old, with: new)
        }
        replaceOnce(rmsReduce("a", into: "local_inv[0]"), with: pairedRmsSource)
        replaceOnce(rmsReduce("b", into: "local_inv[1]"), with: "")
        replaceOnce(
            "const T h1 = w1[wbase + i] * static_cast<T>((float)a[base + i] * inv1);",
            with: "const T h1 = w1[wbase + i] * static_cast<T>(av[i] * inv1);")
        replaceOnce(
            "const T h2 = w2[wbase + i] * static_cast<T>((float)b[base + i] * inv2);",
            with: "const T h2 = w2[wbase + i] * static_cast<T>(bv[i] * inv2);")
        return result
    }

    static let normResidual = """
            const uint row = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            threadgroup float local_inv[1];
            threadgroup float local_sums[32];
            const uint base = row * 2816 + lid * 4;
            const uint wbase = lid * 4;
        \(rmsReduce("x", into: "local_inv[0]"))
            const float inv = local_inv[0];
            for (int i = 0; i < 4; i++) {
                // The stock chain rounds the norm's output to T in memory
                // before the residual add reads it; reproduce both roundings.
                const T normed = static_cast<T>(
                    w[wbase + i] * static_cast<T>((float)x[base + i] * inv));
                out[base + i] = res[base + i] + normed;
            }
        """

    static let normResidual1024 = """
            const uint row = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            threadgroup float local_inv[1];
            threadgroup float local_sums[32];
            const uint base = row * 1024 + lid * 4;
            const uint wbase = lid * 4;
        \(rmsReduce("x", into: "local_inv[0]", axis: 1024, simdGroups: 8))
            const float inv = local_inv[0];
            for (int i = 0; i < 4; i++) {
                // The stock chain rounds the norm's output to T in memory
                // before the residual add reads it; reproduce both roundings.
                const T normed = static_cast<T>(
                    w[wbase + i] * static_cast<T>((float)x[base + i] * inv));
                out[base + i] = res[base + i] + normed;
            }
        """

    static let dualPreNorm = """
                const uint row = threadgroup_position_in_grid.x;
                const uint lid = thread_position_in_threadgroup.x;
                const uint simd_lane_id = thread_index_in_simdgroup;
                const uint simd_group_id = simdgroup_index_in_threadgroup;
                threadgroup float local_inv[1];
                threadgroup float local_sums[32];
                const uint base = row * 2816 + lid * 4;
                const uint wbase = lid * 4;
            \(rmsReduce("x", into: "local_inv[0]"))
                const float inv = local_inv[0];
                for (int i = 0; i < 4; i++) {
                    const T nx = static_cast<T>((float)x[base + i] * inv);
                    out1[base + i] = w1[wbase + i] * nx;
                    out2[base + i] = w2[wbase + i] * nx;
                }
            """

    static let tail = """
            const uint row = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            threadgroup float local_inv[2];
            threadgroup float local_sums[32];
            const uint base = row * 2816 + lid * 4;
            const uint wbase = lid * 4;
        \(rmsReduce("a", into: "local_inv[0]"))
        \(rmsReduce("b", into: "local_inv[1]"))
            const float inv1 = local_inv[0];
            const float inv2 = local_inv[1];
            T sv[4];
            for (int i = 0; i < 4; i++) {
                const T h1 = w1[wbase + i] * static_cast<T>((float)a[base + i] * inv1);
                const T h2 = w2[wbase + i] * static_cast<T>((float)b[base + i] * inv2);
                sv[i] = h1 + h2;
            }
        \(rmsReduce("sv", into: "local_inv[0]").replacingOccurrences(
            of: "(float)sv[base + i]", with: "(float)sv[i]"))
            const float inv3 = local_inv[0];
            const T scalar = s[0];
            for (int i = 0; i < 4; i++) {
                // Same double rounding as the stock norm-then-add pair, then
                // the layer-scalar multiply with its own stock rounding: the
                // residual sum rounds to T in a register exactly where the
                // stock graph stored it to memory, and the T*T product rounds
                // once on the store exactly like the stock multiply kernel.
                const T normed = static_cast<T>(
                    w3[wbase + i] * static_cast<T>((float)sv[i] * inv3));
                const T summed = res[base + i] + normed;
                out[base + i] = summed * scalar;
            }
        """

    static let tailChained = """
            const uint row = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            threadgroup float local_inv[2];
            threadgroup float local_sums[32];
            const uint base = row * 2816 + lid * 4;
            const uint wbase = lid * 4;
        \(rmsReduce("a", into: "local_inv[0]"))
        \(rmsReduce("b", into: "local_inv[1]"))
            const float inv1 = local_inv[0];
            const float inv2 = local_inv[1];
            T sv[4];
            for (int i = 0; i < 4; i++) {
                const T h1 = w1[wbase + i] * static_cast<T>((float)a[base + i] * inv1);
                const T h2 = w2[wbase + i] * static_cast<T>((float)b[base + i] * inv2);
                sv[i] = h1 + h2;
            }
        \(rmsReduce("sv", into: "local_inv[0]").replacingOccurrences(
            of: "(float)sv[base + i]", with: "(float)sv[i]"))
            const float inv3 = local_inv[0];
            const T scalar = s[0];
            T outv[4];
            for (int i = 0; i < 4; i++) {
                const T normed3 = static_cast<T>(
                    w3[wbase + i] * static_cast<T>((float)sv[i] * inv3));
                const T summed = res[base + i] + normed3;
                outv[i] = summed * scalar;
                out[base + i] = outv[i];
            }
        \(rmsReduce("outv", into: "local_inv[0]").replacingOccurrences(
            of: "(float)outv[base + i]", with: "(float)outv[i]"))
            const float inv4 = local_inv[0];
            for (int i = 0; i < 4; i++) {
                normed[base + i] =
                    wn[wbase + i] * static_cast<T>((float)outv[i] * inv4);
            }
        """
}
