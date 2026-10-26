// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import MLXLMCommon

/// One decode step's attention read straight from a `PagedAttentionKVPool`'s
/// pages: MLX's own single-query kernels (`sdpa_vector`, and
/// `sdpa_vector_2pass_1` + `sdpa_vector_2pass_2` past MLX's two-pass
/// threshold; `mlx/backend/metal/kernels/sdpa_vector.h`), reproduced
/// statement for statement, with each key and value row addressed through
/// the page table instead of a strided buffer. Same simdgroup layout, same
/// loop order, same float accumulators, same `fast::exp`, same block count
/// (MLX's host rule for this device, `sdpa_vector_2pass`): every output is
/// the bits `scaledDotProductAttention` computes on the gathered
/// `[1, kvHeads, N, headDim]` arrays, without gathering them.
///
/// `plan` returns nil wherever MLX would not run these kernels (a query of
/// more than one token, a mask, a head dim the vector kernel does not take,
/// a GQA factor past 32, or MLX's GQA-specialised two-pass variant): the
/// caller then gathers and calls `scaledDotProductAttention` as before.
enum PagedAttentionDecode {
    struct Plan: Equatable {
        /// nil: the one-pass kernel; else the two-pass kernel's block count.
        let blocks: Int?
    }

    /// The last character of this GPU's architecture name, as MLX reads it
    /// (`metal::Device::get_architecture().back()`).
    static let architectureClass: Character? = GPU.deviceInfo().architecture.last

    /// MLX's routing for a single-query `scaledDotProductAttention` of
    /// `queryHeads` heads over `N` keys of `kvHeads` heads, or nil where MLX
    /// would not take `sdpa_vector` / `sdpa_vector_2pass` (generic `_1`).
    static func plan(
        queryHeads: Int, kvHeads: Int, headDim: Int, valueHeadDim: Int, N: Int, dtype: DType,
        architecture: Character? = architectureClass,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Plan? {
        guard N >= 1, kvHeads > 0, queryHeads % kvHeads == 0, headDim == valueHeadDim,
            [64, 96, 128, 256].contains(headDim), [.float32, .float16, .bfloat16].contains(dtype)
        else { return nil }
        let gqa = queryHeads / kvHeads
        guard gqa <= 32 else { return nil }
        let twoPass =
            ((architecture == "d" || architecture == "s") && N >= 1024)
            || (kvHeads < queryHeads && N >= 4096)
        guard twoPass else { return Plan(blocks: nil) }
        // MLX's duplication-free GQA pass is another kernel: not reproduced.
        if queryHeads == 8 * kvHeads && (headDim == 64 || headDim == 128) && N >= 8192 {
            return nil
        }
        let nSimds = gqa
        var blocks: Int
        switch architecture ?? " " {
        case "s":
            blocks = 64
            if N > 1024 && nSimds > 4 {
                blocks = N <= 8192 ? 128 : N <= 32768 ? 256 : N <= 65536 ? 512 : 1024
            }
        case "d":
            blocks = 128
            if nSimds <= 2 && N > 8192 {
                blocks = 256
            } else if nSimds >= 6 {
                if N >= 16384 && N < 65536 {
                    blocks = 512
                } else if N >= 65536 {
                    blocks = 1024
                }
            }
        default:
            blocks = nSimds >= 4 ? 64 : 32
        }
        if let value = environment["MLX_SDPA_BLOCKS"], let requested = Int(value), requested > 0 {
            blocks = (requested + 31) / 32 * 32
        }
        return Plan(blocks: blocks)
    }

    /// `[1, queryHeads, 1, headDim]` attention of `queries` over the first
    /// `count` tokens of the pages `pages` (after `fence`), and the read's fence.
    static func attend(
        queries: MLXArray, scale: Float, slabs: [MLXArray], pageSize: Int, kvHeads: Int,
        headDim: Int, pages: MLXArray, count: Int, fence: MLXArray, plan: Plan
    ) -> (output: MLXArray, fence: MLXArray) {
        let queryHeads = queries.dim(1)
        let dtype = queries.dtype
        let constants = GLM5NextKernelConstants.shared
        let n = constants.int32s([Int32(count)])
        let scaleArray = constants.floats([scale])
        let template: [(String, any KernelTemplateArg)] = [
            ("T", dtype), ("D", headDim), ("V", headDim), ("PS", pageSize), ("HKV", kvHeads),
        ]
        guard let blocks = plan.blocks else {
            var oneTemplate = template
            oneTemplate.append(("GQA", queryHeads / kvHeads))
            let outputs = onePass(
                [queries, slabs[0], slabs[1], pages, n, scaleArray, fence],
                template: oneTemplate,
                grid: (1024 * queryHeads, 1, 1), threadGroup: (1024, 1, 1),
                outputShapes: [[1, queryHeads, 1, headDim], [1]], outputDTypes: [dtype, .int32])
            return (outputs[0], outputs[1])
        }
        let gqa = queryHeads / kvHeads
        var firstTemplate = template
        firstTemplate.append(("BLOCKS", blocks))
        let secondTemplate: [(String, any KernelTemplateArg)] = [
            ("T", dtype), ("D", headDim), ("BLOCKS", blocks),
        ]
        let first = twoPassFirst(
            [queries, slabs[0], slabs[1], pages, n, scaleArray, fence],
            template: firstTemplate,
            grid: (32 * kvHeads, gqa, blocks), threadGroup: (32, gqa, 1),
            outputShapes: [
                [1, queryHeads, 1, blocks, headDim], [1, queryHeads, 1, blocks],
                [1, queryHeads, 1, blocks], [1],
            ],
            outputDTypes: [.float32, .float32, .float32, .int32])
        let output = twoPassSecond(
            [first[0], first[1], first[2]],
            template: secondTemplate,
            grid: (1024 * queryHeads, 1, 1), threadGroup: (1024, 1, 1),
            outputShapes: [[1, queryHeads, 1, headDim]], outputDTypes: [dtype])[0]
        return (output, first[3])
    }

    /// `sdpa_vector` (no mask, no sinks, row-contiguous query) with paged keys and values.
    private static let onePassSource = """
        constexpr int BN = 32;
        constexpr int BD = 32;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread = V / BD;
        const uint3 tid = threadgroup_position_in_grid;
        const uint3 tpg = threadgroups_per_grid;
        const uint simd_gid = simdgroup_index_in_threadgroup;
        const uint simd_lid = thread_index_in_simdgroup;
        const int N = count[0];

        typedef float U;

        thread U q[qk_per_thread];
        thread U k[qk_per_thread];
        thread U o[v_per_thread];

        threadgroup U outputs[BN * BD];
        threadgroup U max_scores[BN];
        threadgroup U sum_exp_scores[BN];

        // Adjust positions
        const int q_batch_head_idx = tid.x;
        const int q_seq_idx = tid.y;
        const int kv_head_idx = q_batch_head_idx / GQA;
        const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
        const int q_offset = o_offset;
        const device T* query_row = queries + q_offset * D + simd_lid * qk_per_thread;
        device T* out_row = out + o_offset * V + simd_gid * v_per_thread;

        // Read the query and 0 the output accumulator
        for (int i = 0; i < qk_per_thread; i++) {
          q[i] = static_cast<U>(scale[0]) * query_row[i];
        }
        for (int i = 0; i < v_per_thread; i++) {
          o[i] = 0;
        }

        U max_score = Limits<U>::finite_min;
        U sum_exp_score = 0;

        // For each key (its page read one key ahead: addresses only)
        size_t page_base = int(simd_gid) < N ? size_t(pages[int(simd_gid) / PS]) * PS : 0;
        for (int i = simd_gid; i < N; i += BN) {
          const size_t row = page_base + size_t(i % PS);
          const int next = i + BN;
          page_base = next < N ? size_t(pages[next / PS]) * PS : 0;
          const device T* keys = k_slab + row * (HKV * D) + kv_head_idx * D + simd_lid * qk_per_thread;
          const device T* values = v_slab + row * (HKV * V) + kv_head_idx * V + simd_lid * v_per_thread;
          // Read the key
          for (int j = 0; j < qk_per_thread; j++) {
            k[j] = keys[j];
          }

          // Compute the i-th score
          U score = 0;
          for (int j = 0; j < qk_per_thread; j++) {
            score += q[j] * k[j];
          }
          score = simd_sum(score);

          // Update the accumulators
          U new_max = max(max_score, score);
          U factor = fast::exp(max_score - new_max);
          U exp_score = fast::exp(score - new_max);

          max_score = new_max;
          sum_exp_score = sum_exp_score * factor + exp_score;

          // Update the output accumulator
          for (int j = 0; j < v_per_thread; j++) {
            o[j] = o[j] * factor + exp_score * values[j];
          }
        }

        // Each thread has a partial part of the output so we need to combine them.

        // First let's communicate the max and sum_exp
        if (simd_lid == 0) {
          max_scores[simd_gid] = max_score;
          sum_exp_scores[simd_gid] = sum_exp_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_score = max_scores[simd_lid];
        U new_max = simd_max(max_score);
        U factor = fast::exp(max_score - new_max);
        sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

        // Now we need to aggregate all the outputs
        for (int i = 0; i < v_per_thread; i++) {
          outputs[simd_lid * BD + simd_gid] = o[i];
          threadgroup_barrier(mem_flags::mem_threadgroup);
          o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
          o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // And write the output
        if (simd_lid == 0) {
          for (int i = 0; i < v_per_thread; i++) {
            out_row[i] = static_cast<T>(o[i]);
          }
        }
        if (tid.x == 0 && simd_gid == 0 && simd_lid == 0) { fence_out[0] = fence[0] ^ 1; }
        """

    /// `sdpa_vector_2pass_1` (no mask, no sinks, row-contiguous query) with
    /// paged keys and values.
    private static let twoPassFirstSource = """
        constexpr int BD = 32;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread = V / BD;
        const uint3 tptg = threads_per_threadgroup;
        const uint3 tidtg = thread_position_in_threadgroup;
        const uint3 tid = threadgroup_position_in_grid;
        const uint3 tpg = threadgroups_per_grid;
        const uint simd_lid = thread_index_in_simdgroup;
        const int N = count[0];
        constexpr int blocks = BLOCKS;

        typedef float U;

        thread U q[qk_per_thread];
        thread U o[v_per_thread] = {0};

        // Adjust positions
        const int kv_head_idx = tid.x;
        const int batch_idx = tid.y;
        const int block_idx = tid.z;
        const int gqa_factor = tptg.y;
        const int q_seq_len = tptg.z;
        const int q_seq_idx = tidtg.z;
        const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
        const int num_kv_heads = tpg.x;
        const int num_q_heads = num_kv_heads * gqa_factor;
        const int q_batch_head_idx = (batch_idx * num_q_heads + q_head_idx);
        const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
        const int q_offset = o_offset;

        const device T* query_row = queries + q_offset * D + simd_lid * qk_per_thread;
        device float* out_row = partials + o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
        device float* sums_row = sums + o_offset * blocks + block_idx;
        device float* maxs_row = maxs + o_offset * blocks + block_idx;

        // Read the query
        for (int i = 0; i < qk_per_thread; i++) {
          q[i] = static_cast<U>(scale[0]) * query_row[i];
        }

        U max_score = Limits<U>::finite_min;
        U sum_exp_score = 0;

        // For each key (its page read one key ahead: addresses only)
        size_t page_base = block_idx < N ? size_t(pages[block_idx / PS]) * PS : 0;
        for (int i = block_idx; i < N; i += blocks) {
          const size_t row = page_base + size_t(i % PS);
          const int next = i + blocks;
          page_base = next < N ? size_t(pages[next / PS]) * PS : 0;
          const device T* keys = k_slab + row * (HKV * D) + kv_head_idx * D + simd_lid * qk_per_thread;
          const device T* values = v_slab + row * (HKV * V) + kv_head_idx * V + simd_lid * v_per_thread;
          // Compute the i-th score
          U score = 0;
          for (int i = 0; i < qk_per_thread; i++) {
            score += q[i] * keys[i];
          }
          score = simd_sum(score);

          // Update the accumulators
          U new_max = max(max_score, score);
          U factor = fast::exp(max_score - new_max);
          U exp_score = fast::exp(score - new_max);

          max_score = new_max;
          sum_exp_score = sum_exp_score * factor + exp_score;

          // Update the output accumulator
          for (int i = 0; i < v_per_thread; i++) {
            o[i] = o[i] * factor + exp_score * values[i];
          }
        }

        // Write the sum and max and outputs
        if (simd_lid == 0) {
          sums_row[0] = sum_exp_score;
          maxs_row[0] = max_score;
        }

        for (int i = 0; i < v_per_thread; i++) {
          out_row[i] = o[i];
        }
        if (tid.x == 0 && tid.z == 0 && tidtg.y == 0 && simd_lid == 0) { fence_out[0] = fence[0] ^ 1; }
        """

    /// `sdpa_vector_2pass_2`, verbatim.
    private static let twoPassSecondSource = """
        constexpr int BN = 32;
        constexpr int BD = 32;
        constexpr int elem_per_thread = D / BD;
        const uint3 tid = threadgroup_position_in_grid;
        const uint3 tpg = threadgroups_per_grid;
        const uint simd_gid = simdgroup_index_in_threadgroup;
        const uint simd_lid = thread_index_in_simdgroup;
        constexpr int blocks = BLOCKS;

        typedef float U;

        thread U o[elem_per_thread] = {0};
        threadgroup U outputs[BN * BD];

        // Adjust positions
        const int head_idx = tid.x;
        const int q_seq_idx = tid.y;
        const int q_offset = head_idx * tpg.y + q_seq_idx;
        const device float* partials_row = partials + q_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
        const device float* sums_row = sums + q_offset * blocks;
        const device float* maxs_row = maxs + q_offset * blocks;
        device T* out_row = out + q_offset * D + simd_gid * elem_per_thread;

        // Set defaults
        U sum_exp_score = 0.0;
        U max_score = Limits<U>::finite_min;

        // Reduce the max
        for (int b = 0; b < blocks / BN; ++b) {
          max_score = max(max_score, maxs_row[simd_lid + BN * b]);
        }
        max_score = simd_max(max_score);

        // Reduce the d
        for (int b = 0; b < blocks / BN; ++b) {
          U factor = fast::exp(maxs_row[simd_lid + BN * b] - max_score);
          sum_exp_score += factor * sums_row[simd_lid + BN * b];
        }
        sum_exp_score = simd_sum(sum_exp_score);

        // Reduce the sum exp and partials
        for (int b = 0; b < blocks / BN; ++b) {
          U factor = fast::exp(maxs_row[simd_gid] - max_score);

          // Update the output accumulator
          for (int i = 0; i < elem_per_thread; i++) {
            o[i] += factor * static_cast<U>(partials_row[i]);
          }
          maxs_row += BN;
          sums_row += BN;
          partials_row += BN * D;
        }

        // Use shared memory to transpose and reduce the final block
        for (int i = 0; i < elem_per_thread; i++) {
          outputs[simd_lid * BD + simd_gid] = o[i];
          threadgroup_barrier(mem_flags::mem_threadgroup);
          o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
          o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // And write the output
        if (simd_lid == 0) {
          for (int i = 0; i < elem_per_thread; i++) {
            out_row[i] = static_cast<T>(o[i]);
          }
        }
        """

    private static let onePass = MLXFast.metalKernel(
        name: "paged_sdpa_vector",
        inputNames: ["queries", "k_slab", "v_slab", "pages", "count", "scale", "fence"],
        outputNames: ["out", "fence_out"], source: onePassSource,
        mutableInputs: ["k_slab", "v_slab"])
    private static let twoPassFirst = MLXFast.metalKernel(
        name: "paged_sdpa_vector_2pass_1",
        inputNames: ["queries", "k_slab", "v_slab", "pages", "count", "scale", "fence"],
        outputNames: ["partials", "sums", "maxs", "fence_out"], source: twoPassFirstSource,
        mutableInputs: ["k_slab", "v_slab"])
    private static let twoPassSecond = MLXFast.metalKernel(
        name: "paged_sdpa_vector_2pass_2", inputNames: ["partials", "sums", "maxs"],
        outputNames: ["out"], source: twoPassSecondSource)
}
