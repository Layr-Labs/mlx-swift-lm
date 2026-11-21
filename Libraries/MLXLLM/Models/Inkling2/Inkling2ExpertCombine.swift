// Inkling 2 routed-expert combine straight from the sorted expert rows.
import Foundation
import MLX

/// `bf16(sum_k fp32(expert_k) * w_k)` read from the SORTED expert rows through
/// the inverse permutation, without the unsort gather, the FP32 widening copy
/// and the `[tokens, K, hidden]` FP32 product.
///
/// Exactness contract (byte-identical to `scatterUnsort` followed by
/// `(rows.asType(.float32) * weights[..., None]).sum(axis: -2).asType(.bfloat16)`,
/// the same argument as the Flint combine in MLXLMCommon):
///  - `float(bf16) * w` is one FP32 rounding in both (contraction is off here).
///  - MLX reduces `sum(axis: -2)` over K <= 8 slots with `col_reduce_small`,
///    folding `(p_k + 0) + acc` in slot order; the loop below does the same,
///    including the `+ 0` that turns `-0` into `+0`.
///  - The narrowing is `static_cast<bfloat16_t>`, as in MLX's copy kernel.
/// Inkling2ExpertCombineTests checks this per device.
enum Inkling2ExpertCombine {
    /// `DARKBLOOM_INKLING2_FUSED_COMBINE=0` keeps the unsort gather and the MLX reduction.
    static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_FUSED_COMBINE"]?
            .lowercased()
        return raw.map { !["0", "false", "off", "no"].contains($0) } ?? true
    }()

    private static let kernel = MLXFast.metalKernel(
        name: "inkling2_expert_weighted_unsort",
        inputNames: ["sorted_outputs", "inverse_order", "weights"],
        outputNames: ["output"],
        source: """
            #pragma clang fp contract(off)
            uint feature = thread_position_in_grid.x;
            uint token = thread_position_in_grid.y;
            const uint base = token * (uint)K;
            float accumulator = 0.0f;
            for (uint slot = 0; slot < (uint)K; ++slot) {
                const uint row = inverse_order[base + slot];
                const float product = float(sorted_outputs[row * (uint)D + feature]) * weights[base + slot];
                const float partial = product + 0.0f;
                accumulator = partial + accumulator;
            }
            output[token * (uint)D + feature] = static_cast<bfloat16_t>(accumulator);
            """,
        ensureRowContiguous: true
    )

    /// Whether the kernel reproduces the MLX expression for these arrays.
    static func supported(sortedOutputs: MLXArray, inverseOrder: MLXArray, weights: MLXArray)
        -> Bool
    {
        let k = weights.dim(-1)
        return enabled && sortedOutputs.dtype == .bfloat16 && weights.dtype == .float32
            && inverseOrder.dtype == .uint32 && (2 ... 8).contains(k)
            && inverseOrder.size == weights.size
            && sortedOutputs.size == weights.size * sortedOutputs.dim(-1)
    }

    /// `sortedOutputs`: `[tokens * K, ..., hidden]` BF16 in sorted order;
    /// `inverseOrder`: `[tokens * K]` UInt32; `weights`: `[tokens, K]` FP32.
    /// Returns `[tokens, hidden]` BF16.
    static func combine(sortedOutputs: MLXArray, inverseOrder: MLXArray, weights: MLXArray)
        -> MLXArray
    {
        let hidden = sortedOutputs.dim(-1)
        let k = weights.dim(-1)
        let tokens = weights.size / k
        return kernel(
            [sortedOutputs.reshaped(-1, hidden), inverseOrder, weights.reshaped(tokens, k)],
            template: [("K", k), ("D", hidden)],
            grid: (hidden, tokens, 1), threadGroup: (hidden.isMultiple(of: 256) ? 256 : 64, 1, 1),
            outputShapes: [[tokens, hidden]], outputDTypes: [.bfloat16]
        )[0]
    }
}

/// The sorted MoE path's permutation by counting, in blocks of 1,024
/// assignments: per block, one thread per expert counts its assignments,
/// then emits them in flat order at the expert's offset (all smaller experts,
/// then the same expert in earlier blocks). Stable, so the order equals a
/// stable argsort of the expert ids (what `gatherSort` computes with two
/// argsorts); Inkling2PrefillKernelTests checks it.
enum Inkling2ExpertSort {
    /// Off by default: on the M5 Max (quiet host, 3 interleaved rounds) the
    /// two argsorts are ~5 ms per 2,048-token chunk faster than this sort.
    /// `DARKBLOOM_INKLING2_COUNTING_SORT=1` turns it on.
    static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_COUNTING_SORT"]?
            .lowercased()
        return raw.map { ["1", "true", "on", "yes"].contains($0) } ?? false
    }()

    static let block = 1024

    private static let load = """
        threadgroup uchar local[BLK];
        const uint n = uint(ids_shape[0]);
        const uint t = thread_position_in_threadgroup.x;
        const uint b = threadgroup_position_in_grid.y;
        const uint first = b * BLK;
        const uint count = min(uint(BLK), n - first);
        for (uint i = t; i < count; i += E) {
            local[i] = uchar(ids[first + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        """

    private static let countKernel = MLXFast.metalKernel(
        name: "inkling2_expert_sort_count",
        inputNames: ["ids"],
        outputNames: ["counts"],
        source: load + """
            uint c = 0;
            for (uint i = 0; i < count; ++i) {
                c += local[i] == t ? 1u : 0u;
            }
            counts[b * E + t] = c;
            """,
        ensureRowContiguous: true
    )

    private static let scatterKernel = MLXFast.metalKernel(
        name: "inkling2_expert_sort_scatter",
        inputNames: ["ids", "counts"],
        outputNames: ["order", "sorted_ids", "inverse"],
        source: load + """
            const uint blocks = uint(counts_shape[0]) / E;
            uint w = 0;
            for (uint blk = 0; blk < blocks; ++blk) {
                for (uint e = 0; e < t; ++e) {
                    w += counts[blk * E + e];
                }
                if (blk < b) {
                    w += counts[blk * E + t];
                }
            }
            for (uint i = 0; i < count; ++i) {
                if (local[i] == t) {
                    order[w] = first + i;
                    sorted_ids[w] = t;
                    inverse[first + i] = w;
                    ++w;
                }
            }
            """,
        ensureRowContiguous: true
    )

    static func supported(_ indices: MLXArray, experts: Int) -> Bool {
        enabled && indices.dtype == .uint32 && experts <= 256 && experts % 32 == 0
    }

    /// `indices` `[..., k]` UInt32 expert ids. Returns the flat order, the
    /// sorted ids and the inverse permutation, all `[size]` UInt32.
    static func sort(_ indices: MLXArray, experts: Int) -> (MLXArray, MLXArray, MLXArray) {
        let flat = indices.flattened()
        let n = flat.size
        let blocks = (n + block - 1) / block
        let template: [(String, any KernelTemplateArg)] = [("E", experts), ("BLK", block)]
        let counts = countKernel(
            [flat], template: template, grid: (experts, blocks, 1),
            threadGroup: (experts, 1, 1), outputShapes: [[blocks * experts]],
            outputDTypes: [.uint32])[0]
        let out = scatterKernel(
            [flat, counts], template: template, grid: (experts, blocks, 1),
            threadGroup: (experts, 1, 1), outputShapes: [[n], [n], [n]],
            outputDTypes: [.uint32, .uint32, .uint32])
        return (out[0], out[1], out[2])
    }
}
