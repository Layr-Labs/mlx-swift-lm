// Copyright © 2026 Eigen Labs.
// Flint routed-expert combine. The arithmetic it reproduces is the FP32 expert
// reduction of Prince Canuma's mlx-vlm Flint (MIT); see ThirdPartyNotices/Flint.md.
import Foundation
import MLX

/// One pass over the SORTED expert rows that produces Flint's MoE output
/// `bf16(Σ_k fp32(expert_k) · w_k) + addend` without the unsort gather, the
/// FP32 widening copy, the `[tokens, K, hidden]` FP32 product or the separate
/// narrowing and shared-expert add.
///
/// Exactness contract (byte-identical to the unfused MLX expression):
///  - `float(bf16) * w` is one FP32 rounding in both (contraction is off here;
///    MLX's binary kernels are separate dispatches).
///  - MLX reduces `sum(axis: -2)` over K <= 8 slots with `col_reduce_small`:
///    one row per simdgroup lane `y`, each `p_k + 0`, then folded in slot
///    order as `(p_k + 0) + acc`. The loop below performs exactly that
///    sequence, including the `+ 0` that turns `-0` into `+0`.
///  - The narrowing is `static_cast<bfloat16_t>` (MLX's copy kernel) and the
///    add is `bfloat16_t + bfloat16_t` (MLX's `Add`), both in the same native
///    `bfloat` type MLX's own kernels use on every Metal 3.1+ device.
/// Kernels compile in MLX's default safe math mode; nothing here depends on
/// NAX or on a particular Apple GPU generation.
private let flintFP32WeightedUnsortAddKernel = MLXFast.metalKernel(
    name: "flint_fp32_weighted_unsort_add",
    inputNames: ["sorted_outputs", "inverse_order", "weights", "addend"],
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
        const uint index = token * (uint)D + feature;
        const bfloat16_t narrowed = static_cast<bfloat16_t>(accumulator);
        output[index] = narrowed + addend[index];
        """,
    ensureRowContiguous: true
)

/// Metadata-only gate for the fused combine. `inverseOrder` must be the genuine
/// SwitchGLU argsort inverse. Returns `false` for anything the kernel does not
/// reproduce bit for bit; callers then evaluate the original expression.
func flintFP32WeightedUnsortAddSupported(
    sortedOutputs: MLXArray, inverseOrder: MLXArray, weights: MLXArray, addend: MLXArray
) -> Bool {
    guard
        mimoV26FP32WeightedUnsortRefusal(
            sortedOutputs: sortedOutputs, inverseOrder: inverseOrder, weights: weights) == nil,
        sortedOutputs.dtype == .bfloat16, addend.dtype == .bfloat16,
        addend.shape == [weights.dim(0), weights.dim(1), sortedOutputs.dim(1)]
    else { return false }
    return true
}

/// `sortedOutputs`: `[tokens * K, hidden]` BF16 expert rows in sorted order.
/// `inverseOrder`: `[tokens * K]` UInt32 argsort inverse. `weights`:
/// `[B, L, K]` FP32 routing weights. `addend`: `[B, L, hidden]` BF16.
func flintFP32WeightedUnsortAdd(
    sortedOutputs: MLXArray, inverseOrder: MLXArray, weights: MLXArray, addend: MLXArray
) -> MLXArray {
    precondition(
        flintFP32WeightedUnsortAddSupported(
            sortedOutputs: sortedOutputs, inverseOrder: inverseOrder, weights: weights,
            addend: addend))
    let hidden = sortedOutputs.dim(1)
    let topK = weights.dim(2)
    let threads = hidden.isMultiple(of: 256) ? 256 : 64
    return flintFP32WeightedUnsortAddKernel(
        [sortedOutputs, inverseOrder, weights, addend],
        template: [("K", topK), ("D", hidden)],
        grid: (hidden, weights.size / topK, 1), threadGroup: (threads, 1, 1),
        outputShapes: [addend.shape], outputDTypes: [.bfloat16],
        stream: .gpu
    )[0]
}
