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

/// The decode-layout twin of `flintFP32WeightedUnsortAddKernel`: the expert
/// rows are already in slot order (`SwitchGLU` did not sort, or unsorted them),
/// so row `token * K + slot` is read directly. Same exactness contract as
/// above; the reduction is again MLX's `col_reduce_small` for K <= 8 (one row
/// per simdgroup lane `y`, `p_k + 0`, folded in slot order). With `residual`,
/// the decoder layer's `h + mlp(...)` (`bfloat16_t + bfloat16_t`, MLX's `Add`)
/// is folded in after the shared-expert add, in that order.
private func flintFP32WeightedAddKernel(residual: Bool) -> MLXFast.MLXFastKernel {
    MLXFast.metalKernel(
        name: residual ? "flint_fp32_weighted_add_residual" : "flint_fp32_weighted_add",
        inputNames: ["expert_outputs", "weights", "addend"] + (residual ? ["residual"] : []),
        outputNames: ["output"],
        source: """
            #pragma clang fp contract(off)
            uint feature = thread_position_in_grid.x;
            uint token = thread_position_in_grid.y;
            const uint base = token * (uint)K;
            float accumulator = 0.0f;
            for (uint slot = 0; slot < (uint)K; ++slot) {
                const float product = float(expert_outputs[(base + slot) * (uint)D + feature]) * weights[base + slot];
                const float partial = product + 0.0f;
                accumulator = partial + accumulator;
            }
            const uint index = token * (uint)D + feature;
            const bfloat16_t narrowed = static_cast<bfloat16_t>(accumulator);
            const bfloat16_t combined = narrowed + addend[index];
            \(residual ? "output[index] = residual[index] + combined;" : "output[index] = combined;")
            """,
        ensureRowContiguous: true
    )
}

private let flintFP32WeightedAdd = flintFP32WeightedAddKernel(residual: false)
private let flintFP32WeightedAddResidual = flintFP32WeightedAddKernel(residual: true)

/// Metadata-only gate for `flintFP32WeightedAdd`. `expertOutputs` is
/// `[B, L, K, hidden]` BF16 in slot order (what `SwitchGLU.callAsFunction`
/// returns), `weights` `[B, L, K]` FP32, `addend` and `residual` `[B, L,
/// hidden]` BF16. `2 <= K <= 8` keeps MLX on the one-row-per-lane
/// `col_reduce_small` order the kernel mirrors (K = 1 is an identity copy in
/// MLX, without the `+ 0`).
func flintFP32WeightedAddSupported(
    expertOutputs: MLXArray, weights: MLXArray, addend: MLXArray, residual: MLXArray?
) -> Bool {
    guard expertOutputs.ndim == 4, weights.ndim == 3,
        Array(expertOutputs.shape.prefix(3)) == weights.shape,
        (2 ... 8).contains(weights.dim(2)),
        expertOutputs.dim(3) > 0, expertOutputs.dim(3).isMultiple(of: 64),
        expertOutputs.dtype == .bfloat16, weights.dtype == .float32,
        addend.dtype == .bfloat16,
        addend.shape == [weights.dim(0), weights.dim(1), expertOutputs.dim(3)],
        residual.map({ $0.dtype == .bfloat16 && $0.shape == addend.shape }) ?? true,
        expertOutputs.size <= Int(Int32.max)
    else { return false }
    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
    return MLXProcessStreams.isDefaultGPU(.default)
    #else
    return false
    #endif
}

/// `bf16(Σ_k fp32(expertOutputs[..., k, :]) · weights[..., k]) + addend`
/// (then `residual + ...` when given), byte-identical to MLX evaluating
/// `(expertOutputs.asType(.float32) * weights[.ellipsis, .newAxis]).sum(axis: -2)
/// .asType(.bfloat16) + addend` (and `residual + that`) in one dispatch.
func flintFP32WeightedAdd(
    expertOutputs: MLXArray, weights: MLXArray, addend: MLXArray, residual: MLXArray? = nil
) -> MLXArray {
    precondition(
        flintFP32WeightedAddSupported(
            expertOutputs: expertOutputs, weights: weights, addend: addend, residual: residual))
    let hidden = expertOutputs.dim(3)
    let topK = weights.dim(2)
    let threads = hidden.isMultiple(of: 256) ? 256 : 64
    let kernel = residual == nil ? flintFP32WeightedAdd : flintFP32WeightedAddResidual
    return kernel(
        [expertOutputs, weights, addend] + (residual.map { [$0] } ?? []),
        template: [("K", topK), ("D", hidden)],
        grid: (hidden, weights.size / topK, 1), threadGroup: (threads, 1, 1),
        outputShapes: [addend.shape], outputDTypes: [.bfloat16],
        stream: .gpu
    )[0]
}
