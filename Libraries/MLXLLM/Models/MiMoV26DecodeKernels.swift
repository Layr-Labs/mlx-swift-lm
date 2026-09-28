// SPDX-License-Identifier: Apache-2.0
// Adapted from jundot/omlx PR #3990, decode_fast.py at
// e58cab4d5db6c3ca27742a45a50ab6287b1b1bcb (2026).
// RMS reduction derived from MLX rms_norm.metal, Copyright © 2024 Apple Inc.
// Swift adaptation Copyright © 2026 Eigen Labs.
// Attribution: docs/mimo-v26/DECODE-KERNEL-ATTRIBUTION.md.

import MLX
import MLXLMCommon
import MLXNN

/// Source-only candidate for the residual/norm portion of the short-forward
/// path. No routing, expert projection, attention or cache arithmetic changes.
/// All weights are read on each call; no reload-sensitive tensor cache exists.
enum MiMoV26DecodeKernels {
    struct Result {
        let residual: MLXArray
        let normalized: MLXArray
    }

    /// Matches MLX's one-row RMS reduction: four values per lane, 32 lanes
    /// per SIMD group and at most 32 SIMD groups per row. Batching is deliberately
    /// excluded until the complete batched-state gate is qualified.
    static func supports(shape: [Int], dtype: DType, device: DeviceType?) -> Bool {
        guard device == .gpu, shape.count == 3, shape[0] == 1,
              (1...7).contains(shape[1]), (4...4096).contains(shape[2]),
              shape[2] % 4 == 0 else { return false }
        return dtype == .bfloat16 || dtype == .float16
    }

    private static func supports(_ x: MLXArray, norm: RMSNorm,
                                 stream: StreamOrDevice = .default) -> Bool {
        supports(shape: x.shape, dtype: x.dtype, device: MiMoV26DecodeStream.deviceType(of: stream))
            && ObjectIdentifier(type(of: norm)) == ObjectIdentifier(RMSNorm.self)
            && norm.weight.shape == [x.dim(-1)] && norm.weight.dtype == x.dtype
            && norm.eps.isFinite && norm.eps > 0
    }

    /// Optional result is the dispatch witness used by component tests. A nil
    /// result has not constructed a custom kernel or changed a cache.
    static func addRMS(_ x: MLXArray, _ y: MLXArray, norm: RMSNorm) -> Result? {
        let stream = StreamOrDevice.default
        guard supports(x, norm: norm, stream: stream), y.shape == x.shape, y.dtype == x.dtype else {
            return nil
        }
        let width = x.dim(-1), threads = ((width + 127) / 128) * 32
        let out = addKernel(
            [x, y, norm.weight, MLXArray([norm.eps])],
            template: [("T", x.dtype), ("AXIS", width)],
            grid: (threads * x.dim(1), 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [x.shape, x.shape], outputDTypes: [x.dtype, x.dtype], stream: stream)
        return Result(residual: out[0], normalized: out[1])
    }

    /// The reference widens expert activations for FP32 weighting/reduction,
    /// rounds the sum to the activation dtype, adds the residual in that dtype,
    /// then normalizes. Retain each of those rounding points explicitly.
    static func combineRMS(_ h: MLXArray, experts: MLXArray, weights: MLXArray,
                           norm: RMSNorm) -> Result? {
        let stream = StreamOrDevice.default
        guard supports(h, norm: norm, stream: stream), experts.ndim == 4, weights.ndim == 3,
              experts.shape == [1, h.dim(1), weights.dim(-1), h.dim(2)],
              experts.dtype == h.dtype,
              weights.shape == [1, h.dim(1), experts.dim(2)],
              weights.dtype == .float32, (1...32).contains(experts.dim(2)),
              weights.size < 64 else { return nil }
        let width = h.dim(-1), threads = ((width + 127) / 128) * 32
        let out = combineKernel(
            [h, experts, weights, norm.weight, MLXArray([norm.eps])],
            template: [("T", h.dtype), ("AXIS", width), ("TOPK", experts.dim(2))],
            grid: (threads * h.dim(1), 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [h.shape, h.shape], outputDTypes: [h.dtype, h.dtype], stream: stream)
        return Result(residual: out[0], normalized: out[1])
    }

    /// Complete the layer after its existing attention operation. The caller
    /// carries `normalized` into the next layer (or returns it as final target
    /// post-norm hidden state), while feature captures retain `residual`.
    static func finishLayer(_ hidden: MLXArray, attentionOutput: MLXArray,
                            layer: MiMoV26DecoderLayer, nextNorm: RMSNorm,
                            enabled: Bool) -> Result? {
        guard enabled, supports(hidden, norm: nextNorm),
              supports(hidden, norm: layer.postAttentionNorm),
              let post = addRMS(hidden, attentionOutput, norm: layer.postAttentionNorm)
        else { return nil }

        if let moe = layer.mlp as? MiMoV26MoE,
           ObjectIdentifier(type(of: moe.switchMLP)) == ObjectIdentifier(SwitchGLU.self),
           moe.gate.config.expertsPerToken <= 32,
           hidden.dim(1) * moe.gate.config.expertsPerToken < 64 {
            // Invoke the original router and SwitchGLU. In particular, retain
            // the declared router operand precision and its exact selection.
            let routed = moe.gate(post.normalized)
            let experts = moe.switchMLP(post.normalized, routed.indices)
            if let combined = combineRMS(post.residual, experts: experts,
                                         weights: routed.weights, norm: nextNorm) {
                return combined
            }
            // Same already-computed rows and weights if a future module changes
            // their dtype/layout. Never rerun the router or drop a projection.
            let output = (experts * routed.weights[.ellipsis, .newAxis])
                .sum(axis: -2).asType(hidden.dtype)
            if let added = addRMS(post.residual, output, norm: nextNorm) { return added }
            let residual = post.residual + output
            return Result(residual: residual, normalized: nextNorm(residual))
        }
        let output = layer.mlp(post.normalized)
        if let added = addRMS(post.residual, output, norm: nextNorm) { return added }
        let residual = post.residual + output
        return Result(residual: residual, normalized: nextNorm(residual))
    }

    // Static kernels initialize lazily, only after the GPU/shape/dtype gate.
    // This is ordinary SIMD Metal, with no NAX or M5-specific dispatch.
    private static let rmsTail = """
        acc = simd_sum(acc);
        if (simd_group_id == 0) {
            local_sums[simd_lane_id] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane_id == 0) {
            local_sums[simd_group_id] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group_id == 0) {
            acc = simd_sum(local_sums[simd_lane_id]);
            if (simd_lane_id == 0) {
                local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < N_READS; i++) {
            if (lid * N_READS + i < axis_size) {
                n_out[base + i] = w[lid * N_READS + i]
                    * static_cast<T>(thread_x[i] * local_inv_mean[0]);
            }
        }
        """

    private static let addKernel = MLXFast.metalKernel(
        name: "mimo_v26_decode_add_rms",
        inputNames: ["x", "y", "w", "eps_in"], outputNames: ["h_out", "n_out"],
        source: """
            constexpr int N_READS = 4;
            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[32];
            uint gid = threadgroup_position_in_grid.x;
            uint lid = thread_position_in_threadgroup.x;
            uint simd_lane_id = thread_index_in_simdgroup;
            uint simd_group_id = simdgroup_index_in_threadgroup;
            const uint axis_size = AXIS;
            const float eps = eps_in[0];
            size_t base = size_t(gid) * axis_size + lid * N_READS;
            float acc = 0;
            float thread_x[N_READS];
            for (int i = 0; i < N_READS; i++) {
                if (lid * N_READS + i < axis_size) {
                    T hv = x[base + i] + y[base + i];
                    h_out[base + i] = hv;
                    thread_x[i] = hv;
                } else {
                    thread_x[i] = 0;
                }
                acc += thread_x[i] * thread_x[i];
            }
            """ + rmsTail, ensureRowContiguous: true)

    private static let combineKernel = MLXFast.metalKernel(
        name: "mimo_v26_decode_combine_residual_rms",
        inputNames: ["h", "y", "scores", "w", "eps_in"],
        outputNames: ["h_out", "n_out"],
        source: """
            constexpr int N_READS = 4;
            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[32];
            uint gid = threadgroup_position_in_grid.x;
            uint lid = thread_position_in_threadgroup.x;
            uint simd_lane_id = thread_index_in_simdgroup;
            uint simd_group_id = simdgroup_index_in_threadgroup;
            const uint axis_size = AXIS;
            const float eps = eps_in[0];
            size_t base = size_t(gid) * axis_size + lid * N_READS;
            size_t ybase = size_t(gid) * TOPK * axis_size + lid * N_READS;
            float acc = 0;
            float thread_x[N_READS];
            for (int i = 0; i < N_READS; i++) {
                if (lid * N_READS + i < axis_size) {
                    float total = 0;
                    for (int k = 0; k < TOPK; k++) {
                        // The materialized reference rounds each FP32 product
                        // before the sum; volatile prohibits FMA contraction.
                        volatile float product = static_cast<float>(
                            y[ybase + size_t(k) * axis_size + i]) * scores[gid * TOPK + k];
                        total = product + total;
                    }
                    T yv = static_cast<T>(total);
                    T hv = h[base + i] + yv;
                    h_out[base + i] = hv;
                    thread_x[i] = hv;
                } else {
                    thread_x[i] = 0;
                }
                acc += thread_x[i] * thread_x[i];
            }
            """ + rmsTail, ensureRowContiguous: true)
}
