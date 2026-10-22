// Copyright 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN

/// An affine quantized linear layer (4 or 8 bits, no bias) applied to several
/// independent one-row inputs in one launch, each output computed exactly as
/// MLX's one-row `qmv_fast` computes it alone: the same lane mapping (two
/// packs per lane, 32 lanes per block), block order, per-lane accumulation,
/// four rows per SIMD group and `simd_sum`. Each weight block is loaded once
/// for every vector, so `V` requests' one-row products (the NextN proposal
/// heads of lanes decoding together) read the weights once instead of `V`
/// times, and each vector's logits are bitwise those of `linear(x_v)` alone.
enum GLM5NextMultiVectorQMV {
    /// The most vectors one launch takes.
    static let maximumVectors = 8

    /// `linear` applied to each row of `x` (`[V, K]`, `1 ... maximumVectors`
    /// rows), `[V, N]`; nil when `linear` or `x` is not one this kernel reads
    /// with MLX's one-row `qmv_fast` arithmetic (then the caller applies
    /// `linear` to each row itself).
    static func apply(_ linear: Linear, _ x: MLXArray) -> MLXArray? {
        guard let layer = linear as? QuantizedLinear, layer.mode == .affine,
            [4, 8].contains(layer.bits), layer.bias == nil, let biases = layer.biases,
            x.ndim == 2, (1 ... maximumVectors).contains(x.dim(0)),
            glm5NextOnDefaultGPUStream()
        else { return nil }
        let (n, k) = layer.shape
        // MLX's `qmv_fast` conditions (`quantized.cpp` qmv): whole 8-row
        // output tiles and whole two-pack blocks.
        let block = 32 * 2 * 32 / layer.bits
        guard x.dim(1) == k, k % block == 0, n % 8 == 0, k % layer.groupSize == 0,
            layer.groupSize % (2 * 32 / layer.bits) == 0,
            x.dtype == layer.scales.dtype, biases.dtype == x.dtype,
            [.bfloat16, .float16, .float32].contains(x.dtype)
        else { return nil }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            let vectors = x.dim(0)
            return kernel(
                [contiguous(x), layer.weight, layer.scales, biases],
                template: [
                    ("T", x.dtype), ("K", k), ("N", n), ("GS", layer.groupSize),
                    ("Bits", layer.bits), ("V", vectors),
                ],
                grid: (32, 2 * (n / 8), 1), threadGroup: (32, 2, 1),
                outputShapes: [[vectors, n]], outputDTypes: [x.dtype])[0]
        #else
            return nil
        #endif
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        private static let kernel = MLXFast.metalKernel(
            name: "glm5_next_multi_vector_qmv",
            inputNames: ["x", "w", "scales", "biases"],
            outputNames: ["y"],
            source: """
                const uint simd_lid = thread_index_in_simdgroup;
                const uint simd_gid = simdgroup_index_in_threadgroup;
                // qmv_fast: eight output rows a threadgroup, four a SIMD group.
                const int out_row = (int(threadgroup_position_in_grid.y) * GLM5D_NSG + int(simd_gid)) * GLM5D_RPS;
                constexpr int vpt = 2 * 32 / Bits;
                constexpr int block = 32 * vpt;
                constexpr int row_bytes = K * Bits / 8;
                constexpr int row_groups = K / GS;
                const device uint8_t* ws = (const device uint8_t*)w + size_t(out_row) * row_bytes
                    + simd_lid * vpt * Bits / 8;
                const device T* sl = scales + size_t(out_row) * row_groups + simd_lid / (GS / vpt);
                const device T* bl = biases + size_t(out_row) * row_groups + simd_lid / (GS / vpt);
                const device T* xs = x + simd_lid * vpt;
                float result[V][GLM5D_RPS];
                for (int v = 0; v < V; ++v) {
                    for (int row = 0; row < GLM5D_RPS; ++row) { result[v][row] = 0.0f; }
                }
                thread float x_thread[vpt];
                for (int k = 0; k < K; k += block) {
                    for (int v = 0; v < V; ++v) {
                        // Each vector's activations, sum and row products, as its own qmv_fast.
                        const float sum = glm5d_load_vector<T, vpt, Bits>(xs + size_t(v) * K + k, x_thread);
                        for (int row = 0; row < GLM5D_RPS; ++row) {
                            result[v][row] += glm5d_qdot<vpt, Bits>(
                                ws + row * row_bytes, x_thread, float(sl[row * row_groups]),
                                float(bl[row * row_groups]), sum);
                        }
                    }
                    ws += block * Bits / 8;
                    sl += block / GS;
                    bl += block / GS;
                }
                for (int v = 0; v < V; ++v) {
                    for (int row = 0; row < GLM5D_RPS; ++row) {
                        const float total = simd_sum(result[v][row]);
                        if (simd_lid == 0) {
                            y[size_t(v) * N + out_row + row] = static_cast<T>(total);
                        }
                    }
                }
                """,
            header: GLM5NextMoEDecodeKernel.header,
            ensureRowContiguous: true)
    #endif
}
