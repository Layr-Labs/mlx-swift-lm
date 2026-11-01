// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import MLX

/// The indexer key LayerNorm (epsilon 1e-6) in one launch for
/// `GLM5NextExecutionPolicy.Normalization.fusedKernel`: FP32 mean, centered
/// mean of squares, rsqrt, FP32 affine and one final cast, as the reference.
/// Its sums use this kernel's order, so not bitwise. Returns nil off the
/// default GPU stream or for unsupported shapes, leaving the reference graph.
enum GLM5NextIndexNormKernel {
    static let threads = 128

    static func apply(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray? {
        let width = x.dim(-1)
        guard x.ndim >= 1, width > 0, width <= 4 * threads, weight.shape == [width],
            bias.shape == [width], [.float16, .bfloat16, .float32].contains(x.dtype),
            x.size <= Int(Int32.max), glm5NextOnDefaultGPUStream()
        else { return nil }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            let rows = x.size / width
            return kernel(
                [x, weight, bias],
                template: [("T", x.dtype), ("Width", width), ("Threads", threads)],
                grid: (threads, rows, 1), threadGroup: (threads, 1, 1),
                outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
        #else
            return nil
        #endif
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        private static let kernel = MLXFast.metalKernel(
            name: "glm5_next_index_layer_norm",
            inputNames: ["x", "weight", "bias"], outputNames: ["y"],
            source: """
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                constexpr uint Groups = uint(Threads) / 32;
                constexpr uint PerThread = (uint(Width) + uint(Threads) - 1) / uint(Threads);
                threadgroup float totals[Groups];
                const uint row = threadgroup_position_in_grid.y;
                const uint tid = thread_position_in_threadgroup.x;
                const uint lane = thread_index_in_simdgroup;
                const uint group = simdgroup_index_in_threadgroup;
                const size_t offset = size_t(row) * Width;
                float values[PerThread];
                float sum = 0.0f;
                for (uint i = 0; i < PerThread; ++i) {
                    const uint c = tid + i * Threads;
                    values[i] = c < uint(Width) ? static_cast<float>(x[offset + c]) : 0.0f;
                    sum += values[i];
                }
                sum = simd_sum(sum);
                if (lane == 0) { totals[group] = sum; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float total = 0.0f;
                for (uint i = 0; i < Groups; ++i) { total += totals[i]; }
                const float mean = total * (1.0f / float(Width));
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float squares = 0.0f;
                for (uint i = 0; i < PerThread; ++i) {
                    const uint c = tid + i * Threads;
                    if (c < uint(Width)) {
                        values[i] = values[i] - mean;
                        const float square = values[i] * values[i];
                        squares += square;
                    }
                }
                squares = simd_sum(squares);
                if (lane == 0) { totals[group] = squares; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float variance = 0.0f;
                for (uint i = 0; i < Groups; ++i) { variance += totals[i]; }
                variance = variance * (1.0f / float(Width));
                const float inverse = metal::precise::rsqrt(variance + 1e-6f);
                for (uint i = 0; i < PerThread; ++i) {
                    const uint c = tid + i * Threads;
                    if (c < uint(Width)) {
                        const float normalized = values[i] * inverse;
                        const float scaled = normalized * static_cast<float>(weight[c]);
                        const float shifted = scaled + static_cast<float>(bias[c]);
                        y[offset + c] = static_cast<T>(shifted);
                    }
                }
                """,
            ensureRowContiguous: true)
    #endif
}
