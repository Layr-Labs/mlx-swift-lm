// DIAGNOSTIC (scratch): the hyper-connection mixer's non-matmul ops as
// custom kernels around the built-in quantized matmuls. Per mixer today:
// grouped rmsNorm, scale, mixDown qmm, divide, silu, mixUp qmm, sigmoid,
// reshape·multiply, mean, blockInject qmm, divide, sigmoid, 2·, and the
// inject's broadcast multiply and add ≈ 18 launches. Here: norm(1) + qmm +
// scaledSilu(1) + qmm + mix(1) + qmm + inject(1) = 7.
import Foundation
import MLX
import MLXFast
import MLXNN

enum Qwen4ExpHCScratchKernels {
    /// Grouped RMS norm over HC groups of D, times (weight + offset), one
    /// simdgroup per (row, group).
    static let norm: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "qwen4exp_hc_norm_scratch", inputNames: ["x", "w", "eps"], outputNames: ["y"],
        source: """
                auto lane = thread_position_in_threadgroup.x;
                auto g = thread_position_in_grid.y;
                auto row = thread_position_in_grid.z;
                constexpr int per = D / 32;
                auto base = (row * HC + g) * D;
                float ss = 0.0f;
                float vals[per];
                for (int i = 0; i < per; ++i) { vals[i] = static_cast<float>(x[base + lane * per + i]); ss += vals[i] * vals[i]; }
                ss = simd_sum(ss);
                float inv = metal::rsqrt(ss / float(D) + eps[0]);
                for (int i = 0; i < per; ++i) {
                    auto d = lane * per + i;
                    y[base + d] = static_cast<InT>(vals[i] * inv * (static_cast<float>(w[g * D + d]) + OFFSET));
                }
            """)
    static func norm(_ x: MLXArray, weight: MLXArray, hc: Int, dims: Int, eps: Float, offsetZero: Bool) -> MLXArray {
        let rows = x.size / (hc * dims)
        return norm([x, weight, MLXArray([eps])],
            template: [("InT", x.dtype), ("D", dims), ("HC", hc), ("OFFSET", offsetZero ? 0 : 1)],
            grid: (32, hc, rows), threadGroup: (32, 1, 1), outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
    }

    /// silu(x * scale), elementwise.
    static let scaledSilu: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "qwen4exp_hc_scaled_silu_scratch", inputNames: ["x", "scale"], outputNames: ["y"],
        source: """
                auto i = thread_position_in_grid.x;
                float v = static_cast<float>(x[i]) * scale[0];
                y[i] = static_cast<InT>(v / (1.0f + metal::exp(-v)));
            """)
    static func scaledSilu(_ x: MLXArray, scale: Float) -> MLXArray {
        scaledSilu([x, MLXArray([scale])], template: [("InT", x.dtype)], grid: (x.size, 1, 1),
            threadGroup: (min(x.size, 256), 1, 1), outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
    }

    /// mean over HC of sigmoid(w) * normed, one thread per (row, d).
    static let mix: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "qwen4exp_hc_mix_scratch", inputNames: ["w", "normed"], outputNames: ["y"],
        source: """
                auto d = thread_position_in_grid.x;
                auto row = thread_position_in_grid.y;
                if (d >= D) return;
                float acc = 0.0f;
                for (int g = 0; g < HC; ++g) {
                    auto i = (row * HC + g) * D + d;
                    float wv = static_cast<float>(w[i]);
                    acc += (1.0f / (1.0f + metal::exp(-wv))) * static_cast<float>(normed[i]);
                }
                y[row * D + d] = static_cast<InT>(acc / float(HC));
            """)
    static func mix(_ w: MLXArray, normed: MLXArray, hc: Int, dims: Int) -> MLXArray {
        let rows = w.size / (hc * dims)
        let lead = Array(w.shape.dropLast())
        return mix([w, normed], template: [("InT", w.dtype), ("D", dims), ("HC", hc)],
            grid: (dims, rows, 1), threadGroup: (min(dims, 256), 1, 1),
            outputShapes: [lead + [dims]], outputDTypes: [w.dtype])[0]
    }

    /// residual + output ⊗ (2·sigmoid(injectLogit * scale)), one thread per (row, g, d).
    static let inject: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "qwen4exp_hc_inject_scratch", inputNames: ["residual", "output", "logit", "scale"], outputNames: ["y"],
        source: """
                auto d = thread_position_in_grid.x;
                auto g = thread_position_in_grid.y;
                auto row = thread_position_in_grid.z;
                if (d >= D) return;
                float l = static_cast<float>(logit[row * HC + g]) * scale[0];
                float inj = 2.0f / (1.0f + metal::exp(-l));
                auto i = (row * HC + g) * D + d;
                y[i] = static_cast<InT>(static_cast<float>(residual[i]) + static_cast<float>(output[row * D + d]) * inj);
            """)
    static func inject(residual: MLXArray, output: MLXArray, logit: MLXArray, scale: Float, hc: Int, dims: Int) -> MLXArray {
        let rows = residual.size / (hc * dims)
        return inject([residual, output, logit, MLXArray([scale])],
            template: [("InT", residual.dtype), ("D", dims), ("HC", hc)],
            grid: (dims, hc, rows), threadGroup: (min(dims, 256), 1, 1),
            outputShapes: [residual.shape], outputDTypes: [residual.dtype])[0]
    }
}
