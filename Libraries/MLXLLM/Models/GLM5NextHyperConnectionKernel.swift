// Copyright 2026 the HuggingFace Team. All rights reserved.
// Copyright 2026 Eigen Labs.
// Modified in 2026: opt-in FP32 coefficient/Sinkhorn Metal candidate. Trained
// projection, RMS normalization, weighted collapse and expansion stay outside.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import MLX

/// Unactivated candidate for the coefficient stage of four-stream mHC only.
///
/// Reference: `Glm5NextTextHyperConnection.forward` at
/// https://github.com/huggingface/transformers/blob/f0705f991a71f05dad9569336b181119f5546f93/src/transformers/models/glm5_next/modeling_glm5_next.py
/// The caller supplies `projected = matmul(RMSNorm(flatten(hidden)), fn.T)` in
/// FP32, with the SAME normalization-before-projection order as the baseline.
/// Moving the RMS factor across that dot product is not permitted by this API.
///
/// For p = projected, the output is:
///   pre  = sigmoid(p[...,0:4] * scale[0] + base[0:4]) + epsilon
///   post = 2 * sigmoid(p[...,4:8] * scale[1] + base[4:8])
///   C = rowSoftmax(reshape(p[...,8:24], 4, 4) * scale[2] + base[8:24]) + epsilon
///   C /= columnSum(C) + epsilon
///   repeat (iterations - 1): C /= rowSum(C) + epsilon; C /= columnSum(C) + epsilon
/// Every affine multiply/add and epsilon addition retains its FP32 boundary.
/// MLX math reference: mlx-swift@6923a80f624f5c91fbf456efe4e00e9698a72961,
/// nested MLX `unary_ops.h` (Sigmoid) and `softmax.h` (FP32 softmax). Softmax
/// uses its existing fast exponential, not a new reduced-precision variant.
///
/// `pre` must multiply FP32 hidden streams BEFORE their stream sum and final
/// activation-dtype cast. `post` and `comb` remain FP32 until the baseline expand
/// casts them BEFORE its output product and residual matmul. This kernel does
/// not fuse those cancellation-sensitive operations or perform any such casts.
enum GLM5NextHyperConnectionKernel {
    enum Failure: Error, Equatable {
        case invalidGeometry(String)
        case invalidTensor(String)
        case unsupportedDeviceOrStream
    }

    struct Coefficients {
        let pre: MLXArray  // FP32 [B,T,4], unnormalized
        let post: MLXArray  // FP32 [B,T,4], no epsilon or normalization
        let comb: MLXArray  // FP32 [B,T,4,4], source row / destination column

        var evaluationTargets: [MLXArray] { [pre, post, comb] }
    }

    /// Only metadata is inspected. No allocation, evaluation or data readback.
    struct Layout {
        let batch: Int
        let tokens: Int
        let rows: Int

        init(
            projectedShape: [Int], hcMult: Int, epsilon: Float, sinkhornIterations: Int
        ) throws {
            guard hcMult == 4 else {
                throw Failure.invalidGeometry("this candidate supports hcMult == 4 only")
            }
            guard projectedShape.count == 3, projectedShape[2] == 24 else {
                throw Failure.invalidGeometry("projected must be [B,T,24]")
            }
            guard epsilon.isFinite, epsilon > 0, (1 ... 128).contains(sinkhornIterations) else {
                throw Failure.invalidGeometry(
                    "epsilon must be positive and finite; iterations must be 1...128")
            }
            var elements = 1
            for dimension in projectedShape {
                let (next, overflow) = elements.multipliedReportingOverflow(by: dimension)
                guard dimension > 0, dimension <= Int(Int32.max), !overflow,
                    next <= Int(Int32.max)
                else { throw Failure.invalidGeometry("nonempty Int32-indexed tensors required") }
                elements = next
            }
            batch = projectedShape[0]
            tokens = projectedShape[1]
            rows = elements / 24
        }
    }

    /// All three inputs must already be FP32. In particular, widening an FP16
    /// or BF16 base/scale is caller-owned, as in GLM5NextHyperConnection.collapse.
    /// Managed MLX outputs are newly allocated; input buffers are read-only and
    /// retained by the lazy graph. Strided inputs use MLXFast's contiguous copy.
    static func apply(
        projected: MLXArray, base: MLXArray, scale: MLXArray,
        hcMult: Int = 4, epsilon: Float, sinkhornIterations: Int
    ) throws -> Coefficients {
        let layout = try Layout(
            projectedShape: projected.shape, hcMult: hcMult, epsilon: epsilon,
            sinkhornIterations: sinkhornIterations)
        for (name, array, shape) in [
            ("projected", projected, projected.shape), ("base", base, [24]), ("scale", scale, [3]),
        ] {
            guard array.dtype == .float32, array.shape == shape else {
                throw Failure.invalidTensor("\(name) must be FP32 \(shape)")
            }
        }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            guard Device.defaultDevice().deviceType == .gpu,
                StreamOrDevice.default.stream == Stream.gpu
            else { throw Failure.unsupportedDeviceOrStream }
            let outputs = kernel(
                [projected, base, scale, MLXArray(epsilon)],
                template: [("Iterations", sinkhornIterations)],
                grid: (32, layout.rows, 1), threadGroup: (32, 1, 1),
                outputShapes: [
                    [layout.batch, layout.tokens, 4], [layout.batch, layout.tokens, 4],
                    [layout.batch, layout.tokens, 4, 4],
                ],
                outputDTypes: [.float32, .float32, .float32])
            return Coefficients(pre: outputs[0], post: outputs[1], comb: outputs[2])
        #else
            throw Failure.unsupportedDeviceOrStream
        #endif
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        // One 32-thread group owns one token. Threads 0...3 share a SIMD group
        // and own matrix rows; all threads enter the shuffles with defined data.
        private static let kernel = MLXFast.metalKernel(
            name: "glm5_next_hc4_coefficients_fp32",
            inputNames: ["projected", "base", "scale", "epsilon"],
            outputNames: ["pre", "post", "comb"],
            source: """
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                const size_t token = thread_position_in_grid.y;
                const uint row = thread_position_in_threadgroup.x;
                float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                if (row < 4) {
                    const float pre_product = projected[token * 24 + row] * scale[0];
                    const float pre_logit = pre_product + base[row];
                    const float post_product = projected[token * 24 + 4 + row] * scale[1];
                    const float post_logit = post_product + base[4 + row];
                    pre[token * 4 + row] = glm5_hc_sigmoid(pre_logit) + epsilon;
                    post[token * 4 + row] = 2.0f * glm5_hc_sigmoid(post_logit);

                    float logits[4];
                    float maximum = -INFINITY;
                    for (int j = 0; j < 4; ++j) {
                        const float product = projected[token * 24 + 8 + row * 4 + j] * scale[2];
                        logits[j] = product + base[8 + row * 4 + j];
                        maximum = metal::max(maximum, logits[j]);
                    }
                    float normalizer = 0.0f;
                    for (int j = 0; j < 4; ++j) {
                        // Matches MLX softmax.h, including precise:true FP32 accumulation.
                        c[j] = metal::fast::exp(logits[j] - maximum);
                        normalizer += c[j];
                    }
                    const float inverse = 1.0f / normalizer;
                    for (int j = 0; j < 4; ++j) { c[j] = c[j] * inverse + epsilon; }
                }

                // The first iteration is column-only; epsilon is AFTER each sum.
                for (int j = 0; j < 4; ++j) {
                    const float denominator = glm5_hc_column_sum(c[j]) + epsilon;
                    c[j] = c[j] / denominator;
                }
                for (int iteration = 1; iteration < Iterations; ++iteration) {
                    float row_sum = 0.0f;
                    for (int j = 0; j < 4; ++j) { row_sum += c[j]; }
                    const float row_denominator = row_sum + epsilon;
                    for (int j = 0; j < 4; ++j) { c[j] = c[j] / row_denominator; }
                    for (int j = 0; j < 4; ++j) {
                        const float denominator = glm5_hc_column_sum(c[j]) + epsilon;
                        c[j] = c[j] / denominator;
                    }
                }
                if (row < 4) {
                    for (int j = 0; j < 4; ++j) { comb[token * 16 + row * 4 + j] = c[j]; }
                }
                """,
            header: """
                inline float glm5_hc_sigmoid(float x) {
                    #pragma clang fp reassociate(off)
                    #pragma clang fp contract(off)
                    // MLX's FP32 unary Sigmoid, never a reduced-precision sigmoid.
                    const float tail = 1.0f / (1.0f + metal::exp(metal::abs(x)));
                    return x < 0.0f ? tail : 1.0f - tail;
                }
                inline float glm5_hc_column_sum(float x) {
                    #pragma clang fp reassociate(off)
                    #pragma clang fp contract(off)
                    float total = 0.0f;
                    total += simd_shuffle(x, ushort(0));
                    total += simd_shuffle(x, ushort(1));
                    total += simd_shuffle(x, ushort(2));
                    total += simd_shuffle(x, ushort(3));
                    return total;
                }
                """,
            ensureRowContiguous: true)
    #endif
}
