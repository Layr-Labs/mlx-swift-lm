// Copyright 2026 the HuggingFace Team. All rights reserved.
// Modified by Eigen Labs in 2026: native Swift/MLX projected-tensor primitives.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import MLX
import MLXNN

/// Unfused GLM5Next mHC primitives, not a registered model implementation.
///
/// Equations and dtype boundaries: `Glm5NextTextHyperConnection.forward` and
/// `Glm5NextTextDecoderLayer.forward` in the official model card's linked reference:
/// https://github.com/huggingface/transformers/blob/e0eb2810ac1b2caf6d082f247d8d37162ecf67a0/src/transformers/models/glm5_next/modeling_glm5_next.py
/// Model/config pin: zai-org/GLM-5.3-Flash@91b69215804f2d2036a8867f6ce44848e278dcda.
enum GLM5NextHyperConnection {

    /// Collapse `[B, S, N, D]` streams to a `[B, S, D]` sublayer input.
    /// `fn` is `[2*N + N*N, N*D]`, `base` is `[2*N + N*N]`, `scale` is `[3]`.
    /// The coefficients remain float32; only the collapsed input is cast back.
    static func collapse(
        _ hiddenStreams: MLXArray,
        fn: MLXArray,
        base: MLXArray,
        scale: MLXArray,
        rmsNormEpsilon: Float,
        epsilon: Float,
        sinkhornIterations: Int
    ) -> (post: MLXArray, comb: MLXArray, collapsed: MLXArray) {
        precondition(hiddenStreams.ndim == 4 && hiddenStreams.dtype.isFloatingPoint)
        precondition(rmsNormEpsilon.isFinite && rmsNormEpsilon > 0)
        precondition(epsilon.isFinite && epsilon > 0 && sinkhornIterations >= 1)
        let batch = hiddenStreams.dim(0)
        let length = hiddenStreams.dim(1)
        let streams = hiddenStreams.dim(2)
        let width = hiddenStreams.dim(3)
        precondition(streams > 0 && width > 0)
        let coefficients = (2 + streams) * streams
        precondition(fn.shape == [coefficients, streams * width])
        precondition(base.shape == [coefficients] && scale.shape == [3])
        precondition(fn.dtype.isFloatingPoint && base.dtype.isFloatingPoint)
        precondition(scale.dtype.isFloatingPoint)

        // RMS is over all N*D channels, using rms_norm_eps, not hc_eps.
        // Normalize before projecting, retaining the reference's float32 boundary.
        let flattened = hiddenStreams.reshaped([batch, length, streams * width]).asType(.float32)
        let normalized =
            flattened * rsqrt(flattened.square().mean(axis: -1, keepDims: true) + rmsNormEpsilon)
        let projected = matmul(normalized, fn.asType(.float32).T)
        let base = base.asType(.float32)
        let scale = scale.asType(.float32)

        // Pre is deliberately not normalized. Post has neither epsilon nor a sum normalization.
        let pre =
            sigmoid(projected[.ellipsis, ..<streams] * scale[0] + base[..<streams]) + epsilon
        let post =
            2
            * sigmoid(
                projected[.ellipsis, streams ..< 2 * streams] * scale[1]
                    + base[streams ..< 2 * streams])
        let logits =
            projected[.ellipsis, (2 * streams)...].reshaped([batch, length, streams, streams])
            * scale[2] + base[(2 * streams)...].reshaped([streams, streams])

        // Row softmax + epsilon, then one column normalization; only the remaining
        // iterations perform row/column pairs. Epsilon belongs in every denominator.
        var comb = softmax(logits, axis: -1, precise: true) + epsilon
        comb = comb / (comb.sum(axis: -2, keepDims: true) + epsilon)
        for _ in 1 ..< sinkhornIterations {
            comb = comb / (comb.sum(axis: -1, keepDims: true) + epsilon)
            comb = comb / (comb.sum(axis: -2, keepDims: true) + epsilon)
        }

        let collapsed = (pre.expandedDimensions(axis: -1) * hiddenStreams.asType(.float32))
            .sum(axis: 2).asType(hiddenStreams.dtype)
        return (post, comb, collapsed)
    }

    /// Expand a `[B, S, D]` sublayer output and mix its original residual streams.
    /// Matches `post[..., None] * output[..., None, :] + comb.T @ residual`.
    static func expand(
        _ output: MLXArray,
        residual: MLXArray,
        post: MLXArray,
        comb: MLXArray
    ) -> MLXArray {
        precondition(residual.ndim == 4 && residual.dtype.isFloatingPoint)
        let batch = residual.dim(0)
        let length = residual.dim(1)
        let streams = residual.dim(2)
        let width = residual.dim(3)
        precondition(output.shape == [batch, length, width] && output.dtype == residual.dtype)
        precondition(post.shape == [batch, length, streams])
        precondition(comb.shape == [batch, length, streams, streams])

        // Unlike collapse, the reference casts both coefficients BEFORE the
        // sublayer-output product and residual matmul, not just their final sum.
        let injected =
            post.asType(residual.dtype).expandedDimensions(axis: -1)
            * output.expandedDimensions(axis: -2)
        let mixed = matmul(comb.asType(residual.dtype).swappedAxes(-1, -2), residual)
        return injected + mixed
    }
}
