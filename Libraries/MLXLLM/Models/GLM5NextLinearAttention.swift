// Copyright 2026 the HuggingFace Team. All rights reserved.
// Copyright 2026 Eigen Labs.
// Modified in 2026: native Swift/MLX recurrence with explicit functional state.
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

import Foundation
import MLX
import MLXNN

/// The projected-tensor KDA primitive, not a model or a serving adapter.
///
/// Reference: `Glm5NextTextLinearAttention`, `Glm5NextTextForgetGate`,
/// `recurrent_kimi_delta_attention`, `causal_conv1d_update`, and
/// `Glm5NextTextRMSNormGated` in:
/// https://github.com/huggingface/transformers/blob/f0705f991a71f05dad9569336b181119f5546f93/src/transformers/models/glm5_next/modeling_glm5_next.py
/// Artifact configuration:
/// https://huggingface.co/zai-org/GLM-5.3-Flash/blob/91b69215804f2d2036a8867f6ce44848e278dcda/config.json
///
/// Learned projections remain caller-owned: Q/K/V, f_b(f_a(x)), b(x), and
/// g_b(g_a(x)) are inputs; the result precedes o_proj. Any padding must be
/// zeroed on x BEFORE those bias-free projections, as in the reference. This
/// primitive neither skips padded timesteps nor owns mutable request history.
enum GLM5NextLinearAttention {
    /// An explicit per-call choice, never an environment flag or automatic
    /// fallback. Unsupported forced-native calls throw the kernel's reason.
    enum ExecutionPolicy: Sendable, Equatable {
        case reference
        case nativeRequired
    }

    struct State {
        /// Raw, pre-convolution Q/K/V history, [B, 3*H*D, K], oldest first.
        /// Keep K entries, matching LinearAttentionLayer.update_conv_state;
        /// the oldest one drops out when the next input arrives.
        let convolution: MLXArray
        /// FP32 KDA state [B, H, D_key, D_value], NOT the GDN [B,H,V,K] layout.
        let recurrent: MLXArray
    }

    struct Weights {
        /// Depthwise cross-correlation weights [3*H*D, K], in Q, K, V order.
        /// Kernel index zero multiplies the oldest entry; there is no bias.
        let convolution: MLXArray
        let aLog: MLXArray  // [H]
        let dtBias: MLXArray  // [H*D]
        let outputNorm: MLXArray  // [D]
    }

    struct Output {
        /// Gated RMS-normalized heads [B, T, H, D], before the output projection.
        let values: MLXArray
        let state: State
        let execution: ExecutionPolicy
    }

    enum Failure: Error, Equatable {
        case invalidGeometry
        case invalidTensor(String)
    }

    /// Reference recurrence by default, with an explicit opt-in native candidate.
    /// Neither path is the reference's differently associated chunk-matrix
    /// optimization. Every operation is lazy and the input state is read-only.
    static func apply(
        query: MLXArray, key: MLXArray, value: MLXArray,
        forgetLogits: MLXArray, betaLogits: MLXArray, outputGate: MLXArray,
        weights: Weights, state: State? = nil,
        normEpsilon: Float = 1e-5, gateLowerBound: Float? = -5,
        execution: ExecutionPolicy = .reference
    ) throws -> Output {
        guard query.ndim == 4, query.shape.allSatisfy({ $0 > 0 }),
            normEpsilon.isFinite, normEpsilon > 0,
            gateLowerBound.map({ $0.isFinite && $0 < 0 }) ?? true
        else { throw Failure.invalidGeometry }
        let batch = query.dim(0)
        let length = query.dim(1)
        let heads = query.dim(2)
        let dimension = query.dim(3)
        let (width, widthOverflow) = heads.multipliedReportingOverflow(by: dimension)
        let (channels, channelsOverflow) = width.multipliedReportingOverflow(by: 3)
        guard !widthOverflow, !channelsOverflow else { throw Failure.invalidGeometry }
        for (name, array) in [
            ("query", query), ("key", key), ("value", value),
            ("forgetLogits", forgetLogits), ("outputGate", outputGate),
        ] {
            try validate(array, shape: query.shape, name: name, dtype: query.dtype)
        }
        try validate(
            betaLogits, shape: [batch, length, heads], name: "betaLogits", dtype: query.dtype)
        guard weights.convolution.ndim == 2, weights.convolution.dim(1) > 0 else {
            throw Failure.invalidTensor("convolution weights")
        }
        let kernel = weights.convolution.dim(1)
        try validate(weights.convolution, shape: [channels, kernel], name: "convolution weights")
        try validate(weights.aLog, shape: [heads], name: "A_log")
        try validate(weights.dtBias, shape: [width], name: "dt_bias")
        try validate(weights.outputNorm, shape: [dimension], name: "output norm")
        if let state {
            try validate(
                state.convolution, shape: [batch, channels, kernel],
                name: "convolution state", dtype: query.dtype)
            try validate(
                state.recurrent, shape: [batch, heads, dimension, dimension],
                name: "recurrent state", dtype: .float32)
        }
        if execution == .nativeRequired {
            // Refuse an unsupported forced path before preprocessing allocates
            // its graph; never silently run a different arithmetic implementation.
            _ = try GLM5NextLinearKernel.Layout(
                queryShape: query.shape, valueShape: value.shape, capturePrefixCount: 0)
            try GLM5NextLinearKernel.requireDefaultGPUStream()
        }

        let raw = concatenated(
            [query, key, value].map { $0.reshaped(batch, length, width) }, axis: -1)
        let previous =
            state?.convolution.transposed(0, 2, 1)
            ?? MLXArray.zeros([batch, kernel, channels], dtype: query.dtype)
        let history = concatenated([previous, raw], axis: 1)
        // [out_channel, kernel, 1] with groups == channels is exactly the
        // reference's independent temporal cross-correlation for each channel.
        // Both input and weight are FP32; activation precedes the cast back.
        let convolution = MLX.conv1d(
            history[0..., 1..., 0...].asType(.float32),
            weights.convolution.asType(.float32).reshaped(channels, kernel, 1),
            groups: channels)
        let convolved = silu(convolution).asType(query.dtype)
        let parts = split(convolved, indices: [width, 2 * width], axis: -1)
            .map { $0.reshaped(batch, length, heads, dimension).asType(.float32) }
        // FLA's reference deliberately uses division by sqrt, not rsqrt or
        // Qwen's RMS scaling. Epsilon is added to the sum, not used as a clamp.
        let q =
            parts[0] / sqrt(parts[0].square().sum(axis: -1, keepDims: true) + 1e-6)
            * (1 / Foundation.sqrt(Float(dimension)))
        let k = parts[1] / sqrt(parts[1].square().sum(axis: -1, keepDims: true) + 1e-6)
        let v = parts[2]
        let gateInput =
            forgetLogits.asType(.float32)
            + weights.dtBias.asType(.float32).reshaped(1, 1, heads, dimension)
        let rate = exp(weights.aLog.asType(.float32)).reshaped(1, 1, heads, 1)
        let logDecay: MLXArray
        if let gateLowerBound {
            logDecay = gateLowerBound * sigmoid(rate * gateInput)
        } else {
            // Preserve the reference's threshold and operation ordering.
            let softplus = MLX.where(gateInput .> 20, gateInput, log(1 + exp(gateInput)))
            logDecay = -rate * softplus
        }
        let beta = betaGate(betaLogits).asType(.float32)
        let recurrent: MLXArray
        let recurrenceOutput: MLXArray
        switch execution {
        case .reference:
            var current =
                state?.recurrent
                ?? MLXArray.zeros([batch, heads, dimension, dimension], dtype: .float32)
            var outputs: [MLXArray] = []
            outputs.reserveCapacity(length)
            for position in 0 ..< length {
                let qt = q[0..., position, 0..., 0...]
                let kt = k[0..., position, 0..., 0...]
                let vt = v[0..., position, 0..., 0...]
                // Per-KEY-channel decay: GDN's scalar-per-head kernel is not valid.
                current = current * exp(logDecay[0..., position, 0..., 0...])[.ellipsis, .newAxis]
                let memory = (current * kt[.ellipsis, .newAxis]).sum(axis: -2)
                let delta = (vt - memory) * beta[0..., position, 0...][.ellipsis, .newAxis]
                current = current + kt[.ellipsis, .newAxis] * delta[0..., 0..., .newAxis, 0...]
                outputs.append((current * qt[.ellipsis, .newAxis]).sum(axis: -2))
            }
            recurrent = current
            recurrenceOutput = stacked(outputs, axis: 1)
        case .nativeRequired:
            let native = try GLM5NextLinearKernel.apply(
                query: q, key: k, value: v, logDecay: logDecay, beta: beta,
                state: state?.recurrent)
            recurrent = native.state
            recurrenceOutput = native.values
        }
        // The recurrent output crosses the activation-dtype boundary before
        // the norm re-enters FP32. Its weights and sigmoid gate stay FP32.
        let core = recurrenceOutput.asType(query.dtype).asType(.float32)
        let normalized = core * rsqrt(core.square().mean(axis: -1, keepDims: true) + normEpsilon)
        let result =
            (normalized * weights.outputNorm.asType(.float32)
            * sigmoid(outputGate.asType(.float32))).asType(query.dtype)
        let nextConvolution = history[0..., (history.dim(1) - kernel)..., 0...].transposed(0, 2, 1)
        return Output(
            values: result,
            state: State(convolution: nextConvolution, recurrent: recurrent), execution: execution)
    }

    /// PyTorch's reduced-precision sigmoid computes in FP32 and rounds its
    /// result once. Keep that projection-dtype boundary before KDA's FP32 cast;
    /// native low-precision MLX sigmoid is not equivalent on the CPU path.
    static func betaGate(_ logits: MLXArray) -> MLXArray {
        sigmoid(logits.asType(.float32)).asType(logits.dtype)
    }

    private static func validate(
        _ array: MLXArray, shape: [Int], name: String, dtype: DType? = nil
    ) throws {
        guard array.shape == shape,
            [.float16, .bfloat16, .float32].contains(array.dtype),
            dtype == nil || array.dtype == dtype
        else { throw Failure.invalidTensor(name) }
    }
}
