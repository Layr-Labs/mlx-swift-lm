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
/// zeroed on x BEFORE those bias-free projections, as in the reference. Without
/// `validity` this primitive does not skip padded timesteps; it never owns
/// mutable request history.
enum GLM5NextLinearAttention {
    /// An explicit per-call choice, never an environment flag or automatic
    /// fallback. Unsupported forced-native calls throw the kernel's reason.
    enum ExecutionPolicy: Sendable, Equatable {
        case reference
        case nativeRequired
        /// `GLM5NextLinearChunkedRecurrence` in FP32 MLX operations, on any
        /// device. Requires a gate lower bound that admits the chunk size.
        case chunked(size: Int)
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
        /// With `capture`, entry t is the state after window token t (an
        /// invalid token repeats its predecessor); the last entry is `state`.
        var captured: [State] = []
    }

    enum Failure: Error, Equatable {
        case invalidGeometry
        case invalidTensor(String)
        case unsupportedChunk
        /// The chunked recurrence materializes only chunk-boundary states.
        case unsupportedCapture
    }

    /// Reference recurrence by default, with explicit native and chunked
    /// alternatives. Every operation is lazy and the input state is read-only.
    ///
    /// `validity` [B,T] makes a whole step equivalent to applying each valid
    /// token in order and skipping the others: an invalid token leaves the
    /// convolution history and recurrent state unchanged, a later valid token
    /// convolves only earlier valid inputs, and invalid positions return zero.
    ///
    /// `capture` additionally returns every per-position state (MTP
    /// capture-verify). The reference loop and the native kernel produce them
    /// within the same recurrence; the chunked recurrence refuses explicitly.
    static func apply(
        query: MLXArray, key: MLXArray, value: MLXArray,
        forgetLogits: MLXArray, betaLogits: MLXArray, outputGate: MLXArray,
        weights: Weights, state: State? = nil, validity: MLXArray? = nil,
        normEpsilon: Float = 1e-5, gateLowerBound: Float? = -5,
        execution: ExecutionPolicy = .reference, capture: Bool = false
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
        if let validity, validity.shape != [batch, length] || validity.dtype != .bool {
            throw Failure.invalidTensor("validity")
        }
        switch execution {
        case .reference: break
        case .nativeRequired:
            // Refuse an unsupported forced path before preprocessing allocates
            // its graph; never silently run a different arithmetic implementation.
            _ = try GLM5NextLinearKernel.Layout(
                queryShape: query.shape, valueShape: value.shape,
                capturePrefixCount: capture ? length : 0)
            try GLM5NextLinearKernel.requireDefaultGPUStream()
        case .chunked(let size):
            guard !capture else { throw Failure.unsupportedCapture }
            guard
                GLM5NextLinearChunkedRecurrence.supports(
                    chunkSize: size, gateLowerBound: gateLowerBound)
            else { throw Failure.unsupportedChunk }
        }

        let raw = concatenated(
            [query, key, value].map { $0.reshaped(batch, length, width) }, axis: -1)
        let previous =
            state?.convolution.transposed(0, 2, 1)
            ?? MLXArray.zeros([batch, kernel, channels], dtype: query.dtype)
        // One token needs no reordering; its retained window still uses validity.
        let order = length > 1 ? validity.map { validFirstOrder($0) } : nil
        let history = concatenated(
            [
                previous,
                order.map { takeAlong(raw, $0.order[.ellipsis, .newAxis], axis: 1) } ?? raw,
            ],
            axis: 1)
        // [out_channel, kernel, 1] with groups == channels is exactly the
        // reference's independent temporal cross-correlation for each channel.
        // Both input and weight are FP32; activation precedes the cast back.
        let convolution = MLX.conv1d(
            history[0..., 1..., 0...].asType(.float32),
            weights.convolution.asType(.float32).reshaped(channels, kernel, 1),
            groups: channels)
        let activated = silu(convolution).asType(query.dtype)
        let convolved =
            order.map { takeAlong(activated, $0.restore[.ellipsis, .newAxis], axis: 1) }
            ?? activated
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
        var logDecay: MLXArray
        if let gateLowerBound {
            logDecay = gateLowerBound * sigmoid(rate * gateInput)
        } else {
            // Preserve the reference's threshold and operation ordering.
            let softplus = MLX.where(gateInput .> 20, gateInput, log(1 + exp(gateInput)))
            logDecay = -rate * softplus
        }
        var beta = betaGate(betaLogits).asType(.float32)
        if let validity {
            // exp(0) = 1 and a zero beta make an invalid step the exact identity.
            logDecay = MLX.where(validity.reshaped(batch, length, 1, 1), logDecay, 0)
            beta = MLX.where(validity.reshaped(batch, length, 1), beta, 0)
        }
        let recurrent: MLXArray
        let recurrenceOutput: MLXArray
        var capturedRecurrent: [MLXArray] = []
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
                if capture { capturedRecurrent.append(current) }
            }
            recurrent = current
            recurrenceOutput = stacked(outputs, axis: 1)
        case .nativeRequired:
            let native = try GLM5NextLinearKernel.apply(
                query: q, key: k, value: v, logDecay: logDecay, beta: beta,
                state: state?.recurrent, capturePrefixCount: capture ? length : 0)
            recurrent = native.state
            recurrenceOutput = native.values
            if capture {
                // Entry t is the kernel's state after token t; the full window
                // returns its own final-state output, not a captured slice.
                capturedRecurrent =
                    try (1 ..< length).map { try native.state(acceptingPrefix: $0) } + [recurrent]
            }
        case .chunked(let size):
            let chunked = GLM5NextLinearChunkedRecurrence.apply(
                query: q, key: k, value: v, logDecay: logDecay, beta: beta,
                state: state?.recurrent
                    ?? MLXArray.zeros([batch, heads, dimension, dimension], dtype: .float32),
                chunkSize: size)
            recurrent = chunked.state
            recurrenceOutput = chunked.values
        }
        // The recurrent output crosses the activation-dtype boundary before
        // the norm re-enters FP32. Its weights and sigmoid gate stay FP32.
        let core = recurrenceOutput.asType(query.dtype).asType(.float32)
        let normalized = core * rsqrt(core.square().mean(axis: -1, keepDims: true) + normEpsilon)
        let gated =
            (normalized * weights.outputNorm.asType(.float32)
            * sigmoid(outputGate.asType(.float32))).asType(query.dtype)
        let result =
            validity.map { MLX.where($0.reshaped(batch, length, 1, 1), gated, 0) } ?? gated
        let retained: MLXArray
        if let validity {
            // The K entries after the row's valid count in [previous, compacted].
            let starts = validity.asType(.int32).sum(axis: 1, keepDims: true)
            let window = starts + MLXArray((0 ..< kernel).map { Int32($0) }).reshaped(1, kernel)
            retained = takeAlong(history, window[.ellipsis, .newAxis], axis: 1)
        } else {
            retained = history[0..., (history.dim(1) - kernel)..., 0...]
        }
        let nextConvolution = retained.transposed(0, 2, 1)
        var captured: [State] = []
        if capture {
            // After token t the retained window starts at the count of valid
            // tokens through t, exactly as the final window above.
            let ends =
                validity.map { $0.asType(.int32).cumsum(axis: 1) }
                ?? MLXArray((1 ... length).map { Int32($0) }).reshaped(1, length)
            let offsets = MLXArray((0 ..< kernel).map { Int32($0) }).reshaped(1, kernel)
            captured = (0 ..< length).map { position in
                let window = ends[0..., position ..< position + 1] + offsets
                let convolution = takeAlong(
                    history, broadcast(window, to: [batch, kernel])[.ellipsis, .newAxis], axis: 1
                ).transposed(0, 2, 1)
                return State(
                    convolution: position == length - 1 ? nextConvolution : convolution,
                    recurrent: capturedRecurrent[position])
            }
        }
        return Output(
            values: result,
            state: State(convolution: nextConvolution, recurrent: recurrent), execution: execution,
            captured: captured)
    }

    /// A stable permutation placing each row's valid tokens first, in order,
    /// and its inverse.
    private static func validFirstOrder(_ validity: MLXArray) -> (
        order: MLXArray, restore: MLXArray
    ) {
        let length = validity.dim(1)
        let positions = MLXArray((0 ..< length).map { Int32($0) }).reshaped(1, length)
        let keys = MLX.where(validity, positions, positions + Int32(length))
        let order = argSort(keys, axis: 1)
        return (order, argSort(order, axis: 1))
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
