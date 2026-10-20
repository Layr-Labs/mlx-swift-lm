// Copyright 2026 the HuggingFace Team. All rights reserved.
// Copyright 2026 Eigen Labs.
// Modified in 2026: opt-in single-launch KDA decode/verify step.
// Structure of the in-kernel f_b/g_b rows follows oMLX's `_KDA_SOURCE`
// (github.com/jundot/omlx, Apache-2.0, tag 33e0b0c).
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

/// One Metal launch for a KDA step of 1...`maximumTokens` tokens, selected
/// only by `GLM5NextExecutionPolicy.LinearDecodeStep.fusedKernel`.
///
/// It computes the equations of `GLM5NextLinearAttention.apply` with a gate
/// lower bound, applying each valid token in order and skipping invalid ones:
/// causal convolution over the retained raw history, SiLU and the
/// activation-dtype boundary, query/key L2 norms, decay and beta gates, the
/// FP32 delta-rule recurrence, the gated output RMS norm and the retained
/// convolution window. Every rounding boundary is preserved; head reductions
/// use a fixed tree, so it is not bitwise. Optionally it also returns the
/// state after every token (MTP verification windows).
///
/// One threadgroup owns one (row, head). Recurrent-state accesses are
/// coalesced along the value channel.
enum GLM5NextLinearDecodeKernel {
    enum Failure: Error, Equatable {
        case invalidGeometry(String)
        case invalidTensor(String)
        case unsupportedDeviceOrStream
    }

    static let maximumTokens = 8
    static let singleTokenThreads = 512
    /// Threads of the low-rank or multi-token variants: four lanes per
    /// f_b/g_b row of a head, and one thread per (token, channel) at the end.
    static let wideThreads = 1024
    /// Input width of the in-kernel f_b/g_b rows (4-bit, group 64).
    static let lowRank = 128

    /// Steps this kernel runs: one token of head width up to 128, or several
    /// tokens (any count, in blocks of `maximumTokens`) of a head width that
    /// divides into SIMD groups.
    static func supports(tokens: Int, dimension: Int) -> Bool {
        guard dimension > 0, dimension <= 128, tokens >= 1 else { return false }
        return tokens == 1 || dimension % 32 == 0
    }

    /// The forget and output gates entering the step.
    enum Gates {
        /// f_b(f_a(x)) and g_b(g_a(x)), [B,T,H,D] in the activation dtype.
        case projected(forget: MLXArray, output: MLXArray)
        /// f_a(x) and g_a(x), [B,T,128]; the Q4 g64 f_b/g_b rows are applied
        /// in-kernel with the reference's activation-dtype rounding of each
        /// projected value (four lanes per row, then a quad sum).
        case lowRank(
            forgetInput: MLXArray, outputInput: MLXArray, forget: QuantizedLinear,
            output: QuantizedLinear)

        /// The in-kernel form when both rows are affine 4-bit group-64 over 128
        /// inputs; nil otherwise.
        static func inKernel(
            forgetInput: MLXArray, outputInput: MLXArray, forget: Linear, output: Linear
        ) -> Gates? {
            guard let forget = forget as? QuantizedLinear, let output = output as? QuantizedLinear,
                isInKernelReadable(forget), isInKernelReadable(output),
                forgetInput.dim(-1) == GLM5NextLinearDecodeKernel.lowRank,
                outputInput.dim(-1) == GLM5NextLinearDecodeKernel.lowRank
            else { return nil }
            return .lowRank(
                forgetInput: forgetInput, outputInput: outputInput, forget: forget, output: output)
        }

        private static func isInKernelReadable(_ linear: QuantizedLinear) -> Bool {
            let inputs: Int = linear.shape.1
            return linear.mode == .affine && linear.bits == 4 && linear.groupSize == 64
                && linear.bias == nil && linear.biases != nil
                && inputs == GLM5NextLinearDecodeKernel.lowRank
                && linear.scales.dtype.isFloatingPoint
        }
    }

    struct Inputs {
        /// Raw projections [B,T,H,D] (q, k, v) and [B,T,H] beta.
        let query: MLXArray
        let key: MLXArray
        let value: MLXArray
        let gates: Gates
        let betaLogits: MLXArray
        /// Checkpoint depthwise weights, [H*D,1,K] each, oldest tap first.
        let queryConvolution: MLXArray
        let keyConvolution: MLXArray
        let valueConvolution: MLXArray
        let aLog: MLXArray
        let dtBias: MLXArray
        let outputNorm: MLXArray
        /// Convolution history in the activation dtype, or FP32 holding
        /// activation-dtype values (the native CBv2 storage); the retained
        /// window is returned in the same dtype.
        let state: GLM5NextLinearAttention.State?
        let validity: MLXArray
        /// The fused input projection that `query`, `key`, `value` and
        /// `betaLogits` are column views of; long steps read them in place
        /// instead of copying each view to contiguous storage.
        var projectionRows: ProjectionRows? = nil
    }

    /// `[B·T, N]` (or `[B, T, N]`) fused projection rows holding q, k and v
    /// in columns `[0, 3·H·D)`, in that order, and beta logits in columns
    /// `[betaColumn, betaColumn + H)`.
    struct ProjectionRows {
        let buffer: MLXArray
        let betaColumn: Int
    }

    struct Result {
        let output: GLM5NextLinearAttention.Output
        /// With capture: FP32 [B,T,H,D,D] recurrent state and [B,T,3HD,K]
        /// convolution window after each token (unchanged across invalid ones).
        let capturedRecurrent: MLXArray?
        let capturedConvolution: MLXArray?
    }

    static func apply(
        _ inputs: Inputs, normEpsilon: Float, gateLowerBound: Float
    ) throws -> GLM5NextLinearAttention.Output {
        try apply(inputs, normEpsilon: normEpsilon, gateLowerBound: gateLowerBound, capture: false)
            .output
    }

    static func apply(
        _ inputs: Inputs, normEpsilon: Float, gateLowerBound: Float, capture: Bool
    ) throws -> Result {
        let query = inputs.query
        guard query.ndim == 4, query.shape.allSatisfy({ $0 > 0 }) else {
            throw Failure.invalidGeometry("query must be [B,T,H,D]")
        }
        let batch = query.dim(0)
        let tokens = query.dim(1)
        let heads = query.dim(2)
        let dimension = query.dim(3)
        let width = heads * dimension
        guard supports(tokens: tokens, dimension: dimension) else {
            throw Failure.invalidGeometry("D <= 128, and D % 32 == 0 for T > 1")
        }
        // Captured states are written per window, at most one token block.
        guard !capture || tokens <= maximumTokens else {
            throw Failure.invalidGeometry("captured windows hold at most 8 tokens")
        }
        let block = min(tokens, maximumTokens)
        guard normEpsilon.isFinite, normEpsilon > 0, gateLowerBound.isFinite, gateLowerBound < 0
        else { throw Failure.invalidGeometry("epsilon and gate lower bound") }
        guard [.float16, .bfloat16, .float32].contains(query.dtype) else {
            throw Failure.invalidTensor("activation dtype")
        }
        for (name, array) in [("key", inputs.key), ("value", inputs.value)]
        where array.shape != query.shape || array.dtype != query.dtype {
            throw Failure.invalidTensor(name)
        }
        let gateInputs: [MLXArray]
        let lowRankGates: Bool
        switch inputs.gates {
        case .projected(let forget, let output):
            guard forget.shape == query.shape, forget.dtype == query.dtype,
                output.shape == query.shape, output.dtype == query.dtype
            else { throw Failure.invalidTensor("gates") }
            gateInputs = [forget, output]
            lowRankGates = false
        case .lowRank(let forgetInput, let outputInput, let forget, let output):
            guard forgetInput.shape == [batch, tokens, lowRank], forgetInput.dtype == query.dtype,
                outputInput.shape == [batch, tokens, lowRank], outputInput.dtype == query.dtype,
                forget.shape == (width, lowRank), output.shape == (width, lowRank),
                8 * dimension <= wideThreads
            else { throw Failure.invalidTensor("low-rank gates") }
            gateInputs = [
                forgetInput, outputInput, forget.weight, forget.scales, forget.biases!,
                output.weight, output.scales, output.biases!,
            ]
            lowRankGates = true
        }
        let threads = lowRankGates || tokens > 1 ? wideThreads : singleTokenThreads
        guard inputs.betaLogits.shape == [batch, tokens, heads],
            inputs.betaLogits.dtype == query.dtype
        else { throw Failure.invalidTensor("beta") }
        if let rows = inputs.projectionRows {
            let stride = rows.buffer.dim(-1)
            guard rows.buffer.dtype == query.dtype, rows.buffer.size == batch * tokens * stride,
                stride >= 3 * width, rows.betaColumn >= 0, rows.betaColumn + heads <= stride
            else { throw Failure.invalidTensor("projection rows") }
        }
        let kernel = inputs.queryConvolution.dim(-1)
        for (name, array) in [
            ("q_conv1d", inputs.queryConvolution), ("k_conv1d", inputs.keyConvolution),
            ("v_conv1d", inputs.valueConvolution),
        ] where array.shape != [width, 1, kernel] || !array.dtype.isFloatingPoint {
            throw Failure.invalidTensor(name)
        }
        guard kernel >= 1, kernel <= 8 else { throw Failure.invalidGeometry("convolution taps") }
        guard inputs.aLog.shape == [heads], inputs.dtBias.shape == [width],
            inputs.outputNorm.shape == [dimension]
        else { throw Failure.invalidTensor("A_log, dt_bias or output norm") }
        guard inputs.validity.shape == [batch, tokens], inputs.validity.dtype == .bool else {
            throw Failure.invalidTensor("validity")
        }
        let convolutionShape = [batch, 3 * width, kernel]
        let stateShape = [batch, heads, dimension, dimension]
        let convolutionType = inputs.state?.convolution.dtype ?? query.dtype
        if let state = inputs.state {
            guard state.convolution.shape == convolutionShape,
                convolutionType == query.dtype || convolutionType == .float32,
                state.recurrent.shape == stateShape, state.recurrent.dtype == .float32
            else { throw Failure.invalidTensor("state") }
        }
        // Indices are 64-bit; only grid extents and the captured stacks, which
        // scale with tokens x state, need Int32 bounds.
        let (rows, overflow) = (batch * heads).multipliedReportingOverflow(by: threads)
        guard !overflow, rows <= Int(Int32.max), batch * tokens <= Int(Int32.max),
            !capture
                || (batch * tokens * 3 * width * kernel <= Int(Int32.max)
                    && batch * tokens * heads * dimension * dimension <= Int(Int32.max))
        else { throw Failure.invalidGeometry("tensor exceeds Int32 indexing") }
        guard glm5NextOnDefaultGPUStream() else { throw Failure.unsupportedDeviceOrStream }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            let convolution =
                inputs.state?.convolution ?? MLXArray.zeros(convolutionShape, dtype: query.dtype)
            let recurrent =
                inputs.state?.recurrent ?? MLXArray.zeros(stateShape, dtype: .float32)
            // Same FP32 value as the composed path's `1 / Foundation.sqrt(Float(D))`.
            let queryScale = 1 / Foundation.sqrt(Float(dimension))
            if tokens > maximumTokens {
                let forget: MLXArray
                let outputGate: MLXArray
                switch inputs.gates {
                case .projected(let projectedForget, let projectedOutput):
                    (forget, outputGate) = (projectedForget, projectedOutput)
                case .lowRank(let forgetInput, let outputInput, let forgetRows, let outputRows):
                    // Long steps run f_b/g_b as quantized GEMMs (activation dtype).
                    forget = forgetRows(forgetInput).reshaped(query.shape)
                    outputGate = outputRows(outputInput).reshaped(query.shape)
                }
                return Result(
                    output: longStep(
                        inputs, forget: forget, outputGate: outputGate, convolution: convolution,
                        recurrent: recurrent, normEpsilon: normEpsilon,
                        gateLowerBound: gateLowerBound, queryScale: queryScale),
                    capturedRecurrent: nil, capturedConvolution: nil)
            }
            var shapes = [query.shape, convolutionShape, stateShape]
            var types = [query.dtype, convolutionType, DType.float32]
            if capture {
                shapes += [
                    [batch, tokens, heads, dimension, dimension],
                    [batch, tokens, 3 * width, kernel],
                ]
                types += [.float32, convolutionType]
            }
            let outputs = kernels[Variant(lowRank: lowRankGates, capture: capture)]!(
                [inputs.query, inputs.key, inputs.value] + gateInputs + [
                    inputs.betaLogits, inputs.queryConvolution,
                    inputs.keyConvolution, inputs.valueConvolution, inputs.aLog,
                    inputs.dtBias, inputs.outputNorm, convolution, recurrent, inputs.validity,
                    MLXArray([gateLowerBound, normEpsilon, queryScale, 1 / Float(dimension)]),
                    MLXArray(Int32(tokens)),
                ],
                template: [
                    ("T", query.dtype), ("C", convolutionType), ("H", heads), ("D", dimension),
                    ("K", kernel), ("TK", block), ("Threads", threads),
                ],
                grid: (threads * batch * heads, 1, 1),
                threadGroup: (threads, 1, 1),
                outputShapes: shapes, outputDTypes: types)
            return Result(
                output: GLM5NextLinearAttention.Output(
                    values: outputs[0],
                    state: .init(convolution: outputs[1], recurrent: outputs[2]),
                    execution: .nativeRequired),
                capturedRecurrent: capture ? outputs[3] : nil,
                capturedConvolution: capture ? outputs[4] : nil)
        #else
            throw Failure.unsupportedDeviceOrStream
        #endif
    }

    /// Value rows of one recurrence threadgroup in `longStep`, and its lanes per row.
    static let longStepRows = 32
    static let longStepLanes = 8
    /// Tokens staged per recurrence block in `longStep`.
    static let longStepBlock = 16
    /// Tokens per thread of `longStep`'s blocked convolution walk.
    static let longConvolutionBlock = 16

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        /// Steps longer than one token block: (1) one thread per (raw channel,
        /// `longConvolutionBlock` tokens) walks its tokens through the causal
        /// convolution from the window entering them; (2) one threadgroup per
        /// (row, head, 32 value rows) runs the recurrence with its state in
        /// registers, recomputing the head's norms, decay and beta per staged
        /// block; (3) the gated output norm per (token, head). Same equations,
        /// rounding boundaries and validity semantics as the single launch.
        /// With `projectionRows`, q/k/v and beta are read in place from the
        /// fused projection (row stride N) rather than from contiguous copies.
        private static func longStep(
            _ inputs: Inputs, forget: MLXArray, outputGate: MLXArray, convolution: MLXArray,
            recurrent: MLXArray, normEpsilon: Float, gateLowerBound: Float, queryScale: Float
        ) -> GLM5NextLinearAttention.Output {
            let walked = longConvolution(inputs, convolution: convolution)
            let recurrence = longRecurrence(
                inputs, activated: walked.activated, forget: forget, recurrent: recurrent,
                gateLowerBound: gateLowerBound, queryScale: queryScale)
            let values = longOutput(
                inputs, core: recurrence.core, outputGate: outputGate, normEpsilon: normEpsilon)
            return GLM5NextLinearAttention.Output(
                values: values,
                state: .init(convolution: walked.convolution, recurrent: recurrence.state),
                execution: .nativeRequired)
        }

        /// Stage 1: activated q/k/v `[B, T, 3W]` and the retained window.
        /// `blocked` (the default) splits every channel's walk into blocks of
        /// `longConvolutionBlock` tokens, one thread each; `false` runs the
        /// serial per-channel walk, the bitwise oracle of the blocked form.
        static func longConvolution(
            _ inputs: Inputs, convolution: MLXArray, blocked: Bool = true
        ) -> (activated: MLXArray, convolution: MLXArray) {
            let query = inputs.query
            let (batch, tokens, heads, dimension) = (
                query.dim(0), query.dim(1), query.dim(2), query.dim(3)
            )
            let width = heads * dimension
            let channels = batch * 3 * width
            let sources: [MLXArray]
            let layout: [Int32]
            if let rows = inputs.projectionRows {
                sources = [rows.buffer, rows.buffer, rows.buffer]
                layout = [Int32(rows.buffer.dim(-1)), 0, Int32(width), Int32(2 * width)]
            } else {
                sources = [inputs.query, inputs.key, inputs.value]
                layout = [Int32(width), 0, 0, 0]
            }
            var template: [(String, any KernelTemplateArg)] = [
                ("T", query.dtype), ("C", convolution.dtype), ("H", heads), ("D", dimension),
                ("K", inputs.queryConvolution.dim(-1)),
            ]
            if blocked { template.append(("TB", longConvolutionBlock)) }
            let blocks = blocked ? (tokens + longConvolutionBlock - 1) / longConvolutionBlock : 1
            let walked = (blocked ? blockedConvolutionKernel : longConvolutionKernel)(
                sources + [
                    inputs.queryConvolution, inputs.keyConvolution, inputs.valueConvolution,
                    convolution, inputs.validity, MLXArray(Int32(tokens)),
                    MLXArray(Int32(channels)), MLXArray(layout),
                ],
                template: template,
                grid: (channels, blocks, 1), threadGroup: (min(256, channels), 1, 1),
                outputShapes: [[batch, tokens, 3 * width], convolution.shape],
                outputDTypes: [query.dtype, convolution.dtype])
            return (walked[0], walked[1])
        }

        /// Stage 2: the recurrence's core outputs `[B, T, H, D]` and final state.
        /// On Neural-Accelerator GPUs, steps of at least
        /// `GLM5NextKDAPrefillKernel.minimumTokens` tokens of 128-wide heads run
        /// `GLM5NextKDAPrefillKernel` (one value-row range per GPU core), whose
        /// FP32 operations and their order are those of `valueSplitRecurrence`.
        static func longRecurrence(
            _ inputs: Inputs, activated: MLXArray, forget: MLXArray, recurrent: MLXArray,
            gateLowerBound: Float, queryScale: Float
        ) -> (core: MLXArray, state: MLXArray) {
            let query = inputs.query
            let beta = inputs.projectionRows?.buffer ?? inputs.betaLogits
            let betaLayout: [Int32] =
                inputs.projectionRows.map {
                    [Int32($0.buffer.dim(-1)), Int32($0.betaColumn)]
                } ?? [Int32(query.dim(2)), 0]
            if let launch = GLM5NextKDAPrefillKernel.launch(
                tokens: query.dim(1), heads: query.dim(2), dimension: query.dim(3))
            {
                return GLM5NextKDAPrefillKernel.recurrence(
                    activated: activated, forget: forget, betaLogits: beta,
                    betaLayout: betaLayout, aLog: inputs.aLog, dtBias: inputs.dtBias,
                    state: recurrent, validity: inputs.validity,
                    gateLowerBound: gateLowerBound, queryScale: queryScale, launch: launch)
            }
            return valueSplitRecurrence(
                activated: activated, forget: forget, betaLogits: beta, betaLayout: betaLayout,
                aLog: inputs.aLog, dtBias: inputs.dtBias, state: recurrent,
                validity: inputs.validity, gateLowerBound: gateLowerBound, queryScale: queryScale)
        }

        /// One threadgroup per (row, head, 32 value rows); each 16-token block
        /// stages the head's normalized q/k, decay and beta.
        static func valueSplitRecurrence(
            activated: MLXArray, forget: MLXArray, betaLogits: MLXArray, betaLayout: [Int32],
            aLog: MLXArray, dtBias: MLXArray, state: MLXArray, validity: MLXArray,
            gateLowerBound: Float, queryScale: Float
        ) -> (core: MLXArray, state: MLXArray) {
            let (batch, heads, dimension) = (state.dim(0), state.dim(1), state.dim(2))
            let tokens = activated.dim(1)
            let threads = longStepRows * longStepLanes
            let recurrence = longRecurrenceKernel(
                [
                    activated, forget, betaLogits, aLog, dtBias, state, validity,
                    MLXArray([gateLowerBound, queryScale]), MLXArray(Int32(tokens)),
                    MLXArray(betaLayout),
                ],
                template: [
                    ("T", activated.dtype), ("H", heads), ("D", dimension),
                    ("TB", longStepBlock), ("Rows", longStepRows), ("Lanes", longStepLanes),
                ],
                grid: (threads * batch * heads * (dimension / longStepRows), 1, 1),
                threadGroup: (threads, 1, 1),
                outputShapes: [[batch, tokens, heads, dimension], state.shape],
                outputDTypes: [activated.dtype, .float32])
            return (recurrence[0], recurrence[1])
        }

        /// Stage 3: the gated output norm.
        static func longOutput(
            _ inputs: Inputs, core: MLXArray, outputGate: MLXArray, normEpsilon: Float
        ) -> MLXArray {
            let query = inputs.query
            let (batch, tokens, heads, dimension) = (
                query.dim(0), query.dim(1), query.dim(2), query.dim(3)
            )
            let pairs = batch * tokens * heads
            return longOutputKernel(
                [
                    core, outputGate, inputs.outputNorm, inputs.validity,
                    MLXArray([normEpsilon, 1 / Float(dimension)]), MLXArray(Int32(pairs)),
                ],
                template: [("T", query.dtype), ("H", heads), ("D", dimension)],
                grid: (32 * pairs, 1, 1), threadGroup: (min(256, 32 * pairs), 1, 1),
                outputShapes: [query.shape], outputDTypes: [query.dtype])[0]
        }

        private static let longConvolutionKernel = MLXFast.metalKernel(
            name: "glm5_next_kda_long_convolution",
            inputNames: [
                "query", "key", "value", "q_conv", "k_conv", "v_conv", "conv_in", "validity",
                "token_count", "channel_count", "layout",
            ],
            outputNames: ["activated", "conv_out"],
            source: """
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                constexpr uint W = H * D;
                const uint index = thread_position_in_grid.x;
                if (index >= uint(channel_count)) { return; }
                const uint total = uint(token_count);
                const uint b = index / (3 * W);
                const uint channel = index % (3 * W);
                const uint part = channel / W;
                const uint local = channel % W;
                const size_t conv_base = (size_t(b) * 3 * W + channel) * K;
                // q/k/v rows: contiguous [B·T, W] each, or in place in the
                // fused projection (row stride N, part offsets 0, W, 2W).
                const size_t row_stride = size_t(layout[0]);
                const size_t column = size_t(layout[1 + part]) + local;
                float taps[K];
                float window[K];
                for (uint j = 0; j < K; ++j) {
                    taps[j] = part == 0 ? static_cast<float>(q_conv[local * K + j])
                        : part == 1 ? static_cast<float>(k_conv[local * K + j])
                        : static_cast<float>(v_conv[local * K + j]);
                    window[j] = static_cast<float>(conv_in[conv_base + j]);
                }
                for (uint t = 0; t < total; ++t) {
                    const size_t token = size_t(b) * total + t;
                    const size_t source = token * row_stride + column;
                    const T raw = part == 0 ? query[source] : part == 1 ? key[source] : value[source];
                    // A valid token enters the window; tap 0 is the oldest.
                    if (validity[token]) {
                        for (uint j = 0; j + 1 < K; ++j) { window[j] = window[j + 1]; }
                        window[K - 1] = static_cast<float>(raw);
                    }
                    float conv_total = 0.0f;
                    for (uint j = 0; j < K; ++j) {
                        const float product = window[j] * taps[j];
                        conv_total += product;
                    }
                    const float sig = 1.0f / (1.0f + metal::precise::exp(metal::abs(conv_total)));
                    const float activated_value = conv_total * (conv_total < 0.0f ? sig : 1.0f - sig);
                    activated[token * 3 * W + channel] = static_cast<T>(activated_value);
                }
                for (uint j = 0; j < K; ++j) { conv_out[conv_base + j] = static_cast<C>(window[j]); }
                """,
            ensureRowContiguous: true)

        /// `longConvolutionKernel` split along the tokens: thread (channel,
        /// block) produces tokens [block·TB, block·TB + TB) of one raw channel.
        /// It loads its block's raw values and validity (and the K tokens before
        /// the block) before any store, so the loads overlap instead of one
        /// dependent load per token. The window entering the block is the
        /// serial walk's window there: the K tokens before the block when all
        /// are valid, otherwise a replay of the stored window and every valid
        /// token from the K-th valid one before the block (only the last K
        /// valid tokens survive the shifts). Each token then runs the serial
        /// walk's operations on the same values in the same order, so every
        /// output and the retained window (written by the last block's thread)
        /// keep their bits.
        private static let blockedConvolutionKernel = MLXFast.metalKernel(
            name: "glm5_next_kda_long_convolution_blocked",
            inputNames: [
                "query", "key", "value", "q_conv", "k_conv", "v_conv", "conv_in", "validity",
                "token_count", "channel_count", "layout",
            ],
            outputNames: ["activated", "conv_out"],
            source: """
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                constexpr uint W = H * D;
                const uint index = thread_position_in_grid.x;
                const uint block = thread_position_in_grid.y;
                if (index >= uint(channel_count)) { return; }
                const uint total = uint(token_count);
                const uint first = block * uint(TB);
                if (first >= total) { return; }
                const uint count = metal::min(uint(TB), total - first);
                const uint b = index / (3 * W);
                const uint channel = index % (3 * W);
                const uint part = channel / W;
                const uint local = channel % W;
                const size_t conv_base = (size_t(b) * 3 * W + channel) * K;
                // q/k/v rows: contiguous [B·T, W] each, or in place in the
                // fused projection (row stride N, part offsets 0, W, 2W).
                const size_t row_stride = size_t(layout[0]);
                const size_t column = size_t(layout[1 + part]) + local;
                const size_t row0 = size_t(b) * total;
                const device T* rows = part == 0 ? query : part == 1 ? key : value;
                // Every load of the common path, issued before any store.
                T raw[TB];
                bool valid[TB];
                for (uint i = 0; i < uint(TB); ++i) {
                    const bool inside = i < count;
                    const size_t token = row0 + first + (inside ? i : 0u);
                    raw[i] = inside ? rows[token * row_stride + column] : T(0);
                    valid[i] = inside && validity[token];
                }
                bool replay = first < uint(K);
                T before[K];
                for (uint j = 0; j < uint(K); ++j) {
                    before[j] = T(0);
                    if (first >= uint(K)) {
                        const size_t token = row0 + first - uint(K) + j;
                        before[j] = rows[token * row_stride + column];
                        replay = replay || !validity[token];
                    }
                }
                float taps[K];
                for (uint j = 0; j < uint(K); ++j) {
                    taps[j] = part == 0 ? static_cast<float>(q_conv[local * K + j])
                        : part == 1 ? static_cast<float>(k_conv[local * K + j])
                        : static_cast<float>(v_conv[local * K + j]);
                }
                // The window entering `first`; tap 0 is the oldest.
                float window[K];
                if (!replay) {
                    for (uint j = 0; j < uint(K); ++j) { window[j] = static_cast<float>(before[j]); }
                } else {
                    uint start = first;
                    for (uint seen = 0; start > 0 && seen < uint(K);) {
                        --start;
                        if (validity[row0 + start]) { ++seen; }
                    }
                    for (uint j = 0; j < uint(K); ++j) {
                        window[j] = static_cast<float>(conv_in[conv_base + j]);
                    }
                    for (uint t = start; t < first; ++t) {
                        const size_t token = row0 + t;
                        if (validity[token]) {
                            const T entering = rows[token * row_stride + column];
                            for (uint j = 0; j + 1 < uint(K); ++j) { window[j] = window[j + 1]; }
                            window[K - 1] = static_cast<float>(entering);
                        }
                    }
                }
                for (uint i = 0; i < count; ++i) {
                    // A valid token enters the window.
                    if (valid[i]) {
                        for (uint j = 0; j + 1 < uint(K); ++j) { window[j] = window[j + 1]; }
                        window[K - 1] = static_cast<float>(raw[i]);
                    }
                    float conv_total = 0.0f;
                    for (uint j = 0; j < uint(K); ++j) {
                        const float product = window[j] * taps[j];
                        conv_total += product;
                    }
                    const float sig = 1.0f / (1.0f + metal::precise::exp(metal::abs(conv_total)));
                    const float activated_value = conv_total * (conv_total < 0.0f ? sig : 1.0f - sig);
                    activated[(row0 + first + i) * 3 * W + channel] = static_cast<T>(activated_value);
                }
                if (first + count == total) {
                    for (uint j = 0; j < uint(K); ++j) {
                        conv_out[conv_base + j] = static_cast<C>(window[j]);
                    }
                }
                """,
            ensureRowContiguous: true)

        // Thread t owns value row t / Lanes of its group and keys
        // i * Lanes + t % Lanes (bank-conflict-free staged reads); the key sums
        // reduce by shuffles inside each aligned 8-lane group.
        private static let longRecurrenceKernel = MLXFast.metalKernel(
            name: "glm5_next_kda_long_recurrence",
            inputNames: [
                "activated", "forget", "beta_logits", "a_log", "dt_bias", "state_in", "validity",
                "params", "token_count", "beta_layout",
            ],
            outputNames: ["core", "state_out"],
            source: """
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                constexpr uint W = H * D;
                constexpr uint Threads = Rows * Lanes;
                constexpr uint Splits = D / Rows;
                constexpr uint KeysPerLane = D / Lanes;
                threadgroup float qs[TB][D];
                threadgroup float ks[TB][D];
                threadgroup float decay[TB][D];
                threadgroup float vs[TB][Rows];
                threadgroup float beta_shared[TB];
                threadgroup float norms[TB][2];
                const uint n = threadgroup_position_in_grid.x;
                const uint split = n % Splits;
                const uint bh = n / Splits;
                const uint b = bh / H;
                const uint h = bh % H;
                const uint tid = thread_position_in_threadgroup.x;
                const uint lane = thread_index_in_simdgroup;
                const uint group = simdgroup_index_in_threadgroup;
                const uint row = tid / Lanes;
                const uint part = tid % Lanes;
                const uint dv = split * Rows + row;
                const float bound = params[0];
                const float query_scale = params[1];
                const uint total = uint(token_count);
                const float rate = metal::precise::exp(static_cast<float>(a_log[h]));
                const size_t state_base = size_t(bh) * D * D;
                float s[KeysPerLane];
                for (uint i = 0; i < KeysPerLane; ++i) {
                    s[i] = state_in[state_base + size_t(i * Lanes + part) * D + dv];
                }
                for (uint first = 0; first < total; first += TB) {
                    const uint count = metal::min(uint(TB), total - first);
                    for (uint e = tid; e < count * D; e += Threads) {
                        const uint u = e / D;
                        const uint c = e % D;
                        const size_t token = size_t(b) * total + first + u;
                        const size_t base = token * 3 * W + size_t(h) * D + c;
                        qs[u][c] = static_cast<float>(activated[base]);
                        ks[u][c] = static_cast<float>(activated[base + W]);
                        const float gate_input = static_cast<float>(forget[token * W + size_t(h) * D + c])
                            + static_cast<float>(dt_bias[h * D + c]);
                        const float x = rate * gate_input;
                        const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
                        const float log_decay = validity[token] ? bound * (x < 0.0f ? y : 1.0f - y) : 0.0f;
                        decay[u][c] = metal::precise::exp(log_decay);
                    }
                    for (uint e = tid; e < count * Rows; e += Threads) {
                        const uint u = e / Rows;
                        const uint r = e % Rows;
                        const size_t token = size_t(b) * total + first + u;
                        vs[u][r] = static_cast<float>(
                            activated[token * 3 * W + 2 * W + size_t(h) * D + split * Rows + r]);
                    }
                    if (tid < count) {
                        const size_t token = size_t(b) * total + first + tid;
                        const float x = static_cast<float>(
                            beta_logits[token * size_t(beta_layout[0]) + size_t(beta_layout[1]) + h]);
                        const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
                        const float rounded_beta = static_cast<float>(static_cast<T>(x < 0.0f ? y : 1.0f - y));
                        beta_shared[tid] = validity[token] ? rounded_beta : 0.0f;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint pair = group; pair < 2 * count; pair += Threads / 32) {
                        const uint u = pair / 2;
                        threadgroup float* source = pair % 2 == 0 ? qs[u] : ks[u];
                        float squares = 0.0f;
                        for (uint i = lane; i < D; i += 32) {
                            const float square = source[i] * source[i];
                            squares += square;
                        }
                        squares = simd_sum(squares);
                        if (lane == 0) { norms[u][pair % 2] = squares; }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint e = tid; e < count * D; e += Threads) {
                        const uint u = e / D;
                        const uint c = e % D;
                        qs[u][c] = qs[u][c] / metal::precise::sqrt(norms[u][0] + 1e-6f) * query_scale;
                        ks[u][c] = ks[u][c] / metal::precise::sqrt(norms[u][1] + 1e-6f);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint u = 0; u < count; ++u) {
                        float memory = 0.0f;
                        for (uint i = 0; i < KeysPerLane; ++i) {
                            const uint k = i * Lanes + part;
                            s[i] = s[i] * decay[u][k];
                            const float product = s[i] * ks[u][k];
                            memory += product;
                        }
                        for (uint offset = Lanes / 2; offset > 0; offset /= 2) {
                            memory += simd_shuffle_xor(memory, ushort(offset));
                        }
                        const float delta = (vs[u][row] - memory) * beta_shared[u];
                        float out = 0.0f;
                        for (uint i = 0; i < KeysPerLane; ++i) {
                            const uint k = i * Lanes + part;
                            const float update = ks[u][k] * delta;
                            s[i] = s[i] + update;
                            const float product = s[i] * qs[u][k];
                            out += product;
                        }
                        for (uint offset = Lanes / 2; offset > 0; offset /= 2) {
                            out += simd_shuffle_xor(out, ushort(offset));
                        }
                        if (part == 0) {
                            core[(size_t(b) * total + first + u) * W + size_t(h) * D + dv] = static_cast<T>(out);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                for (uint i = 0; i < KeysPerLane; ++i) {
                    state_out[state_base + size_t(i * Lanes + part) * D + dv] = s[i];
                }
                """,
            ensureRowContiguous: true)

        // One SIMD group per (token, head); the RMS sum is the single launch's
        // order (SIMD sums of 32-channel groups, added in order).
        private static let longOutputKernel = MLXFast.metalKernel(
            name: "glm5_next_kda_long_output",
            inputNames: ["core", "gate", "out_norm", "validity", "params", "pair_count"],
            outputNames: ["values"],
            source: """
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                constexpr uint W = H * D;
                constexpr uint Groups = D / 32;
                const uint pair = thread_position_in_grid.x / 32;
                if (pair >= uint(pair_count)) { return; }
                const uint lane = thread_index_in_simdgroup;
                const size_t token = pair / H;
                const uint h = pair % H;
                const size_t base = token * W + size_t(h) * D;
                const float epsilon = params[0];
                const float inverse_count = params[1];
                float squares = 0.0f;
                for (uint g = 0; g < Groups; ++g) {
                    const float value = static_cast<float>(core[base + g * 32 + lane]);
                    const float square = value * value;
                    squares += simd_sum(square);
                }
                const float mean = squares * inverse_count;
                const bool valid = validity[token];
                for (uint g = 0; g < Groups; ++g) {
                    const uint c = g * 32 + lane;
                    const float normalized = static_cast<float>(core[base + c]) * metal::precise::rsqrt(mean + epsilon);
                    const float g_value = static_cast<float>(gate[base + c]);
                    const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(g_value)));
                    const float gate_value = g_value < 0.0f ? y : 1.0f - y;
                    const float weighted = normalized * static_cast<float>(out_norm[c]);
                    const float gated = weighted * gate_value;
                    values[base + c] = valid ? static_cast<T>(gated) : static_cast<T>(0.0f);
                }
                """,
            ensureRowContiguous: true)
    #endif

    private struct Variant: Hashable {
        let lowRank: Bool
        let capture: Bool
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        private static let kernels: [Variant: MLXFast.MLXFastKernel] = Dictionary(
            uniqueKeysWithValues: [false, true].flatMap { lowRank in
                [false, true].map { capture in
                    (
                        Variant(lowRank: lowRank, capture: capture),
                        makeKernel(lowRank: lowRank, capture: capture)
                    )
                }
            })

        // Threads 0..3D-1 each own one raw channel (q, k, v) of the head and
        // walk the tokens; the recurrence gives thread t < Slices*D value
        // channel t % D and key slice t / D, so state reads/writes are
        // contiguous across a SIMD group. Head reductions: lane-strided
        // partials, simd_sum, then partials in fixed order. Exponentials and
        // rsqrt use the precise forms MLX uses.
        private static func makeKernel(lowRank: Bool, capture: Bool) -> MLXFast.MLXFastKernel {
            let gateNames =
                lowRank
                ? ["forget_in", "gate_in", "fb_w", "fb_s", "fb_b", "gb_w", "gb_s", "gb_b"]
                : ["forget", "gate"]
            // Quad lanes keep their f_b/g_b row's packed words, scale and
            // offset in registers for every token block.
            let lowRankSetup =
                lowRank
                ? """
                const uint quad = tid / 4;
                const uint ql = tid % 4;
                const bool gate_row = quad < 2 * D;
                const uint which = gate_row ? quad / D : 0;
                const uint gate_index = gate_row ? quad % D : 0;
                uint packed[4];
                float gate_scale = 0.0f;
                float gate_offset = 0.0f;
                if (gate_row) {
                    const size_t weight_row = size_t(h) * D + gate_index;
                    const size_t word_base = weight_row * 16 + ql * 4;
                    for (uint w = 0; w < 4; ++w) {
                        packed[w] = which == 0 ? fb_w[word_base + w] : gb_w[word_base + w];
                    }
                    const size_t group_index = weight_row * 2 + ql / 2;
                    gate_scale = static_cast<float>(which == 0 ? fb_s[group_index] : gb_s[group_index]);
                    gate_offset = static_cast<float>(which == 0 ? fb_b[group_index] : gb_b[group_index]);
                }
                """ : ""
            // Each gate value is stored as its activation-dtype rounding: f_b/g_b
            // rows as four lanes x 32 inputs (affine 4-bit, low nibble first,
            // one scale/offset per 64) and a quad sum.
            let loadGates =
                lowRank
                ? """
                if (gate_row) {
                    for (uint u = 0; u < count; ++u) {
                        const size_t x_base = (size_t(b) * total + first + u) * 128 + ql * 32;
                        float accumulated = 0.0f;
                        float x_sum = 0.0f;
                        for (uint w = 0; w < 4; ++w) {
                            for (uint j = 0; j < 8; ++j) {
                                const float x = static_cast<float>(
                                    which == 0 ? forget_in[x_base + w * 8 + j] : gate_in[x_base + w * 8 + j]);
                                const float product = x * static_cast<float>((packed[w] >> (4 * j)) & 0xFu);
                                accumulated += product;
                                x_sum += x;
                            }
                        }
                        const float scaled = gate_scale * accumulated;
                        const float shifted = gate_offset * x_sum;
                        float result = scaled + shifted;
                        result += simd_shuffle_xor(result, ushort(1));
                        result += simd_shuffle_xor(result, ushort(2));
                        if (ql == 0) {
                            const float rounded = static_cast<float>(static_cast<T>(result));
                            if (which == 0) { forget_s[u][gate_index] = rounded; } else { gate_s[u][gate_index] = rounded; }
                        }
                    }
                }
                """
                : """
                for (uint e = tid; e < count * D; e += Threads) {
                    const uint u = e / D;
                    const uint c = e % D;
                    const size_t index = (size_t(b) * total + first + u) * W + size_t(h) * D + c;
                    forget_s[u][c] = static_cast<float>(forget[index]);
                    gate_s[u][c] = static_cast<float>(gate[index]);
                }
                """
            // Captures exist only for single-block windows (t == u).
            let captureConvolution =
                capture
                ? """
                for (uint j = 0; j < K; ++j) {
                    conv_capture[((size_t(b) * total + t) * 3 * W + part * W + local) * K + j] =
                        static_cast<C>(window[j]);
                }
                """ : ""
            let captureState =
                capture
                ? """
                if (owner && k < D) {
                    state_capture[((size_t(b) * total + t) * H + h) * D * D + size_t(k) * D + dv] = s[i];
                }
                """ : ""
            return MLXFast.metalKernel(
                name: "glm5_next_kda_step" + (lowRank ? "_low_rank" : "")
                    + (capture ? "_capture" : ""),
                inputNames: ["query", "key", "value"] + gateNames + [
                    "beta_logits", "q_conv", "k_conv", "v_conv", "a_log", "dt_bias", "out_norm",
                    "conv_in", "state_in", "validity", "params", "token_count",
                ],
                outputNames: ["values", "conv_out", "state_out"]
                    + (capture ? ["state_capture", "conv_capture"] : []),
                source: """
                    #pragma clang fp reassociate(off)
                    #pragma clang fp contract(off)
                    constexpr uint W = H * D;
                    // Shuffle lanes: thread t owns value row t / Lanes and keys
                    // (t % Lanes) * KeysPerLane..., reduced by shuffles inside its
                    // aligned lane group. Otherwise key slices reduce through
                    // threadgroup memory (single-token, odd head widths).
                    constexpr uint Lanes = Threads / D;
                    constexpr bool ShuffleLanes =
                        Threads % D == 0 && Lanes <= 32 && (Lanes & (Lanes - 1)) == 0;
                    constexpr uint KeysPerLane = ShuffleLanes ? D / Lanes : 1;
                    constexpr uint Slices = Threads / D;
                    constexpr uint KeysPerSlice = (D + Slices - 1) / Slices;
                    constexpr uint StateEntries = ShuffleLanes ? KeysPerLane : KeysPerSlice;
                    constexpr uint Groups = (D + 31) / 32;
                    threadgroup float qs[TK][D];
                    threadgroup float ks[TK][D];
                    threadgroup float vs[TK][D];
                    // forget_s holds the projected forget gate, then its decay
                    // in place (32 KiB of threadgroup memory at TK = 8).
                    threadgroup float forget_s[TK][D];
                    threadgroup float gate_s[TK][D];
                    threadgroup float outs[TK][D];
                    threadgroup float partial[ShuffleLanes ? 1 : Slices * D];
                    threadgroup float norms[TK][8];
                    threadgroup float beta_shared[TK];
                    const uint n = threadgroup_position_in_grid.x;
                    const uint b = n / H;
                    const uint h = n % H;
                    const uint tid = thread_position_in_threadgroup.x;
                    const uint lane = thread_index_in_simdgroup;
                    const uint group = simdgroup_index_in_threadgroup;
                    const float bound = params[0];
                    const float epsilon = params[1];
                    const float query_scale = params[2];
                    const float inverse_count = params[3];
                    const uint total = uint(token_count);
                    const float rate = metal::precise::exp(static_cast<float>(a_log[h]));

                    // Per-thread carried state: the convolution window of one raw
                    // channel, and this thread's recurrent-state entries.
                    const bool convolves = tid < 3 * D;
                    const uint part = tid / D;
                    const uint d = tid % D;
                    const uint local = h * D + d;
                    const size_t channel = size_t(b) * 3 * W + part * W + local;
                    float taps[K];
                    float window[K];
                    if (convolves) {
                        for (uint j = 0; j < K; ++j) {
                            taps[j] = part == 0 ? static_cast<float>(q_conv[local * K + j])
                                : part == 1 ? static_cast<float>(k_conv[local * K + j])
                                : static_cast<float>(v_conv[local * K + j]);
                            window[j] = static_cast<float>(conv_in[channel * K + j]);
                        }
                    }
                    \(lowRankSetup)
                    const size_t state_base = size_t(n) * D * D;
                    const uint dv = ShuffleLanes ? tid / Lanes : tid % D;
                    const uint key_part = ShuffleLanes ? tid % Lanes : tid / D;
                    const bool owner = ShuffleLanes ? true : tid < Slices * D;
                    float s[StateEntries];
                    for (uint i = 0; i < StateEntries; ++i) {
                        const uint k = key_part * StateEntries + i;
                        s[i] = owner && k < D ? state_in[state_base + size_t(k) * D + dv] : 0.0f;
                    }

                    for (uint first = 0; first < total; first += TK) {
                        const uint count = metal::min(uint(TK), total - first);
                        if (convolves) {
                            for (uint u = 0; u < count; ++u) {
                                const uint t = first + u;
                                const size_t index = (size_t(b) * total + t) * W + local;
                                const T raw = part == 0 ? query[index] : part == 1 ? key[index] : value[index];
                                // A valid token enters the window; tap 0 is the oldest.
                                if (validity[b * total + t]) {
                                    for (uint j = 0; j + 1 < K; ++j) { window[j] = window[j + 1]; }
                                    window[K - 1] = static_cast<float>(raw);
                                }
                                float conv_total = 0.0f;
                                for (uint j = 0; j < K; ++j) {
                                    const float product = window[j] * taps[j];
                                    conv_total += product;
                                }
                                const float sig = 1.0f / (1.0f + metal::precise::exp(metal::abs(conv_total)));
                                const float activated = conv_total * (conv_total < 0.0f ? sig : 1.0f - sig);
                                const float rounded = static_cast<float>(static_cast<T>(activated));
                                if (part == 0) { qs[u][d] = rounded; } else if (part == 1) { ks[u][d] = rounded; } else { vs[u][d] = rounded; }
                                \(captureConvolution)
                            }
                        }
                        \(loadGates)
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        for (uint e = tid; e < count * D; e += Threads) {
                            const uint u = e / D;
                            const uint c = e % D;
                            const float gate_input = forget_s[u][c] + static_cast<float>(dt_bias[h * D + c]);
                            const float x = rate * gate_input;
                            const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
                            const float log_decay =
                                validity[b * total + first + u] ? bound * (x < 0.0f ? y : 1.0f - y) : 0.0f;
                            forget_s[u][c] = metal::precise::exp(log_decay);
                        }
                        if (tid < count) {
                            const uint t = first + tid;
                            const float x = static_cast<float>(beta_logits[(size_t(b) * total + t) * H + h]);
                            const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
                            const float rounded_beta = static_cast<float>(static_cast<T>(x < 0.0f ? y : 1.0f - y));
                            beta_shared[tid] = validity[b * total + t] ? rounded_beta : 0.0f;
                        }
                        if (group < 2 * count) {
                            const uint u = group / 2;
                            threadgroup float* source = group % 2 == 0 ? qs[u] : ks[u];
                            float squares = 0.0f;
                            for (uint i = lane; i < D; i += 32) {
                                const float square = source[i] * source[i];
                                squares += square;
                            }
                            squares = simd_sum(squares);
                            if (lane == 0) { norms[u][group % 2] = squares; }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        for (uint e = tid; e < count * D; e += Threads) {
                            const uint u = e / D;
                            const uint c = e % D;
                            qs[u][c] = qs[u][c] / metal::precise::sqrt(norms[u][0] + 1e-6f) * query_scale;
                            ks[u][c] = ks[u][c] / metal::precise::sqrt(norms[u][1] + 1e-6f);
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        for (uint u = 0; u < count; ++u) {
                            const uint t = first + u;
                            float memory = 0.0f;
                            for (uint i = 0; i < StateEntries; ++i) {
                                const uint k = key_part * StateEntries + i;
                                if (owner && k < D) {
                                    s[i] = s[i] * forget_s[u][k];
                                    const float product = s[i] * ks[u][k];
                                    memory += product;
                                }
                            }
                            float memory_total = 0.0f;
                            if (ShuffleLanes) {
                                memory_total = memory;
                                for (uint offset = Lanes / 2; offset > 0; offset /= 2) {
                                    memory_total += simd_shuffle_xor(memory_total, ushort(offset));
                                }
                            } else {
                                if (owner) { partial[key_part * D + dv] = memory; }
                                threadgroup_barrier(mem_flags::mem_threadgroup);
                                for (uint i = 0; i < Slices; ++i) { memory_total += partial[i * D + dv]; }
                            }
                            const float delta = (vs[u][dv] - memory_total) * beta_shared[u];
                            float out = 0.0f;
                            for (uint i = 0; i < StateEntries; ++i) {
                                const uint k = key_part * StateEntries + i;
                                if (owner && k < D) {
                                    const float update = ks[u][k] * delta;
                                    s[i] = s[i] + update;
                                    const float product = s[i] * qs[u][k];
                                    out += product;
                                }
                                \(captureState)
                            }
                            if (ShuffleLanes) {
                                for (uint offset = Lanes / 2; offset > 0; offset /= 2) {
                                    out += simd_shuffle_xor(out, ushort(offset));
                                }
                                if (key_part == 0) { outs[u][dv] = static_cast<float>(static_cast<T>(out)); }
                            } else {
                                threadgroup_barrier(mem_flags::mem_threadgroup);
                                if (owner) { partial[key_part * D + dv] = out; }
                                threadgroup_barrier(mem_flags::mem_threadgroup);
                                if (tid < D) {
                                    float out_total = 0.0f;
                                    for (uint i = 0; i < Slices; ++i) { out_total += partial[i * D + tid]; }
                                    outs[u][tid] = static_cast<float>(static_cast<T>(out_total));
                                }
                                threadgroup_barrier(mem_flags::mem_threadgroup);
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        if (tid < count * D) {
                            const uint u = tid / D;
                            const uint c = tid % D;
                            const float core = outs[u][c];
                            const float square = core * core;
                            const float squares = simd_sum(square);
                            if (lane == 0) { norms[u][2 + c / 32] = squares; }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        if (tid < count * D) {
                            const uint u = tid / D;
                            const uint c = tid % D;
                            const uint t = first + u;
                            float squares = 0.0f;
                            for (uint i = 0; i < Groups; ++i) { squares += norms[u][2 + i]; }
                            const float mean = squares * inverse_count;
                            const float normalized = outs[u][c] * metal::precise::rsqrt(mean + epsilon);
                            const float g = gate_s[u][c];
                            const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(g)));
                            const float gate_value = g < 0.0f ? y : 1.0f - y;
                            const float weighted = normalized * static_cast<float>(out_norm[c]);
                            const float gated = weighted * gate_value;
                            values[(size_t(b) * total + t) * W + size_t(h) * D + c] =
                                validity[b * total + t] ? static_cast<T>(gated) : static_cast<T>(0.0f);
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    for (uint i = 0; i < StateEntries; ++i) {
                        const uint k = key_part * StateEntries + i;
                        if (owner && k < D) { state_out[state_base + size_t(k) * D + dv] = s[i]; }
                    }
                    if (convolves) {
                        for (uint j = 0; j < K; ++j) { conv_out[channel * K + j] = static_cast<C>(window[j]); }
                    }
                    """,
                ensureRowContiguous: true)
        }
    #endif
}
