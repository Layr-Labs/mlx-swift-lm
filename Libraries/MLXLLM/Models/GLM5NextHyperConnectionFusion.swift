// Copyright 2026 the HuggingFace Team. All rights reserved.
// Copyright 2026 Eigen Labs.
// Modified in 2026: opt-in fused four-stream mHC collapse/expand Metal path.
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

/// Fused replacement for `GLM5NextHyperConnection.collapse` / `expand`, selected
/// only by `GLM5NextExecutionPolicy.HyperConnection.nativeRequired`.
///
/// Collapse is three launches: inverse RMS per row, split-K projection partials
/// and one coefficient/Sinkhorn/weighted-collapse kernel. Expand is one launch.
/// Every rounding boundary of the reference is retained: RMS normalization
/// precedes the FP32 projection, `pre` multiplies FP32 streams before their
/// stream sum and single activation-dtype cast, and expand casts `post`/`comb`
/// to the activation dtype BEFORE the output product and residual mix.
/// The trained `fn`/`base`/`scale` are read in their stored dtype and widened
/// exactly in registers, so no per-call or retained FP32 copy is needed.
///
/// FP32 sums use a fixed tree that is independent of batch and sequence shape
/// (so token-by-token and chunked forwards agree bitwise with each other) but
/// differs from MLX's reduction order; the result is not bitwise to reference.
extension GLM5NextHyperConnectionKernel {
    /// Channels per projection partial. Part of the fixed reduction tree.
    static let projectionChunk = 256
    /// Rows per projection threadgroup (one SIMD group each).
    static let projectionRowsPerGroup = 8
    /// Output columns per collapse threadgroup.
    static let collapseColumnsPerGroup = 1024

    struct FusedLayout {
        let batch: Int
        let tokens: Int
        let rows: Int
        let width: Int
        let channels: Int
        let chunks: Int

        init(streamsShape shape: [Int]) throws {
            guard shape.count == 4, shape[2] == 4 else {
                throw Failure.invalidGeometry("streams must be [B,T,4,D]")
            }
            var elements = 1
            for dimension in shape {
                let (next, overflow) = elements.multipliedReportingOverflow(by: dimension)
                guard dimension > 0, !overflow, next <= Int(Int32.max) else {
                    throw Failure.invalidGeometry("nonempty Int32-indexed streams required")
                }
                elements = next
            }
            batch = shape[0]
            tokens = shape[1]
            rows = batch * tokens
            width = shape[3]
            channels = 4 * width
            chunks = (channels + projectionChunk - 1) / projectionChunk
            let (partials, overflow) = (rows * chunks).multipliedReportingOverflow(by: 24)
            guard !overflow, partials <= Int(Int32.max) else {
                throw Failure.invalidGeometry("projection partials exceed Int32 indexing")
            }
        }
    }

    /// Same contract and outputs as `GLM5NextHyperConnection.collapse` with
    /// hcMult == 4: FP32 `post` [B,T,4] and `comb` [B,T,4,4], and the collapsed
    /// input [B,T,D] in the streams' dtype. Validation precedes any graph.
    static func collapse(
        _ hiddenStreams: MLXArray, fn: MLXArray, base: MLXArray, scale: MLXArray,
        rmsNormEpsilon: Float, epsilon: Float, sinkhornIterations: Int
    ) throws -> (post: MLXArray, comb: MLXArray, collapsed: MLXArray) {
        let layout = try FusedLayout(streamsShape: hiddenStreams.shape)
        guard rmsNormEpsilon.isFinite, rmsNormEpsilon > 0, epsilon.isFinite, epsilon > 0,
            (1 ... 128).contains(sinkhornIterations)
        else {
            throw Failure.invalidGeometry(
                "epsilons must be positive and finite; iterations must be 1...128")
        }
        try requireActivation(hiddenStreams, name: "streams")
        for (name, array, shape) in [
            ("fn", fn, [24, layout.channels]), ("base", base, [24]), ("scale", scale, [3]),
        ] {
            guard array.shape == shape, Self.activationTypes.contains(array.dtype) else {
                throw Failure.invalidTensor("\(name) must be floating \(shape)")
            }
        }
        guard glm5NextOnDefaultGPUStream() else { throw Failure.unsupportedDeviceOrStream }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            let streams = hiddenStreams.reshaped(layout.rows, layout.channels)
            let inverseRMS = inverseRMSKernel(
                [
                    streams, MLXArray(Float(1.0 / Double(layout.channels))),
                    MLXArray(rmsNormEpsilon),
                ],
                template: [("Channels", layout.channels)],
                grid: (256, layout.rows, 1), threadGroup: (256, 1, 1),
                outputShapes: [[layout.rows]], outputDTypes: [.float32])[0]
            let rowGroups =
                (layout.rows + projectionRowsPerGroup - 1) / projectionRowsPerGroup
            let partials = projectionKernel(
                [streams, inverseRMS, fn, MLXArray(Int32(layout.rows))],
                template: [("Channels", layout.channels), ("Chunks", layout.chunks)],
                grid: (layout.chunks * 256, rowGroups, 1), threadGroup: (256, 1, 1),
                outputShapes: [[layout.rows, layout.chunks, 24]], outputDTypes: [.float32])[0]
            let columnGroups =
                (layout.width + collapseColumnsPerGroup - 1) / collapseColumnsPerGroup
            let outputs = collapseKernel(
                [streams, partials, base, scale, MLXArray(epsilon)],
                template: [
                    ("T", hiddenStreams.dtype), ("Width", layout.width),
                    ("Chunks", layout.chunks), ("Iterations", sinkhornIterations),
                ],
                grid: (columnGroups * 256, layout.rows, 1), threadGroup: (256, 1, 1),
                outputShapes: [
                    [layout.batch, layout.tokens, 4], [layout.batch, layout.tokens, 4, 4],
                    [layout.batch, layout.tokens, layout.width],
                ],
                outputDTypes: [.float32, .float32, hiddenStreams.dtype])
            return (outputs[0], outputs[1], outputs[2])
        #else
            throw Failure.unsupportedDeviceOrStream
        #endif
    }

    /// Same contract as `GLM5NextHyperConnection.expand`, with the FP32
    /// coefficients produced by the fused collapse.
    static func expand(
        _ output: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray
    ) throws -> MLXArray {
        let layout = try FusedLayout(streamsShape: residual.shape)
        try requireActivation(residual, name: "residual")
        guard output.shape == [layout.batch, layout.tokens, layout.width],
            output.dtype == residual.dtype
        else { throw Failure.invalidTensor("output must match residual [B,T,D] and dtype") }
        guard post.shape == [layout.batch, layout.tokens, 4], post.dtype == .float32,
            comb.shape == [layout.batch, layout.tokens, 4, 4], comb.dtype == .float32
        else { throw Failure.invalidTensor("post/comb must be FP32 [B,T,4] / [B,T,4,4]") }
        guard glm5NextOnDefaultGPUStream() else { throw Failure.unsupportedDeviceOrStream }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            return expandKernel(
                [output, residual, post, comb],
                template: [("T", residual.dtype), ("Width", layout.width)],
                grid: (layout.width, layout.rows, 1), threadGroup: (256, 1, 1),
                outputShapes: [residual.shape], outputDTypes: [residual.dtype])[0]
        #else
            throw Failure.unsupportedDeviceOrStream
        #endif
    }

    static let activationTypes: [DType] = [.float32, .float16, .bfloat16]

    static func requireActivation(_ array: MLXArray, name: String) throws {
        guard activationTypes.contains(array.dtype) else {
            throw Failure.invalidTensor("\(name) must be FP32, FP16 or BF16")
        }
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        // One 256-thread group per row. Thread t sums channels t, t+256, ...
        // in order; then SIMD sums and the eight SIMD partials in order.
        // MLX mean is sum * (1/count) and rsqrt is metal::precise::rsqrt.
        private static let inverseRMSKernel = MLXFast.metalKernel(
            name: "glm5_next_hc_inverse_rms",
            inputNames: ["streams", "inverse_count", "rms_epsilon"],
            outputNames: ["inverse_rms"],
            source: """
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                threadgroup float simd_totals[8];
                const uint row = threadgroup_position_in_grid.y;
                const uint index = thread_position_in_threadgroup.x;
                const size_t offset = size_t(row) * Channels;
                float total = 0.0f;
                for (uint channel = index; channel < uint(Channels); channel += 256) {
                    const float value = static_cast<float>(streams[offset + channel]);
                    const float square = value * value;
                    total += square;
                }
                total = simd_sum(total);
                if (thread_index_in_simdgroup == 0) {
                    simd_totals[simdgroup_index_in_threadgroup] = total;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (index == 0) {
                    float sum = 0.0f;
                    for (int group = 0; group < 8; ++group) { sum += simd_totals[group]; }
                    const float mean = sum * inverse_count;
                    inverse_rms[row] = metal::precise::rsqrt(mean + rms_epsilon);
                }
                """,
            ensureRowContiguous: true)

        // Group (chunk, row block): the 24 x 256 weight tile is widened once into
        // threadgroup memory; SIMD group s owns row 8*block+s. Lane l sums
        // channels l, l+32, ... of the chunk in order, then one SIMD sum.
        // Normalization (x * inverse_rms) is rounded BEFORE each product.
        private static let projectionKernel = MLXFast.metalKernel(
            name: "glm5_next_hc_projection_partials",
            inputNames: ["streams", "inverse_rms", "fn", "rows"],
            outputNames: ["partials"],
            source: """
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                threadgroup float weights[24 * 256];
                const uint chunk = threadgroup_position_in_grid.x;
                const uint index = thread_position_in_threadgroup.x;
                const uint column = chunk * 256 + index;
                for (int output = 0; output < 24; ++output) {
                    weights[output * 256 + index] = column < uint(Channels)
                        ? static_cast<float>(fn[size_t(output) * Channels + column]) : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const uint row = threadgroup_position_in_grid.y * 8 + simdgroup_index_in_threadgroup;
                if (row < uint(rows)) {
                    const uint lane = thread_index_in_simdgroup;
                    const float inverse = inverse_rms[row];
                    const size_t offset = size_t(row) * Channels;
                    float accumulators[24];
                    for (int output = 0; output < 24; ++output) { accumulators[output] = 0.0f; }
                    for (int step = 0; step < 8; ++step) {
                        const uint local = lane + 32 * step;
                        const uint channel = chunk * 256 + local;
                        if (channel < uint(Channels)) {
                            const float normalized =
                                static_cast<float>(streams[offset + channel]) * inverse;
                            for (int output = 0; output < 24; ++output) {
                                const float product = normalized * weights[output * 256 + local];
                                accumulators[output] += product;
                            }
                        }
                    }
                    float mine = 0.0f;
                    for (int output = 0; output < 24; ++output) {
                        const float total = simd_sum(accumulators[output]);
                        if (lane == uint(output)) { mine = total; }
                    }
                    if (lane < 24) {
                        partials[(size_t(row) * Chunks + chunk) * 24 + lane] = mine;
                    }
                }
                """,
            ensureRowContiguous: true)

        // Group (column block, row): chunk partials are summed in order, SIMD
        // group 0 runs the shared coefficient stage, and column block 0 alone
        // stores post/comb. The collapse sums FP32 pre*stream products in stream
        // order (MLX's column reduction) and casts once to the activation dtype.
        private static let collapseKernel = MLXFast.metalKernel(
            name: "glm5_next_hc4_collapse",
            inputNames: ["streams", "partials", "base", "scale", "epsilon"],
            outputNames: ["post", "comb", "collapsed"],
            source: """
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                threadgroup float projected[24];
                threadgroup float pre_weights[4];
                const uint row = threadgroup_position_in_grid.y;
                const uint block = threadgroup_position_in_grid.x;
                const uint index = thread_position_in_threadgroup.x;
                if (index < 24) {
                    float total = 0.0f;
                    for (int chunk = 0; chunk < Chunks; ++chunk) {
                        total += partials[(size_t(row) * Chunks + chunk) * 24 + index];
                    }
                    projected[index] = total;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simdgroup_index_in_threadgroup == 0) {
                    const uint lane = thread_index_in_simdgroup;
                    const bool owns_row = lane < 4;
                    const uint matrix_row = owns_row ? lane : 0;
                    float pre_logit = 0.0f;
                    float post_logit = 0.0f;
                    float logits[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    if (owns_row) {
                        const float pre_product =
                            projected[matrix_row] * static_cast<float>(scale[0]);
                        pre_logit = pre_product + static_cast<float>(base[matrix_row]);
                        const float post_product =
                            projected[4 + matrix_row] * static_cast<float>(scale[1]);
                        post_logit = post_product + static_cast<float>(base[4 + matrix_row]);
                        for (int j = 0; j < 4; ++j) {
                            const float product =
                                projected[8 + matrix_row * 4 + j] * static_cast<float>(scale[2]);
                            logits[j] = product + static_cast<float>(base[8 + matrix_row * 4 + j]);
                        }
                    }
                    float pre_value;
                    float post_value;
                    float c[4];
                    glm5_hc4_coefficients(
                        owns_row, pre_logit, post_logit, logits, epsilon, Iterations,
                        pre_value, post_value, c);
                    if (owns_row) {
                        pre_weights[matrix_row] = pre_value;
                        if (block == 0) {
                            post[size_t(row) * 4 + matrix_row] = post_value;
                            for (int j = 0; j < 4; ++j) {
                                comb[(size_t(row) * 4 + matrix_row) * 4 + j] = c[j];
                            }
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const size_t offset = size_t(row) * 4 * Width;
                for (int step = 0; step < 4; ++step) {
                    const uint column = block * 1024 + step * 256 + index;
                    if (column < uint(Width)) {
                        float total = 0.0f;
                        for (int stream = 0; stream < 4; ++stream) {
                            const float product = pre_weights[stream]
                                * static_cast<float>(streams[offset + stream * Width + column]);
                            total += product;
                        }
                        collapsed[size_t(row) * Width + column] = static_cast<T>(total);
                    }
                }
                """,
            header: coefficientHeader,
            ensureRowContiguous: true)

        // One thread per (column, row) writes all four destination streams with
        // the reference's activation-dtype boundaries: rounded post * output,
        // rounded FP32-accumulated comb^T @ residual, then their rounded sum.
        /// `output` can be the last layer's tensor-parallel sum: a fenced
        /// input (see `MLXFast.metalKernel(...fencedInputs:)`).
        private static let expandKernel = MLXFast.metalKernel(
            name: "glm5_next_hc4_expand",
            inputNames: ["output", "residual", "post", "comb"],
            outputNames: ["expanded"],
            source: """
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                MLX_FENCED_INPUT_WAIT(output);
                const uint column = thread_position_in_grid.x;
                const uint row = thread_position_in_grid.y;
                const size_t offset = size_t(row) * 4 * Width;
                const T value = output[size_t(row) * Width + column];
                T sources[4];
                for (int source = 0; source < 4; ++source) {
                    sources[source] = residual[offset + source * Width + column];
                }
                for (int destination = 0; destination < 4; ++destination) {
                    const T coefficient = static_cast<T>(post[size_t(row) * 4 + destination]);
                    const T injected = static_cast<T>(coefficient * value);
                    float total = 0.0f;
                    for (int source = 0; source < 4; ++source) {
                        const T weight =
                            static_cast<T>(comb[(size_t(row) * 4 + source) * 4 + destination]);
                        const float product =
                            static_cast<float>(weight) * static_cast<float>(sources[source]);
                        total += product;
                    }
                    const T mixed = static_cast<T>(total);
                    expanded[offset + destination * Width + column] = static_cast<T>(injected + mixed);
                }
                """,
            ensureRowContiguous: true, fencedInputs: ["output"])
    #endif
}
