// Copyright 2026 the HuggingFace Team. All rights reserved.
// Copyright 2026 Eigen Labs.
// Modified in 2026: opt-in fused mHC collapse with its RMSNorm, and an
// expand folded into the following collapse; decode-sized sites in one launch,
// waiting themselves for a cross-rank sum (MLX fenced input).
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

/// `GLM5NextExecutionPolicy.HyperConnection.nativeNormFused`: each collapse and
/// the RMSNorm that consumes it run fused, and an expand whose only consumer is
/// the next collapse is computed inside that collapse, which also stores the
/// expanded streams for the residual path. Prefill steps take one launch
/// (`prefillCollapse`); decode/verify steps take the split form below, in one
/// launch per site (`SplitForm.oneLaunch`).
///
/// Rounding boundaries follow the reference: expanded streams and the collapsed
/// input are rounded to the activation dtype exactly as `expand`/`collapse` do,
/// and the norm rounds `x * rsqrt` before the weight product. Two reorderings
/// make it not bitwise: the mHC RMS factor multiplies each projected sum
/// instead of every streamed value (exact in real arithmetic), and the norm's
/// mean of squares uses this kernel's fixed tree.
extension GLM5NextHyperConnectionKernel {
    enum StreamsSource {
        case streams(MLXArray)
        /// `expand(output, residual:, post:, comb:)`, materialized by the collapse.
        case expand(output: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray)
    }

    struct NormalizedCollapse {
        /// The collapse's input streams: the given ones, or the expanded ones.
        let streams: MLXArray
        let post: MLXArray
        let comb: MLXArray
        /// `RMSNorm(collapsed)` in the activation dtype, [B,T,D].
        let normalized: MLXArray
    }

    /// Threads of one normalized-collapse threadgroup (one row).
    static let normalizedCollapseThreads = 1024

    /// How a step of at most `splitCollapseMaximumRows` rows runs a site.
    enum SplitForm {
        /// One launch (`oneLaunchCollapse`): the split form's arithmetic spread
        /// over one threadgroup per SIMD group of its tree, finished by the
        /// row's last threadgroup to arrive. Bitwise to `twoLaunch`.
        case oneLaunch
        /// The pre/collapse/norm launch on the critical path and the post/comb
        /// launch beside the branch (`splitCollapse`): the oracle of `oneLaunch`.
        case twoLaunch
    }

    static func collapseNormalized(
        _ source: StreamsSource, fn: MLXArray, base: MLXArray, scale: MLXArray,
        rmsNormEpsilon: Float, epsilon: Float, sinkhornIterations: Int,
        normWeight: MLXArray, normEpsilon: Float, splitForm: SplitForm = .oneLaunch,
        workspace: SiteWorkspace = .shared
    ) throws -> NormalizedCollapse {
        let reference: MLXArray
        switch source {
        case .streams(let streams): reference = streams
        case .expand(let output, let residual, let post, let comb):
            reference = residual
            let layout = try FusedLayout(streamsShape: residual.shape)
            guard output.shape == [layout.batch, layout.tokens, layout.width],
                output.dtype == residual.dtype
            else { throw Failure.invalidTensor("output must match residual [B,T,D] and dtype") }
            guard post.shape == [layout.batch, layout.tokens, 4], post.dtype == .float32,
                comb.shape == [layout.batch, layout.tokens, 4, 4], comb.dtype == .float32
            else { throw Failure.invalidTensor("post/comb must be FP32 [B,T,4] / [B,T,4,4]") }
        }
        let layout = try FusedLayout(streamsShape: reference.shape)
        guard rmsNormEpsilon.isFinite, rmsNormEpsilon > 0, epsilon.isFinite, epsilon > 0,
            normEpsilon.isFinite, normEpsilon > 0, (1 ... 128).contains(sinkhornIterations)
        else {
            throw Failure.invalidGeometry(
                "epsilons must be positive and finite; iterations must be 1...128")
        }
        try requireActivation(reference, name: "streams")
        for (name, array, shape) in [
            ("fn", fn, [24, layout.channels]), ("base", base, [24]), ("scale", scale, [3]),
            ("norm weight", normWeight, [layout.width]),
        ] {
            guard array.shape == shape, Self.activationTypes.contains(array.dtype) else {
                throw Failure.invalidTensor("\(name) must be floating \(shape)")
            }
        }
        // The reference product `weight * x` would promote a wider weight.
        guard normWeight.dtype == reference.dtype else {
            throw Failure.invalidTensor("norm weight must have the activation dtype")
        }
        guard glm5NextOnDefaultGPUStream() else { throw Failure.unsupportedDeviceOrStream }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            if layout.rows <= splitCollapseMaximumRows, splitForm == .oneLaunch {
                return oneLaunchCollapse(
                    source, layout: layout, fn: fn, base: base, scale: scale,
                    rmsNormEpsilon: rmsNormEpsilon, epsilon: epsilon,
                    sinkhornIterations: sinkhornIterations, normWeight: normWeight,
                    normEpsilon: normEpsilon, dtype: reference.dtype, workspace: workspace)
            }
            if layout.rows <= splitCollapseMaximumRows {
                return splitCollapse(
                    source, layout: layout, fn: fn, base: base, scale: scale,
                    rmsNormEpsilon: rmsNormEpsilon, epsilon: epsilon,
                    sinkhornIterations: sinkhornIterations, normWeight: normWeight,
                    normEpsilon: normEpsilon, dtype: reference.dtype)
            }
            return prefillCollapse(
                source, layout: layout, fn: fn, base: base, scale: scale,
                rmsNormEpsilon: rmsNormEpsilon, epsilon: epsilon,
                sinkhornIterations: sinkhornIterations, normWeight: normWeight,
                normEpsilon: normEpsilon, dtype: reference.dtype)
        #else
            throw Failure.unsupportedDeviceOrStream
        #endif
    }

    /// Rows (decode and verify steps) that take the split form: one launch per
    /// row on the critical path (pre coefficients, collapse, RMSNorm, and the
    /// folded expand), and one launch for post/comb that only the next expand
    /// reads, ordered before the branch so it overlaps it. `SplitForm.oneLaunch`
    /// runs the same arithmetic as one launch.
    static let splitCollapseMaximumRows = 8

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        private static func splitCollapse(
            _ source: StreamsSource, layout: FusedLayout, fn: MLXArray, base: MLXArray,
            scale: MLXArray, rmsNormEpsilon: Float, epsilon: Float, sinkhornIterations: Int,
            normWeight: MLXArray, normEpsilon: Float, dtype: DType
        ) -> NormalizedCollapse {
            let threads = normalizedCollapseThreads
            let perThread = (layout.width + threads - 1) / threads
            let params = MLXArray([
                epsilon, Float(1.0 / Double(layout.channels)), rmsNormEpsilon,
                Float(1.0 / Double(layout.width)), normEpsilon,
            ])
            let template: [(String, any KernelTemplateArg)] = [
                ("T", dtype), ("Width", layout.width), ("Threads", threads),
                ("PerThread", perThread), ("Iterations", sinkhornIterations),
            ]
            let grid = (threads, layout.rows, 1)
            let rowShape = [layout.batch, layout.tokens, layout.width]
            let streams: MLXArray
            let normalized: MLXArray
            switch source {
            case .streams(let input):
                streams = input
                normalized =
                    preCollapseKernel(
                        [input, fn, base, scale, normWeight, params], template: template,
                        grid: grid,
                        threadGroup: (threads, 1, 1), outputShapes: [rowShape],
                        outputDTypes: [dtype])[0]
            case .expand(let output, let residual, let post, let comb):
                let outputs = expandPreCollapseKernel(
                    [output, residual, post, comb, fn, base, scale, normWeight, params],
                    template: template, grid: grid, threadGroup: (threads, 1, 1),
                    outputShapes: [rowShape, residual.shape], outputDTypes: [dtype, dtype])
                normalized = outputs[0]
                streams = outputs[1]
            }
            let coefficients = postCombKernel(
                [streams, fn, base, scale, params], template: template, grid: grid,
                threadGroup: (threads, 1, 1),
                outputShapes: [
                    [layout.batch, layout.tokens, 4], [layout.batch, layout.tokens, 4, 4],
                ],
                outputDTypes: [.float32, .float32])
            // Ordering only: the branch input reads the collapse's buffer, so
            // post/comb run beside the branch instead of before the next expand.
            return NormalizedCollapse(
                streams: streams, post: coefficients[0], comb: coefficients[1],
                normalized: depends(input: normalized, dependencies: coefficients))
        }

        /// Shared head of the split kernels: thread t holds all four streams of
        /// columns t, t+Threads, ... (`load` sets `value` for `stream` at
        /// `column`), and sums of squares plus the requested projection rows in
        /// a fixed tree (thread order, SIMD sums, SIMD partials in order).
        private static func splitHead(load: String, rows: Int, firstRow: Int) -> String {
            """
            #pragma clang fp reassociate(off)
            #pragma clang fp contract(off)
            constexpr uint Groups = uint(Threads) / 32;
            constexpr uint Sums = \(rows) + 1;
            threadgroup float partial[Groups * Sums];
            threadgroup float totals[Sums];
            const float epsilon = params[0];
            const float inverse_channels = params[1];
            const float rms_epsilon = params[2];
            const float inverse_width = params[3];
            const float norm_epsilon = params[4];
            const uint row = threadgroup_position_in_grid.y;
            const uint tid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint group = simdgroup_index_in_threadgroup;
            const size_t offset = size_t(row) * 4 * Width;
            float values[4][PerThread];
            float sums[Sums];
            for (uint i = 0; i < Sums; ++i) { sums[i] = 0.0f; }
            for (uint k = 0; k < uint(PerThread); ++k) {
                const uint column = tid + k * uint(Threads);
                for (uint stream = 0; stream < 4; ++stream) {
                    float value = 0.0f;
                    if (column < uint(Width)) {
                        \(load)
                        const size_t channel = size_t(stream) * Width + column;
                        const float square = value * value;
                        sums[\(rows)] += square;
                        for (uint o = 0; o < \(rows); ++o) {
                            const float product =
                                value * static_cast<float>(fn[size_t(\(firstRow) + o) * 4 * Width + channel]);
                            sums[o] += product;
                        }
                    }
                    values[stream][k] = value;
                }
            }
            for (uint i = 0; i < Sums; ++i) {
                const float total = simd_sum(sums[i]);
                if (lane == 0) { partial[group * Sums + i] = total; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < Sums) {
                float total = 0.0f;
                for (uint g = 0; g < Groups; ++g) { total += partial[g * Sums + tid]; }
                totals[tid] = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float inverse_rms = metal::precise::rsqrt(totals[\(rows)] * inverse_channels + rms_epsilon);

            """
        }

        private static let streamLoad =
            "value = static_cast<float>(streams[offset + size_t(stream) * Width + column]);"

        // The expand arithmetic is `expandKernel`'s, element for element.
        private static let expandLoad = """
            const T injected_value = output[size_t(row) * Width + column];
            const T coefficient = static_cast<T>(post[size_t(row) * 4 + stream]);
            const T injected = static_cast<T>(coefficient * injected_value);
            float mix = 0.0f;
            for (uint source = 0; source < 4; ++source) {
                const T weight = static_cast<T>(comb[(size_t(row) * 4 + source) * 4 + stream]);
                const T residual_value = residual[offset + size_t(source) * Width + column];
                const float product = static_cast<float>(weight) * static_cast<float>(residual_value);
                mix += product;
            }
            const T mixed = static_cast<T>(mix);
            const T expanded_value = static_cast<T>(injected + mixed);
            expanded[offset + size_t(stream) * Width + column] = expanded_value;
            value = static_cast<float>(expanded_value);
            """

        // Pre coefficients from rows 0..3, the weighted collapse rounded to the
        // activation dtype, then RMSNorm with its own fixed tree.
        private static func preCollapseSource(load: String) -> String {
            splitHead(load: load, rows: 4, firstRow: 0) + """
                float pre[4];
                for (uint s = 0; s < 4; ++s) {
                    const float projected = totals[s] * inverse_rms;
                    const float product = projected * static_cast<float>(scale[0]);
                    const float logit = product + static_cast<float>(base[s]);
                    pre[s] = glm5_hc_sigmoid(logit) + epsilon;
                }
                float collapsed[PerThread];
                float squares = 0.0f;
                for (uint k = 0; k < uint(PerThread); ++k) {
                    const uint column = tid + k * uint(Threads);
                    collapsed[k] = 0.0f;
                    if (column < uint(Width)) {
                        float total = 0.0f;
                        for (uint s = 0; s < 4; ++s) {
                            const float product = pre[s] * values[s][k];
                            total += product;
                        }
                        collapsed[k] = static_cast<float>(static_cast<T>(total));
                        const float square = collapsed[k] * collapsed[k];
                        squares += square;
                    }
                }
                const float simd_total = simd_sum(squares);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lane == 0) { partial[group] = simd_total; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float sum = 0.0f;
                for (uint g = 0; g < Groups; ++g) { sum += partial[g]; }
                const float inverse = metal::precise::rsqrt(sum * inverse_width + norm_epsilon);
                for (uint k = 0; k < uint(PerThread); ++k) {
                    const uint column = tid + k * uint(Threads);
                    if (column < uint(Width)) {
                        const T scaled = static_cast<T>(collapsed[k] * inverse);
                        const T weight = static_cast<T>(norm_weight[column]);
                        normalized[size_t(row) * Width + column] = static_cast<T>(weight * scaled);
                    }
                }
                """
        }

        private static let preCollapseKernel = MLXFast.metalKernel(
            name: "glm5_next_hc4_pre_collapse_norm",
            inputNames: ["streams", "fn", "base", "scale", "norm_weight", "params"],
            outputNames: ["normalized"], source: preCollapseSource(load: streamLoad),
            header: coefficientHeader, ensureRowContiguous: true)

        private static let expandPreCollapseKernel = MLXFast.metalKernel(
            name: "glm5_next_hc4_expand_pre_collapse_norm",
            inputNames: [
                "output", "residual", "post", "comb", "fn", "base", "scale", "norm_weight",
                "params",
            ],
            outputNames: ["normalized", "expanded"], source: preCollapseSource(load: expandLoad),
            header: coefficientHeader, ensureRowContiguous: true)

        // Post and comb (rows 4..23) with the coefficient stage on SIMD group 0.
        private static let postCombKernel = MLXFast.metalKernel(
            name: "glm5_next_hc4_post_comb",
            inputNames: ["streams", "fn", "base", "scale", "params"],
            outputNames: ["post", "comb"],
            source: splitHead(load: streamLoad, rows: 20, firstRow: 4) + """
                if (group == 0) {
                    const bool owns_row = lane < 4;
                    const uint matrix_row = owns_row ? lane : 0;
                    float post_logit = 0.0f;
                    float logits[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    if (owns_row) {
                        const float post_projected = totals[matrix_row] * inverse_rms;
                        const float post_product = post_projected * static_cast<float>(scale[1]);
                        post_logit = post_product + static_cast<float>(base[4 + matrix_row]);
                        for (int j = 0; j < 4; ++j) {
                            const float projected = totals[4 + matrix_row * 4 + j] * inverse_rms;
                            const float product = projected * static_cast<float>(scale[2]);
                            logits[j] = product + static_cast<float>(base[8 + matrix_row * 4 + j]);
                        }
                    }
                    float pre_value;
                    float post_value;
                    float c[4];
                    glm5_hc4_coefficients(
                        owns_row, 0.0f, post_logit, logits, epsilon, Iterations,
                        pre_value, post_value, c);
                    if (owns_row) {
                        post[size_t(row) * 4 + matrix_row] = post_value;
                        for (int j = 0; j < 4; ++j) {
                            comb[(size_t(row) * 4 + matrix_row) * 4 + j] = c[j];
                        }
                    }
                }
                """,
            header: coefficientHeader, ensureRowContiguous: true)
    #endif
}

// MARK: - The split form in one launch

extension GLM5NextHyperConnectionKernel {
    /// Sums of a one-launch site: fn rows 0..<24, then the streams' sum of squares.
    static let oneLaunchSums = 25

    /// Threadgroups of a one-launch site row: one per SIMD group of the split
    /// form's one-threadgroup tree.
    static let oneLaunchGroups = normalizedCollapseThreads / 32

    /// Cross-threadgroup state of `SplitForm.oneLaunch`, shared by every site:
    /// per-row arrival counters in two alternating slots, and per-row partial
    /// sums. Every launch passes both as mutable inputs, so MLX orders all
    /// launches that use them one after another (a barrier within a command
    /// buffer, a fence across command buffers); none of them run concurrently.
    ///
    /// Counters are zero between launches: a row's last threadgroup clears its
    /// slot and the other one, so a launch that never finished (a failed
    /// command buffer) is cleared by the next launch that completes, which uses
    /// the other slot.
    ///
    /// Production uses `shared`; a test may run sites on its own workspace.
    final class SiteWorkspace: @unchecked Sendable {
        static let shared = SiteWorkspace()

        /// UInt32 `[2, splitCollapseMaximumRows]`.
        let counters: MLXArray
        /// Float32 `[splitCollapseMaximumRows, oneLaunchSums, oneLaunchGroups]`.
        let partials: MLXArray
        private let lock = NSLock()
        private var launches: UInt64 = 0

        init() {
            let rows = GLM5NextHyperConnectionKernel.splitCollapseMaximumRows
            counters = MLXArray([UInt32](repeating: 0, count: 2 * rows))
            partials = MLXArray(
                [Float](
                    repeating: 0,
                    count: rows * GLM5NextHyperConnectionKernel.oneLaunchSums
                        * GLM5NextHyperConnectionKernel.oneLaunchGroups))
        }

        /// The counter slot of the next launch, alternating 0 and 1.
        func nextSlot() -> Int {
            lock.lock()
            defer { lock.unlock() }
            launches &+= 1
            return Int(launches & 1)
        }

        /// The slot `nextSlot` returns next, without taking it (a test seam).
        var upcomingSlot: Int {
            lock.lock()
            defer { lock.unlock() }
            return Int((launches &+ 1) & 1)
        }
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        /// The split form (`splitCollapse`) in one launch of
        /// `oneLaunchGroups` 1,024-thread threadgroups per row.
        ///
        /// Threadgroup g computes, for the threads 32g..<32g+32 of the split
        /// form's single threadgroup, the 24 projection sums and the sum of
        /// squares: SIMD group s < 24 sums fn row s and SIMD group 24 the
        /// squares, in the split form's (column, stream) order, then `simd_sum`.
        /// It publishes the 25 SIMD totals and counts itself in. The row's last
        /// threadgroup to arrive adds the totals in SIMD-group order, as the
        /// split form does, then runs the post/comb coefficient stage on SIMD
        /// group 0 while SIMD groups 1...31 collapse and normalize the row with
        /// the split form's thread-to-column map and tree (SIMD group 1 also
        /// takes SIMD group 0's columns). Every value, sum and rounding is the
        /// split form's, in its order: the outputs are bitwise those of
        /// `splitCollapse`. No threadgroup waits for another.
        private static func oneLaunchCollapse(
            _ source: StreamsSource, layout: FusedLayout, fn: MLXArray, base: MLXArray,
            scale: MLXArray, rmsNormEpsilon: Float, epsilon: Float, sinkhornIterations: Int,
            normWeight: MLXArray, normEpsilon: Float, dtype: DType, workspace: SiteWorkspace
        ) -> NormalizedCollapse {
            let threads = normalizedCollapseThreads
            let perThread = (layout.width + threads - 1) / threads
            let params = GLM5NextKernelConstants.shared.floats([
                epsilon, Float(1.0 / Double(layout.channels)), rmsNormEpsilon,
                Float(1.0 / Double(layout.width)), normEpsilon, Float(workspace.nextSlot()),
            ])
            let template: [(String, any KernelTemplateArg)] = [
                ("T", dtype), ("Width", layout.width), ("Threads", threads),
                ("PerThread", perThread), ("Iterations", sinkhornIterations),
                ("MaxRows", splitCollapseMaximumRows),
            ]
            let grid = (threads * oneLaunchGroups, layout.rows, 1)
            let rowShape = [layout.batch, layout.tokens, layout.width]
            let coefficientShapes = [
                [layout.batch, layout.tokens, 4], [layout.batch, layout.tokens, 4, 4],
            ]
            switch source {
            case .streams(let input):
                let outputs = oneLaunchKernel(
                    [
                        input, fn, base, scale, normWeight, params, workspace.counters,
                        workspace.partials,
                    ],
                    template: template, grid: grid, threadGroup: (threads, 1, 1),
                    outputShapes: [rowShape] + coefficientShapes,
                    outputDTypes: [dtype, .float32, .float32])
                return NormalizedCollapse(
                    streams: input, post: outputs[1], comb: outputs[2], normalized: outputs[0])
            case .expand(let output, let residual, let post, let comb):
                let outputs = expandOneLaunchKernel(
                    [
                        output, residual, post, comb, fn, base, scale, normWeight, params,
                        workspace.counters, workspace.partials,
                    ],
                    template: template, grid: grid, threadGroup: (threads, 1, 1),
                    outputShapes: [rowShape, residual.shape] + coefficientShapes,
                    outputDTypes: [dtype, dtype, .float32, .float32])
                return NormalizedCollapse(
                    streams: outputs[1], post: outputs[2], comb: outputs[3],
                    normalized: outputs[0])
            }
        }

        /// `GLM5_HC_SITE_EXPAND` selects the source: the streams input, or the
        /// folded expand (`expandLoad`'s arithmetic; phase 1 also stores it).
        private static let oneLaunchSource = """
            #pragma clang fp reassociate(off)
            #pragma clang fp contract(off)
            constexpr uint Groups = uint(Threads) / 32;
            constexpr uint Sums = 25;
            threadgroup float staged[4][PerThread][32];
            threadgroup float gathered[Sums][Groups];
            threadgroup float totals[Sums];
            threadgroup float partial[Groups];
            threadgroup uint arrival[1];
            const float epsilon = params[0];
            const float inverse_channels = params[1];
            const float rms_epsilon = params[2];
            const float inverse_width = params[3];
            const float norm_epsilon = params[4];
            const uint slot = uint(params[5]) & 1u;
            const uint row = threadgroup_position_in_grid.y;
            // The split form's SIMD group whose threads this threadgroup stands for.
            const uint tile = threadgroup_position_in_grid.x;
            const uint tid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint group = simdgroup_index_in_threadgroup;
            const size_t offset = size_t(row) * 4 * Width;
            device atomic_uint* counter =
                reinterpret_cast<device atomic_uint*>(counters) + slot * uint(MaxRows) + row;
            device atomic_uint* spare =
                reinterpret_cast<device atomic_uint*>(counters) + (1u - slot) * uint(MaxRows) + row;
            device atomic_float* row_partials =
                reinterpret_cast<device atomic_float*>(partials) + size_t(row) * Sums * Groups;
            #if GLM5_HC_SITE_EXPAND
            // `output` can be a CPU-stream sum: wait for it here, not in a
            // separate MLX fence-wait launch (a no-op unless MLX enables it).
            MLX_FENCED_INPUT_WAIT(output);
            #endif

            // Phase 1. SIMD group `stream` < 4 stages that stream's values of
            // the split form's threads 32 * tile + lane.
            if (group < 4) {
                const uint stream = group;
                for (uint k = 0; k < uint(PerThread); ++k) {
                    const uint column = 32 * tile + lane + k * uint(Threads);
                    float value = 0.0f;
                    if (column < uint(Width)) {
            #if GLM5_HC_SITE_EXPAND
                        const T injected_value = output[size_t(row) * Width + column];
                        const T coefficient = static_cast<T>(post[size_t(row) * 4 + stream]);
                        const T injected = static_cast<T>(coefficient * injected_value);
                        float mix = 0.0f;
                        for (uint source = 0; source < 4; ++source) {
                            const T weight = static_cast<T>(comb[(size_t(row) * 4 + source) * 4 + stream]);
                            const T residual_value = residual[offset + size_t(source) * Width + column];
                            const float product = static_cast<float>(weight) * static_cast<float>(residual_value);
                            mix += product;
                        }
                        const T mixed = static_cast<T>(mix);
                        const T expanded_value = static_cast<T>(injected + mixed);
                        expanded[offset + size_t(stream) * Width + column] = expanded_value;
                        value = static_cast<float>(expanded_value);
            #else
                        value = static_cast<float>(streams[offset + size_t(stream) * Width + column]);
            #endif
                    }
                    staged[stream][k][lane] = value;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // SIMD group s < 24 sums fn row s, SIMD group 24 the squares, each in
            // the split form's (column, stream) order.
            if (group < Sums) {
                float sum = 0.0f;
                for (uint k = 0; k < uint(PerThread); ++k) {
                    const uint column = 32 * tile + lane + k * uint(Threads);
                    for (uint stream = 0; stream < 4; ++stream) {
                        if (column < uint(Width)) {
                            const float value = staged[stream][k][lane];
                            if (group + 1 == Sums) {
                                const float square = value * value;
                                sum += square;
                            } else {
                                const size_t channel = size_t(stream) * Width + column;
                                const float product =
                                    value * static_cast<float>(fn[size_t(group) * 4 * Width + channel]);
                                sum += product;
                            }
                        }
                    }
                }
                const float total = simd_sum(sum);
                if (lane == 0) {
                    // Publish the SIMD total before this threadgroup counts in.
                    atomic_store_explicit(&row_partials[group * Groups + tile], total, memory_order_relaxed);
                    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
                }
            }
            // Count this threadgroup in. Only the row's last threadgroup to
            // arrive continues; no threadgroup waits for another.
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            if (tid == 0) {
                atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
                arrival[0] = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
                atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            if (arrival[0] != Groups - 1) {
                return;
            }

            // Phase 2, the row's last threadgroup: every count is in, so clear
            // both slots; gather the 25 x 32 SIMD totals in one load each, then
            // add each sum's totals in SIMD-group order, as the split form does.
            if (tid == 0) {
                atomic_store_explicit(counter, 0u, memory_order_relaxed);
                atomic_store_explicit(spare, 0u, memory_order_relaxed);
            }
            if (tid < Sums * Groups) {
                gathered[tid / Groups][tid % Groups] =
                    atomic_load_explicit(&row_partials[tid], memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < Sums) {
                float total = 0.0f;
                for (uint g = 0; g < Groups; ++g) { total += gathered[tid][g]; }
                totals[tid] = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float inverse_rms = metal::precise::rsqrt(totals[Sums - 1] * inverse_channels + rms_epsilon);
            float collapsed[2][PerThread];
            if (group == 0) {
                // `postCombKernel`'s coefficient stage (its totals are rows 4...23).
                const bool owns_row = lane < 4;
                const uint matrix_row = owns_row ? lane : 0;
                float post_logit = 0.0f;
                float logits[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                if (owns_row) {
                    const float post_projected = totals[4 + matrix_row] * inverse_rms;
                    const float post_product = post_projected * static_cast<float>(scale[1]);
                    post_logit = post_product + static_cast<float>(base[4 + matrix_row]);
                    for (int j = 0; j < 4; ++j) {
                        const float projected = totals[8 + matrix_row * 4 + j] * inverse_rms;
                        const float product = projected * static_cast<float>(scale[2]);
                        logits[j] = product + static_cast<float>(base[8 + matrix_row * 4 + j]);
                    }
                }
                float pre_value;
                float post_value;
                float c[4];
                glm5_hc4_coefficients(
                    owns_row, 0.0f, post_logit, logits, epsilon, Iterations,
                    pre_value, post_value, c);
                if (owns_row) {
                    post_out[size_t(row) * 4 + matrix_row] = post_value;
                    for (int j = 0; j < 4; ++j) {
                        comb_out[(size_t(row) * 4 + matrix_row) * 4 + j] = c[j];
                    }
                }
            } else {
                // `preCollapseSource`'s tail for the split form's SIMD groups
                // `group` and, on SIMD group 1, also 0.
                float pre[4];
                for (uint s = 0; s < 4; ++s) {
                    const float projected = totals[s] * inverse_rms;
                    const float product = projected * static_cast<float>(scale[0]);
                    const float logit = product + static_cast<float>(base[s]);
                    pre[s] = glm5_hc_sigmoid(logit) + epsilon;
                }
                const uint passes = group == 1 ? 2u : 1u;
                for (uint pass = 0; pass < passes; ++pass) {
                    const uint tree_group = pass == 0 ? group : 0u;
                    float squares = 0.0f;
                    for (uint k = 0; k < uint(PerThread); ++k) {
                        const uint column = 32 * tree_group + lane + k * uint(Threads);
                        collapsed[pass][k] = 0.0f;
                        if (column < uint(Width)) {
                            float values[4];
                            for (uint stream = 0; stream < 4; ++stream) {
            #if GLM5_HC_SITE_EXPAND
                                const T injected_value = output[size_t(row) * Width + column];
                                const T coefficient = static_cast<T>(post[size_t(row) * 4 + stream]);
                                const T injected = static_cast<T>(coefficient * injected_value);
                                float mix = 0.0f;
                                for (uint source = 0; source < 4; ++source) {
                                    const T weight = static_cast<T>(comb[(size_t(row) * 4 + source) * 4 + stream]);
                                    const T residual_value = residual[offset + size_t(source) * Width + column];
                                    const float product = static_cast<float>(weight) * static_cast<float>(residual_value);
                                    mix += product;
                                }
                                const T mixed = static_cast<T>(mix);
                                const T expanded_value = static_cast<T>(injected + mixed);
                                values[stream] = static_cast<float>(expanded_value);
            #else
                                values[stream] = static_cast<float>(streams[offset + size_t(stream) * Width + column]);
            #endif
                            }
                            float total = 0.0f;
                            for (uint s = 0; s < 4; ++s) {
                                const float product = pre[s] * values[s];
                                total += product;
                            }
                            collapsed[pass][k] = static_cast<float>(static_cast<T>(total));
                            const float square = collapsed[pass][k] * collapsed[pass][k];
                            squares += square;
                        }
                    }
                    const float simd_total = simd_sum(squares);
                    if (lane == 0) { partial[tree_group] = simd_total; }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (group != 0) {
                float sum = 0.0f;
                for (uint g = 0; g < Groups; ++g) { sum += partial[g]; }
                const float inverse = metal::precise::rsqrt(sum * inverse_width + norm_epsilon);
                const uint passes = group == 1 ? 2u : 1u;
                for (uint pass = 0; pass < passes; ++pass) {
                    const uint tree_group = pass == 0 ? group : 0u;
                    for (uint k = 0; k < uint(PerThread); ++k) {
                        const uint column = 32 * tree_group + lane + k * uint(Threads);
                        if (column < uint(Width)) {
                            const T scaled = static_cast<T>(collapsed[pass][k] * inverse);
                            const T weight = static_cast<T>(norm_weight[column]);
                            normalized[size_t(row) * Width + column] = static_cast<T>(weight * scaled);
                        }
                    }
                }
            }
            """

        private static let oneLaunchKernel = MLXFast.metalKernel(
            name: "glm5_next_hc4_site_collapse_norm",
            inputNames: [
                "streams", "fn", "base", "scale", "norm_weight", "params", "counters", "partials",
            ],
            outputNames: ["normalized", "post_out", "comb_out"],
            source: "#define GLM5_HC_SITE_EXPAND 0\n" + oneLaunchSource,
            header: coefficientHeader, ensureRowContiguous: true,
            mutableInputs: ["counters", "partials"])

        /// `output`, the branch output, is the tensor-parallel sum of the
        /// ranks' partials: a fenced input (see `MLXFast.metalKernel(...fencedInputs:)`).
        private static let expandOneLaunchKernel = MLXFast.metalKernel(
            name: "glm5_next_hc4_site_expand_collapse_norm",
            inputNames: [
                "output", "residual", "post", "comb", "fn", "base", "scale", "norm_weight",
                "params", "counters", "partials",
            ],
            outputNames: ["normalized", "expanded", "post_out", "comb_out"],
            source: "#define GLM5_HC_SITE_EXPAND 1\n" + oneLaunchSource,
            header: coefficientHeader, ensureRowContiguous: true,
            mutableInputs: ["counters", "partials"], fencedInputs: ["output"])
    #endif
}
