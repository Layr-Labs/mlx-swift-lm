// Copyright 2026 Eigen Labs.
// Modified in 2026: one-launch prefill form of the normalized mHC collapse,
// adapted from oMLX "HC prefill kernels (batch-invariant)" (hc_prefill.py,
// Apache-2.0): row tiles per 1024-thread threadgroup, the 24-wide mix on FP32
// 8x8 SIMD-group matrices over a fixed K split with a sequential partial sum,
// and the coefficient stage and collapse in the same kernel.
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

/// Prefill steps (more than `splitCollapseMaximumRows` rows) of
/// `collapseNormalized`: one launch per site. A threadgroup owns `TileRows`
/// rows and, for an `.expand` source, first stores their expanded streams
/// with `expandKernel`'s arithmetic, then reads them back (from cache).
///
/// Fixed reduction tree, independent of the row count, tile height and chunk
/// offset, so chunked and whole prefills agree bitwise for every row:
/// the channels split into 32 slices of `Slice` (a multiple of 8); a slice's
/// mix sums accumulate 8 channels per SIMD-group matrix product, in channel
/// order; its sum of squares accumulates per lane in channel order, then
/// across the four lanes sharing a row as (c0 + c2) + (c4 + c6); the 32 slice
/// partials add sequentially. Products of BF16 streams and BF16 weights are
/// exact in FP32; the matrix unit's internal order of each 8-term sum is not
/// specified, so this tree is deterministic but differs from every other
/// form (split decode path, `collapse`, the reference).
extension GLM5NextHyperConnectionKernel {
    /// Rows per prefill threadgroup (measured on M5 Ultra at hidden 4096:
    /// 16 is fastest up to 2048 rows; 8 keeps each tile's re-read of its
    /// streams in cache at 8192). Results do not depend on this choice.
    static func prefillTileRows(rows: Int) -> Int { rows >= 4096 ? 8 : 16 }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        static func prefillCollapse(
            _ source: StreamsSource, layout: FusedLayout, fn: MLXArray, base: MLXArray,
            scale: MLXArray, rmsNormEpsilon: Float, epsilon: Float, sinkhornIterations: Int,
            normWeight: MLXArray, normEpsilon: Float, dtype: DType
        ) -> NormalizedCollapse {
            let tileRows = prefillTileRows(rows: layout.rows)
            let params = MLXArray([
                epsilon, Float(1.0 / Double(layout.channels)), rmsNormEpsilon,
                Float(1.0 / Double(layout.width)), normEpsilon,
            ])
            let template: [(String, any KernelTemplateArg)] = [
                ("T", dtype), ("Width", layout.width), ("Channels", layout.channels),
                ("Slice", (layout.channels + 255) / 256 * 8), ("TileRows", tileRows),
                ("Iterations", sinkhornIterations),
            ]
            let groups = (layout.rows + tileRows - 1) / tileRows
            let grid = (groups * prefillThreads, 1, 1)
            let rowCount = MLXArray(Int32(layout.rows))
            let coefficientShapes = [
                [layout.batch, layout.tokens, 4], [layout.batch, layout.tokens, 4, 4],
                [layout.batch, layout.tokens, layout.width],
            ]
            switch source {
            case .streams(let streams):
                let outputs = prefillKernel(
                    [streams, fn, base, scale, normWeight, params, rowCount],
                    template: template, grid: grid, threadGroup: (prefillThreads, 1, 1),
                    outputShapes: coefficientShapes,
                    outputDTypes: [.float32, .float32, dtype])
                return NormalizedCollapse(
                    streams: streams, post: outputs[0], comb: outputs[1],
                    normalized: outputs[2])
            case .expand(let output, let residual, let post, let comb):
                let outputs = expandPrefillKernel(
                    [
                        output, residual, post, comb, fn, base, scale, normWeight, params,
                        rowCount,
                    ],
                    template: template, grid: grid, threadGroup: (prefillThreads, 1, 1),
                    outputShapes: [residual.shape] + coefficientShapes,
                    outputDTypes: [dtype, .float32, .float32, dtype])
                return NormalizedCollapse(
                    streams: outputs[0], post: outputs[1], comb: outputs[2],
                    normalized: outputs[3])
            }
        }

        private static let prefillThreads = 1024

        private static let prefillHeader =
            "#include <metal_simdgroup_matrix>\n" + coefficientHeader

        // The expand arithmetic is `expandKernel`'s, element for element. The
        // device barrier publishes the stored streams to the whole threadgroup.
        private static let expandStage = """
            for (uint m = 0; m < uint(TileRows); ++m) {
                const uint row = first_row + m;
                if (row >= uint(row_count)) { break; }
                const size_t offset = size_t(row) * Channels;
                for (uint column = tid; column < uint(Width); column += Threads) {
                    const T value = output[size_t(row) * Width + column];
                    T sources[4];
                    for (int source = 0; source < 4; ++source) {
                        sources[source] = residual[offset + source * Width + column];
                    }
                    for (int destination = 0; destination < 4; ++destination) {
                        const T coefficient =
                            static_cast<T>(previous_post[size_t(row) * 4 + destination]);
                        const T injected = static_cast<T>(coefficient * value);
                        float total = 0.0f;
                        for (int source = 0; source < 4; ++source) {
                            const T weight = static_cast<T>(
                                previous_comb[(size_t(row) * 4 + source) * 4 + destination]);
                            const float product =
                                static_cast<float>(weight) * static_cast<float>(sources[source]);
                            total += product;
                        }
                        const T mixed = static_cast<T>(total);
                        expanded[offset + destination * Width + column] =
                            static_cast<T>(injected + mixed);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_device);
            const device T* streams = expanded;

            """

        /// `stage` defines `streams`, the [rows, Channels] collapse input.
        private static func prefillSource(stage: String) -> String {
            """
            #pragma clang fp reassociate(off)
            #pragma clang fp contract(off)
            constexpr uint Threads = 1024;
            constexpr uint Groups = Threads / 32;
            constexpr uint Sums = 25;
            constexpr uint RowTiles = uint(TileRows) / 8;
            // Slice partials staged per round: Buffers x TileRows x Sums floats.
            constexpr uint Buffers = 256 / uint(TileRows);
            constexpr uint PerThread = (uint(Width) + Threads - 1) / Threads;
            threadgroup float part[Buffers][TileRows][Sums];
            threadgroup float raw[TileRows][Sums];
            threadgroup float pre_weights[TileRows][4];
            threadgroup float simd_totals[2][Groups];
            const float epsilon = params[0];
            const float inverse_channels = params[1];
            const float rms_epsilon = params[2];
            const float inverse_width = params[3];
            const float norm_epsilon = params[4];
            const uint tid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint group = simdgroup_index_in_threadgroup;
            const uint first_row = threadgroup_position_in_grid.x * uint(TileRows);
            \(stage)
            // Raw mix sums and sums of squares: SIMD group g owns slice g. The
            // 8x8 fragment of lane l holds (fragment_row, fragment_column + 0/1).
            const uint quad = lane / 4;
            const uint fragment_row = (quad & 4) + ((lane / 2) % 4);
            const uint fragment_column = (quad & 2) * 2 + (lane % 2) * 2;
            simdgroup_matrix<float, 8, 8> sums[RowTiles][3];
            float squares[RowTiles];
            bool valid[RowTiles];
            size_t row_offset[RowTiles];
            for (uint t = 0; t < RowTiles; ++t) {
                const uint row = first_row + t * 8 + fragment_row;
                valid[t] = row < uint(row_count);
                row_offset[t] = size_t(valid[t] ? row : first_row) * Channels;
                squares[t] = 0.0f;
                for (uint j = 0; j < 3; ++j) {
                    sums[t][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                }
            }
            const uint begin = group * uint(Slice);
            const uint end = min(begin + uint(Slice), uint(Channels));
            for (uint k = begin; k < end; k += 8) {
                const uint weight_channel = k + fragment_row;
                simdgroup_matrix<float, 8, 8> weights[3];
                for (uint j = 0; j < 3; ++j) {
                    float2 w = float2(0.0f);
                    if (weight_channel < uint(Channels)) {
                        const uint mix_row = 8 * j + fragment_column;
                        w.x = static_cast<float>(fn[size_t(mix_row) * Channels + weight_channel]);
                        w.y = static_cast<float>(fn[size_t(mix_row + 1) * Channels + weight_channel]);
                    }
                    reinterpret_cast<thread float2&>(weights[j].thread_elements()) = w;
                }
                const uint channel = k + fragment_column;
                for (uint t = 0; t < RowTiles; ++t) {
                    float2 a = float2(0.0f);
                    if (valid[t] && channel < uint(Channels)) {
                        a.x = static_cast<float>(streams[row_offset[t] + channel]);
                        a.y = static_cast<float>(streams[row_offset[t] + channel + 1]);
                    }
                    const float square_x = a.x * a.x;
                    squares[t] += square_x;
                    const float square_y = a.y * a.y;
                    squares[t] += square_y;
                    simdgroup_matrix<float, 8, 8> values;
                    reinterpret_cast<thread float2&>(values.thread_elements()) = a;
                    for (uint j = 0; j < 3; ++j) {
                        simdgroup_multiply_accumulate(sums[t][j], values, weights[j], sums[t][j]);
                    }
                }
            }
            for (uint t = 0; t < RowTiles; ++t) {
                const float pair = squares[t] + simd_shuffle_xor(squares[t], ushort(1));
                squares[t] = pair + simd_shuffle_xor(pair, ushort(8));
            }
            // Slice partials in slice order, Buffers slices per round.
            const uint sum_row = tid / Sums;
            const uint sum_index = tid % Sums;
            float total = 0.0f;
            for (uint round = 0; round < Groups / Buffers; ++round) {
                if (group / Buffers == round) {
                    const uint buffer = group % Buffers;
                    for (uint t = 0; t < RowTiles; ++t) {
                        const uint m = t * 8 + fragment_row;
                        for (uint j = 0; j < 3; ++j) {
                            const float2 e =
                                reinterpret_cast<thread float2&>(sums[t][j].thread_elements());
                            part[buffer][m][8 * j + fragment_column] = e.x;
                            part[buffer][m][8 * j + fragment_column + 1] = e.y;
                        }
                        if ((lane & 9) == 0) { part[buffer][m][24] = squares[t]; }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < uint(TileRows) * Sums) {
                    for (uint buffer = 0; buffer < Buffers; ++buffer) {
                        total += part[buffer][sum_row][sum_index];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid < uint(TileRows) * Sums) { raw[sum_row][sum_index] = total; }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // SIMD group m: the RMS factor on each raw sum and the coefficients.
            if (group < uint(TileRows) && first_row + group < uint(row_count)) {
                const uint m = group;
                const uint row = first_row + m;
                const float mean = raw[m][24] * inverse_channels;
                const float inverse_rms = metal::precise::rsqrt(mean + rms_epsilon);
                const bool owns_row = lane < 4;
                const uint matrix_row = owns_row ? lane : 0;
                float pre_logit = 0.0f;
                float post_logit = 0.0f;
                float logits[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                if (owns_row) {
                    const float pre_projected = raw[m][matrix_row] * inverse_rms;
                    const float pre_product = pre_projected * static_cast<float>(scale[0]);
                    pre_logit = pre_product + static_cast<float>(base[matrix_row]);
                    const float post_projected = raw[m][4 + matrix_row] * inverse_rms;
                    const float post_product = post_projected * static_cast<float>(scale[1]);
                    post_logit = post_product + static_cast<float>(base[4 + matrix_row]);
                    for (int j = 0; j < 4; ++j) {
                        const float projected = raw[m][8 + matrix_row * 4 + j] * inverse_rms;
                        const float product = projected * static_cast<float>(scale[2]);
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
                    pre_weights[m][matrix_row] = pre_value;
                    post[size_t(row) * 4 + matrix_row] = post_value;
                    for (int j = 0; j < 4; ++j) {
                        comb[(size_t(row) * 4 + matrix_row) * 4 + j] = c[j];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Per row: the weighted collapse rounded to the activation dtype,
            // then RMSNorm (SIMD sums, then the SIMD partials in order).
            for (uint m = 0; m < uint(TileRows); ++m) {
                const uint row = first_row + m;
                if (row >= uint(row_count)) { break; }
                const size_t offset = size_t(row) * Channels;
                float collapsed[PerThread];
                float collapsed_squares = 0.0f;
                for (uint step = 0; step < PerThread; ++step) {
                    const uint column = step * Threads + tid;
                    collapsed[step] = 0.0f;
                    if (column < uint(Width)) {
                        float weighted = 0.0f;
                        for (int stream = 0; stream < 4; ++stream) {
                            const float product = pre_weights[m][stream]
                                * static_cast<float>(streams[offset + stream * Width + column]);
                            weighted += product;
                        }
                        collapsed[step] = static_cast<float>(static_cast<T>(weighted));
                        const float square = collapsed[step] * collapsed[step];
                        collapsed_squares += square;
                    }
                }
                const float simd_total = simd_sum(collapsed_squares);
                if (lane == 0) { simd_totals[m & 1][group] = simd_total; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float sum = 0.0f;
                for (uint i = 0; i < Groups; ++i) { sum += simd_totals[m & 1][i]; }
                const float norm_mean = sum * inverse_width;
                const float inverse = metal::precise::rsqrt(norm_mean + norm_epsilon);
                for (uint step = 0; step < PerThread; ++step) {
                    const uint column = step * Threads + tid;
                    if (column < uint(Width)) {
                        const T scaled = static_cast<T>(collapsed[step] * inverse);
                        const T weight = static_cast<T>(norm_weight[column]);
                        normalized[size_t(row) * Width + column] = static_cast<T>(weight * scaled);
                    }
                }
            }
            """
        }

        private static let prefillKernel = MLXFast.metalKernel(
            name: "glm5_next_hc4_prefill_collapse_norm",
            inputNames: ["streams", "fn", "base", "scale", "norm_weight", "params", "row_count"],
            outputNames: ["post", "comb", "normalized"],
            source: prefillSource(stage: ""), header: prefillHeader, ensureRowContiguous: true)

        private static let expandPrefillKernel = MLXFast.metalKernel(
            name: "glm5_next_hc4_expand_prefill_collapse_norm",
            inputNames: [
                "output", "residual", "previous_post", "previous_comb", "fn", "base", "scale",
                "norm_weight", "params", "row_count",
            ],
            outputNames: ["expanded", "post", "comb", "normalized"],
            source: prefillSource(stage: expandStage), header: prefillHeader,
            ensureRowContiguous: true)
    #endif
}
