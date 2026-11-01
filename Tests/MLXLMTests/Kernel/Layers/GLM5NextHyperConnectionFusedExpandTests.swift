// Copyright 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    /// The prefill hyper-connection site after the fused expand
    /// (`fusedExpandPrefillSource`, and the two-pass form rebuilt from shared
    /// pieces) against the site's kernel text before it, embedded verbatim
    /// (`Original`): every bit of the expanded streams, post, comb and
    /// normalized rows, over the rank's width (4,096) and others (stream
    /// boundaries inside a slice, tiny widths), bf16/f16/f32 activations,
    /// bf16 and f32 `fn`, serving micro-batch sizes (8,192, 4,096, the
    /// 1,414-row tail) and odd row counts, the `.streams` form, and
    /// micro-batch splits of one call. Plus an opt-in timing report
    /// (`GLM53_HC_FUSED_EXPAND_BENCH=1`).
    @Suite(.serialized)
    struct GLM5NextHyperConnectionFusedExpandTests {
        typealias Kernel = GLM5NextHyperConnectionKernel

        /// The prefill site's kernel text before the fused expand (SDK
        /// c2f2faf6, `prefillSource(stage:)` and `expandStage`), verbatim:
        /// the oracle every production form must equal bit for bit.
        enum Original {
            static let expandStage = """
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

            static func source(stage: String) -> String {
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

            static let header =
                "#include <metal_simdgroup_matrix>\n"
                + GLM5NextHyperConnectionKernel.coefficientHeader

            static let streamsKernel = MLXFast.metalKernel(
                name: "glm5_next_hc4_prefill_collapse_norm_c2f2faf6",
                inputNames: [
                    "streams", "fn", "base", "scale", "norm_weight", "params", "row_count",
                ],
                outputNames: ["post", "comb", "normalized"], source: source(stage: ""),
                header: header, ensureRowContiguous: true)

            static let expandKernel = MLXFast.metalKernel(
                name: "glm5_next_hc4_expand_prefill_collapse_norm_c2f2faf6",
                inputNames: [
                    "output", "residual", "previous_post", "previous_comb", "fn", "base", "scale",
                    "norm_weight", "params", "row_count",
                ],
                outputNames: ["expanded", "post", "comb", "normalized"],
                source: source(stage: expandStage), header: header, ensureRowContiguous: true)
        }

        struct Site {
            let output: MLXArray
            let residual: MLXArray
            let post: MLXArray
            let comb: MLXArray
            let fn: MLXArray
            let base: MLXArray
            let scale: MLXArray
            let weight: MLXArray

            init(rows: Int, width: Int, dtype: DType, fnType: DType = .bfloat16, seed: UInt64) {
                let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 8)
                residual = MLXRandom.normal([1, rows, 4, width], key: keys[0]).asType(dtype)
                output = MLXRandom.normal([1, rows, width], key: keys[1]).asType(dtype)
                fn = (MLXRandom.normal([24, 4 * width], key: keys[2]) * 0.02).asType(fnType)
                base = MLXRandom.normal([24], key: keys[3]) * 0.5
                scale = MLXRandom.uniform(low: Float(0.5), high: 1.5, [3], key: keys[4])
                weight = MLXRandom.uniform(low: Float(0.5), high: 1.5, [width], key: keys[5])
                    .asType(dtype)
                post = MLXRandom.uniform(low: Float(0), high: 2, [1, rows, 4], key: keys[6])
                comb = MLXRandom.uniform(low: Float(0), high: 0.5, [1, rows, 4, 4], key: keys[7])
                eval(residual, output, fn, base, scale, weight, post, comb)
            }

            func rows(_ range: Range<Int>) -> Site {
                Site(
                    output: output[0..., range], residual: residual[0..., range],
                    post: post[0..., range], comb: comb[0..., range], fn: fn, base: base,
                    scale: scale, weight: weight)
            }

            private init(
                output: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray,
                fn: MLXArray, base: MLXArray, scale: MLXArray, weight: MLXArray
            ) {
                (self.output, self.residual, self.post, self.comb) = (output, residual, post, comb)
                (self.fn, self.base, self.scale, self.weight) = (fn, base, scale, weight)
            }

            /// `Original`'s expand (or, with `streams`, collapse-only) kernel
            /// at the tile height the site had: expanded streams (the input
            /// streams for `streams`), post, comb, normalized.
            func original(streams: Bool = false) throws -> [MLXArray] {
                let layout = try Kernel.FusedLayout(streamsShape: residual.shape)
                let tileRows = Kernel.prefillTileRows(rows: layout.rows)
                let params = MLXArray([
                    Float(1e-6), Float(1.0 / Double(layout.channels)), 1e-5,
                    Float(1.0 / Double(layout.width)), 1e-5,
                ])
                let template: [(String, any KernelTemplateArg)] = [
                    ("T", residual.dtype), ("Width", layout.width), ("Channels", layout.channels),
                    ("Slice", (layout.channels + 255) / 256 * 8), ("TileRows", tileRows),
                    ("Iterations", 20),
                ]
                let grid = ((layout.rows + tileRows - 1) / tileRows * 1024, 1, 1)
                let rowCount = MLXArray(Int32(layout.rows))
                let shapes = [
                    [1, layout.rows, 4], [1, layout.rows, 4, 4], [1, layout.rows, layout.width],
                ]
                if streams {
                    return [residual]
                        + Original.streamsKernel(
                            [residual, fn, base, scale, weight, params, rowCount],
                            template: template, grid: grid, threadGroup: (1024, 1, 1),
                            outputShapes: shapes,
                            outputDTypes: [.float32, .float32, residual.dtype])
                }
                return Original.expandKernel(
                    [output, residual, post, comb, fn, base, scale, weight, params, rowCount],
                    template: template, grid: grid, threadGroup: (1024, 1, 1),
                    outputShapes: [residual.shape] + shapes,
                    outputDTypes: [residual.dtype, .float32, .float32, residual.dtype])
            }

            /// The production `.streams` site.
            func streamsSite() throws -> [MLXArray] {
                let result = try Kernel.collapseNormalized(
                    .streams(residual), fn: fn, base: base, scale: scale, rmsNormEpsilon: 1e-5,
                    epsilon: 1e-6, sinkhornIterations: 20, normWeight: weight, normEpsilon: 1e-5)
                return [result.streams, result.post, result.comb, result.normalized]
            }

            /// Expanded streams, post, comb, normalized.
            func run(fused: Bool) throws -> [MLXArray] {
                let layout = try Kernel.FusedLayout(streamsShape: residual.shape)
                let result = Kernel.prefillCollapse(
                    .expand(output: output, residual: residual, post: post, comb: comb),
                    layout: layout, fn: fn, base: base, scale: scale, rmsNormEpsilon: 1e-5,
                    epsilon: 1e-6, sinkhornIterations: 20, normWeight: weight, normEpsilon: 1e-5,
                    dtype: residual.dtype, fusedExpand: fused)
                return [result.streams, result.post, result.comb, result.normalized]
            }
        }

        /// Bit patterns (signed zeros and NaN payloads included).
        static func bitwise(_ a: MLXArray, _ b: MLXArray) -> Bool {
            let bits: DType = a.dtype == .float32 ? .uint32 : .uint16
            return a.dtype == b.dtype && a.shape == b.shape
                && arrayEqual(a.view(dtype: bits), b.view(dtype: bits)).item(Bool.self)
        }

        static let outputs = ["streams", "post", "comb", "normalized"]

        @Test func prefillSitesAreTheOriginalKernelBitForBit() throws {
            let cases: [(rows: Int, width: Int, dtype: DType, fnType: DType)] = [
                (8192, 4096, .bfloat16, .bfloat16), (4096, 4096, .bfloat16, .bfloat16),
                (1414, 4096, .bfloat16, .bfloat16), (2051, 4096, .bfloat16, .bfloat16),
                (9, 4096, .bfloat16, .bfloat16), (4096, 4096, .float16, .bfloat16),
                (1031, 4096, .float32, .float32), (4100, 1000, .bfloat16, .bfloat16),
                (37, 1000, .float16, .float32), (333, 96, .bfloat16, .bfloat16),
                (17, 96, .float32, .bfloat16), (64, 8, .bfloat16, .bfloat16),
                (11, 8, .float32, .float32),
            ]
            try Device.withDefaultDevice(.gpu) {
                for (index, item) in cases.enumerated() {
                    #expect(Kernel.takesFusedExpand(width: item.width) == Kernel.fusedExpandEnabled)
                    let site = Site(
                        rows: item.rows, width: item.width, dtype: item.dtype, fnType: item.fnType,
                        seed: UInt64(17 + index))
                    let expected = try site.original()
                    let fused = try site.run(fused: true)
                    let twoPass = try site.run(fused: false)
                    eval(expected + fused + twoPass)
                    let label =
                        "rows \(item.rows) width \(item.width) \(item.dtype) fn \(item.fnType)"
                    for (name, (actual, reference)) in zip(Self.outputs, zip(fused, expected)) {
                        #expect(Self.bitwise(actual, reference), "\(label) fused \(name)")
                    }
                    for (name, (actual, reference)) in zip(Self.outputs, zip(twoPass, expected)) {
                        #expect(Self.bitwise(actual, reference), "\(label) two-pass \(name)")
                    }
                    if index % 3 == 0 {
                        let expectedStreams = try site.original(streams: true)
                        let streams = try site.streamsSite()
                        eval(expectedStreams + streams)
                        for (name, (actual, reference)) in zip(
                            Self.outputs, zip(streams, expectedStreams))
                        {
                            #expect(Self.bitwise(actual, reference), "\(label) streams \(name)")
                        }
                    }
                    if index == 0 {
                        // Control: the comparison sees a one-ulp change.
                        let changed = expected[3].view(dtype: .uint16) + MLXArray(UInt16(1))
                        #expect(!Self.bitwise(fused[3], changed.view(dtype: .bfloat16)), "control")
                    }
                    Memory.clearCache()
                }
            }
        }

        /// Rows of one call equal the same rows in calls over parts of it:
        /// the two serving micro-batches of an 8,192-token chunk, an uneven
        /// cut, and the 1,414-row tail beside its 6,778-row head.
        @Test func fusedExpandRowsDoNotDependOnTheCall() throws {
            try Device.withDefaultDevice(.gpu) {
                let site = Site(rows: 8192, width: 4096, dtype: .bfloat16, seed: 5)
                let whole = try site.run(fused: true)
                eval(whole)
                for cut in [4096, 4095, 1414] {
                    let parts = [0 ..< cut, cut ..< 8192]
                    for part in parts {
                        let piece = try site.rows(part).run(fused: true)
                        eval(piece)
                        for (name, pair) in zip(Self.outputs, zip(piece, whole)) {
                            #expect(
                                Self.bitwise(pair.0, pair.1[0..., part]),
                                "cut \(cut) rows \(part) \(name)")
                        }
                    }
                }
            }
        }

        // MARK: Timing (opt-in)

        /// Best wall time per site over chains of 16 dependent `.expand`
        /// sites (each takes the previous site's streams, post, comb and
        /// normalized rows), two-pass against fused, at the rank's width.
        /// "per 4,096" scales to 89 expand sites per 4,096 rows (45 layers,
        /// two sites each, the first one from materialized streams).
        @Test(
            .enabled(
                if: ProcessInfo.processInfo.environment["GLM53_HC_FUSED_EXPAND_BENCH"] == "1"))
        func timing() throws {
            try Device.withDefaultDevice(.gpu) {
                func best(_ body: () throws -> [MLXArray]) rethrows -> Double {
                    var best = Double.infinity
                    for _ in 0 ..< 6 {
                        let start = DispatchTime.now().uptimeNanoseconds
                        eval(try body())
                        best = min(best, Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
                    }
                    return best
                }
                for rows in [4096, 8192, 1414] {
                    let site = Site(rows: rows, width: 4096, dtype: .bfloat16, seed: 7)
                    let layout = try Kernel.FusedLayout(streamsShape: site.residual.shape)
                    func chain(fused: Bool) -> [MLXArray] {
                        var (output, residual, post, comb) = (
                            site.output, site.residual, site.post, site.comb
                        )
                        var last: [MLXArray] = []
                        for _ in 0 ..< 16 {
                            let r = Kernel.prefillCollapse(
                                .expand(output: output, residual: residual, post: post, comb: comb),
                                layout: layout, fn: site.fn, base: site.base, scale: site.scale,
                                rmsNormEpsilon: 1e-5, epsilon: 1e-6, sinkhornIterations: 20,
                                normWeight: site.weight, normEpsilon: 1e-5, dtype: .bfloat16,
                                fusedExpand: fused)
                            (output, residual, post, comb) = (
                                r.normalized, r.streams, r.post, r.comb
                            )
                            last = [r.streams, r.post, r.comb, r.normalized]
                        }
                        return last
                    }
                    let floor = best { [MLXArray.zeros([1]) + 1] }
                    var old = Double.infinity
                    var new = Double.infinity
                    for _ in 0 ..< 4 {
                        old = min(old, best { chain(fused: false) })
                        new = min(new, best { chain(fused: true) })
                    }
                    let (o, n) = ((old - floor) / 16, (new - floor) / 16)
                    let per4096 = 89 * (o - n) * 4096 / Double(rows)
                    print(
                        String(
                            format:
                                "GLM53_HC_FUSED_EXPAND rows %d: two-pass %.3f ms, fused %.3f ms per site (x%.3f): -%.1f ms per 4,096 rows per rank (89 sites)",
                            rows, o, n, o / n, per4096))
                    Memory.clearCache()
                }
            }
        }
    }
}
