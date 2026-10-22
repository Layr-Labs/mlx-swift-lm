// Copyright 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {
    /// The shortened `SplitForm.oneLaunch` kernel against the one it replaced
    /// (F1, e7a5f18d, embedded below as it was): same inputs, same geometry,
    /// separate workspaces; every output must match bit for bit. Covers both
    /// sources, BF16/F16/F32 activations, every one-launch geometry, Sinkhorn
    /// iteration counts other than the model's 20, saturating inputs, and a
    /// 90-site decode chain at T = 1 and 3. `GLM5NextHyperConnectionOneLaunchTests`
    /// separately keeps the new kernel bitwise to the two-launch oracle.
    @Suite(.serialized)
    struct GLM5NextHyperConnectionSiteRevisionTests {
        private typealias Kernel = GLM5NextHyperConnectionKernel

        private struct Inputs {
            let streams: MLXArray
            let output: MLXArray
            let post: MLXArray
            let comb: MLXArray
            let fn: MLXArray
            let base: MLXArray
            let scale: MLXArray
            let weight: MLXArray
        }

        /// `magnitude` scales fn and base (1 = the contract tests' ranges;
        /// large values saturate the sigmoids and the softmax).
        private static func inputs(
            dtype: DType, fnType: DType, batch: Int, tokens: Int, width: Int, seed: UInt64,
            magnitude: Float = 1
        ) -> Inputs {
            let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 8)
            return Inputs(
                streams: MLXRandom.normal([batch, tokens, 4, width], key: keys[0]).asType(dtype),
                output: MLXRandom.normal([batch, tokens, width], key: keys[1]).asType(dtype),
                post: MLXRandom.uniform(low: Float(0), high: 2, [batch, tokens, 4], key: keys[2]),
                comb: MLXRandom.uniform(
                    low: Float(0), high: 0.5, [batch, tokens, 4, 4], key: keys[3]),
                fn: (MLXRandom.normal([24, 4 * width], key: keys[4]) * (0.05 * magnitude))
                    .asType(fnType),
                base: (MLXRandom.normal([24], key: keys[5]) * (0.5 * magnitude)).asType(fnType),
                scale: MLXRandom.uniform(low: Float(0.5), high: Float(1.5), [3], key: keys[6])
                    .asType(fnType),
                weight: MLXRandom.uniform(low: Float(0.5), high: Float(1.5), [width], key: keys[7])
                    .asType(dtype))
        }

        private static func new(
            _ source: Kernel.StreamsSource, _ x: Inputs, iterations: Int,
            workspace: Kernel.SiteWorkspace
        ) throws -> [MLXArray] {
            let result = try Kernel.collapseNormalized(
                source, fn: x.fn, base: x.base, scale: x.scale, rmsNormEpsilon: 1e-5,
                epsilon: 1e-6, sinkhornIterations: iterations, normWeight: x.weight,
                normEpsilon: 1e-5, splitForm: .oneLaunch, workspace: workspace)
            return [result.normalized, result.streams, result.post, result.comb]
        }

        /// F1's `oneLaunchCollapse`, host side as it was.
        private static func old(
            _ source: Kernel.StreamsSource, _ x: Inputs, iterations: Int,
            workspace: Kernel.SiteWorkspace
        ) throws -> [MLXArray] {
            let reference: MLXArray
            switch source {
            case .streams(let streams): reference = streams
            case .expand(_, let residual, _, _): reference = residual
            }
            let shape = reference.shape
            let (batch, tokens, width) = (shape[0], shape[1], shape[3])
            let threads = Kernel.normalizedCollapseThreads
            let params = MLXArray([
                Float(1e-6), Float(1.0 / Double(4 * width)), Float(1e-5),
                Float(1.0 / Double(width)), Float(1e-5), Float(workspace.nextSlot()),
            ])
            let template: [(String, any KernelTemplateArg)] = [
                ("T", reference.dtype), ("Width", width), ("Threads", threads),
                ("PerThread", (width + threads - 1) / threads), ("Iterations", iterations),
                ("MaxRows", Kernel.splitCollapseMaximumRows),
            ]
            let grid = (threads * Kernel.oneLaunchGroups, batch * tokens, 1)
            let rowShape = [batch, tokens, width]
            let coefficientShapes = [[batch, tokens, 4], [batch, tokens, 4, 4]]
            switch source {
            case .streams(let input):
                let outputs = oldKernel(
                    [
                        input, x.fn, x.base, x.scale, x.weight, params, workspace.counters,
                        workspace.partials,
                    ],
                    template: template, grid: grid, threadGroup: (threads, 1, 1),
                    outputShapes: [rowShape] + coefficientShapes,
                    outputDTypes: [reference.dtype, .float32, .float32])
                return [outputs[0], input, outputs[1], outputs[2]]
            case .expand(let output, let residual, let post, let comb):
                let outputs = oldExpandKernel(
                    [
                        output, residual, post, comb, x.fn, x.base, x.scale, x.weight, params,
                        workspace.counters, workspace.partials,
                    ],
                    template: template, grid: grid, threadGroup: (threads, 1, 1),
                    outputShapes: [rowShape, residual.shape] + coefficientShapes,
                    outputDTypes: [reference.dtype, reference.dtype, .float32, .float32])
                return outputs
            }
        }

        private static let names = ["normalized", "streams", "post", "comb"]

        /// Float32 bit patterns; widening FP16/BF16 to FP32 is exact.
        private static func bits(_ array: MLXArray) -> [UInt32] {
            array.asType(.float32).asArray(Float.self).map(\.bitPattern)
        }

        private static func differing(_ a: MLXArray, _ b: MLXArray) -> Int {
            zip(bits(a), bits(b)).filter { $0 != $1 }.count
        }

        private static func compare(_ new: [MLXArray], _ old: [MLXArray], _ label: String) -> [Int]
        {
            var counts: [Int] = []
            for (index, name) in names.enumerated() {
                #expect(new[index].shape == old[index].shape, "\(label) \(name) shape")
                #expect(new[index].dtype == old[index].dtype, "\(label) \(name) dtype")
                let count = differing(new[index], old[index])
                #expect(count == 0, "\(label) \(name): \(count) elements differ")
                counts.append(count)
            }
            return counts
        }

        private static func expectCleared(_ workspace: Kernel.SiteWorkspace, _ label: String) {
            let counters = workspace.counters.asArray(UInt32.self)
            #expect(counters.allSatisfy { $0 == 0 }, "\(label) counters \(counters)")
        }

        private static let geometries = [
            [1, 1, 4096], [1, 3, 4096], [2, 4, 4096], [1, 8, 4096], [2, 3, 1000], [1, 8, 33],
            [2, 4, 64], [1, 1, 4], [1, 2, 5000], [1, 1, 6144],
        ]

        @Test(arguments: [DType.bfloat16, .float16, .float32])
        func newOneLaunchIsBitwiseToF1OnGPU(dtype: DType) throws {
            try Device.withDefaultDevice(.gpu) {
                let newWorkspace = Kernel.SiteWorkspace()
                let oldWorkspace = Kernel.SiteWorkspace()
                var fnTypes = [dtype]
                for other in [DType.bfloat16, .float32] where !fnTypes.contains(other) {
                    fnTypes.append(other)
                }
                var cases = 0
                for geometry in Self.geometries {
                    let (batch, tokens, width) = (geometry[0], geometry[1], geometry[2])
                    for fnType in fnTypes {
                        for (iterations, magnitude, seed) in [
                            (20, Float(1), UInt64(0)), (20, 1, 1), (20, 40, 2), (1, 1, 3),
                            (2, 4, 4), (33, 1, 5),
                        ] {
                            let x = Self.inputs(
                                dtype: dtype, fnType: fnType, batch: batch, tokens: tokens,
                                width: width,
                                seed: UInt64(width * 31 + tokens * 7 + batch) * 16 + seed,
                                magnitude: magnitude)
                            for (name, source) in [
                                ("streams", Kernel.StreamsSource.streams(x.streams)),
                                (
                                    "expand",
                                    .expand(
                                        output: x.output, residual: x.streams, post: x.post,
                                        comb: x.comb)
                                ),
                            ] {
                                let new = try Self.new(
                                    source, x, iterations: iterations, workspace: newWorkspace)
                                let old = try Self.old(
                                    source, x, iterations: iterations, workspace: oldWorkspace)
                                try withError { error in
                                    eval(new + old)
                                    try error.check()
                                }
                                let label =
                                    "\(dtype) fn \(fnType) B\(batch) T\(tokens) D\(width) "
                                    + "\(name) iterations \(iterations) magnitude \(magnitude)"
                                let counts = Self.compare(new, old, label)
                                cases += 1
                                if counts.contains(where: { $0 != 0 }) {
                                    print("GLM53_HC_SITE_REVISION \(label) differing=\(counts)")
                                }
                            }
                        }
                    }
                }
                print("GLM53_HC_SITE_REVISION dtype=\(dtype) cases=\(cases) done")
                Self.expectCleared(newWorkspace, "new \(dtype)")
                Self.expectCleared(oldWorkspace, "old \(dtype)")
            }
        }

        /// A decode step's chain of 90 dependent sites (each folding the
        /// previous expand, a quantized branch between them), new against F1.
        @Test func chainIsBitwiseToF1OnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                for tokens in [1, 3] {
                    let width = 4096
                    let x = Self.inputs(
                        dtype: .bfloat16, fnType: .bfloat16, batch: 1, tokens: tokens,
                        width: width, seed: 190 + UInt64(tokens))
                    let weight = MLXRandom.normal([width, width], key: MLXRandom.key(7)) * 0.02
                    let (words, scales, biases) = MLX.quantized(weight, groupSize: 64, bits: 4)
                    func branch(_ value: MLXArray) -> MLXArray {
                        quantizedMatmul(
                            value, words, scales: scales, biases: biases, transpose: true,
                            groupSize: 64, bits: 4
                        ).asType(value.dtype)
                    }
                    var finals: [[MLXArray]] = []
                    for useNew in [true, false] {
                        let workspace = Kernel.SiteWorkspace()
                        var source = Kernel.StreamsSource.streams(x.streams)
                        var seen: [MLXArray] = []
                        for _ in 0 ..< 90 {
                            let result =
                                useNew
                                ? try Self.new(source, x, iterations: 20, workspace: workspace)
                                : try Self.old(source, x, iterations: 20, workspace: workspace)
                            seen += [result[0], result[2], result[3]]
                            source = .expand(
                                output: branch(result[0]), residual: result[1], post: result[2],
                                comb: result[3])
                        }
                        guard case .expand(let output, let residual, let post, let comb) = source
                        else { throw Kernel.Failure.invalidGeometry("chain") }
                        finals.append(
                            [try Kernel.expand(output, residual: residual, post: post, comb: comb)]
                                + seen)
                        eval(finals.last!)
                        Self.expectCleared(workspace, "chain T\(tokens) new=\(useNew)")
                    }
                    #expect(isFinite(finals[0][0]).all().item(Bool.self), "T\(tokens) finite")
                    let differing = zip(finals[0], finals[1]).map { Self.differing($0, $1) }
                    #expect(differing.allSatisfy { $0 == 0 }, "T\(tokens) chain \(differing)")
                    print(
                        "GLM53_HC_SITE_REVISION chain T=\(tokens) sites=90 "
                            + "differing=\(differing.reduce(0, +))")
                }
            }
        }

        /// Opt-in timing (`GLM53_HC_ONE_LAUNCH_BENCH=1`): a chain of 90 dependent
        /// sites with nothing between them (site k + 1 folds site k's expand of
        /// its own normalized row), so the GPU, not the host, bounds the eval.
        /// Prints wall-clock microseconds per site for F1 and the new kernel at
        /// T = 1 and 3.
        @Test func siteChainTiming() throws {
            guard ProcessInfo.processInfo.environment["GLM53_HC_ONE_LAUNCH_BENCH"] == "1" else {
                return
            }
            // `GLM53_HC_SITE_TRACE=1`: for a Metal System Trace attached by pid
            // (one dispatch per command buffer with MLX_MAX_OPS_PER_BUFFER=0):
            // wait, then run each form for about two seconds, 1 s apart.
            let trace = ProcessInfo.processInfo.environment["GLM53_HC_SITE_TRACE"] == "1"
            if trace {
                print("GLM53_HC_SITE_TRACE pid=\(ProcessInfo.processInfo.processIdentifier)")
                fflush(stdout)
                Thread.sleep(forTimeInterval: 6)
            }
            try Device.withDefaultDevice(.gpu) {
                for tokens in [1, 3] {
                    let x = Self.inputs(
                        dtype: .bfloat16, fnType: .bfloat16, batch: 1, tokens: tokens,
                        width: 4096, seed: 3)
                    if trace {
                        for useNew in [false, true] {
                            let workspace = Kernel.SiteWorkspace()
                            let start = Date()
                            var launches = 0
                            while Date().timeIntervalSince(start) < 2 {
                                var source = Kernel.StreamsSource.expand(
                                    output: x.output, residual: x.streams, post: x.post,
                                    comb: x.comb)
                                var last = x.output
                                for _ in 0 ..< 90 {
                                    let result =
                                        useNew
                                        ? try Self.new(
                                            source, x, iterations: 20, workspace: workspace)
                                        : try Self.old(
                                            source, x, iterations: 20, workspace: workspace)
                                    last = result[0]
                                    source = .expand(
                                        output: result[0], residual: result[1], post: result[2],
                                        comb: result[3])
                                }
                                eval(last)
                                launches += 90
                            }
                            print(
                                "GLM53_HC_SITE_TRACE T=\(tokens) form=\(useNew ? "new" : "f1") "
                                    + "launches=\(launches) end=\(Date().timeIntervalSince1970)")
                            fflush(stdout)
                            Thread.sleep(forTimeInterval: 1)
                        }
                        continue
                    }
                    func microseconds(_ useNew: Bool) throws -> Double {
                        let workspace = Kernel.SiteWorkspace()
                        var samples: [Double] = []
                        for iteration in 0 ..< 40 {
                            var source = Kernel.StreamsSource.expand(
                                output: x.output, residual: x.streams, post: x.post, comb: x.comb)
                            var last = x.output
                            for _ in 0 ..< 90 {
                                let result =
                                    useNew
                                    ? try Self.new(source, x, iterations: 20, workspace: workspace)
                                    : try Self.old(source, x, iterations: 20, workspace: workspace)
                                last = result[0]
                                source = .expand(
                                    output: result[0], residual: result[1], post: result[2],
                                    comb: result[3])
                            }
                            let start = DispatchTime.now().uptimeNanoseconds
                            eval(last)
                            let elapsed =
                                Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3 / 90
                            if iteration >= 8 { samples.append(elapsed) }
                        }
                        return samples.sorted()[samples.count / 2]
                    }
                    let old = try microseconds(false)
                    let new = try microseconds(true)
                    print(
                        "GLM53_HC_SITE_REVISION_BENCH T=\(tokens) sites=90 f1_us_per_site=\(old) "
                            + "new_us_per_site=\(new) saved_us_per_site=\(old - new)")
                }
            }
        }

        // F1's kernels (e7a5f18d), verbatim:
        // `GLM5NextHyperConnectionNormFusion.oneLaunchSource` as it was.
        private static let oldSource = """
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

        private static let oldKernel = MLXFast.metalKernel(
            name: "glm5_next_test_hc4_site_collapse_norm_f1",
            inputNames: [
                "streams", "fn", "base", "scale", "norm_weight", "params", "counters", "partials",
            ],
            outputNames: ["normalized", "post_out", "comb_out"],
            source: "#define GLM5_HC_SITE_EXPAND 0\n" + oldSource,
            header: Kernel.coefficientHeader, ensureRowContiguous: true,
            mutableInputs: ["counters", "partials"])

        private static let oldExpandKernel = MLXFast.metalKernel(
            name: "glm5_next_test_hc4_site_expand_collapse_norm_f1",
            inputNames: [
                "output", "residual", "post", "comb", "fn", "base", "scale", "norm_weight",
                "params", "counters", "partials",
            ],
            outputNames: ["normalized", "expanded", "post_out", "comb_out"],
            source: "#define GLM5_HC_SITE_EXPAND 1\n" + oldSource,
            header: Kernel.coefficientHeader, ensureRowContiguous: true,
            mutableInputs: ["counters", "partials"])
    }
}
