// Copyright 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {
    /// `SplitForm.oneLaunch` (one launch per decode-sized mHC site) must
    /// reproduce the two-launch split form bit for bit: the normalized row, the
    /// folded expand's streams, post and comb. Its counters must be zero
    /// after every completed launch and recover from an unfinished one.
    @Suite(.serialized)
    struct GLM5NextHyperConnectionOneLaunchTests {
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

        private static func inputs(
            dtype: DType, fnType: DType, batch: Int, tokens: Int, width: Int, seed: UInt64
        ) -> Inputs {
            let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 8)
            return Inputs(
                streams: MLXRandom.normal([batch, tokens, 4, width], key: keys[0]).asType(dtype),
                output: MLXRandom.normal([batch, tokens, width], key: keys[1]).asType(dtype),
                post: MLXRandom.uniform(low: Float(0), high: 2, [batch, tokens, 4], key: keys[2]),
                comb: MLXRandom.uniform(
                    low: Float(0), high: 0.5, [batch, tokens, 4, 4], key: keys[3]),
                fn: (MLXRandom.normal([24, 4 * width], key: keys[4]) * 0.05).asType(fnType),
                base: (MLXRandom.normal([24], key: keys[5]) * 0.5).asType(fnType),
                scale: MLXRandom.uniform(low: Float(0.5), high: Float(1.5), [3], key: keys[6])
                    .asType(fnType),
                weight: MLXRandom.uniform(low: Float(0.5), high: Float(1.5), [width], key: keys[7])
                    .asType(dtype))
        }

        private static func site(
            _ source: Kernel.StreamsSource, _ x: Inputs, form: Kernel.SplitForm,
            workspace: Kernel.SiteWorkspace = .shared
        ) throws -> [MLXArray] {
            let result = try Kernel.collapseNormalized(
                source, fn: x.fn, base: x.base, scale: x.scale, rmsNormEpsilon: 1e-5,
                epsilon: 1e-6, sinkhornIterations: 20, normWeight: x.weight, normEpsilon: 1e-5,
                splitForm: form, workspace: workspace)
            return [result.normalized, result.streams, result.post, result.comb]
        }

        private static let names = ["normalized", "streams", "post", "comb"]

        /// Float32 bit patterns; widening FP16/BF16 to FP32 is exact.
        private static func bits(_ array: MLXArray) -> [UInt32] {
            array.asType(.float32).asArray(Float.self).map(\.bitPattern)
        }

        private static func differing(_ a: MLXArray, _ b: MLXArray) -> Int {
            zip(bits(a), bits(b)).filter { $0 != $1 }.count
        }

        private static func expectBitwise(_ one: [MLXArray], _ two: [MLXArray], _ label: String) {
            for (index, name) in names.enumerated() {
                #expect(one[index].shape == two[index].shape, "\(label) \(name) shape")
                #expect(one[index].dtype == two[index].dtype, "\(label) \(name) dtype")
                let count = differing(one[index], two[index])
                #expect(count == 0, "\(label) \(name): \(count) elements differ")
            }
        }

        private static func expectCleared(_ workspace: Kernel.SiteWorkspace, _ label: String) {
            let counters = workspace.counters.asArray(UInt32.self)
            #expect(counters.allSatisfy { $0 == 0 }, "\(label) counters \(counters)")
        }

        /// (B, T, D): serving decode, verify windows up to the row bound, the
        /// existing contract widths (1000, 33, 64, 4) and widths of 5 and 6
        /// columns per thread.
        private static let geometries = [
            [1, 1, 4096], [1, 3, 4096], [2, 4, 4096], [1, 8, 4096], [2, 3, 1000], [1, 8, 33],
            [2, 4, 64], [1, 1, 4], [1, 2, 5000], [1, 1, 6144],
        ]

        @Test(arguments: [DType.bfloat16, .float16, .float32])
        func oneLaunchIsBitwiseToTwoLaunchOnGPU(dtype: DType) throws {
            try Device.withDefaultDevice(.gpu) {
                let workspace = Kernel.SiteWorkspace()
                var fnTypes = [dtype]
                for other in [DType.bfloat16, .float32] where !fnTypes.contains(other) {
                    fnTypes.append(other)
                }
                for geometry in Self.geometries {
                    let (batch, tokens, width) = (geometry[0], geometry[1], geometry[2])
                    for fnType in fnTypes {
                        let x = Self.inputs(
                            dtype: dtype, fnType: fnType, batch: batch, tokens: tokens,
                            width: width, seed: UInt64(width * 31 + tokens * 7 + batch))
                        let sources: [(String, Kernel.StreamsSource)] = [
                            ("streams", .streams(x.streams)),
                            (
                                "expand",
                                .expand(
                                    output: x.output, residual: x.streams, post: x.post,
                                    comb: x.comb)
                            ),
                        ]
                        for (name, source) in sources {
                            let one = try Self.site(
                                source, x, form: .oneLaunch, workspace: workspace)
                            let two = try Self.site(source, x, form: .twoLaunch)
                            try withError { error in
                                eval(one + two)
                                try error.check()
                            }
                            Self.expectBitwise(
                                one, two,
                                "\(dtype) fn \(fnType) B\(batch) T\(tokens) D\(width) \(name)")
                            print(
                                "GLM53_HC_ONE_LAUNCH dtype=\(dtype) fn=\(fnType) B=\(batch) "
                                    + "T=\(tokens) D=\(width) source=\(name) differing="
                                    + "\(zip(one, two).map { Self.differing($0, $1) })")
                        }
                    }
                }
                Self.expectCleared(workspace, "\(dtype)")
            }
        }

        /// A decode step's chain: 90 dependent sites in one graph, each folding
        /// the previous expand, with a quantized branch between them (so the
        /// two-launch post/comb runs beside it, as in decode). Every launch
        /// reuses one workspace's counters and partials.
        @Test func chainedSitesStayBitwiseOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                for tokens in [1, 3] {
                    let width = 4096
                    let x = Self.inputs(
                        dtype: .bfloat16, fnType: .bfloat16, batch: 1, tokens: tokens,
                        width: width, seed: 90 + UInt64(tokens))
                    let branch = Self.branch(width: width, seed: 7)
                    let workspace = Kernel.SiteWorkspace()
                    var finals: [[MLXArray]] = []
                    for form in [Kernel.SplitForm.oneLaunch, .twoLaunch] {
                        var source = Kernel.StreamsSource.streams(x.streams)
                        var seen: [MLXArray] = []
                        for _ in 0 ..< 90 {
                            let result = try Self.site(
                                source, x, form: form, workspace: workspace)
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
                    }
                    try withError { error in
                        eval(finals[0] + finals[1])
                        try error.check()
                    }
                    #expect(isFinite(finals[0][0]).all().item(Bool.self), "T\(tokens) finite")
                    let differing = zip(finals[0], finals[1]).map { Self.differing($0, $1) }
                    #expect(differing.allSatisfy { $0 == 0 }, "T\(tokens) chain \(differing)")
                    print(
                        "GLM53_HC_ONE_LAUNCH chain T=\(tokens) sites=90 "
                            + "differing=\(differing.reduce(0, +))")
                    Self.expectCleared(workspace, "chain T\(tokens)")
                }
            }
        }

        /// Independent sites in one evaluation (no data dependency between
        /// them): MLX must still order every launch on their one workspace.
        @Test func independentSitesInOneGraphAreBitwiseOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                let workspace = Kernel.SiteWorkspace()
                var one: [MLXArray] = []
                var two: [MLXArray] = []
                for index in 0 ..< 12 {
                    let x = Self.inputs(
                        dtype: .bfloat16, fnType: .bfloat16, batch: 1, tokens: 1 + index % 8,
                        width: 4096, seed: 500 + UInt64(index))
                    let source = Kernel.StreamsSource.expand(
                        output: x.output, residual: x.streams, post: x.post, comb: x.comb)
                    one += try Self.site(source, x, form: .oneLaunch, workspace: workspace)
                    two += try Self.site(source, x, form: .twoLaunch)
                }
                try withError { error in
                    eval(one + two)
                    try error.check()
                }
                let differing = zip(one, two).map { Self.differing($0, $1) }
                #expect(differing.allSatisfy { $0 == 0 }, "independent \(differing)")
                Self.expectCleared(workspace, "independent")
            }
        }

        /// A launch that never finished leaves an arrival count behind. The
        /// next completed launch on the other slot clears both slots; every
        /// launch after it is bitwise again.
        @Test func anUnfinishedLaunchIsClearedByTheNextOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                let x = Self.inputs(
                    dtype: .bfloat16, fnType: .bfloat16, batch: 1, tokens: 2, width: 4096,
                    seed: 77)
                let source = Kernel.StreamsSource.expand(
                    output: x.output, residual: x.streams, post: x.post, comb: x.comb)
                let expected = try Self.site(source, x, form: .twoLaunch)
                eval(expected)
                let rows = Kernel.splitCollapseMaximumRows
                for poisonNext in [false, true] {
                    let workspace = Kernel.SiteWorkspace()
                    let warm = try Self.site(source, x, form: .oneLaunch, workspace: workspace)
                    eval(warm)
                    Self.expectBitwise(warm, expected, "warm")
                    // An unfinished launch's count in one slot, for both rows.
                    let slot = poisonNext ? workspace.upcomingSlot : 1 - workspace.upcomingSlot
                    var values = [UInt32](repeating: 0, count: 2 * rows)
                    values[slot * rows] = 7
                    values[slot * rows + 1] = 19
                    Self.poke(workspace, values)
                    #expect(workspace.counters.asArray(UInt32.self) == values)
                    let first = try Self.site(source, x, form: .oneLaunch, workspace: workspace)
                    eval(first)
                    if !poisonNext {
                        // The launch used the clean slot and cleared the stale one.
                        Self.expectBitwise(first, expected, "after a stale other slot")
                        Self.expectCleared(workspace, "after a stale other slot")
                    }
                    // Either way the next launch uses a clean slot and clears both.
                    let second = try Self.site(source, x, form: .oneLaunch, workspace: workspace)
                    eval(second)
                    Self.expectBitwise(second, expected, "recovered poisonNext=\(poisonNext)")
                    Self.expectCleared(workspace, "recovered poisonNext=\(poisonNext)")
                    let third = try Self.site(source, x, form: .oneLaunch, workspace: workspace)
                    eval(third)
                    Self.expectBitwise(third, expected, "steady poisonNext=\(poisonNext)")
                    Self.expectCleared(workspace, "steady poisonNext=\(poisonNext)")
                }
            }
        }

        /// Opt-in timing (`GLM53_HC_ONE_LAUNCH_BENCH=1`): a 90-site decode chain
        /// with a 4-bit 4096x4096 branch between sites, for each form, against
        /// the branch-only chain. Prints per-site microseconds.
        @Test func oneLaunchChainTiming() throws {
            guard ProcessInfo.processInfo.environment["GLM53_HC_ONE_LAUNCH_BENCH"] == "1" else {
                return
            }
            try Device.withDefaultDevice(.gpu) {
                let width = 4096
                let sites = 90
                let x = Self.inputs(
                    dtype: .bfloat16, fnType: .bfloat16, batch: 1, tokens: 1, width: width,
                    seed: 3)
                let branch = Self.branch(width: width, seed: 11)
                func chain(_ form: Kernel.SplitForm?) throws -> MLXArray {
                    guard let form else {
                        var value = x.output
                        for _ in 0 ..< sites { value = branch(value) }
                        return value
                    }
                    var source = Kernel.StreamsSource.expand(
                        output: x.output, residual: x.streams, post: x.post, comb: x.comb)
                    var last = x.output
                    for _ in 0 ..< sites {
                        let result = try Self.site(source, x, form: form)
                        last = branch(result[0])
                        source = .expand(
                            output: last, residual: result[1], post: result[2], comb: result[3])
                    }
                    return last
                }
                func microseconds(_ form: Kernel.SplitForm?) throws -> Double {
                    var samples: [Double] = []
                    for iteration in 0 ..< 40 {
                        let value = try chain(form)
                        let start = DispatchTime.now().uptimeNanoseconds
                        eval(value)
                        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3
                        if iteration >= 8 { samples.append(elapsed) }
                    }
                    return samples.sorted()[samples.count / 2]
                }
                let floor = try microseconds(nil)
                let one = try microseconds(.oneLaunch)
                let two = try microseconds(.twoLaunch)
                print(
                    "GLM53_HC_ONE_LAUNCH_BENCH sites=\(sites) branch_only_us=\(floor) "
                        + "one_launch_us=\(one) two_launch_us=\(two) "
                        + "one_per_site_us=\((one - floor) / Double(sites)) "
                        + "two_per_site_us=\((two - floor) / Double(sites)) "
                        + "saved_per_site_us=\((two - one) / Double(sites))")
            }
        }

        /// A 4-bit 4096x4096 branch (one quantized matmul), rounding back to
        /// the activation dtype.
        private static func branch(width: Int, seed: UInt64) -> (MLXArray) -> MLXArray {
            let weight = MLXRandom.normal([width, width], key: MLXRandom.key(seed)) * 0.02
            let (words, scales, biases) = MLX.quantized(weight, groupSize: 64, bits: 4)
            eval([words, scales] + (biases.map { [$0] } ?? []))
            return { value in
                quantizedMatmul(
                    value, words, scales: scales, biases: biases, transpose: true, groupSize: 64,
                    bits: 4
                ).asType(value.dtype)
            }
        }

        /// Writes `values` into the workspace counters (a test-only kernel).
        private static func poke(_ workspace: Kernel.SiteWorkspace, _ values: [UInt32]) {
            let fence = pokeKernel(
                [workspace.counters, MLXArray(values)], grid: (values.count, 1, 1),
                threadGroup: (values.count, 1, 1), outputShapes: [[1]], outputDTypes: [.uint32]
            )[0]
            eval(fence)
        }

        private static let pokeKernel = MLXFast.metalKernel(
            name: "glm5_next_test_hc_site_counters",
            inputNames: ["counters", "values"], outputNames: ["fence"],
            source: """
                const uint i = thread_position_in_grid.x;
                counters[i] = values[i];
                if (i == 0) { fence[0] = 1; }
                """, mutableInputs: ["counters"])
    }
}
