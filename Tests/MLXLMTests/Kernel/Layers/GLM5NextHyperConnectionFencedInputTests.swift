// Copyright 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {
    /// The one-launch site reads its `output` as an MLX fenced input: when the
    /// output comes from a CPU stream in the same evaluation (a tensor-parallel
    /// sum), the kernel waits for the CPU itself instead of a separate MLX
    /// fence-wait launch. Synchronization only, so every result must be bitwise
    /// the result of the same site run on the already evaluated output.
    ///
    /// Run with `MLX_METAL_FAST_SYNCH=1 MLX_METAL_FENCED_INPUTS=1` (and
    /// `MLX_METAL_FUSED_FENCE_UPDATE=1` for the GPU-to-CPU direction) to test
    /// the new paths; without them the same checks cover the existing fences.
    /// A hang here is a synchronization bug: see `agent-work/sumfuse/NOTES.md`.
    @Suite(.serialized)
    struct GLM5NextHyperConnectionFencedInputTests {
        private typealias Kernel = GLM5NextHyperConnectionKernel

        private static var mode: String {
            let environment = ProcessInfo.processInfo.environment
            return [
                "MLX_METAL_FAST_SYNCH", "MLX_METAL_FENCED_INPUTS", "MLX_METAL_FUSED_FENCE_UPDATE",
            ]
            .map { "\($0)=\(environment[$0] ?? "0")" }.joined(separator: " ")
        }

        private struct Inputs {
            let streams: MLXArray
            let post: MLXArray
            let comb: MLXArray
            let fn: MLXArray
            let base: MLXArray
            let scale: MLXArray
            let weight: MLXArray
        }

        private static func inputs(tokens: Int, width: Int, seed: UInt64) -> Inputs {
            let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 7)
            let x = Inputs(
                streams: MLXRandom.normal([1, tokens, 4, width], key: keys[0]).asType(.bfloat16),
                post: MLXRandom.uniform(low: Float(0), high: 2, [1, tokens, 4], key: keys[1]),
                comb: MLXRandom.uniform(low: Float(0), high: 0.5, [1, tokens, 4, 4], key: keys[2]),
                fn: (MLXRandom.normal([24, 4 * width], key: keys[3]) * 0.05).asType(.bfloat16),
                base: (MLXRandom.normal([24], key: keys[4]) * 0.5).asType(.bfloat16),
                scale: MLXRandom.uniform(low: Float(0.5), high: Float(1.5), [3], key: keys[5])
                    .asType(.bfloat16),
                weight: MLXRandom.uniform(low: Float(0.5), high: Float(1.5), [width], key: keys[6])
                    .asType(.bfloat16))
            eval(x.streams, x.post, x.comb, x.fn, x.base, x.scale, x.weight)
            return x
        }

        private static func site(
            output: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray, _ x: Inputs,
            form: Kernel.SplitForm = .oneLaunch
        ) throws -> [MLXArray] {
            let result = try Kernel.collapseNormalized(
                .expand(output: output, residual: residual, post: post, comb: comb), fn: x.fn,
                base: x.base, scale: x.scale, rmsNormEpsilon: 1e-5, epsilon: 1e-6,
                sinkhornIterations: 20, normWeight: x.weight, normEpsilon: 1e-5, splitForm: form)
            return [result.normalized, result.streams, result.post, result.comb]
        }

        private static func bits(_ array: MLXArray) -> [UInt32] {
            array.asType(.float32).asArray(Float.self).map(\.bitPattern)
        }

        private static func differing(_ a: [MLXArray], _ b: [MLXArray]) -> [Int] {
            zip(a, b).map { zip(bits($0), bits($1)).filter { $0 != $1 }.count }
        }

        /// A branch output made on the CPU stream, slowly (a K = 8192 matmul),
        /// so the GPU reaches the site long before the CPU writes it.
        private static func cpuOutput(tokens: Int, width: Int, seed: UInt64) -> MLXArray {
            let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 2)
            let a = MLXRandom.normal([tokens, 8192], key: keys[0]) * 0.01
            let b = MLXRandom.normal([8192, width], key: keys[1]) * 0.01
            eval(a, b)
            return matmul(a, b, stream: .cpu).asType(.bfloat16, stream: .cpu)
                .reshaped([1, tokens, width], stream: .cpu)
        }

        /// The site and the stand-alone expand on a CPU-stream output in the
        /// same evaluation, then on that output once evaluated (no fence):
        /// bitwise, for several geometries and fresh values each time, both
        /// split forms.
        @Test func cpuStreamOutputIsBitwiseOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                for (index, geometry) in [[1, 4096], [3, 4096], [8, 4096], [2, 1000], [1, 33]]
                    .enumerated()
                {
                    let (tokens, width) = (geometry[0], geometry[1])
                    for form in [Kernel.SplitForm.oneLaunch, .twoLaunch] {
                        for repeat_ in 0 ..< 4 {
                            let seed = UInt64(1000 * index + 10 * repeat_)
                            let x = Self.inputs(tokens: tokens, width: width, seed: seed)
                            let output = Self.cpuOutput(
                                tokens: tokens, width: width, seed: seed + 1)
                            // The site, and the stand-alone expand (the last layer's sum).
                            let fenced =
                                try Self.site(
                                    output: output, residual: x.streams, post: x.post, comb: x.comb,
                                    x,
                                    form: form)
                                + [
                                    try Kernel.expand(
                                        output, residual: x.streams, post: x.post, comb: x.comb)
                                ]
                            try withError { error in
                                eval(fenced + [output])
                                try error.check()
                            }
                            let reference =
                                try Self.site(
                                    output: output, residual: x.streams, post: x.post, comb: x.comb,
                                    x,
                                    form: form)
                                + [
                                    try Kernel.expand(
                                        output, residual: x.streams, post: x.post, comb: x.comb)
                                ]
                            eval(reference)
                            let counts = Self.differing(fenced, reference)
                            #expect(
                                counts.allSatisfy { $0 == 0 },
                                "T\(tokens) D\(width) \(form) #\(repeat_): \(counts)")
                        }
                    }
                    print("GLM53_FENCED_INPUT site T=\(tokens) D=\(width) ok \(Self.mode)")
                }
            }
        }

        /// A decode chain with a GPU -> CPU -> GPU hop at every site, as the
        /// tensor-parallel sum makes: 90 sites in one evaluation, each branch
        /// output copied on the CPU stream (`* 1`, exact). Bitwise to the same
        /// chain with the copy on the GPU, and it must not hang.
        @Test func chainWithCPUHopsIsBitwiseOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                for tokens in [1, 3] {
                    let width = 4096
                    let x = Self.inputs(tokens: tokens, width: width, seed: 77 + UInt64(tokens))
                    let branch = Self.branch(width: width, seed: 5)
                    let start = MLXRandom.normal([1, tokens, width], key: MLXRandom.key(9))
                        .asType(.bfloat16)
                    let one = MLXArray(1).asType(.bfloat16)
                    eval(start, one)
                    var finals: [[MLXArray]] = []
                    for device in [StreamOrDevice.cpu, .gpu] {
                        var output = start
                        var residual = x.streams
                        var post = x.post
                        var comb = x.comb
                        var seen: [MLXArray] = []
                        for _ in 0 ..< 90 {
                            let result = try Self.site(
                                output: output, residual: residual, post: post, comb: comb, x)
                            seen += [result[0], result[2]]
                            output = multiply(branch(result[0]), one, stream: device)
                            (residual, post, comb) = (result[1], result[2], result[3])
                        }
                        finals.append(
                            [try Kernel.expand(output, residual: residual, post: post, comb: comb)]
                                + seen)
                    }
                    let started = Date()
                    try withError { error in
                        eval(finals[0])
                        try error.check()
                    }
                    let seconds = Date().timeIntervalSince(started)
                    eval(finals[1])
                    let counts = Self.differing(finals[0], finals[1])
                    #expect(
                        counts.allSatisfy { $0 == 0 }, "T\(tokens) chain \(counts.reduce(0, +))")
                    print(
                        "GLM53_FENCED_INPUT chain T=\(tokens) sites=90 cpu_hops_s=\(seconds) "
                            + "differing=\(counts.reduce(0, +)) \(Self.mode)")
                }
            }
        }

        /// A fenced input that is not row-contiguous is copied before the
        /// kernel, so MLX must wait before the copy (the separate wait).
        @Test func transposedCPUOutputWaitsBeforeTheCopyOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                let (tokens, width) = (3, 4096)
                let x = Self.inputs(tokens: tokens, width: width, seed: 31)
                let rows = Self.cpuOutput(tokens: width, width: tokens, seed: 32)
                let output = transposed(rows, axes: [0, 2, 1], stream: .cpu)
                #expect(output.shape == [1, tokens, width])
                let fenced = try Self.site(
                    output: output, residual: x.streams, post: x.post, comb: x.comb, x)
                try withError { error in
                    eval(fenced + [output])
                    try error.check()
                }
                let reference = try Self.site(
                    output: contiguous(output), residual: x.streams, post: x.post, comb: x.comb, x)
                eval(reference)
                let counts = Self.differing(fenced, reference)
                #expect(counts.allSatisfy { $0 == 0 }, "transposed \(counts)")
            }
        }

        /// The fenced-input contract outside the site: a CPU-stream input and a
        /// GPU input (count 0, no wait) in one copy kernel.
        @Test func copyKernelReadsCPUAndGPUInputsOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                for round in 0 ..< 8 {
                    let fromCPU = Self.cpuOutput(tokens: 2, width: 512, seed: 600 + UInt64(round))
                    let fromGPU =
                        MLXRandom.normal([1, 2, 512], key: MLXRandom.key(UInt64(round)))
                        .asType(.bfloat16) * 2
                    let outputs = Self.copyKernel(
                        [fromCPU, fromGPU], grid: (1024, 1, 1), threadGroup: (256, 1, 1),
                        outputShapes: [[1, 2, 512], [1, 2, 512]],
                        outputDTypes: [.bfloat16, .bfloat16])
                    try withError { error in
                        eval(outputs + [fromCPU, fromGPU])
                        try error.check()
                    }
                    #expect(
                        Self.differing(outputs, [fromCPU, fromGPU]) == [0, 0], "round \(round)")
                }
            }
        }

        @Test func fencedInputsAreValidated() throws {
            func make(_ fenced: [String], source: String) throws {
                try withError {
                    _ = MLXFast.metalKernel(
                        name: "glm5_next_test_fenced_validation", inputNames: ["a", "b"],
                        outputNames: ["out"], source: source, fencedInputs: fenced)
                }
            }
            let source = "MLX_FENCED_INPUT_WAIT(a); out[0] = a[0];"
            try make(["a"], source: source)
            #expect(throws: (any Error).self) { try make(["c"], source: source) }
            #expect(throws: (any Error).self) { try make(["b"], source: source) }
            #expect(throws: (any Error).self) { try make(["a", "a"], source: source) }
        }

        /// Opt-in timing (`GLM53_FENCED_INPUT_BENCH=1`): the 90-site chain with a
        /// CPU hop per site. Compare one run with the MLX variables against one
        /// without.
        @Test func chainWithCPUHopsTiming() throws {
            guard ProcessInfo.processInfo.environment["GLM53_FENCED_INPUT_BENCH"] == "1" else {
                return
            }
            try Device.withDefaultDevice(.gpu) {
                let width = 4096
                let x = Self.inputs(tokens: 1, width: width, seed: 3)
                let branch = Self.branch(width: width, seed: 11)
                let one = MLXArray(1).asType(.bfloat16)
                let start = MLXRandom.normal([1, 1, width], key: MLXRandom.key(4)).asType(.bfloat16)
                eval(one, start)
                var samples: [Double] = []
                for iteration in 0 ..< 40 {
                    var output = start
                    var (residual, post, comb) = (x.streams, x.post, x.comb)
                    for _ in 0 ..< 90 {
                        let result = try Self.site(
                            output: output, residual: residual, post: post, comb: comb, x)
                        output = multiply(branch(result[0]), one, stream: .cpu)
                        (residual, post, comb) = (result[1], result[2], result[3])
                    }
                    let start = DispatchTime.now().uptimeNanoseconds
                    eval(output, residual)
                    let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3
                    if iteration >= 8 { samples.append(elapsed) }
                }
                let median = samples.sorted()[samples.count / 2]
                print(
                    "GLM53_FENCED_INPUT_BENCH sites=90 chain_us=\(median) "
                        + "per_site_us=\(median / 90) \(Self.mode)")
            }
        }

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

        private static let copyKernel = MLXFast.metalKernel(
            name: "glm5_next_test_fenced_copy",
            inputNames: ["a", "b"], outputNames: ["out_a", "out_b"],
            source: """
                MLX_FENCED_INPUT_WAIT(a);
                MLX_FENCED_INPUT_WAIT(b);
                const uint i = thread_position_in_grid.x;
                if (i < 1024) {
                    out_a[i] = a[i];
                    out_b[i] = b[i];
                }
                """, fencedInputs: ["a", "b"])
    }
}
