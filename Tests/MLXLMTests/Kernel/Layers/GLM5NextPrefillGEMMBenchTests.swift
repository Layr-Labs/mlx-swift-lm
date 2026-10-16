// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {
    fileprivate enum PrefillGEMM {
        static func linear(n: Int, k: Int, seed: UInt64) -> QuantizedLinear {
            let key = MLXRandom.key(seed)
            let weight = (MLXRandom.normal([n, k], key: key) * 0.02).asType(.bfloat16)
            let linear = QuantizedLinear(weight: weight, bias: nil, groupSize: 64, bits: 4)
            eval(linear)
            return linear
        }

        static func input(m: Int, k: Int, seed: UInt64) -> MLXArray {
            MLXRandom.normal([m, k], key: MLXRandom.key(seed)).asType(.bfloat16)
        }

        /// (max |a - b| / max |b|, fraction of differing elements), on the GPU.
        static func difference(_ actual: MLXArray, _ expected: MLXArray) -> (Float, Float) {
            let a = actual.asType(.float32)
            let b = expected.asType(.float32)
            let worst = abs(a - b).max().item(Float.self)
            let peak = abs(b).max().item(Float.self)
            let differing = (a .!= b).asType(.float32).mean().item(Float.self)
            return (worst / max(peak, .leastNormalMagnitude), differing)
        }

        static func fence(_ arrays: [MLXArray]) throws {
            try withError { error in
                eval(arrays)
                StreamOrDevice.default.stream.synchronize()
                try error.check()
            }
        }

        /// A long KDA step whose q/k/v/beta are column views of one fused
        /// projection output `[T, 3W + 2·128 + H]`, as the fused layer makes them.
        struct LongStep {
            let fused: MLXArray
            let inputs: GLM5NextLinearDecodeKernel.Inputs
            let rows: GLM5NextLinearDecodeKernel.ProjectionRows

            init(tokens: Int, heads: Int, dimension: Int, seed: UInt64) {
                let width = heads * dimension
                let n = 3 * width + 256 + heads
                let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 6)
                fused = MLXRandom.normal([tokens, n], key: keys[0]).asType(.bfloat16)
                let shape = [1, tokens, heads, dimension]
                let parts = split(fused, indices: [width, 2 * width, 3 * width, 3 * width + 128,
                    3 * width + 256], axis: -1)
                let gates = MLXRandom.normal([2, 1, tokens, heads, dimension], key: keys[1])
                    .asType(.bfloat16)
                rows = .init(buffer: fused, betaColumn: 3 * width + 256)
                inputs = .init(
                    query: parts[0].reshaped(shape), key: parts[1].reshaped(shape),
                    value: parts[2].reshaped(shape),
                    gates: .projected(forget: gates[0], output: gates[1]),
                    betaLogits: parts[5].reshaped(1, tokens, heads),
                    queryConvolution: MLXRandom.normal([width, 1, 4], key: keys[2]) * 0.3,
                    keyConvolution: MLXRandom.normal([width, 1, 4], key: keys[3]) * 0.3,
                    valueConvolution: MLXRandom.normal([width, 1, 4], key: keys[4]) * 0.3,
                    aLog: MLXArray.zeros([heads]), dtBias: MLXArray.zeros([width]),
                    outputNorm: MLXArray.ones([dimension]).asType(.bfloat16), state: nil,
                    validity: MLXArray.ones([1, tokens], dtype: .bool))
            }

            func inPlace(_ enabled: Bool) -> GLM5NextLinearDecodeKernel.Inputs {
                var copy = inputs
                copy.projectionRows = enabled ? rows : nil
                return copy
            }
        }
    }

    /// The native prefill GEMM against MLX's quantized matmul and an FP32
    /// product of the same dequantized weights, including row and column tails.
    @Suite(.serialized)
    struct GLM5NextPrefillGEMMTests {
        @Test(arguments: [(130, 200, 256), (300, 256, 512), (512, 4160, 1024)])
        func matchesQuantizedMatmul(m: Int, n: Int, k: Int) throws {
            guard GLM5NextPrefillGEMMKernel.naxAvailable else { return }
            try Device.withDefaultDevice(.gpu) {
                let linear = PrefillGEMM.linear(n: n, k: k, seed: 3)
                let x = PrefillGEMM.input(m: m, k: k, seed: 4)
                let dense = dequantized(
                    linear.weight, scales: linear.scales, biases: linear.biases,
                    groupSize: 64, bits: 4)
                let exact = matmul(x.asType(.float32), dense.asType(.float32).T)
                let mlx = linear(x)
                var outputs = [exact, mlx]
                let configs: [GLM5NextPrefillGEMMKernel.Config] = [
                    .default,
                    .init(bm: 64, bn: 64, bk: 64, wm: 2, wn: 2, swizzle: 0, doubleBuffer: false),
                    .init(bm: 128, bn: 64, bk: 64, wm: 4, wn: 2, swizzle: 1, doubleBuffer: true),
                ]
                for config in configs {
                    outputs.append(try GLM5NextPrefillGEMMKernel.matmul(x, linear, config: config))
                }
                try PrefillGEMM.fence(outputs)
                let (mlxError, _) = PrefillGEMM.difference(mlx, exact)
                for (config, native) in zip(configs, outputs.dropFirst(2)) {
                    #expect(native.shape == [m, n])
                    let (error, _) = PrefillGEMM.difference(native, exact)
                    // One BF16 output rounding (2^-8 relative) bounds both paths.
                    #expect(error <= max(mlxError, 1e-6) * 2 + 4e-3, "\(config)")
                    let (vsMLX, differing) = PrefillGEMM.difference(native, mlx)
                    #expect(vsMLX <= 8e-3, "\(config) differing=\(differing)")
                }
                let batched = try GLM5NextPrefillGEMMKernel.matmul(
                    x.reshaped(2, m / 2, k), linear)
                #expect(batched.shape == [2, m / 2, n])
            }
        }
    }

    /// Long KDA steps reading q/k/v/beta in place from the fused projection
    /// produce the same bits as reading contiguous copies of the views.
    /// Not yet run on hardware; enabled with `GLM53_GEMM_BENCH` until qualified.
    @Suite(
        .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["GLM53_GEMM_BENCH"] != nil))
    struct GLM5NextKDAInPlaceReadTests {
        @Test(arguments: [(40, 2, 32), (70, 4, 64)])
        func inPlaceReadsAreBitwise(tokens: Int, heads: Int, dimension: Int) throws {
            try Device.withDefaultDevice(.gpu) {
                let step = PrefillGEMM.LongStep(
                    tokens: tokens, heads: heads, dimension: dimension, seed: 11)
                let results = try [false, true].map {
                    try GLM5NextLinearDecodeKernel.apply(
                        step.inPlace($0), normEpsilon: 1e-5, gateLowerBound: -5)
                }
                let arrays = results.flatMap {
                    [$0.values, $0.state.convolution, $0.state.recurrent]
                }
                try PrefillGEMM.fence(arrays)
                for (copied, inPlace) in zip(arrays.prefix(3), arrays.suffix(3)) {
                    #expect(copied.shape == inPlace.shape)
                    #expect(arrayEqual(copied, inPlace).item(Bool.self))
                }
                #expect(abs(results[0].values.asType(.float32)).sum().item(Float.self) > 0)
            }
        }

        @Test func mismatchedRowsAreRejected() throws {
            try Device.withDefaultDevice(.gpu) {
                let step = PrefillGEMM.LongStep(tokens: 20, heads: 2, dimension: 32, seed: 2)
                var inputs = step.inputs
                inputs.projectionRows = .init(buffer: step.fused, betaColumn: step.fused.dim(-1) - 1)
                #expect(throws: GLM5NextLinearDecodeKernel.Failure.self) {
                    try GLM5NextLinearDecodeKernel.apply(inputs, normEpsilon: 1e-5, gateLowerBound: -5)
                }
            }
        }
    }

    /// TFLOPS of BF16 dense matmul, MLX's Q4 quantized matmul and the native
    /// prefill GEMM at GLM-5.3 projection shapes. Diagnostic only; enabled by
    /// `GLM53_GEMM_BENCH` (comma-separated row counts, e.g. `512,2048,8192`).
    @Suite(
        .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["GLM53_GEMM_BENCH"] != nil))
    struct GLM5NextPrefillGEMMBenchTests {
        /// Best per-call milliseconds over five evaluations of `repeats` calls.
        private static func time(repeats: Int, _ body: () throws -> MLXArray) rethrows -> Double {
            var best = Double.infinity
            for _ in 0 ..< 5 {
                let start = ContinuousClock.now
                var outputs: [MLXArray] = []
                for _ in 0 ..< repeats {
                    outputs.append(try body())
                }
                eval(outputs)
                StreamOrDevice.default.stream.synchronize()
                let elapsed = ContinuousClock.now - start
                best = min(
                    best,
                    (Double(elapsed.components.seconds) * 1000
                        + Double(elapsed.components.attoseconds) / 1e15) / Double(repeats))
            }
            return best
        }

        static let shapes: [(String, Int, Int)] = [
            ("kda_in", 24896, 4096), ("kda_o", 4096, 8192), ("q_a", 1536, 4096),
            ("q_b", 16384, 1536), ("shared_gu", 4096, 4096), ("shared_down", 4096, 2048),
            ("dsa_o", 4096, 16384), ("kda_fb", 8192, 128), ("mlp_gu", 24576, 4096),
            ("mlp_down", 4096, 12288),
        ]

        static let configs: [GLM5NextPrefillGEMMKernel.Config] = [
            .default,
            .init(bm: 64, bn: 64, bk: 64, wm: 2, wn: 2, swizzle: 0, doubleBuffer: false),
            .init(bm: 128, bn: 64, bk: 64, wm: 4, wn: 2, swizzle: 0, doubleBuffer: true),
            .init(bm: 128, bn: 64, bk: 64, wm: 4, wn: 2, swizzle: 1, doubleBuffer: true),
            .init(bm: 128, bn: 64, bk: 64, wm: 4, wn: 2, swizzle: 3, doubleBuffer: true),
            .init(bm: 128, bn: 64, bk: 64, wm: 2, wn: 2, swizzle: 2, doubleBuffer: true),
            .init(bm: 256, bn: 64, bk: 64, wm: 4, wn: 2, swizzle: 2, doubleBuffer: true),
            .init(bm: 256, bn: 64, bk: 64, wm: 8, wn: 1, swizzle: 2, doubleBuffer: true),
            .init(bm: 256, bn: 64, bk: 64, wm: 8, wn: 1, swizzle: 1, doubleBuffer: true),
            .init(bm: 128, bn: 128, bk: 32, wm: 2, wn: 2, swizzle: 2, doubleBuffer: true),
        ]

        /// KDA long-step stages at GLM-5.3 geometry (H 64, D 128).
        @Test func kdaLongStepStages() throws {
            let rowCounts = (ProcessInfo.processInfo.environment["GLM53_GEMM_BENCH"] ?? "")
                .split(separator: ",").compactMap { Int($0) }
            try Device.withDefaultDevice(.gpu) {
                for tokens in rowCounts {
                    let step = PrefillGEMM.LongStep(tokens: tokens, heads: 64, dimension: 128, seed: 5)
                    try PrefillGEMM.fence([step.fused] + [step.inputs.query, step.inputs.queryConvolution])
                    let zeros = MLXArray.zeros([1, 3 * 8192, 4], dtype: .bfloat16)
                    let state = MLXArray.zeros([1, 64, 128, 128], dtype: .float32)
                    guard case .projected(let forget, let gate) = step.inputs.gates else { return }
                    var stages: [(String, Double)] = []
                    for inPlace in [false, true] {
                        let inputs = step.inPlace(inPlace)
                        stages.append(("apply inPlace=\(inPlace)", try Self.time(repeats: 3) {
                            try GLM5NextLinearDecodeKernel.apply(
                                inputs, normEpsilon: 1e-5, gateLowerBound: -5).values
                        }))
                        stages.append(("convolution inPlace=\(inPlace)", Self.time(repeats: 3) {
                            GLM5NextLinearDecodeKernel.longConvolution(inputs, convolution: zeros).activated
                        }))
                    }
                    let activated = GLM5NextLinearDecodeKernel.longConvolution(
                        step.inputs, convolution: zeros).activated
                    let core = GLM5NextLinearDecodeKernel.longRecurrence(
                        step.inputs, activated: activated, forget: forget, recurrent: state,
                        gateLowerBound: -5, queryScale: 0.088).core
                    try PrefillGEMM.fence([activated, core])
                    stages.append(("recurrence", Self.time(repeats: 3) {
                        GLM5NextLinearDecodeKernel.longRecurrence(
                            step.inputs, activated: activated, forget: forget, recurrent: state,
                            gateLowerBound: -5, queryScale: 0.088).core
                    }))
                    stages.append(("output_norm", Self.time(repeats: 3) {
                        GLM5NextLinearDecodeKernel.longOutput(
                            step.inputs, core: core, outputGate: gate, normEpsilon: 1e-5)
                    }))
                    stages.append(("copy_one_view", Self.time(repeats: 3) { step.inputs.query + 0 }))
                    for (name, ms) in stages {
                        print(String(format: "GLM53_GEMM_BENCH kda tokens=%d %@ ms=%.3f", tokens, name, ms))
                    }
                }
            }
        }

        @Test func projectionThroughput() throws {
            let rowCounts = (ProcessInfo.processInfo.environment["GLM53_GEMM_BENCH"] ?? "")
                .split(separator: ",").compactMap { Int($0) }
            try Device.withDefaultDevice(.gpu) {
                for (name, n, k) in Self.shapes {
                    let linear = PrefillGEMM.linear(n: n, k: k, seed: 7)
                    let dense = dequantized(
                        linear.weight, scales: linear.scales, biases: linear.biases,
                        groupSize: 64, bits: 4)
                    let packedBytes = Double(
                        linear.weight.nbytes + linear.scales.nbytes + linear.biases!.nbytes)
                    try PrefillGEMM.fence([dense])
                    for m in rowCounts {
                        let x = PrefillGEMM.input(m: m, k: k, seed: 8)
                        let mlx = linear(x)
                        try PrefillGEMM.fence([x, mlx])
                        let flops = 2.0 * Double(m) * Double(n) * Double(k)
                        let repeats = max(1, min(20, Int(4e11 / flops)))
                        func report(_ label: String, _ ms: Double, weightBytes: Double) {
                            let bytes = weightBytes + Double(m * (k + n) * 2)
                            print(String(
                                format: "GLM53_GEMM_BENCH %@ m=%d n=%d k=%d %@ ms=%.3f tflops=%.1f gbps=%.0f",
                                name, m, n, k, label, ms, flops / ms / 1e9, bytes / ms / 1e6))
                        }
                        report("bf16_dense", Self.time(repeats: repeats) { matmul(x, dense.T) },
                            weightBytes: Double(dense.nbytes))
                        report("mlx_qmm", Self.time(repeats: repeats) { linear(x) },
                            weightBytes: packedBytes)
                        if name.hasPrefix("mlp_") {
                            // Dense MLPs of the NVFP4 artifact: MLX nvfp4 (group 16,
                            // FP8 scales), alone and with GLM5NextNVFP4Linear's
                            // FP32 global-scale epilogue.
                            let fp = quantized(dense, groupSize: 16, bits: 4, mode: .nvfp4)
                            let nvfp4 = try GLM5NextNVFP4Linear(
                                weight: fp.wq, scales: fp.scales, globalScale: MLXArray([Float(1)]))
                            let fpBytes = Double(fp.wq.nbytes + fp.scales.nbytes)
                            try PrefillGEMM.fence([fp.wq, fp.scales])
                            report("mlx_nvfp4_qmm", Self.time(repeats: repeats) {
                                quantizedMM(
                                    x, fp.wq, scales: fp.scales, biases: nil, transpose: true,
                                    groupSize: 16, bits: 4, mode: .nvfp4)
                            }, weightBytes: fpBytes)
                            report("nvfp4_linear(+global_scale)", Self.time(repeats: repeats) {
                                nvfp4(x)
                            }, weightBytes: fpBytes)
                        }
                        for config in Self.configs where config.bm <= max(64, m) {
                            let native = try GLM5NextPrefillGEMMKernel.matmul(
                                x, linear, config: config)
                            try PrefillGEMM.fence([native])
                            let (error, differing) = PrefillGEMM.difference(native, mlx)
                            let ms = try Self.time(repeats: repeats) {
                                try GLM5NextPrefillGEMMKernel.matmul(x, linear, config: config)
                            }
                            report("native_\(config) vs_mlx=\(String(format: "%.2e/%.4f", error, differing))", ms,
                                weightBytes: packedBytes)
                        }
                    }
                }
            }
        }
    }
}
