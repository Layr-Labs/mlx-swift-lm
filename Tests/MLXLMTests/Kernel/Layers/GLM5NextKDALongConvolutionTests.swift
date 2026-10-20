// Copyright 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    /// The blocked long-step convolution against the serial per-channel walk:
    /// every bit of the activated rows and of the retained window, over
    /// validity holes, stored windows in both storage dtypes, batches and
    /// in-place projection rows. Plus an opt-in timing report
    /// (`GLM53_KDA_CONV_BENCH`, comma-separated token counts, e.g. `2048,4096`).
    @Suite(.serialized)
    struct GLM5NextKDALongConvolutionTests {
        typealias Kernel = GLM5NextLinearDecodeKernel

        enum Holes: String, CaseIterable, Sendable {
            /// Every token valid (prefill chunks).
            case allValid
            /// Isolated invalid tokens, including inside the K tokens before blocks.
            case scattered
            /// Row 1 starts with 20 invalid tokens.
            case leading
            /// The last third of every row is padding.
            case trailing
            /// One valid token in five (windows replay across several blocks).
            case sparse

            func valid(row: Int, token t: Int, tokens: Int) -> Bool {
                switch self {
                case .allValid: true
                case .scattered: !(t % 7 == 3 || t % 11 == 0 || t % 16 == 14)
                case .leading: !(row == 1 && t < 20)
                case .trailing: t < tokens - tokens / 3
                case .sparse: t % 5 == 2
                }
            }
        }

        /// A long step whose q/k/v are column views of one fused projection
        /// `[B·T, 3W + 37]`, as the fused KDA layer makes them.
        struct Step {
            let inputs: Kernel.Inputs
            let rows: Kernel.ProjectionRows
            let convolution: MLXArray

            init(
                batch: Int, tokens: Int, heads: Int, dimension: Int, holes: Holes,
                dtype: DType = .bfloat16, storage: DType = .bfloat16, seed: UInt64
            ) {
                let width = heads * dimension
                let stride = 3 * width + 37
                let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 5)
                let fused = (MLXRandom.normal([batch * tokens, stride], key: keys[0]) * 2)
                    .asType(dtype)
                let shape = [batch, tokens, heads, dimension]
                let parts = split(fused, indices: [width, 2 * width, 3 * width], axis: -1)
                let weights = (0 ..< 3).map {
                    MLXRandom.normal([width, 1, 4], key: MLXRandom.key(seed &+ UInt64($0) &+ 17))
                        * 0.4
                }
                let valid = (0 ..< batch * tokens).map { index in
                    let (row, t) = index.quotientAndRemainder(dividingBy: tokens)
                    return holes.valid(row: row, token: t, tokens: tokens)
                }
                // Stored windows hold activation-dtype values (FP32 storage too).
                convolution = MLXRandom.normal([batch, 3 * width, 4], key: keys[1])
                    .asType(dtype).asType(storage)
                rows = .init(buffer: fused, betaColumn: 3 * width)
                inputs = .init(
                    query: parts[0].reshaped(shape), key: parts[1].reshaped(shape),
                    value: parts[2].reshaped(shape),
                    gates: .projected(
                        forget: MLXArray.zeros(shape, dtype: dtype),
                        output: MLXArray.zeros(shape, dtype: dtype)),
                    betaLogits: MLXArray.zeros([batch, tokens, heads], dtype: dtype),
                    queryConvolution: weights[0], keyConvolution: weights[1],
                    valueConvolution: weights[2], aLog: MLXArray.zeros([heads]),
                    dtBias: MLXArray.zeros([width]),
                    outputNorm: MLXArray.ones([dimension]).asType(dtype), state: nil,
                    validity: MLXArray(valid, [batch, tokens]))
            }

            func inPlace(_ enabled: Bool) -> Kernel.Inputs {
                var copy = inputs
                copy.projectionRows = enabled ? rows : nil
                return copy
            }
        }

        static func fence(_ arrays: [MLXArray]) throws {
            try withError { error in
                eval(arrays)
                StreamOrDevice.default.stream.synchronize()
                try error.check()
            }
        }

        static func same(_ a: MLXArray, _ b: MLXArray) -> Bool {
            a.shape == b.shape && a.dtype == b.dtype && arrayEqual(a, b).item(Bool.self)
        }

        @Test(
            arguments: [
                (1, 9, 2, 32), (1, 16, 2, 32), (1, 17, 2, 32), (2, 33, 2, 32), (1, 100, 2, 32),
                (2, 515, 2, 32), (1, 300, 4, 64),
            ])
        func blockedWalkMatchesSerialWalkBitwise(
            batch: Int, tokens: Int, heads: Int, dimension: Int
        ) throws {
            try Device.withDefaultDevice(.gpu) {
                for holes in Holes.allCases {
                    for storage in [DType.bfloat16, .float32] {
                        let step = Step(
                            batch: batch, tokens: tokens, heads: heads, dimension: dimension,
                            holes: holes, storage: storage, seed: UInt64(tokens * 31 + heads))
                        var outputs: [MLXArray] = []
                        for inPlace in [false, true] {
                            for blocked in [false, true] {
                                let walked = Kernel.longConvolution(
                                    step.inPlace(inPlace), convolution: step.convolution,
                                    blocked: blocked)
                                outputs += [walked.activated, walked.convolution]
                            }
                        }
                        try Self.fence(outputs)
                        let label = "B\(batch) T\(tokens) H\(heads) \(holes) \(storage)"
                        // outputs: [copied serial, copied blocked, in-place serial, in-place blocked]
                        for pair in stride(from: 2, to: outputs.count, by: 2) {
                            #expect(Self.same(outputs[pair], outputs[0]), "activated \(label) \(pair)")
                            #expect(
                                Self.same(outputs[pair + 1], outputs[1]), "window \(label) \(pair)")
                        }
                        #expect(abs(outputs[0].asType(.float32)).sum().item(Float.self) > 0)
                    }
                }
            }
        }

        /// Serving geometry, also in FP32 activations (the 1024-row tails and
        /// a block-straddling hole run in the same launch).
        @Test func servingGeometryBitwise() throws {
            try Device.withDefaultDevice(.gpu) {
                for (dtype, holes) in [(DType.bfloat16, Holes.allValid), (.float32, .scattered)] {
                    let step = Step(
                        batch: 1, tokens: 1043, heads: 32, dimension: 128, holes: holes,
                        dtype: dtype, storage: dtype, seed: 9)
                    let serial = Kernel.longConvolution(
                        step.inputs, convolution: step.convolution, blocked: false)
                    let blocked = Kernel.longConvolution(
                        step.inPlace(true), convolution: step.convolution)
                    try Self.fence([
                        serial.activated, serial.convolution, blocked.activated,
                        blocked.convolution,
                    ])
                    #expect(Self.same(blocked.activated, serial.activated), "\(dtype)")
                    #expect(Self.same(blocked.convolution, serial.convolution), "\(dtype)")
                }
            }
        }

        /// Median-free best-of-five milliseconds per call of `repeats` calls.
        private static func time(repeats: Int, _ body: () -> MLXArray) -> Double {
            var best = Double.infinity
            for _ in 0 ..< 5 {
                let start = ContinuousClock.now
                var outputs: [MLXArray] = []
                for _ in 0 ..< repeats { outputs.append(body()) }
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

        @Test(
            .enabled(if: ProcessInfo.processInfo.environment["GLM53_KDA_CONV_BENCH"] != nil))
        func timing() throws {
            let rowCounts = (ProcessInfo.processInfo.environment["GLM53_KDA_CONV_BENCH"] ?? "")
                .split(separator: ",").compactMap { Int($0) }
            try Device.withDefaultDevice(.gpu) {
                for tokens in rowCounts {
                    for heads in [32, 64] {
                        let step = Step(
                            batch: 1, tokens: tokens, heads: heads, dimension: 128,
                            holes: .allValid, seed: 5)
                        try Self.fence([step.rows.buffer, step.convolution])
                        var line = "GLM53_KDA_CONV_BENCH T\(tokens) H\(heads)"
                        for (name, inPlace, blocked) in [
                            ("serial_copied", false, false), ("serial_inplace", true, false),
                            ("blocked_copied", false, true), ("blocked_inplace", true, true),
                        ] {
                            let inputs = step.inPlace(inPlace)
                            let ms = Self.time(repeats: 3) {
                                Kernel.longConvolution(
                                    inputs, convolution: step.convolution, blocked: blocked
                                ).activated
                            }
                            line += String(format: " | %@ %.3f ms", name, ms)
                        }
                        print(line)
                    }
                }
            }
        }
    }
}
