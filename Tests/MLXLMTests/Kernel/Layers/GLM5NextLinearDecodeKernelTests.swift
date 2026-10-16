// Copyright 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {
    /// The single-launch decode step must match the composed single-token step:
    /// values and recurrent state within reduction-order tolerance, and the
    /// retained convolution window exactly.
    @Suite(.serialized)
    struct GLM5NextLinearDecodeKernelTests {
        private typealias Composed = GLM5NextLinearAttention

        private static func tensor(_ shape: [Int], salt: Int, scale: Float = 1) -> MLXArray {
            let count = shape.reduce(1, *)
            return MLXArray(
                (0 ..< count).map { Float(($0 * 17 + salt * 13) % 101 - 50) / 51 * scale }, shape)
        }

        private func maximumError(_ actual: MLXArray, _ expected: MLXArray) -> Float {
            abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
        }

        private func compare(
            dtype: DType, batch: Int, heads: Int, dimension: Int, withState: Bool,
            validity: [Bool], tokens: Int = 1
        ) throws {
            let kernel = 4
            let width = heads * dimension
            let shape = [batch, tokens, heads, dimension]
            let raw = (1 ... 3).map { Self.tensor(shape, salt: $0).asType(dtype) }
            let forget = Self.tensor(shape, salt: 4, scale: 3).asType(dtype)
            let betaLogits = Self.tensor([batch, tokens, heads], salt: 5, scale: 2).asType(dtype)
            let gate = Self.tensor(shape, salt: 6, scale: 2).asType(dtype)
            let convolutions = (7 ... 9).map {
                Self.tensor([width, 1, kernel], salt: $0, scale: 0.4).asType(dtype)
            }
            let aLog = Self.tensor([heads], salt: 10)
            let dtBias = Self.tensor([width], salt: 11).asType(dtype)
            let outputNorm = Self.tensor([dimension], salt: 12).asType(dtype)
            let state =
                withState
                ? Composed.State(
                    convolution: Self.tensor([batch, 3 * width, kernel], salt: 13).asType(dtype),
                    recurrent: Self.tensor(
                        [batch, heads, dimension, dimension], salt: 14, scale: 0.1))
                : nil
            let valid = MLXArray(validity, [batch, tokens])
            // The composed path masks x before the bias-free projections.
            let mask = valid.reshaped(batch, tokens, 1, 1)
            let masked = raw.map { MLX.where(mask, $0, MLXArray(Float(0)).asType(dtype)) }
            let maskedForget = MLX.where(mask, forget, MLXArray(Float(0)).asType(dtype))
            let maskedBeta = MLX.where(
                valid.reshaped(batch, tokens, 1), betaLogits, MLXArray(Float(0)).asType(dtype))
            let maskedGate = MLX.where(mask, gate, MLXArray(Float(0)).asType(dtype))
            let composedState =
                state
                ?? Composed.State(
                    convolution: MLXArray.zeros([batch, 3 * width, kernel], dtype: dtype),
                    recurrent: MLXArray.zeros(
                        [batch, heads, dimension, dimension], dtype: .float32))
            let expected = try Composed.apply(
                query: masked[0], key: masked[1], value: masked[2], forgetLogits: maskedForget,
                betaLogits: maskedBeta, outputGate: maskedGate,
                weights: .init(
                    convolution: concatenated(convolutions, axis: 0).squeezed(axis: 1),
                    aLog: aLog, dtBias: dtBias, outputNorm: outputNorm),
                state: composedState, validity: valid, normEpsilon: 1e-5, gateLowerBound: -5,
                execution: .nativeRequired)
            let actual = try GLM5NextLinearDecodeKernel.apply(
                .init(
                    query: masked[0], key: masked[1], value: masked[2],
                    gates: .projected(forget: maskedForget, output: maskedGate),
                    betaLogits: maskedBeta,
                    queryConvolution: convolutions[0], keyConvolution: convolutions[1],
                    valueConvolution: convolutions[2], aLog: aLog, dtBias: dtBias,
                    outputNorm: outputNorm, state: state, validity: valid),
                normEpsilon: 1e-5, gateLowerBound: -5)
            eval(
                expected.values, expected.state.convolution, expected.state.recurrent,
                actual.values, actual.state.convolution, actual.state.recurrent)
            let label = "\(dtype) B\(batch) H\(heads) D\(dimension) state=\(withState)"
            #expect(actual.values.dtype == dtype && actual.values.shape == shape, "\(label)")
            #expect(actual.state.convolution.dtype == dtype, "\(label)")
            #expect(
                actual.state.convolution.asType(.float32).asArray(Float.self)
                    == expected.state.convolution.asType(.float32).asArray(Float.self),
                "\(label) convolution window")
            let low = dtype != .float32
            // FP32 reduction-order differences accumulate along the recurrence.
            let steps = Float(max(1, tokens / 8))
            let valueError = maximumError(actual.values, expected.values)
            #expect(valueError <= (low ? 0.03 : 5e-5 * steps), "\(label) values \(valueError)")
            let stateError = maximumError(actual.state.recurrent, expected.state.recurrent)
            #expect(
                stateError <= (low ? 3e-3 * steps : 2e-5 * steps), "\(label) state \(stateError)")
            if let state, dtype != .float32 {
                // Native CBv2 keeps the raw history in FP32; it passes through.
                let canonical = try GLM5NextLinearDecodeKernel.apply(
                    .init(
                        query: masked[0], key: masked[1], value: masked[2],
                        gates: .projected(forget: maskedForget, output: maskedGate),
                        betaLogits: maskedBeta,
                        queryConvolution: convolutions[0], keyConvolution: convolutions[1],
                        valueConvolution: convolutions[2], aLog: aLog, dtBias: dtBias,
                        outputNorm: outputNorm,
                        state: .init(
                            convolution: state.convolution.asType(.float32),
                            recurrent: state.recurrent),
                        validity: valid),
                    normEpsilon: 1e-5, gateLowerBound: -5)
                eval(canonical.values, canonical.state.convolution, canonical.state.recurrent)
                #expect(canonical.state.convolution.dtype == .float32, "\(label) canonical dtype")
                #expect(
                    canonical.state.convolution.asArray(Float.self)
                        == actual.state.convolution.asType(.float32).asArray(Float.self),
                    "\(label) canonical window")
                #expect(
                    maximumError(canonical.values, actual.values) == 0
                        && maximumError(canonical.state.recurrent, actual.state.recurrent) == 0,
                    "\(label) canonical results")
            }
            for (row, isValid) in validity.enumerated() where !isValid && tokens == 1 {
                #expect(
                    actual.values[row].asType(.float32).abs().max().item(Float.self) == 0,
                    "\(label) invalid row \(row) output")
                #expect(
                    maximumError(actual.state.recurrent[row], composedState.recurrent[row]) == 0,
                    "\(label) invalid row \(row) state")
            }
        }

        @Test func decodeStepMatchesComposedStepOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                for dtype: DType in [.float32, .bfloat16, .float16] {
                    for dimension in [2, 33, 64, 128] {
                        for withState in [false, true] {
                            try compare(
                                dtype: dtype, batch: 3, heads: 2, dimension: dimension,
                                withState: withState, validity: [true, false, true])
                        }
                    }
                }
                // Real GLM-5.3 geometry, one row.
                try compare(
                    dtype: .bfloat16, batch: 1, heads: 64, dimension: 128, withState: true,
                    validity: [true])
                // Verification windows: several tokens, including skipped ones.
                for dtype: DType in [.float32, .bfloat16] {
                    // Prefill lengths walk several 8-token blocks.
                    for (tokens, dimension) in [
                        (2, 128), (4, 64), (8, 128), (3, 32), (20, 128), (37, 64),
                    ] {
                        try compare(
                            dtype: dtype, batch: 2, heads: 2, dimension: dimension,
                            withState: true,
                            validity: (0 ..< 2 * tokens).map { $0 != 1 && $0 != 2 * tokens - 2 },
                            tokens: tokens)
                    }
                }
            }
        }

        /// The in-kernel Q4 f_b/g_b rows match running them as separate
        /// quantized matmuls before the kernel.
        @Test func lowRankGatesMatchProjectedGatesOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                for (dtype, heads, dimension, batch) in [
                    (DType.bfloat16, 64, 128, 1), (.bfloat16, 3, 128, 2), (.float32, 2, 64, 2),
                ] {
                    let width = heads * dimension
                    let shape = [batch, 1, heads, dimension]
                    let keys = MLXRandom.split(key: MLXRandom.key(UInt64(width + batch)), into: 16)
                    func packed(_ key: MLXArray) -> QuantizedLinear {
                        let weight = MLXRandom.normal([width, 128], key: key) * 0.2
                        let (words, scales, biases) = MLX.quantized(weight, groupSize: 64, bits: 4)
                        return QuantizedLinear(
                            weight: words, scales: scales.asType(dtype),
                            biases: biases?.asType(dtype), groupSize: 64, bits: 4)
                    }
                    let fB = packed(keys[0])
                    let gB = packed(keys[1])
                    let fA = (MLXRandom.normal([batch, 1, 128], key: keys[2]) * 2).asType(dtype)
                    let gA = (MLXRandom.normal([batch, 1, 128], key: keys[3]) * 2).asType(dtype)
                    let raw = (4 ... 6).map {
                        MLXRandom.normal(shape, key: keys[$0]).asType(dtype)
                    }
                    let state = Composed.State(
                        convolution: MLXRandom.normal([batch, 3 * width, 4], key: keys[7]).asType(
                            dtype),
                        recurrent: MLXRandom.normal(
                            [batch, heads, dimension, dimension], key: keys[8]) * 0.1)
                    func run(_ gates: GLM5NextLinearDecodeKernel.Gates) throws -> Composed.Output {
                        try GLM5NextLinearDecodeKernel.apply(
                            .init(
                                query: raw[0], key: raw[1], value: raw[2], gates: gates,
                                betaLogits: MLXRandom.normal([batch, 1, heads], key: keys[9])
                                    .asType(dtype),
                                queryConvolution: (MLXRandom.normal([width, 1, 4], key: keys[10])
                                    * 0.4)
                                    .asType(dtype),
                                keyConvolution: (MLXRandom.normal([width, 1, 4], key: keys[11])
                                    * 0.4)
                                    .asType(dtype),
                                valueConvolution: (MLXRandom.normal([width, 1, 4], key: keys[12])
                                    * 0.4)
                                    .asType(dtype),
                                aLog: MLXRandom.normal([heads], key: keys[13]),
                                dtBias: MLXRandom.normal([width], key: keys[14]),
                                outputNorm: MLXRandom.normal([dimension], key: keys[15]).asType(
                                    dtype),
                                state: state, validity: MLXArray.ones([batch, 1], dtype: .bool)),
                            normEpsilon: 1e-5, gateLowerBound: -5)
                    }
                    let lowRank = try #require(
                        GLM5NextLinearDecodeKernel.Gates.inKernel(
                            forgetInput: fA, outputInput: gA, forget: fB, output: gB))
                    let actual = try run(lowRank)
                    let expected = try run(
                        .projected(
                            forget: fB(fA).reshaped(shape), output: gB(gA).reshaped(shape)))
                    eval(
                        actual.values, actual.state.recurrent, expected.values,
                        expected.state.recurrent)
                    let label = "\(dtype) H\(heads) D\(dimension) B\(batch)"
                    let low = dtype != .float32
                    let valueError = maximumError(actual.values, expected.values)
                    #expect(valueError <= (low ? 0.05 : 1e-4), "\(label) values \(valueError)")
                    let stateError = maximumError(actual.state.recurrent, expected.state.recurrent)
                    #expect(stateError <= (low ? 0.02 : 1e-4), "\(label) state \(stateError)")
                }
            }
        }

        /// The captured per-token states of a window equal running its tokens
        /// one step at a time, and the last capture is the returned state.
        @Test func windowCaptureMatchesTokenByTokenSteps() throws {
            try Device.withDefaultDevice(.gpu) {
                let (batch, tokens, heads, dimension, width) = (1, 3, 2, 128, 256)
                let shape = [batch, tokens, heads, dimension]
                let raw = (1 ... 3).map { Self.tensor(shape, salt: $0) }
                let forget = Self.tensor(shape, salt: 4, scale: 3)
                let gate = Self.tensor(shape, salt: 6, scale: 2)
                let beta = Self.tensor([batch, tokens, heads], salt: 5, scale: 2)
                let convolutions = (7 ... 9).map {
                    Self.tensor([width, 1, 4], salt: $0, scale: 0.4)
                }
                let aLog = Self.tensor([heads], salt: 10)
                let dtBias = Self.tensor([width], salt: 11)
                let norm = Self.tensor([dimension], salt: 12)
                func inputs(_ range: Range<Int>, _ state: Composed.State)
                    -> GLM5NextLinearDecodeKernel.Inputs
                {
                    .init(
                        query: raw[0][0..., range], key: raw[1][0..., range],
                        value: raw[2][0..., range],
                        gates: .projected(forget: forget[0..., range], output: gate[0..., range]),
                        betaLogits: beta[0..., range], queryConvolution: convolutions[0],
                        keyConvolution: convolutions[1], valueConvolution: convolutions[2],
                        aLog: aLog, dtBias: dtBias, outputNorm: norm, state: state,
                        validity: MLXArray.ones([batch, range.count], dtype: .bool))
                }
                let initial = Composed.State(
                    convolution: Self.tensor([batch, 3 * width, 4], salt: 13),
                    recurrent: Self.tensor(
                        [batch, heads, dimension, dimension], salt: 14, scale: 0.1))
                let window = try GLM5NextLinearDecodeKernel.apply(
                    inputs(0 ..< tokens, initial), normEpsilon: 1e-5, gateLowerBound: -5,
                    capture: true)
                let captured = try #require(window.capturedRecurrent)
                let capturedWindows = try #require(window.capturedConvolution)
                var state = initial
                for t in 0 ..< tokens {
                    let step = try GLM5NextLinearDecodeKernel.apply(
                        inputs(t ..< t + 1, state), normEpsilon: 1e-5, gateLowerBound: -5)
                    state = step.state
                    eval(step.values, state.recurrent, state.convolution, captured, capturedWindows)
                    #expect(maximumError(captured[0..., t], state.recurrent) <= 2e-5, "state \(t)")
                    #expect(
                        capturedWindows[0..., t].asArray(Float.self)
                            == state.convolution.asArray(Float.self), "window \(t)")
                    #expect(
                        maximumError(window.output.values[0..., t ..< t + 1], step.values) <= 5e-5,
                        "values \(t)")
                }
                eval(window.output.state.recurrent)
                #expect(
                    maximumError(captured[0..., tokens - 1], window.output.state.recurrent) == 0)
            }
        }

        @Test func refusesCPUAndUnsupportedSteps() throws {
            func inputs(tokens: Int) -> GLM5NextLinearDecodeKernel.Inputs {
                let shape = [1, tokens, 1, 32]
                return .init(
                    query: MLXArray.zeros(shape), key: MLXArray.zeros(shape),
                    value: MLXArray.zeros(shape),
                    gates: .projected(forget: MLXArray.zeros(shape), output: MLXArray.zeros(shape)),
                    betaLogits: MLXArray.zeros([1, tokens, 1]),
                    queryConvolution: MLXArray.zeros([32, 1, 4]),
                    keyConvolution: MLXArray.zeros([32, 1, 4]),
                    valueConvolution: MLXArray.zeros([32, 1, 4]), aLog: MLXArray.zeros([1]),
                    dtBias: MLXArray.zeros([32]), outputNorm: MLXArray.zeros([32]), state: nil,
                    validity: MLXArray.ones([1, tokens], dtype: .bool))
            }
            // Captured windows hold one token block; odd head widths run one token.
            #expect(throws: GLM5NextLinearDecodeKernel.Failure.self) {
                try GLM5NextLinearDecodeKernel.apply(
                    inputs(tokens: GLM5NextLinearDecodeKernel.maximumTokens + 1), normEpsilon: 1e-5,
                    gateLowerBound: -5, capture: true)
            }
            #expect(!GLM5NextLinearDecodeKernel.supports(tokens: 2, dimension: 33))
            try Device.withDefaultDevice(.cpu) {
                #expect(throws: GLM5NextLinearDecodeKernel.Failure.unsupportedDeviceOrStream) {
                    try GLM5NextLinearDecodeKernel.apply(
                        inputs(tokens: 1), normEpsilon: 1e-5, gateLowerBound: -5)
                }
            }
        }
    }
}
