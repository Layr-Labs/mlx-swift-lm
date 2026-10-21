// Copyright 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {
    /// Candidate qualification only; no test enables a model/default kernel path.
    @Suite(.serialized)
    struct GLM5NextLinearKernelTests {
        private typealias Candidate = GLM5NextLinearKernel
        private typealias Baseline = GLM5NextLinearAttention

        private struct Inputs {
            let q: MLXArray
            let k: MLXArray
            let v: MLXArray
            let logDecay: MLXArray
            let beta: MLXArray
            let state: MLXArray

            func native(capture: Int = 0, initial: MLXArray? = nil) throws -> Candidate.Output {
                try Candidate.apply(
                    query: q, key: k, value: v, logDecay: logDecay, beta: beta,
                    state: initial ?? state, capturePrefixCount: capture)
            }

            func slice(_ range: Range<Int>) -> Inputs {
                Inputs(
                    q: q[0..., range], k: k[0..., range], v: v[0..., range],
                    logDecay: logDecay[0..., range], beta: beta[0..., range], state: state)
            }

            /// Literal recurrence in GLM5NextLinearAttention.apply, after its
            /// precomputed q/k/v/gate boundary. No GDN or candidate helper is used.
            /// Materializing each step bounds test memory on the full H=64 shape;
            /// this schedule is NOT a valid performance comparator.
            func baseline(initial: MLXArray? = nil, materializeEachStep: Bool = false)
                -> (values: MLXArray, state: MLXArray)
            {
                var recurrent = initial ?? state
                var outputs: [MLXArray] = []
                for t in 0 ..< q.dim(1) {
                    let qt = q[0..., t]
                    let kt = k[0..., t]
                    recurrent = recurrent * exp(logDecay[0..., t])[.ellipsis, .newAxis]
                    let memory = (recurrent * kt[.ellipsis, .newAxis]).sum(axis: -2)
                    let delta = (v[0..., t] - memory) * beta[0..., t][.ellipsis, .newAxis]
                    recurrent =
                        recurrent + kt[.ellipsis, .newAxis] * delta[0..., 0..., .newAxis, 0...]
                    let output = (recurrent * qt[.ellipsis, .newAxis]).sum(axis: -2)
                    outputs.append(output)
                    if materializeEachStep { eval(recurrent, output) }
                }
                return (stacked(outputs, axis: 1), recurrent)
            }
        }

        private static func values(_ count: Int, salt: Int, scale: Float = 1) -> [Float] {
            (0 ..< count).map { Float(($0 * 17 + salt * 13) % 101 - 50) / 51 * scale }
        }

        private static func tensor(_ shape: [Int], salt: Int, scale: Float = 1) -> MLXArray {
            MLXArray(values(shape.reduce(1, *), salt: salt, scale: scale), shape)
        }

        private static func fixture(batch: Int, tokens: Int, heads: Int, key: Int, value: Int)
            -> Inputs
        {
            let rows = batch * tokens * heads
            func normalized(_ salt: Int, query: Bool) -> MLXArray {
                var data = values(rows * key, salt: salt)
                for row in 0 ..< rows {
                    var squared: Float = 0
                    for d in 0 ..< key { squared += data[row * key + d] * data[row * key + d] }
                    let denominator = Foundation.sqrt(squared + 1e-6)
                    for d in 0 ..< key {
                        data[row * key + d] /= denominator
                        if query { data[row * key + d] *= 1 / Foundation.sqrt(Float(key)) }
                    }
                }
                return MLXArray(data, [batch, tokens, heads, key])
            }
            let decay = (0 ..< rows * key).map { index -> Float in
                // Deliberately unequal key gates, including no decay and near reset.
                if index % 19 == 0 { return 0 }
                if index % 23 == 0 { return -5 }
                return -Float(index % 31 + 1) / 97
            }
            return Inputs(
                q: normalized(1, query: true), k: normalized(4, query: false),
                v: tensor([batch, tokens, heads, value], salt: 3),
                logDecay: MLXArray(decay, [batch, tokens, heads, key]),
                beta: MLXArray(
                    (0 ..< rows).map { Float($0 % 17 + 1) / 19 }, [batch, tokens, heads]),
                state: tensor([batch, heads, key, value], salt: 9, scale: 0.2))
        }

        private func expectClose(
            _ actual: MLXArray, _ expected: MLXArray, tolerance: Float = 2e-5,
            sourceLocation: SourceLocation = #_sourceLocation
        ) {
            #expect(actual.shape == expected.shape, sourceLocation: sourceLocation)
            let a = actual.asType(.float32).asArray(Float.self)
            let e = expected.asType(.float32).asArray(Float.self)
            #expect(a.count == e.count, sourceLocation: sourceLocation)
            #expect(
                a.allSatisfy { $0.isFinite } && e.allSatisfy { $0.isFinite },
                sourceLocation: sourceLocation)
            let maximum = zip(a, e).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
            #expect(
                maximum <= tolerance, "max absolute error \(maximum)",
                sourceLocation: sourceLocation)
        }

        private func expectExact(
            _ actual: MLXArray, _ expected: MLXArray,
            sourceLocation: SourceLocation = #_sourceLocation
        ) {
            #expect(actual.shape == expected.shape, sourceLocation: sourceLocation)
            #expect(actual.dtype == expected.dtype, sourceLocation: sourceLocation)
            #expect(
                actual.asArray(Float.self).map(\.bitPattern)
                    == expected.asArray(Float.self).map(\.bitPattern),
                sourceLocation: sourceLocation)
        }

        @Test func scalarFixtureProvesPerKeyDecayAndKeyValueOrientation() throws {
            // Non-square K=2,V=3 and distinct row decays make a transposed state
            // or scalar-per-head gate visibly wrong. Host FP32 scalar oracle.
            let input = Inputs(
                q: MLXArray([Float(0.2), -0.7], [1, 1, 1, 2]),
                k: MLXArray([Float(0.6), 0.8], [1, 1, 1, 2]),
                v: MLXArray([Float(0.5), -0.25, 0.75], [1, 1, 1, 3]),
                logDecay: MLXArray([Float(0), -2], [1, 1, 1, 2]),
                beta: MLXArray([Float(0.3)], [1, 1, 1]),
                state: MLXArray([Float(1), 2, 3, 4, 5, 6], [1, 1, 2, 3]))
            var state: [Float] = [1, 2, 3, 4, 5, 6]
            var output = [Float](repeating: 0, count: 3)
            let key: [Float] = [0.6, 0.8]
            let query: [Float] = [0.2, -0.7]
            let value: [Float] = [0.5, -0.25, 0.75]
            for k in 0 ..< 2 {
                for v in 0 ..< 3 { state[k * 3 + v] *= Foundation.exp(k == 0 ? Float(0) : -2) }
            }
            for v in 0 ..< 3 {
                var memory: Float = 0
                for k in 0 ..< 2 { memory += state[k * 3 + v] * key[k] }
                let delta = (value[v] - memory) * 0.3
                for k in 0 ..< 2 {
                    state[k * 3 + v] += key[k] * delta
                    output[v] += state[k * 3 + v] * query[k]
                }
            }
            let result = try input.native(capture: 1)
            expectClose(result.values, MLXArray(output, [1, 1, 1, 3]), tolerance: 1e-6)
            expectClose(result.state, MLXArray(state, [1, 1, 2, 3]), tolerance: 1e-6)
            expectExact(try result.state(acceptingPrefix: 1), result.state)
        }

        @Test func shortAndLongChunksMatchLiteralBaseline() throws {
            // Covers masked-off SIMD lanes, multiple register entries per lane,
            // independent batches/heads, asymmetric dimensions and long recurrence.
            for (tokens, key, value) in [
                (1, 1, 1), (7, 3, 5), (9, 31, 7), (3, 32, 4), (17, 33, 13), (257, 128, 128),
            ] {
                let input = Self.fixture(batch: 2, tokens: tokens, heads: 3, key: key, value: value)
                for initial in [input.state, MLXArray.zeros(input.state.shape)] {
                    let result = try input.native(initial: initial)
                    let baseline = input.baseline(initial: initial, materializeEachStep: true)
                    expectClose(result.values, baseline.values)
                    expectClose(result.state, baseline.state)
                    #expect(result.values.dtype == .float32)
                    #expect(result.state.dtype == .float32)
                    #expect(result.capturedPrefixStates == nil)
                }
            }
        }

        @Test func fullH64K128V128GeometryMatchesBaseline() throws {
            for tokens in [1, 7, 65] {
                let input = Self.fixture(batch: 2, tokens: tokens, heads: 64, key: 128, value: 128)
                let captured = min(tokens, 2)
                let result = try input.native(capture: captured)
                let baseline = input.baseline(materializeEachStep: true)
                expectClose(result.values, baseline.values)
                expectClose(result.state, baseline.state)
                let prefix = input.slice(0 ..< captured).baseline(materializeEachStep: true)
                expectClose(try result.state(acceptingPrefix: captured), prefix.state)
            }
        }

        @Test func chunkEdgesAndNoncontiguousViewsPreserveNativeResults() throws {
            let input = Self.fixture(batch: 2, tokens: 129, heads: 3, key: 128, value: 128)
            let whole = try input.native()
            var state = input.state
            var outputs: [MLXArray] = []
            for range in [0 ..< 1, 1 ..< 18, 18 ..< 65, 65 ..< 129] {
                let part = try input.slice(range).native(initial: state)
                state = part.state
                outputs.append(part.values)
            }
            // Same per-token arithmetic and FP32 state at every chunk edge.
            expectExact(concatenated(outputs, axis: 1), whole.values)
            expectExact(state, whole.state)

            let wide = Self.fixture(batch: 2, tokens: 5, heads: 2, key: 66, value: 14)
            let strided = Inputs(
                q: wide.q[.ellipsis, .stride(by: 2)],
                k: wide.k[.ellipsis, .stride(by: 2)],
                v: wide.v[.ellipsis, .stride(by: 2)],
                logDecay: wide.logDecay[.ellipsis, .stride(by: 2)], beta: wide.beta,
                state: wide.state[0..., 0..., .stride(by: 2), .stride(by: 2)])
            let result = try strided.native()
            let baseline = strided.baseline()
            expectClose(result.values, baseline.values)
            expectClose(result.state, baseline.state)
        }

        @Test func boundedCaptureAcceptsZeroPartialAndAllWithoutMutatingInputs() throws {
            let input = Self.fixture(batch: 2, tokens: 5, heads: 2, key: 33, value: 7)
            let originals = [input.q, input.k, input.v, input.logDecay, input.beta, input.state]
            let snapshots = originals.map { $0.asArray(Float.self).map(\.bitPattern) }
            var prefixes: [MLXArray] = []
            var state = input.state
            for t in 0 ..< 5 {
                state = try input.slice(t ..< t + 1).native(initial: state).state
                prefixes.append(state)
            }
            let plain = try input.native()
            for count in [0, 2, 5] {
                let result = try input.native(capture: count)
                expectExact(result.values, plain.values)
                expectExact(result.state, plain.state)
                expectExact(try result.state(acceptingPrefix: 0), input.state)
                expectExact(try result.state(acceptingPrefix: 5), prefixes[4])
                if count > 0 {
                    let captured = try #require(result.capturedPrefixStates)
                    #expect(captured.shape == [2, count, 2, 33, 7])
                    #expect(captured.dtype == .float32)
                    for accepted in 1 ... count {
                        expectExact(
                            try result.state(acceptingPrefix: accepted), prefixes[accepted - 1])
                        expectExact(captured[0..., accepted - 1], prefixes[accepted - 1])
                    }
                }
                if count < 4 {
                    #expect(
                        throws: Candidate.Failure.invalidCapture("accepted prefix was not captured")
                    ) {
                        try result.state(acceptingPrefix: count + 1)
                    }
                }
                for invalid in [-1, 6, Int.max] {
                    #expect(
                        throws: Candidate.Failure.invalidCapture(
                            "accepted prefix is outside the window")
                    ) {
                        try result.state(acceptingPrefix: invalid)
                    }
                }
            }
            // Forks share only read-only input. Evaluate another branch before
            // re-reading the first result to expose accidental input donation.
            let fork = try input.slice(0 ..< 2).native(capture: 2)
            eval(fork.values, fork.state)
            let repeated = try input.native()
            expectExact(repeated.values, plain.values)
            expectExact(repeated.state, plain.state)
            for (array, snapshot) in zip(originals, snapshots) {
                #expect(array.asArray(Float.self).map(\.bitPattern) == snapshot)
            }
        }

        @Test func lazyOutputsRetainTemporaryInputsAndZeroInitialState() throws {
            func temporary() throws -> Candidate.Output {
                let input = Self.fixture(batch: 2, tokens: 5, heads: 2, key: 33, value: 7)
                return try Candidate.apply(
                    query: input.q, key: input.k, value: input.v,
                    logDecay: input.logDecay, beta: input.beta, capturePrefixCount: 2)
            }
            let result = try temporary()
            let input = Self.fixture(batch: 2, tokens: 5, heads: 2, key: 33, value: 7)
            let baseline = input.baseline(initial: MLXArray.zeros(input.state.shape))
            expectClose(result.values, baseline.values)
            expectClose(result.state, baseline.state)
            expectExact(try result.state(acceptingPrefix: 0), MLXArray.zeros(input.state.shape))
            let firstTwo = input.slice(0 ..< 2).baseline(initial: MLXArray.zeros(input.state.shape))
            expectClose(try result.state(acceptingPrefix: 2), firstTwo.state)
        }

        @Test func reducedPrecisionCallerBoundariesMatchUnchangedFullBaseline() throws {
            let batch = 2
            let tokens = 5
            let heads = 2
            let dimension = 33
            let kernel = 4
            let shape = [batch, tokens, heads, dimension]
            let width = heads * dimension
            for dtype: DType in [.float32, .float16, .bfloat16] {
                let raw = (1 ... 3).map { Self.tensor(shape, salt: $0).asType(dtype) }
                let forget = Self.tensor(shape, salt: 4).asType(dtype)
                let logits = Self.tensor([batch, tokens, heads], salt: 5).asType(dtype)
                let outputGate = Self.tensor(shape, salt: 6).asType(dtype)
                let weights = Baseline.Weights(
                    convolution: Self.tensor([3 * width, kernel], salt: 7, scale: 0.3),
                    aLog: Self.tensor([heads], salt: 8),
                    dtBias: Self.tensor([width], salt: 9),
                    outputNorm: Self.tensor([dimension], salt: 10))
                let initial = Baseline.State(
                    convolution: Self.tensor([batch, 3 * width, kernel], salt: 11).asType(dtype),
                    recurrent: Self.tensor(
                        [batch, heads, dimension, dimension], salt: 12, scale: 0.1))
                let history = concatenated(
                    [
                        initial.convolution.transposed(0, 2, 1),
                        concatenated(raw.map { $0.reshaped(batch, tokens, width) }, axis: -1),
                    ],
                    axis: 1)
                let convolved = silu(
                    MLX.conv1d(
                        history[0..., 1...].asType(.float32),
                        weights.convolution.reshaped(3 * width, kernel, 1), groups: 3 * width)
                ).asType(dtype)
                let parts = split(convolved, indices: [width, 2 * width], axis: -1)
                    .map { $0.reshaped(shape).asType(.float32) }
                let q =
                    parts[0] / sqrt(parts[0].square().sum(axis: -1, keepDims: true) + 1e-6)
                    * (1 / Foundation.sqrt(Float(dimension)))
                let k = parts[1] / sqrt(parts[1].square().sum(axis: -1, keepDims: true) + 1e-6)
                let gateInput =
                    forget.asType(.float32) + weights.dtBias.reshaped(1, 1, heads, dimension)
                let rate = exp(weights.aLog).reshaped(1, 1, heads, 1)
                for lowerBound: Float? in [-5, nil] {
                    let logDecay: MLXArray
                    if let lowerBound {
                        logDecay = lowerBound * sigmoid(rate * gateInput)
                    } else {
                        logDecay =
                            -rate * MLX.where(gateInput .> 20, gateInput, log(1 + exp(gateInput)))
                    }
                    let result = try Candidate.apply(
                        query: q, key: k, value: parts[2], logDecay: logDecay,
                        beta: Baseline.betaGate(logits).asType(.float32), state: initial.recurrent)
                    let core = result.values.asType(dtype).asType(.float32)
                    let normalized =
                        core * rsqrt(core.square().mean(axis: -1, keepDims: true) + 1e-5)
                    let actual =
                        (normalized * weights.outputNorm * sigmoid(outputGate.asType(.float32)))
                        .asType(dtype)
                    let baseline = try Baseline.apply(
                        query: raw[0], key: raw[1], value: raw[2], forgetLogits: forget,
                        betaLogits: logits, outputGate: outputGate, weights: weights,
                        state: initial, gateLowerBound: lowerBound)
                    #expect(actual.dtype == dtype)
                    #expect(result.state.dtype == .float32)
                    expectClose(
                        actual, baseline.values, tolerance: dtype == .float32 ? 3e-5 : 0.015)
                    expectClose(result.state, baseline.state.recurrent)
                }
            }
        }

        @Test func rejectsInvalidMetadataBeforeAllocationOrDispatch() throws {
            // This checks shapes without allocating the oversized tensors.
            for (query, value) in [
                ([1, 1, 2], [1, 1, 2, 3]), ([0, 1, 1, 3], [0, 1, 1, 3]),
                ([1, 0, 1, 3], [1, 0, 1, 3]), ([1, 1, 0, 3], [1, 1, 0, 3]),
                ([1, 1, 1, 0], [1, 1, 1, 3]), ([1, 1, 1, 3], [1, 1, 1, 0]),
                ([1, 1, 1, 129], [1, 1, 1, 3]), ([1, 1, 1, 3], [1, 1, 1, 129]),
                ([1, 1, 1, 3], [2, 1, 1, 3]),
                ([Int.max, 1, 1, 1], [Int.max, 1, 1, 1]),
                ([65_536, 65_536, 1, 1], [65_536, 65_536, 1, 1]),
                ([Int(Int32.max), 1, 1, 2], [Int(Int32.max), 1, 1, 2]),
                // Input counts fit Int32 but the recurrent-state product does not.
                ([1_024, 1, 1_024, 128], [1_024, 1, 1_024, 128]),
            ] {
                #expect(throws: Candidate.Failure.self) {
                    try Candidate.Layout(
                        queryShape: query, valueShape: value, capturePrefixCount: 0)
                }
            }
            for count in [-1, 6, Int.max] {
                #expect(throws: Candidate.Failure.self) {
                    try Candidate.Layout(
                        queryShape: [1, 5, 1, 3], valueShape: [1, 5, 1, 7],
                        capturePrefixCount: count)
                }
            }
            #expect(throws: Candidate.Failure.invalidCapture("capture must be in 0...min(T,32)")) {
                try Candidate.Layout(
                    queryShape: [1, 64, 1, 3], valueShape: [1, 64, 1, 7], capturePrefixCount: 33)
            }
            #expect(throws: Candidate.Failure.invalidCapture("captured states exceed 64 MiB")) {
                try Candidate.Layout(
                    queryShape: [2, 9, 64, 128], valueShape: [2, 9, 64, 128], capturePrefixCount: 9)
            }
            let maximum = try Candidate.Layout(
                queryShape: [2, 8, 64, 128], valueShape: [2, 8, 64, 128], capturePrefixCount: 8)
            #expect(maximum.captureShape == [2, 8, 64, 128, 128])
        }

        @Test func rejectsWrongDtypesAndTensorShapesWithoutImplicitCasts() throws {
            let input = Self.fixture(batch: 2, tokens: 5, heads: 2, key: 3, value: 7)
            let arrays = [input.q, input.k, input.v, input.logDecay, input.beta, input.state]
            func run(_ arrays: [MLXArray]) throws -> Candidate.Output {
                try Candidate.apply(
                    query: arrays[0], key: arrays[1], value: arrays[2], logDecay: arrays[3],
                    beta: arrays[4], state: arrays[5])
            }
            for index in arrays.indices {
                for dtype: DType in [.float16, .bfloat16, .int32] {
                    var invalid = arrays
                    invalid[index] = invalid[index].asType(dtype)
                    #expect(throws: Candidate.Failure.self) { try run(invalid) }
                }
            }
            for index in [1, 3, 4, 5] {
                var invalid = arrays
                invalid[index] = invalid[index][0 ..< 1]
                #expect(throws: Candidate.Failure.self) { try run(invalid) }
            }
            var transposed = arrays
            transposed[5] = input.state.transposed(0, 1, 3, 2)
            #expect(throws: Candidate.Failure.self) { try run(transposed) }
            Device.withDefaultDevice(.cpu) {
                #expect(throws: Candidate.Failure.metalUnavailable) { try run(arrays) }
            }
            Stream.withNewDefaultStream(device: .cpu) {
                #expect(throws: Candidate.Failure.metalUnavailable) { try run(arrays) }
            }
        }
    }
}
