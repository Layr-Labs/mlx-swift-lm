// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

extension KernelTests {
    /// Whole-step KDA (`GLM5NextExecutionPolicy.linearAttention`) against the
    /// per-token reference: the chunk-parallel recurrence, validity skipping,
    /// continuation, refusal, and model-level wiring. CPU tests exercise the
    /// operation-only chunk form; native kernel tests require the GPU.
    @Suite(.serialized)
    struct GLM5NextWholeStepLinearAttentionTests {
        private typealias Primitive = GLM5NextLinearAttention
        private typealias Chunked = GLM5NextLinearChunkedRecurrence

        /// Host-generated uniform values in [0, 1): no MLX random state or device work.
        private static func uniform(_ shape: [Int], seed: UInt64) -> MLXArray {
            var state = seed &* 0x9E37_79B9_7F4A_7C15 &+ 1
            let values = (0 ..< shape.reduce(1, *)).map { _ -> Float in
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                return Float(state >> 40) / Float(1 << 24)
            }
            return MLXArray(values, shape)
        }

        /// Zero-mean values with unit variance.
        private static func normal(_ shape: [Int], seed: UInt64, scale: Float = 1) -> MLXArray {
            (uniform(shape, seed: seed) - 0.5) * (Foundation.sqrt(Float(12)) * scale)
        }

        // MARK: Recurrence-only inputs

        private struct Recurrence {
            let q, k, v, logDecay, beta, state: MLXArray

            init(
                q: MLXArray, k: MLXArray, v: MLXArray, logDecay: MLXArray, beta: MLXArray,
                state: MLXArray
            ) {
                self.q = q
                self.k = k
                self.v = v
                self.logDecay = logDecay
                self.beta = beta
                self.state = state
            }

            static func unit(_ x: MLXArray) -> MLXArray {
                x / sqrt(x.square().sum(axis: -1, keepDims: true) + 1e-6)
            }

            /// Nearly parallel keys, slow decay and beta near one make the
            /// in-chunk system (I + M) nearly singular in the Neumann sense:
            /// M's off-diagonal entries approach one.
            static func correlated(
                tokens: Int, heads: Int, key: Int, value: Int, seed: UInt64
            ) -> Recurrence {
                let shape = [1, tokens, heads, key]
                let base = normal([1, 1, heads, key], seed: seed)
                return Recurrence(
                    q: unit(normal(shape, seed: seed + 1)) * (1 / Foundation.sqrt(Float(key))),
                    k: unit(base + 0.05 * normal(shape, seed: seed + 2)),
                    v: normal([1, tokens, heads, value], seed: seed + 3),
                    logDecay: -0.01 * (0.5 + uniform(shape, seed: seed + 4)),
                    beta: 0.8 + 0.2 * uniform([1, tokens, heads], seed: seed + 5),
                    state: normal([1, heads, key, value], seed: seed + 6, scale: 0.3))
            }

            init(batch: Int, tokens: Int, heads: Int, key: Int, value: Int, seed: UInt64) {
                let unit = Self.unit
                let shape = [batch, tokens, heads, key]
                q = unit(normal(shape, seed: seed)) * (1 / Foundation.sqrt(Float(key)))
                k = unit(normal(shape, seed: seed + 1))
                v = normal([batch, tokens, heads, value], seed: seed + 2)
                // Mostly slow decay, so tokens interact across whole chunks,
                // plus channels at the -5 bound: a 32-token chunk then spans
                // the largest admitted in-chunk decay ratios.
                let gate = uniform(shape, seed: seed + 3)
                logDecay = MLX.where(gate .> 0.9, Float(-5), -0.2 * gate)
                beta = uniform([batch, tokens, heads], seed: seed + 4)
                state = normal([batch, heads, key, value], seed: seed + 5, scale: 0.3)
            }

            /// The literal per-token recurrence of `GLM5NextLinearAttention`.
            func literal() -> (values: MLXArray, state: MLXArray) {
                var current = state
                var outputs: [MLXArray] = []
                for t in 0 ..< q.dim(1) {
                    let kt = k[0..., t]
                    current = current * exp(logDecay[0..., t])[.ellipsis, .newAxis]
                    let memory = (current * kt[.ellipsis, .newAxis]).sum(axis: -2)
                    let delta = (v[0..., t] - memory) * beta[0..., t][.ellipsis, .newAxis]
                    current = current + kt[.ellipsis, .newAxis] * delta[0..., 0..., .newAxis, 0...]
                    outputs.append((current * q[0..., t][.ellipsis, .newAxis]).sum(axis: -2))
                    eval(current, outputs[outputs.count - 1])
                }
                return (stacked(outputs, axis: 1), current)
            }

            func chunked(_ size: Int) -> (values: MLXArray, state: MLXArray) {
                Chunked.apply(
                    query: q, key: k, value: v, logDecay: logDecay, beta: beta, state: state,
                    chunkSize: size)
            }
        }

        private static func maxError(_ actual: MLXArray, _ expected: MLXArray) -> Float {
            abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
        }

        private func expectClose(
            _ actual: MLXArray, _ expected: MLXArray, tolerance: Float, _ label: String,
            sourceLocation: SourceLocation = #_sourceLocation
        ) {
            #expect(
                actual.shape == expected.shape, "\(label) shape", sourceLocation: sourceLocation)
            let error = Self.maxError(actual, expected)
            #expect(
                error.isFinite && error <= tolerance, "\(label) max error \(error)",
                sourceLocation: sourceLocation)
        }

        @Test func chunkedRecurrenceMatchesLiteralAcrossChunkSizesAndPadding() throws {
            try Device.withDefaultDevice(.cpu) {
                for (tokens, key, value) in [
                    (1, 3, 5), (7, 4, 4), (32, 8, 6), (33, 16, 16), (70, 9, 7),
                ] {
                    let input = Recurrence(
                        batch: 2, tokens: tokens, heads: 3, key: key, value: value,
                        seed: UInt64(tokens * 31 + key))
                    let expected = input.literal()
                    for size in [1, 2, 3, 8, 16, 32] {
                        let actual = input.chunked(size)
                        let label = "T=\(tokens) K=\(key) C=\(size)"
                        expectClose(actual.values, expected.values, tolerance: 1e-4, "\(label) out")
                        expectClose(actual.state, expected.state, tolerance: 1e-4, "\(label) state")
                        #expect(actual.values.dtype == .float32 && actual.state.dtype == .float32)
                    }
                }
            }
        }

        @Test func chunkedRecurrenceMatchesLiteralAtFullHeadDimension() throws {
            try Device.withDefaultDevice(.cpu) {
                let input = Recurrence(
                    batch: 1, tokens: 65, heads: 2, key: 128, value: 128, seed: 77)
                let expected = input.literal()
                let actual = input.chunked(GLM5NextExecutionPolicy.linearAttentionChunkSize)
                expectClose(actual.values, expected.values, tolerance: 1e-4, "K=V=128 out")
                expectClose(actual.state, expected.state, tolerance: 1e-4, "K=V=128 state")
            }
        }

        /// Truncated-Neumann (repeated-squaring) inversion grows like
        /// (1 + a)^(C-2) on this input; forward substitution does not.
        @Test func chunkedRecurrenceIsStableForCorrelatedKeysAndBetaNearOne() throws {
            try Device.withDefaultDevice(.cpu) {
                let input = Recurrence.correlated(tokens: 64, heads: 2, key: 16, value: 8, seed: 41)
                let expected = input.literal()
                for size in [8, 16, 32] {
                    let actual = input.chunked(size)
                    expectClose(actual.values, expected.values, tolerance: 1e-4, "C=\(size) out")
                    expectClose(actual.state, expected.state, tolerance: 1e-4, "C=\(size) state")
                }
            }
        }

        @Test func chunkAdmissionFollowsTheGateLowerBound() {
            #expect(Chunked.supports(chunkSize: 32, gateLowerBound: -5))
            #expect(Chunked.supports(chunkSize: 33, gateLowerBound: -5))
            #expect(!Chunked.supports(chunkSize: 34, gateLowerBound: -5))
            #expect(!Chunked.supports(chunkSize: 64, gateLowerBound: -5))
            #expect(Chunked.supports(chunkSize: 64, gateLowerBound: -2.5))
            #expect(!Chunked.supports(chunkSize: 65, gateLowerBound: -1))
            #expect(!Chunked.supports(chunkSize: 0, gateLowerBound: -5))
            #expect(!Chunked.supports(chunkSize: 8, gateLowerBound: nil))
            #expect(!Chunked.supports(chunkSize: 8, gateLowerBound: -.infinity))
        }

        // MARK: Projected-input primitive with validity

        private struct Projected {
            let batch, tokens, heads, dimension: Int
            let dtype: DType
            let query, key, value, forget, gate, beta: MLXArray
            let weights: Primitive.Weights
            let state: Primitive.State

            init(batch: Int, tokens: Int, heads: Int, dimension: Int, dtype: DType, seed: UInt64) {
                self.batch = batch
                self.tokens = tokens
                self.heads = heads
                self.dimension = dimension
                self.dtype = dtype
                let shape = [batch, tokens, heads, dimension]
                func input(_ offset: UInt64, scale: Float = 1) -> MLXArray {
                    normal(shape, seed: seed + offset, scale: scale).asType(dtype)
                }
                query = input(0)
                key = input(1)
                value = input(2)
                forget = input(3, scale: 2)
                gate = input(4)
                beta = normal([batch, tokens, heads], seed: seed + 5, scale: 2).asType(dtype)
                let channels = 3 * heads * dimension
                weights = .init(
                    convolution: normal([channels, 4], seed: seed + 6, scale: 0.5),
                    aLog: normal([heads], seed: seed + 7, scale: 0.5),
                    dtBias: normal([heads * dimension], seed: seed + 8, scale: 0.5),
                    outputNorm: normal([dimension], seed: seed + 9))
                state = .init(
                    convolution: normal([batch, channels, 4], seed: seed + 10).asType(dtype),
                    recurrent: normal(
                        [batch, heads, dimension, dimension], seed: seed + 11, scale: 0.3))
            }

            func whole(
                _ range: Range<Int>? = nil, state: Primitive.State? = nil,
                validity: MLXArray?, execution: Primitive.ExecutionPolicy
            ) throws -> Primitive.Output {
                let range = range ?? 0 ..< tokens
                return try Primitive.apply(
                    query: query[0..., range], key: key[0..., range], value: value[0..., range],
                    forgetLogits: forget[0..., range], betaLogits: beta[0..., range],
                    outputGate: gate[0..., range], weights: weights, state: state ?? self.state,
                    validity: validity.map { $0[0..., range] }, execution: execution)
            }

            /// The per-token schedule of `GLM5NextKDASublayer`'s reference loop.
            func perToken(validity: MLXArray) throws -> (values: MLXArray, state: Primitive.State) {
                var current = state
                var rows: [MLXArray] = []
                for t in 0 ..< tokens {
                    let slice = t ..< t + 1
                    let result = try Primitive.apply(
                        query: query[0..., slice], key: key[0..., slice],
                        value: value[0..., slice], forgetLogits: forget[0..., slice],
                        betaLogits: beta[0..., slice], outputGate: gate[0..., slice],
                        weights: weights, state: current)
                    let active = validity[0..., t]
                    current = .init(
                        convolution: MLX.where(
                            active.reshaped(batch, 1, 1), result.state.convolution,
                            current.convolution),
                        recurrent: MLX.where(
                            active.reshaped(batch, 1, 1, 1), result.state.recurrent,
                            current.recurrent))
                    rows.append(MLX.where(active.reshaped(batch, 1, 1, 1), result.values, 0))
                    eval(current.convolution, current.recurrent, rows[rows.count - 1])
                }
                return (concatenated(rows, axis: 1), current)
            }
        }

        /// Row 0 all valid, row 1 left padded and interleaved, row 2 trailing
        /// padding, row 3 entirely padding.
        private static func validity(tokens: Int) -> MLXArray {
            let rows: [[Bool]] = [
                Array(repeating: true, count: tokens),
                (0 ..< tokens).map { $0 >= 3 && $0 % 5 != 2 },
                (0 ..< tokens).map { $0 < tokens - 4 },
                Array(repeating: false, count: tokens),
            ]
            return MLXArray(rows.flatMap { $0 }, [4, tokens])
        }

        /// Every row pattern of `validity(tokens:)` against the per-token loop.
        private func expectValiditySkipping(
            dtype: DType, executions: [Primitive.ExecutionPolicy], recurrentTolerance: Float
        ) throws {
            let tokens = 37
            let input = Projected(
                batch: 4, tokens: tokens, heads: 2, dimension: 8, dtype: dtype, seed: 11)
            let validity = Self.validity(tokens: tokens)
            let expected = try input.perToken(validity: validity)
            let tolerance: Float = dtype == .float32 ? recurrentTolerance : 3e-2
            for execution in executions {
                let actual = try input.whole(validity: validity, execution: execution)
                let label = "\(dtype) \(execution)"
                expectClose(actual.values, expected.values, tolerance: tolerance, "\(label) out")
                // The convolution history is a selection of input values.
                #expect(
                    Self.maxError(actual.state.convolution, expected.state.convolution) == 0,
                    "\(label) convolution history")
                #expect(actual.state.convolution.dtype == dtype)
                expectClose(
                    actual.state.recurrent, expected.state.recurrent,
                    tolerance: recurrentTolerance, "\(label) recurrent")
                // A padding-only row keeps its input state bit for bit and
                // returns zeros.
                func bits(_ array: MLXArray) -> [UInt32] {
                    array.asType(.float32).asArray(Float.self).map(\.bitPattern)
                }
                #expect(
                    bits(actual.state.recurrent[3]) == bits(input.state.recurrent[3]),
                    "\(label) padding-only recurrent")
                #expect(
                    bits(actual.state.convolution[3]) == bits(input.state.convolution[3]),
                    "\(label) padding-only convolution")
                #expect(Self.maxError(actual.values[3], MLXArray.zeros([1])) == 0)
            }
        }

        @Test(arguments: [DType.float32, .bfloat16])
        func wholeStepValiditySkipsPaddingLikeThePerTokenLoop(dtype: DType) throws {
            try Device.withDefaultDevice(.cpu) {
                try expectValiditySkipping(
                    dtype: dtype,
                    executions: [
                        .reference, .chunked(size: 1), .chunked(size: 8), .chunked(size: 32),
                    ], recurrentTolerance: 1e-4)
            }
        }

        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            /// The native kernel under the same row patterns: all valid,
            /// left/interleaved, trailing and padding-only.
            @Test(arguments: [DType.float32, .bfloat16])
            func nativeWholeStepValiditySkipsPaddingLikeThePerTokenLoopOnGPU(dtype: DType) throws {
                try Device.withDefaultDevice(.gpu) {
                    try expectValiditySkipping(
                        dtype: dtype, executions: [.nativeRequired, .chunked(size: 32)],
                        recurrentTolerance: 2e-4)
                }
            }
        #endif

        @Test func wholeStepContinuationMatchesOneStepAcrossChunkBoundaries() throws {
            try Device.withDefaultDevice(.cpu) {
                let tokens = 45
                let input = Projected(
                    batch: 4, tokens: tokens, heads: 2, dimension: 8, dtype: .float32, seed: 23)
                let validity = Self.validity(tokens: tokens)
                let whole = try input.whole(validity: validity, execution: .chunked(size: 16))
                for cuts in [[1, 45], [5, 40, 45], [16, 32, 45], [33, 45]] {
                    var state: Primitive.State?
                    var start = 0
                    var values: [MLXArray] = []
                    for end in cuts {
                        let part = try input.whole(
                            start ..< end, state: state, validity: validity,
                            execution: end - start >= 16 ? .chunked(size: 16) : .reference)
                        values.append(part.values)
                        state = part.state
                        start = end
                    }
                    let final = try #require(state)
                    expectClose(
                        concatenated(values, axis: 1), whole.values, tolerance: 1e-4, "\(cuts) out")
                    #expect(Self.maxError(final.convolution, whole.state.convolution) == 0)
                    expectClose(
                        final.recurrent, whole.state.recurrent, tolerance: 1e-4, "\(cuts) state")
                }
            }
        }

        @Test func wholeStepWithoutValidityIsTheExistingReferenceGraph() throws {
            try Device.withDefaultDevice(.cpu) {
                let input = Projected(
                    batch: 2, tokens: 9, heads: 2, dimension: 4, dtype: .float32, seed: 5)
                let unmasked = try input.whole(validity: nil, execution: .reference)
                let masked = try input.whole(
                    validity: MLXArray.ones([2, 9], dtype: .bool), execution: .reference)
                expectClose(masked.values, unmasked.values, tolerance: 1e-6, "out")
                #expect(Self.maxError(masked.state.convolution, unmasked.state.convolution) == 0)
                expectClose(masked.state.recurrent, unmasked.state.recurrent, tolerance: 1e-6, "s")
            }
        }

        @Test func refusesUnadmittedChunksAndValidityBeforeWork() throws {
            try Device.withDefaultDevice(.cpu) {
                let input = Projected(
                    batch: 2, tokens: 4, heads: 1, dimension: 2, dtype: .float32, seed: 3)
                #expect(throws: Primitive.Failure.unsupportedChunk) {
                    try input.whole(validity: nil, execution: .chunked(size: 64))
                }
                #expect(throws: Primitive.Failure.unsupportedChunk) {
                    try Primitive.apply(
                        query: input.query, key: input.key, value: input.value,
                        forgetLogits: input.forget, betaLogits: input.beta,
                        outputGate: input.gate, weights: input.weights, gateLowerBound: nil,
                        execution: .chunked(size: 4))
                }
                #expect(throws: Primitive.Failure.invalidTensor("validity")) {
                    try input.whole(
                        validity: MLXArray.ones([2, 4], dtype: .int32), execution: .reference)
                }
                #expect(throws: GLM5NextLinearKernel.Failure.metalUnavailable) {
                    try input.whole(validity: nil, execution: .nativeRequired)
                }
            }
        }

        // MARK: Model-level policy

        private static let native = GLM5NextExecutionPolicy(linearAttention: .nativeRecurrence)
        private static let chunked = GLM5NextExecutionPolicy(
            linearAttention: .nativeRecurrenceChunkedPrefill)

        private static func inventory() throws -> GLM5NextWeightInventory {
            let config = try GLM5NextInventoryFixture.configuration(layers: 4)
            let headers = try GLM5NextWeightInventory.expectedTensors(
                configuration: config, storage: .float32, shard: "tiny.safetensors")
            return try GLM5NextWeightInventory.validate(
                configuration: config, tensors: headers, storage: .float32)
        }

        private func makeModel(_ policy: GLM5NextExecutionPolicy = .reference) throws
            -> GLM5NextTextModel
        {
            let inventory = try Self.inventory()
            return try GLM5NextTextModel(
                configuration: inventory.configuration.textConfig, inventory: inventory,
                tensor: { name in
                    let shape = try #require(inventory.entries[name]).tensor.shape
                    let count = shape.reduce(1, *)
                    if name.hasSuffix("_scale") || name.contains("norm.weight") {
                        return MLXArray.ones(shape)
                    }
                    if name.hasSuffix("A_log") {
                        return MLXArray(Array(repeating: Float(-0.7), count: count), shape)
                    }
                    if name.hasSuffix(".bias") { return MLXArray.zeros(shape) }
                    let salt = name.utf8.reduce(0) { ($0 + Int($1)) % 13 }
                    return MLXArray(
                        (0 ..< count).map { Float(($0 + salt) % 11 - 5) * 0.025 }, shape)
                }, executionPolicy: policy)
        }

        private func fence(_ arrays: [MLXArray]) throws {
            try withError { error in
                eval(arrays)
                StreamOrDevice.default.stream.synchronize()
                try error.check()
            }
        }

        private static func tokens(_ count: Int, batch: Int = 1) -> MLXArray {
            MLXArray((0 ..< count * batch).map { Int32($0 * 7 % 29 + 1) }, [batch, count])
        }

        @Test func policyReachesEveryKDALayerAndRefusesCPUWithoutChangingTheCache() throws {
            try Device.withDefaultDevice(.cpu) {
                #expect(GLM5NextExecutionPolicy().linearAttention == .reference)
                for policy in [Self.native, Self.chunked] {
                    let model = try makeModel(policy)
                    let kda = model.model.layers.compactMap {
                        $0.attention as? GLM5NextKDASublayer
                    }
                    #expect(
                        !kda.isEmpty
                            && kda.allSatisfy { $0.linearAttention == policy.linearAttention })
                    for length in [1, 40] {
                        let cache = model.newTextCache()
                        let revision = cache.revision()
                        #expect(throws: GLM5NextLinearKernel.Failure.metalUnavailable) {
                            try model.prepareForward(tokens: Self.tokens(length), cache: cache)
                        }
                        #expect(cache.revision() == revision && cache.offset == 0)
                    }
                    // The reference policy's per-call seam cannot override it.
                    #expect(throws: GLM5NextLoadError.self) {
                        try model.forward(
                            tokens: Self.tokens(2), cache: nil, recurrenceExecution: .nativeRequired
                        )
                    }
                }
            }
        }

        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            /// GPU FP32 matmuls may run at TF32 on Neural-Accelerator GPUs, so
            /// the model-level tolerance matches the execution-policy tests.
            private func close(
                _ actual: MLXArray, _ expected: MLXArray, _ label: String,
                tolerance: Float = 2e-3
            ) {
                expectClose(actual, expected, tolerance: tolerance, label)
            }

            @Test func wholeStepPoliciesMatchReferenceForwardAndContinuationOnGPU() throws {
                try Device.withDefaultDevice(.gpu) {
                    let reference = try makeModel()
                    // 40 tokens exceed one 32-token chunk; row 1 is left padded.
                    let tokens = Self.tokens(40, batch: 2)
                    let validity = MLXArray((0 ..< 80).map { $0 < 40 || $0 >= 45 }, [2, 40])
                    let expected = try reference.forward(
                        tokens: tokens, cache: nil, validity: validity)
                    for (label, policy) in [("native", Self.native), ("chunked", Self.chunked)] {
                        let model = try makeModel(policy)
                        let output = try model.forward(
                            tokens: tokens, cache: nil, validity: validity)
                        try fence([expected.logits, output.logits, output.hidden])
                        close(output.logits, expected.logits, "\(label) logits")
                        close(output.hidden, expected.hidden, "\(label) hidden")

                        // Prefill, then decode and a two-token verify window.
                        let cache = model.newTextCache()
                        let referenceCache = reference.newTextCache()
                        for range in [0 ..< 33, 33 ..< 34, 34 ..< 36, 36 ..< 40] {
                            let piece = tokens[0..., range]
                            let mask = validity[0..., range]
                            let actual = try model.forward(
                                tokens: piece, cache: cache, validity: mask)
                            let wanted = try reference.forward(
                                tokens: piece, cache: referenceCache, validity: mask)
                            try fence([actual.hidden, wanted.hidden])
                            close(actual.hidden, wanted.hidden, "\(label) \(range)")
                        }
                        var firstKDA = true
                        for (index, layer) in cache.layers.enumerated() {
                            guard let state = (layer as? GLM5NextKDACache)?.current,
                                let wanted = (referenceCache.layers[index] as? GLM5NextKDACache)?
                                    .current
                            else { continue }
                            // Only the first KDA layer sees bitwise-equal inputs;
                            // later layers inherit the earlier layers' differences.
                            if firstKDA {
                                #expect(
                                    Self.maxError(state.convolution, wanted.convolution) == 0,
                                    "\(label) layer \(index) convolution")
                                firstKDA = false
                            } else {
                                close(
                                    state.convolution, wanted.convolution,
                                    "\(label) layer \(index) convolution")
                            }
                            close(state.recurrent, wanted.recurrent, "\(label) layer \(index)")
                        }
                    }
                }
            }
        #endif
    }

    /// The official cases have 17 and 19 tokens, below the 32-token chunk of
    /// `.nativeRecurrenceChunkedPrefill`, so the member suite below exercises
    /// only its kernel branch. This suite runs the chunk form itself against
    /// the oracle at smaller chunk sizes, through the reference policy's
    /// whole-step qualification seam. Operation-only: it runs on the CPU.
    @Suite(
        .serialized,
        .enabled(
            if: GLM5NextReferenceFixture.enabled,
            "Set GLM53_REFERENCE_FIXTURE_DIR to the prepared canonical official CPU oracle"))
    struct GLM5NextChunkedLinearAttentionOfficialReferenceTests {
        @Test(arguments: [4, 8, 16])
        func chunkFormMatchesOfficialEveryLayer(chunkSize: Int) throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextReferenceFixture()
                let model = try fixture.textModel()
                for caseName in ["text_full", "text_left_padding"] {
                    let test = try fixture.caseObject(caseName)
                    let tokens = try fixture.tensor(fixture.key(test, "input_ids"))
                    #expect(tokens.dim(1) > chunkSize, "\(caseName) spans several chunks")
                    let padded = caseName == "text_left_padding"
                    let validity =
                        padded
                        ? try fixture.tensor(fixture.key(test, "attention_mask")).asType(.bool)
                        : MLXArray.ones(tokens.shape, dtype: .bool)
                    let mask = padded ? validity : nil
                    var streams = broadcast(
                        model.embed(tokens).expandedDimensions(axis: 2),
                        to: [
                            tokens.dim(0), tokens.dim(1), model.configuration.hcMult,
                            model.configuration.hiddenSize,
                        ])
                    let cache = model.newTextCache()
                    let layerKeys = try #require(test["layer_outputs"] as? [String])
                    guard layerKeys.count == model.model.layers.count else {
                        throw GLM5NextReferenceFixture.Failure("target layer count differs")
                    }
                    for (index, layer) in model.model.layers.enumerated() {
                        if let kda = layer.attention as? GLM5NextKDASublayer {
                            streams = try layer.forward(streams) { normalized in
                                try kda.forward(
                                    normalized, state: nil, validity: validity,
                                    wholeStepQualification: .chunked(size: chunkSize)
                                ).values
                            }
                        } else {
                            streams = try layer.forward(
                                streams, cache: cache.layers[index], validity: validity)
                        }
                        eval(streams)
                        try fixture.compare(
                            streams, key: layerKeys[index], mask: mask)
                    }
                }
            }
        }
    }

    /// Official-oracle parity of both whole-step KDA members: full target,
    /// every layer, left padding, and token and chunk continuation.
    @Suite(
        .serialized,
        .enabled(
            if: GLM5NextReferenceFixture.enabled,
            "Set GLM53_REFERENCE_FIXTURE_DIR to the prepared canonical official CPU oracle"))
    struct GLM5NextWholeStepLinearAttentionOfficialReferenceTests {
        private typealias Checks = GLM5NextNativeKernelReferenceChecks

        @Test(arguments: [
            GLM5NextExecutionPolicy.LinearAttention.nativeRecurrence,
            .nativeRecurrenceChunkedPrefill,
        ])
        func fullTargetEachLayerAndContinuationMatchOfficialOnGPU(
            member: GLM5NextExecutionPolicy.LinearAttention
        ) throws {
            try Device.withDefaultDevice(.gpu) {
                let fixture = try GLM5NextReferenceFixture()
                let model = try fixture.textModel(
                    executionPolicy: .init(linearAttention: member))
                for caseName in ["text_full", "text_left_padding"] {
                    let test = try fixture.caseObject(caseName)
                    let tokens = try fixture.tensor(fixture.key(test, "input_ids"))
                    let padded = caseName == "text_left_padding"
                    let validity =
                        padded
                        ? try fixture.tensor(fixture.key(test, "attention_mask")).asType(.bool)
                        : MLXArray.ones(tokens.shape, dtype: .bool)
                    let mask = padded ? validity : nil
                    let output = try model.forward(tokens: tokens, cache: nil, validity: validity)
                    try Checks.fence([output.logits, output.hidden])
                    try fixture.compare(output.logits, key: fixture.key(test, "logits"), mask: mask)
                    try fixture.compare(
                        output.hidden, key: fixture.key(test, "final_hidden"), mask: mask)

                    var streams = broadcast(
                        model.embed(tokens).expandedDimensions(axis: 2),
                        to: [
                            tokens.dim(0), tokens.dim(1), model.configuration.hcMult,
                            model.configuration.hiddenSize,
                        ])
                    let cache = model.newTextCache()
                    let layerKeys = try #require(test["layer_outputs"] as? [String])
                    guard layerKeys.count == model.model.layers.count else {
                        throw GLM5NextReferenceFixture.Failure("target layer count differs")
                    }
                    for (index, layer) in model.model.layers.enumerated() {
                        streams = try layer.forward(
                            streams, cache: cache.layers[index], validity: validity)
                        try Checks.fence([streams])
                        try fixture.compare(streams, key: layerKeys[index], mask: mask)
                    }
                }

                let full = try fixture.caseObject("text_full")
                let tokens = try fixture.tensor(fixture.key(full, "input_ids"))
                let steps = try #require(
                    try fixture.caseObject("text_tokenwise")["steps"] as? [[String: Any]])
                guard steps.count == tokens.dim(1) else {
                    throw GLM5NextReferenceFixture.Failure("token step count differs from oracle")
                }
                let tokenCache = model.newTextCache()
                for (index, step) in steps.enumerated() {
                    let output = try model.forward(
                        tokens: tokens[0..., index ..< index + 1], cache: tokenCache)
                    try Checks.fence([output.logits, output.hidden])
                    try fixture.compare(output.logits, key: fixture.key(step, "logits"))
                    try fixture.compare(output.hidden, key: fixture.key(step, "final_hidden"))
                }
                let chunkCache = model.newTextCache()
                var start = 0
                var pieces: [MLXArray] = []
                for end in Set([3, 4, 5, 11, tokens.dim(1)].filter { $0 <= tokens.dim(1) })
                    .sorted()
                {
                    let output = try model.forward(
                        tokens: tokens[0..., start ..< end], cache: chunkCache)
                    try Checks.fence([output.logits])
                    pieces.append(output.logits)
                    start = end
                }
                try fixture.compare(concatenated(pieces, axis: 1), key: fixture.key(full, "logits"))
            }
        }
    }
}
