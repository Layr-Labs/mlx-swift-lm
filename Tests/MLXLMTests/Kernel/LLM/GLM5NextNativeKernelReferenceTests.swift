// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

private typealias Recurrence = GLM5NextLinearAttention
private typealias NativeKernel = GLM5NextLinearKernel

/// The frozen synthetic oracle exercises the actual model and layer forwards.
/// It is numerical qualification, not a full-artifact throughput measurement.
enum GLM5NextNativeKernelReferenceChecks {
    static func fullTargetAndEachLayer() throws {
        try Device.withDefaultDevice(.gpu) {
            let fixture = try GLM5NextReferenceFixture()
            let model = try fixture.textModel()
            for caseName in ["text_full", "text_left_padding"] {
                let test = try fixture.caseObject(caseName)
                let tokens = try fixture.tensor(fixture.key(test, "input_ids"))
                let validity: MLXArray
                if caseName == "text_left_padding" {
                    validity = try fixture.tensor(fixture.key(test, "attention_mask")).asType(.bool)
                } else {
                    validity = MLXArray.ones(tokens.shape, dtype: .bool)
                }
                let mask = caseName == "text_left_padding" ? validity : nil
                let reference = try model.forward(
                    tokens: tokens, cache: nil, validity: validity, recurrenceExecution: .reference)
                let native = try model.forward(
                    tokens: tokens, cache: nil, validity: validity,
                    recurrenceExecution: .nativeRequired)
                try fence([reference.logits, reference.hidden, native.logits, native.hidden])
                for (label, output) in [("reference", reference), ("native", native)] {
                    try fixture.compare(output.logits, key: fixture.key(test, "logits"), mask: mask)
                    try fixture.compare(
                        output.hidden, key: fixture.key(test, "final_hidden"), mask: mask)
                    try fixture.compare(
                        output.logits, expected: reference.logits,
                        label: "\(caseName) \(label) vs reference logits", mask: mask)
                }

                let initial = broadcast(
                    model.embed(tokens).expandedDimensions(axis: 2),
                    to: [
                        tokens.dim(0), tokens.dim(1), model.configuration.hcMult,
                        model.configuration.hiddenSize,
                    ])
                var referenceStreams = initial
                var nativeStreams = initial
                let referenceCache = model.newTextCache()
                let nativeCache = model.newTextCache()
                let layerKeys = try #require(test["layer_outputs"] as? [String])
                guard layerKeys.count == model.model.layers.count else {
                    throw GLM5NextReferenceFixture.Failure("target layer count differs from oracle")
                }
                for (index, layer) in model.model.layers.enumerated() {
                    referenceStreams = try layer.forward(
                        referenceStreams, cache: referenceCache.layers[index], validity: validity,
                        recurrenceExecution: .reference)
                    nativeStreams = try layer.forward(
                        nativeStreams, cache: nativeCache.layers[index], validity: validity,
                        recurrenceExecution: .nativeRequired)
                    try fence([referenceStreams, nativeStreams])
                    try fixture.compare(referenceStreams, key: layerKeys[index], mask: mask)
                    try fixture.compare(nativeStreams, key: layerKeys[index], mask: mask)
                    try fixture.compare(
                        nativeStreams, expected: referenceStreams,
                        label: "\(caseName) layer \(index) native vs reference", mask: mask)
                }
                try compareKDACaches(
                    nativeCache, reference: referenceCache, fixture: fixture, label: caseName)
                // The frozen left-padding case has layer/logit tensors but no
                // cache snapshot. Do not invent an official cache expectation.
                if caseName == "text_full" {
                    try compareKDAStates(nativeCache, description: test, fixture: fixture)
                }
            }
        }
    }

    static func tokenContinuation() throws {
        try Device.withDefaultDevice(.gpu) {
            let fixture = try GLM5NextReferenceFixture()
            let model = try fixture.textModel()
            let full = try fixture.caseObject("text_full")
            let tokens = try fixture.tensor(fixture.key(full, "input_ids"))
            let tokenwise = try fixture.caseObject("text_tokenwise")
            let steps = try #require(tokenwise["steps"] as? [[String: Any]])
            guard steps.count == tokens.dim(1) else {
                throw GLM5NextReferenceFixture.Failure("token step count differs from oracle")
            }
            let referenceCache = model.newTextCache()
            let nativeCache = model.newTextCache()
            for (index, step) in steps.enumerated() {
                let token = tokens[0..., index ..< index + 1]
                let reference = try model.forward(
                    tokens: token, cache: referenceCache, recurrenceExecution: .reference)
                let native = try model.forward(
                    tokens: token, cache: nativeCache, recurrenceExecution: .nativeRequired)
                try fence(
                    [reference.logits, reference.hidden, native.logits, native.hidden]
                        + nativeCache.layers.flatMap(\.state))
                try fixture.compare(native.logits, key: fixture.key(step, "logits"))
                try fixture.compare(native.hidden, key: fixture.key(step, "final_hidden"))
                try fixture.compare(
                    native.logits, expected: reference.logits, label: "token \(index) logits")
                try compareKDAStates(nativeCache, description: step, fixture: fixture)
            }
        }
    }

    static func compareKDAStates(
        _ cache: GLM5NextTextCache, description: [String: Any],
        fixture: GLM5NextReferenceFixture
    ) throws {
        let expected = try #require(description["cache"] as? [String: Any])
        let convolution = try #require(expected["conv_states"] as? [String])
        let recurrent = try #require(expected["kda_states"] as? [String])
        let layers = fixture.configuration.textConfig.linearAttention.kdaLayers
        guard convolution.count == layers.count, recurrent.count == layers.count else {
            throw GLM5NextReferenceFixture.Failure("KDA state count differs from oracle")
        }
        for (slot, index) in layers.enumerated() {
            let state = cache.layers[index].state
            guard state.count == 2 else {
                throw GLM5NextReferenceFixture.Failure("missing KDA state at layer \(index)")
            }
            try fixture.compare(state[0], key: convolution[slot])
            try fixture.compare(state[1], key: recurrent[slot])
        }
    }

    static func compareKDACaches(
        _ actual: GLM5NextTextCache, reference: GLM5NextTextCache,
        fixture: GLM5NextReferenceFixture, label: String
    ) throws {
        for index in fixture.configuration.textConfig.linearAttention.kdaLayers {
            let state = actual.layers[index].state
            let expected = reference.layers[index].state
            guard state.count == 2, expected.count == 2 else {
                throw GLM5NextReferenceFixture.Failure("missing KDA state at layer \(index)")
            }
            for slot in 0 ..< 2 {
                try fixture.compare(
                    state[slot], expected: expected[slot],
                    label: "\(label) layer \(index) state \(slot)")
            }
        }
    }

    static func fence(_ arrays: [MLXArray]) throws {
        try withError { error in
            eval(arrays)
            StreamOrDevice.default.stream.synchronize()
            try error.check()
        }
    }
}

extension KernelTests {
    @Suite(
        .serialized,
        .enabled(
            if: GLM5NextReferenceFixture.enabled,
            "Set GLM53_REFERENCE_FIXTURE_DIR to the prepared canonical official CPU oracle"))
    struct GLM5NextNativeKernelReferenceTests {
        @Test func fullTargetAndEachLayerMatchOfficialOnGPU() throws {
            try GLM5NextNativeKernelReferenceChecks.fullTargetAndEachLayer()
        }

        @Test func tokenContinuationMatchesOfficialOnGPU() throws {
            try GLM5NextNativeKernelReferenceChecks.tokenContinuation()
        }

        @Test func chunkContinuationMatchesOfficialOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                let fixture = try GLM5NextReferenceFixture()
                let model = try fixture.textModel()
                let full = try fixture.caseObject("text_full")
                let tokens = try fixture.tensor(fixture.key(full, "input_ids"))
                let officialLogits = try fixture.tensor(fixture.key(full, "logits"))
                let referenceCache = model.newTextCache()
                let nativeCache = model.newTextCache()
                var start = 0
                var pieces: [MLXArray] = []
                let ends = Set([3, 4, 5, 11, tokens.dim(1)].filter { $0 <= tokens.dim(1) }).sorted()
                for end in ends {
                    let range = start ..< end
                    let reference = try model.forward(
                        tokens: tokens[0..., range], cache: referenceCache,
                        recurrenceExecution: .reference)
                    let native = try model.forward(
                        tokens: tokens[0..., range], cache: nativeCache,
                        recurrenceExecution: .nativeRequired)
                    try GLM5NextNativeKernelReferenceChecks.fence(
                        [reference.logits, reference.hidden, native.logits, native.hidden]
                            + nativeCache.layers.flatMap(\.state))
                    try fixture.compare(
                        native.logits, expected: officialLogits[0..., range],
                        label: "chunk \(range) official logits")
                    try fixture.compare(
                        native.logits, expected: reference.logits,
                        label: "chunk \(range) native vs reference logits")
                    try fixture.compare(
                        native.hidden, expected: reference.hidden,
                        label: "chunk \(range) native vs reference hidden")
                    try GLM5NextNativeKernelReferenceChecks.compareKDACaches(
                        nativeCache, reference: referenceCache, fixture: fixture,
                        label: "chunk \(range)")
                    pieces.append(native.logits)
                    start = end
                }
                try fixture.compare(concatenated(pieces, axis: 1), key: fixture.key(full, "logits"))
                try GLM5NextNativeKernelReferenceChecks.compareKDAStates(
                    nativeCache, description: full, fixture: fixture)
            }
        }

        @Test func preparedTokenAndEmbeddingForwardsPreserveCommitBoundaryOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                let fixture = try GLM5NextReferenceFixture()
                let model = try fixture.textModel()
                let full = try fixture.caseObject("text_full")
                let tokens = try fixture.tensor(fixture.key(full, "input_ids"))
                guard tokens.dim(1) > 1 else {
                    throw GLM5NextReferenceFixture.Failure("continuation fixture needs two tokens")
                }
                let split = min(3, tokens.dim(1) - 1)
                let referenceCache = model.newTextCache()
                let nativeCache = model.newTextCache()
                var pieces: [MLXArray] = []
                for (part, range) in [0 ..< split, split ..< tokens.dim(1)].enumerated() {
                    let reference = try model.forward(
                        tokens: tokens[0..., range], cache: referenceCache,
                        recurrenceExecution: .reference)
                    let revision = nativeCache.revision()
                    let candidate: GLM5NextTextCandidate
                    if part == 0 {
                        candidate = try model.prepareForward(
                            tokens: tokens[0..., range], cache: nativeCache,
                            recurrenceExecution: .nativeRequired)
                    } else {
                        let embeddings = model.embed(tokens[0..., range])
                        let discarded = try model.prepareForward(
                            embeddings: embeddings, cache: nativeCache,
                            recurrenceExecution: .nativeRequired)
                        discarded.discard()
                        #expect(nativeCache.revision() == revision)
                        candidate = try model.prepareForward(
                            embeddings: embeddings, cache: nativeCache,
                            recurrenceExecution: .nativeRequired)
                    }
                    #expect(nativeCache.revision() == revision)
                    let native = try candidate.commit(
                        afterEvaluating: GLM5NextNativeKernelReferenceChecks.fence)
                    #expect(nativeCache.offset == range.upperBound)
                    try fixture.compare(
                        native.logits, expected: reference.logits,
                        label: "prepared \(range) native vs reference logits")
                    try fixture.compare(
                        native.hidden, expected: reference.hidden,
                        label: "prepared \(range) native vs reference hidden")
                    pieces.append(native.logits)
                }
                try fixture.compare(concatenated(pieces, axis: 1), key: fixture.key(full, "logits"))
                try GLM5NextNativeKernelReferenceChecks.compareKDAStates(
                    nativeCache, description: full, fixture: fixture)
            }
        }

        @Test func forcedModelPolicyReachesKernelAndRejectsCPU() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextReferenceFixture()
                let model = try fixture.textModel()
                let full = try fixture.caseObject("text_full")
                let tokens = try fixture.tensor(fixture.key(full, "input_ids"))[0..., 0 ..< 1]
                let cache = model.newTextCache()
                let revision = cache.revision()
                let reference = try model.forward(
                    tokens: tokens, cache: nil, recurrenceExecution: .reference)
                let defaultOutput = try model.forward(tokens: tokens, cache: nil)
                try fixture.compare(
                    defaultOutput.logits, expected: reference.logits,
                    label: "public forward defaults to reference")
                let defaultCandidate = try model.prepareForward(
                    tokens: tokens, cache: model.newTextCache())
                let prepared = try defaultCandidate.commit(
                    afterEvaluating: GLM5NextNativeKernelReferenceChecks.fence)
                try fixture.compare(
                    prepared.logits, expected: reference.logits,
                    label: "public prepareForward defaults to reference")
                #expect(throws: NativeKernel.Failure.metalUnavailable) {
                    try model.forward(
                        tokens: tokens, cache: cache, recurrenceExecution: .nativeRequired)
                }
                #expect(throws: NativeKernel.Failure.metalUnavailable) {
                    try model.prepareForward(
                        tokens: tokens, cache: cache, recurrenceExecution: .nativeRequired)
                }
                #expect(throws: NativeKernel.Failure.metalUnavailable) {
                    try model.prepareForward(
                        embeddings: model.embed(tokens), cache: cache,
                        recurrenceExecution: .nativeRequired)
                }
                #expect(cache.revision() == revision)
            }
        }
    }

    @Suite(.serialized)
    struct GLM5NextNativeKernelPolicyTests {
        @Test func referenceRemainsDefaultAndForcedNativePreservesCallerMath() throws {
            try Device.withDefaultDevice(.gpu) {
                for dtype: DType in [.float32, .float16, .bfloat16] {
                    let input = LinearPolicyFixture(dtype: dtype)
                    let original = input.state.recurrent.asArray(Float.self)
                    let implicit = try input.run()
                    let reference = try input.run(execution: .reference)
                    let native = try input.run(execution: .nativeRequired)
                    try GLM5NextNativeKernelReferenceChecks.fence([
                        implicit.values, reference.values, native.values, native.state.recurrent,
                    ])
                    #expect(implicit.execution == .reference)
                    #expect(reference.execution == .reference)
                    #expect(native.execution == .nativeRequired)
                    #expect(
                        implicit.values.asArray(Float.self) == reference.values.asArray(Float.self))
                    try compare(
                        native.values, reference.values, tolerance: dtype == .float32 ? 3e-5 : 0.015
                    )
                    try compare(native.state.recurrent, reference.state.recurrent, tolerance: 2e-5)
                    #expect(
                        native.state.convolution.asArray(Float.self)
                            == reference.state.convolution.asArray(Float.self))
                    #expect(input.state.recurrent.asArray(Float.self) == original)
                    #expect(
                        native.values.dtype == dtype && native.state.recurrent.dtype == .float32)
                }
            }
        }

        @Test func forcedNativeRejectsCPUWithoutFallingBack() throws {
            try Device.withDefaultDevice(.cpu) {
                let input = LinearPolicyFixture(dtype: .float32)
                let result = try input.run()
                #expect(result.execution == .reference)
                #expect(throws: NativeKernel.Failure.metalUnavailable) {
                    try input.run(execution: .nativeRequired)
                }
            }
        }

        @Test func forcedNativeRejectsUnsupportedGeometry() throws {
            let input = LinearPolicyFixture(dtype: .float32, dimension: 129)
            #expect(throws: NativeKernel.Failure.invalidGeometry("K and V must be in 1...128")) {
                try input.run(execution: .nativeRequired)
            }
        }

        @Test func admittedCaptureConsumesRealLeaseAndRetainsItThroughRetirement() throws {
            try Device.withDefaultDevice(.gpu) {
                let q = MLXArray.ones([1, 2, 1, 3], dtype: .float32) * 0.1
                let v = MLXArray.ones([1, 2, 1, 5], dtype: .float32)
                let initial = MLXArray.ones([1, 1, 3, 5], dtype: .float32) * 0.2
                let layout = try NativeKernel.Layout(
                    queryShape: q.shape, valueShape: v.shape, capturePrefixCount: 2)
                #expect(layout.outputAllocation.map(\.logicalBytes) == [40, 60, 120])
                let allocator = try #require(Memory.allocationFootprintPolicy())
                let limits = CBv2MTPAllocationLimits(maximumPrefillTokens: 2, maximumDraftTokens: 2)
                let resolution = CBv2MTPBoundedAdmission.resolve(
                    spec: .init(
                        resident: layout.outputAllocation,
                        working: [.init(logicalBytes: 4)], hostBytes: 0),
                    limits: limits, policy: allocator)
                guard case .bounded(let resolved) = resolution else {
                    throw GLM5NextReferenceFixture.Failure("capture declaration failed to resolve")
                }
                let process = NativePageTestProcessOwner(maximum: 64 << 20)
                let ledger = AdmissionV2(
                    layerKinds: [], bytesCapacity: 64 << 20,
                    config: .init(
                        watermarkFraction: 0, fixedBytesPerRequest: resolved.fixedBytesPerRequest),
                    processMemoryOwner: process)
                let id = CBv2RequestID(91)
                try ledger.reserve(id: id, additionalTokens: 1)
                let target = NSObject()
                let drafter = NSObject()
                let unowned = AdmissionV2(
                    layerKinds: [], bytesCapacity: 64 << 20,
                    config: .init(
                        watermarkFraction: 0, fixedBytesPerRequest: resolved.fixedBytesPerRequest))
                try unowned.reserve(id: id, additionalTokens: 1)
                #expect(throws: CBv2MTPBindingError.missingAdmission) {
                    try unowned.issueMTPBinding(
                        id: id, maximumInputs: 2, target: ObjectIdentifier(target),
                        drafter: ObjectIdentifier(drafter), resolved: resolved, policy: allocator)
                }
                unowned.releaseAll(id: id)
                let admission = try ledger.issueMTPBinding(
                    id: id, maximumInputs: 2, target: ObjectIdentifier(target),
                    drafter: ObjectIdentifier(drafter), resolved: resolved, policy: allocator)
                #expect(
                    throws: NativeKernel.Failure.invalidCapture(
                        "admitted capture requires a nonempty prefix")
                ) {
                    try NativeKernel.applyAdmittedCapture(
                        query: q, key: q, value: v, logDecay: -q,
                        beta: MLXArray.ones([1, 2, 1], dtype: .float32), state: initial,
                        capturePrefixCount: 0, admission: admission)
                }
                #expect(admission.allocatedBytes == 0)
                func capture() throws -> NativeKernel.AdmittedCapture {
                    try NativeKernel.applyAdmittedCapture(
                        query: q, key: q, value: v, logDecay: -q,
                        beta: MLXArray.ones([1, 2, 1], dtype: .float32), state: initial,
                        capturePrefixCount: 2, admission: admission)
                }
                var output: NativeKernel.AdmittedCapture? = try capture()
                #expect(admission.allocatedBytes == resolved.residentBytes)
                #expect(throws: CBv2MTPBindingError.capacityExceeded) { try capture() }
                #expect(admission.allocatedBytes == resolved.residentBytes)
                try output?.withEvaluationLease { result in
                    try GLM5NextNativeKernelReferenceChecks.fence([
                        result.values, result.state, try #require(result.capturedPrefixStates),
                    ])
                    #expect(result.state.dtype == .float32)
                }
                ledger.releaseAll(id: id)
                #expect(ledger.bytesReserved == resolved.fixedBytesPerRequest)
                #expect(process.bytes == UInt64(resolved.fixedBytesPerRequest))
                #expect(throws: CBv2MTPBindingError.staleRequest) { try capture() }
                output = nil
                #expect(admission.allocatedBytes == 0)
                #expect(ledger.bytesReserved == 0)
                #expect(process.bytes == 0)
            }
        }

        private func compare(_ a: MLXArray, _ b: MLXArray, tolerance: Float) throws {
            let left = a.asType(.float32).asArray(Float.self)
            let right = b.asType(.float32).asArray(Float.self)
            #expect(a.shape == b.shape)
            #expect(left.allSatisfy(\.isFinite) && right.allSatisfy(\.isFinite))
            #expect(zip(left, right).allSatisfy { abs($0.0 - $0.1) <= tolerance })
        }
    }

    /// This isolated stage benchmark cannot establish the full-model acceptance
    /// floors. Both gates are required so an ordinary test pass performs no timing.
    @Suite(
        .serialized,
        .enabled(
            if:
                ProcessInfo.processInfo.environment["GLM53_RUN_LINEAR_BENCHMARK"] == "1"
                && ProcessInfo.processInfo.environment["GLM53_GPU_LANE_LOCKED"] == "1"))
    struct GLM5NextLinearKernelBenchmark {
        @Test func referenceVsNativeH64() throws {
            try Device.withDefaultDevice(.gpu) {
                for tokens in [1, 128, 512] {
                    let shape = [1, tokens, 64, 128]
                    let ramp = MLXArray(0 ..< shape.reduce(1, *)).asType(.float32).reshaped(shape)
                    let q0 = sin(ramp * 0.013)
                    let k0 = cos(ramp * 0.017)
                    let q =
                        q0 / sqrt(q0.square().sum(axis: -1, keepDims: true) + 1e-6)
                        * (1 / sqrt(Float(128)))
                    let k = k0 / sqrt(k0.square().sum(axis: -1, keepDims: true) + 1e-6)
                    let v = sin(ramp * 0.019)
                    let decay = -abs(cos(ramp * 0.011)) * 0.1
                    let beta = MLXArray.ones([1, tokens, 64], dtype: .float32) * 0.6
                    let state =
                        cos(MLXArray(0 ..< 64 * 128 * 128).asType(.float32) * 0.007)
                        .reshaped(1, 64, 128, 128) * 0.1
                    try GLM5NextNativeKernelReferenceChecks.fence([q, k, v, decay, beta, state])
                    func run(_ native: Bool) throws -> [MLXArray] {
                        if native {
                            let result = try NativeKernel.apply(
                                query: q, key: k, value: v, logDecay: decay, beta: beta,
                                state: state)
                            return [result.values, result.state]
                        }
                        var current = state
                        var values: [MLXArray] = []
                        for t in 0 ..< tokens {
                            let kt = k[0..., t]
                            current = current * exp(decay[0..., t])[.ellipsis, .newAxis]
                            let memory = (current * kt[.ellipsis, .newAxis]).sum(axis: -2)
                            let delta = (v[0..., t] - memory) * beta[0..., t][.ellipsis, .newAxis]
                            current =
                                current + kt[.ellipsis, .newAxis]
                                * delta[0..., 0..., .newAxis, 0...]
                            values.append((current * q[0..., t][.ellipsis, .newAxis]).sum(axis: -2))
                        }
                        return [stacked(values, axis: 1), current]
                    }
                    do {
                        let reference = try run(false)
                        let native = try run(true)
                        let errors = zip(reference, native).map { abs($0.0 - $0.1).max() }
                        try GLM5NextNativeKernelReferenceChecks.fence(reference + native + errors)
                        guard errors.allSatisfy({ $0.item(Float.self) <= 2e-5 }) else {
                            throw GLM5NextReferenceFixture.Failure("benchmark parity failed")
                        }
                    }
                    // Compile and populate caches independently of timed samples.
                    for _ in 0 ..< 3 {
                        for native in [false, true] {
                            try GLM5NextNativeKernelReferenceChecks.fence(run(native))
                        }
                    }
                    let clock = ContinuousClock()
                    for sample in 0 ..< 12 {
                        for native in (sample.isMultiple(of: 2) ? [false, true] : [true, false]) {
                            Stream.gpu.synchronize()
                            Memory.peakMemory = 0
                            let active = Memory.activeMemory
                            let start = clock.now
                            try GLM5NextNativeKernelReferenceChecks.fence(run(native))
                            let elapsed = start.duration(to: clock.now).components
                            let milliseconds =
                                Double(elapsed.seconds) * 1_000
                                + Double(elapsed.attoseconds) / 1e15
                            let record: [String: Any] = [
                                "scope": "recurrence-only-not-full-model-throughput",
                                "backend": native ? "native" : "reference", "sample": sample,
                                "batch": 1, "heads": 64, "key": 128, "value": 128, "tokens": tokens,
                                "milliseconds": milliseconds, "activeBeforeBytes": active,
                                "peakBytes": Memory.peakMemory,
                                "activeAfterBytes": Memory.activeMemory,
                            ]
                            print(
                                String(
                                    decoding: try JSONSerialization.data(
                                        withJSONObject: record, options: [.sortedKeys]),
                                    as: UTF8.self))
                        }
                    }
                }
            }
        }
    }
}

private struct LinearPolicyFixture {
    let query: MLXArray
    let key: MLXArray
    let value: MLXArray
    let forget: MLXArray
    let beta: MLXArray
    let gate: MLXArray
    let weights: Recurrence.Weights
    let state: Recurrence.State

    init(dtype: DType, dimension: Int = 33) {
        let shape = [2, 5, 2, dimension]
        let width = 2 * dimension
        func tensor(_ dimensions: [Int], salt: Int) -> MLXArray {
            MLXArray(
                (0 ..< dimensions.reduce(1, *)).map {
                    Float(($0 * 17 + salt) % 71 - 35) / 71
                }, dimensions)
        }
        query = tensor(shape, salt: 1).asType(dtype)
        key = tensor(shape, salt: 2).asType(dtype)
        value = tensor(shape, salt: 3).asType(dtype)
        forget = tensor(shape, salt: 4).asType(dtype)
        beta = tensor([2, 5, 2], salt: 5).asType(dtype)
        gate = tensor(shape, salt: 6).asType(dtype)
        weights = .init(
            convolution: tensor([3 * width, 4], salt: 7), aLog: tensor([2], salt: 8),
            dtBias: tensor([width], salt: 9), outputNorm: tensor([dimension], salt: 10))
        state = .init(
            convolution: tensor([2, 3 * width, 4], salt: 11).asType(dtype),
            recurrent: tensor([2, 2, dimension, dimension], salt: 12))
    }

    func run(execution: Recurrence.ExecutionPolicy? = nil) throws -> Recurrence.Output {
        if let execution {
            return try Recurrence.apply(
                query: query, key: key, value: value, forgetLogits: forget, betaLogits: beta,
                outputGate: gate, weights: weights, state: state, execution: execution)
        }
        return try Recurrence.apply(
            query: query, key: key, value: value, forgetLogits: forget, betaLogits: beta,
            outputGate: gate, weights: weights, state: state)
    }
}
