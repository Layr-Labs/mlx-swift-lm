// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

extension KernelTests {
    /// Model-level wiring of the construction-time `GLM5NextExecutionPolicy` on
    /// the deterministic tiny target. Real-checkpoint and official-oracle
    /// qualification remain separate.
    @Suite(.serialized)
    struct GLM5NextExecutionPolicyTests {
        private static let hyperConnection = GLM5NextExecutionPolicy(
            hyperConnection: .nativeRequired)
        private static let routedExperts = GLM5NextExecutionPolicy(routedExperts: .nativeCombine)
        private static let allNative = GLM5NextExecutionPolicy(
            hyperConnection: .nativeRequired, routedExperts: .nativeCombineSortedPrefill)
        private static let everyOptIn = GLM5NextExecutionPolicy(
            sparseAttention: .absorbed,
            sparseSelection: .init(selection: .parallelThreshold, deviceRanges: true),
            pagedMetadataReads: .deviceTable, hyperConnection: .nativeRequired,
            routedExperts: .nativeCombineSortedPrefill,
            linearAttention: .nativeRecurrenceChunkedPrefill)

        private static let normFused = GLM5NextExecutionPolicy(hyperConnection: .nativeNormFused)
        private static let fusedDecode = GLM5NextExecutionPolicy(
            hyperConnection: .nativeNormFused, routedExperts: .fusedDecode,
            linearAttention: .nativeRecurrence, normalization: .fusedKernel,
            linearDecodeStep: .fusedKernel, routing: .nativeKernel, projections: .fusedInputs)

        private static func inventory() throws -> GLM5NextWeightInventory {
            let config = try GLM5NextInventoryFixture.configuration(layers: 4)
            let headers = try GLM5NextWeightInventory.expectedTensors(
                configuration: config, storage: .float32, shard: "tiny.safetensors")
            return try GLM5NextWeightInventory.validate(
                configuration: config, tensors: headers, storage: .float32)
        }

        private static func tensor(_ inventory: GLM5NextWeightInventory, _ name: String) throws
            -> MLXArray
        {
            let entry = try #require(inventory.entries[name])
            let shape = entry.tensor.shape
            let count = shape.reduce(1, *)
            let values: [Float]
            if name.hasSuffix("_scale") || name.contains("norm.weight") {
                values = Array(repeating: 1, count: count)
            } else if name.hasSuffix("A_log") {
                values = Array(repeating: -0.7, count: count)
            } else if name.hasSuffix("dt_bias") {
                values = Array(repeating: -0.2, count: count)
            } else if name.hasSuffix(".bias") {
                values = Array(repeating: 0, count: count)
            } else {
                let salt = name.utf8.reduce(0) { ($0 + Int($1)) % 13 }
                values = (0 ..< count).map { Float(($0 + salt) % 11 - 5) * 0.025 }
            }
            return MLXArray(values, shape)
        }

        private func makeModel(_ policy: GLM5NextExecutionPolicy = .reference) throws
            -> GLM5NextTextModel
        {
            let inventory = try Self.inventory()
            return try GLM5NextTextModel(
                configuration: inventory.configuration.textConfig, inventory: inventory,
                tensor: { try Self.tensor(inventory, $0) }, executionPolicy: policy)
        }

        private func makeMTP(_ target: GLM5NextTextModel) throws -> GLM5NextMTP {
            try GLM5NextMTP(target: target, tensor: { try Self.tensor(target.inventory, $0) })
        }

        private func fence(_ arrays: [MLXArray]) throws {
            try withError { error in
                eval(arrays)
                StreamOrDevice.default.stream.synchronize()
                try error.check()
            }
        }

        private static func bits(_ array: MLXArray) -> [UInt32] {
            array.asType(.float32).asArray(Float.self).map(\.bitPattern)
        }

        /// GPU FP32 matmuls with M>1 may run at TF32 precision (MLX_ENABLE_TF32
        /// defaults on), so prefill and token-wise reference graphs differ too.
        private func close(
            _ actual: MLXArray, _ expected: MLXArray, tolerance: Float = 2e-3, _ label: String
        ) {
            #expect(actual.shape == expected.shape, "\(label) shape")
            let error = abs(actual.asType(.float32) - expected.asType(.float32)).max()
                .item(Float.self)
            #expect(error.isFinite && error < tolerance, "\(label) max error \(error)")
        }

        private let tokens = MLXArray([Int32(2), 5, 3, 9, 7, 1, 4], [1, 7])

        /// The default is the reference everywhere, and a configured policy is
        /// held unchanged by every layer, sparse attention and the MTP layer.
        @Test func policyIsFixedAtConstructionForEveryLayerAndTheMTP() throws {
            try Device.withDefaultDevice(.cpu) {
                #expect(GLM5NextExecutionPolicy() == .reference)
                for policy in [
                    GLM5NextExecutionPolicy.reference, Self.everyOptIn, Self.fusedDecode,
                ] {
                    let target = try makeModel(policy)
                    #expect(target.executionPolicy == policy)
                    #expect(target.glm5NextPagedMetadataReads == policy.pagedMetadataReads)
                    #expect(target.model.layers.allSatisfy { $0.executionPolicy == policy })
                    let kda = target.model.layers.compactMap {
                        $0.attention as? GLM5NextKDASublayer
                    }
                    #expect(
                        !kda.isEmpty
                            && kda.allSatisfy {
                                $0.linearAttention == policy.linearAttention
                                    && $0.decodeStep == policy.linearDecodeStep
                            })
                    let sparse = target.model.layers.compactMap(\.sparseAttention)
                    #expect(!sparse.isEmpty && sparse.allSatisfy { $0.executionPolicy == policy })
                    #expect(try makeMTP(target).executionPolicy == policy)
                }
            }
        }

        @Test func nativePoliciesMatchReferenceForwardOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                let model = try makeModel()
                let reference = try model.forward(tokens: tokens, cache: nil)
                for (label, policy) in [
                    ("mHC", Self.hyperConnection), ("experts", Self.routedExperts),
                    ("all", Self.allNative), ("mHC norm fused", Self.normFused),
                    ("fused decode", Self.fusedDecode),
                ] {
                    let native = try makeModel(policy)
                    let output = try native.forward(tokens: tokens, cache: nil)
                    try fence([reference.logits, reference.hidden, output.logits, output.hidden])
                    close(output.logits, reference.logits, "\(label) logits")
                    close(output.hidden, reference.hidden, "\(label) hidden")

                    let cache = native.newTextCache()
                    let singles = try (0 ..< tokens.dim(1)).map {
                        try native.forward(tokens: tokens[0..., $0 ..< $0 + 1], cache: cache).hidden
                    }
                    close(concatenated(singles, axis: 1), output.hidden, "\(label) tokenwise")

                    // The admitted candidate entry point serves the same policy.
                    let candidate = try native.prepareForward(
                        tokens: tokens, cache: native.newTextCache())
                    let prepared = try candidate.commit(afterEvaluating: fence)
                    #expect(Self.bits(prepared.logits) == Self.bits(output.logits), "\(label)")
                }
            }
        }

        @Test func nativePoliciesRefuseCPUWithoutChangingTheCache() throws {
            try Device.withDefaultDevice(.cpu) {
                func expectRefusal<Failure: Error & Equatable>(
                    _ policy: GLM5NextExecutionPolicy, _ failure: Failure
                ) throws {
                    let model = try makeModel(policy)
                    let cache = model.newTextCache()
                    let revision = cache.revision()
                    #expect(throws: failure) {
                        try model.prepareForward(tokens: tokens, cache: cache)
                    }
                    #expect(throws: failure) { try model.forward(tokens: tokens, cache: cache) }
                    #expect(cache.revision() == revision && cache.offset == 0)
                }
                try expectRefusal(
                    Self.hyperConnection,
                    GLM5NextHyperConnectionKernel.Failure.unsupportedDeviceOrStream)
                try expectRefusal(
                    Self.routedExperts, GLM5NextMoEKernel.Failure.unsupportedDeviceOrStream)
            }
        }

        /// Fused input projections keep the checkpoint's parameter tree (as row
        /// views of one buffer). The matmul kernels choose their reduction
        /// split by output width, so results agree to rounding, not bitwise.
        @Test(arguments: [DeviceType.cpu, .gpu])
        func fusedInputProjectionsKeepTreeAndOutputs(device: DeviceType) throws {
            try Device.withDefaultDevice(Device(device)) {
                let separate = try makeModel()
                let fused = try makeModel(GLM5NextExecutionPolicy(projections: .fusedInputs))
                func tree(_ model: GLM5NextTextModel) -> [String: [Int]] {
                    Dictionary(
                        uniqueKeysWithValues: model.parameters().flattened().map {
                            ($0.0, $0.1.shape)
                        })
                }
                #expect(tree(fused) == tree(separate))
                let kda = fused.model.layers.compactMap { $0.attention as? GLM5NextKDASublayer }
                #expect(!kda.isEmpty && kda.allSatisfy { $0.fusedInputs != nil })
                #expect(
                    fused.model.layers.compactMap(\.sparseAttention).allSatisfy {
                        $0.fusedInputs != nil
                    })
                let expected = try separate.forward(tokens: tokens, cache: nil)
                let actual = try fused.forward(tokens: tokens, cache: nil)
                try fence([expected.logits, actual.logits])
                close(actual.logits, expected.logits, tolerance: 1e-5, "fused prefill")
                let separateCache = separate.newTextCache()
                let fusedCache = fused.newTextCache()
                for index in 0 ..< tokens.dim(1) {
                    let step = tokens[0..., index ..< index + 1]
                    let a = try fused.forward(tokens: step, cache: fusedCache).logits
                    let b = try separate.forward(tokens: step, cache: separateCache).logits
                    try fence([a, b])
                    close(a, b, tolerance: 1e-5, "fused token \(index)")
                }
            }
        }

        /// The MTP readout's MoE follows its target's routed-expert choice. The
        /// unsorted native combine keeps every reference rounding boundary.
        @Test func mtpReadoutFollowsTargetRoutedExpertsOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                let referenceTarget = try makeModel()
                let nativeTarget = try makeModel(Self.routedExperts)
                let referenceMTP = try makeMTP(referenceTarget)
                let nativeMTP = try makeMTP(nativeTarget)
                #expect(nativeMTP.executionPolicy == Self.routedExperts)
                // One trusted target boundary feeds both predictors, isolating MTP.
                let hidden = try referenceTarget.forward(tokens: tokens, cache: nil).hidden
                let carry = MLXArray([Int32(7)], [1, 1])
                let expected = try referenceMTP.primePrompt(
                    tokens: tokens, targetHidden: hidden, carryToken: carry,
                    state: try referenceMTP.newRequestState())
                let actual = try nativeMTP.primePrompt(
                    tokens: tokens, targetHidden: hidden, carryToken: carry,
                    state: try nativeMTP.newRequestState())
                try fence([expected.hidden, expected.logits, actual.hidden, actual.logits])
                #expect(Self.bits(actual.hidden) == Self.bits(expected.hidden))
                #expect(Self.bits(actual.logits) == Self.bits(expected.logits))
            }
        }
    }

    /// Model-level official-oracle parity for the native mHC/MoE policy: full
    /// target, every layer, token and chunk continuation (the G3 shape used by
    /// the KDA candidate). Requires the prepared canonical CPU oracle.
    @Suite(
        .serialized,
        .enabled(
            if: GLM5NextReferenceFixture.enabled,
            "Set GLM53_REFERENCE_FIXTURE_DIR to the prepared canonical official CPU oracle"))
    struct GLM5NextExecutionPolicyOfficialReferenceTests {
        private static let policy = GLM5NextExecutionPolicy(
            hyperConnection: .nativeRequired, routedExperts: .nativeCombineSortedPrefill)
        /// The serving policy with early asynchronous step evaluation and
        /// `fusedDecode` routed experts.
        private static let fastest: GLM5NextExecutionPolicy = {
            var policy = GLM5NextExecutionPolicy.serving
            policy.sparseAttention = .absorbedDenseHistory
            policy.normalization = .fusedKernel
            policy.linearDecodeStep = .fusedEveryStep
            policy.routing = .nativeKernel
            policy.hyperConnection = .nativeNormFused
            policy.projections = .fusedInputs
            policy.stepEvaluation = .earlyAsync
            policy.routedExperts = .fusedDecode
            return policy
        }()
        static let policies = [policy, .serving, fastest]
        private typealias Checks = GLM5NextNativeKernelReferenceChecks

        @Test(arguments: KernelTests.GLM5NextExecutionPolicyOfficialReferenceTests.policies)
        func fullTargetAndEachLayerMatchOfficialOnGPU(policy: GLM5NextExecutionPolicy) throws {
            try Device.withDefaultDevice(.gpu) {
                let fixture = try GLM5NextReferenceFixture()
                let model = try fixture.textModel(executionPolicy: policy)
                for caseName in ["text_full", "text_left_padding"] {
                    let test = try fixture.caseObject(caseName)
                    let tokens = try fixture.tensor(fixture.key(test, "input_ids"))
                    let padded = caseName == "text_left_padding"
                    let validity =
                        padded
                        ? try fixture.tensor(fixture.key(test, "attention_mask")).asType(.bool)
                        : MLXArray.ones(tokens.shape, dtype: .bool)
                    let mask = padded ? validity : nil
                    let native = try model.forward(tokens: tokens, cache: nil, validity: validity)
                    try Checks.fence([native.logits, native.hidden])
                    try fixture.compare(native.logits, key: fixture.key(test, "logits"), mask: mask)
                    try fixture.compare(
                        native.hidden, key: fixture.key(test, "final_hidden"), mask: mask)

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
            }
        }

        @Test(arguments: KernelTests.GLM5NextExecutionPolicyOfficialReferenceTests.policies)
        func tokenAndChunkContinuationMatchOfficialOnGPU(policy: GLM5NextExecutionPolicy) throws {
            try Device.withDefaultDevice(.gpu) {
                let fixture = try GLM5NextReferenceFixture()
                let model = try fixture.textModel(executionPolicy: policy)
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
