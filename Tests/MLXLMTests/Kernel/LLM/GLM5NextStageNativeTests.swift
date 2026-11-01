// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

extension KernelTests {
    /// Native serving uses BF16 activations and genuine NVFP4/affine checkpoint
    /// weights. The independent full-model load is the oracle, not fixture constants.
    @Suite(.serialized)
    struct GLM5NextStageNativeTests {
        private func configuration() throws -> GLM5NextConfiguration {
            let base = try GLM5NextInventoryFixture.configuration(layers: 8, q4: true)
            var json = try #require(
                JSONSerialization.jsonObject(with: JSONEncoder().encode(base)) as? [String: Any])
            let policy = GLM5NextNVFP4InventoryFixture.policy(base.textConfig)
            json["quantization"] = policy
            json["quantization_config"] = policy
            return try JSONDecoder().decode(
                GLM5NextConfiguration.self, from: JSONSerialization.data(withJSONObject: json))
        }

        private func fence(_ arrays: [MLXArray]) throws {
            try MLX.withError { error in
                eval(arrays)
                StreamOrDevice.default.stream.synchronize()
                try error.check()
            }
        }

        private func identical(_ actual: MLXArray, _ expected: MLXArray, _ label: String) throws {
            try #require(
                actual.shape == expected.shape && actual.dtype == expected.dtype, "\(label)")
            #expect(
                actual.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite }, "\(label)")
            #expect(
                expected.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite }, "\(label)"
            )
            #expect(
                actual.asData(access: .copy).data == expected.asData(access: .copy).data,
                "\(label)")
        }

        private func servingPolicy(_ stage: GLM5NextTextStage) throws {
            #expect(stage.executionPolicy == .serving)
            #expect(stage.executionPolicy.hyperConnection == .nativeNormFused)
            #expect(stage.layers.count == stage.plan.layers.count)
            for (globalIndex, layer) in zip(stage.plan.layers, stage.layers) {
                #expect(layer.executionPolicy == .serving)
                if let kda = layer.attention as? GLM5NextKDASublayer {
                    #expect(kda.linearAttention == GLM5NextExecutionPolicy.serving.linearAttention)
                    #expect(kda.decodeStep == GLM5NextExecutionPolicy.serving.linearDecodeStep)
                } else {
                    let sparse = try #require(layer.sparseAttention)
                    #expect(sparse.executionPolicy == .serving)
                    #expect(sparse.modelLayerIndex == globalIndex)
                }
            }
            let modules = stage.modules()
            let norms = modules.compactMap { $0 as? GLM5NextRMSNorm }
            #expect(!norms.isEmpty && norms.allSatisfy { $0.usesFusedKernel })
            let routers = modules.compactMap { $0 as? GLM5NextRouter }
            #expect(routers.count == stage.layers.filter { $0.mlp is GLM5NextMoE }.count)
            #expect(routers.allSatisfy { $0.usesNativeKernel })
            #expect(!modules.contains { $0 is GLM5NextTextModel })
        }

        // Global layer 3 is DSA, so these boundaries start with DSA and KDA,
        // respectively. Both halves also encounter native recurrent continuation.
        @Test(arguments: [3, 4])
        func servingSplitPreservesBF16NVFP4BoundaryAndCachedReadout(cut: Int) throws {
            try Device.withDefaultDevice(.gpu) {
                try #require(glm5NextOnDefaultGPUStream())
                let fixture = try GLM5NextNVFP4ArtifactFixture(configuration: configuration())
                let c = fixture.declared.textConfig
                let fullLoader = try fixture.loader(policy: .serving)
                let full = try fullLoader.loadTextModel()
                let firstLoader = try fixture.loader(policy: .serving)
                let lastLoader = try fixture.loader(policy: .serving)
                let first = try firstLoader.loadTextStage(
                    plan: .init(configuration: c, layers: 0 ..< cut))
                let last = try lastLoader.loadTextStage(
                    plan: .init(
                        configuration: c, layers: cut ..< c.hiddenLayers, loadMTPEmbedding: true))
                try servingPolicy(first)
                try servingPolicy(last)
                let banks = (first.modules() + last.modules()).compactMap {
                    $0 as? QuantizedSwitchLinear
                }
                #expect(!banks.isEmpty)
                #expect(
                    banks.allSatisfy {
                        $0.mode == .nvfp4 && $0.globalScales?.shape == [c.nRoutedExperts]
                    })
                #expect(first.norm == nil && first.head == nil)
                #expect(last.embedding != nil && last.norm != nil && last.head != nil)

                let fullCache = full.newTextCache()
                let firstCache = first.newCache()
                let lastCache = last.newCache()
                let chunks: [(MLXArray, MLXArray)] = [
                    (
                        MLXArray([Int32(0), 0, 2, 5, 3, 9, 7, 1, 4], [1, 9]),
                        MLXArray([false, false, true, true, true, true, true, true, true], [1, 9])
                    ),
                    (MLXArray([Int32(9), 7], [1, 2]), MLXArray.ones([1, 2], dtype: .bool)),
                    (MLXArray([Int32(4)], [1, 1]), MLXArray.ones([1, 1], dtype: .bool)),
                ]
                // Exercise both the prefill and split/decode norm-fusion kernels.
                #expect(chunks[0].0.dim(1) > GLM5NextHyperConnectionKernel.splitCollapseMaximumRows)
                for (chunk, (tokens, validity)) in chunks.enumerated() {
                    let expected = try full.prepareForward(
                        tokens: tokens, cache: fullCache, validity: validity
                    ).commit(afterEvaluating: fence)
                    let before = firstCache.offset
                    let candidate = try first.prepareForward(
                        tokens: tokens, cache: firstCache, validity: validity)
                    guard case .streams(let boundary) = candidate.output else {
                        Issue.record("intermediate stage returned a final readout")
                        return
                    }
                    #expect(boundary.shape == [1, tokens.dim(1), c.hcMult, c.hiddenSize])
                    #expect(boundary.dtype == .bfloat16)
                    let tail = try last.prepareForward(
                        streams: boundary, cache: lastCache, validity: validity)
                    #expect(firstCache.offset == before && lastCache.offset == before)
                    var actual: GLM5NextTextOutput?
                    try candidate.commit { roots in
                        try fence(roots)
                        if case .final(let output) = try tail.commit(afterEvaluating: fence) {
                            actual = output
                        }
                    }
                    let output = try #require(actual)
                    // Folded expand/collapse and the materialized stage boundary
                    // have the same dtype rounding contract (norm-fusion tests).
                    let label = "cut \(cut), chunk \(chunk)"
                    try identical(output.hidden, expected.hidden, "\(label) hidden")
                    try identical(output.logits, expected.logits, "\(label) logits")
                    #expect(firstCache.offset == fullCache.offset)
                    #expect(lastCache.offset == fullCache.offset)
                    for (layer, (split, original)) in zip(
                        firstCache.layers + lastCache.layers, fullCache.layers
                    ).enumerated() {
                        #expect(split.offset == original.offset)
                        try #require(split.state.count == original.state.count)
                        for (index, (a, b)) in zip(split.state, original.state).enumerated() {
                            try identical(a, b, "\(label) layer \(layer) state \(index)")
                        }
                    }
                }

                let mtp = try lastLoader.loadMTP(stage: last)
                #expect(mtp.executionPolicy == .serving)
                let attention = try #require(
                    mtp.modules().compactMap { $0 as? GLM5NextDSAAttention }.first)
                #expect(attention.executionPolicy == .serving)
                #expect(attention.modelLayerIndex == c.hiddenLayers)
                #expect(
                    !mtp.modules().contains { $0 is GLM5NextTextModel || $0 is GLM5NextTextStage })
                try fence([])
            }
        }

        @Test func servingStagesRefuseCPUAndNondefaultGPUStreamWithoutPublishing() throws {
            try Device.withDefaultDevice(.gpu) {
                try #require(glm5NextOnDefaultGPUStream())
                let fixture = try GLM5NextNVFP4ArtifactFixture()
                let c = fixture.declared.textConfig
                let first = try fixture.loader(policy: .serving).loadTextStage(
                    plan: .init(configuration: c, layers: 0 ..< 3))
                let last = try fixture.loader(policy: .serving).loadTextStage(
                    plan: .init(configuration: c, layers: 3 ..< c.hiddenLayers))
                let firstCache = first.newCache()
                let lastCache = last.newCache()
                let tokens = MLXArray([Int32(2), 5], [1, 2])
                let prefix = try first.prepareForward(tokens: tokens, cache: firstCache)
                guard case .streams(let boundary) = prefix.output else {
                    Issue.record("missing native boundary")
                    return
                }
                let tail = try last.prepareForward(streams: boundary, cache: lastCache)
                // Prove this geometry and dtype actually succeed before testing refusal.
                try prefix.commit(afterEvaluating: fence)
                try tail.commit(afterEvaluating: fence)
                func snapshot() -> [[Data]] {
                    (firstCache.layers + lastCache.layers).map {
                        $0.state.map { $0.asData(access: .copy).data }
                    }
                }
                let before = snapshot()
                func refused() {
                    #expect(throws: GLM5NextHyperConnectionKernel.Failure.unsupportedDeviceOrStream)
                    {
                        try first.prepareForward(tokens: tokens, cache: firstCache)
                    }
                    #expect(throws: GLM5NextHyperConnectionKernel.Failure.unsupportedDeviceOrStream)
                    {
                        try last.prepareForward(streams: boundary, cache: lastCache)
                    }
                    #expect(firstCache.offset == 2 && lastCache.offset == 2)
                }
                Device.withDefaultDevice(.cpu) { refused() }
                #expect(snapshot() == before)
                Stream.withNewDefaultStream(device: .gpu) { refused() }
                #expect(snapshot() == before)
                try fence([])
            }
        }
    }
}
