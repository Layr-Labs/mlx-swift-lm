// Copyright 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    /// The two-launch router selects the reference experts (as a set per row)
    /// with matching weights whenever no near-tie straddles the K boundary.
    @Suite(.serialized)
    struct GLM5NextRouterKernelTests {
        private func router(
            experts: Int, hidden: Int, topK: Int, normalize: Bool, dtype: DType, seed: UInt64
        ) throws -> GLM5NextRouter {
            let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 2)
            return try GLM5NextRouter(
                hiddenSize: hidden, numExperts: experts, topK: topK,
                weight: (MLXRandom.normal([experts, hidden], key: keys[0]) * 0.05).asType(dtype),
                correctionBias: MLXRandom.normal([experts], key: keys[1]) * 0.1,
                normTopkProb: normalize, routedScalingFactor: 2.5)
        }

        @Test func nativeRoutingMatchesReferenceOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                for (experts, hidden, topK, normalize) in [
                    (288, 4096, 8, true), (64, 96, 4, false), (33, 64, 3, true),
                ] {
                    for tokens in [1, 3, 16] {
                        for dtype: DType in [.bfloat16, .float32] {
                            let gate = try router(
                                experts: experts, hidden: hidden, topK: topK,
                                normalize: normalize, dtype: dtype, seed: UInt64(experts + tokens))
                            let x = MLXRandom.normal(
                                [1, tokens, hidden], key: MLXRandom.key(UInt64(7 + tokens))
                            ).asType(dtype)
                            let expected = gate(x)
                            gate.usesNativeKernel = true
                            let actual = try gate.route(x)
                            gate.usesNativeKernel = false
                            eval(
                                expected.logits, expected.weights, expected.indices,
                                actual.logits, actual.weights, actual.indices)
                            let label = "E\(experts) D\(hidden) K\(topK) N\(tokens) \(dtype)"
                            #expect(actual.indices.dtype == .uint32, "\(label)")
                            #expect(actual.weights.shape == [tokens, topK], "\(label)")
                            let logitError = abs(actual.logits - expected.logits).max()
                                .item(Float.self)
                            #expect(logitError < 1e-4, "\(label) logits \(logitError)")
                            let expectedIndices = expected.indices.asType(.int32).asArray(
                                Int32.self)
                            let actualIndices = actual.indices.asType(.int32).asArray(Int32.self)
                            let expectedWeights = expected.weights.asArray(Float.self)
                            let actualWeights = actual.weights.asArray(Float.self)
                            for row in 0 ..< tokens {
                                let slots = row * topK ..< (row + 1) * topK
                                let reference = Dictionary(
                                    uniqueKeysWithValues: slots.map {
                                        (expectedIndices[$0], expectedWeights[$0])
                                    })
                                let native = Dictionary(
                                    uniqueKeysWithValues: slots.map {
                                        (actualIndices[$0], actualWeights[$0])
                                    })
                                #expect(
                                    Set(native.keys) == Set(reference.keys), "\(label) row \(row)")
                                for (expert, weight) in native {
                                    let error = abs(weight - (reference[expert] ?? .nan))
                                    #expect(error < 1e-5, "\(label) row \(row) expert \(expert)")
                                }
                            }
                        }
                    }
                }
            }
        }

        /// `Routing.nativeKernelPrefill`: tensor-op FP32 logits and the
        /// per-row selection against the reference graph on prefill-length
        /// steps. Logits differ only in FP32 summation order. A row may pick a
        /// different expert set only where the reference's K-th and (K+1)-th
        /// corrected scores are a near-tie; weights of matching sets agree.
        @Test(.enabled(if: GLM5NextMoEPrefillKernel.naxAvailable))
        func prefillRoutingMatchesReferenceOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                for (experts, hidden, topK, normalize) in [
                    (288, 4096, 8, true), (64, 256, 4, false), (33, 64, 3, true),
                ] {
                    for tokens in [17, 300, 2048] {
                        for dtype: DType in [.bfloat16, .float16] {
                            let gate = try router(
                                experts: experts, hidden: hidden, topK: topK,
                                normalize: normalize, dtype: dtype, seed: UInt64(experts + tokens))
                            let x = MLXRandom.normal(
                                [1, tokens, hidden], key: MLXRandom.key(UInt64(11 + tokens))
                            ).asType(dtype)
                            let expected = gate(x)
                            gate.usesNativeKernel = true
                            gate.usesPrefillKernel = true
                            let actual = try gate.route(x)
                            (gate.usesNativeKernel, gate.usesPrefillKernel) = (false, false)
                            eval(
                                expected.logits, expected.weights, expected.indices,
                                actual.logits, actual.weights, actual.indices)
                            let label = "E\(experts) D\(hidden) K\(topK) N\(tokens) \(dtype)"
                            #expect(actual.logits.shape == [tokens, experts], "\(label)")
                            let peak = abs(expected.logits).max().item(Float.self)
                            let logitError = abs(actual.logits - expected.logits).max()
                                .item(Float.self)
                            #expect(
                                logitError <= 1e-5 * max(1, peak), "\(label) logits \(logitError)")
                            let corrected =
                                (sigmoid(expected.logits)
                                + gate.correctionBias.asType(.float32)).asArray(Float.self)
                            let expectedIndices = expected.indices.asType(.int32).asArray(
                                Int32.self)
                            let actualIndices = actual.indices.asType(.int32).asArray(Int32.self)
                            let expectedWeights = expected.weights.asArray(Float.self)
                            let actualWeights = actual.weights.asArray(Float.self)
                            var nearTies = 0
                            for row in 0 ..< tokens {
                                let slots = row * topK ..< (row + 1) * topK
                                let reference = Dictionary(
                                    uniqueKeysWithValues: slots.map {
                                        (expectedIndices[$0], expectedWeights[$0])
                                    })
                                let native = Dictionary(
                                    uniqueKeysWithValues: slots.map {
                                        (actualIndices[$0], actualWeights[$0])
                                    })
                                guard Set(native.keys) == Set(reference.keys) else {
                                    let scores = corrected[row * experts ..< (row + 1) * experts]
                                        .sorted(by: >)
                                    #expect(
                                        scores[topK - 1] - scores[topK] <= 1e-5,
                                        "\(label) row \(row) differs without a near-tie")
                                    nearTies += 1
                                    continue
                                }
                                for (expert, weight) in native {
                                    let error = abs(weight - (reference[expert] ?? .nan))
                                    #expect(error < 1e-5, "\(label) row \(row) expert \(expert)")
                                }
                            }
                            print(
                                "GLM53_ROUTER_PREFILL \(label) logits max \(logitError) near-tie rows \(nearTies)"
                            )
                        }
                    }
                }
                // FP32 activations or a weight of another dtype keep the reference.
                let gate = try router(
                    experts: 32, hidden: 64, topK: 2, normalize: true, dtype: .float32, seed: 5)
                (gate.usesNativeKernel, gate.usesPrefillKernel) = (true, true)
                let x = MLXRandom.normal([1, 40, 64])
                let routed = try gate.route(x)
                let reference = gate(x)
                #expect(
                    routed.indices.asArray(UInt32.self) == reference.indices.asArray(UInt32.self))
            }
        }

        @Test func longStepsKeepTheReferenceAndCPURefuses() throws {
            let gate = try router(
                experts: 32, hidden: 64, topK: 2, normalize: true, dtype: .float32, seed: 3)
            gate.usesNativeKernel = true
            try Device.withDefaultDevice(.gpu) {
                let x = MLXRandom.normal([1, GLM5NextRouterKernel.maximumTokens + 1, 64])
                let routed = try gate.route(x)
                let reference = gate(x)
                #expect(
                    routed.indices.asArray(UInt32.self) == reference.indices.asArray(UInt32.self))
            }
            try Device.withDefaultDevice(.cpu) {
                #expect(throws: GLM5NextRouterKernel.Failure.unsupportedDeviceOrStream) {
                    try gate.route(MLXArray.zeros([1, 1, 64]))
                }
            }
        }
    }
}
