// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

extension KernelTests {
    /// Model-level wiring of `RoutedExperts.fusedDecode` on the deterministic
    /// tiny target. Its FP32 expert banks are not affine-quantized, so every
    /// step keeps `nativeCombineSortedPrefill` bitwise; kernel arithmetic is
    /// covered by `GLM5NextMoEDecodeKernelTests` and the real-checkpoint probes.
    @Suite(.serialized)
    struct GLM5NextMoEDecodePolicyTests {
        private static let fusedDecode = GLM5NextExecutionPolicy(
            routedExperts: .fusedDecode, routing: .nativeKernel)
        private static let sortedPrefill = GLM5NextExecutionPolicy(
            routedExperts: .nativeCombineSortedPrefill, routing: .nativeKernel)

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

        private func makeModel(_ policy: GLM5NextExecutionPolicy) throws -> GLM5NextTextModel {
            let inventory = try Self.inventory()
            return try GLM5NextTextModel(
                configuration: inventory.configuration.textConfig, inventory: inventory,
                tensor: { try Self.tensor(inventory, $0) }, executionPolicy: policy)
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

        private let tokens = MLXArray([Int32(2), 5, 3, 9, 7, 1, 4], [1, 7])

        @Test func unreadableBanksKeepSortedPrefillForEveryStepOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                let expected = try makeModel(Self.sortedPrefill)
                let actual = try makeModel(Self.fusedDecode)
                #expect(actual.executionPolicy == Self.fusedDecode)
                let moes = actual.model.layers.compactMap { $0.mlp as? GLM5NextMoE }
                #expect(!moes.isEmpty)
                #expect(moes.allSatisfy { GLM5NextMoEDecodeKernel.Plan($0, dtype: .float32) == nil })

                let full = try actual.forward(tokens: tokens, cache: nil)
                let reference = try expected.forward(tokens: tokens, cache: nil)
                try fence([full.logits, reference.logits])
                #expect(Self.bits(full.logits) == Self.bits(reference.logits))

                let actualCache = actual.newTextCache()
                let expectedCache = expected.newTextCache()
                for index in 0 ..< tokens.dim(1) {
                    let step = tokens[0..., index ..< index + 1]
                    let a = try actual.forward(tokens: step, cache: actualCache).logits
                    let b = try expected.forward(tokens: step, cache: expectedCache).logits
                    try fence([a, b])
                    #expect(Self.bits(a) == Self.bits(b), "token \(index)")
                }
            }
        }

        @Test func cpuRefusesWithoutChangingTheCache() throws {
            try Device.withDefaultDevice(.cpu) {
                let model = try makeModel(Self.fusedDecode)
                let cache = model.newTextCache()
                let revision = cache.revision()
                #expect(throws: GLM5NextMoEKernel.Failure.unsupportedDeviceOrStream) {
                    try model.forward(tokens: tokens[0..., ..<1], cache: cache)
                }
                #expect(cache.revision() == revision && cache.offset == 0)
            }
        }
    }
}
