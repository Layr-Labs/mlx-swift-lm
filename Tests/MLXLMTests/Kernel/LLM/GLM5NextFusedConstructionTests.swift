// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Cmlx
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextFusedConstructionTests {
        enum Family: CaseIterable, Sendable { case dense, shared, dsa, kda }
        enum Entry: CaseIterable, Sendable { case model, modelMTP }
        private enum Refusal: Error { case injected }
        private final class Witness {
            weak var array: MLXArray?
            init(_ array: MLXArray) { self.array = array }
        }

        private func available(_ array: MLXArray) throws -> Bool {
            var result = false
            try #require(_mlx_array_is_available(&result, array.ctx) == 0)
            return result
        }

        private func retire(_ roots: GLM5NextArtifactLoadRoots) throws {
            try MLX.withError { error in
                Stream.gpu.synchronize()
                Stream.cpu.synchronize()
                try error.check()
                roots.releaseAfterConfirmedFence()
            }
        }

        @Test(arguments: Family.allCases, [false, true])
        func eachProjectionFamilyRetainsFailedMaterialization(
            family: Family, evaluateThenThrow: Bool
        ) throws {
            let fixture = try GLM5NextLocalArtifactFixture()
            let artifact = try fixture.loader()
            let c = fixture.configuration.textConfig
            var refused: [Witness] = []
            weak var observedRoots: GLM5NextArtifactLoadRoots?
            let roots = GLM5NextArtifactLoadRoots { arrays in
                try #require(arrays.count > 1)
                #expect(try #require(observedRoots).retainedArrayCount >= arrays.count)
                #expect(try !available(arrays[0]))
                refused = arrays.map { Witness($0) }
                if evaluateThenThrow {
                    try MLX.withError { error in
                        eval(arrays)
                        Stream.gpu.synchronize()
                        try error.check()
                    }
                }
                throw Refusal.injected
            }
            observedRoots = roots
            defer {
                do { try retire(roots) } catch {
                    Issue.record(error, "fused construction cleanup failed")
                }
            }
            let loader = GLM5NextTensorLoader(
                artifact.inventory, artifact.tensor,
                materializeFusedProjections: roots.evaluate)
            #expect(throws: Refusal.self) {
                switch family {
                case .dense:
                    _ = try loader.mlp(
                        c, prefix: "model.language_model.layers.0.mlp",
                        intermediate: c.intermediateSize, projections: .fusedInputs)
                case .shared:
                    _ = try loader.moe(
                        c, prefix: "model.language_model.layers.3.mlp",
                        materialize: artifact.materializeExpertStack, projections: .fusedInputs)
                case .dsa:
                    // Exercise the public standalone initializer as well as DSA's loader.
                    _ = try GLM5NextDSAAttention(
                        configuration: c, sourcePrefix: "model.language_model.layers.3.self_attn",
                        inventory: artifact.inventory, tensor: artifact.tensor,
                        executionPolicy: .serving, materializeFusedProjections: roots.evaluate)
                case .kda:
                    _ = try GLM5NextKDASublayer(
                        c, prefix: "model.language_model.layers.0.self_attn", loader: loader,
                        linearAttention: .reference, projections: .fusedInputs)
                }
            }
            try #require(!refused.isEmpty)
            #expect(roots.retainedArrayCount == refused.count)
            for witness in refused {
                // The failed constructor has unwound; only the construction owner
                // retains these wrappers, including after completed native work.
                #expect(try available(#require(witness.array)) == evaluateThenThrow)
            }
            try retire(roots)
            #expect(roots.retainedArrayCount == 0)
            #expect(refused.allSatisfy { $0.array == nil })
        }

        @Test(arguments: Entry.allCases, [false, true])
        func artifactOwnsFusedGroupsAcrossPartialFailure(entry: Entry, packed: Bool) throws {
            let fixture = try GLM5NextLocalArtifactFixture(q4: packed)
            let isMTP = entry == .modelMTP
            let refuseAt = isMTP ? 1 : 2
            var armed = false
            var groups: [[Witness]] = []
            weak var observedRoots: GLM5NextArtifactLoadRoots?
            let roots = GLM5NextArtifactLoadRoots { arrays in
                // Tensor reads and expert banks each evaluate one array. Fused
                // groups contain the concatenated storage and all member views.
                guard armed, arrays.count > 1 else { return }
                #expect(try #require(observedRoots).retainedArrayCount >= arrays.count)
                #expect(try !available(arrays[0]))
                groups.append(arrays.map { Witness($0) })
                if groups.count == refuseAt { throw Refusal.injected }
            }
            observedRoots = roots
            defer {
                do { try retire(roots) } catch {
                    Issue.record(error, "artifact construction cleanup failed")
                }
            }
            let loader = try GLM5NextArtifactLoader(
                directory: fixture.directory, expectedIdentity: fixture.identity(),
                storage: fixture.storage, constructionRoots: roots, executionPolicy: .serving)
            let model = entry == .modelMTP ? try loader.loadTextModel() : nil
            let baselineOwners = roots.retainedOwnerCount
            armed = true
            #expect(throws: Refusal.self) {
                switch entry {
                case .model: _ = try loader.loadTextModel()
                case .modelMTP: _ = try loader.loadMTP(target: #require(model))
                }
            }
            try #require(groups.count == refuseAt)
            #expect(roots.retainedOwnerCount == baselineOwners)
            #expect(roots.retainedArrayCount >= groups.flatMap { $0 }.count)
            for (index, group) in groups.enumerated() {
                let first = try #require(group.first?.array)
                #expect(try available(first) == (index < refuseAt - 1))
                #expect(group.allSatisfy { $0.array != nil })
            }
            try retire(roots)
            #expect(roots.retainedArrayCount == 0)
            #expect(groups.flatMap { $0 }.allSatisfy { $0.array == nil })
        }
    }
}
