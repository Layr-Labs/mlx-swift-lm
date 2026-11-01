// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextStageLoadOwnershipTests {
        enum Partition: CaseIterable, Sendable { case stage, mtp }
        enum FailurePoint: CaseIterable, Sendable { case admission, expertBank }
        private enum InjectedFailure: Error, Equatable { case admission, expertBank }

        /// Observe Swift allocation ownership without keeping the observed array alive.
        private final class ArrayWitness {
            weak var value: MLXArray?
            init(_ value: MLXArray) { self.value = value }
        }

        private func clearAfterFence(_ roots: GLM5NextArtifactLoadRoots) throws {
            try MLX.withError { error in
                StreamOrDevice.default.stream.synchronize()
                try error.check()
                roots.releaseAfterConfirmedFence()
            }
        }

        @Test(arguments: Partition.allCases, FailurePoint.allCases)
        func partialLoadKeepsRootsUntilCleanupAndRequiresFreshOwner(
            partition: Partition, failurePoint: FailurePoint
        ) throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextNVFP4ArtifactFixture()
                let identity = try fixture.loader().identity
                let c = fixture.declared.textConfig
                let plan = try GLM5NextTextStagePlan(
                    configuration: c, layers: 3 ..< c.hiddenLayers, loadMTPEmbedding: true)
                var armed = false
                var bankFailures = 0
                var witnesses: [ArrayWitness] = []
                let roots = GLM5NextArtifactLoadRoots { arrays in
                    guard armed else { return }
                    witnesses += arrays.map { ArrayWitness($0) }
                    // A member is a 2D packed matrix; this is the real stacked bank,
                    // retained before its evaluation hook and before source retirement.
                    if failurePoint == .expertBank,
                        arrays.contains(where: {
                            $0.dtype == .uint32 && $0.ndim == 3
                                && $0.dim(0) == c.nRoutedExperts
                        })
                    {
                        bankFailures += 1
                        throw InjectedFailure.expertBank
                    }
                }
                defer {
                    do { try clearAfterFence(roots) } catch {
                        Issue.record(error, "construction-root cleanup fence failed")
                    }
                }
                let loader = try GLM5NextArtifactLoader(
                    directory: fixture.directory, expectedIdentity: identity,
                    storage: .nvfp4Hybrid, constructionRoots: roots)
                var target: GLM5NextTextStage?
                weak var targetWitness: GLM5NextTextStage?
                if partition == .mtp {
                    target = try loader.loadTextStage(plan: plan)
                    targetWitness = target
                }
                let baselineOwners = partition == .mtp ? 1 : 0
                #expect(roots.retainedOwnerCount == baselineOwners)
                #expect(roots.retainedArrayCount == 0)
                let part: GLM5NextWeightInventory.Part = partition == .mtp ? .mtp : .text
                let otherPart: GLM5NextWeightInventory.Part = partition == .mtp ? .text : .mtp
                let unresolvedBefore = Set(loader.unresolved(in: part))
                let otherBefore = Set(loader.unresolved(in: otherPart))
                let visionBefore = Set(loader.unresolved(in: .vision))
                let selected: Set<String>
                if partition == .mtp {
                    selected = Set(loader.inventory.entries(in: .mtp).map { $0.tensor.name })
                } else {
                    selected = try plan.tensorNames(in: loader.inventory)
                }
                var requests: [String] = []
                func admission(_ name: String, _ bytes: Int) throws {
                    #expect(selected.contains(name))
                    #expect(loader.inventory.entries[name]?.bytes == bytes)
                    requests.append(name)
                    // Five actual payloads have been allocated/evaluated, but no
                    // routed bank has been completed. This is not a first-read refusal.
                    if failurePoint == .admission && requests.count == 6 {
                        throw InjectedFailure.admission
                    }
                }
                func attempt() throws {
                    if partition == .mtp {
                        _ = try loader.loadMTP(
                            stage: #require(target), beforeTensorRead: admission)
                    } else {
                        _ = try loader.loadTextStage(plan: plan, beforeTensorRead: admission)
                    }
                }
                armed = true
                let failure: InjectedFailure =
                    failurePoint == .admission ? .admission : .expertBank
                #expect(throws: failure) { try attempt() }

                let unresolvedAfter = Set(loader.unresolved(in: part))
                let consumed = unresolvedBefore.subtracting(unresolvedAfter)
                try #require(!consumed.isEmpty && !witnesses.isEmpty)
                #expect(consumed.isSubset(of: selected))
                #expect(selected.subtracting(consumed).isSubset(of: unresolvedAfter))
                #expect(Set(loader.unresolved(in: otherPart)) == otherBefore)
                #expect(Set(loader.unresolved(in: .vision)) == visionBefore)
                #expect(roots.retainedOwnerCount == baselineOwners)
                #expect(witnesses.allSatisfy { $0.value != nil })
                if failurePoint == .admission {
                    #expect(requests.count == 6 && bankFailures == 0)
                    #expect(consumed == Set(requests.prefix(5)))
                    let refusedRead = try #require(requests.last)
                    #expect(unresolvedAfter.contains(refusedRead))
                    #expect(roots.retainedArrayCount == 5)
                    #expect(roots.retainedExpertSourceCount == 0)
                } else {
                    #expect(bankFailures == 1)
                    #expect(consumed == Set(requests))
                    #expect(roots.retainedArrayCount == consumed.count + 1)
                    #expect(roots.retainedExpertSourceCount == c.nRoutedExperts)
                    let globalLayer = partition == .mtp ? c.hiddenLayers : plan.layers.lowerBound
                    let members = Set(
                        (0 ..< c.nRoutedExperts).map {
                            "model.language_model.layers.\(globalLayer).mlp.experts.\($0).gate_proj.weight"
                        })
                    #expect(members.isSubset(of: consumed))
                }

                // A partially consumed owner must fail closed, without allocating
                // another tensor or transferring partial roots into a completed owner.
                let readsBeforeRetry = requests.count
                let rootsBeforeRetry = roots.retainedArrayCount
                let witnessesBeforeRetry = witnesses.count
                if partition == .stage {
                    #expect(
                        throws: GLM5NextArtifactError.invalid(
                            "text stage requires an unconsumed artifact owner")
                    ) { try attempt() }
                    #expect(requests.count == readsBeforeRetry)
                } else {
                    let firstRead = try #require(requests.first)
                    #expect(throws: GLM5NextArtifactError.alreadyConsumed(firstRead)) {
                        try attempt()
                    }
                    // MTP admission precedes the resolver's single-consumption check.
                    #expect(requests.count == readsBeforeRetry + 1)
                }
                #expect(Set(loader.unresolved(in: part)) == unresolvedAfter)
                #expect(roots.retainedArrayCount == rootsBeforeRetry)
                #expect(roots.retainedOwnerCount == baselineOwners)
                #expect(witnesses.count == witnessesBeforeRetry)
                target = nil
                if partition == .mtp { #expect(targetWitness != nil) }
                #expect(witnesses.allSatisfy { $0.value != nil })
                try clearAfterFence(roots)
                #expect(roots.retainedArrayCount == 0 && roots.retainedOwnerCount == 0)
                #expect(roots.retainedExpertSourceCount == 0)
                #expect(witnesses.allSatisfy { $0.value == nil })
                #expect(targetWitness == nil)

                // The same unchanged artifact succeeds with a fresh owner and roots.
                // This also proves the failure did not damage the artifact or schema.
                let freshRoots = GLM5NextArtifactLoadRoots()
                defer {
                    do { try clearAfterFence(freshRoots) } catch {
                        Issue.record(error, "fresh-owner cleanup fence failed")
                    }
                }
                let fresh = try GLM5NextArtifactLoader(
                    directory: fixture.directory, expectedIdentity: identity,
                    storage: .nvfp4Hybrid, constructionRoots: freshRoots)
                let stage = try fresh.loadTextStage(plan: plan)
                if partition == .mtp {
                    _ = try fresh.loadMTP(stage: stage)
                    #expect(fresh.unresolved(in: .mtp).isEmpty)
                }
                #expect(
                    Set(fresh.unresolved(in: .text))
                        == Set(fresh.inventory.entries(in: .text).map { $0.tensor.name })
                        .subtracting(try plan.tensorNames(in: fresh.inventory)))
                #expect(freshRoots.retainedArrayCount == 0)
                #expect(freshRoots.retainedExpertSourceCount == 0)
                #expect(freshRoots.retainedOwnerCount == (partition == .mtp ? 2 : 1))
                #expect(Set(fresh.unresolved(in: .vision)) == visionBefore)
                try fresh.verifyUnchanged()
                try clearAfterFence(freshRoots)
                #expect(freshRoots.retainedArrayCount == 0 && freshRoots.retainedOwnerCount == 0)
            }
        }
    }
}
