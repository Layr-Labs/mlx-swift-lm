import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {
    /// Real DSA/MoE modules with fixed small weights. These are equation and state
    /// checks, not trained-checkpoint quality, stochastic acceptance or speed tests.
    @Suite(.serialized)
    struct GLM5NextMTPTests {
        @Test func trainedLoaderSharesTargetHeadAndConsumesOnlyMTPPartition() throws {
            let f = try Fixture()
            let expected = Set(f.inventory.entries(in: .mtp).map { $0.tensor.name })
            #expect(Set(f.requests.names) == expected)
            #expect(!f.requests.names.contains("lm_head.weight"))
            #expect(!f.requests.names.contains("model.language_model.embed_tokens.weight"))
            #expect(f.mtp.targetIdentity == ObjectIdentifier(f.target))
            #expect(f.mtp.maximumDraftTokens == 1)
            let names = f.mtp.parameters().flattened().map(\.0)
            #expect(names.contains("shared_head.norm.weight"))
            #expect(!names.contains { $0.contains("hc_") })
            var released = Set<String>()
            _ = try GLM5NextMTP(
                target: f.target, tensor: f.tensor,
                releaseSourceTensors: { released.formUnion($0) })
            #expect(released == expected)
            #expect(throws: GLM5NextLoadError.self) {
                try GLM5NextMTP(target: f.target) { name in
                    if name.hasSuffix(".hnorm.weight") { throw GLM5NextLoadError.missing(name) }
                    return try f.tensor(name)
                }
            }
        }

        @Test func q4MTPRequiresBothLifetimeCallbacksBeforeReadingPayloads() throws {
            let base = try Fixture()
            var root = try #require(
                JSONSerialization.jsonObject(
                    with: JSONEncoder().encode(base.inventory.configuration)) as? [String: Any])
            var text = try #require(root["text_config"] as? [String: Any])
            for key in [
                "hidden_size", "intermediate_size", "moe_intermediate_size", "q_lora_rank",
                "kv_lora_rank", "qk_head_dim", "qk_nope_head_dim", "v_head_dim", "index_head_dim",
            ] {
                text[key] = 64
            }
            var linear = try #require(text["linear_attn_config"] as? [String: Any])
            linear["head_dim"] = 64
            text["linear_attn_config"] = linear
            root["text_config"] = text
            var vision = try #require(root["vision_config"] as? [String: Any])
            vision["out_hidden_size"] = 64
            root["vision_config"] = vision
            root["quantization"] = ["bits": 4, "group_size": 64, "mode": "affine"]
            let c = try JSONDecoder().decode(
                GLM5NextConfiguration.self,
                from: JSONSerialization.data(withJSONObject: root))
            let headers = try GLM5NextWeightInventory.expectedTensors(
                configuration: c, storage: .vontraQ4, shard: "packed-fixed.safetensors")
            let inventory = try GLM5NextWeightInventory.validate(
                configuration: c, tensors: headers, storage: .vontraQ4)
            func tensor(_ name: String) throws -> MLXArray {
                let header = try #require(inventory.entries[name]).tensor
                let dtype: DType =
                    header.dtype == "U32" ? .uint32 : header.dtype == "F32" ? .float32 : .bfloat16
                let one =
                    name.hasSuffix(".scales") || name.contains("norm.weight")
                    || name.hasSuffix("_scale")
                return one
                    ? MLXArray.ones(header.shape, dtype: dtype)
                    : MLXArray.zeros(header.shape, dtype: dtype)
            }
            func materialize(_ array: MLXArray) throws {
                try withError { error in
                    eval(array)
                    try error.check()
                }
            }
            let target = try GLM5NextTextModel(
                configuration: c.textConfig, inventory: inventory,
                tensor: tensor, materializeExpertStack: materialize, releaseSourceTensors: { _ in })
            var reads = 0
            func counted(_ name: String) throws -> MLXArray {
                reads += 1
                return try tensor(name)
            }
            #expect(throws: GLM5NextMTPError.self) {
                try GLM5NextMTP(
                    target: target, tensor: counted, materializeExpertStack: materialize)
            }
            #expect(reads == 0)
            #expect(throws: GLM5NextMTPError.self) {
                try GLM5NextMTP(target: target, tensor: counted, releaseSourceTensors: { _ in })
            }
            #expect(reads == 0)
            var released = Set<String>()
            _ = try GLM5NextMTP(
                target: target, tensor: counted, materializeExpertStack: materialize,
                releaseSourceTensors: { released.formUnion($0) })
            #expect(reads > 0)
            #expect(released == Set(inventory.entries(in: .mtp).map { $0.tensor.name }))
        }

        @Test func targetBoundaryProjectionResidualAndSingleReadoutMatchScalarOracle() throws {
            let f = try Fixture()
            let ids = [1, 2, 3]
            let targetCache = f.target.newTextCache()
            let target = try f.target.forward(tokens: tokens(ids), cache: targetCache)
            let targetState = targetCache.layers.map { $0.state.map(values) }
            let trusted = ids.map { id in
                normalize(
                    Fixture.embedding(id).map { $0 * Fixture.targetResidualScale },
                    weight: Fixture.targetNorm)
            }
            close(target.hidden, trusted.flatMap { $0 })
            let shifted = [2, 3, 4]
            let oracle = reference(shifted: shifted, hidden: trusted)
            let projected = try f.mtp.projectInput(
                shiftedEmbeddings: f.target.embed(tokens(shifted)),
                targetHidden: target.hidden, firstPosition: 0)
            close(projected, oracle.projected.flatMap { $0 })
            let state = try f.mtp.newRequestState()
            let before = f.head.calls
            let output = try f.mtp.primePrompt(
                tokens: tokens(ids), targetHidden: target.hidden, carryToken: tokens([4]),
                state: state)
            #expect(f.head.calls == before + 1)
            #expect(state.committedInputCount == ids.count && state.stagedInputCount == 0)
            close(output.hidden, oracle.hidden.flatMap { $0 })
            #expect(output.logits.shape == [1, 1, 16])
            close(output.logits, oracle.logits.last!)
            close(try #require(state.recycledHidden), oracle.hidden.last!)
            #expect(targetCache.layers.map { $0.state.map(values) } == targetState)
        }

        @Test func fullShiftedPrimingMatchesChunksAcrossPoolBoundary() throws {
            let f = try Fixture()
            let shifted = [2, 3, 4, 5, 6]
            let trusted = features(shifted.count)
            let fullState = try f.mtp.newRequestState()
            let full = try f.mtp.prime(
                shiftedTokens: tokens(shifted), targetHidden: rows(trusted), at: 0, state: fullState
            )
            let oracle = reference(shifted: shifted, hidden: trusted)
            close(full.hidden, oracle.hidden.flatMap { $0 })
            #expect(full.logits.shape == [1, 1, 16])
            close(full.logits, oracle.logits.last!)
            let splitState = try f.mtp.newRequestState()
            var outputs = [MLXArray]()
            for range in [0 ..< 2, 2 ..< 4, 4 ..< 5] {
                let result = try f.mtp.prime(
                    shiftedTokens: tokens(Array(shifted[range])),
                    targetHidden: rows(Array(trusted[range])), at: range.lowerBound,
                    state: splitState)
                outputs.append(result.hidden)
            }
            close(concatenated(outputs, axis: 1), values(full.hidden).map(Double.init))
            #expect(splitState.committedInputCount == 5)
            #expect(try f.mtp.evaluationTargets(state: splitState).first?.shape == [1, 5, 2])

            let e = f.target.embed(tokens([3]))
            let h = rows([trusted[1]])
            let first = try f.mtp.projectInput(
                shiftedEmbeddings: e, targetHidden: h, firstPosition: 0)
            let later = try f.mtp.projectInput(
                shiftedEmbeddings: e, targetHidden: h, firstPosition: 2)
            close(
                first,
                project(embedding: Fixture.embedding(3), hidden: trusted[1], zeroEmbedding: true))
            close(
                later,
                project(embedding: Fixture.embedding(3), hidden: trusted[1], zeroEmbedding: false))
            #expect(values(first) != values(later))
        }

        @Test func roundsRollbackExactlyAndRejectUnimplementedIndexSharingDepth() throws {
            let f = try Fixture()
            let state = try f.mtp.newRequestState()
            _ = try f.mtp.prime(
                shiftedTokens: tokens([2, 3, 4, 5]), targetHidden: rows(features(4)), at: 0,
                state: state)
            let prefix = try f.mtp.capture(state: state)
            let saved = try f.mtp.evaluationTargets(state: state).map(values)
            #expect(throws: GLM5NextMTPError.unsupportedDraftDepth(2)) {
                try f.mtp.beginRound(state: state, candidates: 2)
            }
            #expect(state.committedInputCount == 4 && state.stagedInputCount == 0)
            try f.mtp.beginRound(state: state)
            let proposed = try f.mtp.draftStep(
                shiftedToken: tokens([6]), targetHidden: rows([features(5)[4]]), state: state)
            #expect(state.committedInputCount == 4 && state.stagedInputCount == 1)
            #expect(throws: GLM5NextMTPError.activeRound) { try f.mtp.capture(state: state) }
            #expect(throws: GLM5NextMTPError.unsupportedDraftDepth(2)) {
                try f.mtp.draftStep(
                    shiftedToken: proposed.greedyTokens, targetHidden: proposed.hidden, state: state
                )
            }
            #expect(throws: GLM5NextMTPError.self) {
                try f.mtp.commitRound(state: state, acceptingInputs: 2)
            }
            try f.mtp.rollbackRound(state: state)
            #expect(try f.mtp.evaluationTargets(state: state).map(values) == saved)
            #expect(prefix.inputCount == 4)
            try f.mtp.beginRound(state: state)
            let replay = try f.mtp.draftStep(
                shiftedToken: tokens([6]), targetHidden: rows([features(5)[4]]), state: state)
            close(replay.logits, values(proposed.logits).map(Double.init))
            try f.mtp.commitRound(state: state, acceptingInputs: 1)
            #expect(state.committedInputCount == 5 && state.stagedInputCount == 0)
            try f.mtp.restore(prefix, into: state)
            try f.mtp.beginRound(state: state)
            _ = try f.mtp.draftStep(
                shiftedToken: tokens([6]), targetHidden: rows([features(5)[4]]), state: state)
            try f.mtp.commitRound(state: state, acceptingInputs: 0)
            #expect(try f.mtp.evaluationTargets(state: state).map(values) == saved)
        }

        @Test func captureRestoreTrimResetAndReleaseAreRequestLocal() throws {
            let f = try Fixture()
            let original = try f.mtp.newRequestState(maximumInputs: 8)
            _ = try f.mtp.prime(
                shiftedTokens: tokens([2, 3, 4, 5]), targetHidden: rows(features(4)), at: 0,
                state: original)
            let checkpoint = try f.mtp.capture(state: original)
            let copy = try f.mtp.newRequestState(maximumInputs: 8)
            try f.mtp.restore(checkpoint, into: copy)
            let originalArrays = try f.mtp.evaluationTargets(state: original).map(values)
            _ = try f.mtp.prime(
                shiftedTokens: tokens([7]), targetHidden: rows([features(5)[4]]), at: 4, state: copy
            )
            #expect(original.committedInputCount == 4 && copy.committedInputCount == 5)
            #expect(try f.mtp.evaluationTargets(state: original).map(values) == originalArrays)
            #expect(throws: GLM5NextLoadError.self) { try f.mtp.trim(state: copy, to: 6) }
            #expect(copy.committedInputCount == 5)
            try f.mtp.trim(state: copy, to: 3)
            #expect(copy.committedInputCount == 3 && copy.recycledHidden == nil)
            try f.mtp.restore(checkpoint, into: copy)
            #expect(copy.committedInputCount == 4)
            try f.mtp.reset(state: copy)
            #expect(copy.committedInputCount == 0 && copy.recycledHidden == nil)
            #expect(try f.mtp.evaluationTargets(state: copy).isEmpty)
            let other = try Fixture()
            let foreign = try other.mtp.newRequestState()
            #expect(throws: GLM5NextMTPError.incompatibleState) {
                try other.mtp.restore(checkpoint, into: foreign)
            }
            try f.mtp.release(state: copy)
            #expect(throws: GLM5NextMTPError.releasedState) { try f.mtp.reset(state: copy) }
        }

        @Test func invalidAppendLeavesNoPartialDraftHistory() throws {
            let f = try Fixture()
            let state = try f.mtp.newRequestState(maximumInputs: 2)
            #expect(throws: GLM5NextMTPError.self) {
                try f.mtp.prime(
                    shiftedTokens: tokens([2, 3]), targetHidden: rows(features(1)), at: 0,
                    state: state)
            }
            #expect(state.committedInputCount == 0)
            _ = try f.mtp.prime(
                shiftedTokens: tokens([2, 3]), targetHidden: rows(features(2)), at: 0, state: state)
            let saved = try f.mtp.evaluationTargets(state: state).map(values)
            #expect(throws: GLM5NextMTPError.self) {
                try f.mtp.prime(
                    shiftedTokens: tokens([4]), targetHidden: rows(features(1)), at: 1, state: state
                )
            }
            #expect(throws: GLM5NextMTPError.self) {
                try f.mtp.prime(
                    shiftedTokens: tokens([4]), targetHidden: rows(features(1)), at: 2, state: state
                )
            }
            #expect(try f.mtp.evaluationTargets(state: state).map(values) == saved)
        }

        @Test func adapterUsesShiftedHistoryAndRequiresTheCommonEvaluationFence() throws {
            let f = try Fixture()
            let factory = ReferenceFactory(f.mtp)
            let adapter = try GLM5NextMTPAdapter(
                target: f.target, predictor: f.mtp, factory: factory)
            let state = try #require(adapter.makeRequestState() as? GLM5NextMTPAdapter.State)
            try adapter.configureRequestState(state, maximumSequenceLength: 16)
            adapter.observeCommittedTarget(
                .init(tokens: tokens([1, 2, 3]), hidden: rows(features(3))), requestState: state)
            #expect(state.committedInputCount == 0)
            #expect(
                adapter.capturePrefixCheckpoint(requestState: state, targetInputCount: 3) == nil)
            try fence(adapter, state)
            #expect(state.committedInputCount == 2)
            let proposal = try adapter.draft(
                tokens: tokens([4]), hidden: rows([features(3)[2]]), state: state)
            #expect(factory.owners[0].committedInputCount == 0)
            #expect(throws: GLM5NextMTPError.self) {
                try adapter.finalize(
                    state: state, confirmedInputs: 1,
                    acceptedTokens: tokens([]), targetHidden: rows([]))
            }
            #expect(throws: GLM5NextMTPError.unsupportedDraftDepth(2)) {
                try adapter.draft(tokens: tokens([5]), hidden: proposal.hidden, state: state)
            }
            try fence(adapter, state)
            #expect(factory.owners[0].fedTokens!.asArray(Int32.self) == [2, 3, 4])
            adapter.finalizeRound(
                requestState: state, confirmedInputTokens: 1,
                committedDraftTokens: tokens([]), committedTargetHidden: rows([]))
            #expect(factory.owners[0].committedInputCount == 3)
            #expect(state.committedInputCount == 3 && state.stagedInputCount == 0)
            #expect(adapter.maximumDraftTokens == 1 && adapter.maximumSpeculativeBatch == 1)
            #expect(!adapter.supportsTargetPrefixAcceptance)
            // Test-only owner has no admission declaration. It cannot enable an engine path.
            #expect(
                adapter.boundedRequestAllocation(
                    limits: .init(maximumPrefillTokens: 8, maximumDraftTokens: 1)) == nil)
            adapter.releaseRequestState(state)
            #expect(factory.owners[0].closed)
        }

        @Test func adapterAcceptedTokensUseVerifiedHiddenAndHistoryCarryIsNotDuplicated() throws {
            let f = try Fixture()
            let factory = ReferenceFactory(f.mtp)
            let adapter = try GLM5NextMTPAdapter(
                target: f.target, predictor: f.mtp, factory: factory)
            let state = try #require(adapter.makeRequestState() as? GLM5NextMTPAdapter.State)
            try adapter.configureRequestState(state, maximumSequenceLength: 16)
            try adapter.observe(
                .init(tokens: tokens([1, 2, 3]), hidden: rows(features(3))), state: state)
            try fence(adapter, state)
            _ = try adapter.draft(tokens: tokens([4]), hidden: rows([features(3)[2]]), state: state)
            try fence(adapter, state)
            let seedHidden = [0.25, -0.75, 1.0, 0.5]
            try adapter.finalize(
                state: state, confirmedInputs: 2,
                acceptedTokens: tokens([5]), targetHidden: rows([seedHidden]))
            #expect(state.committedInputCount == 4)
            let draftHidden = [0.5, 0.25, -1.0, 0.75]
            let carryHidden = [-0.5, 0.5, 0.75, 1.0]
            // Exact Common ordering: history carry (token 6, hidden of token 5),
            // then ordinary observation (token 6, hidden of token 6).
            try adapter.observe(
                .init(tokens: tokens([6]), hidden: rows([draftHidden])), state: state)
            try adapter.observe(
                .init(tokens: tokens([6]), hidden: rows([carryHidden])), state: state)
            #expect(state.committedInputCount == 4)
            try fence(adapter, state)
            #expect(state.committedInputCount == 5)
            _ = try adapter.draft(tokens: tokens([7]), hidden: rows([carryHidden]), state: state)
            try fence(adapter, state)
            let owner = factory.owners[0]
            #expect(owner.fedTokens!.asArray(Int32.self) == [5, 6, 7])
            close(owner.fedHidden!, (seedHidden + draftHidden + carryHidden))
            try adapter.discard(state: state)
            #expect(owner.committedInputCount == 3 && state.committedInputCount == 5)
            _ = try adapter.draft(tokens: tokens([7]), hidden: rows([carryHidden]), state: state)
            try fence(adapter, state)
            #expect(owner.fedTokens!.asArray(Int32.self) == [5, 6, 7])
            try adapter.discard(state: state)
            adapter.releaseRequestState(state)
        }

        @Test func adapterEvaluationFailureCannotPublishCandidateOrPrefix() throws {
            let f = try Fixture()
            let factory = ReferenceFactory(f.mtp)
            let adapter = try GLM5NextMTPAdapter(
                target: f.target, predictor: f.mtp, factory: factory)
            let state = try #require(adapter.makeRequestState() as? GLM5NextMTPAdapter.State)
            try adapter.configureRequestState(state, maximumSequenceLength: 16)
            try adapter.observe(
                .init(tokens: tokens([1, 2]), hidden: rows(features(2))), state: state)
            try fence(adapter, state)
            _ = try adapter.draft(tokens: tokens([3]), hidden: rows([features(2)[1]]), state: state)
            #expect(
                adapter.capturePrefixCheckpoint(requestState: state, targetInputCount: 2) == nil)
            factory.owners[0].failEvaluation = true
            #expect(throws: GLM5NextMTPError.self) { try fence(adapter, state) }
            #expect(factory.owners[0].committedInputCount == 0)
            #expect(state.committedInputCount == 1)
            #expect(throws: GLM5NextMTPError.self) {
                try adapter.finalize(
                    state: state, confirmedInputs: 1,
                    acceptedTokens: tokens([]), targetHidden: rows([]))
            }
            try adapter.discard(state: state)
            adapter.releaseRequestState(state)
        }

        @Test func admittedConcreteOwnerRunsTrainedMathWithoutAnExtraReservation() throws {
            let h = try AdmittedHarness()
            let owner = try h.factory.makeOwner(admission: h.context)
            let incremental = try #require(owner as? any GLM5NextMTPIncrementalExecutionOwner)
            let initial = h.admission.bytesReserved
            let empty = try owner.checkpoint()
            #expect(empty.inputCount == 0 && empty.evaluationTargets.isEmpty)
            let beforeRejectedStage = h.context.allocatedBytes
            #expect(throws: GLM5NextMTPError.self) {
                try incremental.stageHistory(
                    (0 ..< 4).map { _ in
                        .init(shiftedTokens: tokens([2]), targetHidden: rows(features(1)))
                    })
            }
            #expect(h.context.allocatedBytes == beforeRejectedStage)
            try incremental.stageHistory([
                .init(shiftedTokens: tokens([2, 3]), targetHidden: rows(features(2)))
            ])
            #expect(owner.committedInputCount == 0)
            try withError { fault in
                eval(owner.evaluationTargets)
                try fault.check()
            }
            try owner.didFinishEvaluation()
            #expect(owner.committedInputCount == 2)
            #expect(throws: GLM5NextMTPError.self) {
                try owner.prepare(shiftedTokens: tokens([4, 5]), targetHidden: rows(features(2)))
            }
            let prepared = try owner.prepare(
                shiftedTokens: tokens([4]), targetHidden: rows([features(3)[2]]))
            #expect(owner.committedInputCount == 2 && h.admission.bytesReserved == initial)
            #expect(throws: GLM5NextMTPError.self) { try prepared.commit() }
            try withError { fault in
                eval(prepared.evaluationTargets)
                try fault.check()
            }
            try owner.didFinishEvaluation()
            try prepared.didFinishEvaluation()
            let oracle = reference(shifted: [2, 3, 4], hidden: features(3))
            close(prepared.output.logits, oracle.logits.last!)
            try prepared.commit()
            #expect(owner.committedInputCount == 3)
            #expect(h.admission.bytesReserved == initial)
            #expect(throws: GLM5NextMTPError.self) { try prepared.commit() }
            let rejected = try owner.prepare(
                shiftedTokens: tokens([5]), targetHidden: rows(features(1)))
            try rejected.discard()
            #expect(owner.committedInputCount == 3)
            #expect(throws: GLM5NextMTPError.self) { try rejected.commit() }
            try owner.didFinishEvaluation()
            try owner.restore(empty)
            #expect(owner.committedInputCount == 0)
            try owner.close()
            h.admission.releaseAll(id: h.id)
        }

        @Test func admittedBindingRejectsWrongIdentityStaleGenerationAndExcessCapacity() throws {
            let h = try AdmittedHarness()
            #expect(throws: CBv2MTPBindingError.identityMismatch) {
                try h.context.validate(
                    target: ObjectIdentifier(h.factory),
                    drafter: ObjectIdentifier(h.adapter), maximumInputs: 8)
            }
            #expect(throws: CBv2MTPBindingError.identityMismatch) {
                try h.context.validate(
                    target: h.fixture.mtp.targetIdentity,
                    drafter: ObjectIdentifier(h.factory), maximumInputs: 8)
            }
            #expect(throws: CBv2MTPBindingError.changedLimit) {
                try h.context.validate(
                    target: h.fixture.mtp.targetIdentity,
                    drafter: ObjectIdentifier(h.adapter), maximumInputs: 9)
            }
            #expect(throws: (any Error).self) {
                try h.context.reserve(.resident, buffers: [.init(logicalBytes: Int.max)])
            }
            #expect(throws: CBv2MTPBindingError.capacityExceeded) {
                try h.context.reserve(
                    .resident,
                    buffers: [.init(logicalBytes: h.context.resolved.residentBytes + 1)])
            }
            var loan: CBv2MTPAllocationLease? = try h.context.reserve(
                .resident,
                buffers: [.init(logicalBytes: 64)])
            let charged = h.admission.bytesReserved
            h.admission.releaseAll(id: h.id)
            #expect(h.admission.bytesReserved == charged)
            #expect(throws: CBv2MTPBindingError.staleRequest) {
                try h.context.validate(
                    target: h.fixture.mtp.targetIdentity,
                    drafter: ObjectIdentifier(h.adapter), maximumInputs: 8)
            }
            try h.admission.reserve(id: h.id, additionalTokens: 1)
            let next = try h.issuer.context(id: h.id, maximumInputs: 8)
            #expect(next.requestGeneration != h.context.requestGeneration)
            #expect(h.admission.bytesReserved == 2 * charged)
            loan = nil
            #expect(loan == nil && h.admission.bytesReserved == charged)
            h.admission.releaseAll(id: h.id)
            #expect(h.admission.bytesReserved == 0 && h.process.bytes == 0)
        }

        @Test func admittedArrayLeaseOutlivesCancelledOwnerWithoutRefundingItsBacking() throws {
            let h = try AdmittedHarness()
            let owner = try h.factory.makeOwner(admission: h.context)
            var held: (any GLM5NextMTPArrayLease)? = try owner.hold([rows(features(2))])
            try withError { fault in
                eval(held!.arrays)
                try fault.check()
            }
            try owner.didFinishEvaluation()
            let charged = h.admission.bytesReserved
            h.admission.releaseAll(id: h.id)
            #expect(throws: CBv2MTPBindingError.staleRequest) {
                try owner.prepare(shiftedTokens: tokens([2]), targetHidden: rows(features(1)))
            }
            try owner.close()
            #expect(h.admission.bytesReserved == charged && held != nil)
            held = nil
            #expect(h.admission.bytesReserved == 0 && h.process.bytes == 0)
        }

        @Test func bothDriverConfigurationPathsUseTheSameIssuedRequestGeneration() throws {
            let h = try AdmittedHarness()
            let driver = try #require(
                CBv2MTPRoundDriver.build(
                    model: CBv2SteppableLanguageModelAdapter(h.fixture.target), drafter: h.adapter,
                    config: .init(
                        enabled: true, maxDraftTokens: 1, maxSpeculativeBatch: 1,
                        fixedDraftTokens: 1, verificationMode: .serialTarget)))
            driver.admittedMTPBindingIssuer = h.issuer
            let preparedState = try driver.takeOrMakeAssistantState(
                for: h.id, maximumSequenceLength: 8)
            let state = try #require(preparedState)
            try driver.configureAssistantState(state, id: h.id, maximumSequenceLength: 8)
            #expect(throws: CBv2MTPBindingError.changedLimit) {
                try driver.configureAssistantState(state, id: h.id, maximumSequenceLength: 9)
            }
            h.admission.releaseAll(id: h.id)
            try h.admission.reserve(id: h.id, additionalTokens: 1)
            #expect(throws: CBv2MTPBindingError.staleRequest) {
                try driver.configureAssistantState(state, id: h.id, maximumSequenceLength: 8)
            }
            h.adapter.releaseRequestState(state)
            h.admission.releaseAll(id: h.id)
            #expect(h.admission.bytesReserved == 0)
        }

        @Test func admittedAdapterPublishesOnlyAfterItsRealFenceAndRetainsAliasCharges() throws {
            let h = try AdmittedHarness()
            let state = try #require(h.adapter.makeRequestState() as? GLM5NextMTPAdapter.State)
            #expect(throws: CBv2MTPBindingError.missingAdmission) {
                try h.adapter.configureRequestState(state, maximumSequenceLength: 8)
            }
            try h.adapter.configureRequestState(state, admission: h.context)
            try h.adapter.observe(
                .init(tokens: tokens([1, 2, 3]), hidden: rows(features(3))), state: state)
            #expect(state.committedInputCount == 0)
            try fence(h.adapter, state)
            #expect(state.committedInputCount == 2)
            _ = try h.adapter.draft(
                tokens: tokens([4]), hidden: rows([features(3)[2]]), state: state)
            #expect(throws: GLM5NextMTPError.self) {
                try h.adapter.finalize(
                    state: state, confirmedInputs: 1,
                    acceptedTokens: tokens([]), targetHidden: rows([]))
            }
            try fence(h.adapter, state)
            try h.adapter.finalize(
                state: state, confirmedInputs: 1,
                acceptedTokens: tokens([]), targetHidden: rows([]))
            #expect(state.committedInputCount == 3)
            // The actual prepaid amount remains owned until deferred state/alias retirement.
            let charged = h.admission.bytesReserved
            h.admission.releaseAll(id: h.id)
            #expect(h.admission.bytesReserved == charged)
            h.adapter.releaseRequestState(state)
            #expect(h.admission.bytesReserved == 0 && h.process.bytes == 0)
        }

        @Test func concreteOwnerCountsHeldResidencyOnceAcrossObservedGenerations() throws {
            let h = try AdmittedHarness()
            let state = try #require(h.adapter.makeRequestState() as? GLM5NextMTPAdapter.State)
            defer {
                h.adapter.releaseRequestState(state)
                h.admission.releaseAll(id: h.id)
            }
            try h.adapter.configureRequestState(state, admission: h.context)
            let prepaid = h.admission.bytesReserved
            let declaration = try #require(
                h.adapter.boundedRequestAllocation(limits: h.context.resolved.limits))
            try h.adapter.observe(
                .init(tokens: tokens([1, 2, 3]), hidden: rows(features(3))), state: state)
            #expect(state.hasUnmeasuredResidency)
            try fence(h.adapter, state)
            #expect(!state.hasUnmeasuredResidency)
            let first = try deduplicatedMeasuredBytes(h.adapter, state)
            #expect(first.bytes > 0 && first.duplicateRoots > 0)
            #expect(state.materializedBytes == first.bytes)
            #expect(h.admission.bytesReserved == prepaid)

            try h.adapter.observe(
                .init(tokens: tokens([4]), hidden: rows(features(1))), state: state)
            #expect(state.hasUnmeasuredResidency)
            try fence(h.adapter, state)
            #expect(!state.hasUnmeasuredResidency)
            let second = try deduplicatedMeasuredBytes(h.adapter, state)
            #expect(second.bytes > 0 && second.duplicateRoots > 0)
            #expect(state.materializedBytes == second.bytes)
            #expect(h.admission.bytesReserved == prepaid)
            let unchanged = try #require(
                h.adapter.boundedRequestAllocation(limits: h.context.resolved.limits))
            #expect(unchanged.resident == declaration.resident)
            #expect(unchanged.working == declaration.working)
            #expect(unchanged.hostBytes == declaration.hostBytes)
        }

        @Test func restoredPrefixStillCountsInputLeaseBorrowedFromAnotherConcreteOwner() throws {
            let h = try AdmittedHarness()
            let state = try #require(h.adapter.makeRequestState() as? GLM5NextMTPAdapter.State)
            let restoredID = CBv2RequestID(712)
            defer {
                h.adapter.releaseRequestState(state)
                h.admission.releaseAll(id: h.id)
                h.admission.releaseAll(id: restoredID)
            }
            try h.adapter.configureRequestState(state, admission: h.context)
            try h.adapter.observe(
                .init(tokens: tokens([1, 2, 3]), hidden: rows(features(3))), state: state)
            try fence(h.adapter, state)
            do {
                let prefix = try #require(
                    h.adapter.capturePrefixCheckpoint(requestState: state, targetInputCount: 3))
                let restored = try #require(
                    h.adapter.restorePrefixCheckpoint(prefix) as? GLM5NextMTPAdapter.State)
                defer { h.adapter.releaseRequestState(restored) }
                try h.admission.reserve(id: restoredID, additionalTokens: 1)
                let context = try h.issuer.context(id: restoredID, maximumInputs: 8)
                try h.adapter.configureRequestState(restored, admission: context)
                #expect(!restored.hasUnmeasuredResidency)
                let measured = try deduplicatedMeasuredBytes(h.adapter, restored)
                #expect(measured.bytes > 0)
                #expect(restored.materializedBytes == measured.bytes)
                #expect(restored.materializedBytes == state.materializedBytes)
            }
        }

        private func deduplicatedMeasuredBytes(
            _ adapter: GLM5NextMTPAdapter, _ state: GLM5NextMTPAdapter.State
        ) throws -> (bytes: Int, duplicateRoots: Int) {
            let roots = adapter.evaluationTargets(for: state)
            var seen = Set<ObjectIdentifier>()
            var bytes = 0
            var duplicates = 0
            for root in roots where root.size > 0 {
                guard seen.insert(ObjectIdentifier(root)).inserted else {
                    duplicates += 1
                    continue
                }
                let evaluated = try root.evaluatedBufferInfo()
                let buffer = try #require(evaluated)
                // Distinct wrappers must not secretly share backing in this
                // oracle. Repeated wrappers are the adapter/owner overlap.
                #expect(buffer.isUnique && buffer.isRowContiguous)
                #expect(buffer.dataOffset == 0 && buffer.dataElements == root.size)
                bytes += buffer.allocatedBytes
            }
            return (bytes, duplicates)
        }

        @Test func retiredAdmissionCanDrainButCannotPublishPrimedHistory() throws {
            let h = try AdmittedHarness()
            let state = try #require(h.adapter.makeRequestState() as? GLM5NextMTPAdapter.State)
            try h.adapter.configureRequestState(state, admission: h.context)
            try h.adapter.observe(
                .init(tokens: tokens([1, 2, 3]), hidden: rows(features(3))), state: state)
            try withError { fault in
                eval(h.adapter.evaluationTargets(for: state))
                try fault.check()
            }
            let charged = h.admission.bytesReserved
            h.admission.releaseAll(id: h.id)
            #expect(!h.context.isActive && h.admission.bytesReserved == charged)
            try h.adapter.fenceRequestStateForNativeCompletion(state)
            try h.adapter.commitRequestStateNativeCompletion(state)
            #expect(state.committedInputCount == 0)
            #expect(h.admission.bytesReserved == charged)
            #expect(throws: CBv2MTPBindingError.staleRequest) {
                try h.adapter.draft(tokens: tokens([4]), hidden: rows(features(1)), state: state)
            }
            h.adapter.releaseRequestState(state)
            #expect(h.admission.bytesReserved == 0 && h.process.bytes == 0)
        }

        @Test func serializedPrefixRestoresExactFencedStateAndIndependentContinuation() throws {
            let h = try AdmittedHarness()
            let destinationID = CBv2RequestID(713)
            do {
                let source = try #require(h.adapter.makeRequestState() as? GLM5NextMTPAdapter.State)
                defer { h.adapter.releaseRequestState(source) }
                try h.adapter.configureRequestState(source, admission: h.context)
                let prompt = [1, 2, 3]
                let target = try h.fixture.target.forward(tokens: tokens(prompt), cache: nil)
                try h.adapter.observe(
                    .init(tokens: tokens(prompt), hidden: target.hidden), state: source)
                #expect(
                    h.adapter.capturePrefixCheckpoint(requestState: source, targetInputCount: 3)
                        == nil)
                try fence(h.adapter, source)
                let captured = try #require(
                    h.adapter.capturePrefixCheckpoint(
                        requestState: source, targetInputCount: 3))
                let wire = try #require(h.adapter.encodePrefixCheckpoint(captured))
                let descriptors = try #require(
                    h.adapter.prefixCheckpointTensorDescriptors(targetInputCount: 3))
                #expect(h.adapter.prefixCheckpointCodecID == "glm5next-mtp-contiguous-prefix-v1")
                #expect(
                    descriptors.map(\.shape) == [
                        [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2], [1, 1, 4], [2],
                    ])
                #expect(
                    descriptors.map(\.dtype) == [
                        .float32, .float32, .float32, .uint8, .float32, .int32,
                    ])
                #expect(wire[3].asArray(UInt8.self) == [1, 1])
                #expect(wire[5].asArray(Int32.self) == [3, 2])
                #expect(wire[4].asData().data == target.hidden[0..., (-1)..., 0...].asData().data)
                let originalBytes = wire.map { $0.asData().data }
                let decoded = try #require(
                    h.adapter.decodePrefixCheckpoint(tensors: wire, prefixTokens: prompt))
                #expect(h.adapter.encodePrefixCheckpoint(decoded) == nil)
                let restored = try #require(
                    h.adapter.restorePrefixCheckpoint(decoded) as? GLM5NextMTPAdapter.State)
                defer { h.adapter.releaseRequestState(restored) }
                #expect(restored.committedInputCount == 0 && restored.hasUnmeasuredResidency)
                #expect(throws: CBv2MTPBindingError.missingAdmission) {
                    try h.adapter.configureRequestState(restored, maximumSequenceLength: 8)
                }
                try h.admission.reserve(id: destinationID, additionalTokens: 1)
                let context = try h.issuer.context(id: destinationID, maximumInputs: 8)
                try h.adapter.configureRequestState(restored, admission: context)
                #expect(restored.committedInputCount == 0 && restored.hasUnmeasuredResidency)
                #expect(context.allocatedBytes > 0)
                try fence(h.adapter, restored)
                #expect(restored.committedInputCount == 2 && !restored.hasUnmeasuredResidency)
                let recaptured = try #require(
                    h.adapter.capturePrefixCheckpoint(requestState: restored, targetInputCount: 3))
                let rewritten = try #require(h.adapter.encodePrefixCheckpoint(recaptured))
                #expect(rewritten.map { $0.asData().data } == originalBytes)

                let hidden = target.hidden[0..., (-1)..., 0...]
                let coldDraft = try h.adapter.draft(
                    tokens: tokens([4]), hidden: hidden, state: source)
                let restoredDraft = try h.adapter.draft(
                    tokens: tokens([4]), hidden: hidden, state: restored)
                try fence(h.adapter, source)
                try fence(h.adapter, restored)
                #expect(
                    coldDraft.tokens.asArray(Int32.self) == restoredDraft.tokens.asArray(Int32.self)
                )
                #expect(coldDraft.hidden.asData().data == restoredDraft.hidden.asData().data)
                try h.adapter.discard(state: source)
                try h.adapter.discard(state: restored)
                let next = try h.fixture.target.forward(tokens: tokens([1, 2, 3, 4]), cache: nil)
                try h.adapter.observe(
                    .init(tokens: tokens([4]), hidden: next.hidden[0..., (-1)..., 0...]),
                    state: restored)
                try fence(h.adapter, restored)
                #expect(restored.committedInputCount == 3 && source.committedInputCount == 2)
                #expect(wire.map { $0.asData().data } == originalBytes)
            }
            h.admission.releaseAll(id: h.id)
            h.admission.releaseAll(id: destinationID)
            #expect(h.admission.bytesReserved == 0 && h.process.bytes == 0)
        }

        @Test func serializedPrefixRejectsMalformedFrontiersDtypesShapesAndForeignOwners() throws {
            let h = try AdmittedHarness()
            do {
                let source = try #require(h.adapter.makeRequestState() as? GLM5NextMTPAdapter.State)
                defer { h.adapter.releaseRequestState(source) }
                try h.adapter.configureRequestState(source, admission: h.context)
                try h.adapter.observe(
                    .init(tokens: tokens([1, 2, 3]), hidden: rows(features(3))), state: source)
                try fence(h.adapter, source)
                let captured = try #require(
                    h.adapter.capturePrefixCheckpoint(requestState: source, targetInputCount: 3))
                let wire = try #require(h.adapter.encodePrefixCheckpoint(captured))
                #expect(h.adapter.prefixCheckpointTensorDescriptors(targetInputCount: 1) == nil)
                #expect(h.adapter.prefixCheckpointTensorDescriptors(targetInputCount: 9) == nil)
                #expect(
                    h.adapter.decodePrefixCheckpoint(
                        tensors: Array(wire.dropLast()), prefixTokens: [1, 2, 3]) == nil)
                for (index, replacement) in [
                    (0, wire[0].asType(.float16)),
                    (1, wire[1][0..., ..<1, 0...]),
                    (3, MLXArray([UInt8(1), 2], [1, 2])),
                    (4, wire[4].asType(.bfloat16)),
                    (5, MLXArray([Int32(3), 3])),
                    (5, MLXArray([Int32(4), 3])),
                    (5, MLXArray([Int32(3), -1])),
                ] {
                    var bad = wire
                    bad[index] = replacement
                    eval(replacement)
                    #expect(
                        h.adapter.decodePrefixCheckpoint(tensors: bad, prefixTokens: [1, 2, 3])
                            == nil)
                }
                var lazy = wire
                lazy[0] = wire[0] + Float(1)
                #expect(try lazy[0].evaluatedBufferInfo() == nil)
                #expect(
                    h.adapter.decodePrefixCheckpoint(tensors: lazy, prefixTokens: [1, 2, 3]) == nil)
                #expect(try lazy[0].evaluatedBufferInfo() == nil)
                #expect(
                    h.adapter.decodePrefixCheckpoint(tensors: wire, prefixTokens: [1, 2, 16]) == nil
                )
                #expect(
                    h.adapter.decodePrefixCheckpoint(tensors: wire, prefixTokens: [1, -2, 3]) == nil
                )
                let foreign = try AdmittedHarness()
                #expect(foreign.adapter.encodePrefixCheckpoint(captured) == nil)
                #expect(foreign.adapter.restorePrefixCheckpoint(captured) == nil)
                foreign.admission.releaseAll(id: foreign.id)

                let decoded = try #require(
                    h.adapter.decodePrefixCheckpoint(tensors: wire, prefixTokens: [1, 2, 3]))
                let tooShort = try #require(
                    h.adapter.restorePrefixCheckpoint(decoded) as? GLM5NextMTPAdapter.State)
                let shortID = CBv2RequestID(715)
                try h.admission.reserve(id: shortID, additionalTokens: 1)
                let shortContext = try h.issuer.context(id: shortID, maximumInputs: 2)
                #expect(throws: GLM5NextMTPError.incompatibleState) {
                    try h.adapter.configureRequestState(tooShort, admission: shortContext)
                }
                #expect(tooShort.committedInputCount == 0)
                h.adapter.releaseRequestState(tooShort)
                h.admission.releaseAll(id: shortID)

                _ = try h.adapter.draft(
                    tokens: tokens([4]), hidden: rows(features(1)), state: source)
                try fence(h.adapter, source)
                try h.adapter.finalize(
                    state: source, confirmedInputs: 2,
                    acceptedTokens: tokens([5]), targetHidden: rows(features(1)))
                #expect(
                    h.adapter.capturePrefixCheckpoint(requestState: source, targetInputCount: 5)
                        == nil)
            }
            h.admission.releaseAll(id: h.id)
            #expect(h.admission.bytesReserved == 0)
        }

        @Test func serializedPrefixPreservesActualBFloat16Payloads() throws {
            let h = try AdmittedHarness(activationDType: .bfloat16)
            do {
                let state = try #require(h.adapter.makeRequestState() as? GLM5NextMTPAdapter.State)
                defer { h.adapter.releaseRequestState(state) }
                try h.adapter.configureRequestState(state, admission: h.context)
                let target = try h.fixture.target.forward(tokens: tokens([1, 2, 3]), cache: nil)
                try h.adapter.observe(
                    .init(tokens: tokens([1, 2, 3]), hidden: target.hidden), state: state)
                try fence(h.adapter, state)
                let prefix = try #require(
                    h.adapter.capturePrefixCheckpoint(requestState: state, targetInputCount: 3))
                let wire = try #require(h.adapter.encodePrefixCheckpoint(prefix))
                #expect(
                    wire.map(\.dtype) == [
                        .bfloat16, .bfloat16, .bfloat16, .uint8, .bfloat16, .int32,
                    ])
                #expect(wire[4].asData().data == target.hidden[0..., (-1)..., 0...].asData().data)
                let decoded = try #require(
                    h.adapter.decodePrefixCheckpoint(tensors: wire, prefixTokens: [1, 2, 3]))
                #expect(decoded.targetInputCount == 3)
            }
            h.admission.releaseAll(id: h.id)
            #expect(h.admission.bytesReserved == 0)
        }

        @Test func cancelledDecodedPrefixNeverPublishesAndRetainsCopiesUntilFence() throws {
            let h = try AdmittedHarness()
            let destinationID = CBv2RequestID(714)
            do {
                let source = try #require(h.adapter.makeRequestState() as? GLM5NextMTPAdapter.State)
                defer { h.adapter.releaseRequestState(source) }
                try h.adapter.configureRequestState(source, admission: h.context)
                try h.adapter.observe(
                    .init(tokens: tokens([1, 2, 3]), hidden: rows(features(3))), state: source)
                try fence(h.adapter, source)
                let prefix = try #require(
                    h.adapter.capturePrefixCheckpoint(requestState: source, targetInputCount: 3))
                let wire = try #require(h.adapter.encodePrefixCheckpoint(prefix))
                let decoded = try #require(
                    h.adapter.decodePrefixCheckpoint(tensors: wire, prefixTokens: [1, 2, 3]))
                let cold = try #require(
                    h.adapter.restorePrefixCheckpoint(decoded) as? GLM5NextMTPAdapter.State)
                let allocated = h.context.allocatedBytes
                h.adapter.releaseRequestState(cold)
                #expect(h.context.allocatedBytes == allocated)
                let restored = try #require(
                    h.adapter.restorePrefixCheckpoint(decoded) as? GLM5NextMTPAdapter.State)
                try h.admission.reserve(id: destinationID, additionalTokens: 1)
                let context = try h.issuer.context(id: destinationID, maximumInputs: 8)
                try h.adapter.configureRequestState(restored, admission: context)
                let charged = h.admission.bytesReserved
                h.admission.releaseAll(id: destinationID)
                #expect(h.admission.bytesReserved == charged && !context.isActive)
                try fence(h.adapter, restored)
                #expect(restored.committedInputCount == 0)
                h.adapter.releaseRequestState(restored)
                #expect(h.admission.bytesReserved < charged)
                #expect(source.committedInputCount == 2)
            }
            h.admission.releaseAll(id: h.id)
            #expect(h.admission.bytesReserved == 0 && h.process.bytes == 0)
        }

        @Test func nativePublicationRequiresItsSeparateCompletionFence() throws {
            let h = try AdmittedHarness()
            let state = try #require(h.adapter.makeRequestState() as? GLM5NextMTPAdapter.State)
            try h.adapter.configureRequestState(state, admission: h.context)
            try h.adapter.observe(
                .init(tokens: tokens([1, 2, 3]), hidden: rows(features(3))), state: state)
            #expect(throws: GLM5NextMTPError.self) {
                try h.adapter.commitRequestStateNativeCompletion(state)
            }
            #expect(state.committedInputCount == 0)
            try withError { fault in
                eval(h.adapter.evaluationTargets(for: state))
                try fault.check()
            }
            try h.adapter.fenceRequestStateForNativeCompletion(state)
            #expect(state.committedInputCount == 0)
            try h.adapter.commitRequestStateNativeCompletion(state)
            #expect(state.committedInputCount == 2)
            h.adapter.releaseRequestState(state)
            h.admission.releaseAll(id: h.id)
            #expect(h.admission.bytesReserved == 0 && h.process.bytes == 0)
        }

        @Test func boundedObservationPrimingConsumesEveryShiftedRowWithoutReadout() throws {
            let h = try AdmittedHarness(maximumInputs: 16, maximumPrefillTokens: 2)
            let state = try #require(h.adapter.makeRequestState() as? GLM5NextMTPAdapter.State)
            try h.adapter.configureRequestState(state, admission: h.context)
            let hidden = features(5)
            let headCalls = h.fixture.head.calls
            var consumed = 0
            for range in [0 ..< 2, 2 ..< 4, 4 ..< 5] {
                try h.adapter.observe(
                    .init(
                        tokens: tokens(range.map { $0 + 1 }),
                        hidden: rows(Array(hidden[range]))), state: state)
                #expect(state.committedInputCount == consumed)
                try fence(h.adapter, state)
                consumed = range.upperBound - 1
                #expect(state.committedInputCount == consumed)
                #expect(h.fixture.head.calls == headCalls)
            }
            let allocationBefore = h.context.allocatedBytes
            #expect(throws: GLM5NextMTPError.self) {
                try h.adapter.observe(
                    .init(tokens: tokens([6, 7, 8]), hidden: rows(features(3))), state: state)
            }
            #expect(h.context.allocatedBytes == allocationBefore && state.committedInputCount == 4)
            let result = try h.adapter.draft(
                tokens: tokens([6]), hidden: rows([hidden[4]]), state: state)
            try fence(h.adapter, state)
            let oracle = reference(shifted: [2, 3, 4, 5, 6], hidden: hidden)
            close(result.hidden, oracle.hidden.last!)
            #expect(h.fixture.head.calls == headCalls + 1)
            try h.adapter.finalize(
                state: state, confirmedInputs: 1,
                acceptedTokens: tokens([]), targetHidden: rows([]))
            #expect(state.committedInputCount == 5)
            h.adapter.releaseRequestState(state)
            h.admission.releaseAll(id: h.id)
        }

        @Test func acceptedHistoryCanPrecedeOneQueryAndSurvivesDiscardBeforeFence() throws {
            let h = try AdmittedHarness(maximumInputs: 16, maximumPrefillTokens: 2)
            let state = try #require(h.adapter.makeRequestState() as? GLM5NextMTPAdapter.State)
            try h.adapter.configureRequestState(state, admission: h.context)
            let hidden = features(4)
            try h.adapter.observe(
                .init(tokens: tokens([1, 2]), hidden: rows(Array(hidden.prefix(2)))), state: state)
            try fence(h.adapter, state)
            _ = try h.adapter.draft(tokens: tokens([3]), hidden: rows([hidden[1]]), state: state)
            try fence(h.adapter, state)
            try h.adapter.finalize(
                state: state, confirmedInputs: 2,
                acceptedTokens: tokens([4]), targetHidden: rows([hidden[2]]))
            #expect(state.committedInputCount == 2)
            _ = try h.adapter.draft(tokens: tokens([5]), hidden: rows([hidden[3]]), state: state)
            try h.adapter.discard(state: state)
            try fence(h.adapter, state)
            // Accepted token 4 was trusted history even though the following
            // query was discarded. It must not be appended a second time.
            #expect(state.committedInputCount == 3)
            let replay = try h.adapter.draft(
                tokens: tokens([5]), hidden: rows([hidden[3]]), state: state)
            try fence(h.adapter, state)
            let oracle = reference(shifted: [2, 3, 4, 5], hidden: hidden)
            close(replay.hidden, oracle.hidden.last!)
            try h.adapter.finalize(
                state: state, confirmedInputs: 1,
                acceptedTokens: tokens([]), targetHidden: rows([]))
            #expect(state.committedInputCount == 4)
            h.adapter.releaseRequestState(state)
            h.admission.releaseAll(id: h.id)
        }

        @Test func boundedCarryAndOrdinaryObservationPrimeThreeChunksWithoutDuplication() throws {
            let h = try AdmittedHarness(maximumInputs: 16, maximumPrefillTokens: 2)
            let state = try #require(h.adapter.makeRequestState() as? GLM5NextMTPAdapter.State)
            try h.adapter.configureRequestState(state, admission: h.context)
            let hidden = features(6)
            try h.adapter.observe(
                .init(tokens: tokens([1, 2]), hidden: rows(Array(hidden.prefix(2)))), state: state)
            try fence(h.adapter, state)
            _ = try h.adapter.draft(tokens: tokens([3]), hidden: rows([hidden[1]]), state: state)
            try fence(h.adapter, state)
            try h.adapter.finalize(
                state: state, confirmedInputs: 2,
                acceptedTokens: tokens([4]), targetHidden: rows([hidden[2]]))
            let calls = h.fixture.head.calls
            try h.adapter.observe(
                .init(tokens: tokens([5]), hidden: rows([hidden[3]])), state: state)
            try h.adapter.observe(
                .init(tokens: tokens([5, 6]), hidden: rows(Array(hidden[4 ..< 6]))), state: state)
            try fence(h.adapter, state)
            #expect(state.committedInputCount == 5 && h.fixture.head.calls == calls)
            let draft = try h.adapter.draft(
                tokens: tokens([7]), hidden: rows([hidden[5]]), state: state)
            try fence(h.adapter, state)
            let oracle = reference(shifted: [2, 3, 4, 5, 6, 7], hidden: hidden)
            close(draft.hidden, oracle.hidden.last!)
            try h.adapter.discard(state: state)
            h.adapter.releaseRequestState(state)
            h.admission.releaseAll(id: h.id)
        }

        @Test func officialAllocationReproductionUsesBoundedQueriesAt512And4096() throws {
            let c = try officialMemoryConfiguration()
            let limits = CBv2MTPAllocationLimits(maximumPrefillTokens: 128, maximumDraftTokens: 1)
            func plan(_ n: Int) throws -> GLM5NextMTPExecution.MemoryPlan {
                try GLM5NextMTPExecution.memoryPlan(
                    configuration: c, maximumInputs: n, limits: limits)
            }
            func bytes(_ items: [CBv2MTPFixedBufferSpec]) -> Int {
                items.reduce(0) { $0 + $1.logicalBytes * $1.allocationCount }
            }
            let small = try plan(512)
            let middle = try plan(2304)
            let large = try plan(4096)
            #expect(small.maximumObservationTokens == 128 && large.maximumObservationTokens == 128)
            #expect(small.maximumDraftQueryTokens == 1 && large.maximumDraftQueryTokens == 1)
            #expect(
                bytes(small.draftWorkBuffers) + bytes(large.draftWorkBuffers) == 2
                    * bytes(middle.draftWorkBuffers))
            #expect(
                bytes(small.historyWorkBuffers) + bytes(large.historyWorkBuffers) == 2
                    * bytes(middle.historyWorkBuffers))
            // The largest selected projection is one query at both contexts.
            #expect(
                small.draftWorkBuffers.map(\.logicalBytes).max()
                    == large.draftWorkBuffers.map(\.logicalBytes).max())
            #expect(large.specification.hostBytes > small.specification.hostBytes)
        }

        @Test(arguments: [1.0, -1.0])
        func protectedPagedEngineUsesOneRealProposalAndMatchesTargetGreedy(projectionSign: Double)
            async throws
        {
            let h = try NativeHarness(projectionSign: projectionSign)
            let tracking = try #require(h.engine.loopForTesting.nativeShutdownState)
            var observedPrimedHistory = false
            h.engine.loopForTesting.onEngineQueueSync {
                tracking.afterAssistantFenceForTesting = {
                    [weak tracking, process = h.process, assistant = h.resources.assistant] value in
                    guard let state = value as? GLM5NextMTPAdapter.State else {
                        Issue.record("native GLM fence received a foreign assistant")
                        return
                    }
                    #expect((tracking?.debugRetainedRootCount ?? 0) > 0)
                    #expect(process.bytes > 0)
                    // A bounded prompt chunk has real staged roots before its
                    // completion metadata may advance the published frontier.
                    observedPrimedHistory =
                        observedPrimedHistory
                        || (state.committedInputCount > 0
                            && !assistant.evaluationTargets(for: state).isEmpty)
                }
            }
            let prompt = [1, 2, 3, 4]
            let expected = try greedy(h.fixture.target, prompt: prompt, count: 6)
            let plain = try await targetOnlyPagedEngine(h.fixture.target, prompt: prompt, count: 6)
            let stream = try h.engine.submit(
                .init(
                    id: .init(1901), promptTokens: prompt,
                    sampling: .init(temperature: 0), maxTokens: 6))
            var actual: [Int] = []
            for await event in stream {
                if case .delta(_, let values, _) = event { actual += values }
            }
            let outcome = await h.engine.shutdownReportingNativeCompletion()
            #expect(actual == expected && actual == plain)
            #expect(observedPrimedHistory)
            let metrics = try #require(h.engine.mtpMetricsSnapshot())
            #expect(metrics.proposedTokens > 0 && metrics.serialVerificationRounds > 0)
            if projectionSign < 0 {
                #expect(metrics.acceptedTokens == 0)
            } else {
                #expect(metrics.acceptedTokens > 0)
            }
            guard case .quiescent(let receipt) = outcome else {
                Issue.record("protected GLM native shutdown did not complete: \(outcome)")
                return
            }
            #expect(receipt.executionContractID == h.resources.contract.id)
            #expect(h.resources.contract.profile == "glm5_next_native_paged_serial_mtp_v1")
            #expect(!h.resources.contract.supportsNativeCompletePrefix)
            #expect(tracking.debugRetainedRootCount == 0)
            #expect(h.resources.backend.bytesInUse == 0 && h.resources.backend.bytesReserved == 0)
            #expect(h.engine.loopForTesting.recurrentStates.isEmpty)
            #expect(h.process.bytes == 0)
        }

        @Test func protectedPagedEngineInterleavesRequestsAndCancelsWithoutLeakingOwners()
            async throws
        {
            let h = try NativeHarness(projectionSign: 1)
            let expected = try greedy(h.fixture.target, prompt: [5, 6, 7], count: 5)
            let tracking = try #require(h.engine.loopForTesting.nativeShutdownState)
            var requested = false
            h.engine.loopForTesting.onEngineQueueSync {
                h.engine.loopForTesting.suspendStepExecutionAtCountForTesting = 0
                tracking.afterAssistantFenceForTesting = { [weak engine = h.engine] _ in
                    guard !requested else { return }
                    requested = true
                    // Queue cancellation while real priming roots are still
                    // owned, rather than race a fast request's buffered output.
                    engine?.cancel(.init(1911))
                }
            }
            let first = try h.engine.submit(
                .init(
                    id: .init(1911), promptTokens: [1, 2],
                    sampling: .init(temperature: 0), maxTokens: 10))
            let second = try h.engine.submit(
                .init(
                    id: .init(1912), promptTokens: [5, 6, 7],
                    sampling: .init(temperature: 0), maxTokens: 5))
            h.engine.loopForTesting.onEngineQueueSync {
                h.engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil
            }
            var cancelled = false
            for await event in first {
                if case .finished(let reason, _) = event, case .cancelled = reason {
                    cancelled = true
                }
            }
            var actual: [Int] = []
            for await event in second {
                if case .delta(_, let values, _) = event { actual += values }
            }
            let outcome = await h.engine.shutdownReportingNativeCompletion()
            #expect(requested && cancelled && actual == expected)
            guard case .quiescent = outcome else {
                Issue.record("interleaved GLM cancellation left native work unresolved")
                return
            }
            #expect(h.process.bytes == 0 && h.resources.backend.bytesReserved == 0)
        }

        @Test func protectedGLMTicketRejectsForeignTargetStorageAndProcessOwner() throws {
            let f = try Fixture(layers: 4)
            let factory = try GLM5NextMTPExecution(
                target: f.target, predictor: f.mtp,
                maximumInputs: 16, storage: .contiguous)
            let assistant = try GLM5NextMTPAdapter(
                target: f.target, predictor: f.mtp, factory: factory)
            let process = ProcessBudget()
            let config = try nativeConfiguration(f.target, dtype: .float16)
            #expect(throws: GLM5NextMTPError.self) {
                try factory.makeNativeResources(
                    assistant: assistant, config: config,
                    processMemoryOwner: process, construction: NativeConstructionScope())
            }
            let resources = try factory.makeNativeResources(
                assistant: assistant,
                config: nativeConfiguration(f.target), processMemoryOwner: process,
                construction: NativeConstructionScope())
            let foreign = try Fixture(layers: 4)
            #expect(
                !resources.contract.consume(
                    model: CBv2SteppableLanguageModelAdapter(foreign.target),
                    backend: resources.backend, cacheProvider: resources.cacheProvider,
                    assistant: assistant,
                    processMemoryOwner: process))
            #expect(
                !resources.contract.consume(
                    model: resources.model,
                    backend: resources.backend,
                    cacheProvider: CBv2LayerCacheBank(layerKinds: f.target.cbv2LayerKinds),
                    assistant: assistant, processMemoryOwner: process))
            #expect(
                !resources.contract.consume(
                    model: resources.model,
                    backend: resources.backend, cacheProvider: resources.cacheProvider,
                    assistant: assistant,
                    processMemoryOwner: ProcessBudget()))
            #expect(
                !resources.contract.consume(
                    model: resources.model,
                    backend: resources.backend, cacheProvider: resources.cacheProvider,
                    assistant: assistant,
                    mtpVerificationMode: .rectangular, processMemoryOwner: process))
            #expect(
                !resources.contract.consume(
                    model: resources.model,
                    backend: resources.backend, cacheProvider: resources.cacheProvider,
                    assistant: nil,
                    processMemoryOwner: process))
            #expect(
                resources.contract.consume(
                    model: resources.model,
                    backend: resources.backend, cacheProvider: resources.cacheProvider,
                    assistant: assistant,
                    processMemoryOwner: process))
            #expect(
                !resources.contract.consume(
                    model: resources.model,
                    backend: resources.backend, cacheProvider: resources.cacheProvider,
                    assistant: assistant,
                    processMemoryOwner: process))
        }

        // MARK: Rectangular verification

        static let linearMembers: [GLM5NextExecutionPolicy.LinearAttention] = [
            .reference, .nativeRecurrence, .nativeRecurrenceChunkedPrefill,
        ]

        /// KDA arithmetic of a captured window: each `linearAttention` member's
        /// own recurrence, or the fused decode-step kernel (which needs a head
        /// width divisible by 32 for multi-token windows).
        static let captureArithmetic = ["reference", "native", "chunked", "fused"]

        private static func capturePolicy(_ name: String) -> GLM5NextExecutionPolicy {
            switch name {
            case "native": .init(linearAttention: .nativeRecurrence)
            case "chunked": .init(linearAttention: .nativeRecurrenceChunkedPrefill)
            case "fused": .init(linearAttention: .nativeRecurrence, linearDecodeStep: .fusedKernel)
            default: .reference
            }
        }

        /// Every arithmetic captures inside the recurrence that produced the
        /// window's final state: the reference loop, the native kernel (which
        /// the chunked member also uses for windows shorter than its chunk) or
        /// the fused step kernel's per-token capture.
        @Test(arguments: captureArithmetic)
        func kdaCapturedWindowStatesEqualOneTokenSteps(arithmetic: String) throws {
            let f = try Fixture(
                layers: 4, targetNoise: true, linearHeadDimension: arithmetic == "fused" ? 32 : 2,
                executionPolicy: Self.capturePolicy(arithmetic))
            let kda = try #require(f.target.model.layers[0].attention as? GLM5NextKDASublayer)
            let x = MLXArray((0 ..< 12).map { Float(sin(Double($0) * 0.9)) }).reshaped([1, 3, 4])
            let valid = MLXArray.ones([1, 3], dtype: .bool)
            let window = try kda.forwardCapturing(
                x, state: nil, validity: valid, canonicalConvolution: false, capture: true)
            let captured = try #require(window.captured)
            #expect(captured.convolution.dim(1) == 3 && captured.recurrent.dim(1) == 3)
            // [1,3] and [1,1] projections may use different matmul kernels, so
            // the one-token chain is a numerical oracle here, not a bitwise one.
            var state: GLM5NextLinearAttention.State?
            for position in 0 ..< 3 {
                let step = try kda.forward(
                    x[0..., position ..< position + 1], state: state,
                    validity: valid[0..., position ..< position + 1])
                state = step.state
                #expect(
                    maxDifference(window.values[0..., position ..< position + 1], step.values)
                        < 1e-5)
                #expect(
                    maxDifference(captured.convolution[0..., position], step.state.convolution)
                        < 1e-5)
                #expect(
                    maxDifference(captured.recurrent[0..., position], step.state.recurrent)
                        < 1e-5)
            }
            // The final captured entry IS the returned state.
            expectBitwise(captured.convolution[0..., 2], window.state.convolution)
            expectBitwise(captured.recurrent[0..., 2], window.state.recurrent)
        }

        /// The chunked recurrence has no per-step states; a capture request
        /// through it is refused before any graph is built.
        @Test func chunkedRecurrenceRefusesPerPositionCapture() {
            let shape = [1, 32, 1, 2]
            let zeros = MLXArray.zeros(shape, dtype: .float32)
            #expect(throws: GLM5NextLinearAttention.Failure.unsupportedCapture) {
                try GLM5NextLinearAttention.apply(
                    query: zeros, key: zeros, value: zeros, forgetLogits: zeros,
                    betaLogits: MLXArray.zeros([1, 32, 1], dtype: .float32), outputGate: zeros,
                    weights: .init(
                        convolution: MLXArray.zeros([6, 4], dtype: .float32),
                        aLog: MLXArray.zeros([1], dtype: .float32),
                        dtBias: MLXArray.zeros([2], dtype: .float32),
                        outputNorm: MLXArray.ones([2], dtype: .float32)),
                    execution: .chunked(size: 32), capture: true)
            }
        }

        /// Contiguous CBv2 rows exercise the same DSA payload/selection and KDA
        /// staging as paged rows, but expose every committed row for comparison.
        /// Committing any accepted prefix of one captured window (or rolling it
        /// all back) must leave exactly the state of replaying that prefix as
        /// decode steps, and the frontier must continue like that replay. The
        /// window is one batched graph, so the decode chain is a numerical
        /// oracle: projections of `[1,4]` and `[1,1]` may use different kernels.
        @Test(arguments: captureArithmetic, [0, 1, 2, 4])
        func capturedWindowCommitsEveryPrefixLikeDecodeReplay(arithmetic: String, keep: Int)
            throws
        {
            let f = try Fixture(
                layers: 4, targetNoise: true, linearHeadDimension: arithmetic == "fused" ? 32 : 2,
                executionPolicy: Self.capturePolicy(arithmetic))
            let prompt = Array(1 ... 6)
            let window = [5, 9, 2, 7]
            let tolerance: Float = 1e-4
            let serial = try ContiguousTarget(f.target)
            serial.step(prompt)
            let serialLogits = window.map { serial.step([$0]) }
            let control = try ContiguousTarget(f.target)
            control.step(prompt)
            for token in window.prefix(keep) { control.step([token]) }

            let rectangular = try ContiguousTarget(f.target)
            rectangular.step(prompt)
            let captured = try rectangular.capturedWindow(window, keep: keep)
            for (position, logits) in serialLogits.enumerated() {
                #expect(
                    maxDifference(captured[0..., position ..< position + 1], logits) < tolerance)
            }
            for layer in f.target.cbv2RecurrentStateSpec.layers.map(\.modelLayerIndex) {
                let actual = try #require(rectangular.recurrent.state(modelLayerIndex: layer))
                let expected = try #require(control.recurrent.state(modelLayerIndex: layer))
                #expect(actual.conv!.shape == expected.conv!.shape)
                #expect(actual.ssm!.shape == expected.ssm!.shape)
                #expect(maxDifference(actual.conv!, expected.conv!) < tolerance)
                #expect(maxDifference(actual.ssm!, expected.ssm!) < tolerance)
            }
            for (actual, expected) in zip(rectangular.rows, control.rows) {
                let a = actual.snapshot()
                let e = expected.snapshot()
                #expect(a.offset == prompt.count + keep && a.offset == e.offset)
                #expect(maxDifference(a.keys, e.keys) < tolerance)
                #expect(maxDifference(a.values, e.values) < tolerance)
            }
            // The committed or rolled-back frontier continues like the replay.
            #expect(maxDifference(rectangular.step([11]), control.step([11])) < tolerance)
            try rectangular.close()
            try control.close()
            try serial.close()
        }

        /// With 3 observed tokens step 0 selects all history. With 8, two complete
        /// pools compete for the one-pool budget, so step 0 selects 4 of 8 rows;
        /// continuations must attend exactly those rows, not the full history.
        @Test(arguments: [3, 8])
        func rectangularProposalsReuseStepZeroSelectionAndNeverPublishHistory(observed: Int)
            throws
        {
            let h = try AdmittedHarness(maximumInputs: 16, verification: .rectangular(depth: 3))
            #expect(h.adapter.maximumDraftTokens == 3)
            #expect(h.adapter.requiredVerificationMode == .rectangular)
            let state = try #require(h.adapter.makeRequestState() as? GLM5NextMTPAdapter.State)
            try h.adapter.configureRequestState(state, admission: h.context)
            let hidden = features(observed + 2)
            let shifted = Array(2 ... observed + 1)
            try h.adapter.observe(
                .init(
                    tokens: tokens(Array(1 ... observed)),
                    hidden: rows(Array(hidden.prefix(observed)))),
                state: state)
            try fence(h.adapter, state)
            #expect(state.committedInputCount == observed - 1)
            let first = try h.adapter.draft(
                tokens: tokens([observed + 1]), hidden: rows([hidden[observed - 1]]),
                state: state)
            let second = try h.adapter.draft(
                tokens: first.tokens.reshaped([1, 1]), hidden: first.hidden, state: state)
            let third = try h.adapter.draft(
                tokens: second.tokens.reshaped([1, 1]), hidden: second.hidden, state: state)
            #expect(throws: GLM5NextMTPError.unsupportedDraftDepth(4)) {
                try h.adapter.draft(
                    tokens: third.tokens.reshaped([1, 1]), hidden: third.hidden, state: state)
            }
            #expect(state.committedInputCount == observed - 1 && state.stagedInputCount == 1)
            try fence(h.adapter, state)

            // The same trained graph over the same rows yields step 0's
            // selection, including its deterministic tie membership.
            let stepZero = try h.fixture.mtp.prepareExecution(
                shiftedTokens: tokens(shifted), hidden: rows(Array(hidden.prefix(observed))),
                cache: GLM5NextDSACache())
            guard case .sparse(let selection) = stepZero.selection else {
                Issue.record("the reference fixture always selects sparsely")
                return
            }
            let selected = selection.indices.asArray(Int32.self).filter { $0 >= 0 }.map(Int.init)
            #expect(selected.count == (observed == 8 ? 4 : observed))
            #expect(Set(selected).count == selected.count && selected.allSatisfy { $0 < observed })
            let base = reference(shifted: shifted, hidden: Array(hidden.prefix(observed)))
            let all = base.projected.map(latent)
            let shared = selected.map { all[$0] }
            close(
                first.hidden,
                referenceReadout(projected: base.projected.last!, latents: shared).hidden)
            var previous = values(first.hidden).map(Double.init)
            for (proposal, next) in [(first, second), (second, third)] {
                let projected = project(
                    embedding: Fixture.embedding(Int(proposal.tokens.item(Int32.self))),
                    hidden: previous, zeroEmbedding: false)
                let oracle = referenceReadout(projected: projected, latents: shared)
                close(next.hidden, oracle.hidden)
                if observed == 8 {
                    // The restriction is observable: the full history differs.
                    let unrestricted = referenceReadout(projected: projected, latents: all)
                    #expect(
                        zip(oracle.hidden, unrestricted.hidden).contains { abs($0 - $1) > 1e-3 })
                }
                previous = oracle.hidden
            }

            // Two accepted proposals are queued with verified target hidden rows;
            // only the step-0 inputs become history now.
            try h.adapter.finalize(
                state: state, confirmedInputs: 3,
                acceptedTokens: concatenated([first.tokens, second.tokens]).reshaped([1, 2]),
                targetHidden: rows([hidden[observed], hidden[observed + 1]]))
            #expect(state.committedInputCount == observed && state.stagedInputCount == 0)
            try h.adapter.observe(
                .init(tokens: tokens([7]), hidden: rows([hidden[observed + 1]])), state: state)
            try fence(h.adapter, state)
            #expect(state.committedInputCount == observed + 3)
            h.adapter.releaseRequestState(state)
            h.admission.releaseAll(id: h.id)
        }

        /// Host-only scalar check of the binary-head fixture's acceptance claim,
        /// at every attention mixture on the latent circle (the extremes of any mean).
        @Test func binaryHeadFixtureDecidesEveryProposal() {
            let mixtures = (0 ..< 360).map { degree -> [Double] in
                let angle = Double(degree) * .pi / 180
                return [sqrt(2) * cos(angle), sqrt(2) * sin(angle)]
            }
            func target(_ token: Int) -> [Double] {
                normalize(Fixture.embedding(token), weight: Fixture.targetNorm)
            }
            func proposal(_ token: Int, _ hidden: [Double], sign: Double, _ mixture: [Double])
                -> [Double]
            {
                let projected = project(
                    embedding: Fixture.embedding(token), hidden: hidden, zeroEmbedding: false
                ).map { $0 * sign }
                return referenceReadout(projected: projected, latents: [mixture]).hidden
            }
            #expect((0 ..< 16).allSatisfy { target($0)[3] < 0 })
            for sign in [1.0, -1.0] {
                for token in 0 ..< 16 {
                    for previous in 0 ..< 16 {
                        let stepZero = mixtures.map {
                            proposal(token, target(previous), sign: sign, $0)[3]
                        }
                        #expect(sign > 0 ? stepZero.max()! < 0 : stepZero.min()! > 0)
                        guard sign > 0 else { continue }
                        // Continuations: any step-0 mixture, then any later mixture.
                        for stepZeroMixture in stride(from: 0, to: 360, by: 30) {
                            let draft = proposal(
                                token, target(previous), sign: sign, mixtures[stepZeroMixture])
                            for next in 0 ..< 16 {
                                #expect(
                                    stride(from: 0, to: 360, by: 4).allSatisfy {
                                        proposal(next, draft, sign: sign, mixtures[$0])[3] < 0
                                    })
                            }
                        }
                    }
                }
            }
        }

        @Test func rectangularAllocationPricesOnlyRetainedContinuationWork() throws {
            let c = try officialMemoryConfiguration()
            func plan(_ depth: Int) throws -> GLM5NextMTPExecution.MemoryPlan {
                try GLM5NextMTPExecution.memoryPlan(
                    configuration: c, maximumInputs: 4096,
                    limits: .init(maximumPrefillTokens: 128, maximumDraftTokens: depth))
            }
            func bytes(_ items: [CBv2MTPFixedBufferSpec]) -> Int {
                items.reduce(0) { $0 + $1.logicalBytes * $1.allocationCount }
            }
            let one = try plan(1)
            let three = try plan(3)
            // Depth one keeps the pre-rectangular formula: draft work twice,
            // history work three times, and no continuation term.
            #expect(
                bytes(one.specification.working)
                    == 2 * bytes(one.draftWorkBuffers) + 3 * bytes(one.historyWorkBuffers))
            #expect(one.draftWorkBuffers.count == 19)
            #expect(one.specification.resident == three.specification.resident)
            #expect(one.draftWorkBuffers == three.draftWorkBuffers)
            #expect(bytes(three.continuationWorkBuffers) < bytes(three.draftWorkBuffers))
            #expect(
                bytes(three.specification.working) - bytes(one.specification.working)
                    == 4 * bytes(three.continuationWorkBuffers))
            #expect(throws: GLM5NextMTPError.self) { try plan(4) }
        }

        @Test func rectangularProfileIsSeparateFromSerialAndRefusesUnsupportedTuples() throws {
            let f = try Fixture(layers: 4)
            #expect(throws: GLM5NextMTPError.unsupportedDraftDepth(4)) {
                try GLM5NextMTPExecution(
                    target: f.target, predictor: f.mtp, maximumInputs: 16,
                    storage: .contiguous, verification: .rectangular(depth: 4))
            }
            let factory = try GLM5NextMTPExecution(
                target: f.target, predictor: f.mtp, maximumInputs: 16,
                storage: .contiguous, verification: .rectangular(depth: 2))
            let assistant = try GLM5NextMTPAdapter(
                target: f.target, predictor: f.mtp, factory: factory)
            #expect(assistant.requiredVerificationMode == .rectangular)
            #expect(assistant.maximumDraftTokens == 2 && assistant.maximumSpeculativeBatch == 1)
            #expect(
                assistant.boundedRequestAllocation(
                    limits: .init(maximumPrefillTokens: 8, maximumDraftTokens: 3)) == nil)
            #expect(
                assistant.boundedRequestAllocation(
                    limits: .init(maximumPrefillTokens: 8, maximumDraftTokens: 2)) != nil)
            let process = ProcessBudget()
            let resources = try factory.makeNativeResources(
                assistant: assistant, config: nativeConfiguration(f.target),
                processMemoryOwner: process, construction: NativeConstructionScope())
            #expect(resources.contract.profile == "glm5_next_native_paged_rectangular_mtp_v1")
            #expect(resources.contract.mtpVerificationMode == .rectangular)
            #expect(resources.contract.supportsNativePagedRectangularMTP)
            #expect(!resources.contract.supportsNativePagedSerialMTP)
            #expect(
                !resources.contract.consume(
                    model: resources.model,
                    backend: resources.backend, cacheProvider: resources.cacheProvider,
                    assistant: assistant, mtpVerificationMode: .serialTarget,
                    processMemoryOwner: process))
            #expect(
                resources.contract.consume(
                    model: resources.model,
                    backend: resources.backend, cacheProvider: resources.cacheProvider,
                    assistant: assistant, mtpVerificationMode: .rectangular,
                    processMemoryOwner: process))
        }

        /// The binary-head fixture makes acceptance deterministic: +1 accepts every
        /// proposal, -1 rejects every first proposal.
        @Test(arguments: [1, 2, 3], [1.0, -1.0])
        func protectedRectangularEngineMatchesTargetGreedy(depth: Int, projectionSign: Double)
            async throws
        {
            let h = try NativeHarness(
                projectionSign: projectionSign, verification: .rectangular(depth: depth),
                binaryHead: true)
            let prompt = [1, 2, 3, 4]
            let expected = try greedy(h.fixture.target, prompt: prompt, count: 8)
            let plain = try await targetOnlyPagedEngine(h.fixture.target, prompt: prompt, count: 8)
            let actual = try await generate(h.engine, id: 1921, prompt: prompt, count: 8)
            let outcome = await h.engine.shutdownReportingNativeCompletion()
            #expect(actual == expected && actual == plain)
            #expect(actual == Array(repeating: Fixture.binaryTargetToken, count: 8))
            let metrics = try #require(h.engine.mtpMetricsSnapshot())
            #expect(metrics.verificationMode == .rectangular)
            #expect(metrics.rectangularVerificationRounds > 0)
            #expect(metrics.serialVerificationRounds == 0 && metrics.proposedTokens > 0)
            if projectionSign < 0 {
                #expect(metrics.acceptedTokens == 0)
            } else {
                #expect(metrics.acceptedTokens > 0)
                if depth >= 2 {
                    // Continuations were accepted, not only step-0 proposals.
                    #expect(metrics.acceptedTokens > metrics.rectangularVerificationRounds)
                }
            }
            expectQuiescentRectangular(h, outcome)
        }

        /// A stateful (nonzero) target with four-token pages: windows cross page
        /// and index-pool boundaries, and rejected suffixes rewind real KDA/DSA state.
        /// The binary head lets the proposal agree only where the target does.
        @Test(arguments: [2, 3])
        func protectedRectangularEngineIsExactAcrossPagesWithStatefulTarget(depth: Int)
            async throws
        {
            let prompt = [1, 2, 3]
            let count = 12
            // The proposal always predicts B; pick the first fixed noise level
            // whose target greedy output also contains A, so both an accepted
            // and a rejected proposal must occur.
            let amplitude = try [0.35, 0.7, 1.0, 1.5, 2.5, 4.0].first { amplitude in
                let target = try Fixture(
                    layers: 4, targetNoise: true, binaryHead: true, noiseAmplitude: amplitude
                ).target
                // The first two tokens precede any speculated position.
                let tokens = try greedy(target, prompt: prompt, count: count).dropFirst(2)
                return tokens.contains(Fixture.binaryTargetToken)
                    && tokens.contains { $0 != Fixture.binaryTargetToken }
            }
            let noise = try #require(amplitude)
            let serial = try NativeHarness(
                projectionSign: 1, pageSize: 4, targetNoise: true, binaryHead: true,
                noiseAmplitude: noise)
            let plain = try await targetOnlyPagedEngine(
                serial.fixture.target, prompt: prompt, count: count)
            let serialTokens = try await generate(
                serial.engine, id: 1931, prompt: prompt, count: count)
            _ = await serial.engine.shutdownReportingNativeCompletion()
            let h = try NativeHarness(
                projectionSign: 1, verification: .rectangular(depth: depth), pageSize: 4,
                targetNoise: true, binaryHead: true, noiseAmplitude: noise)
            let actual = try await generate(h.engine, id: 1932, prompt: prompt, count: count)
            let outcome = await h.engine.shutdownReportingNativeCompletion()
            #expect(actual == plain && serialTokens == plain)
            let metrics = try #require(h.engine.mtpMetricsSnapshot())
            #expect(metrics.rectangularVerificationRounds > 0)
            // Both an accepted and a rejected proposal must occur.
            #expect(metrics.acceptedTokens > 0)
            #expect(metrics.acceptedTokens < metrics.proposedTokens)
            expectQuiescentRectangular(h, outcome)
        }

        /// Target-prefix acceptance: a sampled row speculates, and because the
        /// verifier re-derives each position's ordinary draw from the request's
        /// own RNG stream, the output equals target-only sampling with the same
        /// request id and seed, token for token.
        @Test(arguments: [1, 3])
        func protectedRectangularEngineSamplesExactlyLikeTargetOnly(depth: Int) async throws {
            let prompt = [1, 2, 3]
            let count = 12
            let sampling = CBv2SamplingParams(temperature: 1, seed: 29)
            let h = try NativeHarness(
                projectionSign: 1, verification: .rectangular(depth: depth), pageSize: 4,
                targetNoise: true, binaryHead: true, sampled: true)
            let plain = try await targetOnlyPagedEngine(
                h.fixture.target, prompt: prompt, count: count, id: 1961, sampling: sampling)
            let actual = try await generate(
                h.engine, id: 1961, prompt: prompt, count: count, sampling: sampling)
            let outcome = await h.engine.shutdownReportingNativeCompletion()
            #expect(actual == plain && actual.count == count)
            let metrics = try #require(h.engine.mtpMetricsSnapshot())
            #expect(metrics.rectangularVerificationRounds > 0 && metrics.proposedTokens > 0)
            expectQuiescentRectangular(h, outcome)
        }

        /// Rectangular verification with a complete-prefix store is not yet
        /// qualified, so the protected factory refuses that tuple before issuing
        /// any native resource; serial verification keeps its prefix profile.
        @Test(arguments: [1, 3])
        func protectedRectangularEngineRefusesCompletePrefix(depth: Int) throws {
            let fixture = try Fixture(layers: 4, targetNoise: true, binaryHead: true)
            let factory = try GLM5NextMTPExecution(
                target: fixture.target, predictor: fixture.mtp, maximumInputs: 32,
                storage: .contiguous, verification: .rectangular(depth: depth))
            let assistant = try GLM5NextMTPAdapter(
                target: fixture.target, predictor: fixture.mtp, factory: factory)
            let limits = try CBv2PagedGatheredAttentionLimits(
                maximumBatchSize: 1, maximumQueryTokens: 32, maximumContextTokens: 32,
                maximumInFlightGraphs: 2, maximumScratchBytes: 128 << 20,
                admissionMode: .stepOwned(.pinnedMetal))
            #expect(throws: GLM5NextMTPError.self) {
                try factory.makeNativeResources(
                    assistant: assistant,
                    config: .init(
                        capacityBytes: 64 << 20, dtype: .float32,
                        maxPrefillChunk: 32, nominalMaxSequenceLength: 32,
                        maxBufferLength: 16 << 20, segmentSizeBytes: 64 << 10,
                        layerDTypes: fixture.target.cbv2CompleteCheckpointKVDTypes,
                        gatheredAttention: limits),
                    processMemoryOwner: ProcessBudget(), construction: NativeConstructionScope(),
                    completePrefixCache: PrefixStore(.init(segmentBytes: 28)))
            }
        }

        /// Native store adapter over the shared fixture store.
        private final class PrefixStore: CBv2NativeCompletePrefixCache, @unchecked Sendable {
            let base: CompleteCheckpointFixtureStore
            init(_ base: CompleteCheckpointFixtureStore) { self.base = base }
            var identity: CBv2CompleteCheckpointIdentity { base.identity }
            func acceptsCheckpoint(position: Int, packedBytes: Int) -> Bool {
                base.acceptsCheckpoint(position: position, packedBytes: packedBytes)
            }
            func takeStaged(
                requestID: CBv2RequestID, tokens: [Int], cacheSalt: String?,
                maximumSequenceLength: Int
            ) -> CBv2StagedCompleteCheckpoint? {
                base.takeStaged(
                    requestID: requestID, tokens: tokens, cacheSalt: cacheSalt,
                    maximumSequenceLength: maximumSequenceLength)
            }
            func donate(
                _ source: CBv2CompleteCheckpointExport, requestID: CBv2RequestID?,
                tokens: [Int], cacheSalt: String?, completion: @escaping @Sendable ([Int]) -> Void
            ) {
                base.donate(
                    source, requestID: requestID, tokens: tokens, cacheSalt: cacheSalt,
                    completion: completion)
            }
            func close() { base.close() }
            func closeAndWait() async { base.close() }
        }

        /// Several requests speculate together: one captured forward verifies
        /// every row's window, while each row keeps its own assistant history and
        /// KDA transaction. Every output equals its own target greedy decode.
        @Test(arguments: [1, 2])
        func protectedRectangularEngineBatchesSpeculatingRowsExactly(depth: Int) async throws {
            let fixture = try Fixture(layers: 4, targetNoise: true, binaryHead: true)
            let process = ProcessBudget()
            let factory = try GLM5NextMTPExecution(
                target: fixture.target, predictor: fixture.mtp, maximumInputs: 16,
                storage: .contiguous, verification: .rectangular(depth: depth),
                maximumSpeculativeBatch: 2)
            let assistant = try GLM5NextMTPAdapter(
                target: fixture.target, predictor: fixture.mtp, factory: factory)
            #expect(assistant.maximumSpeculativeBatch == 2)
            let resources = try factory.makeNativeResources(
                assistant: assistant, config: nativeConfiguration(fixture.target),
                processMemoryOwner: process, construction: NativeConstructionScope())
            let engine = EngineV2(
                model: resources.model, layerKinds: fixture.target.cbv2LayerKinds,
                backend: resources.backend, cacheProvider: resources.cacheProvider,
                sampler: CBv2DefaultSampler(),
                schedulerConfig: .init(
                    maxConcurrentRequests: 2, maxBatchedTokensPerStep: 8,
                    prefillChunkSize: 2, maxWaiting: 4, enablePrefixCache: false),
                admissionConfig: fixture.target.cbv2NativeAdmissionConfig,
                mtpDrafter: assistant,
                mtpConfig: .init(
                    enabled: true, maxDraftTokens: depth, maxSpeculativeBatch: 2,
                    fixedDraftTokens: depth, verificationMode: .rectangular),
                processMemoryOwner: process, nativeCompletionTracking: true,
                nativeExecutionContract: resources.contract)
            #expect(engine.mtpInactiveReason == nil)
            let prompts = [[1, 2, 3], [5, 6, 7, 8]]
            let sampled = CBv2SamplingParams(temperature: 1, seed: 5)
            let expectedGreedy = try greedy(fixture.target, prompt: prompts[0], count: 8)
            let expectedSampled = try await targetOnlyPagedEngine(
                fixture.target, prompt: prompts[1], count: 8, id: 1982, sampling: sampled)
            async let first = generate(engine, id: 1981, prompt: prompts[0], count: 8)
            async let second = generate(
                engine, id: 1982, prompt: prompts[1], count: 8, sampling: sampled)
            let (greedyTokens, sampledTokens) = try await (first, second)
            let metrics = try #require(engine.mtpMetricsSnapshot())
            let outcome = await engine.shutdownReportingNativeCompletion()
            #expect(greedyTokens == expectedGreedy)
            #expect(sampledTokens == expectedSampled)
            #expect(metrics.rectangularVerificationRounds > 0)
            guard case .quiescent = outcome else {
                Issue.record("batched rectangular shutdown: \(outcome)")
                return
            }
            #expect(resources.backend.bytesReserved == 0 && process.bytes == 0)
        }

        /// Captured KDA residency of one k=3 window (position-major stacks plus the
        /// per-position states that build them) stays within the engine's
        /// 2(k+1)-generation captured reservation. Wider KDA geometry makes the
        /// state, not tiny activations or allocator rounding, the measured term.
        @Test func capturedDepthThreeWindowFitsItsRecurrentReservation() throws {
            let f = try Fixture(
                layers: 4, targetNoise: true, linearHeads: 4, linearHeadDimension: 64)
            let policy = try #require(Memory.allocationFootprintPolicy())
            let window = [5, 9, 2, 7]
            let reservation =
                try f.target.cbv2RecurrentStateSpec.allocationBytesPerGeneration(policy: policy)
                * 2 * window.count
            func peak(captured: Bool) throws -> Int {
                let target = try ContiguousTarget(f.target)
                target.step([1, 2, 3])
                Memory.clearCache()
                let baseline = Memory.activeMemory
                Memory.peakMemory = 0
                if captured {
                    _ = try target.capturedWindow(window, keep: window.count)
                } else {
                    target.step(window)
                }
                let used = max(Memory.peakMemory, baseline) - baseline
                try target.close()
                return used
            }
            let plain = try peak(captured: false)
            let captured = try peak(captured: true)
            #expect(captured - plain <= reservation)
        }

        /// The full engine under each native `linearAttention` member: a stateful
        /// target's rectangular depth-3 output equals serial MTP and target-only
        /// decode with the same policy.
        @Test(arguments: [
            GLM5NextExecutionPolicy.LinearAttention.nativeRecurrence,
            .nativeRecurrenceChunkedPrefill,
        ])
        func protectedRectangularEngineMatchesSerialUnderNativeLinearAttention(
            member: GLM5NextExecutionPolicy.LinearAttention
        ) async throws {
            let policy = GLM5NextExecutionPolicy(linearAttention: member)
            let prompt = [1, 2, 3]
            let serial = try NativeHarness(
                projectionSign: 1, targetNoise: true, binaryHead: true, executionPolicy: policy)
            let plain = try await targetOnlyPagedEngine(
                serial.fixture.target, prompt: prompt, count: 10)
            let serialTokens = try await generate(
                serial.engine, id: 1951, prompt: prompt, count: 10)
            _ = await serial.engine.shutdownReportingNativeCompletion()
            let h = try NativeHarness(
                projectionSign: 1, verification: .rectangular(depth: 3), targetNoise: true,
                binaryHead: true, executionPolicy: policy)
            let actual = try await generate(h.engine, id: 1952, prompt: prompt, count: 10)
            let outcome = await h.engine.shutdownReportingNativeCompletion()
            #expect(actual == plain && serialTokens == plain)
            let metrics = try #require(h.engine.mtpMetricsSnapshot())
            #expect(metrics.rectangularVerificationRounds > 0)
            expectQuiescentRectangular(h, outcome)
        }

        @Test func protectedRectangularEngineCancelsWithoutLeakingOwners() async throws {
            let h = try NativeHarness(projectionSign: 1, verification: .rectangular(depth: 3))
            let expected = try greedy(h.fixture.target, prompt: [5, 6, 7], count: 5)
            let tracking = try #require(h.engine.loopForTesting.nativeShutdownState)
            var requested = false
            h.engine.loopForTesting.onEngineQueueSync {
                h.engine.loopForTesting.suspendStepExecutionAtCountForTesting = 0
                tracking.afterAssistantFenceForTesting = { [weak engine = h.engine] _ in
                    guard !requested else { return }
                    requested = true
                    engine?.cancel(.init(1941))
                }
            }
            let first = try h.engine.submit(
                .init(
                    id: .init(1941), promptTokens: [1, 2],
                    sampling: .init(temperature: 0), maxTokens: 10))
            let second = try h.engine.submit(
                .init(
                    id: .init(1942), promptTokens: [5, 6, 7],
                    sampling: .init(temperature: 0), maxTokens: 5))
            h.engine.loopForTesting.onEngineQueueSync {
                h.engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil
            }
            var cancelled = false
            for await event in first {
                if case .finished(let reason, _) = event, case .cancelled = reason {
                    cancelled = true
                }
            }
            var actual: [Int] = []
            for await event in second {
                if case .delta(_, let values, _) = event { actual += values }
            }
            let outcome = await h.engine.shutdownReportingNativeCompletion()
            #expect(requested && cancelled && actual == expected)
            expectQuiescentRectangular(h, outcome)
        }

        private func generate(
            _ engine: EngineV2, id: UInt64, prompt: [Int], count: Int,
            sampling: CBv2SamplingParams = .init(temperature: 0)
        )
            async throws -> [Int]
        {
            let stream = try engine.submit(
                .init(id: .init(id), promptTokens: prompt, sampling: sampling, maxTokens: count))
            var result: [Int] = []
            for await event in stream {
                if case .delta(_, let values, _) = event { result += values }
            }
            return result
        }

        private func expectQuiescentRectangular(
            _ h: NativeHarness, _ outcome: CBv2NativeShutdownOutcome
        ) {
            guard case .quiescent(let receipt) = outcome else {
                Issue.record("rectangular GLM native shutdown did not complete: \(outcome)")
                return
            }
            #expect(receipt.executionContractID == h.resources.contract.id)
            #expect(h.resources.contract.profile == "glm5_next_native_paged_rectangular_mtp_v1")
            #expect(h.engine.loopForTesting.nativeShutdownState?.debugRetainedRootCount == 0)
            #expect(h.resources.backend.bytesInUse == 0 && h.resources.backend.bytesReserved == 0)
            #expect(h.engine.loopForTesting.recurrentStates.isEmpty)
            #expect(h.process.bytes == 0)
        }

        private func expectBitwise(_ actual: MLXArray, _ expected: MLXArray) {
            #expect(actual.shape == expected.shape)
            #expect(
                actual.asType(.float32).asArray(Float.self)
                    == expected.asType(.float32).asArray(Float.self))
        }

        private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
            abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
        }

        /// One request's contiguous CBv2 rows and request-owned KDA state.
        private final class ContiguousTarget {
            let model: GLM5NextTextModel
            let rows: [CBv2FullSequenceKV]
            let caches: [CBv2LayerCache]
            let recurrent: CBv2RecurrentRequestState

            init(_ model: GLM5NextTextModel) throws {
                self.model = model
                let kinds = model.cbv2LayerKinds
                let rows = kinds.map {
                    CBv2FullSequenceKV(
                        promptLength: 0, maxLength: 32, kvHeads: $0.kvHeads, headDim: $0.headDim,
                        valueHeadDim: $0.valueHeadDim)
                }
                self.rows = rows
                caches = kinds.indices.map {
                    CBv2LayerCache(layerIndex: $0, kind: kinds[$0], rows: [rows[$0]])
                }
                recurrent = try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
            }

            /// Ordinary committed forward (prefill chunk or one-token decode).
            @discardableResult
            func step(_ ids: [Int]) -> MLXArray {
                do {
                    let evaluation = try recurrent.bind()
                    let output = try model.prepareCBv2Forward(
                        tokens: Self.tokens(ids), caches: caches, recurrentState: [evaluation])
                    try fence([output.logits] + evaluation.evaluate())
                    try evaluation.commit()
                    return output.logits
                } catch {
                    Issue.record("contiguous GLM step failed: \(error)")
                    return MLXArray.zeros([1])
                }
            }

            /// One captured window; keep 0 rolls the whole window back.
            func capturedWindow(_ ids: [Int], keep: Int) throws -> MLXArray {
                for row in rows { row.beginSpeculativeWrite() }
                let evaluation = try recurrent.bind()
                let output = try model.prepareCBv2Forward(
                    tokens: Self.tokens(ids), caches: caches, recurrentState: [evaluation],
                    captureRecurrent: true)
                #expect(evaluation.isCaptured)
                try fence([output.logits] + evaluation.evaluate())
                if keep == 0 {
                    try evaluation.rollback()
                } else {
                    try evaluation.commit(keepPositions: keep)
                }
                for row in rows {
                    row.rollback(ids.count - keep)
                    row.commitSpeculativeWrite()
                }
                return output.logits
            }

            func close() throws {
                for cache in caches { cache.setRows([]) }
                try recurrent.release()
            }

            private func fence(_ roots: [MLXArray]) throws {
                try withError { fault in
                    eval(roots + caches.flatMap { $0.innerState() })
                    try fault.check()
                }
            }

            private static func tokens(_ ids: [Int]) -> MLXArray {
                MLXArray(ids.map(Int32.init)).reshaped([1, ids.count])
            }
        }

        private func greedy(_ model: GLM5NextTextModel, prompt: [Int], count: Int) throws -> [Int] {
            let cache = model.newTextCache()
            var input = prompt
            var result: [Int] = []
            for _ in 0 ..< count {
                let next = try model.forward(tokens: tokens(input), cache: cache)
                    .logits[0, -1].argMax().item(Int.self)
                result.append(next)
                input = [next]
            }
            return result
        }

        private func targetOnlyPagedEngine(
            _ model: GLM5NextTextModel, prompt: [Int], count: Int, id: UInt64 = 1900,
            sampling: CBv2SamplingParams = .init(temperature: 0)
        )
            async throws -> [Int]
        {
            let backend = try model.makeCBv2PagedBackend(
                kvBytesCapacity: 64 << 20,
                maximumBatchSize: 2, maximumQueryTokens: 8, maximumContextTokens: 16,
                maximumScratchBytes: 128 << 20, segmentBytes: 64 << 10, maximumBufferBytes: 32 << 20
            )
            let process = ProcessBudget()
            let engine = EngineV2(
                model: CBv2SteppableLanguageModelAdapter(model),
                layerKinds: model.cbv2LayerKinds, backend: backend,
                cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
                sampler: sampling.temperature == 0
                    ? CBv2GreedySampler() as CBv2StepSampler : CBv2DefaultSampler(),
                schedulerConfig: .init(
                    maxConcurrentRequests: 2, maxBatchedTokensPerStep: 8,
                    prefillChunkSize: 2, maxWaiting: 4, enablePrefixCache: false),
                admissionConfig: model.cbv2NativeAdmissionConfig, processMemoryOwner: process)
            let stream = try engine.submit(
                .init(id: .init(id), promptTokens: prompt, sampling: sampling, maxTokens: count))
            var result: [Int] = []
            for await event in stream {
                if case .delta(_, let values, _) = event { result += values }
            }
            await engine.shutdown()
            #expect(backend.bytesInUse == 0 && backend.bytesReserved == 0 && process.bytes == 0)
            #expect(engine.loopForTesting.recurrentStates.isEmpty)
            return result
        }

        private func nativeConfiguration(_ target: GLM5NextTextModel, dtype: DType = .float32)
            throws -> PagedKVPoolConfig
        {
            let limits = try CBv2PagedGatheredAttentionLimits(
                maximumBatchSize: 2, maximumQueryTokens: 8,
                maximumContextTokens: 16, maximumInFlightGraphs: 2, maximumScratchBytes: 128 << 20,
                admissionMode: .stepOwned(.pinnedMetal))
            return .init(
                capacityBytes: 64 << 20, dtype: dtype, maxPrefillChunk: 8,
                nominalMaxSequenceLength: 16, maxBufferLength: 32 << 20, segmentSizeBytes: 64 << 10,
                layerDTypes: Array(repeating: dtype, count: target.cbv2LayerKinds.count),
                gatheredAttention: limits)
        }

        private final class NativeHarness {
            let fixture: Fixture
            let resources: GLM5NextMTPExecution.NativeResources
            let engine: EngineV2
            let process: ProcessBudget
            init(
                projectionSign: Double, verification: GLM5NextMTPVerification = .serialTarget,
                pageSize: Int = CBv2PagedDefaults.pageSize, targetNoise: Bool = false,
                binaryHead: Bool = false, executionPolicy: GLM5NextExecutionPolicy = .reference,
                sampled: Bool = false, noiseAmplitude: Double = 0.35
            ) throws {
                fixture = try Fixture(
                    layers: 4, mtpProjectionSign: projectionSign, targetNoise: targetNoise,
                    binaryHead: binaryHead, executionPolicy: executionPolicy,
                    noiseAmplitude: noiseAmplitude)
                process = ProcessBudget()
                let factory = try GLM5NextMTPExecution(
                    target: fixture.target, predictor: fixture.mtp,
                    maximumInputs: 16, storage: .contiguous, verification: verification)
                let assistant = try GLM5NextMTPAdapter(
                    target: fixture.target, predictor: fixture.mtp, factory: factory)
                let limits = try CBv2PagedGatheredAttentionLimits(
                    maximumBatchSize: 2, maximumQueryTokens: 8,
                    maximumContextTokens: 16, maximumInFlightGraphs: 2,
                    maximumScratchBytes: 128 << 20,
                    admissionMode: .stepOwned(.pinnedMetal))
                let config = PagedKVPoolConfig(
                    pageSize: pageSize, capacityBytes: 64 << 20, dtype: .float32,
                    maxPrefillChunk: 8,
                    nominalMaxSequenceLength: 16, maxBufferLength: 32 << 20,
                    segmentSizeBytes: 64 << 10,
                    layerDTypes: Array(
                        repeating: .float32, count: fixture.target.cbv2LayerKinds.count),
                    gatheredAttention: limits)
                resources = try factory.makeNativeResources(
                    assistant: assistant, config: config,
                    processMemoryOwner: process, construction: NativeConstructionScope())
                engine = EngineV2(
                    model: resources.model, layerKinds: fixture.target.cbv2LayerKinds,
                    backend: resources.backend, cacheProvider: resources.cacheProvider,
                    sampler: sampled ? CBv2DefaultSampler() as CBv2StepSampler : CBv2GreedySampler(),
                    schedulerConfig: .init(
                        maxConcurrentRequests: 2, maxBatchedTokensPerStep: 8,
                        prefillChunkSize: 2, maxWaiting: 4, enablePrefixCache: false),
                    admissionConfig: fixture.target.cbv2NativeAdmissionConfig,
                    mtpDrafter: assistant,
                    mtpConfig: .init(
                        enabled: true, maxDraftTokens: verification.depth,
                        maxSpeculativeBatch: 1, fixedDraftTokens: verification.depth,
                        verificationMode: verification.engineMode),
                    processMemoryOwner: process, nativeCompletionTracking: true,
                    nativeExecutionContract: resources.contract)
                #expect(engine.mtpInactiveReason == nil)
                #expect(backendPrepared)
            }
            private var backendPrepared: Bool {
                resources.backend.pool.attentionWorkEnginePrepared
                    && resources.backend.pool.attentionWorkEngineRefusal == nil
            }
        }

        private func officialMemoryConfiguration() throws -> GLM5NextTextConfiguration {
            let base = try Fixture.configuration()
            var text = try #require(
                JSONSerialization.jsonObject(
                    with: JSONEncoder().encode(base.textConfig)) as? [String: Any])
            let layers = (0 ..< 45).map {
                $0 % 4 == 3 ? "deepseek_sparse_attention" : "linear_attention"
            }
            for (key, value) in [
                "hidden_size": 4096, "intermediate_size": 12288,
                "moe_intermediate_size": 2048, "num_hidden_layers": 45, "num_attention_heads": 64,
                "num_key_value_heads": 64, "qk_head_dim": 256, "qk_nope_head_dim": 256,
                "v_head_dim": 256, "q_lora_rank": 1536, "kv_lora_rank": 512, "index_head_dim": 128,
                "index_n_heads": 32, "index_topk": 2048, "first_k_dense_replace": 3,
                "n_routed_experts": 288, "num_experts_per_tok": 8, "vocab_size": 154880,
                "max_position_embeddings": 1_048_576,
            ] { text[key] = value }
            text["layer_types"] = layers
            text["mlp_layer_types"] = (0 ..< 45).map { $0 < 3 ? "dense" : "sparse" }
            text["indexer_types"] = Array(repeating: "full", count: 45)
            text["linear_attn_config"] = [
                "num_heads": 64, "head_dim": 128, "gate_lower_bound": -5,
                "short_conv_kernel_size": 4,
                "kda_layers": layers.indices.filter { layers[$0] == "linear_attention" },
                "full_attn_layers": layers.indices.filter {
                    layers[$0] == "deepseek_sparse_attention"
                },
            ]
            return try JSONDecoder().decode(
                GLM5NextTextConfiguration.self,
                from: JSONSerialization.data(withJSONObject: text))
        }

        private final class ProcessBudget: CBv2ProcessMemoryOwner, @unchecked Sendable {
            private let lock = NSLock()
            private var charge: UInt64 = 0
            private var coverage: UInt64 = 0
            private var retired = false
            private let limit: UInt64
            init(limit: UInt64 = 512 << 20) { self.limit = limit }
            var bytes: UInt64 { lock.withLock { charge } }
            func replaceCharge(_ value: UInt64) throws {
                try lock.withLock {
                    guard value <= limit, value >= coverage, !retired || value <= charge
                    else {
                        throw CBv2MTPBindingError.capacityExceeded
                    }
                    charge = value
                }
            }
            func recordMaterialization(_ value: UInt64) throws {
                try lock.withLock {
                    guard value >= coverage && value <= charge else {
                        throw CBv2MTPBindingError.capacityExceeded
                    }
                    coverage = value
                }
            }
            func withdrawCoverage(_ value: UInt64) throws {
                try lock.withLock {
                    guard value <= coverage else { throw CBv2MTPBindingError.capacityExceeded }
                    coverage -= value
                }
            }
            func retire() { lock.withLock { retired = true } }
        }

        /// A real resolved whole-request MTP reservation in AdmissionV2, with a
        /// bounded process owner. No production factory constructs a parallel ledger.
        private final class AdmittedHarness {
            let fixture: Fixture
            let factory: GLM5NextMTPExecution
            let adapter: GLM5NextMTPAdapter
            let admission: AdmissionV2
            let context: CBv2AdmittedMTPContext
            let issuer: CBv2MTPAdmissionIssuer
            let process = ProcessBudget()
            let id = CBv2RequestID(711)
            init(
                maximumInputs: Int = 8, maximumPrefillTokens: Int = 8,
                activationDType: DType = .float32,
                verification: GLM5NextMTPVerification = .serialTarget
            ) throws {
                fixture = try Fixture()
                if activationDType != .float32 {
                    // Synthetic dtype coverage only; production loading never
                    // converts a checkpoint's trained payload for this codec.
                    try fixture.target.update(
                        parameters: .unflattened(
                            fixture.target.parameters().flattened().map {
                                ($0.0, $0.1.asType(activationDType))
                            }), verify: .all)
                    try fixture.mtp.update(
                        parameters: .unflattened(
                            fixture.mtp.parameters().flattened().map {
                                ($0.0, $0.1.asType(activationDType))
                            }), verify: .all)
                }
                factory = try GLM5NextMTPExecution(
                    target: fixture.target, predictor: fixture.mtp,
                    maximumInputs: maximumInputs, storage: .contiguous,
                    verification: verification)
                adapter = try GLM5NextMTPAdapter(
                    target: fixture.target, predictor: fixture.mtp, factory: factory)
                let policy = try #require(Memory.allocationFootprintPolicy())
                let limits = CBv2MTPAllocationLimits(
                    maximumPrefillTokens: maximumPrefillTokens,
                    maximumDraftTokens: verification.depth)
                let resolution = CBv2MTPBoundedAdmission.resolve(
                    spec: adapter.boundedRequestAllocation(limits: limits), limits: limits,
                    policy: policy)
                guard case .bounded(let resolved) = resolution else {
                    throw CBv2MTPBindingError.missingAdmission
                }
                admission = AdmissionV2(
                    layerKinds: [], bytesCapacity: 512 << 20,
                    config: .init(fixedBytesPerRequest: resolved.fixedBytesPerRequest),
                    processMemoryOwner: process)
                try admission.reserve(id: id, additionalTokens: 1)
                issuer = CBv2MTPAdmissionIssuer(
                    admission: admission, target: fixture.mtp.targetIdentity,
                    drafter: ObjectIdentifier(adapter), resolved: resolved, policy: policy)
                context = try issuer.context(id: id, maximumInputs: maximumInputs)
            }
        }

        private func fence(_ adapter: GLM5NextMTPAdapter, _ state: GLM5NextMTPAdapter.State) throws
        {
            try withError { error in
                eval(adapter.evaluationTargets(for: state))
                try error.check()
            }
            try adapter.requestStateDidFinishEvaluation(state)
        }

        /// Real trained-module array execution only. Deliberately unadmitted and
        /// not a fake native-page implementation; the native graph-only owner is pending.
        private final class ReferenceFactory: GLM5NextMTPExecutionFactory {
            let model: GLM5NextMTP
            var owners = [ReferenceOwner]()
            init(_ model: GLM5NextMTP) { self.model = model }
            var targetIdentity: ObjectIdentifier { model.targetIdentity }
            var predictorIdentity: ObjectIdentifier { ObjectIdentifier(model) }
            func allocation(limits: CBv2MTPAllocationLimits) -> CBv2MTPBoundedAllocationSpec? {
                nil
            }
            func makeOwner(maximumInputs: Int) throws -> any GLM5NextMTPExecutionOwner {
                let owner = try ReferenceOwner(model, maximumInputs)
                owners.append(owner)
                return owner
            }
        }

        private final class ReferenceLease: GLM5NextMTPArrayLease {
            let arrays: [MLXArray]
            init(_ arrays: [MLXArray]) { self.arrays = arrays }
            var materializedBytes: Int { 0 }
            var hasUnmeasuredResidency: Bool { true }
        }

        private final class ReferenceCheckpoint: GLM5NextMTPExecutionCheckpoint {
            let value: GLM5NextMTPCheckpoint
            init(_ value: GLM5NextMTPCheckpoint) { self.value = value }
            var inputCount: Int { value.inputCount }
            var materializedBytes: Int { 0 }
            var evaluationTargets: [MLXArray] { [] }
        }

        private final class ReferenceOwner: GLM5NextMTPExecutionOwner {
            let model: GLM5NextMTP
            let maximumInputs: Int
            var state: GLM5NextMTPState
            var generation = 0
            var closed = false
            var failEvaluation = false
            var discarded = [ReferencePrepared]()
            var fedTokens: MLXArray?
            var fedHidden: MLXArray?
            init(_ model: GLM5NextMTP, _ maximum: Int) throws {
                self.model = model
                maximumInputs = maximum
                state = try model.newRequestState(maximumInputs: maximum)
            }
            var targetIdentity: ObjectIdentifier { model.targetIdentity }
            var predictorIdentity: ObjectIdentifier { ObjectIdentifier(model) }
            var committedInputCount: Int { state.committedInputCount }
            var materializedBytes: Int { 0 }
            var hasUnmeasuredResidency: Bool { true }
            func hold(_ arrays: [MLXArray]) throws -> any GLM5NextMTPArrayLease {
                ReferenceLease(arrays)
            }
            func prepare(shiftedTokens: MLXArray, targetHidden: MLXArray) throws
                -> any GLM5NextMTPPreparedExecution
            {
                let working = try model.newRequestState(maximumInputs: maximumInputs)
                try model.restore(model.capture(state: state), into: working)
                let output = try model.prime(
                    shiftedTokens: shiftedTokens, targetHidden: targetHidden,
                    at: committedInputCount, state: working)
                fedTokens = shiftedTokens
                fedHidden = targetHidden
                return ReferencePrepared(
                    owner: self, working: working, output: output,
                    count: shiftedTokens.dim(1), generation: generation)
            }
            func didFinishEvaluation() throws {
                if failEvaluation { throw GLM5NextMTPError.invalid("injected owner fence failure") }
                discarded.removeAll()
            }
            func checkpoint() throws -> any GLM5NextMTPExecutionCheckpoint {
                ReferenceCheckpoint(try model.capture(state: state))
            }
            func restore(_ checkpoint: any GLM5NextMTPExecutionCheckpoint) throws {
                guard let checkpoint = checkpoint as? ReferenceCheckpoint else {
                    throw GLM5NextMTPError.incompatibleState
                }
                try model.restore(checkpoint.value, into: state)
                generation += 1
            }
            func close() throws {
                try model.release(state: state)
                discarded.removeAll()
                closed = true
            }
        }

        private final class ReferencePrepared: GLM5NextMTPPreparedExecution {
            unowned let owner: ReferenceOwner
            let working: GLM5NextMTPState
            let output: GLM5NextMTPOutput
            let inputCount: Int
            let generation: Int
            var evaluated = false
            init(
                owner: ReferenceOwner, working: GLM5NextMTPState, output: GLM5NextMTPOutput,
                count: Int, generation: Int
            ) {
                self.owner = owner
                self.working = working
                self.output = output
                inputCount = count
                self.generation = generation
            }
            var evaluationTargets: [MLXArray] {
                (try? owner.model.evaluationTargets(state: working)) ?? []
            }
            func didFinishEvaluation() throws { evaluated = true }
            func commit() throws {
                guard evaluated, owner.generation == generation, !owner.closed else {
                    throw GLM5NextMTPError.incompatibleState
                }
                owner.state = working
                owner.generation += 1
            }
            func discard() throws { owner.discarded.append(self) }
        }

        private func tokens(_ ids: [Int]) -> MLXArray {
            MLXArray(ids.map(Int32.init)).reshaped([1, ids.count])
        }
        private func rows(_ vectors: [[Double]]) -> MLXArray {
            MLXArray(vectors.flatMap { $0.map(Float.init) }).reshaped([1, vectors.count, 4])
        }
        private func features(_ count: Int) -> [[Double]] {
            (0 ..< count).map { index in
                [0.5 + Double(index) * 0.1, -1, 0.75, -0.25 - Double(index) * 0.05]
            }
        }
        private func values(_ x: MLXArray) -> [Float] { x.asType(.float32).asArray(Float.self) }
        private func close(_ actual: MLXArray, _ expected: [Double]) {
            let array = values(actual)
            #expect(array.count == expected.count)
            for (a, b) in zip(array, expected) {
                #expect(a.isFinite && abs(Double(a) - b) < 1e-4)
            }
        }

        private func normalize(_ x: [Double], weight: [Double]) -> [Double] {
            let scale = 1 / sqrt(x.reduce(0) { $0 + $1 * $1 } / Double(x.count) + 1e-5)
            return zip(x, weight).map { $0 * scale * $1 }
        }
        private func matvec(_ matrix: [Double], _ x: [Double]) -> [Double] {
            (0 ..< matrix.count / x.count).map { row in
                x.indices.reduce(0) { $0 + matrix[row * x.count + $1] * x[$1] }
            }
        }
        private func project(embedding: [Double], hidden: [Double], zeroEmbedding: Bool) -> [Double]
        {
            let e = normalize(zeroEmbedding ? [0, 0, 0, 0] : embedding, weight: Fixture.enorm)
            return matvec(Fixture.eh, e + normalize(hidden, weight: Fixture.hnorm))
        }
        private func reference(shifted: [Int], hidden: [[Double]])
            -> (projected: [[Double]], hidden: [[Double]], logits: [[Double]])
        {
            var projections = [[Double]]()
            var outputs = [[Double]]()
            var logits = [[Double]]()
            var latents = [[Double]]()
            for position in shifted.indices {
                let p = project(
                    embedding: Fixture.embedding(shifted[position]),
                    hidden: hidden[position], zeroEmbedding: position == 0)
                projections.append(p)
                latents.append(latent(p))
                let step = referenceReadout(projected: p, latents: latents)
                outputs.append(step.hidden)
                logits.append(step.logits)
            }
            return (projections, outputs, logits)
        }

        private func latent(_ projected: [Double]) -> [Double] {
            let input = normalize(projected, weight: Fixture.inputNorm)
            return normalize(Array(input.prefix(2)), weight: [1, 1])
        }

        /// q=0 means uniform attention over `latents`; <=7 rows retain the whole
        /// first complete pool and its raw tail, without a dense fallback.
        private func referenceReadout(projected p: [Double], latents: [[Double]])
            -> (hidden: [Double], logits: [Double])
        {
            let mean = (0 ..< 2).map { channel in
                latents.reduce(0) { $0 + $1[channel] } / Double(latents.count)
            }
            let attended = matvec(Fixture.attentionOutput, mean)
            let residual = zip(p, attended).map(+)
            let post = normalize(residual, weight: Fixture.postNorm)
            func diagonalMLP(_ gate: [Double], _ up: [Double], _ down: [Double]) -> [Double] {
                post.indices.map { i in
                    let g = min(post[i] * gate[i], 10)
                    return down[i] * (g / (1 + exp(-g))) * min(max(post[i] * up[i], -10), 10)
                }
            }
            let expert = diagonalMLP(Fixture.expertGate, Fixture.expertUp, Fixture.expertDown)
            let shared = diagonalMLP(Fixture.sharedGate, Fixture.sharedUp, Fixture.sharedDown)
            let total = post.indices.map { residual[$0] + 2.5 * expert[$0] + shared[$0] }
            let output = normalize(total, weight: Fixture.outputNorm)
            return (output, matvec(Fixture.headWeights, output))
        }

        private final class CountingHead: Linear {
            var calls = 0
            override func callAsFunction(_ x: MLXArray) -> MLXArray {
                calls += 1
                return super.callAsFunction(x)
            }
        }
        private final class Requests { var names = [String]() }

        private struct Fixture {
            let inventory: GLM5NextWeightInventory
            let target: GLM5NextTextModel
            let mtp: GLM5NextMTP
            let head: CountingHead
            let requests: Requests
            let mtpProjectionSign: Double

            static let targetNorm = [1.5, 0.5, 1.25, 0.75]
            static let enorm = [1.0, 1.5, 0.75, 1.25]
            static let hnorm = [0.5, 1.25, 1.0, 1.5]
            static let inputNorm = [1.0, 0.75, 1.25, 0.5]
            static let postNorm = [0.8, 1.1, 0.9, 1.2]
            static let outputNorm = [1.25, 0.75, 1.5, 0.5]
            static let eh: [Double] = [
                1, 0, 0, 0, 0.25, 0, 0, 0,
                0, 0.5, 0, 0, 0, -0.5, 0, 0,
                0, 0, -0.75, 0, 0, 0, 0.75, 0,
                0, 0, 0, 1.25, 0, 0, 0, 0.25,
            ]
            static let attentionOutput = [0.2, 0.0, 0, 0.3, 0.1, -0.1, -0.15, 0.05]
            static let expertGate = [0.5, 0.75, -0.5, 0.25]
            static let expertUp = [1.0, -0.5, 0.75, 1.25]
            static let expertDown = [0.2, -0.1, 0.15, 0.1]
            static let sharedGate = [0.4, 0.3, 0.2, 0.1]
            static let sharedUp = [0.5, 0.75, 1.0, 1.25]
            static let sharedDown = [0.05, 0.06, -0.04, 0.03]
            static let headWeights: [Double] = {
                var weights: [Double] = []
                for row in 0 ..< 16 {
                    weights.append(0.03 * Double(row))
                    weights.append(-0.02 * Double(row + 1))
                    weights.append(0.01 * Double(row % 3 - 1))
                    weights.append(0.04)
                }
                return weights
            }()
            static func embedding(_ id: Int) -> [Double] {
                [
                    Double(id + 1) * 0.2, -0.3 + Double(id) * 0.03,
                    0.4 - Double(id) * 0.02, -0.1 - Double(id) * 0.01,
                ]
            }
            static var targetResidualScale: Double {
                var c = (0.25 + 1e-6) / (1 + 5e-6)
                for _ in 1 ..< 20 {
                    c /= 4 * c + 1e-6
                    c /= 4 * c + 1e-6
                }
                return pow(4 * c, 2)
            }

            /// Head rows 5 (A) and 6 (B) read only hidden channel 3, with opposite
            /// signs; every other row is zero. The zero-weight target's channel 3
            /// is negative for every token, so it always predicts B. Under this
            /// head the trained proposal always predicts B for projection sign +1
            /// and always A for -1, at any attention mixture of its latents
            /// (`binaryHeadFixtureDecidesEveryProposal` checks this with the scalar oracle).
            static let binaryHeadWeights: [Double] = (0 ..< 16).flatMap { row -> [Double] in
                [0, 0, 0, row == 5 ? 1 : row == 6 ? -1 : 0]
            }
            /// B: the zero-weight target's only greedy token under this head.
            static let binaryTargetToken = 6

            /// `targetNoise` fills otherwise-zero TARGET projections with fixed
            /// nonzero values so KDA/DSA/MoE state actually depends on history.
            /// `binaryHead` replaces the shared target/proposal head (see above).
            init(
                layers: Int = 1, mtpProjectionSign: Double = 1, targetNoise: Bool = false,
                binaryHead: Bool = false, linearHeads: Int = 1, linearHeadDimension: Int = 2,
                executionPolicy: GLM5NextExecutionPolicy = .reference,
                noiseAmplitude: Double = 0.35
            ) throws {
                let configuration = try Self.configuration(
                    layers: layers, linearHeads: linearHeads,
                    linearHeadDimension: linearHeadDimension)
                self.mtpProjectionSign = mtpProjectionSign
                let headers = try GLM5NextWeightInventory.expectedTensors(
                    configuration: configuration, storage: .float32, shard: "fixed.safetensors")
                let inventory = try GLM5NextWeightInventory.validate(
                    configuration: configuration, tensors: headers, storage: .float32)
                self.inventory = inventory
                let target = try GLM5NextTextModel(
                    configuration: configuration.textConfig,
                    inventory: inventory,
                    tensor: { name in
                        let value = try Self.tensor(
                            name, inventory: inventory, mtpProjectionSign: mtpProjectionSign)
                        return targetNoise
                            ? Self.noisy(name, value, amplitude: noiseAmplitude) : value
                    },
                    executionPolicy: executionPolicy)
                let head = CountingHead(
                    weight: MLXArray(
                        (binaryHead ? Self.binaryHeadWeights : Self.headWeights).map(Float.init)
                    ).reshaped([16, 4]))
                target.update(modules: ModuleChildren.unflattened([("lm_head", head as Module)]))
                self.target = target
                self.head = head
                let requests = Requests()
                self.requests = requests
                mtp = try GLM5NextMTP(target: target) { name in
                    requests.names.append(name)
                    return try Self.tensor(
                        name, inventory: inventory, mtpProjectionSign: mtpProjectionSign)
                }
            }

            func tensor(_ name: String) throws -> MLXArray {
                try Self.tensor(name, inventory: inventory, mtpProjectionSign: mtpProjectionSign)
            }
            /// Deterministic per-name values for zero-filled target tensors only.
            /// Norms, scales, embeddings and the head keep their fixed values.
            static func noisy(_ name: String, _ value: MLXArray, amplitude: Double = 0.35)
                -> MLXArray
            {
                guard name.hasPrefix("model.language_model.layers."), !name.contains("norm"),
                    !name.hasSuffix("_scale")
                else { return value }
                let seed = Double(name.utf8.reduce(0) { ($0 &* 31 &+ Int($1)) & 0xffff })
                let noise = (0 ..< value.size).map {
                    Float(amplitude * sin(seed + 1.7 * Double($0)))
                }
                return value + MLXArray(noise).reshaped(value.shape)
            }

            private static func diagonal(_ entries: [Double]) -> [Double] {
                (0 ..< 4).flatMap { row in (0 ..< 4).map { row == $0 ? entries[row] : 0 } }
            }
            private static func tensor(
                _ name: String, inventory: GLM5NextWeightInventory,
                mtpProjectionSign: Double
            ) throws
                -> MLXArray
            {
                guard let entry = inventory.entries[name] else {
                    throw GLM5NextLoadError.missing(name)
                }
                let count = entry.tensor.shape.reduce(1, *)
                var data = Array(repeating: 0.0, count: count)
                if name == "model.language_model.embed_tokens.weight" {
                    data = (0 ..< 16).flatMap(embedding)
                } else if name == "lm_head.weight" {
                    data = headWeights
                } else if name == "model.language_model.norm.weight" {
                    data = targetNorm
                } else if name.contains("norm.weight") || name.hasSuffix("_scale") {
                    data = Array(repeating: 1, count: count)
                }
                let p =
                    "model.language_model.layers.\(inventory.configuration.textConfig.hiddenLayers)."
                if name.hasPrefix(p) {
                    switch String(name.dropFirst(p.count)) {
                    case "enorm.weight": data = enorm
                    case "hnorm.weight": data = hnorm
                    case "input_layernorm.weight": data = inputNorm
                    case "post_attention_layernorm.weight": data = postNorm
                    case "shared_head.norm.weight": data = outputNorm
                    case "eh_proj.weight": data = eh.map { $0 * mtpProjectionSign }
                    case "self_attn.kv_a_proj_with_mqa.weight": data = [1, 0, 0, 0, 0, 1, 0, 0]
                    case "self_attn.kv_b_proj.weight": data = [0, 0, 0, 0, 1, 0, 0, 1]
                    case "self_attn.o_proj.weight": data = attentionOutput
                    case "mlp.gate.e_score_correction_bias": data = [1, 0]
                    case "mlp.experts.0.gate_proj.weight": data = diagonal(expertGate)
                    case "mlp.experts.0.up_proj.weight": data = diagonal(expertUp)
                    case "mlp.experts.0.down_proj.weight": data = diagonal(expertDown)
                    case "mlp.experts.1.gate_proj.weight": data = diagonal([1, 2, 3, 4])
                    case "mlp.experts.1.up_proj.weight": data = diagonal([4, 3, 2, 1])
                    case "mlp.experts.1.down_proj.weight": data = diagonal([1, 1, 1, 1])
                    case "mlp.shared_experts.gate_proj.weight": data = diagonal(sharedGate)
                    case "mlp.shared_experts.up_proj.weight": data = diagonal(sharedUp)
                    case "mlp.shared_experts.down_proj.weight": data = diagonal(sharedDown)
                    default: break
                    }
                }
                guard data.count == count else {
                    throw GLM5NextLoadError.invalid("fixed tensor shape")
                }
                return MLXArray(data.map(Float.init)).reshaped(entry.tensor.shape)
            }

            static func configuration(
                layers: Int = 1, linearHeads: Int = 1, linearHeadDimension: Int = 2
            ) throws -> GLM5NextConfiguration {
                let kinds = (0 ..< layers).map {
                    $0 % 4 == 3 ? "deepseek_sparse_attention" : "linear_attention"
                }
                let dense = min(layers, 3)
                let text: [String: Any] = [
                    "model_type": "glm5_next_text", "vocab_size": 16, "hidden_size": 4,
                    "intermediate_size": 8, "moe_intermediate_size": 4, "num_hidden_layers": layers,
                    "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 0,
                    "qk_head_dim": 2, "qk_nope_head_dim": 2, "qk_rope_head_dim": 0,
                    "v_head_dim": 2, "q_lora_rank": 4, "kv_lora_rank": 2, "mla_use_nope": true,
                    "layer_types": kinds,
                    "mlp_layer_types": (0 ..< layers).map { $0 < dense ? "dense" : "sparse" },
                    "indexer_types": Array(repeating: "full", count: layers),
                    "linear_attn_config": [
                        "num_heads": linearHeads,
                        "head_dim": linearHeadDimension, "gate_lower_bound": -5,
                        "short_conv_kernel_size": 4,
                        "kda_layers": kinds.indices.filter { kinds[$0] == "linear_attention" },
                        "full_attn_layers": kinds.indices.filter {
                            kinds[$0] == "deepseek_sparse_attention"
                        },
                    ],
                    "index_head_dim": 2, "index_n_heads": 1, "index_topk": 4, "index_kpool": 4,
                    "index_kpool_compress": true, "index_kpool_always_select_tail": true,
                    "indexer_rope_interleave": true, "index_share_for_mtp_iteration": true,
                    "mhc": true, "hc_mult": 4, "hc_eps": 1e-6, "hc_sinkhorn_iters": 20,
                    "first_k_dense_replace": dense, "n_routed_experts": 2, "n_shared_experts": 1,
                    "num_experts_per_tok": 1, "n_group": 1, "topk_group": 1, "norm_topk_prob": true,
                    "routed_scaling_factor": 2.5, "scoring_func": "sigmoid",
                    "topk_method": "noaux_tc",
                    "moe_router_dtype": "float32", "output_router_logits": false,
                    "router_aux_loss_coef": 0.001, "num_nextn_predict_layers": 1,
                    "hidden_act": "silu", "swiglu_limit": 10, "rms_norm_eps": 1e-5,
                    "attention_bias": false, "attention_dropout": 0, "max_position_embeddings": 32,
                    "initializer_range": 0.02, "use_cache": true, "tie_word_embeddings": false,
                    "dtype": "float32", "pad_token_id": 0, "eos_token_id": [1, 2, 3],
                ]
                let root: [String: Any] = [
                    "model_type": "glm5_next",
                    "architectures": ["Glm5NextForConditionalGeneration"],
                    "tie_word_embeddings": false, "text_config": text,
                    "image_token_id": 10, "video_token_id": 11, "image_start_token_id": 12,
                    "image_end_token_id": 13, "video_start_token_id": 14, "video_end_token_id": 15,
                    "vision_config": [
                        "model_type": "glm5_next_vision", "depth": 1, "hidden_size": 8,
                        "intermediate_size": 12, "num_heads": 2, "image_size": 8, "patch_size": 2,
                        "temporal_patch_size": 2, "spatial_merge_size": 2, "in_channels": 3,
                        "out_hidden_size": 4, "projection_intermediate_size": 8,
                        "hidden_act": "silu",
                        "swiglu_limit": 10, "rms_norm_eps": 1e-5, "attention_bias": true,
                        "attention_dropout": 0, "initializer_range": 0.02,
                    ],
                ]
                return try JSONDecoder().decode(
                    GLM5NextConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
            }
        }
    }
}
