import Foundation
import MLX
import Testing

@testable import MLXLMCommon

private final class QwenMTPFixtureState: CBv2MTPRequestState {
    var committedInputCount = 0
    var stagedInputCount = 0
    var materializedBytes = 0
    var stagedSeeds: [MLXArray] = []
    var stagedShortlists: [MLXArray?] = []
}

private final class QwenMTPFixtureDrafter: CBv2MTPRequestStatefulDrafter {
    private final class Prepared: CBv2MTPPreparedCapture {}

    let targetIdentity: ObjectIdentifier
    let correctionOffset: Int
    let verification: CBv2MTPVerificationMode
    let targetPrefix: Bool
    let maxDraft: Int?
    private(set) var created = 0
    private(set) var released = 0
    private(set) var finalized: [Int] = []
    private(set) var committedBeforeDraft: [Int] = []
    private(set) var discarded = 0
    private(set) var observedWidths: [Int] = []
    private(set) var finalizedDraftWidths: [Int] = []

    init(
        target: AnyObject, correctionOffset: Int,
        verification: CBv2MTPVerificationMode = .serialTarget,
        targetPrefix: Bool = false, maxDraft: Int? = 1
    ) {
        self.targetIdentity = ObjectIdentifier(target)
        self.correctionOffset = correctionOffset
        self.verification = verification
        self.targetPrefix = targetPrefix
        self.maxDraft = maxDraft
    }

    var mtpTargetIdentity: ObjectIdentifier? { targetIdentity }
    var requiredVerificationMode: CBv2MTPVerificationMode? { verification }
    var maximumDraftTokens: Int? { maxDraft }
    var maximumSpeculativeBatch: Int? { 1 }
    var supportsTargetPrefixAcceptance: Bool { targetPrefix }
    var requestStateBytesPerToken: Int { 8 }
    var requestStateTokenGranularity: Int { 256 }
    var requestStateTokenAllocationPadding: Int { 1 }

    func makeRequestState() -> any CBv2MTPRequestState {
        created += 1
        return QwenMTPFixtureState()
    }
    func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState
    ) {
        observedWidths.append(observation.tokens.dim(1))
    }

    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture { Prepared() }

    func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure("request-stateful fixture used frozen capture")
    }

    /// Shortlists observed per round (nil ⇒ full-head draft) and the round
    /// seed tokens, for plumbing assertions.
    private(set) var shortlists: [[Int32]?] = []
    private(set) var seeds: [Int32] = []
    var shortlistSize: Int?
    var draftShortlistSize: Int? { shortlistSize }

    func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        let state = requestState as! QwenMTPFixtureState
        committedBeforeDraft.append(state.committedInputCount)
        state.stagedSeeds.append(tokens.asType(.int32).reshaped([1]))
        state.stagedShortlists.append(shortlist)
        state.stagedInputCount += 2
        let next =
            (tokens.asType(.int32).reshaped([1])
                + hidden.asType(.int32).reshaped([1])
                + Int32(1 + correctionOffset)) % Int32(32)
        return (next.reshaped([1]), hidden)
    }

    func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] { [] }

    func finalizeRound(
        requestState: any CBv2MTPRequestState,
        confirmedInputTokens: Int,
        committedDraftTokens: MLXArray,
        committedTargetHidden: MLXArray
    ) {
        let state = requestState as! QwenMTPFixtureState
        finalized.append(confirmedInputTokens)
        for (seed, shortlist) in zip(state.stagedSeeds, state.stagedShortlists) {
            seeds.append(seed.asArray(Int32.self)[0])
            shortlists.append(shortlist.map { $0.asArray(Int32.self) })
        }
        finalizedDraftWidths.append(committedDraftTokens.dim(1))
        state.stagedSeeds.removeAll(keepingCapacity: false)
        state.stagedShortlists.removeAll(keepingCapacity: false)
        state.committedInputCount += confirmedInputTokens
        state.stagedInputCount = 0
    }

    func discardRound(requestState: any CBv2MTPRequestState) {
        discarded += 1
        (requestState as! QwenMTPFixtureState).stagedInputCount = 0
        let state = requestState as! QwenMTPFixtureState
        state.stagedSeeds.removeAll(keepingCapacity: false)
        state.stagedShortlists.removeAll(keepingCapacity: false)
    }

    func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        released += 1
        let state = requestState as! QwenMTPFixtureState
        state.committedInputCount = 0
        state.stagedInputCount = 0
        state.stagedSeeds.removeAll(keepingCapacity: false)
        state.stagedShortlists.removeAll(keepingCapacity: false)
    }
}

private class QwenMTPFixtureModel: CBv2RecurrentMTPSteppableModel,
    CBv2PositionedRecurrentSteppableModel, CBv2PositionedMultimodalSteppableModel,
    CBv2PositionAxisProviding, CBv2MTPPolicyTopTwoProviding,
    CBv2MTPPolicyTopTwoCapabilityProviding
{
    let cbv2Capabilities: CBv2ModelCapabilities
    let recurrentStateSpec: CBv2RecurrentStateSpec? = .init(layers: [
        .init(
            modelLayerIndex: 0,
            convShape: [1, 1, 1], convDType: .float32,
            ssmShape: [1, 1, 1, 1], ssmDType: .float32)
    ])

    func cbv2MTPTopTwo(
        _ logits: MLXArray
    ) -> (ids: MLXArray, values: MLXArray) {
        topTwoCalls += 1
        let vocabulary = logits.dim(-1)
        let first = argMax(logits, axis: -1).asType(.int32)
        let vocabularyIDs = MLXArray(Int32(0) ..< Int32(vocabulary))
            .reshaped([1, 1, vocabulary])
        let masked = MLX.where(
            vocabularyIDs .== first[0..., 0..., .newAxis],
            MLXArray(-Float.infinity),
            logits)
        let second = argMax(masked, axis: -1).asType(.int32)
        let ids = stacked([first, second], axis: -1)
        return (ids, takeAlong(logits, ids, axis: -1))
    }
    let mtpCaptureLayers: CBv2MTPCaptureLayers? = .init(full: 0, sliding: 0)
    /// True: `forwardWithHiddenCaptured` stages per-position captured
    /// stacks (the capture-verify rectangular path).
    let captureWindows: Bool
    /// True: logits are a deterministic moderate-entropy function of
    /// (token, state) instead of a ±10 one-hot — exercises real sampling.
    let spreadLogits: Bool
    let compactReplay: Bool
    let cbv2MTPPolicyTopTwoAvailable: Bool
    var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(self) }
    var supportsRequestStatefulMTP: Bool { true }
    var supportsCapturedVerifyWindow: Bool { captureWindows }
    var cbv2PositionAxisCount: Int? { 3 }
    private(set) var hiddenPositionIDs: [[Int32]] = []
    private(set) var capturedVerifyWidths: [Int] = []
    private(set) var topTwoCalls = 0

    init(
        captureWindows: Bool = false, spreadLogits: Bool = false,
        compactReplay: Bool = false, policyTopTwoAvailable: Bool = true
    ) {
        self.captureWindows = captureWindows
        self.spreadLogits = spreadLogits
        self.compactReplay = compactReplay
        self.cbv2MTPPolicyTopTwoAvailable = policyTopTwoAvailable
        self.cbv2Capabilities = CBv2ModelCapabilities(
            supportsPrefixReuse: false, supportsPagedKV: false,
            supportsCompiledDecode: false, supportsPackedPrefill: false,
            supportsMTP: true,
            supportsCompactRecurrentMTPReplay: compactReplay)
    }

    func embedPromptTokens(_ tokens: MLXArray) -> MLXArray {
        tokens.asType(.float32)[0..., 0..., .newAxis]
    }

    func forward(
        tokens: MLXArray, inputEmbeddings: MLXArray,
        caches: [CBv2AttendingLayerCache]
    ) -> MLXArray {
        preconditionFailure("fixture requires recurrent multimodal forward")
    }

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        preconditionFailure("fixture requires recurrent forward")
    }

    func forwardWithHidden(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        preconditionFailure("fixture requires recurrent hidden forward")
    }

    func forward(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        fixtureForward(
            tokens: tokens, caches: caches, recurrentState: recurrentState,
            positionIds: nil, recordsHiddenPositions: false
        ).logits
    }

    func forward(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        fixtureForward(
            tokens: tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds, recordsHiddenPositions: false
        ).logits
    }

    func forward(
        tokens: MLXArray, inputEmbeddings: MLXArray,
        caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        forward(
            tokens: tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds)
    }

    func forwardWithHidden(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        fixtureForward(
            tokens: tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds, recordsHiddenPositions: true)
    }

    private func fixtureForward(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        recordsHiddenPositions: Bool
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        if recordsHiddenPositions, let positionIds {
            hiddenPositionIDs.append(positionIds[0].asArray(Int32.self))
        }
        let batch = tokens.dim(0)
        let length = tokens.dim(1)
        let qkv = tokens.asType(.float32).reshaped([batch, 1, length, 1])
        for cache in caches {
            _ = cache.updateAndAttend(
                queries: qkv, keys: qkv, values: qkv, scale: 1, sinks: nil)
        }
        var stateRows: [MLXArray] = []
        for evaluation in recurrentState {
            let previous =
                evaluation.inputState(modelLayerIndex: 0)?.conv
                ?? MLXArray.zeros([1, 1, 1])
            let next = previous + Float(length)
            try! evaluation.stage(
                modelLayerIndex: 0, conv: next,
                ssm: next.reshaped([1, 1, 1, 1]))
            stateRows.append(next)
        }

        let stateValues = concatenated(stateRows, axis: 0).asType(.int32)
            .reshaped([batch, 1])
        let targetIDs =
            (tokens.asType(.int32)
                + broadcast(stateValues, to: [batch, length])) % Int32(32)
        let logits = fixtureLogits(targetIDs: targetIDs, batch: batch, length: length)
        let hidden = concatenated(stateRows, axis: 0).reshaped([batch, 1, 1])
        return (logits, hidden)
    }

    /// Capture-verify window: per-position states `previous + s + 1` are
    /// staged as `[length, ...]` stacks; per-position logits/hidden use the
    /// per-position state so the rectangular path is semantically identical
    /// to the serial per-column oracle.
    func forwardWithHiddenCaptured(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        precondition(captureWindows, "fixture built without capture-window support")
        if let positionIds {
            hiddenPositionIDs.append(positionIds[0].asArray(Int32.self))
        }
        let batch = tokens.dim(0)
        let length = tokens.dim(1)
        capturedVerifyWidths.append(length)
        let qkv = tokens.asType(.float32).reshaped([batch, 1, length, 1])
        for cache in caches {
            _ = cache.updateAndAttend(
                queries: qkv, keys: qkv, values: qkv, scale: 1, sinks: nil)
        }
        var perRowStates: [MLXArray] = []
        for evaluation in recurrentState {
            let previous =
                evaluation.inputState(modelLayerIndex: 0)?.conv
                ?? MLXArray.zeros([1, 1, 1])
            let stack = concatenated(
                (0 ..< length).map { previous + Float($0 + 1) }, axis: 0)
            try! evaluation.stageCaptured(
                modelLayerIndex: 0, conv: stack,
                ssm: stack.reshaped([length, 1, 1, 1]), positions: length)
            perRowStates.append(stack.reshaped([1, length]))
        }
        let states = concatenated(perRowStates, axis: 0)
        let targetIDs = (tokens.asType(.int32) + states.asType(.int32)) % Int32(32)
        let logits = fixtureLogits(targetIDs: targetIDs, batch: batch, length: length)
        let hidden = states.reshaped([batch, length, 1])
        return (logits, hidden)
    }

    private func fixtureLogits(
        targetIDs: MLXArray, batch: Int, length: Int
    ) -> MLXArray {
        let vocab = MLXArray(Int32(0) ..< Int32(32)).reshaped([1, 32])
        let targets = targetIDs.reshaped([-1, 1])
        if spreadLogits {
            return ((targets * 7 + vocab * 5) % 13)
                .asType(.float32).reshaped([batch, length, 32]) * 0.5
        }
        return MLX.where(vocab .== targets, 10, -10).reshaped([batch, length, 32])
    }
}

private final class QwenMTPIncompatibleTarget: QwenMTPFixtureModel {
    override var supportsRequestStatefulMTP: Bool { false }
}

@Suite("CBv2 Qwen-style request-stateful MTP", .serialized)
struct CBv2QwenMTPIntegrationTests {
    private func engine(
        correctionOffset: Int, enabled: Bool = true,
        mtpConfig: CBv2MTPConfig? = nil,
        captureWindows: Bool = false, spreadLogits: Bool = false,
        compactReplay: Bool = false,
        verification: CBv2MTPVerificationMode = .serialTarget,
        targetPrefix: Bool = false, maxDraft: Int? = 1,
        sampler: (any CBv2StepSampler)? = nil
    ) -> (EngineV2, QwenMTPFixtureDrafter, QwenMTPFixtureModel) {
        let model = QwenMTPFixtureModel(
            captureWindows: captureWindows, spreadLogits: spreadLogits,
            compactReplay: compactReplay)
        let drafter = QwenMTPFixtureDrafter(
            target: model, correctionOffset: correctionOffset,
            verification: verification, targetPrefix: targetPrefix,
            maxDraft: maxDraft)
        let kinds = [
            CBv2LayerKind(
                attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
        ]
        return (
            EngineV2(
                model: model, layerKinds: kinds,
                backend: CBv2ContiguousKVBackend(
                    config: .init(bytesCapacity: 1 << 20, kvDType: .float32)),
                cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
                sampler: sampler ?? CBv2GreedySampler(),
                schedulerConfig: .init(
                    maxConcurrentRequests: 2, maxBatchedTokensPerStep: 32,
                    prefillChunkSize: 8, maxWaiting: 4),
                admissionConfig: .init(watermarkFraction: 0),
                mtpDrafter: enabled ? drafter : nil,
                mtpConfig: mtpConfig
                    ?? .init(
                        enabled: enabled, maxDraftTokens: 7,
                        maxSpeculativeBatch: 8, fixedDraftTokens: 7,
                        verificationMode: .rectangular,
                        maxAutomaticRectangularTokens: 64)),
            drafter, model
        )
    }

    private func run(
        _ engine: EngineV2, id: UInt64 = 1,
        sampling: CBv2SamplingParams = .init(temperature: 0),
        maxTokens: Int = 10
    ) async throws -> CBv2SchedCollected {
        await cbv2SchedCollect(
            try engine.submit(
                CBv2Request(
                    id: CBv2RequestID(id), promptTokens: [2, 4, 6],
                    sampling: sampling, maxTokens: maxTokens)))
    }

    @Test("policy is forced to serial B1 depth one")
    func forcedPolicy() async throws {
        let (engine, _, _) = engine(correctionOffset: 0)
        let driver = try #require(engine.loopForTesting.mtp)
        #expect(driver.config.verificationMode == .serialTarget)
        #expect(driver.config.maxAutomaticRectangularTokens == 0)
        #expect(driver.config.maxDraftTokens == 1)
        #expect(driver.config.fixedDraftTokens == 1)
        #expect(driver.config.maxSpeculativeBatch == 1)
        #expect(engine.admissionForTesting.auxiliaryBytesPerToken == 8)
        #expect(engine.admissionForTesting.auxiliaryTokenGranularity == 256)
        #expect(engine.admissionForTesting.auxiliaryTokenAllocationPadding == 1)
        #expect(engine.admissionForTesting.fixedBytesPerRequest == 24)
        #expect(engine.admissionForTesting.allocatedBytes(forTokens: 1) == 2_076)
        #expect(engine.admissionForTesting.allocatedBytes(forTokens: 256) == 5_144)
        #expect(engine.admissionForTesting.allocatedBytes(forTokens: 257) == 5_148)
        #expect(engine.admissionForTesting.allocatedBytes(forTokens: 511) == 6_164)
        #expect(engine.admissionForTesting.allocatedBytes(forTokens: 512) == 8_216)
        let state = QwenMTPFixtureState()
        state.materializedBytes = 1_234
        driver.restoreAssistantState(state, for: CBv2RequestID(403))
        #expect(driver.materializedAssistantBytes() == 1_234)
        let detached = QwenMTPFixtureState()
        detached.materializedBytes = 321
        #expect(driver.materializedAssistantBytes(detachedStates: [state, detached]) == 1_555)
        driver.invalidateCarry(CBv2RequestID(403))
        let id = CBv2RequestID(404)
        try engine.admissionForTesting.reserve(id: id, additionalTokens: 3)
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.publishGauges()
        }
        #expect(
            engine.capacity().kvBytesReserved
                == engine.admissionForTesting.bytesReserved)
        #expect(engine.capacity().kvBytesReserved > 3 * 8)
        engine.admissionForTesting.releaseAll(id: id)
        await engine.shutdown()
    }

    @Test("request-stateful drafter fails safe without an exact target seam")
    func incompatibleTargetFallback() async {
        let model = QwenMTPIncompatibleTarget()
        let drafter = QwenMTPFixtureDrafter(target: model, correctionOffset: 0)
        let kinds = [
            CBv2LayerKind(
                attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
        ]
        let engine = EngineV2(
            model: model, layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(
                config: .init(bytesCapacity: 1 << 20, kvDType: .float32)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            sampler: CBv2GreedySampler(),
            admissionConfig: .init(watermarkFraction: 0),
            mtpDrafter: drafter,
            mtpConfig: .init(enabled: true, fixedDraftTokens: 1))
        #expect(engine.mtpMetricsSnapshot() == nil)
        #expect(engine.mtpInactiveReason != nil)
        await engine.shutdown()
    }

    @Test("default and zero caller policies clamp to serial B1 depth <= 1")
    func forcedDefaultAndZeroPolicy() async throws {
        // Contract update (capture-verify): the driver no longer force-pins
        // `fixedDraftTokens = 1` for request-stateful recurrent drafters. A
        // serial-mode drafter still clamps depth/batch to the proven k<=1 /
        // B=1 shape, but a nil fixed depth stays adaptive (the controller
        // works within maxDraftTokens == 1) and an explicit caller 0 stays
        // an explicit target-only probe mode instead of being promoted.
        let configs: [(CBv2MTPConfig, Int?)] = [
            (CBv2MTPConfig(enabled: true), nil),
            (
                CBv2MTPConfig(
                    enabled: true, maxDraftTokens: 7, maxSpeculativeBatch: 8,
                    fixedDraftTokens: 0, verificationMode: .rectangular,
                    maxAutomaticRectangularTokens: 64), 0
            ),
        ]
        for (index, (config, expectedFixed)) in configs.enumerated() {
            let (engine, _, _) = engine(correctionOffset: 0, mtpConfig: config)
            let driver = try #require(engine.loopForTesting.mtp)
            #expect(driver.config.verificationMode == .serialTarget)
            #expect(driver.config.maxAutomaticRectangularTokens == 0)
            #expect(driver.config.maxDraftTokens == 1)
            #expect(driver.config.fixedDraftTokens == expectedFixed)
            #expect(driver.config.maxSpeculativeBatch == 1)
            _ = try await run(engine, id: UInt64(500 + index))
            await engine.shutdown()
        }
    }

    @Test("recurrent targets reject non-stateful drafters")
    func nonStatefulDrafterFallback() {
        final class FrozenDrafter: CBv2MTPDrafter {
            final class Prepared: CBv2MTPPreparedCapture {}
            let target: ObjectIdentifier
            init(_ target: AnyObject) { self.target = ObjectIdentifier(target) }
            var mtpTargetIdentity: ObjectIdentifier? { target }
            func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture { Prepared() }
            func draftStep(
                tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
            ) -> (tokens: MLXArray, hidden: MLXArray) { (tokens, hidden) }
        }

        let model = QwenMTPFixtureModel()
        #expect(
            CBv2MTPRoundDriver.build(
                model: model, drafter: FrozenDrafter(model),
                config: .init(enabled: true, fixedDraftTokens: 1)) == nil)
    }

    @Test("incompatible Qwen position axes fail before model execution")
    func incompatiblePositionAxesRejected() async {
        let (engine, _, _) = engine(correctionOffset: 0)
        let positions = CBv2PositionState(
            promptPositionIds: MLXArray.zeros([2, 1, 3], dtype: .int32),
            decodeDeltas: [0])
        let media = CBv2MultimodalInput(
            spans: [CBv2ImageSpan(tokenOffset: 1, length: 1)],
            attention: .causal,
            positionState: positions,
            embeddings: { [MLXArray.zeros([1, 1, 1])] })
        #expect(throws: CBv2MultimodalError.self) {
            _ = try engine.submit(
                CBv2Request(
                    id: CBv2RequestID(506), promptTokens: [1, 2, 3], maxTokens: 1,
                    multimodal: media, positionState: positions))
        }
        await engine.shutdown()
    }

    @Test("accepted and rejected drafts preserve target token authority and state")
    func acceptedRejectedParity() async throws {
        let (baseline, _, _) = engine(correctionOffset: 0, enabled: false)
        let expected = try await run(baseline)
        await baseline.shutdown()

        for correctionOffset in [0, 1] {
            let (mtp, drafter, _) = engine(correctionOffset: correctionOffset)
            let actual = try await run(mtp)
            let metrics = try #require(mtp.mtpMetricsSnapshot())
            await mtp.shutdown()

            #expect(actual.tokens == expected.tokens)
            #expect(metrics.rectangularVerificationRounds == 0)
            #expect(metrics.serialVerificationRounds > 0)
            #expect(drafter.created > 0)
            #expect(drafter.released == drafter.created)
            if correctionOffset == 0 {
                #expect(drafter.finalized.contains(2))
            } else {
                #expect(drafter.finalized.contains(1))
            }
            #expect(drafter.committedBeforeDraft.count > 1)
            for index in drafter.finalized.indices.dropLast() {
                #expect(
                    drafter.committedBeforeDraft[index + 1]
                        == drafter.committedBeforeDraft[index] + drafter.finalized[index])
            }
            #expect(mtp.loopForTesting.recurrentStates.isEmpty)
            #expect(mtp.loopForTesting.mtp?.requestStateCountForTesting == 0)
        }
    }

    @Test("multimodal history gate preserves stateless Gemma eligibility")
    func statelessMultimodalEligibility() {
        #expect(
            EngineLoopV2.mtpMultimodalHistoryEligible(
                hasSpans: true, tracksPersistentHistory: false))
        #expect(
            !EngineLoopV2.mtpMultimodalHistoryEligible(
                hasSpans: true, tracksPersistentHistory: true))
        #expect(
            EngineLoopV2.mtpMultimodalHistoryEligible(
                hasSpans: false, tracksPersistentHistory: true))
    }

    @Test("multimodal prompt remains target-only without trusted hidden capture")
    func positionedCausalMediaMTP() async throws {
        let model = QwenMTPFixtureModel()
        let drafter = QwenMTPFixtureDrafter(target: model, correctionOffset: 0)
        let kinds = [
            CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
        ]
        let engine = EngineV2(
            model: model, layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(
                config: .init(bytesCapacity: 1 << 20, kvDType: .float32)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 8,
                prefillChunkSize: 8, maxWaiting: 2),
            admissionConfig: .init(watermarkFraction: 0),
            mtpDrafter: drafter,
            mtpConfig: .init(enabled: true, fixedDraftTokens: 1))
        let positions = CBv2PositionState(
            promptPositionIds: MLXArray([
                Int32(0), 1, 2,
                Int32(0), 1, 2,
                Int32(0), 1, 2,
            ]).reshaped([3, 1, 3]),
            decodeDeltas: [100])
        let media = CBv2MultimodalInput(
            spans: [.init(tokenOffset: 1, length: 1)], attention: .causal,
            positionState: positions
        ) { [MLXArray.ones([1, 1, 1])] }

        let result = await cbv2SchedCollect(
            try engine.submit(
                CBv2Request(
                    id: CBv2RequestID(405), promptTokens: [2, 4, 6],
                    sampling: .init(temperature: 0), maxTokens: 6,
                    multimodal: media)))
        let metrics = try #require(engine.mtpMetricsSnapshot())
        #expect(result.finishReason == .length)
        #expect(drafter.created == 0)
        #expect(metrics.rounds == 0)
        #expect(engine.loopForTesting.mtp?.requestStateCountForTesting == 0)
        await engine.shutdown()
    }

    @Test("MTP verification can share a plan with recurrent prompt prefill")
    func mixedVerificationAndPrefill() async throws {
        let (engine, _, _) = engine(correctionOffset: 0)
        let shortStream = try engine.submit(
            CBv2Request(
                id: CBv2RequestID(406), promptTokens: [2],
                sampling: .init(temperature: 0), maxTokens: 12))
        let longStream = try engine.submit(
            CBv2Request(
                id: CBv2RequestID(407),
                promptTokens: Array(repeating: 3, count: 48),
                sampling: .init(temperature: 0), maxTokens: 2))
        async let short = cbv2SchedCollect(shortStream)
        async let long = cbv2SchedCollect(longStream)
        let (shortResult, longResult) = await (short, long)
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()

        #expect(shortResult.finishReason == .length)
        #expect(longResult.finishReason == .length)
        #expect(metrics.serialVerificationRounds > 0)
    }

    @Test("cancel releases assistant and recurrent request state")
    func cancellationCleanup() async throws {
        let (engine, drafter, _) = engine(correctionOffset: 0)
        let request = CBv2Request(
            id: CBv2RequestID(99), promptTokens: [1, 3, 5],
            sampling: .init(temperature: 0), maxTokens: 512)
        let stream = try engine.submit(request)
        var finish: CBv2FinishReason?
        for await event in stream {
            switch event {
            case .delta:
                let admission = engine.admissionForTesting
                let target = admission.targetBytesReserved(
                    partitionedBy: Set(engine.loopForTesting.kvStates.keys))
                let expected =
                    max(
                        engine.loopForTesting.backend.bytesReserved, target.materialized)
                    + target.unmaterialized + admission.nonBackendBytesReserved
                #expect(engine.loopForTesting.backend.bytesReserved > target.materialized)
                #expect(engine.capacity().kvBytesReserved == expected)
                engine.cancel(request.id)
            case .finished(let reason, _):
                finish = reason
            }
        }
        #expect(finish == .cancelled)
        await engine.shutdown()
        #expect(drafter.released == drafter.created)
        #expect(engine.loopForTesting.recurrentStates.isEmpty)
        #expect(engine.loopForTesting.mtp?.requestStateCountForTesting == 0)
    }

    @Test("preemption invalidation releases assistant history")
    func preemptionCleanup() throws {
        let model = QwenMTPFixtureModel()
        let drafter = QwenMTPFixtureDrafter(target: model, correctionOffset: 0)
        let driver = try #require(
            CBv2MTPRoundDriver.build(
                model: model, drafter: drafter,
                config: .init(enabled: true, fixedDraftTokens: 1)))
        let id = CBv2RequestID(201)
        driver.storeCarry(
            id: id, token: 7, hidden: MLXArray.zeros([1, 1, 1]),
            tokensCount: 4, kvOffset: 3)
        #expect(drafter.created == 1)
        #expect(driver.requestStateCountForTesting > 0)

        // `EngineLoopV2.handlePreemptions` calls this exact hook before
        // releasing target KV and recurrent state for a full restart.
        driver.invalidateCarry(id)
        #expect(drafter.released == 1)
        #expect(driver.requestStateCountForTesting == 0)
    }

    // MARK: - Capture-verify (rectangular recurrent verification)

    @Test("capture-verify rounds preserve target token authority and state")
    func captureVerifyParity() async throws {
        // Baseline never speculates. Fixture logits are a function of the
        // request-owned recurrent STATE at every position, so token parity
        // with the baseline proves both commit paths restore/select exactly
        // the right captured state: correctionOffset 0 drafts always match
        // (accepted commit selects position 2), offset 1 never matches
        // (rejected commit restores the post-seed snapshot).
        let (baseline, _, _) = engine(correctionOffset: 0, enabled: false)
        let expected = try await run(baseline)
        await baseline.shutdown()

        for correctionOffset in [0, 1] {
            let (mtp, drafter, _) = engine(
                correctionOffset: correctionOffset,
                captureWindows: true, verification: .rectangular)
            let driver = try #require(mtp.loopForTesting.mtp)
            #expect(driver.config.verificationMode == .rectangular)
            #expect(driver.config.maxDraftTokens == 1)
            #expect(driver.config.maxSpeculativeBatch == 1)
            let actual = try await run(mtp)
            let metrics = try #require(mtp.mtpMetricsSnapshot())
            await mtp.shutdown()

            #expect(actual.tokens == expected.tokens)
            #expect(metrics.rectangularVerificationRounds > 0)
            #expect(metrics.serialVerificationRounds == 0)
            if correctionOffset == 0 {
                #expect(metrics.acceptedTokens > 0)
            } else {
                #expect(metrics.acceptedTokens == 0)
                #expect(metrics.rounds > 0)
            }
            #expect(drafter.released == drafter.created)
            #expect(mtp.loopForTesting.recurrentStates.isEmpty)
            #expect(mtp.loopForTesting.mtp?.requestStateCountForTesting == 0)
        }
    }

    @Test("fixed depth zero is target-only with no assistant state")
    func fixedZeroParityAndNoState() async throws {
        let (baseline, _, _) = engine(correctionOffset: 0, enabled: false)
        let expected = try await run(baseline, id: 900, maxTokens: 12)
        await baseline.shutdown()

        let config = CBv2MTPConfig(
            enabled: true, maxDraftTokens: 4, maxSpeculativeBatch: 1,
            fixedDraftTokens: 0, verificationMode: .rectangular,
            maxAutomaticRectangularTokens: 8)
        let (targetOnly, drafter, model) = engine(
            correctionOffset: 0, mtpConfig: config,
            captureWindows: true, verification: .rectangular, maxDraft: 4)
        let actual = try await run(targetOnly, id: 901, maxTokens: 12)
        #expect(actual.tokens == expected.tokens)
        #expect(targetOnly.admissionForTesting.auxiliaryBytesPerToken == 0)
        #expect(drafter.created == 0)
        #expect(drafter.observedWidths.isEmpty)
        #expect(model.capturedVerifyWidths.isEmpty)
        #expect(model.topTwoCalls == 0)
        let metrics = try #require(targetOnly.mtpMetricsSnapshot())
        #expect(metrics.costInputs.isEmpty)
        #expect(targetOnly.chainedStepCount > 0)
        await targetOnly.shutdown()
    }

    @Test("fixed stateful depth does not require adaptive top-two policy")
    func fixedStatefulDepthWithoutTopTwoPolicy() {
        let model = QwenMTPFixtureModel(
            captureWindows: true, policyTopTwoAvailable: false)
        let drafter = QwenMTPFixtureDrafter(
            target: model, correctionOffset: 0, verification: .rectangular,
            maxDraft: 2)
        let fixed = CBv2MTPRoundDriver.build(
            model: model, drafter: drafter,
            config: CBv2MTPConfig(
                enabled: true, maxDraftTokens: 2, maxSpeculativeBatch: 1,
                fixedDraftTokens: 2, verificationMode: .rectangular,
                maxAutomaticRectangularTokens: 4))
        #expect(fixed?.config.fixedDraftTokens == 2)
        #expect(fixed?.config.maxDraftTokens == 2)

        let adaptive = CBv2MTPRoundDriver.build(
            model: model, drafter: drafter,
            config: CBv2MTPConfig(
                enabled: true, maxDraftTokens: 2, maxSpeculativeBatch: 1,
                fixedDraftTokens: nil, verificationMode: .rectangular,
                maxAutomaticRectangularTokens: 4))
        #expect(adaptive == nil)
    }

    @Test("fixed positive depth captures decode history while batch pressure forces zero")
    func fixedDepthTemporaryZeroCapturesHistory() async throws {
        let (baseline, _, _) = engine(correctionOffset: 0, enabled: false)
        let expected = try await run(baseline, id: 930, maxTokens: 10)
        await baseline.shutdown()

        let config = CBv2MTPConfig(
            enabled: true, maxDraftTokens: 4, maxSpeculativeBatch: 1,
            fixedDraftTokens: 2, verificationMode: .rectangular,
            maxAutomaticRectangularTokens: 8)
        let (mtp, drafter, _) = engine(
            correctionOffset: 0, mtpConfig: config,
            captureWindows: true, verification: .rectangular, maxDraft: 4)
        async let first = run(mtp, id: 931, maxTokens: 10)
        async let second = run(mtp, id: 932, maxTokens: 10)
        let values = try await (first, second)
        #expect(values.0.tokens == expected.tokens)
        #expect(values.1.tokens == expected.tokens)
        #expect(drafter.observedWidths.contains(1))
        await mtp.shutdown()
        #expect(drafter.released == drafter.created)
    }

    @Test("terminal seed cost is invalidated before same request id reuse")
    func terminalSeedCostDoesNotLeak() async throws {
        let config = CBv2MTPConfig(
            enabled: true, maxDraftTokens: 2, maxSpeculativeBatch: 1,
            fixedDraftTokens: 2, verificationMode: .rectangular,
            maxAutomaticRectangularTokens: 4)
        let (mtp, _, _) = engine(
            correctionOffset: 0, mtpConfig: config,
            captureWindows: true, verification: .rectangular, maxDraft: 2)
        let first = try await run(mtp, id: 940, maxTokens: 1)
        #expect(first.finishReason == .length)
        #expect(mtp.loopForTesting.mtp?.pendingSeedCostCountForTesting == 0)
        #expect(mtp.loopForTesting.mtp?.requestStateCountForTesting == 0)

        let reused = try await run(mtp, id: 940, maxTokens: 1)
        #expect(reused.finishReason == .length)
        #expect(mtp.loopForTesting.mtp?.pendingSeedCostCountForTesting == 0)
        #expect(mtp.loopForTesting.mtp?.requestStateCountForTesting == 0)
        await mtp.shutdown()
    }

    @Test(
        "fixed depths one through four stay target-authoritative with one target window",
        arguments: [1, 2, 3, 4])
    func fixedDepthRectangularParity(_ depth: Int) async throws {
        let (baseline, _, _) = engine(correctionOffset: 0, enabled: false)
        let expected = try await run(
            baseline, id: UInt64(910 + depth), maxTokens: 16)
        await baseline.shutdown()

        let config = CBv2MTPConfig(
            enabled: true, maxDraftTokens: 4, maxSpeculativeBatch: 1,
            fixedDraftTokens: depth, verificationMode: .rectangular,
            maxAutomaticRectangularTokens: 8)
        let (mtp, drafter, model) = engine(
            correctionOffset: depth.isMultiple(of: 2) ? 0 : 1,
            mtpConfig: config, captureWindows: true,
            verification: .rectangular, maxDraft: 4)
        let actual = try await run(
            mtp, id: UInt64(920 + depth), maxTokens: 16)
        let metrics = try #require(mtp.mtpMetricsSnapshot())
        #expect(actual.tokens == expected.tokens)
        #expect(metrics.rounds > 0)
        #expect(model.capturedVerifyWidths.count == metrics.rounds)
        #expect(
            model.capturedVerifyWidths.allSatisfy {
                (2 ... 1 + depth).contains($0)
            })
        #expect(model.capturedVerifyWidths.contains(1 + depth))
        #expect(model.topTwoCalls == metrics.rounds)
        #expect(
            mtp.admissionForTesting.fixedBytesPerRequest
                == 24 + 8 * (depth - 1))
        #expect(drafter.finalizedDraftWidths.allSatisfy { (0 ... depth).contains($0) })
        await mtp.shutdown()
        #expect(drafter.released == drafter.created)
    }

    @Test("only explicit compact replay avoids depth-dependent recurrent admission")
    func compactReplayAdmissionCapability() async throws {
        for depth in [1, 4] {
            let config = CBv2MTPConfig(
                enabled: true, maxDraftTokens: 4, maxSpeculativeBatch: 1,
                fixedDraftTokens: depth, verificationMode: .rectangular,
                maxAutomaticRectangularTokens: 8)
            let (captured, _, _) = engine(
                correctionOffset: 0, mtpConfig: config, captureWindows: true,
                compactReplay: false, verification: .rectangular, maxDraft: 4)
            let (compact, _, _) = engine(
                correctionOffset: 0, mtpConfig: config, captureWindows: true,
                compactReplay: true, verification: .rectangular, maxDraft: 4)

            #expect(
                captured.admissionForTesting.fixedBytesPerRequest
                    == 24 + 8 * (depth - 1))
            #expect(
                compact.admissionForTesting.fixedBytesPerRequest
                    == (depth >= 2 ? 32 : 24))
            await captured.shutdown()
            await compact.shutdown()
        }

        let serialConfig = CBv2MTPConfig(
            enabled: true, maxDraftTokens: 4, maxSpeculativeBatch: 1,
            fixedDraftTokens: 4, verificationMode: .serialTarget,
            maxAutomaticRectangularTokens: 8)
        let (serial, _, _) = engine(
            correctionOffset: 0, mtpConfig: serialConfig, captureWindows: true,
            compactReplay: true, verification: .serialTarget, maxDraft: 4)
        #expect(
            serial.admissionForTesting.fixedBytesPerRequest == 48,
            "serial verification retains six recurrent generations at k=4")
        await serial.shutdown()
    }

    @Test("capture-verify without the captured seam clamps to target-only")
    func captureVerifyClampsWithoutSeam() async throws {
        // A positive-depth recurrent round must never fall back to serial
        // target columns after draft construction. Missing captured-window
        // support is resolved before planning and owns no assistant state.
        let (engine, drafter, _) = engine(
            correctionOffset: 0, captureWindows: false, verification: .rectangular)
        let driver = try #require(engine.loopForTesting.mtp)
        #expect(driver.config.verificationMode == .rectangular)
        #expect(driver.config.maxDraftTokens == 0)
        #expect(driver.config.fixedDraftTokens == 0)
        #expect(engine.admissionForTesting.auxiliaryBytesPerToken == 0)
        let result = try await run(engine, id: 601)
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()
        #expect(result.finishReason == .length)
        #expect(metrics.serialVerificationRounds == 0)
        #expect(metrics.rectangularVerificationRounds == 0)
        #expect(drafter.created == 0)
    }

    @Test("uncertified cache provider clamps captured Qwen to target-only")
    func captureVerifyClampsWithoutCacheCertification() async throws {
        final class UncertifiedCacheProvider: CBv2LayerCacheProvider {
            let bank: CBv2LayerCacheBank

            init(layerKinds: [CBv2LayerKind]) {
                bank = CBv2LayerCacheBank(layerKinds: layerKinds)
            }

            func layerCaches(
                rowStates: [[CBv2SequenceKV?]]
            ) -> [CBv2AttendingLayerCache] {
                bank.layerCaches(rowStates: rowStates)
            }
        }

        let model = QwenMTPFixtureModel(captureWindows: true)
        let drafter = QwenMTPFixtureDrafter(
            target: model, correctionOffset: 0,
            verification: .rectangular, maxDraft: 4)
        let kinds = [
            CBv2LayerKind(
                attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
        ]
        let engine = EngineV2(
            model: model, layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(
                config: .init(bytesCapacity: 1 << 20, kvDType: .float32)),
            cacheProvider: UncertifiedCacheProvider(layerKinds: kinds),
            sampler: CBv2GreedySampler(),
            admissionConfig: .init(watermarkFraction: 0),
            mtpDrafter: drafter,
            mtpConfig: .init(
                enabled: true, maxDraftTokens: 4, fixedDraftTokens: 4,
                verificationMode: .rectangular))
        let driver = try #require(engine.loopForTesting.mtp)
        #expect(driver.config.maxDraftTokens == 0)
        #expect(driver.config.fixedDraftTokens == 0)
        #expect(engine.admissionForTesting.auxiliaryBytesPerToken == 0)
        let result = try await run(engine, id: 602)
        #expect(result.finishReason == .length)
        #expect(drafter.created == 0)
        await engine.shutdown()
    }

    @Test("persistent-history planner fallback is a replacement seed")
    func persistentHistoryPlannerFallbackPreservesCarryPipeline() throws {
        let model = QwenMTPFixtureModel(captureWindows: true, compactReplay: true)
        let drafter = QwenMTPFixtureDrafter(
            target: model, correctionOffset: 0,
            verification: .rectangular, targetPrefix: true, maxDraft: 2)
        let driver = try #require(
            CBv2MTPRoundDriver.build(
                model: model, drafter: drafter,
                config: .init(
                    enabled: true, maxDraftTokens: 2,
                    maxSpeculativeBatch: 1, fixedDraftTokens: 2,
                    verificationMode: .rectangular)))
        #expect(driver.tracksPersistentHistory)

        let kinds = [
            CBv2LayerKind(
                attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
        ]
        let scheduler = SchedulerV2(
            config: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 2,
                prefillChunkSize: 2, maxWaiting: 1))
        var request = CBv2SchedFixtures.request(prompt: [1, 2], maxTokens: 8)
        request.sampling.temperature = 0
        try scheduler.enqueue(request)
        CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())
        #expect(scheduler.record(for: request.id)?.isDecodeReady == true)

        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 1 << 20, kvDType: .float32))
        let loop = EngineLoopV2(
            model: model, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            sampler: CBv2GreedySampler(),
            detokenizerFactory: CBv2NullDetokenizerFactory(),
            scheduler: scheduler, capacity: nil, mtp: driver,
            config: CBv2EngineLoopConfig(),
            gauges: CBv2EngineGauges(kvBytesCapacity: backend.bytesCapacity))

        driver.beginPlan(plannedDecodeRows: 1, canSpeculate: true)
        scheduler.speculationPlanner = { rec in
            driver.markRound(rec.id, k: 2)
            return 2
        }
        let plan = scheduler.plan()
        #expect(plan.assignments.map(\.numTokens) == [1])
        #expect(plan.speculationFallbacks[request.id] == .tokenBudget)

        let step = try #require(loop.executeMTPRound(plan))
        #expect(step.mtpRound?.seedRows.map(\.id) == [request.id])
        #expect(step.mtpRound?.verify == nil)
        let metrics = driver.metricsSnapshot()
        #expect(metrics.skippedRows["token_budget"] == 1)
        #expect(metrics.controllerFallbacks["step_reservation_race"] == nil)

        let host = try #require(step.sampledTokens).asArray(Int32.self)
        #expect(host.count == 1)
        for evaluation in step.recurrentEvaluations.values {
            try evaluation.commit()
        }
        scheduler.recordSampled(id: request.id, token: Int(host[0]))
        loop.finalizeMTPRound(step)

        let advanced = try #require(scheduler.record(for: request.id))
        #expect(driver.hasValidCarry(for: advanced))
        driver.beginPlan(plannedDecodeRows: 1, canSpeculate: true)
        #expect(loop.mtpPlanSpeculation(for: advanced) == 2)
        #expect(driver.roundMark(for: request.id) == 2)
        #expect(!driver.isSeedMarked(request.id))
    }

    @Test("adaptive cost discards one cold head sample then recovers or rejects")
    func adaptiveColdCostRecovery() throws {
        func driver() throws -> CBv2MTPRoundDriver {
            let model = QwenMTPFixtureModel(captureWindows: true)
            let drafter = QwenMTPFixtureDrafter(
                target: model, correctionOffset: 0,
                verification: .rectangular, maxDraft: 4)
            return try #require(
                CBv2MTPRoundDriver.build(
                    model: model, drafter: drafter,
                    config: .init(
                        enabled: true, maxDraftTokens: 4,
                        fixedDraftTokens: nil,
                        verificationMode: .rectangular)))
        }

        func record(
            _ driver: CBv2MTPRoundDriver, depth: Int, nanos: UInt64
        ) {
            driver.recordStepCost(
                .init(
                    decision: .init(
                        depth: depth, decodeRowBucket: 1,
                        reason: depth == 0 ? "warmup_baseline" : "explore_cost",
                        isExploration: true),
                    actualDepth: depth, costEligible: true,
                    chained: false, seedOnly: false),
                wallTimeNanos: nanos,
                finalizedPlainWork: depth == 0,
                finalizedSeedIDs: [],
                finalizedVerification: depth > 0,
                claimedSeedCostNanos: 0)
        }

        let recovered = try driver()
        record(recovered, depth: 0, nanos: 100)
        recovered.beginPlan(plannedDecodeRows: 1, canSpeculate: true)
        #expect(recovered.planDecision.isExploration)
        #expect(!recovered.shouldApplyMarginalPolicyToPlan)

        record(recovered, depth: 1, nanos: 10_000)
        #expect(
            recovered.metricsSnapshot().costInputs.allSatisfy {
                $0.depth != 1
            })
        recovered.beginPlan(plannedDecodeRows: 1, canSpeculate: true)
        #expect(recovered.planDecision.depth == 1)
        #expect(recovered.planDecision.isExploration)
        #expect(!recovered.shouldApplyMarginalPolicyToPlan)

        record(recovered, depth: 1, nanos: 120)
        let recoveredCost = try #require(
            recovered.metricsSnapshot().costInputs.first { $0.depth == 1 })
        #expect(recoveredCost.samples == 1)
        #expect(recoveredCost.ewmaWallTimeNanos == 120)
        #expect(
            recovered.marginalDepth(
                for: CBv2RequestID(950), offeredDepth: 4,
                remainingTokens: 5, verificationLimit: 4,
                decodeRowBucket: 1) > 0)

        let expensive = try driver()
        record(expensive, depth: 0, nanos: 100)
        record(expensive, depth: 1, nanos: 50_000)
        record(expensive, depth: 1, nanos: 10_000)
        #expect(
            expensive.marginalDepth(
                for: CBv2RequestID(951), offeredDepth: 4,
                remainingTokens: 5, verificationLimit: 4,
                decodeRowBucket: 1) == 0)
    }

    @Test("adaptive captured-window depth supports the k equals zero through four contract")
    func adaptiveCapturedWindowDepthRange() async throws {
        // A stateful recurrent drafter may chain up to four proposals. The
        // engine's rectangular plan clamps the shared shape to this bound
        // even when the caller and drafter offer more.
        let model = QwenMTPFixtureModel(captureWindows: true)
        let drafter = QwenMTPFixtureDrafter(
            target: model, correctionOffset: 0,
            verification: .rectangular, maxDraft: nil)
        let config = CBv2MTPConfig(
            enabled: true, maxDraftTokens: 7, maxSpeculativeBatch: 8,
            fixedDraftTokens: nil, verificationMode: .rectangular,
            maxAutomaticRectangularTokens: 64)
        let driver = try #require(
            CBv2MTPRoundDriver.build(model: model, drafter: drafter, config: config))
        #expect(driver.config.verificationMode == .rectangular)
        #expect(driver.config.maxDraftTokens == 4)
        #expect(driver.config.fixedDraftTokens == nil)

        // Control: an UNCLAMPED controller under this exact pressure (flat
        // cost curve, perfect acceptance) genuinely prefers k > 1 — the
        // scenario the clamp defends against is real, not hypothetical.
        let unclamped = CBv2MTPDepthController(
            maxDepth: config.maxDraftTokens, fixedDepth: config.fixedDraftTokens)
        for depth in 0 ... 4 {
            unclamped.observeCost(
                decodeRowBucket: 1, depth: depth,
                wallTimeNanos: UInt64(100_000_000 + depth * 1_000_000))
        }
        for _ in 0 ..< 20 {
            unclamped.observeAcceptance(decodeRowBucket: 1, drafted: 4, accepted: 4)
        }
        #expect(unclamped.select(plannedDecodeRows: 1, canSpeculate: true).depth > 1)

        // The driver's request-stateful controller is built after the cap:
        // every warmup, exploration, and marginal offer stays in 0...4.
        for _ in 0 ..< 20 {
            driver.recordStepAcceptance(
                drafted: 4, accepted: 4, observedDrafts: 4, decodeRowBucket: 1)
        }
        for _ in 0 ..< 32 {
            driver.beginPlan(plannedDecodeRows: 1, canSpeculate: true)
            #expect((0 ... 4).contains(driver.planDecision.depth))
        }

        // End-to-end: adaptive capture-verify reaches completion with one
        // rectangular target window per positive-depth round.
        let (engine, adaptiveDrafter, adaptiveModel) = engine(
            correctionOffset: 0, mtpConfig: config,
            captureWindows: true, verification: .rectangular, maxDraft: nil)
        let result = try await run(engine, id: 650, maxTokens: 20)
        let metrics = try #require(engine.mtpMetricsSnapshot())
        #expect(engine.admissionForTesting.fixedBytesPerRequest == 48)
        await engine.shutdown()
        #expect(result.finishReason == .length)
        #expect(metrics.rounds > 0)
        #expect(metrics.rectangularVerificationRounds > 0)
        #expect(metrics.depthSelections.keys.allSatisfy { (0 ... 4).contains($0) })
        #expect((metrics.depthSelections[0] ?? 0) > 0)
        #expect(adaptiveDrafter.observedWidths.contains(3))
        #expect(adaptiveModel.topTwoCalls > metrics.rounds)
        #expect(adaptiveDrafter.observedWidths.contains(1))
    }

    // MARK: - target-prefix acceptance (temperature > 0)

    @Test("target-prefix keeps MTP-on output token-identical to MTP-off")
    func targetPrefixExactness() async throws {
        // Spread logits give a genuine multi-candidate distribution; the
        // fixture computes them from exact integers, so MTP-off and MTP-on
        // see bitwise-identical logits at equal (token, state) positions.
        // With per-request seeds, target-prefix pre-sampling must reproduce
        // the ordinary keyed draws exactly — greedy AND temp 0.7.
        let samplings: [CBv2SamplingParams] = [
            .init(temperature: 0, seed: 42),
            .init(temperature: 0.7, topP: 0.9, topK: 8, seed: 42),
        ]
        for (index, sampling) in samplings.enumerated() {
            let (baseline, _, _) = engine(
                correctionOffset: 0, enabled: false, spreadLogits: true,
                sampler: CBv2DefaultSampler(fallbackSeed: 1717))
            let expected = try await run(
                baseline, id: UInt64(700 + index), sampling: sampling)
            await baseline.shutdown()

            let (mtp, _, _) = engine(
                correctionOffset: 0, captureWindows: true, spreadLogits: true,
                verification: .rectangular, targetPrefix: true,
                sampler: CBv2DefaultSampler(fallbackSeed: 1717))
            // Same request id as the baseline: the keyed RNG stream is a
            // function of (seed, requestID, per-request step). Separate
            // engines, so the reuse is legal.
            let actual = try await run(
                mtp, id: UInt64(700 + index), sampling: sampling)
            let metrics = try #require(mtp.mtpMetricsSnapshot())
            await mtp.shutdown()

            #expect(actual.tokens == expected.tokens, "sampling \(sampling.temperature)")
            // Eligibility admitted the row (greedy always; temp 0.7 through
            // the lifted gate) and rounds actually ran.
            #expect(metrics.rounds > 0, "sampling \(sampling.temperature)")
        }
    }

    @Test("temperature gate stays without sampler target-prefix support")
    func temperatureGateWithoutSamplerSupport() async throws {
        // Drafter opted in, but CBv2GreedySampler cannot pre-sample verify
        // positions — non-greedy rows must remain ineligible (no rounds).
        let (engine, _, _) = engine(
            correctionOffset: 0, captureWindows: true, spreadLogits: true,
            verification: .rectangular, targetPrefix: true)
        let result = try await run(
            engine, id: 720, sampling: .init(temperature: 0.7, seed: 9))
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()
        #expect(result.finishReason == .length)
        #expect(metrics.rounds == 0)
        #expect(metrics.draftedTokens == 0)
    }
    // MARK: - Draft-head shortlist (draft trim)

    @Test("verify rounds thread coverage-gated shortlists into the next carry")
    func shortlistPlumbing() async throws {
        // One-hot ±10 fixture logits concentrate essentially all probability
        // mass on the argmax, so every verify position clears the coverage
        // gate. The first round's carry comes from a SEED step (no verify
        // shortlist); every chained round after it must receive the target
        // top-K ids captured at the carry position — which by construction
        // contain the very token the target committed there (the next
        // round's seed). Output tokens must be untouched by the plumbing.
        let (baseline, _, _) = engine(correctionOffset: 0, enabled: false)
        let expected = try await run(baseline, id: 801)
        await baseline.shutdown()

        let (mtp, drafter, _) = engine(
            correctionOffset: 0, captureWindows: true, verification: .rectangular)
        drafter.shortlistSize = 8
        let actual = try await run(mtp, id: 801)
        let metrics = try #require(mtp.mtpMetricsSnapshot())
        await mtp.shutdown()

        #expect(actual.tokens == expected.tokens)
        #expect(metrics.rectangularVerificationRounds > 0)
        #expect(drafter.shortlists.count == drafter.seeds.count)
        #expect(drafter.shortlists.count > 2)
        #expect(drafter.shortlists[0] == nil)
        for round in 1 ..< drafter.shortlists.count {
            let shortlist = try #require(drafter.shortlists[round])
            #expect(shortlist.count == 8, "round \(round)")
            // Correct column: the carry position's argmax IS this round's
            // seed token. A wrong column would only contain it by tie luck.
            #expect(shortlist.contains(drafter.seeds[round]), "round \(round)")
        }
    }

    @Test("shortlists stay off for drafters that do not opt in")
    func shortlistRequiresOptIn() async throws {
        let (mtp, drafter, _) = engine(
            correctionOffset: 0, captureWindows: true, verification: .rectangular)
        _ = try await run(mtp, id: 802)
        let metrics = try #require(mtp.mtpMetricsSnapshot())
        await mtp.shutdown()
        #expect(metrics.rectangularVerificationRounds > 0)
        #expect(drafter.shortlists.allSatisfy { $0 == nil })
    }

    @Test("shortlist coverage math: ids and parts-per-million mass")
    func shortlistCoverageMath() throws {
        // Column 0: one dominant logit — top-2 mass ≈ (e^10 + 1) / (e^10 + 7)
        // ≈ 0.999728 ⇒ clears the 0.90 gate. Column 1: flat — top-2 mass is
        // exactly 2/8 = 0.25 ⇒ must fall back to the full head.
        var values = [Float](repeating: 0, count: 8)
        values[5] = 10
        let logits = concatenated(
            [
                MLXArray(values).reshaped([1, 1, 8]),
                MLXArray.zeros([1, 1, 8]),
            ], axis: 1)
        let shortlist = try #require(
            EngineLoopV2.mtpDraftShortlist(logits: logits, size: 2))
        #expect(shortlist.ids.shape == [1, 2, 2])
        #expect(shortlist.massScaled.shape == [1, 2])
        let ids = shortlist.ids.asArray(Int32.self)
        #expect(ids[0 ..< 2].contains(5))
        let mass = shortlist.massScaled.asArray(Int32.self)
        #expect(mass[0] >= EngineLoopV2.mtpShortlistMassThresholdPPM)
        #expect(abs(Int(mass[0]) - 999_728) < 200)
        #expect(mass[1] < EngineLoopV2.mtpShortlistMassThresholdPPM)
        #expect(abs(Int(mass[1]) - 250_000) < 200)

        // A shortlist as wide as the vocabulary is refused (no byte win).
        #expect(EngineLoopV2.mtpDraftShortlist(logits: logits, size: 8) == nil)
        #expect(EngineLoopV2.mtpDraftShortlist(logits: logits, size: 0) == nil)
    }
}
