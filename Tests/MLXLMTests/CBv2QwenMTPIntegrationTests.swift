import Foundation
import MLX
import Testing

@testable import MLXLMCommon

private final class QwenMTPFixtureState: CBv2MTPRequestState {
    var committedInputCount = 0
    var stagedInputCount = 0
    var materializedBytes = 0
}

private final class QwenMTPFixtureDrafter: CBv2MTPRequestStatefulDrafter {
    private final class Prepared: CBv2MTPPreparedCapture {}

    let targetIdentity: ObjectIdentifier
    let correctionOffset: Int
    let verification: CBv2MTPVerificationMode
    let targetPrefix: Bool
    private(set) var created = 0
    private(set) var released = 0
    private(set) var finalized: [Int] = []
    private(set) var committedBeforeDraft: [Int] = []
    private(set) var discarded = 0

    init(
        target: AnyObject, correctionOffset: Int,
        verification: CBv2MTPVerificationMode = .serialTarget,
        targetPrefix: Bool = false
    ) {
        self.targetIdentity = ObjectIdentifier(target)
        self.correctionOffset = correctionOffset
        self.verification = verification
        self.targetPrefix = targetPrefix
    }

    var mtpTargetIdentity: ObjectIdentifier? { targetIdentity }
    var requiredVerificationMode: CBv2MTPVerificationMode? { verification }
    var maximumDraftTokens: Int? { 1 }
    var maximumSpeculativeBatch: Int? { 1 }
    var supportsTargetPrefixAcceptance: Bool { targetPrefix }
    var requestStateBytesPerToken: Int { 8 }
    var requestStateTokenGranularity: Int { 256 }
    var requestStateTokenAllocationPadding: Int { 1 }

    func makeRequestState() -> any CBv2MTPRequestState {
        created += 1
        return QwenMTPFixtureState()
    }

    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture { Prepared() }

    func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure("request-stateful fixture used frozen capture")
    }

    func draftStep(
        tokens: MLXArray, hidden: MLXArray,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        let state = requestState as! QwenMTPFixtureState
        committedBeforeDraft.append(state.committedInputCount)
        state.stagedInputCount += 2
        let next = (
            tokens.asType(.int32).reshaped([1])
                + hidden.asType(.int32).reshaped([1])
                + Int32(1 + correctionOffset)) % Int32(32)
        return (next.reshaped([1]), hidden)
    }

    func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] { [] }

    func finalizeRound(
        requestState: any CBv2MTPRequestState, confirmedInputTokens: Int
    ) {
        let state = requestState as! QwenMTPFixtureState
        finalized.append(confirmedInputTokens)
        state.committedInputCount += confirmedInputTokens
        state.stagedInputCount = 0
    }

    func discardRound(requestState: any CBv2MTPRequestState) {
        discarded += 1
        (requestState as! QwenMTPFixtureState).stagedInputCount = 0
    }

    func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        released += 1
        let state = requestState as! QwenMTPFixtureState
        state.committedInputCount = 0
        state.stagedInputCount = 0
    }
}

private class QwenMTPFixtureModel: CBv2RecurrentMTPSteppableModel,
    CBv2PositionedRecurrentSteppableModel, CBv2PositionedMultimodalSteppableModel,
    CBv2PositionAxisProviding
{
    let cbv2Capabilities = CBv2ModelCapabilities(
        supportsPrefixReuse: false, supportsPagedKV: false,
        supportsCompiledDecode: false, supportsPackedPrefill: false,
        supportsMTP: true)
    let recurrentStateSpec: CBv2RecurrentStateSpec? = .init(layers: [
        .init(
            modelLayerIndex: 0,
            convShape: [1, 1, 1], convDType: .float32,
            ssmShape: [1, 1, 1, 1], ssmDType: .float32)
    ])
    let mtpCaptureLayers: CBv2MTPCaptureLayers? = .init(full: 0, sliding: 0)
    /// True: `forwardWithHiddenCaptured` stages per-position captured
    /// stacks (the capture-verify rectangular path).
    let captureWindows: Bool
    /// True: logits are a deterministic moderate-entropy function of
    /// (token, state) instead of a ±10 one-hot — exercises real sampling.
    let spreadLogits: Bool
    var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(self) }
    var supportsRequestStatefulMTP: Bool { true }
    var supportsCapturedVerifyWindow: Bool { captureWindows }
    var cbv2PositionAxisCount: Int? { 3 }
    private(set) var hiddenPositionIDs: [[Int32]] = []

    init(captureWindows: Bool = false, spreadLogits: Bool = false) {
        self.captureWindows = captureWindows
        self.spreadLogits = spreadLogits
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
            positionIds: nil, recordsHiddenPositions: false).logits
    }

    func forward(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        fixtureForward(
            tokens: tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds, recordsHiddenPositions: false).logits
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
            let previous = evaluation.inputState(modelLayerIndex: 0)?.conv
                ?? MLXArray.zeros([1, 1, 1])
            let next = previous + Float(length)
            try! evaluation.stage(
                modelLayerIndex: 0, conv: next,
                ssm: next.reshaped([1, 1, 1, 1]))
            stateRows.append(next)
        }

        let stateValues = concatenated(stateRows, axis: 0).asType(.int32)
            .reshaped([batch, 1])
        let targetIDs = (
            tokens.asType(.int32)
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
        let qkv = tokens.asType(.float32).reshaped([batch, 1, length, 1])
        for cache in caches {
            _ = cache.updateAndAttend(
                queries: qkv, keys: qkv, values: qkv, scale: 1, sinks: nil)
        }
        var perRowStates: [MLXArray] = []
        for evaluation in recurrentState {
            let previous = evaluation.inputState(modelLayerIndex: 0)?.conv
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
        verification: CBv2MTPVerificationMode = .serialTarget,
        targetPrefix: Bool = false,
        sampler: (any CBv2StepSampler)? = nil
    ) -> (EngineV2, QwenMTPFixtureDrafter, QwenMTPFixtureModel) {
        let model = QwenMTPFixtureModel(
            captureWindows: captureWindows, spreadLogits: spreadLogits)
        let drafter = QwenMTPFixtureDrafter(
            target: model, correctionOffset: correctionOffset,
            verification: verification, targetPrefix: targetPrefix)
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
                mtpConfig: mtpConfig ?? .init(
                    enabled: enabled, maxDraftTokens: 7,
                    maxSpeculativeBatch: 8, fixedDraftTokens: 7,
                    verificationMode: .rectangular,
                    maxAutomaticRectangularTokens: 64)),
            drafter, model)
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

    @Test("positioned causal media reaches MTP seed and serial verification")
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

        _ = await cbv2SchedCollect(
            try engine.submit(
                CBv2Request(
                    id: CBv2RequestID(405), promptTokens: [2, 4, 6],
                    sampling: .init(temperature: 0), maxTokens: 6,
                    multimodal: media)))
        await engine.shutdown()

        #expect(model.hiddenPositionIDs.count >= 3)
        let forwarded = model.hiddenPositionIDs.flatMap { $0 }
        #expect(forwarded.allSatisfy { $0 >= 103 })
        #expect(Set(forwarded).count > 1)
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
                let expected = max(
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

    @Test("capture-verify without the captured seam degrades to serial")
    func captureVerifyDegradesWithoutSeam() async throws {
        // Rectangular requested but the model lacks the captured-window
        // forward: the driver must fall back to the serial oracle instead of
        // trapping or silently running stateless.
        let (engine, _, _) = engine(
            correctionOffset: 0, captureWindows: false, verification: .rectangular)
        let driver = try #require(engine.loopForTesting.mtp)
        #expect(driver.config.verificationMode == .serialTarget)
        let result = try await run(engine, id: 601)
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()
        #expect(result.finishReason == .length)
        #expect(metrics.serialVerificationRounds > 0)
        #expect(metrics.rectangularVerificationRounds == 0)
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
}
