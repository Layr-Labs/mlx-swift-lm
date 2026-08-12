import Foundation
import MLX
import Testing

@testable import MLXLMCommon

private final class QwenMTPFixtureState: CBv2MTPRequestState {
    var committedInputCount = 0
    var stagedInputCount = 0
}

private final class QwenMTPFixtureDrafter: CBv2MTPRequestStatefulDrafter {
    private final class Prepared: CBv2MTPPreparedCapture {}

    let targetIdentity: ObjectIdentifier
    let correctionOffset: Int
    private(set) var created = 0
    private(set) var released = 0
    private(set) var finalized: [Int] = []
    private(set) var committedBeforeDraft: [Int] = []
    private(set) var discarded = 0

    init(target: AnyObject, correctionOffset: Int) {
        self.targetIdentity = ObjectIdentifier(target)
        self.correctionOffset = correctionOffset
    }

    var mtpTargetIdentity: ObjectIdentifier? { targetIdentity }
    var requiredVerificationMode: CBv2MTPVerificationMode? { .serialTarget }
    var maximumDraftTokens: Int? { 1 }
    var maximumSpeculativeBatch: Int? { 1 }
    var requestStateBytesPerToken: Int { 8 }

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
    CBv2PositionedRecurrentSteppableModel, CBv2PositionedMultimodalSteppableModel
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
    var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(self) }
    var supportsRequestStatefulMTP: Bool { true }
    private(set) var hiddenPositionIDs: [[Int32]] = []

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
        let vocab = MLXArray(Int32(0) ..< Int32(32)).reshaped([1, 32])
        let targets = targetIDs.reshaped([-1, 1])
        let logits = MLX.where(vocab .== targets, 10, -10).reshaped([batch, length, 32])
        let hidden = concatenated(stateRows, axis: 0).reshaped([batch, 1, 1])
        return (logits, hidden)
    }
}

private final class QwenMTPIncompatibleTarget: QwenMTPFixtureModel {
    override var supportsRequestStatefulMTP: Bool { false }
}

@Suite("CBv2 Qwen-style request-stateful MTP", .serialized)
struct CBv2QwenMTPIntegrationTests {
    private func engine(
        correctionOffset: Int, enabled: Bool = true
    ) -> (EngineV2, QwenMTPFixtureDrafter) {
        let model = QwenMTPFixtureModel()
        let drafter = QwenMTPFixtureDrafter(
            target: model, correctionOffset: correctionOffset)
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
                sampler: CBv2GreedySampler(),
                schedulerConfig: .init(
                    maxConcurrentRequests: 2, maxBatchedTokensPerStep: 32,
                    prefillChunkSize: 8, maxWaiting: 4),
                admissionConfig: .init(watermarkFraction: 0),
                mtpDrafter: enabled ? drafter : nil,
                mtpConfig: .init(
                    enabled: enabled, maxDraftTokens: 7,
                    maxSpeculativeBatch: 8, fixedDraftTokens: 7,
                    verificationMode: .rectangular,
                    maxAutomaticRectangularTokens: 64)),
            drafter)
    }

    private func run(_ engine: EngineV2, id: UInt64 = 1) async throws -> CBv2SchedCollected {
        await cbv2SchedCollect(
            try engine.submit(
                CBv2Request(
                    id: CBv2RequestID(id), promptTokens: [2, 4, 6],
                    sampling: .init(temperature: 0), maxTokens: 10)))
    }

    @Test("policy is forced to serial B1 depth one")
    func forcedPolicy() async throws {
        let (engine, _) = engine(correctionOffset: 0)
        let driver = try #require(engine.loopForTesting.mtp)
        #expect(driver.config.verificationMode == .serialTarget)
        #expect(driver.config.maxAutomaticRectangularTokens == 0)
        #expect(driver.config.maxDraftTokens == 1)
        #expect(driver.config.fixedDraftTokens == 1)
        #expect(driver.config.maxSpeculativeBatch == 1)
        #expect(engine.admissionForTesting.auxiliaryBytesPerToken == 8)
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

    @Test("accepted and rejected drafts preserve target token authority and state")
    func acceptedRejectedParity() async throws {
        let (baseline, _) = engine(correctionOffset: 0, enabled: false)
        let expected = try await run(baseline)
        await baseline.shutdown()

        for correctionOffset in [0, 1] {
            let (mtp, drafter) = engine(correctionOffset: correctionOffset)
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
        let (engine, _) = engine(correctionOffset: 0)
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
        let (engine, drafter) = engine(correctionOffset: 0)
        let request = CBv2Request(
            id: CBv2RequestID(99), promptTokens: [1, 3, 5],
            sampling: .init(temperature: 0), maxTokens: 512)
        let stream = try engine.submit(request)
        var finish: CBv2FinishReason?
        for await event in stream {
            switch event {
            case .delta:
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
}
