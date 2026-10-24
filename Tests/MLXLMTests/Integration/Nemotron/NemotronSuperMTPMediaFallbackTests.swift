import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Image requests on the speculating Super owner are request-scoped
/// target-only work: their causal embedding prefill and decode run the
/// ordinary adapter inside the same engine and never draft, verify, carry or
/// own assistant state, while text requests keep speculating. Greedy image
/// output must equal the non-speculative image path, and text output must not
/// change when an image request runs beside it. Synthetic weights.
@Suite(.tags(.integration))
struct NemotronSuperMTPMediaFallbackTests {
    typealias Collected = NemotronSuperMTPEngineTests.Collected
    typealias Batched = NemotronSuperBatchedEngineTests
    typealias Speculative = NemotronSuperMTPBatchedEngineTests
    typealias CoScheduled = NemotronSuperCoScheduledServingTests

    static let depth = 2
    static let rows = CoScheduled.rows
    static let mediaID = CBv2RequestID(900)
    static let textIDs = [CBv2RequestID(901), CBv2RequestID(902)]
    static let textPrompts = [Speculative.requests[0].prompt, Speculative.requests[2].prompt]
    static let textMaxTokens = 48
    static let mediaMaxTokens = 12
    /// The cancelled image request's emitted-token count at cancellation.
    static let cancelAfter = 3
    /// The placeholder run starts in the first prefill chunk and ends in the
    /// second, so one causal image span is split across chunks.
    static let mediaPrompt: [Int] =
        [5, 9, 1, 4, 22, 7] + [Int](repeating: 18, count: 4)
        + [3, 8, 30, 12, 2, 40, 41, 13, 17, 19, 23]

    /// Four image features from the fixture's own tower and projector, in the
    /// language model's embedding dtype.
    static func features(_ model: NemotronSuperModel) throws -> MLXArray {
        let pixels = MLXArray((0 ..< 64).map { Float($0 % 7) / 7 - 0.5 }).reshaped(1, 1, 8, 8)
        let dtype = model.languageModel.inputEmbeddings(MLXArray([Int32(0)])).dtype
        let features = try model.imageComposition.imageFeatures(normalizedPixels: pixels)
            .asType(dtype)
        eval(features)
        #expect(features.shape == [1, 4, NemotronSuperMTPFixture.hidden])
        return features
    }

    static func mediaRequest(_ features: MLXArray, id: CBv2RequestID = mediaID) -> CBv2Request {
        CBv2Request(
            id: id, promptTokens: mediaPrompt, sampling: .init(temperature: 0),
            maxTokens: mediaMaxTokens,
            multimodal: CBv2MultimodalInput(
                spans: [CBv2ImageSpan(tokenOffset: 6, length: 4)], attention: .causal
            ) { [features] })
    }

    static func textRequest(_ index: Int) -> CBv2Request {
        CBv2Request(
            id: textIDs[index], promptTokens: textPrompts[index],
            sampling: .init(temperature: 0), maxTokens: textMaxTokens)
    }

    /// The existing non-speculative image path: the row-exact plain adapter
    /// under the same co-scheduling policy.
    static func plainImage(_ model: NemotronSuperModel, features: MLXArray, kvDType: DType)
        async throws -> Collected
    {
        let result = try await servePlainImage(model, features: features, kvDType: kvDType)
        #expect(result.finish == .length && result.tokens.count == mediaMaxTokens)
        return result
    }

    /// `plainImage` without the expectation of a full-length answer.
    static func servePlainImage(
        _ model: NemotronSuperModel, features: MLXArray, kvDType: DType
    ) async throws -> Collected {
        let adapter = try NemotronSuperRecurrentAdapter(
            model, maximumDecodeRows: rows, decodeProducts: .rowExact)
        let (engine, backend) = CoScheduled.engine(
            adapter, kinds: adapter.cbv2LayerKinds,
            caches: adapter.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) },
            kvDType: kvDType, config: CoScheduled.policy(depth: 0, decodeSteps: 0),
            drafter: nil, depth: 0)
        let result = try await Batched.collect(try engine.submit(mediaRequest(features))) { _ in }
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
        return result
    }

    /// Each text request served alone by the serial plain adapter.
    static func soloText(_ model: NemotronSuperModel, kvDType: DType) async throws -> [Collected] {
        let (engine, backend) = NemotronSuperMTPEngineTests.plainEngine(model, kvDType: kvDType)
        var results: [Collected] = []
        for index in textIDs.indices {
            results.append(
                try await Batched.collect(try engine.submit(textRequest(index))) { _ in })
        }
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
        return results
    }

    static func speculativeEngine(
        _ stack: NemotronSuperMTPFixture.Stack, kvDType: DType,
        sampler: any CBv2StepSampler = CBv2GreedySampler()
    ) -> (EngineV2, CBv2ContiguousKVBackend) {
        CoScheduled.engine(
            stack.target, kinds: stack.target.cbv2LayerKinds,
            caches: stack.target.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) },
            kvDType: kvDType, config: CoScheduled.policy(depth: depth, decodeSteps: 0),
            drafter: stack.assistant, depth: depth, sampler: sampler)
    }

    /// Every carry the driver stores, by request, with the image request's
    /// emitted-token count at that finalize (nil once it has left).
    final class CarryLog: @unchecked Sendable {
        private let lock = NSLock()
        private var entries: [(id: CBv2RequestID, mediaTokens: Int?)] = []
        func record(_ id: CBv2RequestID, mediaTokens: Int?) {
            lock.withLock { entries.append((id, mediaTokens)) }
        }
        var stored: Set<CBv2RequestID> { lock.withLock { Set(entries.map(\.id)) } }
        /// Distinct image token counts seen while text rows stored carries.
        /// Two or more mean the image request decoded in steps whose text
        /// rows speculated: every step of a live text request is an MTP step.
        var mediaDecodeCountsBesideSpeculation: Int {
            lock.withLock { Set(entries.compactMap { $0.mediaTokens }.filter { $0 >= 1 }).count }
        }
    }

    static func observeCarries(_ loop: EngineLoopV2, into log: CarryLog) {
        loop.onEngineQueueSync {
            loop.mtp?.carryStoredObserverForTesting = { [unowned loop] carry in
                log.record(
                    carry.id,
                    mediaTokens: loop.scheduler.record(for: mediaID)?.generatedTokenCount)
            }
        }
    }

    /// The live image request's driver state, read on the engine queue. A
    /// delta may arrive after the request finished and left the driver.
    struct MediaObservation {
        var liveObservations = 0
        var targetOnly = true
        var acquiredState = false

        mutating func observe(_ loop: EngineLoopV2) {
            let id = NemotronSuperMTPMediaFallbackTests.mediaID
            loop.onEngineQueueSync {
                guard let driver = loop.mtp, loop.scheduler.record(for: id) != nil else { return }
                liveObservations += 1
                targetOnly = targetOnly && driver.isTargetOnlyMediaForTesting(id)
                if driver.assistantStateCountsForTesting(id) != nil
                    || driver.roundMark(for: id) != nil || driver.isSeedMarked(id)
                    || loop.scheduler.record(for: id).map({ driver.hasValidCarry(for: $0) })
                        == true
                {
                    acquiredState = true
                }
            }
        }
    }

    struct Case: Sendable, CustomTestStringConvertible {
        let quantized: Bool
        let cancelImage: Bool
        var kvDType: DType { quantized ? .bfloat16 : .float32 }
        var testDescription: String {
            "\(quantized ? "bf16-modelopt" : "fp32")\(cancelImage ? " cancelled image" : "")"
        }
    }

    @Test(
        .timeLimit(.minutes(5)),
        arguments: [
            Case(quantized: true, cancelImage: false), Case(quantized: true, cancelImage: true),
            Case(quantized: false, cancelImage: false),
        ])
    func imageBesideSpeculatingTextIsTargetOnlyAndExact(_ testCase: Case) async throws {
        let stack = try Speculative.stack(
            .init(
                head: .predictive, depth: Self.depth, quantized: testCase.quantized, rows: Self.rows
            ))
        let features = try Self.features(stack.model)
        let plainImage = try await Self.plainImage(
            stack.model, features: features, kvDType: testCase.kvDType)
        let soloText = try await Self.soloText(stack.model, kvDType: testCase.kvDType)

        // Text alone on the speculating owner.
        let (textOnly, textOnlyBackend) = Self.speculativeEngine(stack, kvDType: testCase.kvDType)
        var withoutImage: [Collected] = []
        try await withThrowingTaskGroup(of: (Int, Collected).self) { group in
            for index in Self.textIDs.indices {
                let stream = try textOnly.submit(Self.textRequest(index))
                group.addTask { (index, try await Batched.collect(stream) { _ in }) }
            }
            var results: [Int: Collected] = [:]
            for try await (index, collected) in group { results[index] = collected }
            withoutImage = Self.textIDs.indices.map { results[$0] ?? .init() }
        }
        await textOnly.shutdown()
        #expect(textOnlyBackend.bytesReserved == 0)

        // The same text requests with an image request beside them.
        let (engine, backend) = Self.speculativeEngine(stack, kvDType: testCase.kvDType)
        let loop = engine.loopForTesting
        let driver = try #require(loop.mtp)
        #expect(driver.supportsRequestScopedMediaFallback)
        let carries = CarryLog()
        Self.observeCarries(loop, into: carries)
        let textTasks = try Self.textIDs.indices.map { index in
            let stream = try engine.submit(Self.textRequest(index))
            return Task { try await Batched.collect(stream) { _ in } }
        }
        let (mediaStream, mediaRetirement) = try engine.submitWithRetirement(
            Self.mediaRequest(features))
        var observation = MediaObservation()
        let media = try await Batched.collect(mediaStream) { count in
            if testCase.cancelImage, count == Self.cancelAfter { engine.cancel(Self.mediaID) }
            observation.observe(loop)
        }
        var withImage: [Collected] = []
        for task in textTasks { withImage.append(try await task.value) }
        await mediaRetirement.wait()
        let metrics = try #require(engine.mtpMetricsSnapshot())
        loop.onEngineQueueSync {
            loop.mtp?.carryStoredObserverForTesting = nil
            #expect(loop.mtp?.isTargetOnlyMediaForTesting(Self.mediaID) == false)
            #expect(loop.mtp?.requestStateCountForTesting == 0)
            #expect(loop.recurrentStates.isEmpty)
        }
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)

        if testCase.cancelImage {
            #expect(media.finish == .cancelled)
            #expect(media.tokens.count >= Self.cancelAfter)
            #expect(Array(plainImage.tokens.prefix(media.tokens.count)) == media.tokens)
        } else {
            #expect(media.tokens == plainImage.tokens, "\(testCase) image tokens")
            #expect(media.finish == .length)
        }
        for index in Self.textIDs.indices {
            #expect(withImage[index].tokens == soloText[index].tokens, "\(testCase) text \(index)")
            #expect(withoutImage[index].tokens == soloText[index].tokens)
            #expect(withImage[index].finish == .length)
        }
        #expect(observation.liveObservations > 0)
        #expect(observation.targetOnly && !observation.acquiredState)
        #expect(!carries.stored.contains(Self.mediaID))
        #expect(carries.stored.isSuperset(of: Self.textIDs))
        // The image request decoded beside speculating text rows, so the
        // mixed hidden/target-only decode path ran.
        let besideSpeculation = carries.mediaDecodeCountsBesideSpeculation
        #expect(besideSpeculation >= 2)
        #expect(metrics.rounds > 0 && metrics.rectangularVerificationRounds > 0)
        #expect(metrics.acceptedTokens > 0)
        print(
            "SYNTHETIC super-mtp-media \(testCase) rounds=\(metrics.rounds) "
                + "accepted=\(metrics.acceptedTokens) emitted=\(metrics.emittedTokens) "
                + "imageTokenCountsBesideSpeculation=\(besideSpeculation)")
    }

    /// The fallback is opt-in per recurrent target: a recurrent target without
    /// `CBv2RecurrentMTPRequestScopedMediaFallback`, and one adopting only the
    /// attention-only `CBv2MTPRequestScopedMediaFallback`, report no fallback.
    @Test func recurrentTargetsWithoutTheRecurrentOptInReportNoFallback() throws {
        let stack = try NemotronSuperMTPFixture.stack(kind: .predictive)
        let optedIn = CBv2SteppableLanguageModelAdapter(stack.target)
        #expect(
            CBv2MTPRoundDriver.requestScopedMediaFallback(
                model: optedIn, drafter: stack.assistant))
        let plain = CBv2SteppableLanguageModelAdapter(NemotronSuperRecurrentAdapter(stack.model))
        #expect(plain.recurrentStateSpec != nil)
        #expect(
            !CBv2MTPRoundDriver.requestScopedMediaFallback(model: plain, drafter: stack.assistant))
        let attentionOnly = AttentionOnlyMediaFallbackTarget(optedIn)
        #expect(attentionOnly.recurrentStateSpec != nil)
        #expect(
            !CBv2MTPRoundDriver.requestScopedMediaFallback(
                model: attentionOnly, drafter: stack.assistant))
    }

    /// The serial owner serves one request at a time: an image request runs
    /// target-only and equals the plain image path, and the next text request
    /// still speculates and equals its solo output.
    @Test(.timeLimit(.minutes(3)))
    func serialTargetServesImagesTargetOnly() async throws {
        let stack = try NemotronSuperMTPFixture.stack(
            kind: .predictive, depth: Self.depth, mixerScale: 0.25, quantized: true,
            verification: .serial)
        let features = try Self.features(stack.model)
        let plainImage = try await Self.plainImage(
            stack.model, features: features, kvDType: .bfloat16)
        let soloText = try await Self.soloText(stack.model, kvDType: .bfloat16)

        let (engine, backend) = NemotronSuperMTPEngineTests.speculativeEngine(
            stack, depth: Self.depth, kvDType: .bfloat16)
        let loop = engine.loopForTesting
        let driver = try #require(loop.mtp)
        #expect(driver.supportsRequestScopedMediaFallback)
        let carries = CarryLog()
        Self.observeCarries(loop, into: carries)
        var observation = MediaObservation()
        let media = try await Batched.collect(try engine.submit(Self.mediaRequest(features))) {
            _ in observation.observe(loop)
        }
        let text = try await Batched.collect(try engine.submit(Self.textRequest(0))) { _ in }
        let metrics = try #require(engine.mtpMetricsSnapshot())
        loop.onEngineQueueSync {
            loop.mtp?.carryStoredObserverForTesting = nil
            #expect(loop.mtp?.requestStateCountForTesting == 0)
            #expect(loop.recurrentStates.isEmpty)
        }
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
        #expect(media.tokens == plainImage.tokens)
        #expect(text.tokens == soloText[0].tokens)
        #expect(observation.liveObservations > 0)
        #expect(observation.targetOnly && !observation.acquiredState)
        #expect(carries.stored == [Self.textIDs[0]])
        #expect(metrics.rounds > 0 && metrics.serialVerificationRounds == metrics.rounds)
    }
}

/// A recurrent MTP target that adopts only the attention-only fallback
/// protocol. Only its declarations are inspected; it never runs a forward.
private final class AttentionOnlyMediaFallbackTarget:
    CBv2RecurrentMTPSteppableModel, CBv2MTPRequestScopedMediaFallback
{
    let base: CBv2SteppableLanguageModelAdapter
    init(_ base: CBv2SteppableLanguageModelAdapter) { self.base = base }

    var cbv2Capabilities: CBv2ModelCapabilities { base.cbv2Capabilities }
    var recurrentStateSpec: CBv2RecurrentStateSpec? { base.recurrentStateSpec }
    var mtpCaptureLayers: CBv2MTPCaptureLayers? { nil }
    var supportsRequestStatefulMTP: Bool { true }
    var supportsMultimodalPrefill: Bool { true }
    var causalPositionRequirement: CBv2CausalPositionRequirement { .scalarCacheOffset }
    func supportsMultimodalPrefill(attention: CBv2MultimodalAttention) -> Bool { true }
    func embedPromptTokens(_ tokens: MLXArray) -> MLXArray { unused() }
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray { unused() }
    func forward(
        tokens: MLXArray, inputEmbeddings: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> MLXArray { unused() }
    func forward(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray { unused() }
    func forwardWithHidden(tokens: MLXArray, caches: [CBv2AttendingLayerCache])
        -> (logits: MLXArray, lastHidden: MLXArray)
    { unused() }
    func forwardWithHidden(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) { unused() }
    private func unused() -> Never { preconditionFailure("declaration-only test target") }
}
