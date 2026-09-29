import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

// DIRECT / UNTRACKED numerical and scheduler fixtures, including AUTO mode.
// A retained probe scope proves only its own real construction completion; it
// does not mint the SDK's strict-loaded execution contract or qualify managed
// media retirement. Observer wrappers and retained numeric carry views remain
// outside that tracked profile. Do not weaken the F1 request/issuer guards.
private func withMiMoMediaMTPConstructionScope<Value>(
    _ body: (NativeConstructionScope) throws -> Value
) rethrows -> Value {
    let work = NativeConstructionScope()
    defer {
        // Same real failed-probe ownership pattern as the current multimodal
        // tests: a genuine retained completion fault lives to test-process exit.
        if work.snapshot.isRetainedFault { _ = Unmanaged.passRetained(work) }
    }
    return try body(work)
}

/// Ordinary coexistence records metadata only. A separately enabled NUMERIC
/// cell retains one genuine seed carry/reference pair until engine drain; that
/// cell is not lifecycle/memory/performance evidence. No hot-path readback,
/// fabricated hidden row, scripted proposal or replacement logits are used.
private final class MiMoMediaMTPTrace {
    struct Row {
        let id: CBv2RequestID
        let arrival: UInt64
        let media: Bool
    }
    struct State {
        let step: Int
        let earlyDrafts: Int
        let rounds: Int
        let drafted: Int
        let liveMedia: [Row]
        let roundDepths: [CBv2RequestID: Int]
        let seedIDs: Set<CBv2RequestID>
        let marginal: Bool
    }
    struct Forward {
        let kind: String
        let rows: [Row]
        let shape: [Int]
        let state: State?
    }
    struct Sample {
        let step: Int
        let ids: [CBv2RequestID]
    }
    struct Witness {
        let step: Int
        let media: Row
        let text: Row
        let depth: Int
        let completedStep: Int
    }
    struct NumericSeed {
        let id: CBv2RequestID
        let step: Int
        let expectedHidden: MLXArray
        var sampleIDs: [CBv2RequestID] = []
        var policyValues: MLXArray?
        var stored: CBv2MTPCarryObservationForTesting?
        var actualHidden: MLXArray?
    }
    var identify: (([CBv2AttendingLayerCache]) -> [Row])?
    var inspect: (([CBv2RequestID]) -> Bool)?
    var state: (() -> State?)?
    let numeric: Bool
    private let lock = NSLock()
    private var forwards: [Forward] = [], samples: [Sample] = []
    private var cleanMediaHistory = true
    private var witnesses: [Witness] = []
    private var scalarCarries: [CBv2MTPCarryObservationForTesting] = []
    private var numericSeed: NumericSeed?
    private var compactPolicyCalls = 0
    init(numeric: Bool = false) { self.numeric = numeric }
    private func completedBoundary(_ current: State) {
        // Called on the engine queue. A round is not qualified by a mark or
        // launch: require the immediately following step's completed counters
        // and the SAME still-live, unpaused media generation.
        for verify in forwards where verify.kind == "hidden" && verify.rows.count == 1 {
            guard let before = verify.state, current.step == before.step + 1,
                let text = verify.rows.first, !text.media,
                let depth = before.roundDepths[text.id], depth > 0,
                current.rounds == before.rounds + 1, current.drafted == before.drafted + depth,
                !witnesses.contains(where: { $0.step == before.step })
            else { continue }
            for media in before.liveMedia {
                guard
                    current.liveMedia.contains(where: {
                        $0.id == media.id && $0.arrival == media.arrival
                    }),
                    let ordinary = forwards.first(where: {
                        $0.kind == "ordinary" && $0.state?.step == before.step
                            && $0.rows.contains(where: {
                                $0.id == media.id && $0.arrival == media.arrival
                            })
                    }), let ordinaryState = ordinary.state,
                    before.earlyDrafts == ordinaryState.earlyDrafts + 1,
                    samples.contains(where: { $0.step == before.step && $0.ids.contains(media.id) })
                else { continue }
                witnesses.append(
                    .init(
                        step: before.step, media: media, text: text, depth: depth,
                        completedStep: current.step))
            }
        }
    }
    @discardableResult
    func forward(_ kind: String, _ tokens: MLXArray, _ caches: [CBv2AttendingLayerCache])
        -> Forward?
    {
        guard let identify else { return nil }
        let rows = identify(caches)
        let current = state?()
        let value = Forward(kind: kind, rows: rows, shape: tokens.shape, state: current)
        lock.lock()
        if let current { completedBoundary(current) }
        forwards.append(value)
        lock.unlock()
        return value
    }
    func sample(_ ids: [CBv2RequestID], step: Int) {
        let clean = inspect?(ids) ?? true
        let current = state?()
        lock.lock()
        if let current { completedBoundary(current) }
        samples.append(.init(step: step, ids: ids))
        cleanMediaHistory = cleanMediaHistory && clean
        lock.unlock()
    }
    func hidden(_ hidden: MLXArray, frame: Forward?) {
        guard numeric, let frame, let current = frame.state, current.marginal,
            frame.kind == "hidden", frame.rows.count == 1,
            let row = frame.rows.first, current.seedIDs.contains(row.id),
            !current.liveMedia.isEmpty
        else { return }
        lock.lock()
        defer { lock.unlock() }
        if let pending = numericSeed, pending.policyValues == nil, pending.step < current.step {
            numericSeed = nil
        }
        guard numericSeed == nil else { return }
        // Reference is the actual singleton text target output, not a zero or
        // media placeholder. No additional graph operation is built here.
        numericSeed = .init(id: row.id, step: current.step, expectedHidden: hidden)
    }
    func policy(_ values: MLXArray) {
        guard let current = state?(), current.marginal else { return }
        lock.lock()
        defer { lock.unlock() }
        guard let sample = samples.last, sample.step == current.step, sample.ids.count == 2,
            current.liveMedia.contains(where: { sample.ids.contains($0.id) })
        else { return }
        compactPolicyCalls += 1
        guard numeric, numericSeed?.step == current.step, numericSeed?.policyValues == nil else {
            return
        }
        numericSeed?.sampleIDs = sample.ids
        numericSeed?.policyValues = values
    }
    func carry(_ value: CBv2MTPCarryObservationForTesting) {
        let current = state?()
        lock.lock()
        defer { lock.unlock() }
        scalarCarries.append(value)
        if numeric, numericSeed?.id == value.id, numericSeed?.stored == nil,
            numericSeed?.policyValues != nil,
            numericSeed?.step == (current?.step ?? -1) - 1, value.previousTopTwoMargin != nil
        {
            numericSeed?.stored = value
        }
    }
    func carryHidden(_ id: CBv2RequestID, _ hidden: MLXArray) {
        guard numeric else { return }
        lock.lock()
        defer { lock.unlock() }
        guard numericSeed?.id == id, numericSeed?.stored != nil, numericSeed?.actualHidden == nil
        else { return }
        numericSeed?.actualHidden = hidden
    }
    func snapshot() -> ([Forward], [Sample], Bool) {
        lock.lock()
        defer { lock.unlock() }
        return (forwards, samples, cleanMediaHistory)
    }
    func proof() -> ([Witness], [CBv2MTPCarryObservationForTesting], Int) {
        lock.lock()
        defer { lock.unlock() }
        return (witnesses, scalarCarries, compactPolicyCalls)
    }
    func takeNumericSeedAfterDrain() -> NumericSeed? {
        lock.lock()
        defer { lock.unlock() }
        let value = numericSeed
        numericSeed = nil
        return value
    }
    func clearAfterDrain() {
        lock.lock()
        numericSeed = nil
        lock.unlock()
        identify = nil
        inspect = nil
        state = nil
    }
}

/// A transparent test interposer, not another Module/parameter tree. The
/// actual MiMo adapter, loaded target identity and trained heads are unchanged.
private final class MiMoMediaMTPObservedTarget: CBv2MTPRequestScopedMediaFallback,
    CBv2MTPPrefillSteppableModel, CBv2PrefillSteppableModel,
    CBv2ModelCapabilityProviding, CBv2MTPPolicyTopTwoProviding
{
    let native: MiMoV26CBv2Adapter
    let trace: MiMoMediaMTPTrace
    init(_ native: MiMoV26CBv2Adapter, _ trace: MiMoMediaMTPTrace) {
        self.native = native
        self.trace = trace
    }
    var cbv2Capabilities: CBv2ModelCapabilities { native.cbv2Capabilities }
    var mtpCaptureLayers: CBv2MTPCaptureLayers? { native.mtpCaptureLayers }
    var mtpTargetIdentity: ObjectIdentifier? { native.mtpTargetIdentity }
    var supportsRequestStatefulMTP: Bool { native.supportsRequestStatefulMTP }
    var supportsMultimodalPrefill: Bool { native.supportsMultimodalPrefill }
    var causalPositionRequirement: CBv2CausalPositionRequirement {
        native.causalPositionRequirement
    }
    func supportsMultimodalPrefill(attention: CBv2MultimodalAttention) -> Bool {
        native.supportsMultimodalPrefill(attention: attention)
    }
    func embedPromptTokens(_ tokens: MLXArray) -> MLXArray { native.embedPromptTokens(tokens) }
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        trace.forward("ordinary", tokens, caches)
        return native.forward(tokens: tokens, caches: caches)
    }
    func forward(tokens: MLXArray, inputEmbeddings: MLXArray, caches: [CBv2AttendingLayerCache])
        -> MLXArray
    {
        trace.forward("embedding", tokens, caches)
        return native.forward(tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches)
    }
    func prefill(
        tokens: MLXArray, inputEmbeddings: MLXArray?, caches: [CBv2AttendingLayerCache],
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        trace.forward(inputEmbeddings == nil ? "prefill" : "media-prefill", tokens, caches)
        return native.prefill(
            tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches,
            requirement: requirement)
    }
    func forwardWithHidden(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> (
        logits: MLXArray, lastHidden: MLXArray
    ) {
        let frame = trace.forward("hidden", tokens, caches)
        let output = native.forwardWithHidden(tokens: tokens, caches: caches)
        trace.hidden(output.lastHidden, frame: frame)
        return output
    }
    func forwardWithHiddenForPrefill(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        trace.forward("hidden-prefill", tokens, caches)
        return native.forwardWithHiddenForPrefill(
            tokens: tokens, caches: caches, requirement: requirement)
    }
    func cbv2MTPTopTwo(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
        let output = native.cbv2MTPTopTwo(logits)
        trace.policy(output.values)
        return output
    }
}

enum MiMoMediaMTPFixture {
    static func installed(_ f: MiMoMediaFixture.Models) throws -> (
        MiMoV26MTP, MiMoV26MTPAssistant, MiMoV26CBv2Adapter
    ) {
        let predictor = try MiMoV26MTP(target: f.target)
        try predictor.loadConvertedWeights(
            Dictionary(
                uniqueKeysWithValues: predictor.parameters().flattened().map { name, array in
                    let dtype: DType =
                        name.hasSuffix("e_score_correction_bias")
                        ? .float32
                        : name.hasSuffix("mlp.gate.weight") ? .bfloat16 : f.target.activationDType
                    return ("mtp." + name, MiMoMediaFixture.values(name, array.shape, dtype))
                }))
        let assistant = try MiMoV26MTPAssistant(
            target: f.target, predictor: predictor, retaining: f.owner)
        return (
            predictor, assistant,
            try MiMoV26CBv2Adapter(target: f.target, assistant: assistant, retaining: f.owner)
        )
    }
    static func binding(
        _ f: MiMoMediaFixture.Models, _ adapter: MiMoV26CBv2Adapter, stops: Set<Int> = []
    ) -> MiMoV26CBv2Binding {
        .init(
            adapter: adapter, assistant: adapter.assistant, stopTokenIDs: stops,
            mediaGeneration: f.generation)
    }
    static func processor(
        _ f: MiMoMediaFixture.Models, _ adapter: MiMoV26CBv2Adapter,
        stops: Set<Int> = []
    ) throws -> MiMoV26MultimodalProcessor {
        try .init(
            configuration: f.target.configuration, tokenizer: f.tokenizer,
            chatTemplate: f.processor.chatTemplate, templateSHA256: f.processor.templateSHA256,
            limits: MiMoMediaFixture.limits, vision: f.vision, audioPatch: f.patch,
            adapter: adapter,
            stopTokens: stops, generation: f.generation, retaining: f.owner, audioCodec: nil)
    }
    static func prepared(
        _ f: MiMoMediaFixture.Models, _ adapter: MiMoV26CBv2Adapter,
        video: Bool = false, output: Int = 18
    ) throws -> MiMoV26PreparedMultimodal {
        let image = MiMoMediaFixture.image(70)
        let content: [MiMoV26MultimodalContent] =
            video
            ? [
                .silentVideo(
                    .init(
                        frames: [image, MiMoMediaFixture.image(110), image], timestamps: [0, 1, 2])),
                .text("tail"),
            ]
            : [.image(image)]
        let input = MiMoV26MultimodalInput(
            messages: [.init(role: .user, content: content)], maximumOutputTokens: output)
        let processor = try processor(f, adapter)
        return try processor.prepare(
            processor.plan(input), authorize: { MiMoMediaFixture.Reservation($0) })
    }
    static func engine(
        _ adapter: MiMoV26CBv2Adapter, model: (any CBv2SteppableModel)? = nil,
        sampler: (any CBv2StepSampler)? = nil, chunk: Int = 3, events: Int = 256,
        fixedDraftTokens: Int? = 2
    ) throws -> (EngineV2, MiMoV26CBv2Backend) {
        _ = try withMiMoMediaMTPConstructionScope { work in
            try adapter.probeNativeKVTypes(retaining: work)
        }
        let backend = try adapter.makeBackend(bytesCapacity: 32 << 20)
        let engine = EngineV2(
            model: model ?? adapter, layerKinds: adapter.layerKinds, backend: backend,
            cacheProvider: try adapter.makeMultimodalCacheProvider(),
            sampler: sampler ?? CBv2DefaultSampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 3, maxBatchedTokensPerStep: 8, prefillChunkSize: chunk,
                maxWaiting: 8),
            loopConfig: .init(
                requestTimeout: 40, stepTimeout: 20, eventBufferCapacity: events,
                shutdownTimeout: 5, useLegacyRequestTimeout: true),
            mtpDrafter: adapter.assistant,
            mtpConfig: .init(
                enabled: adapter.assistant != nil, maxDraftTokens: 2,
                maxSpeculativeBatch: 1, fixedDraftTokens: fixedDraftTokens,
                verificationMode: fixedDraftTokens == nil ? .automatic : .serialTarget),
            nativeCompletionTracking: false)
        return (engine, backend)
    }
    static func collect(_ stream: AsyncStream<CBv2Event>) async -> ([Int], CBv2FinishReason?) {
        var tokens: [Int] = []
        var reason: CBv2FinishReason?
        for await event in stream {
            switch event {
            case .delta(_, let ids, _): tokens += ids
            case .finished(let value, _): reason = value
            }
        }
        return (tokens, reason)
    }
    static func text(_ id: UInt64, length: Int = 13, output: Int = 18) -> CBv2Request {
        var prompt = (0 ..< length).map { 20 + $0 % 32 }
        prompt[prompt.count - 1] = 53
        return .init(
            id: .init(id), promptTokens: prompt, sampling: .init(temperature: 0), maxTokens: output,
            prefixCacheEnabled: false)
    }
}

final class MiMoV26MediaMTPFallbackTests: XCTestCase {
    func testHistoricalMultimodalEligibilityTruthTableRemainsUnchanged() {
        XCTAssertFalse(
            EngineLoopV2.mtpMultimodalHistoryEligible(hasSpans: true, tracksPersistentHistory: true)
        )
        XCTAssertTrue(
            EngineLoopV2.mtpMultimodalHistoryEligible(
                hasSpans: true, tracksPersistentHistory: false))
        XCTAssertTrue(
            EngineLoopV2.mtpMultimodalHistoryEligible(
                hasSpans: false, tracksPersistentHistory: true))
        XCTAssertTrue(
            EngineLoopV2.mtpMultimodalHistoryEligible(
                hasSpans: false, tracksPersistentHistory: false))
    }
    private func attach(_ trace: MiMoMediaMTPTrace, to engine: EngineV2) {
        // These closures execute ON the engine queue; do not sync-dispatch it.
        trace.identify = { [weak engine] caches in
            guard let loop = engine?.loopForTesting, let cache = caches.first else { return [] }
            return cache.rows.compactMap { row in
                for (id, states) in loop.kvStates where states.first.flatMap({ $0 }) === row {
                    guard let rec = loop.scheduler.record(for: id) else { continue }
                    return .init(
                        id: id, arrival: rec.arrivalSeq,
                        media: rec.request.multimodal?.spans.isEmpty == false)
                }
                return nil
            }
        }
        trace.inspect = { [weak engine] ids in
            guard let loop = engine?.loopForTesting, let driver = loop.mtp else { return false }
            for id in ids {
                guard let rec = loop.scheduler.record(for: id),
                    rec.request.multimodal?.spans.isEmpty == false
                else { continue }
                if !driver.isTargetOnlyMediaForTesting(id)
                    || driver.assistantStateCountsForTesting(id) != nil
                    || driver.hasValidCarry(for: rec) || driver.isSeedMarked(id)
                    || driver.roundMark(for: id) != nil
                {
                    return false
                }
            }
            return true
        }
        trace.state = { [weak engine] in
            guard let loop = engine?.loopForTesting, let driver = loop.mtp else { return nil }
            let rows = loop.scheduler.running
            let liveMedia = rows.filter {
                !$0.isPaused && !$0.cancelRequested && $0.request.multimodal?.spans.isEmpty == false
            }.map { MiMoMediaMTPTrace.Row(id: $0.id, arrival: $0.arrivalSeq, media: true) }
            let metrics = driver.metricsSnapshot()
            return .init(
                step: loop.stepCount, earlyDrafts: metrics.earlyDraftSubmissions,
                rounds: metrics.rounds, drafted: metrics.draftedTokens, liveMedia: liveMedia,
                roundDepths: Dictionary(
                    uniqueKeysWithValues: rows.compactMap { rec in
                        driver.roundMark(for: rec.id).map { (rec.id, $0) }
                    }), seedIDs: Set(rows.filter { driver.isSeedMarked($0.id) }.map(\.id)),
                marginal: driver.usesMarginalPolicy)
        }
        engine.loopForTesting.onEngineQueueSync {
            let driver = engine.loopForTesting.mtp
            XCTAssertNotNil(driver, "the real accepted sampler must leave native MTP active")
            driver?.carryStoredObserverForTesting = { [weak trace] in trace?.carry($0) }
            if trace.numeric {
                driver?.carryHiddenObserverForTesting = { [weak trace] in trace?.carryHidden($0, $1)
                }
            }
        }
    }
    private func detachAfterDrain(
        _ trace: MiMoMediaMTPTrace, engine: EngineV2, sampler: CBv2DefaultSampler
    ) {
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.mtp?.carryStoredObserverForTesting = nil
            engine.loopForTesting.mtp?.carryHiddenObserverForTesting = nil
            sampler.sampleObserverForTesting = nil
        }
        trace.clearAfterDrain()
    }
    /// Separate call boundary releases the single retained numerical pair
    /// before the caller makes any subsequent lifecycle assertions.
    private func assertNumericMappingAfterDrain(
        _ trace: MiMoMediaMTPTrace, dtype: String,
        mediaFirst: Bool, mediaID: CBv2RequestID, textID: CBv2RequestID, hiddenSize: Int
    ) throws {
        let seed = try XCTUnwrap(trace.takeNumericSeedAfterDrain())
        let stored = try XCTUnwrap(seed.stored)
        let actualHidden = try XCTUnwrap(seed.actualHidden)
        let values = try XCTUnwrap(seed.policyValues)
        XCTAssertEqual(seed.sampleIDs, mediaFirst ? [mediaID, textID] : [textID, mediaID])
        XCTAssertEqual(values.shape, [1, 2, 2])
        XCTAssertEqual(stored.hiddenShape, [1, 1, hiddenSize])
        XCTAssertEqual(actualHidden.shape, seed.expectedHidden.shape)
        eval(values, actualHidden, seed.expectedHidden)
        let table = values.asArray(Float.self)
        let textIndex = mediaFirst ? 1 : 0
        let mediaIndex = 1 - textIndex
        let textMargin = Double(table[textIndex * 2] - table[textIndex * 2 + 1])
        let mediaMargin = Double(table[mediaIndex * 2] - table[mediaIndex * 2 + 1])
        XCTAssertEqual(stored.previousTopTwoMargin, textMargin)
        XCTAssertNotEqual(
            textMargin, mediaMargin, "native scores must discriminate a wrong policy gather")
        let actualValues = actualHidden.asArray(Float.self)
        let expectedValues = seed.expectedHidden.asArray(Float.self)
        XCTAssertEqual(
            actualValues, expectedValues,
            "stored carry must use the genuine text hidden, not a same-shaped media placeholder")
        print(
            "MIMO_MEDIA_ADAPTIVE_MAPPING dtype=\(dtype) media_first=\(mediaFirst) step=\(seed.step) text_margin=\(textMargin) media_margin=\(mediaMargin) hidden_exact=\(actualValues == expectedValues)"
        )
    }
    func testRealMediaOFFONParityWithTextDraftingAndOneOrderedSamplerInvocation() async throws {
        for dtype in ["float32", "bfloat16"] {
            for mediaFirst in [true, false] {
                try await mixedWitness(
                    dtype: dtype, mediaFirst: mediaFirst, fixedDraftTokens: 2, numeric: false)
            }
        }
    }
    func testAdaptiveCompactPolicyAndHiddenMappingUseRealSerialMiMoInBothRowOrders() async throws {
        // Raw references are enabled ONLY in this numerical mapping cell.
        // It is not lifecycle/residency/peak/timing/performance evidence.
        for dtype in ["float32", "bfloat16"] {
            for mediaFirst in [true, false] {
                try await mixedWitness(
                    dtype: dtype, mediaFirst: mediaFirst, fixedDraftTokens: nil, numeric: true)
            }
        }
    }
    func testTwoGenuinelyEligibleTextRowsBesideLiveMediaKeepNativeHeadBatchOneGate() async throws {
        for mediaFirst in [true, false] {
            let f = try MiMoMediaFixture.models()
            let (_, _, on) = try MiMoMediaMTPFixture.installed(f)
            let offPrepared = try MiMoMediaMTPFixture.prepared(f, f.adapter)
            let onPrepared = try MiMoMediaMTPFixture.prepared(f, on)
            let mediaID = CBv2RequestID(81)
            let aID = CBv2RequestID(82)
            let bID = CBv2RequestID(83)
            let a = MiMoMediaMTPFixture.text(82, length: offPrepared.plan.promptTokens.count)
            let b = MiMoMediaMTPFixture.text(83, length: offPrepared.plan.promptTokens.count)
            let trace = MiMoMediaMTPTrace()
            let observed = MiMoMediaMTPObservedTarget(on, trace)
            let sampler = CBv2DefaultSampler()
            sampler.sampleObserverForTesting = { [weak trace] in trace?.sample($0, step: $1) }
            let (off, offBackend) = try MiMoMediaMTPFixture.engine(f.adapter, chunk: 1)
            var startedEngine: EngineV2?
            func run(
                _ value: EngineV2, _ prepared: MiMoV26PreparedMultimodal,
                _ binding: MiMoV26CBv2Binding
            ) async throws -> [[Int]] {
                value.loopForTesting.onEngineQueueSync {
                    value.loopForTesting.suspendStepExecutionAtCountForTesting = 0
                }
                let media = try prepared.makeRequest(
                    binding: binding, id: mediaID, sampling: .init(temperature: 0))
                let requests = mediaFirst ? [media, a, b] : [a, media, b]
                let streams = try requests.map { try value.submit($0) }
                value.loopForTesting.onEngineQueueSync {
                    XCTAssertEqual(value.loopForTesting.stepCount, 0)
                    for id in [mediaID, aID, bID] {
                        XCTAssertNotNil(value.loopForTesting.scheduler.record(for: id))
                    }
                    value.loopForTesting.suspendStepExecutionAtCountForTesting = nil
                }
                async let one = MiMoMediaMTPFixture.collect(streams[0])
                async let two = MiMoMediaMTPFixture.collect(streams[1])
                async let three = MiMoMediaMTPFixture.collect(streams[2])
                let results = await (one, two, three)
                let all = [results.0, results.1, results.2]
                for result in all { XCTAssertEqual(result.1, .length) }
                return all.map { $0.0 }
            }
            do {
                let (engine, backend) = try MiMoMediaMTPFixture.engine(
                    on, model: observed, sampler: sampler, chunk: 1)
                startedEngine = engine
                attach(trace, to: engine)
                let expected = try await run(off, offPrepared, f.binding)
                await off.shutdown()
                XCTAssertEqual(offBackend.bytesReserved, 0)
                let actual = try await run(engine, onPrepared, MiMoMediaMTPFixture.binding(f, on))
                XCTAssertEqual(actual, expected)
                let metrics = try XCTUnwrap(engine.mtpMetricsSnapshot())
                XCTAssertTrue(metrics.active)
                XCTAssertEqual(metrics.draftedTokens, 0)
                let (forwards, samples, clean) = trace.snapshot()
                XCTAssertTrue(clean)
                XCTAssertTrue(
                    forwards.contains {
                        $0.kind == "hidden" && $0.rows.count == 2
                            && Set($0.rows.map(\.id)) == Set([aID, bID])
                    }, "both real eligible text rows must share target work beside media")
                let shared = samples.filter { Set($0.ids) == Set([mediaID, aID, bID]) }
                XCTAssertFalse(shared.isEmpty)
                for sample in shared {
                    XCTAssertEqual(
                        sample.ids, mediaFirst ? [mediaID, aID, bID] : [aID, mediaID, bID])
                }
                for calls in Dictionary(
                    grouping: samples.filter { $0.ids.contains(mediaID) }, by: \.step
                ).values { XCTAssertEqual(calls.count, 1) }
                XCTAssertTrue(trace.proof().1.allSatisfy { $0.id != mediaID })
                await engine.shutdown()
                detachAfterDrain(trace, engine: engine, sampler: sampler)
                XCTAssertEqual(backend.bytesReserved, 0)
                XCTAssertTrue(
                    engine.loopForTesting.onEngineQueueSync {
                        let loop = engine.loopForTesting
                        return loop.mtp?.carryHiddenObserverForTesting == nil
                            && [mediaID, aID, bID].allSatisfy {
                                loop.mtp?.assistantStateCountsForTesting($0) == nil
                            }
                    })
            } catch {
                off.loopForTesting.onEngineQueueSync {
                    off.loopForTesting.suspendStepExecutionAtCountForTesting = nil
                }
                await off.shutdown()
                if let engine = startedEngine {
                    engine.loopForTesting.onEngineQueueSync {
                        engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil
                    }
                    await engine.shutdown()
                    detachAfterDrain(trace, engine: engine, sampler: sampler)
                } else {
                    sampler.sampleObserverForTesting = nil
                    trace.clearAfterDrain()
                }
                throw error
            }
        }
    }
    private func mixedWitness(
        dtype: String, mediaFirst: Bool, fixedDraftTokens: Int?, numeric: Bool
    ) async throws {
        let f = try MiMoMediaFixture.models(dtype)
        let (_, assistant, on) = try MiMoMediaMTPFixture.installed(f)
        XCTAssertTrue(on.target === f.adapter.target)
        XCTAssertEqual(assistant.mtpTargetIdentity, ObjectIdentifier(f.target))
        let output = fixedDraftTokens == nil ? 48 : 18
        let offPrepared = try MiMoMediaMTPFixture.prepared(f, f.adapter, output: output)
        let onPrepared = try MiMoMediaMTPFixture.prepared(f, on, output: output)
        let mediaID = CBv2RequestID(1)
        let textID = CBv2RequestID(2)
        let text = MiMoMediaMTPFixture.text(
            2, length: offPrepared.plan.promptTokens.count, output: output)
        let windows = on.layerKinds.compactMap { kind -> Int? in
            if case .slidingWindow(let window) = kind.attention { return window }
            return nil
        }
        XCTAssertGreaterThan(offPrepared.plan.promptTokens.count, try XCTUnwrap(windows.max()))
        XCTAssertTrue(
            on.layerKinds.contains {
                if case .full = $0.attention { return true }
                return false
            })
        let (off, offBackend) = try MiMoMediaMTPFixture.engine(f.adapter)
        off.loopForTesting.onEngineQueueSync {
            off.loopForTesting.suspendStepExecutionAtCountForTesting = 0
        }
        do {
            let media = try off.submit(
                offPrepared.makeRequest(
                    binding: f.binding, id: mediaID, sampling: .init(temperature: 0)))
            let ordinary = try off.submit(text)
            off.loopForTesting.onEngineQueueSync {
                XCTAssertEqual(off.loopForTesting.stepCount, 0)
                XCTAssertNotNil(off.loopForTesting.scheduler.record(for: mediaID))
                XCTAssertNotNil(off.loopForTesting.scheduler.record(for: textID))
                off.loopForTesting.suspendStepExecutionAtCountForTesting = nil
            }
            async let mediaValue = MiMoMediaMTPFixture.collect(media)
            async let textValue = MiMoMediaMTPFixture.collect(ordinary)
            let expected = await (mediaValue, textValue)
            await off.shutdown()
            XCTAssertEqual(offBackend.bytesReserved, 0)
            XCTAssertEqual(expected.0.1, .length)
            XCTAssertEqual(expected.1.1, .length)
            XCTAssertNotEqual(
                expected.0.0, expected.1.0, "fixture must discriminate a swapped sampler row")
            let trace = MiMoMediaMTPTrace(numeric: numeric)
            let observed = MiMoMediaMTPObservedTarget(on, trace)
            let sampler = CBv2DefaultSampler()
            sampler.sampleObserverForTesting = { [weak trace] in trace?.sample($0, step: $1) }
            let (engine, backend) = try MiMoMediaMTPFixture.engine(
                on, model: observed, sampler: sampler,
                fixedDraftTokens: fixedDraftTokens)
            attach(trace, to: engine)
            engine.loopForTesting.onEngineQueueSync {
                let loop = engine.loopForTesting
                loop.suspendStepExecutionAtCountForTesting = 0
                XCTAssertEqual(loop.mtp?.config.fixedDraftTokens, fixedDraftTokens)
                XCTAssertEqual(loop.mtp?.config.verificationMode, .serialTarget)
                XCTAssertEqual(loop.mtp?.usesMarginalPolicy, fixedDraftTokens == nil)
                XCTAssertEqual(loop.mtp?.config.maxSpeculativeBatch, 1)
                if !numeric { XCTAssertNil(loop.mtp?.carryHiddenObserverForTesting) }
            }
            do {
                let mediaRequest = try onPrepared.makeRequest(
                    binding: MiMoMediaMTPFixture.binding(f, on), id: mediaID,
                    sampling: .init(temperature: 0))
                let first = try engine.submit(mediaFirst ? mediaRequest : text)
                let second = try engine.submit(mediaFirst ? text : mediaRequest)
                engine.loopForTesting.onEngineQueueSync {
                    let loop = engine.loopForTesting
                    XCTAssertEqual(loop.stepCount, 0)
                    let media = loop.scheduler.record(for: mediaID)
                    let text = loop.scheduler.record(for: textID)
                    XCTAssertNotNil(media)
                    XCTAssertNotNil(text)
                    if let media, let text {
                        XCTAssertEqual(media.arrivalSeq < text.arrivalSeq, mediaFirst)
                    }
                    loop.suspendStepExecutionAtCountForTesting = nil
                }
                async let firstValue = MiMoMediaMTPFixture.collect(first)
                async let secondValue = MiMoMediaMTPFixture.collect(second)
                let actual = await (firstValue, secondValue)
                XCTAssertEqual(actual.0.0, mediaFirst ? expected.0.0 : expected.1.0)
                XCTAssertEqual(actual.1.0, mediaFirst ? expected.1.0 : expected.0.0)
                XCTAssertEqual(actual.0.1, .length)
                XCTAssertEqual(actual.1.1, .length)
                let metrics = try XCTUnwrap(engine.mtpMetricsSnapshot())
                XCTAssertGreaterThan(metrics.draftedTokens, 0)
                XCTAssertGreaterThan(metrics.serialVerificationRounds, 0)
                XCTAssertEqual(metrics.rectangularVerificationRounds, 0)
                await engine.shutdown()
                let (forwards, samples, clean) = trace.snapshot()
                XCTAssertTrue(clean)
                XCTAssertFalse(forwards.isEmpty)
                XCTAssertTrue(
                    forwards.allSatisfy { $0.rows.count == $0.shape[0] },
                    "trace must identify every actual target row")
                XCTAssertTrue(
                    forwards.filter { $0.kind.hasPrefix("hidden") }.allSatisfy {
                        !$0.rows.contains(where: \.media)
                    })
                XCTAssertTrue(
                    forwards.contains { $0.kind == "ordinary" && $0.rows.contains(where: \.media) })
                let shared = samples.filter { $0.ids.contains(mediaID) && $0.ids.contains(textID) }
                XCTAssertFalse(shared.isEmpty, "must exercise the mixed decode/seed cohort")
                for sample in shared {
                    XCTAssertEqual(sample.ids, mediaFirst ? [mediaID, textID] : [textID, mediaID])
                }
                for calls in Dictionary(
                    grouping: samples.filter { $0.ids.contains(mediaID) }, by: \.step
                ).values {
                    XCTAssertEqual(calls.count, 1, "media row sampled twice in one engine step")
                }
                let (witnesses, carries, policyCalls) = trace.proof()
                XCTAssertTrue(
                    witnesses.contains {
                        $0.media.id == mediaID && $0.text.id == textID && $0.depth > 0
                            && $0.completedStep == $0.step + 1
                    },
                    "require a COMPLETED real text draft/serial verification in the SAME step as live, unpaused media forward+sample"
                )
                XCTAssertFalse(carries.isEmpty)
                XCTAssertTrue(
                    carries.allSatisfy { $0.id == textID }, "media must never acquire any carry")
                XCTAssertTrue(
                    forwards.contains {
                        $0.kind == "media-prefill" && $0.rows.contains(where: \.media)
                    },
                    "the media history must contain actual embedding-spliced prefill")
                if fixedDraftTokens == nil {
                    XCTAssertGreaterThan(
                        metrics.controllerFallbacks["marginal_offer", default: 0], 0)
                    XCTAssertGreaterThan(
                        policyCalls, 0, "the actual mixed compact top-two branch must run")
                }
                if numeric {
                    // The engine is drained BEFORE any numeric observer
                    // readback. Do not treat this retained-view run as a
                    // memory, ownership, liveness or performance result.
                    try assertNumericMappingAfterDrain(
                        trace, dtype: dtype, mediaFirst: mediaFirst,
                        mediaID: mediaID, textID: textID,
                        hiddenSize: f.target.configuration.hiddenSize)
                } else {
                    XCTAssertNil(trace.takeNumericSeedAfterDrain())
                }
                detachAfterDrain(trace, engine: engine, sampler: sampler)
                XCTAssertEqual(backend.bytesReserved, 0)
                XCTAssertTrue(
                    engine.loopForTesting.onEngineQueueSync {
                        let driver = engine.loopForTesting.mtp!
                        return !driver.isTargetOnlyMediaForTesting(mediaID)
                            && driver.assistantStateCountsForTesting(mediaID) == nil
                            && driver.assistantStateCountsForTesting(textID) == nil
                    })
            } catch {
                engine.loopForTesting.onEngineQueueSync {
                    engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil
                }
                await engine.shutdown()
                detachAfterDrain(trace, engine: engine, sampler: sampler)
                throw error
            }
        } catch {
            off.loopForTesting.onEngineQueueSync {
                off.loopForTesting.suspendStepExecutionAtCountForTesting = nil
            }
            await off.shutdown()
            throw error
        }
    }
    func testVideoChunksAndTextBeforeAfterStillUseRealHeadsOnSameTarget() async throws {
        let f = try MiMoMediaFixture.models()
        let (_, _, on) = try MiMoMediaMTPFixture.installed(f)
        let offPrepared = try MiMoMediaMTPFixture.prepared(f, f.adapter, video: true)
        let onPrepared = try MiMoMediaMTPFixture.prepared(f, on, video: true)
        let (off, _) = try MiMoMediaMTPFixture.engine(f.adapter, chunk: 2)
        let expected: ([Int], CBv2FinishReason?)
        do {
            expected = await MiMoMediaMTPFixture.collect(
                try off.submit(
                    offPrepared.makeRequest(
                        binding: f.binding, id: .init(10), sampling: .init(temperature: 0))))
            await off.shutdown()
        } catch {
            await off.shutdown()
            throw error
        }
        let (engine, backend) = try MiMoMediaMTPFixture.engine(on, chunk: 2)
        do {
            let before = await MiMoMediaMTPFixture.collect(
                try engine.submit(MiMoMediaMTPFixture.text(20)))
            XCTAssertEqual(before.1, .length)
            let draftsBefore = try XCTUnwrap(engine.mtpMetricsSnapshot()).draftedTokens
            XCTAssertGreaterThan(draftsBefore, 0)
            let actual = await MiMoMediaMTPFixture.collect(
                try engine.submit(
                    onPrepared.makeRequest(
                        binding: MiMoMediaMTPFixture.binding(f, on), id: .init(10),
                        sampling: .init(temperature: 0))))
            XCTAssertEqual(actual.0, expected.0)
            XCTAssertEqual(actual.1, .length)
            XCTAssertEqual(
                engine.mtpMetricsSnapshot()?.draftedTokens, draftsBefore,
                "media itself must never draft")
            let after = await MiMoMediaMTPFixture.collect(
                try engine.submit(MiMoMediaMTPFixture.text(21)))
            XCTAssertEqual(after.0, before.0)
            XCTAssertEqual(after.1, .length)
            XCTAssertGreaterThan(
                try XCTUnwrap(engine.mtpMetricsSnapshot()).draftedTokens, draftsBefore)
            await engine.shutdown()
            XCTAssertEqual(backend.bytesReserved, 0)
        } catch {
            await engine.shutdown()
            throw error
        }
    }
    func testActualAudioPatchSpansUseTargetOnlyRouteWithTextHeadsInstalled() async throws {
        let f = try MiMoMediaFixture.models()
        let (_, _, on) = try MiMoMediaMTPFixture.installed(f)
        let codes = MiMoV26AudioCodeClip(
            codes: (0 ..< (9 * f.patch.configuration.channels)).map { Int32($0 % 6) }, frameCount: 9
        )
        let output = try f.patch.forward(clips: [codes], limits: MiMoMediaFixture.limits.audioPatch)
        eval(output.features)
        XCTAssertEqual(output.clipPatchRanges.map(\.count), [3])
        let prompt = [20, 7, 6, 6, 6, 8, 21, 22]
        func request(_ id: UInt64) -> CBv2Request {
            .init(
                id: .init(id), promptTokens: prompt, sampling: .init(temperature: 0), maxTokens: 18,
                prefixCacheEnabled: false,
                multimodal: .init(spans: [.init(tokenOffset: 2, length: 3)], attention: .causal) {
                    [features = output.features] in [features]
                })
        }
        let (off, offBackend) = try MiMoMediaMTPFixture.engine(f.adapter, chunk: 2)
        let expected: ([Int], CBv2FinishReason?)
        do {
            expected = await MiMoMediaMTPFixture.collect(try off.submit(request(70)))
            await off.shutdown()
        } catch {
            await off.shutdown()
            throw error
        }
        XCTAssertEqual(offBackend.bytesReserved, 0)
        let (engine, backend) = try MiMoMediaMTPFixture.engine(on, chunk: 2)
        do {
            let actual = await MiMoMediaMTPFixture.collect(try engine.submit(request(70)))
            XCTAssertEqual(actual.0, expected.0)
            XCTAssertEqual(actual.1, .length)
            XCTAssertEqual(engine.mtpMetricsSnapshot()?.draftedTokens, 0)
            let text = await MiMoMediaMTPFixture.collect(
                try engine.submit(MiMoMediaMTPFixture.text(71)))
            XCTAssertEqual(text.1, .length)
            XCTAssertGreaterThan(try XCTUnwrap(engine.mtpMetricsSnapshot()).draftedTokens, 0)
            await engine.shutdown()
            XCTAssertEqual(backend.bytesReserved, 0)
        } catch {
            await engine.shutdown()
            throw error
        }
        // This is the real discrete-code AudioPatch seam, not a PCM codec or
        // encoded-audio qualification. Those remain separate root gates.
    }
    func testCancellationAndNumericIDReuseDoNotSeedMediaOrPoisonLaterText() async throws {
        let f = try MiMoMediaFixture.models()
        let (_, _, on) = try MiMoMediaMTPFixture.installed(f)
        let media = try MiMoMediaMTPFixture.prepared(f, on, output: 80)
        let (engine, backend) = try MiMoMediaMTPFixture.engine(on, chunk: 1, events: 1)
        let id = CBv2RequestID(50)
        do {
            let initial = await MiMoMediaMTPFixture.collect(
                try engine.submit(MiMoMediaMTPFixture.text(50)))
            XCTAssertEqual(initial.1, .length)
            let before = try XCTUnwrap(engine.mtpMetricsSnapshot()).draftedTokens
            let stream = try engine.submit(
                media.makeRequest(
                    binding: MiMoMediaMTPFixture.binding(f, on), id: id,
                    sampling: .init(temperature: 0)))
            var cancelled = false
            var finish: CBv2FinishReason?
            for await event in stream {
                switch event {
                case .delta:
                    if !cancelled {
                        engine.cancel(id)
                        cancelled = true
                    }
                case .finished(let reason, _): finish = reason
                }
            }
            XCTAssertTrue(cancelled)
            XCTAssertEqual(finish, .cancelled)
            XCTAssertEqual(engine.mtpMetricsSnapshot()?.draftedTokens, before)
            XCTAssertTrue(
                engine.loopForTesting.onEngineQueueSync {
                    let loop = engine.loopForTesting
                    return loop.kvStates[id] == nil
                        && loop.mtp?.assistantStateCountsForTesting(id) == nil
                        && loop.mtp?.isTargetOnlyMediaForTesting(id) == false
                })
            let later = await MiMoMediaMTPFixture.collect(
                try engine.submit(MiMoMediaMTPFixture.text(50)))
            XCTAssertEqual(later.0, initial.0)
            XCTAssertEqual(later.1, .length)
            XCTAssertGreaterThan(try XCTUnwrap(engine.mtpMetricsSnapshot()).draftedTokens, before)
            await engine.shutdown()
            XCTAssertEqual(backend.bytesReserved, 0)
        } catch {
            await engine.shutdown()
            throw error
        }
    }
    func testSplitTargetLogitsRecordDifferencesAgainstRectangularNativeForward() throws {
        for dtype in ["float32", "bfloat16"] {
            let f = try MiMoMediaFixture.models(dtype)
            let (_, _, on) = try MiMoMediaMTPFixture.installed(f)
            _ = try withMiMoMediaMTPConstructionScope { work in
                try f.adapter.probeNativeKVTypes(retaining: work)
            }
            _ = try withMiMoMediaMTPConstructionScope { work in
                try on.probeNativeKVTypes(retaining: work)
            }
            let referenceBackend = try f.adapter.makeBackend(bytesCapacity: 16 << 20)
            let splitBackend = try on.makeBackend(bytesCapacity: 16 << 20)
            let referenceRows = try (0 ..< 2).map { _ in
                try referenceBackend.makeSequenceState(
                    layerKinds: f.adapter.layerKinds, promptLength: 16, maxLength: 64)
            }
            let splitRows = try (0 ..< 2).map { _ in
                try splitBackend.makeSequenceState(
                    layerKinds: on.layerKinds, promptLength: 16, maxLength: 64)
            }
            let referenceCaches = f.adapter.makeCaches()
            let splitCaches = on.makeCaches()
            for step in 0 ..< 20 {
                let tokens = [Int32(20 + step % 12), Int32(53 - step % 12)]
                try f.adapter.bindRows(referenceRows, caches: referenceCaches)
                let expected = try f.adapter.forwardValidated(
                    tokens: MLXArray(tokens, [2, 1]), caches: referenceCaches)
                try on.bindRows([splitRows[0]], caches: splitCaches)
                let media = on.forward(tokens: MLXArray([tokens[0]], [1, 1]), caches: splitCaches)
                try on.bindRows([splitRows[1]], caches: splitCaches)
                let text = on.forwardWithHidden(
                    tokens: MLXArray([tokens[1]], [1, 1]), caches: splitCaches
                ).logits
                let actual = concatenated([media, text], axis: 0)
                eval(actual, expected)
                let difference = abs(actual - expected)
                let maximum = difference.max().item(Float.self)
                print("MIMO_MEDIA_MTP_SPLIT dtype=\(dtype) step=\(step) max_abs=\(maximum)")
                let absolute: Float = dtype == "bfloat16" ? 0.02 : 5e-5
                let relative: Float = dtype == "bfloat16" ? 0.02 : 1e-4
                XCTAssertTrue(
                    all(difference .<= (absolute + relative * abs(expected))).item(Bool.self))
                XCTAssertEqual(
                    argMax(actual, axis: -1).asArray(Int32.self),
                    argMax(expected, axis: -1).asArray(Int32.self))
            }
            for rows in referenceRows { referenceBackend.release(rows) }
            for rows in splitRows { splitBackend.release(rows) }
        }
    }
    func testLoadedAssistantInvalidationAndEOSUnionStillFailClosedOrPreserve() throws {
        let f = try MiMoMediaFixture.models()
        let (_, assistant, on) = try MiMoMediaMTPFixture.installed(f)
        let stops: Set<Int> = [1, 57, 63]
        let processor = try MiMoMediaMTPFixture.processor(f, on, stops: stops)
        let plan = try processor.plan(MiMoMediaFixture.request([.image(MiMoMediaFixture.image())]))
        let first = try processor.prepare(plan, authorize: { MiMoMediaFixture.Reservation($0) })
        let request = try first.makeRequest(
            binding: MiMoMediaMTPFixture.binding(f, on, stops: stops), id: .init(77))
        XCTAssertEqual(request.stopTokens, stops)
        XCTAssertFalse(request.prefixCacheEnabled)
        assistant.invalidateLoadedSession()
        XCTAssertFalse(on.supportsMultimodalPrefill)
        XCTAssertThrowsError(try request.multimodal!.embeddings())
        var admitted = 0
        XCTAssertThrowsError(
            try processor.prepare(
                plan,
                authorize: {
                    admitted += 1
                    return MiMoMediaFixture.Reservation($0)
                }))
        XCTAssertEqual(admitted, 0)
    }

    /// Keep the original token-only split oracle above intact. This additional
    /// cell starts from genuine media-conditioned history in BOTH branches,
    /// then compares rectangular ordinary decode with the media/text split.
    func testMediaConditionedSplitLogitsPreserveFP32BF16OraclesAfterSWAWrap() throws {
        for dtype in ["float32", "bfloat16"] {
            let f = try MiMoMediaFixture.models(dtype)
            let (_, _, on) = try MiMoMediaMTPFixture.installed(f)
            _ = try withMiMoMediaMTPConstructionScope { work in
                try f.adapter.probeNativeKVTypes(retaining: work)
            }
            _ = try withMiMoMediaMTPConstructionScope { work in
                try on.probeNativeKVTypes(retaining: work)
            }
            let prepared = try MiMoMediaMTPFixture.prepared(f, f.adapter)
            let request = try prepared.makeRequest(
                binding: f.binding, id: .init(601), sampling: .init(temperature: 0))
            let spans = try XCTUnwrap(request.multimodal).spans
            let features = try request.multimodal!.embeddings()
            XCTAssertFalse(spans.isEmpty)
            XCTAssertEqual(features.count, spans.count)
            XCTAssertTrue(features.allSatisfy { $0.dtype == f.target.activationDType })
            // Independent host assembly, not the engine's splice/resolve path.
            let base = f.target.model.embedTokens(
                MLXArray(request.promptTokens.map(Int32.init), [1, request.promptTokens.count]))
            var pieces: [MLXArray] = []
            var cursor = 0
            for (span, feature) in zip(spans, features) {
                if cursor < span.tokenOffset {
                    pieces.append(base[0..., cursor ..< span.tokenOffset, 0...])
                }
                pieces.append(feature.expandedDimensions(axis: 0))
                cursor = span.tokenOffset + span.length
            }
            if cursor < request.promptTokens.count {
                pieces.append(base[0..., cursor ..< request.promptTokens.count, 0...])
            }
            let mediaEmbeddings = concatenated(pieces, axis: 1)
            let textPrompt = MiMoMediaMTPFixture.text(602, length: request.promptTokens.count)
                .promptTokens
            let referenceBackend = try f.adapter.makeBackend(bytesCapacity: 16 << 20)
            let splitBackend = try on.makeBackend(bytesCapacity: 16 << 20)
            let referenceRows = try (0 ..< 2).map { _ in
                try referenceBackend.makeSequenceState(
                    layerKinds: f.adapter.layerKinds, promptLength: request.promptTokens.count,
                    maxLength: 128)
            }
            let splitRows = try (0 ..< 2).map { _ in
                try splitBackend.makeSequenceState(
                    layerKinds: on.layerKinds, promptLength: request.promptTokens.count,
                    maxLength: 128)
            }
            defer {
                for rows in referenceRows { referenceBackend.release(rows) }
                for rows in splitRows { splitBackend.release(rows) }
            }
            let referenceCaches = f.adapter.makeCaches()
            let splitCaches = on.makeCaches()
            func compare(_ actual: MLXArray, _ expected: MLXArray, phase: String, step: Int) {
                eval(actual, expected)
                let difference = abs(actual - expected)
                let maximum = difference.max().item(Float.self)
                print(
                    "MIMO_MEDIA_CONDITIONED_SPLIT dtype=\(dtype) phase=\(phase) step=\(step) max_abs=\(maximum)"
                )
                let absolute: Float = dtype == "bfloat16" ? 0.02 : 5e-5
                let relative: Float = dtype == "bfloat16" ? 0.02 : 1e-4
                XCTAssertTrue(
                    all(difference .<= (absolute + relative * abs(expected))).item(Bool.self))
                XCTAssertEqual(
                    argMax(actual, axis: -1).asArray(Int32.self),
                    argMax(expected, axis: -1).asArray(Int32.self))
            }
            for start in stride(from: 0, to: request.promptTokens.count, by: 3) {
                let end = min(start + 3, request.promptTokens.count)
                let mediaIDs = MLXArray(
                    request.promptTokens[start ..< end].map(Int32.init), [1, end - start])
                let textIDs = MLXArray(textPrompt[start ..< end].map(Int32.init), [1, end - start])
                let chunk = mediaEmbeddings[0..., start ..< end, 0...]
                try f.adapter.bindRows([referenceRows[0]], caches: referenceCaches)
                let expectedMedia = try f.adapter.forwardValidated(
                    tokens: mediaIDs, inputEmbeddings: chunk, caches: referenceCaches)
                try on.bindRows([splitRows[0]], caches: splitCaches)
                let actualMedia = try on.forwardValidated(
                    tokens: mediaIDs, inputEmbeddings: chunk, caches: splitCaches)
                compare(actualMedia, expectedMedia, phase: "media-prefill", step: start)
                try f.adapter.bindRows([referenceRows[1]], caches: referenceCaches)
                let expectedText = try f.adapter.forwardValidated(
                    tokens: textIDs, caches: referenceCaches)
                try on.bindRows([splitRows[1]], caches: splitCaches)
                let actualText = on.forwardWithHidden(tokens: textIDs, caches: splitCaches)
                compare(actualText.logits, expectedText, phase: "text-prefill", step: start)
                eval(actualText.lastHidden)
            }
            let windows = f.adapter.layerKinds.compactMap { kind -> Int? in
                if case .slidingWindow(let window) = kind.attention { return window }
                return nil
            }
            XCTAssertFalse(windows.isEmpty)
            XCTAssertTrue(
                f.adapter.layerKinds.contains {
                    if case .full = $0.attention { return true }
                    return false
                })
            XCTAssertGreaterThan(
                request.promptTokens.count, try XCTUnwrap(windows.max()),
                "media history must wrap SWA before mixed decode")
            for step in 0 ..< 20 {
                let tokens = [Int32(20 + step % 12), Int32(53 - step % 12)]
                try f.adapter.bindRows(referenceRows, caches: referenceCaches)
                let expected = try f.adapter.forwardValidated(
                    tokens: MLXArray(tokens, [2, 1]), caches: referenceCaches)
                try on.bindRows([splitRows[0]], caches: splitCaches)
                let media = on.forward(tokens: MLXArray([tokens[0]], [1, 1]), caches: splitCaches)
                try on.bindRows([splitRows[1]], caches: splitCaches)
                let text = on.forwardWithHidden(
                    tokens: MLXArray([tokens[1]], [1, 1]), caches: splitCaches)
                compare(
                    concatenated([media, text.logits], axis: 0), expected, phase: "mixed-decode",
                    step: step)
                eval(text.lastHidden)
                for rows in referenceRows + splitRows {
                    XCTAssertEqual(
                        rows.compactMap { $0?.absoluteOffset },
                        Array(repeating: request.promptTokens.count + step + 1, count: rows.count))
                }
            }
        }
    }
}
