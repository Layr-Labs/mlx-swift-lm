import Foundation
import MLX
import MLXNN
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon

/// Metadata-only observation around the actual target; never supplies logits,
/// features, proposal tokens or a fabricated native execution contract.
private final class MiMoRectangularObservedTarget: CBv2MTPPrefillSteppableModel,
    CBv2PrefillSteppableModel, CBv2ModelCapabilityProviding, CBv2MTPPolicyTopTwoProviding {
    let native: MiMoV26CBv2Adapter
    var depth: (() -> Int?)?
    var calls: [(shape: [Int], depth: Int)] = []
    init(_ native: MiMoV26CBv2Adapter) { self.native = native }
    var cbv2Capabilities: CBv2ModelCapabilities { native.cbv2Capabilities }
    var mtpCaptureLayers: CBv2MTPCaptureLayers? { native.mtpCaptureLayers }
    var mtpTargetIdentity: ObjectIdentifier? { native.mtpTargetIdentity }
    var supportsRequestStatefulMTP: Bool { native.supportsRequestStatefulMTP }
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        native.forward(tokens: tokens, caches: caches)
    }
    func prefill(tokens: MLXArray, inputEmbeddings: MLXArray?, caches: [CBv2AttendingLayerCache],
        requirement: CBv2PrefillRequirement) -> MLXArray {
        native.prefill(tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches, requirement: requirement)
    }
    func forwardWithHidden(tokens: MLXArray, caches: [CBv2AttendingLayerCache])
        -> (logits: MLXArray, lastHidden: MLXArray) {
        if let depth = depth?() { calls.append((tokens.shape, depth)) }
        return native.forwardWithHidden(tokens: tokens, caches: caches)
    }
    func forwardWithHiddenForPrefill(tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        requirement: CBv2PrefillRequirement) -> (logits: MLXArray, lastHidden: MLXArray) {
        native.forwardWithHiddenForPrefill(tokens: tokens, caches: caches, requirement: requirement)
    }
    func cbv2MTPTopTwo(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
        native.cbv2MTPTopTwo(logits)
    }
}

/// Candidate source only. Every numerical cell requires the exclusive lane;
/// no replacement target/head output is used for engagement or parity.
private final class MiMoV26RectangularScriptedAssistant: CBv2MTPRequestStatefulDrafter {
    let native: MiMoV26MTPAssistant
    let script: [Int]?
    let promptLength: Int
    let rejectPosition: Int
    var states: [MiMoV26MTPState] = []
    var observations = 0
    var observationFenceReads = 0
    var fenceHook: ((MiMoV26MTPState) throws -> Void)?
    init(_ native: MiMoV26MTPAssistant, script: [Int]? = nil,
         promptLength: Int = 0, rejectPosition: Int = 3) {
        self.native = native; self.script = script
        self.promptLength = promptLength; self.rejectPosition = rejectPosition
    }
    var mtpTargetIdentity: ObjectIdentifier? { native.mtpTargetIdentity }
    var requiredVerificationMode: CBv2MTPVerificationMode? { native.requiredVerificationMode }
    var maximumDraftTokens: Int? { 3 }
    var maximumSpeculativeBatch: Int? { 1 }
    var requiresCommittedObservationFence: Bool { true }
    var requestStateBytesPerToken: Int { native.requestStateBytesPerToken }
    var requestStateTokenGranularity: Int { native.requestStateTokenGranularity }
    var requestStateTokenAllocationPadding: Int { native.requestStateTokenAllocationPadding }
    func makeRequestState() -> any CBv2MTPRequestState {
        let state = native.makeRequestState() as! MiMoV26MTPState
        states.append(state)
        return state
    }
    func configureRequestState(_ state: any CBv2MTPRequestState, maximumSequenceLength: Int) throws {
        try native.configureRequestState(state, maximumSequenceLength: maximumSequenceLength)
    }
    func observeCommittedTarget(_ observation: CBv2MTPCommittedTargetObservation,
                                requestState: any CBv2MTPRequestState) {
        observations += 1
        native.observeCommittedTarget(observation, requestState: requestState)
    }
    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture { native.prepare(rows: rows) }
    func draftStep(tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture)
        -> (tokens: MLXArray, hidden: MLXArray) {
        native.draftStep(tokens: tokens, hidden: hidden, prepared: prepared)
    }
    func draftStep(tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
                   requestState: any CBv2MTPRequestState) -> (tokens: MLXArray, hidden: MLXArray) {
        let result = native.draftStep(tokens: tokens, hidden: hidden, shortlist: shortlist,
                                      requestState: requestState)
        guard let script else { return result }
        let state = requestState as! MiMoV26MTPState
        let depth = state.stagedInputCount - 1
        let index = state.observedCount - promptLength + depth + 1
        precondition(script.indices.contains(index))
        let token = (script[index] + (depth == rejectPosition ? 1 : 0)) % 32
        return (MLXArray([Int32(token)]), result.hidden)
    }
    func evaluationTargets(for state: any CBv2MTPRequestState) -> [MLXArray] {
        if state.stagedInputCount == 0 { observationFenceReads += 1 }
        return native.evaluationTargets(for: state)
    }
    func requestStateDidFinishEvaluation(_ state: any CBv2MTPRequestState) throws {
        if let state = state as? MiMoV26MTPState { try fenceHook?(state) }
        try native.requestStateDidFinishEvaluation(state)
    }
    func finalizeRound(requestState: any CBv2MTPRequestState, confirmedInputTokens: Int,
                       committedDraftTokens: MLXArray, committedTargetHidden: MLXArray) {
        native.finalizeRound(requestState: requestState, confirmedInputTokens: confirmedInputTokens,
                             committedDraftTokens: committedDraftTokens, committedTargetHidden: committedTargetHidden)
    }
    func discardRound(requestState: any CBv2MTPRequestState) { native.discardRound(requestState: requestState) }
    func releaseRequestState(_ state: any CBv2MTPRequestState) { native.releaseRequestState(state) }
}

final class MiMoV26RectangularVerifyTests: XCTestCase {
    private func lane() throws {
        guard ProcessInfo.processInfo.environment["MIMO_V26_RECTANGULAR_NATIVE_TESTS"] == "1" else {
            throw XCTSkip("Requires an authorized exclusive native candidate process")
        }
    }
    private func ids(_ tokens: [Int]) -> MLXArray {
        MLXArray(tokens.map(Int32.init), [1, tokens.count])
    }
    private func prompt(_ count: Int) -> [Int] { (0..<count).map { 1 + ($0 * 7) % 31 } }
    private func fixture(_ dtype: DType) throws -> (MiMoV26TextModel, MiMoV26MTP) {
        var fields = try JSONSerialization.jsonObject(with:
            JSONEncoder().encode(MiMoV26MTPChecks.config())) as! [String: Any]
        fields["max_position_embeddings"] = 256
        fields["sliding_window"] = 128; fields["sliding_window_size"] = 128
        fields["head_dim"] = 192; fields["swa_head_dim"] = 192
        fields["v_head_dim"] = 128; fields["swa_v_head_dim"] = 128
        fields["add_full_attention_sink_bias"] = true
        fields["dtype"] = dtype == .float32 ? "float32" : "bfloat16"
        fields["moe_router_dtype"] = fields["dtype"]
        let config = try JSONDecoder().decode(MiMoV26Configuration.self,
            from: JSONSerialization.data(withJSONObject: fields))
        let target = try MiMoV26TextModel(config)
        try target.update(parameters: .unflattened(MiMoV26MTPChecks.fixtureWeights(target)
            .mapValues { $0.asType(dtype) }), verify: .all)
        let predictor = try MiMoV26MTP(target: target)
        try predictor.loadConvertedWeights(MiMoV26MTPChecks.fixtureWeights(predictor, prefix: "mtp.")
            .mapValues { $0.asType(dtype) })
        eval(target, predictor)
        return (target, predictor)
    }
    private func probe(_ adapter: MiMoV26CBv2Adapter) throws {
        let scope = NativeConstructionScope()
        defer { if scope.snapshot.isRetainedFault { _ = Unmanaged.passRetained(scope) } }
        _ = try adapter.probeNativeKVTypes(retaining: scope)
    }
    /// Existing component absolute gate for FP32; existing native adapter's
    /// BF16 abs/relative gate. Exact target IDs/continuation are separate.
    private func compare(_ actual: MLXArray, _ reference: MLXArray, _ name: String) throws {
        eval(actual, reference)
        XCTAssertEqual(actual.shape, reference.shape, name)
        XCTAssertEqual(actual.dtype, reference.dtype, name)
        let a = actual.asType(.float32).asArray(Float.self)
        let b = reference.asType(.float32).asArray(Float.self)
        guard a.count == b.count else { return }
        var maxULP: UInt32 = 0
        func ordered(_ value: Float) -> UInt32 {
            let bits = value.bitPattern
            return bits & 0x80000000 == 0 ? bits | 0x80000000 : ~bits
        }
        for (x, y) in zip(a, b) {
            XCTAssertTrue(x.isFinite && y.isFinite, name)
            let u = ordered(x), v = ordered(y)
            maxULP = max(maxULP, u >= v ? u - v : v - u)
            let limit: Float = actual.dtype == .bfloat16 ? 0.02 + 0.02 * abs(y) : 8e-5
            XCTAssertLessThanOrEqual(abs(x - y), limit, name)
        }
        var nativeULP = maxULP
        if actual.dtype == .bfloat16 {
            func ordered16(_ bits: UInt16) -> UInt16 {
                bits & 0x8000 == 0 ? bits | 0x8000 : ~bits
            }
            nativeULP = zip(actual.view(dtype: .uint16).asArray(UInt16.self),
                reference.view(dtype: .uint16).asArray(UInt16.self)).reduce(0) { result, pair in
                    let x = ordered16(pair.0), y = ordered16(pair.1)
                    return max(result, UInt32(x >= y ? x - y : y - x))
                }
        }
        print("MIMO_RECTANGULAR_STATE name=\(name) dtype=\(actual.dtype) maxNativeULP=\(nativeULP) maxFloat32ULP=\(maxULP)")
    }
    private func roots(_ caches: [any CBv2AttendingLayerCache]) -> [MLXArray] {
        caches.flatMap { ($0 as? any KVCache)?.innerState() ?? [] }
    }
    private func compareRows(_ actual: [CBv2SequenceKV?], _ reference: [CBv2SequenceKV?]) throws {
        for (a, b) in zip(actual, reference) {
            let a = try XCTUnwrap(a), b = try XCTUnwrap(b)
            XCTAssertEqual(a.absoluteOffset, b.absoluteOffset)
            XCTAssertEqual(a.retainedCount, b.retainedCount)
            let x = a.snapshot(), y = b.snapshot()
            try compare(x.keys, y.keys, "target K")
            try compare(x.values, y.values, "target V")
        }
    }
    private func settle(_ assistant: MiMoV26MTPAssistant, _ state: MiMoV26MTPState) throws {
        eval(assistant.evaluationTargets(for: state))
        try assistant.requestStateDidFinishEvaluation(state)
    }
    private func compareHeads(_ actual: MiMoV26MTPState, _ reference: MiMoV26MTPState) throws {
        XCTAssertEqual(actual.observedCount, reference.observedCount)
        XCTAssertEqual(actual.committedInputCount, reference.committedInputCount)
        XCTAssertEqual(actual.stagedInputCount, reference.stagedInputCount)
        XCTAssertEqual(actual.headInputCounts, reference.headInputCounts)
        XCTAssertEqual(actual.generation, reference.generation)
        XCTAssertEqual(actual.ownedArrays.count, reference.ownedArrays.count)
        for (a, b) in zip(actual.ownedArrays, reference.ownedArrays) {
            if a.dtype == .int32 || a.dtype == .uint32 {
                XCTAssertEqual(a.asType(.int32).asArray(Int32.self), b.asType(.int32).asArray(Int32.self))
            } else { try compare(a, b, "head KV/features") }
        }
    }
    func testImmutableModeDefaultsAndUnsupportedModes() throws {
        try lane()
        let (target, predictor) = try fixture(.float32)
        XCTAssertEqual(try MiMoV26MTPAssistant(target: target, predictor: predictor).requiredVerificationMode, .serialTarget)
        let candidate = try MiMoV26MTPAssistant(target: target, predictor: predictor, verificationMode: .rectangular)
        XCTAssertEqual(candidate.requiredVerificationMode, .rectangular)
        for mode: CBv2MTPVerificationMode in [.automatic, .rectangularExact] {
            XCTAssertThrowsError(try MiMoV26MTPAssistant(target: target, predictor: predictor, verificationMode: mode))
        }
        let adapter = try MiMoV26CBv2Adapter(target: target, assistant: candidate)
        try probe(adapter)
        XCTAssertTrue(CBv2LayerCacheBank(caches: adapter.makeCaches()).supportsMTPRectangularVerification)
        XCTAssertTrue(try adapter.makeMultimodalCacheProvider().supportsMTPRectangularVerification)
        XCTAssertTrue(adapter.supportsMultimodalPrefill(attention: .causal))
        candidate.invalidateLoadedSession()
        XCTAssertNil(candidate.mtpTargetIdentity)
        XCTAssertThrowsError(try MiMoV26CBv2Adapter(target: target, assistant: candidate))
    }

    func testActualTargetInvocationCountAndShapeWithRealNativeProposals() async throws {
        try lane()
        let (target, predictor) = try fixture(.float32)
        for depth in 1...3 {
            var streams: [[Int]] = []
            for mode: CBv2MTPVerificationMode in [.serialTarget, .rectangular] {
                let assistant = try MiMoV26MTPAssistant(target: target, predictor: predictor, verificationMode: mode)
                let adapter = try MiMoV26CBv2Adapter(target: target, assistant: assistant)
                try probe(adapter)
                let backend = try adapter.makeBackend(bytesCapacity: 32 << 20)
                let observed = MiMoRectangularObservedTarget(adapter)
                let engine = EngineV2(model: observed, layerKinds: adapter.layerKinds, backend: backend,
                    cacheProvider: CBv2LayerCacheBank(caches: adapter.makeCaches()),
                    schedulerConfig: .init(maxConcurrentRequests: 1, maxBatchedTokensPerStep: 16,
                        prefillChunkSize: 3, maxWaiting: 4, enablePrefixCache: false),
                    mtpDrafter: assistant, mtpConfig: .init(enabled: true, maxDraftTokens: depth,
                        maxSpeculativeBatch: 1, fixedDraftTokens: depth, verificationMode: mode))
                observed.depth = { [weak engine] in
                    engine?.loopForTesting.mtp?.roundMark(for: .init(1))
                }
                let request = CBv2Request(id: .init(1), promptTokens: prompt(11),
                    sampling: .init(temperature: 0), maxTokens: 16, prefixCacheEnabled: false)
                let result = await cbv2SchedCollect(try engine.submit(request))
                let metrics = try XCTUnwrap(engine.mtpMetricsSnapshot())
                await engine.shutdown()
                observed.depth = nil
                XCTAssertEqual(result.finishReason, .length)
                XCTAssertGreaterThan(metrics.draftedTokens, 0)
                XCTAssertGreaterThan(metrics.rounds, 0)
                XCTAssertFalse(observed.calls.isEmpty)
                if mode == .rectangular {
                    XCTAssertEqual(metrics.serialVerificationRounds, 0)
                    XCTAssertEqual(observed.calls.count, metrics.rectangularVerificationRounds)
                    for call in observed.calls { XCTAssertEqual(call.shape, [1, call.depth + 1]) }
                } else {
                    XCTAssertEqual(metrics.rectangularVerificationRounds, 0)
                    for call in observed.calls { XCTAssertEqual(call.shape, [1, 1]) }
                    // Tail depth may shrink with remaining output budget.
                    XCTAssertEqual(observed.calls.reduce(0.0) { $0 + 1 / Double($1.depth + 1) },
                        Double(metrics.serialVerificationRounds), accuracy: 1e-9)
                }
                XCTAssertEqual(backend.bytesReserved, 0)
                streams.append(result.tokens)
            }
            XCTAssertEqual(streams[1], streams[0])
        }
    }

    func testPackageTicketRejectsUnbackedRectangularClaimAndWrongModeReplay() throws {
        let model = NSObject(), backend = NSObject(), bank = NSObject(), assistant = NSObject()
        let work = NativeConstructionScope()
        let serial = try work.withPhase(.nativeSetup) {
            try CBv2NativeExecutionContract(model: model, backend: backend,
                cacheProvider: bank, assistant: assistant, construction: work)
        }
        XCTAssertEqual(serial.mtpVerificationMode, .serialTarget)
        XCTAssertFalse(serial.consume(model: model, backend: backend, cacheProvider: bank,
            assistant: assistant, mtpVerificationMode: .rectangular))
        XCTAssertTrue(serial.consume(model: model, backend: backend, cacheProvider: bank,
            assistant: assistant, mtpVerificationMode: .serialTarget))
        XCTAssertFalse(serial.consume(model: model, backend: backend, cacheProvider: bank,
            assistant: assistant, mtpVerificationMode: .serialTarget))
        for mode: CBv2MTPVerificationMode in [.rectangular, .rectangularExact, .automatic] {
            XCTAssertThrowsError(try work.withPhase(.nativeSetup) {
                try CBv2NativeExecutionContract(model: model, backend: backend,
                    cacheProvider: bank, assistant: assistant, construction: work, mtpVerificationMode: mode)
            })
        }
    }

    func testActualRectangularAcceptWalkEveryRejectPositionAndEOS() async throws {
        try lane()
        let (target, predictor) = try MiMoV26MTPChecks.fixture()
        let native = try MiMoV26MTPAssistant(target: target, predictor: predictor, verificationMode: .rectangular)
        func make(_ drafter: (any CBv2MTPDrafter)?) throws -> (EngineV2, MiMoV26CBv2Backend) {
            let adapter = try MiMoV26CBv2Adapter(target: target, assistant: drafter == nil ? nil : native)
            try probe(adapter)
            let backend = try adapter.makeBackend(bytesCapacity: 32 << 20)
            return (EngineV2(model: adapter, layerKinds: adapter.layerKinds, backend: backend,
                cacheProvider: CBv2LayerCacheBank(caches: adapter.makeCaches()),
                schedulerConfig: .init(maxConcurrentRequests: 1, maxBatchedTokensPerStep: 16,
                    prefillChunkSize: 3, maxWaiting: 4, enablePrefixCache: false),
                mtpDrafter: drafter, mtpConfig: .init(enabled: drafter != nil, maxDraftTokens: 3,
                    maxSpeculativeBatch: 1, fixedDraftTokens: 3, verificationMode: .rectangular)), backend)
        }
        let request = CBv2Request(id: .init(1), promptTokens: prompt(11),
            sampling: .init(temperature: 0), maxTokens: 24, prefixCacheEnabled: false)
        let (offEngine, offBackend) = try make(nil)
        let off = await cbv2SchedCollect(try offEngine.submit(request))
        await offEngine.shutdown()
        XCTAssertEqual(off.finishReason, .length); XCTAssertEqual(offBackend.bytesReserved, 0)
        for rejectedAt in 0...3 {
            let script = MiMoV26RectangularScriptedAssistant(native, script: off.tokens,
                promptLength: request.promptTokens.count, rejectPosition: rejectedAt)
            let (engine, backend) = try make(script)
            let result = await cbv2SchedCollect(try engine.submit(request))
            let metrics = try XCTUnwrap(engine.mtpMetricsSnapshot())
            await engine.shutdown()
            XCTAssertEqual(result.tokens, off.tokens)
            XCTAssertEqual(result.finishReason, off.finishReason)
            XCTAssertGreaterThan(metrics.rectangularVerificationRounds, 0)
            XCTAssertEqual(metrics.serialVerificationRounds, 0)
            if rejectedAt == 0 { XCTAssertEqual(metrics.acceptedTokens, 0) }
            else { XCTAssertGreaterThan(metrics.perPositionAccepted[rejectedAt - 1], 0) }
            if rejectedAt < 3 { XCTAssertEqual(metrics.perPositionAccepted[rejectedAt], 0) }
            XCTAssertTrue(script.states.allSatisfy { $0.isReleased && $0.materializedBytes == 0 })
            XCTAssertEqual(backend.bytesReserved, 0)
        }
        var stop = request
        stop.stopTokens = [off.tokens[2]]
        let (baseline, _) = try make(nil)
        let expected = await cbv2SchedCollect(try baseline.submit(stop)); await baseline.shutdown()
        let script = MiMoV26RectangularScriptedAssistant(native, script: off.tokens,
            promptLength: request.promptTokens.count)
        let (engine, backend) = try make(script)
        let actual = await cbv2SchedCollect(try engine.submit(stop)); await engine.shutdown()
        XCTAssertEqual(actual.tokens, expected.tokens); XCTAssertEqual(actual.finishReason, .stop)
        XCTAssertEqual(backend.bytesReserved, 0)
    }

    func testSerialAuthorityCompleteTargetAndHeadStateAcrossEveryRollbackPrefix() throws {
        try lane()
        for dtype: DType in [.float32, .bfloat16] {
            let (target, predictor) = try fixture(dtype)
            for length in [1, 2, 3, 127, 128, 129] {
                for depth in 1...3 {
                    for accepted in 0...depth {
                        let serial = try MiMoV26MTPAssistant(target: target, predictor: predictor)
                        let fast = try MiMoV26MTPAssistant(target: target, predictor: predictor, verificationMode: .rectangular)
                        let adapters = try [serial, fast].map { try MiMoV26CBv2Adapter(target: target, assistant: $0) }
                        for adapter in adapters { try probe(adapter) }
                        let backends = try adapters.map { try $0.makeBackend(bytesCapacity: 32 << 20) }
                        let banks = adapters.map { $0.makeCaches() }
                        let rows = try backends.enumerated().map { index, backend in
                            try backend.makeSequenceState(layerKinds: adapters[index].layerKinds,
                                promptLength: length, maxLength: 256)
                        }
                        defer { for (backend, row) in zip(backends, rows) { backend.release(row) } }
                        for index in 0..<2 { try adapters[index].bindRows([rows[index]], caches: banks[index]) }
                        let assistants = [serial, fast]
                        let states = assistants.map { $0.makeRequestState() as! MiMoV26MTPState }
                        defer { for (assistant, state) in zip(assistants, states) { assistant.releaseRequestState(state) } }
                        for index in 0..<2 { try assistants[index].configureRequestState(states[index], maximumSequenceLength: 256) }
                        var last: [(logits: MLXArray, lastHidden: MLXArray)] = []
                        for index in 0..<2 {
                            let output = adapters[index].forwardWithHidden(tokens: ids(prompt(length)), caches: banks[index])
                            eval([output.logits, output.lastHidden] + roots(banks[index]))
                            assistants[index].observeCommittedTarget(.init(tokens: ids(prompt(length)), hidden: output.lastHidden), requestState: states[index])
                            try settle(assistants[index], states[index])
                            last.append(output)
                        }
                        let seed = Int(argMax(last[0].logits[0, -1], axis: -1).item(Int32.self))
                        var proposals: [[Int]] = []
                        for index in 0..<2 {
                            var token = ids([seed]), hidden = last[index].lastHidden[0..., (-1)..., 0...], values: [Int] = []
                            for _ in 0..<depth {
                                let draft = assistants[index].draftStep(tokens: token, hidden: hidden, shortlist: nil, requestState: states[index])
                                eval([draft.tokens, draft.hidden] + assistants[index].evaluationTargets(for: states[index]))
                                try assistants[index].requestStateDidFinishEvaluation(states[index])
                                values.append(Int(draft.tokens.item(Int32.self)))
                                token = draft.tokens.reshaped([1, 1]); hidden = draft.hidden
                            }
                            proposals.append(values)
                        }
                        XCTAssertEqual(proposals[0], proposals[1])
                        let tokens = [seed] + proposals[0]
                        for row in rows.flatMap({ $0.compactMap { $0 } }) { row.beginSpeculativeWrite() }
                        var serialLogits: [MLXArray] = [], serialHidden: [MLXArray] = []
                        for token in tokens {
                            let output = adapters[0].forwardWithHidden(tokens: ids([token]), caches: banks[0])
                            eval([output.logits, output.lastHidden] + roots(banks[0]))
                            serialLogits.append(output.logits); serialHidden.append(output.lastHidden)
                        }
                        let serializers = banks[1].compactMap { $0 as? any CBv2MTPRectangularSerializing }
                        XCTAssertEqual(serializers.count, banks[1].count)
                        for cache in serializers { cache.mtpSerializesRectangularAttention = true }
                        let candidate = adapters[1].forwardWithHidden(tokens: ids(tokens), caches: banks[1])
                        for cache in serializers { cache.mtpSerializesRectangularAttention = false }
                        eval([candidate.logits, candidate.lastHidden] + roots(banks[1]))
                        let referenceLogits = concatenated(serialLogits, axis: 1)
                        let referenceHidden = concatenated(serialHidden, axis: 1)
                        try compare(candidate.logits, referenceLogits, "verify logits")
                        try compare(candidate.lastHidden, referenceHidden, "post-final-norm hidden")
                        XCTAssertEqual(argMax(candidate.logits, axis: -1).asArray(Int32.self),
                            argMax(referenceLogits, axis: -1).asArray(Int32.self))
                        for row in rows.flatMap({ $0.compactMap { $0 } }) {
                            if accepted < depth { row.rollback(depth - accepted) }
                            row.commitSpeculativeWrite()
                        }
                        for index in 0..<2 {
                            try adapters[index].bindRows([rows[index]], caches: banks[index])
                            eval(roots(banks[index]))
                            let hidden = index == 0 ? referenceHidden : candidate.lastHidden
                            assistants[index].finalizeRound(requestState: states[index], confirmedInputTokens: accepted + 1,
                                committedDraftTokens: ids(Array(proposals[index].prefix(accepted))),
                                committedTargetHidden: hidden[0..., 0..<accepted, 0...])
                            try settle(assistants[index], states[index])
                        }
                        try compareRows(rows[1], rows[0]); try compareHeads(states[1], states[0])
                        // Carry transition and real subsequent proposal/target continuation.
                        let next = Int(argMax(referenceLogits[0, accepted], axis: -1).item(Int32.self))
                        var nextDrafts: [Int32] = [], continuation: [MLXArray] = []
                        for index in 0..<2 {
                            let hidden = index == 0 ? referenceHidden : candidate.lastHidden
                            let draft = assistants[index].draftStep(tokens: ids([next]),
                                hidden: hidden[0..., accepted..<accepted + 1, 0...], shortlist: nil, requestState: states[index])
                            eval([draft.tokens, draft.hidden] + assistants[index].evaluationTargets(for: states[index]))
                            try assistants[index].requestStateDidFinishEvaluation(states[index])
                            nextDrafts.append(draft.tokens.item(Int32.self))
                            let output = adapters[index].forwardWithHidden(tokens: ids([next]), caches: banks[index])
                            eval([output.logits, output.lastHidden] + roots(banks[index]))
                            continuation.append(output.logits)
                        }
                        XCTAssertEqual(nextDrafts[0], nextDrafts[1])
                        try compareHeads(states[1], states[0]); try compareRows(rows[1], rows[0])
                        XCTAssertEqual(argMax(continuation[0], axis: -1).asArray(Int32.self),
                            argMax(continuation[1], axis: -1).asArray(Int32.self))
                    }
                }
            }
        }
    }
}
