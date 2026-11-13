import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

// Observe the public assistant contract while forwarding every operation to
// the real native head. Array values are read only after the stream finishes.
private final class ObservedQwenSpanAssistant: CBv2MTPRequestStatefulDrafter {
    let native: Qwen35InlineMTPAssistant
    private(set) var starts: [(token: MLXArray, history: Int)] = []
    private var rounds: Set<ObjectIdentifier> = []

    init(_ native: Qwen35InlineMTPAssistant) { self.native = native }
    var mtpTargetIdentity: ObjectIdentifier? { native.mtpTargetIdentity }
    var requiredVerificationMode: CBv2MTPVerificationMode? { native.requiredVerificationMode }
    var maximumDraftTokens: Int? { native.maximumDraftTokens }
    var maximumSpeculativeBatch: Int? { native.maximumSpeculativeBatch }
    var supportsTargetPrefixAcceptance: Bool { native.supportsTargetPrefixAcceptance }
    var supportsEarlyDraftSubmission: Bool { native.supportsEarlyDraftSubmission }
    var prefersBatchedRectangularAttention: Bool { native.prefersBatchedRectangularAttention }
    var commitsAcceptedPrefixPerRow: Bool { native.commitsAcceptedPrefixPerRow }
    var supportsOrdinaryDecodeHistory: Bool { native.supportsOrdinaryDecodeHistory }
    var requiresCommittedObservationFence: Bool { native.requiresCommittedObservationFence }
    var draftShortlistSize: Int? { native.draftShortlistSize }
    var submitsEveryDraftStepEarly: Bool { native.submitsEveryDraftStepEarly }
    var requestStateBytesPerToken: Int { native.requestStateBytesPerToken }
    var requestStateTokenGranularity: Int { native.requestStateTokenGranularity }
    var requestStateTokenAllocationPadding: Int { native.requestStateTokenAllocationPadding }
    var requestStateAllocationSpecs: [CBv2AuxiliaryAllocationSpec]? {
        native.requestStateAllocationSpecs
    }

    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        native.prepare(rows: rows)
    }

    func draftStep(tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture)
        -> (tokens: MLXArray, hidden: MLXArray)
    {
        native.draftStep(tokens: tokens, hidden: hidden, prepared: prepared)
    }

    func makeRequestState() -> any CBv2MTPRequestState { native.makeRequestState() }
    func configureRequestState(_ state: any CBv2MTPRequestState, maximumSequenceLength: Int)
        throws
    {
        try native.configureRequestState(state, maximumSequenceLength: maximumSequenceLength)
    }

    func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation, requestState: any CBv2MTPRequestState
    ) {
        native.observeCommittedTarget(observation, requestState: requestState)
    }

    func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        if rounds.insert(ObjectIdentifier(requestState)).inserted {
            starts.append((tokens, requestState.committedInputCount))
        }
        return native.draftStep(
            tokens: tokens, hidden: hidden, shortlist: shortlist, requestState: requestState)
    }

    func evaluationTargets(for state: any CBv2MTPRequestState) -> [MLXArray] {
        native.evaluationTargets(for: state)
    }

    func requestStateDidFinishEvaluation(_ state: any CBv2MTPRequestState) throws {
        try native.requestStateDidFinishEvaluation(state)
    }

    func finalizeRound(
        requestState: any CBv2MTPRequestState, confirmedInputTokens: Int,
        committedDraftTokens: MLXArray, committedTargetHidden: MLXArray
    ) {
        native.finalizeRound(
            requestState: requestState, confirmedInputTokens: confirmedInputTokens,
            committedDraftTokens: committedDraftTokens, committedTargetHidden: committedTargetHidden
        )
        rounds.remove(ObjectIdentifier(requestState))
    }

    func discardRound(requestState: any CBv2MTPRequestState) {
        native.discardRound(requestState: requestState)
        rounds.remove(ObjectIdentifier(requestState))
    }

    func releaseRequestState(_ state: any CBv2MTPRequestState) {
        native.releaseRequestState(state)
        rounds.remove(ObjectIdentifier(state))
    }
}

// A tiny real hybrid target and inline head loaded through public weights.
// The final projections are zero, leaving a wide-margin successor code.
func makeQwenFreeSpanFixture() throws -> (Qwen35TextModel, Qwen35InlineMTPAssistant) {
    let json = Data(
        #"""
        {"model_type":"qwen3_5_moe",
         "mtplx_mtp":{"included":true,"prefix":"mtp.","block_size":3},
         "mtplx_mtp_quantization":{},
         "text_config":{"model_type":"qwen3_5_moe_text","hidden_size":64,
          "num_hidden_layers":4,"intermediate_size":64,"num_attention_heads":2,
          "num_key_value_heads":1,"head_dim":64,"linear_num_value_heads":1,
          "linear_num_key_heads":1,"linear_key_head_dim":64,"linear_value_head_dim":64,
          "linear_conv_kernel_dim":4,"full_attention_interval":2,"vocab_size":64,
          "num_experts":0,"num_experts_per_tok":0,"moe_intermediate_size":32,
          "shared_expert_intermediate_size":32,"norm_topk_prob":true,
          "mtp_num_hidden_layers":1,"tie_word_embeddings":false}}
        """#.utf8)
    let object = try JSONSerialization.jsonObject(with: json) as! [String: Any]
    let configuration = try JSONDecoder().decode(
        Qwen35TextConfiguration.self,
        from: JSONSerialization.data(withJSONObject: object["text_config"]!))
    let target = Qwen35TextModel(configuration)
    var code = [Float](repeating: 0, count: 64 * 64)
    var head = code
    for token in 0 ..< 64 {
        for column in 0 ..< 64 {
            code[token * 64 + column] = token & (1 << (column % 6)) == 0 ? -0.25 : 0.25
        }
    }
    for token in 0 ..< 64 {
        for column in 0 ..< 64 {
            head[token * 64 + column] = code[((token + 63) % 64) * 64 + column]
        }
    }
    var parameters = Dictionary(uniqueKeysWithValues: target.parameters().flattened())
    for (key, value) in parameters {
        parameters[key] = value.asType(.bfloat16)
        if key.hasSuffix(".self_attn.o_proj.weight")
            || key.hasSuffix(".linear_attn.out_proj.weight")
            || key.hasSuffix(".mlp.down_proj.weight")
        {
            parameters[key] = MLXArray.zeros(value.shape, dtype: .bfloat16)
        }
    }
    parameters["model.embed_tokens.weight"] = MLXArray(code, [64, 64]).asType(.bfloat16)
    parameters["lm_head.weight"] = MLXArray(head, [64, 64]).asType(.bfloat16)
    target.update(parameters: ModuleParameters.unflattened(parameters))

    // The head's full-attention block shares the public checkpoint layout.
    var weights: [String: MLXArray] = Dictionary(
        uniqueKeysWithValues: parameters.compactMap { key, value in
            key.hasPrefix("model.layers.1.")
                ? ("mtp.layers.0." + String(key.dropFirst("model.layers.1.".count)), value) : nil
        })
    for key in ["pre_fc_norm_hidden", "pre_fc_norm_embedding", "norm"] {
        weights["mtp.\(key).weight"] = MLXArray.ones([64], dtype: .bfloat16)
    }
    var fusion = [Float](repeating: 0, count: 64 * 128)
    for index in 0 ..< 64 { fusion[index * 128 + index] = 1 }
    weights["mtp.fc.weight"] = MLXArray(fusion, [64, 128]).asType(.bfloat16)
    quantize(model: target, groupSize: 32, bits: 4) { _, module in module is Embedding }
    eval(target)

    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("qwen-free-span-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try json.write(to: directory.appendingPathComponent("config.json"))
    let shard = "model-00001-of-00001.safetensors"
    try save(arrays: weights, url: directory.appendingPathComponent(shard))
    try JSONSerialization.data(withJSONObject: [
        "weight_map": Dictionary(uniqueKeysWithValues: weights.keys.map { ($0, shard) })
    ]).write(to: directory.appendingPathComponent("model.safetensors.index.json"))
    let assistant = try Qwen35InlineMTPAssistant.load(
        from: directory, target: target, verificationMode: .serialTarget)
    return (target, assistant)
}

@Suite("Qwen free-span assistant history", .serialized, .tags(.integration))
struct QwenFreeSpanHistoryTests {
    @Test(arguments: [false, true], [false, true])
    func protectedSpanRetainsTheNativeAssistantHistory(
        ordinaryHistory: Bool, enabled: Bool
    ) async throws {
        let (target, native) = try makeQwenFreeSpanFixture()
        let assistant = ObservedQwenSpanAssistant(native)
        let kinds = target.cbv2LayerKinds
        let adapter = CBv2SteppableLanguageModelAdapter(target)
        let observed = try CBv2NativeKVTypeProbe.run(
            model: adapter, layerKinds: kinds,
            caches: target.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 64 << 20, maxPrefillChunk: 32,
                nominalMaxSequenceLength: 512, segmentSizeBytes: 64 << 10,
                layerDTypes: observed.layerDTypes))
        let storage = backend.makeLayerCaches()
        let indices = Dictionary(
            uniqueKeysWithValues: kinds.enumerated().map {
                ($0.element.modelLayerIndex ?? $0.offset, $0.offset)
            })
        let engine = EngineV2(
            model: adapter, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(
                caches: target.newCacheV2 { index, _ in storage[indices[index]!] }),
            schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 32,
                prefillChunkSize: 32, maxWaiting: 4, enablePrefixCache: false),
            admissionConfig: .init(watermarkFraction: 0),
            mtpDrafter: assistant,
            mtpConfig: .init(
                enabled: true, maxDraftTokens: 2, fixedDraftTokens: 2,
                verificationMode: .serialTarget, ordinaryDecodeHistory: ordinaryHistory,
                prefillCarry: ordinaryHistory, freeSpanSpeculation: enabled))
        let result = await cbv2SchedCollect(
            try engine.submit(
                .init(
                    id: .init(42), promptTokens: [1, 2, 3], sampling: .init(temperature: 0),
                    maxTokens: 20, stopTokens: [63], tokenConstraint: QwenSpanConstraint())))
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()
        let expected = [4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
        #expect(result.tokens == expected)
        #expect(result.finishReason == .length)
        if !enabled {
            #expect(metrics.rounds == 0 && assistant.starts.isEmpty)
            return
        }
        #expect(metrics.acceptedTokens > 0)
        let starts = assistant.starts.map {
            (seed: Int($0.token.item(Int32.self)), history: $0.history)
        }
        #expect(starts.contains { (4 ... 8).contains($0.seed) })
        #expect(starts.contains { $0.seed >= 22 })
        #expect(!starts.contains { [9, 20, 21].contains($0.seed) })
        for start in starts {
            let outputIndex = try #require(expected.firstIndex(of: start.seed))
            #expect(
                start.history == 3 + outputIndex - 1,
                "seed \(start.seed) has \(start.history) trusted transitions")
        }
    }
}

private final class QwenSpanConstraint: CBv2FreeSpanTokenConstraint, @unchecked Sendable {
    let mode = CBv2TokenConstraintMode.auto
    let initialState = 0
    let maxTokens = 20
    let fallbackTokenID = 63

    func allowedTokenIDs(state: Int, remainingTokens: Int) -> [Int] {
        guard remainingTokens > 0 else { return [] }
        switch state {
        case 0, 4: return Array(0 ..< 63)
        case 1: return [20]
        case 2: return [21]
        case 3: return [22]
        default: return []
        }
    }

    func nextState(state: Int, tokenID: Int) -> Int? {
        switch (state, tokenID) {
        case (0, 9): return 1
        case (1, 20): return 2
        case (2, 21): return 3
        case (3, 22): return 4
        case (0, 0 ..< 63), (4, 0 ..< 63): return state
        default: return nil
        }
    }

    func isFreeSpan(state: Int) -> Bool { state == 0 || state == 4 }
    func isFreeSpanContinuation(state: Int, tokenID: Int) -> Bool {
        isFreeSpan(state: state) && tokenID != 9 && (0 ..< 63).contains(tokenID)
    }
}
