import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

@Suite("Qwen MTP paged assistant history", .serialized, .tags(.integration))
struct QwenMTPPagedHistoryTests {
    private func exact(_ a: MLXArray, _ b: MLXArray) {
        eval(a, b)
        #expect(a.shape == b.shape && a.dtype == b.dtype)
        #expect(a.asData().data == b.asData().data)
    }

    private func fixture(_ body: (Qwen35InlineMTPAssistant) throws -> Void) throws {
        try fixtureSources { assistant, _, _, _ in try body(assistant) }
    }

    private func fixtureSources(
        _ body: (Qwen35InlineMTPAssistant, Qwen35TextModel, Data, [String: MLXArray]) throws -> Void
    ) throws {
        let text: [String: Any] = [
            "model_type": "qwen3_5_text", "hidden_size": 64,
            "num_hidden_layers": 1, "intermediate_size": 128, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 64,
            "linear_num_value_heads": 2, "linear_num_key_heads": 1,
            "linear_key_head_dim": 16, "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4, "full_attention_interval": 1,
            "num_experts": 0, "num_experts_per_tok": 0, "mtp_num_hidden_layers": 1,
        ]
        let root: [String: Any] = [
            "model_type": "qwen3_5",
            "text_config": text,
            "mtplx_mtp": ["included": true, "prefix": "mtp.", "block_size": 3],
            "mtplx_mtp_quantization": ["group_size": 64, "bits": 4, "mode": "affine"],
        ]
        let config = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: JSONSerialization.data(withJSONObject: text))
        MLXRandom.seed(441)
        let target = Qwen35TextModel(config)
        let donor = Qwen35MTPModule(config)
        for model in [target as Module, donor as Module] {
            model.update(
                parameters: ModuleParameters.unflattened(
                    model.parameters().flattened().map {
                        ($0.0, $0.1.asType(.bfloat16))
                    }))
            quantize(model: model, groupSize: 64, bits: 4)
        }
        eval(target, donor)
        let weights = Dictionary(
            uniqueKeysWithValues: donor.parameters().flattened().map { ("mtp." + $0.0, $0.1) })
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "qwen-mtp-paged-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try JSONSerialization.data(withJSONObject: root).write(
            to: directory.appendingPathComponent("config.json"))
        try save(arrays: weights, url: directory.appendingPathComponent("mtp.safetensors"))
        try JSONSerialization.data(withJSONObject: [
            "weight_map": weights.mapValues { _ in "mtp.safetensors" }
        ])
        .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        try body(
            Qwen35InlineMTPAssistant.load(from: directory, target: target), target,
            JSONSerialization.data(withJSONObject: root),
            Dictionary(
                uniqueKeysWithValues: weights.map { (String($0.key.dropFirst(4)), $0.value) }))
    }

    @Test("verified arrays preserve path loading and reject incomplete weights")
    func verifiedArrayConstruction() throws {
        try Device.withDefaultDevice(.cpu) {
            try fixtureSources { original, target, configuration, weights in
                let loaded = try Qwen35InlineMTPAssistant.load(
                    configurationData: configuration,
                    indexedWeights: weights, target: target, allowsShortlist: false)
                let caches = [original.makeCache(), loaded.makeCache()]
                for count in [3, 1, 65, 2] {
                    let tokens = MLXArray((0 ..< count).map { Int32(($0 * 3 + 7) % 64) }).reshaped([
                        1, count,
                    ])
                    let hidden = MLXArray((0 ..< (count * 64)).map { Float($0 % 11) * 0.03 })
                        .reshaped([1, count, 64]).asType(.bfloat16)
                    let first = original.forward(hidden: hidden, tokens: tokens, cache: caches[0])
                    let second = loaded.forward(hidden: hidden, tokens: tokens, cache: caches[1])
                    exact(first.logits, second.logits)
                    exact(first.hidden, second.hidden)
                }
                #expect(throws: (any Error).self) {
                    try Qwen35InlineMTPAssistant.load(
                        configurationData: configuration, indexedWeights: [:], target: target)
                }
                var incomplete = weights
                incomplete.removeValue(forKey: "fc.weight")
                #expect(throws: (any Error).self) {
                    try Qwen35InlineMTPAssistant.load(
                        configurationData: configuration, indexedWeights: incomplete, target: target
                    )
                }
            }
        }
    }

    @Test("verified loader transfers construction custody before a failing update")
    func failedConstructionCustody() throws {
        try Device.withDefaultDevice(.cpu) {
            try fixtureSources { _, target, configuration, weights in
                var retained: Qwen35InlineMTPAssistant?
                weak var observed: Qwen35InlineMTPAssistant?
                var calls = 0
                var incomplete = weights
                incomplete.removeValue(forKey: "fc.weight")
                #expect(throws: (any Error).self) {
                    try Qwen35InlineMTPAssistant.load(
                        configurationData: configuration,
                        indexedWeights: incomplete, target: target,
                        retainDuringLoad: { assistant in
                            calls += 1
                            retained = assistant
                            observed = assistant
                        })
                }
                #expect(calls == 1 && retained != nil && observed != nil)
                Stream.cpu.synchronize()
                retained = nil
                #expect(observed == nil)
                let loaded = try Qwen35InlineMTPAssistant.load(
                    configurationData: configuration,
                    indexedWeights: weights, target: target,
                    retainDuringLoad: { assistant in
                        calls += 1
                        retained = assistant
                    })
                #expect(calls == 2 && retained === loaded)
                retained = nil
                #expect(throws: (any Error).self) {
                    try Qwen35InlineMTPAssistant.load(
                        configurationData: configuration,
                        indexedWeights: [:], target: target, retainDuringLoad: { _ in calls += 1 })
                }
                #expect(calls == 2)
            }
        }
    }

    @Test("pages match growing cache across boundaries, trim, copies and restore")
    func cacheStorageAndRollback() {
        Device.withDefaultDevice(.cpu) {
            for dtype in [DType.bfloat16, .float16, .float32] {
                let paged = Qwen35MTPPagedKVCache(capacityTokens: 512)
                let ordinary = KVCacheSimple()
                #expect(paged.pageCount == 0 && paged.innerState().isEmpty)
                var cursor = 0
                for count in [1, 62, 1, 65, 8] {
                    let keys = MLXArray((cursor ..< (cursor + count * 16)).map { Float($0 % 19) })
                        .reshaped([1, 2, count, 8]).asType(dtype)
                    let values = -keys
                    let a = paged.update(keys: keys, values: values)
                    let b = ordinary.update(keys: keys, values: values)
                    exact(a.0, b.0)
                    exact(a.1, b.1)
                    #expect(
                        paged.offset == ordinary.offset
                            && paged.pageCount == (paged.offset + 63) / 64)
                    cursor += count * 16
                }
                let snapshot = paged.copy()
                let before = snapshot.state.map { $0.asData().data }
                #expect(paged.trim(74) == ordinary.trim(74))
                #expect(paged.offset == 63 && paged.pageCount == 1)
                let next = MLXArray((0 ..< (70 * 16)).map { Float($0 % 13) }).reshaped([
                    1, 2, 70, 8,
                ]).asType(dtype)
                let a = paged.update(keys: next, values: -next)
                let b = ordinary.update(keys: next, values: -next)
                exact(a.0, b.0)
                exact(a.1, b.1)
                #expect(snapshot.state.map { $0.asData().data } == before)
                let restored = Qwen35MTPPagedKVCache(capacityTokens: 512)
                restored.state = paged.state
                restored.metaState = paged.metaState
                exact(restored.state[0], paged.state[0])
                exact(restored.state[1], paged.state[1])
                #expect(restored.offset == paged.offset)
                if case .array(let a) = paged.makeMask(n: 3, windowSize: nil, returnArray: true),
                    case .array(let b) = ordinary.makeMask(n: 3, windowSize: nil, returnArray: true)
                {
                    exact(a, b)
                } else {
                    Issue.record("Missing causal offset masks")
                }
                #expect(
                    paged.trim(Int.max) == 133 && paged.pageCount == 0 && paged.innerState().isEmpty
                )
            }
        }
    }

    @Test("real packed assistant preserves draft, accepted-prefix and discard history")
    func realAssistantRounds() throws {
        try Device.withDefaultDevice(.cpu) {
            try fixture { assistant in
                let paged = try assistant.makePagedRequestState(maximumSequenceLength: 512)
                let ordinary = assistant.makeRequestState()
                let states = [paged, ordinary]
                var offset = 0
                for count in [63, 1, 65, 11] {
                    let tokens = MLXArray((offset ..< (offset + count)).map { Int32($0 % 64) })
                        .reshaped([1, count])
                    let hidden = MLXArray(
                        (offset ..< (offset + count * 64)).map { Float($0 % 17) * 0.01 }
                    ).reshaped([1, count, 64]).asType(.bfloat16)
                    for state in states {
                        assistant.observeCommittedTarget(
                            .init(tokens: tokens, hidden: hidden), requestState: state)
                    }
                    offset += count
                }
                for (round, depth) in [1, 4, 2, 3, 4].enumerated() {
                    let carry = MLXArray(Array(repeating: Float(round + 1) * 0.03, count: 64))
                        .reshaped([1, 1, 64]).asType(.bfloat16)
                    var hidden = [carry, carry]
                    var tokens = states.map { _ in MLXArray([Int32(round + 7)]).reshaped([1, 1]) }
                    var drafts: [[MLXArray]] = [[], []]
                    for _ in 0 ..< depth {
                        for i in 0 ..< 2 {
                            let result = assistant.draftStep(
                                tokens: tokens[i], hidden: hidden[i], shortlist: nil,
                                requestState: states[i])
                            eval(
                                [result.tokens, result.hidden]
                                    + assistant.evaluationTargets(for: states[i]))
                            tokens[i] = result.tokens.reshaped([1, 1])
                            hidden[i] = result.hidden
                            drafts[i].append(tokens[i])
                        }
                        exact(tokens[0], tokens[1])
                        exact(hidden[0], hidden[1])
                    }
                    if round == 2 {
                        for state in states { assistant.discardRound(requestState: state) }
                    } else {
                        let accepted = min(round, depth)
                        let trusted = MLXArray(
                            Array(repeating: Float(round) * 0.02, count: accepted * 64)
                        ).reshaped([1, accepted, 64]).asType(.bfloat16)
                        for i in 0 ..< 2 {
                            let selected = concatenated(drafts[i], axis: 1)[0..., 0 ..< accepted]
                            assistant.finalizeRound(
                                requestState: states[i], confirmedInputTokens: accepted + 1,
                                committedDraftTokens: selected, committedTargetHidden: trusted)
                        }
                    }
                    #expect(
                        paged.committedInputCount == ordinary.committedInputCount
                            && paged.stagedInputCount == 0)
                    let a = try #require(paged as? Qwen35InlineMTPAssistant.RequestState)
                    let b = try #require(ordinary as? Qwen35InlineMTPAssistant.RequestState)
                    #expect(a.caches.allSatisfy { $0 is Qwen35MTPPagedKVCache })
                    for (x, y) in zip(a.caches, b.caches) {
                        exact(x.state[0], y.state[0])
                        exact(x.state[1], y.state[1])
                    }
                }
                for state in states { assistant.releaseRequestState(state) }
                #expect(paged.materializedBytes == 0 && ordinary.materializedBytes == 0)
                #expect(assistant.evaluationTargets(for: paged).isEmpty)
            }
        }
    }

    @Test("prefix restoration uses fresh pages and invalid capacities refuse")
    func prefixOwnershipAndBounds() throws {
        try Device.withDefaultDevice(.cpu) {
            try fixture { assistant in
                for bad in [Int.min, -1, 0, (1 << 20) + 1, Int.max] {
                    #expect(throws: Qwen35InlineMTPError.self) {
                        try assistant.makePagedRequestState(maximumSequenceLength: bad)
                    }
                }
                let original = try assistant.makePagedRequestState(maximumSequenceLength: 256)
                let tokens = MLXArray((0 ..< 65).map { Int32($0 % 64) }).reshaped([1, 65])
                let hidden = MLXArray(Array(repeating: Float(0.1), count: 65 * 64)).reshaped([
                    1, 65, 64,
                ]).asType(.bfloat16)
                assistant.observeCommittedTarget(
                    .init(tokens: tokens, hidden: hidden), requestState: original)
                let checkpoint = try #require(
                    assistant.capturePrefixCheckpoint(requestState: original, targetInputCount: 65))
                eval(checkpoint.evaluationTargets)
                let first = try #require(
                    try assistant.restorePagedPrefixCheckpoint(
                        checkpoint, maximumSequenceLength: 256)
                        as? Qwen35InlineMTPAssistant.RequestState)
                let second = try #require(
                    try assistant.restorePagedPrefixCheckpoint(
                        checkpoint, maximumSequenceLength: 256)
                        as? Qwen35InlineMTPAssistant.RequestState)
                #expect(first.caches[0] is Qwen35MTPPagedKVCache && second.caches[0].offset == 0)
                let draft = assistant.draftStep(
                    tokens: MLXArray([Int32(5)]).reshaped([1, 1]),
                    hidden: hidden[0..., 64 ..< 65, 0...], shortlist: nil, requestState: first)
                eval([draft.tokens, draft.hidden] + assistant.evaluationTargets(for: first))
                #expect(first.caches[0].offset == 65 && second.caches[0].offset == 0)
                #expect(
                    try assistant.restorePagedPrefixCheckpoint(
                        checkpoint, maximumSequenceLength: 64) == nil)
                for state in [original, first, second] { assistant.releaseRequestState(state) }
                #expect(first.materializedBytes == 0 && second.materializedBytes == 0)
            }
        }
    }
}
