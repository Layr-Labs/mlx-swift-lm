// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Tiny LFM2-MoE fixtures shared by the CBv2 parity suites. The topology is
/// the production one in miniature: conv and full-attention layers, two dense
/// feed-forward layers followed by sigmoid-routed MoE layers with expert bias,
/// GQA with head_dim 64, QK norm, RoPE theta 1e7 and a tied embedding head.
enum LFM2MoEFixture {
    enum Precision: String {
        /// Every parameter in float32.
        case float32
        /// bfloat16 activations; router gate and expert bias stay float32
        /// exactly like the published MLX artifact.
        case bfloat16
        /// The artifact's mixed quantization in miniature: affine 4-bit g64
        /// projections and experts, 6-bit g64 tied embedding, float32 router.
        case quantizedMixed
    }

    static let layerTypes = ["conv", "full_attention", "conv", "conv", "full_attention", "conv"]

    static func configuration(convLCache: Int = 3) throws -> LFM2MoEConfiguration {
        let types = layerTypes.map { "\"\($0)\"" }.joined(separator: ",")
        let json = """
            {"model_type":"lfm2_moe","vocab_size":96,"hidden_size":256,
             "intermediate_size":128,"moe_intermediate_size":64,"num_hidden_layers":6,
             "num_experts":8,"num_experts_per_tok":2,"norm_topk_prob":true,
             "num_attention_heads":4,"num_key_value_heads":1,"max_position_embeddings":4096,
             "use_expert_bias":true,"num_dense_layers":2,"norm_eps":1e-5,"conv_bias":false,
             "conv_L_cache":\(convLCache),
             "rope_parameters":{"rope_theta":10000000.0,"rope_type":"default"},
             "routed_scaling_factor":1.0,"tie_word_embeddings":true,
             "layer_types":[\(types)]}
            """
        return try JSONDecoder().decode(LFM2MoEConfiguration.self, from: Data(json.utf8))
    }

    static func isRouterParameter(_ key: String) -> Bool {
        key.hasSuffix("feed_forward.gate.weight") || key.hasSuffix("expert_bias")
    }

    static func model(_ precision: Precision, seed: UInt64 = 2_524_002) throws -> LFM2MoEModel {
        MLXRandom.seed(seed)
        let model = LFM2MoEModel(try configuration())
        // Random selection-only bias so expert choice genuinely depends on it.
        var updates: [(String, MLXArray)] = []
        for (key, value) in model.parameters().flattened() where key.hasSuffix("expert_bias") {
            updates.append((key, MLXRandom.uniform(low: -0.05, high: 0.05, value.shape)))
        }
        model.update(parameters: ModuleParameters.unflattened(updates))
        if precision != .float32 {
            model.update(
                parameters: ModuleParameters.unflattened(
                    model.parameters().flattened().map { key, value in
                        (key, isRouterParameter(key) ? value : value.asType(.bfloat16))
                    }))
        }
        if precision == .quantizedMixed {
            quantize(model: model) { path, module in
                if path.hasSuffix("feed_forward.gate") { return nil }
                if path.hasSuffix("embed_tokens") { return (64, 6, .affine) }
                return module is Linear || module is SwitchLinear ? (64, 4, .affine) : nil
            }
        }
        eval(model)
        return model
    }

    /// Deterministic token IDs (never the pad/edge IDs) of any length.
    static func prompt(_ count: Int, salt: Int = 0) -> [Int] {
        (0 ..< count).map { 1 + (($0 * 37 + salt * 11 + ($0 / 7) * 5) % 90) }
    }

    static var chunk: Int { max(256, CBv2AttentionV1.queryBlockSize) }

    /// Serial (non-CBv2) logits for one prompt plus forced decode tokens.
    /// Returns per-position prompt logits [L, V] and one [V] row per decode step.
    static func serial(
        _ model: LFM2MoEModel, prompt: [Int], decode: [Int], prefillChunk: Int? = nil
    ) -> (prompt: MLXArray, decode: [MLXArray]) {
        let cache = model.newCache(parameters: nil)
        var promptRows: [MLXArray] = []
        let step = prefillChunk ?? prompt.count
        for start in stride(from: 0, to: prompt.count, by: step) {
            let piece = Array(prompt[start ..< min(start + step, prompt.count)])
            let logits = model(MLXArray(piece.map(Int32.init)).reshaped(1, piece.count), cache: cache)
            eval(logits)
            promptRows.append(logits[0])
        }
        var decodeRows: [MLXArray] = []
        for token in decode {
            let logits = model(MLXArray([Int32(token)]).reshaped(1, 1), cache: cache)
            eval(logits)
            decodeRows.append(logits[0, 0])
        }
        let promptLogits = promptRows.count == 1 ? promptRows[0] : concatenated(promptRows, axis: 0)
        return (promptLogits, decodeRows)
    }

    static func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
}

/// One CBv2 KV backend (contiguous or native paged) plus its layer caches.
final class LFM2MoECBv2Storage {
    let backend: any CBv2KVBackend
    let bank: CBv2LayerCacheBank
    let kinds: [CBv2LayerKind]
    let types: [DType]

    init(
        model: LFM2MoEModel, paged: Bool, maximumLength: Int, capacity: Int = 512 << 20,
        maxPrefillChunk: Int = LFM2MoEFixture.chunk
    ) throws {
        kinds = model.cbv2LayerKinds
        types = try XCTUnwrap(model.cbv2CompleteCheckpointKVDTypes, "missing native KV dtypes")
        XCTAssertEqual(types.count, kinds.count)
        let caches: [any CBv2AttendingLayerCache]
        if paged {
            let pages = try PagedKVBackend(
                layerKinds: kinds,
                config: .init(
                    capacityBytes: capacity, maxPrefillChunk: maxPrefillChunk,
                    nominalMaxSequenceLength: maximumLength, segmentSizeBytes: 64 << 10,
                    layerDTypes: types))
            backend = pages
            let storage = pages.makeLayerCaches()
            let indices = Dictionary(
                uniqueKeysWithValues: kinds.enumerated().map {
                    ($0.element.modelLayerIndex ?? $0.offset, $0.offset)
                })
            caches = try model.newCacheV2 { index, _ in storage[try XCTUnwrap(indices[index])] }
        } else {
            backend = CBv2ContiguousKVBackend(
                config: .init(
                    bytesCapacity: capacity, kvDType: types.max { $0.size < $1.size }!))
            caches = model.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) }
        }
        bank = CBv2LayerCacheBank(caches: caches)
    }

    var paged: PagedKVBackend? { backend as? PagedKVBackend }
}

/// One request: its own KV rows and request-owned recurrent state.
final class LFM2MoECBv2Request {
    let storage: LFM2MoECBv2Storage
    var recurrent: CBv2RecurrentRequestState
    var rows: [CBv2SequenceKV?]

    init(
        model: LFM2MoEModel, storage: LFM2MoECBv2Storage, promptLength: Int, maximumLength: Int,
        recurrent: CBv2RecurrentRequestState? = nil, rows: [CBv2SequenceKV?]? = nil
    ) throws {
        self.storage = storage
        self.recurrent = try recurrent ?? CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
        self.rows = try rows ?? storage.backend.makeSequenceState(
            layerKinds: model.cbv2LayerKinds, promptLength: promptLength, maxLength: maximumLength)
    }

    func close() {
        storage.backend.release(rows)
        rows.removeAll()
        recurrent.discardPendingAfterSynchronization()
        try? recurrent.release()
    }
}

/// Drives the production adapter exactly as the engine does: bind each
/// row's recurrent transaction, forward the rectangle, evaluate, commit.
enum LFM2MoECBv2Driver {
    @discardableResult
    static func forward(
        _ model: LFM2MoEModel, storage: LFM2MoECBv2Storage, requests: [LFM2MoECBv2Request],
        tokens: [[Int]], commit: Bool = true
    ) throws -> (logits: MLXArray, transactions: [CBv2RecurrentStateEvaluation]) {
        precondition(requests.count == tokens.count)
        let width = tokens[0].count
        precondition(tokens.allSatisfy { $0.count == width })
        let adapter = CBv2SteppableLanguageModelAdapter(model)
        let caches = storage.bank.layerCaches(rowStates: requests.map(\.rows))
        let bindings = try requests.map { try $0.recurrent.bind() }
        let input = MLXArray(tokens.flatMap { $0 }.map(Int32.init)).reshaped(tokens.count, width)
        let output = adapter.forward(tokens: input, caches: caches, recurrentState: bindings)
        try storage.paged?.pool.writeValidation.check()
        var roots: [MLXArray] = [output]
        for binding in bindings { roots.append(contentsOf: try binding.evaluate()) }
        for cache in caches { roots.append(contentsOf: try XCTUnwrap(cache as? KVCache).innerState()) }
        eval(roots)
        if commit { for binding in bindings { try binding.commit() } }
        return (output, bindings)
    }

    /// Prefill one request in `chunk`-sized pieces; returns [L, V] logits.
    static func prefill(
        _ model: LFM2MoEModel, storage: LFM2MoECBv2Storage, request: LFM2MoECBv2Request,
        prompt: [Int], chunk: Int
    ) throws -> MLXArray {
        var rows: [MLXArray] = []
        for start in stride(from: 0, to: prompt.count, by: chunk) {
            let piece = Array(prompt[start ..< min(start + chunk, prompt.count)])
            rows.append(
                try forward(model, storage: storage, requests: [request], tokens: [piece]).logits[0])
        }
        return rows.count == 1 ? rows[0] : concatenated(rows, axis: 0)
    }
}

/// EngineV2 runs over LFM2-MoE with hybrid checkpoints. The complete-checkpoint
/// store is the in-memory test fixture (`CompleteCheckpointFixtureStore`),
/// standing in for the provider's SSD tier; the codec and engine are real.
struct LFM2MoEEngineHarness {
    final class Positions: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [[Int]] = []
        func append(_ positions: [Int]) { lock.withLock { values.append(positions) } }
        var snapshot: [[Int]] { lock.withLock { values } }
    }

    struct Run {
        let tokens: [Int]
        let archives: [CompleteCheckpointFixtureStore.Archive]
    }

    let model: LFM2MoEModel
    var chunk = LFM2MoEFixture.chunk
    var capacity = 512 << 20
    var maxTokens = 6
    var timeout: TimeInterval = 180

    func storage(paged: Bool, length: Int) throws -> LFM2MoECBv2Storage {
        try LFM2MoECBv2Storage(
            model: model, paged: paged, maximumLength: length + maxTokens + 8, capacity: capacity,
            maxPrefillChunk: chunk)
    }

    func engine(
        storage: LFM2MoECBv2Storage, resident: Bool, store: CompleteCheckpointFixtureStore? = nil
    ) -> EngineV2 {
        EngineV2(
            model: CBv2SteppableLanguageModelAdapter(model), layerKinds: model.cbv2LayerKinds,
            backend: storage.backend, cacheProvider: storage.bank, sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: chunk,
                prefillChunkSize: chunk, maxWaiting: 2, enablePrefixCache: resident || store != nil),
            admissionConfig: .init(watermarkFraction: 0),
            hybridPrefixCache: resident
                ? .init(
                    maximumBytes: max(64 << 20, capacity / 2), modelID: "lfm2moe-fixture",
                    promptContractID: "fixture-template", buildID: "fixture-build")
                : nil,
            completePrefixCache: store)
    }

    /// A cold run with no prefix machinery at all.
    func cold(prompt: [Int], paged: Bool) async throws -> [Int] {
        let storage = try storage(paged: paged, length: prompt.count)
        let engine = engine(storage: storage, resident: false)
        let result = await cbv2SchedCollect(
            try engine.submit(.init(
                id: .init(1), promptTokens: prompt, sampling: .init(temperature: 0),
                maxTokens: maxTokens)),
            timeoutSeconds: timeout)
        XCTAssertEqual(result.finishReason, .length)
        XCTAssertEqual(result.usage?.prefixCachePrefillTokensSaved ?? 0, 0)
        await engine.shutdown()
        XCTAssertEqual(storage.backend.bytesReserved, 0)
        return result.tokens
    }

    /// One request through the complete-checkpoint codec. With `archives`, the
    /// request first stages the deepest matching checkpoint and must save
    /// exactly `expectedSaved` prompt tokens.
    func completeRun(
        paged: Bool, prompt: [Int], archives: [CompleteCheckpointFixtureStore.Archive] = [],
        expectedSaved: Int = 0
    ) async throws -> Run {
        let store = CompleteCheckpointFixtureStore(archives: archives, segmentBytes: 64 << 10)
        let storage = try storage(paged: paged, length: prompt.count)
        let engine = engine(storage: storage, resident: false, store: store)
        XCTAssertNotNil(engine.completeCheckpointCodec, "paged=\(paged): complete checkpoint codec unavailable")
        let published = Positions()
        engine.setCompletePrefixPublicationHandler { _, positions in published.append(positions) }
        let request = CBv2Request(
            id: .init(25), promptTokens: prompt, sampling: .init(temperature: 0), maxTokens: maxTokens,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(125))
        if expectedSaved > 0 {
            XCTAssertTrue(try store.stage(engine: engine, request: request), "checkpoint was not staged")
        }
        let result = await cbv2SchedCollect(try engine.submit(request), timeoutSeconds: timeout)
        XCTAssertEqual(result.finishReason, .length)
        XCTAssertEqual(result.usage?.prefixCachePrefillTokensSaved ?? 0, expectedSaved)
        XCTAssertEqual(result.usage?.prefixCacheReplayTokens ?? 0, 0)
        let eligible = stride(from: chunk, to: prompt.count, by: chunk).filter { $0 > expectedSaved }
        if let last = eligible.last {
            let settled = await cbv2SchedWait(timeoutSeconds: 30) {
                Set(published.snapshot.flatMap { $0 }).contains(last)
            }
            XCTAssertTrue(settled, "paged=\(paged): publication did not settle")
        }
        let saved = Array(store.saved.dropFirst(archives.count))
        await engine.shutdown()
        XCTAssertEqual(storage.backend.bytesReserved, 0)
        XCTAssertTrue(engine.loopForTesting.recurrentStates.isEmpty)
        if let pages = storage.paged { XCTAssertEqual(pages.bytesWired, 0) }
        return Run(tokens: result.tokens, archives: saved)
    }
}
