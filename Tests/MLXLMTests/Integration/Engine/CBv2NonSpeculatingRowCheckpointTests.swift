import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Miniature Qwen 3.5 and native Qwen4 targets with their inline assistants on
/// a durable complete-checkpoint store. A stop string, a penalty, a logit
/// bias, logprobs or a token constraint keeps a row from speculating; its
/// prompt must still save the checkpoints, assistant history included, that an
/// unconstrained request saves. Synthetic weights, no model load.
@Suite("Checkpoints of rows that cannot speculate", .serialized, .tags(.integration))
struct CBv2NonSpeculatingRowCheckpointTests {
    typealias Control = CBv2NonSpeculatingControl
    typealias Store = CompleteCheckpointFixtureStore

    static let chunk: Int = max(32, CBv2AttentionV1.queryBlockSize)

    private static func bfloat16(_ model: Module) {
        model.update(
            parameters: ModuleParameters.unflattened(
                model.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
    }

    private static func engine(
        _ adapter: any CBv2SteppableModel, kinds: [CBv2LayerKind], backend: PagedKVBackend,
        caches: [any CBv2AttendingLayerCache], store: Store?, drafter: any CBv2MTPDrafter,
        verification: CBv2MTPVerificationMode
    ) -> (EngineV2, PagedKVBackend) {
        let engine = EngineV2(
            model: adapter, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: caches), sampler: CBv2DefaultSampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: chunk,
                prefillChunkSize: chunk, maxWaiting: 4, enablePrefixCache: store != nil),
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store,
            mtpDrafter: drafter,
            mtpConfig: .init(
                enabled: true, maxDraftTokens: 2, fixedDraftTokens: 2,
                verificationMode: verification))
        #expect(engine.hybridPrefixCache == nil)
        #expect((engine.completeCheckpointCodec != nil) == (store != nil))
        return (engine, backend)
    }

    private static func qwen35Engine(_ store: Store?) throws -> (EngineV2, PagedKVBackend) {
        let json = Data(
            """
            {
              "model_type": "qwen3_5_moe", "mtplx_mtp": {
                "included": true, "prefix": "mtp.", "block_size": 3
              }, "mtplx_mtp_quantization": {},
              "text_config": {
                "model_type": "qwen3_5_moe_text", "hidden_size": 64,
                "num_hidden_layers": 4, "intermediate_size": 64,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
                "linear_num_value_heads": 1, "linear_num_key_heads": 1,
                "linear_key_head_dim": 64, "linear_value_head_dim": 64,
                "linear_conv_kernel_dim": 4, "full_attention_interval": 2,
                "vocab_size": 64, "num_experts": 0, "num_experts_per_tok": 0,
                "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
                "norm_topk_prob": true, "mtp_num_hidden_layers": 1
              }
            }
            """.utf8)
        let object = try JSONSerialization.jsonObject(with: json) as! [String: Any]
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: JSONSerialization.data(withJSONObject: object["text_config"]!))
        MLXRandom.seed(7029)
        let model = Qwen35TextModel(configuration)
        bfloat16(model)
        quantize(model: model, groupSize: 32, bits: 4) { _, module in module is Embedding }
        let weights = Dictionary(
            uniqueKeysWithValues: Qwen35MTPModule(configuration).parameters().flattened().map {
                ("mtp." + $0.0, $0.1.asType(.bfloat16))
            })
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("cbv2-controlled-rows-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try json.write(to: directory.appendingPathComponent("config.json"))
        let shard = "model-00001-of-00001.safetensors"
        try save(arrays: weights, url: directory.appendingPathComponent(shard))
        let index = [
            "weight_map": Dictionary(uniqueKeysWithValues: weights.keys.map { ($0, shard) })
        ]
        try JSONSerialization.data(withJSONObject: index)
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        eval(model)
        let assistant = try Qwen35InlineMTPAssistant.load(
            from: directory, target: model, verificationMode: .serialTarget)

        let kinds = model.cbv2LayerKinds
        let adapter = CBv2SteppableLanguageModelAdapter(model)
        let observed = try CBv2NativeKVTypeProbe.run(
            model: adapter, layerKinds: kinds,
            caches: model.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 96 << 20, maxPrefillChunk: chunk, nominalMaxSequenceLength: 512,
                segmentSizeBytes: 64 << 10, layerDTypes: observed.layerDTypes))
        let storage = backend.makeLayerCaches()
        let indices = Dictionary(
            uniqueKeysWithValues: kinds.enumerated().map {
                ($0.element.modelLayerIndex ?? $0.offset, $0.offset)
            })
        return engine(
            adapter, kinds: kinds, backend: backend,
            caches: model.newCacheV2 { index, _ in storage[indices[index]!] }, store: store,
            drafter: assistant, verification: .serialTarget)
    }

    private static func qwen4Engine(_ store: Store?, verification: CBv2MTPVerificationMode) throws
        -> (EngineV2, PagedKVBackend)
    {
        var c = Qwen4ExpTextConfiguration()
        c.hiddenSize = 64
        c.hiddenLayers = 2
        c.attentionHeads = 2
        c.kvHeads = 1
        c.headDim = 64
        c.linearNumValueHeads = 2
        c.linearNumKeyHeads = 1
        c.linearKeyHeadDim = 64
        c.linearValueHeadDim = 64
        c.vocabularySize = 64
        c.maxPositionEmbeddings = 512
        c.fullAttentionInterval = 2
        c.layerTypes = ["linear_attention", "qwen_sparse_attention"]
        c.hcCount = 2
        c.hcLowrank = 8
        c.pleLayerIds = []
        c.pleEmbedDim = 64
        c.indexerNHeads = 2
        c.indexerKVHeads = 1
        c.indexerHeadDim = 32
        c.indexerBudget = 16
        c.indexerCompressRatio = 4
        c.numExperts = 1
        c.numExpertsPerTok = 1
        c.sharedExpertIntermediateSize = 32
        c.moeIntermediateSize = 32
        c.mropeSection = [2, 1, 1]
        c.partialRotaryFactor = 0.25
        c.mtpNumHiddenLayers = 1
        MLXRandom.seed(8301)
        let model = Qwen4ExpTextModel(c)
        bfloat16(model)
        eval(model)

        let kinds = model.cbv2LayerKinds
        let adapter = CBv2SteppableLanguageModelAdapter(model)
        let observed = try CBv2NativeKVTypeProbe.run(
            model: adapter, layerKinds: kinds,
            caches: model.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 256 << 20, maxPrefillChunk: chunk, nominalMaxSequenceLength: 512,
                segmentSizeBytes: 64 << 10, layerDTypes: observed.layerDTypes))
        let assistant = try Qwen4ExpInlineMTPAssistant(
            configuration: model.configuration, blockSize: 3, target: model,
            verificationMode: verification)
        eval(assistant)
        return engine(
            adapter, kinds: kinds, backend: backend, caches: backend.makeLayerCaches(),
            store: store, drafter: assistant, verification: verification)
    }

    /// Every charge and every assistant request state is back.
    private static func released(_ engine: EngineV2, _ backend: PagedKVBackend) {
        let state = engine.loopForTesting.onEngineQueueSync {
            (
                engine.admissionForTesting.bytesReserved, backend.bytesReserved,
                backend.bytesWired, engine.loopForTesting.recurrentStates.isEmpty,
                engine.loopForTesting.mtp?.requestStateCountForTesting
            )
        }
        #expect(state.0 == 0 && state.1 == 0 && state.2 == 0)
        #expect(state.3)
        #expect(state.4 == 0)
    }

    @Test(.timeLimit(.minutes(5)), arguments: Control.allCases)
    func qwen35RowSavesAndReusesCheckpoints(_ control: Control) async throws {
        try await cbv2ExpectNonSpeculatingDonorCheckpoints(
            control, chunk: Self.chunk, maxTokens: 12, segmentBytes: 4096,
            make: { try Self.qwen35Engine($0) }, released: Self.released)
    }

    @Test(
        .timeLimit(.minutes(5)),
        arguments: [CBv2MTPVerificationMode.serialTarget, .rectangularExact], Control.allCases)
    func qwen4RowSavesAndReusesNativeCheckpoints(
        _ verification: CBv2MTPVerificationMode, _ control: Control
    ) async throws {
        try await cbv2ExpectNonSpeculatingDonorCheckpoints(
            control, chunk: Self.chunk, maxTokens: 8, segmentBytes: 1024,
            make: { try Self.qwen4Engine($0, verification: verification) },
            released: Self.released)
    }
}
