#if canImport(Metal) && canImport(ObjectiveC)
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A tiny Lightning (Nemotron H 3.5) target with its request-stateful MTP
/// assistant on paged storage. The assistant keeps lazy views of every
/// observed step (`pendingHidden`, `frontier`) until it drafts, and drafts
/// one row at a time, so rows that decode together only feed its history.
@Suite("Nemotron H 3.5 MTP history Metal events", .serialized, .tags(.integration))
struct NemotronH35MTPHistoryEventTests {
    private func engine() throws -> EngineV2 {
        let data = Data(
            """
            {"model_type":"nemotron_h","vocab_size":128,"hidden_size":64,
             "num_hidden_layers":2,"num_attention_heads":1,"num_key_value_heads":1,
             "head_dim":64,"mamba_num_heads":4,"mamba_head_dim":16,"ssm_state_size":16,
             "conv_kernel":4,"n_groups":2,"intermediate_size":128,"moe_intermediate_size":64,
             "moe_shared_expert_intermediate_size":64,"n_routed_experts":4,
             "num_experts_per_tok":2,"layers_block_type":["mamba","attention"],
             "mamba_ssm_cache_dtype":"float32","num_nextn_predict_layers":1,
             "mtp_layers_block_type":["attention","moe"]}
            """.utf8)
        MLXRandom.seed(35_196)
        let model = NemotronH35Model(
            try JSONDecoder().decode(NemotronH35Configuration.self, from: data))
        let assistant = NemotronH35MTPAssistant(target: model, maximumDraftTokens: 2)
        eval(model, assistant)
        let kinds = model.cbv2LayerKinds
        let adapter = CBv2SteppableLanguageModelAdapter(model)
        let observed = try CBv2NativeKVTypeProbe.run(
            model: adapter, layerKinds: kinds,
            caches: model.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
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
        return EngineV2(
            model: adapter, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(
                caches: model.newCacheV2 { index, _ in storage[indices[index]!] }),
            sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 2, maxBatchedTokensPerStep: 32,
                prefillChunkSize: 32, maxWaiting: 4, enablePrefixCache: false),
            admissionConfig: .init(watermarkFraction: 0),
            mtpDrafter: assistant,
            mtpConfig: .init(enabled: true, maxDraftTokens: 2, fixedDraftTokens: 2))
    }

    /// Production Lightning restarted once right after a 30-minute stretch of
    /// two or more rows (about 36K steps). The live count must not grow with
    /// the steps the rows decode together.
    @Test("batch-gated rows keep no Metal event per decode step")
    func batchGatedHistoryKeepsNoEventPerStep() async throws {
        _ = try #require(MetalSharedEventCounter.live, "needs a Metal device")
        let requests = [
            CBv2Request(
                id: .init(961), promptTokens: (1 ... 17).map { $0 },
                sampling: .init(temperature: 0), maxTokens: 48),
            CBv2Request(
                id: .init(962), promptTokens: (3 ... 15).map { $0 },
                sampling: .init(temperature: 0), maxTokens: 48),
        ]
        let steps = 12 ... 36
        let sample = try #require(
            await MetalSharedEventCounter.sampleBatchGatedStretch(
                try engine(), requests: requests, steps: steps))

        #expect(sample.outputs.map(\.count) == [48, 48])
        #expect((sample.metrics?.skippedRows["batch_gate"] ?? 0) > 0)
        #expect(
            sample.late - sample.early <= 4,
            "live Metal shared events \(sample.early) -> \(sample.late) over \(steps)")
    }
}
#endif
