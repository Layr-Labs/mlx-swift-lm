import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A real tiny Qwen3.5 target with the production-loaded inline assistant on
/// paged storage: batch-gated rows chain as ordinary decode while feeding the
/// assistant's persistent history, and prompts seed the first carry. These are
/// mechanics tests, not numerical or latency evidence.
@Suite("Qwen3.5 inline MTP ordinary-decode history", .serialized)
struct Qwen35MTPOrdinaryHistoryTests {
    private func fixture() throws -> (Qwen35TextModel, Qwen35InlineMTPAssistant) {
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
        MLXRandom.seed(7031)
        let model = Qwen35TextModel(configuration)
        model.update(
            parameters: ModuleParameters.unflattened(
                model.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
        quantize(model: model, groupSize: 32, bits: 4) { _, module in module is Embedding }
        let draft = Qwen35MTPModule(configuration)
        let weights = Dictionary(
            uniqueKeysWithValues: draft.parameters().flattened().map {
                ("mtp." + $0.0, $0.1.asType(.bfloat16))
            })
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-mtp-history-\(UUID().uuidString)", isDirectory: true)
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
        return (model, assistant)
    }

    private func engine(mtpConfig: CBv2MTPConfig) throws -> (EngineV2, PagedKVBackend) {
        let (model, assistant) = try fixture()
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
        let engine = EngineV2(
            model: adapter, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(
                caches: model.newCacheV2 { index, _ in storage[indices[index]!] }),
            sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 2, maxBatchedTokensPerStep: 32,
                prefillChunkSize: 32, maxWaiting: 4, enablePrefixCache: false),
            admissionConfig: .init(watermarkFraction: 0),
            mtpDrafter: assistant, mtpConfig: mtpConfig)
        return (engine, backend)
    }

    /// Admit both requests atomically so they decode together from step two.
    private func runTogether(
        _ engine: EngineV2, _ requests: [CBv2Request]
    ) async throws -> [[Int]] {
        let streams = try engine.loopForTesting.onEngineQueueSync {
            try requests.map { try engine.submit($0) }
        }
        var tokens: [[Int]] = []
        for stream in streams { tokens.append(await cbv2SchedCollect(stream).tokens) }
        return tokens
    }

    /// Two prompts decode above the assistant's speculative batch of one;
    /// the shorter finishes and the longer resumes rounds. Serial
    /// verification keeps MTP-on tokens equal to target-only decode. Every
    /// stored carry (prefill, seed, verify or chained finalize) must find the
    /// production assistant's history holding exactly the transitions before
    /// the carry token: no gap and no duplicate.
    @Test("batch-gated rows chain with exact tokens and contiguous history")
    func batchGatedRowsChainWithContiguousHistory() async throws {
        final class CarryAudit: @unchecked Sendable {
            var carries = 0
            var firstOffsets: [CBv2RequestID: Int] = [:]
            var mismatches: [String] = []
        }
        let requests = [
            CBv2Request(
                id: .init(931), promptTokens: (1 ... 17).map { $0 },
                sampling: .init(temperature: 0), maxTokens: 20),
            CBv2Request(
                id: .init(932), promptTokens: (3 ... 15).map { $0 },
                sampling: .init(temperature: 0), maxTokens: 5),
        ]
        let (reference, _) = try engine(mtpConfig: .init(enabled: false))
        let expected = try await runTogether(reference, requests)
        await reference.shutdown()

        for (ordinary, prefillCarry) in [(true, true), (false, false)] {
            let (candidate, backend) = try engine(
                mtpConfig: .init(
                    enabled: true, maxDraftTokens: 2, fixedDraftTokens: 2,
                    verificationMode: .serialTarget, ordinaryDecodeHistory: ordinary,
                    prefillCarry: prefillCarry))
            let audit = CarryAudit()
            candidate.loopForTesting.onEngineQueueSync {
                let driver = candidate.loopForTesting.mtp
                driver?.carryStoredObserverForTesting = { [weak driver] carry in
                    audit.carries += 1
                    if audit.firstOffsets[carry.id] == nil {
                        audit.firstOffsets[carry.id] = carry.kvOffset
                    }
                    let committed = driver?.assistantStateCountsForTesting(carry.id)?.committed
                    if committed != carry.kvOffset - 1 {
                        audit.mismatches.append(
                            "\(carry.id.raw)@\(carry.kvOffset): \(String(describing: committed))")
                    }
                }
            }
            let actual = try await runTogether(candidate, requests)
            let metrics = try #require(candidate.mtpMetricsSnapshot())
            let chained = candidate.chainedStepCount
            let released = candidate.loopForTesting.onEngineQueueSync {
                (
                    candidate.loopForTesting.mtp?.requestStateCountForTesting,
                    candidate.loopForTesting.recurrentStates.isEmpty,
                    backend.bytesReserved
                )
            }
            await candidate.shutdown()

            #expect(actual == expected)
            #expect(metrics.rounds > 0)
            #expect(audit.carries > metrics.rounds)
            #expect(audit.mismatches.isEmpty, "\(audit.mismatches)")
            #expect(released.0 == 0 && released.1 && released.2 == 0)
            if ordinary {
                // Prefill carries anchor at the prompt lengths; batch-gated
                // steps chain; rounds resume from a finalized carry. Only the
                // lone long row's output-budget tail may still seed.
                #expect(audit.firstOffsets == [.init(931): 17, .init(932): 13])
                #expect(chained > 0)
                #expect(metrics.controllerFallbacks["synchronized_seed"] == nil)
                #expect(metrics.seedSteps <= 1)
            } else {
                #expect(audit.firstOffsets == [.init(931): 18, .init(932): 14])
                #expect(chained == 0)
                #expect(metrics.seedSteps > 0)
            }
        }
    }
}
