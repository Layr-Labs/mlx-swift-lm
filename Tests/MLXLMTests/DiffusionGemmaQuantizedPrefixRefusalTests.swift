import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

@Suite("Diffusion packed KV refuses native prefix codecs before mutation", .serialized)
struct DiffusionGemmaQuantizedPrefixRefusalTests {
    private let capacity = 64 << 20

    private func context() throws -> DiffusionGemmaContext {
        let original = try DiffusionGemmaPagedStateTests().configuration()
        var text = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(original)) as? [String: Any])
        text["head_dim"] = 64
        text["global_head_dim"] = 64
        text["num_attention_heads"] = 4
        text["num_key_value_heads"] = 2
        text["num_global_key_value_heads"] = 2
        text["sliding_window"] = 256
        let fields: [String: Any] = [
            "model_type": "diffusion_gemma", "text_config": text,
            "canvas_length": 4, "tie_word_embeddings": true, "eos_token_id": [1],
        ]
        let config = try JSONDecoder().decode(
            DiffusionGemmaConfiguration.self, from: JSONSerialization.data(withJSONObject: fields))
        let model = DiffusionGemma(config)
        model.train(false)
        eval(model)
        return .init(
            configuration: .init(directory: Bundle.module.bundleURL), model: model,
            generationConfiguration: try .init(
                maxNewTokens: 4, maxDenoisingSteps: 4, eosTokenIds: [1]),
            tokenizer: TestTokenizer(vocabularySize: 128), chatTemplate: "fixture", processor: nil)
    }

    private func pageConfig(
        _ context: DiffusionGemmaContext, allNative: Bool = false
    ) -> PagedKVPoolConfig {
        let kinds = context.model.configuration.textConfig.diffusionPagedLayerKinds
        let nativeOwners =
            allNative
            ? Set(kinds.indices.filter { kinds[$0].sharesKVWithLayer == nil }) : Set<Int>()
        return .init(
            capacityBytes: capacity, dtype: .float32, maxPrefillChunk: 32,
            nominalMaxSequenceLength: 320, segmentSizeBytes: 1 << 20,
            quantization: .init(), nativeLayerIndices: nativeOwners)
    }

    private func prefixConfig(
        _ store: NativeCheckpointMemoryTransport
    ) throws -> DiffusionGemmaResidentPrefixConfiguration {
        try .init(
            maximumBytes: 8 << 20, artifactIdentity: store.identity.modelAggregateHash,
            templateIdentity: store.identity.promptContractID,
            numericalProfile: store.identity.numericsFingerprint)
    }

    private func run(_ engine: CBv2NativeBlockEngine, id: UInt64) async throws -> CBv2Usage {
        let request = CBv2Request(
            id: .init(id), promptTokens: (0 ..< 161).map { $0 % 100 + 2 },
            sampling: .init(seed: 341), maxTokens: 4, cacheSalt: "tenant")
        var terminal: CBv2Usage?
        for await event in try engine.submit(request) {
            if case .finished(let reason, let usage) = event {
                #expect(reason == .stop || reason == .length)
                #expect(terminal == nil)
                terminal = usage
            }
        }
        return try #require(terminal)
    }

    @Test(arguments: [0, 1, 2])
    func residentOrPersistentNativeCodecRefusesBeforeProcessCharge(selection: Int) async throws {
        let context = try context()
        let owner = NativePageTestProcessOwner(maximum: UInt64(capacity))
        let store = NativeCheckpointMemoryTransport()
        let resident = try prefixConfig(store)
        do {
            let unexpected = try context.makeNativeEngine(
                kvBytesCapacity: capacity, prefillChunkSize: 32,
                prefixCache: selection == 1 ? nil : resident,
                completePrefixCache: selection == 0 ? nil : store,
                pagedConfiguration: pageConfig(context), processMemoryOwner: owner)
            await unexpected.shutdown()
            Issue.record("Packed history must refuse the native prefix codec at construction")
        } catch let error as CBv2KVError {
            guard case .backendIneligible(let reason) = error else {
                Issue.record("Expected an explicit packed-prefix eligibility refusal: \(error)")
                return
            }
            #expect(reason == "packed DiffusionGemma legacy prefix caches are unsupported")
        }
        #expect(owner.bytes == 0 && owner.coverage == 0)
        #expect(!store.isClosed && store.counts.writes == 0 && store.counts.pending == 0)
    }

    @Test func packedNativeEngineStillGeneratesWithPrefixCachingOmitted() async throws {
        let context = try context()
        let owner = NativePageTestProcessOwner(maximum: UInt64(capacity))
        let engine = try context.makeNativeEngine(
            kvBytesCapacity: capacity, prefillChunkSize: 32,
            pagedConfiguration: pageConfig(context), processMemoryOwner: owner)
        let usage = try await run(engine, id: 101)
        await engine.shutdown()
        #expect(usage.completionTokens > 0 && usage.prefixCachePrefillTokensSaved == 0)
        #expect(owner.bytes == 0 && owner.coverage == 0)
    }

    @Test func allNativeOwnerExemptionsPreserveResidentPrefixReuse() async throws {
        let context = try context()
        let owner = NativePageTestProcessOwner(maximum: UInt64(capacity))
        let store = NativeCheckpointMemoryTransport()
        let engine = try context.makeNativeEngine(
            kvBytesCapacity: capacity, prefillChunkSize: 32,
            prefixCache: prefixConfig(store),
            pagedConfiguration: pageConfig(context, allNative: true),
            processMemoryOwner: owner)
        let cold = try await run(engine, id: 201)
        let warm = try await run(engine, id: 202)
        await engine.shutdown()
        #expect(cold.completionTokens > 0 && cold.prefixCachePrefillTokensSaved == 0)
        #expect(warm.prefixCachePrefillTokensSaved == 161)
        #expect(owner.bytes == 0 && owner.coverage == 0)
    }

    @Test func longNativeCheckpointRefusesPackedRestoreBeforeDestinationAllocation() throws {
        let context = try context()
        let model = context.model
        let tokens = MLXArray((0 ..< 257).map { Int32($0 % 100 + 2) }).reshaped(1, 257)
        let donor = try model.makeCache(expectedPromptLength: 257, maximumSequenceLength: 320)
        for start in stride(from: 0, to: 257, by: 32) {
            _ = try model.encode(
                tokenIds: tokens[0..., start ..< min(257, start + 32)], cache: donor)
        }
        let identity = try DiffusionGemmaPrefixIdentity(
            tenantScope: "fixture", artifact: "generated", template: "fixture", media: "text-only",
            numericalProfile: "native-float32", epoch: "one")
        let checkpoint = try model.model.decoder.checkpoint(
            cache: donor, identity: identity, compact: true)
        let packed = try PagedKVBackend(
            layerKinds: model.configuration.textConfig.diffusionPagedLayerKinds,
            config: pageConfig(context))
        #expect(packed.usesQuantizedStorage)
        #expect(
            throws: DiffusionGemmaModelError.invalidInput(
                "native DiffusionGemma prefix checkpoints do not support packed KV")
        ) {
            _ = try model.model.decoder.restorePrefix(
                checkpoint, identity: identity, promptTokenIds: tokens,
                maximumSequenceLength: 320, pagedBackend: packed)
        }
        #expect(packed.bytesReserved == 0 && packed.bytesInUse == 0 && packed.bytesWired == 0)
        #expect(packed.pool.nativeRecentBytesInUse == 0 && !packed.pool.writeValidation.isFaulted)

        let native = try PagedKVBackend(
            layerKinds: model.configuration.textConfig.diffusionPagedLayerKinds,
            config: pageConfig(context, allNative: true))
        #expect(!native.usesQuantizedStorage)
        do {
            let restored = try model.model.decoder.restorePrefix(
                checkpoint, identity: identity, promptTokenIds: tokens,
                maximumSequenceLength: 320, pagedBackend: native)
            #expect(restored.position == 257 && restored.windowOrder == donor.windowOrder)
            for (source, destination) in zip(donor.snapshots(), restored.snapshots()) {
                eval(source.keys, source.values, destination.keys, destination.values)
                #expect(source.keys.asArray(Float.self) == destination.keys.asArray(Float.self))
                #expect(source.values.asArray(Float.self) == destination.values.asArray(Float.self))
            }
        }
        #expect(native.bytesReserved == 0 && native.bytesInUse == 0 && native.bytesWired == 0)
    }
}
