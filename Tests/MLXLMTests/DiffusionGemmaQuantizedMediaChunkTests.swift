import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

@Suite("Packed diffusion whole-media chunks", .serialized)
struct DiffusionGemmaQuantizedMediaChunkTests {
    @Test func wholeBlockBoundExcludesMarkersAndRejectsLargerCoalescedSpans() throws {
        let maximum = DiffusionGemmaPrefillGeometry.maximumVisualBlockTokens
        let spans = [
            CBv2ImageSpan(tokenOffset: 2, length: 560), .init(tokenOffset: 562, length: 560),
        ]
        let geometry = try DiffusionGemmaPrefillGeometry(
            promptCount: maximum + 6, chunkSize: 512, spans: spans)
        #expect(geometry.boundaries == [2, maximum + 2, maximum + 6])
        #expect(try geometry.chunkLength(start: 2) == maximum)
        #expect(try geometry.chunkLength(start: maximum + 2) == 4)
        try DiffusionGemmaVisualEmbeddings.validate(spans: spans, promptCount: maximum + 6)

        let oversized = [
            CBv2ImageSpan(tokenOffset: 2, length: 560), .init(tokenOffset: 562, length: 561),
        ]
        #expect(throws: (any Error).self) {
            try DiffusionGemmaPrefillGeometry(
                promptCount: maximum + 7, chunkSize: 512, spans: oversized)
        }
        #expect(throws: (any Error).self) {
            try DiffusionGemmaVisualEmbeddings.validate(spans: oversized, promptCount: maximum + 7)
        }
        let square = try DiffusionGemmaMediaGeometry.resized(
            width: 512, height: 512, patchSize: 16, poolingSize: 3, maxSoftTokens: 560)
        #expect(square.softTokens == 529 && square.softTokens > 512)
    }

    private func model() throws -> DiffusionGemma {
        let metadata = try #require(
            Bundle.module.url(forResource: "diffusiongemma-vision-oracle", withExtension: "json"))
        let fixture = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: metadata)) as? [String: Any])
        var root = try #require(fixture["model_config"] as? [String: Any])
        var text = try #require(root["text_config"] as? [String: Any])
        // Keep the actual native media model/decoder route, with generated
        // small weights and supported packed attention dimensions. Prepared
        // features exercise the binding contract; this is not a tower oracle.
        text["head_dim"] = 64
        text["global_head_dim"] = 64
        text["num_attention_heads"] = 4
        text["num_key_value_heads"] = 2
        text["num_global_key_value_heads"] = 2
        text["sliding_window"] = 256
        text["max_position_embeddings"] = 2048
        root["text_config"] = text
        let configuration = try JSONDecoder().decode(
            DiffusionGemmaConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
        let model = DiffusionGemma(configuration)
        model.train(false)
        eval(model)
        return model
    }

    private func run(
        _ engine: CBv2NativeBlockEngine, request: CBv2Request
    ) async throws -> CBv2Usage {
        var terminal: CBv2Usage?
        for await event in try engine.submit(request) {
            switch event {
            case .delta(_, let tokens, _):
                #expect(tokens.allSatisfy { $0 >= 0 && $0 < 128 })
            case .finished(let reason, let usage):
                #expect(
                    reason == .length || reason == .stop,
                    "An accepted whole visual block must complete: \(reason)")
                #expect(terminal == nil)
                terminal = usage
            }
        }
        return try #require(terminal)
    }

    @Test(arguments: [529, 1120])
    func acceptedMediaLargerThanTextChunkCompletesPackedNativeEngine(mediaTokens: Int) async throws
    {
        let model = try model()
        let configuration = model.configuration
        let context = DiffusionGemmaContext(
            configuration: .init(directory: FileManager.default.temporaryDirectory), model: model,
            generationConfiguration: try .init(
                maxNewTokens: 4, maxDenoisingSteps: 4, eosTokenIds: []),
            tokenizer: TestTokenizer(vocabularySize: 128), chatTemplate: "unused", processor: nil)
        let capacity = 128 << 20
        let owner = NativePageTestProcessOwner(maximum: UInt64(capacity))
        let pageConfig = PagedKVPoolConfig(
            capacityBytes: capacity, dtype: .float32,
            maxPrefillChunk: max(
                512, configuration.canvasLength,
                DiffusionGemmaPrefillGeometry.maximumVisualBlockTokens),
            nominalMaxSequenceLength: configuration.textConfig.maxPositionEmbeddings,
            segmentSizeBytes: 1 << 20, quantization: .init())
        let packed = try context.makeNativeEngine(
            kvBytesCapacity: capacity, maxConcurrentRequests: 1, prefillChunkSize: 512,
            retainMemoryPrefixes: false, pagedConfiguration: pageConfig, processMemoryOwner: owner)
        let native = try context.makeNativeEngine(
            kvBytesCapacity: capacity, maxConcurrentRequests: 1, prefillChunkSize: 512,
            retainMemoryPrefixes: false)
        let prompt =
            [2, configuration.beginImageTokenId]
            + Array(repeating: configuration.imageTokenId, count: mediaTokens)
            + [configuration.endImageTokenId, 17, 19]
        let features = MLXArray(
            (0 ..< mediaTokens * configuration.textConfig.hiddenSize).map {
                Float(($0 * 17) % 71) / 71 - 0.5
            }
        ).reshaped(1, mediaTokens, configuration.textConfig.hiddenSize)
        eval(features)
        let request = CBv2Request(
            id: .init(1), promptTokens: prompt, sampling: .init(temperature: 0, seed: 341),
            maxTokens: 4,
            multimodal: .init(
                spans: [.init(tokenOffset: 2, length: mediaTokens)], embeddings: { [features] }))
        do {
            let control = try await run(native, request: request)
            let actual = try await run(packed, request: request)
            #expect(control.promptTokens == prompt.count && actual.promptTokens == prompt.count)
            #expect(control.completionTokens > 0 && actual.completionTokens > 0)
            #expect(actual.prefixCacheOutcome == .disabled)
        } catch {
            await packed.shutdown()
            await native.shutdown()
            throw error
        }
        await packed.shutdown()
        await native.shutdown()
        #expect(packed.capacity().kvBytesInUse == 0 && packed.capacity().kvBytesReserved == 0)
        #expect(owner.bytes == 0 && owner.coverage == 0)
    }
}
