import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

@Suite("Native diffusion media resident prefix", .serialized)
struct DiffusionGemmaMediaPrefixTests {
    private func run(_ engine: CBv2NativeBlockEngine, _ request: CBv2Request) async throws -> ([Int], CBv2Usage) {
        var tokens = [Int](), result: CBv2Usage?
        for await event in try engine.submit(request) {
            switch event {
            case .delta(_, let ids, _): tokens += ids
            case .finished(let reason, let usage):
                #expect(reason == .length || reason == .stop)
                #expect(result == nil)
                result = usage
            }
        }
        return (tokens, try #require(result))
    }

    @Test func coldBoundariesPreserveWholeBlocksAndTextBehavior() throws {
        let text = try DiffusionGemmaPrefillGeometry(promptCount: 11, chunkSize: 4)
        #expect(text.boundaries == [4, 8, 11] && text.capturePositions == [4, 8, 11])
        let spans = [CBv2ImageSpan(tokenOffset: 2, length: 6)]
        let original = try DiffusionGemmaPrefillGeometry(promptCount: 11, chunkSize: 4, spans: spans)
        #expect(original.boundaries == [2, 8, 11])
        #expect(original.capturePositions == [2, 8, 11])
        #expect(try original.chunkLength(start: 11) == 0)
        #expect(throws: (any Error).self) { try original.chunkLength(start: 4) }
        let appended = try DiffusionGemmaPrefillGeometry(promptCount: 14, chunkSize: 4, spans: spans)
        #expect(appended.boundaries == [2, 8, 12, 14])
        #expect(appended.permitsRestore(position: 8) && !appended.permitsRestore(position: 11))
        let shifted = try DiffusionGemmaPrefillGeometry(promptCount: 20, chunkSize: 4,
            spans: [.init(tokenOffset: 9, length: 6)])
        #expect(shifted.boundaries == [4, 8, 9, 15, 19, 20])
        #expect(shifted.capturePositions == [4, 8, 19, 20])
        #expect(shifted.lastAlignedStableBoundary == 8)
        #expect(shifted.permitsPersistentCapture(position: 8))
        #expect(shifted.permitsPersistentCapture(position: 19))
        #expect(!shifted.permitsPersistentCapture(position: 20))
        for invalid in [
            [CBv2ImageSpan(tokenOffset: 10, length: 2)],
            [CBv2ImageSpan(tokenOffset: 2, length: 4), .init(tokenOffset: 5, length: 2)],
        ] {
            #expect(throws: (any Error).self) {
                try DiffusionGemmaPrefillGeometry(promptCount: 11, chunkSize: 4, spans: invalid)
            }
        }
    }

    @Test(arguments: [2, 6, 8], [false, true])
    func boundMediaExactAndAppendMatchColdWithoutCrossImageOrTenantReuse(_ mediaLength: Int, _ pageBacked: Bool) async throws {
        let (fixture, _, oracleModel) = try DiffusionGemmaVisionOracleTests().fixture()
        let model: DiffusionGemma
        if pageBacked {
            // Keep the independent tiny oracle unchanged. Paging needs its
            // supported production head dimensions, not a weakened backend gate.
            var root = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(fixture.model_config)) as? [String: Any])
            root["text_config"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(
                DiffusionGemmaPagedStateTests().configuration()))
            model = try DiffusionGemma(JSONDecoder().decode(DiffusionGemmaConfiguration.self,
                from: JSONSerialization.data(withJSONObject: root)))
            model.train(false)
            eval(model)
        } else { model = oracleModel }
        let context = DiffusionGemmaContext(
            configuration: .init(directory: FileManager.default.temporaryDirectory),
            model: model,
            generationConfiguration: try .init(maxNewTokens: 4, maxDenoisingSteps: 4, eosTokenIds: []),
            tokenizer: TestTokenizer(vocabularySize: 128), chatTemplate: "unused", processor: nil)
        let cache = try DiffusionGemmaResidentPrefixConfiguration(maximumBytes: 1 << 20,
            artifactIdentity: "generated-media-fixture", templateIdentity: "fixture",
            numericalProfile: "frozen-native-fp32")
        let owner = NativePageTestProcessOwner(maximum: 64 << 20)
        let engine = try context.makeNativeEngine(kvBytesCapacity: 32 << 20,
            prefillChunkSize: 4, prefixCache: cache,
            pagedConfiguration: pageBacked ? .init(capacityBytes: 32 << 20, dtype: .float32,
                maxPrefillChunk: 32, nominalMaxSequenceLength: 256, segmentSizeBytes: 1 << 20) : nil,
            processMemoryOwner: pageBacked ? owner : nil)
        let control = try context.makeNativeEngine(kvBytesCapacity: 32 << 20, prefillChunkSize: 4)
        let span = CBv2ImageSpan(tokenOffset: 2, length: mediaLength)
        let prompt = [2, 7] + Array(repeating: model.configuration.imageTokenId, count: mediaLength) + [17, 19, 23]
        let values = (MLXArray(0..<(mediaLength * 32)).asType(.float32) / 97).reshaped(1, mediaLength, 32)
        let changed = -values
        eval(values, changed)
        func request(_ id: UInt64, appended: Bool = false, tenant: String = "a",
                     changedMedia: Bool = false, bound: Bool = true) throws -> CBv2Request {
            let embedding = changedMedia ? changed : values
            var bytes = embedding.asData(access: .copy).data
            bytes.append(contentsOf: "offset=2,length=\(mediaLength)".utf8)
            let identity = try CBv2HybridPrefixIdentity(digest: Data(SHA256.hash(data: bytes)))
            return .init(id: .init(id), promptTokens: prompt + (appended ? [29, 31, 37] : []),
                sampling: .init(seed: 341), maxTokens: 4, cacheSalt: tenant,
                multimodal: .init(spans: [span], embeddings: { [embedding] }),
                hybridPrefixIdentity: bound ? identity : nil)
        }
        let base = try request(1)
        let reference = try await run(control, base)
        let cold = try await run(engine, base)
        #expect(cold.0 == reference.0 && cold.1.prefixCachePrefillTokensSaved == 0)
        let warm = try await run(engine, request(2))
        #expect(warm.0 == reference.0 && warm.1.completionTokens == reference.1.completionTokens)
        #expect(warm.1.prefixCachePrefillTokensSaved == prompt.count)
        let extendedReference = try await run(control, request(3, appended: true))
        let extended = try await run(engine, request(4, appended: true))
        let boundaries = try DiffusionGemmaPrefillGeometry(promptCount: prompt.count, chunkSize: 4, spans: [span])
        #expect(extended.0 == extendedReference.0)
        #expect(extended.1.prefixCachePrefillTokensSaved == boundaries.lastStableBoundary)
        let other = try await run(engine, request(5, tenant: "b"))
        #expect(other.0 == reference.0 && other.1.prefixCachePrefillTokensSaved == 0)
        let changedReference = try await run(control, request(6, changedMedia: true))
        let changedResult = try await run(engine, request(7, changedMedia: true))
        #expect(changedResult.0 == changedReference.0 && changedResult.1.prefixCachePrefillTokensSaved == 0)
        let unbound = try await run(engine, request(8, bound: false))
        #expect(unbound.0 == reference.0 && unbound.1.prefixCacheOutcome == .skippedPolicy)
        await engine.shutdown()
        await control.shutdown()
        #expect(engine.capacity().kvBytesInUse == 0 && engine.capacity().kvBytesReserved == 0)
        #expect(owner.bytes == 0 && owner.coverage == 0)
    }

    @Test func mediaCheckpointPreservesRawStateAfterDonorMutationAndCancellationDoesNotPublish() throws {
        let (_, _, model) = try DiffusionGemmaVisionOracleTests().fixture()
        let tokens: [Int32] = [2, 7] + Array(repeating: 100, count: 6) + [17, 19, 23]
        let input = MLXArray(tokens).reshaped(1, tokens.count)
        let span = CBv2ImageSpan(tokenOffset: 2, length: 6)
        let values = (MLXArray(0..<192).asType(.float32) / 97).reshaped(1, 6, 32)
        let media = CBv2MultimodalInput(spans: [span], embeddings: { [values] })
        let digest = try CBv2HybridPrefixIdentity(digest: Data(repeating: 7, count: 32))
        let config = try DiffusionGemmaResidentPrefixConfiguration(maximumBytes: 1 << 20,
            artifactIdentity: "generated", templateIdentity: "fixture", numericalProfile: "native")
        let store = DiffusionGemmaResidentPrefixCache(configuration: config, chunkSize: 4)
        let request = CBv2Request(id: .init(1), promptTokens: tokens.map(Int.init), maxTokens: 4,
            cacheSalt: "tenant", multimodal: media, hybridPrefixIdentity: digest)
        let identity = try #require(try store.identity(for: request))
        let geometry = try DiffusionGemmaPrefillGeometry(promptCount: tokens.count, chunkSize: 4, spans: [span])
        let prepared = try DiffusionGemmaVisualEmbeddings(input: media, promptCount: tokens.count, hiddenSize: 32)
        let state = try model.makeCache(expectedPromptLength: tokens.count)
        let ticket = UUID()
        var start = 0
        for end in geometry.boundaries {
            try prepared.encode(model: model, tokens: input[0..., start..<end], cache: state, start: start)
            eval(state.stateArrays())
            try store.capture(model: model, cache: state, promptTokens: tokens,
                request: ticket, identity: identity, geometry: geometry)
            start = end
        }
        #expect(store.retainedBytes > 0)
        #expect(store.lookup(tokens: tokens, identity: identity, geometry: geometry) == nil)
        store.finish(request: ticket, successful: false)
        #expect(store.retainedBytes == 0)
        #expect(store.lookup(tokens: tokens, identity: identity, geometry: geometry) == nil)
        let original = state.stateArrays().map { $0.asArray(Float.self).map(\.bitPattern) }
        let canvas = MLXArray([Int32(3), 5, 7, 9]).reshaped(1, 4)
        let logits = try model.denoise(canvasIds: canvas, cache: state).asArray(Float.self).map(\.bitPattern)
        let checkpoint = try model.model.decoder.checkpoint(cache: state, identity: identity, compact: true)
        _ = try model.encode(tokenIds: MLXArray([Int32(29), 31, 37]).reshaped(1, 3), cache: state)
        eval(state.stateArrays())
        let restored = try model.model.decoder.restorePrefix(checkpoint, identity: identity, promptTokenIds: input)
        #expect(restored.stateArrays().map { $0.asArray(Float.self).map(\.bitPattern) } == original)
        #expect(try model.denoise(canvasIds: canvas, cache: restored).asArray(Float.self).map(\.bitPattern) == logits)
        store.clear()
    }
}
