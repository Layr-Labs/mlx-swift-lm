import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

@Suite("Native diffusion media persistent transport", .serialized)
struct DiffusionGemmaPersistentMediaTests {
    private func run(_ engine: CBv2NativeBlockEngine, _ request: CBv2Request) async throws -> ([Int], CBv2Usage) {
        var ids = [Int](), usage: CBv2Usage?
        for await event in try engine.submit(request) {
            switch event {
            case .delta(_, let tokens, _): ids += tokens
            case .finished(let reason, let value):
                #expect(reason == .length || reason == .stop)
                #expect(usage == nil)
                usage = value
            }
        }
        return (ids, try #require(usage))
    }

    @Test(arguments: [2, 6, 8])
    func mediaWireRoundTripUsesExactNativeBoundaries(_ length: Int) async throws {
        let (_, _, model) = try DiffusionGemmaVisionOracleTests().fixture()
        let context = DiffusionGemmaContext(configuration: .init(directory: FileManager.default.temporaryDirectory),
            model: model, generationConfiguration: try .init(maxNewTokens: 4, maxDenoisingSteps: 4, eosTokenIds: []),
            tokenizer: TestTokenizer(vocabularySize: 128), chatTemplate: "unused", processor: nil)
        let store = NativeCheckpointMemoryTransport(nativeBoundaries: true)
        let prefix = try DiffusionGemmaResidentPrefixConfiguration(maximumBytes: 1 << 20,
            artifactIdentity: store.identity.modelAggregateHash, templateIdentity: store.identity.promptContractID,
            numericalProfile: store.identity.numericsFingerprint)
        let engine = try context.makeNativeEngine(kvBytesCapacity: 32 << 20, prefillChunkSize: 4,
            prefixCache: prefix, completePrefixCache: store, retainMemoryPrefixes: false)
        let control = try context.makeNativeEngine(kvBytesCapacity: 32 << 20, prefillChunkSize: 4)
        let tokens = [2, 7] + Array(repeating: model.configuration.imageTokenId, count: length) + [17, 19, 23]
        let span = CBv2ImageSpan(tokenOffset: 2, length: length)
        let values = (MLXArray(0..<(length * 32)).asType(.float32) / 97).reshaped(1, length, 32)
        let changed = -values
        eval(values, changed)
        func request(_ id: UInt64, extra: [Int] = [], tenant: String = "tenant", other: Bool = false) throws -> CBv2Request {
            let features = other ? changed : values
            var bytes = features.asData(access: .copy).data
            bytes.append(contentsOf: "image:2:\(length)".utf8)
            var request = CBv2Request(id: .init(id), promptTokens: tokens + extra,
                sampling: .init(seed: 341), maxTokens: 4, cacheSalt: tenant,
                multimodal: .init(spans: [span], embeddings: { [features] }),
                hybridPrefixIdentity: try .init(digest: Data(SHA256.hash(data: bytes))))
            request.prefixCacheReceiptID = .init(id + 100)
            return request
        }
        do {
            let cold = try await run(engine, request(1))
            let reference = try await run(control, request(2))
            #expect(cold.0 == reference.0 && cold.1.prefixCachePrefillTokensSaved == 0)
            let geometry = try DiffusionGemmaPrefillGeometry(promptCount: tokens.count, chunkSize: 4, spans: [span])
            let boundary = try #require(geometry.lastStableBoundary)
            try #require(store.counts.writes == 1)
            let manifest = try store.manifest()
            #expect(manifest.position == boundary && manifest.mediaIdentity != nil && !manifest.mediaTargetOnly)
            #expect(manifest.backendLayout == "native-block-diffusiongemma-v2")
            if length == 8 {
                #expect(boundary == 10 && !boundary.isMultiple(of: 4))
                var unbound = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(manifest)) as? [String: Any])
                unbound.removeValue(forKey: "mediaIdentity")
                #expect(throws: (any Error).self) {
                    let decoded = try JSONDecoder().decode(CBv2CompleteCheckpointManifest.self,
                        from: JSONSerialization.data(withJSONObject: unbound))
                    _ = try decoded.validateStructure()
                }
            }
            var obsolete = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(manifest)) as? [String: Any])
            obsolete["backendLayout"] = "native-block-diffusiongemma-v1"
            #expect(throws: (any Error).self) {
                let decoded = try JSONDecoder().decode(CBv2CompleteCheckpointManifest.self,
                    from: JSONSerialization.data(withJSONObject: obsolete))
                _ = try decoded.validateStructure()
            }
            let repeated = try request(3)
            try store.stage(engine: engine, request: repeated)
            let warm = try await run(engine, repeated)
            #expect(warm.0 == reference.0 && warm.1.prefixCachePrefillTokensSaved == boundary)
            #expect(warm.1.prefixCacheTier == .snapshot)
            let extended = try request(4, extra: [29, 31, 37])
            let extendedReference = try await run(control, extended)
            try store.stage(engine: engine, request: extended)
            let appended = try await run(engine, extended)
            #expect(appended.0 == extendedReference.0 && appended.1.prefixCachePrefillTokensSaved == boundary)
            // Wrong media or tenant must be rejected by planning before allocation.
            let reserved = engine.capacity().kvBytesReserved
            for negative in [try request(5, extra: [29, 31, 37], tenant: "other"),
                             try request(6, extra: [29, 31, 37], other: true)] {
                #expect(throws: (any Error).self) { try store.stage(engine: engine, request: negative) }
                #expect(engine.capacity().kvBytesReserved == reserved)
            }
            // A nominally aligned position is not necessarily a native cold
            // boundary after bidirectional geometry changes.
            var split = try request(7)
            split.multimodal = .init(spans: [.init(tokenOffset: length == 2 ? 3 : 1, length: length)], embeddings: { [values] })
            #expect(throws: (any Error).self) { try engine.planNativeCheckpointImport(manifest: manifest, request: split) }
            #expect(engine.capacity().kvBytesReserved == reserved)
            let unstaged = try await run(engine, request(8))
            #expect(unstaged.0 == reference.0 && unstaged.1.prefixCachePrefillTokensSaved == 0)
            #expect(store.counts.failures == 0)
        } catch { await engine.shutdown(); await control.shutdown(); throw error }
        await engine.shutdown(); await control.shutdown()
        #expect(engine.capacity().kvBytesReserved == 0 && store.isClosed)
    }
}
