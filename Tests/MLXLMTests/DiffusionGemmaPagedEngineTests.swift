import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@testable import MLXVLM

@Suite("Native diffusion page-admitted engine", .serialized)
struct DiffusionGemmaPagedEngineTests {
    private func context() throws -> DiffusionGemmaContext {
        let config = try DiffusionGemmaPagedStateTests().configuration()
        let root: [String: Any] = [
            "model_type": "diffusion_gemma",
            "text_config": try JSONSerialization.jsonObject(with: JSONEncoder().encode(config)),
            "canvas_length": 4, "tie_word_embeddings": true, "eos_token_id": [1],
        ]
        let model = try DiffusionGemma(
            JSONDecoder().decode(
                DiffusionGemmaConfiguration.self, from: JSONSerialization.data(withJSONObject: root)
            ))
        eval(model)
        return .init(
            configuration: .init(directory: Bundle.module.bundleURL), model: model,
            generationConfiguration: try .init(maxNewTokens: 12, eosTokenIds: [1]),
            tokenizer: TestTokenizer(vocabularySize: 128), chatTemplate: "fixture", processor: nil)
    }
    private func run(_ engine: CBv2NativeBlockEngine, _ request: CBv2Request) async throws -> (
        [Int], CBv2Usage
    ) {
        var tokens: [Int] = []
        var usage: CBv2Usage?
        for await event in try engine.submit(request) {
            switch event {
            case .delta(_, let ids, _): tokens += ids
            case .finished(let reason, let counts):
                #expect(reason == .stop || reason == .length)
                usage = counts
            }
        }
        return (tokens, try #require(usage))
    }
    @Test func sharedProcessBudgetPreservesIndependentRequestsAndResidentReuse() async throws {
        let context = try context()
        let owner = NativePageTestProcessOwner(maximum: 128 << 20)
        let cache = try DiffusionGemmaResidentPrefixConfiguration(
            maximumBytes: 8 << 20,
            artifactIdentity: "generated", templateIdentity: "fixture",
            numericalProfile: "native-sdpa")
        let engine = try context.makeNativeEngine(
            kvBytesCapacity: 128 << 20, prefillChunkSize: 32,
            prefixCache: cache,
            pagedConfiguration: .init(
                capacityBytes: 128 << 20, dtype: .float32,
                maxPrefillChunk: 32, nominalMaxSequenceLength: 128, segmentSizeBytes: 1 << 20),
            processMemoryOwner: owner)
        let cold = try context.makeNativeEngine(kvBytesCapacity: 128 << 20, prefillChunkSize: 32)
        let a = CBv2Request(
            id: .init(1), promptTokens: (0 ..< 67).map { $0 % 100 + 2 }, sampling: .init(seed: 341),
            maxTokens: 12, cacheSalt: "a")
        let b = CBv2Request(
            id: .init(2), promptTokens: (0 ..< 81).map { ($0 * 7) % 100 + 2 },
            sampling: .init(seed: 7419), maxTokens: 12, cacheSalt: "b")
        let referenceA = try await run(cold, a)
        let referenceB = try await run(cold, b)
        async let resultA = run(engine, a)
        async let resultB = run(engine, b)
        let (actualA, actualB) = try await (resultA, resultB)
        #expect(actualA.0 == referenceA.0 && actualB.0 == referenceB.0)
        #expect(
            actualA.1.completionTokens == referenceA.1.completionTokens
                && actualB.1.completionTokens == referenceB.1.completionTokens)
        #expect(owner.bytes == 8 << 20 && owner.coverage == 0)
        var repeated = a
        repeated.id = .init(3)
        let hit = try await run(engine, repeated)
        #expect(
            hit.0 == referenceA.0 && hit.1.prefixCachePrefillTokensSaved == a.promptTokens.count)
        #expect(engine.capacity().kvBytesReserved == Int(owner.bytes))
        engine.updateKVBytesCapacity(1 << 19)
        let deadline = ContinuousClock.now.advanced(by: .seconds(2))
        while owner.bytes != 0 && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(1))
        }
        #expect(owner.bytes == 0 && engine.capacity().kvBytesReserved == 0)
        #expect(throws: CBv2KVError.self) { try engine.submit(a) }
        engine.updateKVBytesCapacity(128 << 20)
        let readmitted = try await run(engine, a)
        #expect(readmitted.0 == referenceA.0)
        await engine.shutdown()
        await cold.shutdown()
        #expect(owner.bytes == 0 && owner.coverage == 0 && engine.capacity().kvBytesReserved == 0)
    }
}
