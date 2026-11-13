import MLX
import MLXLMCommon
import Testing

/// A transition table is the model oracle; the engine, sampler and KV owner are real.
private final class BudgetTransitionModel: CBv2SteppableModel {
    let transitions = MLXArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0])
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        let kv = broadcast(
            tokens.asType(.float32).reshaped([tokens.dim(0), 1, tokens.dim(1), 1]),
            to: [tokens.dim(0), 1, tokens.dim(1), 64])
        for cache in caches {
            _ = cache.updateAndAttend(queries: kv, keys: kv, values: kv, scale: 1, sinks: nil)
        }
        return take(eye(16), take(transitions, tokens, axis: 0), axis: 0)
    }
}

@Suite("Native reasoning budget", .tags(.integration), .serialized)
struct CBv2ReasoningBudgetTests {
    private func engine(enabled: Bool = true) -> EngineV2 {
        let kinds = [CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 1)]
        return EngineV2(
            model: BudgetTransitionModel(), layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 20)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            reasoningBudgetsEnabled: enabled)
    }

    @Test("N thought tokens are followed by the close and an answer in the same request")
    func closesAtBudget() async throws {
        let engine = engine()
        var request = CBv2Request(
            id: .init(808), promptTokens: [0], sampling: .init(temperature: 0),
            maxTokens: 6)
        request.reasoningBudget = try .init(
            maximumTokens: 2, openTokenID: 14, closeTokenID: 8,
            startsInsideThought: true)
        let result = await cbv2SchedCollect(try engine.submit(request))
        await engine.shutdown()
        #expect(result.tokens == [1, 2, 8, 9, 10, 11])
        #expect(result.finishReason == .length)
        #expect(result.usage?.completionTokens == 6)
    }
}
