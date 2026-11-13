import Foundation
import MLX
import MLXLMCommon
import Testing

/// A transition table is the model oracle; the engine, sampler and KV owner are real.
private final class BudgetTransitionModel: CBv2SteppableModel {
    let transitions: MLXArray
    init(next: [Int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]) {
        transitions = MLXArray(next)
    }
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

private struct BudgetTextTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [0] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(String.init).joined()
    }
    func convertTokenToId(_ token: String) -> Int? { Int(token) }
    func convertIdToToken(_ id: Int) -> String? { String(id) }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [0] }
}

@Suite("Native reasoning budget", .tags(.integration), .serialized)
struct CBv2ReasoningBudgetTests {
    private func engine(
        enabled: Bool = true, sampler: any CBv2StepSampler = CBv2DefaultSampler(),
        detokenizer: any CBv2DetokenizerFactory = CBv2NullDetokenizerFactory(),
        next: [Int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
    ) -> EngineV2 {
        let kinds = [CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 1)]
        return EngineV2(
            model: BudgetTransitionModel(next: next), layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 20)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds), sampler: sampler,
            detokenizerFactory: detokenizer,
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

    private func request(
        id: UInt64 = 808, budget: Int = 2, prompt: Int = 0,
        inside: Bool = true, maximumOutput: Int = 6
    ) throws -> CBv2Request {
        var request = CBv2Request(
            id: .init(id), promptTokens: [prompt],
            sampling: .init(temperature: 0), maxTokens: maximumOutput)
        request.reasoningBudget = try .init(
            maximumTokens: budget,
            openTokenID: 14, closeTokenID: 8, startsInsideThought: inside)
        return request
    }

    @Test("Adjacent budgets preserve the exact N boundary", arguments: [1, 2, 3])
    func adjacentBudgets(_ n: Int) async throws {
        let engine = engine()
        let result = await cbv2SchedCollect(try engine.submit(request(budget: n)))
        await engine.shutdown()
        let expected = [
            1: [1, 8, 9, 10, 11, 12], 2: [1, 2, 8, 9, 10, 11],
            3: [1, 2, 3, 8, 9, 10],
        ]
        #expect(result.tokens == expected[n])
        #expect(result.usage?.completionTokens == 6)
    }

    @Test("A natural close ends the thought before its budget")
    func naturalClose() async throws {
        let engine = engine()
        let result = await cbv2SchedCollect(try engine.submit(request(budget: 10, prompt: 6)))
        await engine.shutdown()
        #expect(result.tokens == [7, 8, 9, 10, 11, 12])
    }

    @Test("Text before the first open and the opener do not spend thought tokens")
    func opensAfterContent() async throws {
        let engine = engine()
        let result = await cbv2SchedCollect(try engine.submit(request(prompt: 12, inside: false)))
        await engine.shutdown()
        #expect(result.tokens == [13, 14, 15, 0, 8, 9])
    }

    @Test("A request without a thought preserves its normal output")
    func noThought() async throws {
        let engine = engine()
        let result = await cbv2SchedCollect(try engine.submit(request(inside: false)))
        await engine.shutdown()
        #expect(result.tokens == [1, 2, 3, 4, 5, 6])
    }

    @Test("Total output exhaustion remains authoritative", arguments: [1, 2, 3])
    func totalOutputLimit(_ maximum: Int) async throws {
        let engine = engine()
        let result = await cbv2SchedCollect(try engine.submit(request(maximumOutput: maximum)))
        await engine.shutdown()
        #expect(result.tokens == Array([1, 2, 8].prefix(maximum)))
        #expect(result.finishReason == .length)
        #expect(result.usage?.completionTokens == maximum)
    }

    @Test("Later open markers cannot create a second quota")
    func repeatedOpen() async throws {
        let engine = engine(next: [1, 2, 3, 4, 5, 6, 7, 8, 14, 10, 11, 12, 13, 14, 15, 0])
        let result = await cbv2SchedCollect(try engine.submit(request(budget: 1)))
        await engine.shutdown()
        #expect(result.tokens == [1, 8, 0, 1, 2, 3])
    }

    @Test("A repeated opener inside the thought cannot evade N")
    func repeatedOpenInside() async throws {
        let engine = engine(next: [14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0])
        let result = await cbv2SchedCollect(try engine.submit(request(budget: 2)))
        await engine.shutdown()
        #expect(result.tokens == [0, 0, 8, 9, 10, 11])
    }

    @Test("Stop tokens and raw logprobs survive a forced delimiter")
    func seededStopAndRawLogprobs() async throws {
        let engine = engine()
        var request = try request()
        request.stopTokens = [8]
        request.sampling = .init(temperature: 0.7, topP: 0.8, topK: 1, seed: 42, topLogprobs: 1)
        var tokens: [Int] = []
        var probabilities: [CBv2TokenLogprob] = []
        var finish: CBv2FinishReason?
        var usage: CBv2Usage?
        for await event in try engine.submit(request) {
            switch event {
            case .delta(_, let value, let logprobs):
                tokens += value
                probabilities += logprobs ?? []
            case .finished(let reason, let value):
                finish = reason
                usage = value
            }
        }
        await engine.shutdown()
        #expect(tokens == [1, 2, 8])
        #expect(finish == .stop)
        #expect(usage?.completionTokens == 3)
        let forced = try #require(probabilities.last)
        #expect(forced.token == 8)
        #expect(abs(forced.logprob - Float(-log(exp(1.0) + 15))) < 0.00001)
        #expect(forced.topLogprobs.first?.token == 3)
    }

    @Test("Mixed requests and reused IDs keep independent budgets")
    func rowIsolationAndReuse() async throws {
        let engine = engine()
        let first = try engine.submit(request(id: 41, budget: 1))
        let second = try engine.submit(request(id: 42, budget: 3))
        var ordinary = try request(id: 43)
        ordinary.reasoningBudget = nil
        let third = try engine.submit(ordinary)
        async let a = cbv2SchedCollect(first)
        async let b = cbv2SchedCollect(second)
        async let c = cbv2SchedCollect(third)
        let results = await (a, b, c)
        #expect(results.0.tokens == [1, 8, 9, 10, 11, 12])
        #expect(results.1.tokens == [1, 2, 3, 8, 9, 10])
        #expect(results.2.tokens == [1, 2, 3, 4, 5, 6])
        let reused = await cbv2SchedCollect(try engine.submit(request(id: 41, budget: 2)))
        await engine.shutdown()
        #expect(reused.tokens == [1, 2, 8, 9, 10, 11])
    }

    @Test("Off and unsupported samplers refuse the control")
    func capabilityRefusal() async throws {
        let off = engine(enabled: false)
        let unsupported = engine(sampler: CBv2GreedySampler())
        let request = try request()
        #expect(!off.supportsReasoningBudgets)
        #expect(!unsupported.supportsReasoningBudgets)
        #expect(throws: CBv2ReasoningBudgetError.unsupported) { try off.submit(request) }
        #expect(throws: CBv2ReasoningBudgetError.unsupported) { try unsupported.submit(request) }
        await off.shutdown()
        await unsupported.shutdown()
    }

    @Test("A marker outside model vocabulary emits no token or completion charge")
    func vocabularyRefusal() async throws {
        let engine = engine()
        var request = try request()
        request.reasoningBudget = try .init(
            maximumTokens: 2, openTokenID: 16, closeTokenID: 8,
            startsInsideThought: true)
        let result = await cbv2SchedCollect(try engine.submit(request))
        await engine.shutdown()
        #expect(result.tokens.isEmpty)
        #expect(result.usage?.completionTokens == 0)
        #expect(result.finishReason == .error("reasoning budget: marker outside model vocabulary"))
    }

    @Test("Configuration rejects unbounded budgets before native work")
    func oversizedBudget() {
        #expect(throws: CBv2ReasoningBudgetError.invalidConfiguration) {
            try CBv2ReasoningBudget(
                maximumTokens: Int.max, openTokenID: 14, closeTokenID: 8,
                startsInsideThought: true)
        }
    }

    @Test("No-budget requests have the same output with the engine switch off or on")
    func noBudgetPreservesOff() async throws {
        let off = engine(enabled: false)
        let on = engine()
        var request = try request()
        request.reasoningBudget = nil
        request.sampling = .init(temperature: 0.7, topK: 1, seed: 123)
        let baseline = await cbv2SchedCollect(try off.submit(request))
        let optedIn = await cbv2SchedCollect(try on.submit(request))
        await off.shutdown()
        await on.shutdown()
        #expect(baseline.tokens == [1, 2, 3, 4, 5, 6])
        #expect(optedIn.tokens == baseline.tokens)
        #expect(optedIn.usage?.completionTokens == baseline.usage?.completionTokens)
    }

    @Test("A stop string spanning answer tokens keeps its normal holdback and usage")
    func answerStopString() async throws {
        let engine = engine(
            detokenizer: CBv2TextDetokenizerFactory(tokenizer: BudgetTextTokenizer()))
        var request = try request()
        request.stopStrings = ["910"]
        let result = await cbv2SchedCollect(try engine.submit(request))
        await engine.shutdown()
        #expect(result.tokens == [1, 2, 8, 9, 10])
        #expect(result.text == "128")
        #expect(result.finishReason == .stop)
        #expect(result.usage?.completionTokens == 5)
    }

}
