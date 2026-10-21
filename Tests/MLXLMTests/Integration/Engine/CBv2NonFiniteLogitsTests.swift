import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// A sampled position whose logits hold NaN or infinity fails its request
/// with the generation-failure terminal and emits no token from that
/// position, on the contiguous and the paged backend, at the prompt frontier
/// and on the chained decode path. A co-batched healthy request keeps its
/// solo tokens and every reservation is returned.
@Suite(.tags(.integration))
struct CBv2NonFiniteLogitsTests {
    enum Backend: String, CaseIterable, Sendable { case contiguous, paged }

    struct Case: Sendable, CustomTestStringConvertible {
        let backend: Backend
        let poison: Float
        var testDescription: String { "\(backend.rawValue) \(poison)" }
    }

    static let cases: [Case] = Backend.allCases.flatMap { backend in
        [Float.nan, .infinity, -.infinity].map { Case(backend: backend, poison: $0) }
    }

    static let kind = CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2)
    static let trigger = 6

    /// Attends through every layer cache, so both backends write real KV,
    /// and scores `(token + 1) % vocabulary`. One logit of a position whose
    /// input token is `trigger` becomes `poison`.
    final class TriggerModel: CBv2SteppableModel, @unchecked Sendable {
        static let vocabulary = 32
        let poison: Float

        init(poison: Float) { self.poison = poison }

        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
            let b = tokens.dim(0)
            let n = tokens.dim(1)
            var hidden = tokens.asType(.float32).reshaped([b, 1, n, 1]) / Float(Self.vocabulary)
            for cache in caches {
                let kind = cache.kind
                let output = cache.updateAndAttend(
                    queries: MLXArray.zeros([b, kind.queryHeads, n, kind.headDim]),
                    keys: broadcast(hidden, to: [b, kind.kvHeads, n, kind.headDim]),
                    values: broadcast(hidden, to: [b, kind.kvHeads, n, kind.valueHeadDim]),
                    scale: 0.125, sinks: nil)
                hidden = mean(output, axes: [1, 3], keepDims: true)
            }
            let ids = MLXArray(Int32(0) ..< Int32(Self.vocabulary))
            let next = ((tokens + 1) % Self.vocabulary).expandedDimensions(axis: -1)
            // The attention output stays in the graph as a finite zero.
            let logits =
                which(ids .== next, Float(10), Float(-10)) + 0 * hidden.reshaped([b, n, 1])
            let poisoned =
                (tokens .== Int32(CBv2NonFiniteLogitsTests.trigger))
                .expandedDimensions(axis: -1) .&& (ids .== Int32(3))
            return which(poisoned, MLXArray(poison), logits)
        }
    }

    /// A host sampler that bans token 0 by writing -infinity into the logits
    /// it is given, in place, then samples greedily.
    final class InPlaceMaskingSampler: CBv2StepSampler {
        let greedy = CBv2GreedySampler()

        func sample(
            logits: MLXArray, params: [CBv2SamplingParams], requestIDs: [CBv2RequestID],
            stepIndex: Int, pendingSampledTokens: MLXArray?,
            rowContext: () -> [CBv2SamplerRow]
        ) -> MLXArray {
            logits[0..., 0] = MLXArray(-Float.infinity)
            return greedy.sample(
                logits: logits, params: params, requestIDs: requestIDs, stepIndex: stepIndex,
                pendingSampledTokens: pendingSampledTokens, rowContext: rowContext)
        }
    }

    struct Engine {
        let engine: EngineV2
        let bytesReserved: () -> Int
    }

    static func engine(
        _ backend: Backend, poison: Float, sampler: CBv2StepSampler = CBv2GreedySampler()
    ) throws -> Engine {
        let model = TriggerModel(poison: poison)
        let scheduler = CBv2SchedulerConfig(
            maxConcurrentRequests: 4, maxBatchedTokensPerStep: 64, prefillChunkSize: 16,
            maxWaiting: 8)
        switch backend {
        case .contiguous:
            let storage = CBv2ContiguousKVBackend(
                config: .init(bytesCapacity: 8 << 20, kvDType: .float32))
            let engine = EngineV2(
                model: model, layerKinds: [kind], backend: storage,
                cacheProvider: CBv2LayerCacheBank(caches: [
                    CBv2LayerCache(layerIndex: 0, kind: kind)
                ]),
                sampler: sampler, schedulerConfig: scheduler,
                admissionConfig: .init(watermarkFraction: 0))
            return Engine(engine: engine, bytesReserved: { storage.bytesReserved })
        case .paged:
            let storage = try PagedKVBackend(
                layerKinds: [kind],
                config: .init(
                    capacityBytes: 8 << 20, dtype: .float32, maxPrefillChunk: 16,
                    nominalMaxSequenceLength: 512))
            let engine = EngineV2(
                model: model, layerKinds: [kind], backend: storage,
                cacheProvider: CBv2LayerCacheBank(caches: storage.makeLayerCaches()),
                sampler: sampler, schedulerConfig: scheduler,
                admissionConfig: .init(watermarkFraction: 0))
            return Engine(engine: engine, bytesReserved: { storage.bytesReserved })
        }
    }

    static func request(_ id: UInt64, prompt: [Int], maxTokens: Int) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt, sampling: .init(temperature: 0),
            maxTokens: maxTokens)
    }

    static func failures(_ engine: EngineV2) -> (sampled: Int, verify: Int, chained: Int) {
        let loop = engine.loopForTesting
        return loop.onEngineQueueSync {
            (loop.nonFiniteSampledFailures, loop.nonFiniteVerifyFailures, loop.chainedStepCount)
        }
    }

    @Test("flags mark NaN and infinity, never large finite values")
    func flagsMarkOnlyNonFiniteRows() {
        for dtype in [DType.float32, .bfloat16, .float16] {
            let large: Float = dtype == .float16 ? 65_504 : 3e38
            let rows: [[Float]] = [
                [1, -2, large, -large], [1, .nan, 0, 0], [.infinity, 1, 2, 3],
                [0, 0, 0, -.infinity], [0, 0, 0, 0],
            ]
            let logits = MLXArray(rows.flatMap { $0 }, [rows.count, 4]).asType(dtype)
            let flags = CBv2NonFiniteLogits.flags(logits).asArray(Int32.self)
            #expect(flags == [0, 1, 1, 1, 0], "\(dtype)")
            let positions = CBv2NonFiniteLogits.flags(
                logits.reshaped([1, rows.count, 4]), keptAxes: 2)
            #expect(positions.shape == [1, rows.count])
            #expect(positions.asArray(Int32.self) == [0, 1, 1, 1, 0], "\(dtype)")
        }
    }

    @Test(.timeLimit(.minutes(2)), arguments: cases)
    func promptFrontierFailsBeforeAnyToken(_ testCase: Case) async throws {
        let fixture = try Self.engine(testCase.backend, poison: testCase.poison)
        let result = await cbv2SchedCollect(
            try fixture.engine.submit(
                Self.request(1, prompt: [4, 5, Self.trigger], maxTokens: 8)))
        #expect(result.tokens.isEmpty)
        #expect(result.finishReason == CBv2NonFiniteLogits.finishReason)
        #expect(result.usage?.completionTokens == 0)
        #expect(Self.failures(fixture.engine).sampled == 1)
        await fixture.engine.shutdown()
        #expect(fixture.bytesReserved() == 0)
        #expect(fixture.engine.capacity().activeRequests == 0)
    }

    /// The trigger is the faulty request's fourth token. The decode step that
    /// consumes it runs chained beside a healthy request; the faulty request
    /// stops after that token and the healthy one keeps its solo tokens.
    @Test(.timeLimit(.minutes(2)), arguments: cases)
    func chainedDecodeFailsOnlyTheFaultyRow(_ testCase: Case) async throws {
        let healthyPrompt = [20, 21]
        let solo = try Self.engine(testCase.backend, poison: testCase.poison)
        let alone = await cbv2SchedCollect(
            try solo.engine.submit(Self.request(1, prompt: healthyPrompt, maxTokens: 10)))
        await solo.engine.shutdown()
        #expect(alone.tokens == Array(22 ..< 32) && alone.finishReason == .length)

        let fixture = try Self.engine(testCase.backend, poison: testCase.poison)
        let faulty = try fixture.engine.submit(Self.request(2, prompt: [1, 2], maxTokens: 10))
        let healthy = try fixture.engine.submit(
            Self.request(3, prompt: healthyPrompt, maxTokens: 10))
        async let faultyResult = cbv2SchedCollect(faulty)
        async let healthyResult = cbv2SchedCollect(healthy)
        let (failed, kept) = await (faultyResult, healthyResult)
        #expect(failed.tokens == [3, 4, 5, Self.trigger])
        #expect(failed.finishReason == CBv2NonFiniteLogits.finishReason)
        #expect(failed.usage?.completionTokens == 4)
        #expect(kept.tokens == alone.tokens && kept.finishReason == .length)
        let counts = Self.failures(fixture.engine)
        #expect(counts.sampled == 1 && counts.verify == 0)
        #expect(counts.chained > 0, "pure decode steps chain")
        await fixture.engine.shutdown()
        #expect(fixture.bytesReserved() == 0)
        #expect(fixture.engine.capacity().activeRequests == 0)
    }

    /// The flags read the model's logits before the sampler runs, so a host
    /// sampler that masks a token in place flags no healthy row, at the
    /// prompt frontier or on the chained decode path. The ban never touches
    /// this prompt's tokens.
    @Test(.timeLimit(.minutes(2)), arguments: Backend.allCases)
    func aSamplerMaskingInPlaceFlagsNoHealthyRow(_ backend: Backend) async throws {
        let fixture = try Self.engine(backend, poison: .nan, sampler: InPlaceMaskingSampler())
        let result = await cbv2SchedCollect(
            try fixture.engine.submit(Self.request(1, prompt: [20, 21], maxTokens: 10)))
        #expect(result.tokens == Array(22 ..< 32))
        #expect(result.finishReason == .length)
        let counts = Self.failures(fixture.engine)
        #expect(counts.sampled == 0)
        #expect(counts.chained > 0, "pure decode steps chain")
        await fixture.engine.shutdown()
        #expect(fixture.bytesReserved() == 0)
    }
}
