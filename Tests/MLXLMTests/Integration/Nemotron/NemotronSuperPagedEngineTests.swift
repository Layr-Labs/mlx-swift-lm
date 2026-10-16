import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// EngineV2 serving of the Super target over segmented native paged KV on
/// tiny synthetic weights. Speculative greedy output equals plain paged
/// greedy output token for token (serial and captured verification), and
/// concurrent row-exact serving equals each request served alone, all on
/// paged storage. Every engine retires its pages, reservations and state.
@Suite(.tags(.integration))
struct NemotronSuperPagedEngineTests {
    typealias Collected = NemotronSuperMTPEngineTests.Collected

    static func engine(
        _ model: any CBv2SteppableModel, language: NemotronHModel, rows: Int = 1,
        drafter: NemotronSuperMTPAssistant? = nil, depth: Int = 1
    ) throws -> (EngineV2, PagedKVBackend) {
        let storage = try NemotronSuperPagedStorage(language)
        let engine = EngineV2(
            model: model, layerKinds: language.cbv2LayerKinds, backend: storage.backend,
            cacheProvider: CBv2LayerCacheBank(caches: storage.caches),
            sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: rows, maxBatchedTokensPerStep: 32,
                prefillChunkSize: 8, maxWaiting: 8),
            admissionConfig: .init(watermarkFraction: 0),
            mtpDrafter: drafter,
            mtpConfig: .init(
                enabled: drafter != nil, maxDraftTokens: depth, maxSpeculativeBatch: 1,
                fixedDraftTokens: depth,
                verificationMode: drafter?.requiredVerificationMode ?? .serialTarget))
        return (engine, storage.backend)
    }

    static func retire(_ engine: EngineV2, _ backend: PagedKVBackend) async {
        engine.loopForTesting.onEngineQueueSync {
            #expect(engine.loopForTesting.recurrentStates.isEmpty)
        }
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
        #expect(backend.bytesWired == 0)
    }

    static func serve(_ engine: EngineV2, base: UInt64) async throws -> [Collected] {
        var results: [Collected] = []
        var id = base
        for prompt in NemotronSuperMTPEngineTests.prompts {
            for maxTokens in [1, 7, 24] {
                results.append(
                    try await NemotronSuperMTPEngineTests.run(
                        engine, id: id, prompt: prompt, maxTokens: maxTokens))
                id += 1
            }
        }
        return results
    }

    @Test(
        .timeLimit(.minutes(5)),
        arguments: [
            NemotronSuperMTPEngineTests.Case(
                head: .predictive, depth: 3, quantized: false, verification: .serial),
            NemotronSuperMTPEngineTests.Case(
                head: .predictive, depth: 3, quantized: true, verification: .serial),
            NemotronSuperMTPEngineTests.Case(
                head: .predictive, depth: 2, quantized: true, verification: .rectangularExact),
            NemotronSuperMTPEngineTests.Case(
                head: .random, depth: 3, quantized: false, verification: .rectangularExact),
        ])
    func pagedSpeculationMatchesPlainPagedDecode(_ testCase: NemotronSuperMTPEngineTests.Case)
        async throws
    {
        let stack = try NemotronSuperMTPFixture.$attentionHeadDim.withValue(64) {
            try testCase.stack()
        }
        let language = stack.model.languageModel
        let (plain, plainBackend) = try Self.engine(
            CBv2SteppableLanguageModelAdapter(NemotronSuperRecurrentAdapter(stack.model)),
            language: language)
        let expected = try await Self.serve(plain, base: 1)
        await Self.retire(plain, plainBackend)

        let (speculative, backend) = try Self.engine(
            CBv2SteppableLanguageModelAdapter(stack.target), language: language,
            drafter: stack.assistant, depth: testCase.depth)
        let actual = try await Self.serve(speculative, base: 100)
        for (index, (lhs, rhs)) in zip(actual, expected).enumerated() {
            #expect(lhs.tokens == rhs.tokens, "\(testCase.testDescription) request \(index)")
            #expect(lhs.finish == rhs.finish)
        }
        let metrics = try #require(speculative.mtpMetricsSnapshot())
        #expect(metrics.rounds > 0 && metrics.acceptedTokens > 0)
        let rectangular = testCase.verification == .rectangularExact
        #expect(metrics.rectangularVerificationRounds == (rectangular ? metrics.rounds : 0))
        speculative.loopForTesting.onEngineQueueSync {
            #expect(speculative.loopForTesting.mtp?.requestStateCountForTesting == 0)
        }
        await Self.retire(speculative, backend)
    }

    /// Concurrent requests (a mid-stream join and a cancellation included)
    /// share row-exact `[B, 1]` decode steps over paged rows and equal each
    /// request served alone on paged storage.
    @Test(.timeLimit(.minutes(5)))
    func pagedConcurrentServingMatchesSoloPagedServing() async throws {
        let model = try NemotronSuperBatchFixture.model(
            seed: 17, dtype: .bfloat16, products: .rowwise)
        let language = model.languageModel
        let (solo, soloBackend) = try Self.engine(
            CBv2SteppableLanguageModelAdapter(NemotronSuperRecurrentAdapter(model)),
            language: language)
        var expected: [Collected] = []
        for index in NemotronSuperBatchedEngineTests.requests.indices {
            expected.append(
                try await NemotronSuperBatchedEngineTests.collect(
                    try solo.submit(NemotronSuperBatchedEngineTests.request(index, base: 100))
                ) { _ in })
        }
        await Self.retire(solo, soloBackend)

        let adapter = try NemotronSuperRecurrentAdapter(model, maximumDecodeRows: 4)
        let (concurrent, backend) = try Self.engine(
            CBv2SteppableLanguageModelAdapter(adapter), language: language, rows: 4)
        let (actual, shapes) = try await NemotronSuperBatchedEngineTests.concurrent(
            engine: concurrent)
        let cancelled = NemotronSuperBatchedEngineTests.cancelledRequest
        for index in expected.indices where index != cancelled {
            #expect(actual[index].tokens == expected[index].tokens, "request \(index)")
            #expect(actual[index].finish == .length)
        }
        #expect(actual[cancelled].finish == .cancelled)
        #expect(
            Array(expected[cancelled].tokens.prefix(actual[cancelled].tokens.count))
                == actual[cancelled].tokens)
        #expect(NemotronSuperBatchedEngineTests.widestDecode(shapes) >= 3)
        await Self.retire(concurrent, backend)
    }
}
