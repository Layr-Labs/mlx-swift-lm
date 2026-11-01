import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Real EngineV2 serving with the Super text target and its embedded-head
/// assistant on tiny synthetic weights. Greedy output must equal the ordinary
/// serial Super adapter token for token. Timings printed here are synthetic.
@Suite(.tags(.integration))
struct NemotronSuperMTPEngineTests {
    struct Collected {
        var tokens: [Int] = []
        var finish: CBv2FinishReason?
    }

    static func collect(_ stream: AsyncStream<CBv2Event>) async -> Collected {
        var result = Collected()
        for await event in stream {
            switch event {
            case .delta(_, let tokens, _): result.tokens += tokens
            case .finished(let reason, _): result.finish = reason
            }
        }
        return result
    }

    static func engine(
        _ model: any CBv2SteppableModel, kinds: [CBv2LayerKind],
        caches: [any CBv2AttendingLayerCache],
        drafter: NemotronSuperMTPAssistant? = nil, depth: Int = 1, kvDType: DType = .float32,
        sampler: any CBv2StepSampler = CBv2GreedySampler()
    ) -> (EngineV2, CBv2ContiguousKVBackend) {
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 64 << 20, kvDType: kvDType))
        let engine = EngineV2(
            model: model, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: caches),
            sampler: sampler,
            schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 16,
                prefillChunkSize: 8, maxWaiting: 4),
            admissionConfig: .init(watermarkFraction: 0),
            mtpDrafter: drafter,
            mtpConfig: .init(
                enabled: drafter != nil, maxDraftTokens: depth, maxSpeculativeBatch: 1,
                fixedDraftTokens: depth,
                verificationMode: drafter?.requiredVerificationMode ?? .serialTarget))
        return (engine, backend)
    }

    static func plainEngine(
        _ model: NemotronSuperModel, kvDType: DType = .float32,
        sampler: any CBv2StepSampler = CBv2GreedySampler()
    ) -> (EngineV2, CBv2ContiguousKVBackend) {
        let adapter = NemotronSuperRecurrentAdapter(model)
        return engine(
            CBv2SteppableLanguageModelAdapter(adapter), kinds: adapter.cbv2LayerKinds,
            caches: adapter.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) },
            kvDType: kvDType, sampler: sampler)
    }

    static func speculativeEngine(
        _ stack: NemotronSuperMTPFixture.Stack, depth: Int, kvDType: DType = .float32
    ) -> (EngineV2, CBv2ContiguousKVBackend) {
        engine(
            CBv2SteppableLanguageModelAdapter(stack.target), kinds: stack.target.cbv2LayerKinds,
            caches: stack.target.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) },
            drafter: stack.assistant, depth: depth, kvDType: kvDType)
    }

    /// One fixture and verification strategy under test.
    struct Case: Sendable, CustomTestStringConvertible {
        let head: NemotronSuperMTPFixture.Head
        let depth: Int
        let quantized: Bool
        let verification: NemotronSuperMTPAssistant.Settings.Verification
        var testDescription: String {
            "\(head) depth \(depth) \(quantized ? "bf16-modelopt" : "fp32") \(verification)"
        }
        var kvDType: DType { quantized ? .bfloat16 : .float32 }
        func stack() throws -> NemotronSuperMTPFixture.Stack {
            try NemotronSuperMTPFixture.stack(
                kind: head, depth: depth, mixerScale: head == .predictive ? 0.25 : 1,
                quantized: quantized, verification: verification)
        }
    }

    static let cases: [Case] =
        [
            (NemotronSuperMTPFixture.Head.copy, 1), (.random, 1), (.copy, 2), (.copy, 3),
            (.random, 3), (.predictive, 1), (.predictive, 3),
        ].map { Case(head: $0.0, depth: $0.1, quantized: false, verification: .serial) }
        + [
            (NemotronSuperMTPFixture.Head.copy, 1, false), (.predictive, 1, false),
            (.predictive, 3, false), (.random, 3, false), (.predictive, 2, true),
            (.predictive, 3, true), (.random, 1, true),
        ].map { Case(head: $0.0, depth: $0.1, quantized: $0.2, verification: .rectangularExact) }
        + [Case(head: .predictive, depth: 3, quantized: true, verification: .serial)]

    static let prompts: [[Int]] = [
        [1, 2, 3],
        [5, 9, 1, 4, 22, 7, 3, 8, 30, 12, 2],
        (0 ..< 40).map { ($0 * 7 + 3) % 61 + 1 }.filter { $0 != 18 },
    ]

    static func run(_ engine: EngineV2, id: UInt64, prompt: [Int], maxTokens: Int) async throws
        -> Collected
    {
        await collect(
            try engine.submit(
                CBv2Request(
                    id: CBv2RequestID(id), promptTokens: prompt,
                    sampling: .init(temperature: 0), maxTokens: maxTokens)))
    }

    @Test(.timeLimit(.minutes(5)), arguments: Self.cases)
    func greedySpeculationMatchesPlainTarget(_ testCase: Case) async throws {
        let head = testCase.head
        let depth = testCase.depth
        let stack = try testCase.stack()
        let (plain, plainBackend) = Self.plainEngine(stack.model, kvDType: testCase.kvDType)
        var expected: [Collected] = []
        var id: UInt64 = 1
        let plainStarted = DispatchTime.now().uptimeNanoseconds
        for prompt in Self.prompts {
            for maxTokens in [1, 7, 24] {
                expected.append(
                    try await Self.run(plain, id: id, prompt: prompt, maxTokens: maxTokens))
                id += 1
            }
        }
        let plainElapsed = DispatchTime.now().uptimeNanoseconds - plainStarted
        await plain.shutdown()
        #expect(plainBackend.bytesReserved == 0)

        let (speculative, backend) = Self.speculativeEngine(
            stack, depth: depth, kvDType: testCase.kvDType)
        let driver = try #require(speculative.loopForTesting.mtp)
        #expect(driver.config.verificationMode == stack.assistant.requiredVerificationMode)
        #expect(driver.config.maxDraftTokens == depth)
        #expect(driver.config.maxSpeculativeBatch == 1)
        var index = 0
        let started = DispatchTime.now().uptimeNanoseconds
        for prompt in Self.prompts {
            for maxTokens in [1, 7, 24] {
                let actual = try await Self.run(
                    speculative, id: id, prompt: prompt, maxTokens: maxTokens)
                #expect(
                    actual.tokens == expected[index].tokens,
                    "prompt \(prompt.count) maxTokens \(maxTokens) head \(head) depth \(depth)")
                #expect(actual.finish == expected[index].finish)
                index += 1
                id += 1
            }
        }
        let elapsed = DispatchTime.now().uptimeNanoseconds - started
        let metrics = try #require(speculative.mtpMetricsSnapshot())
        // Bookkeeping: a round drafts at most `depth` (the engine shortens the
        // final rounds near the length bound), emits accepted + 1 unless that
        // bound truncates it, and per-position acceptance is monotone.
        #expect(metrics.rounds > 0)
        #expect(metrics.draftedTokens <= metrics.rounds * depth)
        #expect(metrics.draftedTokens >= metrics.rounds)
        #expect(metrics.acceptedTokens <= metrics.draftedTokens)
        #expect(metrics.emittedTokens <= metrics.acceptedTokens + metrics.rounds)
        #expect(metrics.emittedTokens >= metrics.rounds)
        let rectangular = testCase.verification == .rectangularExact
        #expect(metrics.serialVerificationRounds == (rectangular ? 0 : metrics.rounds))
        #expect(metrics.rectangularVerificationRounds == (rectangular ? metrics.rounds : 0))
        let positions = metrics.perPositionAccepted
        #expect(zip(positions, positions.dropFirst()).allSatisfy { $0 >= $1 })
        #expect(positions.reduce(0, +) == metrics.acceptedTokens)
        if head != .random { #expect(metrics.acceptedTokens > 0) }
        if head == .predictive && depth > 1 {
            #expect(positions.count == depth && positions[1] > 0)
        }
        print(
            "SYNTHETIC super-mtp \(testCase.testDescription) rounds=\(metrics.rounds) "
                + "drafted=\(metrics.draftedTokens) accepted=\(metrics.acceptedTokens) "
                + "emitted=\(metrics.emittedTokens) seeds=\(metrics.seedSteps) "
                + "perPosition=\(positions) roundWallNs=\(metrics.totalRoundWallTimeNanos) "
                + "speculativeWallNs=\(elapsed) plainWallNs=\(plainElapsed)")
        speculative.loopForTesting.onEngineQueueSync {
            #expect(speculative.loopForTesting.mtp?.requestStateCountForTesting == 0)
            #expect(speculative.loopForTesting.recurrentStates.isEmpty)
        }
        await speculative.shutdown()
        #expect(backend.bytesReserved == 0)
    }

    @Test(
        .timeLimit(.minutes(2)),
        arguments: [
            (1, NemotronSuperMTPAssistant.Settings.Verification.serial), (3, .serial),
            (1, .rectangularExact), (3, .rectangularExact),
        ])
    func cancellationMidGenerationRetiresAndEngineStaysExact(
        depth: Int, verification: NemotronSuperMTPAssistant.Settings.Verification
    ) async throws {
        let stack = try NemotronSuperMTPFixture.stack(
            kind: .copy, depth: depth, verification: verification)
        let prompt = Self.prompts[1]
        let (plain, _) = Self.plainEngine(stack.model)
        let expected = try await Self.run(plain, id: 1, prompt: prompt, maxTokens: 12)
        await plain.shutdown()

        let (engine, backend) = Self.speculativeEngine(stack, depth: depth)
        for cancelAfter in [0, 1, 3] {
            let id = CBv2RequestID(UInt64(100 + cancelAfter))
            let (events, retirement) = try engine.submitWithRetirement(
                CBv2Request(
                    id: id, promptTokens: prompt, sampling: .init(temperature: 0),
                    maxTokens: 64))
            if cancelAfter == 0 { engine.cancel(id) }
            var deltas = 0
            var finish: CBv2FinishReason?
            for await event in events {
                switch event {
                case .delta:
                    deltas += 1
                    if deltas == cancelAfter { engine.cancel(id) }
                case .finished(let reason, _): finish = reason
                }
            }
            await retirement.wait()
            #expect(finish != nil)
            engine.loopForTesting.onEngineQueueSync {
                #expect(engine.loopForTesting.mtp?.requestStateCountForTesting == 0)
                #expect(engine.loopForTesting.recurrentStates.isEmpty)
                #expect(engine.loopForTesting.scheduler.record(for: id) == nil)
            }
            #expect(backend.bytesReserved == 0)
        }
        let resumed = try await Self.run(engine, id: 200, prompt: prompt, maxTokens: 12)
        #expect(resumed.tokens == expected.tokens)
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
    }

    /// Causal image requests are served target-only
    /// (`NemotronSuperMTPMediaFallbackTests`); bidirectional spans stay refused.
    @Test func bidirectionalMediaIsRefusedBeforeAnyAssistantState() async throws {
        let stack = try NemotronSuperMTPFixture.stack(kind: .copy)
        let (engine, backend) = Self.speculativeEngine(stack, depth: 1)
        // Refused at validation, before the embeddings provider is called.
        let multimodal = CBv2MultimodalInput(
            spans: [CBv2ImageSpan(tokenOffset: 1, length: 1)], attention: .bidirectionalSpans
        ) { [MLXArray.zeros([1, NemotronSuperMTPFixture.hidden])] }
        #expect(throws: CBv2MultimodalError.self) {
            _ = try engine.submit(
                CBv2Request(
                    id: CBv2RequestID(1), promptTokens: [1, 18, 2],
                    sampling: .init(temperature: 0), maxTokens: 2, multimodal: multimodal))
        }
        engine.loopForTesting.onEngineQueueSync {
            #expect(engine.loopForTesting.mtp?.requestStateCountForTesting == 0)
        }
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
    }

    @Test func ordinaryAdapterAndForeignTargetsNeverSpeculate() throws {
        let stack = try NemotronSuperMTPFixture.stack(kind: .copy)
        let config = CBv2MTPConfig(
            enabled: true, maxDraftTokens: 1, fixedDraftTokens: 1, verificationMode: .serialTarget)
        #expect(
            CBv2MTPRoundDriver.build(
                model: CBv2SteppableLanguageModelAdapter(stack.target),
                drafter: stack.assistant, config: config) != nil)
        // The ordinary serving adapter stays fail-closed for MTP.
        #expect(
            CBv2MTPRoundDriver.build(
                model: CBv2SteppableLanguageModelAdapter(
                    NemotronSuperRecurrentAdapter(stack.model)),
                drafter: stack.assistant, config: config) == nil)
        #expect(!NemotronSuperRecurrentAdapter(stack.model).cbv2Capabilities.supportsMTP)
        // Same architecture, different loaded target: identity must not match.
        let other = NemotronSuperMTPTextTarget(try NemotronSuperMTPFixture.target(seed: 8))
        #expect(
            CBv2MTPRoundDriver.build(
                model: CBv2SteppableLanguageModelAdapter(other),
                drafter: stack.assistant, config: config) == nil)
    }
}
