import Foundation
import MLX
import Testing

@testable import MLXLLM
@_spi(Benchmarking) @testable import MLXLMCommon
@testable import MLXVLM

/// Real EngineV2 speculation for several concurrent Super text requests: one
/// captured `[B, 1 + k]` verify window per round, each request drafting from
/// its own assistant state and committing its own accepted prefix. Greedy
/// output must equal each request served alone by the plain serial adapter,
/// including a request that joins mid-stream and a cancelled one. Synthetic
/// weights; timings are not model evidence.
@Suite(.tags(.integration))
struct NemotronSuperMTPBatchedEngineTests {
    typealias Collected = NemotronSuperMTPEngineTests.Collected
    typealias Batched = NemotronSuperBatchedEngineTests

    struct Case: Sendable, CustomTestStringConvertible {
        let head: NemotronSuperMTPFixture.Head
        let depth: Int
        let quantized: Bool
        let rows: Int
        var testDescription: String {
            "\(head) depth \(depth) \(quantized ? "bf16-modelopt" : "fp32") rows \(rows)"
        }
    }

    static let cases: [Case] = [
        Case(head: .predictive, depth: 3, quantized: true, rows: 4),
        Case(head: .predictive, depth: 2, quantized: false, rows: 4),
        Case(head: .copy, depth: 1, quantized: true, rows: 2),
        Case(head: .random, depth: 3, quantized: true, rows: 3),
    ]

    static let requests: [(prompt: [Int], maxTokens: Int)] = [
        ([5, 9, 1, 4, 22, 7, 3, 8, 30, 12, 2], 24),
        ([1, 2, 3], 17),
        ((0 ..< 20).map { ($0 * 7 + 3) % 61 + 1 }.filter { $0 != 18 }, 30),
        ([40, 41, 42, 43, 44, 45, 46], 9),
        ([11, 13, 17, 19, 23], 40),
    ]
    /// Submitted once request 0 has emitted three tokens.
    static let lateRequest = 3
    /// Cancelled once it has emitted four tokens.
    static let cancelledRequest = 4

    static func request(_ index: Int, base: UInt64) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(base + UInt64(index)), promptTokens: requests[index].prompt,
            sampling: .init(temperature: 0), maxTokens: requests[index].maxTokens)
    }

    static func speculativeEngine(_ stack: NemotronSuperMTPFixture.Stack, testCase: Case)
        -> (EngineV2, CBv2ContiguousKVBackend)
    {
        let backend = CBv2ContiguousKVBackend(
            config: .init(
                bytesCapacity: 64 << 20, kvDType: testCase.quantized ? .bfloat16 : .float32))
        let engine = EngineV2(
            model: CBv2SteppableLanguageModelAdapter(stack.target),
            layerKinds: stack.target.cbv2LayerKinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(
                caches: stack.target.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) }),
            sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: testCase.rows, maxBatchedTokensPerStep: 32,
                prefillChunkSize: 8, maxWaiting: 8),
            admissionConfig: .init(watermarkFraction: 0),
            mtpDrafter: stack.assistant,
            mtpConfig: .init(
                enabled: true, maxDraftTokens: testCase.depth,
                maxSpeculativeBatch: testCase.rows, fixedDraftTokens: testCase.depth,
                verificationMode: .rectangularExact))
        return (engine, backend)
    }

    static func stack(_ testCase: Case) throws -> NemotronSuperMTPFixture.Stack {
        let model = try NemotronSuperMTPFixture.target(
            mixerScale: testCase.head == .predictive ? 0.25 : 1, quantized: testCase.quantized)
        let target = try NemotronSuperMTPCaptureTarget(
            capturing: model, maximumRequests: testCase.rows)
        let head = try NemotronSuperMTPFixture.head(
            for: model.languageModel, kind: testCase.head,
            dtype: testCase.quantized ? .bfloat16 : .float32)
        var settings = NemotronSuperMTPFixture.settings(
            depth: testCase.depth, verification: .rectangularExact)
        settings.maximumSpeculativeRows = testCase.rows
        let assistant = try target.makeAssistant(head: head, settings: settings)
        return .init(model: model, target: target, head: head, assistant: assistant)
    }

    static func solo(_ model: NemotronSuperModel, kvDType: DType) async throws -> [Collected] {
        let (engine, backend) = NemotronSuperMTPEngineTests.plainEngine(model, kvDType: kvDType)
        var results: [Collected] = []
        for index in requests.indices {
            results.append(
                try await Batched.collect(try engine.submit(request(index, base: 100))) { _ in })
        }
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
        return results
    }

    @Test(.timeLimit(.minutes(5)), arguments: Self.cases)
    func concurrentSpeculationMatchesSoloPlainDecode(_ testCase: Case) async throws {
        let stack = try Self.stack(testCase)
        let solo = try await Self.solo(
            stack.model, kvDType: testCase.quantized ? .bfloat16 : .float32)
        let (engine, backend) = Self.speculativeEngine(stack, testCase: testCase)
        let driver = try #require(engine.loopForTesting.mtp)
        #expect(driver.config.maxSpeculativeBatch == testCase.rows)
        #expect(driver.config.verificationMode == .rectangularExact)
        _ = try engine.beginForwardShapeObservation()
        let late = Batched.LateJoin()
        let initial = Self.requests.indices.filter { $0 != Self.lateRequest }
        let streams = try initial.map { try engine.submit(Self.request($0, base: 200)) }
        var results: [Int: Collected] = [:]
        try await withThrowingTaskGroup(of: (Int, Collected).self) { group in
            for (stream, index) in zip(streams, initial) {
                group.addTask {
                    var cancelSent = false
                    let collected = try await Batched.collect(stream) { count in
                        if index == Self.cancelledRequest, count >= 4, !cancelSent {
                            cancelSent = true
                            engine.cancel(CBv2RequestID(200 + UInt64(index)))
                        }
                        if index == 0, count >= 3 {
                            late.start {
                                try engine.submit(Self.request(Self.lateRequest, base: 200))
                            }
                        }
                    }
                    return (index, collected)
                }
            }
            for try await (index, collected) in group { results[index] = collected }
        }
        results[Self.lateRequest] = try await late.value()
        let shapes = engine.forwardShapeSnapshot()
        let metrics = try #require(engine.mtpMetricsSnapshot())
        for index in Self.requests.indices where index != Self.cancelledRequest {
            #expect(
                results[index]?.tokens == solo[index].tokens, "\(testCase) request \(index)")
            #expect(results[index]?.finish == .length)
        }
        let cancelled = try #require(results[Self.cancelledRequest])
        #expect(cancelled.finish == .cancelled)
        #expect(
            Array(solo[Self.cancelledRequest].tokens.prefix(cancelled.tokens.count))
                == cancelled.tokens)
        let widestVerify =
            shapes.entries.filter { $0.axes.phase == .mtpVerification }
            .map(\.axes.liveBatchRows).max() ?? 0
        #expect(widestVerify >= 2, "\(testCase) widest verify \(widestVerify)")
        // `rounds` counts request rounds; a verify step carries several.
        #expect(metrics.rounds > 0)
        #expect(metrics.rectangularVerificationRounds > 0)
        #expect(metrics.rectangularVerificationRounds < metrics.rounds)
        #expect(metrics.serialVerificationRounds == 0)
        if testCase.head != .random { #expect(metrics.acceptedTokens > 0) }
        print(
            "SYNTHETIC super-mtp-batched \(testCase) widestVerify=\(widestVerify) "
                + "verifySteps=\(metrics.rectangularVerificationRounds) "
                + "rounds=\(metrics.rounds) drafted=\(metrics.draftedTokens) "
                + "accepted=\(metrics.acceptedTokens) emitted=\(metrics.emittedTokens) "
                + "seeds=\(metrics.seedSteps) skipped=\(metrics.skippedRows)")
        engine.loopForTesting.onEngineQueueSync {
            #expect(engine.loopForTesting.mtp?.requestStateCountForTesting == 0)
            #expect(engine.loopForTesting.recurrentStates.isEmpty)
        }
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
    }
}
