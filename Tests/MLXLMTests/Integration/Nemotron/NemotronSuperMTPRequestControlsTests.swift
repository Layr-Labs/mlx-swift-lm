import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Request controls on the speculating Super owner. A sampled request
/// speculates through target-prefix acceptance and draws exactly the tokens
/// target-only sampling draws with the same seed. A request the engine cannot
/// draft for (stop strings, a penalty, a logit bias) decodes one ordinary token
/// per step while the text beside it keeps speculating. Synthetic weights.
@Suite(.tags(.integration))
struct NemotronSuperMTPRequestControlsTests {
    typealias Collected = NemotronSuperMTPEngineTests.Collected
    typealias Batched = NemotronSuperBatchedEngineTests
    typealias Fallback = NemotronSuperMTPMediaFallbackTests

    struct SampledCase: Sendable, CustomTestStringConvertible {
        let quantized: Bool
        let verification: NemotronSuperMTPAssistant.Settings.Verification
        var testDescription: String {
            "\(quantized ? "bf16-modelopt" : "fp32") \(verification)"
        }
    }

    @Test(
        .timeLimit(.minutes(5)),
        arguments: [
            SampledCase(quantized: true, verification: .rectangularExact),
            SampledCase(quantized: false, verification: .rectangularExact),
            SampledCase(quantized: true, verification: .serial),
        ])
    func sampledRequestSpeculatesAndDrawsTheTargetOnlyTokens(_ testCase: SampledCase)
        async throws
    {
        let depth = 2
        let kvDType: DType = testCase.quantized ? .bfloat16 : .float32
        let stack = try NemotronSuperMTPFixture.stack(
            kind: .predictive, depth: depth, mixerScale: 0.25, quantized: testCase.quantized,
            verification: testCase.verification)
        var request = Fallback.textRequest(0)
        request.sampling = .init(temperature: 0.7, topP: 0.95, seed: 29)

        let (plain, plainBackend) = NemotronSuperMTPEngineTests.plainEngine(
            stack.model, kvDType: kvDType, sampler: CBv2DefaultSampler())
        let reference = await NemotronSuperMTPEngineTests.collect(try plain.submit(request))
        await plain.shutdown()
        #expect(plainBackend.bytesReserved == 0)
        #expect(reference.finish == .length && reference.tokens.count == request.maxTokens)

        let (engine, backend) = NemotronSuperMTPEngineTests.engine(
            CBv2SteppableLanguageModelAdapter(stack.target), kinds: stack.target.cbv2LayerKinds,
            caches: stack.target.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) },
            drafter: stack.assistant, depth: depth, kvDType: kvDType,
            sampler: CBv2DefaultSampler())
        let served = await NemotronSuperMTPEngineTests.collect(try engine.submit(request))
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)

        #expect(served.tokens == reference.tokens, "\(testCase) sampled tokens")
        #expect(served.finish == .length)
        // Speculation ran for the sampled request; before target-prefix
        // acceptance its row was ineligible and every step was target-only.
        #expect(metrics.rounds > 0, "\(testCase) rounds=\(metrics.rounds)")
        print(
            "SYNTHETIC super-mtp-sampled \(testCase) rounds=\(metrics.rounds) "
                + "proposed=\(metrics.draftedTokens) accepted=\(metrics.acceptedTokens)")
    }

    enum Control: String, CaseIterable, Sendable, CustomTestStringConvertible {
        case stopString, repetitionPenalty, logitBias
        var testDescription: String { rawValue }

        func applied(to request: CBv2Request) -> CBv2Request {
            var request = request
            switch self {
            case .stopString: request.stopStrings = ["\u{1}never matches\u{1}"]
            case .repetitionPenalty:
                request.sampling = .init(temperature: 0, repetitionPenalty: 1.3)
            case .logitBias: request.sampling = .init(temperature: 0, logitBias: [5: -40])
            }
            return request
        }
    }

    /// The controlled row never drafts, verifies, carries or owns assistant
    /// state and equals the plain engine's output; the text row beside it keeps
    /// speculating and its output does not change.
    @Test(.timeLimit(.minutes(5)), arguments: Control.allCases)
    func ineligibleRowDecodesBesideSpeculatingText(_ control: Control) async throws {
        let kvDType = DType.bfloat16
        let stack = try Fallback.Speculative.stack(
            .init(head: .predictive, depth: Fallback.depth, quantized: true, rows: Fallback.rows))
        let soloText = try await Fallback.soloText(stack.model, kvDType: kvDType)
        let controlled = control.applied(to: Fallback.textRequest(0))
        let (plain, plainBackend) = NemotronSuperMTPEngineTests.plainEngine(
            stack.model, kvDType: kvDType, sampler: CBv2DefaultSampler())
        let reference = await NemotronSuperMTPEngineTests.collect(try plain.submit(controlled))
        await plain.shutdown()
        #expect(plainBackend.bytesReserved == 0)
        #expect(reference.finish == .length && reference.tokens.count == controlled.maxTokens)

        let (engine, backend) = Fallback.speculativeEngine(
            stack, kvDType: kvDType, sampler: CBv2DefaultSampler())
        let loop = engine.loopForTesting
        let carries = Fallback.CarryLog()
        Fallback.observeCarries(loop, into: carries)
        // Both rows join the first step and the text row is shorter, so all of
        // its speculation happens beside the controlled row.
        var text = Fallback.textRequest(1)
        text.maxTokens = Fallback.textMaxTokens / 2
        loop.onEngineQueueSync { loop.suspendStepExecutionAtCountForTesting = 0 }
        let controlledStream = try engine.submit(controlled)
        let textStream = try engine.submit(text)
        loop.onEngineQueueSync { loop.suspendStepExecutionAtCountForTesting = nil }
        let controlledID = controlled.id
        let textTask = Task {
            let collected = try await Batched.collect(textStream) { _ in }
            let loop = engine.loopForTesting
            var controlledRowLive = false
            loop.onEngineQueueSync {
                controlledRowLive = loop.scheduler.record(for: controlledID) != nil
            }
            let metrics = engine.mtpMetricsSnapshot()
            return (
                collected, metrics?.rounds ?? 0, metrics?.acceptedTokens ?? 0, controlledRowLive
            )
        }
        var drafted = false
        let served = try await Batched.collect(controlledStream) { _ in
            loop.onEngineQueueSync {
                guard let driver = loop.mtp, let record = loop.scheduler.record(for: controlledID)
                else { return }
                if driver.assistantStateCountsForTesting(controlledID) != nil
                    || driver.roundMark(for: controlledID) != nil
                    || driver.isSeedMarked(controlledID) || driver.hasValidCarry(for: record)
                {
                    drafted = true
                }
            }
        }
        let (textRun, rounds, accepted, controlledRowLive) = try await textTask.value
        loop.onEngineQueueSync {
            loop.mtp?.carryStoredObserverForTesting = nil
            #expect(loop.mtp?.requestStateCountForTesting == 0)
            #expect(loop.recurrentStates.isEmpty)
        }
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)

        #expect(textRun.tokens == Array(soloText[1].tokens.prefix(text.maxTokens)), "\(control)")
        #expect(textRun.finish == .length)
        #expect(served.tokens == reference.tokens && served.finish == .length, "\(control)")
        #expect(!drafted && !carries.stored.contains(controlledID))
        // The text row speculated while the controlled row was still decoding;
        // before, one ineligible row kept every row of the step target-only.
        #expect(controlledRowLive && rounds > 0 && accepted > 0, "\(control) rounds=\(rounds)")
        print(
            "SYNTHETIC super-mtp-controls \(control) roundsBesideRow=\(rounds) accepted=\(accepted)"
        )
    }
}
