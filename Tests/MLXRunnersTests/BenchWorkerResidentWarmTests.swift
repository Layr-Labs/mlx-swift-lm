import Foundation
import MLXLMCommon
import Testing

@testable import MLXRunners

/// A stepper that records what the warm pass drove through it.
private final class RecordingStepper: TeacherForcedStepper {
    private let inner = MockStepper()
    private(set) var begins = 0
    private(set) var forwardedCounts: [Int] = []
    private(set) var allTokens: [Int] = []
    var forwards: Int { inner.forwards }

    func begin() throws {
        begins += 1
        try inner.begin()
    }

    func forward(_ tokens: [Int]) throws -> StepOutput {
        forwardedCounts.append(tokens.count)
        allTokens += tokens
        return try inner.forward(tokens)
    }
}

private final class RecordingRunner: Runner, @unchecked Sendable {
    static let manifest = MockRunner.manifest
    private let inner = MockRunner()
    let stepper = RecordingStepper()
    private(set) var steppersMade = 0

    var servingModel: any LanguageModel { inner.servingModel }
    var tokenizer: any MLXLMCommon.Tokenizer { inner.tokenizer }
    var eosTokenIDs: Set<Int> { inner.eosTokenIDs }
    var layerKinds: [CBv2LayerKind] { inner.layerKinds }
    var loadedDecoders: [DecoderID] { inner.loadedDecoders }
    var headProvenance: HeadProvenance? { inner.headProvenance }
    var loadedModelType: String { inner.loadedModelType }

    static func adopt(
        model: any LanguageModel,
        tokenizer: any MLXLMCommon.Tokenizer,
        configuration: ModelConfiguration,
        directory: URL,
        options: RunnerLoadOptions
    ) throws -> RecordingRunner {
        RecordingRunner()
    }

    func makeEngine(_ build: EngineBuild) throws -> any CBv2Engine { try inner.makeEngine(build) }
    func makeStepper() throws -> any TeacherForcedStepper {
        steppersMade += 1
        return stepper
    }
}

@Suite("Resident warm pass")
struct BenchWorkerResidentWarmTests {
    @Test("one session: a fixed prefill, then fixed single-token steps")
    func fixedShape() throws {
        let runner = RecordingRunner()
        let seconds = try BenchWorkerResidentWarm.run(runner: runner, memory: FixtureMemoryReporter())
        #expect(seconds >= 0)
        #expect(runner.steppersMade == 1)
        #expect(runner.stepper.begins == 1)
        #expect(
            runner.stepper.forwardedCounts
                == [BenchWorkerResidentWarm.prefillTokens]
                + Array(repeating: 1, count: BenchWorkerResidentWarm.decodeSteps))
        // The prefill tokens are ours; the step tokens are whatever the
        // stepper's argmax returned.
        let prefill = runner.stepper.allTokens.prefix(BenchWorkerResidentWarm.prefillTokens)
        #expect(prefill.allSatisfy { $0 >= 1 && $0 < 1000 })
    }

    @Test("the pass is the same every time: same tokens, same count")
    func deterministic() {
        #expect(BenchWorkerResidentWarm.tokens(count: 16) == BenchWorkerResidentWarm.tokens(count: 16))
        #expect(BenchWorkerResidentWarm.tokens(count: 1024).count == 1024)
    }
}
