import Foundation
import MLX
import MLXLMCommon
import Testing

@Suite("DiffusionGemma entropy-bound sampler", .serialized)
struct DiffusionGemmaSamplerTests {
    @Test func entropyMatchesIndependentAnalyticDistribution() throws {
        // P=(1/4,3/4); independent scalar entropy, not another MLX helper.
        let logits = MLXArray([Float(0), log(Float(3))]).reshaped(1, 1, 2)
        let value = try DiffusionGemmaSampling.tokenEntropy(logits).item(Float.self)
        let expected = -(0.25 * log(0.25) + 0.75 * log(0.75))
        #expect(abs(Double(value) - expected) < 1e-6)
        let masked = MLXArray([Float(0), -.infinity]).reshaped(1, 1, 2)
        #expect(try DiffusionGemmaSampling.tokenEntropy(masked).item(Float.self) == 0)
    }

    @Test func acceptanceUsesSumExcludingLargestAndRecomputes() throws {
        let first = MLXArray([Float(0.2), 0.01, 0.07, 0.8]).reshaped(1, 4)
        let mask = try DiffusionGemmaSampling.acceptanceMask(entropy: first, bound: 0.1)
        #expect(mask.asArray(Bool.self) == [true, true, true, false])
        let changed = MLXArray([Float(0.8), 0.01, 0.07, 0.2]).reshaped(1, 4)
        let next = try DiffusionGemmaSampling.acceptanceMask(entropy: changed, bound: 0.1)
        #expect(next.asArray(Bool.self) == [false, true, true, true])
        let tied = MLXArray.ones([2, 4])
        let count = try DiffusionGemmaSampling.acceptanceMask(entropy: tied, bound: 0.1)
            .asType(.int32).sum(axis: -1).asArray(Int32.self)
        #expect(count == [1, 1])
    }

    @Test func finishedRowsAreFrozenAndDraftIsNeverFinal() throws {
        let config = try DiffusionGemmaGenerationConfiguration(maxDenoisingSteps: 4)
        let initial = MLXArray([Int32(3), 3, 3, 3]).reshaped(2, 2)
        let state = try DiffusionGemmaDenoisingState(
            initialCanvas: initial, vocabularySize: 4, embeddingDType: .bfloat16,
            configuration: config)
        #expect(throws: DiffusionGemmaSamplingError.canvasNotFinal) { try state.finalizedCanvas() }
        // Row zero converges; row one stays uncertain and must keep running.
        let logits = MLXArray([
            Float(12), -12, -12, -12, -12, 12, -12, -12,
            0, 0, 0, 0, 0, 0, 0, 0,
        ]).reshaped(2, 2, 4)
        let draws = MLXArray([Int32(2), 3, 1, 2]).reshaped(2, 2)
        for _ in 0 ..< 2 {
            try state.step(rawLogits: logits, sample: { _ in draws }, noise: { initial })
        }
        #expect(state.finishedRows.asArray(Bool.self) == [true, false])
        #expect(state.argmaxCanvas[0].asArray(Int32.self) == [0, 1])
        #expect(state.currentCanvas[0].asArray(Int32.self) == [2, 3])
        #expect(state.selfConditioningLogits?.dtype == .bfloat16)
        let frozenConditioning = state.selfConditioningLogits![0].asType(.float32).asArray(
            Float.self)
        #expect(throws: DiffusionGemmaSamplingError.canvasNotFinal) { try state.finalizedCanvas() }
        for _ in 0 ..< 2 {
            try state.step(rawLogits: -logits, sample: { _ in initial }, noise: { draws })
        }
        #expect(state.stepsUsed.asArray(Int32.self) == [2, 4])
        #expect(state.currentCanvas[0].asArray(Int32.self) == [2, 3])
        #expect(
            state.selfConditioningLogits![0].asType(.float32).asArray(Float.self)
                == frozenConditioning)
        #expect(try state.finalizedCanvas()[0].asArray(Int32.self) == [0, 1])
        #expect(throws: DiffusionGemmaSamplingError.exhaustedCanvas) {
            try state.step(rawLogits: logits, sample: { _ in draws }, noise: { initial })
        }
    }

    @Test func failuresDoNotAdvanceTheCanvas() throws {
        let initial = MLXArray([Int32(0), 1]).reshaped(1, 2)
        let state = try DiffusionGemmaDenoisingState(
            initialCanvas: initial, vocabularySize: 3, embeddingDType: .float32,
            configuration: .init(maxDenoisingSteps: 2))
        let invalid = MLXArray.full([1, 2, 3], values: MLXArray(-Float.infinity))
        #expect(throws: DiffusionGemmaSamplingError.invalidDistribution) {
            try state.step(rawLogits: invalid, sample: { _ in initial }, noise: { initial })
        }
        #expect(throws: DiffusionGemmaSamplingError.invalidToken) {
            try state.step(
                rawLogits: .zeros([1, 2, 3]),
                sample: { _ in MLXArray([Int32(0), 3]).reshaped(1, 2) }, noise: { initial })
        }
        #expect(state.remainingSteps == 2)
        #expect(state.currentCanvas.asArray(Int32.self) == [0, 1])
        #expect(state.stepsUsed.asArray(Int32.self) == [0])
        #expect(state.selfConditioningLogits == nil)
    }

    @Test func zeroStabilityCanFinishFirstConfidentStepAndStatesAreIndependent() throws {
        let config = try DiffusionGemmaGenerationConfiguration(
            maxDenoisingSteps: 2, stabilityThreshold: 0)
        let first = try DiffusionGemmaDenoisingState(
            initialCanvas: MLXArray([Int32(0)]).reshaped(1, 1), vocabularySize: 2,
            embeddingDType: .float32, configuration: config)
        let untouched = try DiffusionGemmaDenoisingState(
            initialCanvas: MLXArray([Int32(1)]).reshaped(1, 1), vocabularySize: 2,
            embeddingDType: .float32, configuration: config)
        let logits = MLXArray([Float(12), -12]).reshaped(1, 1, 2)
        try first.step(
            rawLogits: logits,
            sample: { processed in
                #expect(processed.asArray(Float.self) == [15, -15])
                return MLXArray([Int32(0)]).reshaped(1, 1)
            }, noise: { MLXArray([Int32(1)]).reshaped(1, 1) })
        #expect(try first.finalizedCanvas().item(Int32.self) == 0)
        #expect(untouched.remainingSteps == 2 && untouched.selfConditioningLogits == nil)
        #expect(throws: DiffusionGemmaSamplingError.canvasNotFinal) {
            try untouched.finalizedCanvas()
        }
    }
}
