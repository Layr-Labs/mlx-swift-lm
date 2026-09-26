import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

@Suite("DiffusionGemma sorted expert reduction", .serialized)
struct DiffusionGemmaExpertReductionTests {
    private func fixture(
        tokens: Int = 8, hidden: Int = 2816, topK: Int = 8,
        dtype: DType = .bfloat16
    ) -> (MLXArray, MLXArray, MLXArray) {
        let count = tokens * topK
        let values: [Float] = (0 ..< (count * hidden)).map { index in
            let centered: Int = (index * 37 + 17) % 1009 - 504
            return Float(centered) / 137
        }
        let rows = MLXArray(values).reshaped(count, 1, hidden).asType(dtype)
        let inverse = MLXArray((0 ..< count).reversed().map(UInt32.init))
        let weights = softmax(
            MLXArray((0 ..< count).map { Float(($0 * 13) % 17) / 8 })
                .reshaped(tokens, topK), axis: -1, precise: true
        ).asType(dtype)
        eval(rows, inverse, weights)
        return (rows, inverse, weights)
    }

    private func legacy(_ rows: MLXArray, inverse: MLXArray?, weights: MLXArray) -> MLXArray {
        let restored =
            inverse.map { scatterUnsort(x: rows, invOrder: $0, shape: weights.shape) } ?? rows
        return (restored.squeezed(axis: -2) * weights.expandedDimensions(axis: -1)).sum(axis: -2)
    }

    @Test func defaultAndRollbackAreExplicit() {
        #expect(DiffusionGemmaExpertReduction.enabled(from: nil))
        for value in ["1", "TRUE", "yes", "on"] {
            #expect(DiffusionGemmaExpertReduction.enabled(from: value))
        }
        for value in ["0", "false", "no", "off", "", "unexpected"] {
            #expect(!DiffusionGemmaExpertReduction.enabled(from: value))
        }
    }

    @Test func shaderIndexRangeIsCheckedWithoutLargeAllocations() {
        let safeAssignments = Int(UInt32.max) / 2816
        #expect(
            DiffusionGemmaExpertReduction.shaderIndexFits(
                assignmentCount: safeAssignments, hidden: 2816))
        #expect(
            !DiffusionGemmaExpertReduction.shaderIndexFits(
                assignmentCount: safeAssignments + 1, hidden: 2816))
        #expect(
            DiffusionGemmaExpertReduction.shaderIndexFits(
                assignmentCount: 1024 * 8, hidden: 2816))
        #expect(
            !DiffusionGemmaExpertReduction.shaderIndexFits(
                assignmentCount: 262_143 * 8, hidden: 2816))
        #expect(
            !DiffusionGemmaExpertReduction.shaderIndexFits(
                assignmentCount: Int.max, hidden: 2))
        #expect(!DiffusionGemmaExpertReduction.shaderIndexFits(assignmentCount: 0, hidden: 2816))
        #expect(!DiffusionGemmaExpertReduction.shaderIndexFits(assignmentCount: -1, hidden: 2816))
        #expect(!DiffusionGemmaExpertReduction.shaderIndexFits(assignmentCount: 64, hidden: 0))
    }

    @Test(arguments: [8, 9, 31, 32, 33, 63, 64, 255, 256, 257, 511, 512, 513, 1024])
    func nativeEligibleShapesAreExact(_ tokens: Int) {
        let (rows, inverse, weights) = fixture(tokens: tokens)
        #expect(
            DiffusionGemmaExpertReduction.eligible(
                rows, inverse: inverse, weights: weights, enabled: true))
        resetWeightedExpertUnsortStats()
        let result = DiffusionGemmaExpertReduction.reduce(
            rows, inverse: inverse,
            indicesShape: weights.shape, weights: weights, enabled: true)
        let expected = legacy(rows, inverse: inverse, weights: weights)
        eval(result, expected)
        #expect(result.dtype == .bfloat16 && result.shape == expected.shape)
        #expect(
            result.asArray(Float.self).map(\.bitPattern)
                == expected.asArray(Float.self).map(\.bitPattern))
        #expect(weightedExpertUnsortStats().effectiveCalls == 1)
    }

    @Test func rollbackAndNonNativeGeometriesKeepLegacyGraph() {
        for (tokens, hidden, topK, dtype) in [
            (7, 2816, 8, DType.bfloat16), (8, 128, 8, .bfloat16),
            (8, 2816, 10, .bfloat16), (8, 2816, 8, .float32), (8, 2816, 8, .float16),
        ] {
            let (rows, inverse, weights) = fixture(
                tokens: tokens, hidden: hidden, topK: topK, dtype: dtype)
            #expect(
                !DiffusionGemmaExpertReduction.eligible(
                    rows, inverse: inverse, weights: weights, enabled: true))
            resetWeightedExpertUnsortStats()
            let result = DiffusionGemmaExpertReduction.reduce(
                rows, inverse: inverse,
                indicesShape: weights.shape, weights: weights, enabled: true)
            let expected = legacy(rows, inverse: inverse, weights: weights)
            eval(result, expected)
            #expect(
                result.asArray(Float.self).map(\.bitPattern)
                    == expected.asArray(Float.self).map(\.bitPattern))
            #expect(weightedExpertUnsortStats().effectiveCalls == 0)
        }
        let (rows, inverse, weights) = fixture()
        #expect(
            !DiffusionGemmaExpertReduction.eligible(
                rows, inverse: inverse, weights: weights, enabled: false))
        resetWeightedExpertUnsortStats()
        let actual = DiffusionGemmaExpertReduction.reduce(
            rows, inverse: inverse,
            indicesShape: weights.shape, weights: weights, enabled: false)
        let expected = legacy(rows, inverse: inverse, weights: weights)
        eval(actual, expected)
        #expect(
            actual.asArray(Float.self).map(\.bitPattern)
                == expected.asArray(Float.self).map(\.bitPattern))
        #expect(weightedExpertUnsortStats().effectiveCalls == 0)
    }

    @Test func missingInverseDoesNotSelectSortedKernel() {
        let (rows, _, weights) = fixture()
        let unsorted = rows.reshaped(8, 8, 1, 2816)
        #expect(
            !DiffusionGemmaExpertReduction.eligible(
                unsorted, inverse: nil, weights: weights, enabled: true))
        resetWeightedExpertUnsortStats()
        let result = DiffusionGemmaExpertReduction.reduce(
            unsorted, inverse: nil,
            indicesShape: weights.shape, weights: weights, enabled: true)
        let expected = legacy(unsorted, inverse: nil, weights: weights)
        eval(result, expected)
        #expect(
            result.asArray(Float.self).map(\.bitPattern)
                == expected.asArray(Float.self).map(\.bitPattern))
        #expect(weightedExpertUnsortStats().effectiveCalls == 0)
    }

    @Test func trainingPreservesLegacyAndInferenceSelectsCandidate() throws {
        let url = try #require(
            Bundle.module.url(forResource: "diffusiongemma-block-oracle", withExtension: "json"))
        let root = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        var values = try #require(root["model_config"] as? [String: Any])
        values["hidden_size"] = 2816
        values["num_experts"] = 8
        values["top_k_experts"] = 8
        let config = try JSONDecoder().decode(
            DiffusionGemmaTextConfiguration.self,
            from: JSONSerialization.data(withJSONObject: values))
        let experts = DiffusionGemmaExperts(config)
        experts.apply { $0.asType(.bfloat16) }
        let input = MLXArray((0 ..< (8 * 2816)).map { Float($0 % 101 - 50) / 101 })
            .reshaped(8, 2816).asType(.bfloat16)
        let indices = MLXArray((0 ..< 64).map { UInt32($0 % 8) }).reshaped(8, 8)
        let weights = MLXArray.full([8, 8], values: MLXArray(Float(0.125)), dtype: .bfloat16)
        eval(experts)
        eval(input, indices, weights)
        #expect(experts.training)
        resetWeightedExpertUnsortStats()
        let training = experts(input, indices: indices, weights: weights)
        eval(training)
        #expect(weightedExpertUnsortStats().effectiveCalls == 0)
        experts.train(false)
        resetWeightedExpertUnsortStats()
        let inference = experts(input, indices: indices, weights: weights)
        eval(inference)
        #expect(weightedExpertUnsortStats().effectiveCalls == 1)
        #expect(
            training.asArray(Float.self).map(\.bitPattern)
                == inference.asArray(Float.self).map(\.bitPattern))
        experts.train(true)
        resetWeightedExpertUnsortStats()
        let returnedToTraining = experts(input, indices: indices, weights: weights)
        eval(returnedToTraining)
        #expect(weightedExpertUnsortStats().effectiveCalls == 0)
        #expect(
            training.asArray(Float.self).map(\.bitPattern)
                == returnedToTraining.asArray(Float.self).map(\.bitPattern))
    }

    @Test func cpuDefaultDeclinesKernel() {
        let (bfRows, bfInverse, bfWeights) = fixture()
        Device.withDefaultDevice(.cpu) {
            #expect(
                !DiffusionGemmaExpertReduction.eligible(
                    bfRows, inverse: bfInverse, weights: bfWeights, enabled: true))
            let (rows, inverse, weights) = fixture(dtype: .float32)
            #expect(
                !DiffusionGemmaExpertReduction.eligible(
                    rows, inverse: inverse, weights: weights, enabled: true))
            let actual = DiffusionGemmaExpertReduction.reduce(
                rows, inverse: inverse,
                indicesShape: weights.shape, weights: weights, enabled: true)
            let expected = legacy(rows, inverse: inverse, weights: weights)
            eval(actual, expected)
            #expect(
                actual.asArray(Float.self).map(\.bitPattern)
                    == expected.asArray(Float.self).map(\.bitPattern))
        }
    }

    @Test func scopedCPUAndCustomGPUStreamsDeclineKernel() {
        let (rows, inverse, weights) = fixture()
        Device.withDefaultDevice(.gpu) {
            Stream.withNewDefaultStream(device: .cpu) {
                #expect(
                    !DiffusionGemmaExpertReduction.eligible(
                        rows, inverse: inverse, weights: weights, enabled: true))
            }
            Stream.withNewDefaultStream(device: .gpu) {
                #expect(
                    !DiffusionGemmaExpertReduction.eligible(
                        rows, inverse: inverse, weights: weights, enabled: true))
                resetWeightedExpertUnsortStats()
                let actual = DiffusionGemmaExpertReduction.reduce(
                    rows, inverse: inverse,
                    indicesShape: weights.shape, weights: weights, enabled: true)
                let expected = legacy(rows, inverse: inverse, weights: weights)
                eval(actual, expected)
                #expect(
                    actual.asArray(Float.self).map(\.bitPattern)
                        == expected.asArray(Float.self).map(\.bitPattern))
                #expect(weightedExpertUnsortStats().effectiveCalls == 0)
            }
        }
    }
}
