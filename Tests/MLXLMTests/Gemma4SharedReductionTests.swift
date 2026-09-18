import MLX
import Testing

@testable import MLXLMCommon

/// The Gemma extraction must retain the current shared primitive's variable K.
/// Small integer inputs make the host reference exact in both FP32 and BF16;
/// arithmetic-boundary coverage remains in the existing numerical suites.
@Suite("Gemma integration preserves shared weighted reduction", .serialized)
struct Gemma4SharedReductionTests {
    private func check(tokens: Int, topK: Int, hidden: Int, explicitStream: Bool) {
        let assignments = tokens * topK
        let values = (0..<(assignments * hidden)).map { Float($0 % 5 - 2) }
        let order = Array((0..<assignments).reversed())
        let scores = (0..<assignments).map { Float($0 % 2 + 1) }
        let outputs = MLXArray(values, [assignments, hidden]).asType(.bfloat16)
        let inverse = MLXArray(order.map(UInt32.init))
        let weights = MLXArray(scores, [tokens, topK]).asType(.bfloat16)
        var reference = [Float](repeating: 0, count: tokens * hidden)
        for token in 0..<tokens {
            for feature in 0..<hidden {
                for slot in 0..<topK {
                    let assignment = token * topK + slot
                    reference[token * hidden + feature] +=
                        values[order[assignment] * hidden + feature] * scores[assignment]
                }
            }
        }
        let actual = explicitStream
            ? weightedExpertUnsortOnStream(sortedOutputs: outputs, inverseOrder: inverse,
                                          weights: weights, stream: .default)
            : weightedExpertUnsort(sortedOutputs: outputs, inverseOrder: inverse, weights: weights)
        let expected = MLXArray(reference, [tokens, hidden]).asType(.bfloat16)
        eval(actual, expected)
        #expect(actual.shape == expected.shape)
        #expect(actual.dtype == .bfloat16)
        #expect(actual.asData(access: .copy).data == expected.asData(access: .copy).data)
    }

    @Test func publicPrimitiveRetainsCurrentSharedTopKContract() {
        check(tokens: 64, topK: 1, hidden: 64, explicitStream: false)
        check(tokens: 8, topK: 8, hidden: 2816, explicitStream: false)
        check(tokens: 8, topK: 10, hidden: 2560, explicitStream: false)
    }

    @Test func explicitStreamTwinRetainsCurrentSharedTopKContract() {
        check(tokens: 8, topK: 8, hidden: 2816, explicitStream: true)
        check(tokens: 8, topK: 10, hidden: 2560, explicitStream: true)
    }
}
