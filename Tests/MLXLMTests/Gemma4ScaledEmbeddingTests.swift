// Copyright © 2026 Eigen Labs.
import MLX
import MLXNN
import Testing
@testable import MLXLMCommon

/// Native-only tests, authored but not executed in the source-only phase.
@Suite("Gemma4 scaled Q4 embedding", .serialized)
struct Gemma4ScaledEmbeddingTests {
    @Test func sameQuantizationAndNegativeWrapProduceExactBytes() throws {
        let count = 32 * 2816
        let values: [Float] = (0..<count).map { index in
            let code = (index * 17) % 113 - 56
            return Float(code) / Float(64)
        }
        let raw = MLXArray(values, [32, 2816]).asType(.bfloat16)
        let embedding = QuantizedEmbedding(weight: raw, groupSize: 64, bits: 4)
        let policy = Gemma4ScaledEmbeddingPolicy(environment: ["DARKBLOOM_GEMMA4_SCALED_EMBEDDING": "1",
            "DARKBLOOM_GEMMA4_SCALED_EMBEDDING_DECODE": "1"])
        let scale = Float(2816).squareRoot()
        for shape in [[1, 1], [1, 8], [4, 1], [8, 1024]] {
            let ids: [Int32] = [-32, -1, 0, 1, 16, 31]
            let tokens = MLXArray((0..<shape.reduce(1, *)).map { ids[$0 % ids.count] }, shape)
            let actual = try #require(Gemma4ScaledEmbeddingV1.apply(tokens: tokens, embedding: embedding,
                embedScale: scale, hidden: 2816, targetEligible: true, policy: policy))
            let expected = embedding(tokens) * scale
            #expect(actual.shape == expected.shape && actual.dtype == expected.dtype)
            #expect(actual.asData(access: .copy).data == expected.asData(access: .copy).data)
        }
    }
    @Test func unsupportedEmbeddingsAndPoliciesFallBack() {
        let plain = Embedding(embeddingCount: 32, dimensions: 2816)
        let tokens = MLXArray([Int32(0), 1], [1, 2])
        let policy = Gemma4ScaledEmbeddingPolicy(environment: ["DARKBLOOM_GEMMA4_SCALED_EMBEDDING": "1"])
        #expect(Gemma4ScaledEmbeddingV1.apply(tokens: tokens, embedding: plain,
            embedScale: 1, hidden: 2816, targetEligible: true, policy: policy) == nil)
        let eight = QuantizedEmbedding(weight: MLXArray.zeros([32, 2816], dtype: .bfloat16), bits: 8)
        #expect(Gemma4ScaledEmbeddingV1.apply(tokens: tokens, embedding: eight,
            embedScale: 1, hidden: 2816, targetEligible: true, policy: policy) == nil)
    }
}
