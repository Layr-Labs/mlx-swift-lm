import MLX
import Testing

@testable import MLXVLM

/// Expected selections from vLLM evs.py `compute_retention_mask` (PyTorch 2.12).
@Suite(.tags(.integration))
struct NemotronSuperEVSTests {
    /// Three tubelets of four 2-wide tokens. Against their predecessors the later
    /// tubelets score [0, 1, 0, 0.04] and [2, 0, 1, 0.4]: equal unit vectors,
    /// orthogonal ones, an opposite one and two partial turns.
    private func features(_ dtype: DType = .float32) -> MLXArray {
        MLXArray([
            Float(1), 0, 1, 0, 0, 1, 3, 4,
            1, 0, 0, 1, 0, 1, 4, 3,
            -1, 0, 0, 1, 1, 0, 0, 5,
        ]).reshaped(3, 4, 2).asType(dtype)
    }

    @Test func scoresAreOneMinusCosineAgainstThePreviousTubelet() {
        let scores = NemotronSuperEVS.dissimilarity(features()).asArray(Float.self)
        let expected: [Float] = [0, 1, 0, 0.04, 2, 0, 1, 0.4]
        #expect(scores.count == expected.count)
        for (actual, reference) in zip(scores, expected) {
            #expect(abs(actual - reference) < 0.00001)
        }
    }

    @Test func keptPositionsMatchTheReferenceSelection() {
        let cases: [(rate: Double, kept: [[Int]])] = [
            (0.5, [[0, 1, 2, 3], [1], [0]]),
            (0.4, [[0, 1, 2, 3], [1], [0, 2]]),
            (0.25, [[0, 1, 2, 3], [1, 3], [0, 2, 3]]),
            (0.1, [[0, 1, 2, 3], [0, 1, 3], [0, 2, 3]]),
            (0.0001, [[0, 1, 2, 3], [0, 1, 2, 3], [0, 2, 3]]),
            (1, [[0, 1, 2, 3], [], []]),
            (0, [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]),
        ]
        for (rate, expected) in cases {
            #expect(
                NemotronSuperEVS.keptPositions(features: features(), pruningRate: rate) == expected,
                "rate \(rate)")
        }
        // The reference makes the same BF16 selection.
        #expect(
            NemotronSuperEVS.keptPositions(features: features(.bfloat16), pruningRate: 0.25) == [
                [0, 1, 2, 3], [1, 3], [0, 2, 3],
            ])
    }

    @Test func zeroTokensUseTheClampedNormAndInvalidShapesKeepNothing() {
        // Zero rows have cosine 0, not NaN: every later token scores 1 and the
        // earliest one is kept.
        #expect(
            NemotronSuperEVS.keptPositions(features: MLXArray.zeros([2, 3, 2]), pruningRate: 0.25)
                == [[0, 1, 2], [0]])
        #expect(
            NemotronSuperEVS.keptPositions(features: MLXArray.zeros([1, 3, 2]), pruningRate: 0.5)
                == [[0, 1, 2]])
        #expect(
            NemotronSuperEVS.keptPositions(features: MLXArray.zeros([3, 2]), pruningRate: 0.5)
                .isEmpty)
    }
}
