// Copyright © 2026 Eigen Labs.
import MLX
import Testing
@testable import MLXLMCommon

/// Native GPU tests authored only. Scores, not arbitrary expert IDs, are the
/// specialized producer's input. No checkpoint is needed to test its outputs.
@Suite("Gemma4 bounded prefill route sort", .serialized)
struct Gemma4RouteSortTests {
    private func context(bitset: Bool, parallel: Bool) -> Gemma4PrefillGluePolicy.Context {
        Gemma4PrefillGluePolicy(environment: ["DARKBLOOM_GEMMA4_PREFILL_GLUE": "1",
            "DARKBLOOM_ROUTE_CSORT_PREFILL": "1", "DARKBLOOM_ROUTE_CSORT_PREFILL_BITSET": bitset ? "1" : "0",
            "DARKBLOOM_GEMMA4_PROMPT_GLUE2": parallel ? "1" : "0"])
            .context(scheduledPrefill: true, eligibleModel: true)!
    }

    private func identical(_ lhs: MLXArray, _ rhs: MLXArray) {
        #expect(lhs.shape == rhs.shape && lhs.dtype == rhs.dtype)
        #expect(lhs.asData(access: .copy).data == rhs.asData(access: .copy).data)
    }

    @Test func allSortOutputsAndRouterWeightsMatchOriginal() throws {
        for bitset in [false, true] {
            for parallel in [false, true] {
                for rows in [9, 31, 33, 512, 1024, 1025, 8192] {
                    for tied in [false, true] {
                        let scoreValues = (0..<(rows * 128)).map { Float(tied ? 0 : ($0 * 17 + $0 / 128) % 257) }
                        let scores = MLXArray(scoreValues, [1, rows, 128]).asType(.bfloat16)
                        let scale = MLXArray((0..<128).map { Float($0 % 7 + 1) / 8 }).asType(.bfloat16)
                        let ctx = context(bitset: bitset, parallel: parallel)
                        let candidate = try #require(Gemma4PrefillRouting.make(scores: scores, perExpertScale: scale, context: ctx))
                        var indices = argPartition(scores, kth: 120, axis: -1)
                        indices = indices[.ellipsis, 120...]
                        var weights = takeAlong(scores, indices, axis: -1)
                        weights = softmax(weights, axis: -1, precise: true)
                        weights = weights * scale[indices]
                        let flat = indices.flattened(), order = argSort(flat), inverse = argSort(order)
                        identical(candidate.indices, indices)
                        identical(candidate.weights, weights)
                        identical(candidate.rowOrder, order.floorDivide(8))
                        identical(candidate.sortedIndices, flat[order])
                        identical(candidate.inverseOrder, inverse)
                        let owned = Gemma4PrefillExpertOrder.fromRouting(candidate)
                        let x = MLXArray((0..<(rows * 2816)).map { Float($0 % 17) }, [rows, 2816]).asType(.bfloat16)
                        identical(owned.gatherNormalized(x), x.reshaped(rows, 1, 2816)[order.floorDivide(8)])
                    }
                }
            }
        }
    }

    @Test func publicWrapperUpdatesCannotRewriteBoundedInputs() throws {
        let scores = MLXArray.zeros([1, 9, 128], dtype: .bfloat16)
        let route = try #require(Gemma4PrefillRouting.make(scores: scores,
            perExpertScale: MLXArray.ones([128], dtype: .bfloat16), context: context(bitset: true, parallel: true)))
        let original = route.flatIndices.asArray(UInt32.self)
        route.indices._updateInternal(MLXArray(Array(repeating: UInt32.max, count: 72), [1, 9, 8]))
        route.weights._updateInternal(MLXArray.zeros([1, 9, 8], dtype: .bfloat16))
        #expect(route.flatIndices.asArray(UInt32.self) == original)
        #expect(route.sortedIndices.asArray(UInt32.self).allSatisfy { $0 < 128 })
        #expect(route.flatWeights.asArray(Float.self).contains { $0 != 0 })
    }

    @Test func invalidScoreDomainAndDecodeShapeDecline() {
        let ctx = context(bitset: true, parallel: true)
        for shape in [[1, 1, 128], [1, 8, 128], [1, 1024, 127], [1, 1024, 129]] {
            #expect(Gemma4PrefillRouting.make(scores: MLXArray.zeros(shape),
                perExpertScale: MLXArray.ones([128]), context: ctx) == nil)
        }
    }
}
