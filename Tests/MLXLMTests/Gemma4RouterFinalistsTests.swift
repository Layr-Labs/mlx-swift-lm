// Copyright © 2026 Eigen Labs.
import MLX
import Testing
@testable import MLXLMCommon

/// Native tests authored only. CPU ordering models do not prove Metal FTZ,
/// softmax, NaN payload or subgroup behavior; these cases gate promotion.
@Suite("Gemma4 router finalists", .serialized)
struct Gemma4RouterFinalistsTests {
    private func policy(weights: Bool, keys: Bool) -> Gemma4RouterFinalistsPolicy {
        Gemma4RouterFinalistsPolicy(environment: ["DARKBLOOM_GEMMA4_ROUTER_FINALISTS32": "1",
            "DARKBLOOM_GEMMA4_ROUTER_FINALISTS32_PREFILL": "1",
            "DARKBLOOM_GEMMA4_ROUTER_WEIGHTS32_PREFILL": weights ? "1" : "0",
            "DARKBLOOM_GEMMA4_PREFILL_ROUTE_ORDER_KEYS": keys ? "1" : "0"])
    }
    private func identical(_ actual: MLXArray, _ expected: MLXArray) {
        #expect(actual.shape == expected.shape && actual.dtype == expected.dtype)
        #expect(actual.asData(access: .copy).data == expected.asData(access: .copy).data)
    }
    private func reference(_ scores: MLXArray, scale: MLXArray) -> (MLXArray, MLXArray) {
        var indices = argPartition(scores, kth: 120, axis: -1)
        indices = indices[.ellipsis, 120...]
        let weights = softmax(takeAlong(scores, indices, axis: -1), axis: -1, precise: true) * scale[indices]
        return (indices, weights)
    }

    @Test func variantsMatchStableSelectionAndOriginalWeights() throws {
        let special: [UInt16] = [0, 0x8000, 1, 0x7f, 0x80, 0x8001, 0x807f, 0x8080,
                                 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc0, 0xffc0]
        for shape in [[1, 1, 128], [8, 1, 128], [1, 17, 128], [8, 1024, 128]] {
            for useSpecial in [false, true] {
                let count = shape.reduce(1, *)
                let scores = useSpecial
                    ? MLXArray((0..<count).map { special[$0 % special.count] }, shape).view(dtype: .bfloat16)
                    : MLXArray((0..<count).map { Float(($0 * 17 + $0 / 128) % 257 - 128) / 16 }, shape).asType(.bfloat16)
                let scale = MLXArray((0..<128).map { Float($0 % 11 + 1) / 16 }).asType(.bfloat16)
                let expected = reference(scores, scale: scale)
                for weights in [false, true] {
                    for keys in [false, true] {
                        let plan = try #require(policy(weights: weights, keys: keys).plan(targetEligible: true,
                            shape: shape, scoresBF16: true, scaleBF16: true, scaleShape: [128],
                            topK: 8, scheduledPrefill: shape[1] > 1))
                        let actual = try #require(Gemma4RouterFinalistsV1.apply(scores: scores, perExpertScale: scale, plan: plan))
                        identical(actual.indices, expected.0)
                        identical(actual.weights, expected.1)
                    }
                }
            }
        }
    }

    @Test func finalistOutputsRemainCompatibleWithBoundedCountingSort() throws {
        let scores = MLXArray((0..<(1024 * 128)).map { Float($0 % 37) }, [1, 1024, 128]).asType(.bfloat16)
        let scale = MLXArray.ones([128], dtype: .bfloat16)
        let plan = try #require(policy(weights: true, keys: true).plan(targetEligible: true,
            shape: scores.shape, scoresBF16: true, scaleBF16: true, scaleShape: [128], topK: 8, scheduledPrefill: true))
        let context = Gemma4PrefillGluePolicy(environment: ["DARKBLOOM_GEMMA4_PREFILL_GLUE": "1",
            "DARKBLOOM_ROUTE_CSORT_PREFILL": "1"]).context(scheduledPrefill: true, eligibleModel: true)!
        let routes = try #require(Gemma4PrefillRouting.make(scores: scores, perExpertScale: scale,
            context: context, finalists: plan))
        let expected = reference(scores, scale: scale)
        identical(routes.indices, expected.0)
        identical(routes.weights, expected.1)
        let flat = expected.0.flattened(), order = argSort(flat)
        identical(routes.sortedIndices, flat[order])
        identical(routes.inverseOrder, argSort(order))
    }
}
