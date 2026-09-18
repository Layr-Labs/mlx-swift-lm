// Copyright © 2026 Eigen Labs.
import MLX
import Testing

@testable import MLXLMCommon

/// Native tests authored only. The host harness never imports or runs this suite.
@Suite("Gemma4 B8 bounded routing", .serialized)
struct Gemma4B8RouteTests {
    private func policy(prefix: Bool, direct: Bool = true, fold: Bool = false,
                        native: Bool = true) -> Gemma4B8RoutePolicy {
        .init(environment: ["DARKBLOOM_GEMMA4_B8_EXPERT_EXECUTION": "1",
                            "DARKBLOOM_GEMMA4_B8_ROUTE_RANK": "1",
                            "DARKBLOOM_GEMMA4_B8_ROUTE_PREFIX": prefix ? "1" : "0",
                            "DARKBLOOM_GEMMA4_B8_ROUTE_DIRECT": direct ? "1" : "0",
                            "DARKBLOOM_GEMMA4_B8_ROUTE_FOLD": fold ? "1" : "0",
                            "DARKBLOOM_GEMMA4_B8_ROUTE_ORDER_KEYS": native ? "1" : "0"])
    }

    private func scores() -> MLXArray {
        MLXArray((0..<1024).map { Float(($0 * 37) % 31 - 15) / 8 }, [8, 1, 128]).asType(.bfloat16)
    }

    private func same(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.dtype == b.dtype && a.shape == b.shape && a.asData(access: .copy).data == b.asData(access: .copy).data
    }

    @Test func rankedAndTaggedRoutesMatchOrdinarySort() throws {
        let input = scores()
        let scale = MLXArray.ones([128], dtype: .bfloat16)
        let expected = try #require(Gemma4B8ExpertRouting.make(scores: input, perExpertScale: scale,
                                                             policy: .init(environment: [:])))
        for prefix in [false, true] {
            for direct in [false, true] {
                let actual = try #require(Gemma4B8ExpertRouting.make(scores: input, perExpertScale: scale,
                    policy: policy(prefix: prefix, direct: direct)))
                #expect(same(actual.rowOrder, expected.rowOrder))
                #expect(same(actual.sortedKeys, expected.sortedKeys))
                #expect(same(actual.inverseOrder, expected.inverseOrder))
                #expect(same(actual.reductionWeights, expected.reductionWeights))
                #expect(actual.usesPrefixBounds == prefix)
                let preserved = actual.executionKeys.asData(access: .copy).data
                actual.indices._updateInternal(MLXArray.zeros([8, 1, 8], dtype: .uint32))
                actual.weights._updateInternal(MLXArray.zeros([8, 1, 8], dtype: .bfloat16))
                #expect(actual.executionKeys.asData(access: .copy).data == preserved)
                #expect(same(actual.reductionWeights, expected.reductionWeights))
            }
        }
    }

    @Test func foldedSelectionAndWeightTapeMatchOriginalChain() throws {
        let scale = (MLXArray(0..<128).asType(.float32) / 64 + 0.5).asType(.bfloat16)
        let selection = Gemma4RouterFinalistsPolicy(environment: ["DARKBLOOM_GEMMA4_ROUTER_FINALISTS32": "1"])
        let plan = try #require(selection.plan(targetEligible: true, shape: [8, 1, 128], scoresBF16: true,
            scaleBF16: true, scaleShape: [128], topK: 8, scheduledPrefill: false))
        for input in [scores(), MLXArray.zeros([8, 1, 128], dtype: .bfloat16)] {
            let expected = try #require(Gemma4B8ExpertRouting.make(scores: input, perExpertScale: scale,
                finalists: plan, policy: .init(environment: [:])))
            for native in [false, true] {
                for prefix in [false, true] {
                    let actual = try #require(Gemma4B8ExpertRouting.make(scores: input, perExpertScale: scale,
                        finalists: plan, policy: policy(prefix: prefix, fold: true, native: native)))
                    #expect(same(actual.indices, expected.indices))
                    #expect(same(actual.weights, expected.weights))
                    #expect(same(actual.rowOrder, expected.rowOrder))
                    #expect(same(actual.sortedKeys, expected.sortedKeys))
                    #expect(same(actual.inverseOrder, expected.inverseOrder))
                }
            }
        }
    }
}
