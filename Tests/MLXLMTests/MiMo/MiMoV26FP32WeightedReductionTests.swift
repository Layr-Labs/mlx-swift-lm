// Prepared source only. Run these real Metal cells serially under root's
// exclusive native test grant. No skip or scripted effective-route substitute.
import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class MiMoV26FP32WeightedReductionTests: XCTestCase {
    private func exact(
        _ actual: MLXArray, _ expected: MLXArray,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        eval(actual, expected)
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertEqual(actual.dtype, expected.dtype, file: file, line: line)
        XCTAssertEqual(
            actual.asData(access: .copy).data, expected.asData(access: .copy).data,
            "Exact bytes are a separate lossless gate, not an allclose claim", file: file,
            line: line)
    }

    private func glu(
        _ profile: SwitchGLUWeightedReductionProfile = .mimoV26FP32,
        hidden: Int = 64, dtype: DType = .bfloat16
    ) throws -> SwitchGLU {
        let value = SwitchGLU(
            inputDims: hidden, hiddenDims: 32, numExperts: 16,
            weightedReductionProfile: profile)
        let weights = Dictionary(
            uniqueKeysWithValues: value.parameters().flattened().map { name, array in
                let salt = name.utf8.reduce(0) { ($0 + Int($1)) % 31 }
                let values = (0 ..< array.size).map { sin(Float($0 * 7 + salt)) * 0.075 }
                return (name, MLXArray(values, array.shape).asType(dtype))
            })
        try value.update(parameters: .unflattened(weights), verify: .all)
        return value
    }

    private func inputs(
        batch: Int = 1, length: Int, hidden: Int = 64,
        topK: Int = 8, dtype: DType = .bfloat16
    ) -> (MLXArray, MLXArray, MLXArray) {
        let rows = batch * length
        let x = MLXArray(
            (0 ..< rows * hidden).map { cos(Float($0 * 3 + 7)) },
            [batch, length, hidden]
        ).asType(dtype)
        // Deliberate duplicates within a token and across tokens. This tests
        // assignment/row mapping, not a claim that the real router emits ties.
        let idx = MLXArray(
            (0 ..< rows * topK).map { UInt32(($0 / 2 + $0 / topK * 3) % 16) },
            [batch, length, topK])
        let weights = MLXArray(
            (0 ..< rows * topK).map { Float(($0 * 13) % 101 + 1) / 137.0 },
            [batch, length, topK])
        return (x, idx, weights)
    }

    private func baseline(
        _ value: SwitchGLU, _ x: MLXArray, _ indices: MLXArray,
        _ weights: MLXArray
    ) -> MLXArray {
        // EXACT current MiMo unfused expression; deliberately not compiled
        // weightedExpertSum and never a pre-cast of the FP32 router weights.
        (value(x, indices) * weights[.ellipsis, .newAxis]).sum(axis: -2)
    }

    func testOptInIsOffByDefaultAndRequiresExplicitTrue() {
        XCTAssertFalse(MiMoV26FP32WeightedReduction.isEnabled(environment: [:]))
        for value in ["", "0", "false", "off", "unknown"] {
            XCTAssertFalse(
                MiMoV26FP32WeightedReduction.isEnabled(
                    environment: [MiMoV26FP32WeightedReduction.envFlag: value]))
        }
        for value in ["1", "true", "on", " TRUE "] {
            XCTAssertTrue(
                MiMoV26FP32WeightedReduction.isEnabled(
                    environment: [MiMoV26FP32WeightedReduction.envFlag: value]))
        }
    }

    func testRealSortedSwitchRouteEngagesForBothTypesAndTopK() throws {
        for dtype: DType in [.float16, .bfloat16] {
            let value = try glu(dtype: dtype)
            for topK in [6, 8] {
                for (batch, length) in [(1, 12), (2, 7)] {
                    let (x, indices, weights) = inputs(
                        batch: batch, length: length, topK: topK, dtype: dtype)
                    let actual = value.callAndMiMoFP32WeightedReduce(
                        x, indices, weights: weights, enabled: true)
                    XCTAssertEqual(
                        actual.route, .fused, "This test MUST NOT silently pass fallback")
                    XCTAssertEqual(actual.output.dtype, .float32)
                    let expected = baseline(value, x, indices, weights)
                    exact(actual.output, expected)
                    exact(actual.output.asType(dtype), expected.asType(dtype))
                }
            }
        }
    }

    func testOffUnsortedDecodeAndUnsupportedCallsKeepExactUnfusedFallback() throws {
        let value = try glu()
        let cases: [(Int, Int, Int, Bool, MiMoV26FP32WeightedReductionRoute)] = [
            (1, 12, 8, false, .disabled), (1, 4, 8, true, .notSorted),
            (1, 1, 8, true, .notPrefill), (16, 1, 8, true, .notPrefill),
            (1, 12, 2, true, .unsupportedTopK),
        ]
        for (batch, length, topK, enabled, route) in cases {
            let (x, idx, w) = inputs(batch: batch, length: length, topK: topK)
            let actual = value.callAndMiMoFP32WeightedReduce(x, idx, weights: w, enabled: enabled)
            XCTAssertEqual(actual.route, route)
            exact(actual.output, baseline(value, x, idx, w))
        }
        let (x, idx, w) = inputs(length: 12)
        let rounded = w.asType(.bfloat16)
        let badWeights = value.callAndMiMoFP32WeightedReduce(
            x, idx, weights: rounded, enabled: true)
        XCTAssertEqual(badWeights.route, .unsupportedDType)
        exact(badWeights.output, baseline(value, x, idx, rounded))
        let fp32 = try glu(dtype: .float32)
        let badType = fp32.callAndMiMoFP32WeightedReduce(
            x.asType(.float32), idx, weights: w, enabled: true)
        XCTAssertEqual(badType.route, .unsupportedDType)
        exact(badType.output, baseline(fp32, x.asType(.float32), idx, w))
        let odd = try glu(hidden: 65)
        let (oddX, oddIdx, oddW) = inputs(length: 12, hidden: 65)
        let badShape = odd.callAndMiMoFP32WeightedReduce(oddX, oddIdx, weights: oddW, enabled: true)
        XCTAssertEqual(badShape.route, .unsupportedShape)
        exact(badShape.output, baseline(odd, oddX, oddIdx, oddW))
    }

    private func reductionFixture(topK: Int, dtype: DType, cancellation: Bool)
        -> (original: MLXArray, sorted: MLXArray, inverse: MLXArray, weights: MLXArray)
    {
        let tokens = 16
        let hidden = 64
        let count = tokens * topK
        MLXRandom.seed(UInt64(700 + topK))
        var original = MLXRandom.normal([2, tokens / 2, topK, hidden]).asType(dtype)
        var weights = MLXRandom.uniform(low: -1, high: 1, [2, tokens / 2, topK])
        if cancellation {
            let terms: [Float] = [256, -256, 1, -1, 2, -2, 4, -4]
            let scores: [Float] = [1.001, 1, 1.1, 1, 1, 1, 1, 1]
            original = MLXArray(
                (0 ..< count * hidden).map { terms[($0 / hidden) % topK] },
                [2, tokens / 2, topK, hidden]
            ).asType(dtype)
            weights = MLXArray((0 ..< count).map { scores[$0 % topK] }, [2, tokens / 2, topK])
        }
        let assignments = MLXArray((0 ..< count).map { UInt32(($0 * 17 + $0 / topK) % 13) })
        let order = argSort(assignments)
        let inverse = argSort(order)
        return (original, original.reshaped(count, hidden)[order], inverse, weights)
    }

    func testExactFP32BytesRemainHardGateForRandomAndCancellationSensitiveValues() {
        for dtype: DType in [.float16, .bfloat16] {
            for topK in [6, 8] {
                for cancellation in [false, true] {
                    let f = reductionFixture(topK: topK, dtype: dtype, cancellation: cancellation)
                    XCTAssertNil(
                        mimoV26FP32WeightedUnsortRefusal(
                            sortedOutputs: f.sorted, inverseOrder: f.inverse, weights: f.weights))
                    let actual = mimoV26FP32WeightedUnsort(
                        sortedOutputs: f.sorted, inverseOrder: f.inverse, weights: f.weights)
                    let expected = (f.original * f.weights[.ellipsis, .newAxis]).sum(axis: -2)
                    // If this fails under a real compiler/device, retain the
                    // numerical counterexample; do not replace with allclose.
                    exact(actual, expected)
                    exact(actual.asType(dtype), expected.asType(dtype))
                }
            }
        }
    }

    func testSeparateReferenceToleranceAndDiscriminatingBF16Negative() {
        for topK in [6, 8] {
            let f = reductionFixture(topK: topK, dtype: .bfloat16, cancellation: true)
            let actual = mimoV26FP32WeightedUnsort(
                sortedOutputs: f.sorted, inverseOrder: f.inverse, weights: f.weights)
            let expected = (f.original * f.weights[.ellipsis, .newAxis]).sum(axis: -2)
            eval(actual, expected)
            // PR17's existing numerical check, reported separately. This is
            // not substituted for either exact-byte or trained-token gates.
            XCTAssertTrue(allClose(actual, expected, rtol: 2e-2, atol: 2e-2).item(Bool.self))
            let wrong = weightedExpertUnsort(
                sortedOutputs: f.sorted, inverseOrder: f.inverse,
                weights: f.weights.reshaped(16, topK).asType(.bfloat16)
            ).reshaped(2, 8, 64)
            eval(wrong)
            XCTAssertGreaterThan(
                max(abs(expected - wrong.asType(.float32))).item(Float.self), 0.1,
                "Fixture must detect the forbidden BF16 product/weight path")
            XCTAssertEqual(
                wrong.dtype, .bfloat16, "Old other-family reducer keeps its old contract")
        }
    }

    func testMetadataAndCPURefusalDoNotDispatchTheKernel() {
        let f = reductionFixture(topK: 8, dtype: .bfloat16, cancellation: false)
        XCTAssertEqual(
            mimoV26FP32WeightedUnsortRefusal(
                sortedOutputs: f.sorted,
                inverseOrder: f.inverse.asType(.int32), weights: f.weights), .unsupportedDType)
        XCTAssertEqual(
            mimoV26FP32WeightedUnsortRefusal(
                sortedOutputs: f.sorted,
                inverseOrder: f.inverse, weights: f.weights.asType(.float16)), .unsupportedDType)
        XCTAssertEqual(
            mimoV26FP32WeightedUnsortRefusal(
                sortedOutputs: f.sorted[0 ..< 1],
                inverseOrder: f.inverse, weights: f.weights), .unsupportedShape)
        MLX.Stream.withNewDefaultStream(device: .cpu) {
            XCTAssertEqual(
                mimoV26FP32WeightedUnsortRefusal(
                    sortedOutputs: f.sorted, inverseOrder: f.inverse, weights: f.weights),
                .unsupportedStream)
        }
    }

    func testOtherFamilyProfilesNeverEnterMiMoRouteAndKeepExistingEntryPoint() throws {
        for profile: SwitchGLUWeightedReductionProfile in [
            .generic, .gemma4ProductionGeGLU, .qwen35ProductionSwiGLU, .qwen4ProductionSwiGLU,
        ] {
            let value = try glu(profile)
            let (x, indices, weights) = inputs(length: 12)
            let candidate = value.callAndMiMoFP32WeightedReduce(
                x, indices, weights: weights, enabled: true)
            XCTAssertEqual(candidate.route, .unsupportedProfile)
            exact(candidate.output, baseline(value, x, indices, weights))
            let legacy = value.callAndWeightedReduce(
                x, indices, weights: weights.asType(.bfloat16),
                fuseSortedReduction: true)
            exact(legacy, weightedExpertSum(value(x, indices), weights.asType(.bfloat16)))
        }
    }

    func testCustomActivationCannotBorrowMiMoProfile() {
        let value = SwitchGLU(
            inputDims: 64, hiddenDims: 32, numExperts: 16,
            activation: { $0 + 0.25 }, weightedReductionProfile: .mimoV26FP32)
        let (x, indices, weights) = inputs(length: 12, dtype: .float32)
        let actual = value.callAndMiMoFP32WeightedReduce(
            x, indices, weights: weights, enabled: true)
        XCTAssertEqual(actual.route, .unsupportedActivation)
        exact(actual.output, baseline(value, x, indices, weights))
    }

    func testRealMXFP4SwitchProjectionEngagesWithoutChangingQuantizedParameters() throws {
        let value = try glu()
        quantize(
            model: value,
            filter: { _, module in
                module is SwitchLinear ? (groupSize: 32, bits: 4, mode: .mxfp4) : nil
            })
        let parameters = value.parameters().flattened()
        XCTAssertTrue(parameters.contains { $0.0.hasSuffix("scales") })
        let (x, indices, weights) = inputs(length: 12)
        let actual = value.callAndMiMoFP32WeightedReduce(
            x, indices, weights: weights, enabled: true)
        XCTAssertEqual(actual.route, .fused)
        exact(actual.output, baseline(value, x, indices, weights))
        let after = Dictionary(uniqueKeysWithValues: value.parameters().flattened())
        XCTAssertEqual(after.count, parameters.count)
        for (name, original) in parameters { XCTAssertTrue(after[name] === original, name) }
    }

    private func nativeConfig() throws -> MiMoV26Configuration {
        try JSONDecoder().decode(
            MiMoV26Configuration.self,
            from: Data(
                """
                {"model_type":"mimo_v2","architectures":["MiMoV2ForCausalLM"],
                 "hidden_size":64,"intermediate_size":128,"moe_intermediate_size":32,
                 "vocab_size":32,"num_hidden_layers":2,"max_position_embeddings":128,
                 "sliding_window_size":4,"sliding_window":4,"num_nextn_predict_layers":3,
                 "hybrid_layer_pattern":[0,1],"moe_layer_freq":[0,1],
                 "partial_rotary_factor":1,"attention_value_scale":0.707,
                 "layernorm_epsilon":0.000001,"attention_projection_layout":"split",
                 "moe_router_dtype":"bfloat16","hidden_act":"silu","dtype":"bfloat16",
                 "attention_bias":false,"tie_word_embeddings":false,"attention_dropout":0,
                 "scoring_func":"sigmoid","topk_method":"noaux_tc","n_routed_experts":16,
                 "num_experts_per_tok":8,"n_group":1,"topk_group":1,"norm_topk_prob":true,
                 "n_shared_experts":null,"routed_scaling_factor":1.7,
                 "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"v_head_dim":16,
                 "swa_num_attention_heads":2,"swa_num_key_value_heads":1,"swa_head_dim":32,
                 "swa_v_head_dim":16,"rope_theta":10000000,"swa_rope_theta":10000,
                 "add_full_attention_sink_bias":false,"add_swa_attention_sink_bias":true,
                 "eos_token_id":3,"pad_token_id":0}
                """.utf8))
    }

    func testActualMiMoRouterAndMoEUseFP32WeightsAndOriginalFinalCast() throws {
        let config = try nativeConfig()
        let off = MiMoV26MoE(config, fp32WeightedReduction: false)
        let on = MiMoV26MoE(config, fp32WeightedReduction: true)
        let parameters = Dictionary(
            uniqueKeysWithValues: off.parameters().flattened().map { name, value in
                let dtype: DType = name.hasSuffix("e_score_correction_bias") ? .float32 : .bfloat16
                return (
                    name,
                    MLXArray(
                        (0 ..< value.size).map { sin(Float($0 * 11 + 17)) * 0.05 },
                        value.shape
                    ).asType(dtype)
                )
            })
        try off.update(parameters: .unflattened(parameters), verify: .all)
        try on.update(parameters: .unflattened(parameters), verify: .all)
        let (x, _, _) = inputs(length: 12)
        let before = off.gate(x)
        let after = on.gate(x)
        XCTAssertEqual(before.weights.dtype, .float32)
        XCTAssertEqual(after.weights.dtype, .float32)
        exact(before.weights, after.weights)
        exact(before.indices, after.indices)
        eval(after.weights)
        XCTAssertTrue(
            any(after.weights .!= after.weights.asType(.bfloat16).asType(.float32)).item(Bool.self),
            "Actual router fixture must retain information beyond BF16")
        let candidate = on.forwardWithWeightedReductionRoute(x)
        XCTAssertEqual(
            candidate.route, .fused, "Real MiMo route must engage, not just direct helper")
        XCTAssertEqual(candidate.output.dtype, x.dtype)
        let reference = (off.switchMLP(x, before.indices) * before.weights[.ellipsis, .newAxis])
            .sum(axis: -2).asType(x.dtype)
        exact(candidate.output, reference)
        exact(off(x), reference)
        exact(on(x), candidate.output)
    }
}
