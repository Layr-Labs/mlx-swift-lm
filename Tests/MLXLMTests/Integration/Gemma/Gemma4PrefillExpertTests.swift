import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.tags(.integration), .serialized)
struct Gemma4PrefillExpertTests {
    private func bytes(_ value: MLXArray) -> Data {
        eval(value)
        return value.asData(access: .copy).data
    }

    private func fixture(bits: Int, dtype: DType = .bfloat16, zeroShared: Bool = false) throws -> (
        Gemma4DecoderLayer, ObservedPrefillExpert
    ) {
        let values: [String: Any] = [
            "hidden_size": 128, "num_hidden_layers": 1, "intermediate_size": 128,
            "num_attention_heads": 2, "head_dim": 64, "num_key_value_heads": 1,
            "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
            "vocab_size": 128, "enable_moe_block": true, "num_experts": 16,
            "top_k_experts": 8, "moe_intermediate_size": 64, "sliding_window": 512,
            "layer_types": ["sliding_attention"], "use_double_wide_mlp": false,
            "quantization": ["bits": bits, "group_size": 64, "mode": "affine"],
        ]
        let config = try JSONDecoder().decode(
            Gemma4TextConfiguration.self,
            from: JSONSerialization.data(withJSONObject: values))
        MLXRandom.seed(6137)
        let layer = Gemma4DecoderLayer(config, layerIdx: 0, fuseWeightedUnsort: true)
        // Isolate token-local composition without replacing any expert/router,
        // normalization or shared-MLP operation. Attention still executes; its
        // zero output projection removes sequence context from this oracle.
        layer.update(
            parameters: .unflattened(
                layer.parameters().flattened().map { name, value in
                    (
                        name,
                        (name == "self_attn.o_proj.weight"
                            || zeroShared && name == "mlp.down_proj.weight")
                            ? MLXArray.zeros(value.shape, dtype: dtype) : value.asType(dtype)
                    )
                }))
        quantize(
            model: layer,
            filter: { path, _ -> (Int, Int, QuantizationMode)? in
                (64, path.hasPrefix("mlp.") || path == "router.proj" ? 8 : bits, .affine)
            })
        let gate = try #require(
            layer.leafModules().flattened().first {
                $0.0 == "experts.switch_glu.gate_proj"
            }?.1 as? QuantizedSwitchLinear)
        let observed = try ObservedPrefillExpert(
            inputDims: 128, outputDims: 64, numExperts: 16,
            weight: gate.weight, scales: gate.scales, biases: try #require(gate.biases),
            groupSize: gate.groupSize, bits: gate.bits)
        layer.update(modules: .unflattened(["experts.switch_glu.gate_proj": observed]))
        eval(layer)
        return (layer, observed)
    }

    /// An independently assembled oracle retains every existing rectangular
    /// normalization and elementwise operation. Only projections and the whole
    /// expert reduction use explicit original singleton module calls.
    private func expectedComposition(_ layer: Gemma4DecoderLayer, _ x: MLXArray) throws -> MLXArray
    {
        let modules = Dictionary(uniqueKeysWithValues: layer.namedModules())
        let parameters = Dictionary(uniqueKeysWithValues: layer.parameters().flattened())
        func linear(_ name: String) throws -> Linear { try #require(modules[name] as? Linear) }
        func norm(_ name: String, _ value: MLXArray) throws -> MLXArray {
            let module = try #require(modules[name] as? RMSNorm)
            return module(value)
        }
        func rows(_ input: MLXArray, _ forward: (MLXArray) -> MLXArray) -> MLXArray {
            concatenated(
                (0 ..< input.dim(1)).map { forward(input[0..., $0 ..< $0 + 1, 0...]) }, axis: 1)
        }
        let h1 = try norm("pre_feedforward_layernorm", x)
        let gate = try linear("mlp.gate_proj")
        let up = try linear("mlp.up_proj")
        let down = try linear("mlp.down_proj")
        let activated =
            safeGeluApproximate(rows(h1, gate.callAsFunction)) * rows(h1, up.callAsFunction)
        let shared = try norm("post_feedforward_layernorm_1", rows(activated, down.callAsFunction))
        let scale = try #require(parameters["router.scale"])
        let routerInput = MLXFast.rmsNorm(x, weight: scale * pow(Float(128), -0.5), eps: 1e-6)
        let router = try linear("router.proj")
        let scores = rows(routerInput, router.callAsFunction)
        let indices = MLX.argPartition(scores, kth: 8, axis: -1)[.ellipsis, 8...]
        let expertScale = try #require(parameters["router.per_expert_scale"])
        let weights =
            MLX.softmax(MLX.takeAlong(scores, indices, axis: -1), axis: -1, precise: true)
            * expertScale[indices]
        let h2 = try norm("pre_feedforward_layernorm_2", x)
        let experts = try #require(modules["experts.switch_glu"] as? SwitchGLU)
        let reduced = concatenated(
            (0 ..< x.dim(1)).map { token in
                experts.callAndWeightedReduce(
                    h2[0..., token ..< token + 1, 0...].reshaped([1, 128]),
                    indices[0..., token ..< token + 1, 0...].reshaped([1, 8]),
                    weights: weights[0..., token ..< token + 1, 0...].reshaped([1, 8]),
                    fuseSortedReduction: true, isProductionPrefill: true)
            }, axis: 0
        ).reshaped(x.shape)
        let sparse = try norm("post_feedforward_layernorm_2", reduced)
        let combined = try norm("post_feedforward_layernorm", shared + sparse)
        let scalar = try #require(parameters["layer_scalar"])
        return (x + combined) * scalar
    }

    @Test func actualGemmaMoECompositionPreservesScopedOperatorsAndRoutes() throws {
        try Device.withDefaultDevice(.cpu) {
            for bits in [4, 8] {
                let (layer, gate) = try fixture(bits: bits)
                for count in [7, 8, 127, 128, 511, 512] {
                    let input = MLXArray(
                        (0 ..< count * 128).map {
                            sin(Float($0) * 0.071) + Float($0 % 17 - 8) / 64
                        }, [1, count, 128]
                    ).asType(.bfloat16)
                    gate.calls = []
                    let expected = try expectedComposition(layer, input)
                    let referenceCalls = gate.calls
                    gate.calls = []
                    let actual = Gemma4PrefillProjectionScope.withPrefillFrame(
                        tokens: MLXArray.zeros([1, count], dtype: .int32), enabled: true
                    ) { layer(input, isExpertPrefill: true).0 }
                    #expect(gate.calls.count == count)
                    #expect(gate.calls == referenceCalls)
                    #expect(bytes(actual) == bytes(expected), "bits=\(bits) tokens=\(count)")
                    let finite = actual.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite)
                    #expect(finite)
                }
            }
        }
    }

    @Test func completeExpertBranchMatchesSingletonCompositionAtLargePromptBoundaries() throws {
        try Device.withDefaultDevice(.cpu) {
            for bits in [4, 8] {
                // Shared-branch zeroing isolates the new complete expert policy
                // from an existing CPU RMSNorm rectangle/singleton discrepancy.
                let (layer, gate) = try fixture(bits: bits, zeroShared: true)
                for count in [7, 8, 127, 128, 511, 512] {
                    let input = MLXArray(
                        (0 ..< count * 128).map {
                            sin(Float($0) * 0.071) + Float($0 % 17 - 8) / 64
                        }, [1, count, 128]
                    ).asType(.bfloat16)
                    let expected = concatenated(
                        (0 ..< count).map {
                            layer(input[0..., $0 ..< $0 + 1, 0...], isExpertPrefill: true).0
                        }, axis: 1)
                    gate.calls = []
                    let actual = Gemma4PrefillProjectionScope.withPrefillFrame(
                        tokens: MLXArray.zeros([1, count], dtype: .int32), enabled: true
                    ) { layer(input, isExpertPrefill: true).0 }
                    #expect(gate.calls.count == count)
                    #expect(bytes(actual) == bytes(expected), "bits=\(bits) tokens=\(count)")
                }
            }
        }
    }

    @Test func ordinaryVerificationSingletonAndOtherDtypeKeepOriginalExpertCall() throws {
        try Device.withDefaultDevice(.cpu) {
            for dtype in [DType.bfloat16, .float32] {
                let (layer, gate) = try fixture(bits: 4, dtype: dtype)
                for (count, enabled, prefill) in [
                    (8, false, true), (8, true, false), (1, true, true),
                ] {
                    let input = MLXArray.ones([1, count, 128], dtype: dtype)
                    gate.calls = []
                    let result = Gemma4PrefillProjectionScope.withPrefillFrame(
                        tokens: MLXArray.zeros([1, count], dtype: .int32), enabled: enabled
                    ) { layer(input, isExpertPrefill: prefill).0 }
                    eval(result)
                    #expect(gate.calls.count == 1)
                }
                if dtype == .float32 {
                    gate.calls = []
                    let result = Gemma4PrefillProjectionScope.withPrefillFrame(
                        tokens: MLXArray.zeros([1, 8], dtype: .int32), enabled: true
                    ) { layer(MLXArray.ones([1, 8, 128], dtype: dtype), isExpertPrefill: true).0 }
                    eval(result)
                    #expect(gate.calls.count == 1)
                }
            }
        }
    }

    @Test func incompatibleRoutesAndNonPromptFramesDoNotOptIn() {
        Device.withDefaultDevice(.cpu) {
            let input = MLXArray.ones([1, 8, 128], dtype: .bfloat16)
            let indices = MLXArray.zeros([1, 8, 8], dtype: .uint32)
            let weights = MLXArray.ones([1, 8, 8], dtype: .bfloat16)
            Gemma4PrefillProjection.$active.withValue(true) {
                #expect(
                    Gemma4PrefillProjection.appliesExperts(
                        input, indices: indices, weights: weights, isExpertPrefill: true))
                for bad in [
                    indices.asType(.int32), MLXArray.zeros([1, 7, 8], dtype: .uint32),
                    MLXArray.zeros([8, 8], dtype: .uint32),
                ] {
                    #expect(
                        !Gemma4PrefillProjection.appliesExperts(
                            input, indices: bad, weights: weights, isExpertPrefill: true))
                }
                #expect(
                    !Gemma4PrefillProjection.appliesExperts(
                        input, indices: indices, weights: weights.asType(.float32),
                        isExpertPrefill: true))
                #expect(
                    !Gemma4PrefillProjection.appliesExperts(
                        input, indices: indices, weights: weights, isExpertPrefill: false))
                #expect(
                    !Gemma4PrefillProjection.appliesExperts(
                        MLXArray.ones([2, 1, 128], dtype: .bfloat16), indices: indices,
                        weights: weights, isExpertPrefill: true))
            }
            #expect(
                !Gemma4PrefillProjection.appliesExperts(
                    input, indices: indices, weights: weights, isExpertPrefill: true))
        }
    }
}

private final class ObservedPrefillExpert: QuantizedSwitchLinear {
    var calls: [[UInt32]] = []
    override func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false)
        -> MLXArray
    {
        eval(indices)
        calls.append(indices.asArray(UInt32.self))
        return super.callAsFunction(x, indices, sortedIndices: sortedIndices)
    }
}
