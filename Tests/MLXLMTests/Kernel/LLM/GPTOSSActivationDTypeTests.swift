import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM

extension KernelTests {

    /// The activation and K/V dtypes of a tiny GPT-OSS model in the layout of
    /// the published checkpoints: every floating parameter in the model dtype
    /// and MXFP4 experts.
    ///
    /// The expert SwiGLU once clamped with float32 `MLXArray` bounds. They
    /// promoted the bf16 expert output to float32, the residual add carried
    /// float32 into every later layer, and those layers projected and stored
    /// their K/V in float32. A float32 model must stay float32.
    @Suite(.serialized)
    struct GPTOSSActivationDTypeTests {

        static let vocabularySize = 64
        static let layerCount = 4
        static let width = 64

        /// Four layers that alternate sliding and full attention, starting
        /// with sliding, as in gpt-oss-20b. MXFP4 needs a width that is a
        /// multiple of its group size of 32.
        static var base: [String: Any] {
            [
                "model_type": "gpt_oss",
                "num_hidden_layers": layerCount,
                "num_local_experts": 4,
                "num_experts_per_tok": 2,
                "vocab_size": vocabularySize,
                "rms_norm_eps": 1e-5,
                "hidden_size": width,
                "intermediate_size": width,
                "head_dim": 16,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "sliding_window": 8,
            ]
        }

        /// Seeded random weights, including nonzero sinks, converted to
        /// `dtype`, and then MXFP4 experts with group size 32.
        static func makeModel(dtype: DType) throws -> GPTOSSModel {
            let model = GPTOSSModel(
                try SyntheticModel.configuration(GPTOSSConfiguration.self, base))
            SyntheticModel.randomize(model, seed: 0x6770_746F)
            convertFloatingParameters(of: model, to: dtype)
            quantize(
                model: model, groupSize: 32, bits: 4, mode: .mxfp4,
                filter: { path, _ in path.contains(".experts.") })
            eval(model)
            return model
        }

        static func makeExperts(dtype: DType, fusedGateUp: Bool) -> SwiGLUSwitchGLU {
            let experts = SwiGLUSwitchGLU(
                inputDims: width, hiddenDims: width, numExperts: 4, bias: true,
                fusedGateUp: fusedGateUp)
            SyntheticModel.randomize(experts, seed: 0x6578_7073)
            convertFloatingParameters(of: experts, to: dtype)
            quantize(model: experts, groupSize: 32, bits: 4, mode: .mxfp4)
            eval(experts)
            return experts
        }

        static func convertFloatingParameters(of module: Module, to dtype: DType) {
            let converted = module.parameters().flattened().compactMap { name, value in
                value.dtype.isFloatingPoint ? (name, value.asType(dtype)) : nil
            }
            module.update(parameters: ModuleParameters.unflattened(converted))
        }

        /// Activations `[2, length, width]` with two expert indices per token.
        static func expertInputs(length: Int, dtype: DType) -> (MLXArray, MLXArray) {
            let x = MLXRandom.normal([2, length, width], key: MLXRandom.key(11)).asType(dtype)
            let ids = (0 ..< (2 * length * 2)).map { UInt32(($0 * 3 + 1) % 4) }
            return (x, MLXArray(ids).reshaped(2, length, 2))
        }

        @Test(arguments: [DType.bfloat16, .float32], [false, true])
        func expertsKeepTheActivationDType(dtype: DType, fusedGateUp: Bool) {
            let experts = Self.makeExperts(dtype: dtype, fusedGateUp: fusedGateUp)
            // 12 indices take the unsorted gather and 64 take the sorted one.
            for length in [3, 16] {
                let (x, ids) = Self.expertInputs(length: length, dtype: dtype)
                let output = experts.uncompiledExpertForward(x, ids)
                eval(output)
                #expect(output.dtype == dtype, "eager experts, \(length) tokens")
                #expect(output.shape == [2, length, 2, Self.width])
                #expect(SyntheticModel.maxAbs(output).isFinite)
            }
            // The opt-in compiled single-token graph wraps the same SwiGLU.
            let (x, ids) = Self.expertInputs(length: 1, dtype: dtype)
            let compiled = experts.compiledExpertForward(x, ids)
            eval(compiled)
            #expect(compiled.dtype == dtype, "compiled experts")
        }

        @Test(arguments: [DType.bfloat16, .float32])
        func everyLayerKeepsTheActivationDType(dtype: DType) throws {
            let model = try Self.makeModel(dtype: dtype)
            let tokens = SyntheticModel.batch([
                SyntheticModel.tokens(count: 7, vocabularySize: Self.vocabularySize, seed: 1)
            ])
            var hidden = model.model.embedTokens(tokens)
            #expect(hidden.dtype == dtype)
            // Seven tokens fit in the sliding window of 8, so the causal mask
            // is the exact mask of both layer types.
            for (index, layer) in model.loraLayers.enumerated() {
                let block = try #require(layer as? GPTOSSTransformerBlock)
                hidden = block(hidden, mask: .causal)
                eval(hidden)
                #expect(hidden.dtype == dtype, "output of layer \(index)")
            }
            let logits = model(tokens)
            eval(logits)
            #expect(logits.dtype == dtype)
            #expect(SyntheticModel.maxAbs(logits).isFinite)
        }

        /// Production admission, paged storage and checkpoint identity read
        /// the native K/V types from this probe, which runs the serving
        /// forward with the model's own CBv2 caches.
        @Test(arguments: [DType.bfloat16, .float32])
        func nativeKVProbeObservesTheModelDTypeOnEveryLayer(dtype: DType) throws {
            let model = try Self.makeModel(dtype: dtype)
            let probe = try CBv2NativeKVTypeProbe.run(
                model: CBv2SteppableLanguageModelAdapter(model),
                layerKinds: model.cbv2LayerKinds,
                caches: model.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
            // Layer 1, the first full-attention layer, is the first whose
            // input has passed through the experts.
            #expect(probe.layerDTypes == Array(repeating: dtype, count: Self.layerCount))
            // Every layer owns its K/V and is observed once in prefill and
            // once in decode.
            #expect(probe.observations.count == 2 * Self.layerCount)
            #expect(
                probe.observations.allSatisfy {
                    $0.keysDType == dtype && $0.valuesDType == dtype
                })
        }
    }
}
