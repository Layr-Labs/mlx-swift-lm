import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// Regression test: without a cache, the attention mask of `NemotronHModel`
/// was `.none`, so the full forward pass was not causal.
///
/// The test is copied from `NemotronHRuntimeTests.attentionIsCausalOnlyWithACache`
/// of PR #240 (`Tests/MLXLMTests/Kernel/LLM/NemotronHRuntimeTests.swift`),
/// without the known issue. The model is tiny and has seeded random weights:
/// a Mamba layer, an attention layer and an MLP layer, hidden size 64,
/// vocabulary 64. The prompt has 11 tokens.
///
/// Tolerance 1e-4: float32, and the two passes differ only in the order of
/// the sums. The differences are near 1e-6.
@Suite
struct NemotronHCausalMaskTests {

    static let vocabularySize = 64
    static let tolerance: Float = 1e-4

    static func makeModel(_ blocks: [String], seed: UInt64 = 1) throws -> NemotronHModel {
        let values: [String: Any] = [
            "model_type": "nemotron_h",
            "vocab_size": vocabularySize,
            "hidden_size": 64,
            "num_hidden_layers": blocks.count,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "mamba_num_heads": 4,
            "mamba_head_dim": 16,
            "ssm_state_size": 16,
            "conv_kernel": 4,
            "n_groups": 2,
            "intermediate_size": 64,
            "moe_intermediate_size": 32,
            "moe_shared_expert_intermediate_size": 32,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "layers_block_type": blocks,
            "mamba_ssm_cache_dtype": "float32",
        ]
        let configuration = try JSONDecoder().decode(
            NemotronHConfiguration.self, from: JSONSerialization.data(withJSONObject: values))
        let model = NemotronHModel(configuration)
        NemotronHMaskTinyModel.randomize(model, seed: seed)
        return model
    }

    /// Without a cache, a change at position 6 must not change the logits
    /// of positions 0 to 5. The same check with a new cache is the control.
    @Test func attentionIsCausalWithAndWithoutACache() throws {
        let model = try Self.makeModel(["mamba", "attention", "mlp"])
        let row = NemotronHMaskTinyModel.tokens(
            count: 11, vocabularySize: Self.vocabularySize, seed: 1)
        var changed = row
        changed[6] = (row[6] + 1) % Self.vocabularySize

        let original = NemotronHMaskTinyModel.logits(model, [row])
        let modified = NemotronHMaskTinyModel.logits(model, [changed])
        let before = NemotronHMaskTinyModel.maxAbsDifference(
            original[0..., ..<6], modified[0..., ..<6])
        let after = NemotronHMaskTinyModel.maxAbsDifference(
            original[0..., 6...], modified[0..., 6...])
        #expect(before <= Self.tolerance, "positions before 6 changed by \(before)")
        #expect(after > 1e-3, "the change at 6 must change its own logits")

        // Control: the same pass with a new cache is causal.
        let cachedOriginal = NemotronHMaskTinyModel.logits(
            model, [row], cache: model.newCache(parameters: nil))
        let cachedModified = NemotronHMaskTinyModel.logits(
            model, [changed], cache: model.newCache(parameters: nil))
        let cachedBefore = NemotronHMaskTinyModel.maxAbsDifference(
            cachedOriginal[0..., ..<6], cachedModified[0..., ..<6])
        #expect(cachedBefore <= Self.tolerance, "cached positions before 6")
    }
}

/// Tiny-model helpers for this file. They are copied from the kernel test
/// support of PR #183 (`Tests/MLXLMTests/Kernel/Support/SyntheticModel.swift`
/// and `ForwardPassChecks.swift`) and are private, so that this file does
/// not depend on that PR or collide with it.
private enum NemotronHMaskTinyModel {

    /// Replaces every floating-point parameter with seeded random values:
    /// norm scales near 1, other 1-D values near 0, and matrices with a
    /// standard deviation of `1 / sqrt(fan-in)`.
    static func randomize(_ model: Module, seed: UInt64) {
        let parameters = model.parameters().flattened().sorted { $0.0 < $1.0 }
        var updated: [(String, MLXArray)] = []
        for (index, (name, value)) in parameters.enumerated() where value.dtype.isFloatingPoint {
            let noise = MLXRandom.normal(
                value.shape, key: MLXRandom.key(seed &* 1_000_003 &+ UInt64(index)))
            let random: MLXArray
            if value.ndim <= 1 {
                random = name.hasSuffix("weight") ? 1 + 0.1 * noise : 0.1 * noise
            } else {
                let fanIn =
                    name.contains("conv") ? value.shape.dropFirst().reduce(1, *) : value.dim(-1)
                random = noise * (1 / Float(fanIn).squareRoot())
            }
            // The GPU has no float64. A float64 initial value becomes float32.
            let dtype: DType = value.dtype == .float64 ? .float32 : value.dtype
            updated.append((name, random.asType(dtype)))
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
        eval(model)
    }

    /// Token IDs from a fixed linear congruential generator.
    static func tokens(count: Int, vocabularySize: Int, seed: Int) -> [Int] {
        var state = UInt64(truncatingIfNeeded: seed) &+ 0x9E37_79B9
        return (0 ..< count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int((state >> 33) % UInt64(vocabularySize))
        }
    }

    static func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// Runs the model on `rows` and returns the logits.
    static func logits(
        _ model: any LanguageModel, _ rows: [[Int]], cache: [KVCache]? = nil
    ) -> MLXArray {
        let batch = MLXArray(rows.flatMap { $0.map { Int32($0) } })
            .reshaped(rows.count, rows[0].count)
        let output = model(batch, cache: cache)
        eval(output)
        return output
    }
}
