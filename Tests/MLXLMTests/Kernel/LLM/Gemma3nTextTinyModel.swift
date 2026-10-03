import Foundation
import MLX
import MLXLMCommon
import MLXNN

@testable import MLXLLM

/// A tiny `Gemma3nTextModel` with seeded random weights for the Gemma 3n
/// regression tests.
///
/// The model has 4 layers: sliding, full, sliding, full. The last 2 layers
/// share the KV caches of the first 2. It has 4 AltUp streams, the LAuReL
/// block, per-layer inputs, activation sparsity in layer 0 and a final logit
/// soft cap. The weights are synthetic. No test downloads or reads real
/// model weights.
///
/// This file copies the minimal parts of `SyntheticModel` and
/// `ForwardPassChecks` from the kernel forward-pass tests (PR #183 and
/// PR #184). It has its own name, so that it does not collide with them.
enum Gemma3nTextTinyModel {

    static let vocabularySize = 64

    // Tolerance of the float32 comparisons: the cached path and the full
    // path differ in the order of the attention sums.
    static let tolerance: Float = 1e-4

    static var base: [String: Any] {
        [
            "model_type": "gemma3n_text", "hidden_size": 32, "num_hidden_layers": 4,
            "intermediate_size": 64, "num_attention_heads": 2, "head_dim": 16,
            "rms_norm_eps": 1e-6, "vocab_size": vocabularySize, "num_key_value_heads": 1,
            "num_kv_shared_layers": 2, "vocab_size_per_layer_input": vocabularySize,
            "sliding_window": 16, "max_position_embeddings": 256,
            "rope_local_base_freq": 10000, "rope_theta": 1_000_000,
            "final_logit_softcapping": 30,
            "layer_types": [
                "sliding_attention", "full_attention", "sliding_attention", "full_attention",
            ],
            "activation_sparsity_pattern": [0.95, 0, 0, 0],
            "hidden_size_per_layer_input": 8, "altup_num_inputs": 4, "altup_coef_clip": 120,
            "altup_correct_scale": true, "altup_active_idx": 0, "laurel_rank": 4,
        ]
    }

    /// Builds the model. A key in `overrides` replaces the same key in `base`.
    static func make(_ overrides: [String: Any] = [:], seed: UInt64 = 1) throws
        -> Gemma3nTextModel
    {
        let merged = base.merging(overrides) { _, new in new }
        let data = try JSONSerialization.data(withJSONObject: merged)
        let configuration = try JSONDecoder().decode(Gemma3nTextConfiguration.self, from: data)
        let model = Gemma3nTextModel(config: configuration)
        randomize(model, seed: seed)
        return model
    }

    /// Replaces every floating-point parameter with seeded random values.
    /// A 1-D `weight` (a norm scale) gets values near 1, another 1-D
    /// parameter gets values near 0, and a matrix gets a standard deviation
    /// of `1 / sqrt(fan-in)`.
    static func randomize(_ model: Module, seed: UInt64) {
        let parameters = model.parameters().flattened().sorted { $0.0 < $1.0 }
        var updated: [(String, MLXArray)] = []
        for (index, (name, value)) in parameters.enumerated() where value.dtype.isFloatingPoint {
            let key = MLXRandom.key(seed &* 1_000_003 &+ UInt64(index))
            let noise = MLXRandom.normal(value.shape, key: key)
            let random: MLXArray
            if value.ndim <= 1 {
                random = name.hasSuffix("weight") ? 1 + 0.1 * noise : 0.1 * noise
            } else {
                random = noise * (1 / Float(value.dim(-1)).squareRoot())
            }
            // The GPU has no float64. A float64 initial value becomes float32.
            let dtype: DType = value.dtype == .float64 ? .float32 : value.dtype
            updated.append((name, random.asType(dtype)))
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
        eval(model)
    }

    /// Token IDs from a fixed linear congruential generator.
    static func row(_ seed: Int, count: Int = 11) -> [Int] {
        var state = UInt64(truncatingIfNeeded: seed) &+ 0x9E37_79B9
        return (0 ..< count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int((state >> 33) % UInt64(vocabularySize))
        }
    }

    /// Runs the model on `rows` (all of the same length) and returns the
    /// logits.
    static func logits(
        _ model: Gemma3nTextModel, _ rows: [[Int]], cache: [KVCache]? = nil
    ) -> MLXArray {
        let length = rows[0].count
        precondition(rows.allSatisfy { $0.count == length })
        let tokens = MLXArray(rows.flatMap { $0.map { Int32($0) } }).reshaped(rows.count, length)
        let output = model(tokens, cache: cache)
        eval(output)
        return output
    }

    /// Runs `rows` in `chunks` with one new cache and returns the logits of
    /// all positions.
    static func chunkedLogits(
        _ model: Gemma3nTextModel, _ rows: [[Int]], chunks: [Int]
    ) -> MLXArray {
        precondition(chunks.reduce(0, +) == rows[0].count, "the chunks must cover the rows")
        let cache = model.newCache(parameters: nil)
        var start = 0
        var parts: [MLXArray] = []
        for chunk in chunks {
            parts.append(
                logits(model, rows.map { Array($0[start ..< start + chunk]) }, cache: cache))
            start += chunk
        }
        return concatenated(parts, axis: 1)
    }

    /// The largest absolute difference between two arrays.
    static func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
}
