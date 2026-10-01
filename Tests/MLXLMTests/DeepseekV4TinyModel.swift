import Foundation
import MLX
import MLXLMCommon
import MLXNN

@testable import MLXLLM

/// A tiny `DeepseekV4Model` with seeded random weights for the DeepSeek V4
/// regression tests. No test downloads or reads real model weights.
///
/// The model has 3 layers, one of each attention type: local attention
/// (compress ratio 0), sparse compressed attention with an indexer (ratio 4)
/// and compressed attention (ratio 128). Layer 0 routes its experts by token
/// ID (hash routing). Every layer uses the hyper connection with 4 streams.
/// The index top-k is 2, so the sparse layer takes the gathered top-k path
/// once it has more than 2 pooled entries.
///
/// The pieces are copied from the kernel test helpers of PR #183
/// (`SyntheticModel`, `ForwardPassChecks`) and the model setup of PR #184
/// (`DeepseekV4ForwardPassTests`), with other names so that they do not
/// collide when those pull requests merge.
enum DeepseekV4TinyModel {

    static let vocabularySize = 64

    static var base: [String: Any] {
        [
            "vocab_size": vocabularySize,
            "hidden_size": 32,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "head_dim": 16,
            "q_lora_rank": 16,
            "qk_rope_head_dim": 8,
            "rms_norm_eps": 1e-6,
            "o_groups": 2,
            "o_lora_rank": 8,
            "sliding_window": 8,
            "compress_ratios": [0, 4, 128],
            "compress_rope_theta": 160000,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.5,
            "swiglu_limit": 10.0,
            "num_hash_layers": 1,
            "num_nextn_predict_layers": 0,
            "norm_topk_prob": true,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 3,
            "hc_eps": 1e-6,
            "rope_theta": 10000,
            "max_position_embeddings": 4096,
            "index_n_heads": 2,
            "index_head_dim": 16,
            "index_topk": 2,
        ]
    }

    /// Builds the model from `base` with `overrides`, and gives it random
    /// weights from `seed`.
    static func make(_ overrides: [String: Any] = [:], seed: UInt64 = 1) throws
        -> DeepseekV4Model
    {
        let merged = base.merging(overrides) { _, new in new }
        let data = try JSONSerialization.data(withJSONObject: merged)
        let configuration = try JSONDecoder().decode(DeepseekV4Configuration.self, from: data)
        let model = DeepseekV4Model(configuration)
        randomize(model, seed: seed)
        // The hash layer maps each token to 2 different experts. The random
        // pass does not touch this integer table.
        let table = (0 ..< vocabularySize).flatMap { [Int32($0 % 4), Int32(($0 + 1) % 4)] }
        model.update(
            parameters: ModuleParameters.unflattened([
                "model.layers.0.ffn.gate.tid2eid": MLXArray(table).reshaped(vocabularySize, 2)
            ]))
        eval(model)
        return model
    }

    /// Replaces every floating-point parameter with seeded random values. A
    /// 1-D `weight` (a norm scale) gets values near 1, another 1-D parameter
    /// values near 0, and a matrix a standard deviation of `1 / sqrt(fan-in)`.
    static func randomize(_ model: Module, seed: UInt64) {
        let parameters = model.parameters().flattened().sorted { $0.0 < $1.0 }
        var updated: [(String, MLXArray)] = []
        for (index, (name, value)) in parameters.enumerated() {
            guard value.dtype.isFloatingPoint else { continue }
            let key = MLXRandom.key(seed &* 1_000_003 &+ UInt64(index))
            let noise = MLXRandom.normal(value.shape, key: key)
            let random: MLXArray
            if value.ndim <= 1 {
                random = name.hasSuffix("weight") ? 1 + 0.1 * noise : 0.1 * noise
            } else {
                random = noise * (1 / Float(value.dim(-1)).squareRoot())
            }
            updated.append((name, random.asType(value.dtype)))
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
        eval(model)
    }

    /// Token IDs from a fixed linear congruential generator.
    static func row(_ seed: Int, count: Int) -> [Int] {
        var state = UInt64(truncatingIfNeeded: seed) &+ 0x9E37_79B9
        return (0 ..< count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int((state >> 33) % UInt64(vocabularySize))
        }
    }

    /// Runs the model on `rows` and returns the logits.
    static func logits(_ model: DeepseekV4Model, _ rows: [[Int]], cache: [KVCache]? = nil)
        -> MLXArray
    {
        let tokens = MLXArray(rows.flatMap { $0.map { Int32($0) } })
            .reshaped(rows.count, rows[0].count)
        let output = model(tokens, cache: cache)
        eval(output)
        return output
    }

    /// The largest absolute difference between two arrays.
    static func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
}
