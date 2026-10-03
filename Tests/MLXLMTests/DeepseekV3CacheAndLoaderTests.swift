import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// Regression tests for the cache and the loader of `DeepseekV3Model`
/// (issues #203 and #233).
///
/// The tests use a tiny DeepSeek V3 model with seeded random weights:
/// multi-head latent attention (LoRA ranks 16), one dense layer and one
/// layer with 4 routed experts in 2 groups and a shared expert. They use no
/// real weights and no network.
///
/// The tests are copies of `newCacheGivesOneCachePerLayer`,
/// `cachedDecodeMatchesTheFullForwardPass` and
/// `loaderStacksPerExpertWeights` of `DeepseekV3ForwardPassTests` in
/// PR #185, without the `withKnownIssue` blocks. The helpers are small
/// copies of `SyntheticModel`, `ForwardPassChecks` and `CheckpointLayout`
/// from PR #183. They are private to this suite, so that they do not collide
/// with those types.
@Suite
struct DeepseekV3CacheAndLoaderTests {

    static let vocabularySize = 64

    // Tolerance of the float32 comparisons: the paths differ only in the
    // order of the attention sums; differences are near 1e-6.
    static let tolerance: Float = 1e-4

    static var configuration: [String: Any] {
        [
            "vocab_size": vocabularySize, "hidden_size": 32, "intermediate_size": 48,
            "moe_intermediate_size": 16, "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 4, "routed_scaling_factor": 1.0, "kv_lora_rank": 16,
            "q_lora_rank": 16, "qk_rope_head_dim": 8, "v_head_dim": 8,
            "qk_nope_head_dim": 8, "norm_topk_prob": true, "moe_layer_freq": 1,
            "first_k_dense_replace": 1, "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6, "rope_theta": 10000, "attention_bias": false,
            "n_shared_experts": 1, "n_routed_experts": 4, "n_group": 2, "topk_group": 1,
            "num_experts_per_tok": 2,
        ]
    }

    /// `newCache(parameters:)` gives one cache for each layer.
    @Test func newCacheGivesOneCachePerLayer() throws {
        let cache = try Self.makeModel(seed: 1).newCache(parameters: nil)
        #expect(cache.count == 2, "cache count")
    }

    /// A prompt in chunks and decode steps must give the logits of the full
    /// pass, with the cache of `newCache(parameters:)` and with an explicit
    /// list of `KVCacheSimple`. Each cache must hold each token once.
    @Test func cachedDecodeMatchesTheFullForwardPass() throws {
        let model = try Self.makeModel(seed: 1)
        let row = Self.tokens(count: 11, seed: 1)
        let full = Self.logits(model, [row])
        for cache in [model.newCache(parameters: nil), [KVCacheSimple(), KVCacheSimple()]] {
            var start = 0
            for chunk in [5, 3, 1, 1, 1] {
                let stepped = Self.logits(
                    model, [Array(row[start ..< start + chunk])], cache: cache)
                let difference = Self.maxAbsDifference(
                    stepped, full[0..., start ..< start + chunk, 0...])
                #expect(
                    difference <= Self.tolerance,
                    "positions \(start) ..< \(start + chunk): cached logits differ by \(difference)"
                )
                start += chunk
            }
            #expect(cache.map(\.offset) == [11, 11], "cache offsets")
        }
    }

    /// The original checkpoint stores one tensor per expert.
    /// `sanitize(weights:)` must stack them and remove the per-expert keys,
    /// so that the strict load accepts the checkpoint and gives the logits
    /// of the reference model.
    @Test func loaderStacksPerExpertWeights() throws {
        let reference = try Self.makeModel(seed: 5)
        let checkpoint = Self.splitExperts(
            Dictionary(uniqueKeysWithValues: reference.parameters().flattened()))
        #expect(checkpoint.keys.contains("model.layers.1.mlp.experts.3.up_proj.weight"))
        let loaded = try Self.makeModel(seed: 6)
        let sanitized = loaded.sanitize(weights: checkpoint)
        #expect(
            sanitized["model.layers.1.mlp.switch_mlp.up_proj.weight"]?.shape == [4, 16, 32])
        #expect(!sanitized.keys.contains { $0.contains(".experts.") }, "per-expert keys kept")
        try Self.load(checkpoint, into: loaded)
        let rows = [Self.tokens(count: 11, seed: 3)]
        #expect(
            Self.maxAbsDifference(Self.logits(reference, rows), Self.logits(loaded, rows)) == 0,
            "loaded logits")
    }

    /// An fp8 checkpoint stores each block-quantized weight with a
    /// `weight_scale_inv` key. `sanitize(weights:)` must dequantize the
    /// weight and drop the scale key, as mlx-lm `deepseek_v3.py` does, so
    /// that the strict load accepts the checkpoint (issue #233).
    ///
    /// The test stores each 2-D attention weight `W` as `W / 2` with a
    /// scale of 2. All weights are smaller than one 128 x 128 block, so the
    /// scale has shape `[1, 1]`, and `(W / 2) * 2` is exact in float32.
    @Test func loaderDropsFp8ScaleKeys() throws {
        let reference = try Self.makeModel(seed: 7)
        var checkpoint = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())
        let scaled = checkpoint.keys.filter {
            $0.contains(".self_attn.") && $0.hasSuffix(".weight") && checkpoint[$0]!.ndim == 2
        }
        #expect(scaled.contains("model.layers.0.self_attn.o_proj.weight"))
        for key in scaled {
            checkpoint[key] = checkpoint[key]! / 2
            checkpoint[key + "_scale_inv"] = MLXArray([Float(2)]).reshaped(1, 1)
        }
        let loaded = try Self.makeModel(seed: 8)
        let sanitized = loaded.sanitize(weights: checkpoint)
        #expect(
            !sanitized.keys.contains { $0.contains("weight_scale_inv") },
            "weight_scale_inv keys kept")
        let original = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())
        for key in scaled {
            #expect(
                sanitized[key].map { Self.maxAbsDifference($0, original[key]!) } == 0,
                "\(key) not dequantized")
        }
        try Self.load(checkpoint, into: loaded)
        let rows = [Self.tokens(count: 11, seed: 3)]
        #expect(
            Self.maxAbsDifference(Self.logits(reference, rows), Self.logits(loaded, rows)) == 0,
            "loaded logits")
    }

    // MARK: - Helpers (copied from PR #183 Kernel/Support/)

    /// Builds the tiny model and gives each floating-point parameter seeded
    /// random values: a norm scale near 1, another 1-D parameter near 0, and
    /// a matrix with a standard deviation of `1 / sqrt(fan-in)`.
    private static func makeModel(seed: UInt64) throws -> DeepseekV3Model {
        let data = try JSONSerialization.data(withJSONObject: configuration)
        let model = DeepseekV3Model(
            try JSONDecoder().decode(DeepseekV3Configuration.self, from: data))
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
                let fanIn =
                    name.contains("conv")
                    ? value.shape.dropFirst().reduce(1, *) : value.dim(-1)
                random = noise * (1 / Float(fanIn).squareRoot())
            }
            let dtype: DType = value.dtype == .float64 ? .float32 : value.dtype
            updated.append((name, random.asType(dtype)))
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
        eval(model)
        return model
    }

    /// Token IDs from a fixed linear congruential generator.
    private static func tokens(count: Int, seed: Int) -> [Int] {
        var state = UInt64(truncatingIfNeeded: seed) &+ 0x9E37_79B9
        return (0 ..< count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int((state >> 33) % UInt64(vocabularySize))
        }
    }

    private static func logits(
        _ model: DeepseekV3Model, _ rows: [[Int]], cache: [KVCache]? = nil
    ) -> MLXArray {
        let input = MLXArray(rows.flatMap { $0.map { Int32($0) } })
            .reshaped(rows.count, rows[0].count)
        let output = model(input, cache: cache)
        eval(output)
        return output
    }

    private static func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// Splits each stacked expert tensor `<prefix>switch_mlp.<name>.<suffix>`
    /// with shape `[E, ...]` into `E` tensors
    /// `<prefix>experts.<e>.<name>.<suffix>`.
    private static func splitExperts(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights {
            var matched = false
            for name in ["gate_proj", "up_proj", "down_proj"]
            where key.contains(".switch_mlp.\(name).") {
                for expert in 0 ..< value.dim(0) {
                    let newKey = key.replacingOccurrences(
                        of: ".switch_mlp.\(name).", with: ".experts.\(expert).\(name).")
                    result[newKey] = value[expert]
                }
                matched = true
            }
            if !matched {
                result[key] = value
            }
        }
        return result
    }

    /// Writes `weights` to a `.safetensors` file in a new temporary folder
    /// and loads them into `model` through `loadWeights`, which calls
    /// `sanitize(weights:)` and then updates the model with
    /// `verify: [.all]`. The load fails when a key is missing, a key is not
    /// used, or a shape does not match.
    private static func load(_ weights: [String: MLXArray], into model: DeepseekV3Model) throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("deepseek-v3-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        try MLX.save(arrays: weights, url: folder.appendingPathComponent("model.safetensors"))
        try loadWeights(modelDirectory: folder, model: model)
    }
}
