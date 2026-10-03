import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// Regression test for issue #200: `JambaModel.sanitize(weights:)` wrote
/// the stacked experts under `block_sparse_moe`, but the module path is
/// `feed_forward`, so a checkpoint with one tensor per expert did not load.
///
/// The test is copied from `jambaLoaderStacksPerExpertWeights` in the
/// kernel forward-pass tests of PR #185, without the known issue.
@Suite
struct JambaSanitizeExpertsTests {

    static func model(seed: UInt64) throws -> JambaModel {
        let configuration = try TinyModel.configuration(
            JambaConfiguration.self,
            [
                "model_type": "jamba", "hidden_size": 32, "intermediate_size": 48,
                "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                "attn_layer_offset": 1, "attn_layer_period": 2, "expert_layer_offset": 1,
                "expert_layer_period": 2, "mamba_d_conv": 4, "mamba_d_state": 8,
                "mamba_expand": 2, "num_experts": 4, "num_experts_per_tok": 2,
                "rms_norm_eps": 1e-6, "max_position_embeddings": 256, "vocab_size": 64,
                "tie_word_embeddings": false,
            ])
        let model = JambaModel(configuration)
        TinyModel.randomize(model, seed: seed)
        return model
    }

    /// A Hugging Face Jamba checkpoint stores each expert as
    /// `feed_forward.experts.N.{gate,up,down}_proj`. The loader must stack
    /// them into `feed_forward.switch_mlp`, and the loaded model must give
    /// the logits of the model the checkpoint came from.
    @Test func loaderStacksPerExpertWeights() throws {
        let reference = try Self.model(seed: 5)
        var checkpoint: [String: MLXArray] = [:]
        for (key, value) in reference.parameters().flattened() {
            if key.contains(".switch_mlp.") {
                for expert in 0 ..< value.dim(0) {
                    let perExpert = key.replacingOccurrences(
                        of: ".switch_mlp.", with: ".experts.\(expert).")
                    checkpoint[perExpert] = value[expert]
                }
            } else {
                checkpoint[key] = value
            }
        }
        #expect(checkpoint["model.layers.1.feed_forward.experts.3.down_proj.weight"] != nil)
        #expect(checkpoint["model.layers.1.feed_forward.switch_mlp.down_proj.weight"] == nil)

        let loaded = try Self.model(seed: 6)
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("jamba-sanitize-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        try MLX.save(arrays: checkpoint, url: folder.appendingPathComponent("model.safetensors"))
        // loadWeights calls sanitize(weights:) and then a strict update, so
        // a key under the wrong module path makes it throw.
        try loadWeights(modelDirectory: folder, model: loaded)

        let rows = [TinyModel.tokens(count: 11, vocabularySize: 64, seed: 3)]
        #expect(
            TinyModel.maxAbsDifference(
                TinyModel.logits(reference, rows), TinyModel.logits(loaded, rows)) == 0,
            "loaded logits")
    }
}

/// Tiny-model helpers for this file. They are copied from the kernel test
/// support of PR #183 (`Tests/MLXLMTests/Kernel/Support/SyntheticModel.swift`)
/// and are private, so that this file does not depend on that PR.
private enum TinyModel {

    /// Decodes a model configuration from a JSON dictionary.
    static func configuration<C: Decodable>(_ type: C.Type, _ values: [String: Any]) throws -> C {
        try JSONDecoder().decode(C.self, from: JSONSerialization.data(withJSONObject: values))
    }

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
            updated.append((name, random.asType(value.dtype)))
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

    /// A `[rows, length]` int32 array of token IDs.
    static func batch(_ rows: [[Int]]) -> MLXArray {
        MLXArray(rows.flatMap { $0.map { Int32($0) } }).reshaped(rows.count, rows[0].count)
    }

    static func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// Runs the model on `rows` and returns the logits.
    static func logits(
        _ model: any LanguageModel, _ rows: [[Int]], cache: [KVCache]? = nil
    ) -> MLXArray {
        let output = model(batch(rows), cache: cache)
        eval(output)
        return output
    }
}
