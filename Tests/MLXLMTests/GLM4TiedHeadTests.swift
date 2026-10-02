import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// Regression tests for issue #271: with `tie_word_embeddings` true,
/// `sanitize(weights:)` of GLM-4 dropped `lm_head.weight`, but the model
/// always has its own head, so the strict load failed.
///
/// The model is tiny and has seeded random weights: hidden size 32, 2
/// layers, 4 query heads of size 8 over 2 key heads, vocabulary 64, partial
/// RoPE. The check is copied from the loading check of the "GLM-4 tied"
/// case of the kernel forward-pass tests in PR #236, without the known
/// issue.
@Suite
struct GLM4TiedHeadTests {

    static let vocabularySize = 64

    static func model(seed: UInt64) throws -> GLM4Model {
        let configuration = try TinyModel.configuration(
            GLM4Configuration.self,
            [
                "model_type": "glm4", "hidden_size": 32, "num_hidden_layers": 2,
                "intermediate_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2,
                "attention_bias": false, "head_dim": 8, "rms_norm_eps": 1e-6,
                "vocab_size": vocabularySize, "partial_rotary_factor": 0.5,
                "tie_word_embeddings": true,
            ])
        let model = GLM4Model(configuration)
        TinyModel.randomize(model, seed: seed)
        return model
    }

    /// A checkpoint of a tied model has its own `lm_head.weight`, as in the
    /// reference. It loads through `loadWeights`, and the loaded model gives
    /// the logits of the model that wrote it, exactly. A head with a wrong
    /// shape is rejected.
    @Test func tiedCheckpointLoadsItsHead() throws {
        let reference = try Self.model(seed: 5)
        let checkpoint = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())
        #expect(checkpoint["lm_head.weight"]?.shape == [64, 32])

        let loaded = try Self.model(seed: 6)
        try TinyModel.load(checkpoint, into: loaded)
        let rows = [TinyModel.tokens(count: 11, vocabularySize: Self.vocabularySize, seed: 3)]
        #expect(
            TinyModel.maxAbsDifference(
                TinyModel.logits(reference, rows), TinyModel.logits(loaded, rows)) == 0,
            "loaded logits")

        var wrong = checkpoint
        wrong["lm_head.weight"] = MLXArray.zeros([65, 32])
        #expect(throws: (any Error).self, "lm_head.weight with a wrong shape") {
            try TinyModel.load(wrong, into: try Self.model(seed: 6))
        }
    }
}

/// Tiny-model helpers for this file. They are copied from the kernel test
/// support of PR #183 (`Tests/MLXLMTests/Kernel/Support/SyntheticModel.swift`)
/// and are private, so that this file does not depend on that PR.
private enum TinyModel {

    /// Decodes a model configuration from a JSON dictionary. A key in
    /// `overrides` replaces the same key in `base`.
    static func configuration<C: Decodable>(
        _ type: C.Type, _ base: [String: Any], overrides: [String: Any] = [:]
    ) throws -> C {
        let merged = base.merging(overrides) { _, new in new }
        return try JSONDecoder().decode(
            C.self, from: JSONSerialization.data(withJSONObject: merged))
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

extension TinyModel {

    /// Writes `weights` to a `.safetensors` file in a new temporary folder,
    /// loads them into `model` through `loadWeights`, and deletes the
    /// folder. The loader calls `sanitize(weights:)` and then updates the
    /// model with `verify: [.all]`, so the load fails when a key is missing,
    /// a key is not used, or a shape does not match.
    static func load(_ weights: [String: MLXArray], into model: any LanguageModel) throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("tiny-model-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        try MLX.save(arrays: weights, url: folder.appendingPathComponent("model.safetensors"))
        try loadWeights(modelDirectory: folder, model: model)
    }
}
