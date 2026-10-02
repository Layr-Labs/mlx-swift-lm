import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// Regression tests for issue #265: without shared embeddings, the OpenELM
/// head took `num_transformer_layers` inputs instead of `model_dim`.
///
/// The model is tiny and has seeded random weights: model dimension 48,
/// head size 8, 2 layers, vocabulary 64. The shape checks are copied from
/// `openELMConfigurationShapesTheLayers` of the kernel forward-pass tests
/// in PR #235, without the known issue. The forward-pass check is new: the
/// recording test could not run the untied model, because the wrong head
/// shape made it crash.
@Suite
struct OpenELMUntiedHeadTests {

    static let vocabularySize = 64

    static func model(_ overrides: [String: Any], seed: UInt64) throws -> OpenELMModel {
        let configuration = try TinyModel.configuration(
            OpenElmConfiguration.self,
            [
                "model_type": "openelm", "head_dim": 8, "num_transformer_layers": 2,
                "model_dim": 48, "vocab_size": vocabularySize, "ffn_dim_divisor": 8,
            ], overrides: overrides)
        let model = OpenELMModel(configuration)
        TinyModel.randomize(model, seed: seed)
        return model
    }

    /// Without q/k norm, OpenELM has no norm weights. Without shared
    /// embeddings, the head maps the model dimension to the vocabulary.
    @Test func configurationShapesTheLayers() throws {
        let plain = Self.parameters(try Self.model(["normalize_qk_projections": false], seed: 1))
        #expect(plain["transformer.layers.0.attn.q_norm.weight"] == nil)
        #expect(plain["transformer.layers.1.attn.k_norm.weight"] == nil)

        let untied = Self.parameters(try Self.model(["share_input_output_layers": false], seed: 1))
        #expect(untied["lm_head.weight"]?.shape == [64, 48], "lm_head shape")
    }

    /// The untied model runs: the logits have the shape `[2, 7, 64]` and
    /// are finite.
    @Test func untiedModelRunsTheForwardPass() throws {
        let model = try Self.model(["share_input_output_layers": false], seed: 1)
        let rows = [1, 2].map {
            TinyModel.tokens(count: 7, vocabularySize: Self.vocabularySize, seed: $0)
        }
        let logits = TinyModel.logits(model, rows)
        #expect(logits.shape == [2, 7, Self.vocabularySize])
        #expect(isFinite(logits).all().item(Bool.self))
    }

    static func parameters(_ model: Module) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: model.parameters().flattened())
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

    /// Runs the model on `rows` and returns the logits.
    static func logits(
        _ model: any LanguageModel, _ rows: [[Int]], cache: [KVCache]? = nil
    ) -> MLXArray {
        let output = model(batch(rows), cache: cache)
        eval(output)
        return output
    }
}
