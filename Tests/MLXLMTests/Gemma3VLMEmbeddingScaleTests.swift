import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// Regression test for issue #204: the Gemma 3 VLM text model cast the
/// embedding scale `sqrt(hidden_size)` to the dtype of the token IDs. A
/// decode step with int32 token IDs scaled the embeddings by `int32(5.66) =
/// 5`. The prompt with the image passes embeddings and got 5.66.
///
/// The model is tiny and has seeded random weights: a text model with
/// hidden size 32 and 6 layers, and a vision tower with 1 layer. A random
/// 8 x 8 image gives 4 image tokens. The test is copied from
/// `decodeAfterThePromptMatchesALongerPrompt` for the "Gemma3" case of the
/// kernel forward-pass tests in PR #186, without the known issue.
@Suite
struct Gemma3VLMEmbeddingScaleTests {

    static let imageToken = 262_144

    static func model(seed: UInt64) throws -> Gemma3 {
        let configuration = try TinyModel.configuration(
            Gemma3Configuration.self,
            [
                "model_type": "gemma3", "mm_tokens_per_image": 4,
                "text_config": [
                    "model_type": "gemma3_text", "hidden_size": 32,
                    "num_hidden_layers": 6, "intermediate_size": 48,
                    "sliding_window": 4, "num_attention_heads": 2,
                    "num_key_value_heads": 1, "head_dim": 16,
                    "query_pre_attn_scalar": 16,
                ],
                "vision_config": [
                    "model_type": "siglip_vision_model", "num_hidden_layers": 1,
                    "hidden_size": 32, "intermediate_size": 48,
                    "num_attention_heads": 4, "patch_size": 2, "image_size": 8,
                ],
            ])
        let model = Gemma3(configuration)
        TinyModel.randomize(model, seed: seed)
        return model
    }

    /// Text, the image tokens, then more text.
    static let prompt = [5, 7] + Array(repeating: imageToken, count: 4) + [9, 11, 13]

    /// Runs `prepare` on `prompt` with the image and returns the logits.
    static func prefill(
        _ model: Gemma3, prompt: [Int], pixels: MLXArray, cache: [KVCache]
    ) throws -> MLXArray {
        let tokens = TinyModel.batch([prompt])
        let input = LMInput(
            text: .init(tokens: tokens, mask: MLXArray.ones(tokens.shape, dtype: .int32)),
            image: .init(pixels: pixels))
        guard case .logits(let output) = try model.prepare(input, cache: cache, windowSize: nil)
        else {
            Issue.record("prepare must return logits")
            return MLXArray.zeros([1])
        }
        eval(output.logits)
        return output.logits
    }

    /// After the prompt, a decode step with the cache gives the logits of
    /// the last position of the same prompt with that token appended.
    ///
    /// Tolerance 1e-4: the paths differ in the order of the attention sums.
    /// The differences are near 1e-6. The defect gave 0.68.
    @Test func decodeAfterThePromptMatchesALongerPrompt() throws {
        let model = try Self.model(seed: 1)
        let pixels = MLXRandom.normal([1, 3, 8, 8], key: MLXRandom.key(1))
        let cache = model.newCache(parameters: nil)
        _ = try Self.prefill(model, prompt: Self.prompt, pixels: pixels, cache: cache)
        let step = model(TinyModel.batch([[17]]), cache: cache)
        eval(step)
        let longer = try Self.prefill(
            model, prompt: Self.prompt + [17], pixels: pixels,
            cache: model.newCache(parameters: nil))
        let difference = TinyModel.maxAbsDifference(step[0..., -1], longer[0..., -1])
        #expect(difference <= 1e-4, "differs by \(difference)")
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
