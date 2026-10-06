import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

extension KernelTests {

    /// Forward-pass tests of `PaliGemma` with a tiny random model and a
    /// synthetic 16 x 16 image: a SigLIP tower with patch size 4 (16 image
    /// tokens), a linear projector and a 2-layer Gemma text model.
    @Suite
    struct PaliGemmaForwardPassTests {

        static let vocabularySize = 64
        static let imageToken = 60
        static let padToken = 0

        static var base: [String: Any] {
            [
                "model_type": "paligemma", "vocab_size": vocabularySize, "ignore_index": -100,
                "image_token_index": imageToken, "hidden_size": 32, "pad_token_id": padToken,
                "text_config": [
                    "model_type": "gemma", "hidden_size": 32, "num_hidden_layers": 2,
                    "intermediate_size": 64, "num_attention_heads": 4,
                    "num_key_value_heads": 1, "vocab_size": vocabularySize,
                ] as [String: Any],
                "vision_config": [
                    "model_type": "siglip_vision_model", "hidden_size": 32,
                    "num_hidden_layers": 2, "intermediate_size": 64, "num_attention_heads": 4,
                    "patch_size": 4, "projection_dim": 32, "image_size": 16,
                ] as [String: Any],
            ]
        }

        static func makeModel(seed: UInt64 = 1) throws -> PaliGemma {
            let model = PaliGemma(
                try SyntheticModel.configuration(PaliGemmaConfiguration.self, base))
            SyntheticModel.randomize(model, seed: seed)
            return model
        }

        /// 16 image tokens, then text tokens.
        static func prompt(_ text: [Int] = [5, 7, 9, 11]) -> [Int] {
            Array(repeating: imageToken, count: 16) + text
        }

        static func pixels(seed: UInt64) -> MLXArray {
            let x = MLXRandom.normal([1, 3, 16, 16], key: MLXRandom.key(seed))
            eval(x)
            return x
        }

        static func prefill(
            _ model: PaliGemma, prompt: [Int], pixels: MLXArray, cache: [KVCache]? = nil
        ) throws -> MLXArray {
            let tokens = SyntheticModel.batch([prompt])
            let input = LMInput(
                text: .init(tokens: tokens, mask: MLXArray.ones(tokens.shape, dtype: .int32)),
                image: .init(pixels: pixels))
            let result = try model.prepare(
                input, cache: cache ?? model.newCache(parameters: nil), windowSize: nil)
            guard case .logits(let output) = result else {
                Issue.record("prepare must return logits")
                return MLXArray.zeros([1])
            }
            eval(output.logits)
            return output.logits
        }

        @Test func prefillLogitsHaveTheExpectedShapeAndAreFinite() throws {
            let model = try Self.makeModel()
            let logits = try Self.prefill(
                model, prompt: Self.prompt(), pixels: Self.pixels(seed: 1))
            #expect(logits.shape == [1, 20, Self.vocabularySize])
            #expect(isFinite(logits).all().item(Bool.self))
        }

        @Test func theImageChangesTheTextLogits() throws {
            let model = try Self.makeModel()
            let a = try Self.prefill(model, prompt: Self.prompt(), pixels: Self.pixels(seed: 1))
            let b = try Self.prefill(model, prompt: Self.prompt(), pixels: Self.pixels(seed: 2))
            #expect(SyntheticModel.maxAbsDifference(a[0..., 16...], b[0..., 16...]) > 1e-3)
            let again = try Self.prefill(model, prompt: Self.prompt(), pixels: Self.pixels(seed: 1))
            #expect(SyntheticModel.maxAbsDifference(a, again) == 0)
        }

        /// The image features replace the embeddings of the image tokens, so
        /// the embedding row of the image token has no effect.
        @Test func imageTokenEmbeddingIsReplacedByImageFeatures() throws {
            let model = try Self.makeModel()
            let before = try Self.prefill(
                model, prompt: Self.prompt(), pixels: Self.pixels(seed: 1))
            let key = "language_model.model.embed_tokens.weight"
            let table = SyntheticModel.flatParameters(model)[key]!
            let changed = concatenated(
                [
                    table[..<Self.imageToken], table[Self.imageToken ..< Self.imageToken + 1] + 5,
                    table[(Self.imageToken + 1)...],
                ], axis: 0)
            model.update(parameters: ModuleParameters.unflattened([key: changed]))
            eval(model)
            // The tied head reads the changed row, so compare every logit
            // but the image token's own.
            let after = try Self.prefill(model, prompt: Self.prompt(), pixels: Self.pixels(seed: 1))
            let columns = MLXArray(
                (0 ..< Self.vocabularySize).filter { $0 != Self.imageToken }.map { Int32($0) })
            #expect(
                SyntheticModel.maxAbsDifference(
                    take(before, columns, axis: -1), take(after, columns, axis: -1)) <= 1e-5)
        }

        @Test func decodeAfterThePromptUsesTheCache() throws {
            let model = try Self.makeModel()
            let cache = model.newCache(parameters: nil)
            _ = try Self.prefill(
                model, prompt: Self.prompt(), pixels: Self.pixels(seed: 1), cache: cache)
            #expect(cache.allSatisfy { $0.offset == 20 })
            let step = ForwardPassChecks.logits(model, [[13]], cache: cache)
            #expect(step.shape == [1, 1, Self.vocabularySize])
            #expect(isFinite(step).all().item(Bool.self))
            #expect(cache.allSatisfy { $0.offset == 21 })
        }

        /// A checkpoint stores the patch embedding in the PyTorch layout
        /// `[out, in, kH, kW]` and has `position_ids`. The sanitizer moves
        /// the channels last and drops the position IDs.
        @Test func loaderConvertsThePatchEmbedding() throws {
            let reference = try Self.makeModel(seed: 5)
            var checkpoint = SyntheticModel.flatParameters(reference)
            let key = "vision_tower.vision_model.embeddings.patch_embedding.weight"
            #expect(checkpoint[key]?.shape == [32, 4, 4, 3])
            checkpoint[key] = checkpoint[key]!.transposed(0, 3, 1, 2)
            checkpoint["vision_tower.vision_model.embeddings.position_ids"] = MLXArray(0 ..< 16)
            let loaded = try Self.makeModel(seed: 6)
            try SyntheticModel.load(checkpoint, into: loaded)
            let pixels = Self.pixels(seed: 1)
            #expect(
                SyntheticModel.maxAbsDifference(
                    try Self.prefill(reference, prompt: Self.prompt(), pixels: pixels),
                    try Self.prefill(loaded, prompt: Self.prompt(), pixels: pixels)) == 0)
        }
    }
}
