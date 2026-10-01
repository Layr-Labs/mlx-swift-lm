import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// Regression tests for two defects in `Pixtral.swift`:
///
/// - `PixtralProcessor` transposed the `[1, C, H, W]` pixels of
///   `MediaProcessing.asMLXArray` again when the width was 3, which gave
///   `[1, W, C, H]`.
/// - With more than one image in the pixel batch, the merge of the image
///   features into the prompt sliced the features per image but kept the
///   text batch of 1, and the concatenation stopped on a batch-size
///   mismatch.
///
/// `processorKeepsTheLayoutOfAThreePixelWideImage` is copied from
/// `VisionProcessorTests.pixtralKeepsTheLayoutOfAThreePixelWideImage` of
/// PR #244, without the known issue. The model test uses the tiny Pixtral
/// configuration of the same PR (`VisionModelExtraPathTests.pixtral`):
/// hidden size 32, 2 text layers, a vision tower with patch size 2 and
/// image size 8, seeded random weights.
@Suite
struct PixtralPixelLayoutAndImageBatchTests {

    static let imageToken = 60

    // MARK: - Processor

    @Test func processorKeepsTheLayoutOfAThreePixelWideImage() throws {
        let config = try JSONDecoder().decode(
            PixtralProcessorConfiguration.self,
            from: JSONSerialization.data(withJSONObject: [
                "image_processor": [
                    "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                    "size": ["longest_edge": 6] as [String: Any], "patch_size": 3,
                ] as [String: Any],
                "image_token": "[IMG]", "patch_size": 3,
            ]))
        let processor = PixtralProcessor(
            config, tokenizer: PixtralLayoutTokenizer(specials: [:], imageMarker: "<image>"))
        let image = UserInput.Image.ciImage(
            CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 3, height: 6)))
        let output = try processor.prepare(input: UserInput(prompt: "x", images: [image]))
        #expect(
            output.image?.frames?.map { [$0.t, $0.h, $0.w] } == [[1, 6, 3]],
            "Pixtral: 3 x 6 frame")
        #expect(
            output.image?.pixels.shape == [1, 3, 6, 3],
            "Pixtral: pixel shape for a 3-pixel-wide image")
    }

    // MARK: - Two images in one prompt

    static func model() throws -> PixtralVLM {
        let values: [String: Any] = [
            "model_type": "pixtral", "image_token_index": imageToken, "vocab_size": 64,
            "text_config": [
                "model_type": "mistral", "hidden_size": 32, "num_hidden_layers": 2,
                "intermediate_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 8, "rms_norm_eps": 1e-6, "vocab_size": 64, "rope_theta": 10000,
            ] as [String: Any],
            "vision_config": [
                "model_type": "pixtral", "hidden_size": 32, "num_hidden_layers": 1,
                "num_attention_heads": 4, "intermediate_size": 48, "patch_size": 2,
                "image_size": 8, "head_dim": 8,
            ] as [String: Any],
        ]
        let configuration = try JSONDecoder().decode(
            PixtralConfiguration.self, from: JSONSerialization.data(withJSONObject: values))
        let model = PixtralVLM(configuration)
        PixtralBatchTinyModel.randomize(model, seed: 1)
        return model
    }

    /// Runs `prepare` with a new cache and returns the logits.
    static func prefill(_ model: PixtralVLM, _ prompt: [Int], pixels: MLXArray) throws -> MLXArray {
        let tokens = MLXArray(prompt.map { Int32($0) }).reshaped(1, prompt.count)
        let input = LMInput(
            text: .init(tokens: tokens, mask: MLXArray.ones(tokens.shape, dtype: .int32)),
            image: LMInput.ProcessedImage(pixels: pixels, frames: nil))
        guard
            case .logits(let output) = try model.prepare(
                input, cache: model.newCache(parameters: nil), windowSize: nil)
        else {
            Issue.record("prepare must return logits")
            return MLXArray.zeros([1])
        }
        eval(output.logits)
        return output.logits
    }

    static func pixels(seed: UInt64) -> MLXArray {
        let x = MLXRandom.normal([1, 3, 8, 8], key: MLXRandom.key(seed))
        eval(x)
        return x
    }

    /// Two 8 x 8 images give 16 features each. The prompt has a block of 16
    /// image tokens for each image. The features of the second image go to
    /// the second block: another second image changes the logits after the
    /// second block, and not the logits before it.
    @Test func twoImagesFillTheirOwnImageTokens() throws {
        let model = try Self.model()
        let block = Array(repeating: Self.imageToken, count: 16)
        let prompt = [5] + block + [7] + block + [9, 11]
        let first = Self.pixels(seed: 1)
        let a = try Self.prefill(
            model, prompt, pixels: concatenated([first, Self.pixels(seed: 2)], axis: 0))
        let b = try Self.prefill(
            model, prompt, pixels: concatenated([first, Self.pixels(seed: 3)], axis: 0))

        #expect(a.shape == [1, prompt.count, 64], "two-image logits shape")
        #expect(isFinite(a).all().item(Bool.self), "two-image logits are finite")
        // Position 17 is the token 7 between the two blocks.
        let before = PixtralBatchTinyModel.maxAbsDifference(a[0..., ..<18], b[0..., ..<18])
        let after = PixtralBatchTinyModel.maxAbsDifference(a[0..., -1], b[0..., -1])
        #expect(before <= 1e-4, "the second image changed the logits before it by \(before)")
        #expect(after > 1e-3, "the second image must change the logits after it")
    }
}

// MARK: - Helpers

// Private copies, under other names, of helpers from the kernel test support
// of PR #183 (`SyntheticModel.swift`) and of `VisionProcessorTests.ScriptTokenizer`
// of PR #244, so that this file does not depend on those PRs or collide
// with them.

private enum PixtralBatchTinyModel {

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
            let dtype: DType = value.dtype == .float64 ? .float32 : value.dtype
            updated.append((name, random.asType(dtype)))
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
        eval(model)
    }

    static func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// Runs the model on `rows` and returns the logits.
    static func logits(_ model: any LanguageModel, _ rows: [[Int]], cache: [KVCache]?)
        -> MLXArray
    {
        let batch = MLXArray(rows.flatMap { $0.map { Int32($0) } })
            .reshaped(rows.count, rows[0].count)
        let output = model(batch, cache: cache)
        eval(output)
        return output
    }
}

/// A small tokenizer: a special string is one token, each other Unicode
/// scalar is the token `1000 + scalar value`. The chat template writes
/// `role:`, the content with `imageMarker` for each image part, and a new
/// line after each message.
private struct PixtralLayoutTokenizer: MLXLMCommon.Tokenizer {
    let specials: [String: Int]
    let imageMarker: String

    init(specials: [String: Int], imageMarker: String) {
        self.specials = specials.merging(["<s>": 1]) { current, _ in current }
        self.imageMarker = imageMarker
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var ids: [Int] = []
        let scalars = Array(text.unicodeScalars)
        let keys = specials.map { (Array($0.key.unicodeScalars), $0.value) }
            .sorted { $0.0.count > $1.0.count }
        var i = 0
        while i < scalars.count {
            let match = keys.first { key in
                !key.0.isEmpty && i + key.0.count <= scalars.count
                    && Array(scalars[i ..< i + key.0.count]) == key.0
            }
            if let match {
                ids.append(match.1)
                i += match.0.count
            } else {
                ids.append(1000 + Int(scalars[i].value))
                i += 1
            }
        }
        return ids
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        var text = ""
        for id in tokenIds {
            let isSpecial = specials.contains { $0.value == id }
            if isSpecial && skipSpecialTokens { continue }
            text += convertIdToToken(id) ?? ""
        }
        return text
    }

    func convertTokenToId(_ token: String) -> Int? { specials[token] }

    func convertIdToToken(_ id: Int) -> String? {
        if let special = specials.first(where: { $0.value == id }) {
            return special.key
        }
        guard id >= 1000, let scalar = Unicode.Scalar(UInt32(id - 1000)) else { return nil }
        return String(Character(scalar))
    }

    var bosToken: String? { "<s>" }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        var text = ""
        for message in messages {
            text += (message["role"] as? String ?? "") + ":"
            if let content = message["content"] as? String {
                text += content
            } else if let parts = message["content"] as? [[String: Any]] {
                for part in parts {
                    switch part["type"] as? String {
                    case "image": text += imageMarker
                    case "text": text += part["text"] as? String ?? ""
                    default: break
                    }
                }
            }
            text += "\n"
        }
        return encode(text: text, addSpecialTokens: false)
    }
}
