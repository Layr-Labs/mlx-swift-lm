import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// Regression tests for two defects in `Mistral3.swift`:
///
/// - `Mistral3VLMProcessor` transposed the `[1, C, H, W]` pixels of
///   `MediaProcessing.asMLXArray` again when the width was 3, which gave
///   `[1, W, C, H]`.
/// - The language model indexed an empty cache array when it ran without a
///   cache, and the sliding-window layers got no mask without a cache.
///
/// `processorKeepsTheLayoutOfAThreePixelWideImage` is copied from
/// `VisionProcessorTests.mistral3KeepsTheLayoutOfAThreePixelWideImage` of
/// PR #244, without the known issue. The model tests use the tiny Mistral3
/// configuration of the same PR (`VisionModelExtraPathTests.mistral3`):
/// hidden size 32, 2 text layers, seeded random weights.
@Suite
struct Mistral3PixelLayoutAndMaskTests {

    /// Tolerance 1e-4: the compared passes differ only in the order of the
    /// attention sums (float32). The differences are near 1e-6.
    static let tolerance: Float = 1e-4

    // MARK: - Processor

    @Test func processorKeepsTheLayoutOfAThreePixelWideImage() async throws {
        let t = Mistral3LayoutTokenizer(specials: ["[IMG]": 20], imageMarker: "[IMG]")
        let config = try JSONDecoder().decode(
            Mistral3VLMProcessorConfiguration.self,
            from: JSONSerialization.data(withJSONObject: [
                "image_processor": [
                    "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                    "size": ["longest_edge": 6] as [String: Any], "patch_size": 3,
                ] as [String: Any],
                "image_token": "[IMG]", "patch_size": 3, "spatial_merge_size": 1,
            ]))
        let processor = Mistral3VLMProcessor(config, tokenizer: t)
        let image = UserInput.Image.ciImage(
            CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 3, height: 6)))
        let output = try await processor.prepare(input: UserInput(prompt: "x", images: [image]))
        #expect(
            output.image?.frames?.map { [$0.t, $0.h, $0.w] } == [[1, 6, 3]],
            "Mistral3: 3 x 6 frame")
        #expect(
            output.image?.pixels.shape == [1, 3, 6, 3],
            "Mistral3: pixel shape for a 3-pixel-wide image")
    }

    // MARK: - Language model without a cache

    static func model(layerTypes: [String]? = nil, slidingWindow: Int? = nil) throws
        -> Mistral3VLM
    {
        var text: [String: Any] = [
            "model_type": "ministral3", "hidden_size": 32, "num_hidden_layers": 2,
            "intermediate_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "rms_norm_eps": 1e-6, "vocab_size": 64,
            "rope_parameters": ["rope_type": "default", "rope_theta": 10000] as [String: Any],
        ]
        if let layerTypes { text["layer_types"] = layerTypes }
        if let slidingWindow { text["sliding_window"] = slidingWindow }
        let values: [String: Any] = [
            "model_type": "mistral3", "image_token_index": 60, "vocab_size": 64,
            "spatial_merge_size": 2, "text_config": text,
            "vision_config": [
                "model_type": "pixtral", "hidden_size": 32, "num_hidden_layers": 1,
                "num_attention_heads": 4, "intermediate_size": 48, "patch_size": 2,
                "image_size": 8, "head_dim": 8,
            ] as [String: Any],
        ]
        let configuration = try JSONDecoder().decode(
            Mistral3VLMConfiguration.self, from: JSONSerialization.data(withJSONObject: values))
        let model = Mistral3VLM(configuration)
        Mistral3MaskTinyModel.randomize(model, seed: 1)
        return model
    }

    /// The model runs without a cache, and gives the logits of the same
    /// prompt run in one call with a new cache. The second case has a
    /// sliding-window layer with a window (4) shorter than the prompt (9).
    @Test(arguments: [false, true])
    func aPassWithoutACacheMatchesAPassWithANewCache(sliding: Bool) throws {
        let model =
            sliding
            ? try Self.model(
                layerTypes: ["sliding_attention", "full_attention"], slidingWindow: 4)
            : try Self.model()
        let prompt = [[5, 7, 9, 11, 13, 15, 17, 19, 21]]
        let withoutCache = Mistral3MaskTinyModel.logits(model, prompt, cache: nil)
        let withCache = Mistral3MaskTinyModel.logits(
            model, prompt, cache: model.newCache(parameters: nil))
        #expect(withoutCache.shape == [1, 9, 64], "logits shape without a cache")
        let difference = Mistral3MaskTinyModel.maxAbsDifference(withoutCache, withCache)
        #expect(
            difference <= Self.tolerance,
            "sliding \(sliding): the pass without a cache differs by \(difference)")
    }

    /// Without a cache, a change at position 6 must not change the logits
    /// of positions 0 to 5.
    @Test(arguments: [false, true])
    func aPassWithoutACacheIsCausal(sliding: Bool) throws {
        let model =
            sliding
            ? try Self.model(
                layerTypes: ["sliding_attention", "full_attention"], slidingWindow: 4)
            : try Self.model()
        let row = [5, 7, 9, 11, 13, 15, 17, 19, 21]
        var changed = row
        changed[6] = 40
        let original = Mistral3MaskTinyModel.logits(model, [row], cache: nil)
        let modified = Mistral3MaskTinyModel.logits(model, [changed], cache: nil)
        let before = Mistral3MaskTinyModel.maxAbsDifference(
            original[0..., ..<6], modified[0..., ..<6])
        let after = Mistral3MaskTinyModel.maxAbsDifference(
            original[0..., 6...], modified[0..., 6...])
        #expect(before <= Self.tolerance, "positions before 6 changed by \(before)")
        #expect(after > 1e-3, "the change at 6 must change its own logits")
    }
}

// MARK: - Helpers

// Private copies, under other names, of helpers from the kernel test support
// of PR #183 (`SyntheticModel.swift`) and of `VisionProcessorTests.ScriptTokenizer`
// of PR #244, so that this file does not depend on those PRs or collide
// with them.

private enum Mistral3MaskTinyModel {

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
private struct Mistral3LayoutTokenizer: MLXLMCommon.Tokenizer {
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
