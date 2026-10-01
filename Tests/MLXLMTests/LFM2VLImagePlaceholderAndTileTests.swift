import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// Regression tests for two defects in `LFM2VLProcessor.prepare`
/// (`LFM2VL.swift`):
///
/// - Image placeholder tokens that were next to each other counted as one
///   placeholder. Two images in one message got the tokens of the first
///   image only, and the model then stopped on a token count mismatch.
/// - Two images with different tile counts have different patch counts,
///   and the concatenation of their pixels stopped.
///
/// `adjacentImagePlaceholdersGiveTheTokensOfEachImage` is copied from
/// `VisionProcessorTests.lfm2vlExpandsAdjacentImagePlaceholders` of PR #244,
/// without the known issue. The other tests are new: PR #244 avoids their
/// input because it crashed. The processor has tile size 8, patch size 2,
/// at most 4 tiles and downsample factor 2. The model is the tiny LFM2VL
/// model of PR #244 (`VisionModelExtraPathTests.lfm2vl`): hidden size 32,
/// a convolution layer and an attention layer, seeded random weights.
@Suite
struct LFM2VLImagePlaceholderAndTileTests {

    static let imageToken = 396

    static func processor() throws -> LFM2VLProcessor {
        let config = try JSONDecoder().decode(
            LFM2VLProcessorConfiguration.self,
            from: JSONSerialization.data(withJSONObject: [
                "tile_size": 8, "encoder_patch_size": 2, "max_tiles": 4,
                "downsample_factor": 2,
            ]))
        return LFM2VLProcessor(
            config,
            tokenizer: LFM2VLImagesTokenizer(
                specials: ["<image>": imageToken], imageMarker: "<image>"))
    }

    static func model() throws -> LFM2VL {
        let values: [String: Any] = [
            "model_type": "lfm2_vl", "downsample_factor": 2, "image_token_id": imageToken,
            "projector_hidden_size": 32,
            "text_config": [
                "model_type": "lfm2", "hidden_size": 32, "num_hidden_layers": 2,
                "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 1200,
                "block_ff_dim": 48, "block_auto_adjust_ff_dim": false,
                "layer_types": ["conv", "full_attention"],
            ] as [String: Any],
            "vision_config": [
                "model_type": "siglip2_vision_model", "hidden_size": 32,
                "intermediate_size": 48, "num_hidden_layers": 1, "num_attention_heads": 4,
                "patch_size": 2, "num_patches": 16,
            ] as [String: Any],
        ]
        let configuration = try JSONDecoder().decode(
            LFM2VLConfiguration.self, from: JSONSerialization.data(withJSONObject: values))
        let model = LFM2VL(configuration)
        LFM2VLImagesTinyModel.randomize(model, seed: 1)
        return model
    }

    static func image(_ width: Int, _ height: Int) -> UserInput.Image {
        .ciImage(
            CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: width, height: height)))
    }

    /// Runs `prepare` with a new cache and returns the logits.
    static func run(_ model: LFM2VL, _ input: LMInput) throws -> MLXArray {
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

    /// Two images in one message give two adjacent placeholders. Each
    /// 8 x 8 image has 4 x 4 patches and 2 x 2 = 4 image tokens.
    @Test func adjacentImagePlaceholdersGiveTheTokensOfEachImage() async throws {
        let output = try await Self.processor().prepare(
            input: UserInput(prompt: "hi", images: [Self.image(8, 8), Self.image(8, 8)]))
        #expect(output.image?.frames?.count == 2, "LFM2VL: two frames")
        let imageTokens = output.text.tokens.asArray(Int.self).filter { $0 == Self.imageToken }
        #expect(
            imageTokens.count == 8, "LFM2VL: adjacent image placeholders give 8 tokens")
    }

    /// The output of two adjacent placeholders runs through the model.
    @Test func adjacentImagePlaceholdersRunThroughTheModel() async throws {
        let output = try await Self.processor().prepare(
            input: UserInput(prompt: "hi", images: [Self.image(8, 8), Self.image(8, 8)]))
        let logits = try Self.run(try Self.model(), output)
        #expect(isFinite(logits).all().item(Bool.self), "LFM2VL: logits are finite")
    }

    /// An 8 x 8 image is 1 tile (4 x 4 patches, 4 tokens). A 16 x 8 image is
    /// 2 x 1 tiles (4 x 8 patches, 8 tokens). The pixels of the first image
    /// are padded to 32 patches, and the model runs on the output.
    @Test func imagesWithDifferentTileCountsArePaddedAndRun() async throws {
        let input = UserInput(chat: [
            .user("a", images: [Self.image(8, 8)]), .user("b", images: [Self.image(16, 8)]),
        ])
        let output = try await Self.processor().prepare(input: input)
        #expect(output.image?.pixels.shape == [2, 32, 12], "LFM2VL: padded pixels")
        #expect(
            output.image?.frames?.map { [$0.t, $0.h, $0.w] } == [[1, 4, 4], [1, 4, 8]],
            "LFM2VL: one frame per image")
        let imageTokens = output.text.tokens.asArray(Int.self).filter { $0 == Self.imageToken }
        #expect(imageTokens.count == 12, "LFM2VL: 4 + 8 image tokens")
        // The padding is zero.
        let padding = try #require(output.image?.pixels)[0, 16..., 0...]
        #expect(abs(padding).max().item(Float.self) == 0, "LFM2VL: zero padding patches")

        let logits = try Self.run(try Self.model(), output)
        #expect(isFinite(logits).all().item(Bool.self), "LFM2VL: logits are finite")
    }
}

// MARK: - Helpers

// Private copies, under other names, of helpers from the kernel test support
// of PR #183 (`SyntheticModel.swift`) and of `VisionProcessorTests.ScriptTokenizer`
// of PR #244, so that this file does not depend on those PRs or collide
// with them.

private enum LFM2VLImagesTinyModel {

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
private struct LFM2VLImagesTokenizer: MLXLMCommon.Tokenizer {
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
