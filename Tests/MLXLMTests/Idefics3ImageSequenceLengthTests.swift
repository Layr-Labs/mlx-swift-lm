import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// Regression tests: `Idefics3Processor` inserted one image token and
/// ignored `image_seq_len`. The model puts the image features at groups of
/// `features per image` image tokens, so with one token it dropped the
/// image.
///
/// `processorOutputUsesTheImage` is copied from
/// `VisionModelExtraPathTests.idefics3ProcessorOutputRunsThroughTheModel` of
/// PR #244, without the known issue. One change: the processor has
/// `image_seq_len` 16, the feature count of the tiny model, where the
/// recording test has 64. The model is the tiny Idefics3 model of that test:
/// image size 384 with patch size 48 (8 x 8 patches) and scale factor 2, so
/// 16 features per image; hidden size 32; seeded random weights.
@Suite
struct Idefics3ImageSequenceLengthTests {

    static let imageToken = 49153

    static func processor(imageSequenceLength: Int) throws -> Idefics3Processor {
        let config = try JSONDecoder().decode(
            Idefics3ProcessorConfiguration.self,
            from: JSONSerialization.data(withJSONObject: [
                "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                "size": ["longest_edge": 384] as [String: Any],
                "image_seq_len": imageSequenceLength,
            ]))
        return Idefics3Processor(
            config, tokenizer: Idefics3SeqLenTokenizer(specials: [:], imageMarker: "<image>"))
    }

    static func model() throws -> Idefics3 {
        let values: [String: Any] = [
            "model_type": "idefics3", "vocab_size": 49160, "scale_factor": 2,
            "image_token_id": imageToken,
            "text_config": [
                "model_type": "llama", "hidden_size": 32, "intermediate_size": 48,
                "num_attention_heads": 4, "rms_norm_eps": 1e-6, "vocab_size": 49160,
                "num_key_value_heads": 2, "rope_theta": 10000, "num_hidden_layers": 2,
            ] as [String: Any],
            "vision_config": [
                "model_type": "idefics3_vision", "hidden_size": 32, "num_attention_heads": 4,
                "patch_size": 48, "image_size": 384, "num_hidden_layers": 1,
                "intermediate_size": 48,
            ] as [String: Any],
        ]
        let configuration = try JSONDecoder().decode(
            Idefics3Configuration.self, from: JSONSerialization.data(withJSONObject: values))
        let model = Idefics3(configuration)
        Idefics3SeqLenTinyModel.randomize(model, seed: 1)
        return model
    }

    static func image(_ color: CIColor) -> UserInput.Image {
        .ciImage(CIImage(color: color).cropped(to: CGRect(x: 0, y: 0, width: 20, height: 10)))
    }

    /// Runs `prepare` with a new cache and returns the logits.
    static func run(_ model: Idefics3, _ input: LMInput) throws -> MLXArray {
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

    /// The processor inserts `image_seq_len` image tokens at the middle of
    /// the prompt tokens.
    @Test func processorInsertsImageSequenceLengthTokens() throws {
        let tokenizer = Idefics3SeqLenTokenizer(specials: [:], imageMarker: "<image>")
        let output = try Self.processor(imageSequenceLength: 16).prepare(
            input: UserInput(prompt: "hello", images: [Self.image(.white)]))
        var expected = tokenizer.encode(text: "hello", addSpecialTokens: false)
        expected.insert(contentsOf: Array(repeating: Self.imageToken, count: 16), at: 2)
        #expect(
            output.text.tokens.asArray(Int.self) == expected,
            "Idefics3: 16 image tokens at count / 2")
        #expect(output.image?.pixels.shape == [1, 384, 384, 3], "Idefics3: NHWC pixels")
    }

    /// A white and a black image through the processor and the model give
    /// other logits: the model uses the image of the processor output.
    @Test func processorOutputUsesTheImage() throws {
        let model = try Self.model()
        let processor = try Self.processor(imageSequenceLength: 16)
        let white = try processor.prepare(
            input: UserInput(prompt: "hi", images: [Self.image(.white)]))
        let black = try processor.prepare(
            input: UserInput(prompt: "hi", images: [Self.image(.black)]))
        let whiteLogits = try Self.run(model, white)
        let blackLogits = try Self.run(model, black)
        #expect(
            isFinite(whiteLogits).all().item(Bool.self),
            "Idefics3: processor output gives finite logits")
        let difference = Idefics3SeqLenTinyModel.maxAbsDifference(
            whiteLogits[0..., -1], blackLogits[0..., -1])
        #expect(difference > 1e-3, "Idefics3: the image of the processor output is used")
    }
}

// MARK: - Helpers

// Private copies, under other names, of helpers from the kernel test support
// of PR #183 (`SyntheticModel.swift`) and of `VisionProcessorTests.ScriptTokenizer`
// of PR #244, so that this file does not depend on those PRs or collide
// with them.

private enum Idefics3SeqLenTinyModel {

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
private struct Idefics3SeqLenTokenizer: MLXLMCommon.Tokenizer {
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
