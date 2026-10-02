import CoreImage
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

extension KernelTests {

    /// Tests of the `prepare(input:)` paths of the vision processors of
    /// Mistral3, Pixtral, LFM2VL, Gemma3, Idefics3 and FastVLM.
    ///
    /// The images are small uniform `CIImage` values. The tokenizer is
    /// `ScriptTokenizer`: it gives one token per Unicode scalar and one token
    /// per special string, so the expected token lists are exact.
    @Suite
    struct VisionProcessorTests {

        /// A small tokenizer for the processor tests.
        ///
        /// - A special string (for example `"[IMG]"`) is one token with the
        ///   ID in `specials`. `"<s>"` is always the special token 1.
        /// - Each other Unicode scalar is one token with the ID
        ///   `1000 + scalar value`.
        /// - `decode` gives back the same text, so a processor that decodes
        ///   the prompt and encodes it again gets the same tokens.
        /// - The chat template writes `role:` and then the content of each
        ///   message, with `imageMarker` for each image part, and a new line
        ///   after each message.
        struct ScriptTokenizer: MLXLMCommon.Tokenizer {
            let specials: [String: Int]
            let imageMarker: String
            /// `encode(text:addSpecialTokens: true)` puts token 1 first.
            let encodeAddsBOS: Bool
            /// `applyChatTemplate` puts token 1 first.
            let templateAddsBOS: Bool

            init(
                specials: [String: Int] = [:], imageMarker: String = "<image>",
                encodeAddsBOS: Bool = false, templateAddsBOS: Bool = false
            ) {
                self.specials = specials.merging(["<s>": 1]) { current, _ in current }
                self.imageMarker = imageMarker
                self.encodeAddsBOS = encodeAddsBOS
                self.templateAddsBOS = templateAddsBOS
            }

            func encode(text: String, addSpecialTokens: Bool) -> [Int] {
                var ids: [Int] = addSpecialTokens && encodeAddsBOS ? [1] : []
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

            func convertTokenToId(_ token: String) -> Int? {
                specials[token]
            }

            func convertIdToToken(_ id: Int) -> String? {
                if let special = specials.first(where: { $0.value == id }) {
                    return special.key
                }
                guard id >= 1000, let scalar = Unicode.Scalar(UInt32(id - 1000)) else {
                    return nil
                }
                return String(Character(scalar))
            }

            var bosToken: String? { "<s>" }
            var eosToken: String? { nil }
            var unknownToken: String? { nil }

            func applyChatTemplate(
                messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                additionalContext: [String: any Sendable]?
            ) throws -> [Int] {
                (templateAddsBOS ? [1] : [])
                    + encode(text: render(messages), addSpecialTokens: false)
            }

            /// The text of the chat template for `messages`.
            func render(_ messages: [[String: any Sendable]]) -> String {
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
                return text
            }
        }

        // MARK: - Helpers

        /// A uniform image of `width` x `height` pixels.
        static func image(_ width: Int, _ height: Int, _ color: CIColor = .white)
            -> UserInput.Image
        {
            .ciImage(
                CIImage(color: color).cropped(to: CGRect(x: 0, y: 0, width: width, height: height)))
        }

        /// The tokens of `text` without special tokens.
        static func text(_ tokenizer: ScriptTokenizer, _ text: String) -> [Int] {
            tokenizer.encode(text: text, addSpecialTokens: false)
        }

        static func tokens(_ input: LMInput) -> [Int] {
            input.text.tokens.asArray(Int.self)
        }

        static func thw(_ frame: THW) -> [Int] {
            [frame.t, frame.h, frame.w]
        }

        static func mistral3(
            longestEdge: Int?, patch: Int = 2, merge: Int = 2, tokenizer: ScriptTokenizer
        ) throws -> Mistral3VLMProcessor {
            var size: [String: Any] = [:]
            if let longestEdge { size["longest_edge"] = longestEdge }
            let config = try SyntheticModel.configuration(
                Mistral3VLMProcessorConfiguration.self,
                [
                    "image_processor": [
                        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                        "size": size, "patch_size": patch,
                    ] as [String: Any],
                    "image_token": "[IMG]", "patch_size": patch, "spatial_merge_size": merge,
                ])
            return Mistral3VLMProcessor(config, tokenizer: tokenizer)
        }

        static func pixtral(longestEdge: Int, patch: Int = 2, tokenizer: ScriptTokenizer) throws
            -> PixtralProcessor
        {
            let config = try SyntheticModel.configuration(
                PixtralProcessorConfiguration.self,
                [
                    "image_processor": [
                        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                        "size": ["longest_edge": longestEdge] as [String: Any],
                        "patch_size": patch,
                    ] as [String: Any],
                    "image_token": "[IMG]", "patch_size": patch,
                ])
            return PixtralProcessor(config, tokenizer: tokenizer)
        }

        static func lfm2vl(tokenizer: ScriptTokenizer, maxTiles: Int = 4) throws -> LFM2VLProcessor
        {
            let config = try SyntheticModel.configuration(
                LFM2VLProcessorConfiguration.self,
                [
                    "tile_size": 8, "encoder_patch_size": 2, "max_tiles": maxTiles,
                    "downsample_factor": 2,
                ])
            return LFM2VLProcessor(config, tokenizer: tokenizer)
        }

        static func gemma3(tokenizer: ScriptTokenizer) throws -> Gemma3Processor {
            let config = try SyntheticModel.configuration(
                Gemma3ProcessorConfiguration.self,
                [
                    "processor_class": "Gemma3Processor",
                    "image_processor_type": "Gemma3ImageProcessor", "do_normalize": true,
                    "do_rescale": true, "do_resize": true, "image_mean": [0.5, 0.5, 0.5],
                    "image_std": [0.5, 0.5, 0.5], "image_seq_length": 4, "resample": 2,
                    "rescale_factor": 0.00392156862745098,
                    "size": ["height": 8, "width": 8] as [String: Any],
                ])
            return Gemma3Processor(config, tokenizer: tokenizer)
        }

        static func idefics3(tokenizer: ScriptTokenizer) throws -> Idefics3Processor {
            let config = try SyntheticModel.configuration(
                Idefics3ProcessorConfiguration.self,
                [
                    "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                    "size": ["longest_edge": 384] as [String: Any], "image_seq_len": 64,
                ])
            return Idefics3Processor(config, tokenizer: tokenizer)
        }

        static func fastVLM(tokenizer: ScriptTokenizer, crop: Int = 16) throws -> FastVLMProcessor {
            let config = try SyntheticModel.configuration(
                FastVLMProcessorConfiguration.self,
                [
                    "image_mean": [0.0, 0.0, 0.0], "image_std": [1.0, 1.0, 1.0],
                    "crop_size": ["width": crop, "height": crop] as [String: Any],
                ])
            return FastVLMProcessor(config, tokenizer: tokenizer)
        }

        // MARK: - Mistral3

        /// A 13 x 9 image with patch size 2 is padded to 14 x 10. That is
        /// 7 x 5 patches, and 3 x 2 = 6 tokens after the 2 x 2 merge. The
        /// one `[IMG]` placeholder becomes 6 image tokens.
        @Test func mistral3PadsToPatchMultiplesAndExpandsTheImageToken() async throws {
            let t = ScriptTokenizer(specials: ["[IMG]": 20], imageMarker: "[IMG]")
            let processor = try Self.mistral3(longestEdge: 64, tokenizer: t)
            let output = try await processor.prepare(
                input: UserInput(prompt: "Describe", images: [Self.image(13, 9)]))

            let pixels = try #require(output.image?.pixels, "Mistral3: pixels")
            #expect(pixels.shape == [1, 3, 10, 14], "Mistral3: padded pixel shape")
            #expect(
                output.image?.frames?.map(Self.thw) == [[1, 10, 14]], "Mistral3: padded frame")
            let expected =
                Self.text(t, "user:") + Array(repeating: 20, count: 6)
                + Self.text(t, "Describe\n")
            #expect(Self.tokens(output) == expected, "Mistral3: tokens with 6 image tokens")
            #expect(
                output.text.tokens.shape == [1, expected.count], "Mistral3: token batch shape")
            // White is a fixed point of the tone curve; (1 - 0.5) / 0.5 = 1.
            // Tolerance 1e-3: Core Image can use half floats inside its filters.
            let center = pixels[0, 0, 5, 7].item(Float.self)
            #expect(abs(center - 1) <= 1e-3, "Mistral3: normalized white center is \(center)")
        }

        /// Without `longest_edge`, the processor uses 24 patches per side:
        /// 48 for patch size 2. A 100 x 50 image becomes 48 x 24: 24 x 12
        /// patches and 12 x 6 = 72 image tokens.
        @Test func mistral3DownscalesToTheVisionLimitWithoutALongestEdge() async throws {
            let t = ScriptTokenizer(specials: ["[IMG]": 20], imageMarker: "[IMG]")
            let processor = try Self.mistral3(longestEdge: nil, tokenizer: t)
            let output = try await processor.prepare(
                input: UserInput(prompt: "x", images: [Self.image(100, 50)]))
            #expect(
                output.image?.pixels.shape == [1, 3, 24, 48], "Mistral3: downscaled pixel shape")
            #expect(
                output.image?.frames?.map(Self.thw) == [[1, 24, 48]],
                "Mistral3: downscaled frame")
            #expect(
                Self.tokens(output).filter { $0 == 20 }.count == 72,
                "Mistral3: 72 image tokens after the downscale")
        }

        /// When the decoded prompt has no `[IMG]` text, the processor uses
        /// the image token ID (10 when the tokenizer does not know `[IMG]`):
        /// it expands the first token 10, or else inserts the tokens after
        /// the BOS token or at the start.
        @Test func mistral3FallsBackToTheImageTokenIdWhenThePlaceholderIsMissing() async throws {
            let image = [Self.image(13, 9)]

            // The template writes `<img>`, which is token 10.
            let withToken = ScriptTokenizer(specials: ["<img>": 10], imageMarker: "<img>")
            let replaced = try await Self.mistral3(longestEdge: 64, tokenizer: withToken)
                .prepare(input: UserInput(prompt: "Describe", images: image))
            #expect(
                Self.tokens(replaced)
                    == Self.text(withToken, "user:") + Array(repeating: 10, count: 6)
                    + Self.text(withToken, "Describe\n"),
                "Mistral3: token 10 replaced by 6 image tokens")

            // The template writes nothing for the image and starts with BOS.
            let withBOS = ScriptTokenizer(imageMarker: "", templateAddsBOS: true)
            let afterBOS = try await Self.mistral3(longestEdge: 64, tokenizer: withBOS)
                .prepare(input: UserInput(prompt: "Describe", images: image))
            #expect(
                Self.tokens(afterBOS)
                    == [1] + Array(repeating: 10, count: 6)
                    + Self.text(withBOS, "user:Describe\n"),
                "Mistral3: image tokens inserted after BOS")

            // No BOS: the image tokens go first.
            let plain = ScriptTokenizer(imageMarker: "")
            let atStart = try await Self.mistral3(longestEdge: 64, tokenizer: plain)
                .prepare(input: UserInput(prompt: "Describe", images: image))
            #expect(
                Self.tokens(atStart)
                    == Array(repeating: 10, count: 6) + Self.text(plain, "user:Describe\n"),
                "Mistral3: image tokens inserted at the start")
        }

        @Test func mistral3TextOnlyAndTwoImages() async throws {
            let t = ScriptTokenizer(specials: ["[IMG]": 20], imageMarker: "[IMG]")
            let processor = try Self.mistral3(longestEdge: 64, tokenizer: t)

            let textOnly = try await processor.prepare(input: UserInput(prompt: "Hi"))
            #expect(textOnly.image == nil, "Mistral3: text-only input has no image")
            #expect(Self.tokens(textOnly) == Self.text(t, "user:Hi\n"), "Mistral3: text tokens")
            #expect(
                textOnly.text.tokens.shape == [1, Self.text(t, "user:Hi\n").count],
                "Mistral3: text token batch shape")

            let two = UserInput(prompt: "Hi", images: [Self.image(4, 4), Self.image(4, 4)])
            await #expect(throws: VLMError.singleImageAllowed) {
                _ = try await processor.prepare(input: two)
            }
        }

        /// A width of 3 must not be mistaken for a channels-last image.
        @Test func mistral3KeepsTheLayoutOfAThreePixelWideImage() async throws {
            let t = ScriptTokenizer(specials: ["[IMG]": 20], imageMarker: "[IMG]")
            let processor = try Self.mistral3(longestEdge: 6, patch: 3, merge: 1, tokenizer: t)
            let output = try await processor.prepare(
                input: UserInput(prompt: "x", images: [Self.image(3, 6)]))
            #expect(output.image?.frames?.map(Self.thw) == [[1, 6, 3]], "Mistral3: 3 x 6 frame")
            let shape = output.image?.pixels.shape ?? []
            #expect(
                shape == [1, 3, 6, 3], "Mistral3: pixel shape for a 3-pixel-wide image")
        }

        // MARK: - Pixtral

        /// A 10 x 6 image with longest edge 8 becomes 8 x 5 and is padded to
        /// 8 x 6: 4 x 3 = 12 patches and 12 image tokens, after the BOS token.
        @Test func pixtralResizesPadsAndInsertsImageTokensAfterBOS() throws {
            let t = ScriptTokenizer(specials: ["[IMG]": 20], encodeAddsBOS: true)
            let processor = try Self.pixtral(longestEdge: 8, tokenizer: t)
            let output = try processor.prepare(
                input: UserInput(prompt: "hi", images: [Self.image(10, 6)]))
            #expect(output.image?.pixels.shape == [1, 3, 6, 8], "Pixtral: padded pixel shape")
            #expect(output.image?.frames?.map(Self.thw) == [[1, 6, 8]], "Pixtral: padded frame")
            #expect(
                Self.tokens(output) == [1] + Array(repeating: 20, count: 12) + Self.text(t, "hi"),
                "Pixtral: BOS, 12 image tokens, text")
        }

        /// A 4 x 2 image with longest edge 8 is scaled up to 8 x 4: 4 x 2 = 8
        /// image tokens. The tokenizer does not know `[IMG]`, so the ID is 10,
        /// and without BOS the image tokens go first.
        @Test func pixtralUpscalesAndUsesTheDefaultImageTokenId() throws {
            let t = ScriptTokenizer()
            let processor = try Self.pixtral(longestEdge: 8, tokenizer: t)
            let output = try processor.prepare(
                input: UserInput(prompt: "hi", images: [Self.image(4, 2)]))
            #expect(output.image?.pixels.shape == [1, 3, 4, 8], "Pixtral: upscaled pixel shape")
            #expect(
                Self.tokens(output) == Array(repeating: 10, count: 8) + Self.text(t, "hi"),
                "Pixtral: 8 image tokens with ID 10 first")
        }

        /// The processor reads the text of the last message for each prompt
        /// kind.
        @Test func pixtralReadsThePromptTextOfEachPromptKind() throws {
            let t = ScriptTokenizer()
            let processor = try Self.pixtral(longestEdge: 8, tokenizer: t)

            let text = try processor.prepare(input: UserInput(prompt: .text("abc")))
            #expect(Self.tokens(text) == Self.text(t, "abc"), "Pixtral: .text prompt")
            #expect(text.image == nil, "Pixtral: no image for text")

            let messages = try processor.prepare(
                input: UserInput(messages: [["role": "user", "content": "xyz"]]))
            #expect(Self.tokens(messages) == Self.text(t, "xyz"), "Pixtral: .messages prompt")

            let chat = try processor.prepare(
                input: UserInput(chat: [.system("sys"), .user("last")]))
            #expect(Self.tokens(chat) == Self.text(t, "last"), "Pixtral: .chat prompt")

            let two = UserInput(prompt: "Hi", images: [Self.image(4, 4), Self.image(4, 4)])
            #expect(throws: VLMError.singleImageAllowed) {
                _ = try processor.prepare(input: two)
            }
        }

        @Test func pixtralKeepsTheLayoutOfAThreePixelWideImage() throws {
            let processor = try Self.pixtral(longestEdge: 6, patch: 3, tokenizer: ScriptTokenizer())
            let output = try processor.prepare(
                input: UserInput(prompt: "x", images: [Self.image(3, 6)]))
            #expect(output.image?.frames?.map(Self.thw) == [[1, 6, 3]], "Pixtral: 3 x 6 frame")
            let shape = output.image?.pixels.shape ?? []
            #expect(shape == [1, 3, 6, 3], "Pixtral: pixel shape for a 3-pixel-wide image")
        }

        // MARK: - LFM2VL

        /// Tile size 8 and patch size 2 give 4 x 4 patches per tile.
        @Test func lfm2vlSplitsImagesIntoTilesAndPatches() throws {
            let processor = try Self.lfm2vl(tokenizer: ScriptTokenizer())
            struct Case {
                let width: Int, height: Int
                let resize: CGSize?
                let shape: (Int, Int)
            }
            let cases = [
                // 12 x 6: 2 x 1 tiles.
                Case(width: 12, height: 6, resize: nil, shape: (4, 8)),
                // 40 x 8: 5 tiles wide, limited to max_tiles = 4.
                Case(width: 40, height: 8, resize: nil, shape: (4, 16)),
                // 32 x 16 resized to fit 8 x 8 is 8 x 4: 1 x 1 tile.
                Case(width: 32, height: 16, resize: CGSize(width: 8, height: 8), shape: (4, 4)),
            ]
            for c in cases {
                let name = "LFM2VL \(c.width) x \(c.height)"
                let image = CIImage(color: .white)
                    .cropped(to: CGRect(x: 0, y: 0, width: c.width, height: c.height))
                let result = try processor.splitIntoPatchesAndPreprocess(
                    image: image, processing: UserInput.Processing(resize: c.resize))
                let patches = c.shape.0 * c.shape.1
                #expect(
                    result.spatialShape.0 == c.shape.0 && result.spatialShape.1 == c.shape.1,
                    "\(name): spatial shape \(result.spatialShape)")
                #expect(result.pixels.shape == [1, patches, 12], "\(name): patch pixel shape")
                #expect(
                    result.pixelAttentionMask.shape == [1, patches], "\(name): mask shape")
                #expect(
                    result.pixelAttentionMask.sum().item(Int.self) == patches,
                    "\(name): every patch is valid")
            }

            let resized = processor.preprocess(
                image: CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 5, height: 7)),
                targetSize: CGSize(width: 8, height: 4))
            #expect(
                Int(resized.extent.width.rounded()) == 8
                    && Int(resized.extent.height.rounded()) == 4,
                "LFM2VL: preprocess resizes to the target size")
        }

        /// Each image placeholder (token 396) becomes (h / 2) * (w / 2) image
        /// tokens. Two 8 x 8 images give 4 x 4 patches and 4 tokens each.
        @Test func lfm2vlExpandsEachImagePlaceholder() async throws {
            let t = ScriptTokenizer(specials: ["<image>": 396])
            let processor = try Self.lfm2vl(tokenizer: t)
            let input = UserInput(chat: [
                .user("a", images: [Self.image(8, 8)]), .user("b", images: [Self.image(8, 8)]),
            ])
            let output = try await processor.prepare(input: input)
            let expected =
                Self.text(t, "user:") + Array(repeating: 396, count: 4) + Self.text(t, "a\nuser:")
                + Array(repeating: 396, count: 4) + Self.text(t, "b\n")
            #expect(Self.tokens(output) == expected, "LFM2VL: two expanded placeholders")
            #expect(output.image?.pixels.shape == [2, 16, 12], "LFM2VL: pixels of two images")
            #expect(
                output.image?.frames?.map(Self.thw) == [[1, 4, 4], [1, 4, 4]],
                "LFM2VL: one frame per image")
            #expect(output.text.mask?.dtype == .int8, "LFM2VL: mask dtype")

            let textOnly = try await processor.prepare(input: UserInput(prompt: "Hi"))
            #expect(textOnly.image == nil, "LFM2VL: text-only input has no image")
            #expect(
                textOnly.text.tokens.shape == [Self.text(t, "user:Hi\n").count],
                "LFM2VL: text-only tokens are 1-D")
            #expect(Self.tokens(textOnly) == Self.text(t, "user:Hi\n"), "LFM2VL: text tokens")
        }

        /// Two images in one message give two adjacent placeholders. The
        /// processor counts adjacent placeholder tokens as one placeholder.
        @Test func lfm2vlExpandsAdjacentImagePlaceholders() async throws {
            let t = ScriptTokenizer(specials: ["<image>": 396])
            let processor = try Self.lfm2vl(tokenizer: t)
            let output = try await processor.prepare(
                input: UserInput(prompt: "hi", images: [Self.image(8, 8), Self.image(8, 8)]))
            #expect(output.image?.frames?.count == 2, "LFM2VL: two frames")
            let imageTokens = Self.tokens(output).filter { $0 == 396 }.count
            // A synchronous function, so that the synchronous
            // `withKnownIssue` is used.
            func check() {
                withKnownIssue(
                    """
                    LFM2VLProcessor.prepare counts adjacent image tokens as one placeholder \
                    (LFM2VL.swift:790-807). Two images in one message get the tokens of the \
                    first image only, and the model then stops on a token count mismatch.
                    """
                ) {
                    #expect(
                        imageTokens == 8, "LFM2VL: adjacent image placeholders give 8 tokens")
                } matching: {
                    $0.isFailedExpectation(["adjacent image placeholders"])
                }
            }
            check()
        }

        // MARK: - Gemma3

        /// The processor resizes every image to `size` (8 x 8) and ignores
        /// the resize of the user.
        @Test func gemma3PreprocessResizesEachImageToTheConfiguredSize() throws {
            let processor = try Self.gemma3(tokenizer: ScriptTokenizer())
            let images = [
                CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 20, height: 10)),
                CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 16, height: 16)),
            ]
            let (pixels, frame) = try processor.preprocess(
                images: images,
                processing: UserInput.Processing(resize: CGSize(width: 2, height: 2)))
            #expect(pixels.shape == [2, 3, 8, 8], "Gemma3: pixels of two images")
            #expect(Self.thw(frame) == [2, 8, 8], "Gemma3: frame of two images")
            // (1 - 0.5) / 0.5 = 1 for white. Tolerance 1e-3: Core Image can
            // use half floats inside its filters.
            let center = pixels[1, 0, 4, 4].item(Float.self)
            #expect(abs(center - 1) <= 1e-3, "Gemma3: normalized white center is \(center)")
        }

        /// Each `<start_of_image>` (255999) becomes `image_seq_length` (4)
        /// image tokens (262144).
        @Test func gemma3ExpandsEachStartOfImageToken() async throws {
            let t = ScriptTokenizer(
                specials: ["<start_of_image>": 255_999], imageMarker: "<start_of_image>")
            let processor = try Self.gemma3(tokenizer: t)
            let output = try await processor.prepare(
                input: UserInput(chat: [
                    .user("a", images: [Self.image(20, 10)]),
                    .user("b", images: [Self.image(6, 6)]),
                ]))
            let expected =
                Self.text(t, "user:") + Array(repeating: 262_144, count: 4)
                + Self.text(t, "a\nuser:") + Array(repeating: 262_144, count: 4)
                + Self.text(t, "b\n")
            #expect(Self.tokens(output) == expected, "Gemma3: two expanded image tokens")
            #expect(output.image?.pixels.shape == [2, 3, 8, 8], "Gemma3: pixels of two images")
            #expect(
                output.image?.frames?.map(Self.thw) == [[1, 8, 8], [1, 8, 8]],
                "Gemma3: one frame per image")
            #expect(output.text.mask?.dtype == .int8, "Gemma3: mask dtype")

            let textOnly = try await processor.prepare(input: UserInput(prompt: "Hi"))
            #expect(textOnly.image == nil, "Gemma3: text-only input has no image")
            #expect(Self.tokens(textOnly) == Self.text(t, "user:Hi\n"), "Gemma3: text tokens")
        }

        // MARK: - Idefics3

        /// The processor inserts one image token (49153) in the middle of
        /// the prompt tokens and resizes the image to 384 x 384, channels
        /// last.
        @Test func idefics3InsertsOneImageTokenAndGivesChannelsLastPixels() throws {
            let t = ScriptTokenizer()
            let processor = try Self.idefics3(tokenizer: t)
            let output = try processor.prepare(
                input: UserInput(prompt: "hello", images: [Self.image(20, 10)]))
            var expected = Self.text(t, "hello")
            expected.insert(49153, at: 2)
            #expect(Self.tokens(output) == expected, "Idefics3: image token at count / 2")
            #expect(output.image?.pixels.shape == [1, 384, 384, 3], "Idefics3: NHWC pixels")
            #expect(output.image?.frames == nil, "Idefics3: no frames")

            let text = try processor.prepare(input: UserInput(prompt: .text("abc")))
            #expect(Self.tokens(text) == Self.text(t, "abc"), "Idefics3: .text prompt")
            #expect(text.image == nil, "Idefics3: no image for text")
            let messages = try processor.prepare(
                input: UserInput(messages: [["role": "user", "content": "xyz"]]))
            #expect(Self.tokens(messages) == Self.text(t, "xyz"), "Idefics3: .messages prompt")

            let two = UserInput(prompt: "Hi", images: [Self.image(4, 4), Self.image(4, 4)])
            #expect(throws: VLMError.singleImageAllowed) {
                _ = try processor.prepare(input: two)
            }
        }

        // MARK: - FastVLM

        /// The processor replaces the `<image>` text with token -200, pads
        /// the image to a square and resizes it to the crop size.
        @Test func fastVLMReplacesTheImageTextAndPadsToASquare() async throws {
            let t = ScriptTokenizer(specials: ["<image>": 50])
            let processor = try Self.fastVLM(tokenizer: t)
            let output = try await processor.prepare(
                input: UserInput(prompt: "Hi", images: [Self.image(12, 6)]))
            #expect(
                Self.tokens(output) == Self.text(t, "user:") + [-200] + Self.text(t, "Hi\n"),
                "FastVLM: tokens with -200")
            #expect(output.image?.pixels.shape == [1, 3, 16, 16], "FastVLM: square crop pixels")

            let two = UserInput(prompt: "Hi", images: [Self.image(4, 4), Self.image(4, 4)])
            await #expect(throws: VLMError.singleImageAllowed) {
                _ = try await processor.prepare(input: two)
            }
        }

        /// The FastVLM message generator drops an empty system message and
        /// keeps a system message with text.
        @Test func fastVLMDropsOnlyEmptySystemMessages() async throws {
            let t = ScriptTokenizer()
            let processor = try Self.fastVLM(tokenizer: t)
            let empty = try await processor.prepare(
                input: UserInput(chat: [.system(""), .user("Hi")]))
            #expect(empty.image == nil, "FastVLM: text-only input has no image")
            #expect(
                Self.tokens(empty) == Self.text(t, "user:Hi\n"), "FastVLM: empty system dropped")

            let kept = try await processor.prepare(
                input: UserInput(chat: [.system("S"), .user("Hi")]))
            #expect(
                Self.tokens(kept) == Self.text(t, "system:S\nuser:Hi\n"),
                "FastVLM: system with text kept")
        }
    }
}
