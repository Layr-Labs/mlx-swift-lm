import CoreImage
import CoreMedia
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

extension KernelTests {

    /// Tests of the Qwen2-VL, Qwen2.5-VL, Qwen3-VL and GLM-OCR input
    /// processors and of the shared `QwenVL` helpers (smart resize, patches,
    /// placeholder expansion), with small synthetic images and video frames.
    ///
    /// The processors use patch size 4, merge size 2 and temporal patch size
    /// 2, so the resize factor is 8 and one patch row has `3 * 2 * 4 * 4 =
    /// 96` values. A 32x24 image (width x height) keeps its size and gives
    /// the grid `THW(1, 6, 8)`: 48 patch rows and 48 / 2^2 = 12 image tokens.
    @Suite
    struct QwenVLProcessorTests {

        // MARK: - Fixtures

        /// A tokenizer that keeps the placeholder strings as single tokens.
        ///
        /// The chat template writes `<role>:` and then the message parts in
        /// their order: the image and video placeholders as markup, the text
        /// as it is. Other characters become one token each, with IDs 1 to
        /// 40, so they never collide with the special tokens (56 to 61) and
        /// stay below the vocabulary size 64 of the tiny models.
        struct MarkupTokenizer: MLXLMCommon.Tokenizer {
            static let special: [(String, Int)] = [
                ("<|vision_start|>", 57), ("<|vision_end|>", 58), ("<|image_pad|>", 60),
                ("<|video_pad|>", 61), ("<|begin_of_image|>", 56), ("<|end_of_image|>", 59),
                ("<|image|>", 60),
            ]

            static let qwen = MarkupTokenizer(
                imageMarkup: "<|vision_start|><|image_pad|><|vision_end|>",
                videoMarkup: "<|vision_start|><|video_pad|><|vision_end|>")
            static let glm = MarkupTokenizer(
                imageMarkup: "<|begin_of_image|><|image|><|end_of_image|>", videoMarkup: "")

            let imageMarkup: String
            let videoMarkup: String

            func encode(text: String, addSpecialTokens: Bool) -> [Int] {
                var ids: [Int] = []
                var rest = Substring(text)
                while let first = rest.unicodeScalars.first {
                    if let match = Self.special.first(where: { rest.hasPrefix($0.0) }) {
                        ids.append(match.1)
                        rest = rest.dropFirst(match.0.count)
                    } else {
                        ids.append(1 + Int(first.value % 40))
                        rest = rest.dropFirst()
                    }
                }
                return ids
            }

            func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
                tokenIds.map(String.init).joined(separator: " ")
            }

            func convertTokenToId(_ token: String) -> Int? {
                Self.special.first { $0.0 == token }?.1
            }

            func convertIdToToken(_ id: Int) -> String? { nil }

            var bosToken: String? { nil }
            var eosToken: String? { nil }
            var unknownToken: String? { nil }

            func applyChatTemplate(
                messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                additionalContext: [String: any Sendable]?
            ) throws -> [Int] {
                encode(text: render(messages), addSpecialTokens: false)
            }

            func render(_ messages: [[String: any Sendable]]) -> String {
                var text = ""
                for message in messages {
                    text += "\(message["role"] as? String ?? "user"):"
                    if let parts = message["content"] as? [[String: String]] {
                        for part in parts {
                            switch part["type"] {
                            case "image": text += imageMarkup
                            case "video": text += videoMarkup
                            default: text += part["text"] ?? ""
                            }
                        }
                    } else if let content = message["content"] as? String {
                        text += content
                    }
                }
                return text
            }
        }

        static let tokenizer = MarkupTokenizer.qwen

        /// The Qwen image block with `count` image pad tokens.
        static func imageBlock(_ count: Int) -> String {
            "<|vision_start|>" + String(repeating: "<|image_pad|>", count: count)
                + "<|vision_end|>"
        }

        /// The Qwen video block with `count` video pad tokens.
        static func videoBlock(_ count: Int) -> String {
            "<|vision_start|>" + String(repeating: "<|video_pad|>", count: count)
                + "<|vision_end|>"
        }

        static var processorBase: [String: Any] {
            [
                "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "merge_size": 2,
                "patch_size": 4, "temporal_patch_size": 2,
                "image_processor_type": "Qwen2VLImageProcessor", "min_pixels": 64,
                "max_pixels": 4096,
            ]
        }

        static func qwen2(_ overrides: [String: Any] = [:]) throws -> Qwen2VLProcessor {
            Qwen2VLProcessor(
                try SyntheticModel.configuration(
                    Qwen2VLProcessorConfiguration.self, processorBase, overrides: overrides),
                tokenizer: tokenizer)
        }

        static func qwen25(_ overrides: [String: Any] = [:]) throws -> Qwen25VLProcessor {
            Qwen25VLProcessor(
                try SyntheticModel.configuration(
                    Qwen25VLProcessorConfiguration.self, processorBase, overrides: overrides),
                tokenizer: tokenizer)
        }

        static func qwen3(_ base: [String: Any] = processorBase) throws -> Qwen3VLProcessor {
            Qwen3VLProcessor(
                try SyntheticModel.configuration(Qwen3VLProcessorConfiguration.self, base),
                tokenizer: tokenizer)
        }

        static func glm() throws -> GlmOcrProcessor {
            var base = processorBase
            base["size"] = ["shortest_edge": 64, "longest_edge": 4096]
            return GlmOcrProcessor(
                try SyntheticModel.configuration(GlmOcrProcessorConfiguration.self, base),
                tokenizer: MarkupTokenizer.glm)
        }

        /// A solid color image of `width` x `height` pixels.
        static func image(width: Int, height: Int, red: CGFloat = 0.3) -> CIImage {
            CIImage(color: CIColor(red: red, green: 0.5, blue: 0.7))
                .cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
        }

        /// `count` solid color frames, `step` seconds apart, from time 0.
        static func video(count: Int, step: Double, width: Int = 32, height: Int = 24)
            -> UserInput.Video
        {
            .frames(
                (0 ..< count).map { index in
                    UserInput.VideoFrame(
                        frame: image(
                            width: width, height: height, red: 0.1 + 0.05 * CGFloat(index)),
                        timeStamp: CMTime(
                            seconds: Double(index) * step, preferredTimescale: 600))
                })
        }

        /// Four frames at 0, 0.5, 1 and 1.5 seconds. At 2 frames per second
        /// the processors sample round(2 * 1.5) = 3 of them (at 0, 0.75 and
        /// 1.5 seconds), and `patchify` repeats the last one to get 4 frames:
        /// the grid is `THW(2, h, w)`.
        static var fourFrames: UserInput.Video { video(count: 4, step: 0.5) }

        static func grid(_ thw: THW) -> [Int] {
            [thw.t, thw.h, thw.w]
        }

        static func ids(_ array: MLXArray) -> [Int] {
            array.asType(.int32).asArray(Int32.self).map(Int.init)
        }

        // MARK: - Smart resize

        struct ResizeCase: Sendable, CustomTestStringConvertible {
            let name: String
            let height: Int
            let width: Int
            let minPixels: Int
            let maxPixels: Int
            let expected: [Int]
            var testDescription: String { name }
        }

        static let resizeCases: [ResizeCase] = [
            ResizeCase(
                name: "already a multiple of 8", height: 24, width: 32, minPixels: 64,
                maxPixels: 4096, expected: [24, 32]),
            // 20 / 8 = 2.5 rounds to 3, 44 / 8 = 5.5 rounds to 6.
            ResizeCase(
                name: "rounds half away from zero", height: 20, width: 44, minPixels: 64,
                maxPixels: 4096, expected: [24, 48]),
            // 4096 > 1024: scale by 1 / sqrt(4) and floor to the factor.
            ResizeCase(
                name: "scales down to maxPixels", height: 64, width: 64, minPixels: 64,
                maxPixels: 1024, expected: [32, 32]),
            // 256 < 1024: scale by sqrt(4) and ceil to the factor.
            ResizeCase(
                name: "scales up to minPixels", height: 16, width: 16, minPixels: 1024,
                maxPixels: 4096, expected: [32, 32]),
            // 1280 tokens * 8^2 = 81,920 pixels: beta = sqrt(1.6), so 256 and
            // 512 become floor(25.3) * 8 = 200 and floor(50.6) * 8 = 400.
            ResizeCase(
                name: "1280-token budget", height: 256, width: 512, minPixels: 64,
                maxPixels: 81_920, expected: [200, 400]),
        ]

        @Test(arguments: resizeCases) func smartResizeGivesTheExactSize(_ c: ResizeCase) throws {
            let (height, width) = try QwenVL.targetSize(
                height: c.height, width: c.width, factor: 8, minPixels: c.minPixels,
                maxPixels: c.maxPixels)
            #expect([height, width] == c.expected, "\(c.name): resized height and width")
        }

        @Test func smartResizeRejectsBadSizes() {
            #expect(throws: VLMError.self, "height 4 below the factor 8") {
                try QwenVL.targetSize(
                    height: 4, width: 32, factor: 8, minPixels: 64, maxPixels: 4096)
            }
            #expect(throws: VLMError.self, "width 4 below the factor 8") {
                try QwenVL.targetSize(
                    height: 32, width: 4, factor: 8, minPixels: 64, maxPixels: 4096)
            }
            #expect(throws: VLMError.self, "aspect ratio 1700 / 8 = 212 above 200") {
                try QwenVL.targetSize(
                    height: 8, width: 1700, factor: 8, minPixels: 64, maxPixels: 1 << 30)
            }
            // 64 > 1 pixel: beta = 8, floor(8 / 8 / 8) = 0.
            #expect(throws: VLMError.self, "budget of 1 pixel gives a size of 0") {
                try QwenVL.targetSize(height: 8, width: 8, factor: 8, minPixels: 0, maxPixels: 1)
            }
        }

        // MARK: - Patches

        /// The flattened patch layout: rows in the order (t, h block, w
        /// block, h in block, w in block), and in each row the values in the
        /// order (channel, frame, y in patch, x in patch). A missing last
        /// frame of a temporal pair repeats the last frame.
        static func expectedPatches(
            frames: [[Float]], channels: Int, height: Int, width: Int, patch: Int, merge: Int,
            temporal: Int
        ) -> [Float] {
            let gridT = (frames.count + temporal - 1) / temporal
            var result: [Float] = []
            for t in 0 ..< gridT {
                for hBlock in 0 ..< height / patch / merge {
                    for wBlock in 0 ..< width / patch / merge {
                        for hIn in 0 ..< merge {
                            for wIn in 0 ..< merge {
                                for c in 0 ..< channels {
                                    for frame in 0 ..< temporal {
                                        let f = min(t * temporal + frame, frames.count - 1)
                                        for py in 0 ..< patch {
                                            for px in 0 ..< patch {
                                                let y = (hBlock * merge + hIn) * patch + py
                                                let x = (wBlock * merge + wIn) * patch + px
                                                result.append(
                                                    frames[f][(c * height + y) * width + x])
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return result
        }

        /// One 3x16x8 image: grid THW(1, 4, 2), 8 rows of 96 values. The
        /// values are a copy, so the check is exact.
        @Test func patchifyOfOneImageHasTheExactLayout() throws {
            let (channels, height, width) = (3, 16, 8)
            let values = (0 ..< channels * height * width).map { Float($0) }
            let image = MLXArray(values, [1, channels, height, width])
            let (patches, thw) = try QwenVL.patchify(
                images: [image], mergeSize: 2, patchSize: 4, temporalPatchSize: 2)
            #expect(Self.grid(thw) == [1, 4, 2], "one image grid")
            #expect(patches.shape == [8, 96], "one image patch shape")
            let expected = Self.expectedPatches(
                frames: [values], channels: channels, height: height, width: width, patch: 4,
                merge: 2, temporal: 2)
            #expect(patches.asArray(Float.self) == expected, "one image patch values")
        }

        /// Three 3x8x8 frames: the third frame is repeated, grid THW(2, 2, 2).
        @Test func patchifyOfThreeFramesRepeatsTheLastFrame() throws {
            let (channels, height, width) = (3, 8, 8)
            let size = channels * height * width
            let frames = (0 ..< 3).map { f in (0 ..< size).map { Float(f * 1000 + $0) } }
            let images = frames.map { MLXArray($0, [1, channels, height, width]) }
            let (patches, thw) = try QwenVL.patchify(
                images: images, mergeSize: 2, patchSize: 4, temporalPatchSize: 2)
            #expect(Self.grid(thw) == [2, 2, 2], "three frame grid")
            #expect(patches.shape == [8, 96], "three frame patch shape")
            let expected = Self.expectedPatches(
                frames: frames, channels: channels, height: height, width: width, patch: 4,
                merge: 2, temporal: 2)
            #expect(patches.asArray(Float.self) == expected, "three frame patch values")
        }

        @Test func patchifyRejectsNoImages() {
            #expect(throws: VLMError.self, "no images to patch") {
                try QwenVL.patchify(images: [], mergeSize: 2, patchSize: 4, temporalPatchSize: 2)
            }
        }

        // MARK: - Placeholder expansion

        /// Each placeholder block gets t * h * w / merge^2 pad tokens.
        @Test func replacePaddingTokensExpandsEachBlock() throws {
            let block = Self.imageBlock(1)
            let prompt = Self.tokenizer.encode(text: "a" + block + "b" + block + "c")
            let result = try QwenVL.replacePaddingTokens(
                in: prompt, frames: [THW(1, 4, 4), THW(2, 4, 8)], paddingToken: "<|image_pad|>",
                mergeSize: 2, tokenizer: Self.tokenizer)
            // 16 / 4 = 4 and 64 / 4 = 16 pad tokens.
            let expected = Self.tokenizer.encode(
                text: "a" + Self.imageBlock(4) + "b" + Self.imageBlock(16) + "c")
            #expect(result == expected, "two expanded image blocks")

            // A prompt that ends with the block keeps no text after it.
            let ending = try QwenVL.replacePaddingTokens(
                in: Self.tokenizer.encode(text: "a" + block), frames: [THW(1, 2, 2)],
                paddingToken: "<|image_pad|>", mergeSize: 2, tokenizer: Self.tokenizer)
            #expect(
                ending == Self.tokenizer.encode(text: "a" + Self.imageBlock(1)),
                "block at the end of the prompt")
        }

        @Test func replacePaddingTokensRejectsAFrameCountMismatch() {
            let prompt = Self.tokenizer.encode(text: "a" + Self.imageBlock(1))
            #expect(throws: VLMError.self, "1 placeholder for 2 frames") {
                try QwenVL.replacePaddingTokens(
                    in: prompt, frames: [THW(1, 4, 4), THW(1, 4, 4)],
                    paddingToken: "<|image_pad|>", mergeSize: 2, tokenizer: Self.tokenizer)
            }
        }

        // MARK: - Qwen2-VL processor

        @Test func qwen2VLPrepareExpandsOneImage() async throws {
            let processor = try Self.qwen2()
            let input = UserInput(
                chat: [.user("Describe.", images: [.ciImage(Self.image(width: 32, height: 24))])])
            let result = try await processor.prepare(input: input)

            let image = try #require(result.image, "Qwen2VL: image pixels")
            #expect(image.pixels.shape == [48, 96], "Qwen2VL: image patch shape")
            #expect(image.frames?.map(Self.grid) == [[1, 6, 8]], "Qwen2VL: image grid")
            let noVideo = result.video == nil
            #expect(noVideo, "Qwen2VL: no video")
            #expect(result.text.tokens.shape.first == 1, "Qwen2VL: batch axis")
            #expect(
                Self.ids(result.text.tokens)
                    == Self.tokenizer.encode(text: "user:" + Self.imageBlock(12) + "Describe."),
                "Qwen2VL: 12 image pad tokens")
            #expect(result.text.mask?.shape == result.text.tokens.shape, "Qwen2VL: mask shape")
            #expect(result.text.mask?.dtype == .int8, "Qwen2VL: mask dtype")
        }

        @Test func qwen2VLPrepareExpandsVideoAndImages() async throws {
            let processor = try Self.qwen2()
            let input = UserInput(chat: [
                .user(
                    "Compare.",
                    images: [
                        .ciImage(Self.image(width: 32, height: 24)),
                        .ciImage(Self.image(width: 16, height: 16)),
                    ],
                    videos: [Self.fourFrames])
            ])
            let result = try await processor.prepare(input: input)

            let image = try #require(result.image, "Qwen2VL: two images")
            // 48 + 16 patch rows.
            #expect(image.pixels.shape == [64, 96], "Qwen2VL: two image patch shape")
            #expect(
                image.frames?.map(Self.grid) == [[1, 6, 8], [1, 4, 4]], "Qwen2VL: two image grids")
            let video = try #require(result.video, "Qwen2VL: video pixels")
            #expect(video.pixels.shape == [96, 96], "Qwen2VL: video patch shape")
            #expect(video.frames?.map(Self.grid) == [[2, 6, 8]], "Qwen2VL: video grid")
            let expected = Self.tokenizer.encode(
                text: "user:" + Self.imageBlock(12) + Self.imageBlock(4) + Self.videoBlock(24)
                    + "Compare.")
            #expect(Self.ids(result.text.tokens) == expected, "Qwen2VL: image and video tokens")
        }

        @Test func qwen2VLPrepareOfTextIsOneDimensional() async throws {
            let result = try await Self.qwen2().prepare(input: UserInput(prompt: "Hi."))
            #expect(result.text.tokens.ndim == 1, "Qwen2VL text: 1-D tokens")
            let noMask = result.text.mask == nil
            let noMedia = result.image == nil && result.video == nil
            #expect(noMask, "Qwen2VL text: no mask")
            #expect(noMedia, "Qwen2VL text: no media")
            #expect(
                Self.ids(result.text.tokens) == Self.tokenizer.encode(text: "user:Hi."),
                "Qwen2VL text tokens")
        }

        @Test func qwen2VLPrepareRejectsMissingPlaceholders() async throws {
            let processor = try Self.qwen2()
            await #expect(throws: VLMError.self, "Qwen2VL: image without a placeholder") {
                _ = try await processor.prepare(
                    input: UserInput(
                        messages: [["role": "user", "content": "no markup"]],
                        images: [.ciImage(Self.image(width: 32, height: 24))]))
            }
        }

        /// The resize options of `UserInput.Processing` and several images
        /// in one call.
        @Test func qwen2VLPreprocessHonorsTheProcessingOptions() throws {
            let processor = try Self.qwen2()

            let maxed = try processor.preprocess(
                images: [Self.image(width: 64, height: 64)], processing: .init(maxPixels: 1024))
            #expect(Self.grid(maxed.1) == [1, 8, 8], "maxPixels 1024: 64x64 becomes 32x32")
            #expect(maxed.0.shape == [64, 96], "maxPixels 1024: patch shape")

            let raised = try processor.preprocess(
                images: [Self.image(width: 16, height: 16)], processing: .init(minPixels: 1024))
            #expect(Self.grid(raised.1) == [1, 8, 8], "minPixels 1024: 16x16 becomes 32x32")

            let resized = try processor.preprocess(
                images: [Self.image(width: 32, height: 32)],
                processing: .init(resize: CGSize(width: 16, height: 16)))
            #expect(Self.grid(resized.1) == [1, 4, 4], "resize to 16x16")
            #expect(resized.0.shape == [16, 96], "resize: patch shape")

            let pair = try processor.preprocess(
                images: [Self.image(width: 32, height: 24), Self.image(width: 32, height: 24)],
                processing: nil)
            #expect(Self.grid(pair.1) == [1, 6, 8], "two images make one temporal pair")
            #expect(pair.0.shape == [48, 96], "two images: patch shape")

            let three = try processor.preprocess(
                images: (0 ..< 3).map { _ in Self.image(width: 32, height: 24) },
                processing: nil)
            #expect(Self.grid(three.1) == [2, 6, 8], "three images pad to two pairs")
            #expect(three.0.shape == [96, 96], "three images: patch shape")

            // The default budget is min(max_pixels, 1280 * 8^2).
            let budget = try Self.qwen2(["max_pixels": 12_845_056]).preprocess(
                images: [Self.image(width: 512, height: 256)], processing: nil)
            #expect(Self.grid(budget.1) == [1, 50, 100], "Qwen2VL 1280-token budget")
        }

        // MARK: - Qwen2.5-VL processor

        @Test func qwen25VLPrepareExpandsImageAndVideo() async throws {
            let processor = try Self.qwen25()
            let input = UserInput(chat: [
                .user(
                    "Look.", images: [.ciImage(Self.image(width: 32, height: 24))],
                    videos: [Self.fourFrames])
            ])
            let result = try await processor.prepare(input: input)
            let image = try #require(result.image, "Qwen25VL: image pixels")
            #expect(image.pixels.shape == [48, 96], "Qwen25VL: image patch shape")
            #expect(image.frames?.map(Self.grid) == [[1, 6, 8]], "Qwen25VL: image grid")
            let video = try #require(result.video, "Qwen25VL: video pixels")
            #expect(video.pixels.shape == [96, 96], "Qwen25VL: video patch shape")
            #expect(video.frames?.map(Self.grid) == [[2, 6, 8]], "Qwen25VL: video grid")
            #expect(
                Self.ids(result.text.tokens)
                    == Self.tokenizer.encode(
                        text: "user:" + Self.imageBlock(12) + Self.videoBlock(24) + "Look."),
                "Qwen25VL: image and video tokens")
            #expect(result.text.mask?.dtype == .int8, "Qwen25VL: mask dtype")
        }

        @Test func qwen25VLPrepareOfTextAndMismatch() async throws {
            let processor = try Self.qwen25()
            let text = try await processor.prepare(input: UserInput(prompt: "Hi."))
            #expect(text.text.tokens.ndim == 1, "Qwen25VL text: 1-D tokens")
            let noMedia = text.image == nil && text.video == nil
            #expect(noMedia, "Qwen25VL text: no media")

            await #expect(throws: VLMError.self, "Qwen25VL: image without a placeholder") {
                _ = try await processor.prepare(
                    input: UserInput(
                        messages: [["role": "user", "content": "no markup"]],
                        images: [.ciImage(Self.image(width: 32, height: 24))]))
            }
        }

        @Test func qwen25VLPreprocessUsesTheBudgetAndTheOverrides() throws {
            let budget = try Self.qwen25(["max_pixels": 12_845_056]).preprocess(
                images: [Self.image(width: 512, height: 256)], processing: nil)
            #expect(Self.grid(budget.1) == [1, 50, 100], "Qwen25VL 1280-token budget")
            #expect(budget.0.shape == [5000, 96], "Qwen25VL budget: patch shape")

            let processor = try Self.qwen25()
            let maxed = try processor.preprocess(
                images: [Self.image(width: 64, height: 64)], processing: .init(maxPixels: 1024))
            #expect(Self.grid(maxed.1) == [1, 8, 8], "Qwen25VL maxPixels 1024")
            let raised = try processor.preprocess(
                images: [Self.image(width: 16, height: 16)], processing: .init(minPixels: 1024))
            #expect(Self.grid(raised.1) == [1, 8, 8], "Qwen25VL minPixels 1024")
        }

        // MARK: - Qwen3-VL processor

        @Test func qwen3VLPrepareExpandsImageAndVideo() async throws {
            let processor = try Self.qwen3()
            let input = UserInput(chat: [
                .user(
                    "Look.", images: [.ciImage(Self.image(width: 32, height: 24))],
                    videos: [Self.fourFrames])
            ])
            let result = try await processor.prepare(input: input)
            let image = try #require(result.image, "Qwen3VL: image pixels")
            #expect(image.pixels.shape == [48, 96], "Qwen3VL: image patch shape")
            #expect(image.frames?.map(Self.grid) == [[1, 6, 8]], "Qwen3VL: image grid")
            let video = try #require(result.video, "Qwen3VL: video pixels")
            #expect(video.pixels.shape == [96, 96], "Qwen3VL: video patch shape")
            #expect(video.frames?.map(Self.grid) == [[2, 6, 8]], "Qwen3VL: video grid")
            #expect(
                Self.ids(result.text.tokens)
                    == Self.tokenizer.encode(
                        text: "user:" + Self.imageBlock(12) + Self.videoBlock(24) + "Look."),
                "Qwen3VL: image and video tokens")
        }

        /// Qwen3-VL gives text-only prompts a batch axis and a mask.
        @Test func qwen3VLPrepareOfTextHasABatchAxisAndAMask() async throws {
            let result = try await Self.qwen3().prepare(input: UserInput(prompt: "Hi."))
            let count = Self.tokenizer.encode(text: "user:Hi.").count
            #expect(result.text.tokens.shape == [1, count], "Qwen3VL text: [1, N] tokens")
            #expect(result.text.mask?.shape == [1, count], "Qwen3VL text: mask shape")
            #expect(result.text.mask?.dtype == .int8, "Qwen3VL text: mask dtype")
            let noMedia = result.image == nil && result.video == nil
            #expect(noMedia, "Qwen3VL text: no media")
        }

        @Test func qwen3VLPrepareRejectsMissingPlaceholders() async throws {
            let processor = try Self.qwen3()
            await #expect(throws: VLMError.self, "Qwen3VL: image without a placeholder") {
                _ = try await processor.prepare(
                    input: UserInput(
                        messages: [["role": "user", "content": "no markup"]],
                        images: [.ciImage(Self.image(width: 32, height: 24))]))
            }
        }

        /// Ten frames, 1 second apart: 2 frames per second asks for 18,
        /// the cap keeps `maxVideoFrames` = 8, which make 4 temporal pairs.
        @Test func qwen3VLVideoKeepsAtMostEightFrames() async throws {
            #expect(Qwen3VLProcessor.maxVideoFrames == 8, "frame cap")
            let input = UserInput(chat: [
                .user("Watch.", videos: [Self.video(count: 10, step: 1)])
            ])
            let result = try await Self.qwen3().prepare(input: input)
            let video = try #require(result.video, "Qwen3VL capped video")
            #expect(video.frames?.map(Self.grid) == [[4, 6, 8]], "8 frames make 4 pairs")
            #expect(video.pixels.shape == [192, 96], "capped video: patch shape")
            #expect(
                Self.ids(result.text.tokens).filter { $0 == 61 }.count == 48,
                "4 * 6 * 8 / 4 = 48 video pad tokens")
        }

        /// An explicit `maxPixels` of 1024 resizes 64x48 images and video
        /// frames: beta = sqrt(3072 / 1024), so 48 and 64 become 24 and 32.
        @Test func qwen3VLExplicitMaxPixelsNarrowsImagesAndVideo() async throws {
            let input = UserInput(
                chat: [
                    .user(
                        "Look.", images: [.ciImage(Self.image(width: 64, height: 48))],
                        videos: [Self.video(count: 4, step: 0.5, width: 64, height: 48)])
                ], processing: .init(maxPixels: 1024))
            let result = try await Self.qwen3().prepare(input: input)
            #expect(result.image?.frames?.map(Self.grid) == [[1, 6, 8]], "image to 24x32")
            #expect(result.video?.frames?.map(Self.grid) == [[2, 6, 8]], "video frames to 24x32")
        }

        @Test func qwen3VLPreprocessUsesTheTokenBudget() throws {
            var base = Self.processorBase
            base.removeValue(forKey: "min_pixels")
            base.removeValue(forKey: "max_pixels")
            let processor = try Self.qwen3(base)
            let budget = try processor.preprocess(
                images: [Self.image(width: 512, height: 256)], processing: nil)
            #expect(Self.grid(budget.1) == [1, 50, 100], "Qwen3VL 1280-token budget")

            #expect(throws: VLMError.self, "Qwen3VL: no image") {
                try processor.preprocess(images: [], processing: nil)
            }
        }

        // MARK: - GLM-OCR processor

        @Test func glmOcrPrepareExpandsOneImageAfterTheText() async throws {
            let processor = try Self.glm()
            let input = UserInput(
                chat: [.user("Read.", images: [.ciImage(Self.image(width: 32, height: 24))])])
            let result = try await processor.prepare(input: input)
            let image = try #require(result.image, "GlmOcr: image pixels")
            #expect(image.pixels.shape == [48, 96], "GlmOcr: image patch shape")
            #expect(image.frames?.map(Self.grid) == [[1, 6, 8]], "GlmOcr: image grid")
            let expected = MarkupTokenizer.glm.encode(
                text: "user:Read.<|begin_of_image|>" + String(repeating: "<|image|>", count: 12)
                    + "<|end_of_image|>")
            #expect(Self.ids(result.text.tokens) == expected, "GlmOcr: 12 image tokens")
            #expect(result.text.mask?.dtype == .int8, "GlmOcr: mask dtype")
        }

        @Test func glmOcrPrepareOfTextAndMismatch() async throws {
            let processor = try Self.glm()
            let text = try await processor.prepare(input: UserInput(prompt: "Hi."))
            #expect(text.text.tokens.ndim == 1, "GlmOcr text: 1-D tokens")
            let noImage = text.image == nil
            #expect(noImage, "GlmOcr text: no image")

            await #expect(throws: VLMError.self, "GlmOcr: image without a placeholder") {
                _ = try await processor.prepare(
                    input: UserInput(
                        messages: [["role": "user", "content": "no markup"]],
                        images: [.ciImage(Self.image(width: 32, height: 24))]))
            }
        }

        @Test func glmOcrMessageGeneratorPutsTheImageAfterTheText() {
            let message = GlmOcrMessageGenerator().generate(
                message: .user("Read.", images: [.ciImage(Self.image(width: 8, height: 8))]))
            #expect(
                message["content"] as? [[String: String]] == [
                    ["type": "text", "text": "Read."], ["type": "image"],
                ], "GlmOcr: text, then image")
        }

        // MARK: - Processor configurations

        @Test func processorConfigurationsDecodeThePixelBounds() throws {
            var base = Self.processorBase
            base.removeValue(forKey: "min_pixels")
            base.removeValue(forKey: "max_pixels")

            let qwen2Defaults = try SyntheticModel.configuration(
                Qwen2VLProcessorConfiguration.self, base)
            #expect(qwen2Defaults.minPixels == 3136, "Qwen2VL min_pixels default")
            #expect(qwen2Defaults.maxPixels == 12_845_056, "Qwen2VL max_pixels default")
            let qwen2Size = try SyntheticModel.configuration(
                Qwen2VLProcessorConfiguration.self, base,
                overrides: ["size": ["min_pixels": 100, "max_pixels": 200]])
            #expect(qwen2Size.minPixels == 100, "Qwen2VL min_pixels from size")
            #expect(qwen2Size.maxPixels == 200, "Qwen2VL max_pixels from size")
            let qwen2Top = try SyntheticModel.configuration(
                Qwen2VLProcessorConfiguration.self, base,
                overrides: [
                    "size": ["min_pixels": 100, "max_pixels": 200], "min_pixels": 300,
                    "max_pixels": 400,
                ])
            #expect(qwen2Top.minPixels == 300, "Qwen2VL top-level min_pixels wins")
            #expect(qwen2Top.maxPixels == 400, "Qwen2VL top-level max_pixels wins")

            let qwen25 = try SyntheticModel.configuration(
                Qwen25VLProcessorConfiguration.self, Self.processorBase)
            #expect(qwen25.size.minPixels == 64, "Qwen25VL size.minPixels")
            #expect(qwen25.size.maxPixels == 4096, "Qwen25VL size.maxPixels")

            let qwen3Defaults = try SyntheticModel.configuration(
                Qwen3VLProcessorConfiguration.self, base)
            #expect(qwen3Defaults.minPixels == 3136, "Qwen3VL min_pixels default")
            #expect(qwen3Defaults.maxPixels == 12_845_056, "Qwen3VL max_pixels default")
            #expect(qwen3Defaults.size.minPixels == 3136, "Qwen3VL size.minPixels")
            #expect(qwen3Defaults.size.maxPixels == 12_845_056, "Qwen3VL size.maxPixels")
            let qwen3 = try SyntheticModel.configuration(
                Qwen3VLProcessorConfiguration.self, Self.processorBase)
            #expect(qwen3.minPixels == 64, "Qwen3VL min_pixels")
            #expect(qwen3.maxPixels == 4096, "Qwen3VL max_pixels")

            var glmBase = base
            glmBase["size"] = ["shortest_edge": 128, "longest_edge": 2048]
            let glm = try SyntheticModel.configuration(GlmOcrProcessorConfiguration.self, glmBase)
            #expect(glm.minPixels == 128, "GlmOcr min pixels from shortest_edge")
            #expect(glm.maxPixels == 2048, "GlmOcr max pixels from longest_edge")
        }

        // MARK: - Model configurations

        @Test func modelConfigurationsDecodeTheirDefaults() throws {
            let qwen2 = try SyntheticModel.configuration(
                Qwen2VLConfiguration.self,
                [
                    "model_type": "qwen2_vl", "hidden_size": 32, "num_hidden_layers": 2,
                    "intermediate_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2,
                    "vocab_size": 64, "image_token_id": 60, "video_token_id": 61,
                    "vision_config": [
                        "depth": 1, "embed_dim": 32, "hidden_size": 32, "num_heads": 4,
                        "patch_size": 4, "mlp_ratio": 2.0, "spatial_patch_size": 2,
                        "spatial_merge_size": 2, "temporal_patch_size": 2,
                    ] as [String: Any],
                ])
            // Exact: each default is a constant.
            #expect(qwen2.textConfiguration.rmsNormEps == 1e-6, "Qwen2VL rms_norm_eps")
            #expect(
                qwen2.textConfiguration.maxpPositionEmbeddings == 32768,
                "Qwen2VL max_position_embeddings")
            #expect(qwen2.textConfiguration.ropeTheta == 1_000_000, "Qwen2VL rope_theta")
            #expect(!qwen2.textConfiguration.ropeTraditional, "Qwen2VL rope_traditional")
            #expect(qwen2.textConfiguration.ropeScaling == nil, "Qwen2VL no rope_scaling")
            #expect(qwen2.textConfiguration.tieWordEmbeddings, "Qwen2VL tied")
            #expect(qwen2.visionConfiguration.inChannels == 3, "Qwen2VL in_channels")
            #expect(qwen2.visionConfiguration.layerNormEps == 1e-6, "Qwen2VL layer_norm_eps")
            #expect(qwen2.baseConfiguration.imageTokenId == 60, "Qwen2VL image_token_id")

            let qwen25 = try SyntheticModel.configuration(
                Qwen25VLConfiguration.self,
                [
                    "model_type": "qwen2_5_vl", "hidden_size": 32, "num_hidden_layers": 2,
                    "intermediate_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2,
                    "vocab_size": 64, "image_token_id": 60, "video_token_id": 61,
                    "vision_start_token_id": 57, "vision_end_token_id": 58,
                    "vision_token_id": 59, "sliding_window": 16, "use_sliding_window": false,
                    "max_window_layers": 2,
                    "vision_config": [
                        "depth": 1, "hidden_size": 32, "intermediate_size": 48,
                        "out_hidden_size": 32, "num_heads": 4, "patch_size": 4,
                        "spatial_patch_size": 2, "spatial_merge_size": 2,
                        "temporal_patch_size": 2, "window_size": 8,
                        "fullatt_block_indexes": [0], "tokens_per_second": 2,
                    ] as [String: Any],
                ])
            let text25 = qwen25.textConfiguration
            #expect(text25.maxPositionEmbeddings == 128_000, "Qwen25VL max_position_embeddings")
            #expect(text25.ropeTheta == 1_000_000, "Qwen25VL rope_theta")
            #expect(!text25.ropeTraditional, "Qwen25VL rope_traditional")
            #expect(text25.tieWordEmbeddings, "Qwen25VL tied")
            #expect(text25.slidingWindow == 16, "Qwen25VL sliding_window")
            #expect(!text25.useSlidingWindow, "Qwen25VL use_sliding_window")
            #expect(text25.rmsNormEps == 1e-6, "Qwen25VL rms_norm_eps")
            let vision25 = qwen25.visionConfiguration
            #expect(vision25.inChannels == 3, "Qwen25VL in_chans")
            #expect(vision25.layerNormEps == 1e-6, "Qwen25VL layer_norm_eps")
            #expect(!vision25.skipVision, "Qwen25VL skip_vision")
            #expect(vision25.hiddenAct == "silu", "Qwen25VL hidden_act")
            #expect(qwen25.baseConfiguration.maxWindowLayers == 2, "Qwen25VL max_window_layers")

            let glm = try SyntheticModel.configuration(
                GlmOcrConfiguration.self,
                [
                    "model_type": "glm_ocr",
                    "text_config": [
                        "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                        "vocab_size": 64, "rope_parameters": ["mrope_section": [1, 1, 2]],
                    ] as [String: Any],
                    "vision_config": [
                        "depth": 1, "hidden_size": 32, "intermediate_size": 48, "num_heads": 4,
                        "patch_size": 4, "out_hidden_size": 32, "spatial_merge_size": 2,
                        "temporal_patch_size": 2,
                    ] as [String: Any],
                ])
            #expect(glm.baseConfiguration.vocabularySize == 59392, "GlmOcr vocab_size default")
            #expect(glm.baseConfiguration.imageTokenId == 59280, "GlmOcr image_token_id default")
            #expect(glm.baseConfiguration.videoTokenId == 59281, "GlmOcr video_token_id default")
            #expect(
                glm.baseConfiguration.imageStartTokenId == 59256,
                "GlmOcr image_start_token_id default")
            #expect(glm.baseConfiguration.hiddenSize == 1536, "GlmOcr hidden_size default")
            #expect(glm.textConfiguration.rmsNormEps == 1e-5, "GlmOcr text rms_norm_eps")
            #expect(!glm.textConfiguration.tieWordEmbeddings, "GlmOcr untied by default")
            #expect(glm.textConfiguration.ropeTheta == 10_000, "GlmOcr rope_theta default")
            #expect(
                glm.textConfiguration.ropeParameters.partialRotaryFactor == 1.0,
                "GlmOcr partial_rotary_factor default")
            #expect(glm.visionConfiguration.inChannels == 3, "GlmOcr in_channels default")
            #expect(glm.visionConfiguration.rmsNormEps == 1e-5, "GlmOcr vision rms_norm_eps")
        }

        // MARK: - Tiny models

        static var qwen2VLModelConfiguration: [String: Any] {
            [
                "model_type": "qwen2_vl", "hidden_size": 32, "num_hidden_layers": 2,
                "intermediate_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2,
                "vocab_size": 64, "image_token_id": 60, "video_token_id": 61,
                "rope_scaling": ["type": "mrope", "mrope_section": [1, 1, 2]],
                "vision_config": [
                    "depth": 1, "embed_dim": 32, "hidden_size": 32, "num_heads": 4,
                    "patch_size": 4, "mlp_ratio": 2.0, "spatial_patch_size": 2,
                    "spatial_merge_size": 2, "temporal_patch_size": 2,
                ] as [String: Any],
            ]
        }

        static var qwen25VLModelConfiguration: [String: Any] {
            [
                "model_type": "qwen2_5_vl", "hidden_size": 32, "num_hidden_layers": 2,
                "intermediate_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2,
                "vocab_size": 64, "image_token_id": 60, "video_token_id": 61,
                "vision_start_token_id": 57, "vision_end_token_id": 58, "vision_token_id": 59,
                "sliding_window": 16, "use_sliding_window": false, "max_window_layers": 2,
                "rope_scaling": ["type": "mrope", "mrope_section": [1, 1, 2]],
                "vision_config": [
                    "depth": 2, "hidden_size": 32, "intermediate_size": 48,
                    "out_hidden_size": 32, "num_heads": 4, "patch_size": 4,
                    "spatial_patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 2,
                    "window_size": 8, "fullatt_block_indexes": [1], "tokens_per_second": 2,
                ] as [String: Any],
            ]
        }

        static var glmOcrModelConfiguration: [String: Any] {
            [
                "model_type": "glm_ocr", "vocab_size": 64, "image_token_id": 60,
                "video_token_id": 61, "image_start_token_id": 56, "hidden_size": 32,
                "text_config": [
                    "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                    "vocab_size": 64,
                    "rope_parameters": [
                        "mrope_section": [1, 1, 2], "partial_rotary_factor": 1.0,
                        "rope_theta": 10000,
                    ] as [String: Any],
                ] as [String: Any],
                "vision_config": [
                    "depth": 1, "hidden_size": 32, "intermediate_size": 48, "num_heads": 4,
                    "patch_size": 4, "out_hidden_size": 32, "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                ] as [String: Any],
            ]
        }

        static func qwen2VLModel(seed: UInt64) throws -> any LanguageModel {
            try ModelCase.build(
                Qwen2VLConfiguration.self, qwen2VLModelConfiguration, seed: seed, Qwen2VL.init)
        }

        static func qwen25VLModel(seed: UInt64) throws -> any LanguageModel {
            try ModelCase.build(
                Qwen25VLConfiguration.self, qwen25VLModelConfiguration, seed: seed, Qwen25VL.init)
        }

        static func glmOcrModel(seed: UInt64) throws -> any LanguageModel {
            try ModelCase.build(
                GlmOcrConfiguration.self, glmOcrModelConfiguration, seed: seed, GlmOcr.init)
        }

        /// Runs `prepare` with a new cache and returns the logits.
        static func logits(_ model: any LanguageModel, _ input: LMInput) throws -> MLXArray {
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

        static var imageAndVideo: UserInput {
            UserInput(chat: [
                .user(
                    "Look.", images: [.ciImage(image(width: 32, height: 24))],
                    videos: [fourFrames])
            ])
        }

        static var videoOnly: UserInput {
            UserInput(chat: [.user("Watch.", videos: [fourFrames])])
        }

        // Tolerance of the float32 comparisons of prepare against the direct
        // call: the two paths differ only in the order of the attention sums
        // (with or without a cache). Differences are near 1e-6.
        static let tolerance: Float = 1e-4

        // MARK: - Processor output through the models

        @Test func qwen3VLProcessorOutputRunsThroughTheModel() async throws {
            let input = try await Self.qwen3().prepare(input: Self.imageAndVideo)
            let model = try KernelTests.Qwen3VLForwardPassTests.makeModel(seed: 1)
            let logits = try Self.logits(model, input)
            #expect(
                logits.shape == [1, input.text.tokens.dim(1), 64], "Qwen3VL end to end: shape")
            #expect(isFinite(logits).all().item(Bool.self), "Qwen3VL end to end: not finite")
        }

        /// Qwen2-VL `prepare` returns the logits of the last position only.
        @Test func qwen2VLProcessorOutputRunsThroughTheModel() async throws {
            let processor = try Self.qwen2()
            let model = try Self.qwen2VLModel(seed: 1)
            for (name, userInput) in [("video", Self.videoOnly), ("both", Self.imageAndVideo)] {
                let input = try await processor.prepare(input: userInput)
                let logits = try Self.logits(model, input)
                #expect(logits.shape == [1, 1, 64], "Qwen2VL end to end \(name): shape")
                #expect(
                    isFinite(logits).all().item(Bool.self),
                    "Qwen2VL end to end \(name): not finite")
            }
        }

        @Test func qwen25VLProcessorOutputRunsThroughTheModel() async throws {
            let processor = try Self.qwen25()
            let model = try Self.qwen25VLModel(seed: 1)
            for (name, userInput) in [("video", Self.videoOnly), ("both", Self.imageAndVideo)] {
                let input = try await processor.prepare(input: userInput)
                let logits = try Self.logits(model, input)
                #expect(
                    logits.shape == [1, input.text.tokens.dim(1), 64],
                    "Qwen25VL end to end \(name): shape")
                #expect(
                    isFinite(logits).all().item(Bool.self),
                    "Qwen25VL end to end \(name): not finite")
            }
        }

        /// After an image prompt, a text prompt must not use the stored
        /// M-RoPE positions of the image prompt.
        @Test func glmOcrTextAfterAnImageMatchesANewModel() async throws {
            let processor = try Self.glm()
            let imageInput = try await processor.prepare(
                input: UserInput(
                    chat: [.user("Read.", images: [.ciImage(Self.image(width: 32, height: 24))])]
                ))
            let textInput = try await processor.prepare(input: UserInput(prompt: "Hi."))

            let model = try Self.glmOcrModel(seed: 1)
            let imageLogits = try Self.logits(model, imageInput)
            #expect(
                imageLogits.shape == [1, imageInput.text.tokens.dim(1), 64],
                "GlmOcr end to end: shape")
            #expect(
                isFinite(imageLogits).all().item(Bool.self), "GlmOcr end to end: not finite")

            let after = try Self.logits(model, textInput)
            let fresh = try Self.logits(try Self.glmOcrModel(seed: 1), textInput)
            // Exact: the same weights run the same text computation.
            #expect(
                SyntheticModel.maxAbsDifference(after, fresh) == 0,
                "GlmOcr text after an image differs from a new model")
        }

        /// The text-only `prepare` path gives the logits of the direct call.
        @Test func textOnlyPrepareMatchesTheDirectCall() async throws {
            let text = try await Self.qwen2().prepare(input: UserInput(prompt: "Hello there."))
            let tokens = text.text.tokens.expandedDimensions(axis: 0)
            let models: [(String, any LanguageModel)] = try [
                ("Qwen2VL", Self.qwen2VLModel(seed: 1)),
                ("Qwen25VL", Self.qwen25VLModel(seed: 1)),
                ("GlmOcr", Self.glmOcrModel(seed: 1)),
            ]
            for (name, model) in models {
                let prepared = try Self.logits(model, text)
                let direct = model(tokens, cache: nil)
                eval(direct)
                let difference = SyntheticModel.maxAbsDifference(
                    prepared[0..., -1], direct[0..., -1])
                #expect(
                    difference <= Self.tolerance,
                    "\(name): text prepare differs from the direct call by \(difference)")
            }
        }

        // MARK: - Sanitize

        /// Hugging Face keys move under `vision_tower` and `language_model`,
        /// `position_ids` buffers go away, and a convolution weight in the
        /// MLX layout (last axis = 3 channels) stays as it is.
        @Test func qwen2And25SanitizeRenameKeysAndKeepTheMLXLayout() throws {
            let models: [(String, any LanguageModel)] = try [
                ("Qwen2VL", Self.qwen2VLModel(seed: 1)),
                ("Qwen25VL", Self.qwen25VLModel(seed: 1)),
            ]
            for (name, model) in models {
                let sanitized = model.sanitize(weights: [
                    "visual.patch_embed.proj.weight": MLXArray.zeros([32, 2, 4, 4, 3]),
                    "visual.position_ids": MLXArray.zeros([4]),
                    "model.embed_tokens.weight": MLXArray.zeros([64, 32]),
                    "lm_head.weight": MLXArray.zeros([64, 32]),
                ])
                #expect(
                    Set(sanitized.keys) == [
                        "vision_tower.patch_embed.proj.weight",
                        "language_model.model.embed_tokens.weight",
                        "language_model.lm_head.weight",
                    ], "\(name): sanitized keys")
                #expect(
                    sanitized["vision_tower.patch_embed.proj.weight"]?.shape == [32, 2, 4, 4, 3],
                    "\(name): MLX convolution layout is kept")
            }
        }

        /// GLM-OCR: the next-n layer (index = number of layers) goes away,
        /// a PyTorch `downsample` weight `[O, I, kH, kW]` becomes
        /// `[O, kH, kW, I]`, an MLX patch weight stays, and a weight that is
        /// not 4-D or 5-D stays as it is.
        @Test func glmOcrSanitizeConvertsTheCheckpoint() throws {
            let model = try Self.glmOcrModel(seed: 1)
            let sanitized = model.sanitize(weights: [
                "model.visual.downsample.weight": MLXArray.zeros([32, 32, 2, 2]),
                "model.visual.patch_embed.proj.weight": MLXArray.zeros([32, 2, 4, 4, 3]),
                "model.visual.position_ids": MLXArray.zeros([4]),
                "model.language_model.layers.1.mlp.down_proj.weight": MLXArray.zeros([32, 48]),
                "model.language_model.layers.2.mlp.down_proj.weight": MLXArray.zeros([32, 48]),
                "lm_head.weight": MLXArray.zeros([64, 32]),
            ])
            #expect(
                Set(sanitized.keys) == [
                    "vision_tower.downsample.weight", "vision_tower.patch_embed.proj.weight",
                    "language_model.model.layers.1.mlp.down_proj.weight",
                    "language_model.lm_head.weight",
                ], "GlmOcr: sanitized keys")
            #expect(
                sanitized["vision_tower.downsample.weight"]?.shape == [32, 2, 2, 32],
                "GlmOcr: PyTorch downsample weight is transposed")
            #expect(
                sanitized["vision_tower.patch_embed.proj.weight"]?.shape == [32, 2, 4, 4, 3],
                "GlmOcr: MLX patch weight is kept")

            let other = model.sanitize(weights: [
                "vision_tower.downsample.weight": MLXArray.zeros([2, 2, 2])
            ])
            #expect(
                other["vision_tower.downsample.weight"]?.shape == [2, 2, 2],
                "GlmOcr: a 3-D weight is kept")
        }
    }
}
