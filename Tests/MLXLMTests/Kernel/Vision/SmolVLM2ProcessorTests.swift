import CoreImage
import CoreMedia
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

extension KernelTests {

    /// `SmolVLMProcessor.prepare(input:)` for text, one image and one video.
    ///
    /// The tokenizer gives one token per Unicode scalar, so the expected
    /// tokens are the scalars of the expected prompt text.
    @Suite
    struct SmolVLM2ProcessorTests {

        /// One token per Unicode scalar. The chat template writes each
        /// message as `<Role>: <parts>\n`, with `<image>` for an image part
        /// and nothing for a video part.
        private struct ScalarTokenizer: Tokenizer {
            var userPrefix = "User: "
            let bosToken: String? = nil
            let eosToken: String? = nil
            let unknownToken: String? = nil

            func encode(text: String, addSpecialTokens: Bool) -> [Int] {
                text.unicodeScalars.map { Int($0.value) }
            }

            func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
                var text = ""
                text.unicodeScalars.append(
                    contentsOf: tokenIds.compactMap { Unicode.Scalar(UInt32($0)) })
                return text
            }

            func convertTokenToId(_ token: String) -> Int? { nil }
            func convertIdToToken(_ id: Int) -> String? { nil }

            func partText(type: String?, text: String?) -> String {
                switch type {
                case "image": "<image>"
                case "text": text ?? ""
                default: ""
                }
            }

            func render(_ messages: [[String: any Sendable]]) -> String {
                var text = ""
                for message in messages {
                    switch message["role"] as? String {
                    case "user": text += userPrefix
                    case "system": text += "System: "
                    case "assistant": text += "Assistant: "
                    default: text += "Other: "
                    }
                    if let content = message["content"] as? String {
                        text += content
                    } else if let parts = message["content"] as? [[String: String]] {
                        for part in parts {
                            text += partText(type: part["type"], text: part["text"])
                        }
                    } else if let parts = message["content"] as? [[String: any Sendable]] {
                        for part in parts {
                            text += partText(
                                type: part["type"] as? String, text: part["text"] as? String)
                        }
                    }
                    text += "\n"
                }
                return text
            }

            func applyChatTemplate(
                messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                additionalContext: [String: any Sendable]?
            ) throws -> [Int] {
                encode(text: render(messages), addSpecialTokens: false)
            }
        }

        static let fake = "<fake_token_around_image>"
        static let global = "<global-img>"
        static let image = "<image>"

        /// Mean and std 0.5, so a channel of 1 becomes 1 and a channel of 0
        /// becomes -1. Tiles are 32 x 32 and each tile has 2 image tokens.
        private static func processor(userPrefix: String = "User: ") throws -> SmolVLMProcessor {
            let json = """
                {
                    "image_mean": [0.5, 0.5, 0.5],
                    "image_std": [0.5, 0.5, 0.5],
                    "image_seq_len": 2,
                    "size": { "longest_edge": 64 },
                    "max_image_size": { "longest_edge": 32 },
                    "video_sampling": { "fps": 1, "max_frames": 4 }
                }
                """
            let configuration = try JSONDecoder().decode(
                SmolVLMProcessorConfiguration.self, from: Data(json.utf8))
            return SmolVLMProcessor(
                configuration, tokenizer: ScalarTokenizer(userPrefix: userPrefix))
        }

        private static func scalars(_ text: String) -> [Int] {
            text.unicodeScalars.map { Int($0.value) }
        }

        private static func sRGB(_ red: CGFloat, _ green: CGFloat, _ blue: CGFloat) -> CIColor {
            CIColor(
                red: red, green: green, blue: blue,
                colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)!
        }

        private static func solid(_ color: CIColor, width: Int, height: Int) -> CIImage {
            CIImage(color: color).cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
        }

        /// The pixel at row 16, column 16 of tile `index` of `pixels`
        /// (layout [N, H, W, C]). It is far from the tile edges, where the
        /// Lanczos filter reads outside the image.
        private static func center(_ pixels: MLXArray, _ index: Int) -> [Float] {
            pixels[index, 16, 16].asArray(Float.self)
        }

        /// True when each channel of `actual` is within 2e-2 of `expected`.
        /// Tolerance 2e-2: CoreImage resamples in half float with Lanczos weights.
        private static func near(_ actual: [Float], _ expected: [Float]) -> Bool {
            actual.count == expected.count
                && zip(actual, expected).allSatisfy { abs($0 - $1) < 2e-2 }
        }

        private static func frameBlock(_ time: String) -> String {
            "\nFrame from \(time):" + fake + global + image + image + fake
        }

        @Test
        func textOnlyPromptHasNoPixels() async throws {
            let processor = try Self.processor()
            let result = try await processor.prepare(input: UserInput(prompt: "Hi"))

            let expected = Self.scalars("User: Hi\n")
            #expect(result.text.tokens.shape == [1, expected.count], "text token shape")
            #expect(result.text.tokens.asArray(Int.self) == expected, "text tokens")
            #expect(result.text.mask?.shape == [1, expected.count], "text mask shape")
            #expect(
                result.text.mask?.asArray(Int.self) == Array(repeating: 1, count: expected.count),
                "text mask is all ones")
            let hasImage = result.image != nil
            #expect(!hasImage, "no image for text")
        }

        @Test
        func imageIsSplitIntoTilesAndAGlobalImage() async throws {
            let processor = try Self.processor()
            // 64 x 32 fits the 64 edge, so there is 1 row of 2 tiles of 32 x 32.
            let red = Self.solid(Self.sRGB(1, 0, 0), width: 64, height: 32)
            let result = try await processor.prepare(
                input: UserInput(prompt: "Describe", images: [.ciImage(red)]))

            let imageText =
                Self.fake + "<row_1_col_1>" + Self.image + Self.image
                + Self.fake + "<row_1_col_2>" + Self.image + Self.image + "\n"
                + "\n" + Self.fake + Self.global + Self.image + Self.image + Self.fake
            let expected = Self.scalars("User: " + imageText + "Describe\n")
            #expect(result.text.tokens.asArray(Int.self) == expected, "image prompt tokens")
            #expect(result.text.mask?.shape == [1, expected.count], "image mask shape")

            let pixels = try #require(result.image?.pixels)
            #expect(pixels.shape == [3, 32, 32, 3], "2 tiles and 1 global image, channels last")
            let hasFrames = result.image?.frames != nil
            #expect(!hasFrames, "no frame sizes for an image")
            for index in 0 ..< 3 {
                // Tolerance 2e-2: see `near`.
                #expect(
                    Self.near(Self.center(pixels, index), [1, -1, -1]),
                    "normalized red at tile \(index)")
            }
        }

        @Test
        func videoFramesGetTimeStampsAndADefaultSystemMessage() async throws {
            let processor = try Self.processor()
            let colors = [Self.sRGB(1, 0, 0), Self.sRGB(0, 1, 0), Self.sRGB(0, 0, 1)]
            let frames = colors.enumerated().map { index, color in
                VideoFrame(
                    frame: Self.solid(color, width: 32, height: 32),
                    timeStamp: CMTime(value: CMTimeValue(index), timescale: 1))
            }
            let result = try await processor.prepare(
                input: UserInput(prompt: "Describe", videos: [.frames(frames)]))

            // The video is 2 s long, so the rate is (10 - 0.9 * 2) * 1 = 8.2 fps.
            // round(8.2 * 2) = 16 is more than the 3 frames, so all 3 are used.
            let videoText =
                "You are provided the following series of 3 frames from a 0:00:02 [H:MM:SS] video.\n"
                + Self.frameBlock("0:00:00") + Self.frameBlock("0:00:01")
                + Self.frameBlock("0:00:02") + "\n\n"
            let expected = Self.scalars(
                "System: " + processor.defaultVideoSystemMessage + "\n"
                    + "User: " + videoText + "Describe\n")
            #expect(result.text.tokens.asArray(Int.self) == expected, "video prompt tokens")

            let pixels = try #require(result.image?.pixels)
            #expect(pixels.shape == [3, 32, 32, 3], "3 frames, channels last")
            let sizes = result.image?.frames?.map { [$0.t, $0.h, $0.w] }
            #expect(sizes == [[0, 32, 32], [1, 32, 32], [2, 32, 32]], "frame sizes")
            // Tolerance 2e-2: see `near`.
            #expect(Self.near(Self.center(pixels, 0), [1, -1, -1]), "frame 0 is red")
            #expect(Self.near(Self.center(pixels, 1), [-1, 1, -1]), "frame 1 is green")
            #expect(Self.near(Self.center(pixels, 2), [-1, -1, 1]), "frame 2 is blue")
        }

        @Test
        func videoKeepsAnExistingSystemMessage() async throws {
            let processor = try Self.processor()
            let frame = VideoFrame(
                frame: Self.solid(Self.sRGB(1, 0, 0), width: 32, height: 32), timeStamp: .zero)
            let input = UserInput(chat: [
                .system("Be brief"),
                .user("Describe", videos: [.frames([frame])]),
            ])
            let result = try await processor.prepare(input: input)

            // One frame at 0 s: the duration is 0, and one frame is the minimum.
            let videoText =
                "You are provided the following series of 1 frames from a 0:00:00 [H:MM:SS] video.\n"
                + Self.frameBlock("0:00:00") + "\n\n"
            let expected = Self.scalars("System: Be brief\nUser: " + videoText + "Describe\n")
            #expect(result.text.tokens.asArray(Int.self) == expected, "tokens with own system")
            #expect(result.image?.pixels.shape == [1, 32, 32, 3], "one frame")
        }

        @Test
        func videoPromptIsAppendedWhenThereIsNoUserMarker() async throws {
            let processor = try Self.processor(userPrefix: "Human: ")
            let frame = VideoFrame(
                frame: Self.solid(Self.sRGB(1, 0, 0), width: 32, height: 32), timeStamp: .zero)
            let result = try await processor.prepare(
                input: UserInput(prompt: "Describe", videos: [.frames([frame])]))

            let videoText =
                "You are provided the following series of 1 frames from a 0:00:00 [H:MM:SS] video.\n"
                + Self.frameBlock("0:00:00") + "\n\n"
            let decoded =
                "System: " + processor.defaultVideoSystemMessage + "\n" + "Human: Describe\n"
            let expected = Self.scalars(decoded + "\n" + videoText)
            #expect(result.text.tokens.asArray(Int.self) == expected, "fallback video tokens")
        }

        @Test
        func moreThanOneMediaItemIsRefused() async throws {
            let processor = try Self.processor()
            let picture = UserInput.Image.ciImage(
                Self.solid(Self.sRGB(1, 0, 0), width: 32, height: 32))
            let video = UserInput.Video.frames([
                VideoFrame(
                    frame: Self.solid(Self.sRGB(1, 0, 0), width: 32, height: 32),
                    timeStamp: .zero)
            ])

            await #expect(throws: VLMError.singleImageAllowed) {
                _ = try await processor.prepare(
                    input: UserInput(prompt: "Two images", images: [picture, picture]))
            }
            await #expect(throws: VLMError.singleMediaTypeAllowed) {
                _ = try await processor.prepare(
                    input: UserInput(prompt: "Image and video", images: [picture], videos: [video]))
            }
            await #expect(throws: VLMError.singleVideoAllowed) {
                _ = try await processor.prepare(
                    input: UserInput(prompt: "Two videos", videos: [video, video]))
            }
        }
    }
}
