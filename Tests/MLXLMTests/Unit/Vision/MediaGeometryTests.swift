import CoreGraphics
import CoreMedia
import Foundation
import MLXLMCommon
import Testing

@testable import MLXVLM

extension UnitTests {

    /// Size and rectangle helpers of `MediaProcessing`. These helpers use
    /// only `CGSize` and `CGRect`, so the expected values are exact.
    @Suite
    struct MediaGeometryTests {

        @Test
        func bestFitKeepsTheAspectRatio() {
            let scale = MediaProcessing.bestFitScale(
                CGSize(width: 200, height: 100), in: CGSize(width: 50, height: 50))
            #expect(scale == 0.25, "best fit scale of 200x100 in 50x50")

            let wide = MediaProcessing.bestFit(
                CGSize(width: 200, height: 100), in: CGSize(width: 50, height: 50))
            #expect(wide == CGSize(width: 50, height: 25), "best fit of 200x100 in 50x50")

            // 30x20 scales by 100/30. The height 66.67 rounds to 67.
            let up = MediaProcessing.bestFit(
                CGSize(width: 30, height: 20), in: CGSize(width: 100, height: 100))
            #expect(up == CGSize(width: 100, height: 67), "best fit of 30x20 in 100x100")
        }

        @Test
        func fitInShortestEdgeScalesBothOrientations() {
            #expect(
                MediaProcessing.fitIn(CGSize(width: 200, height: 100), shortestEdge: 50)
                    == CGSize(width: 100, height: 50), "shortest edge of a wide size")
            #expect(
                MediaProcessing.fitIn(CGSize(width: 100, height: 200), shortestEdge: 50)
                    == CGSize(width: 50, height: 100), "shortest edge of a tall size")
            // The function also enlarges a small size.
            #expect(
                MediaProcessing.fitIn(CGSize(width: 10, height: 20), shortestEdge: 30)
                    == CGSize(width: 30, height: 60), "shortest edge enlarges")
        }

        @Test
        func fitInLongestEdgeKeepsSmallSizes() {
            #expect(
                MediaProcessing.fitIn(CGSize(width: 40, height: 20), longestEdge: 50)
                    == CGSize(width: 40, height: 20), "longest edge keeps a wide size that fits")
            #expect(
                MediaProcessing.fitIn(CGSize(width: 20, height: 40), longestEdge: 50)
                    == CGSize(width: 20, height: 40), "longest edge keeps a tall size that fits")
        }

        @Test
        func fitInLongestEdgeShrinksLargeSizes() {
            let wide = MediaProcessing.fitIn(CGSize(width: 200, height: 100), longestEdge: 50)
            let tall = MediaProcessing.fitIn(CGSize(width: 100, height: 200), longestEdge: 50)
            #expect(wide == CGSize(width: 50, height: 25), "longest edge shrinks a wide size")
            #expect(tall == CGSize(width: 25, height: 50), "longest edge shrinks a tall size")
        }

        @Test
        func rectSmallerOrEqualComparesBothSides() {
            let rect = CGRect(x: 0, y: 0, width: 10, height: 20)
            #expect(
                MediaProcessing.rectSmallerOrEqual(rect, size: CGSize(width: 10, height: 20)),
                "equal size")
            #expect(
                MediaProcessing.rectSmallerOrEqual(rect, size: CGSize(width: 30, height: 30)),
                "larger size")
            #expect(
                !MediaProcessing.rectSmallerOrEqual(rect, size: CGSize(width: 9, height: 30)),
                "width too small")
            #expect(
                !MediaProcessing.rectSmallerOrEqual(rect, size: CGSize(width: 30, height: 19)),
                "height too small")
        }

        @Test
        func centerCropRectAtTheOrigin() {
            let crop = MediaProcessing.centerCrop(
                CGRect(x: 0, y: 0, width: 8, height: 6), size: CGSize(width: 4, height: 4))
            #expect(crop == CGRect(x: 2, y: 1, width: 4, height: 4), "center crop of 8x6 to 4x4")

            // The crop is never larger than the extent.
            let clamped = MediaProcessing.centerCrop(
                CGRect(x: 0, y: 0, width: 8, height: 6), size: CGSize(width: 20, height: 2))
            #expect(
                clamped == CGRect(x: 0, y: 2, width: 8, height: 2),
                "center crop clamps the width to the extent")
        }

        @Test
        func centerCropRectWithAnOffsetOrigin() {
            // The extent is x 10 ..< 18, y 4 ..< 10. The center is (14, 7).
            let crop = MediaProcessing.centerCrop(
                CGRect(x: 10, y: 4, width: 8, height: 6), size: CGSize(width: 4, height: 4))
            #expect(
                crop == CGRect(x: 12, y: 5, width: 4, height: 4),
                "center crop of an offset extent")
            #expect(crop.size == CGSize(width: 4, height: 4), "size of the offset crop")
        }
    }

    /// The parts of `SmolVLMProcessor` that use only Swift values: the
    /// configuration, the resize sizes, the time stamps and the prompt text.
    @Suite
    struct SmolVLMProcessorLogicTests {

        /// A tokenizer that is never called by these tests.
        private struct UnusedTokenizer: Tokenizer {
            let bosToken: String? = nil
            let eosToken: String? = nil
            let unknownToken: String? = nil
            func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
            func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
            func convertTokenToId(_ token: String) -> Int? { nil }
            func convertIdToToken(_ id: Int) -> String? { nil }
            func applyChatTemplate(
                messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                additionalContext: [String: any Sendable]?
            ) throws -> [Int] { [] }
        }

        private static func processor(imageSequenceLength: String = "")
            throws -> SmolVLMProcessor
        {
            let json = """
                {
                    "image_mean": [0.1, 0.2, 0.3],
                    "image_std": [0.4, 0.5, 0.6],
                    \(imageSequenceLength)
                    "size": { "longest_edge": 96 },
                    "max_image_size": { "longest_edge": 24 },
                    "video_sampling": { "fps": 3, "max_frames": 7 }
                }
                """
            let configuration = try JSONDecoder().decode(
                SmolVLMProcessorConfiguration.self, from: Data(json.utf8))
            return SmolVLMProcessor(configuration, tokenizer: UnusedTokenizer())
        }

        @Test
        func configurationDecodesAndUsesDefaults() throws {
            let json = """
                {
                    "image_mean": [0.1, 0.2, 0.3],
                    "image_std": [0.4, 0.5, 0.6],
                    "size": { "longest_edge": 96 },
                    "max_image_size": { "longest_edge": 24 },
                    "video_sampling": { "fps": 3, "max_frames": 7 }
                }
                """
            let configuration = try JSONDecoder().decode(
                SmolVLMProcessorConfiguration.self, from: Data(json.utf8))
            #expect(configuration.imageSequenceLength == 64, "default image sequence length")
            #expect(configuration.size.longestEdge == 96, "size longest edge")
            #expect(configuration.maxImageSize.longestEdge == 24, "max image size longest edge")
            #expect(configuration.videoSampling.fps == 3, "video fps")
            #expect(configuration.videoSampling.maxFrames == 7, "video max frames")
            let mean = configuration.imageMeanTuple
            let std = configuration.imageStdTuple
            #expect([mean.0, mean.1, mean.2] == [0.1, 0.2, 0.3], "mean tuple")
            #expect([std.0, std.1, std.2] == [0.4, 0.5, 0.6], "std tuple")

            let explicit = try Self.processor(imageSequenceLength: #""image_seq_len": 5,"#)
            #expect(explicit.imageSequenceLength == 5, "decoded image sequence length")

            let memberwise = SmolVLMProcessorConfiguration(
                imageMean: [0, 0, 0], imageStd: [1, 1, 1],
                size: .init(longestEdge: 8), maxImageSize: .init(longestEdge: 4),
                videoSampling: .init(fps: 2, maxFrames: 3), imageSequenceLength: nil)
            #expect(memberwise.imageSequenceLength == 64, "memberwise default sequence length")
        }

        @Test
        func processorReadsItsSizesFromTheConfiguration() throws {
            let processor = try Self.processor()
            #expect(processor.maxProcessingImageSize == 96, "max processing image size")
            #expect(processor.fixedImageSize == 24, "fixed image size")
            #expect(processor.imageSequenceLength == 64, "processor sequence length")
            #expect(processor.maxVideoFrames == 20, "hard-coded max video frames")
            #expect(processor.targetVideoFPS == 3, "target video fps")
        }

        @Test
        func aspectRatioSizeRoundsUpToTheMultiple() throws {
            let processor = try Self.processor()

            // 100x50 fits in 64x64 as 64x32.
            #expect(
                processor.aspectRatioSize(for: CGSize(width: 100, height: 50), longestEdge: 64)
                    == CGSize(width: 64, height: 32), "no multiple (Int)")
            #expect(
                processor.aspectRatioSize(
                    for: CGSize(width: 100, height: 50), longestEdge: CGFloat(64))
                    == CGSize(width: 64, height: 32), "no multiple (CGFloat)")

            // Wide: width ceil(64 / 24) * 24 = 72, height 72 / 2 = 36 -> 48.
            #expect(
                processor.aspectRatioSize(
                    for: CGSize(width: 100, height: 50), longestEdge: 64, multiple: 24)
                    == CGSize(width: 72, height: 48), "wide with multiple 24")

            // Tall: height 72, width 72 * 0.5 = 36 -> 48.
            #expect(
                processor.aspectRatioSize(
                    for: CGSize(width: 50, height: 100), longestEdge: 64, multiple: 24)
                    == CGSize(width: 48, height: 72), "tall with multiple 24")

            // A square takes the wide branch.
            #expect(
                processor.aspectRatioSize(
                    for: CGSize(width: 30, height: 30), longestEdge: CGFloat(30),
                    multiple: CGFloat(16))
                    == CGSize(width: 32, height: 32), "square with multiple 16")
        }

        @Test
        func formatTimestampRoundsUpToWholeSeconds() throws {
            let processor = try Self.processor()
            #expect(processor.formatTimestamp(.zero) == "0:00:00", "zero time")
            #expect(
                processor.formatTimestamp(CMTime(value: 1, timescale: 10)) == "0:00:01",
                "0.1 s rounds up")
            // 3725.2 s rounds up to 3726 s = 1 h 2 min 6 s.
            #expect(
                processor.formatTimestamp(CMTime(value: 37252, timescale: 10)) == "1:02:06",
                "hours, minutes and seconds")
        }

        @Test
        func imagePromptStringHasOneBlockPerTileAndAGlobalBlock() throws {
            let processor = try Self.processor()
            let text = processor.getImagePromptString(
                rows: 2, cols: 2, seqLen: 2, fakeToken: "F", imageToken: "I",
                globalImageToken: "G")
            let expected =
                "F<row_1_col_1>IIF<row_1_col_2>II\n"
                + "F<row_2_col_1>IIF<row_2_col_2>II\n"
                + "\nFGIIF"
            #expect(text == expected, "image prompt string")
        }

        @Test
        func videoPromptStringHasOneBlockPerFrame() throws {
            let processor = try Self.processor()
            let text = processor.getVideoPromptString(
                frameCount: 2, timeStamps: ["0:00:00", "0:00:01"], videoDuration: "0:00:02",
                seqLen: 3, fakeToken: "F", imageToken: "I", globalImageToken: "G")
            let expected =
                "You are provided the following series of 2 frames from a 0:00:02 [H:MM:SS] video.\n"
                + "\nFrame from 0:00:00:FGIIIF"
                + "\nFrame from 0:00:01:FGIIIF"
                + "\n\n"
            #expect(text == expected, "video prompt string")
        }
    }
}
