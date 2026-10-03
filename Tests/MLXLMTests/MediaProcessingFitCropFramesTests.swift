import CoreGraphics
import CoreImage
import CoreMedia
import Foundation
import MLXLMCommon
import Testing

@testable import MLXVLM

/// Regression tests for three helpers of `MediaProcessing`:
///
/// - `fitIn(_:longestEdge:)` must keep the aspect ratio when it shrinks a size.
/// - `centerCrop(_:size:)` for a `CGRect` must center the crop in an extent
///   that does not start at 0.
/// - The `.frames` video path must sample frames whose time stamps do not
///   start at 0.
///
/// The tests are copied from `MediaGeometryTests` and
/// `MediaProcessingVideoTests` of PR #244, without the known issues.
@Suite
struct MediaProcessingFitCropFramesTests {

    @Test
    func fitInLongestEdgeShrinksLargeSizes() {
        let wide = MediaProcessing.fitIn(CGSize(width: 200, height: 100), longestEdge: 50)
        let tall = MediaProcessing.fitIn(CGSize(width: 100, height: 200), longestEdge: 50)
        #expect(wide == CGSize(width: 50, height: 25), "longest edge shrinks a wide size")
        #expect(tall == CGSize(width: 25, height: 50), "longest edge shrinks a tall size")
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

    /// An extent at the origin gives the same crop as before the fix.
    @Test
    func centerCropRectAtTheOrigin() {
        let crop = MediaProcessing.centerCrop(
            CGRect(x: 0, y: 0, width: 8, height: 6), size: CGSize(width: 4, height: 4))
        #expect(crop == CGRect(x: 2, y: 1, width: 4, height: 4), "center crop of 8x6 to 4x4")
    }

    @Test
    func framesThatStartAfterZeroAreSampled() async throws {
        // Frames at 5, 6 and 7 s. The duration is 2 s, so 1 fps asks for
        // round(1 * 2) = 2 frames.
        let frames = (5 ..< 8).map {
            VideoFrame(
                frame: Self.solid(.red), timeStamp: CMTime(value: Int64($0), timescale: 1))
        }
        let processed = try await MediaProcessing.asProcessedSequence(
            .frames(frames), samplesPerSecond: 1)
        #expect(processed.totalDuration == CMTime(value: 2, timescale: 1), "frame range")
        #expect(processed.frames.count == 2, "frames after time zero")
        // The samples are at 5 s and 7 s, the first and the last frame.
        #expect(
            processed.timestamps
                == [CMTime(value: 5, timescale: 1), CMTime(value: 7, timescale: 1)],
            "time stamps of the sampled frames")
    }

    /// A solid 4x4 image. Copied from `MediaProcessingVideoTests.solid` of PR #244.
    private static func solid(_ color: CIColor) -> CIImage {
        CIImage(color: color).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4))
    }
}
