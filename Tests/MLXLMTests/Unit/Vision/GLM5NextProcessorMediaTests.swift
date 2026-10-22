import CoreImage
import CoreMedia
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

extension UnitTests {
    /// `prepareMediaValues` is the media half of `prepareDetailed` as host
    /// values: for every image and video the same patches (bit for bit), grid
    /// and timestamps. A cluster rank that holds no tokenizer prepares media
    /// with it and compares patch digests with its leader's `prepareDetailed`
    /// on every request.
    @Suite struct GLM5NextProcessorMediaTests {
        static func image(width: Int, height: Int, seed: Int) -> CIImage {
            var bytes = [UInt8](repeating: 255, count: width * height * 4)
            for i in 0 ..< width * height {
                for c in 0 ..< 3 { bytes[i * 4 + c] = UInt8(truncatingIfNeeded: (i &* 37) &+ seed &* 11 &+ c &* 71) }
            }
            return CIImage(bitmapData: Data(bytes), bytesPerRow: width * 4,
                size: CGSize(width: width, height: height), format: .RGBA8,
                colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
        }

        /// The items' values, concatenated, against the pixels' bits.
        static func bitwiseEqual(_ items: [GLM5NextPreparedMediaValues], _ pixels: MLXArray) -> Bool {
            let values = items.flatMap(\.values)
            return pixels.dtype == .float32 && pixels.size == values.count
                && pixels.asArray(Float.self).map(\.bitPattern) == values.map(\.bitPattern)
        }

        @Test func prepareMediaValuesAreTheMediaHalfOfPrepareDetailed() async throws {
            try await Device.withDefaultDevice(.cpu) {
                let images: [UserInput.Image] = [
                    .ciImage(Self.image(width: 37, height: 23, seed: 1)),
                    .ciImage(Self.image(width: 64, height: 40, seed: 2)),
                ]
                let frames = (0 ..< 6).map {
                    UserInput.VideoFrame(frame: Self.image(width: 30, height: 20, seed: 10 + $0),
                        timeStamp: CMTime(value: Int64($0), timescale: 4))
                }
                let videos: [UserInput.Video] = [.frames(frames)]
                let processor = try GLM5NextProcessor(configuration: .init(), tokenizer: GLM5NextTestTokenizer())
                let detailed = try await processor.prepareDetailed(
                    input: UserInput(chat: [.user("look", images: images, videos: videos)]))
                let media = try await processor.prepareMediaValues(images: images, videos: videos)
                let image = try #require(detailed.input.image), video = try #require(detailed.input.video)
                #expect(media.images.count == 2 && media.videos.count == 1)
                #expect(Self.bitwiseEqual(media.images, image.pixels))
                #expect(Self.bitwiseEqual(media.videos, video.pixels))
                #expect(media.images.map { [$0.grid.thw.t, $0.grid.thw.h, $0.grid.thw.w] }
                    == image.frames?.map { [$0.t, $0.h, $0.w] })
                #expect(media.videos.map { [$0.grid.thw.t, $0.grid.thw.h, $0.grid.thw.w] }
                    == video.frames?.map { [$0.t, $0.h, $0.w] })
                // The video's pair timestamps are the ones its prompt expanded with.
                #expect(media.videos[0].timestamps.count == media.videos[0].grid.temporal)
                // The same aggregate budget refuses the same media.
                let small = try GLM5NextProcessor(imageConfiguration: .init(
                    maximumPreparedBytes: 6 * 14 * 14 * 4 * 16))
                await #expect(throws: GLM5NextVisionError.self) {
                    _ = try await small.prepareMediaValues(images: images, videos: [])
                }
            }
        }
    }
}
