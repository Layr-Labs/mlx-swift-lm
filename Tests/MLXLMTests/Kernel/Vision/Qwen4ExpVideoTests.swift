@preconcurrency import AVFoundation
import CoreImage
import CoreMedia
import CoreVideo
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

extension KernelTests {

    /// `Qwen4ExpVideoSampler.sample`: frame selection for decoded frames and
    /// for a small clip that the test writes to a temporary folder.
    @Suite
    struct Qwen4ExpVideoSamplerKernelTests {

        private enum ClipError: Error { case setup, append, finish }
        private struct ProcessError: Error {}

        private static func configuration(
            fps: Double = 2, minFrames: Int = 4, maxFrames: Int = 768
        ) throws -> Qwen4ExpVideoConfiguration {
            let object: [String: Any] = [
                "size": ["shortest_edge": 64, "longest_edge": 4096],
                "patch_size": 2, "merge_size": 2, "temporal_patch_size": 2,
                "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                "fps": fps, "min_frames": minFrames, "max_frames": maxFrames,
            ]
            return try JSONDecoder().decode(
                Qwen4ExpVideoConfiguration.self,
                from: JSONSerialization.data(withJSONObject: object))
        }

        private static func frame(width: Int, height: Int, seconds: Double)
            -> UserInput.VideoFrame
        {
            let image = CIImage(color: CIColor(red: 0.2, green: 0.4, blue: 0.6))
                .cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
            return UserInput.VideoFrame(
                frame: image, timeStamp: CMTime(seconds: seconds, preferredTimescale: 600))
        }

        @Test
        func decodedFramesAreSampledEvenly() async throws {
            // 8 frames at 0 s ... 7 s give a source rate of 7 / 7 = 1 fps.
            // The target rate asks for 8 / 1 * 2 = 16 frames; max_frames
            // keeps 3. The step is 7 / 2 = 3.5, and 3.5 rounds to even (4).
            // Frame i is 8 + i pixels wide, so the width tells which frame
            // the sampler took.
            let frames = (0 ..< 8).map {
                Self.frame(width: 8 + $0, height: 8, seconds: Double($0))
            }
            var counts: [Int] = []
            let sample = try await Qwen4ExpVideoSampler.sample(
                .frames(frames), config: Self.configuration(minFrames: 2, maxFrames: 3)
            ) { image, count in
                counts.append(count)
                return image
            }
            #expect(sample.indices == [0, 4, 7], "evenly spaced indices, last frame kept")
            // Exact: 7 / 7 seconds is exact in Double.
            #expect(sample.sourceFPS == 1, "source rate from the timestamps")
            #expect(counts == [3, 3, 3], "the closure gets the selected frame count")
            #expect(
                sample.frames.map(\.shape) == [[1, 3, 8, 8], [1, 3, 8, 12], [1, 3, 8, 15]],
                "each frame is the selected source frame")
        }

        @Test
        func oneDecodedFrameUsesTheFallbackRate() async throws {
            let sample = try await Qwen4ExpVideoSampler.sample(
                .frames([Self.frame(width: 8, height: 8, seconds: 5)]),
                config: Self.configuration()
            ) { image, _ in image }
            #expect(sample.indices == [0], "one frame gives index 0")
            #expect(sample.sourceFPS == 24, "fallback source rate is 24 fps")
            #expect(sample.frames.map(\.shape) == [[1, 3, 8, 8]], "one frame array")
        }

        @Test
        func invalidDecodedFrameListsAreRejected() async throws {
            let config = try Self.configuration()
            await #expect(
                throws: Qwen4ExpVideoSampler.Failure.invalidMetadata, "no frames"
            ) {
                _ = try await Qwen4ExpVideoSampler.sample(.frames([]), config: config) { i, _ in i }
            }
            let same = [
                Self.frame(width: 8, height: 8, seconds: 1),
                Self.frame(width: 8, height: 8, seconds: 1),
            ]
            await #expect(
                throws: Qwen4ExpVideoSampler.Failure.invalidMetadata, "zero time span"
            ) {
                _ = try await Qwen4ExpVideoSampler.sample(.frames(same), config: config) { i, _ in i
                }
            }
            let two = [
                Self.frame(width: 8, height: 8, seconds: 0),
                Self.frame(width: 8, height: 8, seconds: 1),
            ]
            await #expect(throws: ProcessError.self, "an error of the closure goes to the caller") {
                _ = try await Qwen4ExpVideoSampler.sample(.frames(two), config: config) { _, _ in
                    throw ProcessError()
                }
            }
        }

        /// Writes a Motion-JPEG clip of `frames` frames at `fps`. Frame `i`
        /// has red `51 * i` (so red is `i / 5` of full scale), and no green
        /// or blue.
        private static func writeClip(to url: URL, frames: Int, fps: Int32, size: Int)
            async throws
        {
            let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
            let input = AVAssetWriterInput(
                mediaType: .video,
                outputSettings: [
                    AVVideoCodecKey: AVVideoCodecType.jpeg,
                    AVVideoWidthKey: size,
                    AVVideoHeightKey: size,
                    AVVideoCompressionPropertiesKey: [AVVideoQualityKey: 1.0],
                ])
            input.expectsMediaDataInRealTime = false
            let adaptor = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: input,
                sourcePixelBufferAttributes: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                    kCVPixelBufferWidthKey as String: size,
                    kCVPixelBufferHeightKey as String: size,
                ])
            guard writer.canAdd(input) else { throw ClipError.setup }
            writer.add(input)
            guard writer.startWriting() else { throw writer.error ?? ClipError.setup }
            writer.startSession(atSourceTime: .zero)
            for index in 0 ..< frames {
                while !input.isReadyForMoreMediaData {
                    try await Task.sleep(nanoseconds: 1_000_000)
                }
                var buffer: CVPixelBuffer?
                let status = CVPixelBufferCreate(
                    kCFAllocatorDefault, size, size, kCVPixelFormatType_32BGRA, nil, &buffer)
                guard status == kCVReturnSuccess, let buffer else { throw ClipError.setup }
                CVPixelBufferLockBaseAddress(buffer, [])
                let base = CVPixelBufferGetBaseAddress(buffer)!
                    .assumingMemoryBound(to: UInt8.self)
                let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
                for y in 0 ..< size {
                    for x in 0 ..< size {
                        let pixel = base + y * rowBytes + x * 4
                        pixel[0] = 0  // blue
                        pixel[1] = 0  // green
                        pixel[2] = UInt8(51 * index)  // red
                        pixel[3] = 255  // alpha
                    }
                }
                CVPixelBufferUnlockBaseAddress(buffer, [])
                let time = CMTime(value: CMTimeValue(index), timescale: fps)
                guard adaptor.append(buffer, withPresentationTime: time) else {
                    throw writer.error ?? ClipError.append
                }
            }
            input.markAsFinished()
            writer.endSession(atSourceTime: CMTime(value: CMTimeValue(frames), timescale: fps))
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                writer.finishWriting { continuation.resume() }
            }
            guard writer.status == .completed else { throw writer.error ?? ClipError.finish }
        }

        @Test
        func writtenClipIsSampledByFrameOrdinal() async throws {
            let folder = FileManager.default.temporaryDirectory
                .appendingPathComponent("qwen4-video-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(
                at: folder, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: folder) }
            let url = folder.appendingPathComponent("clip.mov")
            try await Self.writeClip(to: url, frames: 6, fps: 2, size: 32)

            // 6 frames in 3 s give 2 fps. The target asks for 6 frames;
            // max_frames keeps 3. The step is 5 / 2 = 2.5, and 2.5 rounds to
            // even (2).
            let config = try Self.configuration(minFrames: 2, maxFrames: 3)
            for video in [UserInput.Video.url(url), .avAsset(AVURLAsset(url: url))] {
                let sample = try await Qwen4ExpVideoSampler.sample(video, config: config) {
                    image, _ in image.toSRGB()
                }
                #expect(sample.indices == [0, 2, 5], "frame ordinals 0, 2 and 5")
                // 1e-6: the duration comes from the track time scale.
                #expect(abs(sample.sourceFPS - 2) < 1e-6, "6 frames in 3 s")
                #expect(
                    sample.frames.map(\.shape) == Array(repeating: [1, 3, 32, 32], count: 3),
                    "three 32 x 32 RGB frames")
                let red = sample.frames.map { frame -> Float in
                    let value = mean(frame[0..., 0, 0..., 0...])
                    eval(value)
                    return value.item(Float.self)
                }
                // 0.08: JPEG coding and the YCbCr round trip change a solid
                // color by a few 8-bit steps.
                for (got, expected) in zip(red, [Float(0), 0.4, 1.0]) {
                    #expect(abs(got - expected) < 0.08, "red of the selected frame: \(red)")
                }
            }
        }

        @Test
        func fileThatIsNotAVideoIsRejected() async throws {
            let folder = FileManager.default.temporaryDirectory
                .appendingPathComponent("qwen4-video-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(
                at: folder, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: folder) }
            let url = folder.appendingPathComponent("not-a-clip.mov")
            try Data("not a movie".utf8).write(to: url)
            let config = try Self.configuration()
            await #expect(throws: (any Error).self, "text bytes are not a clip") {
                _ = try await Qwen4ExpVideoSampler.sample(.url(url), config: config) { i, _ in i }
            }
        }
    }

    /// `Qwen4ExpProcessor.prepare(input:)` with a video, which runs
    /// `Qwen4ExpVideoPreparation.prepare` and `Qwen4ExpVideoPrompt`.
    @Suite
    struct Qwen4ExpVideoPreparationKernelTests {

        /// One token per Unicode scalar. The chat template is a fixed text.
        private struct TemplateTokenizer: Tokenizer {
            let template: String
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

            func applyChatTemplate(
                messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                additionalContext: [String: any Sendable]?
            ) throws -> [Int] {
                encode(text: template, addSpecialTokens: false)
            }
        }

        private static let start = "<|vision_start|>"
        private static let pad = "<|video_pad|>"
        private static let end = "<|vision_end|>"

        /// Patch 2, merge 2, temporal patch 2, mean and std 0.5. The video
        /// budget is 64 ... 4096 pixels for all frames together.
        private static func processor(
            template: String = "<|vision_start|><|video_pad|><|vision_end|>hi",
            withVideo: Bool = true
        ) throws -> Qwen4ExpProcessor {
            var object: [String: Any] = [
                "size": ["shortest_edge": 64, "longest_edge": 4096],
                "patch_size": 2, "merge_size": 2, "temporal_patch_size": 2,
                "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                "image_processor_type": "Qwen2VLImageProcessorFast",
            ]
            if withVideo {
                object["video_processor"] = [
                    "size": ["shortest_edge": 64, "longest_edge": 4096],
                    "patch_size": 2, "merge_size": 2, "temporal_patch_size": 2,
                    "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                    "fps": 2, "min_frames": 4, "max_frames": 768,
                ]
            }
            let config = try JSONDecoder().decode(
                Qwen4ExpProcessorConfiguration.self,
                from: JSONSerialization.data(withJSONObject: object))
            return Qwen4ExpProcessor(config, tokenizer: TemplateTokenizer(template: template))
        }

        /// Frames at 0, 1, 2 and 3 s. `sizes[i]` is the side of frame `i`.
        private static func clip(sizes: [Int]) -> UserInput.Video {
            .frames(
                sizes.enumerated().map { index, side in
                    let image = CIImage(color: CIColor(red: 1, green: 0, blue: 0))
                        .cropped(to: CGRect(x: 0, y: 0, width: side, height: side))
                    return UserInput.VideoFrame(
                        frame: image,
                        timeStamp: CMTime(seconds: Double(index), preferredTimescale: 600))
                })
        }

        private static func scalars(_ text: String) -> [Int] {
            text.unicodeScalars.map { Int($0.value) }
        }

        @Test
        func videoPromptGetsTimestampedFrameBlocks() async throws {
            // 4 frames in 3 s give 1 fps, so the sampler keeps all 4
            // (indices 0 ... 3). 4 frames of 16 x 16 are 1024 pixels, inside
            // the budget, so the frames stay 16 x 16. The grid is
            // T = 4 / 2 = 2, H = W = 16 / 2 = 8; each temporal group has
            // (8 / 2) * (8 / 2) = 16 tokens. The group times are
            // (0 + 1) / 2 = 0.5 s and (2 + 3) / 2 = 2.5 s.
            let input = UserInput(
                prompt: .text("hi"), videos: [Self.clip(sizes: [16, 16, 16, 16])])
            let result = try await Self.processor().prepare(input: input)

            let block = Self.start + String(repeating: Self.pad, count: 16) + Self.end
            let expected = Self.scalars("<0.5 seconds>\(block)<2.5 seconds>\(block)hi")
            #expect(result.text.tokens.shape == [1, expected.count], "token shape")
            #expect(
                result.text.tokens.asType(.int32).asArray(Int32.self).map(Int.init) == expected,
                "expanded prompt tokens")
            let mask = try #require(result.text.mask)
            #expect(mask.dtype == .int8, "mask dtype")
            #expect(
                mask.asType(.int32).sum().item(Int32.self) == Int32(expected.count),
                "every token is visible")
            #expect(result.image == nil, "no image input")

            let video = try #require(result.video)
            #expect(
                video.frames?.map { [$0.t, $0.h, $0.w] } == [[2, 8, 8]], "video grid T x H x W")
            // Rows: 2 * 8 * 8 patches. Columns: 3 channels * 2 * 2 * 2.
            #expect(video.pixels.shape == [128, 24], "patch matrix shape")
            #expect(isFinite(video.pixels).all().item(Bool.self), "every pixel is finite")
        }

        @Test
        func frameSizeChangeInsideAClipIsRejected() async throws {
            // Frame 0 is 16 x 16 and frame 1 is 32 x 32. 4 frames of
            // 32 x 32 are 4096 pixels, inside the budget, so the two frames
            // get different target sizes.
            let input = UserInput(
                prompt: .text("hi"), videos: [Self.clip(sizes: [16, 32, 16, 32])])
            await #expect(
                throws: VLMError.processing("Qwen4 video changes frame geometry within a clip")
            ) {
                _ = try await Self.processor().prepare(input: input)
            }
        }

        @Test
        func frameWithoutFiniteSizeIsRejected() async throws {
            // A color image without a crop has an infinite extent.
            let frames = (0 ..< 4).map {
                UserInput.VideoFrame(
                    frame: CIImage(color: CIColor(red: 1, green: 0, blue: 0)),
                    timeStamp: CMTime(seconds: Double($0), preferredTimescale: 600))
            }
            let input = UserInput(prompt: .text("hi"), videos: [.frames(frames)])
            await #expect(throws: Qwen4ExpMediaGeometry.Failure.invalidGeometry) {
                _ = try await Self.processor().prepare(input: input)
            }
        }

        @Test
        func videoWithoutVideoMetadataIsRejected() async throws {
            let input = UserInput(
                prompt: .text("hi"), videos: [Self.clip(sizes: [16, 16, 16, 16])])
            await #expect(
                throws: VLMError.processing("Qwen4 video processor metadata is unavailable")
            ) {
                _ = try await Self.processor(withVideo: false).prepare(input: input)
            }
        }

        @Test
        func templateWithoutVideoSlotIsRejected() async throws {
            let input = UserInput(
                prompt: .text("hi"), videos: [Self.clip(sizes: [16, 16, 16, 16])])
            await #expect(
                throws: VLMError.processing(
                    "Qwen4 video placeholder count does not match supplied clips")
            ) {
                _ = try await Self.processor(template: "hi").prepare(input: input)
            }
        }
    }
}
