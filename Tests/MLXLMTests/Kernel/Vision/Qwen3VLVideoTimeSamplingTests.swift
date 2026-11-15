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

    /// `Qwen4ExpVideoSampler.sampleByPresentationTime`, the reader of
    /// `Qwen3VLProcessor`'s timestamped layout: reference indices from the
    /// track's shown range and rate, each frame read at its time.
    @Suite
    struct Qwen3VLVideoTimeSamplingTests {

        private enum ClipError: Error { case setup, append, finish }

        private static func configuration(maxFrames: Int) throws -> Qwen4ExpVideoConfiguration {
            let object: [String: Any] = [
                "size": ["shortest_edge": 64, "longest_edge": 1 << 24],
                "patch_size": 16, "merge_size": 2, "temporal_patch_size": 2,
                "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                "fps": 2, "min_frames": 4, "max_frames": maxFrames,
            ]
            return try JSONDecoder().decode(
                Qwen4ExpVideoConfiguration.self,
                from: JSONSerialization.data(withJSONObject: object))
        }

        /// An H.264 clip of `frames` frames at `fps`, a keyframe every 10
        /// frames; frame `i` is gray at level `12 * i % 256`. `first` is the
        /// first frame's time (a later one leaves an empty edit before it);
        /// with `timescale`, a frame lasts `step` ticks of it.
        private static func writeClip(
            to url: URL, frames: Int, fps: Int32, first: CMTime = .zero, step: Int64 = 1,
            timescale: Int32? = nil
        ) async throws {
            let size = 64
            let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
            let input = AVAssetWriterInput(
                mediaType: .video,
                outputSettings: [
                    AVVideoCodecKey: AVVideoCodecType.h264,
                    AVVideoWidthKey: size, AVVideoHeightKey: size,
                    AVVideoCompressionPropertiesKey: [
                        AVVideoMaxKeyFrameIntervalKey: 10,
                        AVVideoAllowFrameReorderingKey: false,
                    ],
                ])
            input.expectsMediaDataInRealTime = false
            let adaptor = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: input,
                sourcePixelBufferAttributes: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                    kCVPixelBufferWidthKey as String: size, kCVPixelBufferHeightKey as String: size,
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
                guard
                    CVPixelBufferCreate(
                        kCFAllocatorDefault, size, size, kCVPixelFormatType_32BGRA, nil, &buffer)
                        == kCVReturnSuccess, let buffer
                else { throw ClipError.setup }
                CVPixelBufferLockBaseAddress(buffer, [])
                let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
                let level = UInt8(12 * index % 256)
                for row in 0 ..< size {
                    let line = base + row * CVPixelBufferGetBytesPerRow(buffer)
                    for x in 0 ..< size {
                        (line + 4 * x).update(repeating: level, count: 3)
                        line[4 * x + 3] = 255
                    }
                }
                CVPixelBufferUnlockBaseAddress(buffer, [])
                let time = CMTimeAdd(
                    first, CMTime(value: CMTimeValue(index) * step, timescale: timescale ?? fps))
                guard adaptor.append(buffer, withPresentationTime: time) else {
                    throw writer.error ?? ClipError.append
                }
            }
            input.markAsFinished()
            writer.endSession(
                atSourceTime: CMTimeAdd(
                    first, CMTime(value: CMTimeValue(frames) * step, timescale: timescale ?? fps)))
            await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
                writer.finishWriting { done.resume() }
            }
            guard writer.status == .completed else { throw writer.error ?? ClipError.finish }
        }

        /// Mean level of each sampled frame, 0...1.
        private static func levels(_ frames: [MLXArray]) -> [Float] {
            frames.map { frame in
                let value = mean(frame)
                eval(value)
                return value.item(Float.self)
            }
        }

        private static func temporaryClip(
            frames: Int, fps: Int32, first: CMTime = .zero, step: Int64 = 1,
            timescale: Int32? = nil
        ) async throws -> (URL, URL) {
            let folder = FileManager.default.temporaryDirectory
                .appendingPathComponent("qwen3vl-video-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let url = folder.appendingPathComponent("clip.mp4")
            try await writeClip(
                to: url, frames: frames, fps: fps, first: first, step: step, timescale: timescale)
            return (folder, url)
        }

        @Test
        func clipIsSampledAtTheReferenceIndices() async throws {
            let (folder, url) = try await Self.temporaryClip(frames: 20, fps: 10)
            defer { try? FileManager.default.removeItem(at: folder) }
            // 20 frames in 2 s at 10 fps: 2 fps asks for 4 frames, and the
            // indices are round(linspace(0, 19, 4)) = 0, 6, 13, 19.
            let sample = try await Qwen4ExpVideoSampler.sampleByPresentationTime(
                .avAsset(AVURLAsset(url: url)), config: Self.configuration(maxFrames: 64)
            ) { image, _ in image.toSRGB() }
            #expect(sample.indices == [0, 6, 13, 19])
            #expect(abs(sample.sourceFPS - 10) < 1e-3)
            let levels = Self.levels(sample.frames)
            for (got, index) in zip(levels, sample.indices) {
                #expect(abs(got - Float(12 * index) / 255) < 0.06, "frame \(index): \(levels)")
            }
        }

        /// A trim that keeps the encoded stream starts between keyframes: the
        /// decoder needs frames 0-2 to show frame 3, and the clip shows 17.
        @Test
        func trimmedClipIsSampledFromItsShownFrames() async throws {
            let (folder, url) = try await Self.temporaryClip(frames: 20, fps: 10)
            defer { try? FileManager.default.removeItem(at: folder) }
            let source = AVURLAsset(url: url)
            let track = try #require(try await source.loadTracks(withMediaType: .video).first)
            let composition = AVMutableComposition()
            let target = try #require(
                composition.addMutableTrack(
                    withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid))
            try target.insertTimeRange(
                CMTimeRange(
                    start: CMTime(value: 3, timescale: 10), end: CMTime(value: 20, timescale: 10)),
                of: track, at: .zero)

            let sample = try await Qwen4ExpVideoSampler.sampleByPresentationTime(
                .avAsset(composition), config: Self.configuration(maxFrames: 64)
            ) { image, _ in image.toSRGB() }
            // 17 shown frames at 10 fps: round(linspace(0, 16, 4)) = 0, 5, 11, 16,
            // which are source frames 3, 8, 14 and 19.
            #expect(sample.indices == [0, 5, 11, 16])
            let levels = Self.levels(sample.frames)
            for (got, index) in zip(levels, sample.indices) {
                #expect(
                    abs(got - Float(12 * (index + 3)) / 255) < 0.06, "shown \(index): \(levels)")
            }
        }

        /// `Qwen3VLProcessor` reads the trimmed clip (a frame-ordinal reader
        /// counts the decoder's lead-in frames and finds 3 missing).
        @Test
        func processorTakesATrimmedClip() async throws {
            let (folder, url) = try await Self.temporaryClip(frames: 20, fps: 10)
            defer { try? FileManager.default.removeItem(at: folder) }
            let source = AVURLAsset(url: url)
            let track = try #require(try await source.loadTracks(withMediaType: .video).first)
            let composition = AVMutableComposition()
            let target = try #require(
                composition.addMutableTrack(
                    withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid))
            try target.insertTimeRange(
                CMTimeRange(
                    start: CMTime(value: 3, timescale: 10), end: CMTime(value: 20, timescale: 10)),
                of: track, at: .zero)
            let input = UserInput(chat: [.user("Watch.", videos: [.avAsset(composition)])])
            let result = try await KernelTests.QwenVLProcessorTests.qwen3().prepare(input: input)
            #expect(result.video?.frames?.map(\.t) == [2], "4 frames make 2 pairs")
        }

        /// Media faults stay the processor's client errors (a 400 at the provider).
        @Test
        func unusableClipsAreProcessorErrors() async throws {
            let processor = try KernelTests.QwenVLProcessorTests.qwen3()
            // 4 px high: below the resize factor of 8.
            let flat = KernelTests.QwenVLProcessorTests.video(
                count: 4, step: 0.5, width: 64, height: 4)
            await #expect(
                throws: VLMError.imageProcessingFailure("Qwen video frames cannot be resized")
            ) {
                _ = try await processor.prepare(
                    input: UserInput(chat: [.user("Watch.", videos: [flat])]))
            }
            await #expect(throws: VLMError.videoNotDecodable, "no video track") {
                _ = try await processor.prepare(
                    input: UserInput(chat: [
                        .user("Watch.", videos: [.avAsset(AVMutableComposition())])
                    ]))
            }
        }

        /// The first frame at 0.5 s leaves an empty edit before it; times count
        /// from the first shown frame (the generator has no image in the gap).
        @Test
        func leadingEmptyEditIsSkipped() async throws {
            let (folder, url) = try await Self.temporaryClip(
                frames: 20, fps: 10, first: CMTime(value: 1, timescale: 2))
            defer { try? FileManager.default.removeItem(at: folder) }
            let sample = try await Qwen4ExpVideoSampler.sampleByPresentationTime(
                .avAsset(AVURLAsset(url: url)), config: Self.configuration(maxFrames: 64)
            ) { image, _ in image.toSRGB() }
            #expect(sample.indices == [0, 6, 13, 19], "20 shown frames")
            let levels = Self.levels(sample.frames)
            for (got, index) in zip(levels, sample.indices) {
                #expect(abs(got - Float(12 * index) / 255) < 0.06, "frame \(index): \(levels)")
            }
        }

        /// 360 frames at 29.97 fps (1,001 ticks of 1/30,000 s): the count, not
        /// the Float nominal rate, sets the rate, so 2 fps asks for 24 frames.
        @Test
        func ntscRateGivesTheReferenceCount() async throws {
            let (folder, url) = try await Self.temporaryClip(
                frames: 360, fps: 30, step: 1_001, timescale: 30_000)
            defer { try? FileManager.default.removeItem(at: folder) }
            let sample = try await Qwen4ExpVideoSampler.sampleByPresentationTime(
                .avAsset(AVURLAsset(url: url)), config: Self.configuration(maxFrames: 64)
            ) { image, _ in image }
            #expect(sample.indices.count == 24)
            #expect(sample.indices.last == 359)
        }
    }
}
