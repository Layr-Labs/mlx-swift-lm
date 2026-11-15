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

        /// The count, not the `Float` nominal rate, sets the rate: 12 s at
        /// 30.000002 fps is 360 frames at 30 fps, and 2 fps then asks for 24
        /// frames where the raw rate gives 23.
        @Test
        func frameCountSetsTheRate() throws {
            let (count, fps) = try Qwen4ExpVideoSampler.shownFrames(
                duration: 12, nominalRate: 30.000002)
            #expect(count == 360)
            #expect(fps == 30)
            let reference = try Qwen4ExpMediaGeometry.sampleIndices(
                totalFrames: count, sourceFPS: fps, maxFrames: 64)
            let raw = try Qwen4ExpMediaGeometry.sampleIndices(
                totalFrames: count, sourceFPS: Double(Float(30.000002)), maxFrames: 64)
            #expect(reference.count == 24)
            #expect(raw.count == 23, "what the raw Float rate would sample")
        }

        @Test
        func timeInAnEmptyEditMovesToTheNearerShownFrame() {
            let shown = [
                CMTimeRange(start: .zero, end: CMTime(value: 1, timescale: 1)),
                CMTimeRange(
                    start: CMTime(value: 2, timescale: 1), end: CMTime(value: 3, timescale: 1)),
            ]
            func moved(_ seconds: Double) -> Double {
                Qwen4ExpVideoSampler.shownTime(
                    CMTime(seconds: seconds, preferredTimescale: 1_000_000), in: shown,
                    halfFrame: 0.05
                ).seconds
            }
            #expect(moved(0.5) == 0.5, "a shown time stays")
            #expect(abs(moved(1.4) - 0.95) < 1e-6, "nearer the first segment's end")
            #expect(abs(moved(1.6) - 2.05) < 1e-6, "nearer the second segment's start")
            #expect(abs(moved(3.5) - 2.95) < 1e-6, "after the last segment")
        }

        /// A pass-through export of 0-1 s, a 1 s empty edit, then 1-2 s of the
        /// clip: two of the six sampled times fall in the gap.
        @Test
        func interiorEmptyEditIsReadFromTheNearerFrames() async throws {
            let (folder, url) = try await Self.temporaryClip(frames: 20, fps: 10)
            defer { try? FileManager.default.removeItem(at: folder) }
            let source = AVURLAsset(url: url)
            let track = try #require(try await source.loadTracks(withMediaType: .video).first)
            let composition = AVMutableComposition()
            let target = try #require(
                composition.addMutableTrack(
                    withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid))
            let second = CMTime(value: 1, timescale: 1)
            try target.insertTimeRange(
                CMTimeRange(start: .zero, duration: second), of: track, at: .zero)
            try target.insertTimeRange(
                CMTimeRange(start: second, duration: second), of: track, at: second)
            target.insertEmptyTimeRange(CMTimeRange(start: second, duration: second))
            let exported = folder.appendingPathComponent("gap.mp4")
            let session = try #require(
                AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetPassthrough)
            )
            session.outputURL = exported
            session.outputFileType = .mp4
            await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
                session.exportAsynchronously { done.resume() }
            }
            #expect(session.status == .completed, "\(String(describing: session.error))")

            let sample = try await Qwen4ExpVideoSampler.sampleByPresentationTime(
                .avAsset(AVURLAsset(url: exported)), config: Self.configuration(maxFrames: 64)
            ) { image, _ in image }
            #expect(sample.frames.count == sample.indices.count)
            #expect(sample.indices.count == 6, "3 s at 10 fps, 2 fps")
        }

        /// 2 s, a 5 s empty edit, then 3 s: ten sampled times fall in the gap;
        /// five move to each edge's frame, which they share.
        @Test
        func longInteriorEmptyEditSharesTheEdgeFrames() async throws {
            let (folder, url) = try await Self.temporaryClip(frames: 50, fps: 10)
            defer { try? FileManager.default.removeItem(at: folder) }
            let source = AVURLAsset(url: url)
            let track = try #require(try await source.loadTracks(withMediaType: .video).first)
            let composition = AVMutableComposition()
            let target = try #require(
                composition.addMutableTrack(
                    withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid))
            let two = CMTime(value: 2, timescale: 1)
            try target.insertTimeRange(
                CMTimeRange(start: .zero, duration: two), of: track, at: .zero)
            try target.insertTimeRange(
                CMTimeRange(start: two, duration: CMTime(value: 3, timescale: 1)), of: track,
                at: two)
            target.insertEmptyTimeRange(
                CMTimeRange(start: two, duration: CMTime(value: 5, timescale: 1)))
            let exported = try await Self.passThrough(
                composition, to: folder.appendingPathComponent("long.mp4"))

            let sample = try await Qwen4ExpVideoSampler.sampleByPresentationTime(
                .avAsset(AVURLAsset(url: exported)), config: Self.configuration(maxFrames: 64)
            ) { image, _ in image.toSRGB() }
            // 10 s at 10 fps, 2 fps: 20 times, every 0.5 s; those in 2-7 s move
            // to source frame 19 (before the gap) or 20 (after it).
            #expect(sample.indices.count == 20)
            let levels = Self.levels(sample.frames)
            let times = sample.indices.map { (Double($0) + 0.5) / 10 }
            for (time, got) in zip(times, levels) {
                let frame: Int =
                    time < 2
                    ? Int(time * 10) : time < 4.5 ? 19 : time < 7 ? 20 : Int(time * 10) - 50
                // 0.02 tells frames 19 and 20 (12 / 255 apart) from each other.
                #expect(abs(got - Float(12 * frame % 256) / 255) < 0.02, "\(time) s: \(levels)")
            }
        }

        private static func passThrough(_ asset: AVAsset, to url: URL) async throws -> URL {
            let session = try #require(
                AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetPassthrough))
            session.outputURL = url
            session.outputFileType = .mp4
            await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
                session.exportAsynchronously { done.resume() }
            }
            #expect(session.status == .completed, "\(String(describing: session.error))")
            return url
        }
    }
}
