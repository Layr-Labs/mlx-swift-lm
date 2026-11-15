// Copyright © 2026 Eigen Labs.
// Native frame extraction for the reference Qwen3-VL / Qwen4 sampling policy.
@preconcurrency import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXLMCommon

enum Qwen4ExpVideoSampler {
    enum Failure: Error {
        case invalidMetadata, decodeFailed, missingFrame, excessiveMetadata
    }
    struct Sample {
        let frames: [MLXArray]
        let indices: [Int]
        let sourceFPS: Double
    }
    // A metadata admission limit, never a truncated frame list.
    private static let maxIndexedFrames = 1_000_000

    static func sample(
        _ video: UserInput.Video, config: Qwen4ExpVideoConfiguration,
        process: (CIImage, Int) throws -> CIImage
    ) async throws -> Sample {
        switch video.source {
        case .memoryBacked(let owner):
            return try await owner.withAsset { asset in
                try await sample(asset: asset, config: config, process: process)
            }
        case .avAsset(let asset):
            return try await sample(asset: asset, config: config, process: process)
        case .url(let url):
            let asset = AVURLAsset(
                url: url,
                options: [
                    AVURLAssetReferenceRestrictionsKey: AVAssetReferenceRestrictions.forbidAll
                        .rawValue
                ])
            return try await sample(asset: asset, config: config, process: process)
        case .frames(let frames):
            guard !frames.isEmpty, frames.count <= maxIndexedFrames else {
                throw Failure.invalidMetadata
            }
            let fps: Double
            if frames.count == 1 {
                // The reference's documented fallback for predecoded input
                // without enough metadata to infer a source rate.
                fps = 24
            } else {
                let span = (frames.last!.timeStamp - frames[0].timeStamp).seconds
                guard span.isFinite, span > 0 else { throw Failure.invalidMetadata }
                fps = Double(frames.count - 1) / span
            }
            let indices = try Qwen4ExpMediaGeometry.sampleIndices(
                totalFrames: frames.count, sourceFPS: fps,
                targetFPS: config.fps, minFrames: config.minFrames, maxFrames: config.maxFrames)
            var processed: [MLXArray] = []
            for index in indices {
                try Task.checkCancellation()
                processed.append(
                    MediaProcessing.asMLXArray(try process(frames[index].frame, indices.count)))
            }
            return .init(frames: processed, indices: indices, sourceFPS: fps)
        }
    }

    private static func sample(
        asset: AVAsset, config: Qwen4ExpVideoConfiguration,
        process: (CIImage, Int) throws -> CIImage
    ) async throws -> Sample {
        try Task.checkCancellation()
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw Failure.invalidMetadata
        }
        let range = try await track.load(.timeRange)
        guard range.duration.seconds.isFinite, range.duration.seconds > 0 else {
            throw Failure.invalidMetadata
        }
        let totalFrames = try await VideoReaderRetry.run {
            try compressedFrameCount(asset, track: track)
        }
        guard totalFrames > 0 else { throw Failure.decodeFailed }
        let fps = Double(totalFrames) / range.duration.seconds
        let indices = try Qwen4ExpMediaGeometry.sampleIndices(
            totalFrames: totalFrames, sourceFPS: fps,
            targetFPS: config.fps, minFrames: config.minFrames, maxFrames: config.maxFrames)
        let transform = try await track.load(.preferredTransform)
        let frames = try await VideoReaderRetry.run {
            try selectedFrames(
                asset, track: track, transform: transform, indices: indices,
                totalFrames: totalFrames, process: process)
        }
        return .init(frames: frames, indices: indices, sourceFPS: fps)
    }

    /// Counts compressed samples without retaining full-resolution frames.
    /// Do not seek using their PTS: MP4 edit lists can place compressed
    /// sample timestamps in a different clock from displayed frame times.
    private static func compressedFrameCount(_ asset: AVAsset, track: AVAssetTrack) throws -> Int {
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        guard reader.canAdd(output) else { throw Failure.decodeFailed }
        reader.add(output)
        guard reader.startReading() else { throw Failure.decodeFailed }
        defer { reader.cancelReading() }
        var totalFrames = 0
        while let buffer = VideoReaderRetry.nextSample(output, of: reader) {
            try Task.checkCancellation()
            let count = CMSampleBufferGetNumSamples(buffer)
            guard count >= 0, count <= maxIndexedFrames - totalFrames else {
                throw Failure.excessiveMetadata
            }
            totalFrames += count
        }
        guard reader.status == .completed else { throw VideoReaderFailure(reader: reader) }
        return totalFrames
    }

    /// Decodes in presentation order and selects the exact frame ordinals,
    /// matching the reference's frame-index contract. Only selected, resized
    /// frames survive each iteration; no full clip is retained.
    private static func selectedFrames(
        _ asset: AVAsset, track: AVAssetTrack, transform: CGAffineTransform, indices: [Int],
        totalFrames: Int, process: (CIImage, Int) throws -> CIImage
    ) throws -> [MLXArray] {
        let decodedReader = try AVAssetReader(asset: asset)
        let decoded = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        decoded.alwaysCopiesSampleData = false
        guard decodedReader.canAdd(decoded) else { throw Failure.decodeFailed }
        decodedReader.add(decoded)
        guard decodedReader.startReading() else { throw Failure.decodeFailed }
        defer { decodedReader.cancelReading() }
        var frames: [MLXArray] = []
        frames.reserveCapacity(indices.count)
        var ordinal = 0
        var selected = 0
        while let buffer = VideoReaderRetry.nextSample(decoded, of: decodedReader) {
            try Task.checkCancellation()
            guard ordinal < totalFrames, let pixel = CMSampleBufferGetImageBuffer(buffer) else {
                throw Failure.invalidMetadata
            }
            if selected < indices.count, ordinal == indices[selected] {
                // The reference preprocesses decoded RGB byte values, not a
                // second color-managed transfer-function conversion. Match
                // the existing native image path's explicit sRGB byte space.
                let ci = CIImage(
                    cvPixelBuffer: pixel,
                    options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!]
                ).transformed(by: transform)
                frames.append(MediaProcessing.asMLXArray(try process(ci, indices.count)))
                selected += 1
            }
            ordinal += 1
        }
        try Task.checkCancellation()
        guard decodedReader.status == .completed else {
            throw VideoReaderFailure(reader: decodedReader)
        }
        guard ordinal == totalFrames, frames.count == indices.count else {
            throw Failure.missingFrame
        }
        return frames
    }

    /// The same reference indices, read at presentation times: the frame count
    /// comes from the track's shown span (its first to its last non-empty edit)
    /// and nominal rate, and an image generator takes each selected frame at the
    /// middle of its interval (a time in an empty edit moves to the nearer shown
    /// frame). A clip trimmed without re-encoding, an empty edit and a long clip
    /// then need no full decode. A variable-rate clip is sampled evenly in time,
    /// where the reference samples it evenly in frame number.
    static func sampleByPresentationTime(
        _ video: UserInput.Video, config: Qwen4ExpVideoConfiguration,
        process: (CIImage, Int) throws -> CIImage
    ) async throws -> Sample {
        switch video.source {
        case .memoryBacked(let owner):
            return try await owner.withAsset { asset in
                try await sampleByTime(asset: asset, config: config, process: process)
            }
        case .avAsset(let asset):
            return try await sampleByTime(asset: asset, config: config, process: process)
        case .url(let url):
            let asset = AVURLAsset(
                url: url,
                options: [
                    AVURLAssetReferenceRestrictionsKey: AVAssetReferenceRestrictions.forbidAll
                        .rawValue
                ])
            return try await sampleByTime(asset: asset, config: config, process: process)
        case .frames:
            return try await sample(video, config: config, process: process)
        }
    }

    private static func sampleByTime(
        asset: AVAsset, config: Qwen4ExpVideoConfiguration,
        process: (CIImage, Int) throws -> CIImage
    ) async throws -> Sample {
        try Task.checkCancellation()
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw Failure.invalidMetadata
        }
        let (range, rate, segments) = try await track.load(
            .timeRange, .nominalFrameRate, .segments)
        let shown = segments.filter { !$0.isEmpty }.map(\.timeMapping.target)
        let start = shown.first?.start ?? range.start
        let duration = CMTimeSubtract(shown.last?.end ?? range.end, start).seconds
        guard duration.isFinite, duration > 0, rate.isFinite, rate > 0 else {
            throw Failure.invalidMetadata
        }
        let (frameCount, fps) = try shownFrames(duration: duration, nominalRate: rate)
        let indices = try Qwen4ExpMediaGeometry.sampleIndices(
            totalFrames: frameCount, sourceFPS: fps,
            targetFPS: config.fps, minFrames: config.minFrames, maxFrames: config.maxFrames)
        let latest = max(0, duration - 0.001 / fps)
        let times = indices.map { index in
            shownTime(
                CMTimeAdd(
                    start,
                    CMTime(
                        seconds: min((Double(index) + 0.5) / fps, latest),
                        preferredTimescale: 1_000_000)),
                in: shown, halfFrame: 0.5 / fps)
        }
        let resize = { (image: CIImage) in try process(image, indices.count) }
        var frames = try await generatedFrames(asset, times: times, resize)
        // One more read of a frame the generator dropped, as the readers retry.
        let missing = frames.indices.filter { frames[$0] == nil }
        if !missing.isEmpty {
            let retried = try await generatedFrames(asset, times: missing.map { times[$0] }, resize)
            for (slot, frame) in zip(missing, retried) { frames[slot] = frame }
        }
        let complete = frames.compactMap { $0 }
        guard complete.count == indices.count else { throw Failure.missingFrame }
        return .init(frames: complete, indices: indices, sourceFPS: fps)
    }

    /// The frame count of a shown span (its duration times the nominal rate,
    /// rounded) and the rate it implies. A `Float` rate of 30.000002 would
    /// otherwise sample one frame fewer than the reference's 30.
    static func shownFrames(duration: Double, nominalRate: Float) throws -> (
        count: Int, fps: Double
    ) {
        let count = (duration * Double(nominalRate)).rounded()
        guard count.isFinite, count >= 1 else { throw Failure.invalidMetadata }
        guard count <= Double(maxIndexedFrames) else { throw Failure.excessiveMetadata }
        return (Int(count), count / duration)
    }

    /// `time` when a shown (non-empty) segment holds it; otherwise the nearer
    /// edge of the shown segments, half a frame inside, since the generator has
    /// no image in an empty edit at any tolerance.
    static func shownTime(_ time: CMTime, in shown: [CMTimeRange], halfFrame: Double) -> CMTime {
        guard !shown.isEmpty, !shown.contains(where: { $0.containsTime(time) }) else { return time }
        let edges = shown.map { range -> CMTime in
            let inset = CMTime(
                seconds: min(halfFrame, range.duration.seconds / 2), preferredTimescale: 1_000_000)
            return CMTimeCompare(time, range.start) < 0
                ? CMTimeAdd(range.start, inset) : CMTimeSubtract(range.end, inset)
        }
        return edges.min { abs(($0 - time).seconds) < abs(($1 - time).seconds) }!
    }

    /// The processed frame at each time, nil where the generator has no image.
    private static func generatedFrames(
        _ asset: AVAsset, times: [CMTime], _ process: (CIImage) throws -> CIImage
    ) async throws -> [MLXArray?] {
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        var frames = [MLXArray?](repeating: nil, count: times.count)
        for await result in generator.images(for: times) {
            try Task.checkCancellation()
            guard case .success(let requested, let image, actualTime: _) = result,
                let slot = times.indices.first(where: {
                    times[$0] == requested && frames[$0] == nil
                })
            else { continue }
            // Decoded bytes as sRGB, as the frame-ordinal reader tags them.
            let frame = CIImage(
                cgImage: image, options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
            frames[slot] = MediaProcessing.asMLXArray(try process(frame))
        }
        return frames
    }
}
