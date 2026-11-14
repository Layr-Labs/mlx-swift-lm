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
    /// and rate come from the track's shown time range and nominal rate, and an
    /// image generator takes each selected frame at the middle of its interval.
    /// A clip trimmed without re-encoding (an edit list that starts between
    /// keyframes) and a long clip then need no full decode.
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
        let (range, rate) = try await track.load(.timeRange, .nominalFrameRate)
        let duration = range.duration.seconds
        let fps = Double(rate)
        guard duration.isFinite, duration > 0, fps.isFinite, fps > 0 else {
            throw Failure.invalidMetadata
        }
        let frameCount = (duration * fps).rounded()
        guard frameCount >= 1 else { throw Failure.invalidMetadata }
        guard frameCount <= Double(maxIndexedFrames) else { throw Failure.excessiveMetadata }
        let indices = try Qwen4ExpMediaGeometry.sampleIndices(
            totalFrames: Int(frameCount), sourceFPS: fps,
            targetFPS: config.fps, minFrames: config.minFrames, maxFrames: config.maxFrames)
        let latest = max(0, duration - 0.001 / fps)
        let times = indices.map { index in
            CMTimeAdd(
                range.start,
                CMTime(
                    seconds: min((Double(index) + 0.5) / fps, latest),
                    preferredTimescale: 1_000_000))
        }
        // One more pass when the generator drops a frame, as the readers retry.
        for attempt in 0 ..< 2 {
            if let frames = try await generatedFrames(
                asset, times: times, process: { try process($0, indices.count) })
            {
                return .init(frames: frames, indices: indices, sourceFPS: fps)
            }
            if attempt == 0 { try Task.checkCancellation() }
        }
        throw Failure.missingFrame
    }

    /// Every requested frame in order, or nil when the generator fails one.
    private static func generatedFrames(
        _ asset: AVAsset, times: [CMTime], process: (CIImage) throws -> CIImage
    ) async throws -> [MLXArray]? {
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        var frames: [MLXArray] = []
        frames.reserveCapacity(times.count)
        for await result in generator.images(for: times) {
            try Task.checkCancellation()
            guard case .success(requestedTime: _, let image, actualTime: _) = result else {
                return nil
            }
            // Decoded bytes as sRGB, as the frame-ordinal reader tags them.
            let frame = CIImage(
                cgImage: image, options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
            frames.append(MediaProcessing.asMLXArray(try process(frame)))
        }
        return frames.count == times.count ? frames : nil
    }
}
