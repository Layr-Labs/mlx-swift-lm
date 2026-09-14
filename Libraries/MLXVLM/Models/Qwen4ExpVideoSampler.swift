// Copyright © 2026 Eigen Labs.
// Native frame extraction for the reference Qwen4 sampling policy.
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

    static func sample(_ video: UserInput.Video, config: Qwen4ExpVideoConfiguration,
                       process: (CIImage, Int) throws -> CIImage) async throws -> Sample {
        switch video {
        case .memoryBacked(let owner):
            return try await owner.withAsset { asset in
                try await sample(asset: asset, config: config, process: process)
            }
        case .avAsset(let asset):
            return try await sample(asset: asset, config: config, process: process)
        case .url(let url):
            let asset = AVURLAsset(url: url, options: [
                AVURLAssetReferenceRestrictionsKey: AVAssetReferenceRestrictions.forbidAll.rawValue,
            ])
            return try await sample(asset: asset, config: config, process: process)
        case .frames(let frames):
            guard !frames.isEmpty, frames.count <= maxIndexedFrames else { throw Failure.invalidMetadata }
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
            let indices = try Qwen4ExpMediaGeometry.sampleIndices(totalFrames: frames.count, sourceFPS: fps,
                targetFPS: config.fps, minFrames: config.minFrames, maxFrames: config.maxFrames)
            var processed: [MLXArray] = []
            for index in indices {
                try Task.checkCancellation()
                processed.append(MediaProcessing.asMLXArray(try process(frames[index].frame, indices.count)))
            }
            return .init(frames: processed, indices: indices, sourceFPS: fps)
        }
    }

    private static func sample(asset: AVAsset, config: Qwen4ExpVideoConfiguration,
                               process: (CIImage, Int) throws -> CIImage) async throws -> Sample {
        try Task.checkCancellation()
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw Failure.invalidMetadata }
        let range = try await track.load(.timeRange)
        guard range.duration.seconds.isFinite, range.duration.seconds > 0 else { throw Failure.invalidMetadata }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        guard reader.canAdd(output) else { throw Failure.decodeFailed }
        reader.add(output)
        guard reader.startReading() else { throw Failure.decodeFailed }
        defer { reader.cancelReading() }
        // Count compressed samples without retaining full-resolution frames.
        // Do not seek using their PTS: MP4 edit lists can place compressed
        // sample timestamps in a different clock from displayed frame times.
        var totalFrames = 0
        while let buffer = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            let count = CMSampleBufferGetNumSamples(buffer)
            guard count >= 0, count <= maxIndexedFrames - totalFrames else { throw Failure.excessiveMetadata }
            totalFrames += count
        }
        guard reader.status == .completed, totalFrames > 0 else { throw Failure.decodeFailed }
        reader.cancelReading()
        let fps = Double(totalFrames) / range.duration.seconds
        let indices = try Qwen4ExpMediaGeometry.sampleIndices(totalFrames: totalFrames, sourceFPS: fps,
            targetFPS: config.fps, minFrames: config.minFrames, maxFrames: config.maxFrames)
        // Decode in presentation order and select the exact frame ordinals,
        // matching the reference's frame-index contract. Only selected,
        // resized frames survive each iteration; no full clip is retained.
        let decodedReader = try AVAssetReader(asset: asset)
        let decoded = AVAssetReaderTrackOutput(track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        decoded.alwaysCopiesSampleData = false
        guard decodedReader.canAdd(decoded) else { throw Failure.decodeFailed }
        decodedReader.add(decoded)
        guard decodedReader.startReading() else { throw Failure.decodeFailed }
        defer { decodedReader.cancelReading() }
        let transform = try await track.load(.preferredTransform)
        var frames: [MLXArray] = []
        frames.reserveCapacity(indices.count)
        var ordinal = 0, selected = 0
        while let buffer = decoded.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard ordinal < totalFrames, let pixel = CMSampleBufferGetImageBuffer(buffer) else {
                throw Failure.invalidMetadata
            }
            if selected < indices.count, ordinal == indices[selected] {
                // The reference preprocesses decoded RGB byte values, not a
                // second color-managed transfer-function conversion. Match
                // the existing native image path's explicit sRGB byte space.
                let ci = CIImage(cvPixelBuffer: pixel,
                    options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!]).transformed(by: transform)
                frames.append(MediaProcessing.asMLXArray(try process(ci, indices.count)))
                selected += 1
            }
            ordinal += 1
        }
        try Task.checkCancellation()
        guard decodedReader.status == .completed, ordinal == totalFrames,
              frames.count == indices.count else { throw Failure.missingFrame }
        return .init(frames: frames, indices: indices, sourceFPS: fps)
    }
}
