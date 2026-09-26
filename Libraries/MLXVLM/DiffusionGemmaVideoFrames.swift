@preconcurrency import AVFoundation
import CoreImage
import Foundation
import MLXLLM
import MLXLMCommon

/// Google documents videos as image sequences, at one frame/s for at most60s.
/// No audio extraction, temporary plaintext file or Gemma4 video-feature path.
enum DiffusionGemmaVideoFrames {
    static func sample<Result>(_ video: UserInput.Video,
        transform: (UserInput.VideoFrame) throws -> Result) async throws -> [Result]
    {
        try Task.checkCancellation()
        switch video {
        case .frames(let frames): return try sampleProvided(frames).map(transform)
        case .avAsset(let asset): return try await sampleAsset(asset, transform: transform)
        case .memoryBacked(let owner): return try await owner.withAsset { try await sampleAsset($0, transform: transform) }
        case .url(let url): return try await sampleAsset(AVURLAsset(url: url), transform: transform)
        }
    }

    private static func seconds(duration: Double) throws -> [Double] {
        guard duration.isFinite, duration >= 0, duration <= 60 else {
            throw DiffusionGemmaModelError.invalidInput("native video duration must not exceed60s")
        }
        return (0..<max(1, Int(ceil(duration)))).map(Double.init)
    }

    private static func sampleProvided(_ frames: [UserInput.VideoFrame]) throws -> [UserInput.VideoFrame] {
        guard let first = frames.first else { throw DiffusionGemmaModelError.invalidInput("empty video frames") }
        let origin = first.timeStamp.seconds
        let times = frames.map { $0.timeStamp.seconds - origin }
        guard origin.isFinite, origin >= 0, times.allSatisfy({ $0.isFinite && $0 >= 0 }),
            zip(times, times.dropFirst()).allSatisfy({ $0 <= $1 })
        else { throw DiffusionGemmaModelError.invalidInput("unordered/nonfinite video timestamps") }
        _ = try seconds(duration: times.last!)
        // Provided frames describe sample instants, so include an integer
        // final sample; an asset duration instead names an exclusive endpoint.
        let selected = (0..<min(60, Int(floor(times.last!)) + 1)).map(Double.init)
        var cursor = 0
        return try selected.map { target in
            try Task.checkCancellation()
            while cursor + 1 < frames.count, times[cursor + 1] <= target { cursor += 1 }
            return .init(frame: frames[cursor].frame,
                timeStamp: CMTime(seconds: times[cursor], preferredTimescale: 1_000_000))
        }
    }

    private static func sampleAsset<Result>(_ asset: AVAsset,
        transform: (UserInput.VideoFrame) throws -> Result) async throws -> [Result]
    {
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let track = tracks.first, try await track.load(.isDecodable) else {
            throw DiffusionGemmaModelError.invalidInput("undecodable video")
        }
        let duration = try await asset.load(.duration)
        let times = try seconds(duration: duration.seconds).map {
            CMTime(seconds: $0, preferredTimescale: 1_000_000)
        }
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        var frames = [(requested: Double, frame: Result)]()
        defer { generator.cancelAllCGImageGeneration() }
        for await result in generator.images(for: times) {
            try Task.checkCancellation()
            switch result {
            case .success(requestedTime: let requested, let image, actualTime: let actual):
                guard actual.seconds.isFinite, actual.seconds >= 0, actual.seconds <= duration.seconds else {
                    throw DiffusionGemmaModelError.invalidInput("invalid decoded video timestamp")
                }
                let pixels = CIImage(cgImage: image, options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
                // Transform/evaluate each frame now. Retaining60 full-resolution
                // CGImages until the final resize would amplify decode memory.
                frames.append((requested.seconds, try transform(.init(frame: pixels, timeStamp: actual))))
            case .failure:
                throw DiffusionGemmaModelError.invalidInput("video frame extraction failed")
            }
        }
        guard frames.count == times.count else { throw DiffusionGemmaModelError.invalidInput("incomplete video extraction") }
        return frames.sorted { $0.requested < $1.requested }.map(\.frame)
    }
}
