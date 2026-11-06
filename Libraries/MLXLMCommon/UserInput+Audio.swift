// Copyright © 2026 Apple Inc.

import Foundation
import MLX

#if canImport(AVFoundation)
@preconcurrency import AVFoundation
#endif

#if canImport(AVFoundation)
extension UserInput.AudioFormat {
    public var asAudioFormatID: AudioFormatID {
        switch self {
        case .linearPCM:
            return kAudioFormatLinearPCM
        }
    }
}
#endif

extension UserInput.Audio {

    /// The samples of the audio. A `.url` reader pass that fails runs once more with a
    /// new reader; a second failure throws `VideoReaderFailure` with medium `.audio`.
    public func asMLXArray(processing: UserInput.AudioProcessing = .init()) async throws -> MLXArray
    {
        switch source {
        case .url(let url):
            #if canImport(AVFoundation)
            let asset = AVURLAsset(url: url)

            guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
                throw UserInputError.noAudioData(url)
            }

            let settings: [String: Any] = [
                AVFormatIDKey: processing.audioFormat.asAudioFormatID,
                AVSampleRateKey: processing.sampleRate,
                AVNumberOfChannelsKey: processing.channels,
                AVLinearPCMBitDepthKey: 32,
                AVLinearPCMIsFloatKey: true,
                AVLinearPCMIsNonInterleaved: false,
            ]

            let samples = try await VideoReaderRetry.run {
                try Self.samples(asset, track: track, settings: settings)
            }
            return MLXArray(samples)
            #else
            fatalError("Audio processing is not supported on this platform.")
            #endif
        case .array(let array):
            return array
        }
    }

    #if canImport(AVFoundation)
    /// One reader pass over `track`. A reader that does not complete throws
    /// `VideoReaderFailure`.
    private static func samples(
        _ asset: AVAsset, track: AVAssetTrack, settings: [String: Any]
    ) throws -> [Float] {
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        let reader = try AVAssetReader(asset: asset)
        reader.add(output)
        reader.startReading()

        var samples: [Float] = []

        while let sampleBuffer = VideoReaderRetry.nextSample(output, of: reader) {
            guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { continue }

            let byteCount = CMBlockBufferGetDataLength(blockBuffer)
            var chunk = [Float](repeating: 0, count: byteCount / MemoryLayout<Float>.size)
            CMBlockBufferCopyDataBytes(
                blockBuffer, atOffset: 0, dataLength: byteCount, destination: &chunk)
            samples.append(contentsOf: chunk)
        }

        guard reader.status == .completed else {
            throw VideoReaderFailure(reader: reader, medium: .audio)
        }
        return samples
    }
    #endif
}
