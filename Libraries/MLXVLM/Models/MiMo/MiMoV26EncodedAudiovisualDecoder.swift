// Copyright © 2026 Eigen Labs.
// Narrow encoded transport for SGLang67bb6a58's whole-audio/interleave0 input.
// No audio converter, resampler, channel mix, peak normalization, MLX or file IO.
@preconcurrency import AVFoundation
import AudioToolbox
import CoreMedia
import CryptoKit
import Foundation
import MLXLMCommon

public enum MiMoV26EncodedAudiovisualDecoder {
    public enum Failure: Error, Equatable, Sendable {
        case missingAudio, unsupportedEncoding, unsupportedChannels, unsupportedSampleRate
        case unsupportedTimeline, invalidAudio, nonfiniteSamples, limit, arithmeticOverflow
    }
    public struct Limits: Sendable {
        public let maximumFrames, maximumWorkingBytes, maximumBuffers: Int
        public init(maximumFrames: Int, maximumWorkingBytes: Int, maximumBuffers: Int = 4096) {
            self.maximumFrames = maximumFrames; self.maximumWorkingBytes = maximumWorkingBytes
            self.maximumBuffers = maximumBuffers
        }
    }
    fileprivate enum Encoding: Equatable, Sendable {
        case pcm16, float32
        var bytesPerFrame: Int { self == .pcm16 ? 2 : 4 }
    }
    public struct Plan: Sendable {
        public let video: MiMoV26EncodedVisualDecoder.VideoPlan
        public let frameCount, audioWorkingByteBound: Int
        public let segmentEnd: MiMoV26DecodedAudiovisual.SegmentEnd
        fileprivate let encoding: Encoding
        fileprivate let trackID: CMPersistentTrackID
        fileprivate let maximumBuffers: Int
        fileprivate init(video: MiMoV26EncodedVisualDecoder.VideoPlan, frames: Int,
                         bound: Int, encoding: Encoding, trackID: CMPersistentTrackID,
                         maximumBuffers: Int, end: Float) {
            self.video = video; frameCount = frames; audioWorkingByteBound = bound
            self.encoding = encoding; self.trackID = trackID
            self.maximumBuffers = maximumBuffers; segmentEnd = .float32(end)
        }
    }
    private static func product(_ a: Int, _ b: Int) throws -> Int {
        let value = a.multipliedReportingOverflow(by: b)
        guard a >= 0, b >= 0, !value.overflow else { throw Failure.arithmeticOverflow }
        return value.partialValue
    }
    private static func sum(_ a: Int, _ b: Int) throws -> Int {
        let value = a.addingReportingOverflow(b)
        guard a >= 0, b >= 0, !value.overflow else { throw Failure.arithmeticOverflow }
        return value.partialValue
    }
    private static func encoding(_ description: CMFormatDescription) throws -> Encoding {
        guard CMFormatDescriptionGetMediaType(description) == kCMMediaType_Audio,
              let pointer = CMAudioFormatDescriptionGetStreamBasicDescription(description) else {
            throw Failure.invalidAudio
        }
        let value = pointer.pointee
        guard value.mChannelsPerFrame == 1 else { throw Failure.unsupportedChannels }
        guard value.mSampleRate == 24000 else { throw Failure.unsupportedSampleRate }
        guard value.mFormatID == kAudioFormatLinearPCM, value.mFramesPerPacket == 1 else {
            throw Failure.unsupportedEncoding
        }
        let selected: Encoding
        if value.mBitsPerChannel == 16,
           value.mFormatFlags == (kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked) {
            selected = .pcm16
        } else if value.mBitsPerChannel == 32,
                  value.mFormatFlags == (kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked) {
            selected = .float32
        } else { throw Failure.unsupportedEncoding }
        guard value.mBytesPerFrame == UInt32(selected.bytesPerFrame),
              value.mBytesPerPacket == UInt32(selected.bytesPerFrame) else { throw Failure.unsupportedEncoding }
        return selected
    }

    /// Mirrors the two Float32 reciprocal operations in process_video, not the
    /// simplified timestamp gap or asset duration (which can change a token end).
    static func individualSegmentEnd(_ timestamps: [Float]) throws -> Float {
        guard timestamps.count >= 2 else { throw Failure.unsupportedTimeline }
        let gap = timestamps[1] - timestamps[0]
        let sampledFPS: Float = 1 / gap
        let step: Float = 1 / sampledFPS
        let end = timestamps[timestamps.count - 1] + step
        guard gap.isFinite, gap > 0, sampledFPS.isFinite, sampledFPS > 0,
              end.isFinite, end > timestamps[timestamps.count - 1] else {
            throw Failure.unsupportedTimeline
        }
        return end
    }

    /// Metadata only; no reader/PCM Float allocation. The provider must price
    /// audioWorkingByteBound + video.decodeWorkingByteBound on the SAME actual
    /// host reservation before calling decode. A Plan is not native admission.
    public static func inspect(_ video: MiMoV26EncodedVisualDecoder.VideoPlan,
                               limits: Limits) async throws -> Plan {
        try Task.checkCancellation()
        guard video.hasAudioTrack else { throw Failure.missingAudio }
        guard limits.maximumFrames > 0, limits.maximumFrames <= Int(Int32.max),
              limits.maximumWorkingBytes > 0, limits.maximumBuffers > 0 else { throw Failure.limit }
        let end = try individualSegmentEnd(video.timestamps)
        return try await video.sourceOwner.withAsset { asset in
            let tracks = try await asset.loadTracks(withMediaType: .audio)
            guard tracks.count == 1, let track = tracks.first else { throw Failure.unsupportedEncoding }
            let descriptions = try await track.load(.formatDescriptions)
            guard descriptions.count == 1, let description = descriptions.first else {
                throw Failure.unsupportedEncoding
            }
            let selected = try encoding(description)
            let range = try await track.load(.timeRange)
            let segments = try await track.load(.segments)
            guard range.start.isNumeric, range.start == .zero, range.duration.isNumeric,
                  range.duration > .zero, segments.count == 1, let segment = segments.first,
                  !segment.isEmpty else { throw Failure.unsupportedTimeline }
            let mapping = segment.timeMapping
            guard mapping.source.start == .zero, mapping.target.start == .zero,
                  mapping.source.duration == range.duration,
                  mapping.target.duration == range.duration else { throw Failure.unsupportedTimeline }
            let scaled = CMTimeConvertScale(range.duration, timescale: 24000, method: .default)
            guard scaled.isNumeric, scaled == range.duration, scaled.value > 0,
                  scaled.value <= Int64(limits.maximumFrames),
                  let frames = Int(exactly: scaled.value) else { throw Failure.limit }
            // Raw PCM payload cannot exceed the retained immutable container.
            guard try product(frames, selected.bytesPerFrame) <= video.sourceOwner.byteCount else {
                throw Failure.invalidAudio
            }
            // Retained Float output + possible raw sample copy/reader storage,
            // bounded metadata and fixed decoder slack. Conservative source
            // accounting, not a measured peak or replacement native reserve.
            let bound = try sum(product(frames, 16),
                sum(product(limits.maximumBuffers, 128), 1 << 20))
            guard bound <= limits.maximumWorkingBytes else { throw Failure.limit }
            return Plan(video: video, frames: frames, bound: bound, encoding: selected,
                trackID: track.trackID, maximumBuffers: limits.maximumBuffers, end: end)
        }
    }

    public static func decode(_ plan: Plan, videoLimits: MiMoV26EncodedVisualDecoder.Limits,
                              audioLimits: Limits) async throws -> MiMoV26DecodedAudiovisual {
        guard plan.frameCount <= audioLimits.maximumFrames,
              plan.audioWorkingByteBound <= audioLimits.maximumWorkingBytes,
              plan.maximumBuffers <= audioLimits.maximumBuffers else { throw Failure.limit }
        guard plan.video.sourceOwner.byteCount <= videoLimits.maximumEncodedBytes,
              videoLimits.maximumControlMarkers >= 0,
              plan.video.codedPixels <= videoLimits.maximumPixels,
              plan.video.sampledIndices.count <= videoLimits.maximumSampledFrames,
              plan.video.sourceFrameCount <= videoLimits.maximumSourceFrames,
              try plan.video.decodeWorkingByteBound() <= videoLimits.maximumWorkingBytes else {
            throw Failure.limit
        }
        try Task.checkCancellation()
        let audio = try await decodeAudio(plan)
        // Only this paired path can access frame decode for an audio-bearing
        // clip. The public silentVideo entrypoint still rejects sound.
        let video = try await MiMoV26EncodedVisualDecoder.audiovisualFrames(
            plan.video, limits: videoLimits)
        try Task.checkCancellation()
        return .init(frames: video.frames, timestamps: video.timestamps,
            wholeAudio: audio, segmentEnd: plan.segmentEnd)
    }

    /// Preserve this decoder's typed timeline/resource refusals for malformed
    /// control buffers while sharing the actual no-media marker predicate.
    static func consumeEmptyMarker(_ sample: CMSampleBuffer,
                                   count: inout Int, limit: Int) throws -> Bool {
        do {
            return try MiMoV26EncodedVisualDecoder.consumeEmptyMarker(
                sample, count: &count, limit: limit)
        } catch MiMoV26EncodedVisualDecoder.Failure.limit {
            throw Failure.limit
        } catch {
            throw Failure.unsupportedTimeline
        }
    }

    private static func decodeAudio(_ plan: Plan) async throws -> MiMoV26DecodedPCM {
        try await plan.video.sourceOwner.withAsset { asset in
            let tracks = try await asset.loadTracks(withMediaType: .audio)
            guard tracks.count == 1, let track = tracks.first, track.trackID == plan.trackID else {
                throw Failure.invalidAudio
            }
            let reader = try AVAssetReader(asset: asset)
            // Passthrough: no platform decoder/converter numerical contract.
            let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
            guard reader.canAdd(output) else { throw Failure.invalidAudio }
            reader.add(output)
            guard reader.startReading() else { throw Failure.invalidAudio }
            defer { reader.cancelReading() }
            var values: [Float] = []; values.reserveCapacity(plan.frameCount)
            var digest = SHA256()
            var buffers = 0, markerCount = 0
            while let sample = output.copyNextSampleBuffer() {
                try Task.checkCancellation()
                buffers += 1
                guard buffers <= plan.maximumBuffers else { throw Failure.limit }
                if try consumeEmptyMarker(sample, count: &markerCount, limit: plan.maximumBuffers) {
                    continue
                }
                guard CMSampleBufferDataIsReady(sample),
                      let description = CMSampleBufferGetFormatDescription(sample),
                      try encoding(description) == plan.encoding,
                      CMGetAttachment(sample, key: kCMSampleBufferAttachmentKey_TrimDurationAtStart,
                                      attachmentModeOut: nil) == nil,
                      CMGetAttachment(sample, key: kCMSampleBufferAttachmentKey_TrimDurationAtEnd,
                                      attachmentModeOut: nil) == nil,
                      CMGetAttachment(sample, key: kCMSampleBufferAttachmentKey_SpeedMultiplier,
                                      attachmentModeOut: nil) == nil,
                      CMGetAttachment(sample, key: kCMSampleBufferAttachmentKey_Reverse,
                                      attachmentModeOut: nil) == nil,
                      CMGetAttachment(sample, key: kCMSampleBufferAttachmentKey_EmptyMedia,
                                      attachmentModeOut: nil) == nil else { throw Failure.unsupportedTimeline }
                let count = CMSampleBufferGetNumSamples(sample)
                let next = try sum(values.count, count)
                guard count > 0, next <= plan.frameCount else { throw Failure.limit }
                // Reject gaps, overlap, edits, retiming or codec priming. No
                // synthesized silence and no dropped/shifted audio samples.
                let pts = CMSampleBufferGetPresentationTimeStamp(sample)
                let duration = CMSampleBufferGetDuration(sample)
                var timing = CMSampleTimingInfo()
                var timingCount: CMItemCount = 0
                // A single descriptor applies to EVERY sample. Refuse a
                // variable-timing buffer instead of trusting its total span.
                guard CMSampleBufferGetSampleTimingInfoArray(sample, entryCount: 1,
                    arrayToFill: &timing, entriesNeededOut: &timingCount) == noErr,
                      timingCount == 1, timing.duration == CMTime(value: 1, timescale: 24000),
                      timing.presentationTimeStamp == pts else { throw Failure.unsupportedTimeline }
                guard pts.isNumeric, duration.isNumeric,
                      pts == CMTime(value: Int64(values.count), timescale: 24000),
                      duration == CMTime(value: Int64(count), timescale: 24000) else {
                    throw Failure.unsupportedTimeline
                }
                let size = try product(count, plan.encoding.bytesPerFrame)
                guard let block = CMSampleBufferGetDataBuffer(sample),
                      CMBlockBufferGetDataLength(block) == size else { throw Failure.invalidAudio }
                var bytes = Data(count: size)
                let status = bytes.withUnsafeMutableBytes {
                    CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: size,
                                              destination: $0.baseAddress!)
                }
                guard status == kCMBlockBufferNoErr else { throw Failure.invalidAudio }
                digest.update(data: bytes)
                try bytes.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                    for index in 0..<count {
                        if index.isMultiple(of: 1024) { try Task.checkCancellation() }
                        let at = index * plan.encoding.bytesPerFrame
                        let low = UInt16(raw[at]) | UInt16(raw[at + 1]) << 8
                        let value: Float
                        switch plan.encoding {
                        case .pcm16: value = Float(Int16(bitPattern: low)) / 32768
                        case .float32:
                            value = Float(bitPattern: UInt32(low)
                                | UInt32(raw[at + 2]) << 16 | UInt32(raw[at + 3]) << 24)
                        }
                        guard value.isFinite else { throw Failure.nonfiniteSamples }
                        values.append(value)
                    }
                }
            }
            guard reader.status == .completed, values.count == plan.frameCount else {
                throw Failure.invalidAudio
            }
            // Exact raw-track provenance, not codec weights/profile authority.
            let identity = "mimo-av-\(plan.encoding)-24k:" +
                digest.finalize().map { String(format: "%02x", $0) }.joined()
            return try .init(samples: values, descriptor: .init(sourceIdentity: identity,
                channels: 1, frameCount: plan.frameCount, sampleRate: 24000))
        }
    }
}
