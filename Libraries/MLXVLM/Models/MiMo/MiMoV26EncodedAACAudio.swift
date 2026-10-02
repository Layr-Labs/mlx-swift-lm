// Copyright © 2026 Eigen Labs.
// Bounded AAC transport decode; native MiMo still owns resampling and mixing.
@preconcurrency import AVFoundation
import AudioToolbox
import CoreMedia
import CryptoKit
import Foundation
import MLXLLM
import MLXLMCommon

enum MiMoV26EncodedAACAudio {
    typealias Failure = MiMoV26EncodedAudiovisualDecoder.Failure
    struct Plan: Sendable {
        let trackID: CMPersistentTrackID
        let channels, sampleRate, minimumFrames, maximumFrames, sampleCount, workingBytes: Int
    }
    private static func multiply(_ a: Int, _ b: Int) throws -> Int {
        let (value, overflow) = a.multipliedReportingOverflow(by: b)
        guard a >= 0, b >= 0, !overflow else { throw Failure.arithmeticOverflow }
        return value
    }
    static func inspect(
        track: AVAssetTrack, description: CMFormatDescription,
        limits: MiMoV26EncodedAudiovisualDecoder.Limits
    ) async throws -> Plan {
        guard let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(description)?.pointee,
            asbd.mFormatID == kAudioFormatMPEG4AAC, asbd.mFramesPerPacket == 1024
        else { throw Failure.unsupportedEncoding }
        let channels = Int(asbd.mChannelsPerFrame)
        guard channels > 0, channels <= min(2, limits.maximumChannels) else {
            throw Failure.unsupportedChannels
        }
        guard asbd.mSampleRate.isFinite, asbd.mSampleRate.rounded() == asbd.mSampleRate,
            asbd.mSampleRate >= Double(MiMoV26EncodedAudioDecoder.minimumSampleRate),
            asbd.mSampleRate
                <= Double(
                    min(MiMoV26EncodedAudioDecoder.maximumSampleRate, limits.maximumSampleRate))
        else { throw Failure.unsupportedSampleRate }
        let rate = Int(asbd.mSampleRate)
        let range = try await track.load(.timeRange)
        let segments = try await track.load(.segments)
        guard range.start.isNumeric, range.start == .zero, range.duration.isNumeric,
            range.duration > .zero, segments.count == 1, let segment = segments.first,
            !segment.isEmpty
        else { throw Failure.unsupportedTimeline }
        let mapping = segment.timeMapping
        // AVAssetReader applies the container's single source trim (including
        // AAC encoder priming). Refuse gaps, multiple edits and speed changes.
        guard mapping.source.start.isNumeric, mapping.source.start >= .zero,
            mapping.target.start == .zero, mapping.source.duration == range.duration,
            mapping.target.duration == range.duration
        else { throw Failure.unsupportedTimeline }
        let upper = CMTimeConvertScale(
            range.duration, timescale: Int32(rate), method: .roundAwayFromZero)
        let lower = CMTimeConvertScale(
            range.duration, timescale: Int32(rate), method: .roundTowardZero)
        guard upper.isNumeric, lower.isNumeric, lower.value > 0,
            upper.value <= Int64(limits.maximumFrames),
            let frames = Int(exactly: upper.value), let minimum = Int(exactly: lower.value)
        else { throw Failure.limit }
        let samples = try multiply(frames, channels)
        let storage = try multiply(samples, 16)
        let metadata = try multiply(limits.maximumBuffers, 128)
        let (partial, overflow1) = storage.addingReportingOverflow(metadata)
        // Fixed host decoder allowance, in addition to retained PCM, sample
        // copies and metadata. This never replaces native activation reserves.
        let (bound, overflow2) = partial.addingReportingOverflow(32 << 20)
        guard !overflow1, !overflow2, bound <= limits.maximumWorkingBytes else {
            throw Failure.limit
        }
        return Plan(
            trackID: track.trackID, channels: channels, sampleRate: rate,
            minimumFrames: minimum, maximumFrames: frames, sampleCount: samples, workingBytes: bound
        )
    }

    static func decode(
        _ plan: Plan, owner: MemoryBackedVideoAsset, maximumBuffers: Int
    ) async throws -> MiMoV26DecodedPCM {
        try await owner.withAsset { asset in
            let tracks = try await asset.loadTracks(withMediaType: .audio)
            guard tracks.count == 1, let track = tracks.first, track.trackID == plan.trackID else {
                throw Failure.invalidAudio
            }
            let reader = try AVAssetReader(asset: asset)
            let output = AVAssetReaderTrackOutput(
                track: track,
                outputSettings: [
                    AVFormatIDKey: kAudioFormatLinearPCM, AVLinearPCMIsFloatKey: true,
                    AVLinearPCMBitDepthKey: 32, AVLinearPCMIsNonInterleaved: false,
                    AVLinearPCMIsBigEndianKey: false,
                ])
            guard reader.canAdd(output) else { throw Failure.invalidAudio }
            reader.add(output)
            guard reader.startReading() else { throw Failure.invalidAudio }
            defer { reader.cancelReading() }
            var values = [Float](repeating: 0, count: plan.sampleCount)
            var frames = 0
            var buffers = 0
            var markers = 0
            var digest = SHA256()
            while let sample = output.copyNextSampleBuffer() {
                try Task.checkCancellation()
                buffers += 1
                guard buffers <= maximumBuffers else { throw Failure.limit }
                if try MiMoV26EncodedAudiovisualDecoder.consumeEmptyMarker(
                    sample, count: &markers, limit: maximumBuffers)
                {
                    continue
                }
                guard CMSampleBufferDataIsReady(sample),
                    let description = CMSampleBufferGetFormatDescription(sample),
                    let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(description)?
                        .pointee,
                    asbd.mFormatID == kAudioFormatLinearPCM,
                    asbd.mSampleRate == Double(plan.sampleRate),
                    asbd.mChannelsPerFrame == UInt32(plan.channels),
                    asbd.mBitsPerChannel == 32, asbd.mBytesPerFrame == UInt32(plan.channels * 4),
                    asbd.mFormatFlags == (kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked)
                else { throw Failure.invalidAudio }
                let count = CMSampleBufferGetNumSamples(sample)
                guard count > 0, count <= plan.maximumFrames - frames else { throw Failure.limit }
                let pts = CMSampleBufferGetPresentationTimeStamp(sample)
                let duration = CMSampleBufferGetDuration(sample)
                var timing = CMSampleTimingInfo()
                var timingCount: CMItemCount = 0
                guard
                    CMSampleBufferGetSampleTimingInfoArray(
                        sample, entryCount: 1,
                        arrayToFill: &timing, entriesNeededOut: &timingCount) == noErr,
                    timingCount == 1,
                    timing.duration == CMTime(value: 1, timescale: Int32(plan.sampleRate)),
                    timing.presentationTimeStamp == pts,
                    pts == CMTime(value: Int64(frames), timescale: Int32(plan.sampleRate)),
                    duration == CMTime(value: Int64(count), timescale: Int32(plan.sampleRate))
                else { throw Failure.unsupportedTimeline }
                for key in [
                    kCMSampleBufferAttachmentKey_TrimDurationAtStart,
                    kCMSampleBufferAttachmentKey_TrimDurationAtEnd,
                    kCMSampleBufferAttachmentKey_SpeedMultiplier,
                    kCMSampleBufferAttachmentKey_Reverse, kCMSampleBufferAttachmentKey_EmptyMedia,
                ] {
                    guard CMGetAttachment(sample, key: key, attachmentModeOut: nil) == nil else {
                        throw Failure.unsupportedTimeline
                    }
                }
                let bytesCount = try multiply(count, plan.channels * 4)
                guard let block = CMSampleBufferGetDataBuffer(sample),
                    CMBlockBufferGetDataLength(block) == bytesCount
                else {
                    throw Failure.invalidAudio
                }
                var bytes = Data(count: bytesCount)
                let status = bytes.withUnsafeMutableBytes {
                    CMBlockBufferCopyDataBytes(
                        block, atOffset: 0, dataLength: bytesCount, destination: $0.baseAddress!)
                }
                guard status == kCMBlockBufferNoErr else { throw Failure.invalidAudio }
                digest.update(data: bytes)
                try bytes.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                    for frame in 0 ..< count {
                        if frame.isMultiple(of: 1024) { try Task.checkCancellation() }
                        for channel in 0 ..< plan.channels {
                            let at = (frame * plan.channels + channel) * 4
                            let bits =
                                UInt32(raw[at]) | UInt32(raw[at + 1]) << 8 | UInt32(raw[at + 2])
                                << 16 | UInt32(raw[at + 3]) << 24
                            let value = Float(bitPattern: bits)
                            guard value.isFinite else { throw Failure.nonfiniteSamples }
                            values[channel * plan.maximumFrames + frames + frame] = value
                        }
                    }
                }
                frames += count
            }
            guard reader.status == .completed, frames >= plan.minimumFrames,
                frames <= plan.maximumFrames
            else {
                throw Failure.invalidAudio
            }
            if frames != plan.maximumFrames {
                values = (0 ..< plan.channels).flatMap { channel in
                    values[
                        (channel * plan.maximumFrames) ..< (channel * plan.maximumFrames + frames)]
                }
            }
            let identity =
                "mimo-av-aac-\(plan.sampleRate)-\(plan.channels):"
                + digest.finalize().map { String(format: "%02x", $0) }.joined()
            return try .init(
                samples: values,
                descriptor: .init(
                    sourceIdentity: identity,
                    channels: plan.channels, frameCount: frames, sampleRate: plan.sampleRate))
        }
    }
}
