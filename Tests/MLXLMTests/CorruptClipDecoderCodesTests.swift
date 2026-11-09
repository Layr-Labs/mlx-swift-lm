// Copyright © 2026 Eigen Labs.

@preconcurrency import AVFoundation
import CoreMedia
import Foundation
import VideoToolbox
import XCTest

@testable import MLXLMCommon

/// Corrupt H.264 and HEVC clips through the real decoders, read as the video
/// readers read them: an in-memory asset, one `AVAssetReader` pass that decodes
/// to 32BGRA, and a `VideoReaderFailure` over the reader's error chain.
///
/// `VideoReaderFailure.transientCodes` holds decoder faults of the machine, with
/// `kVTVideoDecoderMalfunctionErr` (-12911) among them. A provider refuses a
/// transient failure as retryable, so a corrupt clip with a transient code would
/// be refused as retryable by every provider. Each clip keeps a valid container
/// and corrupts only the NAL units of its samples. The test prints each result
/// with the reader's code chain.
final class CorruptClipDecoderCodesTests: XCTestCase {
    /// The first samples in decode order. They decode without later samples.
    private static let sampleCount = 30

    private enum Corruption: String, CaseIterable {
        /// Inverts the second quarter of each NAL unit after its header.
        case flippedBits
        /// Cuts each NAL unit to half its length and rewrites its length field.
        case truncatedNALUnits
        /// The first length field of each sample claims more bytes than the sample has.
        case overlongLengthField
        /// Replaces each NAL unit after its header with pseudo-random bytes.
        case randomPayload
    }

    private struct Track {
        let format: CMFormatDescription
        let samples: [CMSampleBuffer]
    }

    private struct Decoded {
        let frames: Int
        let failure: VideoReaderFailure?
    }

    private struct WriterFailure: Error, CustomStringConvertible {
        let description: String
    }

    func testCorruptH264AndHEVCClipsGiveNoTransientDecoderCode() async throws {
        let fixture = try XCTUnwrap(
            Bundle.module.url(forResource: "1080p_30", withExtension: "mov"))
        let hevc = try await hevcClip(from: fixture)
        defer { try? FileManager.default.removeItem(at: hevc) }
        for (name, url, codec) in [
            ("H.264", fixture, kCMVideoCodecType_H264), ("HEVC", hevc, kCMVideoCodecType_HEVC),
        ] {
            let track = try await compressedTrack(at: url)
            XCTAssertEqual(CMFormatDescriptionGetMediaSubType(track.format), codec, name)
            let hardware = VTIsHardwareDecodeSupported(codec)
            let clean = try await decode(try await clip(track, nil))
            XCTAssertNil(clean.failure, "\(name): the clip without corruption did not decode")
            XCTAssertGreaterThan(clean.frames, 0, name)
            var failed = 0
            for corruption in Corruption.allCases {
                let result = try await decode(try await clip(track, corruption))
                let line =
                    "\(name) \(corruption.rawValue), hardware decoder \(hardware): "
                    + "\(result.frames) of \(clean.frames) frames, "
                    + (result.failure?.description ?? "reader completed")
                print("corrupt clip: \(line)")
                if result.failure != nil { failed += 1 }
                XCTAssertFalse(
                    result.failure?.isTransient ?? false,
                    "a corrupt clip gave a transient decoder code: \(line)")
            }
            XCTAssertGreaterThan(failed, 0, "\(name): no corrupt clip made the reader fail")
        }
    }

    // MARK: - Clips

    /// The fixture's first frames, encoded again as HEVC with a key frame every 10 frames.
    private func hevcClip(from url: URL) async throws -> URL {
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        let track = try XCTUnwrap(tracks.first)
        let size = try await track.load(.naturalSize)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String:
                    kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
            ])
        reader.add(output)
        XCTAssertTrue(reader.startReading(), "\(String(describing: reader.error))")
        defer { reader.cancelReading() }

        let target = temporaryURL()
        let writer = try AVAssetWriter(outputURL: target, fileType: .mov)
        let input = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.hevc,
                AVVideoWidthKey: Int(size.width), AVVideoHeightKey: Int(size.height),
                AVVideoCompressionPropertiesKey: [AVVideoMaxKeyFrameIntervalKey: 10],
            ])
        input.expectsMediaDataInRealTime = false
        writer.add(input)
        guard writer.startWriting() else {
            throw WriterFailure(description: "HEVC writer: \(String(describing: writer.error))")
        }
        var started = false
        for _ in 0 ..< Self.sampleCount {
            guard let sample = output.copyNextSampleBuffer() else { break }
            if !started {
                writer.startSession(atSourceTime: CMSampleBufferGetPresentationTimeStamp(sample))
                started = true
            }
            try await waitUntilReady(input, of: writer)
            XCTAssertTrue(input.append(sample), "\(String(describing: writer.error))")
        }
        input.markAsFinished()
        await writer.finishWriting()
        guard writer.status == .completed else {
            throw WriterFailure(description: "HEVC writer: \(String(describing: writer.error))")
        }
        return target
    }

    /// The first compressed samples of the clip's video track, with its format.
    private func compressedTrack(at url: URL) async throws -> Track {
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        let track = try XCTUnwrap(tracks.first)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        reader.add(output)
        XCTAssertTrue(reader.startReading(), "\(String(describing: reader.error))")
        defer { reader.cancelReading() }
        var samples: [CMSampleBuffer] = []
        while samples.count < Self.sampleCount, let sample = output.copyNextSampleBuffer() {
            if CMSampleBufferGetNumSamples(sample) == 1 { samples.append(sample) }
        }
        XCTAssertEqual(samples.count, Self.sampleCount)
        let format = try XCTUnwrap(samples.first.flatMap(CMSampleBufferGetFormatDescription))
        return Track(format: format, samples: samples)
    }

    /// A QuickTime clip of `track`'s samples, written without decoding, each sample
    /// corrupted by `corruption` when one is given.
    private func clip(_ track: Track, _ corruption: Corruption?) async throws -> Data {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(
            mediaType: .video, outputSettings: nil, sourceFormatHint: track.format)
        input.expectsMediaDataInRealTime = false
        writer.add(input)
        guard writer.startWriting() else {
            throw WriterFailure(description: "clip writer: \(String(describing: writer.error))")
        }
        writer.startSession(
            atSourceTime: track.samples.map(CMSampleBufferGetPresentationTimeStamp)
                .reduce(.positiveInfinity, CMTimeMinimum))
        let lengthSize = try nalLengthFieldSize(track.format)
        let headerSize =
            CMFormatDescriptionGetMediaSubType(track.format) == kCMVideoCodecType_HEVC ? 2 : 1
        for (index, sample) in track.samples.enumerated() {
            var bytes = try sampleBytes(sample)
            if let corruption {
                bytes = Self.corrupt(
                    bytes, corruption, lengthSize: lengthSize, headerSize: headerSize, seed: index)
            }
            let copy = try sampleBuffer(bytes, like: sample)
            try await waitUntilReady(input, of: writer)
            XCTAssertTrue(input.append(copy), "\(String(describing: writer.error))")
        }
        input.markAsFinished()
        await writer.finishWriting()
        guard writer.status == .completed else {
            throw WriterFailure(description: "clip writer: \(String(describing: writer.error))")
        }
        return try Data(contentsOf: url)
    }

    /// One reader pass over the clip, as the video readers decode it.
    private func decode(_ data: Data) async throws -> Decoded {
        let owner = try MemoryBackedVideoAsset(videoData: data)
        return try await owner.withAsset { (asset: AVURLAsset) async throws -> Decoded in
            let tracks = try await asset.loadTracks(withMediaType: .video)
            let track = try XCTUnwrap(tracks.first)
            let reader = try AVAssetReader(asset: asset)
            let output = AVAssetReaderTrackOutput(
                track: track,
                outputSettings: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
                ])
            output.alwaysCopiesSampleData = false
            reader.add(output)
            guard reader.startReading() else {
                return Decoded(frames: 0, failure: VideoReaderFailure(reader: reader))
            }
            defer { reader.cancelReading() }
            var frames = 0
            while let sample = output.copyNextSampleBuffer() {
                if CMSampleBufferGetImageBuffer(sample) != nil { frames += 1 }
            }
            return Decoded(
                frames: frames,
                failure: reader.status == .completed ? nil : VideoReaderFailure(reader: reader))
        }
    }

    // MARK: - NAL units

    /// The NAL units of a length-prefixed sample: the offset of each length field
    /// and the length of the unit after it.
    private static func nalUnits(_ bytes: [UInt8], lengthSize: Int) -> [(offset: Int, length: Int)]
    {
        var units: [(offset: Int, length: Int)] = []
        var offset = 0
        while offset + lengthSize <= bytes.count {
            let length = bytes[offset ..< offset + lengthSize].reduce(0) { $0 << 8 | Int($1) }
            guard length > 0, offset + lengthSize + length <= bytes.count else { break }
            units.append((offset, length))
            offset += lengthSize + length
        }
        return units
    }

    private static func lengthField(_ value: Int, size: Int) -> [UInt8] {
        (0 ..< size).map { UInt8(truncatingIfNeeded: value >> (8 * (size - 1 - $0))) }
    }

    private static func corrupt(
        _ bytes: [UInt8], _ corruption: Corruption, lengthSize: Int, headerSize: Int, seed: Int
    ) -> [UInt8] {
        let units = nalUnits(bytes, lengthSize: lengthSize)
        var out = bytes
        switch corruption {
        case .overlongLengthField:
            out.replaceSubrange(0 ..< lengthSize, with: lengthField(bytes.count, size: lengthSize))
        case .truncatedNALUnits:
            out = []
            for unit in units {
                let kept = min(unit.length, max(headerSize, unit.length / 2))
                let start = unit.offset + lengthSize
                out += lengthField(kept, size: lengthSize) + bytes[start ..< start + kept]
            }
        case .flippedBits, .randomPayload:
            var state = UInt64(seed + 1) &* 0x9E37_79B9_7F4A_7C15
            for unit in units where unit.length > headerSize {
                let start = unit.offset + lengthSize + headerSize
                let count = unit.length - headerSize
                let range =
                    corruption == .flippedBits
                    ? start + count / 4 ..< start + count / 2 : start ..< start + count
                for index in range {
                    if corruption == .flippedBits {
                        out[index] ^= 0xFF
                    } else {
                        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                        out[index] = UInt8(truncatingIfNeeded: state >> 56)
                    }
                }
            }
        }
        return out
    }

    // MARK: - Core Media

    /// The size of the length field before each NAL unit of the format's samples.
    private func nalLengthFieldSize(_ format: CMFormatDescription) throws -> Int {
        var size: Int32 = 0
        let status =
            CMFormatDescriptionGetMediaSubType(format) == kCMVideoCodecType_HEVC
            ? CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
                format, parameterSetIndex: 0, parameterSetPointerOut: nil,
                parameterSetSizeOut: nil, parameterSetCountOut: nil, nalUnitHeaderLengthOut: &size)
            : CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
                format, parameterSetIndex: 0, parameterSetPointerOut: nil,
                parameterSetSizeOut: nil, parameterSetCountOut: nil, nalUnitHeaderLengthOut: &size)
        XCTAssertEqual(status, noErr)
        XCTAssertTrue([1, 2, 4].contains(size), "length field size \(size)")
        return Int(size)
    }

    private func sampleBytes(_ sample: CMSampleBuffer) throws -> [UInt8] {
        let block = try XCTUnwrap(CMSampleBufferGetDataBuffer(sample))
        var bytes = [UInt8](repeating: 0, count: CMBlockBufferGetDataLength(block))
        XCTAssertFalse(bytes.isEmpty)
        let status = bytes.withUnsafeMutableBytes {
            CMBlockBufferCopyDataBytes(
                block, atOffset: 0, dataLength: $0.count, destination: $0.baseAddress!)
        }
        XCTAssertEqual(status, kCMBlockBufferNoErr)
        return bytes
    }

    /// A sample of `bytes` with the timing, format and sync flag of `original`.
    private func sampleBuffer(_ bytes: [UInt8], like original: CMSampleBuffer) throws
        -> CMSampleBuffer
    {
        var block: CMBlockBuffer?
        XCTAssertEqual(
            CMBlockBufferCreateWithMemoryBlock(
                allocator: kCFAllocatorDefault, memoryBlock: nil, blockLength: bytes.count,
                blockAllocator: kCFAllocatorDefault, customBlockSource: nil, offsetToData: 0,
                dataLength: bytes.count, flags: kCMBlockBufferAssureMemoryNowFlag,
                blockBufferOut: &block), kCMBlockBufferNoErr)
        let data = try XCTUnwrap(block)
        let copied = bytes.withUnsafeBytes {
            CMBlockBufferReplaceDataBytes(
                with: $0.baseAddress!, blockBuffer: data, offsetIntoDestination: 0,
                dataLength: $0.count)
        }
        XCTAssertEqual(copied, kCMBlockBufferNoErr)
        var timing = CMSampleTimingInfo()
        XCTAssertEqual(
            CMSampleBufferGetSampleTimingInfo(original, at: 0, timingInfoOut: &timing), noErr)
        var size = bytes.count
        var sample: CMSampleBuffer?
        XCTAssertEqual(
            CMSampleBufferCreateReady(
                allocator: kCFAllocatorDefault, dataBuffer: data,
                formatDescription: CMSampleBufferGetFormatDescription(original), sampleCount: 1,
                sampleTimingEntryCount: 1, sampleTimingArray: &timing, sampleSizeEntryCount: 1,
                sampleSizeArray: &size, sampleBufferOut: &sample), noErr)
        let copy = try XCTUnwrap(sample)
        let attachments =
            CMSampleBufferGetSampleAttachmentsArray(original, createIfNecessary: false)
            as? [[String: Any]]
        let notSync =
            attachments?.first?[kCMSampleAttachmentKey_NotSync as String] as? Bool ?? false
        if notSync,
            let target = CMSampleBufferGetSampleAttachmentsArray(copy, createIfNecessary: true)
        {
            let dictionary = unsafeBitCast(
                CFArrayGetValueAtIndex(target, 0), to: CFMutableDictionary.self)
            CFDictionarySetValue(
                dictionary, Unmanaged.passUnretained(kCMSampleAttachmentKey_NotSync).toOpaque(),
                Unmanaged.passUnretained(kCFBooleanTrue).toOpaque())
        }
        return copy
    }

    private func waitUntilReady(_ input: AVAssetWriterInput, of writer: AVAssetWriter) async throws
    {
        // Appending while the input is not ready raises an Objective-C exception.
        let deadline = ContinuousClock.now.advanced(by: .seconds(30))
        while !input.isReadyForMoreMediaData, writer.status == .writing,
            ContinuousClock.now < deadline
        {
            try await Task.sleep(for: .milliseconds(5))
        }
        guard input.isReadyForMoreMediaData else {
            throw WriterFailure(
                description:
                    "writer status \(writer.status.rawValue): \(String(describing: writer.error))")
        }
    }

    private func temporaryURL() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".mov")
    }
}
