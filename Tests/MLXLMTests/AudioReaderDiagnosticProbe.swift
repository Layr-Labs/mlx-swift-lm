// Copyright © 2026 Eigen Labs.

@preconcurrency import AVFoundation
import CryptoKit
import Foundation
import XCTest

@testable import MLXLMCommon

/// Temporary, bounded raw-PCM experiments. These do not replace the production-path assertions.
enum AudioReaderDiagnosticProbe {
    enum Source {
        case file(URL)
        case memory(Data)

        func withAsset<T>(_ body: (AVAsset) async throws -> T) async throws -> T {
            switch self {
            case .file(let url):
                return try await body(AVURLAsset(url: url))
            case .memory(let data):
                let owner = try MemoryBackedVideoAsset(videoData: data)
                return try await owner.withAsset { try await body($0) }
            }
        }
    }

    private struct PCM {
        var bytes = Data()
        var frames = 0
        var channels: UInt32 = 0
        var rate = 0.0
        var timing: [AudioReaderDiagnosticTimeline.Segment] = []
        var hash: String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
    }

    private final class Readers {
        var held: [AVAssetReader] = []
    }

    static func run(source: Source, label: String, settings: [String: Any]) async throws {
        emit([
            "fixture": label, "event": "environment",
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "framework": Bundle(for: AVAssetReader.self).object(
                forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown",
        ])
        for fresh in [false, true] {
            try await source.withAsset { shared in
                for retention in ["fault-free", "retain", "release"] {
                    for repetition in 0 ..< 3 {
                        let context = "\(label)/fresh=\(fresh)/\(retention)/\(repetition)"
                        func read(cancel: Bool, readers: Readers?) async throws -> PCM {
                            func pass(_ asset: AVAsset) async throws -> PCM {
                                let tracks = try await asset.loadTracks(withMediaType: .audio)
                                let track = try XCTUnwrap(tracks.first)
                                return try autoreleasepool {
                                    try collect(
                                        asset, track: track, settings: settings, cancel: cancel,
                                        readers: readers, context: context)
                                }
                            }
                            if fresh { return try await source.withAsset { try await pass($0) } }
                            return try await pass(shared)
                        }

                        let baseline = try await read(cancel: false, readers: nil)
                        let readers = Readers()
                        let failures = FailedPasses()
                        var attempts = 0
                        let candidate = try await VideoReaderRetry.$failedPassObserver.withValue(
                            failures.observer
                        ) {
                            try await VideoReaderRetry.run {
                                attempts += 1
                                return try await read(
                                    cancel: retention != "fault-free" && attempts == 1,
                                    readers: retention == "retain" ? readers : nil)
                            }
                        }
                        emit([
                            "event": "comparison", "case": context,
                            "baselineHash": baseline.hash, "candidateHash": candidate.hash,
                            "frames": baseline.frames, "candidateFrames": candidate.frames,
                            "bytes": baseline.bytes.count, "candidateBytes": candidate.bytes.count,
                            "channels": baseline.channels, "rate": baseline.rate,
                            "differentBytes": zip(baseline.bytes, candidate.bytes).filter {
                                $0 != $1
                            }.count,
                            "attempts": attempts, "heldReaders": readers.held.count,
                        ])
                        XCTAssertEqual(attempts, retention == "fault-free" ? 1 : 2, context)
                        XCTAssertEqual(
                            failures.retrying, retention == "fault-free" ? [] : [true], context)
                        XCTAssertEqual(
                            failures.all.map { $0.failure.medium },
                            retention == "fault-free" ? [] : [.audio], context)
                        XCTAssertEqual(
                            failures.all.map { $0.failure.status },
                            retention == "fault-free"
                                ? [] : [AVAssetReader.Status.cancelled.rawValue],
                            context)
                        XCTAssertFalse(baseline.bytes.isEmpty, context)
                        XCTAssertEqual(candidate.frames, baseline.frames, context)
                        XCTAssertEqual(candidate.channels, baseline.channels, context)
                        XCTAssertEqual(candidate.rate, baseline.rate, context)
                        let baselineTimeline = AudioReaderDiagnosticTimeline(baseline.timing)
                        let candidateTimeline = AudioReaderDiagnosticTimeline(candidate.timing)
                        XCTAssertNotNil(baselineTimeline, "Invalid baseline timeline: \(context)")
                        XCTAssertNotNil(candidateTimeline, "Invalid candidate timeline: \(context)")
                        XCTAssertEqual(candidateTimeline, baselineTimeline, context)
                        XCTAssertTrue(
                            candidate.bytes == baseline.bytes, "Raw PCM differs: \(context)")
                        XCTAssertEqual(candidate.hash, baseline.hash, context)
                        withExtendedLifetime(readers) {}
                    }
                }
            }
        }
    }

    private static func collect(
        _ asset: AVAsset, track: AVAssetTrack, settings: [String: Any], cancel: Bool,
        readers: Readers?, context: String
    ) throws -> PCM {
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        XCTAssertTrue(reader.canAdd(output))
        reader.add(output)
        XCTAssertTrue(reader.startReading())
        readers?.held.append(reader)
        defer { reader.cancelReading() }
        var result = PCM()
        var reads = 0
        while true {
            reads += 1
            if cancel && reads == 2 { reader.cancelReading() }
            guard let sample = output.copyNextSampleBuffer() else { break }
            guard let block = CMSampleBufferGetDataBuffer(sample) else { continue }
            let description = try XCTUnwrap(CMSampleBufferGetFormatDescription(sample))
            let format = try XCTUnwrap(
                CMAudioFormatDescriptionGetStreamBasicDescription(description)
            )
            .pointee
            XCTAssertEqual(format.mFormatID, kAudioFormatLinearPCM)
            XCTAssertEqual(format.mBitsPerChannel, 32)
            result.channels = format.mChannelsPerFrame
            result.rate = format.mSampleRate
            result.frames += CMSampleBufferGetNumSamples(sample)
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)
            let duration = CMSampleBufferGetDuration(sample)
            result.timing.append(.init(start: pts, duration: duration))
            let count = CMBlockBufferGetDataLength(block)
            var bytes = Data(count: count)
            let status = bytes.withUnsafeMutableBytes {
                CMBlockBufferCopyDataBytes(
                    block, atOffset: 0, dataLength: count, destination: $0.baseAddress!)
            }
            XCTAssertEqual(status, kCMBlockBufferNoErr)
            result.bytes.append(bytes)
        }
        emit([
            "event": "pass", "case": context, "cancel": cancel,
            "status": reader.status.rawValue, "frames": result.frames,
            "buffers": result.timing.count, "hash": result.hash,
        ])
        guard reader.status == .completed else {
            throw VideoReaderFailure(reader: reader, medium: .audio)
        }
        return result
    }

    private static func emit(_ fields: [String: Any]) {
        var fields = fields
        fields["context"] =
            ProcessInfo.processInfo.environment["DARKBLOOM_AUDIO_DIAGNOSTIC_CONTEXT"]
            ?? "shard"
        if let data = try? JSONSerialization.data(withJSONObject: fields, options: [.sortedKeys]),
            let line = String(data: data, encoding: .utf8)
        {
            print("[DEBUG-audio-reader] \(line)")
        }
    }
}
