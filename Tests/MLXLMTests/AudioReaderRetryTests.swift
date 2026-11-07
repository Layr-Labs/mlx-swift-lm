// Copyright © 2026 Eigen Labs.

@preconcurrency import AVFoundation
import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// `UserInput.Audio` over the AAC test clip, with its reader ended through
/// `VideoReaderRetry`'s read hook. The hook also runs before the read that
/// returns no sample, so read 2 comes on any track with one sample buffer.
final class AudioReaderRetryTests: XCTestCase {
    private func clip() throws -> UserInput.Audio {
        .url(try XCTUnwrap(Bundle.module.url(forResource: "audio_only", withExtension: "mov")))
    }

    private func samples(_ audio: UserInput.Audio) async throws -> [Float] {
        try await audio.asMLXArray().asArray(Float.self)
    }

    func testAudioFailureNamesItsMediumAndSharesTheTransientTable() {
        let unavailable = NSError(
            domain: AVFoundationErrorDomain,
            code: AVError.Code.decoderTemporarilyUnavailable.rawValue)
        let audio = VideoReaderFailure(
            status: AVAssetReader.Status.failed.rawValue, error: unavailable, medium: .audio)
        XCTAssertTrue(audio.isTransient)
        XCTAssertEqual(audio.description, "audio reader status 3: AVFoundationErrorDomain -11839")
        XCTAssertNotEqual(
            audio,
            VideoReaderFailure(status: AVAssetReader.Status.failed.rawValue, error: unavailable))
        let stopped = VideoReaderFailure(
            status: AVAssetReader.Status.cancelled.rawValue, error: nil, medium: .audio)
        XCTAssertFalse(stopped.isTransient)
        XCTAssertEqual(stopped.description, "audio reader status 4: no error")
    }

    func testUserInputAudioRetryAfterAFailedReaderPassKeepsTheSamples() async throws {
        let audio = try clip()
        let baseline = try await samples(audio)
        let ender = ReaderEnder(ending: [1], atRead: 2)
        let passes = FailedPasses()
        let retried = try await withReaderFaults(passes, ending: ender) {
            try await samples(audio)
        }
        XCTAssertEqual(ender.ended, 1)
        XCTAssertEqual(passes.retrying, [true])
        XCTAssertEqual(passes.all.map { $0.failure.medium }, [.audio])
        XCTAssertEqual(
            passes.all.map { $0.failure.status }, [AVAssetReader.Status.cancelled.rawValue])
        XCTAssertFalse(baseline.isEmpty)
        XCTAssertEqual(retried.map(\.bitPattern), baseline.map(\.bitPattern))
    }

    func testUserInputAudioThrowsTheReaderFailureWhenBothPassesFail() async throws {
        let audio = try clip()
        let ender = ReaderEnder(ending: [1, 2], atRead: 2)
        let passes = FailedPasses()
        do {
            _ = try await withReaderFaults(passes, ending: ender) { try await samples(audio) }
            XCTFail("both passes failed but the audio decoded")
        } catch let failure as VideoReaderFailure {
            XCTAssertEqual(failure.medium, .audio)
            XCTAssertEqual(failure.status, AVAssetReader.Status.cancelled.rawValue)
            XCTAssertFalse(failure.isTransient)
        }
        XCTAssertEqual(ender.ended, 2)
        XCTAssertEqual(passes.retrying, [true, false])
    }
}
