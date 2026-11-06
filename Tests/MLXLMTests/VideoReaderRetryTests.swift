// Copyright © 2026 Eigen Labs.

@preconcurrency import AVFoundation
import CoreImage
import Foundation
import MLX
import VideoToolbox
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

/// Every failed reader pass, in order. Disarms `fault` when a pass fails and a
/// second pass follows, so that pass reads every byte.
private final class FailedPasses: @unchecked Sendable {
    private let lock = NSLock()
    private let fault: VideoDataRequestFault?
    private var recorded: [(failure: VideoReaderFailure, retrying: Bool)] = []

    init(disarming fault: VideoDataRequestFault? = nil) { self.fault = fault }

    var all: [(failure: VideoReaderFailure, retrying: Bool)] { lock.withLock { recorded } }
    var retrying: [Bool] { all.map { $0.retrying } }

    var observer: VideoReaderRetry.Observer {
        { [self] failure, retrying in
            lock.withLock { recorded.append((failure, retrying)) }
            if retrying { fault?.disarm() }
        }
    }
}

/// Cancels the readers in `ending` (1 is the first reader the task opens)
/// before their read number `read`, on the reading thread between two reads.
/// Each then ends `.cancelled`, as a reader that fails partway ends `.failed`.
private final class ReaderEnder: @unchecked Sendable {
    private let lock = NSLock()
    private let ending: Set<Int>
    private let read: Int
    private var order: [ObjectIdentifier: (ordinal: Int, reads: Int)] = [:]
    private var endedCount = 0

    init(ending: Set<Int>, atRead read: Int) {
        self.ending = ending
        self.read = read
    }

    var ended: Int { lock.withLock { endedCount } }

    var hook: @Sendable (AVAssetReader) -> Void {
        { [self] reader in
            let end = lock.withLock { () -> Bool in
                let id = ObjectIdentifier(reader)
                var entry = order[id] ?? (ordinal: order.count + 1, reads: 0)
                entry.reads += 1
                order[id] = entry
                guard entry.reads == read, ending.contains(entry.ordinal) else { return false }
                endedCount += 1
                return true
            }
            if end { reader.cancelReading() }
        }
    }
}

/// The real decoders over the real in-memory resource loader, with data requests
/// failed partway through the clip. AVFoundation requests a failed range again,
/// so the fault stays armed until a reader pass fails. A compressed-sample pass
/// that loses its bytes ends `.completed` with fewer samples instead, and a
/// later pass reads bytes an earlier pass loaded, so the MiMo and Qwen4 tests
/// end their decoded-frame reader partway instead.
final class VideoReaderRetryTests: XCTestCase {
    private struct Unrelated: Error {}

    private func clip(_ name: String) throws -> Data {
        try Data(
            contentsOf: XCTUnwrap(
                Bundle.module.url(
                    forResource: name, withExtension: nil, subdirectory: "MiMoOpenRouter")))
    }

    /// Both clips keep `moov` after `mdat`, so the middle byte is sample data that
    /// only a full reader pass requests.
    private func midClipFault(_ data: Data, error: any Error & Sendable) -> VideoDataRequestFault {
        VideoDataRequestFault(offset: data.count / 2, error: error)
    }

    private func owner(_ data: Data, _ fault: VideoDataRequestFault? = nil) throws
        -> MemoryBackedVideoAsset
    {
        try VideoDataRequestFault.$current.withValue(fault) {
            try MemoryBackedVideoAsset(videoData: data)
        }
    }

    private func observing<Output>(_ passes: FailedPasses, _ body: () async throws -> Output)
        async throws -> Output
    {
        try await VideoReaderRetry.$failedPassObserver.withValue(passes.observer, operation: body)
    }

    private let decoderUnavailable = NSError(
        domain: AVFoundationErrorDomain,
        code: AVError.Code.decoderTemporarilyUnavailable.rawValue)

    // MARK: - Failure and retry

    func testFailureKeepsStatusAndTheErrorChainCodesOnly() {
        func chain(_ codes: [(String, Int)]) -> NSError? {
            var error: NSError?
            for (domain, code) in codes.reversed() {
                var info: [String: Any] = [NSLocalizedDescriptionKey: "frame of a private clip"]
                if let error { info[NSUnderlyingErrorKey] = error }
                error = NSError(domain: domain, code: code, userInfo: info)
            }
            return error
        }
        let malfunction = VideoReaderFailure(
            status: AVAssetReader.Status.failed.rawValue,
            error: chain([
                (AVFoundationErrorDomain, AVError.Code.unknown.rawValue),
                (NSOSStatusErrorDomain, Int(kVTVideoDecoderMalfunctionErr)),
            ]))
        XCTAssertTrue(malfunction.isTransient)
        XCTAssertEqual(
            malfunction.description,
            "video reader status 3: AVFoundationErrorDomain -11800 < NSOSStatusErrorDomain -12911")
        XCTAssertFalse(malfunction.description.contains("private"))

        let badData = VideoReaderFailure(
            status: AVAssetReader.Status.failed.rawValue,
            error: chain([
                (AVFoundationErrorDomain, AVError.Code.decodeFailed.rawValue),
                (NSOSStatusErrorDomain, Int(kVTVideoDecoderBadDataErr)),
            ]))
        XCTAssertFalse(badData.isTransient)

        let deep = VideoReaderFailure(
            status: AVAssetReader.Status.failed.rawValue,
            error: chain((0 ..< 6).map { ("D\($0)", $0) }))
        XCTAssertEqual(deep.codes.map(\.domain), ["D0", "D1", "D2", "D3"])

        // Every table entry, as the reader's error or under AVFoundation's -11800.
        let transient: [(String, Int)] = [
            (AVFoundationErrorDomain, AVError.Code.outOfMemory.rawValue),
            (AVFoundationErrorDomain, -11819),
            (AVFoundationErrorDomain, AVError.Code.decoderTemporarilyUnavailable.rawValue),
            (AVFoundationErrorDomain, -11847),
            (NSOSStatusErrorDomain, Int(kVTInvalidSessionErr)),
            (NSOSStatusErrorDomain, Int(kVTAllocationFailedErr)),
            (NSOSStatusErrorDomain, Int(kVTVideoDecoderMalfunctionErr)),
            (NSOSStatusErrorDomain, Int(kVTVideoDecoderNotAvailableNowErr)),
            (NSOSStatusErrorDomain, Int(kVTVideoDecoderRemovedErr)),
        ]
        XCTAssertEqual(
            Set(transient.map { "\($0.0) \($0.1)" }),
            Set(
                VideoReaderFailure.transientCodes.flatMap { domain, codes in
                    codes.map { "\(domain) \($0)" }
                }))
        for (domain, code) in transient {
            for codes in [[(domain, code)], [(AVFoundationErrorDomain, -11800), (domain, code)]] {
                XCTAssertTrue(
                    VideoReaderFailure(status: 3, error: chain(codes)).isTransient,
                    "\(codes)")
            }
        }
        for (domain, code) in [
            (AVFoundationErrorDomain, AVError.Code.unknown.rawValue),
            (AVFoundationErrorDomain, AVError.Code.decodeFailed.rawValue),
            (NSOSStatusErrorDomain, -11847),
            ("MLXLMCommon.MemoryBackedVideoAssetError", 2),
        ] {
            XCTAssertFalse(
                VideoReaderFailure(status: 3, error: chain([(domain, code)])).isTransient,
                "\(domain) \(code)")
        }

        let silent = VideoReaderFailure(status: AVAssetReader.Status.reading.rawValue, error: nil)
        XCTAssertFalse(silent.isTransient)
        XCTAssertEqual(silent.description, "video reader status 1: no error")
    }

    func testRetryRunsOneSecondPassAndReportsEachFailedPass() async throws {
        let first = VideoReaderFailure(status: 3, error: decoderUnavailable)
        let second = VideoReaderFailure(status: 4, error: nil)
        var calls = 0
        func run(_ passes: FailedPasses, _ pass: () throws -> Int) async throws -> Int {
            try await observing(passes) {
                try await VideoReaderRetry.run { () async throws -> Int in
                    calls += 1
                    return try pass()
                }
            }
        }

        let recovered = FailedPasses()
        let value = try await run(recovered) {
            if calls == 1 { throw first }
            return 7
        }
        XCTAssertEqual(value, 7)
        XCTAssertEqual(calls, 2)
        XCTAssertEqual(recovered.all.map { $0.failure }, [first])
        XCTAssertEqual(recovered.retrying, [true])

        let failed = FailedPasses()
        calls = 0
        do {
            _ = try await run(failed) { throw calls == 1 ? first : second }
            XCTFail("both passes failed but the run returned")
        } catch {
            XCTAssertEqual(error as? VideoReaderFailure, second)
        }
        XCTAssertEqual(calls, 2)
        XCTAssertEqual(failed.all.map { $0.failure }, [first, second])
        XCTAssertEqual(failed.retrying, [true, false])

        let unrelated = FailedPasses()
        calls = 0
        do {
            _ = try await run(unrelated) { throw Unrelated() }
            XCTFail("the unrelated error was swallowed")
        } catch {
            XCTAssertTrue(error is Unrelated)
        }
        XCTAssertEqual(calls, 1)
        XCTAssertTrue(unrelated.all.isEmpty)
    }

    func testCancelledTaskDoesNotRetry() async throws {
        let failure = VideoReaderFailure(status: 4, error: nil)
        let passes = FailedPasses()
        let task = Task { () -> (calls: Int, error: (any Error)?) in
            withUnsafeCurrentTask { $0?.cancel() }
            var calls = 0
            do {
                _ = try await VideoReaderRetry.$failedPassObserver.withValue(passes.observer) {
                    try await VideoReaderRetry.run { () async throws -> Int in
                        calls += 1
                        throw failure
                    }
                }
                return (calls, nil)
            } catch {
                return (calls, error)
            }
        }
        let result = await task.value
        XCTAssertEqual(result.calls, 1)
        XCTAssertTrue(result.error is CancellationError)
        XCTAssertEqual(passes.retrying, [false])
    }

    /// The fault serves the bytes before its offset, then fails the request,
    /// only while armed.
    func testFaultFailsOnlyArmedRequestsHoldingItsOffset() {
        let fault = VideoDataRequestFault(
            offset: 10, error: MemoryBackedVideoAssetError.invalidRange, armed: false)
        XCTAssertNil(fault.intercept(4 ..< 20))
        fault.arm()
        XCTAssertNil(fault.intercept(0 ..< 10))
        let injected = fault.intercept(4 ..< 20)
        XCTAssertEqual(injected?.served, 4 ..< 10)
        XCTAssertEqual(injected?.error as? MemoryBackedVideoAssetError, .invalidRange)
        fault.disarm()
        XCTAssertNil(fault.intercept(4 ..< 20))
        XCTAssertEqual(fault.injected, 1)
    }

    // MARK: - GLM

    private func glmProcessor() throws -> GLM5NextProcessor {
        try GLM5NextProcessor(configuration: .init(), tokenizer: GLM5NextTestTokenizer())
    }

    func testGLMRetryAfterAFailedReaderPassKeepsTheDecodedFrames() async throws {
        let data = try clip("input-video-url.mp4")
        let processor = try glmProcessor()
        let baseline = try await processor.decode(.memoryBacked(owner(data)))
        let fault = midClipFault(data, error: MemoryBackedVideoAssetError.invalidRange)
        let passes = FailedPasses(disarming: fault)
        let retried = try await observing(passes) {
            try await processor.decode(.memoryBacked(owner(data, fault)))
        }
        XCTAssertGreaterThanOrEqual(fault.injected, 1)
        XCTAssertEqual(passes.retrying, [true])
        XCTAssertFalse(try XCTUnwrap(passes.all.first).failure.codes.isEmpty)
        XCTAssertEqual(retried.sourceFrameCount, baseline.sourceFrameCount)
        XCTAssertEqual(retried.sourceIndices, baseline.sourceIndices)
        XCTAssertEqual(retried.frames.map(\.pixels), baseline.frames.map(\.pixels))
    }

    /// A resource that ends partway is a fault of the media; an unavailable
    /// decoder is a fault of the machine.
    func testGLMRefusesWithTheReaderFailureWhenBothPassesFail() async throws {
        let data = try clip("input-video-url.mp4")
        let processor = try glmProcessor()
        for (error, transient) in [
            (MemoryBackedVideoAssetError.invalidRange as any Error & Sendable, false),
            (decoderUnavailable, true),
        ] {
            let fault = midClipFault(data, error: error)
            let passes = FailedPasses()
            do {
                _ = try await observing(passes) {
                    try await processor.decode(.memoryBacked(owner(data, fault)))
                }
                XCTFail("\(error): both passes failed but the video decoded")
            } catch GLM5NextVisionError.videoReader(let failure) {
                XCTAssertEqual(failure.isTransient, transient, "\(failure)")
                XCTAssertEqual(failure.status, AVAssetReader.Status.failed.rawValue)
            }
            XCTAssertGreaterThanOrEqual(fault.injected, 2)
            XCTAssertEqual(passes.retrying, [true, false])
        }
    }

    // MARK: - MiMo and Qwen4

    private func ending<Output>(_ ender: ReaderEnder, _ body: () async throws -> Output)
        async throws -> Output
    {
        try await VideoReaderRetry.$beforeReadForTesting.withValue(ender.hook, operation: body)
    }

    /// Reader 1 counts the compressed samples; reader 2 decodes and ends partway.
    func testMiMoRetryAfterAFailedReaderPassKeepsThePlanAndFrames() async throws {
        let data = try clip("input-video-base64.mov")
        let sampling = try MiMoV26EncodedVisualDecoder.Sampling(
            fps: 2, minimumFrames: 4, maximumFrames: 64)
        let limits = MiMoV26EncodedVisualDecoder.Limits(
            maximumPixels: 8_388_608, maximumWorkingBytes: 1 << 30,
            maximumSourceFrames: 360_000, maximumSampledFrames: 64,
            maximumEncodedBytes: 32 << 20)
        func decode() async throws -> (indices: [Int], frames: [[Float]]) {
            let plan = try await MiMoV26EncodedVisualDecoder.inspectVideo(
                owner(data), sampling: sampling, limits: limits)
            let video = try await MiMoV26EncodedVisualDecoder.silentVideo(plan, limits: limits)
            return (plan.sampledIndices, video.frames.map(\.planarRGB))
        }
        let baseline = try await decode()
        let ender = ReaderEnder(ending: [2], atRead: 20)
        let passes = FailedPasses()
        let retried = try await observing(passes) { try await ending(ender) { try await decode() } }
        XCTAssertEqual(ender.ended, 1)
        XCTAssertEqual(passes.retrying, [true])
        XCTAssertEqual(
            passes.all.map { $0.failure.status }, [AVAssetReader.Status.cancelled.rawValue])
        XCTAssertEqual(retried.indices, baseline.indices)
        XCTAssertEqual(retried.frames, baseline.frames)
    }

    /// Reader 1 counts the compressed samples; reader 2 decodes and ends partway.
    func testQwen4RetryAfterAFailedReaderPassKeepsTheSampledFrames() async throws {
        let data = try clip("input-video-base64.mov")
        let config = try JSONDecoder().decode(
            Qwen4ExpVideoConfiguration.self, from: Data("{}".utf8))
        func sample() async throws -> Qwen4ExpVideoSampler.Sample {
            try await Qwen4ExpVideoSampler.sample(.memoryBacked(owner(data)), config: config) {
                frame, _ in
                frame.transformed(by: CGAffineTransform(scaleX: 0.125, y: 0.125))
            }
        }
        let baseline = try await sample()
        let ender = ReaderEnder(ending: [2], atRead: 20)
        let passes = FailedPasses()
        let retried = try await observing(passes) { try await ending(ender) { try await sample() } }
        XCTAssertEqual(ender.ended, 1)
        XCTAssertEqual(passes.retrying, [true])
        XCTAssertEqual(retried.indices, baseline.indices)
        XCTAssertEqual(retried.sourceFPS, baseline.sourceFPS)
        XCTAssertEqual(retried.frames.count, baseline.frames.count)
        for (lhs, rhs) in zip(retried.frames, baseline.frames) {
            XCTAssertTrue(arrayEqual(lhs, rhs).item(Bool.self))
        }
    }
}
