// Copyright © 2026 Eigen Labs.

import CoreMedia
import XCTest

final class AudioReaderDiagnosticTimelineTests: XCTestCase {
    // Captured in Fork Tests 43515150576, shard 4, user-input/fresh=true/retain/0.
    // Both reads had 239983 frames and identical raw PCM bytes and SHA256.
    private let regular: [(Int64, Int64)] = [
        (0, 8192),
        (8192, 8192),
        (16384, 8192),
        (24576, 8192),
        (32768, 8192),
        (40960, 8192),
        (49152, 8192),
        (57344, 8192),
        (65536, 8192),
        (73728, 8192),
        (81920, 8192),
        (90112, 8192),
        (98304, 8192),
        (106496, 8192),
        (114688, 8192),
        (122880, 8192),
        (131072, 8192),
        (139264, 8192),
        (147456, 8192),
        (155648, 8192),
        (163840, 8192),
        (172032, 8192),
        (180224, 8192),
        (188416, 8192),
        (196608, 8192),
        (204800, 8192),
        (212992, 8192),
        (221184, 8192),
        (229376, 8192),
        (237568, 2415),
    ]

    private let repartitioned: [(Int64, Int64)] = [
        (0, 8192),
        (8192, 8192),
        (16384, 8192),
        (24576, 8192),
        (32768, 8192),
        (40960, 8192),
        (49152, 8192),
        (57344, 8192),
        (65536, 8192),
        (73728, 8192),
        (81920, 8192),
        (90112, 8192),
        (98304, 8192),
        (106496, 8192),
        (114688, 8192),
        (122880, 8192),
        (131072, 8192),
        (139264, 3382),
        (142646, 8192),
        (150838, 8192),
        (159030, 8192),
        (167222, 8192),
        (175414, 8192),
        (183606, 8192),
        (191798, 8192),
        (199990, 8192),
        (208182, 8192),
        (216374, 8192),
        (224566, 8192),
        (232758, 7225),
    ]

    private func segments(_ pairs: [(Int64, Int64)]) -> [AudioReaderDiagnosticTimeline.Segment] {
        pairs.map {
            .init(
                start: CMTime(value: $0.0, timescale: 48000),
                duration: CMTime(value: $0.1, timescale: 48000))
        }
    }

    func testCapturedDifferentPartitionsHaveExactlyTheSameTimeline() throws {
        let baseline = try XCTUnwrap(AudioReaderDiagnosticTimeline(segments(regular)))
        let candidate = try XCTUnwrap(AudioReaderDiagnosticTimeline(segments(repartitioned)))
        XCTAssertEqual(baseline, candidate)
        XCTAssertEqual(CMTimeCompare(candidate.start, .zero), 0)
        XCTAssertEqual(CMTimeCompare(candidate.end, CMTime(value: 239983, timescale: 48000)), 0)
        XCTAssertEqual(
            CMTimeCompare(candidate.coverage, CMTime(value: 239983, timescale: 48000)), 0)
    }

    func testOneTickGapIsRejectedEvenWithTheSameOuterEndpoints() {
        var damaged = repartitioned
        damaged[18].0 += 1
        XCTAssertNil(AudioReaderDiagnosticTimeline(segments(damaged)))
    }

    func testOneTickOverlapIsRejectedEvenWithTheSameOuterEndpoints() {
        var damaged = repartitioned
        damaged[18].0 -= 1
        XCTAssertNil(AudioReaderDiagnosticTimeline(segments(damaged)))
    }

    func testOneTickMissingStartDiffersDespiteTheSameEnd() throws {
        let baseline = try XCTUnwrap(AudioReaderDiagnosticTimeline(segments(regular)))
        var changed = repartitioned
        changed[0] = (1, 8191)
        let candidate = try XCTUnwrap(AudioReaderDiagnosticTimeline(segments(changed)))
        XCTAssertNotEqual(candidate, baseline)
        XCTAssertEqual(CMTimeCompare(candidate.end, baseline.end), 0)
        XCTAssertEqual(
            CMTimeCompare(candidate.coverage, CMTime(value: 239982, timescale: 48000)), 0)
    }

    func testOneTickTruncatedOrExtendedEndDiffers() throws {
        let baseline = try XCTUnwrap(AudioReaderDiagnosticTimeline(segments(regular)))
        for delta in [Int64(-1), 1] {
            var changed = repartitioned
            changed[changed.count - 1].1 += delta
            let candidate = try XCTUnwrap(AudioReaderDiagnosticTimeline(segments(changed)))
            XCTAssertNotEqual(candidate, baseline)
            XCTAssertEqual(CMTimeCompare(candidate.start, baseline.start), 0)
        }
    }

    func testEquivalentRationalTimescalesCompareEqual() throws {
        let baseline = try XCTUnwrap(AudioReaderDiagnosticTimeline(segments(regular)))
        let rescaled = repartitioned.map {
            AudioReaderDiagnosticTimeline.Segment(
                start: CMTime(value: $0.0 * 2, timescale: 96000),
                duration: CMTime(value: $0.1 * 2, timescale: 96000))
        }
        XCTAssertEqual(try XCTUnwrap(AudioReaderDiagnosticTimeline(rescaled)), baseline)
    }

    func testEmptyOrInvalidTimelinesAreRejected() {
        XCTAssertNil(AudioReaderDiagnosticTimeline([]))
        let invalidTimes: [CMTime] = [
            .invalid, .indefinite, .positiveInfinity, .negativeInfinity,
            CMTime(value: 1, timescale: 1, flags: [.valid, .hasBeenRounded], epoch: 0),
        ]
        for time in invalidTimes {
            XCTAssertNil(
                AudioReaderDiagnosticTimeline([
                    .init(start: time, duration: CMTime(value: 1, timescale: 1))
                ]))
            XCTAssertNil(AudioReaderDiagnosticTimeline([.init(start: .zero, duration: time)]))
        }
        for duration in [
            CMTime.zero, CMTime(value: -1, timescale: 1),
            CMTime(value: 1, timescale: 1, flags: .valid, epoch: 1),
        ] {
            XCTAssertNil(AudioReaderDiagnosticTimeline([.init(start: .zero, duration: duration)]))
        }
    }

    func testRoundedOrOverflowingEndArithmeticIsRejected() {
        XCTAssertNil(
            AudioReaderDiagnosticTimeline([
                .init(
                    start: CMTime(value: 1, timescale: Int32.max),
                    duration: CMTime(value: 1, timescale: Int32.max - 1))
            ]))
        XCTAssertNil(
            AudioReaderDiagnosticTimeline([
                .init(
                    start: CMTime(value: Int64.max, timescale: 1),
                    duration: CMTime(value: 1, timescale: 1))
            ]))
    }
}
