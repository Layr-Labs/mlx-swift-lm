// Copyright © 2026 Eigen Labs.

import CoreMedia
import Testing

extension UnitTests {
    @Suite
    struct AudioReaderDiagnosticTimelineTests {
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

        private func segments(_ pairs: [(Int64, Int64)]) -> [AudioReaderDiagnosticTimeline.Segment]
        {
            pairs.map {
                .init(
                    start: CMTime(value: $0.0, timescale: 48000),
                    duration: CMTime(value: $0.1, timescale: 48000))
            }
        }

        @Test func testCapturedDifferentPartitionsHaveExactlyTheSameTimeline() throws {
            let baseline = try #require(AudioReaderDiagnosticTimeline(segments(regular)))
            let candidate = try #require(AudioReaderDiagnosticTimeline(segments(repartitioned)))
            #expect(baseline == candidate)
            #expect(CMTimeCompare(candidate.start, .zero) == 0)
            #expect(CMTimeCompare(candidate.end, CMTime(value: 239983, timescale: 48000)) == 0)
            #expect(CMTimeCompare(candidate.coverage, CMTime(value: 239983, timescale: 48000)) == 0)
        }

        @Test func testOneTickGapIsRejectedEvenWithTheSameOuterEndpoints() {
            var damaged = repartitioned
            damaged[18].0 += 1
            #expect(AudioReaderDiagnosticTimeline(segments(damaged)) == nil)
        }

        @Test func testOneTickOverlapIsRejectedEvenWithTheSameOuterEndpoints() {
            var damaged = repartitioned
            damaged[18].0 -= 1
            #expect(AudioReaderDiagnosticTimeline(segments(damaged)) == nil)
        }

        @Test func testOneTickMissingStartDiffersDespiteTheSameEnd() throws {
            let baseline = try #require(AudioReaderDiagnosticTimeline(segments(regular)))
            var changed = repartitioned
            changed[0] = (1, 8191)
            let candidate = try #require(AudioReaderDiagnosticTimeline(segments(changed)))
            #expect(candidate != baseline)
            #expect(CMTimeCompare(candidate.end, baseline.end) == 0)
            #expect(CMTimeCompare(candidate.coverage, CMTime(value: 239982, timescale: 48000)) == 0)
        }

        @Test func testOneTickTruncatedOrExtendedEndDiffers() throws {
            let baseline = try #require(AudioReaderDiagnosticTimeline(segments(regular)))
            for delta in [Int64(-1), 1] {
                var changed = repartitioned
                changed[changed.count - 1].1 += delta
                let candidate = try #require(AudioReaderDiagnosticTimeline(segments(changed)))
                #expect(candidate != baseline)
                #expect(CMTimeCompare(candidate.start, baseline.start) == 0)
            }
        }

        @Test func testEquivalentRationalTimescalesCompareEqual() throws {
            let baseline = try #require(AudioReaderDiagnosticTimeline(segments(regular)))
            let rescaled = repartitioned.map {
                AudioReaderDiagnosticTimeline.Segment(
                    start: CMTime(value: $0.0 * 2, timescale: 96000),
                    duration: CMTime(value: $0.1 * 2, timescale: 96000))
            }
            #expect(try #require(AudioReaderDiagnosticTimeline(rescaled)) == baseline)
        }

        @Test func testEmptyOrInvalidTimelinesAreRejected() {
            #expect(AudioReaderDiagnosticTimeline([]) == nil)
            let invalidTimes: [CMTime] = [
                .invalid, .indefinite, .positiveInfinity, .negativeInfinity,
                CMTime(value: 1, timescale: 1, flags: [.valid, .hasBeenRounded], epoch: 0),
            ]
            for time in invalidTimes {
                #expect(
                    AudioReaderDiagnosticTimeline([
                        .init(start: time, duration: CMTime(value: 1, timescale: 1))
                    ]) == nil)
                #expect(AudioReaderDiagnosticTimeline([.init(start: .zero, duration: time)]) == nil)
            }
            for duration in [
                CMTime.zero, CMTime(value: -1, timescale: 1),
                CMTime(value: 1, timescale: 1, flags: .valid, epoch: 1),
            ] {
                #expect(
                    AudioReaderDiagnosticTimeline([.init(start: .zero, duration: duration)]) == nil)
            }
        }

        @Test func testRoundedOrOverflowingEndArithmeticIsRejected() {
            #expect(
                AudioReaderDiagnosticTimeline([
                    .init(
                        start: CMTime(value: 1, timescale: Int32.max),
                        duration: CMTime(value: 1, timescale: Int32.max - 1))
                ]) == nil)
            #expect(
                AudioReaderDiagnosticTimeline([
                    .init(
                        start: CMTime(value: Int64.max, timescale: 1),
                        duration: CMTime(value: 1, timescale: 1))
                ]) == nil)
        }
    }
}
