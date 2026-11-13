// Copyright © 2026 Eigen Labs.

import CoreMedia

/// Exact coverage of contiguous decoded audio, independent of buffer partitioning.
struct AudioReaderDiagnosticTimeline: Equatable {
    struct Segment {
        let start: CMTime
        let duration: CMTime
    }

    let start: CMTime
    let end: CMTime
    let coverage: CMTime

    init?(_ segments: [Segment]) {
        guard let first = segments.first, Self.isExact(first.start) else { return nil }
        var end = first.start
        var coverage = CMTime.zero
        for segment in segments {
            guard Self.isExact(segment.start), Self.isExact(segment.duration),
                segment.duration.epoch == 0, CMTimeCompare(segment.duration, .zero) > 0,
                CMTimeCompare(segment.start, end) == 0
            else { return nil }
            end = CMTimeAdd(segment.start, segment.duration)
            coverage = CMTimeAdd(coverage, segment.duration)
            guard Self.isExact(end), Self.isExact(coverage) else { return nil }
        }
        self.start = first.start
        self.end = end
        self.coverage = coverage
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        CMTimeCompare(lhs.start, rhs.start) == 0
            && CMTimeCompare(lhs.end, rhs.end) == 0
            && CMTimeCompare(lhs.coverage, rhs.coverage) == 0
    }

    private static func isExact(_ time: CMTime) -> Bool {
        time.isNumeric && time.timescale > 0 && !time.flags.contains(.hasBeenRounded)
    }
}
