import Foundation

/// CPU decode allocations retained by the caller, plus one sequential decoder's
/// transient allowance. Codec-private pools still require host/system headroom;
/// this is not an AVFoundation process-RSS limit or a replacement for that reserve.
public struct MiMoV26VisualDecodeMemory: Sendable {
    public let retainedBytes: Int
    public let transientBytes: Int

    public var peakBytes: Int {
        get throws { try Self.sum(retainedBytes, transientBytes) }
    }

    public static func image(pixels: Int) throws -> Self {
        guard pixels > 0 else { throw MiMoV26EncodedVisualDecoder.Failure.limit }
        return Self(
            retainedBytes: try product(pixels, 12),
            transientBytes: try sum(product(pixels, 20), 1 << 20))
    }

    static func video(
        encodedBytes: Int, sourceFrames: Int, sampledFrames: Int,
        pixels: Int, maximumControlMarkers: Int
    ) throws -> Self {
        guard sourceFrames >= sampledFrames, sampledFrames > 0, pixels > 0 else {
            throw MiMoV26EncodedVisualDecoder.Failure.limit
        }
        return Self(
            retainedBytes: try sum(
                encodedBytes, product(sampledFrames, pixels, 12),
                product(sourceFrames, 64), product(maximumControlMarkers, 64), 1 << 20),
            transientBytes: try sum(product(pixels, 32), 1 << 20))
    }

    static func product(_ values: Int...) throws -> Int {
        var result = 1
        for value in values {
            let (next, overflow) = result.multipliedReportingOverflow(by: value)
            guard value >= 0, !overflow else {
                throw MiMoV26EncodedVisualDecoder.Failure.arithmeticOverflow
            }
            result = next
        }
        return result
    }

    static func sum(_ values: Int...) throws -> Int {
        var result = 0
        for value in values {
            let (next, overflow) = result.addingReportingOverflow(value)
            guard value >= 0, !overflow else {
                throw MiMoV26EncodedVisualDecoder.Failure.arithmeticOverflow
            }
            result = next
        }
        return result
    }
}
