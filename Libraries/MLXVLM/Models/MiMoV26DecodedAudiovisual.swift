// Copyright © 2026 Eigen Labs.
// SGLang67bb6a58 mimo_v2.py945–1100: selected interleave0 unit contract.
// Decoded data and scalar planning only; no decoder, resampler or native work.
import Foundation

public struct MiMoV26DecodedAudiovisual: Sendable {
    /// Preserve the source scalar's arithmetic: default individual end is a
    /// Float32 tensor; some metadata/partial paths carry Python Float64.
    public enum SegmentEnd: Sendable, Equatable {
        case float32(Float)
        case float64(Double)
        var value: Double {
            switch self { case .float32(let x): Double(x); case .float64(let x): x }
        }
        var scaledAudioIndex: Double {
            switch self {
            case .float32(let x): Double(x * Float(6.25))
            case .float64(let x): x * 6.25
            }
        }
    }
    /// Complete, already sampled clip: no omitted-frame pixel-budget divisor.
    public let frames: [MiMoV26Pixels.DecodedRGB]
    public let timestamps: [Float]
    public let wholeAudio: MiMoV26DecodedPCM
    public let segmentEnd: SegmentEnd
    public init(frames: [MiMoV26Pixels.DecodedRGB], timestamps: [Float],
                wholeAudio: MiMoV26DecodedPCM, segmentEnd: SegmentEnd) {
        self.frames = frames; self.timestamps = timestamps
        self.wholeAudio = wholeAudio; self.segmentEnd = segmentEnd
    }
}

/// Metadata is not admission or a model/codec ownership capability.
public enum MiMoV26AudiovisualLayout {
    static let profileName = "sglang67-decoded-av-interleave0-whole-audio-v1"
    public struct Unit: Equatable, Sendable {
        public let visualGroup: Int
        public let timestamp: Float
        /// Indices in the WHOLE clip's final AudioPatch features, not RVQ codes.
        public let audioRange: Range<Int>
    }
    public struct Plan: Equatable, Sendable {
        public let units: [Unit]
        public let alignedFrames, duplicatedFrames: Int
        public let wholeAudioPatches, usedAudioPatches, unusedAudioPatches: Int
    }

    public static func make(timestamps: [Float],
        segmentEnd: MiMoV26DecodedAudiovisual.SegmentEnd,
        temporalPatchSize: Int, wholeAudioPatches: Int, maximumUnits: Int
    ) throws -> Plan {
        guard !timestamps.isEmpty, temporalPatchSize > 0, maximumUnits > 0,
              wholeAudioPatches > 0, wholeAudioPatches <= Int(Int32.max),
              segmentEnd.value.isFinite, segmentEnd.value > 0 else {
            throw MiMoV26MultimodalError.invalidInput("decoded AV temporal bounds")
        }
        var previous: Float = -1
        for time in timestamps {
            guard time.isFinite, time >= 0, time > previous else {
                throw MiMoV26MultimodalError.invalidInput("decoded AV source timestamps")
            }
            previous = time
        }
        guard segmentEnd.value > Double(timestamps[timestamps.count-1]) else {
            throw MiMoV26MultimodalError.invalidInput("decoded AV segment end")
        }
        let groups = timestamps.count / temporalPatchSize + (timestamps.count % temporalPatchSize == 0 ? 0 : 1)
        guard groups <= maximumUnits else { throw MiMoV26MultimodalError.limit("decoded AV unit count") }
        let aligned = groups.multipliedReportingOverflow(by:temporalPatchSize)
        guard !aligned.overflow, aligned.partialValue <= Int(Int32.max) else {
            throw MiMoV26MultimodalError.limit("decoded AV aligned frames")
        }
        func scaled(_ time: Float) -> Double { Double(time * Float(6.25)) }
        func startIndex(_ value: Double) throws -> Int {
            guard value.isFinite, value >= 0, value < Double(wholeAudioPatches) else {
                throw MiMoV26MultimodalError.invalidInput("decoded AV audio begins beyond whole clip")
            }
            return Int(value) // nonnegative native int truncation, after Float32 multiplication
        }
        func clippedEnd(_ value: Double) throws -> Int {
            guard value.isFinite, value >= 0 else {
                throw MiMoV26MultimodalError.invalidInput("decoded AV nonrepresentable audio end")
            }
            // Equivalent to min(int(value),wholeAudioPatches), without unsafe
            // Int conversion for a finite very large source segment end.
            return value >= Double(wholeAudioPatches) ? wholeAudioPatches : Int(value)
        }
        var units: [Unit] = []; units.reserveCapacity(groups)
        var used = 0
        for group in 0..<groups {
            let time = timestamps[group * temporalPatchSize]
            let start = try startIndex(scaled(time))
            let next = group+1 < groups
                ? scaled(timestamps[(group+1) * temporalPatchSize]) : segmentEnd.scaledAudioIndex
            let end = try clippedEnd(next)
            guard end > start else {
                throw MiMoV26MultimodalError.invalidInput("decoded AV empty or reversed native audio unit")
            }
            used += end-start // disjoint ordered ranges, bounded by wholeAudioPatches
            units.append(.init(visualGroup:group,timestamp:time,audioRange:start..<end))
        }
        guard used <= wholeAudioPatches else { throw MiMoV26MultimodalError.invalidInput("decoded AV overlapping units") }
        return .init(units:units,alignedFrames:aligned.partialValue,
            duplicatedFrames:aligned.partialValue-timestamps.count,
            wholeAudioPatches:wholeAudioPatches,usedAudioPatches:used,
            unusedAudioPatches:wholeAudioPatches-used)
    }
}
