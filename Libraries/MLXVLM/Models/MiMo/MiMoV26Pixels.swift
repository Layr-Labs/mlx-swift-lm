// Copyright © 2026 Eigen Labs.
// Decoded-RGB scalar preparation derived from SGLang (Apache-2.0), commit
// 67bb6a58d0dad4a39af80fa1b2bf86f0de0cb99b, processors/mimo_v2.py.
// F.interpolate(float, mode="bilinear", align_corners=false, antialias=false),
// then native 0...255 ImageNet normalization and spatiotemporal patch ordering.
// No image codecs, alpha/EXIF policy, frame sampling, MLX arrays or model load.
import Foundation

public enum MiMoV26Pixels {
    public enum Failure: Error, Equatable, Sendable {
        case invalidInput(String)
        case resourceLimit(String)
        case overflow(String)
    }

    /// Already color/alpha/orientation-resolved RGB, planar CHW, range0...255.
    /// A caller must not pass normalized0...1 values as though they were pixels.
    public struct DecodedRGB: Sendable {
        public let height, width: Int
        public let planarRGB: [Float]

        public init(height: Int, width: Int, planarRGB: [Float]) {
            self.height = height
            self.width = width
            self.planarRGB = planarRGB
        }
    }

    /// Explicit caller resource admission. Byte accounting bounds logical
    /// buffers including retained inputs; it is not an allocator/RSS guarantee.
    public struct Limits: Sendable {
        public let maximumInputElements, maximumOutputElements, maximumWorkingBytes: Int

        public init(maximumInputElements: Int, maximumOutputElements: Int, maximumWorkingBytes: Int)
        {
            self.maximumInputElements = maximumInputElements
            self.maximumOutputElements = maximumOutputElements
            self.maximumWorkingBytes = maximumWorkingBytes
        }
    }

    public struct Prepared: Sendable {
        public let geometry: MiMoV26MediaGeometry.Plan
        /// Row-major[patchCount,patchVectorSize], Float32, ready for the native
        /// patch projection. This is not a tensor allocation or a tower result.
        public let patchValues: [Float]
        public let inputElementCount, plannedWorkingBytes: Int
    }

    private struct AxisSample {
        let lower, upper: Int
        let lowerWeight, upperWeight: Float
    }
    private static let means: [Float] = [123.675, 116.28, 103.53]
    private static let deviations: [Float] = [58.395, 57.12, 57.375]

    public static func image(
        _ frame: DecodedRGB, settings: MiMoV26MediaGeometry.Settings,
        limits: Limits
    ) throws -> Prepared {
        let plan = try MiMoV26MediaGeometry.image(
            height: frame.height, width: frame.width, settings: settings)
        return try prepare(frames: [frame], plan: plan, settings: settings, limits: limits)
    }

    /// `frames` contains the entire already-sampled clip. Segment selection,
    /// timestamps and decoder sampling belong to a separate upstream boundary;
    /// a mismatched divisor must not silently alter the native resize budget.
    public static func video(
        frames: [DecodedRGB], sampledFrameCount: Int,
        settings: MiMoV26MediaGeometry.Settings, limits: Limits
    ) throws -> Prepared {
        guard !frames.isEmpty, frames.count == sampledFrameCount, let first = frames.first else {
            throw Failure.invalidInput("sampled frame count must match the decoded clip")
        }
        let plan = try MiMoV26MediaGeometry.video(
            height: first.height, width: first.width,
            sampledFrames: sampledFrameCount, settings: settings)
        return try prepare(frames: frames, plan: plan, settings: settings, limits: limits)
    }

    /// Allocation-free quote shared by actual preparation and managed admission.
    /// A configured resource ceiling is not the amount this request allocates.
    static func workingByteCount(inputElements: Int, frameCount: Int,
                                 plan: MiMoV26MediaGeometry.Plan) throws -> Int {
        let inputBytes = try product([inputElements, MemoryLayout<Float>.stride], "input bytes")
        let outputBytes = try product(
            [plan.patchElementCount, MemoryLayout<Float>.stride], "output bytes")
        let axisEntries = try sum(plan.height, plan.width, "axis entries")
        let axisBytes = try product([axisEntries, MemoryLayout<AxisSample>.stride], "axis bytes")
        let frameBytes = try product(
            [frameCount, MemoryLayout<DecodedRGB>.stride], "frame metadata")
        // Include array/plan descriptors and a fixed allowance for stack/value
        // descriptors. No resized, normalized, padded or copied-frame array is
        // materialized; all pixel work writes directly into the final buffer.
        let overhead = try sum(frameBytes, 1024, "metadata bytes")
        return try sum(
            try sum(inputBytes, outputBytes, "input/output bytes"),
            try sum(axisBytes, overhead, "scratch bytes"), "working bytes")
    }

    private static func prepare(
        frames: [DecodedRGB], plan: MiMoV26MediaGeometry.Plan,
        settings: MiMoV26MediaGeometry.Settings, limits: Limits
    ) throws -> Prepared {
        guard limits.maximumInputElements > 0, limits.maximumOutputElements > 0,
            limits.maximumWorkingBytes > 0, let first = frames.first
        else {
            throw Failure.invalidInput("invalid resource limits or empty frames")
        }
        let frameElements = try product([3, first.height, first.width], "input frame")
        let inputElements = try product([frameElements, frames.count], "input clip")
        guard inputElements <= limits.maximumInputElements,
            plan.patchElementCount <= limits.maximumOutputElements
        else {
            throw Failure.resourceLimit("input/output element bound")
        }
        // Exact integer coordinates within Float's native input arithmetic.
        // Larger decoded/resized axes need a separately qualified path.
        guard
            [first.height, first.width, plan.height, plan.width].allSatisfy({
                $0 > 0 && $0 <= 16_777_216
            })
        else {
            throw Failure.invalidInput("pixel axis exceeds exact Float integer range")
        }
        let bytes = try workingByteCount(inputElements: inputElements, frameCount: frames.count, plan: plan)
        guard bytes <= limits.maximumWorkingBytes else {
            throw Failure.resourceLimit("planned working bytes")
        }
        for frame in frames {
            guard frame.height == first.height, frame.width == first.width,
                frame.planarRGB.count == frameElements
            else {
                throw Failure.invalidInput(
                    "all RGB frames must have matching complete CHW dimensions")
            }
            guard frame.planarRGB.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 255 }) else {
                throw Failure.invalidInput("decoded RGB must contain finite0...255 pixels")
            }
        }

        let horizontal = samples(input: first.width, output: plan.width)
        let vertical = samples(input: first.height, output: plan.height)
        var patches = [Float](repeating: 0, count: plan.patchElementCount)
        var destination = 0
        let patch = settings.patchSize
        let merge = settings.mergeSize
        let temporal = settings.temporalPatchSize
        let sourcePlane = first.height * first.width  // Checked frame product.
        // Reference view(Tgroup,Tinner,C,Hgroup,Hinner,Py,Wgroup,Winner,Px)
        // permutes to(Tgroup,Hgroup,Wgroup,Hinner,Winner,C,Tinner,Py,Px).
        for groupT in 0 ..< plan.gridT {
            for groupH in 0 ..< (plan.gridH / merge) {
                for groupW in 0 ..< (plan.gridW / merge) {
                    for innerH in 0 ..< merge {
                        for innerW in 0 ..< merge {
                            for channel in 0 ..< 3 {
                                for innerT in 0 ..< temporal {
                                    let sourceFrame = min(
                                        groupT * temporal + innerT, frames.count - 1)
                                    let source = frames[sourceFrame].planarRGB
                                    for patchY in 0 ..< patch {
                                        let y = vertical[(groupH * merge + innerH) * patch + patchY]
                                        for patchX in 0 ..< patch {
                                            let x = horizontal[
                                                (groupW * merge + innerW) * patch + patchX]
                                            let base = channel * sourcePlane
                                            let top =
                                                x.lowerWeight
                                                * source[base + y.lower * first.width + x.lower]
                                                + x.upperWeight
                                                * source[base + y.lower * first.width + x.upper]
                                            let bottom =
                                                x.lowerWeight
                                                * source[base + y.upper * first.width + x.lower]
                                                + x.upperWeight
                                                * source[base + y.upper * first.width + x.upper]
                                            let resized =
                                                y.lowerWeight * top + y.upperWeight * bottom
                                            patches[destination] =
                                                (resized - means[channel]) / deviations[channel]
                                            destination += 1
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        guard destination == plan.patchElementCount else {
            throw Failure.invalidInput("internal patch accounting")
        }
        return Prepared(
            geometry: plan, patchValues: patches, inputElementCount: inputElements,
            plannedWorkingBytes: bytes)
    }

    private static func samples(input: Int, output: Int) -> [AxisSample] {
        let ratio = Float(input) / Float(output)
        return (0 ..< output).map { index in
            if input == output {
                return AxisSample(lower: index, upper: index, lowerWeight: 1, upperWeight: 0)
            }
            // Native align_corners=false half-pixel coordinates, clipped at
            // the left edge; the right neighbor is clamped to the last pixel.
            let coordinate = max(Float(0), ratio * (Float(index) + 0.5) - 0.5)
            let lower = min(Int(floor(coordinate)), input - 1)
            let upper = min(lower + 1, input - 1)
            let weight = min(max(coordinate - Float(lower), 0), 1)
            return AxisSample(
                lower: lower, upper: upper, lowerWeight: 1 - weight, upperWeight: weight)
        }
    }

    private static func product(_ values: [Int], _ field: String) throws -> Int {
        var result = 1
        for value in values {
            guard value > 0 else { throw Failure.invalidInput(field) }
            let next = result.multipliedReportingOverflow(by: value)
            guard !next.overflow else { throw Failure.overflow(field) }
            result = next.partialValue
        }
        return result
    }
    private static func sum(_ lhs: Int, _ rhs: Int, _ field: String) throws -> Int {
        let result = lhs.addingReportingOverflow(rhs)
        guard !result.overflow else { throw Failure.overflow(field) }
        return result.partialValue
    }
}
