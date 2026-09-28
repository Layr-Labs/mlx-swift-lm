// Copyright © 2026 Eigen Labs.
// Native scalar contracts derived from SGLang (Apache-2.0), commit
// 67bb6a58d0dad4a39af80fa1b2bf86f0de0cb99b, multimodal/processors/mimo_v2.py.
// No pixel processing, media decoding, tokenization or model execution.
import Foundation

public enum MiMoV26MediaGeometry {
    public enum Failure: Error, Equatable, Sendable {
        case invalidSettings(String)
        case invalidGeometry(String)
        case overflow(String)
        case unusableResize
    }

    /// Explicit effective settings, after native model/request precedence.
    /// Sampling fps/min_frames/max_frames are a separate decoder contract.
    public struct Settings: Codable, Equatable, Sendable {
        public let patchSize, mergeSize, temporalPatchSize, temporalCompressionRatio: Int
        public let imageMinPixels, imageMaxPixels: Int
        public let videoMinPixels, videoMaxPixels, videoTotalMaxPixels: Int

        public init(patchSize: Int, mergeSize: Int, temporalPatchSize: Int,
                    temporalCompressionRatio: Int, imageMinPixels: Int, imageMaxPixels: Int,
                    videoMinPixels: Int, videoMaxPixels: Int, videoTotalMaxPixels: Int) throws {
            self.patchSize = patchSize
            self.mergeSize = mergeSize
            self.temporalPatchSize = temporalPatchSize
            self.temporalCompressionRatio = temporalCompressionRatio
            self.imageMinPixels = imageMinPixels
            self.imageMaxPixels = imageMaxPixels
            self.videoMinPixels = videoMinPixels
            self.videoMaxPixels = videoMaxPixels
            self.videoTotalMaxPixels = videoTotalMaxPixels
            try validate()
        }

        fileprivate func validate() throws {
            guard patchSize > 0, mergeSize > 0, temporalPatchSize > 0,
                  temporalCompressionRatio > 0, imageMinPixels > 0,
                  imageMaxPixels >= imageMinPixels, videoMinPixels > 0,
                  videoMaxPixels >= videoMinPixels, videoTotalMaxPixels > 0 else {
                throw Failure.invalidSettings("positive dimensions and ordered pixel bounds required")
            }
            _ = try dimension(product([patchSize, mergeSize], "spatial factor"), "spatial factor")
            _ = try dimension(product([temporalPatchSize, temporalCompressionRatio], "temporal factor"), "temporal factor")
            _ = try dimension(product([3, temporalPatchSize, patchSize, patchSize], "patch vector"), "patch vector")
            for limit in [imageMinPixels, imageMaxPixels, videoMinPixels, videoMaxPixels, videoTotalMaxPixels] {
                try exactDoubleInteger(limit, "pixel budget")
            }
        }
    }

    /// Counts describe actual native rounding/padding, not admission success.
    /// Video total_max_pixels is a soft resize budget: its minimum floor and
    /// last-frame duplication can exceed it. Report that; never silently clamp.
    public struct Plan: Codable, Equatable, Sendable {
        public let height, width: Int
        public let sampledFrames, segmentFrames, alignedFrames, duplicatedFrames: Int
        public let gridT, gridH, gridW: Int
        public let patchCount, patchVectorSize, patchElementCount: Int
        public let mediaTokens, timestampCount, fixedWrapperTokens: Int
        public let promptTokensExcludingTimestamps: Int
        public let resizedPixelsPerFrame, alignedPixels, aggregateBudgetPixels: Int
        public let effectiveMaxPixelsPerFrame: Int
        public let exceedsPerFrameMaximum, belowPerFrameMinimum, exceedsAggregateBudget: Bool
    }

    public static func image(height: Int, width: Int, settings: Settings) throws -> Plan {
        try settings.validate()
        let size = try resize(height: height, width: width,
                              factor: product([settings.patchSize, settings.mergeSize], "spatial factor"),
                              minimum: settings.imageMinPixels, maximum: settings.imageMaxPixels)
        // The native image flattener repeats a still image temporalPatchSize
        // times; gridT stays one and image tokens have no temporal compression.
        return try plan(size: size, sampled: 1, segment: 1,
                        aligned: settings.temporalPatchSize, settings: settings,
                        minimum: settings.imageMinPixels, maximum: settings.imageMaxPixels, video: false)
    }

    /// Dimensions describe decoded sampled frames. `sampledFrames` is the full
    /// sampled clip count used for the native pixel budget, even when a smaller
    /// temporal segment is selected. Frame selection/timestamps are upstream.
    /// `sampledFrameLimit` is an optional explicit caller admission limit; it is
    /// deliberately not inferred from native sampling min_frames/max_frames.
    public static func video(height: Int, width: Int, sampledFrames: Int,
                             segmentFrames: Int? = nil, sampledFrameLimit: Int? = nil,
                             settings: Settings) throws -> Plan {
        try settings.validate()
        _ = try dimension(sampledFrames, "sampled frames")
        let segment = segmentFrames ?? sampledFrames
        guard segment > 0, segment <= sampledFrames else {
            throw Failure.invalidGeometry("segment frames must be within the sampled clip")
        }
        if let limit = sampledFrameLimit {
            guard limit > 0, sampledFrames <= limit else {
                throw Failure.invalidGeometry("explicit sampled-frame admission limit")
            }
        }
        let temporal = try product([settings.temporalPatchSize, settings.temporalCompressionRatio], "temporal factor")
        let numerator = try product([settings.videoTotalMaxPixels, temporal], "video budget numerator")
        let perFrame = max(settings.videoMinPixels,
                           min(numerator / sampledFrames, settings.videoMaxPixels))
        let groups = segment / temporal + (segment % temporal == 0 ? 0 : 1)
        let aligned = try dimension(product([groups, temporal], "aligned frames"), "aligned frames")
        let size = try resize(height: height, width: width,
                              factor: product([settings.patchSize, settings.mergeSize], "spatial factor"),
                              minimum: settings.videoMinPixels, maximum: perFrame)
        return try plan(size: size, sampled: sampledFrames, segment: segment,
                        aligned: aligned, settings: settings,
                        minimum: settings.videoMinPixels, maximum: perFrame, video: true)
    }

    private static func resize(height: Int, width: Int, factor: Int,
                               minimum: Int, maximum: Int) throws -> (Int, Int) {
        _ = try dimension(height, "input height")
        _ = try dimension(width, "input width")
        try exactDoubleInteger(product([height, width], "input pixels"), "input pixels")
        var h = height, w = width
        if min(h, w) < factor {
            // Keep the reference's branch ordering: tiny axes are scaled first,
            // so its >200 aspect refusal is NOT applied in this branch.
            let scale = Double(factor) / Double(min(h, w))
            h = try roundedInteger(Double(h) * scale, .toNearestOrEven)
            w = try roundedInteger(Double(w) * scale, .toNearestOrEven)
        } else if Double(max(h, w)) / Double(min(h, w)) > 200 {
            throw Failure.invalidGeometry("native aspect ratio exceeds 200")
        }
        let pixels = try product([h, w], "scaled input pixels")
        try exactDoubleInteger(pixels, "scaled input pixels")
        var targetH = try multiple(Double(h) / Double(factor), factor, .toNearestOrEven)
        var targetW = try multiple(Double(w) / Double(factor), factor, .toNearestOrEven)
        let roundedPixels = try product([targetH, targetW], "rounded pixels")
        if roundedPixels > maximum {
            let beta = sqrt(Double(pixels) / Double(maximum))
            targetH = try multiple(Double(h) / beta / Double(factor), factor, .down)
            targetW = try multiple(Double(w) / beta / Double(factor), factor, .down)
        } else if roundedPixels < minimum {
            let beta = sqrt(Double(minimum) / Double(pixels))
            targetH = try multiple(Double(h) * beta / Double(factor), factor, .up)
            targetW = try multiple(Double(w) * beta / Double(factor), factor, .up)
        }
        // The reference can produce a zero axis for extreme/budget-limited
        // geometry. Refuse before allocation; do not invent a one-patch clamp.
        guard targetH > 0, targetW > 0 else { throw Failure.unusableResize }
        _ = try dimension(targetH, "output height")
        _ = try dimension(targetW, "output width")
        return (targetH, targetW)
    }

    private static func plan(size: (Int, Int), sampled: Int, segment: Int, aligned: Int,
                             settings: Settings, minimum: Int, maximum: Int, video: Bool) throws -> Plan {
        let (height, width) = size
        let gridT = aligned / settings.temporalPatchSize
        let gridH = height / settings.patchSize, gridW = width / settings.patchSize
        let count = try dimension(product([gridT, gridH, gridW], "patch count"), "patch count")
        let vector = try dimension(product([3, settings.temporalPatchSize, settings.patchSize,
                                           settings.patchSize], "patch vector"), "patch vector")
        let elements = try product([count, vector], "patch elements")
        let spatialMerge = try product([settings.mergeSize, settings.mergeSize], "merge area")
        let compression = video ? settings.temporalCompressionRatio : 1
        let tokens = count / spatialMerge / compression
        let stamps = video ? gridT / compression : 0
        let wrappers = video ? try sum(product([2, stamps], "video wrappers"), 2, "video wrappers") : 2
        let totalTokens = try sum(tokens, wrappers, "prompt tokens")
        let pixels = try product([height, width], "resized pixels")
        let alignedPixels = try product([pixels, aligned], "aligned pixels")
        let budgetPixels = alignedPixels / settings.temporalPatchSize / compression
        return Plan(height: height, width: width, sampledFrames: sampled, segmentFrames: segment,
                    alignedFrames: aligned, duplicatedFrames: aligned - segment,
                    gridT: gridT, gridH: gridH, gridW: gridW,
                    patchCount: count, patchVectorSize: vector, patchElementCount: elements,
                    mediaTokens: tokens, timestampCount: stamps, fixedWrapperTokens: wrappers,
                    promptTokensExcludingTimestamps: totalTokens,
                    resizedPixelsPerFrame: pixels, alignedPixels: alignedPixels, aggregateBudgetPixels: budgetPixels,
                    effectiveMaxPixelsPerFrame: maximum, exceedsPerFrameMaximum: pixels > maximum,
                    belowPerFrameMinimum: pixels < minimum,
                    exceedsAggregateBudget: video && budgetPixels > settings.videoTotalMaxPixels)
    }

    private static func dimension(_ value: Int, _ field: String) throws -> Int {
        guard value > 0 else { throw Failure.invalidGeometry(field) }
        guard value <= Int(Int32.max) else { throw Failure.overflow(field) }
        return value
    }

    private static func exactDoubleInteger(_ value: Int, _ field: String) throws {
        guard value > 0, value <= 9_007_199_254_740_991 else { throw Failure.overflow(field) }
    }

    private static func product(_ values: [Int], _ field: String) throws -> Int {
        var result = 1
        for value in values {
            guard value > 0 else { throw Failure.invalidGeometry(field) }
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

    private static func roundedInteger(_ value: Double, _ rule: FloatingPointRoundingRule) throws -> Int {
        let result = value.rounded(rule)
        guard result.isFinite, result >= 0, result <= 9_007_199_254_740_991,
              let integer = Int(exactly: result) else { throw Failure.overflow("resize rounding") }
        return integer
    }

    private static func multiple(_ value: Double, _ factor: Int,
                                 _ rule: FloatingPointRoundingRule) throws -> Int {
        let rounded = try roundedInteger(value, rule)
        if rounded == 0 { return 0 }
        return try product([rounded, factor], "rounded dimension")
    }
}
