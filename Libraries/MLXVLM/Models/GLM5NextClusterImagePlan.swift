import CoreImage
// Copyright © 2026 Eigen Labs.
import Foundation
import ImageIO
import MLXLLM

/// Bounded still-image planning/decoding. Inspection creates no raster, MLX
/// array, or model. PNG only; animated, oriented and multi-frame inputs refuse.
public enum GLM5NextClusterImagePlan {
    public static let maximumFeatures = 1024
    public static let maximumPixels = 1024 * 1024
    public static let maximumEncodedBytes = 96 * 1024
    public static let maximumPatches = maximumFeatures * 4
    public struct Inspection {
        public let width, height: Int
        public let resize: GLM5NextMediaGeometry.ResizePlan
        public let patchBytes, hostWorkingBytes: Int
    }
    public static func requireConfiguration(_ c: GLM5NextConfiguration) throws {
        let v = c.visionConfig
        guard v.inChannels == 3, v.temporalPatchSize == 2, v.spatialMergeSize == 2,
            v.attentionDropout == 0, v.outHiddenSize == c.textConfig.hiddenSize,
            v.depth > 0, v.depth <= 128, v.hiddenSize <= 8192,
            v.intermediateSize <= 32768, v.projectionIntermediateSize <= 32768,
            v.outHiddenSize <= 8192, v.numHeads > 0, v.numHeads <= 128,
            v.hiddenSize.isMultiple(of: v.numHeads)
        else {
            throw GLM5NextVisionError.unsupported("cluster still-image configuration")
        }
    }
    public static func processorConfiguration(_ c: GLM5NextConfiguration) throws
        -> GLM5NextProcessorConfiguration
    {
        try requireConfiguration(c)
        return try .init(
            patchSize: c.visionConfig.patchSize, imageMaximumTokens: maximumFeatures,
            maximumDecodedBytes: 128 * 1024 * 1024, maximumPreparedBytes: 128 * 1024 * 1024)
    }
    public static func inspectPNG(_ data: Data, configuration c: GLM5NextConfiguration) throws
        -> Inspection
    {
        try requireConfiguration(c)
        guard data.count > 32, data.count <= maximumEncodedBytes,
            data.prefix(8) == Data([137, 80, 78, 71, 13, 10, 26, 10]),
            let source = CGImageSourceCreateWithData(
                data as CFData, [kCGImageSourceShouldCache: false] as CFDictionary),
            CGImageSourceGetType(source) as String? == "public.png",
            CGImageSourceGetCount(source) == 1,
            let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
            let w = props[kCGImagePropertyPixelWidth] as? NSNumber,
            let h = props[kCGImagePropertyPixelHeight] as? NSNumber,
            w.doubleValue == Double(w.intValue), h.doubleValue == Double(h.intValue),
            w.intValue > 0, h.intValue > 0, w.intValue <= maximumPixels / h.intValue,
            (props[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1 == 1
        else {
            throw GLM5NextVisionError.invalidInput("bounded single-frame PNG required")
        }
        // APNG can expose one frame through some decoders. Reject its animation
        // control chunk structurally, before a pixel decoder runs. A slice keeps
        // its parent's indices, so subscripts add `base` to the payload offset.
        let base = data.startIndex
        var offset = 8
        var ended = false
        while offset <= data.count - 12 {
            let n = data[base + offset ..< base + offset + 4].reduce(0) { ($0 << 8) | Int($1) }
            guard n <= data.count - offset - 12 else {
                throw GLM5NextVisionError.invalidInput("PNG chunk bound")
            }
            let type = String(
                bytes: data[base + offset + 4 ..< base + offset + 8], encoding: .ascii)
            guard type != "acTL", type != "fcTL", type != "fdAT" else {
                throw GLM5NextVisionError.unsupported("animated PNG")
            }
            offset += n + 12
            if type == "IEND" {
                ended = true
                break
            }
        }
        guard ended, offset == data.count else {
            throw GLM5NextVisionError.invalidInput("PNG framing")
        }
        let resize = try GLM5NextMediaGeometry.resizePlan(
            width: w.intValue, height: h.intValue,
            patchSize: c.visionConfig.patchSize, maximumTokens: maximumFeatures)
        guard try resize.grid.featureCount() <= maximumFeatures else {
            throw GLM5NextVisionError.budgetExceeded("cluster image features")
        }
        let patch = try product([
            resize.grid.patchCount(), 6, c.visionConfig.patchSize, c.visionConfig.patchSize, 4,
        ])
        // Source RGBA+RGB, resize source/intermediate/content/canvas and
        // coefficient tables plus patch float array. Bound before decode.
        let host = try sum([
            product([w.intValue, h.intValue, 16]),
            product([resize.width, h.intValue, 6]), product([resize.width, resize.height, 12]),
            patch, patch, 8 * 1024 * 1024,
        ])
        guard host <= 128 * 1024 * 1024 else {
            throw GLM5NextVisionError.budgetExceeded("image host workspace")
        }
        return .init(
            width: w.intValue, height: h.intValue, resize: resize, patchBytes: patch,
            hostWorkingBytes: host)
    }
    public static func decodePNG(
        _ data: Data, inspection: Inspection,
        configuration: GLM5NextConfiguration
    ) throws -> GLM5NextRGBImage {
        let verified = try inspectPNG(data, configuration: configuration)
        guard verified.width == inspection.width, verified.height == inspection.height,
            verified.resize == inspection.resize
        else { throw GLM5NextVisionError.invalidInput("image inspection changed") }
        try Task.checkCancellation()
        guard let image = CIImage(data: data, options: [.applyOrientationProperty: false]),
            image.extent.width == CGFloat(verified.width),
            image.extent.height == CGFloat(verified.height)
        else {
            throw GLM5NextVisionError.invalidInput("PNG pixel decode")
        }
        return try GLM5NextRGBDecoder.render(
            image, maximumBytes: product([verified.width, verified.height, 4]),
            useSoftwareRenderer: true)
    }

    public static func product(_ values: [Int]) throws -> Int {
        try values.reduce(1) { a, b in
            let n = a.multipliedReportingOverflow(by: b)
            guard a > 0, b > 0, !n.overflow else {
                throw GLM5NextVisionError.budgetExceeded("image byte overflow")
            }
            return n.partialValue
        }
    }
    public static func sum(_ values: [Int]) throws -> Int {
        try values.reduce(0) { a, b in
            let n = a.addingReportingOverflow(b)
            guard b >= 0, !n.overflow else {
                throw GLM5NextVisionError.budgetExceeded("image byte overflow")
            }
            return n.partialValue
        }
    }
}
