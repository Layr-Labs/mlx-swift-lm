import Foundation
import ImageIO
import MLXLLM
import MLXLMCommon

/// Allocation-free SDK geometry and codec-header inspection. This path never
/// creates CGImage, raster buffers or MLX arrays. Limits are admission bounds,
/// not a measured memory/quality assertion.
public enum MiMoV26ClusterImagePlan {
    public struct Inspection: Equatable, Sendable {
        public let width, height: Int
        public let geometry: MiMoV26MediaGeometry.Plan
        public let decodeBytes, pixelWorkingBytes, towerWorkingBytes: Int
    }
    public static func settings(configuration c: MiMoV26Configuration) throws
        -> MiMoV26MediaGeometry.Settings
    {
        guard let p = c.processorFields, let v = c.vision,
            p["rope_type"] == .string("rope")
        else { throw MiMoV26MultimodalError.unsupportedProfile("causal image profile") }
        func n(_ k: String) throws -> Int { try MiMoV26MultimodalProfile.integer(p, k) }
        let s = try MiMoV26MediaGeometry.Settings(
            patchSize: n("patch_size"), mergeSize: n("merge_size"),
            temporalPatchSize: n("temporal_patch_size"),
            temporalCompressionRatio: n("temporal_compression_ratio"),
            imageMinPixels: n("image_min_pixels"), imageMaxPixels: n("image_max_pixels"),
            videoMinPixels: n("video_min_pixels"), videoMaxPixels: n("video_max_pixels"),
            videoTotalMaxPixels: n("video_total_max_pixels"))
        guard s.patchSize == v.patchSize, s.mergeSize == v.spatialMergeSize,
            s.temporalPatchSize == v.temporalPatchSize, s.temporalCompressionRatio == 1
        else {
            throw MiMoV26MultimodalError.unsupportedProfile("image tower geometry")
        }
        return s
    }
    public static func inspectPNG(
        _ data: Data, configuration c: MiMoV26Configuration,
        maximumEncodedBytes: Int, maximumPixels: Int
    ) throws -> Inspection {
        guard data.count > 32, data.count <= maximumEncodedBytes,
            data.prefix(8) == Data([137, 80, 78, 71, 13, 10, 26, 10]),
            let source = CGImageSourceCreateWithData(
                data as CFData,
                [kCGImageSourceShouldCache: false] as CFDictionary),
            CGImageSourceGetCount(source) == 1, CGImageSourceGetStatus(source) == .statusComplete,
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
            let w = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
            let h = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue,
            w > 0, h > 0, h <= maximumPixels, w <= maximumPixels / h,
            ((properties[kCGImagePropertyDepth] as? NSNumber)?.intValue ?? 8) <= 8
        else {
            throw MiMoV26EncodedVisualDecoder.Failure.invalidImage
        }
        let orientation = (properties[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
        guard (1 ... 8).contains(orientation), let v = c.vision else {
            throw MiMoV26EncodedVisualDecoder.Failure.invalidImage
        }
        let width = orientation >= 5 ? h : w
        let height = orientation >= 5 ? w : h
        let geometry = try MiMoV26MediaGeometry.image(
            height: height, width: width, settings: settings(configuration: c))
        return try .init(
            width: width, height: height, geometry: geometry,
            decodeBytes: MiMoV26VisualDecodeMemory.image(pixels: w * h).peakBytes,
            pixelWorkingBytes: MiMoV26Pixels.workingByteCount(
                inputElements: 3 * w * h, frameCount: 1, plan: geometry),
            towerWorkingBytes: MiMoV26VisionWorkingSet.frameBytes(
                geometry, configuration: v, stream: .cpu))
    }
    /// Same tokenizer namespace/geometry as native single-Mac preprocessing.
    /// Input is the already normalized template message sequence with ordered
    /// image markers. No media strings or external URLs are rendered as text.
    public static func expand(
        symbolic: [Int], inspections: [Inspection], configuration c: MiMoV26Configuration,
        tokenizer: any Tokenizer, maximumPromptTokens: Int,
        maximumOutputTokens: Int
    ) throws -> (tokens: [Int], spans: [CBv2ImageSpan]) {
        let p = try MiMoV26MultimodalProfile(configuration: c, tokenizer: tokenizer)
        let start = p.tokens["vision_start_token_id"]!
        let pad = p.tokens["image_token_id"]!
        let end = p.tokens["vision_end_token_id"]!
        let reserved = Set(p.tokens.values)
        var out: [Int] = []
        var spans: [CBv2ImageSpan] = []
        var cursor = 0
        guard maximumOutputTokens > 0, maximumOutputTokens < c.maxPositionEmbeddings else {
            throw MiMoV26MultimodalError.limit("image context")
        }
        let cap = min(maximumPromptTokens, c.maxPositionEmbeddings - maximumOutputTokens)
        while cursor < symbolic.count {
            if reserved.contains(symbolic[cursor]) {
                guard spans.count < inspections.count, cursor + 2 < symbolic.count,
                    Array(symbolic[cursor ..< cursor + 3]) == [start, pad, end]
                else {
                    throw MiMoV26MultimodalError.invalidInput("unbound image marker")
                }
                let count = inspections[spans.count].geometry.mediaTokens
                guard count > 0, out.count <= cap - 2, count <= cap - out.count - 2 else {
                    throw MiMoV26MultimodalError.limit("expanded image prompt")
                }
                out.append(start)
                spans.append(.init(tokenOffset: out.count, length: count))
                out.append(contentsOf: repeatElement(pad, count: count))
                out.append(end)
                cursor += 3
            } else {
                guard out.count < cap, (0 ..< c.vocabularySize).contains(symbolic[cursor]) else {
                    throw MiMoV26MultimodalError.limit("image prompt vocabulary/context")
                }
                out.append(symbolic[cursor])
                cursor += 1
            }
        }
        guard spans.count == inspections.count else {
            throw MiMoV26MultimodalError.invalidInput("missing image marker")
        }
        return (out, spans)
    }
}
