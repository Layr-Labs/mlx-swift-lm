// Copyright © 2026 Eigen Labs.

import CoreImage
import Foundation
import MLX
import MLXLMCommon

/// Video geometry of the reference processor (`MuseGlimmerVideoProcessor`
/// with the checkpoint's processor config): frames sampled at 2 fps, at most
/// 96, paired into temporal groups of two; every frame resized to one grid
/// of at most 144 merged tokens. CPU only.
public enum MuseGlimmerVideoGeometry {
    public static let samplingFPS = 2.0
    public static let maximumFrames = 96
    public static let temporalPatchSize = MuseGlimmerImageGeometry.temporalPatchSize
    public static let maximumGroups = maximumFrames / temporalPatchSize
    /// Merged tokens of one temporal group (`max_video_frame_tokens`).
    public static let maximumTokensPerGroup = 144
    /// 6,912: the most `<|video|>` tokens one video adds.
    public static let maximumVideoTokens = maximumGroups * maximumTokensPerGroup
    /// The reference's source rate when the container reports none.
    public static let fallbackSourceFPS = 24.0

    /// One video: `groups` temporal groups, each encoded as one image of `frame`.
    public struct Grid: Sendable, Equatable {
        public let groups: Int
        public let frame: MuseGlimmerImageGeometry.Grid

        public init(groups: Int, frame: MuseGlimmerImageGeometry.Grid) throws {
            guard (1 ... maximumGroups).contains(groups),
                frame.featureRows <= maximumTokensPerGroup
            else {
                throw MuseGlimmerImageError.geometry(
                    "video of \(groups) groups of \(frame.featureRows) tokens")
            }
            self.groups = groups
            self.frame = frame
        }

        /// `<|video|>` tokens of one group, and its span length.
        public var tokensPerGroup: Int { frame.featureRows }
        /// Feature rows of the whole video.
        public var featureRows: Int { groups * tokensPerGroup }
        public var thw: THW { THW(groups, frame.patchRows, frame.patchColumns) }
    }

    /// The source frame indices the reference samples (`sample_frames`):
    /// min(total × 2 / source fps, 96, total) frames, rounded down to a
    /// multiple of 2 but at least 2, capped at `totalFrames`, at
    /// `linspace(0, total − 1, count)` truncated, in float32 as torch computes it.
    public static func sampledFrameIndices(totalFrames: Int, sourceFPS: Double?) throws -> [Int] {
        guard totalFrames >= 1, totalFrames <= Int(Int32.max) else {
            throw MuseGlimmerImageError.geometry("video of \(totalFrames) frames")
        }
        let fps = sourceFPS ?? fallbackSourceFPS
        guard fps.isFinite, fps > 0 else {
            throw MuseGlimmerImageError.geometry("video source rate \(fps)")
        }
        let byRate = Double(totalFrames) * samplingFPS / fps
        var count = min(byRate >= Double(maximumFrames) ? maximumFrames : Int(byRate), totalFrames)
        count = max(temporalPatchSize, (count / temporalPatchSize) * temporalPatchSize)
        count = min(count, totalFrames)
        return linspaceTruncated(end: totalFrames - 1, count: count)
    }

    /// torch's CPU `linspace(0, end, count)` in float32 (the first half
    /// counted up from the start, the second half down from the end), then `.long()`.
    static func linspaceTruncated(end: Int, count: Int) -> [Int] {
        guard count > 1 else { return [0] }
        let start: Float = 0
        let stop = Float(end)
        let step = (stop - start) / Float(count - 1)
        let halfway = count / 2
        return (0 ..< count).map { i in
            let value = i < halfway ? start + step * Float(i) : stop - step * Float(count - i - 1)
            return Int(value)
        }
    }

    /// The grid of a video of `frames` sampled frames whose frames are
    /// `sourceHeight x sourceWidth`: the image path's `smart_resize` with at
    /// most 144 merged tokens, one group per two frames (an odd count is
    /// padded with the last frame).
    public static func grid(frames: Int, sourceHeight: Int, sourceWidth: Int) throws -> Grid {
        guard (1 ... maximumFrames).contains(frames) else {
            throw MuseGlimmerImageError.geometry("video of \(frames) sampled frames")
        }
        let frame = try MuseGlimmerImageGeometry.grid(
            sourceHeight: sourceHeight, sourceWidth: sourceWidth,
            maximumImageTokens: maximumTokensPerGroup)
        return try Grid(groups: (frames + temporalPatchSize - 1) / temporalPatchSize, frame: frame)
    }

    /// One timestamp per group, as vLLM writes them: group × 2 / 2 fps.
    public static func timestamps(groups: Int) -> [Double] {
        (0 ..< groups).map { Double($0 * temporalPatchSize) / samplingFPS }
    }

    /// HF's timestamps, kept so tests show the difference: the source times
    /// of every second sampled frame (`indices / source fps`), padded with the last.
    public static func huggingFaceTimestamps(
        sampledIndices: [Int], sourceFPS: Double?, groups: Int
    ) -> [Double] {
        let fps = sourceFPS ?? fallbackSourceFPS
        var times = stride(from: 0, to: sampledIndices.count, by: temporalPatchSize).map {
            Double(sampledIndices[$0]) / fps
        }
        times = Array(times.prefix(groups))
        while times.count < groups { times.append(times.last ?? 0) }
        return times
    }

    /// `"Time: {timestamp:.1f}s"`.
    public static func timestampText(_ seconds: Double) -> String {
        String(format: "Time: %.1fs", seconds)
    }
}

/// Expansion of a prompt holding images and videos, in placeholder order.
public enum MuseGlimmerMediaPrompt {
    public static let videoStartTokenID = 200082
    public static let videoEndTokenID = 200083
    public static let frameSeparatorTokenID = 200087

    public enum Item: Sendable, Equatable {
        case image(MuseGlimmerImageGeometry.Grid)
        case video(MuseGlimmerVideoGeometry.Grid)

        /// Spans (and feature arrays) the item contributes.
        public var spanCount: Int {
            switch self {
            case .image: 1
            case .video(let grid): grid.groups
            }
        }
    }

    /// Replaces the i-th placeholder (`<|patch|>` for an image, `<|video|>`
    /// for a video) with `media[i]`'s tokens. An image is framed as in
    /// ``MuseGlimmerImagePrompt``. A video is `<|vid_start|>`, then for each
    /// group the tokens of `"Time: …s"` (vLLM's timestamps, tokenized by
    /// `timestampTokens` without special tokens) and its `<|video|>` run, then
    /// `<|vid_frame_separator|>`, the last group ending with `<|vid_end|>`.
    /// One span per image and per video group, in prompt order.
    public static func expand(
        promptTokens: [Int], media: [Item], timestampTokens: (String) throws -> [Int]
    ) throws -> MuseGlimmerImagePrompt.Expanded {
        let image = MuseGlimmerImagePrompt.imageTokenID
        let video = MuseGlimmerImagePrompt.videoTokenID
        let placeholders = promptTokens.filter { $0 == image || $0 == video }
        guard placeholders.count == media.count, !media.isEmpty else {
            throw MuseGlimmerImageError.placeholders(
                "\(placeholders.count) media placeholders for \(media.count) items")
        }
        for (index, (token, item)) in zip(placeholders, media).enumerated() {
            switch (token, item) {
            case (image, .image), (video, .video): continue
            default:
                throw MuseGlimmerImageError.placeholders(
                    "placeholder \(index) is not its item's kind")
            }
        }
        var tokens = [Int]()
        var spans = [CBv2ImageSpan]()
        var next = 0
        for token in promptTokens {
            guard token == image || token == video else {
                tokens.append(token)
                continue
            }
            switch media[next] {
            case .image(let grid):
                tokens.append(MuseGlimmerImagePrompt.imageStartTokenID)
                spans.append(CBv2ImageSpan(tokenOffset: tokens.count, length: grid.featureRows))
                tokens.append(contentsOf: repeatElement(image, count: grid.featureRows))
                tokens.append(MuseGlimmerImagePrompt.imageEndTokenID)
            case .video(let grid):
                tokens.append(videoStartTokenID)
                for (group, seconds) in MuseGlimmerVideoGeometry.timestamps(groups: grid.groups)
                    .enumerated()
                {
                    let text = try timestampTokens(MuseGlimmerVideoGeometry.timestampText(seconds))
                    guard !text.contains(where: { $0 == image || $0 == video }) else {
                        throw MuseGlimmerImageError.placeholders("timestamp tokenizes to media")
                    }
                    tokens.append(contentsOf: text)
                    spans.append(
                        CBv2ImageSpan(tokenOffset: tokens.count, length: grid.tokensPerGroup))
                    tokens.append(contentsOf: repeatElement(video, count: grid.tokensPerGroup))
                    tokens.append(group < grid.groups - 1 ? frameSeparatorTokenID : videoEndTokenID)
                }
            }
            next += 1
        }
        return MuseGlimmerImagePrompt.Expanded(tokens: tokens, spans: spans)
    }
}

extension MuseGlimmerImageEncoder {
    /// The reference's preprocessing of sampled frames (in order, one size):
    /// each frame as an image is preprocessed to the video grid, frames
    /// paired into groups (the last repeated for an odd count) and each
    /// group patchified `(temporal, channel, y, x)`. Returns one evaluated
    /// float32 `[patchRows * patchColumns, 1176]` array per group.
    public static func preprocessVideo(frames: [CIImage]) throws -> (
        groups: [MLXArray], grid: MuseGlimmerVideoGeometry.Grid
    ) {
        guard let first = frames.first else {
            throw MuseGlimmerImageError.geometry("video without frames")
        }
        let height = Int(first.extent.height.rounded())
        let width = Int(first.extent.width.rounded())
        guard
            frames.allSatisfy({
                Int($0.extent.height.rounded()) == height && Int($0.extent.width.rounded()) == width
            })
        else { throw MuseGlimmerImageError.geometry("video frames of different sizes") }
        let grid = try MuseGlimmerVideoGeometry.grid(
            frames: frames.count, sourceHeight: height, sourceWidth: width)
        var groups = [MLXArray]()
        for group in 0 ..< grid.groups {
            try Task.checkCancellation()
            let a = frames[2 * group]
            let b = 2 * group + 1 < frames.count ? frames[2 * group + 1] : a
            let pixels = museGlimmerPatchifyPair(
                normalizedPixels(a, grid: grid.frame), normalizedPixels(b, grid: grid.frame),
                grid: grid.frame)
            try MLX.withError { errors in
                eval(pixels)
                try errors.check()
            }
            groups.append(pixels)
        }
        return (groups, grid)
    }

    /// One evaluated `[tokensPerGroup, featureWidth]` array per group, in
    /// order. Each group is encoded on its own, as vLLM does (the reference
    /// encodes temporal groups independently), with a cancellation check
    /// before each.
    public func videoFeatures(groups: [MLXArray], grid: MuseGlimmerVideoGeometry.Grid) throws
        -> [MLXArray]
    {
        guard groups.count == grid.groups else {
            throw MuseGlimmerImageError.geometry("\(groups.count) groups for \(grid.groups)")
        }
        return try groups.map { try features(pixels: $0, grid: grid.frame) }
    }
}

/// Two frames' `[1, 3, H, W]` pixels as one temporal group: each patch row
/// holds frame `a` then frame `b`, `(temporal, channel, y, x)`, like
/// `museGlimmerPatchify` with a second frame instead of a copy.
func museGlimmerPatchifyPair(
    _ a: MLXArray, _ b: MLXArray, grid: MuseGlimmerImageGeometry.Grid
) -> MLXArray {
    let p = MuseGlimmerImageGeometry.patchSize
    func patches(_ x: MLXArray) -> MLXArray {
        x.reshaped(x.dim(-3), grid.patchRows, p, grid.patchColumns, p).transposed(1, 3, 0, 2, 4)
    }
    return stacked([patches(a), patches(b)], axis: 2).reshaped(
        grid.patchRows * grid.patchColumns, MuseGlimmerImageGeometry.patchValues)
}
