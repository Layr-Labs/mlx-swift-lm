// Copyright © 2026 Eigen Labs.
//
// Inkling 2 image preprocessing. The reference processor never resamples an
// image: it pads the RGB raster with -1 (on the 0-255 scale) up to a multiple
// of 40 px per side and cuts one 40x40 patch per language-model image token,
// normalized to [-1, 1] with mean = std = 0.5.
//
// Reference: transformers PR 51050 `image_processing_inkling2.py`
// (`Inkling2ImageProcessor._preprocess`, `convert_to_rgb`), vLLM PR 69028
// `processing.py` (`Inkling2ImageProcessor`, `_composite_on_checkerboard`).

import CoreGraphics
import CoreImage
import Foundation
import MLX
import MLXLMCommon

/// Token-grid geometry of one image. Images up to `maximumSide` px per side
/// keep their exact pixels (the reference path); a larger image is first
/// scaled down so its longer side is `maximumSide` (outside the model card's
/// 40-4096 px range, where no reference rule exists).
public struct Inkling2ImageLayout: Sendable, Equatable {
    /// Pixels per side of one image token.
    public static let tokenSide = 40
    /// Longest side the model card recommends.
    public static let maximumSide = 4096

    public let width: Int
    public let height: Int

    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
    }

    public var rows: Int { (height + Self.tokenSide - 1) / Self.tokenSide }
    public var columns: Int { (width + Self.tokenSide - 1) / Self.tokenSide }
    public var tokenCount: Int { rows * columns }
    public var grid: THW { THW(1, rows, columns) }

    /// The size an image of `width x height` is processed at: unchanged when
    /// both sides fit `bound`, otherwise scaled (aspect kept, sides rounded,
    /// at least 1 px) so the longer side fits.
    public static func processedSize(
        width: Int, height: Int, bound: (width: Int, height: Int)? = nil
    ) -> (width: Int, height: Int) {
        let limitWidth = min(bound?.width ?? maximumSide, maximumSide)
        let limitHeight = min(bound?.height ?? maximumSide, maximumSide)
        guard width > limitWidth || height > limitHeight else { return (width, height) }
        let scale = min(Double(limitWidth) / Double(width), Double(limitHeight) / Double(height))
        return (
            min(limitWidth, max(1, Int((Double(width) * scale).rounded()))),
            min(limitHeight, max(1, Int((Double(height) * scale).rounded())))
        )
    }
}

/// Raw 8-bit RGB pixels as PIL decodes them: no color management, and any
/// transparency composited onto the reference 8 px gray checkerboard.
public struct Inkling2RGBImage: Sendable {
    public let width: Int
    public let height: Int
    /// Row-major `[height, width, 3]`.
    public let bytes: [UInt8]

    public init(width: Int, height: Int, bytes: [UInt8]) {
        precondition(bytes.count == width * height * 3)
        self.width = width
        self.height = height
        self.bytes = bytes
    }

    /// Straight-alpha RGBA8 composited onto the checkerboard exactly as
    /// `Inkling2ImageProcessor.convert_to_rgb`:
    /// `(rgb * a + bg * (255 - a) + 127) / 255`, bg 204 or 153 per 8 px cell.
    public init(width: Int, height: Int, straightRGBA: [UInt8]) {
        precondition(straightRGBA.count == width * height * 4)
        var bytes = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let source = (y * width + x) * 4
                let target = (y * width + x) * 3
                let alpha = Int(straightRGBA[source + 3])
                let background = ((y / 8) + (x / 8)) % 2 == 0 ? 204 : 153
                for channel in 0 ..< 3 {
                    let value = Int(straightRGBA[source + channel])
                    bytes[target + channel] = UInt8(
                        (value * alpha + background * (255 - alpha) + 127) / 255)
                }
            }
        }
        self.init(width: width, height: height, bytes: bytes)
    }

    #if compiler(>=6.2)
    private static let context = CIContext(options: [
        .workingColorSpace: NSNull(), .workingFormat: CIFormat.RGBAf,
        .cacheIntermediates: false,
    ])
    #else
    nonisolated(unsafe) private static let context = CIContext(options: [
        .workingColorSpace: NSNull(), .workingFormat: CIFormat.RGBAf,
        .cacheIntermediates: false,
    ])
    #endif

    /// Renders `image` without color matching in float RGBA, then recovers the
    /// stored 8-bit values (Core Image premultiplies alpha internally).
    public init(_ image: CIImage) throws {
        let extent = image.extent.integral
        guard extent.width >= 1, extent.height >= 1, extent.width.isFinite,
            extent.height.isFinite, extent.width <= 65_536, extent.height <= 65_536
        else {
            throw VLMError.imageProcessingFailure("Inkling 2 cannot read an image of this size")
        }
        let width = Int(extent.width)
        let height = Int(extent.height)
        var floats = [Float](repeating: 0, count: width * height * 4)
        floats.withUnsafeMutableBytes { pointer in
            Self.context.render(
                image, toBitmap: pointer.baseAddress!, rowBytes: width * 16, bounds: extent,
                format: .RGBAf, colorSpace: nil)
        }
        Self.context.clearCaches()
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        func byte(_ value: Float) -> UInt8 {
            UInt8(min(max((value * 255).rounded(), 0), 255))
        }
        for index in 0 ..< width * height {
            let alpha = min(max(floats[index * 4 + 3], 0), 1)
            let alphaByte = byte(alpha)
            rgba[index * 4 + 3] = alphaByte
            guard alphaByte > 0 else { continue }
            for channel in 0 ..< 3 {
                rgba[index * 4 + channel] = byte(floats[index * 4 + channel] / alpha)
            }
        }
        self.init(width: width, height: height, straightRGBA: rgba)
    }
}

public enum Inkling2ImagePatches {
    /// Normalized value of every 0-255 byte, plus the -1 padding value at
    /// index 256, with the reference arithmetic: the torchvision backend fuses
    /// rescale into normalize (`mean / rescale`, `std / rescale` in float32)
    /// and computes `(x - mean') / std'` in float32.
    public static func normalizationTable(
        rescaleFactor: Double = 1 / 255, mean: [Float] = [0.5, 0.5, 0.5],
        std: [Float] = [0.5, 0.5, 0.5]
    ) -> [[Float]] {
        let inverse = Float(1 / rescaleFactor)
        return (0 ..< 3).map { channel in
            let fusedMean = mean[channel] * inverse
            let fusedStd = std[channel] * inverse
            return (0 ... 256).map { value in
                let raw: Float = value == 256 ? -1 : Float(value)
                return (raw - fusedMean) / fusedStd
            }
        }
    }

    /// `[rows * cols, 40, 40, 3]` float32 patches of one image in raster token
    /// order, padded on the bottom and right with the -1 value.
    public static func patches(
        _ image: Inkling2RGBImage, table: [[Float]] = normalizationTable()
    ) -> (pixels: MLXArray, grid: THW) {
        let layout = Inkling2ImageLayout(width: image.width, height: image.height)
        let side = Inkling2ImageLayout.tokenSide
        let rows = layout.rows
        let columns = layout.columns
        var values = [Float](repeating: 0, count: rows * columns * side * side * 3)
        // Flat [channel * 257 + value] lookup; index 256 is the padding value.
        let lookup = table.flatMap { $0 }
        image.bytes.withUnsafeBufferPointer { source in
            lookup.withUnsafeBufferPointer { lookup in
                values.withUnsafeMutableBufferPointer { target in
                    var cursor = 0
                    for row in 0 ..< rows {
                        for column in 0 ..< columns {
                            for py in 0 ..< side {
                                let y = row * side + py
                                for px in 0 ..< side {
                                    let x = column * side + px
                                    let inside = y < image.height && x < image.width
                                    let offset = (y * image.width + x) * 3
                                    for channel in 0 ..< 3 {
                                        let value = inside ? Int(source[offset + channel]) : 256
                                        target[cursor + channel] = lookup[channel * 257 + value]
                                    }
                                    cursor += 3
                                }
                            }
                        }
                    }
                }
            }
        }
        return (MLXArray(values, [rows * columns, side, side, 3]), layout.grid)
    }
}
