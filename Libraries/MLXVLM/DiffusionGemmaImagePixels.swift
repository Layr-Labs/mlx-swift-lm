import CoreImage
import Foundation
import MLX
import MLXLLM

private let diffusionImageContext = CIContext(options: [.cacheIntermediates: false])

/// The released processor resizes RGB8 in its encoded color space, rounds and
/// clips each separable pass, then scales to FP32. A floating CoreImage resize
/// in linear light is not the same operation even when both are called bicubic.
enum DiffusionGemmaImagePixels {
    static func prepare(_ image: CIImage, width: Int, height: Int,
        geometry: DiffusionGemmaMediaGeometry) throws -> MLXArray
    {
        do {
            let count = try DiffusionGemmaBicubicRGB.checkedPixelCount(width, height)
            _ = try DiffusionGemmaBicubicRGB.checkedPixelCount(geometry.width, geometry.height)
            try Task.checkCancellation()
            var rgba = [UInt8](repeating: 0, count: count * 4)
            defer { diffusionImageContext.clearCaches() }
            try rgba.withUnsafeMutableBytes { bytes in
                let destination = CIRenderDestination(bitmapData: bytes.baseAddress!,
                    width: width, height: height, bytesPerRow: width * 4, format: .RGBA8)
                destination.colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
                destination.alphaMode = .unpremultiplied
                // RGB reference buffers store their first row at the top.
                // CIRenderDestination otherwise defaults to Cartesian order.
                destination.isFlipped = true
                destination.isDithered = false
                destination.isClamped = true
                let task = try diffusionImageContext.startTask(toRender: image,
                    from: image.extent, to: destination, at: .zero)
                _ = try task.waitUntilCompleted()
            }
            try Task.checkCancellation()
            var rgb = [UInt8]()
            rgb.reserveCapacity(count * 3)
            for pixel in 0..<count {
                if pixel.isMultiple(of: 16384) { try Task.checkCancellation() }
                rgb.append(contentsOf: rgba[pixel * 4..<(pixel * 4 + 3)])
            }
            let resized = try DiffusionGemmaBicubicRGB.resize(rgb,
                width: width, height: height, targetWidth: geometry.width, targetHeight: geometry.height)
            try Task.checkCancellation()
            let plane = geometry.width * geometry.height
            var values = [Float](repeating: 0, count: plane * 3)
            let scale = Float(1.0 / 255.0)
            for channel in 0..<3 {
                for pixel in 0..<plane {
                    if pixel.isMultiple(of: 16384) { try Task.checkCancellation() }
                    values[channel * plane + pixel] = Float(resized[pixel * 3 + channel]) * scale
                }
            }
            return MLXArray(values).reshaped(1, 3, geometry.height, geometry.width)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            // CoreImage failures may carry input-specific details. Preserve a
            // recoverable native error without exposing arbitrary descriptions.
            throw DiffusionGemmaModelError.invalidInput("native image pixel preparation")
        }
    }
}
