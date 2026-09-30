@preconcurrency import CoreVideo
import CoreGraphics
import Foundation

extension MiMoV26EncodedVisualDecoder {
    static func validateFrame(_ buffer: CVPixelBuffer, plannedPixels: Int, limits: Limits) throws {
        guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA else {
            throw Failure.unsupportedRepresentation
        }
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let pixels = try MiMoV26VisualDecodeMemory.product(width, height)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let bytes = try MiMoV26VisualDecodeMemory.product(stride, height)
        let scratch = try MiMoV26VisualDecodeMemory.sum(
            MiMoV26VisualDecodeMemory.product(pixels, 32), 1 << 20)
        guard width > 0, height > 0, pixels <= plannedPixels, pixels <= limits.maximumPixels,
            stride >= (try MiMoV26VisualDecodeMemory.product(width, 4)), bytes <= scratch,
            try MiMoV26VisualDecodeMemory.sum(bytes, MiMoV26VisualDecodeMemory.product(pixels, 12))
                <= limits.maximumWorkingBytes
        else { throw Failure.limit }
    }

    /// Reads the locked BGRA buffer directly into the retained planar Float RGB.
    /// No Data / CGImage / CFData raster copies survive between reader iterations.
    static func frame(_ buffer: CVPixelBuffer, transform: CGAffineTransform, limits: Limits) throws
        -> MiMoV26Pixels.DecodedRGB
    {
        try validateFrame(buffer, plannedPixels: limits.maximumPixels, limits: limits)
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let pixels = try MiMoV26VisualDecodeMemory.product(width, height)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        // Translation normalizes the origin; only the same eight exact EXIF
        // orientations accepted by the image path are supported.
        let shape = [transform.a, transform.b, transform.c, transform.d]
        let orientations: [[CGFloat]] = [
            [1, 0, 0, 1], [-1, 0, 0, 1], [-1, 0, 0, -1], [1, 0, 0, -1],
            [0, 1, 1, 0], [0, 1, -1, 0], [0, -1, -1, 0], [0, -1, 1, 0],
        ]
        guard let orientation = orientations.firstIndex(of: shape),
            CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess
        else { throw Failure.unsupportedRepresentation }
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let address = CVPixelBufferGetBaseAddress(buffer) else { throw Failure.invalidVideo }
        let pointer = address.assumingMemoryBound(to: UInt8.self)
        let outWidth = orientation >= 4 ? height : width
        let outHeight = orientation >= 4 ? width : height
        var rgb = [Float](repeating: 0, count: try MiMoV26VisualDecodeMemory.product(pixels, 3))
        for y in 0 ..< outHeight {
            try Task.checkCancellation()
            for x in 0 ..< outWidth {
                let sx: Int, sy: Int
                switch orientation {
                case 0: (sx, sy) = (x, y)
                case 1: (sx, sy) = (width - 1 - x, y)
                case 2: (sx, sy) = (width - 1 - x, height - 1 - y)
                case 3: (sx, sy) = (x, height - 1 - y)
                case 4: (sx, sy) = (y, x)
                case 5: (sx, sy) = (y, height - 1 - x)
                case 6: (sx, sy) = (width - 1 - y, height - 1 - x)
                default: (sx, sy) = (width - 1 - y, x)
                }
                let offset = sy * stride + sx * 4
                let destination = y * outWidth + x
                rgb[destination] = Float(pointer[offset + 2])
                rgb[pixels + destination] = Float(pointer[offset + 1])
                rgb[2 * pixels + destination] = Float(pointer[offset])
            }
        }
        return .init(height: outHeight, width: outWidth, planarRGB: rgb)
    }
}
