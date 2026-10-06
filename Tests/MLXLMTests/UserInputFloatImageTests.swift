import CoreImage
import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Regression tests for `UserInput.Image.array(_:).asCIImage()` with a float
/// array: the scaled array stayed float32, and the conversion gave its float
/// bytes to Core Image as RGBA8 bytes, so the pixels were wrong.
///
/// `floatArrayInTheUnitRange` and the helpers are copied from
/// `UserInputImageArrayTests` of PR #237, without the known issue. The
/// helpers are private to this file.
@Suite
struct UserInputFloatImageTests {

    /// A float array in 0 ... 1 is scaled to 0 ... 255 and must give the
    /// same pixels as the equal `uint8` array.
    @Test func floatArrayInTheUnitRange() throws {
        let array = image(height: 2, width: 2, pixel: [1, 1, 1], dtype: .float32)
        let result = try UserInput.Image.array(array).asCIImage()
        #expect(result.extent == CGRect(x: 0, y: 0, width: 2, height: 2))
        let difference = maxDifference(rgba(result), pixel: [255, 255, 255, 255])
        #expect(difference <= 1, "float pixels must convert to white")
    }

    /// A float array with values above 1 is not scaled. Its values are the
    /// bytes of the pixels.
    @Test func floatArrayInTheByteRange() throws {
        let array = image(height: 2, width: 3, pixel: [10, 200, 30], dtype: .float32)
        let result = try UserInput.Image.array(array).asCIImage()
        #expect(result.extent == CGRect(x: 0, y: 0, width: 3, height: 2))
        let difference = maxDifference(rgba(result), pixel: [10, 200, 30, 255])
        #expect(difference <= 1, "float pixels in 0 ... 255 keep their values")
    }

    /// A float value above 255 or below 0 is clipped to the byte range, as
    /// in the Pixtral reference (`np.clip(array, 0, 255)`), and does not wrap
    /// around in the `uint8` conversion.
    @Test func floatArrayOutOfTheByteRangeIsClipped() throws {
        let array = image(height: 2, width: 2, pixel: [300, -20, 128], dtype: .float32)
        let result = try UserInput.Image.array(array).asCIImage()
        #expect(result.extent == CGRect(x: 0, y: 0, width: 2, height: 2))
        let difference = maxDifference(rgba(result), pixel: [255, 0, 128, 255])
        #expect(difference <= 1, "float pixels out of 0 ... 255 must clip to 0 and 255")
    }

    @Test(arguments: [false, true])
    func signedByteRGBGetsAnOpaqueAlpha(channelsFirst: Bool) throws {
        var array = image(height: 2, width: 5, pixel: [10, 80, 30], dtype: .int8)
        if channelsFirst {
            array = array.transposed(2, 0, 1)
        }
        let result = try UserInput.Image.array(array).asCIImage()
        #expect(result.extent == CGRect(x: 0, y: 0, width: 5, height: 2))
        #expect(maxDifference(rgba(result), pixel: [10, 80, 30, 255]) <= 1)
    }

    @Test(arguments: [false, true])
    func signedByteRGBAKeepsItsAlpha(channelsFirst: Bool) throws {
        var array = image(height: 2, width: 5, pixel: [10, 80, 30, 100], dtype: .int8)
        if channelsFirst {
            array = array.transposed(2, 0, 1)
        }
        let result = try UserInput.Image.array(array).asCIImage()
        #expect(result.extent == CGRect(x: 0, y: 0, width: 5, height: 2))
        #expect(maxDifference(rgba(result), pixel: [10, 80, 30, 100]) <= 1)
    }

    @Test func signedByteUnitRangeIsScaledBeforePadding() throws {
        let array = image(height: 2, width: 5, pixel: [0, 1, 1], dtype: .int8)
        let result = try UserInput.Image.array(array).asCIImage()
        #expect(maxDifference(rgba(result), pixel: [0, 255, 255, 255]) <= 1)
    }

    @Test func signedByteNegativeComponentsAreClipped() throws {
        let array = image(height: 2, width: 5, pixel: [-10, 80, 30], dtype: .int8)
        let result = try UserInput.Image.array(array).asCIImage()
        #expect(maxDifference(rgba(result), pixel: [0, 80, 30, 255]) <= 1)
    }

    /// Renders `image` to RGBA bytes without color management, so the bytes
    /// of the input come back unchanged.
    private func rgba(_ image: CIImage) -> [UInt8] {
        let width = Int(image.extent.width)
        let height = Int(image.extent.height)
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let context = CIContext(options: [
            .workingColorSpace: NSNull(), .useSoftwareRenderer: true,
        ])
        context.render(
            image, toBitmap: &bytes, rowBytes: width * 4, bounds: image.extent,
            format: .RGBA8, colorSpace: nil)
        return bytes
    }

    /// The largest difference between the bytes and the pixel values. The
    /// tolerance of 1 allows for rounding in Core Image.
    private func maxDifference(_ bytes: [UInt8], pixel: [UInt8]) -> Int {
        bytes.enumerated().map { abs(Int($0.element) - Int(pixel[$0.offset % 4])) }.max() ?? 0
    }

    /// An `[H, W, C]` array of `pixel` with the given type.
    private func image(height: Int, width: Int, pixel: [Float], dtype: DType) -> MLXArray {
        let values = (0 ..< height * width).flatMap { _ in pixel }
        return MLXArray(values).reshaped(height, width, pixel.count).asType(dtype)
    }
}
