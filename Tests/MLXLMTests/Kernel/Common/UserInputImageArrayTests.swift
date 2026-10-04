import CoreImage
import Foundation
import MLX
import Testing

@testable import MLXLMCommon

extension KernelTests {

    /// Tests of `UserInput.Image.array(_:).asCIImage()`. The conversion
    /// accepts an `[H, W, C]` or `[C, H, W]` array with 3 or 4 channels,
    /// scales values in 0 ... 1 to 0 ... 255 and adds an opaque alpha
    /// channel to RGB input.
    @Suite
    struct UserInputImageArrayTests {

        /// Renders `image` to RGBA bytes without color management, so the
        /// bytes of the input come back unchanged.
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

        /// The largest difference between the bytes and the pixel values.
        /// The tolerance of 1 allows for rounding in Core Image.
        private func maxDifference(_ bytes: [UInt8], pixel: [UInt8]) -> Int {
            bytes.enumerated().map { abs(Int($0.element) - Int(pixel[$0.offset % 4])) }.max()
                ?? 0
        }

        /// An `[H, W, 3]` array of `pixel` with the given type.
        private func image(height: Int, width: Int, pixel: [Float], dtype: DType) -> MLXArray {
            let values = (0 ..< height * width).flatMap { _ in pixel }
            return MLXArray(values).reshaped(height, width, pixel.count).asType(dtype)
        }

        @Test func uint8ChannelsLastGetsAnOpaqueAlpha() throws {
            let array = image(height: 2, width: 3, pixel: [10, 200, 30], dtype: .uint8)
            let result = try UserInput.Image.array(array).asCIImage()
            #expect(result.extent == CGRect(x: 0, y: 0, width: 3, height: 2))
            #expect(maxDifference(rgba(result), pixel: [10, 200, 30, 255]) <= 1)
        }

        @Test func uint8ChannelsFirstIsTransposed() throws {
            // [3, H, W] with H = 2 and W = 5.
            let planes = MLXArray(
                (0 ..< 3).flatMap { channel in
                    Array(repeating: Float([40, 80, 120][channel]), count: 10)
                }
            ).reshaped(3, 2, 5).asType(.uint8)
            let result = try UserInput.Image.array(planes).asCIImage()
            #expect(result.extent == CGRect(x: 0, y: 0, width: 5, height: 2))
            #expect(maxDifference(rgba(result), pixel: [40, 80, 120, 255]) <= 1)
        }

        @Test func uint8FourChannelsAreKept() throws {
            let array = image(height: 2, width: 2, pixel: [1, 2, 3, 255], dtype: .uint8)
            let result = try UserInput.Image.array(array).asCIImage()
            #expect(result.extent == CGRect(x: 0, y: 0, width: 2, height: 2))
            #expect(maxDifference(rgba(result), pixel: [1, 2, 3, 255]) <= 1)
        }

        @Test func arrayWithoutThreeDimensionsThrows() {
            let array = MLXArray([UInt8](repeating: 1, count: 12)).reshaped(3, 4)
            do {
                _ = try UserInput.Image.array(array).asCIImage()
                Issue.record("expected an error")
            } catch UserInputError.arrayError(let message) {
                #expect(message == "array must have 3 dimensions: 2")
            } catch {
                Issue.record("unexpected error \(error)")
            }
        }

        @Test func arrayWithTwoChannelsThrows() {
            let array = MLXArray([UInt8](repeating: 9, count: 8)).reshaped(2, 2, 2)
            do {
                _ = try UserInput.Image.array(array).asCIImage()
                Issue.record("expected an error")
            } catch UserInputError.arrayError(let message) {
                #expect(message == "channel dimension must be last and 3/4: [2, 2, 2]")
            } catch {
                Issue.record("unexpected error \(error)")
            }
        }

        /// A float array in 0 ... 1 is scaled and converted to RGBA8 bytes.
        @Test func floatArrayInTheUnitRange() throws {
            let array = image(height: 2, width: 2, pixel: [1, 1, 1], dtype: .float32)
            let result = try UserInput.Image.array(array).asCIImage()
            #expect(result.extent == CGRect(x: 0, y: 0, width: 2, height: 2))
            let difference = maxDifference(rgba(result), pixel: [255, 255, 255, 255])
            #expect(difference <= 1, "float pixels must convert to white")
        }
    }
}
