// Copyright © 2026 Eigen Labs.
import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import Testing

@testable import MLXVLM

extension UnitTests {
    @Suite struct GLM5NextClusterImageDecodeTests {
        private func png(width: Int, height: Int, rgba: [UInt8]) throws -> Data {
            let provider = try #require(CGDataProvider(data: Data(rgba) as CFData))
            let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
            let image = try #require(
                CGImage(
                    width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                    bytesPerRow: width * 4, space: space,
                    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                    provider: provider, decode: nil, shouldInterpolate: false,
                    intent: .defaultIntent))
            let data = NSMutableData()
            let destination = try #require(
                CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil))
            CGImageDestinationAddImage(destination, image, nil)
            try #require(CGImageDestinationFinalize(destination))
            return data as Data
        }

        private func decode(_ data: Data) throws -> GLM5NextRGBImage {
            let configuration = try GLM5NextInventoryFixture.configuration()
            let inspection = try GLM5NextClusterImagePlan.inspectPNG(
                data, configuration: configuration)
            return try GLM5NextClusterImagePlan.decodePNG(
                data, inspection: inspection, configuration: configuration)
        }

        @Test func pngDecodePreservesTopRowsAndUprightGlyph() throws {
            // Blue top, red bottom, and an upright F distinguish both axes.
            let rows = ["BBBBB", ".FFF.", ".F...", ".FF..", "RRRRR"]
            let colors: [Character: [UInt8]] = [
                "B": [64, 128, 192], "R": [220, 30, 80],
                "F": [255, 255, 255], ".": [0, 0, 0],
            ]
            let rgb: [UInt8] = rows.joined().flatMap { colors[$0]! }
            let rgba: [UInt8] = rows.joined().flatMap { colors[$0]! + [UInt8(255)] }
            let data = try png(width: 5, height: 5, rgba: rgba)
            let image = try decode(data)
            #expect(image.width == 5 && image.height == 5)
            #expect(image.pixels == rgb)
            // Exercise the actual CPU resize used before normalization/patchify.
            let resized = try GLM5NextRGB8Bicubic.resize(
                image.pixels, width: 5, height: 5, targetWidth: 16, targetHeight: 16)
            let expected = try GLM5NextRGB8Bicubic.resize(
                rgb, width: 5, height: 5, targetWidth: 16, targetHeight: 16)
            #expect(resized == expected)
        }

        @Test func pngDecodePreservesUnpremultipliedLowAlphaColors() throws {
            let data = try png(
                width: 3, height: 1,
                rgba: [64, 128, 192, 1, 64, 128, 192, 128, 64, 128, 192, 255])
            let image = try decode(data)
            #expect(image.pixels == [64, 128, 192, 64, 128, 192, 64, 128, 192])
            let nativeImage = try #require(CIImage(data: data))
            let native = try GLM5NextRGBDecoder.render(
                nativeImage, maximumBytes: 12, useSoftwareRenderer: true)
            #expect(image.pixels == native.pixels)
            #expect(throws: GLM5NextVisionError.budgetExceeded("decoded image")) {
                try GLM5NextRGBDecoder.render(
                    nativeImage, maximumBytes: 11, useSoftwareRenderer: true)
            }
        }

        @Test func decodeRetainsInspectionAndEncodedBounds() throws {
            let data = try png(width: 1, height: 1, rgba: [64, 128, 192, 255])
            let configuration = try GLM5NextInventoryFixture.configuration()
            let inspection = try GLM5NextClusterImagePlan.inspectPNG(
                data, configuration: configuration)
            let other = try png(
                width: 2, height: 1, rgba: [64, 128, 192, 255, 64, 128, 192, 255])
            #expect(throws: GLM5NextVisionError.self) {
                try GLM5NextClusterImagePlan.decodePNG(
                    other, inspection: inspection, configuration: configuration)
            }
            #expect(throws: GLM5NextVisionError.self) {
                try GLM5NextClusterImagePlan.inspectPNG(
                    Data(repeating: 0, count: GLM5NextClusterImagePlan.maximumEncodedBytes + 1),
                    configuration: configuration)
            }
            #expect(throws: GLM5NextVisionError.self) {
                try decode(Data(data.dropLast()))
            }
        }

        @Test func cancelledDecodeDoesNotReturnPixels() async throws {
            let data = try png(width: 1, height: 1, rgba: [64, 128, 192, 255])
            let task = Task {
                withUnsafeCurrentTask { $0?.cancel() }
                do {
                    _ = try decode(data)
                    return false
                } catch is CancellationError {
                    return true
                } catch {
                    return false
                }
            }
            #expect(await task.value)
        }
    }
}
