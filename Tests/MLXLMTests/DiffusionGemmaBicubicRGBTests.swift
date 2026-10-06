import CryptoKit
import Foundation
import Testing
@testable import MLXVLM

@Suite("DiffusionGemma reference RGB8 resampling")
struct DiffusionGemmaBicubicRGBTests {
    // Generated with MLX-VLM e79b0e0's actual image processor, Pillow 12.1.0,
    // NumPy 2.4.1 and the released 280-soft-token image budget. Original
    // top-row-first NCHW Float32 output bytes are hashed, not rounded metrics.
    @Test(arguments: ["rgb-gradient", "gray-steps", "asymmetric-edges"])
    func matchesPinnedReferencePixels(_ pattern: String) throws {
        let width: Int, height: Int, targetWidth: Int, targetHeight: Int, digest: String
        switch pattern {
        case "rgb-gradient":
            (width, height, targetWidth, targetHeight, digest) = (97, 61, 1008, 624,
                "16eb5bc781577ecda588f4286b4495e0954c7aef4bc8ede783790f71d36a2f5c")
        case "gray-steps":
            (width, height, targetWidth, targetHeight, digest) = (96, 96, 768, 768,
                "607fdece411f99efad58593473d559625d6f4d5848fc346e5ea192611b75ea06")
        default:
            (width, height, targetWidth, targetHeight, digest) = (961, 769, 864, 672,
                "46b2242a186d8d81174da1d4b79e5c77078bb693e764951679319858022bcbe9")
        }
        var input = [UInt8]()
        input.reserveCapacity(width * height * 3)
        for y in 0..<height {
            for x in 0..<width {
                if pattern == "rgb-gradient" {
                    input.append(contentsOf: [UInt8(x * 255 / (width - 1)),
                        UInt8(y * 255 / (height - 1)), UInt8((x * 17 + y * 31) % 256)])
                } else if pattern == "gray-steps" {
                    let gray: UInt8 = x < width / 3 ? 0 : (x < 2 * width / 3 ? 128 : 255)
                    input.append(contentsOf: [gray, gray, gray])
                } else {
                    input.append(contentsOf: [x < width / 2 ? 255 : 0,
                        y < height / 3 ? 255 : 0, (x / 13 + y / 17) % 2 == 1 ? 96 : 224])
                }
            }
        }
        let output = try DiffusionGemmaBicubicRGB.resize(input, width: width, height: height,
            targetWidth: targetWidth, targetHeight: targetHeight)
        let plane = targetWidth * targetHeight
        var pixels = [Float](repeating: 0, count: plane * 3)
        for channel in 0..<3 {
            for i in 0..<plane { pixels[channel * plane + i] = Float(output[i * 3 + channel]) * Float(1.0 / 255.0) }
        }
        let actual = pixels.withUnsafeBytes { SHA256.hash(data: Data($0)) }
            .map { String(format: "%02x", $0) }.joined()
        #expect(actual == digest)
    }

    @Test func unchangedSizeAndConstantTinyAxesPreserveRGBBytes() throws {
        let original: [UInt8] = [0, 32, 255, 127, 3, 200]
        #expect(try DiffusionGemmaBicubicRGB.resize(original, width: 2, height: 1,
            targetWidth: 2, targetHeight: 1) == original)
        let color: [UInt8] = [17, 127, 239]
        for (w, h, tw, th) in [(1, 1, 13, 7), (1, 31, 7, 3), (31, 1, 3, 7)] {
            let input = Array(repeating: color, count: w * h).flatMap { $0 }
            let result = try DiffusionGemmaBicubicRGB.resize(input, width: w, height: h,
                targetWidth: tw, targetHeight: th)
            let expected = Array(repeating: color, count: tw * th).flatMap { $0 }
            #expect(result == expected)
        }
    }

    @Test func cancelledResizeDoesNotContinuePixelWork() async {
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try DiffusionGemmaBicubicRGB.resize([17, 127, 239], width: 1, height: 1,
                targetWidth: 13, targetHeight: 7)
        }
        await #expect(throws: CancellationError.self) { _ = try await task.value }
    }

    @Test func invalidOrExcessiveBuffersRejectBeforeAllocation() {
        for (w, h, tw, th) in [(0, 1, 1, 1), (-1, 1, 1, 1), (1, 1, 0, 1),
            (Int.max, 2, 1, 1), (1, 1, Int.max, Int.max), (1, 1, 1, 1)] {
            #expect(throws: (any Error).self) {
                _ = try DiffusionGemmaBicubicRGB.resize([], width: w, height: h,
                    targetWidth: tw, targetHeight: th)
            }
        }
    }
}
