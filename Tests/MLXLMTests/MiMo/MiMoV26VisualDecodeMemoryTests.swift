import CoreGraphics
import CoreVideo
import XCTest

@testable import MLXVLM

final class MiMoV26VisualDecodeMemoryTests: XCTestCase {
    func testVideoChargesSampledRGBAndOneTransientRatherThanEverySourceFrame() throws {
        let pixels = 1920 * 1080
        let short = try MiMoV26VisualDecodeMemory.video(
            encodedBytes: 4096, sourceFrames: 30, sampledFrames: 20,
            pixels: pixels, maximumControlMarkers: 4096)
        let long = try MiMoV26VisualDecodeMemory.video(
            encodedBytes: 4096, sourceFrames: 300, sampledFrames: 20,
            pixels: pixels, maximumControlMarkers: 4096)
        XCTAssertEqual(try long.peakBytes - short.peakBytes, (300 - 30) * 64)
        XCTAssertEqual(long.transientBytes, pixels * 32 + (1 << 20))
        XCTAssertEqual(
            long.retainedBytes, 4096 + 20 * pixels * 12 + 300 * 64 + 4096 * 64 + (1 << 20))
        XCTAssertLessThan(try long.peakBytes, 550 << 20)
        XCTAssertGreaterThan(300 * pixels * 32, 18 << 30)
    }

    func testImageCopiesAndOverflowAreBounded() throws {
        let value = try MiMoV26VisualDecodeMemory.image(pixels: 3840 * 2160)
        XCTAssertEqual(value.retainedBytes, 3840 * 2160 * 12)
        XCTAssertEqual(try value.peakBytes, 3840 * 2160 * 32 + (1 << 20))
        XCTAssertThrowsError(try MiMoV26VisualDecodeMemory.image(pixels: Int.max))
        XCTAssertThrowsError(try MiMoV26VisualDecodeMemory.image(pixels: 0))
        XCTAssertThrowsError(
            try MiMoV26VisualDecodeMemory.video(
                encodedBytes: 1, sourceFrames: Int.max, sampledFrames: 2,
                pixels: 1, maximumControlMarkers: 0))
    }

    func testPixelWorkloadQuoteMatchesPreparationRegardlessOfCeiling() throws {
        let settings = try MiMoV26MediaGeometry.Settings(
            patchSize: 2, mergeSize: 2, temporalPatchSize: 2, temporalCompressionRatio: 1,
            imageMinPixels: 16, imageMaxPixels: 256, videoMinPixels: 16,
            videoMaxPixels: 256, videoTotalMaxPixels: 1024)
        let frame = MiMoV26Pixels.DecodedRGB(
            height: 4, width: 4,
            planarRGB: [Float](repeating: 128, count: 48))
        for ceiling in [1 << 20, 230 << 30] {
            let result = try MiMoV26Pixels.image(
                frame, settings: settings,
                limits: .init(
                    maximumInputElements: 100000, maximumOutputElements: 100000,
                    maximumWorkingBytes: ceiling))
            let quote = try MiMoV26Pixels.workingByteCount(
                inputElements: 48, frameCount: 1, plan: result.geometry)
            XCTAssertEqual(quote, result.plannedWorkingBytes)
            XCTAssertLessThan(quote, 4096)
        }
        let video = try MiMoV26MediaGeometry.video(
            height: 16, width: 16,
            sampledFrames: 20, settings: settings)
        let perFrame = video.gridH * video.gridW
        let score = try MiMoV26VisionWorkingSet.scoreBytes(video, queryHeads: 2)
        let expected = perFrame * perFrame * 2 * 16
        XCTAssertEqual(score, expected)
        let oldQuote = video.patchCount * video.patchCount * 2 * 16
        XCTAssertEqual(oldQuote, score * video.gridT * video.gridT)
    }

    func testDirectPaddedBGRAFramesPreserveChannelsAndAllEightOrientations() throws {
        var value: CVPixelBuffer?
        XCTAssertEqual(
            CVPixelBufferCreate(
                kCFAllocatorDefault, 2, 3, kCVPixelFormatType_32BGRA,
                [kCVPixelBufferBytesPerRowAlignmentKey: 64] as CFDictionary, &value),
            kCVReturnSuccess)
        let buffer = try XCTUnwrap(value)
        XCTAssertEqual(CVPixelBufferLockBaseAddress(buffer, []), kCVReturnSuccess)
        let pointer = try XCTUnwrap(CVPixelBufferGetBaseAddress(buffer)).assumingMemoryBound(
            to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        for y in 0 ..< 3 {
            for x in 0 ..< 2 {
                let n = UInt8(y * 2 + x + 1)
                let offset = y * stride + x * 4
                pointer[offset] = n + 80
                pointer[offset + 1] = n + 40
                pointer[offset + 2] = n
                pointer[offset + 3] = 255
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        let shapes: [[CGFloat]] = [
            [1, 0, 0, 1], [-1, 0, 0, 1], [-1, 0, 0, -1], [1, 0, 0, -1],
            [0, 1, 1, 0], [0, 1, -1, 0], [0, -1, -1, 0], [0, -1, 1, 0],
        ]
        let expected: [[Float]] = [
            [1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5], [6, 5, 4, 3, 2, 1], [5, 6, 3, 4, 1, 2],
            [1, 3, 5, 2, 4, 6], [5, 3, 1, 6, 4, 2], [6, 4, 2, 5, 3, 1], [2, 4, 6, 1, 3, 5],
        ]
        let limits = MiMoV26EncodedVisualDecoder.Limits(
            maximumPixels: 6,
            maximumWorkingBytes: 2 << 20, maximumSourceFrames: 2, maximumSampledFrames: 2)
        for (i, s) in shapes.enumerated() {
            let rgb = try MiMoV26EncodedVisualDecoder.frame(
                buffer,
                transform: .init(a: s[0], b: s[1], c: s[2], d: s[3], tx: 0, ty: 0), limits: limits)
            XCTAssertEqual(rgb.width, i >= 4 ? 3 : 2)
            XCTAssertEqual(rgb.height, i >= 4 ? 2 : 3)
            XCTAssertEqual(
                rgb.planarRGB,
                expected[i] + expected[i].map { $0 + 40 } + expected[i].map { $0 + 80 })
        }
        XCTAssertThrowsError(
            try MiMoV26EncodedVisualDecoder.validateFrame(buffer, plannedPixels: 5, limits: limits))
        XCTAssertThrowsError(
            try MiMoV26EncodedVisualDecoder.frame(
                buffer, transform: .identity,
                limits: .init(
                    maximumPixels: 6, maximumWorkingBytes: 6 * 12,
                    maximumSourceFrames: 2, maximumSampledFrames: 2)))
    }
}
