import Foundation
import MLXVLM
import XCTest

/// No MLX arrays or model execution: these exercise the Foundation decoded-RGB
/// component. Package compilation and Swift execution are separate validation steps.
final class MiMoV26PixelsTests: XCTestCase {
    private let limits = MiMoV26Pixels.Limits(maximumInputElements: 1_000_000,
        maximumOutputElements: 1_000_000, maximumWorkingBytes: 32 * 1024 * 1024)

    private func settings() throws -> MiMoV26MediaGeometry.Settings {
        try .init(patchSize: 2, mergeSize: 2, temporalPatchSize: 2, temporalCompressionRatio: 1,
            imageMinPixels: 16, imageMaxPixels: 256, videoMinPixels: 16,
            videoMaxPixels: 256, videoTotalMaxPixels: 1024)
    }
    private func frame(_ value: Float, height: Int = 4, width: Int = 4) -> MiMoV26Pixels.DecodedRGB {
        .init(height: height, width: width, planarRGB: [Float](repeating: value, count: 3 * height * width))
    }

    func testIndependentTorchCPUReference() throws {
        guard let path = ProcessInfo.processInfo.environment["MIMO_V26_PIXELS_ORACLE"] else {
            throw XCTSkip("Set MIMO_V26_PIXELS_ORACLE for the installed-Torch CPU fixture corpus")
        }
        let corpus = try JSONDecoder().decode(Corpus.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        XCTAssertEqual(corpus.schemaVersion, 1)
        XCTAssertEqual(corpus.cases.count, 12)
        XCTAssertEqual(corpus.comparisonAbsoluteTolerance, 3e-5)
        XCTAssertEqual(Set(corpus.cases.map(\.id)).count, corpus.cases.count)
        for fixture in corpus.cases {
            let frames = fixture.frames.map {
                MiMoV26Pixels.DecodedRGB(height: fixture.height, width: fixture.width, planarRGB: $0)
            }
            let result: MiMoV26Pixels.Prepared
            if fixture.kind == "image" {
                XCTAssertEqual(frames.count, 1)
                result = try MiMoV26Pixels.image(XCTUnwrap(frames.first), settings: fixture.settings, limits: limits)
            } else {
                XCTAssertEqual(fixture.kind, "video")
                result = try MiMoV26Pixels.video(frames: frames, sampledFrameCount: frames.count,
                                                settings: fixture.settings, limits: limits)
            }
            XCTAssertEqual([result.geometry.height, result.geometry.width], [fixture.outputHeight, fixture.outputWidth], fixture.id)
            XCTAssertEqual([result.geometry.gridT, result.geometry.gridH, result.geometry.gridW], fixture.grid, fixture.id)
            XCTAssertEqual([result.geometry.patchCount, result.geometry.patchVectorSize], fixture.patchShape, fixture.id)
            XCTAssertEqual(result.geometry.alignedFrames, fixture.alignedFrames, fixture.id)
            XCTAssertEqual(result.patchValues.count, fixture.expected.count, fixture.id)
            for (value, expected) in zip(result.patchValues, fixture.expected) {
                XCTAssertTrue(value.isFinite, fixture.id)
                // Explicit installed-Torch CPU scalar diagnostic tolerance.
                // This is not a bit-exact CUDA/Metal or full-model claim.
                XCTAssertEqual(value, expected, accuracy: corpus.comparisonAbsoluteTolerance, fixture.id)
            }
            if let expectedBits = fixture.expectedBits {
                // Identity-sized fixtures preserve the measured CPU arithmetic
                // exactly; resized cases retain the stated tolerance above.
                XCTAssertEqual(result.patchValues.map(\.bitPattern), expectedBits, fixture.id)
            }
            XCTAssertLessThanOrEqual(result.plannedWorkingBytes, limits.maximumWorkingBytes)
        }
    }

    func testRGBUnitsAndChannelNormalization() throws {
        let pixels = [Float](repeating: 0, count: 16) + [Float](repeating: 128, count: 16)
            + [Float](repeating: 255, count: 16)
        let result = try MiMoV26Pixels.image(.init(height: 4, width: 4, planarRGB: pixels),
                                            settings: settings(), limits: limits)
        XCTAssertEqual(result.patchValues[0], (Float(0) - 123.675) / 58.395)
        XCTAssertEqual(result.patchValues[8], (Float(128) - 116.28) / 57.12)
        XCTAssertEqual(result.patchValues[16], (Float(255) - 103.53) / 57.375)
        XCTAssertGreaterThan(abs(result.patchValues[16] - (Float(1) - 103.53) / 57.375), 4)
    }

    func testTemporalAndSpatialPatchOrder() throws {
        let video = try MiMoV26Pixels.video(frames: [frame(10), frame(20), frame(30)], sampledFrameCount: 3,
                                            settings: settings(), limits: limits)
        XCTAssertEqual(video.geometry.alignedFrames, 4)
        XCTAssertEqual(video.geometry.duplicatedFrames, 1)
        XCTAssertEqual(video.patchValues[0], (Float(10) - 123.675) / 58.395)
        XCTAssertEqual(video.patchValues[4], (Float(20) - 123.675) / 58.395)
        XCTAssertEqual(video.patchValues[96], (Float(30) - 123.675) / 58.395)
        XCTAssertEqual(video.patchValues[100], video.patchValues[96])
        var ramp: [Float] = []
        for channel in 0..<3 {
            for row in 0..<4 {
                for column in 0..<4 {
                    ramp.append(Float(channel * 50 + row * 10 + column))
                }
            }
        }
        let image = try MiMoV26Pixels.image(.init(height: 4, width: 4, planarRGB: ramp),
                                           settings: settings(), limits: limits)
        for (index, pixel) in [0: Float(0), 1: 1, 2: 10, 3: 11, 24: 2, 48: 20, 72: 22] {
            XCTAssertEqual(image.patchValues[index], (pixel - 123.675) / 58.395)
        }
    }

    func testInvalidDecodedInputsAndFrameCountReject() throws {
        let native = try settings()
        for value in [Float.nan, Float.infinity, -1, 256] {
            var pixels = [Float](repeating: 0, count: 48)
            pixels[17] = value
            XCTAssertThrowsError(try MiMoV26Pixels.image(.init(height: 4, width: 4, planarRGB: pixels),
                                                       settings: native, limits: limits))
        }
        for input in [MiMoV26Pixels.DecodedRGB(height: 4, width: 4, planarRGB: [0]),
                      .init(height: 0, width: 4, planarRGB: []),
                      .init(height: Int.max, width: 4, planarRGB: [])] {
            XCTAssertThrowsError(try MiMoV26Pixels.image(input, settings: native, limits: limits))
        }
        XCTAssertThrowsError(try MiMoV26Pixels.video(frames: [], sampledFrameCount: 0, settings: native, limits: limits))
        XCTAssertThrowsError(try MiMoV26Pixels.video(frames: [frame(10)], sampledFrameCount: 2, settings: native, limits: limits))
        XCTAssertThrowsError(try MiMoV26Pixels.video(frames: [frame(10), frame(20, height: 4, width: 8)],
            sampledFrameCount: 2, settings: native, limits: limits))
    }

    func testExplicitResourceLimitsCoverInputsOutputAndScratch() throws {
        let native = try settings(), input = frame(20)
        let measured = try MiMoV26Pixels.image(input, settings: native, limits: limits)
        XCTAssertEqual(measured.inputElementCount, 48)
        XCTAssertGreaterThan(measured.plannedWorkingBytes, (48 + measured.patchValues.count) * MemoryLayout<Float>.stride)
        let exact = MiMoV26Pixels.Limits(maximumInputElements: 48, maximumOutputElements: measured.patchValues.count,
                                        maximumWorkingBytes: measured.plannedWorkingBytes)
        XCTAssertEqual(try MiMoV26Pixels.image(input, settings: native, limits: exact).patchValues, measured.patchValues)
        for restricted in [
            MiMoV26Pixels.Limits(maximumInputElements: 47, maximumOutputElements: 1000, maximumWorkingBytes: 100000),
            .init(maximumInputElements: 1000, maximumOutputElements: measured.patchValues.count - 1, maximumWorkingBytes: 100000),
            .init(maximumInputElements: 1000, maximumOutputElements: 1000, maximumWorkingBytes: measured.plannedWorkingBytes - 1),
            .init(maximumInputElements: 0, maximumOutputElements: 1000, maximumWorkingBytes: 100000),
        ] {
            XCTAssertThrowsError(try MiMoV26Pixels.image(input, settings: native, limits: restricted))
        }
    }

    private struct Corpus: Decodable {
        let schemaVersion: Int
        let comparisonAbsoluteTolerance: Float
        let cases: [Fixture]
    }
    private struct Fixture: Decodable {
        let id, kind: String
        let height, width: Int
        let frames: [[Float]]
        let settings: MiMoV26MediaGeometry.Settings
        let outputHeight, outputWidth, alignedFrames: Int
        let grid, patchShape: [Int]
        let expected: [Float]
        let expectedBits: [UInt32]?
    }
}
