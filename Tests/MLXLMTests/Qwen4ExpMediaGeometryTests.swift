import XCTest
@testable import MLXVLM

final class Qwen4ExpMediaGeometryTests: XCTestCase {
    // Goldens evaluated from the actual smart_resize AST in Transformers
    // v5.8.0 (049d2bf1), without importing or reimplementing its function.
    func testPublishedImageBudgetAndReferenceRounding() throws {
        let cases = [(1120,2240,1120,2240), (224,448,224,448),
                     (48,80,224,352), (64,96,224,320), (1920,1080,1920,1088),
                     (4096,4096,4096,4096), (32,6400,32,6400), (33,6401,32,6400),
                     (1,2,192,384)]
        for (h,w,expectedH,expectedW) in cases {
            let size = try Qwen4ExpMediaGeometry.image(height: h, width: w, factor: 32,
                                                      minPixels: 65536, maxPixels: 16777216)
            XCTAssertEqual(size, .init(height: expectedH, width: expectedW))
        }
        // Python round is ties-to-even, unlike Swift's default round().
        let tie = try Qwen4ExpMediaGeometry.image(height: 48, width: 80, factor: 32,
                                                minPixels: 1, maxPixels: 16777216)
        XCTAssertEqual(tie, .init(height: 64, width: 64))
    }

    func testReferenceTemporalResizeIncludingOddFrameCounts() throws {
        let cases = [(1,1120,2240,1120,2240), (3,1120,2240,1120,2240),
                     (8,1120,2240,1120,2240), (17,1120,2240,832,1696),
                     (33,1120,2240,608,1216), (17,1920,1080,1600,896),
                     (33,1920,1080,1152,640), (1,4096,4096,4992,4992),
                     (3,4096,4096,2880,2880), (8,4096,4096,1760,1760),
                     (17,4096,4096,1216,1216), (33,4096,4096,864,864)]
        for (frames,h,w,expectedH,expectedW) in cases {
            let size = try Qwen4ExpMediaGeometry.video(frames: frames, height: h, width: w,
                temporalFactor: 2, factor: 32, minPixels: 4096, maxPixels: 25165824)
            XCTAssertEqual(size, .init(height: expectedH, width: expectedW))
        }
        // These are geometry results, not promises of admitted GPU capacity.
        // The reference's one-frame ceil/padding behavior is preserved too.
    }

    func testReferenceSamplingAndTemporalTimestamps() throws {
        XCTAssertEqual(try Qwen4ExpMediaGeometry.sampleIndices(totalFrames: 8, sourceFPS: 2), Array(0..<8))
        XCTAssertEqual(try Qwen4ExpMediaGeometry.sampleIndices(totalFrames: 20, sourceFPS: 4),
                       [0,2,4,6,8,11,13,15,17,19])
        XCTAssertEqual(try Qwen4ExpMediaGeometry.sampleIndices(totalFrames: 4, sourceFPS: 30), [0,1,2,3])
        XCTAssertEqual(try Qwen4ExpMediaGeometry.sampleIndices(totalFrames: 1, sourceFPS: 30), [0])
        let longClip = try Qwen4ExpMediaGeometry.sampleIndices(totalFrames: 24000, sourceFPS: 24)
        XCTAssertEqual(longClip.count, 768)
        XCTAssertEqual(longClip.first, 0)
        XCTAssertEqual(longClip.last, 23999)
        XCTAssertEqual(try Qwen4ExpMediaGeometry.timestamps(indices: [0,1,2,3,4], sourceFPS: 2,
                                                          temporalFactor: 2), [0.25,1.25,2.0])
    }

    func testMalformedGeometryFailsBeforeAllocation() throws {
        for (h,w,factor,minP,maxP) in [(0,64,32,1,1024), (64,64,0,1,1024),
                                      (32,6401,32,1,1024), (64,64,32,2048,1024),
                                      (Int.max,Int.max,32,1,1024)] {
            XCTAssertThrowsError(try Qwen4ExpMediaGeometry.image(height: h, width: w,
                factor: factor, minPixels: minP, maxPixels: maxP))
        }
        XCTAssertThrowsError(try Qwen4ExpMediaGeometry.video(frames: 8, height: 1, width: 2,
            temporalFactor: 2, factor: 32, minPixels: 4096, maxPixels: 25165824))
        XCTAssertThrowsError(try Qwen4ExpMediaGeometry.sampleIndices(totalFrames: 1, sourceFPS: 0))
        XCTAssertThrowsError(try Qwen4ExpMediaGeometry.sampleIndices(totalFrames: Int.max, sourceFPS: 24))
        XCTAssertThrowsError(try Qwen4ExpMediaGeometry.timestamps(indices: [], sourceFPS: 2, temporalFactor: 2))
        XCTAssertThrowsError(try Qwen4ExpMediaGeometry.timestamps(indices: [1], sourceFPS: .leastNonzeroMagnitude,
                                                               temporalFactor: 2))
    }
}
