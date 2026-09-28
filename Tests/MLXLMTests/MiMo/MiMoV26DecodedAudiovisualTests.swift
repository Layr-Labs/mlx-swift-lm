import Foundation
import XCTest
@testable import MLXVLM

/// Independent scalar expectations only; no tokenizer, codec, pixels or MLX.
final class MiMoV26DecodedAudiovisualTests: XCTestCase {
    func testInterleaveZeroUsesPatchRateAndKeepsUnusedWholeAudioBacking() throws {
        let value = try MiMoV26AudiovisualLayout.make(timestamps:[0,0.16,0.32],
            segmentEnd:.float32(0.48),temporalPatchSize:2,wholeAudioPatches:4,maximumUnits:2)
        XCTAssertEqual(value.units.map(\.audioRange),[0..<2,2..<3])
        XCTAssertEqual(value.units.map(\.visualGroup),[0,1])
        XCTAssertEqual(value.units.map(\.timestamp),[0,0.32])
        XCTAssertEqual(value.alignedFrames,4); XCTAssertEqual(value.duplicatedFrames,1)
        XCTAssertEqual(value.wholeAudioPatches,4)
        XCTAssertEqual(value.usedAudioPatches,3); XCTAssertEqual(value.unusedAudioPatches,1)
    }
    func testFloat32MultiplyBeforeTruncationIsNotFloat64Promotion() throws {
        let good = try MiMoV26AudiovisualLayout.make(timestamps:[0,0.16,0.32],
            segmentEnd:.float32(0.48),temporalPatchSize:2,wholeAudioPatches:4,maximumUnits:2)
        XCTAssertEqual(good.units[1].audioRange,2..<3)
        // Float32(.48)*6.25 rounds to3. Promoting the stored F32 first
        // yields2.9999999329...: truncation would make the final unit empty.
        XCTAssertThrowsError(try MiMoV26AudiovisualLayout.make(timestamps:[0,0.16,0.32],
            segmentEnd:.float64(Double(Float(0.48))),temporalPatchSize:2,
            wholeAudioPatches:4,maximumUnits:2))
    }
    func testClippedEndIsNotSilencePaddingOrDroppingAnEmptyUnit() throws {
        let clipped = try MiMoV26AudiovisualLayout.make(timestamps:[0,1,2],
            segmentEnd:.float32(3),temporalPatchSize:2,wholeAudioPatches:14,maximumUnits:2)
        XCTAssertEqual(clipped.units.map(\.audioRange),[0..<12,12..<14])
        XCTAssertEqual(clipped.unusedAudioPatches,0)
        XCTAssertThrowsError(try MiMoV26AudiovisualLayout.make(timestamps:[0,1,2],
            segmentEnd:.float32(3),temporalPatchSize:2,wholeAudioPatches:12,maximumUnits:2))
    }
    func testWholeAudioPrefixAndTailAreNotRefundedByARepresentedSlice() throws {
        let value = try MiMoV26AudiovisualLayout.make(timestamps:[1,1.16],
            segmentEnd:.float32(1.32),temporalPatchSize:2,wholeAudioPatches:10,maximumUnits:2)
        XCTAssertEqual(value.units[0].audioRange,6..<8)
        XCTAssertEqual(value.wholeAudioPatches,10)
        XCTAssertEqual(value.usedAudioPatches,2); XCTAssertEqual(value.unusedAudioPatches,8)
    }
    func testSingleFrameAlignsWithoutInventingAnotherAudioUnit() throws {
        let value = try MiMoV26AudiovisualLayout.make(timestamps:[0],
            segmentEnd:.float32(1),temporalPatchSize:2,wholeAudioPatches:1,maximumUnits:1)
        XCTAssertEqual(value.alignedFrames,2); XCTAssertEqual(value.duplicatedFrames,1)
        XCTAssertEqual(value.units.count,1); XCTAssertEqual(value.units[0].audioRange,0..<1)
    }
    func testMalformedAndResourceBoundsRefuseBeforeUnitAllocation() throws {
        for times in [[Float](),[0,0],[1,0],[-1,0],[0,.nan],[0,.infinity]] {
            XCTAssertThrowsError(try MiMoV26AudiovisualLayout.make(timestamps:times,
                segmentEnd:.float32(3),temporalPatchSize:2,wholeAudioPatches:20,maximumUnits:2))
        }
        for end in [MiMoV26DecodedAudiovisual.SegmentEnd.float32(.infinity),.float64(.nan),
                    .float32(0),.float32(0.01)] {
            XCTAssertThrowsError(try MiMoV26AudiovisualLayout.make(timestamps:[0],
                segmentEnd:end,temporalPatchSize:2,wholeAudioPatches:20,maximumUnits:2))
        }
        XCTAssertThrowsError(try MiMoV26AudiovisualLayout.make(timestamps:[0,1,2],
            segmentEnd:.float32(3),temporalPatchSize:2,wholeAudioPatches:20,maximumUnits:1))
        XCTAssertThrowsError(try MiMoV26AudiovisualLayout.make(timestamps:[0],
            segmentEnd:.float32(1),temporalPatchSize:0,wholeAudioPatches:20,maximumUnits:1))
        XCTAssertThrowsError(try MiMoV26AudiovisualLayout.make(timestamps:[0],
            segmentEnd:.float32(1),temporalPatchSize:2,wholeAudioPatches:0,maximumUnits:1))
        let largeFinite = try MiMoV26AudiovisualLayout.make(timestamps:[0],
            segmentEnd:.float64(1e20),temporalPatchSize:2,wholeAudioPatches:4,maximumUnits:1)
        XCTAssertEqual(largeFinite.units[0].audioRange,0..<4)
    }
}
