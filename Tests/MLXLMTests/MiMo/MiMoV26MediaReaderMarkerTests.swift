import CoreMedia
import Foundation
import XCTest
@testable import MLXVLM

/// Real Core Media buffers exercise marker admission; no decoder is mocked.
final class MiMoV26MediaReaderMarkerTests: XCTestCase {
    private func marker(ready: Bool = true, payloadBytes: Int = 0) throws -> CMSampleBuffer {
        var block: CMBlockBuffer?
        if payloadBytes > 0 {
            XCTAssertEqual(CMBlockBufferCreateWithMemoryBlock(allocator: kCFAllocatorDefault,
                memoryBlock: nil, blockLength: payloadBytes, blockAllocator: kCFAllocatorDefault,
                customBlockSource: nil, offsetToData: 0, dataLength: payloadBytes,
                flags: 0, blockBufferOut: &block), kCMBlockBufferNoErr)
        }
        var sample: CMSampleBuffer?
        XCTAssertEqual(CMSampleBufferCreate(allocator: kCFAllocatorDefault, dataBuffer: block,
            dataReady: ready, makeDataReadyCallback: nil, refcon: nil, formatDescription: nil,
            sampleCount: 0, sampleTimingEntryCount: 0, sampleTimingArray: nil,
            sampleSizeEntryCount: 0, sampleSizeArray: nil, sampleBufferOut: &sample), noErr)
        return try XCTUnwrap(sample)
    }

    func testActualEmptyMarkersStayBoundedForVideoAndAudio() throws {
        let sample = try marker()
        XCTAssertTrue(CMSampleBufferIsValid(sample))
        XCTAssertTrue(CMSampleBufferDataIsReady(sample))
        XCTAssertEqual(CMSampleBufferGetNumSamples(sample), 0)
        XCTAssertEqual(CMSampleBufferGetDuration(sample), .zero)
        var videoMarkers = 0
        XCTAssertTrue(try MiMoV26EncodedVisualDecoder.consumeEmptyMarker(
            sample, count: &videoMarkers, limit: 1))
        XCTAssertEqual(videoMarkers, 1)
        XCTAssertThrowsError(try MiMoV26EncodedVisualDecoder.consumeEmptyMarker(
            sample, count: &videoMarkers, limit: 1)) {
            XCTAssertEqual($0 as? MiMoV26EncodedVisualDecoder.Failure, .limit)
        }
        XCTAssertEqual(videoMarkers, 1)
        var audioMarkers = 0
        XCTAssertTrue(try MiMoV26EncodedAudiovisualDecoder.consumeEmptyMarker(
            sample, count: &audioMarkers, limit: 1))
        XCTAssertThrowsError(try MiMoV26EncodedAudiovisualDecoder.consumeEmptyMarker(
            sample, count: &audioMarkers, limit: 1)) {
            XCTAssertEqual($0 as? MiMoV26EncodedAudiovisualDecoder.Failure, .limit)
        }
    }

    func testInvalidUnreadyPayloadAndRetimedMarkersRemainTypedRefusals() throws {
        let invalid = try marker()
        XCTAssertEqual(CMSampleBufferInvalidate(invalid), noErr)
        let unready = try marker(ready: false)
        XCTAssertFalse(CMSampleBufferDataIsReady(unready))
        let payload = try marker(payloadBytes: 1)
        XCTAssertEqual(CMBlockBufferGetDataLength(try XCTUnwrap(CMSampleBufferGetDataBuffer(payload))), 1)
        let retimed = try marker()
        CMSetAttachment(retimed, key: kCMSampleBufferAttachmentKey_SpeedMultiplier,
            value: NSNumber(value: 2), attachmentMode: kCMAttachmentMode_ShouldPropagate)
        for sample in [invalid, unready, payload, retimed] {
            var videoMarkers = 0
            XCTAssertThrowsError(try MiMoV26EncodedVisualDecoder.consumeEmptyMarker(
                sample, count: &videoMarkers, limit: 10)) {
                XCTAssertEqual($0 as? MiMoV26EncodedVisualDecoder.Failure, .invalidVideo)
            }
            XCTAssertEqual(videoMarkers, 0)
            var audioMarkers = 0
            XCTAssertThrowsError(try MiMoV26EncodedAudiovisualDecoder.consumeEmptyMarker(
                sample, count: &audioMarkers, limit: 10)) {
                XCTAssertEqual($0 as? MiMoV26EncodedAudiovisualDecoder.Failure, .unsupportedTimeline)
            }
            XCTAssertEqual(audioMarkers, 0)
        }
    }
}
