import Foundation
import XCTest

@testable import MLXLMCommon

final class CBv2ConfirmedTokenTimingTests: XCTestCase {
    private final class Row {}

    func testRowsBurstsRelativeClockAndReusedIdentity() throws {
        let recorder = CBv2ForwardShapeRecorder()
        try recorder.reset()
        let first = Row()
        let second = Row()
        let step = recorder.beginStep()
        step.confirmTokens(row: ObjectIdentifier(first), firstToken: true, count: 1, nanos: 100)
        step.confirmTokens(row: ObjectIdentifier(second), firstToken: true, count: 1, nanos: 100)
        step.confirmTokens(row: ObjectIdentifier(first), firstToken: false, count: 3, nanos: 200)
        step.confirmTokens(row: ObjectIdentifier(first), firstToken: true, count: 1, nanos: 300)
        step.attach()
        step.complete()
        step.confirmTokens(row: ObjectIdentifier(first), firstToken: false, count: 1, nanos: 400)
        let snapshot = recorder.snapshot()
        let values = try XCTUnwrap(snapshot.confirmedTokenTimings)
        XCTAssertEqual(values.map(\.rowOrdinal), [0, 1, 0, 2])
        XCTAssertEqual(values.map(\.tokenCount), [1, 1, 3, 1])
        XCTAssertEqual(values.map(\.relativeNanos), [0, 0, 100, 200])
        XCTAssertEqual(snapshot.droppedTokenTimings, 0)
        let json = String(decoding: try JSONEncoder().encode(values), as: UTF8.self)
        for forbidden in ["request", "tokenID", "ObjectIdentifier", "text"] {
            XCTAssertFalse(json.contains(forbidden))
        }
        try recorder.reset()
        XCTAssertEqual(recorder.snapshot().confirmedTokenTimings?.count, 0)
        XCTAssertNil(CBv2ForwardShapeSnapshot.disabled.confirmedTokenTimings)
        XCTAssertNil(CBv2ForwardShapeSnapshot.disabled.droppedTokenTimings)
    }

    func testBoundsAndInvalidClocksCannotProduceEligiblePartialHistory() throws {
        let recorder = CBv2ForwardShapeRecorder()
        try recorder.reset()
        let row = Row()
        let step = recorder.beginStep()
        for index in 0 ..< CBv2ConfirmedTokenTimings.maximumReceipts {
            step.confirmTokens(
                row: ObjectIdentifier(row), firstToken: index == 0,
                count: 1, nanos: UInt64(index + 1))
        }
        step.confirmTokens(row: ObjectIdentifier(row), firstToken: false, count: 1, nanos: 100_000)
        step.attach()
        step.complete()
        XCTAssertEqual(
            recorder.snapshot().confirmedTokenTimings?.count,
            CBv2ConfirmedTokenTimings.maximumReceipts)
        XCTAssertEqual(recorder.snapshot().droppedTokenTimings, 1)
        try recorder.reset()
        let invalid = recorder.beginStep()
        invalid.confirmTokens(row: ObjectIdentifier(row), firstToken: true, count: 1, nanos: 10)
        invalid.confirmTokens(row: ObjectIdentifier(row), firstToken: false, count: 1, nanos: 9)
        invalid.confirmTokens(row: ObjectIdentifier(row), firstToken: false, count: 0, nanos: 11)
        invalid.confirmTokens(row: ObjectIdentifier(row), firstToken: false, count: 9, nanos: 11)
        invalid.attach()
        invalid.complete()
        XCTAssertEqual(recorder.snapshot().confirmedTokenTimings?.count, 1)
        XCTAssertEqual(recorder.snapshot().droppedTokenTimings, 3)
    }

    func testExhaustedOrdinalsCannotReuseAnOldRowsHistory() {
        let timings = CBv2ConfirmedTokenTimings()
        let liveRow = Row()
        let reusedAddress = Row()
        timings.record(row: ObjectIdentifier(liveRow), firstToken: true, count: 1, nanos: 1)
        for index in 1 ..< CBv2ConfirmedTokenTimings.maximumRows {
            // Explicit firstToken simulates serial requests allocated at the
            // same address without depending on an allocator reuse heuristic.
            timings.record(
                row: ObjectIdentifier(reusedAddress), firstToken: true,
                count: 1, nanos: UInt64(index + 1))
        }
        timings.record(row: ObjectIdentifier(reusedAddress), firstToken: true, count: 1, nanos: 300)
        timings.record(
            row: ObjectIdentifier(reusedAddress), firstToken: false, count: 3, nanos: 400)
        timings.record(
            row: ObjectIdentifier(reusedAddress), firstToken: false, count: 1, nanos: 500)
        // Exhausting new ordinals must not discard an already-observed live row.
        timings.record(row: ObjectIdentifier(liveRow), firstToken: false, count: 2, nanos: 600)
        XCTAssertEqual(timings.receipts.count, CBv2ConfirmedTokenTimings.maximumRows + 1)
        XCTAssertEqual(timings.receipts.last?.rowOrdinal, 0)
        XCTAssertEqual(timings.receipts.last?.tokenCount, 2)
        XCTAssertEqual(timings.dropped, 3)
    }

    func testInvalidFirstReceiptStillRetiresAReusedIdentity() {
        let timings = CBv2ConfirmedTokenTimings()
        let row = Row()
        timings.record(row: ObjectIdentifier(row), firstToken: true, count: 1, nanos: 10)
        timings.record(row: ObjectIdentifier(row), firstToken: true, count: 1, nanos: 9)
        timings.record(row: ObjectIdentifier(row), firstToken: false, count: 1, nanos: 20)
        XCTAssertEqual(timings.receipts.map(\.rowOrdinal), [0, 1])
        XCTAssertEqual(timings.dropped, 1)
    }
}
