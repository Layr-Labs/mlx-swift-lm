import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class Qwen4MixedBatchPositionTests: XCTestCase {
    func testMixedRowsKeepTextAbsenceAndAllMediaPlanes() throws {
        // [axes=3, rows=2, length=2]. The scheduler fills the absent text
        // row to form a rectangle; those values must not change its path.
        let positions = MLXArray(
            [Int32(61), 62, 7, 8, 61, 62, 9, 10, 61, 62, 11, 12], [3, 2, 2])
        try CBv2Qwen4PositionScope.withExplicitRows([false, true]) {
            XCTAssertNil(Qwen4ExpBatchedQSA.positions(positions, row: 0, batch: 2, length: 2))
            let media = try XCTUnwrap(
                Qwen4ExpBatchedQSA.positions(positions, row: 1, batch: 2, length: 2))
            XCTAssertEqual(media.shape, [3, 1, 2])
            XCTAssertEqual(media.asArray(Int32.self), [7, 8, 9, 10, 11, 12])
        }
        // Explicit positions supplied outside an engine scope retain the
        // existing direct-model API contract; no data-based inference.
        XCTAssertNotNil(Qwen4ExpBatchedQSA.positions(positions, row: 0, batch: 2, length: 2))
    }

    func testEqualMediaPlanesAreStillExplicitAfterReorder() throws {
        let positions = MLXArray([Int32(7), 61, 7, 61, 7, 61], [3, 2, 1])
        try CBv2Qwen4PositionScope.withExplicitRows([true, false]) {
            let media = try XCTUnwrap(
                Qwen4ExpBatchedQSA.positions(positions, row: 0, batch: 2, length: 1))
            XCTAssertEqual(media.asArray(Int32.self), [7, 7, 7])
            XCTAssertNil(Qwen4ExpBatchedQSA.positions(positions, row: 1, batch: 2, length: 1))
        }
    }

    func testNestedAndThrowingScopesRestorePreviousBinding() throws {
        enum Probe: Error { case stop }
        XCTAssertNil(CBv2Qwen4PositionScope.explicitPosition(row: 0, batch: 1))
        try CBv2Qwen4PositionScope.withExplicitRows([false, true]) {
            XCTAssertEqual(CBv2Qwen4PositionScope.explicitPosition(row: 0, batch: 2), false)
            XCTAssertThrowsError(try CBv2Qwen4PositionScope.withExplicitRows([true]) {
                XCTAssertEqual(CBv2Qwen4PositionScope.explicitPosition(row: 0, batch: 1), true)
                throw Probe.stop
            })
            XCTAssertEqual(CBv2Qwen4PositionScope.explicitPosition(row: 1, batch: 2), true)
        }
        XCTAssertNil(CBv2Qwen4PositionScope.explicitPosition(row: 0, batch: 1))
    }

    func testConcurrentSynchronousScopesDoNotShareRowKinds() {
        DispatchQueue.concurrentPerform(iterations: 16) { index in
            let expected = index.isMultiple(of: 2)
            CBv2Qwen4PositionScope.withExplicitRows([expected, !expected]) {
                XCTAssertEqual(CBv2Qwen4PositionScope.explicitPosition(row: 0, batch: 2), expected)
                XCTAssertEqual(CBv2Qwen4PositionScope.explicitPosition(row: 1, batch: 2), !expected)
            }
            XCTAssertNil(CBv2Qwen4PositionScope.explicitPosition(row: 0, batch: 1))
        }
    }
}
