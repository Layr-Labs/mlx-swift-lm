import MLX
import XCTest

@testable import MLXLMCommon

/// Host-only staging witnesses exercise the actual native-frontier commit
/// policy. They contain no arrays and never qualify a codec/native execution;
/// the separate native engine fixtures must prove capture and restoration.
final class CBv2NativeHistoricalRetentionTests: XCTestCase {
    private func fixture() -> CBv2CompleteCheckpointCapture {
        let kinds = [
            CBv2LayerKind(
                attention: .full, headDim: 64,
                valueHeadDim: 32, kvHeads: 1, queryHeads: 1)
        ]
        let store = CompleteCheckpointFixtureStore()
        let codec = CBv2CompleteCheckpointCodec(
            identity: store.identity,
            layerKinds: kinds, recurrentSpec: nil, kvDTypes: [.float16], assistant: nil,
            admission: AdmissionV2(layerKinds: kinds, bytesCapacity: 16 << 20))
        return .init(codec: codec, store: store)
    }

    private func commit(
        _ positions: [Int], hint: Int?, resumedAt: Int = 0,
        capture: CBv2CompleteCheckpointCapture
    ) {
        for position in positions {
            let witness = CBv2CapturedCompleteCheckpoint(
                historical: .init(
                    position: position, chunkSize: 1_024, windows: [:]))
            capture.commitHistorical(
                witness, requestID: .init(1),
                hintTokens: hint, resumedAt: resumedAt)
            XCTAssertLessThanOrEqual(capture.staged[.init(1)]?.count ?? 0, 3)
        }
        capture.queue.sync {}
    }

    private func close(_ capture: CBv2CompleteCheckpointCapture) {
        capture.close()
        for checkpoint in capture.staged.values.flatMap({ $0 }) {
            checkpoint.closeAfterCompletedEvaluation()
        }
        capture.staged.removeAll()
        capture.retentions.removeAll()
        capture.queue.sync {}
        XCTAssertEqual(capture.codec.admission.bytesReserved, 0)
    }

    func testKnownForkSurvivesLaterNativeFrontiers() {
        let capture = fixture()
        defer { close(capture) }
        // No invented 2,300-token state: the deepest ACTUALLY captured
        // frontier under that hint is 2,048.
        commit([1_024, 2_048, 3_072, 4_096], hint: 2_300, capture: capture)
        XCTAssertEqual(capture.staged[.init(1)]?.compactMap(\.position), [1_024, 2_048, 4_096])
        XCTAssertEqual(capture.retentions[.init(1)]?.publication, [4_096, 2_048, 1_024])
    }

    func testNoDemandKeepsExistingFirstLatestPair() {
        for hint in [nil, 0] as [Int?] {
            let capture = fixture()
            defer { close(capture) }
            commit([1_024, 2_048, 3_072, 4_096], hint: hint, capture: capture)
            XCTAssertEqual(capture.staged[.init(1)]?.compactMap(\.position), [1_024, 4_096])
            XCTAssertNil(capture.retentions[.init(1)]?.target)
        }
    }

    func testResumedNativeDonorDoesNotRecaptureAnInteriorFirst() {
        let capture = fixture()
        defer { close(capture) }
        commit([3_072, 4_096, 5_120], hint: 2_048, resumedAt: 2_048, capture: capture)
        XCTAssertEqual(capture.staged[.init(1)]?.compactMap(\.position), [5_120])
        XCTAssertEqual(capture.retentions[.init(1)]?.publication, [5_120])
    }

    func testResumedNativeDonorKeepsOnlyObservedTargetAndLatest() {
        let capture = fixture()
        defer { close(capture) }
        // A refused/absent 4,096 boundary cannot be supplied by a hint.
        commit([3_072, 5_120, 6_144], hint: 4_900, resumedAt: 2_048, capture: capture)
        XCTAssertEqual(capture.staged[.init(1)]?.compactMap(\.position), [3_072, 6_144])
        XCTAssertEqual(capture.retentions[.init(1)]?.publication, [6_144, 3_072])
    }

    func testNativeRetentionShedsFirstThenTargetWithinExistingBytePolicy() {
        var retention = CBv2CheckpointRetention(stride: nil, hintTokens: 2_300)
        let bytes = [1_024: 100, 2_048: 100, 3_072: 100, 4_096: 100]
        XCTAssertEqual(
            retention.commitHistoricalBounded(
                1_024,
                bytesByPosition: bytes, byteBudget: 250), [])
        XCTAssertEqual(
            retention.commitHistoricalBounded(
                2_048,
                bytesByPosition: bytes, byteBudget: 250), [])
        XCTAssertEqual(
            retention.commitHistoricalBounded(
                3_072,
                bytesByPosition: bytes, byteBudget: 250), [1_024])
        XCTAssertEqual(retention.retained, [2_048, 3_072])
        XCTAssertEqual(
            retention.commitHistoricalBounded(
                4_096,
                bytesByPosition: bytes, byteBudget: 150), [2_048, 3_072])
        XCTAssertEqual(retention.retained, [4_096])
        XCTAssertFalse(
            retention.firstIsOpen,
            "shedding must not reopen an adopter/cold first role")
    }

    func testContiguousSlotCapRefusesBeforeCopyAndDoesNotChangeTheLedger() throws {
        let capture = fixture()
        defer { close(capture) }
        let footprint = try CBv2ContiguousHistoricalCheckpoint.reservationFootprint(
            codec: capture.codec, position: 2_048)
        XCTAssertEqual(
            footprint.reservedBytes, (64 << 10) + 512,
            "a full-row capture counts its existing host backing, without copying full K/V")
        capture.historicalSlotStagedByteCapOverride = footprint.reservedBytes - 1
        capture.makeContiguousCheckpoint = { _, _, _, _ in
            XCTFail("a refused slot-cap candidate must not construct a native copy")
            throw CBv2CompleteCheckpointError.allocationFailed
        }
        XCTAssertNil(try capture.prepareHistorical(position: 2_048, chunkSize: 1_024, state: []))
        XCTAssertEqual(capture.inFlightHistoricalBytes, 0)
        XCTAssertEqual(capture.stagedHistoricalBytes, 0)
        XCTAssertEqual(capture.codec.admission.bytesReserved, 0)
    }
}
