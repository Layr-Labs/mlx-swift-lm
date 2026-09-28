import Foundation
import MLX
import XCTest
@testable import MLXLMCommon

final class CBv2ContiguousCheckpointCompletionTests: XCTestCase {
    private func fixture() throws -> (CBv2CompleteCheckpointCodec, CBv2ContiguousHistoricalCheckpoint) {
        let chunk = max(32, CBv2AttentionV1.queryBlockSize)
        let kinds: [CBv2LayerKind] = [
            .init(attention: .slidingWindow(17), headDim: 192, valueHeadDim: 128,
                  kvHeads: 1, queryHeads: 2),
        ]
        let codec = CBv2CompleteCheckpointCodec(
            identity: .init(modelAggregateHash: "synthetic", promptContractID: "test",
                            buildID: "test", numericsFingerprint: "float32"),
            layerKinds: kinds, recurrentSpec: nil, kvDTypes: [.float32], assistant: nil,
            admission: .init(layerKinds: kinds, bytesCapacity: 1 << 20,
                             config: .init(watermarkFraction: 0, elementBytes: 4)))
        let row = CBv2WindowedSequenceKV(window: 17, kvHeads: 1, headDim: 192, valueHeadDim: 128)
        let pair = row.update(
            keys: MLXArray(Int32(0)..<Int32(chunk * 192)).asType(.float32).reshaped([1, 1, chunk, 192]),
            values: MLXArray(Int32(0)..<Int32(chunk * 128)).asType(.float32).reshaped([1, 1, chunk, 128]))
        try withError { eval(pair.0, pair.1) }
        return (codec, try .init(codec: codec, position: chunk, chunkSize: chunk, state: [row]))
    }

    func testFailedRequiredCompletionKeepsActualBuffersAndChargeAfterClose() throws {
        var prepared: (CBv2CompleteCheckpointCodec, CBv2ContiguousHistoricalCheckpoint)? = try fixture()
        let codec = prepared!.0
        var capture: CBv2ContiguousHistoricalCheckpoint? = prepared!.1
        prepared = nil
        weak var wrapper = capture
        weak var actualRoot = capture?.evaluationRoots.first
        let charge = codec.admission.bytesReserved
        XCTAssertGreaterThan(charge, 0)
        XCTAssertNotNil(actualRoot)
        capture?.beforeRequiredDrainForTesting = {
            throw MLXError.caught("intentional required-completion failure")
        }
        XCTAssertThrowsError(try capture!.finishEvaluation())
        XCTAssertTrue(capture!.completionFailed)
        XCTAssertNil(capture!.compactAllocationEvidence)
        capture!.close()
        capture!.close()
        capture = nil
        XCTAssertNil(wrapper)
        XCTAssertNotNil(actualRoot, "a byte ledger alone cannot keep in-flight native buffers alive")
        XCTAssertEqual(codec.admission.bytesReserved, charge,
                       "uncertain required completion cannot refund the owner on close")
    }

    func testRemovingFailureInjectionCannotRetryOrRefundFailedCompletion() throws {
        let (codec, capture) = try fixture()
        let charge = codec.admission.bytesReserved
        var drainAttempts = 0
        capture.beforeRequiredDrainForTesting = {
            drainAttempts += 1
            throw MLXError.caught("intentional required-completion failure")
        }
        XCTAssertThrowsError(try capture.finishEvaluation())
        XCTAssertEqual(drainAttempts, 1)
        capture.beforeRequiredDrainForTesting = nil
        XCTAssertThrowsError(try capture.finishEvaluation())
        capture.close()
        XCTAssertTrue(capture.completionFailed)
        XCTAssertEqual(codec.admission.bytesReserved, charge)
    }

    func testUnsubmittedCloseReleasesWithoutInventingCompletion() throws {
        let (codec, capture) = try fixture()
        capture.beforeRequiredDrainForTesting = {
            XCTFail("unsubmitted lazy copies have no submitted native work to drain")
            throw MLXError.caught("unexpected drain")
        }
        XCTAssertGreaterThan(codec.admission.bytesReserved, 0)
        capture.close()
        XCTAssertFalse(capture.completionFailed)
        XCTAssertEqual(codec.admission.bytesReserved, 0)
    }

    func testSuccessfulNativeCompletionStillReleasesNormally() throws {
        let (codec, capture) = try fixture()
        try capture.finishEvaluation()
        XCTAssertNotNil(capture.compactAllocationEvidence)
        XCTAssertFalse(capture.completionFailed)
        capture.close()
        XCTAssertEqual(codec.admission.bytesReserved, 0)
    }
}
