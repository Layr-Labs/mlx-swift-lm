// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Genuine tiny MiMo target/three-head module and real Common capture buffers.
/// These component tests do NOT mint a native-engine contract or a load receipt.
/// Provider's separate strict-load interior-publication selector covers that seam.
final class MiMoV26HistoricalCaptureIntegrationTests: XCTestCase {
    private enum Failure: Error { case injected }
    private struct Fixture {
        let target: MiMoV26TextModel
        let assistant: MiMoV26MTPAssistant
        let adapter: MiMoV26CBv2Adapter
        let backend: MiMoV26CBv2Backend
        let caches: [any CBv2AttendingLayerCache]
        let rows: [CBv2SequenceKV?]
        let driver: CBv2MTPRoundDriver
        let state: MiMoV26MTPState
        let codec: CBv2CompleteCheckpointCodec
        let capture: CBv2ContiguousHistoricalCheckpoint
        let tokens: [Int]
    }
    private func fixture() throws -> Fixture {
        var object =
            try JSONSerialization.jsonObject(
                with: JSONEncoder().encode(MiMoV26MTPChecks.config())) as! [String: Any]
        object["dtype"] = "bfloat16"
        object["moe_router_dtype"] = "bfloat16"
        // Actual native asymmetric layout accepted by Common, not a different
        // row geometry attached to a tiny4/2 target. Hidden/vocab remain4/32.
        for key in ["head_dim", "swa_head_dim"] { object[key] = 64 }
        for key in ["v_head_dim", "swa_v_head_dim"] { object[key] = 32 }
        let config = try JSONDecoder().decode(
            MiMoV26Configuration.self,
            from: JSONSerialization.data(withJSONObject: object))
        let target = try MiMoV26TextModel(config)
        try target.update(
            parameters: .unflattened(
                MiMoV26MTPChecks.fixtureWeights(target).mapValues { $0.asType(.bfloat16) }),
            verify: .all)
        let predictor = try MiMoV26MTP(target: target)
        try predictor.loadConvertedWeights(
            MiMoV26MTPChecks.fixtureWeights(predictor, prefix: "mtp.").mapValues {
                $0.asType(.bfloat16)
            })
        let assistant = try MiMoV26MTPAssistant(target: target, predictor: predictor)
        let adapter = try MiMoV26CBv2Adapter(target: target, assistant: assistant)
        let construction = NativeConstructionScope()
        defer {
            if construction.snapshot.isRetainedFault { _ = Unmanaged.passRetained(construction) }
        }
        _ = try adapter.probeNativeKVTypes(retaining: construction)
        let backend = try adapter.makeBackend(bytesCapacity: 16 << 20)
        let rows = try backend.makeSequenceState(
            layerKinds: adapter.layerKinds, promptLength: 17, maxLength: 64)
        let caches = adapter.makeCaches()
        try adapter.bindRows([rows], caches: caches)
        let driver = try XCTUnwrap(
            CBv2MTPRoundDriver.build(
                model: adapter, drafter: assistant,
                config: .init(
                    enabled: true, maxDraftTokens: 3, maxSpeculativeBatch: 1,
                    fixedDraftTokens: 3, verificationMode: .serialTarget)))
        let tokens = (0 ..< 17).map { 1 + ($0 * 7) % 31 }
        let state = try XCTUnwrap(
            driver.takeOrMakeAssistantState(
                for: .init(91),
                maximumSequenceLength: 64, historicalPrefixPromptTokens: tokens) as? MiMoV26MTPState
        )
        let prefix = MLXArray(tokens.prefix(8).map(Int32.init), [1, 8])
        let output = adapter.forwardWithHidden(tokens: prefix, caches: caches)
        try withError { fault in
            eval(
                [output.logits, output.lastHidden]
                    + caches.flatMap { ($0 as? KVCache)?.innerState() ?? [] })
            try fault.check()
        }
        assistant.observeCommittedTarget(
            .init(tokens: prefix, hidden: output.lastHidden), requestState: state)
        let codec = CBv2CompleteCheckpointCodec(
            identity: .init(
                modelAggregateHash: "synthetic-tiny-mimo", promptContractID: "test-prompt",
                buildID: "source-regression", numericsFingerprint: "native-bf16"),
            layerKinds: adapter.layerKinds, recurrentSpec: nil,
            kvDTypes: Array(repeating: .bfloat16, count: adapter.layerKinds.count),
            assistant: assistant,
            admission: .init(
                layerKinds: adapter.layerKinds, bytesCapacity: 8 << 20,
                config: .init(watermarkFraction: 0, elementBytes: 2)))
        let capture = try CBv2ContiguousHistoricalCheckpoint(
            codec: codec, position: 8, chunkSize: 8, state: rows)
        return .init(
            target: target, assistant: assistant, adapter: adapter, backend: backend,
            caches: caches, rows: rows, driver: driver, state: state,
            codec: codec, capture: capture, tokens: tokens)
    }
    private func finishAssistant(_ value: Fixture) throws {
        try withError { fault in
            eval(value.assistant.evaluationTargets(for: value.state))
            try fault.check()
        }
        try value.assistant.requestStateDidFinishEvaluation(value.state)
    }
    private func release(_ value: Fixture) {
        value.assistant.releaseRequestState(value.state)
        // Release the ACTUAL bound rows only after all test evaluation/drains.
        XCTAssertNoThrow(try value.backend.releaseValidated(value.rows))
        // The local Fixture still has known passive cache/row aliases. This
        // component assertion is never a physical-free/native-engine receipt.
        withExtendedLifetime(value.target) {}
    }

    func testFreshDriverContextReusedWithoutLateReinstallationAndActualCaptureSettles() throws {
        let value = try fixture()
        defer { release(value) }
        try finishAssistant(value)
        XCTAssertEqual(value.state.observedCount, 8)
        XCTAssertNotNil(
            value.state.prefixCaptureContext, "driver must install before the first observation")
        value.driver.restoreAssistantState(value.state, for: .init(91))
        let again = try XCTUnwrap(
            value.driver.takeOrMakeAssistantState(
                for: .init(91),
                maximumSequenceLength: 64, historicalPrefixPromptTokens: value.tokens)
                as? MiMoV26MTPState)
        XCTAssertTrue(
            again === value.state,
            "existing observed state must not be replaced or reconfigured as fresh")
        try value.capture.captureSettledAssistant(requestState: again)
        let targetWindowCopies =
            value.adapter.layerKinds.filter {
                if case .slidingWindow = $0.attention { return true }
                return false
            }.count * 2
        XCTAssertEqual(value.capture.evaluationRoots.count, targetWindowCopies + 9)
        try value.capture.finishEvaluation()
        XCTAssertNotNil(value.capture.compactAllocationEvidence)
        XCTAssertFalse(value.capture.completionFailed)
        value.capture.close()
        XCTAssertEqual(value.codec.admission.bytesReserved, 0)
    }

    func testDiscardSubmittedTargetCopiesBeforeAssistantAttachmentReallyDrains() throws {
        let value = try fixture()
        defer { release(value) }
        value.capture.markSubmitted()
        try withError { fault in
            asyncEval(value.capture.evaluationRoots)
            try fault.check()
        }
        XCTAssertTrue(value.capture.requiresAssistant)
        XCTAssertThrowsError(
            try value.capture.finishEvaluation(), "publication requires the missing assistant")
        XCTAssertGreaterThan(value.codec.admission.bytesReserved, 0)
        var drains = 0
        value.capture.beforeRequiredDrainForTesting = { drains += 1 }  // observes the REAL following drain
        try value.capture.finishEvaluationForRetirement()
        XCTAssertEqual(drains, 1)
        XCTAssertNil(value.capture.compactAllocationEvidence, "discard is not publication")
        value.capture.close()
        XCTAssertEqual(value.codec.admission.bytesReserved, 0)
        XCTAssertFalse(value.capture.completionFailed)
        // Assistant observation is still owned independently; complete it
        // truthfully before releasing its real cache/state.
        try finishAssistant(value)
    }

    func testRequiredCopyFenceFailureKeepsRealAssistantRootsAndPromiseAfterLaterDrain() throws {
        guard
            ProcessInfo.processInfo.environment["MIMO_PREFIX_CAPTURE_FAULT"]
                == "required_copy_fence"
        else {
            throw XCTSkip(
                "Run alone in a fresh native process; real roots intentionally remain quarantined")
        }
        let value = try fixture()
        defer { release(value) }
        try finishAssistant(value)
        try value.capture.captureSettledAssistant(requestState: value.state)
        weak var actualAssistantRoot = value.capture.evaluationRoots.last
        let charge = value.codec.admission.bytesReserved
        XCTAssertGreaterThan(charge, 0)
        XCTAssertNotNil(actualAssistantRoot)
        value.capture.beforeRequiredDrainForTesting = { throw Failure.injected }
        XCTAssertThrowsError(try value.capture.finishEvaluation())
        XCTAssertTrue(value.capture.completionFailed)
        value.capture.beforeRequiredDrainForTesting = nil
        try withError { fault in
            StreamOrDevice.default.stream.synchronize()
            try fault.check()
        }
        XCTAssertThrowsError(
            try value.capture.finishEvaluation(), "later general drain cannot erase first failure")
        value.capture.close()
        XCTAssertNotNil(
            actualAssistantRoot, "retain actual nine-tensor assistant copy owner, not just C")
        XCTAssertEqual(value.codec.admission.bytesReserved, charge)
        XCTAssertNil(value.capture.compactAllocationEvidence)
    }
}
