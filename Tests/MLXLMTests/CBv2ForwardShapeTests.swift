import Foundation
import XCTest

#if canImport(MLXLMCommon)
    @testable import MLXLMCommon
#endif

final class CBv2ForwardShapeTests: XCTestCase {
    func testDisabledObserverKeepsDispatchUninstrumentedAndOmitsTimingStorage() throws {
        var calls = 0
        CBv2ForwardShapeObservation.dispatch(step: nil, phase: .decode) {
            calls += 1
            XCTAssertFalse(CBv2ForwardShapeObservation.isActive)
            XCTAssertNil(
                CBv2ForwardShapeObservation.beginTarget(liveBatchRows: 1, sequenceWidth: 1))
        }
        XCTAssertEqual(calls, 1)
        let disabled = CBv2ForwardShapeSnapshot.disabled
        XCTAssertFalse(disabled.enabled)
        XCTAssertNil(disabled.completedStepTimings)
        XCTAssertNil(disabled.droppedStepTimings)
        let json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(disabled)) as? [String: Any])
        XCTAssertNil(json["completedStepTimings"])
        XCTAssertNil(json["droppedStepTimings"])
    }

    func testStepTimingsMeasureConfirmedMixedWorkOnceAndReset() throws {
        let recorder = CBv2ForwardShapeRecorder()
        try recorder.reset()
        let step = recorder.beginStep()
        let leaf = LeafSpy()
        CBv2ForwardShapeObservation.dispatch(step: step, phase: .prefill) {
            leaf.forward(rows: 1, columns: 128)
        }
        CBv2ForwardShapeObservation.dispatch(step: step, phase: .decode) {
            leaf.forward(rows: 2, columns: 1)
        }
        step.attach()
        XCTAssertEqual(recorder.snapshot().completedStepTimings?.count, 0)
        step.complete(wallNanos: 75_000_000)
        step.complete(wallNanos: 900_000_000)
        let measured = try XCTUnwrap(recorder.snapshot().completedStepTimings)
        XCTAssertEqual(measured.count, 1)
        XCTAssertEqual(measured.first?.phase, .mixedFrontier)
        XCTAssertEqual(measured.first?.wallNanos, 75_000_000)
        try recorder.reset()
        XCTAssertEqual(recorder.snapshot().completedStepTimings?.count, 0)
        XCTAssertEqual(recorder.snapshot().droppedStepTimings, 0)
    }

    func testStepTimingStorageIsBoundedAndAbandonedWorkNeverClaimsCompletion() throws {
        let recorder = CBv2ForwardShapeRecorder()
        try recorder.reset()
        for _ in 0 ... CBv2ForwardShapeRecorder.maximumStepTimings {
            let step = recorder.beginStep()
            CBv2ForwardShapeObservation.dispatch(step: step, phase: .mtpVerification) {
                LeafSpy().forward(rows: 1, columns: 3)
            }
            step.attach()
            step.complete(wallNanos: 1)
        }
        let abandoned = recorder.beginStep()
        CBv2ForwardShapeObservation.dispatch(step: abandoned, phase: .prefill) {
            LeafSpy().forward(rows: 1, columns: 128)
        }
        abandoned.finishBuilding()
        let snapshot = recorder.snapshot()
        XCTAssertEqual(
            snapshot.completedStepTimings?.count, CBv2ForwardShapeRecorder.maximumStepTimings)
        XCTAssertEqual(snapshot.droppedStepTimings, 1)
        XCTAssertEqual(snapshot.abandonedSteps, 1)
        XCTAssertTrue(snapshot.completedStepTimings?.allSatisfy { $0.phase == .decode } == true)
    }

    func testFullAxisBucketsMarkUnclassifiableStepTimingsDropped() throws {
        let recorder = CBv2ForwardShapeRecorder()
        try recorder.reset()
        // Fill the shape table while timing storage still has ample room.
        for width in 1 ... CBv2ForwardShapeRecorder.maximumBuckets {
            let step = recorder.beginStep()
            CBv2ForwardShapeObservation.dispatch(step: step, phase: .decode) {
                LeafSpy().forward(rows: 1, columns: width)
            }
            step.attach()
            step.complete(wallNanos: 10)
        }
        let unknown = recorder.beginStep()
        CBv2ForwardShapeObservation.dispatch(step: unknown, phase: .prefill) {
            LeafSpy().forward(rows: 1, columns: 1024)
        }
        unknown.attach()
        unknown.complete(wallNanos: 20)
        unknown.complete(wallNanos: 20)  // Retirement must count the drop once.

        let mixed = recorder.beginStep()
        CBv2ForwardShapeObservation.dispatch(step: mixed, phase: .decode) {
            LeafSpy().forward(rows: 1, columns: 1)  // Previously observed axes.
        }
        CBv2ForwardShapeObservation.dispatch(step: mixed, phase: .prefill) {
            LeafSpy().forward(rows: 1, columns: 1024)  // Rejected new target axes.
        }
        mixed.attach()
        mixed.complete(wallNanos: 30)

        let snapshot = recorder.snapshot()
        XCTAssertEqual(snapshot.entries.count, CBv2ForwardShapeRecorder.maximumBuckets)
        XCTAssertEqual(
            snapshot.completedStepTimings?.count, CBv2ForwardShapeRecorder.maximumBuckets)
        XCTAssertEqual(snapshot.droppedStepTimings, 2)
        XCTAssertEqual(snapshot.droppedCalls, 2)
        XCTAssertEqual(snapshot.pendingSteps, 0)
        XCTAssertEqual(snapshot.entries.reduce(0) { $0 + $1.completedCalls }, 257)
    }

    func testMissingTargetDispatchMarksCompletedTimingDropped() throws {
        let recorder = CBv2ForwardShapeRecorder()
        try recorder.reset()
        let step = recorder.beginStep()
        CBv2ForwardShapeObservation.dispatch(step: step, phase: .prefill) {}
        step.attach()
        step.complete(wallNanos: 100)
        XCTAssertEqual(recorder.snapshot().unobservedDispatches, 1)
        XCTAssertEqual(recorder.snapshot().completedStepTimings?.count, 0)
        XCTAssertEqual(recorder.snapshot().droppedStepTimings, 1)
    }

    private final class LeafSpy {
        var calls = 0
        func forward(rows: Int, columns: Int, body: () -> Void = {}) {
            let call = CBv2ForwardShapeObservation.beginTarget(
                liveBatchRows: rows, sequenceWidth: columns)
            defer { call?.end() }
            calls += 1
            body()
        }
    }

    private func counts(_ snapshot: CBv2ForwardShapeSnapshot, kind: CBv2ForwardKind = .target)
        -> [CBv2ForwardShapeCount]
    {
        snapshot.entries.filter { $0.axes.kind == kind }
    }

    func testRowSplittingRecordsFourB1CallsAndPackedRecordsOneB4Call() throws {
        let recorder = CBv2ForwardShapeRecorder()
        try recorder.reset()
        let leaf = LeafSpy()
        let split = recorder.beginStep()
        CBv2ForwardShapeObservation.dispatch(step: split, phase: .decode) {
            // The outer request cohort is four, but the actual leaf is B1.
            for _ in 0 ..< 4 { leaf.forward(rows: 1, columns: 1) }
        }
        split.attach()
        XCTAssertEqual(counts(recorder.snapshot()).map(\.submittedCalls), [4])
        XCTAssertEqual(counts(recorder.snapshot()).map(\.completedCalls), [0])
        XCTAssertEqual(recorder.snapshot().pendingSteps, 1)
        split.complete()
        XCTAssertEqual(counts(recorder.snapshot()).map { $0.axes.liveBatchRows }, [1])
        XCTAssertEqual(counts(recorder.snapshot()).map(\.completedCalls), [4])
        try recorder.reset()
        let packed = recorder.beginStep()
        CBv2ForwardShapeObservation.dispatch(step: packed, phase: .decode) {
            leaf.forward(rows: 4, columns: 1)
        }
        packed.attach()
        packed.complete()
        XCTAssertEqual(counts(recorder.snapshot()).map { $0.axes.liveBatchRows }, [4])
        XCTAssertEqual(counts(recorder.snapshot()).map(\.completedCalls), [1])
    }

    func testLookaheadAndCompiledPhysicalRowsNeverBecomeLiveBatchRows() throws {
        let recorder = CBv2ForwardShapeRecorder()
        try recorder.reset()
        let step = recorder.beginStep()
        CBv2ForwardShapeObservation.dispatch(step: step, phase: .mtpVerification) {
            LeafSpy().forward(rows: 1, columns: 4) {
                CBv2ForwardShapeObservation.compiledComponent(.gptossExperts, physicalRows: 8) {}
            }
        }
        step.attach()
        step.complete()
        let target = try XCTUnwrap(counts(recorder.snapshot()).first)
        XCTAssertEqual(target.axes.phase, .mtpVerification)
        XCTAssertEqual(target.axes.liveBatchRows, 1)
        XCTAssertEqual(target.axes.sequenceWidth, 4)
        XCTAssertEqual(target.axes.physicalBatchRows, 1)
        let compiled = try XCTUnwrap(counts(recorder.snapshot(), kind: .compiledComponent).first)
        XCTAssertEqual(compiled.axes.liveBatchRows, 1)
        XCTAssertEqual(compiled.axes.physicalComponentRows, 8)
    }

    func testCompiledInvocationsCountAfterTraceAndExcludeTraceCallbacks() throws {
        let recorder = CBv2ForwardShapeRecorder()
        try recorder.reset()
        let step = recorder.beginStep()
        let leaf = LeafSpy()
        var traces = 0
        var executions = 0
        func compiled() {
            CBv2ForwardShapeObservation.compiledComponent(.gptossExperts, physicalRows: 4) {
                if traces == 0 {
                    traces += 1
                    // A tracing callback must not fabricate a real B4 target.
                    LeafSpy().forward(rows: 4, columns: 1)
                }
                executions += 1
            }
        }
        CBv2ForwardShapeObservation.dispatch(step: step, phase: .decode) {
            for _ in 0 ..< 2 { leaf.forward(rows: 2, columns: 1, body: compiled) }
        }
        step.attach()
        step.complete()
        XCTAssertEqual(traces, 1)
        XCTAssertEqual(executions, 2)
        XCTAssertEqual(counts(recorder.snapshot()).map { $0.axes.liveBatchRows }, [2])
        XCTAssertEqual(counts(recorder.snapshot()).map(\.completedCalls), [2])
        XCTAssertEqual(
            counts(recorder.snapshot(), kind: .compiledComponent).map(\.completedCalls), [2])
    }

    func testSerialVerificationColumnsStaySeparateB1Calls() throws {
        let recorder = CBv2ForwardShapeRecorder()
        try recorder.reset()
        let step = recorder.beginStep()
        for _ in 0 ..< 4 {
            CBv2ForwardShapeObservation.dispatch(step: step, phase: .mtpVerification) {
                LeafSpy().forward(rows: 1, columns: 1)
            }
        }
        step.attach()
        step.complete()
        let value = try XCTUnwrap(counts(recorder.snapshot()).first)
        XCTAssertEqual(value.axes.liveBatchRows, 1)
        XCTAssertEqual(value.axes.sequenceWidth, 1)
        XCTAssertEqual(value.axes.phase, .mtpVerification)
        XCTAssertEqual(value.completedCalls, 4)
    }

    func testWarmupAndBeforeAfterDeltasExcludeEarlierCalls() throws {
        LeafSpy().forward(rows: 8, columns: 32)  // No dispatch scope: warmup is invisible.
        let recorder = CBv2ForwardShapeRecorder()
        try recorder.reset()
        let before = recorder.snapshot()
        let step = recorder.beginStep()
        CBv2ForwardShapeObservation.dispatch(step: step, phase: .prefill) {
            LeafSpy().forward(rows: 2, columns: 16)
        }
        step.attach()
        step.complete()
        let after = recorder.snapshot()
        let delta = after.delta(since: before)
        XCTAssertTrue(delta.complete)
        XCTAssertEqual(delta.entries.count, 1)
        XCTAssertEqual(delta.entries.first?.submittedCalls, 1)
        XCTAssertTrue(after.delta(since: after).entries.isEmpty)
        try recorder.reset()
        XCTAssertFalse(recorder.snapshot().delta(since: after).complete)
    }

    func testAbandonmentIsNotCompletionAndResetRefusesPendingWork() throws {
        let recorder = CBv2ForwardShapeRecorder()
        try recorder.reset()
        let before = recorder.snapshot()
        let step = recorder.beginStep()
        CBv2ForwardShapeObservation.dispatch(step: step, phase: .decode) {
            LeafSpy().forward(rows: 2, columns: 1)
        }
        XCTAssertThrowsError(try recorder.reset())
        step.finishBuilding()  // Rejected construction: no successful step completion.
        let after = recorder.snapshot()
        XCTAssertEqual(after.pendingSteps, 0)
        XCTAssertEqual(after.abandonedSteps, 1)
        XCTAssertEqual(counts(after).first?.submittedCalls, 1)
        XCTAssertEqual(counts(after).first?.completedCalls, 0)
        XCTAssertFalse(after.delta(since: before).complete)
    }

    func testThrownDispatchRestoresContextAndRetiresEnteredCallsWithoutCompletion() throws {
        enum Refusal: Error { case rejected }
        let recorder = CBv2ForwardShapeRecorder()
        try recorder.reset()
        let before = recorder.snapshot()
        let step = recorder.beginStep()
        XCTAssertThrowsError(
            try CBv2ForwardShapeObservation.dispatch(step: step, phase: .mtpVerification) {
                LeafSpy().forward(rows: 2, columns: 3)
                throw Refusal.rejected
            })
        XCTAssertFalse(CBv2ForwardShapeObservation.isActive)
        step.finishBuilding()
        let after = recorder.snapshot()
        let delta = after.delta(since: before)
        XCTAssertEqual(after.pendingSteps, 0)
        XCTAssertEqual(after.unobservedDispatches, 0)
        XCTAssertEqual(after.droppedCalls, 0)
        XCTAssertEqual(counts(after).first?.submittedCalls, 1)
        XCTAssertEqual(counts(after).first?.completedCalls, 0)
        XCTAssertEqual(Set(delta.reasons), Set(["abandoned_step", "unconfirmed_calls"]))
    }

    func testMissingLeafAndInvalidOrExcessAxesFailClosedWithoutPrivatePayload() throws {
        let recorder = CBv2ForwardShapeRecorder()
        try recorder.reset()
        let step = recorder.beginStep()
        CBv2ForwardShapeObservation.dispatch(step: step, phase: .decode) {}
        CBv2ForwardShapeObservation.dispatch(step: step, phase: .prefill) {
            let leaf = LeafSpy()
            for width in 1 ... 257 { leaf.forward(rows: 1, columns: width) }
            leaf.forward(rows: Int.max, columns: Int.max)
        }
        step.attach()
        step.complete()
        let value = recorder.snapshot()
        XCTAssertEqual(value.entries.count, 256)
        XCTAssertEqual(value.droppedCalls, 2)
        XCTAssertEqual(value.unobservedDispatches, 1)
        let encoded = String(decoding: try JSONEncoder().encode(value), as: UTF8.self)
        XCTAssertLessThan(encoded.utf8.count, 100_000)
        for forbidden in ["token_ids", "request_id", "prompt", "text", "tensor", "\(Int.max)"] {
            XCTAssertFalse(encoded.contains(forbidden))
        }
    }
}
