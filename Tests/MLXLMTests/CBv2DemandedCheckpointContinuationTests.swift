import XCTest

@testable import MLXLMCommon

/// Scheduler, pure forecast, geometry and retention witnesses only. Actual
/// native MTP state, encrypted publication and donor/fork cost remain gates.
final class CBv2DemandedCheckpointContinuationTests: XCTestCase {
    private func config(qualified: Bool = true, concurrent: Int = 1) -> CBv2SchedulerConfig {
        var value = CBv2SchedulerConfig(
            maxConcurrentRequests: concurrent, maxBatchedTokensPerStep: 2_048,
            prefillChunkSize: 512, soloPrefillStripeTokens: 4_096,
            maxConcurrentPartialPrefills: 1, enablePrefixCache: true)
        value.demandedShortCheckpointMinimumTokens = 1_024
        value.demandedCheckpointPartitionIncludesLongPrompts = qualified
        return value
    }

    private func request(prompt: Int = 16_513, hint: Int = 14_336) -> CBv2Request {
        CBv2Request(
            id: .init(7_060), promptTokens: Array(repeating: 1, count: prompt),
            maxTokens: 1, cacheSalt: "tenant-fixture", prefixCheckpointTargetTokens: hint)
    }

    private func expectProjection(
        _ scheduler: SchedulerV2, id: CBv2RequestID, steps: Int,
        inFlight: [(id: CBv2RequestID, numTokens: Int)] = [],
        file: StaticString = #filePath, line: UInt = #line
    ) {
        guard
            case .bounded(let work, _) = scheduler.firstTokenWorkProjection(
                for: id, inFlightAssignments: inFlight)
        else { return XCTFail("qualified text projection must be bounded", file: file, line: line) }
        XCTAssertEqual(work.scheduledSteps, steps, file: file, line: line)
    }

    private func reachTarget(_ scheduler: SchedulerV2, id: CBv2RequestID) -> CBv2StepPlan {
        for _ in 0 ..< 3 {
            let step = scheduler.plan()
            XCTAssertEqual(step.assignments.map(\.numTokens), [4_096])
            XCTAssertTrue(step.demandedShortCheckpointRows.isEmpty)
        }
        let target = scheduler.plan()
        XCTAssertEqual(target.assignments.map(\.numTokens), [2_048])
        XCTAssertEqual(target.demandedShortCheckpointRows, [id])
        return target
    }

    func testLongSplitPreservesActualLatestGeometryAndThreeRetainedRoles() throws {
        for qualified in [false, true] {
            let scheduler = SchedulerV2(config: config(qualified: qualified))
            let request = request()
            let record = try scheduler.enqueue(request)
            let chunks =
                qualified
                ? [4_096, 4_096, 4_096, 2_048, 2_048, 129]
                : [4_096, 4_096, 4_096, 4_096, 129]
            expectProjection(scheduler, id: request.id, steps: chunks.count)
            var geometry = CBv2RecurrentCheckpointGeometry()
            var retention = CBv2CheckpointRetention(stride: nil, hintTokens: 14_336)
            var captured: [Int] = []
            for (index, count) in chunks.enumerated() {
                let start = record.numComputedTokens
                let step = scheduler.plan()
                XCTAssertEqual(step.assignments.map(\.numTokens), [count])
                XCTAssertEqual(
                    step.demandedShortCheckpointRows,
                    qualified && (index == 3 || index == 4) ? [request.id] : [])
                XCTAssertEqual(record.plannedPrefillChunkSize, 4_096)
                if geometry.record(
                    range: start ..< record.numComputedTokens, cap: 4_096,
                    promptLength: request.promptTokens.count, packed: false),
                    record.numComputedTokens < request.promptTokens.count
                {
                    captured.append(record.numComputedTokens)
                    _ = retention.commit(record.numComputedTokens)
                    XCTAssertLessThanOrEqual(retention.retained.count, 3)
                }
            }
            XCTAssertEqual(
                captured,
                qualified
                    ? [4_096, 8_192, 12_288, 14_336, 16_384]
                    : [4_096, 8_192, 12_288, 16_384])
            XCTAssertEqual(
                retention.publication,
                qualified
                    ? [16_384, 14_336, 4_096] : [16_384, 12_288, 4_096])
            XCTAssertEqual(record.numComputedTokens, 16_513)
            XCTAssertEqual(record.preemptionCount, 0)
        }
    }

    func testProjectionChargesInFlightTargetAndResidualExactlyOnceWithoutMutation() throws {
        let scheduler = SchedulerV2(config: config())
        let request = request()
        let record = try scheduler.enqueue(request)
        let target = reachTarget(scheduler, id: request.id)
        expectProjection(scheduler, id: request.id, steps: 3, inFlight: target.assignments)
        XCTAssertEqual(record.numComputedTokens, 14_336)
        expectProjection(scheduler, id: request.id, steps: 2)
        let residual = scheduler.plan()
        XCTAssertEqual(residual.assignments.map(\.numTokens), [2_048])
        XCTAssertEqual(residual.demandedShortCheckpointRows, [request.id])
        expectProjection(scheduler, id: request.id, steps: 2, inFlight: residual.assignments)
        XCTAssertEqual(record.numComputedTokens, 16_384)
        expectProjection(scheduler, id: request.id, steps: 1)
    }

    func testRollbackRetriesTargetAndResidualWithTheSameRangeAndProjection() throws {
        let scheduler = SchedulerV2(config: config())
        let request = request()
        let record = try scheduler.enqueue(request)
        let target = reachTarget(scheduler, id: request.id)
        scheduler.rollback(target)
        XCTAssertEqual(record.numComputedTokens, 12_288)
        expectProjection(scheduler, id: request.id, steps: 3)
        XCTAssertEqual(scheduler.plan().assignments.map(\.numTokens), [2_048])
        let residual = scheduler.plan()
        XCTAssertEqual(residual.assignments.map(\.numTokens), [2_048])
        scheduler.rollback(residual)
        XCTAssertEqual(record.numComputedTokens, 14_336)
        expectProjection(scheduler, id: request.id, steps: 2)
        let retry = scheduler.plan()
        XCTAssertEqual(retry.assignments.map(\.numTokens), [2_048])
        XCTAssertEqual(retry.demandedShortCheckpointRows, [request.id])
        XCTAssertEqual(scheduler.plan().assignments.map(\.numTokens), [129])
    }

    func testAlignedTargetAndTerminalOriginalEndNeverAddAnotherRange() throws {
        for cell in [
            (prompt: 16_513, hint: 12_288, chunks: [4_096, 4_096, 4_096, 4_096, 129]),
            (prompt: 16_513, hint: 1_031, chunks: [1_024, 3_072, 4_096, 4_096, 4_096, 129]),
            (prompt: 7_169, hint: 6_144, chunks: [4_096, 2_048, 1_025]),
        ] {
            let scheduler = SchedulerV2(config: config())
            let request = request(prompt: cell.prompt, hint: cell.hint)
            let record = try scheduler.enqueue(request)
            expectProjection(scheduler, id: request.id, steps: cell.chunks.count)
            for count in cell.chunks {
                XCTAssertEqual(scheduler.plan().assignments.map(\.numTokens), [count])
            }
            XCTAssertEqual(record.numComputedTokens, cell.prompt)
        }
    }

    func testRealKVFallbackDoesNotRememberAnUnassignedRangeEnd() throws {
        let capacity = AdmissionV2(
            layerKinds: [.init(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)],
            bytesCapacity: 12_800 * 8,
            config: .init(watermarkFraction: 0, elementBytes: 4))
        let scheduler = SchedulerV2(config: config(), capacity: capacity)
        let request = request()
        let record = try scheduler.enqueue(request)
        for _ in 0 ..< 3 {
            XCTAssertEqual(scheduler.plan().assignments.map(\.numTokens), [4_096])
        }
        let refusedTarget = scheduler.plan()
        XCTAssertEqual(refusedTarget.assignments.map(\.numTokens), [512])
        XCTAssertTrue(refusedTarget.demandedShortCheckpointRows.isEmpty)
        XCTAssertEqual(record.numComputedTokens, 12_800)
        XCTAssertEqual(capacity.bytesReserved, 12_800 * 8)
        capacity.updateBytesCapacity(16_513 * 8)
        // The newly proposed range ends at the prompt. Remembering the
        // refused 16,384 endpoint here would invent its old geometry.
        expectProjection(scheduler, id: request.id, steps: 2)
        XCTAssertEqual(scheduler.plan().assignments.map(\.numTokens), [1_536])
        let tail = scheduler.plan()
        XCTAssertEqual(tail.assignments.map(\.numTokens), [2_177])
        XCTAssertTrue(tail.demandedShortCheckpointRows.isEmpty)
        scheduler.finish(id: request.id, reason: .cancelled)
        capacity.releaseAll(id: request.id)
        XCTAssertEqual(capacity.bytesReserved, 0)
    }

    func testActualUnarmedProgressDiscardsCarryButPauseWithoutWorkDoesNot() throws {
        let scheduler = SchedulerV2(config: config(concurrent: 2))
        let request = request()
        let record = try scheduler.enqueue(request)
        _ = reachTarget(scheduler, id: request.id)
        scheduler.pause(request.id)
        XCTAssertTrue(scheduler.plan().assignments.isEmpty)
        scheduler.resume(request.id)
        expectProjection(scheduler, id: request.id, steps: 2)
        scheduler.pause(request.id)
        let neighbor = CBv2Request(id: .init(7_061), promptTokens: [1], maxTokens: 2)
        try scheduler.enqueue(neighbor)
        XCTAssertEqual(scheduler.plan().assignments.map(\.numTokens), [1])
        scheduler.markPendingSamples(ids: [neighbor.id])
        scheduler.recordSampled(id: neighbor.id, token: 1)
        scheduler.resume(request.id)
        let mixed = scheduler.plan()
        XCTAssertEqual(mixed.assignments.map(\.numTokens), [512, 1])
        XCTAssertTrue(mixed.demandedShortCheckpointRows.isEmpty)
        XCTAssertEqual(record.numComputedTokens, 14_848)
        scheduler.finish(id: neighbor.id, reason: .cancelled)
        expectProjection(scheduler, id: request.id, steps: 1)
        let ordinary = scheduler.plan()
        XCTAssertEqual(ordinary.assignments.map(\.numTokens), [1_665])
        XCTAssertTrue(ordinary.demandedShortCheckpointRows.isEmpty)
    }

    func testActualPreemptionAndMonotonicCaptureDisarmSuppressContinuation() throws {
        for preempted in [false, true] {
            let scheduler = SchedulerV2(config: config())
            let request = request()
            let record = try scheduler.enqueue(request)
            _ = reachTarget(scheduler, id: request.id)
            if preempted {
                XCTAssertTrue(scheduler.requeueOnCapacity(request.id))
                XCTAssertEqual(record.preemptionCount, 1)
                expectProjection(scheduler, id: request.id, steps: 5)
                XCTAssertEqual(scheduler.plan().assignments.map(\.numTokens), [4_096])
            } else {
                record.shortCheckpointCaptureDisarmed = true
                expectProjection(scheduler, id: request.id, steps: 1)
                XCTAssertEqual(scheduler.plan().assignments.map(\.numTokens), [2_177])
            }
        }
    }
}
