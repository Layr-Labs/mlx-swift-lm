import XCTest

@testable import MLXLMCommon

/// Host-only authoritative scheduling/projection proof; actual recurrent
/// tensor/output parity and donor cost are separately required per architecture.
final class CBv2DemandedCheckpointPartitionTests: XCTestCase {
    private func config(qualified: Bool) -> CBv2SchedulerConfig {
        var value = CBv2SchedulerConfig(
            maxConcurrentRequests: 1, maxBatchedTokensPerStep: 2_048,
            prefillChunkSize: 512, soloPrefillStripeTokens: 4_096,
            maxConcurrentPartialPrefills: 1, enablePrefixCache: true)
        value.demandedShortCheckpointMinimumTokens = 1_024
        value.demandedCheckpointPartitionIncludesLongPrompts = qualified
        return value
    }

    func testDefaultLongGeometryAndQualifiedBoundaryHaveMatchingForecasts() throws {
        for qualified in [false, true] {
            let scheduler = SchedulerV2(config: config(qualified: qualified))
            let request = CBv2Request(
                id: .init(7_050), promptTokens: Array(repeating: 1, count: 7_169),
                maxTokens: 1, cacheSalt: "tenant-fixture", prefixCheckpointTargetTokens: 6_144)
            let record = try scheduler.enqueue(request)
            guard
                case .bounded(let allWork, _) = scheduler.firstTokenWorkProjection(for: request.id)
            else { return XCTFail("whole long-prefix projection must be bounded") }
            XCTAssertEqual(allWork.prefillTokens, 7_169)
            XCTAssertEqual(allWork.scheduledSteps, qualified ? 3 : 2)
            let first = scheduler.plan()
            XCTAssertEqual(first.assignments.map(\.numTokens), [4_096])
            XCTAssertTrue(first.demandedShortCheckpointRows.isEmpty)
            guard
                case .bounded(let tailWork, _) = scheduler.firstTokenWorkProjection(for: request.id)
            else { return XCTFail("remaining long-prefix projection must be bounded") }
            XCTAssertEqual(tailWork.prefillTokens, 3_073)
            XCTAssertEqual(tailWork.scheduledSteps, qualified ? 2 : 1)
            let second = scheduler.plan()
            XCTAssertEqual(second.assignments.map(\.numTokens), qualified ? [2_048] : [3_073])
            XCTAssertEqual(second.demandedShortCheckpointRows, qualified ? [request.id] : [])
            XCTAssertEqual(record.plannedPrefillChunkSize, 4_096)
            if qualified {
                let third = scheduler.plan()
                XCTAssertEqual(third.assignments.map(\.numTokens), [1_025])
                XCTAssertTrue(third.demandedShortCheckpointRows.isEmpty)
            }
            XCTAssertEqual(record.numComputedTokens, 7_169)
            XCTAssertEqual(record.preemptionCount, 0)
        }
    }

    func testQualificationAddsAtMostOneAlignedBoundaryIncludingTheLongTail() throws {
        let cases: [(prompt: Int, hint: Int, chunks: [Int])] = [
            (7_169, 1_031, [1_024, 3_072, 3_073]),
            (7_169, 4_096, [4_096, 3_073]),
            (16_513, 14_343, [4_096, 4_096, 4_096, 2_048, 2_048, 129]),
        ]
        for cell in cases {
            let scheduler = SchedulerV2(config: config(qualified: true))
            let request = CBv2Request(
                id: .init(7_051), promptTokens: Array(repeating: 1, count: cell.prompt),
                maxTokens: 1, cacheSalt: "tenant-fixture", prefixCheckpointTargetTokens: cell.hint)
            let record = try scheduler.enqueue(request)
            guard case .bounded(let work, _) = scheduler.firstTokenWorkProjection(for: request.id)
            else { return XCTFail("qualified aligned projection must be bounded") }
            XCTAssertEqual(work.scheduledSteps, cell.chunks.count)
            var marked = 0
            for expected in cell.chunks {
                let step = scheduler.plan()
                XCTAssertEqual(step.assignments.map(\.numTokens), [expected])
                marked += step.demandedShortCheckpointRows.count
                XCTAssertLessThanOrEqual(expected, 4_096)
            }
            // A target and residual original end can both shorten a
            // proposed range, while replacing only one original range.
            XCTAssertLessThanOrEqual(marked, 2)
            XCTAssertEqual(record.numComputedTokens, cell.prompt)
        }
    }

    func testLongQualificationKeepsExistingParticipationAndCaptureVetoes() throws {
        for disabled in [false, true] {
            let scheduler = SchedulerV2(config: config(qualified: true))
            let request = CBv2Request(
                id: .init(7_052), promptTokens: Array(repeating: 1, count: 7_169),
                maxTokens: 1, cacheSalt: "tenant-fixture", prefixCacheEnabled: !disabled,
                prefixCheckpointTargetTokens: 6_144)
            let record = try scheduler.enqueue(request)
            if !disabled { record.shortCheckpointCaptureDisarmed = true }
            guard case .bounded(let work, _) = scheduler.firstTokenWorkProjection(for: request.id)
            else { return XCTFail("excluded long-prefix projection must be bounded") }
            XCTAssertEqual(work.scheduledSteps, 2)
            XCTAssertEqual(scheduler.plan().assignments.map(\.numTokens), [4_096])
            let second = scheduler.plan()
            XCTAssertEqual(second.assignments.map(\.numTokens), [3_073])
            XCTAssertTrue(second.demandedShortCheckpointRows.isEmpty)
        }
    }
}
