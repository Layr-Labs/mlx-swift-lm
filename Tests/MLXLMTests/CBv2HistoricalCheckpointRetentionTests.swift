import XCTest

@testable import MLXLMCommon

/// Demand-targeted retention for historical donors: the first boundary, the
/// coordinator's fork target and the rolling latest, and nothing else.
final class CBv2HistoricalCheckpointRetentionTests: XCTestCase {
    private let stride = CBv2RecurrentCheckpointGeometry.historicalCheckpointStrideTokens

    /// Commit every stride multiple up to `through`, as prefill captures them.
    private func run(
        hint: Int?, resumedAt: Int = 0, through: Int,
        file: StaticString = #filePath, line: UInt = #line
    ) -> (retention: CBv2HistoricalCheckpointRetention, retired: [Int]) {
        var retention = CBv2HistoricalCheckpointRetention(stride: stride, hintTokens: hint, resumedAt: resumedAt)
        var retired: [Int] = []
        let start = resumedAt / stride + 1
        guard start <= through / stride else { return (retention, retired) }
        for multiple in start ... through / stride {
            retired += retention.commit(multiple * stride)
            XCTAssertLessThanOrEqual(retention.retained.count, CBv2HistoricalCheckpointRetention.maximumRetained,
                                     "after \(multiple * stride)", file: file, line: line)
            XCTAssertEqual(retention.retained, retention.retained.sorted(), file: file, line: line)
        }
        return (retention, retired)
    }

    func testNoHintKeepsFirstAndDeepest() {
        let (retention, retired) = run(hint: nil, through: 6144)
        XCTAssertNil(retention.target)
        XCTAssertEqual(retention.retained, [1024, 6144])
        XCTAssertEqual(retired, [2048, 3072, 4096, 5120])
        XCTAssertEqual(retention.publication.publish, [6144, 1024])
        XCTAssertEqual(retention.publication.drop, [])
    }

    func testFleetNovelHintKeepsFirstAndDeepest() {
        let (retention, _) = run(hint: 0, through: 6144)
        XCTAssertNil(retention.target)
        XCTAssertEqual(retention.retained, [1024, 6144])
    }

    func testForkHintKeepsFirstTargetAndDeepestAndPublishesDeepestFirst() {
        // Other prompts share 2,300 tokens; the fork boundary is 2,048.
        let (retention, retired) = run(hint: 2300, through: 6144)
        XCTAssertEqual(retention.target, 2048)
        XCTAssertEqual(retention.retained, [1024, 2048, 6144])
        XCTAssertEqual(retired, [3072, 4096, 5120])
        XCTAssertEqual(retention.publication.publish, [6144, 2048, 1024])
        XCTAssertEqual(retention.publication.drop, [])
    }

    func testTargetAdjacentToTheFinalDeepestIsDroppedAtPublication() {
        // Staged while the prompt is still growing; adjacent only at the end.
        var (retention, _) = run(hint: 5120, through: 5120)
        XCTAssertEqual(retention.retained, [1024, 5120])
        XCTAssertEqual(retention.commit(6144), [])
        XCTAssertEqual(retention.retained, [1024, 5120, 6144])
        XCTAssertEqual(retention.publication.publish, [6144, 1024])
        XCTAssertEqual(retention.publication.drop, [5120])
        // One more stride and the same target is a real fork again.
        XCTAssertEqual(retention.commit(7168), [6144])
        XCTAssertEqual(retention.publication.publish, [7168, 5120, 1024])
        XCTAssertEqual(retention.publication.drop, [])
    }

    func testTargetAtTheDeepestOrTheFirstIsThatSameCheckpoint() {
        let (deepest, _) = run(hint: 6400, through: 6144)
        XCTAssertEqual(deepest.target, 6144)
        XCTAssertEqual(deepest.retained, [1024, 6144])
        XCTAssertEqual(deepest.publication.publish, [6144, 1024])
        let (first, _) = run(hint: 1500, through: 6144)
        XCTAssertEqual(first.target, 1024)
        XCTAssertEqual(first.retained, [1024, 6144])
        XCTAssertEqual(first.publication.drop, [])
    }

    func testTargetBelowTheFirstOrBeyondThePromptIsIgnored() {
        let (below, _) = run(hint: 900, through: 6144)
        XCTAssertNil(below.target)
        XCTAssertEqual(below.retained, [1024, 6144])
        let (beyond, _) = run(hint: 9000, through: 6144)
        XCTAssertEqual(beyond.target, 8192)
        XCTAssertEqual(beyond.retained, [1024, 6144])
        XCTAssertEqual(beyond.publication.publish, [6144, 1024])
    }

    func testAdopterCapturesOnlyAboveItsRestoredBoundary() {
        // Restored at the fork itself: neither the first nor the target is
        // recaptured; only the rolling latest above 2,048 is kept.
        let (atFork, retired) = run(hint: 2048, resumedAt: 2048, through: 6144)
        XCTAssertFalse(atFork.keepsFirst)
        XCTAssertNil(atFork.target)
        XCTAssertNil(atFork.first)
        XCTAssertEqual(atFork.retained, [6144])
        XCTAssertEqual(retired, [3072, 4096, 5120])
        XCTAssertEqual(atFork.publication.publish, [6144])
        // A fork above the restore point is still worth a checkpoint.
        let (above, _) = run(hint: 4200, resumedAt: 2048, through: 7168)
        XCTAssertEqual(above.target, 4096)
        XCTAssertEqual(above.retained, [4096, 7168])
        XCTAssertEqual(above.publication.publish, [7168, 4096])
        // Restored at the deepest boundary: nothing above it, nothing staged.
        let (nothing, _) = run(hint: 6144, resumedAt: 6144, through: 6144)
        XCTAssertTrue(nothing.retained.isEmpty)
        XCTAssertTrue(nothing.publication.publish.isEmpty)
    }

    func testStagedCountNeverExceedsThreeForAnyHint() {
        for hint in [nil, 0, 1, 1023, 1024, 2048, 3000, 5120, 9_000, 40_000, 1 << 40] as [Int?] {
            for resumedAt in [0, 1024, 4096] {
                let (retention, retired) = run(hint: hint, resumedAt: resumedAt, through: 40 * stride)
                XCTAssertEqual(retention.retained.last, 40 * stride)
                XCTAssertEqual(Set(retention.retained).count, retention.retained.count)
                XCTAssertTrue(Set(retired).isDisjoint(with: retention.retained))
                XCTAssertEqual(retired.count + retention.retained.count, 40 - resumedAt / stride)
            }
        }
    }

    func testFirstIsTheFirstBoundaryThatLandsAndOutOfOrderIsRefused() {
        // The store refused 1,024 (a higher floor): 2,048 is the first.
        var retention = CBv2HistoricalCheckpointRetention(stride: stride, hintTokens: nil)
        XCTAssertTrue(retention.firstIsOpen)
        XCTAssertEqual(retention.commit(2048), [])
        XCTAssertFalse(retention.firstIsOpen)
        XCTAssertEqual(retention.first, 2048)
        XCTAssertEqual(retention.commit(2048), [2048], "a duplicate never displaces a retained boundary")
        XCTAssertEqual(retention.commit(1024), [1024])
        XCTAssertEqual(retention.retained, [2048])
        XCTAssertEqual(retention.commit(3072), [])
        XCTAssertEqual(retention.commit(4096), [3072])
        XCTAssertEqual(retention.retained, [2048, 4096])
    }

    func testBytePressureGivesUpTheTargetAndNeverThePair() {
        var (retention, _) = run(hint: 2048, through: 4096)
        XCTAssertEqual(retention.retained, [1024, 2048, 4096])
        XCTAssertEqual(retention.dropInterior(), 2048)
        XCTAssertEqual(retention.retained, [1024, 4096])
        XCTAssertNil(retention.dropInterior())
        XCTAssertEqual(retention.retained, [1024, 4096])
    }

    func testDegenerateStrideHasNoTarget() {
        XCTAssertNil(CBv2HistoricalCheckpointRetention(stride: 1, hintTokens: 4096).target)
        XCTAssertNil(CBv2HistoricalCheckpointRetention(stride: 0, hintTokens: 4096).target)
        XCTAssertNil(CBv2HistoricalCheckpointRetention(stride: stride, hintTokens: -5).target)
    }
}
