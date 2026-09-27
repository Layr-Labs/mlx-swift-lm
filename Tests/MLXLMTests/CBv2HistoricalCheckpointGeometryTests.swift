import XCTest

@testable import MLXLMCommon

/// Historical (attention-only) checkpoint geometry: every stride-aligned
/// position a computed range covers is a checkpoint, whatever chunk cap
/// produced it. The recurrent rule next to it must not move.
final class CBv2HistoricalCheckpointGeometryTests: XCTestCase {
    private let prompt = 6_700

    func testDefaultStrideMatchesProviderFloorAlignment() {
        XCTAssertEqual(CBv2RecurrentCheckpointGeometry.historicalCheckpointStrideTokens, 1024)
        XCTAssertEqual(CBv2CheckpointRetention.maximumRetained, 3)
    }

    func testCompanyChunksThenSoloStripeCaptureEveryStrideMultiple() {
        var geometry = CBv2RecurrentCheckpointGeometry()
        XCTAssertEqual(geometry.recordHistorical(range: 0 ..< 512, promptLength: prompt, packed: false), [])
        XCTAssertEqual(geometry.recordHistorical(range: 512 ..< 1024, promptLength: prompt, packed: false), [1024])
        // Cap changes 512 -> 2048 when the row becomes the armed solo stripe.
        XCTAssertEqual(geometry.recordHistorical(range: 1024 ..< 3072, promptLength: prompt, packed: false), [2048, 3072])
        XCTAssertEqual(geometry.recordHistorical(range: 3072 ..< 5120, promptLength: prompt, packed: false), [4096, 5120])
        XCTAssertTrue(geometry.isArmed)
        XCTAssertEqual(geometry.position, 5120)
    }

    func testSoloStripeThenCompanyChunksCaptureEveryStrideMultiple() {
        var geometry = CBv2RecurrentCheckpointGeometry()
        XCTAssertEqual(geometry.recordHistorical(range: 0 ..< 2048, promptLength: prompt, packed: false), [1024, 2048])
        // Cap changes 2048 -> 512 when decode company arrives.
        XCTAssertEqual(geometry.recordHistorical(range: 2048 ..< 2560, promptLength: prompt, packed: false), [])
        XCTAssertEqual(geometry.recordHistorical(range: 2560 ..< 3072, promptLength: prompt, packed: false), [3072])
        XCTAssertEqual(geometry.recordHistorical(range: 3072 ..< 3584, promptLength: prompt, packed: false), [])
        XCTAssertEqual(geometry.recordHistorical(range: 3584 ..< 4096, promptLength: prompt, packed: false), [4096])
        XCTAssertTrue(geometry.isArmed)
    }

    func testRaggedFinalRangeStillYieldsItsAlignedBoundary() {
        var geometry = CBv2RecurrentCheckpointGeometry()
        XCTAssertEqual(geometry.recordHistorical(range: 0 ..< 2048, promptLength: prompt, packed: false), [1024, 2048])
        XCTAssertEqual(geometry.recordHistorical(range: 2048 ..< 4096, promptLength: prompt, packed: false), [3072, 4096])
        XCTAssertEqual(geometry.recordHistorical(range: 4096 ..< 6144, promptLength: prompt, packed: false), [5120, 6144])
        // The 556-token tail contains no further multiple of 1,024.
        XCTAssertEqual(geometry.recordHistorical(range: 6144 ..< 6700, promptLength: prompt, packed: false), [])
        XCTAssertTrue(geometry.isArmed)
        var tail = CBv2RecurrentCheckpointGeometry(position: 3072, chunkSize: 2048)
        XCTAssertEqual(tail.recordHistorical(range: 3072 ..< 4200, promptLength: 4200, packed: false), [4096])
    }

    func testAdoptedRequestContinuesFromItsRestoredBoundary() {
        var geometry = CBv2RecurrentCheckpointGeometry(position: 3072, chunkSize: 1024)
        XCTAssertEqual(geometry.recordHistorical(range: 3072 ..< 5120, promptLength: prompt, packed: false), [4096, 5120])
        XCTAssertEqual(geometry.recordHistorical(range: 5120 ..< 5632, promptLength: prompt, packed: false), [])
        XCTAssertEqual(geometry.recordHistorical(range: 5632 ..< 6144, promptLength: prompt, packed: false), [6144])
    }

    func testPackedGapAndOverrunDisarmForTheRestOfThePrompt() {
        var packed = CBv2RecurrentCheckpointGeometry()
        XCTAssertEqual(packed.recordHistorical(range: 0 ..< 1024, promptLength: prompt, packed: true), [])
        XCTAssertFalse(packed.isArmed)
        XCTAssertEqual(packed.recordHistorical(range: 1024 ..< 2048, promptLength: prompt, packed: false), [])

        var gap = CBv2RecurrentCheckpointGeometry()
        XCTAssertEqual(gap.recordHistorical(range: 0 ..< 1024, promptLength: prompt, packed: false), [1024])
        XCTAssertEqual(gap.recordHistorical(range: 1536 ..< 2048, promptLength: prompt, packed: false), [])
        XCTAssertFalse(gap.isArmed)

        var overrun = CBv2RecurrentCheckpointGeometry()
        XCTAssertEqual(overrun.recordHistorical(range: 0 ..< 2048, promptLength: 1500, packed: false), [])
        XCTAssertFalse(overrun.isArmed)

        var empty = CBv2RecurrentCheckpointGeometry()
        XCTAssertEqual(empty.recordHistorical(range: 0 ..< 0, promptLength: prompt, packed: false), [])
        XCTAssertFalse(empty.isArmed)
    }

    func testCustomStrideAlignsToItself() {
        var geometry = CBv2RecurrentCheckpointGeometry()
        XCTAssertEqual(geometry.recordHistorical(range: 0 ..< 100, promptLength: 400, packed: false, stride: 32),
                       [32, 64, 96])
        XCTAssertEqual(geometry.recordHistorical(range: 100 ..< 128, promptLength: 400, packed: false, stride: 32),
                       [128])
        XCTAssertEqual(geometry.recordHistorical(range: 128 ..< 129, promptLength: 400, packed: false, stride: 1), [])
        XCTAssertFalse(geometry.isArmed, "a stride of one is never a checkpoint geometry")
    }

    /// Recurrent capture is chunk-agnostic: a boundary is any contiguous
    /// range end aligned to the 256-token stride and the query block,
    /// whatever chunk produced it or the ranges before it. The parity
    /// experiment showed the state at such a boundary is bit-identical on
    /// the dense Qwen target across every partition.
    func testRecurrentBoundariesAtEveryAlignedRangeEndAcrossMixedSchedules() {
        func ends(_ schedule: [Int], prompt: Int) -> (captured: [Int], geometry: CBv2RecurrentCheckpointGeometry) {
            var geometry = CBv2RecurrentCheckpointGeometry()
            var captured: [Int] = []
            var position = 0
            for cap in schedule {
                let upper = min(position + cap, prompt)
                if geometry.record(range: position ..< upper, cap: cap, promptLength: prompt, packed: false) {
                    captured.append(upper)
                }
                XCTAssertTrue(geometry.isArmed, "schedule \(schedule) disarmed at \(position)")
                XCTAssertNil(geometry.disarmReason)
                XCTAssertEqual(geometry.chunkSize, cap, "the last range's cap is the manifest's provenance")
                position = upper
            }
            return (captured, geometry)
        }
        // A 2,048 stripe, then company arrives and the rest runs in 512s.
        XCTAssertEqual(ends([2048, 512, 512, 512, 512, 512], prompt: 4608).captured,
                       [2048, 2560, 3072, 3584, 4096, 4608])
        // 512s under company, then the company leaves and stripes resume.
        XCTAssertEqual(ends([512, 512, 512, 512, 2048, 2048], prompt: 6144).captured,
                       [512, 1024, 1536, 2048, 4096, 6144])
        // 512, 2,048, 2,048 and a ragged tail: the tail is no boundary but
        // does not disarm either.
        let ragged = ends([512, 2048, 2048, 2048], prompt: 4700)
        XCTAssertEqual(ragged.captured, [512, 2560, 4608])
        XCTAssertTrue(ragged.geometry.isArmed)
        XCTAssertEqual(ragged.geometry.position, 4700)
        // A range end off the stride is a boundary only as the end of a
        // full chunk of its own cap (the old rule's boundaries): 384 is,
        // 512 = 384 + 128 is 256-aligned, 612 = 512 + 100 is neither.
        XCTAssertEqual(ends([384, 128, 100, 412], prompt: 1024).captured, [384, 512, 1024])
        // The production 4,096 stripe alone.
        XCTAssertEqual(ends([4096, 4096], prompt: 9171).captured, [4096, 8192])
    }

    func testRecurrentCapChangeNoLongerDisarms() {
        var geometry = CBv2RecurrentCheckpointGeometry()
        XCTAssertTrue(geometry.record(range: 0 ..< 4096, cap: 4096, promptLength: prompt, packed: false))
        XCTAssertTrue(geometry.record(range: 4096 ..< 4608, cap: 512, promptLength: prompt, packed: false))
        XCTAssertTrue(geometry.isArmed)
        XCTAssertNil(geometry.disarmReason, "a cap change is not a disarm reason")
        XCTAssertEqual(geometry.chunkSize, 512)
        // An adopter restored under one chunk continues under another.
        var adopter = CBv2RecurrentCheckpointGeometry(position: 1024, chunkSize: 1024)
        XCTAssertTrue(adopter.record(range: 1024 ..< 3072, cap: 2048, promptLength: prompt, packed: false))
        XCTAssertNil(adopter.disarmReason)
    }

    /// Packing, a non-contiguous range and an overrun still disarm for the
    /// rest of the prompt, each with its own reason, reported once.
    func testRecurrentPackingGapAndOverrunStillDisarm() {
        var packed = CBv2RecurrentCheckpointGeometry()
        XCTAssertFalse(packed.record(range: 0 ..< 512, cap: 512, promptLength: prompt, packed: true))
        XCTAssertEqual(packed.disarmReason, .packed)
        XCTAssertFalse(packed.record(range: 512 ..< 1024, cap: 512, promptLength: prompt, packed: false))
        XCTAssertEqual(packed.disarmReason, .packed, "the reason is set once")

        var gap = CBv2RecurrentCheckpointGeometry(position: 2048, chunkSize: 2048)
        XCTAssertFalse(gap.record(range: 4096 ..< 6144, cap: 2048, promptLength: prompt, packed: false))
        XCTAssertEqual(gap.disarmReason, .geometry)

        var overrun = CBv2RecurrentCheckpointGeometry()
        XCTAssertTrue(overrun.record(range: 0 ..< 1024, cap: 1024, promptLength: 1024, packed: false))
        // The decode range past the last prompt token.
        XCTAssertFalse(overrun.record(range: 1024 ..< 1025, cap: 1, promptLength: 1024, packed: false))
        XCTAssertEqual(overrun.disarmReason, .geometry)
    }

    func testRecurrentStrideSeamAndQueryBlockAlignment() {
        // A smaller stride (test fixtures) still requires query-block alignment.
        let block = CBv2AttentionV1.queryBlockSize
        XCTAssertTrue(CBv2RecurrentCheckpointGeometry.isRecurrentBoundary(256))
        XCTAssertFalse(CBv2RecurrentCheckpointGeometry.isRecurrentBoundary(384))
        XCTAssertFalse(CBv2RecurrentCheckpointGeometry.isRecurrentBoundary(0))
        if block > 1 {
            XCTAssertTrue(CBv2RecurrentCheckpointGeometry.isRecurrentBoundary(block, stride: block))
            XCTAssertFalse(CBv2RecurrentCheckpointGeometry.isRecurrentBoundary(block / 2, stride: block / 2),
                           "a stride below the query block cannot outrun the block alignment")
        }
        var geometry = CBv2RecurrentCheckpointGeometry()
        XCTAssertTrue(geometry.record(range: 0 ..< 100, cap: 100, promptLength: prompt, packed: false, stride: 256),
                      "a full chunk end of its own cap, whatever the block")
        XCTAssertTrue(CBv2RecurrentCheckpointGeometry.isRecurrentBoundary(384, chunkSize: 384))
        XCTAssertFalse(CBv2RecurrentCheckpointGeometry.isRecurrentBoundary(640, chunkSize: 100, stride: 256))
        // A chunk of 1 (as the overrun case above records) has no chunk
        // clause: only the stride rule applies, else every position would
        // be a boundary.
        XCTAssertFalse(CBv2RecurrentCheckpointGeometry.isRecurrentBoundary(7, chunkSize: 1))
        XCTAssertFalse(CBv2RecurrentCheckpointGeometry.isRecurrentBoundary(1025, chunkSize: 1))
        XCTAssertTrue(CBv2RecurrentCheckpointGeometry.isRecurrentBoundary(1024, chunkSize: 1))
    }
}
