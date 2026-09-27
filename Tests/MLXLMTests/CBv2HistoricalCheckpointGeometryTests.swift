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

    func testRecurrentRuleStillRequiresUniformAlignedChunks() {
        // Qwen keeps the exact-geometry contract: a cap change disarms.
        var mixed = CBv2RecurrentCheckpointGeometry()
        XCTAssertTrue(mixed.record(range: 0 ..< 512, cap: 512, promptLength: prompt, packed: false))
        XCTAssertTrue(mixed.record(range: 512 ..< 1024, cap: 512, promptLength: prompt, packed: false))
        XCTAssertFalse(mixed.record(range: 1024 ..< 3072, cap: 2048, promptLength: prompt, packed: false))
        XCTAssertFalse(mixed.isArmed)
        var ragged = CBv2RecurrentCheckpointGeometry()
        XCTAssertTrue(ragged.record(range: 0 ..< 2048, cap: 2048, promptLength: prompt, packed: false))
        XCTAssertFalse(ragged.record(range: 2048 ..< 2560, cap: 2048, promptLength: prompt, packed: false))
        XCTAssertFalse(ragged.isArmed)
        var misaligned = CBv2RecurrentCheckpointGeometry(position: 1024, chunkSize: 1024)
        XCTAssertFalse(misaligned.record(range: 1024 ..< 3072, cap: 2048, promptLength: prompt, packed: false))
    }

    /// Only a cap change is attributed to the uniform-chunk rule, so its
    /// count sizes what relaxing that rule would recover. Packing and every
    /// other geometry failure keep their own reasons, and a disarmed request
    /// reports its reason once: later ranges return false without a change.
    func testRecurrentDisarmReasonAttributesOnlyCapChangesToTheUniformRule() {
        var solo = CBv2RecurrentCheckpointGeometry()
        XCTAssertNil(solo.disarmReason)
        XCTAssertTrue(solo.record(range: 0 ..< 4096, cap: 4096, promptLength: prompt, packed: false))
        XCTAssertNil(solo.disarmReason)
        // Company arrived: the next range is a plain 512 chunk.
        XCTAssertFalse(solo.record(range: 4096 ..< 4608, cap: 512, promptLength: prompt, packed: false))
        XCTAssertEqual(solo.disarmReason, .chunkSizeChanged)
        XCTAssertFalse(solo.record(range: 4608 ..< 5120, cap: 512, promptLength: prompt, packed: false))
        XCTAssertEqual(solo.disarmReason, .chunkSizeChanged, "the reason is set once")

        // A cap change on a ragged final range is still the cap change.
        var raggedChange = CBv2RecurrentCheckpointGeometry()
        XCTAssertTrue(raggedChange.record(range: 0 ..< 4096, cap: 4096, promptLength: 4400, packed: false))
        XCTAssertFalse(raggedChange.record(range: 4096 ..< 4400, cap: 512, promptLength: 4400, packed: false))
        XCTAssertEqual(raggedChange.disarmReason, .chunkSizeChanged)

        // The ragged final range of a solo prompt under an unchanged cap.
        var ragged = CBv2RecurrentCheckpointGeometry()
        XCTAssertTrue(ragged.record(range: 0 ..< 4096, cap: 4096, promptLength: 6200, packed: false))
        XCTAssertFalse(ragged.record(range: 4096 ..< 6200, cap: 4096, promptLength: 6200, packed: false))
        XCTAssertEqual(ragged.disarmReason, .geometry)

        var packed = CBv2RecurrentCheckpointGeometry()
        XCTAssertFalse(packed.record(range: 0 ..< 512, cap: 512, promptLength: prompt, packed: true))
        XCTAssertEqual(packed.disarmReason, .packed)
        // Packed wins over a simultaneous cap change: the row never ran solo.
        var packedChange = CBv2RecurrentCheckpointGeometry(position: 2048, chunkSize: 2048)
        XCTAssertFalse(packedChange.record(range: 2048 ..< 2560, cap: 512, promptLength: prompt, packed: true))
        XCTAssertEqual(packedChange.disarmReason, .packed)

        var gap = CBv2RecurrentCheckpointGeometry(position: 2048, chunkSize: 2048)
        XCTAssertFalse(gap.record(range: 4096 ..< 6144, cap: 2048, promptLength: prompt, packed: false))
        XCTAssertEqual(gap.disarmReason, .geometry)
        var overrun = CBv2RecurrentCheckpointGeometry()
        XCTAssertFalse(overrun.record(range: 0 ..< 2048, cap: 2048, promptLength: 1000, packed: false))
        XCTAssertEqual(overrun.disarmReason, .geometry)
        // An adopter whose first range is misaligned to the new cap: the cap
        // itself changed from the checkpoint's chunk, and that is the reason.
        var adopter = CBv2RecurrentCheckpointGeometry(position: 1024, chunkSize: 1024)
        XCTAssertFalse(adopter.record(range: 1024 ..< 3072, cap: 2048, promptLength: prompt, packed: false))
        XCTAssertEqual(adopter.disarmReason, .chunkSizeChanged)
    }
}
