// Copyright © 2026 Eigen Labs.
import XCTest
@testable import MLXLMCommon

/// Pure descriptor/checked-arithmetic controls; no MLX arrays or GPU calls.
final class MiMoV26NAXKeyRangeNativePlanTests: XCTestCase {
    private func plan(_ q: Int = 512, _ k: Int = 16_384, window: Int? = nil)
        -> MiMoV26BlockBatchAttention.Plan? {
        MiMoV26BlockBatchAttention.layout(queries: q, keys: k, heads: 64, kvHeads: 4,
            window: window, admittedMaximumQueries: 8192)
    }

    func testActualGroupedOccupancyDoesNotLowerTheOriginalScalarThreshold() throws {
        XCTAssertEqual(MiMoV26NAXAttentionKeyRanges.balancedEdges(
            batch: 1, heads: 64, queries: 128, keys: 16_384), [0, 512])
        for q in [256, 384, 448] {
            let value = try XCTUnwrap(plan(q))
            XCTAssertNil(MiMoV26NAXAttentionKeyRanges.projectedGroupedBytes(plan: value,
                maximumBufferBytes: 1 << 30, upperBound: { $0 }))
            XCTAssertTrue(try XCTUnwrap(MiMoV26NAXAttentionKeyRanges.groupedSchedules(plan: value)).isEmpty)
        }
        let value = try XCTUnwrap(plan())
        let groups = try XCTUnwrap(MiMoV26NAXAttentionKeyRanges.groupedSchedules(plan: value))
        let group = try XCTUnwrap(groups[0])
        XCTAssertEqual(groups.count, 1)
        XCTAssertEqual(group.descriptors.reduce(0) { $0 + 64 * (($1.queryCount + 63) / 64) }, 512)
        XCTAssertEqual(group.dispatches, 6)
        XCTAssertEqual(group.fullStateBuffers, 5)
        XCTAssertEqual(group.stateElements, 4 * 64 * 2 * 64 * 130)
        XCTAssertEqual(Set(group.stateOffsets).count, 4)
        for (d, schedule) in zip(group.descriptors, group.schedules) {
            XCTAssertEqual(schedule.batch, 1)
            XCTAssertEqual(schedule.keys, d.keyCount)
            XCTAssertEqual(schedule.edges,
                MiMoV26NAXAttentionKeyRanges.balancedEdges(batch: 4, heads: 64,
                    queries: 128, keys: d.keyCount))
        }
        XCTAssertEqual(group.descriptors.map(\.keyCount), [16_000, 16_128, 16_256, 16_384])
    }

    func testWindowRaggedAndRangeCountBoundaryKeepExistingFallbacks() throws {
        let windowed = try XCTUnwrap(plan(512, 639, window: 128))
        XCTAssertNil(MiMoV26NAXAttentionKeyRanges.projectedGroupedBytes(plan: windowed,
            maximumBufferBytes: 1 << 30, upperBound: { $0 }))
        XCTAssertNil(plan(513), "the existing <=8-token grouped tail exclusion is unchanged")
        let mixedRanges = try XCTUnwrap(plan(512, 12_544))
        XCTAssertNil(MiMoV26NAXAttentionKeyRanges.projectedGroupedBytes(plan: mixedRanges,
            maximumBufferBytes: 1 << 30, upperBound: { $0 }))
        let raggedGroups = try XCTUnwrap(plan(640))
        let ranged = try XCTUnwrap(MiMoV26NAXAttentionKeyRanges.groupedSchedules(plan: raggedGroups))
        XCTAssertEqual(ranged.count, 1)
        XCTAssertNotNil(ranged[0])
        XCTAssertNil(ranged[1], "one remaining q128 block is not fictional occupancy512")
    }

    func testProjectionPricesEachRealStateAllocationAndRefusesOverflowOrSmallBuffer() throws {
        let value = try XCTUnwrap(plan())
        let logical = try XCTUnwrap(MiMoV26NAXAttentionKeyRanges.projectedGroupedBytes(
            plan: value, maximumBufferBytes: 1 << 30, upperBound: { $0 }))
        let states = 4 * 64 * 2 * 64 * 130 * 4 * 5
        XCTAssertGreaterThan(logical, states)
        let rounded = try XCTUnwrap(MiMoV26NAXAttentionKeyRanges.projectedGroupedBytes(
            plan: value, maximumBufferBytes: 1 << 30, upperBound: { $0 + 4096 }))
        // 5states +Q+O+5dummies+2state scalars+6params+sinks+mask+2scale arrays.
        XCTAssertEqual(rounded - logical, 24 * 4096)
        XCTAssertNil(MiMoV26NAXAttentionKeyRanges.projectedGroupedBytes(
            plan: value, maximumBufferBytes: 16 << 20, upperBound: { $0 }))
        XCTAssertNil(MiMoV26NAXAttentionKeyRanges.projectedGroupedBytes(
            plan: value, maximumBufferBytes: Int.max, upperBound: { _ in Int.max }))
        XCTAssertNil(MiMoV26NAXAttentionKeyRanges.projectedGroupedBytes(
            plan: value, maximumBufferBytes: 1 << 30, upperBound: { $0 - 1 }))
    }

    func testHoistedKeysAndInvalidGeometryCannotCreateAnAdmittedSchedule() throws {
        let value = try XCTUnwrap(plan())
        var group = value.groups[0]
        let old = group[0]
        group[0] = .init(queryStart: old.queryStart, queryCount: old.queryCount,
            keyStart: 0, keyCount: value.keyCount, causal: old.causal)
        let hoisted = MiMoV26BlockBatchAttention.Plan(queryCount: value.queryCount,
            keyCount: value.keyCount, heads: value.heads, kvHeads: value.kvHeads,
            window: value.window, groups: [group])
        XCTAssertNil(MiMoV26NAXAttentionKeyRanges.groupedSchedules(plan: hoisted))
        XCTAssertNil(MiMoV26NAXAttentionKeyRanges.projectedGroupedBytes(
            plan: hoisted, maximumBufferBytes: 1 << 30, upperBound: { $0 }))
        let overflow = MiMoV26BlockBatchAttention.Plan(queryCount: Int.max, keyCount: Int.max,
            heads: 64, kvHeads: 4, window: nil, groups: value.groups)
        XCTAssertNil(MiMoV26NAXAttentionKeyRanges.groupedSchedules(plan: overflow))
        XCTAssertNil(MiMoV26NAXAttentionKeyRanges.projectedGroupedBytes(
            plan: overflow, maximumBufferBytes: Int.max, upperBound: { $0 }))
    }
}
