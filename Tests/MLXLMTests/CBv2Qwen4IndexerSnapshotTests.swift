// Copyright © 2026 Eigen Labs Inc.

import MLX
import XCTest

@testable import MLXLMCommon

final class CBv2Qwen4IndexerSnapshotTests: XCTestCase {
    func testHistoricalBoundaryPreservesInt64PositionsAndDoesNotTrimDonor() throws {
        let donor = makeRow(tokens: 12)
        donor.qwen4IndexTokenCount = 12
        donor.qwen4IndexKeys = MLXArray((0..<48).map(Float.init), [1, 16, 3])
        let origin = Int64(Int32.max) + 100
        donor.qwen4IndexPositionIds = MLXArray((0..<48).map { origin + Int64($0) }, [3, 1, 16])
        donor.qwen4PooledIndexKeys = MLXArray((0..<12).map(Float.init), [1, 4, 3])
        donor.qwen4PooledIndexBlocks = 3
        let checkpoint = try donor.snapshotQwen4Indexer(at: 7, compressRatio: 4)
        eval(checkpoint.arrays)
        XCTAssertEqual(checkpoint.tokenCount, 7)
        XCTAssertEqual(checkpoint.indexKeys.shape, [1, 7, 3])
        XCTAssertEqual(checkpoint.positionIds.dtype, .int64)
        XCTAssertEqual(checkpoint.positionIds.shape, [3, 1, 7])
        XCTAssertEqual(checkpoint.positionIds.asArray(Int64.self),
            (0..<3).flatMap { plane in (0..<7).map { origin + Int64(plane * 16 + $0) } })
        XCTAssertEqual(checkpoint.pooledIndexBlocks, 1)
        XCTAssertEqual(checkpoint.pooledIndexKeys?.shape, [1, 1, 3])
        XCTAssertEqual(donor.absoluteOffset, 12)
        XCTAssertEqual(donor.qwen4IndexKeys?.shape, [1, 16, 3])
        XCTAssertEqual(donor.qwen4PooledIndexBlocks, 3)
        let early = try donor.snapshotQwen4Indexer(at: 3, compressRatio: 4)
        XCTAssertEqual(early.pooledIndexBlocks, 0)
        XCTAssertNil(early.pooledIndexKeys)
        let fresh = makeRow(tokens: 7)
        try fresh.restoreQwen4Indexer(checkpoint)
        XCTAssertEqual(fresh.qwen4IndexPositionIds?.asArray(Int64.self), checkpoint.positionIds.asArray(Int64.self))
    }

    func testHistoricalBoundaryRejectsFutureAndInvalidCompression() throws {
        let donor = makeRow(tokens: 8)
        donor.qwen4IndexKeys = MLXArray.zeros([1, 8, 3])
        donor.qwen4IndexPositionIds = MLXArray.zeros([1, 8], dtype: .int32)
        XCTAssertThrowsError(try donor.snapshotQwen4Indexer(at: 9, compressRatio: 4))
        XCTAssertThrowsError(try donor.snapshotQwen4Indexer(at: 0, compressRatio: 4))
        XCTAssertThrowsError(try donor.snapshotQwen4Indexer(at: 4, compressRatio: 0))
        let lazy = try donor.snapshotQwen4Indexer(at: 4, compressRatio: 4)
        XCTAssertEqual(lazy.pooledIndexBlocks, 0)
        donor.qwen4PooledIndexKeys = MLXArray.zeros([1, 3, 3])
        donor.qwen4PooledIndexBlocks = 3
        XCTAssertThrowsError(try donor.snapshotQwen4Indexer(at: 4, compressRatio: 4))
    }

    func testSnapshotDropsCapacityAndRestoresThreePlanePositions() throws {
        let donor = makeRow(tokens: 4)
        donor.qwen4IndexTokenCount = 4
        donor.qwen4IndexKeys = MLXArray(
            (0 ..< 24).map(Float.init), [1, 8, 3])
        donor.qwen4IndexPositionIds = MLXArray(
            (0 ..< 24).map(Int32.init), [3, 1, 8])
        donor.qwen4PooledIndexKeys = MLXArray(
            (0 ..< 12).map { Float($0 + 30) }, [1, 4, 3])
        donor.qwen4PooledIndexBlocks = 2

        let snapshot = try donor.snapshotQwen4Indexer()
        eval(snapshot.arrays)
        XCTAssertEqual(snapshot.tokenCount, 4)
        XCTAssertEqual(snapshot.indexKeys.shape, [1, 4, 3])
        XCTAssertEqual(snapshot.positionIds.shape, [3, 1, 4])
        XCTAssertEqual(snapshot.pooledIndexKeys?.shape, [1, 2, 3])
        XCTAssertEqual(snapshot.pooledIndexBlocks, 2)

        let restored = makeRow(tokens: 4)
        try restored.restoreQwen4Indexer(snapshot)
        XCTAssertEqual(
            restored.qwen4IndexKeys?.asArray(Float.self),
            snapshot.indexKeys.asArray(Float.self))
        XCTAssertEqual(
            restored.qwen4IndexPositionIds?.asArray(Int32.self),
            snapshot.positionIds.asArray(Int32.self))
        XCTAssertEqual(
            restored.qwen4PooledIndexKeys?.asArray(Float.self),
            snapshot.pooledIndexKeys?.asArray(Float.self))
        XCTAssertEqual(restored.qwen4PooledIndexBlocks, 2)
    }

    func testTrimPreservesLazyPooledFrontierForDensePrompt() throws {
        let donor = makeRow(tokens: 8)
        donor.qwen4IndexTokenCount = 8
        donor.qwen4IndexKeys = MLXArray.zeros([1, 16, 3])
        donor.qwen4IndexPositionIds = MLXArray.zeros(
            [1, 16], dtype: .int32)

        try donor.trimQwen4Indexer(to: 8, compressRatio: 2)
        let snapshot = try donor.snapshotQwen4Indexer()

        XCTAssertEqual(snapshot.indexKeys.shape, [1, 8, 3])
        XCTAssertEqual(snapshot.positionIds.shape, [1, 8])
        XCTAssertNil(snapshot.pooledIndexKeys)
        XCTAssertEqual(snapshot.pooledIndexBlocks, 0)
    }

    func testSnapshotRejectsIncompleteOrStaleSidecars() throws {
        let row = makeRow(tokens: 4)
        row.qwen4IndexKeys = MLXArray.zeros([1, 4, 3])
        XCTAssertThrowsError(try row.snapshotQwen4Indexer())

        row.qwen4IndexPositionIds = MLXArray.zeros([1, 4], dtype: .int32)
        row.qwen4PooledIndexBlocks = 2
        XCTAssertThrowsError(try row.snapshotQwen4Indexer())
    }

    func testRestoreRequiresFreshMatchingKVFrontier() throws {
        let donor = makeRow(tokens: 4)
        donor.qwen4IndexKeys = MLXArray.zeros([1, 4, 3])
        donor.qwen4IndexPositionIds = MLXArray.zeros([1, 4], dtype: .int32)
        let snapshot = try donor.snapshotQwen4Indexer()

        let wrongOffset = makeRow(tokens: 3)
        XCTAssertThrowsError(try wrongOffset.restoreQwen4Indexer(snapshot))

        let occupied = makeRow(tokens: 4)
        occupied.qwen4IndexKeys = MLXArray.zeros([1, 4, 3])
        XCTAssertThrowsError(try occupied.restoreQwen4Indexer(snapshot))

        let malformed = CBv2Qwen4IndexerSnapshot(
            tokenCount: 4,
            indexKeys: snapshot.indexKeys,
            positionIds: snapshot.positionIds,
            pooledIndexKeys: MLXArray.zeros([1, 1, 3]),
            pooledIndexBlocks: 0)
        let fresh = makeRow(tokens: 4)
        XCTAssertThrowsError(try fresh.restoreQwen4Indexer(malformed))
    }

    private func makeRow(tokens: Int) -> CBv2FullSequenceKV {
        let row = CBv2FullSequenceKV(
            promptLength: 0, maxLength: 16, kvHeads: 1, headDim: 2)
        if tokens > 0 {
            _ = row.update(
                keys: MLXArray.zeros([1, 1, tokens, 2]),
                values: MLXArray.zeros([1, 1, tokens, 2]))
        }
        return row
    }
}
