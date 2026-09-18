// Qwen4 gathered-QSA cache extras.
//
// Kept off `CBv2AttendingLayerCache` so the frozen integration contract does
// not grow a model-family method. Production contiguous and paged caches both
// conform. Long-context QSA callers refuse a missing capability instead of
// silently replacing sparse selection with dense attention.

import Foundation
import MLX

/// Write K/V and keep the QSA indexer side-state without paying leftover
/// dense SDPA. One request per view; experimental multirow models obtain
/// independent views through `CBv2Qwen4BatchScopeProviding`.
public protocol CBv2Qwen4GatheredCache: AnyObject {
    /// A recorded backend error must unwind through the engine's typed
    /// failure boundary; model-side row scopes may not trap or publish state.
    var qwen4HasWriteFault: Bool { get }
    func qwen4ValidateProjectedKV(keys: MLXArray, values: MLXArray) -> Bool
    /// Mirrors the engine's temporary rectangular-MTP attention contract.
    /// QSA must preserve canonical per-token dispatch when a window crosses
    /// the dense/sparse boundary, not treat that window as ordinary prefill.
    var qwen4SerializesRectangularAttention: Bool { get }

    func updateKVAndAdvanceOffsets(
        keys: MLXArray, values: MLXArray
    ) -> [(keys: MLXArray, values: MLXArray)]

    var qwen4IndexKeys: MLXArray? { get set }
    /// Initialized logical tokens, distinct from allocated capacity. Nil is
    /// accepted only for an exact-width externally constructed sidecar.
    var qwen4IndexTokenCount: Int? { get set }
    var qwen4IndexPositionIds: MLXArray? { get set }
    /// Completed pooled QSA index-key blocks (Fusion `pooled_indexer_keys`).
    var qwen4PooledIndexKeys: MLXArray? { get set }
    var qwen4PooledIndexBlocks: Int { get set }

    func clearQwen4IndexerState()
}

extension CBv2Qwen4GatheredCache {
    public var qwen4HasWriteFault: Bool { false }
    public func qwen4ValidateProjectedKV(keys: MLXArray, values: MLXArray) -> Bool {
        !qwen4HasWriteFault
    }

    public func clearQwen4IndexerState() {
        qwen4IndexKeys = nil
        qwen4IndexTokenCount = nil
        qwen4IndexPositionIds = nil
        qwen4PooledIndexKeys = nil
        qwen4PooledIndexBlocks = 0
    }
}

/// Qwen4 gathered QSA keeps the indexer sidecar on the layer cache. CBv2
/// rebinds that cache with `setRows` after MTP finalize (`eagerCompositionStale`)
/// even when the B1 row object is unchanged. The sidecar therefore also
/// lives on the sequence row so a rebind can restore it. Wiping it makes
/// the next rectangular verify (`L>1` past the budget) miss indexer state
/// and SIGTRAP instead of running gathered QSA.
public protocol CBv2Qwen4IndexerRow: CBv2SequenceKV {
    var qwen4IndexKeys: MLXArray? { get set }
    var qwen4IndexTokenCount: Int? { get set }
    var qwen4IndexPositionIds: MLXArray? { get set }
    var qwen4PooledIndexKeys: MLXArray? { get set }
    var qwen4PooledIndexBlocks: Int { get set }
}

public enum CBv2Qwen4IndexerSnapshotError: Error, Equatable {
    case incomplete(String)
    case incompatible(String)
}

/// Canonical logical QSA side-state for one row at an exact committed token
/// boundary. Capacity slack is deliberately sliced away before persistence.
public struct CBv2Qwen4IndexerSnapshot {
    public let tokenCount: Int
    public let indexKeys: MLXArray
    public let positionIds: MLXArray
    public let pooledIndexKeys: MLXArray?
    public let pooledIndexBlocks: Int

    public init(
        tokenCount: Int,
        indexKeys: MLXArray,
        positionIds: MLXArray,
        pooledIndexKeys: MLXArray?,
        pooledIndexBlocks: Int
    ) {
        self.tokenCount = tokenCount
        self.indexKeys = indexKeys
        self.positionIds = positionIds
        self.pooledIndexKeys = pooledIndexKeys
        self.pooledIndexBlocks = pooledIndexBlocks
    }

    public var arrays: [MLXArray] {
        [indexKeys, positionIds] + (pooledIndexKeys.map { [$0] } ?? [])
    }
}

extension CBv2Qwen4IndexerRow {
    /// Reconcile row-owned QSA side-state after an out-of-band KV rollback.
    /// Capacity slack and rejected columns are dropped. Pooled keys are a
    /// derivative acceleration structure: short dense-QSA prompts may not
    /// materialize them at all, while long/gathered execution does. Preserve
    /// the actually materialized committed frontier and cap it to the blocks
    /// still covered after rollback; missing blocks are rebuilt
    /// deterministically from raw keys and positions when gathered QSA first
    /// needs them.
    public func trimQwen4Indexer(
        to tokenCount: Int,
        compressRatio: Int
    ) throws {
        guard tokenCount == absoluteOffset, tokenCount >= 0, compressRatio > 0 else {
            throw CBv2Qwen4IndexerSnapshotError.incompatible(
                "Qwen4 index trim does not match the KV frontier")
        }
        if tokenCount == 0 {
            qwen4IndexKeys = nil
            qwen4IndexTokenCount = nil
            qwen4IndexPositionIds = nil
            qwen4PooledIndexKeys = nil
            qwen4PooledIndexBlocks = 0
            return
        }
        guard CBv2Qwen4IndexerFrontier.covers(
                tokenCount, count: qwen4IndexTokenCount,
                keys: qwen4IndexKeys, positions: qwen4IndexPositionIds),
            let keys = qwen4IndexKeys,
            keys.ndim == 3,
            keys.dim(1) >= tokenCount,
            let positions = qwen4IndexPositionIds,
            (positions.ndim == 2 || positions.ndim == 3),
            positions.dim(-1) >= tokenCount
        else {
            throw CBv2Qwen4IndexerSnapshotError.incomplete(
                "Qwen4 index trim lacks raw keys or positions")
        }
        qwen4IndexKeys =
            keys.dim(1) == tokenCount
            ? keys
            : keys[0..., 0 ..< tokenCount, 0...]
        if positions.dim(-1) != tokenCount {
            qwen4IndexPositionIds =
                positions.ndim == 2
                ? positions[0..., 0 ..< tokenCount]
                : positions[0..., 0..., 0 ..< tokenCount]
        }
        let completeBlocks = tokenCount / compressRatio
        guard qwen4PooledIndexBlocks >= 0 else {
            throw CBv2Qwen4IndexerSnapshotError.incomplete(
                "Qwen4 pooled index block frontier is negative")
        }
        let blocks = min(qwen4PooledIndexBlocks, completeBlocks)
        if blocks == 0 {
            qwen4PooledIndexKeys = nil
        } else {
            guard let pooled = qwen4PooledIndexKeys,
                pooled.ndim == 3,
                pooled.dim(1) >= blocks
            else {
                throw CBv2Qwen4IndexerSnapshotError.incomplete(
                    "Qwen4 index trim lacks pooled keys")
            }
            qwen4PooledIndexKeys =
                pooled.dim(1) == blocks
                ? pooled
                : pooled[0..., 0 ..< blocks, 0...]
        }
        qwen4PooledIndexBlocks = blocks
        qwen4IndexTokenCount = tokenCount
    }

    /// Snapshot the logical frontier only. A wider capacity bank must never
    /// become durable cache state because its tail is uninitialized or may
    /// contain rejected speculative columns.
    public func snapshotQwen4Indexer() throws -> CBv2Qwen4IndexerSnapshot {
        let tokens = absoluteOffset
        guard tokens > 0,
            CBv2Qwen4IndexerFrontier.covers(
                tokens, count: qwen4IndexTokenCount,
                keys: qwen4IndexKeys, positions: qwen4IndexPositionIds),
            let storedKeys = qwen4IndexKeys,
            storedKeys.ndim == 3,
            storedKeys.dim(1) >= tokens,
            let storedPositions = qwen4IndexPositionIds,
            (storedPositions.ndim == 2 || storedPositions.ndim == 3),
            storedPositions.dim(-1) >= tokens
        else {
            throw CBv2Qwen4IndexerSnapshotError.incomplete(
                "Qwen4 index keys/positions do not cover the committed row")
        }
        let keys =
            storedKeys.dim(1) == tokens
            ? storedKeys
            : storedKeys[0..., 0 ..< tokens, 0...]
        let positions: MLXArray
        if storedPositions.dim(-1) == tokens {
            positions = storedPositions
        } else if storedPositions.ndim == 2 {
            positions = storedPositions[0..., 0 ..< tokens]
        } else {
            positions = storedPositions[0..., 0..., 0 ..< tokens]
        }

        let blocks = qwen4PooledIndexBlocks
        guard blocks >= 0, blocks <= tokens else {
            throw CBv2Qwen4IndexerSnapshotError.incomplete(
                "Qwen4 pooled index block frontier is invalid")
        }
        let pooled: MLXArray?
        if blocks == 0 {
            pooled = nil
        } else {
            guard let stored = qwen4PooledIndexKeys,
                stored.ndim == 3, stored.dim(1) >= blocks
            else {
                throw CBv2Qwen4IndexerSnapshotError.incomplete(
                    "Qwen4 pooled index keys do not cover their block frontier")
            }
            pooled =
                stored.dim(1) == blocks
                ? stored
                : stored[0..., 0 ..< blocks, 0...]
        }
        return CBv2Qwen4IndexerSnapshot(
            tokenCount: tokens,
            indexKeys: keys,
            positionIds: positions,
            pooledIndexKeys: pooled,
            pooledIndexBlocks: blocks)
    }

    /// Restore onto the KV row representing the same exact token frontier.
    public func restoreQwen4Indexer(_ snapshot: CBv2Qwen4IndexerSnapshot) throws {
        guard qwen4IndexKeys == nil,
            qwen4IndexTokenCount == nil,
            qwen4IndexPositionIds == nil,
            qwen4PooledIndexKeys == nil,
            qwen4PooledIndexBlocks == 0,
            absoluteOffset == snapshot.tokenCount,
            snapshot.tokenCount > 0,
            snapshot.indexKeys.ndim == 3,
            snapshot.indexKeys.dim(1) == snapshot.tokenCount,
            (snapshot.positionIds.ndim == 2 || snapshot.positionIds.ndim == 3),
            snapshot.positionIds.dim(-1) == snapshot.tokenCount,
            snapshot.pooledIndexBlocks >= 0,
            snapshot.pooledIndexBlocks <= snapshot.tokenCount
        else {
            throw CBv2Qwen4IndexerSnapshotError.incompatible(
                "Qwen4 index snapshot does not match the adopted KV frontier")
        }
        if snapshot.pooledIndexBlocks > 0 {
            guard let pooled = snapshot.pooledIndexKeys,
                pooled.ndim == 3,
                pooled.dim(1) == snapshot.pooledIndexBlocks
            else {
                throw CBv2Qwen4IndexerSnapshotError.incompatible(
                    "Qwen4 pooled index snapshot is incomplete")
            }
        } else if snapshot.pooledIndexKeys != nil {
            throw CBv2Qwen4IndexerSnapshotError.incompatible(
                "Qwen4 zero-block snapshot unexpectedly contains pooled keys")
        }
        qwen4IndexKeys = snapshot.indexKeys
        qwen4IndexTokenCount = snapshot.tokenCount
        qwen4IndexPositionIds = snapshot.positionIds
        qwen4PooledIndexKeys = snapshot.pooledIndexKeys
        qwen4PooledIndexBlocks = snapshot.pooledIndexBlocks
    }
}

public enum CBv2Qwen4IndexerBind: Sendable {
    public static func preservesSidecar(
        current: [CBv2SequenceKV], next: [CBv2SequenceKV]
    ) -> Bool {
        current.count == 1 && next.count == 1
            && ObjectIdentifier(current[0]) == ObjectIdentifier(next[0])
    }

    public static func harvest(_ cache: CBv2Qwen4GatheredCache, into row: CBv2SequenceKV) {
        guard let row = row as? CBv2Qwen4IndexerRow else { return }
        row.qwen4IndexKeys = cache.qwen4IndexKeys
        row.qwen4IndexTokenCount = cache.qwen4IndexTokenCount.map { min($0, row.absoluteOffset) }
        row.qwen4IndexPositionIds = cache.qwen4IndexPositionIds
        row.qwen4PooledIndexKeys = cache.qwen4PooledIndexKeys
        row.qwen4PooledIndexBlocks = cache.qwen4PooledIndexBlocks
    }

    public static func restore(_ cache: CBv2Qwen4GatheredCache, from row: CBv2SequenceKV) -> Bool {
        guard let row = row as? CBv2Qwen4IndexerRow, row.qwen4IndexKeys != nil else {
            return false
        }
        cache.qwen4IndexKeys = row.qwen4IndexKeys
        cache.qwen4IndexTokenCount = row.qwen4IndexTokenCount
        cache.qwen4IndexPositionIds = row.qwen4IndexPositionIds
        cache.qwen4PooledIndexKeys = row.qwen4PooledIndexKeys
        cache.qwen4PooledIndexBlocks = row.qwen4PooledIndexBlocks
        return true
    }
}
