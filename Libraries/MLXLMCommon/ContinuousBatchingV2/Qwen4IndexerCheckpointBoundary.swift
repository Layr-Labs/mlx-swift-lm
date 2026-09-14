import MLX

extension CBv2Qwen4IndexerRow {
    /// Read-only logical views at an earlier complete-checkpoint boundary.
    /// Never trim the live row to manufacture a donor. Callers that retain
    /// these beyond the row's lifetime must reserve and compact-copy them;
    /// an array view alone does not transfer backing ownership.
    public func snapshotQwen4Indexer(at position: Int, compressRatio: Int)
        throws -> CBv2Qwen4IndexerSnapshot
    {
        guard position > 0, position <= absoluteOffset, compressRatio > 0 else {
            throw CBv2Qwen4IndexerSnapshotError.incompatible(
                "Qwen4 checkpoint boundary is outside committed history")
        }
        let current = try snapshotQwen4Indexer()
        // Only complete micro-blocks below the requested boundary are reusable.
        // A dense prompt may not have materialized pooled keys at all; retain
        // that exact lazy frontier rather than inventing blocks or recomputing.
        guard current.pooledIndexBlocks <= current.tokenCount / compressRatio else {
            throw CBv2Qwen4IndexerSnapshotError.incompatible(
                "Qwen4 pooled frontier includes an incomplete micro-block")
        }
        let blocks = min(current.pooledIndexBlocks, position / compressRatio)
        let keys = current.indexKeys[0..., 0..<position, 0...]
        let positions = current.positionIds.ndim == 2
            ? current.positionIds[0..., 0..<position]
            : current.positionIds[0..., 0..., 0..<position]
        let pooled = blocks == 0 ? nil : current.pooledIndexKeys.map { $0[0..., 0..<blocks, 0...] }
        return .init(tokenCount: position, indexKeys: keys, positionIds: positions,
                     pooledIndexKeys: pooled, pooledIndexBlocks: blocks)
    }
}
