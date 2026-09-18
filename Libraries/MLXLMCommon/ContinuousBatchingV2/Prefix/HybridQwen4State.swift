import MLX

/// Resident reuse needs the same native QSA side-state as durable reuse.
/// Snapshot views are described before charging; copies are built only after
/// the cache reserves them. The live row remains owned until evaluation ends.
enum CBv2HybridQwen4State {
    static func snapshot(model: any CBv2SteppableModel, layerKinds: [CBv2LayerKind],
                         rows: [CBv2SequenceKV?], position: Int)
        throws -> [Int: CBv2Qwen4IndexerSnapshot] {
        guard let geometries = EngineV2.checkpointQwen4Geometries(model: model, layerKinds: layerKinds)
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        if geometries.isEmpty { return [:] }
        guard rows.count == layerKinds.count else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        var result: [Int: CBv2Qwen4IndexerSnapshot] = [:]
        for geometry in geometries {
            guard let index = layerKinds.indices.first(where: { (layerKinds[$0].modelLayerIndex ?? $0) == geometry.layer }),
                  let row = rows[index] as? any CBv2Qwen4IndexerRow else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            let snapshot = try row.snapshotQwen4Indexer(at: position, compressRatio: geometry.compressRatio)
            _ = try CBv2Qwen4CheckpointTensorCodec.descriptors(snapshot, geometry: geometry)
            result[geometry.layer] = snapshot
        }
        return result
    }

    static func compact(_ snapshots: [Int: CBv2Qwen4IndexerSnapshot]) -> [Int: CBv2Qwen4IndexerSnapshot] {
        snapshots.mapValues { snapshot in
            .init(tokenCount: snapshot.tokenCount,
                  indexKeys: MLX.where(MLXArray(true), snapshot.indexKeys, snapshot.indexKeys),
                  positionIds: MLX.where(MLXArray(true), snapshot.positionIds, snapshot.positionIds),
                  pooledIndexKeys: snapshot.pooledIndexKeys.map { MLX.where(MLXArray(true), $0, $0) },
                  pooledIndexBlocks: snapshot.pooledIndexBlocks)
        }
    }

    static func restore(_ snapshots: [Int: CBv2Qwen4IndexerSnapshot], model: any CBv2SteppableModel,
                        layerKinds: [CBv2LayerKind], rows: [CBv2SequenceKV?]) throws {
        guard let geometries = EngineV2.checkpointQwen4Geometries(model: model, layerKinds: layerKinds),
              Set(snapshots.keys) == Set(geometries.map(\.layer))
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        if geometries.isEmpty { return }
        guard rows.count == layerKinds.count else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        for geometry in geometries {
            guard let snapshot = snapshots[geometry.layer],
                  let index = layerKinds.indices.first(where: { (layerKinds[$0].modelLayerIndex ?? $0) == geometry.layer }),
                  let row = rows[index] as? any CBv2Qwen4IndexerRow else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            _ = try CBv2Qwen4CheckpointTensorCodec.descriptors(snapshot, geometry: geometry)
            try row.restoreQwen4Indexer(snapshot)
        }
    }
}
