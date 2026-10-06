import MLX

extension CBv2CheckpointTensorRole {
    var isQwen4Indexer: Bool {
        self == .indexKeys || self == .indexPositions || self == .pooledIndexKeys
    }
}

extension CBv2CompleteCheckpointCodec {
    func qwen4Descriptors(in manifest: CBv2CompleteCheckpointManifest) -> [CBv2CheckpointTensorDescriptor] {
        manifest.tensors.filter { $0.role.isQwen4Indexer }
    }

    func validateQwen4Descriptors(_ descriptors: [CBv2CheckpointTensorDescriptor], position: Int) throws {
        let expectedLayers = layerKinds.enumerated().compactMap { index, kind in
            kind.qwen4IndexerCompressRatio == nil ? nil : (kind.modelLayerIndex ?? index)
        }
        guard qwen4Geometries.map(\.layer) == expectedLayers else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let ratios = layerKinds.compactMap(\.qwen4IndexerCompressRatio)
        guard qwen4Geometries.map(\.compressRatio) == ratios else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        var cursor = 0
        for geometry in qwen4Geometries {
            let group = Array(descriptors.dropFirst(cursor).prefix { $0.layer == geometry.layer })
            try CBv2Qwen4CheckpointTensorCodec.validate(group, position: position, geometry: geometry)
            cursor += group.count
        }
        guard cursor == descriptors.count else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
    }

    func qwen4Descriptors(_ checkpoint: CBv2RecurrentCheckpoint) throws -> [CBv2CheckpointTensorDescriptor] {
        guard checkpoint.qwen4.count == qwen4Geometries.count else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        return try qwen4Geometries.flatMap { geometry in
            guard let snapshot = checkpoint.qwen4[geometry.layer], snapshot.tokenCount == checkpoint.position else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            return try CBv2Qwen4CheckpointTensorCodec.descriptors(snapshot, geometry: geometry)
        }
    }

    func qwen4Arrays(_ checkpoint: CBv2RecurrentCheckpoint) throws -> [MLXArray] {
        _ = try qwen4Descriptors(checkpoint)
        return qwen4Geometries.flatMap { checkpoint.qwen4[$0.layer]!.arrays }
    }

    func qwen4Snapshots(rows: [CBv2SequenceKV?], position: Int) throws -> [Int: CBv2Qwen4IndexerSnapshot] {
        if qwen4Geometries.isEmpty { return [:] }
        guard rows.count == layerKinds.count else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        var snapshots: [Int: CBv2Qwen4IndexerSnapshot] = [:]
        for (index, kind) in layerKinds.enumerated() where kind.qwen4IndexerCompressRatio != nil {
            guard let row = rows[index] as? any CBv2Qwen4IndexerRow else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            snapshots[kind.modelLayerIndex ?? index] = try row.snapshotQwen4Indexer(
                at: position, compressRatio: kind.qwen4IndexerCompressRatio!)
        }
        return snapshots
    }

    /// Caller holds the complete capture reservation before constructing
    /// these compact copies. No precision conversion or pooling is performed.
    func compactQwen4(_ snapshots: [Int: CBv2Qwen4IndexerSnapshot]) throws -> [Int: CBv2Qwen4IndexerSnapshot] {
        var copies: [Int: CBv2Qwen4IndexerSnapshot] = [:]
        for geometry in qwen4Geometries {
            guard let snapshot = snapshots[geometry.layer] else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            let descriptors = try CBv2Qwen4CheckpointTensorCodec.descriptors(snapshot, geometry: geometry)
            copies[geometry.layer] = try CBv2Qwen4CheckpointTensorCodec.decode(
                snapshot.arrays.map { MLX.where(MLXArray(true), $0, $0) },
                descriptors: descriptors, position: snapshot.tokenCount, geometry: geometry)
        }
        return copies
    }

    func restoreQwen4(_ checkpoint: CBv2RecurrentCheckpoint, rows: [CBv2SequenceKV?]) throws {
        guard rows.count == layerKinds.count else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        try validateQwen4Descriptors(qwen4Descriptors(checkpoint), position: checkpoint.position)
        for (index, kind) in layerKinds.enumerated() where kind.qwen4IndexerCompressRatio != nil {
            guard let row = rows[index] as? any CBv2Qwen4IndexerRow,
                  let snapshot = checkpoint.qwen4[kind.modelLayerIndex ?? index] else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            try row.restoreQwen4Indexer(snapshot)
        }
    }
}
