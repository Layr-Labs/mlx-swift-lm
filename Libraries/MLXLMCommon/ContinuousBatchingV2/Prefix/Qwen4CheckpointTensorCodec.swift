import MLX

public protocol CBv2Qwen4CheckpointGeometryProviding: AnyObject {
    /// Nil means native geometry could not be verified. No checkpoint reuse.
    var cbv2Qwen4CheckpointGeometries: [CBv2Qwen4CheckpointGeometry]? { get }
}

/// Loaded-model geometry, not values trusted from an imported manifest.
/// The provider/model resolves native projection and pooled-key dtypes before
/// constructing the complete codec. Position axes retain their actual type;
/// the enclosing request identity must also bind text versus multimodal state.
public struct CBv2Qwen4CheckpointGeometry: Sendable, Equatable {
    public let layer: Int
    public let headDim: Int
    public let compressRatio: Int
    public let keyDType: CBv2CheckpointDType
    public let pooledDType: CBv2CheckpointDType

    public init(layer: Int, headDim: Int, compressRatio: Int,
                keyDType: DType, pooledDType: DType) throws {
        guard layer >= 0, headDim > 0, headDim <= Int(Int32.max), compressRatio > 0,
            let keys = CBv2CheckpointDType(keyDType), keys.isFloatingPoint,
            let pooled = CBv2CheckpointDType(pooledDType), pooled.isFloatingPoint
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        self.layer = layer
        self.headDim = headDim
        self.compressRatio = compressRatio
        self.keyDType = keys
        self.pooledDType = pooled
    }
}

/// Pure shape/type validation and lossless representation of QSA side-state.
/// This does not allocate device arrays, evaluate graphs, cast positions, or
/// recompute a derived pool. Complete checkpoint capture/import owns memory.
enum CBv2Qwen4CheckpointTensorCodec {
    static func descriptors(_ snapshot: CBv2Qwen4IndexerSnapshot,
                            geometry: CBv2Qwen4CheckpointGeometry)
        throws -> [CBv2CheckpointTensorDescriptor]
    {
        guard let positions = CBv2CheckpointDType(snapshot.positionIds.dtype),
            snapshot.indexKeys.dtype == geometry.keyDType.mlxDType,
            (snapshot.pooledIndexKeys == nil) == (snapshot.pooledIndexBlocks == 0)
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        var result: [CBv2CheckpointTensorDescriptor] = [
            try .init(role: .indexKeys, layer: geometry.layer,
                      shape: snapshot.indexKeys.shape, dtype: geometry.keyDType),
            try .init(role: .indexPositions, layer: geometry.layer,
                      shape: snapshot.positionIds.shape, dtype: positions),
        ]
        if let pooled = snapshot.pooledIndexKeys {
            guard pooled.dtype == geometry.pooledDType.mlxDType,
                pooled.ndim == 3, pooled.dim(1) == snapshot.pooledIndexBlocks
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            result.append(try .init(role: .pooledIndexKeys, layer: geometry.layer,
                                    shape: pooled.shape, dtype: geometry.pooledDType))
        }
        try validate(result, position: snapshot.tokenCount, geometry: geometry)
        return result
    }

    static func validate(_ descriptors: [CBv2CheckpointTensorDescriptor], position: Int,
                         geometry: CBv2Qwen4CheckpointGeometry) throws {
        guard position > 0, descriptors.count == 2 || descriptors.count == 3,
            descriptors.allSatisfy({ $0.layer == geometry.layer })
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        for descriptor in descriptors { try descriptor.validate() }
        let keys = descriptors[0], positions = descriptors[1]
        guard keys.role == .indexKeys, keys.dtype == geometry.keyDType,
            keys.shape == [1, position, geometry.headDim],
            positions.role == .indexPositions,
            positions.dtype == .int32 || positions.dtype == .int64,
            positions.shape == [1, position] || positions.shape == [3, 1, position]
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        if descriptors.count == 3 {
            let pooled = descriptors[2]
            guard pooled.role == .pooledIndexKeys, pooled.dtype == geometry.pooledDType,
                pooled.shape.count == 3, pooled.shape[0] == 1,
                pooled.shape[1] <= position / geometry.compressRatio,
                pooled.shape[2] == geometry.headDim
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        }
    }

    static func decode(_ arrays: [MLXArray], descriptors: [CBv2CheckpointTensorDescriptor],
                       position: Int, geometry: CBv2Qwen4CheckpointGeometry)
        throws -> CBv2Qwen4IndexerSnapshot
    {
        try validate(descriptors, position: position, geometry: geometry)
        guard arrays.count == descriptors.count,
            zip(arrays, descriptors).allSatisfy({ $0.shape == $1.shape && $0.dtype == $1.dtype.mlxDType })
        else { throw CBv2CompleteCheckpointError.incompleteTransfer }
        return .init(tokenCount: position, indexKeys: arrays[0], positionIds: arrays[1],
                     pooledIndexKeys: arrays.count == 3 ? arrays[2] : nil,
                     pooledIndexBlocks: arrays.count == 3 ? descriptors[2].shape[1] : 0)
    }
}
