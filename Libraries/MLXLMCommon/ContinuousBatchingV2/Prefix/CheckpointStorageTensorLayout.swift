import MLX

/// One logical K or V row on disk. Quantized rows retain the complete packed
/// codes, FP32 scales and offsets; copying them never changes numerical values.
/// Native rows preserve the original tensor shape and byte representation.
struct CBv2CheckpointStorageTensorLayout {
    let rowElements: Int
    let dtype: CBv2CheckpointDType
    var itemSize: Int { dtype.mlxDType.size }

    init(key: PagedKVGroupKey, values: Bool) throws {
        if let quantization = key.quantization {
            let packed = try quantization.rowLayout(headDim: key.headDim)
            rowElements = values ? packed.valueRowBytes : packed.keyRowBytes
            dtype = .uint8
        } else {
            guard let native = CBv2CheckpointDType(key.dtype),
                  [.float16, .bfloat16, .float32].contains(native)
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            rowElements = key.headDim
            dtype = native
        }
    }

    func descriptor(key: PagedKVGroupKey, modelLayer: Int?, position: Int,
                    values: Bool) throws -> CBv2CheckpointTensorDescriptor {
        try .init(role: values ? .values : .keys, layer: modelLayer,
                  shape: [1, key.kvHeads, position, rowElements], dtype: dtype)
    }
}

extension CBv2CompleteCheckpointCodec {
    var kvQuantization: PagedKVQuantizationConfig? { pagedConfig?.quantization }

    func storageKey(layerIndex: Int) -> PagedKVGroupKey {
        PagedKVGroupKey(layerKinds[layerIndex], dtype: kvDTypes[layerIndex],
                        separateWindow: true, quantization: kvQuantization)
    }

    func attentionTensorDescriptors(position: Int) throws -> [CBv2CheckpointTensorDescriptor] {
        guard kvDTypes.count == layerKinds.count else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let indices = historicalLayout?.owningIndices ?? Array(layerKinds.indices)
        return try indices.flatMap { index in
            let key = storageKey(layerIndex: index)
            let start = historicalLayout?.layers[index].tokenStart(at: position) ?? 0
            return try [false, true].map { values in
                try CBv2CheckpointStorageTensorLayout(key: key, values: values)
                    .descriptor(key: key, modelLayer: layerKinds[index].modelLayerIndex ?? index,
                                position: position - start, values: values)
            }
        }
    }
}
