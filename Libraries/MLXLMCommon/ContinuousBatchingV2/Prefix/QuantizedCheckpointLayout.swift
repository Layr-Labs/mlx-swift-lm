import MLX

/// Two canonical role streams per owning layer. Each head stores every
/// token-local code verbatim, followed by its exact native recent band. The
/// resolved pool profile is also part of the provider's numerical identity.
struct CBv2CheckpointPagedRoleLayout {
    let key: PagedKVGroupKey
    let values: Bool
    let position: Int
    let tokenStart: Int
    let nativeCount: Int
    let packedCount: Int
    let packedRowBytes: Int
    let nativeRowBytes: Int
    let bytesPerHead: Int
    var isQuantized: Bool { key.quantization != nil }
    var nativeStart: Int { position - nativeCount }
    var width: Int { values ? key.valueHeadDim : key.headDim }

    init(key: PagedKVGroupKey, position: Int, tokenStart: Int = 0, values: Bool) throws {
        guard position > tokenStart, tokenStart >= 0 else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        self.key = key
        self.position = position
        self.tokenStart = tokenStart
        self.values = values
        let width = values ? key.valueHeadDim : key.headDim
        nativeRowBytes = try PagedKVQuantizationConfig.multiply(width, key.dtype.size)
        if let quantization = key.quantization {
            let layout = try quantization.rowLayout(headDim: width)
            packedRowBytes = values ? layout.valueRowBytes : layout.keyRowBytes
            nativeCount = min(quantization.recentTokenCount, position - tokenStart)
            // The mirror of the recent band is preserved too: those exact
            // bytes become authoritative when these tokens age out. Restore
            // never invokes a second encoder with different rounding.
            packedCount = position - tokenStart
            bytesPerHead = try CBv2CheckpointAllocationFootprint.add(
                PagedKVQuantizationConfig.multiply(packedCount, packedRowBytes),
                PagedKVQuantizationConfig.multiply(nativeCount, nativeRowBytes))
        } else {
            packedRowBytes = nativeRowBytes
            nativeCount = 0
            packedCount = position - tokenStart
            bytesPerHead = try PagedKVQuantizationConfig.multiply(packedCount, packedRowBytes)
        }
    }

    func descriptor(layer: Int) throws -> CBv2CheckpointTensorDescriptor {
        try .init(
            role: values ? .values : .keys, layer: layer,
            shape: isQuantized
                ? [1, key.kvHeads, bytesPerHead]
                : [1, key.kvHeads, position - tokenStart, width],
            dtype: isQuantized ? .uint8 : CBv2CheckpointDType(key.dtype)!)
    }
}

extension CBv2CompleteCheckpointCodec {
    var usesQuantizedCheckpoint: Bool { pagedConfig?.quantization != nil }

    func checkpointGroupKey(layer index: Int) -> PagedKVGroupKey {
        let config = pagedConfig
        return PagedKVGroupKey(
            layerKinds[index], dtype: kvDTypes[index], separateWindow: true,
            quantization: config?.nativeLayerIndices.contains(
                layerKinds[index].sharesKVWithLayer ?? index) == true ? nil : config?.quantization)
    }

    func checkpointTargetDescriptors(position: Int) throws -> [CBv2CheckpointTensorDescriptor] {
        guard identity.isValid, position > 1, !layerKinds.isEmpty,
            kvDTypes.count == layerKinds.count,
            historicalLayout != nil
                || layerKinds.allSatisfy({
                    if case .full = $0.attention { return $0.sharesKVWithLayer == nil }
                    return false
                })
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        return try layerKinds.indices.filter { layerKinds[$0].sharesKVWithLayer == nil }.flatMap {
            index in
            let key = checkpointGroupKey(layer: index)
            let start: Int
            switch layerKinds[index].attention {
            case .full: start = 0
            case .slidingWindow(let size): start = max(0, position - size)
            }
            return try [false, true].map {
                try CBv2CheckpointPagedRoleLayout(
                    key: key, position: position, tokenStart: start, values: $0
                ).descriptor(layer: layerKinds[index].modelLayerIndex ?? index)
            }
        }
    }
}
