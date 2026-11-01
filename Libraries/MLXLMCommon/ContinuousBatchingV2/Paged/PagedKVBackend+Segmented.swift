import MLX

extension PagedKVBackend {
    /// Backing grows in segments of this many bytes as admitted demand grows.
    public static let servingSegmentBytes = PagedKVSegmentLayout.defaultTargetBytes

    /// The serving pool, built one way for every owner (the provider's engine
    /// and the cluster's stage hosts). The caller has already admitted this
    /// grant: the pool starts empty and follows it, with no second
    /// physical-RAM fraction, eager-pool budget or model-demand ceiling.
    /// Native construction validates dtype, owner/borrower geometry, window
    /// rings, per-segment Metal buffer limits and kernel addressability; a
    /// packed `quantization` also validates every quantized layer's format.
    public static func segmented(
        admittedGrantBytes: Int,
        layerKinds: [CBv2LayerKind],
        layerDTypes: [DType],
        maxPrefillChunk: Int,
        nominalMaxSequenceLength: Int,
        maxBufferLength: Int,
        quantization: PagedKVQuantizationConfig? = nil,
        nativeLayerIndices: Set<Int> = [],
        residentPrefixCache: CBv2PagedPrefixCacheConfig? = nil
    ) throws -> PagedKVBackend {
        try PagedKVBackend(
            layerKinds: layerKinds,
            config: PagedKVPoolConfig(
                capacityBytes: admittedGrantBytes,
                dtype: layerDTypes.first ?? .float16,
                maxPrefillChunk: maxPrefillChunk,
                nominalMaxSequenceLength: nominalMaxSequenceLength,
                maxBufferLength: maxBufferLength,
                segmentSizeBytes: servingSegmentBytes,
                layerDTypes: layerDTypes,
                quantization: quantization,
                nativeLayerIndices: nativeLayerIndices),
            slabCommitment: .atFirstAdmission,
            residentPrefixCache: residentPrefixCache)
    }

    /// The storage keys `segmented` derives for these layers, in storage order,
    /// before the pool exists, so a caller can size its grant from their real
    /// bytes per token (`PagedKVGroupKey.bytesPerToken`).
    public static func segmentedGroupKeys(
        layerKinds: [CBv2LayerKind], layerDTypes: [DType],
        quantization: PagedKVQuantizationConfig?, nativeLayerIndices: Set<Int> = []
    ) -> [PagedKVGroupKey] {
        PagedKVGroupKey.storageKeys(
            layerKinds: layerKinds, dtypes: layerDTypes, separateWindow: true,
            quantization: quantization, nativeLayerIndices: nativeLayerIndices)
    }
}
