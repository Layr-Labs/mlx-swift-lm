import MLX

/// Checkpoint encoding only. This never selects the live attention backend or
/// changes its page geometry, dtype, admission rates or kernel policy.
public enum CBv2CompleteCheckpointStorageQuantization {
    /// Resolve before constructing a durable store, and use the same result in
    /// its numerical identity and `EngineV2.checkpointQuantization`. Unsupported
    /// layouts retain their existing native codec; live-packed codecs retain
    /// their original format and exact-code round-trip contract.
    public static func resolve(
        _ requested: PagedKVQuantizationConfig?, layerKinds: [CBv2LayerKind],
        layerDTypes: [DType], pagedConfig: PagedKVPoolConfig?, hasAssistantState: Bool
    ) -> PagedKVQuantizationConfig? {
        guard let requested, !hasAssistantState, pagedConfig?.quantization == nil,
            !layerKinds.isEmpty, layerKinds.count == layerDTypes.count,
            layerDTypes.allSatisfy({ [.float16, .bfloat16, .float32].contains($0) }),
            (try? requested.validateParameters()) != nil,
            // Selected-index continuation has a separate native contract.
            layerKinds.allSatisfy({ $0.qwen4IndexerCompressRatio == nil })
        else { return nil }
        if let pagedConfig {
            guard pagedConfig.segmentSizeBytes != nil,
                pagedConfig.layerDTypes == layerDTypes
            else { return nil }
        }
        var hasEncodedOwner = false
        for (index, kind) in layerKinds.enumerated() where kind.sharesKVWithLayer == nil {
            guard kind.kvGeometry != nil else { return nil }
            if pagedConfig?.nativeLayerIndices.contains(index) == true { continue }
            if case .slidingWindow(let size) = kind.attention,
                size <= requested.recentTokenCount
            { continue }
            guard (try? requested.validate(headDim: kind.headDim)) != nil,
                (try? requested.validate(headDim: kind.valueHeadDim)) != nil
            else { return nil }
            hasEncodedOwner = true
        }
        return hasEncodedOwner ? requested : nil
    }
}
