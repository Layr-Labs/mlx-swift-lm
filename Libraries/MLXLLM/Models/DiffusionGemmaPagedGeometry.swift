import MLXLMCommon

extension DiffusionGemmaTextConfiguration {
    /// Committed encoder storage geometry. Canvas K/V is ephemeral and must
    /// never be appended to these rows until the sampler commits a block.
    public var diffusionPagedLayerKinds: [CBv2LayerKind] {
        layerTypes.enumerated().map { index, kind in
            let windowed = kind == "sliding_attention"
            return .init(
                attention: windowed ? .slidingWindow(slidingWindow) : .full,
                headDim: windowed ? headDimension : globalHeadDimension,
                kvHeads: windowed ? keyValueHeads : (globalKeyValueHeads ?? keyValueHeads),
                queryHeads: attentionHeads, modelLayerIndex: index)
        }
    }
}
