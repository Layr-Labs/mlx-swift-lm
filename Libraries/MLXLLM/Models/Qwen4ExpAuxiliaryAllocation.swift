import MLXLMCommon

extension Qwen4ExpTextModel: CBv2TargetAuxiliaryAllocationProviding {
    public var cbv2TargetAuxiliaryAllocationSpecs: [CBv2AuxiliaryAllocationSpec]? {
        let c = configuration
        guard c.indexerHeadDim > 0, c.indexerCompressRatio > 0, c.qsaLayerCount > 0 else { return nil }
        let (raw, overflow) = c.indexerHeadDim.multipliedReportingOverflow(by: 4)
        guard !overflow else { return nil }
        let pooled = raw / c.indexerCompressRatio + (raw % c.indexerCompressRatio == 0 ? 0 : 1)
        // Conservative backing envelope, not an activation-reserve formula:
        // current/successor plus two rolling retained states, each capacity
        // bounded by 2*(logical length + the 8192-token growth step). Float32
        // prices either native key dtype; positions cover all three int64 axes.
        // The checkpoint exporter separately reserves copies it materializes.
        return (0..<c.qsaLayerCount).flatMap { _ in
            [raw, 3 * 8, pooled].map {
                CBv2AuxiliaryAllocationSpec(bytesPerToken: $0, allocationCount: 8,
                    tokenGranularity: 8192, tokenPadding: 8192)
            }
        }
    }
}

extension Qwen4ExpModel: CBv2TargetAuxiliaryAllocationProviding {
    public var cbv2TargetAuxiliaryAllocationSpecs: [CBv2AuxiliaryAllocationSpec]? {
        languageModel.cbv2TargetAuxiliaryAllocationSpecs
    }
}
