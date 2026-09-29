// Copyright © 2026 Eigen Labs.
import MLX
import MLXLMCommon

extension MiMoV26CBv2Adapter: MiMoV26PrefillDefaultProviding {
    package var cbv2MiMoAutomaticPrefillMaximumTokens: Int? {
        let config = target.configuration
        guard cbv2MiMoBlockBatchLayerCount != nil,
            config.hiddenSize == 4096, config.moeIntermediateSize == 2048,
            config.routedExpertCount == 256, config.expertsPerToken == 8,
            config.maxPositionEmbeddings >= 4096,
            let types = cbv2CompleteCheckpointKVDTypes,
            types.count == layerKinds.count,
            types.allSatisfy({ $0 == .bfloat16 || $0 == .float16 })
        else { return nil }
        let experts = target.model.layers.compactMap { $0.mlp as? MiMoV26MoE }
        guard !experts.isEmpty else { return nil }
        // 4096 tokens keep native top-8 routing at <=32768 sorted rows.
        // The wider8192 profile requires the existing requested oversized
        // gather to match every actual loaded expert projection.
        return config.maxPositionEmbeddings >= 8192
            && experts.allSatisfy { $0.switchMLP.mimoV26SupportsOversizedPrefillGather }
            ? 8192 : 4096
    }
}
