// Copyright © 2026 Eigen Labs.
import Foundation
import MLXLMCommon

/// Image-policy admission for the inspected dense 9B pack. This profile only
/// selects preprocessing; it does not qualify arbitrary weights or serving.
/// Other Prism geometries retain their existing image/video policy.
enum PrismHadamardImageProfile: Equatable {
    case legacy
    case checkpoint9B

    static func resolve(configurationData: Data) throws -> Self {
        let checkpoint = try JSONDecoder().decode(
            PrismHadamardCheckpointConfiguration.self, from: configurationData)
        let config = try JSONDecoder.json5().decode(Qwen35Configuration.self, from: configurationData)
        let text = config.textConfiguration
        let vision = config.visionConfiguration
        guard checkpoint.hasVision,
              checkpoint.modules.allSatisfy({ $0.block == 4096 }),
              text.hiddenSize == 4096, text.hiddenLayers == 32,
              text.vocabularySize == 248_320, text.intermediateSize == 12_288,
              text.attentionHeads == 16, text.kvHeads == 4, text.headDim == 256,
              text.fullAttentionInterval == 4,
              text.linearNumKeyHeads == 16, text.linearNumValueHeads == 32,
              text.linearKeyHeadDim == 128, text.linearValueHeadDim == 128,
              text.linearConvKernelDim == 4, !text.tieWordEmbeddings,
              vision.depth == 27, vision.hiddenSize == 1152, vision.numHeads == 16,
              vision.intermediateSize == 4304, vision.numPositionEmbeddings == 2304,
              vision.inChannels == 3,
              vision.outHiddenSize == 4096, vision.patchSize == 16,
              vision.spatialMergeSize == 2, vision.temporalPatchSize == 2,
              vision.deepstackVisualIndexes.isEmpty else { return .legacy }
        return .checkpoint9B
    }
}
