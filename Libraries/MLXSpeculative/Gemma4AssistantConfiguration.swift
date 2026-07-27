// Copyright © 2026 Apple Inc.

import Foundation
import MLXLLM
import MLXLMCommon

/// Configuration for the Gemma 4 Multi-Token Prediction "assistant" drafter.
///
/// Mirrors the HF `Gemma4AssistantConfig` schema (flattened top-level +
/// nested `text_config`). Some fields are drafter-specific
/// (`backboneHiddenSize`, `useOrderedEmbeddings`, `numCentroids`,
/// `centroidIntermediateTopK`, `blockSize`); the nested `textConfig`
/// reuses the same `Gemma4TextConfiguration` that the target model uses,
/// with a post-init clamp that forces `numKvSharedLayers =
/// numHiddenLayers` when the checkpoint omits it (matching HF semantics
/// for drafter checkpoints).
public struct Gemma4AssistantConfiguration: Codable, Sendable {
    public var modelType: String
    public var backboneHiddenSize: Int
    public var useOrderedEmbeddings: Bool
    public var numCentroids: Int
    public var centroidIntermediateTopK: Int
    public var blockSize: Int
    public var textConfig: Gemma4TextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case backboneHiddenSize = "backbone_hidden_size"
        case useOrderedEmbeddings = "use_ordered_embeddings"
        case numCentroids = "num_centroids"
        case centroidIntermediateTopK = "centroid_intermediate_top_k"
        case blockSize = "block_size"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "gemma4_assistant"
        self.backboneHiddenSize = try container.decode(Int.self, forKey: .backboneHiddenSize)
        self.useOrderedEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .useOrderedEmbeddings) ?? false
        self.numCentroids =
            try container.decodeIfPresent(Int.self, forKey: .numCentroids) ?? 2048
        self.centroidIntermediateTopK =
            try container.decodeIfPresent(Int.self, forKey: .centroidIntermediateTopK) ?? 32
        // block_size isn't in the HF config file — it's a runtime hyperparam
        // passed into the round loop. Default 4 matches Google's published
        // recommendation.
        self.blockSize =
            try container.decodeIfPresent(Int.self, forKey: .blockSize) ?? 4

        var textConfig =
            try container.decode(Gemma4TextConfiguration.self, forKey: .textConfig)

        // HF post-init: drafter checkpoints require every layer to consume
        // the target's shared K/V. If the checkpoint omits num_kv_shared_layers
        // (decoder default 20, typically > num_hidden_layers=4 for drafters)
        // or sets it to 0, clamp it to num_hidden_layers so every drafter
        // layer is KV-shared.
        if textConfig.numKvSharedLayers == 0
            || textConfig.numKvSharedLayers > textConfig.numHiddenLayers
        {
            textConfig.numKvSharedLayers = textConfig.numHiddenLayers
        }
        self.textConfig = textConfig
    }
}
