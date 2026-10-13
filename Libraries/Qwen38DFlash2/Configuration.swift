// Copyright 2026 bstnxbt
// SPDX-License-Identifier: Apache-2.0
// Swift port of dflash-mlx DFlash2 at the revision recorded in NOTICE.

import Foundation

public enum DFlash2ConfigurationError: Error, Equatable, CustomStringConvertible {
    case invariant(String)

    public var description: String {
        switch self {
        case .invariant(let message): message
        }
    }
}

public struct DFlash2Configuration: Decodable, Equatable, Sendable {
    public let architectures: [String]
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let headDimension: Int
    public let vocabularySize: Int
    public let targetLayers: Int
    public let maxPositionEmbeddings: Int
    public let slidingWindow: Int
    public let layerTypes: [String]
    public let rmsNormEpsilon: Double
    public let activationDType: String
    public let isCausal: Bool
    public let attentionBias: Bool
    public let tieWordEmbeddings: Bool
    public let ropeTheta: Double
    public let ropeType: String
    public let blockSize: Int
    public let targetLayerIDs: [Int]
    public let maskTokenID: Int
    public let convKernelSize: Int
    public let convGroupSize: Int
    public let selectorRank: Int
    public let selectorTopK: Int

    private enum CodingKeys: String, CodingKey {
        case architectures
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case keyValueHeads = "num_key_value_heads"
        case headDimension = "head_dim"
        case vocabularySize = "vocab_size"
        case targetLayers = "num_target_layers"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case rmsNormEpsilon = "rms_norm_eps"
        case activationDType = "dtype"
        case isCausal = "is_causal"
        case attentionBias = "attention_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeParameters = "rope_parameters"
        case dflashConfiguration = "dflash_config"
    }

    private enum RopeKeys: String, CodingKey {
        case theta = "rope_theta"
        case type
        case ropeType = "rope_type"
    }

    private enum DFlashKeys: String, CodingKey {
        case blockSize = "block_size"
        case targetLayerIDs = "target_layer_ids"
        case maskTokenID = "mask_token_id"
        case convKernelSize = "conv_kernel_size"
        case convGroupSize = "conv_group_size"
        case selectorRank = "selector_rank"
        case selectorTopK = "selector_top_k"
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        architectures = try values.decode([String].self, forKey: .architectures)
        modelType = try values.decode(String.self, forKey: .modelType)
        hiddenSize = try values.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try values.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try values.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try values.decode(Int.self, forKey: .attentionHeads)
        keyValueHeads = try values.decode(Int.self, forKey: .keyValueHeads)
        headDimension = try values.decode(Int.self, forKey: .headDimension)
        vocabularySize = try values.decode(Int.self, forKey: .vocabularySize)
        targetLayers = try values.decode(Int.self, forKey: .targetLayers)
        maxPositionEmbeddings = try values.decode(Int.self, forKey: .maxPositionEmbeddings)
        slidingWindow = try values.decode(Int.self, forKey: .slidingWindow)
        layerTypes = try values.decode([String].self, forKey: .layerTypes)
        rmsNormEpsilon = try values.decode(Double.self, forKey: .rmsNormEpsilon)
        activationDType = try values.decode(String.self, forKey: .activationDType)
        isCausal = try values.decode(Bool.self, forKey: .isCausal)
        attentionBias = try values.decode(Bool.self, forKey: .attentionBias)
        tieWordEmbeddings = try values.decode(Bool.self, forKey: .tieWordEmbeddings)

        let rope = try values.nestedContainer(keyedBy: RopeKeys.self, forKey: .ropeParameters)
        ropeTheta = try rope.decode(Double.self, forKey: .theta)
        ropeType =
            try rope.decodeIfPresent(String.self, forKey: .type)
            ?? rope.decodeIfPresent(String.self, forKey: .ropeType)
            ?? "default"

        let dflash = try values.nestedContainer(
            keyedBy: DFlashKeys.self, forKey: .dflashConfiguration)
        blockSize = try dflash.decode(Int.self, forKey: .blockSize)
        targetLayerIDs = try dflash.decode([Int].self, forKey: .targetLayerIDs)
        maskTokenID = try dflash.decode(Int.self, forKey: .maskTokenID)
        convKernelSize = try dflash.decode(Int.self, forKey: .convKernelSize)
        convGroupSize = try dflash.decode(Int.self, forKey: .convGroupSize)
        selectorRank = try dflash.decode(Int.self, forKey: .selectorRank)
        selectorTopK = try dflash.decode(Int.self, forKey: .selectorTopK)
    }

    public func validatePinnedContract() throws {
        let expectedLayerTypes = Array(repeating: "sliding_attention", count: 5)
        let checks: [(String, Bool)] = [
            ("architectures", architectures == ["DFlash2DraftModel"]),
            ("model_type", modelType == "qwen3"),
            ("hidden_size", hiddenSize == 5_120),
            ("num_hidden_layers", hiddenLayers == 5),
            ("intermediate_size", intermediateSize == 17_408),
            ("num_attention_heads", attentionHeads == 32),
            ("num_key_value_heads", keyValueHeads == 8),
            ("head_dim", headDimension == 128),
            ("vocab_size", vocabularySize == 248_320),
            ("num_target_layers", targetLayers == 64),
            ("max_position_embeddings", maxPositionEmbeddings == 262_144),
            ("sliding_window", slidingWindow == 2_048),
            ("layer_types", layerTypes == expectedLayerTypes),
            ("rms_norm_eps", rmsNormEpsilon == 1e-6),
            ("dtype", activationDType == "bfloat16"),
            ("is_causal", !isCausal),
            ("attention_bias", !attentionBias),
            ("tie_word_embeddings", !tieWordEmbeddings),
            ("rope_theta", ropeTheta == 10_000_000),
            ("rope_type", ropeType == "default"),
            ("block_size", blockSize == 8),
            ("target_layer_ids", targetLayerIDs == [5, 19, 33, 47, 61]),
            ("mask_token_id", maskTokenID == 248_070),
            ("conv_kernel_size", convKernelSize == 2),
            ("conv_group_size", convGroupSize == 16),
            ("selector_rank", selectorRank == 256),
            ("selector_top_k", selectorTopK == 16),
        ]
        if let failed = checks.first(where: { !$0.1 }) {
            throw DFlash2ConfigurationError.invariant(
                "DFlash2 pinned configuration mismatch: \(failed.0)")
        }
    }
}
