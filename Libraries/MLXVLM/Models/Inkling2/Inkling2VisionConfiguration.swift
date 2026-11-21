// Copyright © 2026 Eigen Labs.
//
// Inkling 2 vision configuration (`vision_config` of `inkling2_loop`).
// Reference: transformers PR 51050 `Inkling2VisionConfig`, vLLM PR 69028
// `nvidia/vision.py`, SGLang PR 51926 `inkling2_vision.py`.

import Foundation
import MLXLMCommon

/// `vision_config` of an Inkling 2 checkpoint. Field names follow the
/// checkpoint (`native_patch_size`, `adapter_merge_size`); the transformers
/// aliases (`patch_size`, `spatial_merge_size`) decode too.
public struct Inkling2VisionConfiguration: Codable, Sendable, Equatable {
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var attentionHeads: Int
    public var mlpRatio: Float
    public var patchSize: Int
    public var mergeSize: Int
    public var temporalPatchSize: Int
    public var positionGridSize: Int
    public var projectionHiddenSize: Int
    public var projectorBias: Bool
    public var textHiddenSize: Int
    public var layerNormEps: Float
    public var ropeTheta: Float

    public var headDim: Int { hiddenSize / attentionHeads }
    public var intermediateSize: Int { Int((Float(hiddenSize) * mlpRatio).rounded()) }
    /// Pixels per side of one language-model image token (`patch * merge`).
    public var tokenSide: Int { patchSize * mergeSize }
    public var patchFeatures: Int { 3 * patchSize * patchSize }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case mlpRatio = "mlp_ratio"
        case nativePatchSize = "native_patch_size"
        case patchSize = "patch_size"
        case adapterMergeSize = "adapter_merge_size"
        case spatialMergeSize = "spatial_merge_size"
        case temporalPatchSize = "temporal_patch_size"
        case positionGridSize = "posemb_grid_size"
        case projectionHiddenSize = "projection_hidden_dim"
        case projectorBias = "projector_bias"
        case textHiddenSize = "text_hidden_size"
        case layerNormEps = "layer_norm_eps"
        case ropeParameters = "rope_parameters"
    }

    public init(
        hiddenSize: Int = 1024, hiddenLayers: Int = 23, attentionHeads: Int = 16,
        mlpRatio: Float = 4, patchSize: Int = 20, mergeSize: Int = 2,
        temporalPatchSize: Int = 2, positionGridSize: Int = 32,
        projectionHiddenSize: Int = 4096, projectorBias: Bool = false,
        textHiddenSize: Int = 2304, layerNormEps: Float = 1e-5, ropeTheta: Float = 10_000
    ) {
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.mlpRatio = mlpRatio
        self.patchSize = patchSize
        self.mergeSize = mergeSize
        self.temporalPatchSize = temporalPatchSize
        self.positionGridSize = positionGridSize
        self.projectionHiddenSize = projectionHiddenSize
        self.projectorBias = projectorBias
        self.textHiddenSize = textHiddenSize
        self.layerNormEps = layerNormEps
        self.ropeTheta = ropeTheta
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = Inkling2VisionConfiguration()
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? defaults.hiddenSize
        hiddenLayers =
            try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? defaults.hiddenLayers
        attentionHeads =
            try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? defaults.attentionHeads
        mlpRatio = try c.decodeIfPresent(Float.self, forKey: .mlpRatio) ?? defaults.mlpRatio
        patchSize =
            try c.decodeIfPresent(Int.self, forKey: .nativePatchSize)
            ?? c.decodeIfPresent(Int.self, forKey: .patchSize) ?? defaults.patchSize
        mergeSize =
            try c.decodeIfPresent(Int.self, forKey: .adapterMergeSize)
            ?? c.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? defaults.mergeSize
        temporalPatchSize =
            try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize)
            ?? defaults.temporalPatchSize
        positionGridSize =
            try c.decodeIfPresent(Int.self, forKey: .positionGridSize)
            ?? defaults.positionGridSize
        projectionHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .projectionHiddenSize)
            ?? defaults.projectionHiddenSize
        projectorBias =
            try c.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? defaults.projectorBias
        textHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .textHiddenSize) ?? defaults.textHiddenSize
        layerNormEps =
            try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? defaults.layerNormEps
        let rope = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
        ropeTheta = rope?["rope_theta"]?.asFloat() ?? defaults.ropeTheta
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(mlpRatio, forKey: .mlpRatio)
        try c.encode(patchSize, forKey: .nativePatchSize)
        try c.encode(mergeSize, forKey: .adapterMergeSize)
        try c.encode(temporalPatchSize, forKey: .temporalPatchSize)
        try c.encode(positionGridSize, forKey: .positionGridSize)
        try c.encode(projectionHiddenSize, forKey: .projectionHiddenSize)
        try c.encode(projectorBias, forKey: .projectorBias)
        try c.encode(textHiddenSize, forKey: .textHiddenSize)
        try c.encode(layerNormEps, forKey: .layerNormEps)
        try c.encode(
            ["rope_type": StringOrNumber.string("axial"), "rope_theta": .float(ropeTheta)],
            forKey: .ropeParameters)
    }

    /// Rejects geometry the tower cannot represent. The 2D rotary splits each
    /// head into width and height halves of interleaved pairs, so the head
    /// width must be a multiple of four.
    public func validate() throws {
        guard hiddenSize > 0, hiddenLayers > 0, attentionHeads > 0,
            hiddenSize % attentionHeads == 0, headDim % 4 == 0,
            patchSize > 0, mergeSize > 0, positionGridSize > 0,
            projectionHiddenSize > 0, textHiddenSize > 0,
            mlpRatio > 0, intermediateSize > 0, layerNormEps > 0, ropeTheta > 0
        else {
            throw VLMError.processing("Unsupported Inkling 2 vision configuration")
        }
    }
}
