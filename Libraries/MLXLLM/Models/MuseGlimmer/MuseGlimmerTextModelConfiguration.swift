// Copyright © 2026 Eigen Labs.

import Foundation
import MLXLMCommon

/// The checkpoint's `text_config`: a dense decoder whose layers are sliding or
/// full attention, with RoPE on the layers whose `layer_rope_theta` is not 0.
/// Keys and defaults follow the MLXVLM `MuseGlimmerTextConfiguration`.
public struct MuseGlimmerTextModelConfiguration: Codable, Sendable, Equatable {
    public enum Failure: Error, Equatable { case invalid(String) }

    public static let slidingAttention = "sliding_attention"
    public static let fullAttention = "full_attention"

    public let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDimensions: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let postNormEps: Float
    public let attentionBias: Bool
    public let slidingWindow: Int
    public let qkScaleFactor: Float
    public let outputMultiplier: Float
    public let finalLogitSoftcapping: Float
    public let tieWordEmbeddings: Bool
    /// `sliding_attention` or `full_attention` per layer.
    public let layerTypes: [String]
    /// RoPE base per layer; 0 means the layer has no position encoding.
    public let layerRopeTheta: [Float]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDimensions = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case postNormEps = "post_norm_eps"
        case attentionBias = "attention_bias"
        case slidingWindow = "sliding_window"
        case qkScaleFactor = "qk_scale_factor"
        case outputMultiplier = "output_multiplier"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case layerRopeTheta = "layer_rope_theta"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "muse_glimmer_text"
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 202048
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 6656
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 19968
        let layers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 52
        hiddenLayers = layers
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 2
        headDimensions = try c.decodeIfPresent(Int.self, forKey: .headDimensions) ?? 128
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        postNormEps = try c.decodeIfPresent(Float.self, forKey: .postNormEps) ?? 1e-8
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 2048
        qkScaleFactor = try c.decodeIfPresent(Float.self, forKey: .qkScaleFactor) ?? 3.87
        outputMultiplier =
            try c.decodeIfPresent(Float.self, forKey: .outputMultiplier)
            ?? 0.196_116_135_138_184_04
        finalLogitSoftcapping =
            try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 20.0
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        let rope = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
        // Full attention every fourth layer counted back from the last.
        let types =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? (0 ..< layers).map {
                (layers - 1 - $0) % 4 == 0 ? Self.fullAttention : Self.slidingAttention
            }
        layerTypes = types
        let baseTheta = rope?["rope_theta"]?.asFloat() ?? 500_000.0
        layerRopeTheta =
            try c.decodeIfPresent([Float].self, forKey: .layerRopeTheta)
            ?? types.map { $0 == Self.fullAttention ? 0 : baseTheta }
        try validate()
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(vocabularySize, forKey: .vocabularySize)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(kvHeads, forKey: .kvHeads)
        try c.encode(headDimensions, forKey: .headDimensions)
        try c.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(postNormEps, forKey: .postNormEps)
        try c.encode(attentionBias, forKey: .attentionBias)
        try c.encode(slidingWindow, forKey: .slidingWindow)
        try c.encode(qkScaleFactor, forKey: .qkScaleFactor)
        try c.encode(outputMultiplier, forKey: .outputMultiplier)
        try c.encode(finalLogitSoftcapping, forKey: .finalLogitSoftcapping)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try c.encode(layerTypes, forKey: .layerTypes)
        try c.encode(layerRopeTheta, forKey: .layerRopeTheta)
    }

    /// Decodes `text_config` from a whole `config.json`.
    public static func textConfiguration(checkpointConfiguration data: Data) throws -> Self {
        struct Root: Decodable {
            let text: MuseGlimmerTextModelConfiguration
            enum CodingKeys: String, CodingKey { case text = "text_config" }
        }
        return try JSONDecoder().decode(Root.self, from: data).text
    }

    /// The forward and the CBv2 layer table assume every bound checked here.
    public func validate() throws {
        func require(_ condition: Bool, _ reason: String) throws {
            guard condition else { throw Failure.invalid(reason) }
        }
        try require(
            vocabularySize > 0 && hiddenSize > 0 && intermediateSize > 0 && hiddenLayers > 0,
            "dimensions")
        try require(
            attentionHeads > 0 && kvHeads > 0 && attentionHeads % kvHeads == 0
                && headDimensions > 0 && headDimensions % 2 == 0, "attention geometry")
        try require(slidingWindow > 0 && maxPositionEmbeddings > 0, "window")
        try require(
            layerTypes.count == hiddenLayers && layerRopeTheta.count == hiddenLayers,
            "per-layer tables")
        try require(
            layerTypes.allSatisfy { $0 == Self.slidingAttention || $0 == Self.fullAttention },
            "layer type")
        try require(layerRopeTheta.allSatisfy { $0.isFinite && $0 >= 0 }, "rope theta")
        try require(
            rmsNormEps.isFinite && rmsNormEps > 0 && postNormEps.isFinite && postNormEps > 0
                && qkScaleFactor.isFinite && qkScaleFactor > 0 && outputMultiplier.isFinite
                && finalLogitSoftcapping.isFinite && finalLogitSoftcapping >= 0, "scalars")
        try require(!attentionBias && !tieWordEmbeddings, "bias or tied embeddings")
    }

    public func isSliding(_ layer: Int) -> Bool { layerTypes[layer] == Self.slidingAttention }
}
