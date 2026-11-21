// Copyright © 2026 Eigen Labs.

import Foundation

/// Typed failures of the Muse Glimmer DFlash drafter.
public enum MuseGlimmerDFlashError: Error, Equatable, Sendable {
    /// The configuration is missing a value or holds an unsupported one.
    case configuration(String)
    /// A file is missing, is not a regular file or exceeds a bound.
    case artifact(String)
    /// A file's SHA-256 does not match its pin.
    case checksum(String)
    /// The tensor set is not exactly the drafter's.
    case inventory(String)
    /// A tensor has the wrong shape or dtype.
    case shape(String)
    /// A byte bound would be exceeded.
    case budget
    /// The bound target does not fit this drafter.
    case target(String)
    /// A request state is foreign, released or used out of order.
    case state(String)
}

/// Configuration of the `muse_glimmer_assistant` block drafter.
///
/// The drafter has no embedding table and no output head: it reads both from
/// its target. Every layer is a sliding-attention layer over the block and
/// the cached target context.
public struct MuseGlimmerDFlashConfiguration: Codable, Sendable, Equatable {
    public static let modelType = "muse_glimmer_assistant"

    public var hiddenSize: Int
    public var intermediateSize: Int
    public var hiddenLayers: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var headDimensions: Int
    /// Positions of one block: the anchor and `blockSize - 1` mask tokens.
    public var blockSize: Int
    public var maskTokenID: Int
    /// 0-based target layers whose output residual stream is the context.
    public var targetLayerIDs: [Int]
    public var slidingWindow: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var maxPositionEmbeddings: Int

    /// Width of one context feature row: every tapped layer, concatenated.
    public var contextWidth: Int { targetLayerIDs.count * hiddenSize }
    /// Draft tokens one block proposes.
    public var draftTokensPerBlock: Int { blockSize - 1 }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDimensions = "head_dim"
        case blockSize = "block_size"
        case maskTokenID = "mask_token_id"
        case targetLayerIDs = "target_layer_ids"
        case slidingWindow = "sliding_window"
        case rmsNormEps = "rms_norm_eps"
        case ropeParameters = "rope_parameters"
        case maxPositionEmbeddings = "max_position_embeddings"
        case layerTypes = "layer_types"
        case hiddenActivation = "hidden_act"
    }

    private struct RoPEParameters: Codable {
        var theta: Float
        var type: String?
        enum CodingKeys: String, CodingKey {
            case theta = "rope_theta"
            case type = "rope_type"
        }
    }

    public init(
        hiddenSize: Int, intermediateSize: Int, hiddenLayers: Int, attentionHeads: Int,
        kvHeads: Int, headDimensions: Int, blockSize: Int, maskTokenID: Int,
        targetLayerIDs: [Int], slidingWindow: Int, rmsNormEps: Float, ropeTheta: Float,
        maxPositionEmbeddings: Int = 131_072
    ) throws {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.headDimensions = headDimensions
        self.blockSize = blockSize
        self.maskTokenID = maskTokenID
        self.targetLayerIDs = targetLayerIDs
        self.slidingWindow = slidingWindow
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.maxPositionEmbeddings = maxPositionEmbeddings
        try validate()
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let type = try c.decode(String.self, forKey: .modelType)
        guard type == Self.modelType else {
            throw MuseGlimmerDFlashError.configuration("model_type \(type)")
        }
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDimensions = try c.decode(Int.self, forKey: .headDimensions)
        blockSize = try c.decode(Int.self, forKey: .blockSize)
        maskTokenID = try c.decode(Int.self, forKey: .maskTokenID)
        targetLayerIDs = try c.decode([Int].self, forKey: .targetLayerIDs)
        slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        let rope = try c.decode(RoPEParameters.self, forKey: .ropeParameters)
        guard rope.type ?? "default" == "default" else {
            throw MuseGlimmerDFlashError.configuration("rope_type")
        }
        ropeTheta = rope.theta
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        if let types = try c.decodeIfPresent([String].self, forKey: .layerTypes) {
            guard types.count == hiddenLayers, types.allSatisfy({ $0 == "sliding_attention" })
            else { throw MuseGlimmerDFlashError.configuration("layer_types") }
        }
        if let activation = try c.decodeIfPresent(String.self, forKey: .hiddenActivation) {
            guard activation == "silu" else {
                throw MuseGlimmerDFlashError.configuration("hidden_act")
            }
        }
        try validate()
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(Self.modelType, forKey: .modelType)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(kvHeads, forKey: .kvHeads)
        try c.encode(headDimensions, forKey: .headDimensions)
        try c.encode(blockSize, forKey: .blockSize)
        try c.encode(maskTokenID, forKey: .maskTokenID)
        try c.encode(targetLayerIDs, forKey: .targetLayerIDs)
        try c.encode(slidingWindow, forKey: .slidingWindow)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(RoPEParameters(theta: ropeTheta, type: "default"), forKey: .ropeParameters)
        try c.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try c.encode(
            [String](repeating: "sliding_attention", count: hiddenLayers), forKey: .layerTypes)
        try c.encode("silu", forKey: .hiddenActivation)
    }

    /// Refuses geometry the module and its context ring cannot represent.
    public func validate() throws {
        func require(_ condition: Bool, _ field: String) throws {
            guard condition else { throw MuseGlimmerDFlashError.configuration(field) }
        }
        try require(hiddenSize > 0 && hiddenSize <= 1 << 16, "hidden_size")
        try require(intermediateSize > 0 && intermediateSize <= 1 << 18, "intermediate_size")
        try require((1 ... 64).contains(hiddenLayers), "num_hidden_layers")
        try require(headDimensions > 0 && headDimensions % 2 == 0, "head_dim")
        try require(attentionHeads > 0 && attentionHeads <= 1 << 10, "num_attention_heads")
        try require(
            kvHeads > 0 && attentionHeads % kvHeads == 0, "num_key_value_heads")
        try require((2 ... 64).contains(blockSize), "block_size")
        try require(maskTokenID >= 0 && maskTokenID <= Int(Int32.max), "mask_token_id")
        try require(
            (1 ... 16).contains(targetLayerIDs.count)
                && targetLayerIDs.allSatisfy { $0 >= 0 }
                && zip(targetLayerIDs, targetLayerIDs.dropFirst()).allSatisfy { $0 < $1 },
            "target_layer_ids")
        try require(
            slidingWindow >= blockSize && slidingWindow <= 1 << 16, "sliding_window")
        try require(rmsNormEps.isFinite && rmsNormEps > 0, "rms_norm_eps")
        try require(ropeTheta.isFinite && ropeTheta > 1, "rope_theta")
        try require(maxPositionEmbeddings >= blockSize, "max_position_embeddings")
    }
}

extension MuseGlimmerDFlashConfiguration {
    /// `meta-models/Muse-Glimmer-30B-assistant` at revision `c676cce9`.
    public static let museGlimmer30B = try! MuseGlimmerDFlashConfiguration(
        hiddenSize: 6656, intermediateSize: 19968, hiddenLayers: 5, attentionHeads: 32,
        kvHeads: 8, headDimensions: 128, blockSize: 16, maskTokenID: 201_818,
        targetLayerIDs: [1, 13, 25, 37, 49], slidingWindow: 2048, rmsNormEps: 1e-5,
        ropeTheta: 500_000)
}
