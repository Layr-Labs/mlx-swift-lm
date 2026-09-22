// Native DiffusionGemma text geometry; not an alias of autoregressive Gemma4.
// Reference: Transformers c587bc884db2c2e31fc2b8102314656b17aa07b1.
// See docs/diffusiongemma/implementation-references.md.

import Foundation

public struct DiffusionGemmaTextConfiguration: Codable, Sendable, Equatable {
    public let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let moeIntermediateSize: Int
    public let layerCount: Int
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let globalKeyValueHeads: Int?
    public let headDimension: Int
    public let globalHeadDimension: Int
    public let maxPositionEmbeddings: Int
    public let slidingWindow: Int
    public let expertCount: Int
    public let topKExperts: Int
    public let rmsNormEpsilon: Float
    public let finalLogitSoftcap: Float
    public let hiddenActivation: String
    public let attentionBias: Bool
    public let attentionDropout: Float
    public let tieWordEmbeddings: Bool
    public let bidirectionalAttention: String?
    public let bosTokenId: Int?
    public let padTokenId: Int?
    public let dtype: String?
    public let initializerRange: Float
    public let eosTokenIds: [Int]?
    public let layerTypes: [String]
    public let ropeParameters: [String: RotaryParameters]

    public struct RotaryParameters: Codable, Sendable, Equatable {
        public let type: String
        public let theta: Float
        public let partialRotaryFactor: Float?
        enum CodingKeys: String, CodingKey {
            case type = "rope_type"
            case theta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
        }
        public init(type: String, theta: Float, partialRotaryFactor: Float? = nil) {
            self.type = type
            self.theta = theta
            self.partialRotaryFactor = partialRotaryFactor
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case layerCount = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case keyValueHeads = "num_key_value_heads"
        case globalKeyValueHeads = "num_global_key_value_heads"
        case headDimension = "head_dim"
        case globalHeadDimension = "global_head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case expertCount = "num_experts"
        case topKExperts = "top_k_experts"
        case rmsNormEpsilon = "rms_norm_eps"
        case finalLogitSoftcap = "final_logit_softcapping"
        case hiddenActivation = "hidden_activation"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case tieWordEmbeddings = "tie_word_embeddings"
        case bidirectionalAttention = "use_bidirectional_attention"
        case bosTokenId = "bos_token_id"
        case padTokenId = "pad_token_id"
        case dtype = "dtype"
        case initializerRange = "initializer_range"
        case eosTokenIds = "eos_token_id"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case perLayerConfiguration = "per_layer_config"
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        // This newer reference feature changes per-layer projection geometry.
        // Do not silently load it using only the global dimensions.
        if values.contains(.perLayerConfiguration) {
            throw DiffusionGemmaConfigurationError.unsupported("per_layer_config")
        }
        modelType =
            try values.decodeIfPresent(String.self, forKey: .modelType) ?? "diffusion_gemma_text"
        vocabularySize = try values.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
        hiddenSize = try values.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try values.decode(Int.self, forKey: .intermediateSize)
        moeIntermediateSize = try values.decode(Int.self, forKey: .moeIntermediateSize)
        layerCount = try values.decode(Int.self, forKey: .layerCount)
        attentionHeads = try values.decode(Int.self, forKey: .attentionHeads)
        keyValueHeads = try values.decode(Int.self, forKey: .keyValueHeads)
        globalKeyValueHeads = try values.decodeIfPresent(Int.self, forKey: .globalKeyValueHeads)
        headDimension = try values.decodeIfPresent(Int.self, forKey: .headDimension) ?? 256
        globalHeadDimension =
            try values.decodeIfPresent(Int.self, forKey: .globalHeadDimension) ?? 512
        maxPositionEmbeddings =
            try values.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        slidingWindow = try values.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        expertCount = try values.decode(Int.self, forKey: .expertCount)
        topKExperts = try values.decode(Int.self, forKey: .topKExperts)
        rmsNormEpsilon = try values.decodeIfPresent(Float.self, forKey: .rmsNormEpsilon) ?? 1e-6
        finalLogitSoftcap = try values.decodeIfPresent(Float.self, forKey: .finalLogitSoftcap) ?? 30
        hiddenActivation =
            try values.decodeIfPresent(String.self, forKey: .hiddenActivation)
            ?? "gelu_pytorch_tanh"
        attentionBias = try values.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        attentionDropout = try values.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0
        tieWordEmbeddings =
            try values.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        bidirectionalAttention = try values.decodeIfPresent(
            String.self, forKey: .bidirectionalAttention)
        bosTokenId = try values.decodeIfPresent(Int.self, forKey: .bosTokenId)
        padTokenId = try values.decodeIfPresent(Int.self, forKey: .padTokenId)
        dtype = try values.decodeIfPresent(String.self, forKey: .dtype)
        initializerRange = try values.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
        if try !values.contains(.eosTokenIds) || values.decodeNil(forKey: .eosTokenIds) {
            eosTokenIds = nil
        } else if let token = try? values.decode(Int.self, forKey: .eosTokenIds) {
            eosTokenIds = [token]
        } else {
            eosTokenIds = try values.decode([Int].self, forKey: .eosTokenIds)
        }
        guard layerCount > 0 else {
            throw DiffusionGemmaConfigurationError.invalid("num_hidden_layers")
        }
        if let declared = try values.decodeIfPresent([String].self, forKey: .layerTypes) {
            layerTypes = declared
        } else {
            let count = layerCount
            layerTypes = (0 ..< count).map { index in
                (index + 1).isMultiple(of: 6) || index == count - 1
                    ? "full_attention" : "sliding_attention"
            }
        }
        ropeParameters =
            try values.decodeIfPresent([String: RotaryParameters].self, forKey: .ropeParameters)
            ?? [
                "sliding_attention": .init(type: "default", theta: 10000),
                "full_attention": .init(
                    type: "proportional", theta: 1_000_000, partialRotaryFactor: 0.25),
            ]
        try validate()
    }

    public func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(modelType, forKey: .modelType)
        try values.encode(vocabularySize, forKey: .vocabularySize)
        try values.encode(hiddenSize, forKey: .hiddenSize)
        try values.encode(intermediateSize, forKey: .intermediateSize)
        try values.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try values.encode(layerCount, forKey: .layerCount)
        try values.encode(attentionHeads, forKey: .attentionHeads)
        try values.encode(keyValueHeads, forKey: .keyValueHeads)
        try values.encodeIfPresent(globalKeyValueHeads, forKey: .globalKeyValueHeads)
        try values.encode(headDimension, forKey: .headDimension)
        try values.encode(globalHeadDimension, forKey: .globalHeadDimension)
        try values.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try values.encode(slidingWindow, forKey: .slidingWindow)
        try values.encode(expertCount, forKey: .expertCount)
        try values.encode(topKExperts, forKey: .topKExperts)
        try values.encode(rmsNormEpsilon, forKey: .rmsNormEpsilon)
        try values.encode(finalLogitSoftcap, forKey: .finalLogitSoftcap)
        try values.encode(hiddenActivation, forKey: .hiddenActivation)
        try values.encode(attentionBias, forKey: .attentionBias)
        try values.encode(attentionDropout, forKey: .attentionDropout)
        try values.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try values.encodeIfPresent(bidirectionalAttention, forKey: .bidirectionalAttention)
        try values.encodeIfPresent(bosTokenId, forKey: .bosTokenId)
        try values.encodeIfPresent(padTokenId, forKey: .padTokenId)
        try values.encodeIfPresent(dtype, forKey: .dtype)
        try values.encode(initializerRange, forKey: .initializerRange)
        try values.encodeIfPresent(eosTokenIds, forKey: .eosTokenIds)
        try values.encode(layerTypes, forKey: .layerTypes)
        try values.encode(ropeParameters, forKey: .ropeParameters)
    }

    private func validate() throws {
        guard modelType == "diffusion_gemma_text" else {
            throw DiffusionGemmaConfigurationError.unsupported("model_type")
        }
        for (name, value) in [
            ("vocab_size", vocabularySize), ("hidden_size", hiddenSize),
            ("intermediate_size", intermediateSize), ("moe_intermediate_size", moeIntermediateSize),
            ("num_attention_heads", attentionHeads), ("num_key_value_heads", keyValueHeads),
            ("head_dim", headDimension), ("global_head_dim", globalHeadDimension),
            ("max_position_embeddings", maxPositionEmbeddings), ("sliding_window", slidingWindow),
            ("num_experts", expertCount), ("top_k_experts", topKExperts),
        ] {
            guard value > 0 else { throw DiffusionGemmaConfigurationError.invalid(name) }
        }
        guard vocabularySize <= Int(Int32.max), topKExperts <= expertCount else {
            throw DiffusionGemmaConfigurationError.invalid("vocabulary/expert range")
        }
        let fullHeads = globalKeyValueHeads ?? keyValueHeads
        guard fullHeads > 0, attentionHeads.isMultiple(of: keyValueHeads),
            attentionHeads.isMultiple(of: fullHeads),
            headDimension.isMultiple(of: 2), globalHeadDimension.isMultiple(of: 2)
        else { throw DiffusionGemmaConfigurationError.invalid("attention geometry") }
        guard layerTypes.count == layerCount,
            layerTypes.allSatisfy({ ["sliding_attention", "full_attention"].contains($0) }),
            layerTypes.last == "full_attention"
        else { throw DiffusionGemmaConfigurationError.invalid("layer_types") }
        guard hiddenActivation == "gelu_pytorch_tanh", tieWordEmbeddings else {
            throw DiffusionGemmaConfigurationError.unsupported("activation/untied embeddings")
        }
        guard bidirectionalAttention == nil || bidirectionalAttention == "vision" else {
            throw DiffusionGemmaConfigurationError.unsupported("use_bidirectional_attention")
        }
        guard rmsNormEpsilon.isFinite, rmsNormEpsilon > 0,
            finalLogitSoftcap.isFinite, finalLogitSoftcap > 0,
            attentionDropout.isFinite, (0 ..< 1).contains(attentionDropout),
            initializerRange.isFinite, initializerRange >= 0
        else {
            throw DiffusionGemmaConfigurationError.invalid("normalization/dropout/initialization")
        }
        for (kind, dimension) in [
            ("sliding_attention", headDimension), ("full_attention", globalHeadDimension),
        ] {
            guard let rope = ropeParameters[kind], rope.theta.isFinite, rope.theta > 0,
                rope.type == (kind == "full_attention" ? "proportional" : "default")
            else { throw DiffusionGemmaConfigurationError.unsupported("rope_parameters." + kind) }
            let fraction = rope.partialRotaryFactor ?? 1
            let rotated = Float(dimension) * fraction
            guard fraction.isFinite, fraction > 0, fraction <= 1,
                rotated >= 2, rotated.rounded(.towardZero) == rotated,
                Int(rotated).isMultiple(of: 2)
            else { throw DiffusionGemmaConfigurationError.invalid("partial_rotary_factor") }
        }
        for token in [bosTokenId, padTokenId].compactMap({ $0 }) + (eosTokenIds ?? []) {
            guard token >= 0, token < vocabularySize else {
                throw DiffusionGemmaConfigurationError.invalid("special_token_ids")
            }
        }
    }
}

public enum DiffusionGemmaConfigurationError: Error, Sendable, Equatable {
    case invalid(String)
    case unsupported(String)
}
