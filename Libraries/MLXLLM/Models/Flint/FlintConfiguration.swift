// Flint architecture configuration. See ThirdPartyNotices/Flint.md.
import Foundation
import MLXLMCommon

public struct FlintConfiguration: Decodable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let vocabularySize: Int
    public let rmsNormEps: Float
    public let maxPositionEmbeddings: Int
    public let ropeTheta: Float
    public let layerTypes: [String]
    public let noRopeLayerIndices: [Int]
    public let mlpOnlyLayers: [Int]
    public let slidingWindow: Int
    public let numExperts: Int
    public let expertsPerToken: Int
    public let moeIntermediateSize: Int
    public let sharedExpertIntermediateSize: Int
    public let tieWordEmbeddings: Bool
    public let mlxExport: Export?

    public struct Export: Codable, Sendable {
        public let bits: Int
        public let embeddingBits: Int
        public let routedExpertBits: Int
        enum CodingKeys: String, CodingKey {
            case bits
            case embeddingBits = "embedding_bits"
            case routedExpertBits = "routed_expert_bits"
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case layerTypes = "layer_types"
        case noRopeLayerIndices = "no_rope_layer_indices"
        case mlpOnlyLayers = "mlp_only_layers"
        case slidingWindow = "sliding_window"
        case numExperts = "num_experts"
        case expertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case mlxExport = "mlx_export"
    }
    private enum SemanticKeys: String, CodingKey {
        case hiddenAct = "hidden_act"
        case scoreFunction = "router_score_function"
        case softmaxType = "softmax_type"
        case normTopk = "norm_topk_prob"
        case attentionBias = "attention_bias"
        case useSliding = "use_sliding_window"
        case sparseStep = "decoder_sparse_step"
        case ropeParameters = "rope_parameters"
        case ropeScaling = "rope_scaling"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        layerTypes = try c.decode([String].self, forKey: .layerTypes)
        noRopeLayerIndices = try c.decode([Int].self, forKey: .noRopeLayerIndices)
        mlpOnlyLayers = try c.decode([Int].self, forKey: .mlpOnlyLayers)
        slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        numExperts = try c.decode(Int.self, forKey: .numExperts)
        expertsPerToken = try c.decode(Int.self, forKey: .expertsPerToken)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        sharedExpertIntermediateSize = try c.decode(Int.self, forKey: .sharedExpertIntermediateSize)
        tieWordEmbeddings = try c.decode(Bool.self, forKey: .tieWordEmbeddings)
        mlxExport = try c.decodeIfPresent(Export.self, forKey: .mlxExport)
        let s = try decoder.container(keyedBy: SemanticKeys.self)
        func require(_ condition: Bool, _ description: String) throws {
            if !condition {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: decoder.codingPath, debugDescription: "Flint: \(description)"))
            }
        }
        try require(modelType == "flint", "incorrect model_type")
        try require(
            [
                hiddenSize, intermediateSize, hiddenLayers, attentionHeads, kvHeads,
                headDim, vocabularySize, maxPositionEmbeddings, slidingWindow, numExperts,
                expertsPerToken, moeIntermediateSize, sharedExpertIntermediateSize,
            ].allSatisfy { $0 > 0 },
            "dimensions must be positive")
        try require(
            attentionHeads % kvHeads == 0 && headDim % 2 == 0,
            "invalid grouped-query or rotary dimensions")
        try require(
            expertsPerToken <= numExperts && rmsNormEps > 0 && ropeTheta > 0,
            "invalid routing or normalization parameters")
        try require(
            layerTypes.count == hiddenLayers
                && layerTypes.allSatisfy {
                    $0 == "full_attention" || $0 == "sliding_attention"
                }, "layer_types must describe every layer")
        let full = Set(layerTypes.indices.filter { layerTypes[$0] == "full_attention" })
        try require(
            Set(noRopeLayerIndices) == full && noRopeLayerIndices.count == full.count,
            "RoPE must be disabled exactly on full-attention layers")
        try require(
            Set(mlpOnlyLayers).count == mlpOnlyLayers.count
                && mlpOnlyLayers.allSatisfy {
                    (0 ..< hiddenLayers).contains($0)
                }, "invalid dense-layer indices")
        try require(try s.decode(String.self, forKey: .hiddenAct) == "silu", "requires SiLU")
        try require(
            try s.decode(String.self, forKey: .scoreFunction) == "sigmoid",
            "requires sigmoid routing")
        try require(
            try s.decode(String.self, forKey: .softmaxType) == "learnable", "requires learned sinks"
        )
        try require(
            try s.decode(Bool.self, forKey: .normTopk), "requires normalized selected scores")
        try require(try !s.decode(Bool.self, forKey: .attentionBias), "attention bias unsupported")
        try require(try s.decode(Bool.self, forKey: .useSliding), "requires sliding-window policy")
        try require(try s.decode(Int.self, forKey: .sparseStep) == 1, "unsupported sparse step")
        if let parameters = try s.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        {
            try require(parameters["rope_type"] == .string("default"), "unsupported RoPE extension")
            try require(parameters["rope_theta"]?.asFloat() == ropeTheta, "inconsistent RoPE base")
        }
        try require(
            !s.contains(.ropeScaling) || (try s.decodeNil(forKey: .ropeScaling)),
            "rope_scaling is unsupported")
        let quantization = try s.decodeIfPresent([String: Int].self, forKey: .quantization)
        if let export = mlxExport {
            try require(
                quantization?["bits"] == 4 && quantization?["group_size"] == 64,
                "official export requires group-64 INT4 metadata")
            try require(
                export.bits == 4 && export.embeddingBits == 8 && export.routedExpertBits == 4,
                "only the official mixed INT4/INT8 export is supported")
        } else {
            try require(
                quantization == nil, "quantized checkpoints require explicit mlx_export metadata")
        }
    }

    public var cbv2LayerKinds: [CBv2LayerKind] {
        layerTypes.map {
            CBv2LayerKind(
                attention: $0 == "full_attention" ? .full : .slidingWindow(slidingWindow),
                hasSinks: true, headDim: headDim, kvHeads: kvHeads, queryHeads: attentionHeads)
        }
    }
}
