// Copyright © 2026 Eigen Labs.
// Modified in 2026: native Swift schema validation from the cited configuration.
// Configuration only: decoding this type does not enable model execution.
// Checkpoint: zai-org/GLM-5.3-Flash@91b69215804f2d2036a8867f6ce44848e278dcda/config.json
// Native candidate: mlx-community/GLM-5.3-Flash-4bit@fa04cbc3622b16aa7475404060b05e506d81bd4e/config.json
// Geometry references (not defaults):
// huggingface/transformers@d4f69a869d66eeba7af0cc19a585552b14ddd6f2
//   src/transformers/models/glm5_next/{configuration,modeling}_glm5_next.py
// sgl-project/sglang@a6a8c56015b8dbca4147634fa9ef22c7c836de05
//   python/sglang/srt/{configs/glm5_next,models/glm5_next_nextn}.py

import Foundation
import MLXLMCommon

public enum GLM5NextConfigurationError: Error, Equatable, Sendable, LocalizedError {
    case invalid(field: String, reason: String)

    public var errorDescription: String? {
        switch self {
        case .invalid(let field, let reason): "GLM5Next configuration \(field): \(reason)"
        }
    }
}

private enum GLM5NextValidation {
    static func require(_ condition: Bool, _ field: String, _ reason: String) throws {
        guard condition else {
            throw GLM5NextConfigurationError.invalid(field: field, reason: reason)
        }
    }

    static func positive(_ fields: [(String, Int)]) throws {
        for (field, value) in fields {
            try require(value > 0, field, "must be positive")
        }
    }

    @discardableResult
    static func product(_ lhs: Int, _ rhs: Int, _ field: String) throws -> Int {
        let result = lhs.multipliedReportingOverflow(by: rhs)
        try require(
            !result.overflow && result.partialValue > 0, field, "dimension product overflows")
        return result.partialValue
    }

    @discardableResult
    static func sum(_ lhs: Int, _ rhs: Int, _ field: String) throws -> Int {
        let result = lhs.addingReportingOverflow(rhs)
        try require(!result.overflow, field, "dimension sum overflows")
        return result.partialValue
    }

    static func token(_ token: Int, vocabularySize: Int, field: String) throws {
        try require((0 ..< vocabularySize).contains(token), field, "outside vocabulary")
    }

    // This is the explicit pinned schema, not a permissive Transformers config.
    // Unknown architecture keys must not silently change the eventual tensor layout.
    static func container<K: CodingKey & CaseIterable>(
        _ type: K.Type, from decoder: any Decoder
    ) throws -> KeyedDecodingContainer<K> {
        let all = try decoder.container(keyedBy: Key.self)
        let known = Set(K.allCases.map(\.stringValue))
        if let unknown = all.allKeys.map(\.stringValue).sorted().first(where: {
            !known.contains($0)
        }) {
            let field = (decoder.codingPath.map(\.stringValue) + [unknown]).joined(separator: ".")
            throw GLM5NextConfigurationError.invalid(field: field, reason: "unrecognized field")
        }
        return try decoder.container(keyedBy: type)
    }

    fileprivate struct Key: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }
}

/// The JSON scalar/list distinction is retained on round trip; numeric strings are not token IDs.
public enum GLM5NextTokenIDs: Codable, Equatable, Sendable {
    case single(Int)
    case multiple([Int])

    public var values: [Int] {
        switch self {
        case .single(let value): [value]
        case .multiple(let values): values
        }
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Int.self) {
            self = .single(value)
        } else {
            self = .multiple(try container.decode([Int].self))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .single(let value): try container.encode(value)
        case .multiple(let values): try container.encode(values)
        }
    }
}

public struct GLM5NextLinearAttentionConfiguration: Codable, Equatable, Sendable {
    public let numHeads: Int
    public let headDim: Int
    public let gateLowerBound: Double
    public let shortConvKernelSize: Int
    public let kdaLayers: [Int]
    public let fullAttentionLayers: [Int]

    enum CodingKeys: String, CodingKey, CaseIterable {
        case numHeads = "num_heads"
        case headDim = "head_dim"
        case gateLowerBound = "gate_lower_bound"
        case shortConvKernelSize = "short_conv_kernel_size"
        case kdaLayers = "kda_layers"
        case fullAttentionLayers = "full_attn_layers"
    }

    public init(from decoder: any Decoder) throws {
        let c = try GLM5NextValidation.container(CodingKeys.self, from: decoder)
        numHeads = try c.decode(Int.self, forKey: .numHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        gateLowerBound = try c.decode(Double.self, forKey: .gateLowerBound)
        shortConvKernelSize = try c.decode(Int.self, forKey: .shortConvKernelSize)
        kdaLayers = try c.decode([Int].self, forKey: .kdaLayers)
        fullAttentionLayers = try c.decode([Int].self, forKey: .fullAttentionLayers)
        try GLM5NextValidation.positive([
            ("linear_attn_config.num_heads", numHeads),
            ("linear_attn_config.head_dim", headDim),
            ("linear_attn_config.short_conv_kernel_size", shortConvKernelSize),
        ])
        try GLM5NextValidation.require(
            gateLowerBound.isFinite && Float(gateLowerBound).isFinite && Float(gateLowerBound) < 0,
            "linear_attn_config.gate_lower_bound", "must be finite and negative in Float32")
        let width = try GLM5NextValidation.product(
            numHeads, headDim, "linear_attn_config.projection")
        try GLM5NextValidation.product(width, 3, "linear_attn_config.qkv")
        try GLM5NextValidation.product(
            width, shortConvKernelSize, "linear_attn_config.convolution")
        try GLM5NextValidation.product(width, headDim, "linear_attn_config.recurrent_state")
    }
}

/// Explicit source fields are required: no architecture dimensions or schedules are guessed.
public struct GLM5NextTextConfiguration: Codable, Equatable, Sendable {
    public enum LayerType: String, Codable, Sendable {
        case linearAttention = "linear_attention"
        case deepseekSparseAttention = "deepseek_sparse_attention"
    }
    public enum MLPLayerType: String, Codable, Sendable {
        case dense, sparse
    }
    public enum IndexerType: String, Codable, Sendable {
        case full, shared
    }

    public let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let moeIntermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    /// Source metadata: zero denotes the NoPE rotary dimension, not hiddenSize / attentionHeads.
    public let headDim: Int
    public let qkHeadDim: Int
    public let qkNopeHeadDim: Int
    public let qkRopeHeadDim: Int
    public let vHeadDim: Int
    public let qLoraRank: Int
    public let kvLoraRank: Int
    public let mlaUseNope: Bool
    public let layerTypes: [LayerType]
    public let mlpLayerTypes: [MLPLayerType]
    public let indexerTypes: [IndexerType]
    public let linearAttention: GLM5NextLinearAttentionConfiguration
    public let indexHeadDim: Int
    public let indexNHeads: Int
    public let indexTopK: Int
    public let indexKPool: Int
    public let indexKPoolCompress: Bool
    public let indexKPoolAlwaysSelectTail: Bool
    public let indexerRopeInterleave: Bool
    public let indexShareForMTPIteration: Bool
    public let mhc: Bool
    public let hcMult: Int
    public let hcEps: Double
    public let hcSinkhornIters: Int
    public let firstKDenseReplace: Int
    public let nRoutedExperts: Int
    public let nSharedExperts: Int
    public let numExpertsPerTok: Int
    public let nGroup: Int
    public let topkGroup: Int
    public let normTopkProb: Bool
    public let routedScalingFactor: Double
    public let scoringFunc: String
    public let topkMethod: String
    public let moeRouterDtype: String
    public let outputRouterLogits: Bool
    public let routerAuxLossCoef: Double
    /// Additional checkpoint layers; not included in hiddenLayers or the backbone schedules.
    public let numNextNPredictLayers: Int
    public let hiddenAct: String
    public let swigluLimit: Double
    public let rmsNormEps: Double
    public let attentionBias: Bool
    public let attentionDropout: Double
    public let maxPositionEmbeddings: Int
    public let initializerRange: Double
    public let useCache: Bool
    public let tieWordEmbeddings: Bool
    public let dtype: String
    public let padTokenId: Int?
    public let bosTokenId: Int?
    public let eosTokenIds: GLM5NextTokenIDs?

    enum CodingKeys: String, CodingKey, CaseIterable {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case qkHeadDim = "qk_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case mlaUseNope = "mla_use_nope"
        case layerTypes = "layer_types"
        case mlpLayerTypes = "mlp_layer_types"
        case indexerTypes = "indexer_types"
        case linearAttention = "linear_attn_config"
        case indexHeadDim = "index_head_dim"
        case indexNHeads = "index_n_heads"
        case indexTopK = "index_topk"
        case indexKPool = "index_kpool"
        case indexKPoolCompress = "index_kpool_compress"
        case indexKPoolAlwaysSelectTail = "index_kpool_always_select_tail"
        case indexerRopeInterleave = "indexer_rope_interleave"
        case indexShareForMTPIteration = "index_share_for_mtp_iteration"
        case mhc
        case hcMult = "hc_mult"
        case hcEps = "hc_eps"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case firstKDenseReplace = "first_k_dense_replace"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case normTopkProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case scoringFunc = "scoring_func"
        case topkMethod = "topk_method"
        case moeRouterDtype = "moe_router_dtype"
        case outputRouterLogits = "output_router_logits"
        case routerAuxLossCoef = "router_aux_loss_coef"
        case numNextNPredictLayers = "num_nextn_predict_layers"
        case hiddenAct = "hidden_act"
        case swigluLimit = "swiglu_limit"
        case rmsNormEps = "rms_norm_eps"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case maxPositionEmbeddings = "max_position_embeddings"
        case initializerRange = "initializer_range"
        case useCache = "use_cache"
        case tieWordEmbeddings = "tie_word_embeddings"
        case dtype
        case padTokenId = "pad_token_id"
        case bosTokenId = "bos_token_id"
        case eosTokenIds = "eos_token_id"
    }

    public init(from decoder: any Decoder) throws {
        let c = try GLM5NextValidation.container(CodingKeys.self, from: decoder)
        modelType = try c.decode(String.self, forKey: .modelType)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        qkHeadDim = try c.decode(Int.self, forKey: .qkHeadDim)
        qkNopeHeadDim = try c.decode(Int.self, forKey: .qkNopeHeadDim)
        qkRopeHeadDim = try c.decode(Int.self, forKey: .qkRopeHeadDim)
        vHeadDim = try c.decode(Int.self, forKey: .vHeadDim)
        qLoraRank = try c.decode(Int.self, forKey: .qLoraRank)
        kvLoraRank = try c.decode(Int.self, forKey: .kvLoraRank)
        mlaUseNope = try c.decode(Bool.self, forKey: .mlaUseNope)
        layerTypes = try c.decode([LayerType].self, forKey: .layerTypes)
        mlpLayerTypes = try c.decode([MLPLayerType].self, forKey: .mlpLayerTypes)
        indexerTypes = try c.decode([IndexerType].self, forKey: .indexerTypes)
        linearAttention = try c.decode(
            GLM5NextLinearAttentionConfiguration.self, forKey: .linearAttention)
        indexHeadDim = try c.decode(Int.self, forKey: .indexHeadDim)
        indexNHeads = try c.decode(Int.self, forKey: .indexNHeads)
        indexTopK = try c.decode(Int.self, forKey: .indexTopK)
        indexKPool = try c.decode(Int.self, forKey: .indexKPool)
        indexKPoolCompress = try c.decode(Bool.self, forKey: .indexKPoolCompress)
        indexKPoolAlwaysSelectTail = try c.decode(Bool.self, forKey: .indexKPoolAlwaysSelectTail)
        indexerRopeInterleave = try c.decode(Bool.self, forKey: .indexerRopeInterleave)
        indexShareForMTPIteration = try c.decode(Bool.self, forKey: .indexShareForMTPIteration)
        mhc = try c.decode(Bool.self, forKey: .mhc)
        hcMult = try c.decode(Int.self, forKey: .hcMult)
        hcEps = try c.decode(Double.self, forKey: .hcEps)
        hcSinkhornIters = try c.decode(Int.self, forKey: .hcSinkhornIters)
        firstKDenseReplace = try c.decode(Int.self, forKey: .firstKDenseReplace)
        nRoutedExperts = try c.decode(Int.self, forKey: .nRoutedExperts)
        nSharedExperts = try c.decode(Int.self, forKey: .nSharedExperts)
        numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        nGroup = try c.decode(Int.self, forKey: .nGroup)
        topkGroup = try c.decode(Int.self, forKey: .topkGroup)
        normTopkProb = try c.decode(Bool.self, forKey: .normTopkProb)
        routedScalingFactor = try c.decode(Double.self, forKey: .routedScalingFactor)
        scoringFunc = try c.decode(String.self, forKey: .scoringFunc)
        topkMethod = try c.decode(String.self, forKey: .topkMethod)
        moeRouterDtype = try c.decode(String.self, forKey: .moeRouterDtype)
        outputRouterLogits = try c.decode(Bool.self, forKey: .outputRouterLogits)
        routerAuxLossCoef = try c.decode(Double.self, forKey: .routerAuxLossCoef)
        numNextNPredictLayers = try c.decode(Int.self, forKey: .numNextNPredictLayers)
        hiddenAct = try c.decode(String.self, forKey: .hiddenAct)
        swigluLimit = try c.decode(Double.self, forKey: .swigluLimit)
        rmsNormEps = try c.decode(Double.self, forKey: .rmsNormEps)
        attentionBias = try c.decode(Bool.self, forKey: .attentionBias)
        attentionDropout = try c.decode(Double.self, forKey: .attentionDropout)
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        initializerRange = try c.decode(Double.self, forKey: .initializerRange)
        useCache = try c.decode(Bool.self, forKey: .useCache)
        tieWordEmbeddings = try c.decode(Bool.self, forKey: .tieWordEmbeddings)
        dtype = try c.decode(String.self, forKey: .dtype)
        padTokenId = try c.decodeIfPresent(Int.self, forKey: .padTokenId)
        bosTokenId = try c.decodeIfPresent(Int.self, forKey: .bosTokenId)
        eosTokenIds = try c.decodeIfPresent(GLM5NextTokenIDs.self, forKey: .eosTokenIds)
        try validate()
    }

    private func validate() throws {
        try GLM5NextValidation.require(
            modelType == "glm5_next_text", "text_config.model_type", "expected glm5_next_text")
        try GLM5NextValidation.positive([
            ("vocab_size", vocabularySize), ("hidden_size", hiddenSize),
            ("intermediate_size", intermediateSize), ("moe_intermediate_size", moeIntermediateSize),
            ("num_hidden_layers", hiddenLayers), ("num_attention_heads", attentionHeads),
            ("num_key_value_heads", kvHeads), ("qk_head_dim", qkHeadDim),
            ("qk_nope_head_dim", qkNopeHeadDim), ("v_head_dim", vHeadDim),
            ("q_lora_rank", qLoraRank), ("kv_lora_rank", kvLoraRank),
            ("index_head_dim", indexHeadDim), ("index_n_heads", indexNHeads),
            ("index_topk", indexTopK), ("index_kpool", indexKPool),
            ("hc_mult", hcMult), ("hc_sinkhorn_iters", hcSinkhornIters),
            ("n_routed_experts", nRoutedExperts), ("n_shared_experts", nSharedExperts),
            ("num_experts_per_tok", numExpertsPerTok), ("n_group", nGroup),
            ("topk_group", topkGroup), ("max_position_embeddings", maxPositionEmbeddings),
        ])
        try GLM5NextValidation.require(
            vocabularySize <= Int(Int32.max), "vocab_size", "must fit token Int32 representation")
        try GLM5NextValidation.require(
            attentionHeads == kvHeads, "num_key_value_heads",
            "MLA requires equal query and KV head counts")
        try GLM5NextValidation.require(
            mlaUseNope && qkRopeHeadDim == 0 && headDim == 0 && qkHeadDim == qkNopeHeadDim,
            "qk_head_dim", "NoPE requires zero rotary/head_dim and qk_head_dim == qk_nope_head_dim")
        try GLM5NextValidation.require(
            indexTopK.isMultiple(of: indexKPool), "index_topk", "must be divisible by index_kpool")
        try GLM5NextValidation.require(
            layerTypes.count == hiddenLayers && mlpLayerTypes.count == hiddenLayers
                && indexerTypes.count == hiddenLayers,
            "layer_types",
            "all backbone schedules must contain num_hidden_layers entries (excluding MTP)")
        let linearLayers = layerTypes.indices.filter { layerTypes[$0] == .linearAttention }
        let sparseLayers = layerTypes.indices.filter { layerTypes[$0] == .deepseekSparseAttention }
        try GLM5NextValidation.require(
            linearAttention.kdaLayers == linearLayers
                && linearAttention.fullAttentionLayers == sparseLayers,
            "linear_attn_config", "ordered layer lists must exactly partition and match layer_types"
        )
        // Only full indexing is established by the pinned checkpoint. Keep the spelling
        // typed but refuse cross-layer index reuse until its separate contract is implemented.
        try GLM5NextValidation.require(
            indexerTypes.allSatisfy { $0 == .full }, "indexer_types",
            "only full indexing is supported by this schema")
        try GLM5NextValidation.require(
            firstKDenseReplace >= 0 && firstKDenseReplace <= hiddenLayers,
            "first_k_dense_replace", "outside backbone layer range")
        try GLM5NextValidation.require(
            mlpLayerTypes.enumerated().allSatisfy { index, type in
                type == (index < firstKDenseReplace ? .dense : .sparse)
            }, "mlp_layer_types", "must agree with first_k_dense_replace")
        try GLM5NextValidation.require(
            nRoutedExperts.isMultiple(of: nGroup) && nRoutedExperts / nGroup >= 2,
            "n_group", "must divide experts into groups of at least two for noaux_tc")
        try GLM5NextValidation.require(topkGroup <= nGroup, "topk_group", "exceeds group count")
        let selectableExperts = try GLM5NextValidation.product(
            nRoutedExperts / nGroup, topkGroup, "topk_group")
        try GLM5NextValidation.require(
            numExpertsPerTok <= selectableExperts, "num_experts_per_tok",
            "exceeds selected groups' expert count")
        try GLM5NextValidation.require(
            scoringFunc == "sigmoid" && topkMethod == "noaux_tc" && moeRouterDtype == "float32",
            "scoring_func/topk_method/moe_router_dtype", "requires sigmoid/noaux_tc/float32 routing"
        )
        try GLM5NextValidation.require(
            hiddenAct == "silu", "hidden_act", "requires the clamped SwiGLU activation")
        try GLM5NextValidation.require(
            ["bfloat16", "float16", "float32"].contains(dtype), "dtype",
            "unrecognized floating-point dtype")
        for (field, value) in [
            ("hc_eps", hcEps), ("rms_norm_eps", rmsNormEps),
            ("swiglu_limit", swigluLimit), ("routed_scaling_factor", routedScalingFactor),
        ] {
            try GLM5NextValidation.require(
                value.isFinite && Float(value).isFinite && Float(value) > 0,
                field, "must be finite and positive in Float32")
        }
        try GLM5NextValidation.require(
            attentionDropout.isFinite && (0 ..< 1).contains(attentionDropout),
            "attention_dropout", "must be finite and in [0, 1)")
        for (field, value) in [
            ("initializer_range", initializerRange), ("router_aux_loss_coef", routerAuxLossCoef),
        ] {
            try GLM5NextValidation.require(
                value.isFinite && value >= 0, field, "must be finite and nonnegative")
        }
        try GLM5NextValidation.require(
            numNextNPredictLayers >= 0,
            "num_nextn_predict_layers", "must be nonnegative")
        try GLM5NextValidation.sum(hiddenLayers, numNextNPredictLayers, "checkpoint layer count")
        for (lhs, rhs, field) in [
            (hiddenSize, intermediateSize, "dense projection"),
            (hiddenSize, moeIntermediateSize, "expert projection"),
            (moeIntermediateSize, nSharedExperts, "shared expert width"),
            (attentionHeads, qkHeadDim, "query projection"),
            (attentionHeads, vHeadDim, "value projection"),
            (hiddenSize, qLoraRank, "query low-rank projection"),
            (hiddenSize, kvLoraRank, "KV low-rank projection"),
            (indexNHeads, indexHeadDim, "indexer projection"),
            (hiddenSize, hcMult, "mHC streams"),
            (hcMult, hcMult, "mHC mixing matrix"),
        ] {
            try GLM5NextValidation.product(lhs, rhs, field)
        }
        let hcPlusTwo = try GLM5NextValidation.sum(hcMult, 2, "mHC projection")
        let hcProjection = try GLM5NextValidation.product(hcMult, hcPlusTwo, "mHC projection")
        let hcWidth = try GLM5NextValidation.product(hcMult, hiddenSize, "mHC streams")
        try GLM5NextValidation.product(hcProjection, hcWidth, "mHC projection weights")
        let queryWidth = try GLM5NextValidation.product(attentionHeads, qkHeadDim, "query width")
        let valueWidth = try GLM5NextValidation.product(attentionHeads, vHeadDim, "value width")
        let kvHeadWidth = try GLM5NextValidation.sum(qkNopeHeadDim, vHeadDim, "KV head width")
        let kvWidth = try GLM5NextValidation.product(attentionHeads, kvHeadWidth, "KV width")
        try GLM5NextValidation.product(qLoraRank, queryWidth, "query expansion weights")
        try GLM5NextValidation.product(kvLoraRank, kvWidth, "KV expansion weights")
        try GLM5NextValidation.product(hiddenSize, valueWidth, "output projection weights")
        let linearWidth = try GLM5NextValidation.product(
            linearAttention.numHeads, linearAttention.headDim, "linear projection width")
        try GLM5NextValidation.product(hiddenSize, linearWidth, "linear projection weights")
        try GLM5NextValidation.sum(
            indexTopK, indexKPoolAlwaysSelectTail ? indexKPool - 1 : 0, "selected index width")
        for (field, tokens) in [
            ("pad_token_id", padTokenId.map { [$0] } ?? []),
            ("bos_token_id", bosTokenId.map { [$0] } ?? []),
            ("eos_token_id", eosTokenIds?.values ?? []),
        ] {
            for token in tokens {
                try GLM5NextValidation.token(token, vocabularySize: vocabularySize, field: field)
            }
        }
    }
}

public struct GLM5NextVisionConfiguration: Codable, Equatable, Sendable {
    public let modelType: String
    public let depth: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHeads: Int
    public let imageSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let spatialMergeSize: Int
    public let inChannels: Int
    public let outHiddenSize: Int
    public let projectionIntermediateSize: Int
    public let hiddenAct: String
    public let swigluLimit: Double
    public let rmsNormEps: Double
    public let attentionBias: Bool
    public let attentionDropout: Double
    public let initializerRange: Double

    enum CodingKeys: String, CodingKey, CaseIterable {
        case modelType = "model_type"
        case depth
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHeads = "num_heads"
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case spatialMergeSize = "spatial_merge_size"
        case inChannels = "in_channels"
        case outHiddenSize = "out_hidden_size"
        case projectionIntermediateSize = "projection_intermediate_size"
        case hiddenAct = "hidden_act"
        case swigluLimit = "swiglu_limit"
        case rmsNormEps = "rms_norm_eps"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case initializerRange = "initializer_range"
    }

    public init(from decoder: any Decoder) throws {
        let c = try GLM5NextValidation.container(CodingKeys.self, from: decoder)
        modelType = try c.decode(String.self, forKey: .modelType)
        depth = try c.decode(Int.self, forKey: .depth)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numHeads = try c.decode(Int.self, forKey: .numHeads)
        imageSize = try c.decode(Int.self, forKey: .imageSize)
        patchSize = try c.decode(Int.self, forKey: .patchSize)
        temporalPatchSize = try c.decode(Int.self, forKey: .temporalPatchSize)
        spatialMergeSize = try c.decode(Int.self, forKey: .spatialMergeSize)
        inChannels = try c.decode(Int.self, forKey: .inChannels)
        outHiddenSize = try c.decode(Int.self, forKey: .outHiddenSize)
        projectionIntermediateSize = try c.decode(Int.self, forKey: .projectionIntermediateSize)
        hiddenAct = try c.decode(String.self, forKey: .hiddenAct)
        swigluLimit = try c.decode(Double.self, forKey: .swigluLimit)
        rmsNormEps = try c.decode(Double.self, forKey: .rmsNormEps)
        attentionBias = try c.decode(Bool.self, forKey: .attentionBias)
        attentionDropout = try c.decode(Double.self, forKey: .attentionDropout)
        initializerRange = try c.decode(Double.self, forKey: .initializerRange)
        try GLM5NextValidation.require(
            modelType == "glm5_next_vision", "vision_config.model_type", "expected glm5_next_vision"
        )
        try GLM5NextValidation.positive([
            ("vision_config.depth", depth), ("vision_config.hidden_size", hiddenSize),
            ("vision_config.intermediate_size", intermediateSize),
            ("vision_config.num_heads", numHeads),
            ("vision_config.image_size", imageSize), ("vision_config.patch_size", patchSize),
            ("vision_config.temporal_patch_size", temporalPatchSize),
            ("vision_config.spatial_merge_size", spatialMergeSize),
            ("vision_config.in_channels", inChannels),
            ("vision_config.out_hidden_size", outHiddenSize),
            ("vision_config.projection_intermediate_size", projectionIntermediateSize),
        ])
        try GLM5NextValidation.require(
            hiddenSize.isMultiple(of: numHeads) && (hiddenSize / numHeads).isMultiple(of: 4),
            "vision_config.num_heads", "axial rotary head dimension must be divisible by four")
        try GLM5NextValidation.require(
            imageSize.isMultiple(of: patchSize)
                && (imageSize / patchSize).isMultiple(of: spatialMergeSize),
            "vision_config.image_size", "patch grid must divide evenly into spatial merge groups")
        try GLM5NextValidation.require(
            hiddenAct == "silu", "vision_config.hidden_act", "requires clamped SwiGLU")
        try GLM5NextValidation.require(
            swigluLimit.isFinite && Float(swigluLimit).isFinite && Float(swigluLimit) > 0
                && rmsNormEps.isFinite && Float(rmsNormEps).isFinite && Float(rmsNormEps) > 0,
            "vision_config.swiglu_limit/rms_norm_eps", "must be finite and positive")
        try GLM5NextValidation.require(
            attentionDropout.isFinite && (0 ..< 1).contains(attentionDropout)
                && initializerRange.isFinite && initializerRange >= 0,
            "vision_config.attention_dropout/initializer_range",
            "invalid dropout or initializer range")
        let patchArea = try GLM5NextValidation.product(patchSize, patchSize, "vision patch area")
        let patchVolume = try GLM5NextValidation.product(
            patchArea, temporalPatchSize, "vision patch volume")
        let patchInput = try GLM5NextValidation.product(
            patchVolume, inChannels, "vision patch input")
        try GLM5NextValidation.product(patchInput, hiddenSize, "vision patch weights")
        let qkvWidth = try GLM5NextValidation.product(3, hiddenSize, "vision QKV width")
        try GLM5NextValidation.product(qkvWidth, hiddenSize, "vision QKV weights")
        let mergeArea = try GLM5NextValidation.product(
            spatialMergeSize, spatialMergeSize, "vision merge area")
        let mergeWidth = try GLM5NextValidation.product(
            mergeArea, hiddenSize, "vision merge width")
        try GLM5NextValidation.product(mergeWidth, outHiddenSize, "vision downsample weights")
        try GLM5NextValidation.product(outHiddenSize, outHiddenSize, "vision merger weights")
        try GLM5NextValidation.product(hiddenSize, intermediateSize, "vision MLP")
        try GLM5NextValidation.product(
            projectionIntermediateSize, outHiddenSize, "vision projection")
    }
}

/// Source FP8 metadata only; this does not select a native weight conversion or kernel.
public struct GLM5NextSourceFP8Quantization: Codable, Equatable, Sendable {
    public let activationScheme: String
    public let format: String
    public let quantizationMethod: String
    public let weightBlockSize: [Int]
    public let modulesToNotConvert: [String]

    enum CodingKeys: String, CodingKey, CaseIterable {
        case activationScheme = "activation_scheme"
        case format = "fmt"
        case quantizationMethod = "quant_method"
        case weightBlockSize = "weight_block_size"
        case modulesToNotConvert = "modules_to_not_convert"
    }

    public init(from decoder: any Decoder) throws {
        let c = try GLM5NextValidation.container(CodingKeys.self, from: decoder)
        activationScheme = try c.decode(String.self, forKey: .activationScheme)
        format = try c.decode(String.self, forKey: .format)
        quantizationMethod = try c.decode(String.self, forKey: .quantizationMethod)
        weightBlockSize = try c.decode([Int].self, forKey: .weightBlockSize)
        modulesToNotConvert = try c.decode([String].self, forKey: .modulesToNotConvert)
        try GLM5NextValidation.require(
            activationScheme == "dynamic" && format == "e4m3" && quantizationMethod == "fp8",
            "quantization_config", "expected source dynamic/e4m3/fp8 metadata")
        try GLM5NextValidation.require(
            weightBlockSize.count == 2 && weightBlockSize.allSatisfy { $0 > 0 },
            "quantization_config.weight_block_size", "requires two positive dimensions")
    }
}

/// Native storage policy, using the same precision type and lookup contract as the model loader.
/// Module paths are retained verbatim, including the extra MTP layer and exporter namespaces.
public struct GLM5NextNativeQuantization: Codable, Equatable, Sendable {
    public let defaultQuantization: BaseConfiguration.Quantization
    public let perModule: [String: BaseConfiguration.Quantization]
    public let skippedModules: Set<String>
    public let metadata: [String: String]

    private enum PolicyKeys: String, CodingKey, CaseIterable {
        case groupSize = "group_size"
        case bits, mode
    }

    private static let metadataKeys: Set<String> = [
        "quant_method", "linear_class", "quantization_mode",
    ]

    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization {
        var overrides = perModule.mapValues { BaseConfiguration.QuantizationOption.quantize($0) }
        for module in skippedModules { overrides[module] = .skip }
        return .init(quantization: defaultQuantization, perLayerQuantization: overrides)
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: GLM5NextValidation.Key.self)
        let modeKey = GLM5NextValidation.Key(stringValue: "mode")
        if c.contains(modeKey) { _ = try c.decode(String.self, forKey: modeKey) }
        defaultQuantization = try BaseConfiguration.Quantization(from: decoder)
        try Self.validate(
            defaultQuantization, field: decoder.codingPath.map(\.stringValue).joined(separator: ".")
        )
        var modules: [String: BaseConfiguration.Quantization] = [:]
        var skipped: Set<String> = []
        var metadata: [String: String] = [:]
        for key in c.allKeys.sorted(by: { $0.stringValue < $1.stringValue }) {
            if PolicyKeys(rawValue: key.stringValue) != nil { continue }
            if Self.metadataKeys.contains(key.stringValue) {
                metadata[key.stringValue] = try c.decode(String.self, forKey: key)
                continue
            }
            if let skip = try? c.decode(Bool.self, forKey: key) {
                try GLM5NextValidation.require(
                    !skip, key.stringValue, "module override may be false, not true")
                skipped.insert(key.stringValue)
            } else {
                let policyDecoder = try c.superDecoder(forKey: key)
                let fields = try GLM5NextValidation.container(PolicyKeys.self, from: policyDecoder)
                if fields.contains(.mode) { _ = try fields.decode(String.self, forKey: .mode) }
                let policy = try BaseConfiguration.Quantization(from: policyDecoder)
                try Self.validate(
                    policy,
                    field: policyDecoder.codingPath.map(\.stringValue).joined(separator: "."),
                    allowsNVFP4: true)
                modules[key.stringValue] = policy
            }
        }
        try GLM5NextValidation.require(
            metadata["quant_method"] != "fp8", "quant_method",
            "source FP8 metadata cannot also declare a native affine policy")
        perModule = modules
        skippedModules = skipped
        self.metadata = metadata
    }

    public func encode(to encoder: any Encoder) throws {
        try defaultQuantization.encode(to: encoder)
        var c = encoder.container(keyedBy: GLM5NextValidation.Key.self)
        for (module, policy) in perModule {
            try c.encode(policy, forKey: GLM5NextValidation.Key(stringValue: module))
        }
        for module in skippedModules {
            try c.encode(false, forKey: GLM5NextValidation.Key(stringValue: module))
        }
        for (key, value) in metadata {
            try c.encode(value, forKey: GLM5NextValidation.Key(stringValue: key))
        }
    }

    fileprivate func hasSamePolicy(as other: Self) -> Bool {
        func equal(_ lhs: BaseConfiguration.Quantization, _ rhs: BaseConfiguration.Quantization)
            -> Bool
        {
            lhs.groupSize == rhs.groupSize && lhs.bits == rhs.bits && lhs.mode == rhs.mode
        }
        return equal(defaultQuantization, other.defaultQuantization)
            && skippedModules == other.skippedModules
            && perModule.keys.sorted() == other.perModule.keys.sorted()
            && perModule.allSatisfy { name, policy in
                other.perModule[name].map { equal(policy, $0) } ?? false
            }
    }

    /// Module overrides may also declare NVIDIA NVFP4 (E2M1, FP8-E4M3 group-16
    /// scales); where it is admitted is the weight inventory's decision.
    private static func validate(
        _ policy: BaseConfiguration.Quantization, field: String, allowsNVFP4: Bool = false
    ) throws {
        if allowsNVFP4, policy.mode == .nvfp4 {
            try GLM5NextValidation.require(
                policy.bits == 4 && policy.groupSize == 16, field,
                "nvfp4 requires bits 4 and group_size 16")
            return
        }
        // These are the affine widths/group sizes accepted by the pinned MLX implementation.
        try GLM5NextValidation.require(
            policy.mode == .affine && [2, 3, 4, 5, 6, 8].contains(policy.bits)
                && [32, 64, 128].contains(policy.groupSize),
            field, "requires affine bits 2/3/4/5/6/8 and group_size 32/64/128")
    }
}

/// `quantization_config` may describe native converted storage or original source FP8 storage.
/// The cases remain distinct; source metadata is never interpreted as a native loader policy.
public enum GLM5NextQuantizationConfiguration: Codable, Equatable, Sendable {
    case native(GLM5NextNativeQuantization)
    case sourceFP8(GLM5NextSourceFP8Quantization)

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: GLM5NextValidation.Key.self)
        if c.allKeys.contains(where: { ["bits", "group_size"].contains($0.stringValue) }) {
            self = .native(try GLM5NextNativeQuantization(from: decoder))
        } else {
            self = .sourceFP8(try GLM5NextSourceFP8Quantization(from: decoder))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        switch self {
        case .native(let policy): try policy.encode(to: encoder)
        case .sourceFP8(let metadata): try metadata.encode(to: encoder)
        }
    }
}

/// The nested multimodal checkpoint contract. No factory registration, weight loading, or MTP
/// execution is implied. A converted schema must be reviewed explicitly, not silently defaulted.
public struct GLM5NextConfiguration: Codable, Equatable, Sendable {
    public let modelType: String
    public let architectures: [String]
    public let textConfig: GLM5NextTextConfiguration
    public let visionConfig: GLM5NextVisionConfiguration
    public let imageTokenId: Int
    public let videoTokenId: Int
    public let imageStartTokenId: Int
    public let imageEndTokenId: Int
    public let videoStartTokenId: Int
    public let videoEndTokenId: Int
    public let tieWordEmbeddings: Bool
    public let transformersVersion: String?
    public let quantization: GLM5NextNativeQuantization?
    public let quantizationConfig: GLM5NextQuantizationConfiguration?

    /// Namespace precedence is safe only after conflicting native declarations are rejected.
    public var nativeQuantization: GLM5NextNativeQuantization? {
        if let quantization { return quantization }
        if case .native(let policy)? = quantizationConfig { return policy }
        return nil
    }

    enum CodingKeys: String, CodingKey, CaseIterable {
        case modelType = "model_type"
        case architectures
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case imageStartTokenId = "image_start_token_id"
        case imageEndTokenId = "image_end_token_id"
        case videoStartTokenId = "video_start_token_id"
        case videoEndTokenId = "video_end_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
        case transformersVersion = "transformers_version"
        case quantization
        case quantizationConfig = "quantization_config"
    }

    public init(from decoder: any Decoder) throws {
        let c = try GLM5NextValidation.container(CodingKeys.self, from: decoder)
        modelType = try c.decode(String.self, forKey: .modelType)
        architectures = try c.decode([String].self, forKey: .architectures)
        textConfig = try c.decode(GLM5NextTextConfiguration.self, forKey: .textConfig)
        visionConfig = try c.decode(GLM5NextVisionConfiguration.self, forKey: .visionConfig)
        imageTokenId = try c.decode(Int.self, forKey: .imageTokenId)
        videoTokenId = try c.decode(Int.self, forKey: .videoTokenId)
        imageStartTokenId = try c.decode(Int.self, forKey: .imageStartTokenId)
        imageEndTokenId = try c.decode(Int.self, forKey: .imageEndTokenId)
        videoStartTokenId = try c.decode(Int.self, forKey: .videoStartTokenId)
        videoEndTokenId = try c.decode(Int.self, forKey: .videoEndTokenId)
        tieWordEmbeddings = try c.decode(Bool.self, forKey: .tieWordEmbeddings)
        transformersVersion = try c.decodeIfPresent(String.self, forKey: .transformersVersion)
        quantization = try c.decodeIfPresent(GLM5NextNativeQuantization.self, forKey: .quantization)
        quantizationConfig = try c.decodeIfPresent(
            GLM5NextQuantizationConfiguration.self, forKey: .quantizationConfig)
        if let quantization, case .native(let alias)? = quantizationConfig {
            try GLM5NextValidation.require(
                quantization.hasSamePolicy(as: alias), "quantization/quantization_config",
                "conflicting native default, module, or skip policies")
        }
        try GLM5NextValidation.require(modelType == "glm5_next", "model_type", "expected glm5_next")
        try GLM5NextValidation.require(
            architectures == ["Glm5NextForConditionalGeneration"], "architectures",
            "unsupported architecture")
        try GLM5NextValidation.require(
            visionConfig.outHiddenSize == textConfig.hiddenSize,
            "vision_config.out_hidden_size", "must match text_config.hidden_size")
        try GLM5NextValidation.require(
            tieWordEmbeddings == textConfig.tieWordEmbeddings,
            "tie_word_embeddings", "root and text values disagree")
        let mediaTokens = [
            imageTokenId, videoTokenId, imageStartTokenId, imageEndTokenId, videoStartTokenId,
            videoEndTokenId,
        ]
        for token in mediaTokens {
            try GLM5NextValidation.token(
                token, vocabularySize: textConfig.vocabularySize, field: "media token IDs")
        }
        try GLM5NextValidation.require(
            Set(mediaTokens).count == mediaTokens.count, "media token IDs", "must be distinct")
    }
}
