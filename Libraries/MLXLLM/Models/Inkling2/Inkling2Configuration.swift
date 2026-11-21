// Inkling 2 (`inkling2_loop`) configuration. Reference: transformers PR 51050
// `configuration_inkling2_loop.py` (Apache-2.0).
import Foundation

/// The looped core: layers `coreStart ..< coreEnd` run `passes` times.
public struct Inkling2LoopConfiguration: Codable, Sendable, Equatable {
    public var coreStart: Int
    public var coreEnd: Int
    public var passes: Int
    public var gateRank: Int
    public var isVarloop: Bool

    enum CodingKeys: String, CodingKey {
        case coreStart = "core_start"
        case coreEnd = "core_end"
        case passes = "num_passes"
        case gateRank = "gate_rank"
        case isVarloop = "is_varloop"
    }

    public init(coreStart: Int = 6, coreEnd: Int = 18, passes: Int = 3, gateRank: Int = 64) {
        self.coreStart = coreStart
        self.coreEnd = coreEnd
        self.passes = passes
        self.gateRank = gateRank
        self.isVarloop = false
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        coreStart = try c.decodeIfPresent(Int.self, forKey: .coreStart) ?? 6
        coreEnd = try c.decodeIfPresent(Int.self, forKey: .coreEnd) ?? 18
        passes = try c.decodeIfPresent(Int.self, forKey: .passes) ?? 3
        gateRank = try c.decodeIfPresent(Int.self, forKey: .gateRank) ?? 64
        isVarloop = try c.decodeIfPresent(Bool.self, forKey: .isVarloop) ?? false
    }
}

/// DSpark drafter settings, passed through for the drafter owner.
public struct Inkling2MTPConfiguration: Codable, Sendable, Equatable {
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var intermediateSize: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var localLayerIds: [Int]
    public var slidingWindow: Int
    public var blockSize: Int
    public var markovRank: Int
    public var markovHeadType: String
    /// Layer visits whose input hidden state the drafter reads; the last entry
    /// equals the visit count and names the final-norm output.
    public var targetLayerIds: [Int]

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case localLayerIds = "local_layer_ids"
        case slidingWindow = "sliding_window_size"
        case blockSize = "block_size"
        case markovRank = "markov_rank"
        case markovHeadType = "markov_head_type"
        case targetLayerIds = "target_layer_ids"
    }
}

public struct Inkling2TextConfiguration: Codable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var vocabularySize: Int
    public var unpaddedVocabularySize: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var relativeDimensions: Int
    public var relativeExtent: Int
    public var logScalingFloor: Int
    public var logScalingAlpha: Float
    public var rmsNormEps: Float
    public var qkClipThreshold: Float
    public var relativeClipThreshold: Float
    /// Sliding-window layers; every other layer attends the full history.
    public var localLayerIds: [Int]
    public var convKernelSize: Int
    public var finalLogitSoftcap: Float
    public var slidingWindow: Int
    public var layersPerBlock: Int
    public var experts: Int
    public var expertsPerToken: Int
    public var moeIntermediateSize: Int
    public var gateSoftcap: Float
    public var upSoftcap: Float
    public var routeScale: Float
    public var loop: Inkling2LoopConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case vocabularySize = "vocab_size"
        case unpaddedVocabularySize = "unpadded_vocab_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case relativeDimensions = "d_rel"
        case relativeExtent = "rel_extent"
        case logScalingFloor = "log_scaling_n_floor"
        case logScalingAlpha = "log_scaling_alpha"
        case rmsNormEps = "rms_norm_eps"
        case qkClipThreshold = "qk_l2_clip_threshold"
        case relativeClipThreshold = "rel_l2_clip_threshold"
        case localLayerIds = "local_layer_ids"
        case convKernelSize = "sconv_kernel_size"
        case finalLogitSoftcap = "final_logit_softcapping"
        case slidingWindow = "sliding_window_size"
        case layersPerBlock = "n_layers_per_attn_res_block"
        case experts = "n_routed_experts"
        case expertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "intermediate_size"
        case gateSoftcap = "mlp_gate_softcap"
        case upSoftcap = "mlp_up_softcap"
        case routeScale = "route_scale"
        case loop = "loop_config"
    }

    private enum CheckedKeys: String, CodingKey {
        case sharedExperts = "n_shared_experts"
        case qBias = "q_bias"
        case oBias = "o_bias"
        case valueNorm = "use_value_norm_no_gamma"
        case normalizeValues = "attn_res_normalize_values"
        case gateBias = "use_gate_bias"
        case swaHeadDim = "swa_head_dim"
        case swaHeads = "swa_num_attention_heads"
        case swaKVHeads = "swa_num_key_value_heads"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "inkling2_loop_text"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        unpaddedVocabularySize =
            try c.decodeIfPresent(Int.self, forKey: .unpaddedVocabularySize) ?? vocabularySize
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        relativeDimensions = try c.decodeIfPresent(Int.self, forKey: .relativeDimensions) ?? 16
        relativeExtent = try c.decodeIfPresent(Int.self, forKey: .relativeExtent) ?? 1024
        logScalingFloor = try c.decodeIfPresent(Int.self, forKey: .logScalingFloor) ?? 8192
        logScalingAlpha = try c.decodeIfPresent(Float.self, forKey: .logScalingAlpha) ?? 0.1
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        qkClipThreshold =
            try c.decodeIfPresent(Float.self, forKey: .qkClipThreshold) ?? 22.627416997969522
        relativeClipThreshold =
            try c.decodeIfPresent(Float.self, forKey: .relativeClipThreshold) ?? 8
        localLayerIds =
            try c.decodeIfPresent([Int].self, forKey: .localLayerIds)
            ?? (0 ..< hiddenLayers).filter { ($0 + 1) % 6 != 0 }
        convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 4
        finalLogitSoftcap = try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcap) ?? 30
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 1024
        layersPerBlock = try c.decodeIfPresent(Int.self, forKey: .layersPerBlock) ?? 6
        experts = try c.decode(Int.self, forKey: .experts)
        expertsPerToken = try c.decode(Int.self, forKey: .expertsPerToken)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        gateSoftcap = try c.decodeIfPresent(Float.self, forKey: .gateSoftcap) ?? 4
        upSoftcap = try c.decodeIfPresent(Float.self, forKey: .upSoftcap) ?? 25
        routeScale = try c.decodeIfPresent(Float.self, forKey: .routeScale) ?? 8
        loop = try c.decodeIfPresent(Inkling2LoopConfiguration.self, forKey: .loop) ?? .init()

        let k = try decoder.container(keyedBy: CheckedKeys.self)
        func require(_ condition: Bool, _ description: String) throws {
            if !condition {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: decoder.codingPath, debugDescription: "Inkling2: \(description)"
                    ))
            }
        }
        try require(
            (try k.decodeIfPresent(Int.self, forKey: .sharedExperts) ?? 0) == 0,
            "shared experts are not supported")
        try require(
            (try k.decodeIfPresent(Bool.self, forKey: .qBias) ?? false) == false
                && (try k.decodeIfPresent(Bool.self, forKey: .oBias) ?? false) == false,
            "attention biases are not supported")
        try require(
            (try k.decodeIfPresent(Bool.self, forKey: .valueNorm) ?? true)
                && (try k.decodeIfPresent(Bool.self, forKey: .normalizeValues) ?? true)
                && (try k.decodeIfPresent(Bool.self, forKey: .gateBias) ?? true),
            "only normalized values and biased routing are supported")
        try require(
            (try k.decodeIfPresent(Int.self, forKey: .swaHeadDim) ?? headDim) == headDim
                && (try k.decodeIfPresent(Int.self, forKey: .swaHeads) ?? attentionHeads)
                    == attentionHeads
                && (try k.decodeIfPresent(Int.self, forKey: .swaKVHeads) ?? kvHeads) == kvHeads,
            "sliding and full layers must share the head geometry")
        try require(
            [
                hiddenSize, hiddenLayers, vocabularySize, attentionHeads, kvHeads, headDim,
                experts, expertsPerToken, moeIntermediateSize, slidingWindow, layersPerBlock,
                convKernelSize,
            ]
            .allSatisfy { $0 > 0 } && attentionHeads % kvHeads == 0
                && expertsPerToken <= experts && unpaddedVocabularySize <= vocabularySize,
            "invalid dimensions")
        try require(
            loop.passes >= 1 && loop.coreStart > 0 && loop.coreStart < loop.coreEnd
                && loop.coreEnd <= hiddenLayers && loop.coreStart % layersPerBlock == 0
                && loop.coreEnd % layersPerBlock == 0 && !loop.isVarloop,
            "invalid loop configuration")
        try require(
            localLayerIds.allSatisfy { (0 ..< hiddenLayers).contains($0) },
            "local layer index out of range")
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(vocabularySize, forKey: .vocabularySize)
        try c.encode(unpaddedVocabularySize, forKey: .unpaddedVocabularySize)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(kvHeads, forKey: .kvHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(relativeDimensions, forKey: .relativeDimensions)
        try c.encode(relativeExtent, forKey: .relativeExtent)
        try c.encode(logScalingFloor, forKey: .logScalingFloor)
        try c.encode(logScalingAlpha, forKey: .logScalingAlpha)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(qkClipThreshold, forKey: .qkClipThreshold)
        try c.encode(relativeClipThreshold, forKey: .relativeClipThreshold)
        try c.encode(localLayerIds, forKey: .localLayerIds)
        try c.encode(convKernelSize, forKey: .convKernelSize)
        try c.encode(finalLogitSoftcap, forKey: .finalLogitSoftcap)
        try c.encode(slidingWindow, forKey: .slidingWindow)
        try c.encode(layersPerBlock, forKey: .layersPerBlock)
        try c.encode(experts, forKey: .experts)
        try c.encode(expertsPerToken, forKey: .expertsPerToken)
        try c.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try c.encode(gateSoftcap, forKey: .gateSoftcap)
        try c.encode(upSoftcap, forKey: .upSoftcap)
        try c.encode(routeScale, forKey: .routeScale)
        try c.encode(loop, forKey: .loop)
    }

    // MARK: Derived layout

    public func isSliding(layer: Int) -> Bool { localLayerIds.contains(layer) }
    public func isLoopLayer(_ layer: Int) -> Bool {
        (loop.coreStart ..< loop.coreEnd).contains(layer)
    }
    public func isBlockBoundary(_ layer: Int) -> Bool { layer % layersPerBlock == 0 }
    public var loopLayerCount: Int { loop.coreEnd - loop.coreStart }

    /// Layer applications per token (48 for Inkling 2 Mini).
    public var visitCount: Int { hiddenLayers + (loop.passes - 1) * loopLayerCount }

    /// Cache slot of `layer` in pass `pass` (0-based), matching the transformers cache:
    /// slots `0 ..< layers` hold pass 0, later passes of the loop layers follow.
    public func slot(layer: Int, pass: Int) -> Int {
        guard pass > 0 else { return layer }
        precondition(isLoopLayer(layer) && pass < loop.passes)
        return hiddenLayers + (pass - 1) * loopLayerCount + (layer - loop.coreStart)
    }

    /// The stored layer behind each cache slot.
    public var slotLayers: [Int] {
        Array(0 ..< hiddenLayers)
            + (1 ..< max(loop.passes, 1)).flatMap { _ in Array(loop.coreStart ..< loop.coreEnd) }
    }

    /// `(layer, pass)` for every visit, in execution order.
    public var visits: [(layer: Int, pass: Int)] {
        (0 ..< loop.coreStart).map { ($0, 0) }
            + (0 ..< loop.passes).flatMap { p in (loop.coreStart ..< loop.coreEnd).map { ($0, p) } }
            + (loop.coreEnd ..< hiddenLayers).map { ($0, 0) }
    }
}

/// `inkling2_loop` checkpoint configuration (text tower plus the passthrough
/// vision and drafter sections).
public struct Inkling2Configuration: Codable, Sendable {
    public var modelType: String
    public var text: Inkling2TextConfiguration
    public var mtp: Inkling2MTPConfiguration?
    public var eosTokenId: Int?
    public var imageTokenId: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case text = "text_config"
        case mtp = "mtp_config"
        case eosTokenId = "eos_token_id"
        case imageTokenId = "image_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        if c.contains(.text) {
            text = try c.decode(Inkling2TextConfiguration.self, forKey: .text)
        } else {
            text = try Inkling2TextConfiguration(from: decoder)
        }
        mtp = try c.decodeIfPresent(Inkling2MTPConfiguration.self, forKey: .mtp)
        eosTokenId = try c.decodeIfPresent(Int.self, forKey: .eosTokenId)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 200054
    }
}
