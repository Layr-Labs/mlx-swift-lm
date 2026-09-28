// Copyright © 2026 Eigen Labs.
// Configuration preparation only. This file deliberately has no MLX dependency.
import Foundation

public enum MiMoV26ConfigurationError: Error, Equatable, Sendable, CustomStringConvertible, LocalizedError {
    case invalid(field: String, reason: String)

    public var description: String {
        switch self {
        case .invalid(let field, let reason): "MiMo V2.6 configuration \(field): \(reason)"
        }
    }
    public var errorDescription: String? { description }
}

/// Preserves semantic JSON, including unknown nested fields and explicit nulls.
/// Decimal avoids the loss of integer precision caused by a Double-only tree.
public indirect enum MiMoV26JSONValue: Codable, Equatable, Sendable {
    case object([String: Self]), array([Self]), string(String), number(Decimal), bool(Bool), null

    public init(from decoder: any Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let value = try? c.decode(Bool.self) { self = .bool(value) }
        else if let value = try? c.decode(String.self) { self = .string(value) }
        else if let value = try? c.decode([String: Self].self) { self = .object(value) }
        else if let value = try? c.decode([Self].self) { self = .array(value) }
        else { self = .number(try c.decode(Decimal.self)) }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .object(let value): try c.encode(value)
        case .array(let value): try c.encode(value)
        case .string(let value): try c.encode(value)
        case .number(let value): try c.encode(value)
        case .bool(let value): try c.encode(value)
        case .null: try c.encodeNil()
        }
    }
}

private struct MiMoV26Fields {
    let values: [String: MiMoV26JSONValue]
    var prefix = ""

    func error(_ key: String, _ reason: String) -> MiMoV26ConfigurationError {
        .invalid(field: prefix + key, reason: reason)
    }
    func string(_ key: String) throws -> String {
        guard case .string(let value) = values[key] else { throw error(key, "requires a string") }
        return value
    }
    func int(_ key: String, minimum: Int = 1) throws -> Int {
        guard case .number(let value) = values[key] else { throw error(key, "requires an integer") }
        let result = NSDecimalNumber(decimal: value).int64Value
        guard value == Decimal(result), let exact = Int(exactly: result), exact >= minimum else {
            throw error(key, "requires a representable integer >= \(minimum)")
        }
        return exact
    }
    func number(_ key: String, minimum: Double = 0, inclusive: Bool = false) throws -> Double {
        guard case .number(let value) = values[key] else { throw error(key, "requires a number") }
        let result = NSDecimalNumber(decimal: value).doubleValue
        guard result.isFinite, inclusive ? result >= minimum : result > minimum else {
            throw error(key, "requires a finite number \(inclusive ? ">=" : ">") \(minimum)")
        }
        return result
    }
    func bool(_ key: String) throws -> Bool {
        guard case .bool(let value) = values[key] else { throw error(key, "requires a boolean") }
        return value
    }
    func integers(_ key: String, minimum: Int = 0) throws -> [Int] {
        guard case .array(let values) = values[key] else { throw error(key, "requires an integer array") }
        return try values.enumerated().map { index, value in
            try MiMoV26Fields(values: ["value": value], prefix: prefix + "\(key)[\(index)].")
                .int("value", minimum: minimum)
        }
    }
    func strings(_ key: String) throws -> [String] {
        guard case .array(let values) = values[key] else { throw error(key, "requires a string array") }
        return try values.map {
            guard case .string(let value) = $0 else { throw error(key, "requires a string array") }
            return value
        }
    }
    func object(_ key: String) throws -> MiMoV26Fields {
        guard case .object(let value) = values[key] else { throw error(key, "requires an object") }
        return MiMoV26Fields(values: value, prefix: prefix + key + ".")
    }
    func optional<T>(_ key: String, _ read: (String) throws -> T) throws -> T? {
        if values[key] == nil || values[key] == .null { return nil }
        return try read(key)
    }
    func require(_ condition: Bool, _ key: String, _ reason: String) throws {
        if !condition { throw error(key, reason) }
    }
    func oneOf(_ key: String, _ supported: [String]) throws -> String {
        let value = try string(key)
        try require(supported.contains(value), key, "unsupported value \(value)")
        return value
    }
    func product(_ a: Int, _ b: Int, _ key: String) throws -> Int {
        let product = a.multipliedReportingOverflow(by: b)
        try require(!product.overflow && product.partialValue > 0, key, "dimension product overflows")
        return product.partialValue
    }
}

public struct MiMoV26AttentionGeometry: Equatable, Sendable {
    public let queryHeads: Int
    public let keyValueHeads: Int
    public let headDim: Int
    public let valueHeadDim: Int
    public let rotaryDimensions: Int
    public let ropeTheta: Double
    public let slidingWindow: Int?
    public let hasSinks: Bool

    fileprivate init(_ f: MiMoV26Fields, sliding: Bool, factor: Double, window: Int) throws {
        queryHeads = try f.int(sliding ? "swa_num_attention_heads" : "num_attention_heads")
        keyValueHeads = try f.int(sliding ? "swa_num_key_value_heads" : "num_key_value_heads")
        headDim = try f.int(sliding ? "swa_head_dim" : "head_dim")
        valueHeadDim = try f.int(sliding ? "swa_v_head_dim" : "v_head_dim")
        ropeTheta = try f.number(sliding ? "swa_rope_theta" : "rope_theta")
        hasSinks = try f.bool(sliding ? "add_swa_attention_sink_bias" : "add_full_attention_sink_bias")
        slidingWindow = sliding ? window : nil
        try f.require(queryHeads.isMultiple(of: keyValueHeads), "attention_heads", "Q heads must divide into KV groups")
        let rotary = Double(headDim) * factor
        guard rotary.isFinite, rotary >= 2, rotary < Double(Int.max) else {
            throw f.error("partial_rotary_factor", "invalid rotary dimension")
        }
        rotaryDimensions = Int(rotary)
        try f.require(rotaryDimensions <= headDim && rotaryDimensions.isMultiple(of: 2),
                      "partial_rotary_factor", "rotary dimension must be even and within the head")
        _ = try f.product(queryHeads, headDim, "Q projection")
        _ = try f.product(keyValueHeads, headDim, "K projection")
        _ = try f.product(keyValueHeads, valueHeadDim, "V projection")
        _ = try f.product(queryHeads, valueHeadDim, "O projection")
    }
}

/// Original source FP8 metadata and converted MLX policy coexist. Only the
/// latter is an operational native quantization policy; neither proves layout.
public struct MiMoV26Quantization: Equatable, Sendable {
    public struct Policy: Equatable, Sendable {
        public let mode: String
        public let bits: Int
        public let groupSize: Int

        fileprivate init(_ f: MiMoV26Fields) throws {
            mode = try f.oneOf("mode", ["mxfp4", "affine"])
            bits = try f.int("bits")
            groupSize = try f.int("group_size")
            try f.require((mode == "mxfp4" && bits == 4 && groupSize == 32)
                          || (mode == "affine" && [4, 8].contains(bits) && groupSize == 64),
                          "mode", "unsupported native bits/group_size combination")
        }
    }
    public enum Override: Equatable, Sendable { case skip, quantize(Policy) }
    public let nativeDefault: Policy?
    public let nativeOverrides: [String: Override]
    public let sourceMetadata: [String: MiMoV26JSONValue]?

    fileprivate init(_ f: MiMoV26Fields) throws {
        let declared = try f.optional("quantization_config", f.object)
        let primary = try f.optional("quantization", f.object)
        // MLX exporters may repeat their operational policy in both fields.
        // Keep that distinct from the upstream FP8 conversion metadata, and
        // reject contradictory aliases rather than silently picking one.
        let nativeAlias = declared.map { $0.values["mode"] != nil && $0.values["quant_method"] == nil } ?? false
        if nativeAlias, let primary, primary.values != declared?.values {
            throw f.error("quantization_config", "conflicting native quantization aliases")
        }
        sourceMetadata = nativeAlias ? nil : declared?.values
        if let sourceMetadata {
            let source = MiMoV26Fields(values: sourceMetadata, prefix: "quantization_config.")
            _ = try source.oneOf("quant_method", ["fp8"])
            _ = try source.oneOf("store_dtype", ["mxfp4"])
            _ = try source.oneOf("fmt", ["e4m3"])
            _ = try source.oneOf("activation_scheme", ["dynamic"])
            try source.require(try source.int("mxfp4_block_size") == 32,
                               "mxfp4_block_size", "expected group32")
            try source.require(try source.integers("weight_block_size") == [128, 128],
                               "weight_block_size", "expected block128 by128")
            _ = try source.strings("ignored_layers")
        }
        if let native = primary ?? (nativeAlias ? declared : nil) {
            nativeDefault = try Policy(native)
            var overrides: [String: Override] = [:]
            for key in native.values.keys where !["mode", "bits", "group_size"].contains(key) {
                // Unknown non-module metadata is retained by rawFields, never
                // interpreted as a module override or dropped on roundtrip.
                if native.values[key] == .bool(false) { overrides[key] = .skip }
                else if case .object(let object) = native.values[key],
                        !Set(object.keys).isDisjoint(with: ["mode", "bits", "group_size"]) {
                    overrides[key] = .quantize(try Policy(native.object(key)))
                } else if key.contains(".") || native.values[key] == .bool(true) {
                    throw native.error(key, "module override requires a native policy or false")
                }
            }
            nativeOverrides = overrides
        } else {
            nativeDefault = nil
            nativeOverrides = [:]
        }
    }

    public func policy(for modulePath: String) -> Policy? {
        switch nativeOverrides[modulePath] {
        case .skip: nil
        case .quantize(let policy): policy
        case nil: nativeDefault
        }
    }
}

public struct MiMoV26EmbeddedMTPMetadata: Equatable, Sendable {
    public let architecture: String
    public let numLayers: Int
    public let storage: String
    public let file: String

    fileprivate init(_ f: MiMoV26Fields, expectedLayers: Int) throws {
        architecture = try f.oneOf("architecture", ["mimo_v2_nextn"])
        numLayers = try f.int("num_layers")
        storage = try f.oneOf("storage", ["embedded"])
        file = try f.string("file")
        try f.require(numLayers == expectedLayers, "num_layers", "must equal num_nextn_predict_layers")
        try f.require(!file.isEmpty && !file.hasPrefix("/")
                      && !file.split(separator: "/").contains(".."), "file", "requires a relative component path")
    }
}

public struct MiMoV26VisionConfiguration: Equatable, Sendable {
    public let rawFields: [String: MiMoV26JSONValue]
    public let depth, hiddenSize, intermediateSize, queryHeads, keyValueHeads: Int
    public let outputHiddenSize, patchSize, temporalPatchSize, spatialMergeSize: Int
    public let fullAttentionBlocks, windowAttentionTypes: [Int]
    public let usesSinks: Bool

    fileprivate init(_ f: MiMoV26Fields, targetHidden: Int) throws {
        rawFields = f.values
        depth = try f.int("depth")
        hiddenSize = try f.int("hidden_size")
        intermediateSize = try f.int("intermediate_size")
        queryHeads = try f.int("num_heads")
        keyValueHeads = try f.int("num_key_value_heads")
        outputHiddenSize = try f.int("out_hidden_size")
        patchSize = try f.int("patch_size")
        temporalPatchSize = try f.int("temporal_patch_size")
        spatialMergeSize = try f.int("spatial_merge_size")
        fullAttentionBlocks = try f.integers("fullatt_block_indexes")
        windowAttentionTypes = try f.integers("vit_window_attn_types", minimum: -1)
        usesSinks = try f.bool("use_sink")
        _ = try f.oneOf("hidden_act", ["silu"])
        _ = try f.int("window_size")
        _ = try f.int("visual_token_window_size")
        try f.require(hiddenSize.isMultiple(of: queryHeads) && queryHeads.isMultiple(of: keyValueHeads),
                      "num_heads", "invalid vision GQA geometry")
        try f.require(try f.int("num_query_groups") == queryHeads / keyValueHeads,
                      "num_query_groups", "must equal query/KV ratio")
        try f.require(try f.int("spatial_patch_size") == patchSize,
                      "spatial_patch_size", "must agree with patch_size")
        try f.require(outputHiddenSize == targetHidden, "out_hidden_size", "must match text hidden_size")
        try f.require(windowAttentionTypes.count == depth && windowAttentionTypes.allSatisfy { [-1, 0, 1].contains($0) },
                      "vit_window_attn_types", "requires one supported window type per layer")
        try f.require(Set(fullAttentionBlocks).count == fullAttentionBlocks.count
                      && fullAttentionBlocks.allSatisfy { $0 < depth }
                      && Set(fullAttentionBlocks) == Set(windowAttentionTypes.indices.filter { windowAttentionTypes[$0] == -1 }),
                      "fullatt_block_indexes", "must exactly identify full-attention layers")
    }
}

public struct MiMoV26AudioConfiguration: Equatable, Sendable {
    public let rawFields: [String: MiMoV26JSONValue]
    public let channels, segmentSize, groupSize, hiddenSize, queryHeads, headDim, layers: Int
    public let outputHiddenSize, speechVocabularySize, zeroEmbeddingIndex: Int

    fileprivate init(_ f: MiMoV26Fields, targetHidden: Int) throws {
        rawFields = f.values
        channels = try f.int("audio_channels")
        segmentSize = try f.int("audio_segment_size")
        groupSize = try f.int("group_size")
        hiddenSize = try f.int("input_local_dim")
        queryHeads = try f.int("input_local_attn_heads")
        headDim = try f.int("input_local_head_dim")
        layers = try f.int("input_local_layers")
        outputHiddenSize = try f.int("out_hidden_size")
        // These two fields are strings in the official schema. Keep the
        // original JSON representation while exposing validated integers.
        guard let vocabulary = Int(try f.string("speech_vocab_size")), vocabulary > 0,
              let zero = Int(try f.string("speech_zeroemb_idx")), zero >= 0, zero < vocabulary else {
            throw f.error("speech_vocab_size", "invalid speech vocabulary/zero embedding index")
        }
        speechVocabularySize = vocabulary
        zeroEmbeddingIndex = zero
        _ = try f.bool("add_post_norm")
        _ = try f.bool("input_full_attention")
        _ = try f.int("input_local_intermediate_size")
        _ = try f.int("projection_layers")
        _ = try f.number("rope_theta")
        let dropout = try f.number("input_local_hidden_dropout", inclusive: true)
        let rotaryFactor = try f.number("partial_rotary_factor")
        try f.require(dropout < 1 && rotaryFactor <= 1, "input_local_hidden_dropout", "invalid dropout/RoPE factor")
        try f.require(try f.product(queryHeads, headDim, "input_local_dim") == hiddenSize,
                      "input_local_dim", "must equal heads times head dimension")
        try f.require(outputHiddenSize == targetHidden, "out_hidden_size", "must match text hidden_size")
    }
}

/// Native target configuration. Unknown fields survive encoding; accepting
/// metadata is not factory registration or proof of checkpoint completeness.
public struct MiMoV26Configuration: Codable, Equatable, Sendable {
    public let rawFields: [String: MiMoV26JSONValue]
    public let modelType: String
    public let architectures: [String]
    public let hiddenSize, intermediateSize, moeIntermediateSize, vocabularySize, numHiddenLayers: Int
    public let maxPositionEmbeddings, slidingWindow, numNextnPredictLayers: Int
    public let hybridLayerPattern, moeLayerFrequency: [Int]
    public let partialRotaryFactor, attentionValueScale, layernormEpsilon: Double
    public let attentionProjectionLayout, moeRouterDType, hiddenActivation, dtype: String
    public let attentionBias, tieWordEmbeddings, normalizeTopKProbability: Bool
    public let attentionDropout: Double
    public let routedExpertCount, expertsPerToken, expertGroupCount, topKGroups: Int
    public let sharedExpertCount: Int?
    public let routedScalingFactor: Double?
    public let fullAttention, slidingAttention: MiMoV26AttentionGeometry
    public let vision: MiMoV26VisionConfiguration?
    public let audio: MiMoV26AudioConfiguration?
    public let processorFields: [String: MiMoV26JSONValue]?
    public let quantization: MiMoV26Quantization
    public let embeddedMTP: MiMoV26EmbeddedMTPMetadata?
    /// Complete native stop set; tokenIDs retains its primary EOS convenience
    /// entry for scalar callers, not a substitute for generation stop semantics.
    public let eosTokenIDs: Set<Int>
    public let tokenIDs: [String: Int]
    public var numAttentionHeads: Int { fullAttention.queryHeads }
    public var numKeyValueHeads: Int { fullAttention.keyValueHeads }
    public var headDim: Int { fullAttention.headDim }
    public var valueHeadDim: Int { fullAttention.valueHeadDim }

    public init(from decoder: any Decoder) throws {
        try self.init(rawFields: [String: MiMoV26JSONValue](from: decoder))
    }
    public func encode(to encoder: any Encoder) throws { try rawFields.encode(to: encoder) }

    public init(rawFields: [String: MiMoV26JSONValue]) throws {
        let f = MiMoV26Fields(values: rawFields)
        self.rawFields = rawFields
        modelType = try f.oneOf("model_type", ["mimo_v2"])
        architectures = try f.strings("architectures")
        try f.require(architectures == ["MiMoV2ForCausalLM"], "architectures", "requires the native MiMoV2 target")
        hiddenSize = try f.int("hidden_size")
        intermediateSize = try f.int("intermediate_size")
        moeIntermediateSize = try f.int("moe_intermediate_size")
        vocabularySize = try f.int("vocab_size")
        numHiddenLayers = try f.int("num_hidden_layers")
        maxPositionEmbeddings = try f.int("max_position_embeddings")
        slidingWindow = try f.int("sliding_window_size")
        numNextnPredictLayers = try f.int("num_nextn_predict_layers", minimum: 0)
        hybridLayerPattern = try f.integers("hybrid_layer_pattern")
        moeLayerFrequency = try f.integers("moe_layer_freq")
        for (name, pattern) in [("hybrid_layer_pattern", hybridLayerPattern), ("moe_layer_freq", moeLayerFrequency)] {
            try f.require(pattern.count == numHiddenLayers && pattern.allSatisfy { $0 == 0 || $0 == 1 },
                          name, "requires one binary entry per text layer")
        }
        try f.require(try f.int("sliding_window") == slidingWindow,
                      "sliding_window", "must agree with sliding_window_size")
        partialRotaryFactor = try f.number("partial_rotary_factor")
        try f.require(partialRotaryFactor <= 1, "partial_rotary_factor", "must be <=1")
        attentionValueScale = try f.number("attention_value_scale")
        layernormEpsilon = try f.number("layernorm_epsilon")
        attentionProjectionLayout = try f.oneOf("attention_projection_layout", ["split", "fused_qkv"])
        moeRouterDType = try f.oneOf("moe_router_dtype", ["bfloat16", "float32"])
        hiddenActivation = try f.oneOf("hidden_act", ["silu"])
        dtype = try f.oneOf("dtype", ["bfloat16", "float16", "float32"])
        attentionBias = try f.bool("attention_bias")
        tieWordEmbeddings = try f.bool("tie_word_embeddings")
        attentionDropout = try f.number("attention_dropout", inclusive: true)
        try f.require(attentionDropout < 1, "attention_dropout", "must be <1")
        _ = try f.oneOf("scoring_func", ["sigmoid"])
        _ = try f.oneOf("topk_method", ["noaux_tc"])
        routedExpertCount = try f.int("n_routed_experts")
        expertsPerToken = try f.int("num_experts_per_tok")
        expertGroupCount = try f.int("n_group")
        topKGroups = try f.int("topk_group")
        normalizeTopKProbability = try f.bool("norm_topk_prob")
        sharedExpertCount = try f.optional("n_shared_experts") { try f.int($0) }
        routedScalingFactor = try f.optional("routed_scaling_factor") { try f.number($0) }
        try f.require(routedExpertCount.isMultiple(of: expertGroupCount)
                      && topKGroups <= expertGroupCount && expertsPerToken <= routedExpertCount,
                      "n_group", "invalid expert/topk grouping")
        let groupWidth = routedExpertCount / expertGroupCount
        try f.require(groupWidth >= 2 && expertsPerToken <= groupWidth * topKGroups,
                      "num_experts_per_tok", "group selection cannot supply requested experts")
        fullAttention = try .init(f, sliding: false, factor: partialRotaryFactor, window: slidingWindow)
        slidingAttention = try .init(f, sliding: true, factor: partialRotaryFactor, window: slidingWindow)
        _ = try f.product(hiddenSize, intermediateSize, "dense MLP")
        _ = try f.product(hiddenSize, moeIntermediateSize, "expert MLP")
        if let rope = try f.optional("rope_parameters", f.object) {
            _ = try rope.oneOf("rope_type", ["default"])
            if rope.values["type"] != nil { _ = try rope.oneOf("type", ["default"]) }
            try rope.require(try rope.number("rope_theta") == fullAttention.ropeTheta
                             && rope.number("partial_rotary_factor") == partialRotaryFactor,
                             "rope_theta", "nested RoPE must agree with text geometry")
        }
        let targetHiddenSize = hiddenSize
        let predictorCount = numNextnPredictLayers
        vision = try f.optional("vision_config") { try .init(f.object($0), targetHidden: targetHiddenSize) }
        audio = try f.optional("audio_config") { try .init(f.object($0), targetHidden: targetHiddenSize) }
        processorFields = try f.optional("processor_config") { try f.object($0).values }
        quantization = try .init(f)
        embeddedMTP = try f.optional("omlx_mimo_mtp") { try .init(f.object($0), expectedLayers: predictorCount) }
        var tokens: [String: Int] = [:]
        let eos: [Int]
        if case .array = rawFields["eos_token_id"] { eos = try f.integers("eos_token_id") }
        else { eos = [try f.int("eos_token_id", minimum: 0)] }
        let eosVocabularySize = vocabularySize
        try f.require(!eos.isEmpty && Set(eos).count == eos.count && eos.allSatisfy { $0 < eosVocabularySize },
                      "eos_token_id", "requires nonempty unique IDs within the text vocabulary")
        eosTokenIDs = Set(eos)
        tokens["eos_token_id"] = eos[0]
        let mainTokens = ["pad_token_id", "bos_token_id", "image_token_id", "video_token_id",
                          "vision_start_token_id", "vision_end_token_id", "audio_token_id", "audio_start_token_id", "audio_end_token_id"]
        for key in mainTokens {
            if let value = try f.optional(key, { try f.int($0, minimum: 0) }) {
                try f.require(value < vocabularySize, key, "token exceeds text vocabulary")
                tokens[key] = value
            }
        }
        try f.require(tokens["eos_token_id"] != nil && tokens["pad_token_id"] != nil,
                      "eos_token_id", "EOS and padding token IDs are required")
        tokenIDs = tokens
        if vision != nil || audio != nil {
            let processor = try f.object("processor_config")
            let requiredTokens = (vision != nil ? ["image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"] : [])
                + (audio != nil ? ["audio_token_id", "audio_start_token_id", "audio_end_token_id"] : [])
            for key in requiredTokens {
                guard let token = tokens[key] else { throw f.error(key, "modality token is required") }
                try processor.require(try processor.int(key, minimum: 0) == token, key, "must agree with text token namespace")
            }
            if let audio {
                for (key, expected) in [("audio_channels", audio.channels), ("audio_group_size", audio.groupSize),
                                        ("audio_segment_size", audio.segmentSize)] {
                    try processor.require(try processor.int(key) == expected, key, "must agree with audio_config")
                }
                let zeros = try processor.integers("audio_zeroemb_idx")
                try processor.require(zeros.count == audio.channels && zeros.allSatisfy { $0 == audio.zeroEmbeddingIndex },
                                      "audio_zeroemb_idx", "must preserve every speech-channel zero index")
                _ = try processor.int("audio_sampling_rate")
                _ = try processor.int("audio_n_mels")
            }
        }
    }

    public func attentionGeometry(at layer: Int) throws -> MiMoV26AttentionGeometry {
        guard hybridLayerPattern.indices.contains(layer) else {
            throw MiMoV26ConfigurationError.invalid(field: "layer", reason: "out of range")
        }
        return hybridLayerPattern[layer] == 1 ? slidingAttention : fullAttention
    }
}

/// Separate draft metadata; decoding this never changes the native target's
/// architecture or replaces its num_nextn_predict_layers with draft depth.
public struct MiMoV26DFlashMetadata: Codable, Equatable, Sendable {
    public let rawFields: [String: MiMoV26JSONValue]
    public let modelType: String
    public let numHiddenLayers, targetLayerCount, targetHiddenSize, vocabularySize, blockSize, maskTokenID: Int
    public let targetLayerIDs: [Int]

    public init(from decoder: any Decoder) throws {
        rawFields = try [String: MiMoV26JSONValue](from: decoder)
        let f = MiMoV26Fields(values: rawFields)
        modelType = try f.oneOf("model_type", ["qwen3"])
        try f.require(try f.strings("architectures") == ["DFlashDraftModel"], "architectures", "requires separate DFlash draft metadata")
        numHiddenLayers = try f.int("num_hidden_layers")
        targetLayerCount = try f.int("num_target_layers")
        targetHiddenSize = try f.int("target_hidden_size")
        vocabularySize = try f.int("vocab_size")
        blockSize = try f.int("block_size")
        let draft = try f.object("dflash_config")
        maskTokenID = try draft.int("mask_token_id", minimum: 0)
        targetLayerIDs = try draft.integers("target_layer_ids")
        try draft.require(try draft.int("block_size") == blockSize, "block_size", "must agree with outer draft block size")
        try draft.require(!targetLayerIDs.isEmpty && Set(targetLayerIDs).count == targetLayerIDs.count
                          && targetLayerIDs.allSatisfy { $0 < targetLayerCount }, "target_layer_ids", "invalid target feature layers")
        try f.require(try !f.bool("is_causal"), "is_causal", "DFlash block metadata must be noncausal")
        try f.require(vocabularySize > maskTokenID, "vocab_size", "mask token exceeds vocabulary")
    }
    public func encode(to encoder: any Encoder) throws { try rawFields.encode(to: encoder) }

    public func validateTarget(_ target: MiMoV26Configuration) throws {
        guard targetLayerCount == target.numHiddenLayers, targetHiddenSize == target.hiddenSize,
              vocabularySize == target.vocabularySize else {
            throw MiMoV26ConfigurationError.invalid(field: "dflash.target", reason: "target geometry mismatch")
        }
    }
}
