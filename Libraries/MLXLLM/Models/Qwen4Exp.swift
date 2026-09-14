// Copyright © 2026 Eigen Labs / Darkbloom Assignment 1.
//
// Native Qwen 3.8 Next / Qwen4-Exp text family (`qwen4_exp` / `qwen4_exp_text`).
// Source of truth: Fusion `mlx_vlm/models/qwen4_exp/` (language.py, config.py).
// This is not a wrap of `qwen3_5`. Instant cache / keepwarm stay out.
// N-gram PLE remains packed on SSD; hardware-tier qualification is separate.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Qwen4ExpTextConfiguration: Codable, Sendable {
    public var modelType: String = "qwen4_exp_text"
    public var hiddenSize: Int = 2560
    public var hiddenLayers: Int = 48
    public var attentionHeads: Int = 24
    public var kvHeads: Int = 2
    public var headDim: Int = 256
    public var linearNumValueHeads: Int = 48
    public var linearNumKeyHeads: Int = 16
    public var linearKeyHeadDim: Int = 128
    public var linearValueHeadDim: Int = 128
    public var linearConvKernelDim: Int = 4
    public var rmsNormEps: Float = 1e-6
    public var vocabularySize: Int = 248_320
    public var maxPositionEmbeddings: Int = 262_144
    public var tieWordEmbeddings: Bool = false
    public var attentionBias: Bool = false
    public var fullAttentionInterval: Int = 4
    public var layerTypes: [String] = []
    public var hcCount: Int = 4
    public var hcLowrank: Int = 320
    public var pleLayerIds: [Int] = []
    public var pleEmbedDim: Int = 2560
    public var pleConvKernelSize: Int = 4
    public var ngramSize: Int = 3
    public var headsPerNgram: Int = 8
    public var ngramVocabSizeBase: Int = 20_000_000
    public var makeNgramVocabSizeDivisibleBy: Int = 128
    public var splitNgramParts: Int = 128
    public var indexerNHeads: Int = 4
    public var indexerKVHeads: Int = 1
    public var indexerHeadDim: Int = 128
    public var indexerBudget: Int = 2048
    public var indexerCompressRatio: Int = 4
    public var outputGateType: String = "sigmoid"
    public var numExperts: Int = 512
    public var numExpertsPerTok: Int = 10
    public var sharedExpertIntermediateSize: Int = 640
    public var moeIntermediateSize: Int = 640
    public var normTopkProb: Bool = true
    public var ropeTheta: Float = 10_000_000
    public var partialRotaryFactor: Float = 0.25
    public var mropeSection: [Int] = [11, 11, 10]
    public var eosTokenId: [Int] = []
    public var mtpNumHiddenLayers: Int = 0
    public var seed: Int = Qwen4ExpNGramGeometry.defaultSeed

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case fullAttentionInterval = "full_attention_interval"
        case layerTypes = "layer_types"
        case hcCount = "hc_count"
        case hcLowrank = "hc_lowrank"
        case pleLayerIds = "ple_layer_ids"
        case pleEmbedDim = "ple_embed_dim"
        case pleConvKernelSize = "ple_conv_kernel_size"
        case ngramSize = "ngram_size"
        case headsPerNgram = "heads_per_ngram"
        case ngramVocabSizeBase = "ngram_vocab_size_base"
        case makeNgramVocabSizeDivisibleBy = "make_ngram_vocab_size_divisible_by"
        case splitNgramParts = "split_ngram_parts"
        case indexerNHeads = "indexer_n_heads"
        case indexerKVHeads = "indexer_kv_heads"
        case indexerHeadDim = "indexer_head_dim"
        case indexerBudget = "indexer_budget"
        case indexerCompressRatio = "indexer_compress_ratio"
        case outputGateType = "output_gate_type"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case normTopkProb = "norm_topk_prob"
        case ropeParameters = "rope_parameters"
        case eosTokenId = "eos_token_id"
        case mtpNumHiddenLayers = "mtp_num_hidden_layers"
        case seed
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen4_exp_text"
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2560
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 48
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 24
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 2
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        linearNumValueHeads = try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 48
        linearNumKeyHeads = try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
        linearKeyHeadDim = try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 128
        linearValueHeadDim = try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
        linearConvKernelDim = try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 248_320
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262_144
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        fullAttentionInterval =
            try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        hcCount = try container.decodeIfPresent(Int.self, forKey: .hcCount) ?? 4
        hcLowrank = try container.decodeIfPresent(Int.self, forKey: .hcLowrank) ?? 320
        pleLayerIds = try container.decodeIfPresent([Int].self, forKey: .pleLayerIds) ?? []
        pleEmbedDim = try container.decodeIfPresent(Int.self, forKey: .pleEmbedDim) ?? hiddenSize
        pleConvKernelSize = try container.decodeIfPresent(Int.self, forKey: .pleConvKernelSize) ?? 4
        ngramSize = try container.decodeIfPresent(Int.self, forKey: .ngramSize) ?? 3
        headsPerNgram = try container.decodeIfPresent(Int.self, forKey: .headsPerNgram) ?? 8
        ngramVocabSizeBase =
            try container.decodeIfPresent(Int.self, forKey: .ngramVocabSizeBase) ?? 20_000_000
        makeNgramVocabSizeDivisibleBy =
            try container.decodeIfPresent(Int.self, forKey: .makeNgramVocabSizeDivisibleBy) ?? 128
        splitNgramParts = try container.decodeIfPresent(Int.self, forKey: .splitNgramParts) ?? 128
        indexerNHeads = try container.decodeIfPresent(Int.self, forKey: .indexerNHeads) ?? 4
        indexerKVHeads = try container.decodeIfPresent(Int.self, forKey: .indexerKVHeads) ?? 1
        indexerHeadDim = try container.decodeIfPresent(Int.self, forKey: .indexerHeadDim) ?? 128
        indexerBudget = try container.decodeIfPresent(Int.self, forKey: .indexerBudget) ?? 2048
        indexerCompressRatio =
            try container.decodeIfPresent(Int.self, forKey: .indexerCompressRatio) ?? 4
        outputGateType = try container.decodeIfPresent(String.self, forKey: .outputGateType) ?? "sigmoid"
        numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 512
        numExpertsPerTok = try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 10
        sharedExpertIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 640
        moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 640
        normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        mtpNumHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .mtpNumHiddenLayers) ?? 0
        seed = try container.decodeIfPresent(Int.self, forKey: .seed) ?? Qwen4ExpNGramGeometry.defaultSeed

        if let rope = try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
        {
            ropeTheta = rope["rope_theta"]?.asFloat() ?? 10_000_000
            partialRotaryFactor = rope["partial_rotary_factor"]?.asFloat() ?? 0.25
            mropeSection = rope["mrope_section"]?.asInts() ?? [11, 11, 10]
        }

        if let eos = try container.decodeIfPresent(IntOrIntArray.self, forKey: .eosTokenId) {
            eosTokenId = eos.values
        }

        if layerTypes.isEmpty {
            layerTypes = (0 ..< hiddenLayers).map { index in
                (index + 1) % fullAttentionInterval == 0
                    ? "qwen_sparse_attention" : "linear_attention"
            }
        } else {
            layerTypes = layerTypes.map { kind in
                kind == "full_attention" ? "qwen_sparse_attention" : kind
            }
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(kvHeads, forKey: .kvHeads)
        try container.encode(headDim, forKey: .headDim)
        try container.encode(linearNumValueHeads, forKey: .linearNumValueHeads)
        try container.encode(linearNumKeyHeads, forKey: .linearNumKeyHeads)
        try container.encode(linearKeyHeadDim, forKey: .linearKeyHeadDim)
        try container.encode(linearValueHeadDim, forKey: .linearValueHeadDim)
        try container.encode(linearConvKernelDim, forKey: .linearConvKernelDim)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encode(attentionBias, forKey: .attentionBias)
        try container.encode(fullAttentionInterval, forKey: .fullAttentionInterval)
        try container.encode(layerTypes, forKey: .layerTypes)
        try container.encode(hcCount, forKey: .hcCount)
        try container.encode(hcLowrank, forKey: .hcLowrank)
        try container.encode(pleLayerIds, forKey: .pleLayerIds)
        try container.encode(pleEmbedDim, forKey: .pleEmbedDim)
        try container.encode(numExperts, forKey: .numExperts)
        try container.encode(numExpertsPerTok, forKey: .numExpertsPerTok)
        try container.encode(sharedExpertIntermediateSize, forKey: .sharedExpertIntermediateSize)
        try container.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
    }

    /// QSA / full-attention rows only. GDN stays in recurrent state.
    public var cbv2LayerKinds: [CBv2LayerKind] {
        (0 ..< hiddenLayers).compactMap { modelLayerIndex in
            guard layerTypes[modelLayerIndex] == "qwen_sparse_attention" else { return nil }
            return CBv2LayerKind(
                attention: .full,
                headDim: headDim,
                kvHeads: kvHeads,
                queryHeads: attentionHeads,
                modelLayerIndex: modelLayerIndex,
                extraStorageBytesPerToken: Qwen4ExpPrefillMemory.qsaSidecarBytesPerToken(
                    indexerHeadDim: indexerHeadDim,
                    compressRatio: indexerCompressRatio),
                qwen4IndexerCompressRatio: indexerCompressRatio)
        }
    }

    public func cbv2RecurrentStateSpec(
        activationDType: DType = .bfloat16
    ) -> CBv2RecurrentStateSpec {
        let keyDim = linearNumKeyHeads * linearKeyHeadDim
        let valueDim = linearNumValueHeads * linearValueHeadDim
        let convDim = 2 * keyDim + valueDim
        var layers: [CBv2RecurrentLayerStateSpec] = (0 ..< hiddenLayers).compactMap { index in
            guard layerTypes[index] == "linear_attention" else { return nil }
            return CBv2RecurrentLayerStateSpec(
                modelLayerIndex: index,
                convShape: [1, max(0, linearConvKernelDim - 1), convDim],
                convDType: activationDType,
                ssmShape: [1, linearNumValueHeads, linearValueHeadDim, linearKeyHeadDim],
                ssmDType: .float32)
        }
        let hcHidden = hcCount * hiddenSize
        let shortConv = max(0, (pleConvKernelSize - 1) * ngramSize)
        let contextLen = max(0, ngramSize - 1)
        for layerIndex in 0 ..< hiddenLayers where pleLayerIds.contains(layerIndex + 1) {
            layers.append(
                CBv2RecurrentLayerStateSpec(
                    modelLayerIndex: Qwen4ExpNGramGeometry.recurrentLayerIndex(layerIndex),
                    convShape: [1, shortConv, hcHidden],
                    convDType: activationDType,
                    ssmShape: [1, 1, 1, contextLen],
                    ssmDType: .float32))
        }
        return CBv2RecurrentStateSpec(layers: layers)
    }

    public var cbv2Capabilities: CBv2ModelCapabilities {
        var capabilities = CBv2ModelCapabilities.initialRecurrentTarget
        // Lightning MTP is a native CBv2 seam (Fusion Qwen4ExpMTPModule).
        // Compact GDN replay stays off so PLE and GDN share captured stacks.
        capabilities.supportsMTP = true
        capabilities.supportsCompactRecurrentMTPReplay = false
        capabilities.supportsPackedPrefill = false
        capabilities.supportsPrefixReuse = false
        // Complete checkpoints include QSA side-state, GDN/PLE rows and the
        // assistant history; ordinary KV-only prefix reuse remains disabled.
        capabilities.supportsRecurrentCheckpointReuse = true
        capabilities.supportsPagedKV = true
        capabilities.requiresNativePagedKV = true
        return capabilities
    }

    public var qsaLayerCount: Int {
        layerTypes.filter { $0 == "qwen_sparse_attention" }.count
    }

    /// Bridge into the existing Qwen3.5 GDN / MoE modules (same kernel shapes).
    func qwen35BridgeConfiguration() throws -> Qwen35TextConfiguration {
        let object: [String: Any] = [
            "model_type": "qwen3_5_text",
            "hidden_size": hiddenSize,
            "num_hidden_layers": hiddenLayers,
            "num_attention_heads": attentionHeads,
            "num_key_value_heads": kvHeads,
            "head_dim": headDim,
            "linear_num_value_heads": linearNumValueHeads,
            "linear_num_key_heads": linearNumKeyHeads,
            "linear_key_head_dim": linearKeyHeadDim,
            "linear_value_head_dim": linearValueHeadDim,
            "linear_conv_kernel_dim": linearConvKernelDim,
            "rms_norm_eps": rmsNormEps,
            "vocab_size": vocabularySize,
            "max_position_embeddings": maxPositionEmbeddings,
            "tie_word_embeddings": tieWordEmbeddings,
            "attention_bias": attentionBias,
            "full_attention_interval": fullAttentionInterval,
            "num_experts": numExperts,
            "num_experts_per_tok": numExpertsPerTok,
            "shared_expert_intermediate_size": sharedExpertIntermediateSize,
            "moe_intermediate_size": moeIntermediateSize,
            "norm_topk_prob": normTopkProb,
            "mtp_num_hidden_layers": 0,
            "rope_parameters": [
                "type": "default",
                "mrope_section": mropeSection,
                "rope_theta": ropeTheta,
                "partial_rotary_factor": partialRotaryFactor,
            ] as [String: Any],
        ]
        let data = try JSONSerialization.data(withJSONObject: object)
        return try JSONDecoder().decode(Qwen35TextConfiguration.self, from: data)
    }
}

public struct Qwen4ExpConfiguration: Codable, Sendable {
    public var modelType: String = "qwen4_exp"
    public var textConfig: Qwen4ExpTextConfiguration
    public var imageTokenId: Int?
    public var videoTokenId: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
    }

    public init(textConfig: Qwen4ExpTextConfiguration = Qwen4ExpTextConfiguration()) {
        self.textConfig = textConfig
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen4_exp"
        if let text = try container.decodeIfPresent(Qwen4ExpTextConfiguration.self, forKey: .textConfig)
        {
            textConfig = text
        } else {
            textConfig = try Qwen4ExpTextConfiguration(from: decoder)
        }
        imageTokenId = try container.decodeIfPresent(Int.self, forKey: .imageTokenId)
        videoTokenId = try container.decodeIfPresent(Int.self, forKey: .videoTokenId)
    }
}

private enum IntOrIntArray: Codable, Sendable {
    case int(Int)
    case ints([Int])

    var values: [Int] {
        switch self {
        case .int(let value): return [value]
        case .ints(let values): return values
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else {
            self = .ints(try container.decode([Int].self))
        }
    }
}

// MARK: - RMSNorm (zero-centered checkpoint, 1 + weight)

final class Qwen4ExpRMSNorm: Module {
    let eps: Float
    let groupSize: Int?
    let dimensions: Int
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int, eps: Float, groupSize: Int? = nil) {
        self.eps = eps
        self.groupSize = groupSize
        self.dimensions = dimensions
        _weight.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        var y = x.asType(.float32)
        var scale = weight.asType(.float32)
        if let groupSize {
            let groups = dimensions / groupSize
            y = y.reshaped(y.shape.dropLast() + [groups, groupSize])
            scale = scale.reshaped([groups, groupSize])
        }
        y = y * rsqrt(y.square().mean(axis: -1, keepDims: true) + eps)
        y = y * (1 + scale)
        return y.reshaped(x.shape).asType(dtype)
    }
}

// MARK: - Hyper-connection

/// Gate for the compiled single-token hyper-connection forward (Fusion
/// `compile_hyper_connections`). Same predicate as Fusion's `__call__`:
/// rank-3, `[1, 1, hc·hidden]`, bf16. `DARKBLOOM_QWEN4_HC_COMPILE=0` is a
/// parity kill switch; `MLX_COMPILED_DECODE` (shared with the MoE fusions)
/// is honoured so the M1/M2 + Tahoe opt-out still covers this path.
public enum Qwen4ExpHyperConnectionCompile: Sendable {
    public static let envFlag = "DARKBLOOM_QWEN4_HC_COMPILE"

    public struct Snapshot: Sendable, Equatable {
        public let compiled: Int
        public let canonical: Int
        public init(compiled: Int, canonical: Int) {
            self.compiled = compiled
            self.canonical = canonical
        }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var compiledCalls = 0
    nonisolated(unsafe) private static var canonicalCalls = 0

    public static func isEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        guard MLXHardwareInfo.isCompiledDecodeSupported else { return false }
        guard let raw = environment[envFlag]?.trimmingCharacters(in: .whitespaces).lowercased()
        else { return true }
        return !["0", "false", "off", "no"].contains(raw)
    }

    /// Widest `[1, T, hc·hidden]` shape that runs through the compiled trace.
    /// Covers T=1 decode and every Lightning MTP verify width (1+k ≤ 16);
    /// MLX `compile` keeps one trace per input shape, so this bounds the
    /// cache at 16 traces per mixer. Prefill chunks stay canonical.
    public static let maxCompiledWidth = 16

    static func eligible(_ hyperInput: MLXArray) -> Bool {
        hyperInput.ndim == 3 && hyperInput.dim(0) == 1
            && (1 ... maxCompiledWidth).contains(hyperInput.dim(1))
            && hyperInput.dtype == .bfloat16 && isEnabled()
    }

    static func recordCompiled() {
        lock.withLock { compiledCalls += 1 }
    }

    static func recordCanonical() {
        lock.withLock { canonicalCalls += 1 }
    }

    public static func snapshot() -> Snapshot {
        lock.withLock { Snapshot(compiled: compiledCalls, canonical: canonicalCalls) }
    }

    public static func resetForTesting() {
        lock.withLock {
            compiledCalls = 0
            canonicalCalls = 0
        }
    }
}

final class Qwen4ExpGatedResidual: Module {
    let hcCount: Int
    let hiddenSize: Int
    let hcLowrank: Int
    let useCombine: Bool

    @ModuleInfo(key: "hc_norm") var hcNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "input_mix_weight_down") var inputMixWeightDown: Linear
    @ModuleInfo(key: "input_mix_weight_up") var inputMixWeightUp: Linear
    @ModuleInfo(key: "block_inject_weight") var blockInjectWeight: Linear?

    init(_ args: Qwen4ExpTextConfiguration, useCombine: Bool = true) {
        self.hcCount = args.hcCount
        self.hiddenSize = args.hiddenSize
        self.hcLowrank = args.hcLowrank
        self.useCombine = useCombine
        let hcHidden = args.hcCount * args.hiddenSize
        _hcNorm.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: hcHidden, eps: args.rmsNormEps, groupSize: args.hiddenSize)
        _inputMixWeightDown.wrappedValue = Linear(hcHidden, args.hcLowrank, bias: false)
        _inputMixWeightUp.wrappedValue = Linear(args.hcLowrank, hcHidden, bias: false)
        if useCombine {
            _blockInjectWeight.wrappedValue = Linear(hcHidden, args.hcCount, bias: false)
        }
        super.init()
    }

    /// Fusion `Qwen4ExpGatedResidual._forward`.
    ///
    /// Decoder (`useCombine`): `(mixed_input, hyper_input, 2·σ(inject/hc))`.
    /// Final mixer (`useCombine == false`): call `mix` — it returns only
    /// `[B, T, hidden_size]` and has no `block_inject_weight` in the card.
    func callAsFunction(_ hyperInput: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let projected = project(hyperInput)
        return (projected.mixed, hyperInput, projected.injection)
    }

    func mix(_ hyperInput: MLXArray) -> MLXArray {
        project(hyperInput).mixed
    }

    /// Fusion `compile_hyper_connections` + `Qwen4ExpGatedResidual.__call__`:
    /// B1 bf16 shapes up to `maxCompiledWidth` tokens run through an
    /// `mx.compile` trace of the canonical forward, so the norm / SiLU /
    /// sigmoid / mean chains fuse instead of dispatching ~20 kernels per
    /// mixer per layer. Fusion compiles only T=1; Darkbloom also compiles the
    /// Lightning MTP verify widths because the round is host-encode bound
    /// and 96 canonical mixers per verify cost ~1,300 graph nodes. Fusion is
    /// elementwise-exact at any shape (same per-element ops and dtypes), so
    /// the trace is bit-identical to the canonical graph. Prefill chunks
    /// keep the uncompiled graph. The trace holds the checkpoint banks as
    /// constants (Fusion compiles the bound method the same way); weights
    /// are final before the first decode.
    private let compiledDecodeLock = NSLock()
    private var compiledDecode: (@Sendable ([MLXArray]) -> [MLXArray])?

    #if DEBUG
        private let testPathLock = NSLock()
        private var testCompiledCalls = 0
        private var testCanonicalCalls = 0
        /// Per-instance evidence: process-global counters include other
        /// models and suites, so they cannot prove this module's dispatch.
        var compilationCounts: (compiled: Int, canonical: Int) {
            testPathLock.withLock { (testCompiledCalls, testCanonicalCalls) }
        }
    #endif

    private func project(_ hyperInput: MLXArray) -> (mixed: MLXArray, injection: MLXArray) {
        if Qwen4ExpHyperConnectionCompile.eligible(hyperInput) {
            let out = compiledDecodeForward()([hyperInput])
            if out.count == 2 {
                Qwen4ExpHyperConnectionCompile.recordCompiled()
                #if DEBUG
                    testPathLock.withLock { testCompiledCalls += 1 }
                #endif
                return (out[0], out[1])
            }
        }
        Qwen4ExpHyperConnectionCompile.recordCanonical()
        #if DEBUG
            testPathLock.withLock { testCanonicalCalls += 1 }
        #endif
        return projectCanonical(hyperInput)
    }

    private func compiledDecodeForward() -> @Sendable ([MLXArray]) -> [MLXArray] {
        compiledDecodeLock.withLock {
            if let compiledDecode { return compiledDecode }
            let fn: @Sendable ([MLXArray]) -> [MLXArray] = compile { [unowned self] args in
                let projected = self.projectCanonical(args[0])
                return [projected.mixed, projected.injection]
            }
            compiledDecode = fn
            return fn
        }
    }

    /// Canonical uncompiled graph (Fusion `_forward`). Internal so tests can
    /// pin the compiled trace against it.
    func projectCanonical(_ hyperInput: MLXArray) -> (mixed: MLXArray, injection: MLXArray) {
        let normed = hcNorm(hyperInput)
        let hc = Float(hcCount)
        // Fusion `hybrid_projection`: at the decode shape the raw `down` and
        // `inject` banks come out of one dispatch (bit-identical to the two
        // stock qmv calls). Slices of `combined` are views, no copy kernel.
        var mix: MLXArray
        var injectRaw: MLXArray? = nil
        if let blockInjectWeight,
            let combined = Qwen4ExpHCHybrid.apply(
                normed, down: inputMixWeightDown, injection: blockInjectWeight)
        {
            mix = combined[.ellipsis, 0 ..< hcLowrank]
            injectRaw = combined[.ellipsis, hcLowrank...]
        } else {
            if let blockInjectWeight, Qwen4ExpAffineQMV.logShapes {
                let d = inputMixWeightDown as? QuantizedLinear
                let i = blockInjectWeight as? QuantizedLinear
                Qwen4ExpAffineQMV.logShapeOnce(
                    "hc-hybrid-declined x=\(normed.shape) \(normed.dtype) hyper=\(hyperInput.dtype) "
                        + "down=\(d.map { "\($0.bits)b g\($0.groupSize) \(type(of: $0))" } ?? "dense") "
                        + "inject=\(i.map { "\($0.bits)b g\($0.groupSize) \(type(of: $0))" } ?? "dense")")
            }
            mix = Qwen4ExpAffineQMM.apply(inputMixWeightDown, normed)
        }
        mix = silu(mix / hc)
        mix = sigmoid(Qwen4ExpAffineQMM.apply(inputMixWeightUp, mix))
        let streamShape = Array(normed.shape.dropLast()) + [hcCount, hiddenSize]
        mix = mix.reshaped(streamShape)
        let streams = normed.reshaped(streamShape)
        let mixed = Qwen4ExpActivation.keep((mix * streams).mean(axis: -2))
        let injection: MLXArray
        if let blockInjectWeight {
            let raw = injectRaw ?? Qwen4ExpAffineQMM.apply(blockInjectWeight, normed)
            injection = Qwen4ExpActivation.keep(2.0 * sigmoid(raw / hc))
        } else {
            injection = MLXArray.ones(
                [hyperInput.dim(0), hyperInput.dim(1), hcCount], dtype: hyperInput.dtype)
        }
        return (mixed, injection)
    }
}

// MARK: - QSA indexer + attention

final class Qwen4ExpQSAIndexer: Module {
    let nHeads: Int
    let kvHeads: Int
    let headDim: Int
    let tokenBudget: Int
    let compressRatio: Int

    @ModuleInfo(key: "index_qk_proj") var indexQKProj: Linear
    @ModuleInfo(key: "q_layernorm") var qLayerNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "k_layernorm") var kLayerNorm: Qwen4ExpRMSNorm

    init(_ args: Qwen4ExpTextConfiguration) {
        self.nHeads = args.indexerNHeads
        self.kvHeads = args.indexerKVHeads
        self.headDim = args.indexerHeadDim
        self.tokenBudget = args.indexerBudget
        self.compressRatio = args.indexerCompressRatio
        _indexQKProj.wrappedValue = Linear(
            args.hiddenSize, (nHeads + kvHeads) * headDim, bias: false)
        _qLayerNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kLayerNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        super.init()
    }
}

final class Qwen4ExpAttention: Module {
    var batchedQSAEnabled = Qwen4ExpBatchedQSA.isEnabled()
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer
    let mrope: Qwen35MRoPE

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "indexer") var indexer: Qwen4ExpQSAIndexer

    init(_ args: Qwen4ExpTextConfiguration) {
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDim = args.headDim
        self.scale = pow(Float(args.headDim), -0.5)
        _qProj.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * args.headDim * 2, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * args.headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * args.headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(
            args.attentionHeads * args.headDim, args.hiddenSize, bias: args.attentionBias)
        _qNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
        _indexer.wrappedValue = Qwen4ExpQSAIndexer(args)
        let ropeDims = max(1, Int(Float(args.headDim) * args.partialRotaryFactor))
        self.rope = initializeRope(
            dims: ropeDims,
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: [
                "type": .string("default"),
                "mrope_section": .ints(args.mropeSection),
                "rope_theta": .float(args.ropeTheta),
                "partial_rotary_factor": .float(args.partialRotaryFactor),
            ],
            maxPositionEmbeddings: args.maxPositionEmbeddings)
        self.mrope = Qwen35MRoPE(
            rope: self.rope, dim: ropeDims, base: args.ropeTheta,
            scalingConfig: [
                "type": .string("default"),
                "mrope_section": .ints(args.mropeSection),
                "rope_theta": .float(args.ropeTheta),
                "partial_rotary_factor": .float(args.partialRotaryFactor),
            ],
            sections: args.mropeSection)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let qSplit = qwen4Linear(qProj, x).reshaped(B, L, attentionHeads, -1).split(
            parts: 2, axis: -1)
        var queries = qNorm(qSplit[0]).transposed(0, 2, 1, 3)
        let gate = qSplit[1].reshaped(B, L, -1)
        var keys = kNorm(qwen4Linear(kProj, x).reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        var values = qwen4Linear(vProj, x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)
        let attended = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask)
        let output = attended.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return qwen4Linear(oProj, output * sigmoid(gate))
    }

    func cbv2Forward(
        _ x: MLXArray, cache: any CBv2AttendingLayerCache,
        positionIds: MLXArray? = nil,
        batchedQSAEnabled: Bool? = nil
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        precondition(B == 1 || ((batchedQSAEnabled ?? self.batchedQSAEnabled)
                     && B <= CBv2Qwen4BatchPolicy.maximumCandidateRows),
                     "Qwen4 multirow QSA requires the explicit experimental batching capability")
        let qSplit = qwen4Linear(qProj, x).reshaped(B, L, attentionHeads, -1).split(
            parts: 2, axis: -1)
        let qsaType = Qwen4ExpNativeSparseGQA.activationType(x.dtype)
        let queries = qNorm(qSplit[0]).transposed(0, 2, 1, 3).asType(qsaType)
        let gate = qSplit[1].reshaped(B, L, -1)
        let keys = kNorm(qwen4Linear(kProj, x).reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
            .asType(qsaType)
        let values = qwen4Linear(vProj, x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
            .asType(qsaType)
        let output: MLXArray
        if B == 1 {
            output = attendProjected(
                x, queries: queries, keys: keys, values: values,
                cache: cache, positionIds: positionIds)
        } else {
            guard let rowProvider = cache as? any CBv2Qwen4BatchScopeProviding else {
                preconditionFailure("Qwen4 batched QSA requires request-owned sparse cache views")
            }
            let scope = rowProvider.qwen4BeginBatchScope()
            precondition(scope.caches.count == B, "Qwen4 QSA batch/cache row mismatch")
            let indexProjection = qwen4Linear(indexer.indexQKProj, x).reshaped(
                B, L, indexer.nHeads + indexer.kvHeads, indexer.headDim)
            var rows: [MLXArray] = []
            rows.reserveCapacity(B)
            for row in 0 ..< B {
                let slice = row ..< row + 1
                rows.append(attendProjected(
                    x[slice], queries: queries[slice], keys: keys[slice], values: values[slice],
                    cache: scope.caches[row],
                    positionIds: Qwen4ExpBatchedQSA.positions(positionIds, row: row, batch: B, length: L),
                    indexProjection: indexProjection[slice], batchedRow: true))
            }
            if scope.finish(length: L) {
                Qwen4ExpBatchedQSAInvocation.record(batch: B)
            }
            output = concatenated(rows, axis: 0)
        }
        return qwen4Linear(oProj, output * sigmoid(gate))
    }

    /// The original singleton attention arithmetic, after projection. A
    /// multirow caller supplies its own cache/positions and rejoins the
    /// output before the batched output projection.
    private func attendProjected(
        _ x: MLXArray, queries inputQueries: MLXArray, keys inputKeys: MLXArray,
        values: MLXArray, cache: any CBv2AttendingLayerCache,
        positionIds: MLXArray?, indexProjection: MLXArray? = nil,
        batchedRow: Bool = false
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        precondition(B == 1, "Qwen4 sparse attention requires a request-local view")
        if let validation = cache as? any CBv2Qwen4GatheredCache,
            !validation.qwen4ValidateProjectedKV(keys: inputKeys, values: values)
        {
            // Shape-only placeholder. The engine rejects this entire graph
            // before sampling/evaluation and owns fenced row retirement.
            return MLXArray.zeros([B, L, attentionHeads * headDim], dtype: x.dtype)
        }
        var queries = inputQueries
        var keys = inputKeys
        let textPositions = Qwen4ExpTextPositions.canonicalText(positionIds, length: L)
        if let positionIds {
            (queries, keys) = mrope.apply(
                queries: queries, keys: keys, positionIds: positionIds)
        } else {
            let offsets = cache.positionOffsets + 0
            queries = rope(queries, offset: offsets)
            keys = rope(keys, offset: offsets)
        }
        let offset = (cache as? KVCache)?.offset ?? 0
        let gatheredCache = cache as? CBv2Qwen4GatheredCache
        defer {
            // Index and KV frontiers now describe the same completed graph.
            // The row owns the sidecar across bank recomposition and release.
            if let gatheredCache, !gatheredCache.qwen4HasWriteFault {
                if let layer = cache as? CBv2LayerCache, layer.rows.count == 1 {
                    CBv2Qwen4IndexerBind.harvest(gatheredCache, into: layer.rows[0])
                } else if let layer = cache as? PagedLayerCache, layer.rows.count == 1 {
                    CBv2Qwen4IndexerBind.harvest(gatheredCache, into: layer.rows[0])
                }
            }
        }
        let (indexerReady, indexQueries) = updateIndexer(
            hidden: x, cache: gatheredCache, offset: offset,
            positionIds: positionIds, textPositions: textPositions, length: L,
            indexProjection: indexProjection)
        precondition(!batchedRow || (indexerReady && indexQueries != nil),
                     "Qwen4 batched QSA is missing the row's committed index history")
        let longContextText = gatheredPrefillEligible(
            batch: B, length: L, offset: offset, positionIds: positionIds)
        if longContextText,
            let crossingCache = cache as? (any CBv2AttendingLayerCache & CBv2Qwen4GatheredCache),
            crossingCache.qwen4SerializesRectangularAttention,
            Qwen4ExpGatheredQSA.serialCrossoverEligible(
                offset: offset, length: L, compressRatio: indexer.compressRatio, tokenBudget: indexer.tokenBudget)
        {
            guard indexerReady, let indexQueries else {
                preconditionFailure("Qwen4 MTP crossover is missing trusted indexer state")
            }
            let output = Qwen4ExpGatheredQSA.attendCrossover(
                cache: crossingCache, queries: queries, keys: keys, values: values,
                indexQueries: indexQueries, offset: offset,
                queryHeads: attentionHeads, kvHeads: kvHeads, headDim: headDim,
                indexerHeadDim: indexer.headDim, compressRatio: indexer.compressRatio,
                tokenBudget: indexer.tokenBudget, scale: scale,
                indexKeyNorm: { self.indexer.kLayerNorm($0) },
                applyIndexRope: { states, positions in self.applyIndexRope(states, positions: positions) })
            return output.reshaped(B, L, -1).asType(x.dtype)
        }
        if longContextText, let gatheredCache,
            gatheredCache.qwen4SerializesRectangularAttention,
            !Qwen4ExpNativeSparseGQA.canAttend(
                queries: queries, keys: keys, values: values,
                selectedWidth: indexer.tokenBudget / indexer.compressRatio)
        {
            guard indexerReady, let indexQueries else {
                preconditionFailure("Qwen4 canonical sparse verify is missing trusted indexer state")
            }
            let views = gatheredCache.updateKVAndAdvanceOffsets(keys: keys, values: values)
            if gatheredCache.qwen4HasWriteFault {
                return MLXArray.zeros([B, L, attentionHeads * headDim], dtype: x.dtype)
            }
            precondition(views.count == 1)
            let output = Qwen4ExpGatheredQSA.attendCanonicalSparseColumns(
                cache: gatheredCache, queries: queries, keys: views[0].keys, values: views[0].values,
                indexQueries: indexQueries, queryHeads: attentionHeads, kvHeads: kvHeads,
                headDim: headDim, indexerHeadDim: indexer.headDim,
                compressRatio: indexer.compressRatio, tokenBudget: indexer.tokenBudget,
                indexKeyNorm: { self.indexer.kLayerNorm($0) },
                applyIndexRope: { states, positions in self.applyIndexRope(states, positions: positions) })
            return output.reshaped(B, L, -1).asType(x.dtype)
        }
        if Qwen4ExpCompactQSA.enabled(), indexerReady, let indexQueries,
            indexer.compressRatio == 4, indexer.tokenBudget == 2048,
            Qwen4ExpTextPositions.isBatchOneText(positionIds, length: L),
            let selectedCache = cache as? any CBv2Qwen4SelectedKVCache,
            let output = Qwen4ExpCompactQSA.attend(
                queries: queries, keys: keys, values: values, indexQueries: indexQueries,
                cache: selectedCache, offset: offset, indexerHeadDim: indexer.headDim,
                indexKeyNorm: { self.indexer.kLayerNorm($0) },
                applyIndexRope: { states, positions in self.applyIndexRope(states, positions: positions) })
        {
            return output.reshaped(B, L, -1).asType(x.dtype)
        }
        if gatheredVerifyEligible(
            batch: B, length: L, offset: offset, positionIds: positionIds)
        {
            // mlx-serve `qsaVerifyGatherAttn` (S=2..15). Prefill native GQA
            // still streams the whole KV; that is the 80K Lightning TTFT miss.
            guard let gatheredCache, let indexQueries, indexerReady else {
                preconditionFailure(
                    "Qwen4 QSA long-context text verify cannot fall back to dense SDPA")
            }
            let views = gatheredCache.updateKVAndAdvanceOffsets(keys: keys, values: values)
            precondition(views.count == 1, "Qwen4 gathered QSA is B1-only")
            let kvLen = views[0].keys.dim(2)
            let indexKeyNorm: (MLXArray) -> MLXArray = { self.indexer.kLayerNorm($0) }
            let applyIndexRope: (MLXArray, MLXArray) -> MLXArray = { states, positions in
                self.applyIndexRope(states, positions: positions)
            }
            let pooled = Qwen4ExpPooledIndex.reuseOrCompute(
                cache: gatheredCache,
                compressRatio: indexer.compressRatio,
                logicalTokens: kvLen,
                indexKeyNorm: indexKeyNorm,
                applyIndexRope: applyIndexRope)
            let output = Qwen4ExpGatheredQSA.attendVerify(
                queries: queries,
                keys: views[0].keys,
                values: views[0].values,
                indexQueries: indexQueries,
                pooledIndexKeys: pooled,
                queryHeads: attentionHeads,
                kvHeads: kvHeads,
                headDim: headDim,
                indexerHeadDim: indexer.headDim,
                compressRatio: indexer.compressRatio,
                tokenBudget: indexer.tokenBudget,
                minKeyTokens: Qwen4ExpGatheredQSA.verifyMinKeyTokens())
            return output.reshaped(B, L, -1).asType(x.dtype)
        }
        if longContextText {
            guard let gatheredCache, let indexQueries, indexerReady else {
                let stored = gatheredCache?.qwen4IndexKeys?.dim(1) ?? -1
                let storedPos = gatheredCache?.qwen4IndexPositionIds?.dim(-1) ?? -1
                let detail =
                    "gathered=\(gatheredCache != nil) ready=\(indexerReady) iq=\(indexQueries != nil) B=\(B) L=\(L) offset=\(offset) storedKeys=\(stored) storedPos=\(storedPos)"
                #if DEBUG
                    try? detail.write(
                        to: URL(fileURLWithPath: "/tmp/qwen4-qsa-gathered-prefill-trap.txt"),
                        atomically: true, encoding: .utf8)
                #endif
                preconditionFailure(
                    "Qwen4 QSA long-context text prefill cannot fall back to dense SDPA \(detail)")
            }
            let views = gatheredCache.updateKVAndAdvanceOffsets(keys: keys, values: values)
            precondition(views.count == 1, "Qwen4 gathered QSA is B1-only")
            let indexKeyNorm: (MLXArray) -> MLXArray = { self.indexer.kLayerNorm($0) }
            let applyIndexRope: (MLXArray, MLXArray) -> MLXArray = { states, positions in
                self.applyIndexRope(states, positions: positions)
            }
            let kvLen = views[0].keys.dim(2)
            let pooled = Qwen4ExpPooledIndex.reuseOrCompute(
                cache: gatheredCache,
                compressRatio: indexer.compressRatio,
                logicalTokens: kvLen,
                indexKeyNorm: indexKeyNorm,
                applyIndexRope: applyIndexRope)
            guard let indexKeys = Qwen4ExpIndexCapacity.logicalTokens(
                gatheredCache.qwen4IndexKeys, length: kvLen),
                let indexPositionIds = Qwen4ExpIndexCapacity.logicalPositions(
                    gatheredCache.qwen4IndexPositionIds, length: kvLen)
            else {
                preconditionFailure(
                    "Qwen4 QSA indexer side-state shorter than the gathered KV")
            }
            let output = Qwen4ExpGatheredQSA.attend(
                queries: queries,
                keys: views[0].keys,
                values: views[0].values,
                indexQueries: indexQueries,
                indexKeys: indexKeys,
                indexPositionIds: indexPositionIds,
                queryHeads: attentionHeads,
                kvHeads: kvHeads,
                headDim: headDim,
                indexerHeadDim: indexer.headDim,
                compressRatio: indexer.compressRatio,
                tokenBudget: indexer.tokenBudget,
                indexKeyNorm: indexKeyNorm,
                applyIndexRope: applyIndexRope,
                pooledIndexKeys: pooled)
            Qwen4ExpQSALayerSync.afterAttend(output, keyTokens: views[0].keys.dim(2))
            return output.reshaped(B, L, -1).asType(x.dtype)
        }
        if gatheredDecodeEligible(batch: B, length: L, offset: offset, positionIds: positionIds) {
            // Fusion `_gathered_text_decode`. Past the budget the model is
            // sparse on every step; dense SDPA here would attend to rows QSA
            // never selects. B1 text always carries aligned indexer state from
            // its own prefill, so a miss is a contract break, not a fallback.
            guard let gatheredCache, let indexQueries, indexerReady else {
                preconditionFailure(
                    "Qwen4 QSA long-context text decode cannot fall back to dense SDPA")
            }
            let views = Qwen4ExpDecodeProfile.stageKV("qsa.kv_read") {
                gatheredCache.updateKVAndAdvanceOffsets(keys: keys, values: values)
            }
            precondition(views.count == 1, "Qwen4 gathered QSA is B1-only")
            let kvLen = views[0].keys.dim(2)
            let indexKeyNorm: (MLXArray) -> MLXArray = { self.indexer.kLayerNorm($0) }
            let applyIndexRope: (MLXArray, MLXArray) -> MLXArray = { states, positions in
                self.applyIndexRope(states, positions: positions)
            }
            let pooled = Qwen4ExpDecodeProfile.stage("qsa.pool") { Qwen4ExpPooledIndex.reuseOrCompute(
                cache: gatheredCache,
                compressRatio: indexer.compressRatio,
                logicalTokens: kvLen,
                indexKeyNorm: indexKeyNorm,
                applyIndexRope: applyIndexRope) }
            let output = Qwen4ExpDecodeProfile.stage("qsa.attend") { Qwen4ExpGatheredQSA.attendDecode(
                queries: queries,
                keys: views[0].keys,
                values: views[0].values,
                indexQueries: indexQueries,
                pooledIndexKeys: pooled,
                queryHeads: attentionHeads,
                kvHeads: kvHeads,
                headDim: headDim,
                indexerHeadDim: indexer.headDim,
                compressRatio: indexer.compressRatio,
                tokenBudget: indexer.tokenBudget) }
            return output.reshaped(B, L, -1).asType(x.dtype)
        }
        if batchedRow {
            let sparseRequired = L > 1 ? offset + L > indexer.tokenBudget
                : Qwen4ExpGatheredQSA.decodeCrossesBudget(
                    offset: offset, compressRatio: indexer.compressRatio, tokenBudget: indexer.tokenBudget)
            precondition(!sparseRequired,
                         "Qwen4 batched long-context attention cannot fall back to dense SDPA")
        }
        Qwen4ExpQSAInvocation.recordDense()
        let output = cache.updateAndAttend(
            queries: queries, keys: keys, values: values,
            scale: scale, sinks: nil)
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)
            .asType(x.dtype)
        return output
    }

    private func qwen4Linear(_ linear: Linear, _ x: MLXArray) -> MLXArray {
        Qwen4ExpAffineQMM.apply(linear, x)
    }

    /// Fusion `_gathered_text_prefill_eligible` widened to B1 multimodal:
    /// an image/video window past the QSA budget is indexed with its genuine
    /// three-plane M-RoPE (stored in the sidecar) instead of falling to dense
    /// SDPA, which would attend to rows official QSA never selects.
    private func gatheredPrefillEligible(
        batch: Int, length: Int, offset: Int, positionIds: MLXArray?
    ) -> Bool {
        batch == 1
            && length > 1
            && offset + length > indexer.tokenBudget
            && Qwen4ExpTextPositions.isBatchOnePositions(positionIds, length: length)
    }

    /// mlx-serve verify-width gather: B1 text, S in 2..15, past the QSA
    /// budget and the union-vs-cache floor. Dense SDPA here would attend to
    /// rows QSA never selects.
    private func gatheredVerifyEligible(
        batch: Int, length: Int, offset: Int, positionIds: MLXArray?
    ) -> Bool {
        guard Qwen4ExpGatheredQSA.verifyGatherEnabled() else { return false }
        guard batch == 1,
            length >= 2,
            length <= Qwen4ExpGatheredQSA.verifyMaxQueryTokens,
            Qwen4ExpTextPositions.isBatchOneText(positionIds, length: length)
        else { return false }
        let keyTokens = offset + length
        guard keyTokens > Qwen4ExpGatheredQSA.verifyMinKeyTokens() else { return false }
        guard keyTokens > indexer.tokenBudget else { return false }
        let blockBudget = indexer.tokenBudget / max(indexer.compressRatio, 1)
        guard let geom = Qwen4ExpGatheredQSA.verifyGeometry(
            queryTokens: length, keyTokens: keyTokens,
            compressRatio: indexer.compressRatio, blockBudget: blockBudget),
            geom.gatheredRows > 0, geom.gatheredRows < keyTokens
        else { return false }
        return Qwen4ExpGatheredQSA.decodeCrossesBudget(
            offset: offset, compressRatio: indexer.compressRatio,
            tokenBudget: indexer.tokenBudget)
    }

    /// Request-local singleton whose
    /// completed blocks exceed the budget. Below that, official full attention
    /// is exactly QSA (every block is selected) and stays on dense SDPA.
    private func gatheredDecodeEligible(
        batch: Int, length: Int, offset: Int, positionIds: MLXArray?
    ) -> Bool {
        batch == 1
            && length == 1
            && Qwen4ExpGatheredQSA.decodeCrossesBudget(
                offset: offset, compressRatio: indexer.compressRatio,
                tokenBudget: indexer.tokenBudget)
            && Qwen4ExpTextPositions.isBatchOnePositions(positionIds, length: 1)
    }

    private func applyIndexRope(_ states: MLXArray, positions: MLXArray) -> MLXArray {
        mrope.apply(queries: states, keys: states, positionIds: positions).0
    }

    private func updateIndexer(
        hidden: MLXArray, cache: CBv2Qwen4GatheredCache?, offset: Int,
        positionIds: MLXArray?, textPositions: MLXArray?, length: Int,
        indexProjection: MLXArray? = nil
    ) -> (Bool, MLXArray?) {
        guard let cache else { return (false, nil) }
        let B = hidden.dim(0)
        precondition(B == 1, "Qwen4 index state must be updated per request")
        if cache.qwen4IndexKeys == nil {
            if let layer = cache as? CBv2LayerCache, layer.rows.count == 1 {
                _ = CBv2Qwen4IndexerBind.restore(cache, from: layer.rows[0])
            } else if let layer = cache as? PagedLayerCache, layer.rows.count == 1 {
                _ = CBv2Qwen4IndexerBind.restore(cache, from: layer.rows[0])
            }
        }
        guard CBv2Qwen4IndexerFrontier.covers(
            offset, count: cache.qwen4IndexTokenCount,
            keys: cache.qwen4IndexKeys, positions: cache.qwen4IndexPositionIds)
        else {
            cache.clearQwen4IndexerState()
            return (false, nil)
        }
        Qwen4ExpPooledIndex.reconcileCommittedFrontier(
            cache: cache,
            committedTokens: offset,
            compressRatio: indexer.compressRatio)
        let projected = indexProjection ?? qwen4Linear(indexer.indexQKProj, hidden).reshaped(
            B, length, indexer.nHeads + indexer.kvHeads, indexer.headDim)
        var indexQueries = indexer.qLayerNorm(projected[0..., 0..., 0 ..< indexer.nHeads, 0...])
            .transposed(0, 2, 1, 3)
        let rawKeys = projected[0..., 0..., indexer.nHeads..., 0...].squeezed(axis: 2)
        // Fusion indexer `_apply_rope(x, position_ids)`: index queries and the
        // stored raw keys carry the same M-RoPE the attention path uses, so an
        // image window is roped with its genuine three planes, not a text ramp.
        let positions = Qwen4ExpTextPositions.indexerPositions(
            positionIds: positionIds, textPositions: textPositions,
            offset: offset, length: length)
        indexQueries = applyIndexRope(indexQueries, positions: positions).transposed(0, 2, 1, 3)
        let end = offset + length
        if Qwen4ExpIndexCapacity.isEnabled() {
            if let stored = cache.qwen4IndexKeys, let storedPos = cache.qwen4IndexPositionIds {
                guard stored.dim(1) >= offset, storedPos.dim(-1) >= offset else {
                    cache.clearQwen4IndexerState()
                    return (false, nil)
                }
                let aligned = Qwen4ExpIndexCapacity.alignPositionRanks(
                    stored: storedPos, rows: positions)
                cache.qwen4IndexKeys = Qwen4ExpIndexCapacity.appendTokens(
                    buffer: stored, offset: offset, rows: rawKeys)
                cache.qwen4IndexPositionIds = Qwen4ExpIndexCapacity.appendPositions(
                    buffer: aligned.stored, offset: offset, rows: aligned.rows)
            } else if offset == 0 {
                cache.qwen4IndexKeys = Qwen4ExpIndexCapacity.appendTokens(
                    buffer: nil, offset: 0, rows: rawKeys)
                cache.qwen4IndexPositionIds = Qwen4ExpIndexCapacity.appendPositions(
                    buffer: nil, offset: 0, rows: positions)
            } else {
                cache.clearQwen4IndexerState()
                return (false, nil)
            }
            // Do not eval the capacity bank here. Prefill attend already
            // materializes the logical prefix; a per-layer eval is the same
            // sync that made `DARKBLOOM_QWEN4_QSA_POOLED_INCREMENTAL=1` miss
            // 80K live (late_visible 90.3 s). Default is this buffer;
            // concat is the kill-switch path.
        } else if let stored = cache.qwen4IndexKeys, let storedPos = cache.qwen4IndexPositionIds {
            guard stored.dim(1) >= offset, storedPos.dim(-1) >= offset else {
                cache.clearQwen4IndexerState()
                return (false, nil)
            }
            let prefixKeys = stored.dim(1) == offset ? stored : stored[0..., 0 ..< offset, 0...]
            let prefixPos =
                storedPos.dim(-1) == offset
                ? storedPos : Qwen4ExpIndexCapacity.logicalPositions(storedPos, length: offset)
            guard let prefixPos else {
                cache.clearQwen4IndexerState()
                return (false, nil)
            }
            let aligned = Qwen4ExpIndexCapacity.alignPositionRanks(
                stored: prefixPos, rows: positions)
            cache.qwen4IndexKeys = concatenated([prefixKeys, rawKeys], axis: 1)
            cache.qwen4IndexPositionIds = concatenated(
                [aligned.stored, aligned.rows], axis: -1)
        } else if offset == 0 {
            cache.qwen4IndexKeys = rawKeys
            cache.qwen4IndexPositionIds = positions
        } else {
            cache.clearQwen4IndexerState()
            return (false, nil)
        }
        let keysReady = Qwen4ExpIndexCapacity.logicalTokens(cache.qwen4IndexKeys, length: end) != nil
            && Qwen4ExpIndexCapacity.logicalPositions(cache.qwen4IndexPositionIds, length: end)
                != nil
        if keysReady {
            cache.qwen4IndexTokenCount = end
        }
        return (keysReady, indexQueries)
    }
}

// MARK: - Decoder

final class Qwen4ExpDecoderLayer: Module {
    let isLinear: Bool
    let hasPLE: Bool

    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen35GatedDeltaNet?
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen4ExpAttention?
    @ModuleInfo(key: "mlp") var mlp: Qwen35SparseMoeBlock
    @ModuleInfo(key: "ple") var ple: Qwen4ExpPLELayer?
    @ModuleInfo(key: "attn_hyper_connection") var attnHyperConnection: Qwen4ExpGatedResidual
    @ModuleInfo(key: "mlp_hyper_connection") var mlpHyperConnection: Qwen4ExpGatedResidual

    init(
        _ args: Qwen4ExpTextConfiguration, layerIdx: Int, mmapPLE: Bool,
        fuseGateUp: Bool = true
    ) throws {
        let layerType = args.layerTypes[layerIdx]
        self.isLinear = layerType == "linear_attention"
        self.hasPLE = args.pleLayerIds.contains(layerIdx + 1)
        let bridge = try args.qwen35BridgeConfiguration()
        if isLinear {
            _linearAttn.wrappedValue = Qwen35GatedDeltaNet(
                bridge, qkNormalization: .qwen4L2, outputGate: .sigmoid)
        } else {
            _selfAttn.wrappedValue = Qwen4ExpAttention(args)
        }
        _mlp.wrappedValue = Qwen35SparseMoeBlock(
            bridge,
            fuseGateUp: fuseGateUp,
            weightedReductionProfile: .qwen4ProductionSwiGLU)
        if hasPLE, let pleIndex = args.pleLayerIds.firstIndex(of: layerIdx + 1) {
            _ple.wrappedValue = Qwen4ExpPLELayer(
                args, layerIndex: layerIdx, pleIndex: pleIndex, mmap: mmapPLE)
        }
        _attnHyperConnection.wrappedValue = Qwen4ExpGatedResidual(args)
        _mlpHyperConnection.wrappedValue = Qwen4ExpGatedResidual(args)
        super.init()
    }

    func callAsFunction(
        _ hidden: MLXArray,
        inputIds: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        var hiddenStates = Qwen4ExpActivation.keep(hidden)
        if let ple {
            hiddenStates = hiddenStates + ple(hiddenStates, inputIds: inputIds)
        }
        let (attnMixed, attnHyper, attnInject) = attnHyperConnection(hiddenStates)
        let attnBranch: MLXArray
        if isLinear {
            attnBranch = linearAttn!(
                attnMixed, mask: ssmMask, cache: cache as? MambaCache, nConfirmed: 0)
        } else {
            attnBranch = selfAttn!(attnMixed, mask: attentionMask, cache: cache)
        }
        hiddenStates = applyHyperInjection(
            branch: attnBranch, hyper: attnHyper, injection: attnInject)
        let (mlpMixed, mlpHyper, mlpInject) = mlpHyperConnection(hiddenStates)
        let mlpBranch = mlp(mlpMixed)
        return applyHyperInjection(branch: mlpBranch, hyper: mlpHyper, injection: mlpInject)
    }

    func cbv2Forward(
        _ x: MLXArray,
        inputIds: MLXArray,
        modelLayerIndex: Int,
        attentionCache: (any CBv2AttendingLayerCache)?,
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?,
        captureRecurrentWindow: Bool = false
    ) -> MLXArray {
        // Stage keys mirror Fusion's `_profile_stage` boundaries; no-ops
        // unless DARKBLOOM_QWEN4_DECODE_PROFILE=1 sampled this call.
        typealias P = Qwen4ExpDecodeProfile
        var hiddenStates = Qwen4ExpActivation.keep(x)
        if let ple {
            hiddenStates = P.stage("ple") {
                hiddenStates + ple.cbv2Forward(
                    hiddenStates, inputIds: inputIds, recurrentState: recurrentState,
                    captureRecurrentWindow: captureRecurrentWindow)
            }
        }
        P.dumpStage("input", layer: modelLayerIndex, hiddenStates)
        let (attnMixed, attnHyper, attnInject) = P.stage3("attn_hc") {
            attnHyperConnection(hiddenStates)
        }
        P.dumpStage("attn_hc.mixed", layer: modelLayerIndex, attnMixed)
        P.dumpStage("attn_hc.inject", layer: modelLayerIndex, attnInject)
        let attnBranch: MLXArray
        if isLinear {
            precondition(attentionCache == nil, "Qwen4 recurrent layer received attention KV")
            attnBranch = P.stage("gdn") {
                if captureRecurrentWindow {
                    linearAttn!.cbv2ForwardCaptured(
                        attnMixed, modelLayerIndex: modelLayerIndex,
                        recurrentState: recurrentState,
                        preferCapturedStacks: true)
                } else {
                    linearAttn!.cbv2Forward(
                        attnMixed, modelLayerIndex: modelLayerIndex,
                        recurrentState: recurrentState)
                }
            }
        } else {
            guard let attentionCache else {
                preconditionFailure("Qwen4 QSA layer is missing its CBv2 cache")
            }
            attnBranch = P.stage("qsa") {
                selfAttn!.cbv2Forward(attnMixed, cache: attentionCache, positionIds: positionIds)
            }
        }
        P.dumpStage("attn_branch", layer: modelLayerIndex, attnBranch)
        hiddenStates = P.stage("attn_residual") {
            applyHyperInjection(branch: attnBranch, hyper: attnHyper, injection: attnInject)
        }
        P.dumpStage("attn_residual", layer: modelLayerIndex, hiddenStates)
        let (mlpMixed, mlpHyper, mlpInject) = P.stage3("mlp_hc") {
            mlpHyperConnection(hiddenStates)
        }
        P.dumpStage("mlp_hc.mixed", layer: modelLayerIndex, mlpMixed)
        let mlpBranch = P.stage("moe") { mlp(mlpMixed) }
        P.dumpStage("moe", layer: modelLayerIndex, mlpBranch)
        let out = P.stage("mlp_residual") {
            applyHyperInjection(branch: mlpBranch, hyper: mlpHyper, injection: mlpInject)
        }
        P.dumpStage("mlp_residual", layer: modelLayerIndex, out)
        return out
    }

    private func applyHyperInjection(
        branch: MLXArray, hyper: MLXArray, injection: MLXArray
    ) -> MLXArray {
        if Qwen4ExpFusions.isEnabled {
            return Qwen4ExpFusions.hyperInjection(branch: branch, hyper: hyper, injection: injection)
        }
        let weighted = branch.expandedDimensions(axis: -2) * injection.expandedDimensions(axis: -1)
        return Qwen4ExpActivation.keep(hyper + weighted.reshaped(hyper.shape))
    }
}

final class Qwen4ExpTextModelInner: Module {
    let configuration: Qwen4ExpTextConfiguration
    let batchingPolicy = Qwen4ExpBatchedQSAPolicyState()
    var currentBatchedQSAPolicy: Bool {
        let attention = layers.compactMap(\.selfAttn)
        return !attention.isEmpty && attention.allSatisfy(\.batchedQSAEnabled)
    }
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen4ExpDecoderLayer]
    @ModuleInfo(key: "hyper_connection_mixer") var hyperConnectionMixer: Qwen4ExpGatedResidual

    init(_ args: Qwen4ExpTextConfiguration, mmapPLE: Bool) throws {
        self.configuration = args
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        _layers.wrappedValue = try (0 ..< args.hiddenLayers).map { index in
            try Qwen4ExpDecoderLayer(args, layerIdx: index, mmapPLE: mmapPLE)
        }
        // oQ4e has no final `norm`. Fusion tiles embeddings by `hc_count`
        // and mixes the residual streams here (`use_combine=False`).
        _hyperConnectionMixer.wrappedValue = Qwen4ExpGatedResidual(args, useCombine: false)
        super.init()
    }

    func embedAndTile(_ inputs: MLXArray, inputEmbeddings: MLXArray? = nil) -> MLXArray {
        let embedded = inputEmbeddings ?? embedTokens(inputs)
        return Qwen4ExpActivation.keep(
            tiled(embedded, repetitions: [1, 1, configuration.hcCount]))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        batchingPolicy.sealOnForward { currentBatchedQSAPolicy }
        var hidden = embedAndTile(inputs)
        let mask = createAttentionMask(h: hidden, cache: cache)
        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden,
                inputIds: inputs,
                attentionMask: mask,
                ssmMask: nil,
                cache: cache?[index])
        }
        return hyperConnectionMixer.mix(hidden)
    }

    func cbv2ForwardResidual(
        _ inputs: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        caches: [any CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray? = nil,
        captureRecurrentWindow: Bool = false
    ) -> MLXArray {
        batchingPolicy.sealOnForward { currentBatchedQSAPolicy }
        let shapeCall = CBv2ForwardShapeObservation.isActive
            ? CBv2ForwardShapeObservation.beginTarget(liveBatchRows: inputs.dim(0), sequenceWidth: inputs.dim(1)) : nil
        defer { shapeCall?.end() }
        precondition(
            caches.count == layers.filter({ !$0.isLinear }).count,
            "Qwen4 CBv2 requires only QSA caches")
        var hiddenStates = Qwen4ExpDecodeProfile.stage("embed+mask") {
            embedAndTile(inputs, inputEmbeddings: inputEmbeddings)
        }
        var attentionIndex = 0
        for (modelLayerIndex, layer) in layers.enumerated() {
            let attentionCache: (any CBv2AttendingLayerCache)?
            if layer.isLinear {
                attentionCache = nil
            } else {
                attentionCache = caches[attentionIndex]
                precondition(
                    attentionCache!.kind.modelLayerIndex == nil
                        || attentionCache!.kind.modelLayerIndex == modelLayerIndex,
                    "Qwen4 CBv2 attention cache mapped to the wrong model layer")
                attentionIndex += 1
            }
            hiddenStates = layer.cbv2Forward(
                hiddenStates,
                inputIds: inputs,
                modelLayerIndex: modelLayerIndex,
                attentionCache: attentionCache,
                recurrentState: recurrentState,
                positionIds: positionIds,
                captureRecurrentWindow: captureRecurrentWindow)
        }
        return hiddenStates
    }

    func cbv2Forward(
        _ inputs: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        caches: [any CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray? = nil,
        captureRecurrentWindow: Bool = false
    ) -> MLXArray {
        let residual = cbv2ForwardResidual(
            inputs, inputEmbeddings: inputEmbeddings, caches: caches,
            recurrentState: recurrentState, positionIds: positionIds,
            captureRecurrentWindow: captureRecurrentWindow)
        return Qwen4ExpDecodeProfile.stage("final_hc") { hyperConnectionMixer.mix(residual) }
    }
}

/// Explicit teardown seam for SSD-backed PLE resources. Providers call this
/// after draining the engine and before dropping the model container; object
/// deinitialization remains a backstop, not the primary lifecycle.
public protocol Qwen4ExpExternalPLEReleasing: AnyObject {
    func releaseExternalPLEResources()
}

/// Load-time validation seam for SSD-backed PLE resources. Providers invoke
/// this after checkpoint weights are loaded but before constructing a serving
/// engine, so a missing/corrupt mmap catalog is a model-load error instead of
/// a process trap during the first request.
public protocol Qwen4ExpExternalPLEValidating: AnyObject {
    func validateExternalPLEResources() throws
}

public class Qwen4ExpTextModel:
    Module, LLMModel, KVCacheDimensionProvider
{
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let configuration: Qwen4ExpTextConfiguration
    public let mmapPLE: Bool
    private let pleDirectoryLease: Qwen4ExpPLEDirectoryLease?

    @ModuleInfo(key: "model") var model: Qwen4ExpTextModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    /// Staged by `loadWeights` before `sanitize` so split oQ4e
    /// `switch_mlp.{gate,up}_proj` halves fuse only when they share a policy.
    public var checkpointPerLayerQuantization: BaseConfiguration.PerLayerQuantization?

    public init(_ args: Qwen4ExpTextConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.mmapPLE = Qwen4ExpPLEResidency.useMmap
        self.pleDirectoryLease = mmapPLE ? Qwen4ExpPLEResidency.retainCurrentDirectory() : nil
        do {
            _model.wrappedValue = try Qwen4ExpTextModelInner(args, mmapPLE: mmapPLE)
        } catch {
            fatalError("Qwen4ExpTextModel construction failed: \(error)")
        }
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        headLogits(model(inputs, cache: cache))
    }

    /// Head logits through the exact affine QMV whenever the shape allows.
    /// Stock `QuantizedLinear` picks a different Metal kernel for one row
    /// (qmv) than for a verify window (qmm), and the two round the 8-bit
    /// g64 lm_head differently (3 of 4,096 outputs at N=4,096 in a direct
    /// check). Lightning verify columns must score with the same bits as
    /// serial decode or MTP and non-MTP outputs diverge at temperature 0.
    /// The QMV equals stock qmv at T=1, so serial decode is unchanged.
    func headLogits(_ hidden: MLXArray) -> MLXArray {
        if let lmHead {
            if Qwen4ExpEnvironment.snapshot["DARKBLOOM_QWEN4_EXACT_HEAD"] == "0" {
                return lmHead(hidden)
            }
            return Qwen4ExpAffineQMM.apply(lmHead, hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        model.layers.map { layer in
            layer.isLinear ? MambaCache() : KVCacheSimple()
        }
    }

    public var loraLayers: [Module] { model.layers }

    public var cbv2LayerKinds: [CBv2LayerKind] { configuration.cbv2LayerKinds }

    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec {
        configuration.cbv2RecurrentStateSpec(
            activationDType: Qwen4ExpActivation.hiddenDType(from: model.embedTokens))
    }

    public var cbv2Capabilities: CBv2ModelCapabilities { configuration.cbv2Capabilities }
    public func newCacheV2(
        makeLayerCache: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
            any CBv2AttendingLayerCache
    ) rethrows -> [any CBv2AttendingLayerCache] {
        try cbv2LayerKinds.enumerated().map { storageIndex, kind in
            try makeLayerCache(kind.modelLayerIndex ?? storageIndex, kind)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        weights = weights.filter { key, _ in
            !Qwen4ExpWeightSanitizer.shouldDrop(key, mmapPLE: mmapPLE)
        }
        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)
        for (key, value) in weights {
            var value = value
            if key.contains("conv1d.weight") && value.ndim == 3 && value.dim(-1) != 1 {
                value = value.movedAxis(source: 2, destination: 1)
            }
            sanitized[key] = value
        }
        if configuration.numExperts > 0 {
            sanitized = qwen35FuseSwitchMLPGateUp(
                weights: sanitized,
                perLayerQuantization: checkpointPerLayerQuantization,
                setFused: { qwen35SetSwitchGLUGateUpFused($1, at: $0, in: self) })
        }
        Qwen4ExpWeightSanitizer.injectMissingNgramWeightScale(
            into: &sanitized, pleLayerIds: configuration.pleLayerIds)
        return sanitized
    }
}

public class Qwen4ExpModel:
    Module, LLMModel, KVCacheDimensionProvider
{
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: Qwen4ExpTextModel

    public init(_ args: Qwen4ExpConfiguration) {
        let text = Qwen4ExpTextModel(args.textConfig)
        self.vocabularySize = text.vocabularySize
        self.kvHeads = text.kvHeads
        _languageModel.wrappedValue = text
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    public var loraLayers: [Module] { languageModel.loraLayers }

    public var cbv2LayerKinds: [CBv2LayerKind] { languageModel.cbv2LayerKinds }
    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec {
        languageModel.cbv2RecurrentStateSpec
    }
    public var cbv2Capabilities: CBv2ModelCapabilities { languageModel.cbv2Capabilities }
    public func newCacheV2(
        makeLayerCache: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
            any CBv2AttendingLayerCache
    ) rethrows -> [any CBv2AttendingLayerCache] {
        try languageModel.newCacheV2(makeLayerCache: makeLayerCache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            if Qwen4ExpWeightSanitizer.shouldDrop(key, mmapPLE: languageModel.mmapPLE) {
                continue
            }
            var key = key
            if key.hasPrefix("model.language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if !key.hasPrefix("language_model.") {
                key = "language_model." + key
            }
            sanitized[key] = value
        }
        if languageModel.configuration.numExperts > 0 {
            sanitized = qwen35FuseSwitchMLPGateUp(
                weights: sanitized,
                perLayerQuantization: checkpointPerLayerQuantization,
                setFused: { qwen35SetSwitchGLUGateUpFused($1, at: $0, in: self) })
        }
        return languageModel.sanitize(weights: sanitized)
    }
}

extension Qwen4ExpTextModel: QuantizationPathAliasing {
    public func quantizationPathAliases(for path: String) -> [String] {
        qwen35GateUpQuantizationAliases(for: path)
    }
}

extension Qwen4ExpTextModel: QuantizationPolicyReceiving {}

extension Qwen4ExpTextModel: CheckpointWeightLoadFiltering {
    public var checkpointWeightLoadFilter: CheckpointWeightLoadFilter {
        Qwen4ExpCheckpointLoad.filter(
            mmapPLE: mmapPLE, keepVisionTower: false)
    }

    public var skipWholeShardPrefetch: Bool { mmapPLE }
}

extension Qwen4ExpTextModel: Qwen4ExpExternalPLEReleasing {
    public func releaseExternalPLEResources() {
        for layer in model.layers {
            layer.ple?.releaseExternalResources()
        }
        pleDirectoryLease?.release()
    }
}

extension Qwen4ExpTextModel: Qwen4ExpExternalPLEValidating {
    public func validateExternalPLEResources() throws {
        for layer in model.layers {
            try layer.ple?.validateExternalResources()
        }
    }
}

extension Qwen4ExpModel: QuantizationPathAliasing {
    public func quantizationPathAliases(for path: String) -> [String] {
        qwen35GateUpQuantizationAliases(for: path)
    }
}

extension Qwen4ExpModel: QuantizationPolicyReceiving {
    public var checkpointPerLayerQuantization: BaseConfiguration.PerLayerQuantization? {
        get { languageModel.checkpointPerLayerQuantization }
        set { languageModel.checkpointPerLayerQuantization = newValue }
    }
}

extension Qwen4ExpModel: CheckpointWeightLoadFiltering {
    public var checkpointWeightLoadFilter: CheckpointWeightLoadFilter {
        Qwen4ExpCheckpointLoad.filter(
            mmapPLE: languageModel.mmapPLE, keepVisionTower: false)
    }

    public var skipWholeShardPrefetch: Bool { languageModel.mmapPLE }
}

extension Qwen4ExpModel: Qwen4ExpExternalPLEReleasing {
    public func releaseExternalPLEResources() {
        languageModel.releaseExternalPLEResources()
    }
}

extension Qwen4ExpModel: Qwen4ExpExternalPLEValidating {
    public func validateExternalPLEResources() throws {
        try languageModel.validateExternalPLEResources()
    }
}

public enum Qwen4ExpPLEResidencyError: Error, LocalizedError, Equatable {
    case conflictingActiveModel

    public var errorDescription: String? {
        switch self {
        case .conflictingActiveModel:
            return "A different Qwen4 PLE checkpoint is already resident"
        }
    }
}

public enum Qwen4ExpPLEResidency {
    public static let mmapFlag = "DARKBLOOM_QWEN4_PLE_SSD_OFFLOAD"
    public static let qwen4ExpModelTypes: Set<String> = [
        "qwen4_exp",
        "qwen4_exp_text",
    ]

    private static let directoryLock = NSLock()
    nonisolated(unsafe) private static var _modelDirectory: URL?
    nonisolated(unsafe) private static var _retainCount = 0

    /// A loaded model owns its binding independently of the temporary loader
    /// adoption. Its idempotent lease also closes on model deinitialization.
    public static func retainCurrentDirectory() -> Qwen4ExpPLEDirectoryLease? {
        directoryLock.lock()
        defer { directoryLock.unlock() }
        guard let directory = _modelDirectory, _retainCount < Int.max else { return nil }
        _retainCount += 1
        return Qwen4ExpPLEDirectoryLease(directory: directory)
    }

    public static var modelDirectory: URL? {
        get {
            directoryLock.lock()
            defer { directoryLock.unlock() }
            return _modelDirectory
        }
        set {
            directoryLock.lock()
            _modelDirectory = newValue.map { $0.standardizedFileURL }
            _retainCount = newValue == nil ? 0 : 1
            directoryLock.unlock()
        }
    }

    public static var retainCount: Int {
        directoryLock.lock()
        defer { directoryLock.unlock() }
        return _retainCount
    }

    public static var resolvedModelDirectory: URL? {
        modelDirectory
    }

    public static var useMmap: Bool {
        let raw = ProcessInfo.processInfo.environment[mmapFlag]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if raw == "0" || raw == "false" || raw == "no" || raw == "off" {
            return false
        }
        return true
    }

    public static func isQwen4ExpModelType(_ modelType: String?) -> Bool {
        guard let modelType else { return false }
        return qwen4ExpModelTypes.contains(modelType)
    }

    public static func configDeclaresQwen4Exp(at directory: URL) -> Bool {
        let url = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return false }
        return isQwen4ExpModelType(json["model_type"] as? String)
    }

    /// Bind the process-wide mmap root only for a Qwen4 checkpoint.
    /// A live different tree is left alone (one PLE root per process).
    /// Returns whether this directory is now retained.
    @discardableResult
    public static func adoptIfQwen4Exp(directory: URL) -> Bool {
        guard configDeclaresQwen4Exp(at: directory) else { return false }
        return adopt(directory: directory)
    }

    /// Provider load seam: non-Qwen4 models need no binding, while a Qwen4
    /// checkpoint must either retain its own tree or fail before loading.
    /// Continuing after `adopt` returns false would make its PLE layer resolve
    /// rows from the already-live checkpoint's process-wide root.
    @discardableResult
    public static func adoptForLoadIfQwen4Exp(directory: URL) throws -> Bool {
        guard configDeclaresQwen4Exp(at: directory) else { return false }
        guard adopt(directory: directory) else {
            throw Qwen4ExpPLEResidencyError.conflictingActiveModel
        }
        return true
    }

    @discardableResult
    public static func adopt(directory: URL) -> Bool {
        let url = directory.standardizedFileURL
        directoryLock.lock()
        defer { directoryLock.unlock() }
        if let current = _modelDirectory {
            if current == url {
                _retainCount += 1
                return true
            }
            if _retainCount > 0 {
                return false
            }
        }
        _modelDirectory = url
        _retainCount = 1
        return true
    }

    public static func release(directory: URL) {
        let url = directory.standardizedFileURL
        directoryLock.lock()
        defer { directoryLock.unlock() }
        guard _modelDirectory == url else { return }
        _retainCount = max(0, _retainCount - 1)
        if _retainCount == 0 {
            _modelDirectory = nil
        }
    }

    public static func reset() {
        directoryLock.lock()
        _modelDirectory = nil
        _retainCount = 0
        directoryLock.unlock()
    }
}

extension Qwen4ExpTextModel: CBv2PositionAxisProviding {
    public var cbv2PositionAxisCount: Int? { 3 }
}

extension Qwen4ExpTextModel: CBv2PositionedRecurrentLanguageModelForwardable,
    CBv2PositionedRecurrentEmbeddingForwardable
{
    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        positionedForward(
            tokens, inputEmbedding: nil, cache: caches,
            recurrentState: recurrentState, positionIds: nil)
    }

    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        positionedForward(
            tokens, inputEmbedding: nil, cache: caches,
            recurrentState: recurrentState, positionIds: positionIds)
    }

    public var supportsVisionSpanPrefill: Bool { false }
    public var supportsCausalVisionPrefill: Bool { true }

    public func scaledInputEmbeddings(_ inputs: MLXArray) -> MLXArray {
        model.embedTokens(inputs)
    }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        preconditionFailure("Qwen4 embedding prefill requires request-owned recurrent state")
    }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        positionedForward(
            inputs, inputEmbedding: inputEmbedding, cache: cache,
            recurrentState: recurrentState, positionIds: positionIds)
    }

    private func positionedForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray?, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        let attending = (cache ?? []).map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                preconditionFailure("Qwen4 CBv2 target received a legacy KV cache")
            }
            return attending
        }
        // Fusion `_decode_profile_context_tokens`: the QSA cache offset
        // before this step is the visible context length.
        let contextTokens = (attending.first as? KVCache)?.offset ?? 0
        typealias P = Qwen4ExpDecodeProfile
        P.beginDump(
            inputs: inputs, inputEmbeddings: inputEmbedding, contextTokens: contextTokens,
            kind: "serial")
        defer { P.endDumpIfPending() }
        if let sample = P.beginFine(
            inputs: inputs, inputEmbeddings: inputEmbedding, contextTokens: contextTokens)
        {
            let totalStart = P.now()
            let hidden = model.cbv2Forward(
                inputs, inputEmbeddings: inputEmbedding, caches: attending,
                recurrentState: recurrentState, positionIds: positionIds)
            sample.modelNs = P.now() &- totalStart
            let logits = headLogits(hidden)
            P.endFine(sample, logits: logits, totalStart: totalStart)
            P.endDump(logits: logits)
            return logits
        }
        if let coarse = P.beginCoarse(
            inputs: inputs, inputEmbeddings: inputEmbedding, contextTokens: contextTokens)
        {
            let buildStart = P.now()
            let hidden = model.cbv2Forward(
                inputs, inputEmbeddings: inputEmbedding, caches: attending,
                recurrentState: recurrentState, positionIds: positionIds)
            let logits = headLogits(hidden)
            P.endCoarse(coarse, logits: logits, buildStart: buildStart)
            P.endDump(logits: logits)
            return logits
        }
        let hidden = model.cbv2Forward(
            inputs, inputEmbeddings: inputEmbedding, caches: attending,
            recurrentState: recurrentState, positionIds: positionIds)
        let logits = headLogits(hidden)
        P.endDump(logits: logits)
        return logits
    }
}

extension Qwen4ExpTextModel: CBv2RecurrentLanguageModelPrefillForwardable {
    public var cbv2SupportsPackedPrefill: Bool { false }

    public func cbv2RecurrentPrefill(
        _ inputs: MLXArray, inputEmbedding: MLXArray?, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        let attending = (cache ?? []).map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                preconditionFailure("Qwen4 CBv2 target received a legacy KV cache")
            }
            return attending
        }
        let hidden = model.cbv2Forward(
            inputs, inputEmbeddings: inputEmbedding, caches: attending,
            recurrentState: recurrentState, positionIds: positionIds)
        switch requirement {
        case .evaluationOnly:
            return hidden[0..., -1, 0 ..< 1]
        case .lastPositionLogits:
            let last = hidden[0..., -1, 0...]
            return headLogits(last)
        }
    }
}

extension Qwen4ExpModel: CBv2PositionAxisProviding {
    public var cbv2PositionAxisCount: Int? { languageModel.cbv2PositionAxisCount }
}

extension Qwen4ExpModel: CBv2PositionedRecurrentLanguageModelForwardable,
    CBv2PositionedRecurrentEmbeddingForwardable
{
    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        languageModel.cbv2Forward(tokens, caches: caches, recurrentState: recurrentState)
    }

    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        languageModel.cbv2Forward(
            tokens, caches: caches, recurrentState: recurrentState, positionIds: positionIds)
    }

    public var supportsVisionSpanPrefill: Bool { languageModel.supportsVisionSpanPrefill }
    public var supportsCausalVisionPrefill: Bool { languageModel.supportsCausalVisionPrefill }

    public func scaledInputEmbeddings(_ inputs: MLXArray) -> MLXArray {
        languageModel.scaledInputEmbeddings(inputs)
    }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        languageModel.embeddingForward(inputs, inputEmbedding: inputEmbedding, cache: cache)
    }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        languageModel.embeddingForward(
            inputs, inputEmbedding: inputEmbedding, cache: cache,
            recurrentState: recurrentState, positionIds: positionIds)
    }
}

extension Qwen4ExpModel: CBv2RecurrentLanguageModelPrefillForwardable {
    public var cbv2SupportsPackedPrefill: Bool { languageModel.cbv2SupportsPackedPrefill }

    public func cbv2RecurrentPrefill(
        _ inputs: MLXArray, inputEmbedding: MLXArray?, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        languageModel.cbv2RecurrentPrefill(
            inputs, inputEmbedding: inputEmbedding, cache: cache,
            recurrentState: recurrentState, positionIds: positionIds,
            requirement: requirement)
    }
}

extension Qwen4ExpTextModel: CBv2RecurrentMTPForwardable {
    public func cbv2ForwardWithHidden(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        hiddenCapture(
            tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds, captureRecurrentWindow: false)
    }
}

extension Qwen4ExpTextModel: CBv2RecurrentCaptureMTPForwardable, CBv2RecurrentPrefillHiddenForwardable {
    /// Lightning MTP consumes the pre-mixer residual `[B, T, hc·H]`.
    /// Logits still come from the mixed `[B, T, H]` tower — Fusion
    /// `capture_layer_ids == []`.
    public func cbv2ForwardWithHiddenCaptured(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        hiddenCapture(
            tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds, captureRecurrentWindow: true)
    }

    /// Prefill chunk under Lightning MTP: the same arithmetic as
    /// `cbv2RecurrentPrefill` (residual → final mixer over every position →
    /// head on the LAST position), so the first token is bit-identical to
    /// serial decode's, plus the pre-mixer residual for every position for
    /// the assistant backlog. `hiddenCapture` scored every position with the
    /// 248,320-way head instead — ~6.8 s of the 80K TTFT (88.5 s vs 81.7 s
    /// with MTP off, 2026-09-05) for rows the caller discarded.
    public func cbv2ForwardWithHiddenForPrefill(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        let attending = caches.map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                preconditionFailure("Qwen4 CBv2 MTP target received a legacy KV cache")
            }
            return attending
        }
        let residual = model.cbv2ForwardResidual(
            tokens, inputEmbeddings: nil, caches: attending,
            recurrentState: recurrentState, positionIds: positionIds,
            captureRecurrentWindow: false)
        let mixed = Qwen4ExpDecodeProfile.stage("final_hc") {
            model.hyperConnectionMixer.mix(residual)
        }
        let last = mixed[0..., (mixed.dim(1) - 1)..., 0...]
        switch requirement {
        case .evaluationOnly:
            return (last[0..., 0..., 0 ..< 1], residual)
        case .lastPositionLogits:
            return (headLogits(last), residual)
        }
    }

    fileprivate func hiddenCapture(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        captureRecurrentWindow: Bool
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        let attending = caches.map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                preconditionFailure("Qwen4 CBv2 MTP target received a legacy KV cache")
            }
            return attending
        }
        // Same diagnostic hooks as `positionedForward`, so the Lightning
        // verify width (1+k) can be attributed per stage when the operator
        // raises DARKBLOOM_QWEN4_DECODE_PROFILE_MAX_WIDTH. No-ops otherwise.
        let contextTokens = (attending.first as? KVCache)?.offset ?? 0
        typealias P = Qwen4ExpDecodeProfile
        P.beginDump(
            inputs: tokens, inputEmbeddings: nil, contextTokens: contextTokens,
            kind: captureRecurrentWindow ? "verify" : "seed")
        defer { P.endDumpIfPending() }
        if let sample = P.beginFine(
            inputs: tokens, inputEmbeddings: nil, contextTokens: contextTokens)
        {
            let totalStart = P.now()
            let residual = model.cbv2ForwardResidual(
                tokens, inputEmbeddings: nil, caches: attending,
                recurrentState: recurrentState, positionIds: positionIds,
                captureRecurrentWindow: captureRecurrentWindow)
            let mixed = P.stage("final_hc") { model.hyperConnectionMixer.mix(residual) }
            sample.modelNs = P.now() &- totalStart
            let logits = headLogits(mixed)
            P.endFine(sample, logits: logits, totalStart: totalStart)
            P.endDump(logits: logits)
            return (logits, residual)
        }
        if let coarse = P.beginCoarse(
            inputs: tokens, inputEmbeddings: nil, contextTokens: contextTokens)
        {
            let buildStart = P.now()
            let residual = model.cbv2ForwardResidual(
                tokens, inputEmbeddings: nil, caches: attending,
                recurrentState: recurrentState, positionIds: positionIds,
                captureRecurrentWindow: captureRecurrentWindow)
            let mixed = model.hyperConnectionMixer.mix(residual)
            let logits = headLogits(mixed)
            P.endCoarse(coarse, logits: logits, buildStart: buildStart)
            P.endDump(logits: logits)
            return (logits, residual)
        }
        let residual = model.cbv2ForwardResidual(
            tokens, inputEmbeddings: nil, caches: attending,
            recurrentState: recurrentState, positionIds: positionIds,
            captureRecurrentWindow: captureRecurrentWindow)
        let mixed = model.hyperConnectionMixer.mix(residual)
        let logits = headLogits(mixed)
        P.endDump(logits: logits)
        return (logits, residual)
    }
}

extension Qwen4ExpTextModel: CBv2MTPPolicyTopTwoProviding {
    public func cbv2MTPTopTwo(
        _ logits: MLXArray
    ) -> (ids: MLXArray, values: MLXArray) {
        precondition(logits.ndim == 3, "Qwen4 MTP policy logits must be [B,L,V]")
        let batch = logits.dim(0)
        let length = logits.dim(1)
        let vocabularySize = logits.dim(2)
        precondition(batch > 0 && length > 0 && vocabularySize >= 2)
        let rows = batch * length
        let topTwo = qwen35MTPTopTwoRows(
            logits.reshaped([1, rows, vocabularySize]))
        return (
            topTwo.ids.reshaped([batch, length, 2]),
            topTwo.values.reshaped([batch, length, 2]))
    }
}

extension Qwen4ExpModel: CBv2RecurrentMTPForwardable {
    public var cbv2MTPTargetIdentity: ObjectIdentifier {
        ObjectIdentifier(languageModel)
    }

    public func cbv2ForwardWithHidden(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        languageModel.cbv2ForwardWithHidden(
            tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds)
    }
}

extension Qwen4ExpModel: CBv2RecurrentCaptureMTPForwardable, CBv2RecurrentPrefillHiddenForwardable {
    public func cbv2ForwardWithHiddenCaptured(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        languageModel.cbv2ForwardWithHiddenCaptured(
            tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds)
    }

    public func cbv2ForwardWithHiddenForPrefill(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        languageModel.cbv2ForwardWithHiddenForPrefill(
            tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds, requirement: requirement)
    }
}

extension Qwen4ExpModel: CBv2MTPPolicyTopTwoProviding {
    public func cbv2MTPTopTwo(
        _ logits: MLXArray
    ) -> (ids: MLXArray, values: MLXArray) {
        languageModel.cbv2MTPTopTwo(logits)
    }
}
