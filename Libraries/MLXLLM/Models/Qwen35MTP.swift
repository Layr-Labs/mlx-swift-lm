// Copyright © 2026 Eigen Labs.
//
// Port of omlx commit 696d90a:
//   patches/mlx_lm_mtp/qwen35_model.py  (MTPDecoderLayer, MTPModule)
//   patches/mlx_lm_mtp/__init__.py        (is_mtp_active / set_mtp_active)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Module-level MTP flag

/// Controls whether Qwen3.5/3.6 model inits attach the MTP head.
/// Set to `true` before calling `MLXLLM.load(...)` when MTP should be active.
/// Mirrors omlx `is_mtp_active()` / `set_mtp_active()` from
/// patches/mlx_lm_mtp/__init__.py.
public nonisolated(unsafe) var _qwen35MTPEnabled: Bool = false

/// Fail-loud errors for the production-safe inline Qwen MTP loader.
///
/// Unlike the legacy process-global attachment flag above, this loader builds
/// an assistant explicitly for one already-loaded target and never changes
/// model construction behavior process-wide.
public enum Qwen35InlineMTPError: Error, LocalizedError, Sendable, Equatable {
    case invalidConfiguration(String)
    case incompatibleTarget(field: String, artifact: Int, target: Int)
    case invalidWeightIndex(String)
    case missingWeights
    case duplicateWeight(String)
    case missingQuantization(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail):
            return "Invalid inline Qwen MTP configuration: \(detail)."
        case .incompatibleTarget(let field, let artifact, let target):
            return "Inline Qwen MTP target mismatch at \(field): artifact=\(artifact), target=\(target)."
        case .invalidWeightIndex(let detail):
            return "Invalid inline Qwen MTP weight index: \(detail)."
        case .missingWeights:
            return "The checkpoint declares inline Qwen MTP but contains no matching tensors."
        case .duplicateWeight(let key):
            return "The inline Qwen MTP tensor \(key) appears more than once."
        case .missingQuantization(let path):
            return "The quantized inline Qwen MTP module \(path) has no matching quantization entry."
        }
    }
}

/// Parsed, bounded metadata for an inline Qwen MTP assistant.
/// Internal so focused tests can validate checkpoint interpretation without
/// constructing hundreds of MiB of weights.
struct Qwen35InlineMTPMetadata: Sendable {
    let textConfiguration: Qwen35TextConfiguration
    let prefix: String
    let blockSize: Int
    let defaultQuantization: BaseConfiguration.Quantization?
    let quantizationByPath: [String: BaseConfiguration.Quantization]
}

// MARK: - MTPDecoderLayer

/// Full-attention transformer layer used inside the Qwen3.5/3.6 MTP head.
/// Unlike `Qwen35DecoderLayer`, this always uses full attention (never SSM/linear).
/// MoE config is honoured when `num_experts > 0`.
/// omlx: patches/mlx_lm_mtp/qwen35_model.py MTPDecoderLayer
final class Qwen35MTPDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen35TextConfiguration) {
        _selfAttn.wrappedValue = Qwen35Attention(args)
        if args.numExperts > 0 {
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> MLXArray {
        // omlx: MTPDecoderLayer.__call__
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }
}

// MARK: - MTPModule

/// Multi-Token Prediction head for Qwen3.5/3.6.
///
/// Fuses the backbone's pre-norm hidden state at position t with the embedding of
/// the sampled main token (t+1) to predict the draft token at (t+2).
///
/// Architecture (port of PR #990):
/// ```
/// pre_fc_norm_hidden:    RMSNorm(hidden_size)
/// pre_fc_norm_embedding: RMSNorm(hidden_size)
/// fc:                    Linear(hidden_size * 2 → hidden_size, bias: false)
/// layers:                [MTPDecoderLayer]  × mtp_num_hidden_layers
/// norm:                  RMSNorm(hidden_size)
/// ```
/// omlx: patches/mlx_lm_mtp/qwen35_model.py MTPModule
final class Qwen35MTPModule: Module {
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFcNormHidden: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFcNormEmbedding: RMSNorm
    @ModuleInfo(key: "fc") var fc: Linear
    // `layers` uses the default ModuleInfo key derived from the property name.
    let layers: [Qwen35MTPDecoderLayer]
    let norm: RMSNorm

    init(_ args: Qwen35TextConfiguration) {
        _preFcNormHidden.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFcNormEmbedding.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _fc.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
        self.layers = (0 ..< args.mtpNumHiddenLayers).map { _ in
            Qwen35MTPDecoderLayer(args)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray {
        // omlx: MTPModule.__call__
        // 1. Embed next-token ids and fuse with normed hidden state.
        let embeds = embedTokens(nextTokenIds)
        let e = preFcNormEmbedding(embeds)
        let h = preFcNormHidden(hidden)
        var fused = fc(concatenated([e, h], axis: -1))

        // 2. Compute attention mask from the first cache entry (or nil if empty).
        let firstCache: (any KVCache)? = cache.first
        let mask = createAttentionMask(h: fused, cache: firstCache)

        // 3. Run each MTPDecoderLayer.
        for (i, layer) in layers.enumerated() {
            let c: (any KVCache)? = i < cache.count ? cache[i] : nil
            fused = layer(fused, mask: mask, cache: c)
        }

        // 4. Return pre-lm_head hidden (norm applied; lm_head is in TextModel).
        return norm(fused)
    }
}

// MARK: - Artifact-scoped inline assistant

/// An explicitly loaded Qwen3.5/3.6 MTP assistant bound to one target.
///
/// The assistant owns only the tensors below `mtp.*`. Target embeddings and
/// the LM head are called through the bound target, so loading this object does
/// not duplicate the target checkpoint and does not require
/// `_qwen35MTPEnabled`.
public final class Qwen35InlineMTPAssistant: Module, @unchecked Sendable {
    static let cacheAllocationStep = 256

    private let mtp: Qwen35MTPModule
    private let target: Qwen35TextModel

    public let blockSize: Int
    public var targetIdentity: ObjectIdentifier { ObjectIdentifier(target) }

    private init(
        configuration: Qwen35TextConfiguration,
        blockSize: Int,
        target: Qwen35TextModel
    ) {
        self.mtp = Qwen35MTPModule(configuration)
        self.blockSize = blockSize
        self.target = target
        super.init()
    }

    /// Allocate the assistant's own autoregressive full-attention KV.
    /// Target recurrent and attention state is never reused as assistant KV.
    public func makeCache() -> [any KVCache] {
        mtp.layers.map { _ in
            let cache = KVCacheSimple()
            cache.step = Self.cacheAllocationStep
            return cache as any KVCache
        }
    }

    /// Advance the assistant by one or more already-chosen target/draft tokens.
    /// Returns logits from the target's shared output projection and the
    /// assistant hidden used to continue a draft chain.
    public func forward(
        hidden: MLXArray,
        tokens: MLXArray,
        cache: [any KVCache]
    ) -> (logits: MLXArray, hidden: MLXArray) {
        let output = mtp(
            hidden: hidden,
            nextTokenIds: tokens,
            embedTokens: target.model.embedTokens,
            cache: cache)
        let logits: MLXArray
        if target.configuration.tieWordEmbeddings {
            logits = target.model.embedTokens.asLinear(output)
        } else {
            logits = target.lmHead!(output)
        }
        return (logits, output)
    }

    /// Strictly load the inline assistant declared by a combined checkpoint.
    /// Only indexed keys below the declared prefix are read and retained.
    public static func load(
        from modelDirectory: URL,
        target: any LanguageModel
    ) throws -> Qwen35InlineMTPAssistant {
        let target = try qwen35TextTarget(target)
        let metadata = try loadMetadata(from: modelDirectory)
        try validate(metadata.textConfiguration, against: target.configuration)

        let indexed = try loadIndexedWeights(
            from: modelDirectory, prefix: metadata.prefix)
        let assistant = Qwen35InlineMTPAssistant(
            configuration: metadata.textConfiguration,
            blockSize: metadata.blockSize,
            target: target)

        let scaledPaths = Set(indexed.keys.compactMap { key -> String? in
            guard key.hasSuffix(".scales") else { return nil }
            return String(key.dropLast(".scales".count))
        })
        for path in scaledPaths
        where metadata.quantizationByPath[path] == nil && metadata.defaultQuantization == nil {
            throw Qwen35InlineMTPError.missingQuantization(path)
        }
        if !scaledPaths.isEmpty {
            quantize(model: assistant.mtp) { path, _ in
                guard scaledPaths.contains(path) else { return nil }
                return (metadata.quantizationByPath[path] ?? metadata.defaultQuantization)?.asTuple
            }
        }

        try assistant.mtp.update(
            parameters: ModuleParameters.unflattened(indexed), verify: [.all])
        eval(assistant.mtp)
        return assistant
    }

    private static func qwen35TextTarget(
        _ target: any LanguageModel
    ) throws -> Qwen35TextModel {
        if let target = target as? Qwen35TextModel { return target }
        if let target = target as? Qwen35Model { return target.languageModel }
        throw Qwen35InlineMTPError.invalidConfiguration(
            "target type \(String(describing: type(of: target))) is not Qwen3.5/3.6")
    }

    static func loadMetadata(from directory: URL) throws -> Qwen35InlineMTPMetadata {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let inline = root["mtplx_mtp"] as? [String: Any],
            inline["included"] as? Bool == true,
            let text = root["text_config"] as? [String: Any]
        else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "mtplx_mtp.included and text_config are required")
        }
        let prefix = (inline["prefix"] as? String) ?? "mtp."
        guard prefix == "mtp." else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "only the mtp. prefix is supported")
        }
        let blockSize = (inline["block_size"] as? NSNumber)?.intValue ?? 3
        guard (2...8).contains(blockSize) else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "block_size \(blockSize) is outside 2...8")
        }
        let textData = try JSONSerialization.data(withJSONObject: text)
        let configuration = try JSONDecoder.json5().decode(
            Qwen35TextConfiguration.self, from: textData)
        guard configuration.mtpNumHiddenLayers > 0,
            configuration.mtpNumHiddenLayers <= 4
        else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "mtp_num_hidden_layers must be within 1...4")
        }

        guard let rawQuantization = root["mtplx_mtp_quantization"] as? [String: Any]
        else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "mtplx_mtp_quantization is required")
        }
        var quantizationByPath: [String: BaseConfiguration.Quantization] = [:]
        let defaultKeys = ["group_size", "bits", "mode"]
        let defaultObject = Dictionary(
            uniqueKeysWithValues: defaultKeys.compactMap { key in
                rawQuantization[key].map { (key, $0) }
            })
        let defaultQuantization: BaseConfiguration.Quantization?
        if defaultObject.isEmpty {
            defaultQuantization = nil
        } else {
            let data = try JSONSerialization.data(withJSONObject: defaultObject)
            defaultQuantization = try JSONDecoder().decode(
                BaseConfiguration.Quantization.self, from: data)
        }
        for (path, raw) in rawQuantization {
            guard path.contains(".") else { continue }
            guard let object = raw as? [String: Any] else {
                throw Qwen35InlineMTPError.invalidConfiguration(
                    "quantization entry \(path) is not an object")
            }
            let quantizationData = try JSONSerialization.data(withJSONObject: object)
            quantizationByPath[path] = try JSONDecoder().decode(
                BaseConfiguration.Quantization.self, from: quantizationData)
        }
        return Qwen35InlineMTPMetadata(
            textConfiguration: configuration,
            prefix: prefix,
            blockSize: blockSize,
            defaultQuantization: defaultQuantization,
            quantizationByPath: quantizationByPath)
    }

    private struct WeightIndex: Decodable {
        let weightMap: [String: String]
        enum CodingKeys: String, CodingKey { case weightMap = "weight_map" }
    }

    private static func loadIndexedWeights(
        from directory: URL,
        prefix: String
    ) throws -> [String: MLXArray] {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        let index: WeightIndex
        do {
            index = try JSONDecoder().decode(WeightIndex.self, from: Data(contentsOf: indexURL))
        } catch {
            throw Qwen35InlineMTPError.invalidWeightIndex(String(describing: error))
        }

        var byFile: [String: [(source: String, destination: String)]] = [:]
        for (key, file) in index.weightMap where key.hasPrefix(prefix) {
            guard file == URL(fileURLWithPath: file).lastPathComponent,
                file.hasSuffix(".safetensors")
            else {
                throw Qwen35InlineMTPError.invalidWeightIndex(
                    "unsafe shard path for \(key)")
            }
            let destination = String(key.dropFirst(prefix.count))
            guard !destination.isEmpty else {
                throw Qwen35InlineMTPError.invalidWeightIndex("empty stripped key")
            }
            byFile[file, default: []].append((key, destination))
        }
        guard !byFile.isEmpty else { throw Qwen35InlineMTPError.missingWeights }

        var weights: [String: MLXArray] = [:]
        for file in byFile.keys.sorted() {
            let url = directory.appendingPathComponent(file)
            let (shard, _) = try loadArraysAndMetadata(url: url)
            for entry in byFile[file]! {
                guard let value = shard[entry.source] else {
                    throw Qwen35InlineMTPError.invalidWeightIndex(
                        "indexed tensor \(entry.source) is absent from \(file)")
                }
                guard weights.updateValue(value, forKey: entry.destination) == nil else {
                    throw Qwen35InlineMTPError.duplicateWeight(entry.destination)
                }
            }
        }
        return weights
    }

    private static func validate(
        _ artifact: Qwen35TextConfiguration,
        against target: Qwen35TextConfiguration
    ) throws {
        let fields: [(String, Int, Int)] = [
            ("hidden_size", artifact.hiddenSize, target.hiddenSize),
            ("vocab_size", artifact.vocabularySize, target.vocabularySize),
            ("num_attention_heads", artifact.attentionHeads, target.attentionHeads),
            ("num_key_value_heads", artifact.kvHeads, target.kvHeads),
            ("head_dim", artifact.headDim ?? 0, target.headDim ?? 0),
            ("num_experts", artifact.numExperts, target.numExperts),
            ("num_experts_per_tok", artifact.numExpertsPerTok, target.numExpertsPerTok),
        ]
        for (field, artifactValue, targetValue) in fields
        where artifactValue != targetValue {
            throw Qwen35InlineMTPError.incompatibleTarget(
                field: field, artifact: artifactValue, target: targetValue)
        }
    }
}

extension Qwen35InlineMTPAssistant: CBv2MTPRequestStatefulDrafter {
    private final class RequestState: CBv2MTPRequestState {
        var caches: [any KVCache]
        var stagedInputs = 0
        var committedInputCount: Int { (caches.first?.offset ?? 0) - stagedInputs }
        var stagedInputCount: Int { stagedInputs }
        var materializedBytes: Int {
            caches.flatMap { $0.innerState() }.reduce(0) { total, array in
                let (next, overflow) = total.addingReportingOverflow(array.nbytes)
                return overflow ? Int.max : next
            }
        }

        init(caches: [any KVCache]) { self.caches = caches }
    }

    private final class UnusedPreparedCapture: CBv2MTPPreparedCapture {}

    public var mtpTargetIdentity: ObjectIdentifier? { targetIdentity }
    /// Verification policy. Rectangular selects CAPTURE-VERIFY: the target
    /// scores all 1+k columns in ONE `[B, 1+k]` forward while the 30
    /// GatedDeltaNet layers stage per-position captured conv/SSM states
    /// (`Qwen35GatedDeltaNet.cbv2ForwardCaptured`); finalize commits the
    /// accepted position by device-side slice and rolls back by restoring
    /// the pre-verify snapshot. This replaces the serial per-column loop
    /// whose k+1 full-model re-reads made the round ≤1.0x by construction.
    ///
    /// NUMERICS POLICY: bitwise greedy parity with serial decode is NOT the
    /// bar here — batched verify changes accumulation geometry exactly like
    /// every other CBv2 batch-shape change (chunked prefill, B>1 decode).
    /// Distribution-exactness is the invariant: committed tokens are always
    /// target-authoritative (argmax when greedy, genuine target samples
    /// under target-prefix acceptance). The serial oracle remains available
    /// via `DARKBLOOM_QWEN_MTP_SERIAL=1` for A/B and certification runs.
    public var requiredVerificationMode: CBv2MTPVerificationMode? {
        Self.forceSerialVerification ? .serialTarget : .rectangular
    }
    /// k=1 today: `draftStep` stages exactly [seed, draft]. Deeper chains
    /// (k=2 self-application) extend the staging loop, not the verify path —
    /// the capture window and finalize already handle any 1+k.
    public var maximumDraftTokens: Int? { 1 }
    public var maximumSpeculativeBatch: Int? { 1 }
    /// Target-prefix acceptance is exact at any temperature (every committed
    /// token IS a target sample), so this drafter lifts the greedy gate when
    /// the engine sampler supports verify pre-sampling.
    public var supportsTargetPrefixAcceptance: Bool { true }

    /// `DARKBLOOM_QWEN_MTP_SERIAL=1/true/yes/on` pins the serial per-column
    /// verification oracle (the pre-capture-verify behavior).
    static let forceSerialVerification: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN_MTP_SERIAL"]
        else { return false }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()
    public var requestStateBytesPerToken: Int {
        Self.cacheBytesPerToken(
            configuration: target.configuration,
            layerCount: mtp.layers.count,
            elementBytes: mtp.norm.weight.dtype.size)
    }
    public var requestStateTokenGranularity: Int { Self.cacheAllocationStep }
    public var requestStateTokenAllocationPadding: Int { 1 }

    static func cacheBytesPerToken(
        configuration: Qwen35TextConfiguration,
        layerCount: Int,
        elementBytes: Int
    ) -> Int {
        let (headsByDimension, geometryOverflow) = configuration.kvHeads
            .multipliedReportingOverflow(by: configuration.headDim ?? 0)
        let (kvElements, kvOverflow) = headsByDimension.multipliedReportingOverflow(by: 2)
        let (layerElements, layerOverflow) = kvElements.multipliedReportingOverflow(
            by: layerCount)
        let (bytes, byteOverflow) = layerElements.multipliedReportingOverflow(
            by: elementBytes)
        return geometryOverflow || kvOverflow || layerOverflow || byteOverflow ? Int.max : bytes
    }

    public func makeRequestState() -> any CBv2MTPRequestState {
        RequestState(caches: makeCache())
    }

    public func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        UnusedPreparedCapture()
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure("inline Qwen MTP requires request-owned assistant state")
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        guard let state = requestState as? RequestState else {
            preconditionFailure("inline Qwen MTP received foreign request state")
        }
        precondition(state.stagedInputs == 0, "inline Qwen MTP round already staged")
        precondition(tokens.dim(0) == 1 && tokens.dim(1) == 1)
        let output = forward(hidden: hidden, tokens: tokens, cache: state.caches)
        let draft = argMax(output.logits[0..., -1, 0...], axis: -1).asType(.int32)
        // KVCacheSimple owns mutable buffers. Complete the proposal step
        // before constructing the history-only draft feed so the proposal
        // cannot observe the later cache version.
        eval([draft, output.hidden] + state.caches.flatMap { $0.innerState() })
        // The next round starts after target verification. Stage the proposed
        // draft now as well so acceptance retains [seed, draft], while a
        // rejection trims only the draft and retains the canonical seed.
        _ = forward(
            hidden: output.hidden, tokens: draft.reshaped([1, 1]), cache: state.caches)
        state.stagedInputs = 2
        return (
            draft,
            output.hidden)
    }

    public func evaluationTargets(
        for requestState: any CBv2MTPRequestState
    ) -> [MLXArray] {
        guard let state = requestState as? RequestState else { return [] }
        return state.caches.flatMap { $0.innerState() }
    }

    public func finalizeRound(
        requestState: any CBv2MTPRequestState, confirmedInputTokens: Int
    ) {
        guard let state = requestState as? RequestState else {
            preconditionFailure("inline Qwen MTP received foreign request state")
        }
        precondition((0 ... state.stagedInputs).contains(confirmedInputTokens))
        let rollback = state.stagedInputs - confirmedInputTokens
        if rollback > 0 {
            for cache in state.caches {
                precondition(cache.trim(rollback) == rollback)
            }
        }
        state.stagedInputs = 0
    }

    public func discardRound(requestState: any CBv2MTPRequestState) {
        guard let state = requestState as? RequestState else { return }
        if state.stagedInputs > 0 {
            for cache in state.caches {
                precondition(cache.trim(state.stagedInputs) == state.stagedInputs)
            }
        }
        state.stagedInputs = 0
    }

    public func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        guard let state = requestState as? RequestState else { return }
        state.caches.removeAll(keepingCapacity: false)
        state.stagedInputs = 0
    }
}
