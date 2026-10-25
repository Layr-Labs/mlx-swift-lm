// Copyright © 2026 Eigen Labs.
//
// Port of omlx commit 696d90a:
//   patches/mlx_lm_mtp/qwen35_model.py  (MTPDecoderLayer, MTPModule)
//   patches/mlx_lm_mtp/__init__.py        (is_mtp_active / set_mtp_active)

import CoreFoundation
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
            return
                "Inline Qwen MTP target mismatch at \(field): artifact=\(artifact), target=\(target)."
        case .invalidWeightIndex(let detail):
            return "Invalid inline Qwen MTP weight index: \(detail)."
        case .missingWeights:
            return "The checkpoint declares inline Qwen MTP but contains no matching tensors."
        case .duplicateWeight(let key):
            return "The inline Qwen MTP tensor \(key) appears more than once."
        case .missingQuantization(let path):
            return
                "The quantized inline Qwen MTP module \(path) has no matching quantization entry."
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
    let quantization: BaseConfiguration.PerLayerQuantization?

    func resolvedQuantization(
        for path: String
    ) -> BaseConfiguration.Quantization? {
        quantization?.quantization(layer: path)
    }
}

// MARK: - MTP history KV

extension Qwen35Attention {
    /// Append committed proposal-head history without computing unused query,
    /// gate, attention, or output-projection rows.
    func appendMTPHistoryKV(_ x: MLXArray, cache: any KVCache) {
        let batch = x.dim(0)
        let length = x.dim(1)
        var keys = kNorm(kProj(x).reshaped(batch, length, kvHeads, -1))
            .transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(batch, length, kvHeads, -1)
            .transposed(0, 2, 1, 3)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)
        _ = cache.update(keys: keys, values: values)
    }
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
            // Split gate/up: the assistant's quantization table
            // (`mtplx_mtp_quantization`) and the checkpoint's mtp.* tensors
            // are keyed on the split module paths, and the MTP head loads
            // outside the target sanitizers that perform gate/up fusion.
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args, fuseGateUp: false)
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

    /// Populate this layer's K/V history without computing a dead decoder
    /// output. Only valid when no later MTP layer consumes that output.
    func appendHistoryKV(_ x: MLXArray, cache: any KVCache) {
        selfAttn.appendMTPHistoryKV(inputLayerNorm(x), cache: cache)
    }
}

// MARK: - MTPModule

/// Multi-Token Prediction head for Qwen3.5/3.6.
///
/// Fuses the backbone's final-normalized hidden state at position t with the
/// embedding of the sampled main token (t+1) to predict the draft token at (t+2).
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

    /// Append every leading committed row through a K/V-only path and compute
    /// a full decoder output only for the final proposal row. Multi-layer heads
    /// fail closed before any cache mutation because later layers require the
    /// omitted leading outputs.
    func lastHiddenWithKVOnlyHistory(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray? {
        guard layers.count == 1, cache.count == 1,
            hidden.dim(1) > 1,
            nextTokenIds.dim(1) == hidden.dim(1)
        else { return nil }

        let embeds = embedTokens(nextTokenIds)
        let e = preFcNormEmbedding(embeds)
        let h = preFcNormHidden(hidden)
        let fused = fc(concatenated([e, h], axis: -1))
        let historyCount = fused.dim(1) - 1
        layers[0].appendHistoryKV(
            fused[0..., 0 ..< historyCount, 0...], cache: cache[0])

        let current = fused[0..., historyCount..., 0...]
        let mask = createAttentionMask(h: current, cache: cache[0])
        return norm(layers[0](current, mask: mask, cache: cache[0]))
    }

    /// Append committed rows through the same embedding, pre-FC norms, FC and
    /// K/V-only path, producing no decoder output. False for multi-layer heads,
    /// before any cache mutation.
    func appendHistoryKV(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> Bool {
        guard layers.count == 1, cache.count == 1,
            hidden.dim(1) > 0,
            nextTokenIds.dim(1) == hidden.dim(1)
        else { return false }

        let e = preFcNormEmbedding(embedTokens(nextTokenIds))
        let h = preFcNormHidden(hidden)
        layers[0].appendHistoryKV(fc(concatenated([e, h], axis: -1)), cache: cache[0])
        return true
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
    private let installedVerificationMode: CBv2MTPVerificationMode
    private let allowsShortlist: Bool
    private let maximumDraftDepth: Int
    private let minimumStateElementBytes: Int

    public let blockSize: Int
    public var targetIdentity: ObjectIdentifier { ObjectIdentifier(target) }

    /// Opt-in bound on trusted-history flushes. When set, leading trusted rows
    /// enter the head K/V in blocks of at most this many rows, each evaluated
    /// before the next is built, so no flush holds a whole-history rectangle.
    /// nil keeps the single-rectangle flush. Only draft accumulation geometry
    /// changes; committed tokens stay target-authoritative.
    public var historyFlushRows: Int? {
        didSet {
            precondition(
                historyFlushRows.map { $0 > 0 } ?? true,
                "inline Qwen MTP history flush rows must be positive")
        }
    }

    var prefixCheckpointGeometry: (width: Int, dtype: DType, vocabulary: Int, verification: String)
    {
        (
            target.configuration.hiddenSize, target.model.norm.weight.dtype,
            target.vocabularySize, installedVerificationMode.rawValue
        )
    }

    private init(
        configuration: Qwen35TextConfiguration,
        blockSize: Int,
        target: Qwen35TextModel,
        verificationMode: CBv2MTPVerificationMode?,
        allowsShortlist: Bool = true,
        maximumDraftDepth: Int = 4,
        minimumStateElementBytes: Int = 0
    ) {
        self.mtp = Qwen35MTPModule(configuration)
        self.blockSize = blockSize
        self.target = target
        self.allowsShortlist = allowsShortlist
        self.maximumDraftDepth = maximumDraftDepth
        self.minimumStateElementBytes = minimumStateElementBytes
        self.installedVerificationMode = Self.resolvedVerificationMode(
            requested: verificationMode,
            forceSerialEnvironment: Self.forceSerialVerification)
        super.init()
    }

    static func resolvedVerificationMode(
        requested: CBv2MTPVerificationMode?, forceSerialEnvironment: Bool
    ) -> CBv2MTPVerificationMode {
        if forceSerialEnvironment { return .serialTarget }
        return requested ?? .rectangular
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
        return (headLogits(output), output)
    }

    /// The target's shared output projection over assistant hidden states.
    func headLogits(_ output: MLXArray) -> MLXArray {
        if target.configuration.tieWordEmbeddings {
            return target.model.embedTokens.asLinear(output)
        }
        return target.lmHead!(output)
    }

    /// Apply the exact final norm of the bound target once at the boundary
    /// where target-derived hidden enters assistant history. Target hidden
    /// capture is deliberately pre-norm; recursive assistant hidden is already
    /// post-`mtp.norm` and must never pass through this helper.
    func targetFinalNorm(_ hidden: MLXArray) -> MLXArray {
        target.model.norm(hidden)
    }

    /// Assistant hidden states WITHOUT the output projection — the draft
    /// path applies the head to the last position only (or to a shortlist).
    func moduleForward(
        hidden: MLXArray,
        tokens: MLXArray,
        cache: [any KVCache]
    ) -> MLXArray {
        mtp(
            hidden: hidden,
            nextTokenIds: tokens,
            embedTokens: target.model.embedTokens,
            cache: cache)
    }

    /// History-flush specialization. Returns one final hidden row while every
    /// leading trusted row contributes only K/V state. nil means the head
    /// geometry cannot safely omit intermediate layer outputs.
    func moduleLastHiddenWithKVOnlyHistory(
        hidden: MLXArray,
        tokens: MLXArray,
        cache: [any KVCache]
    ) -> MLXArray? {
        mtp.lastHiddenWithKVOnlyHistory(
            hidden: hidden,
            nextTokenIds: tokens,
            embedTokens: target.model.embedTokens,
            cache: cache)
    }

    /// Draft logits over an engine-provided shortlist of token ids. The
    /// draft needs only an argmax, so score the gathered head rows instead
    /// of streaming the full `[V, H]` output projection (~K/V of the bytes;
    /// the 4-bit Qwen3.6 lm_head alone is ~286 MB per read). Quantized
    /// heads gather packed rows plus their scales/biases and stay on the
    /// quantized-matmul path; float heads (small test fixtures) gather
    /// plain rows.
    func shortlistLogits(hidden: MLXArray, ids: MLXArray) -> MLXArray {
        // Custom projections may transform inputs before packed matmul. Calling
        // the complete module preserves that contract even for direct callers.
        if !allowsShortlist { return headLogits(hidden)[.ellipsis, ids] }
        if target.configuration.tieWordEmbeddings {
            let embed = target.model.embedTokens
            if let quantized = embed as? QuantizedEmbedding {
                return quantizedMM(
                    hidden, quantized.weight[ids],
                    scales: quantized.scales[ids],
                    biases: quantized.biases.map { $0[ids] },
                    transpose: true,
                    groupSize: quantized.groupSize, bits: quantized.bits,
                    mode: quantized.mode)
            }
            return matmul(hidden, embed.weight[ids].transposed(1, 0))
        }
        let head = target.lmHead!
        if let quantized = head as? QuantizedLinear {
            var logits = quantizedMM(
                hidden, quantized.weight[ids],
                scales: quantized.scales[ids],
                biases: quantized.biases.map { $0[ids] },
                transpose: true,
                groupSize: quantized.groupSize, bits: quantized.bits,
                mode: quantized.mode)
            if let bias = quantized.bias { logits = logits + bias[ids] }
            return logits
        }
        var logits = matmul(hidden, head.weight[ids].transposed(1, 0))
        if let bias = head.bias { logits = logits + bias[ids] }
        return logits
    }

    /// Strictly load either an inline assistant declared by a combined
    /// checkpoint or a standalone `qwen3_5_mtp` artifact. Inline artifacts
    /// read only indexed keys below their declared prefix; standalone
    /// artifacts read every tensor from their bounded safetensors directory.
    public static func load(
        from modelDirectory: URL,
        target: any LanguageModel,
        verificationMode: CBv2MTPVerificationMode? = nil
    ) throws -> Qwen35InlineMTPAssistant {
        let target = try qwen35TextTarget(target)
        let resolvedVerificationMode = Self.resolvedVerificationMode(
            requested: verificationMode,
            forceSerialEnvironment: Self.forceSerialVerification)
        if resolvedVerificationMode == .rectangularExact && !target.model.exactTargetVerify {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "rectangular_exact verification requires exact target arithmetic")
        }
        let metadata = try loadMetadata(from: modelDirectory)
        try validate(metadata.textConfiguration, against: target.configuration)

        let indexed: [String: MLXArray]
        if metadata.prefix.isEmpty {
            indexed = try loadStandaloneWeights(from: modelDirectory)
        } else {
            indexed = try loadIndexedWeights(
                from: modelDirectory, prefix: metadata.prefix)
        }
        return try construct(
            metadata: metadata, indexed: indexed, target: target,
            verificationMode: verificationMode, allowsShortlist: true)
    }

    /// Construct from caller-verified metadata and prefix-stripped tensors.
    /// No paths are reopened. The caller owns source provenance, read/LOAD
    /// limits and checked native cleanup. A custom collective output head must
    /// disable shortlist projection so its full module is always invoked.
    /// `retainDuringLoad` runs before quantization/update/evaluation and lets
    /// native callers preserve failed construction until checked retirement.
    public static func load(
        configurationData: Data,
        indexedWeights: [String: MLXArray],
        target: any LanguageModel,
        verificationMode: CBv2MTPVerificationMode? = nil,
        allowsShortlist: Bool = true,
        retainDuringLoad: ((Qwen35InlineMTPAssistant) -> Void)? = nil
    ) throws -> Qwen35InlineMTPAssistant {
        let target = try qwen35TextTarget(target)
        let mode = resolvedVerificationMode(
            requested: verificationMode, forceSerialEnvironment: forceSerialVerification)
        guard mode != .rectangularExact || target.model.exactTargetVerify else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "rectangular_exact verification requires exact target arithmetic")
        }
        let metadata = try loadMetadata(configurationData: configurationData)
        try validate(metadata.textConfiguration, against: target.configuration)
        guard !indexedWeights.isEmpty else { throw Qwen35InlineMTPError.missingWeights }
        return try construct(
            metadata: metadata, indexed: indexedWeights, target: target,
            verificationMode: verificationMode, allowsShortlist: allowsShortlist,
            retainDuringLoad: retainDuringLoad)
    }

    private static func construct(
        metadata: Qwen35InlineMTPMetadata,
        indexed: [String: MLXArray],
        target: Qwen35TextModel,
        verificationMode: CBv2MTPVerificationMode?,
        allowsShortlist: Bool,
        retainDuringLoad: ((Qwen35InlineMTPAssistant) -> Void)? = nil
    ) throws -> Qwen35InlineMTPAssistant {
        let assistant = Qwen35InlineMTPAssistant(
            configuration: metadata.textConfiguration,
            blockSize: metadata.blockSize,
            target: target,
            verificationMode: verificationMode, allowsShortlist: allowsShortlist)
        // A native loader can keep failed construction roots alive until its
        // checked stream drain. Existing path-based callers keep their policy.
        retainDuringLoad?(assistant)

        let scaledPaths = Set(
            indexed.keys.compactMap { key -> String? in
                guard key.hasSuffix(".scales") else { return nil }
                return String(key.dropLast(".scales".count))
            })
        for path in scaledPaths
        where metadata.resolvedQuantization(for: path) == nil {
            throw Qwen35InlineMTPError.missingQuantization(path)
        }
        if !scaledPaths.isEmpty {
            quantize(model: assistant.mtp) { path, _ in
                guard scaledPaths.contains(path) else { return nil }
                return metadata.resolvedQuantization(for: path)?.asTuple
            }
        }

        try assistant.mtp.update(
            parameters: ModuleParameters.unflattened(indexed), verify: [.all])
        eval(assistant.mtp)
        return assistant
    }

    /// Bind a validated one-layer F16 or affine Q4 Bonsai 9B sidecar to an already loaded
    /// Prism target. This does not modify the base checkpoint or enable inline
    /// MTP construction. Callers must separately verify artifact provenance.
    public static func loadPrism9BSidecar(
        from modelDirectory: URL,
        target: Qwen35TextModel,
        targetConfigurationData: Data,
        verificationMode: CBv2MTPVerificationMode? = nil
    ) throws -> Qwen35InlineMTPAssistant {
        let metadata = try loadMetadata(from: modelDirectory)
        let artifactData = try Data(
            contentsOf: modelDirectory.appendingPathComponent("config.json"))
        try validatePrism9BMetadata(metadata, against: target.configuration)
        try validatePrism9BTextConfiguration(
            artifactData: artifactData, targetData: targetConfigurationData)
        guard let embedding = target.model.embedTokens as? HadamardQuantizedEmbedding,
            let head = target.lmHead as? HadamardQuantizedLinear,
            embedding.bits == 2, embedding.groupSize == 128,
            embedding.transform.width == 4096, embedding.transform.blockSize == 4096,
            head.bits == 2, head.groupSize == 128,
            head.transform.width == 4096, head.transform.blockSize == 4096,
            head.gdnLayout == nil, head.bias == nil
        else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "Prism 9B MTP requires loaded packed Hadamard embedding and output modules")
        }
        let mode = resolvedVerificationMode(
            requested: verificationMode, forceSerialEnvironment: forceSerialVerification)
        guard mode != .rectangularExact || target.model.exactTargetVerify else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "rectangular_exact verification requires exact target arithmetic")
        }
        let weights = try loadStandaloneWeights(from: modelDirectory, requiredFormat: "mlx")
        try validatePrism9BWeights(
            weights.mapValues { ($0.shape, $0.dtype) }, quantization: metadata.quantization)
        let assistant = Qwen35InlineMTPAssistant(
            configuration: metadata.textConfiguration, blockSize: metadata.blockSize,
            target: target, verificationMode: verificationMode, allowsShortlist: false,
            maximumDraftDepth: 2, minimumStateElementBytes: DType.float32.size)
        if let quantization = metadata.quantization?.quantization {
            let matrixPaths = Set(
                prism9BWeightShapes.compactMap { key, shape in
                    shape.count == 2 ? String(key.dropLast(".weight".count)) : nil
                })
            quantize(model: assistant.mtp) { path, _ in
                matrixPaths.contains(path) ? quantization.asTuple : nil
            }
        }
        // format=mlx already contains converted RMSNorm weights; never add 1.
        try assistant.mtp.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
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

    private static let globalQuantizationKeys: Set<String> = [
        "group_size", "bits", "mode",
    ]

    /// Scalar keys understood or deliberately ignored by
    /// `BaseConfiguration.QuantizationContainer`; all other keys name modules.
    private static let quantizationMetadataKeys =
        globalQuantizationKeys.union([
            "quant_method", "linear_class", "quantization_mode",
        ])

    private static func selectedJSONObject(
        in root: [String: Any],
        primaryKey: String,
        fallbackKey: String
    ) throws -> [String: Any]? {
        for key in [primaryKey, fallbackKey] {
            guard let raw = root[key], !(raw is NSNull) else { continue }
            guard let object = raw as? [String: Any] else {
                throw Qwen35InlineMTPError.invalidConfiguration(
                    "standalone MTP quantization must be an object")
            }
            return object
        }
        return nil
    }

    private static func decodeQuantization(
        _ rawQuantization: [String: Any]
    ) throws -> BaseConfiguration.PerLayerQuantization {
        var perLayer = [String: BaseConfiguration.QuantizationOption]()
        for (path, raw) in rawQuantization
        where !quantizationMetadataKeys.contains(path) {
            if CFGetTypeID(raw as CFTypeRef) == CFBooleanGetTypeID() {
                guard (raw as? Bool) == false else {
                    throw Qwen35InlineMTPError.invalidConfiguration(
                        "quantization entry \(path) must be false or an object")
                }
                perLayer[path] = .skip
            } else {
                guard let object = raw as? [String: Any] else {
                    throw Qwen35InlineMTPError.invalidConfiguration(
                        "quantization entry \(path) must be false or an object")
                }
                let data = try JSONSerialization.data(withJSONObject: object)
                perLayer[path] = .quantize(
                    try JSONDecoder().decode(
                        BaseConfiguration.Quantization.self, from: data))
            }
        }

        let hasGlobalQuantization = rawQuantization.keys.contains {
            globalQuantizationKeys.contains($0)
        }
        let global: BaseConfiguration.Quantization?
        if hasGlobalQuantization {
            let data = try JSONSerialization.data(withJSONObject: rawQuantization)
            global = try JSONDecoder().decode(
                BaseConfiguration.Quantization.self, from: data)
        } else {
            global = nil
        }
        return BaseConfiguration.PerLayerQuantization(
            quantization: global, perLayerQuantization: perLayer)
    }

    static func loadMetadata(from directory: URL) throws -> Qwen35InlineMTPMetadata {
        try loadMetadata(
            configurationData: Data(contentsOf: directory.appendingPathComponent("config.json")))
    }

    static func loadMetadata(configurationData data: Data) throws -> Qwen35InlineMTPMetadata {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let text = root["text_config"] as? [String: Any]
        else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "text_config is required")
        }

        let prefix: String
        let blockSize: Int
        let rawQuantization: [String: Any]?
        if let inline = root["mtplx_mtp"] as? [String: Any],
            inline["included"] as? Bool == true
        {
            prefix = (inline["prefix"] as? String) ?? "mtp."
            guard prefix == "mtp." else {
                throw Qwen35InlineMTPError.invalidConfiguration(
                    "only the mtp. prefix is supported")
            }
            blockSize = (inline["block_size"] as? NSNumber)?.intValue ?? 3
            guard let quantization = root["mtplx_mtp_quantization"] as? [String: Any]
            else {
                throw Qwen35InlineMTPError.invalidConfiguration(
                    "mtplx_mtp_quantization is required")
            }
            rawQuantization = quantization
        } else {
            guard (root["model_type"] as? String)?.lowercased() == "qwen3_5_mtp"
            else {
                throw Qwen35InlineMTPError.invalidConfiguration(
                    "mtplx_mtp.included=true or model_type=qwen3_5_mtp is required")
            }
            prefix = ""
            blockSize = (root["block_size"] as? NSNumber)?.intValue ?? 3
            rawQuantization = try selectedJSONObject(
                in: root,
                primaryKey: "quantization",
                fallbackKey: "quantization_config")
        }
        guard (2 ... 8).contains(blockSize) else {
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

        let quantization: BaseConfiguration.PerLayerQuantization?
        if let rawQuantization, !rawQuantization.isEmpty {
            quantization = try decodeQuantization(rawQuantization)
        } else {
            quantization = nil
        }
        return Qwen35InlineMTPMetadata(
            textConfiguration: configuration,
            prefix: prefix,
            blockSize: blockSize,
            quantization: quantization)
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

    static func loadStandaloneWeights(
        from directory: URL, requiredFormat: String? = nil
    ) throws -> [String: MLXArray] {
        let urls: [URL]
        do {
            urls = try FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            )
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        } catch {
            throw Qwen35InlineMTPError.invalidWeightIndex(String(describing: error))
        }
        guard !urls.isEmpty, urls.count <= 64 else {
            throw Qwen35InlineMTPError.missingWeights
        }

        var weights: [String: MLXArray] = [:]
        for url in urls {
            let (shard, metadata) = try loadArraysAndMetadata(url: url)
            if let requiredFormat, metadata["format"] != requiredFormat {
                throw Qwen35InlineMTPError.invalidWeightIndex(
                    "standalone tensor metadata must declare format=\(requiredFormat)")
            }
            for (key, value) in shard {
                guard !key.isEmpty, key.utf8.count <= 1024 else {
                    throw Qwen35InlineMTPError.invalidWeightIndex(
                        "standalone tensor key is empty or oversized")
                }
                guard weights.updateValue(value, forKey: key) == nil else {
                    throw Qwen35InlineMTPError.duplicateWeight(key)
                }
            }
        }
        guard !weights.isEmpty else { throw Qwen35InlineMTPError.missingWeights }
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
    final class RequestState: CBv2MTPRequestState {
        var caches: [any KVCache]

        /// Trusted target transitions not yet appended to the persistent head KV.
        /// Each hidden row at position t is paired with the token at t+1.
        var backlogHidden: [MLXArray] = []
        var backlogTokens: [MLXArray] = []
        /// Last trusted target hidden in an observed chunk. It becomes the
        /// preceding row when the next observed target chunk crosses a boundary.
        var targetHiddenFrontier: MLXArray?

        /// Cache geometry captured before and after the round's trusted flush.
        /// Every later head-chain input is speculative and is trimmed to
        /// `roundValidHistoryOffset` at finalize.
        var roundBaseOffset = 0
        var roundValidHistoryOffset = 0
        var roundDraftSteps = 0
        var roundInFlight = false
        var isReleased = false

        /// Trusted inputs moved out of the backlog for this round. Retaining
        /// their original roots both fences lazy concatenation and lets discard
        /// restore them without a host read.
        var roundTrustedHidden: [MLXArray] = []
        var roundTrustedTokens: [MLXArray] = []
        /// Final proposal hidden rows and draft ids that retain each lazy
        /// head-step graph until the engine's existing finalize synchronization.
        var roundRoots: [MLXArray] = []

        var cacheOffset: Int {
            guard let first = caches.first else { return 0 }
            precondition(
                caches.dropFirst().allSatisfy { $0.offset == first.offset },
                "inline Qwen MTP cache offsets diverged")
            return first.offset
        }

        private var backlogInputCount: Int {
            backlogTokens.reduce(0) { total, tokens in
                let (next, overflow) = total.addingReportingOverflow(tokens.dim(1))
                return overflow ? Int.max : next
            }
        }

        var committedInputCount: Int {
            let committedCache =
                roundInFlight ? roundValidHistoryOffset : cacheOffset
            let (total, overflow) = committedCache.addingReportingOverflow(
                backlogInputCount)
            return overflow ? Int.max : total
        }

        var stagedInputCount: Int {
            guard roundInFlight else { return 0 }
            return max(0, cacheOffset - roundValidHistoryOffset)
        }

        var materializedBytes: Int {
            let arrays =
                caches.flatMap { $0.innerState() }
                + backlogHidden + backlogTokens
                + [targetHiddenFrontier].compactMap { $0 }
                + roundTrustedHidden + roundTrustedTokens + roundRoots
            return arrays.reduce(0) { total, array in
                let (next, overflow) = total.addingReportingOverflow(array.nbytes)
                return overflow ? Int.max : next
            }
        }

        init(caches: [any KVCache]) { self.caches = caches }

        func clearRound() {
            roundBaseOffset = cacheOffset
            roundValidHistoryOffset = cacheOffset
            roundDraftSteps = 0
            roundInFlight = false
            roundTrustedHidden.removeAll(keepingCapacity: true)
            roundTrustedTokens.removeAll(keepingCapacity: true)
            roundRoots.removeAll(keepingCapacity: true)
        }

        func clearAll() {
            caches.removeAll(keepingCapacity: false)
            backlogHidden.removeAll(keepingCapacity: false)
            backlogTokens.removeAll(keepingCapacity: false)
            targetHiddenFrontier = nil
            roundTrustedHidden.removeAll(keepingCapacity: false)
            roundTrustedTokens.removeAll(keepingCapacity: false)
            roundRoots.removeAll(keepingCapacity: false)
            roundBaseOffset = 0
            roundValidHistoryOffset = 0
            roundDraftSteps = 0
            roundInFlight = false
            isReleased = true
        }
    }

    private final class UnusedPreparedCapture: CBv2MTPPreparedCapture {}

    public var mtpTargetIdentity: ObjectIdentifier? { targetIdentity }
    /// Verification policy. Rectangular scores all 1+k columns in one
    /// `[B, 1+k]` target forward. Widths one and two stage ordinary captured
    /// recurrent states. At S>=3 each GatedDeltaNet layer runs one full-window
    /// recurrence and retains compact transformed inputs: full acceptance
    /// installs the final state directly, while a strict accepted prefix lazily
    /// replays only that prefix. This replaces the serial per-column loop whose
    /// k+1 full-model re-reads made the round ≤1.0x by construction.
    ///
    /// NUMERICS POLICY: bitwise greedy parity with serial decode is NOT the
    /// bar here — batched verify changes accumulation geometry exactly like
    /// every other CBv2 batch-shape change (chunked prefill, B>1 decode).
    /// Distribution-exactness is the invariant: committed tokens are always
    /// target-authoritative (argmax when greedy, genuine target samples
    /// under target-prefix acceptance). The serial oracle remains available
    /// via `DARKBLOOM_QWEN_MTP_SERIAL=1` for A/B and certification runs.
    public var requiredVerificationMode: CBv2MTPVerificationMode? {
        installedVerificationMode
    }
    /// Drafting self-applies the recurrent MTP head up to the binding's depth
    /// limit. The legacy double-forward oracle stays a k=1 A/B control.
    public var maximumDraftTokens: Int? { Self.forceDoubleForward ? 1 : maximumDraftDepth }
    /// `draftHidden`/`recordDraft` are available (not the legacy double-forward oracle).
    public var splitsDrafts: Bool { !Self.forceDoubleForward }
    public var maximumSpeculativeBatch: Int? { 1 }
    /// Observation only queues lazy normalized views, and every committed
    /// position hands over the pre-norm target hidden that a seed would, so
    /// batch-gated rows may chain and a final prompt row may seed the carry.
    public var supportsOrdinaryDecodeHistory: Bool { true }
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
    /// `DARKBLOOM_QWEN_MTP_DOUBLE_FORWARD=1/true/yes/on` pins the pre-trim
    /// draft step (proposal forward + staging re-forward, both through the
    /// full lm_head) as the paired A/B control arm.
    static let forceDoubleForward: Bool = {
        guard
            let raw = ProcessInfo.processInfo.environment[
                "DARKBLOOM_QWEN_MTP_DOUBLE_FORWARD"]
        else { return false }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()

    /// `DARKBLOOM_QWEN_MTP_SHORTLIST=<K>` opts the draft head into an
    /// engine-provided top-K shortlist (score K gathered lm_head rows
    /// instead of streaming all 248,320). DEFAULT OFF: measured on M4 Max
    /// (release, B=1 greedy, 256 tok, paired session 2026-08-13) the
    /// shortlist is a net LOSS at every practical K because the ids come
    /// from the target's distribution one position BEHIND the draft:
    ///   K=256   acceptance 0.79→0.58,  95.5 tok/s (full head: 116.5)
    ///   K=2048  acceptance      0.68, 110.3 tok/s
    ///   K=16384 acceptance      0.77, 115.2 tok/s — coverage recovered,
    ///           but per-round verify-side top-K + row gather costs more
    ///           than the ~1.3 ms full 286 MB 4-bit head read it replaces.
    /// Kept env-gated for deeper draft chains (k≥2 amortizes the top-K
    /// cost across several head evaluations per round).
    static let shortlistSize: Int? = {
        guard
            let raw = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN_MTP_SHORTLIST"],
            let value = Int(raw), value > 0
        else { return nil }
        return value
    }()

    /// Shortlisting and the double-forward oracle are intentionally separate
    /// controls: the oracle scores the complete shared output head.
    public var draftShortlistSize: Int? {
        Self.resolvedDraftShortlistSize(
            requested: Self.shortlistSize, allowsShortlist: allowsShortlist,
            forceDoubleForward: Self.forceDoubleForward)
    }

    static func resolvedDraftShortlistSize(
        requested: Int?, allowsShortlist: Bool, forceDoubleForward: Bool
    ) -> Int? {
        allowsShortlist && !forceDoubleForward ? requested : nil
    }
    public var requestStateBytesPerToken: Int {
        Self.stateBytesPerToken(
            configuration: target.configuration,
            layerCount: mtp.layers.count,
            cacheElementBytes: requestCacheElementBytes,
            hiddenElementBytes: requestHiddenElementBytes)
    }
    // Prism's FP32 trusted hidden requires FP32 state reservations, independent
    // of F16 or packed Q4 sidecar matrices and their F16 norms/scales.
    private var requestCacheElementBytes: Int {
        max(mtp.norm.weight.dtype.size, minimumStateElementBytes)
    }
    private var requestHiddenElementBytes: Int {
        max(target.model.norm.weight.dtype.size, minimumStateElementBytes)
    }
    public var requestStateTokenGranularity: Int { Self.cacheAllocationStep }
    public var requestStateTokenAllocationPadding: Int { 4 }

    public var requestStateAllocationSpecs: [CBv2AuxiliaryAllocationSpec]? {
        let config = target.configuration
        let (elements, geometryOverflow) = config.kvHeads.multipliedReportingOverflow(
            by: config.headDim ?? 0)
        let (cacheRow, cacheOverflow) = elements.multipliedReportingOverflow(
            by: requestCacheElementBytes)
        let (cacheCount, countOverflow) = mtp.layers.count.multipliedReportingOverflow(by: 2)
        let (hiddenRow, hiddenOverflow) = config.hiddenSize.multipliedReportingOverflow(
            by: requestHiddenElementBytes)
        guard !geometryOverflow, !cacheOverflow, !countOverflow, !hiddenOverflow else { return nil }
        return [
            .init(
                bytesPerToken: cacheRow, allocationCount: cacheCount,
                tokenGranularity: Self.cacheAllocationStep, tokenPadding: 4),
            // Backlog and frontier are views of normalized committed chunks;
            // the bound covers every possible partition of the retained rows.
            .init(bytesPerToken: hiddenRow, tokenPadding: 4, partitioned: true),
            .init(bytesPerToken: MemoryLayout<Int32>.stride, tokenPadding: 4, partitioned: true),
        ]
    }

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

    static func stateBytesPerToken(
        configuration: Qwen35TextConfiguration,
        layerCount: Int,
        cacheElementBytes: Int,
        hiddenElementBytes: Int
    ) -> Int {
        let cacheBytes = cacheBytesPerToken(
            configuration: configuration, layerCount: layerCount,
            elementBytes: cacheElementBytes)
        let (hiddenBytes, hiddenOverflow) = configuration.hiddenSize
            .multipliedReportingOverflow(by: hiddenElementBytes)
        guard cacheBytes != Int.max, !hiddenOverflow else { return Int.max }
        let (withHidden, hiddenAdditionOverflow) = cacheBytes.addingReportingOverflow(
            hiddenBytes)
        let (withToken, tokenAdditionOverflow) = withHidden.addingReportingOverflow(
            MemoryLayout<Int32>.stride)
        return hiddenAdditionOverflow || tokenAdditionOverflow ? Int.max : withToken
    }

    public func makeRequestState() -> any CBv2MTPRequestState {
        RequestState(caches: makeCache())
    }

    /// A bounded, request-owned paged assistant history for cluster owners.
    /// The bound includes any speculative rows that may later be rolled back.
    /// Admission must also cover full-attention gathers, partial-page writes,
    /// and the ordinary trusted-hidden backlog. This does not alter the
    /// standalone engine's default state factory or allocation quotation.
    public func makePagedRequestState(maximumSequenceLength: Int) throws -> any CBv2MTPRequestState
    {
        guard (1 ... Qwen35MTPPagedKVCache.maximumTokens).contains(maximumSequenceLength) else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "paged assistant history exceeds its bounded context")
        }
        return RequestState(
            caches: mtp.layers.map { _ in
                Qwen35MTPPagedKVCache(capacityTokens: maximumSequenceLength)
            })
    }

    public func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState
    ) {
        guard let state = requestState as? RequestState else {
            preconditionFailure("inline Qwen MTP received foreign request state")
        }
        precondition(!state.isReleased, "inline Qwen MTP observed released request state")
        precondition(!state.roundInFlight, "inline Qwen MTP observed target during a round")
        precondition(
            observation.tokens.ndim == 2 && observation.hidden.ndim == 3
                && observation.tokens.dim(0) == 1 && observation.hidden.dim(0) == 1
                && observation.tokens.dim(1) == observation.hidden.dim(1),
            "inline Qwen MTP target observation shape mismatch")

        let count = observation.tokens.dim(1)
        guard count > 0 else { return }
        let normalizedHidden = targetFinalNorm(observation.hidden)

        // Cross-chunk transition: the preceding chunk's final normalized
        // target hidden pairs with this chunk's first target input.
        if let frontier = state.targetHiddenFrontier {
            state.backlogHidden.append(frontier)
            state.backlogTokens.append(observation.tokens[0..., 0 ..< 1])
        }
        // Intra-chunk transitions: finalNorm(hidden[t]) conditions token[t+1].
        if count > 1 {
            state.backlogHidden.append(normalizedHidden[0..., 0 ..< count - 1, 0...])
            state.backlogTokens.append(observation.tokens[0..., 1 ..< count])
        }
        state.targetHiddenFrontier =
            normalizedHidden[0..., (count - 1) ..< count, 0...]
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
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        guard let state = requestState as? RequestState else {
            preconditionFailure("inline Qwen MTP received foreign request state")
        }
        precondition(!state.isReleased, "inline Qwen MTP drafted with released request state")
        precondition(
            tokens.ndim == 2 && hidden.ndim == 3
                && tokens.dim(0) == 1 && tokens.dim(1) == 1
                && hidden.dim(0) == 1 && hidden.dim(1) == 1,
            "inline Qwen MTP draft input shape mismatch")

        if Self.forceDoubleForward {
            return legacyDoubleForwardDraftStep(tokens: tokens, hidden: hidden, state: state)
        }

        let lastHidden = proposalHidden(tokens: tokens, hidden: hidden, state: state)
        let draft = draftToken(hidden: lastHidden, shortlist: shortlist)
        recordProposal(lastHidden: lastHidden, draft: draft, state: state)
        return (draft, lastHidden)
    }

    /// `draftStep` without its head: this request's assistant hidden for the
    /// next proposal, its round and history advanced exactly as `draftStep`
    /// advances them. A caller that chooses several requests' proposals with
    /// one head call (each row's arithmetic that of its own call) passes each
    /// row's choice to `recordDraft` before evaluating `evaluationTargets`.
    /// Not for the legacy double-forward oracle.
    public func draftHidden(
        tokens: MLXArray, hidden: MLXArray, requestState: any CBv2MTPRequestState
    ) -> MLXArray {
        guard let state = requestState as? RequestState else {
            preconditionFailure("inline Qwen MTP received foreign request state")
        }
        precondition(!state.isReleased, "inline Qwen MTP drafted with released request state")
        precondition(
            tokens.ndim == 2 && hidden.ndim == 3
                && tokens.dim(0) == 1 && tokens.dim(1) == 1
                && hidden.dim(0) == 1 && hidden.dim(1) == 1,
            "inline Qwen MTP draft input shape mismatch")
        precondition(
            !Self.forceDoubleForward, "inline Qwen MTP split drafts exclude the legacy oracle")
        return proposalHidden(tokens: tokens, hidden: hidden, state: state)
    }

    /// The proposal chosen for `draftHidden`'s row: a `[1]` int32 token.
    public func recordDraft(
        lastHidden: MLXArray, draft: MLXArray, requestState: any CBv2MTPRequestState
    ) {
        guard let state = requestState as? RequestState else {
            preconditionFailure("inline Qwen MTP received foreign request state")
        }
        precondition(
            !state.isReleased && state.roundInFlight && draft.shape == [1] && draft.dtype == .int32,
            "inline Qwen MTP recorded a draft outside its round")
        recordProposal(lastHidden: lastHidden, draft: draft, state: state)
    }

    private func proposalHidden(tokens: MLXArray, hidden: MLXArray, state: RequestState) -> MLXArray
    {
        let isFirstStep = !state.roundInFlight
        let feed: (tokens: MLXArray, hidden: MLXArray)
        if isFirstStep {
            feed = beginRound(tokens: tokens, hidden: hidden, state: state)
        } else {
            precondition(
                state.roundDraftSteps < 4,
                "inline Qwen MTP exceeded its four-step draft chain")
            feed = (tokens, hidden)
        }

        let output: MLXArray
        if isFirstStep {
            let remainder = flushLeadingHistory(feed, state: state)
            output =
                moduleLastHiddenWithKVOnlyHistory(
                    hidden: remainder.hidden, tokens: remainder.tokens, cache: state.caches)
                ?? moduleForward(
                    hidden: remainder.hidden, tokens: remainder.tokens, cache: state.caches)
        } else {
            output = moduleForward(
                hidden: feed.hidden, tokens: feed.tokens, cache: state.caches)
        }
        return output[0..., (output.dim(1) - 1)..., 0...]
    }

    private func recordProposal(lastHidden: MLXArray, draft: MLXArray, state: RequestState) {
        // `beginRound` zeroes the step count: zero here is the round's first step.
        let isFirstStep = state.roundDraftSteps == 0
        state.roundRoots.append(contentsOf: [lastHidden, draft])
        state.roundDraftSteps += 1

        if isFirstStep {
            // The engine immediately submits evaluationTargets for this first
            // cache generation before constructing a deeper draft step.
            state.roundValidHistoryOffset = state.cacheOffset
        }
    }

    private func beginRound(
        tokens: MLXArray, hidden: MLXArray, state: RequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        precondition(!state.roundInFlight, "inline Qwen MTP round already in flight")
        precondition(
            state.backlogHidden.count == state.backlogTokens.count,
            "inline Qwen MTP trusted backlog diverged")

        state.roundBaseOffset = state.cacheOffset
        state.roundValidHistoryOffset = state.cacheOffset
        state.roundDraftSteps = 0
        state.roundInFlight = true
        state.roundTrustedHidden = state.backlogHidden
        state.roundTrustedTokens = state.backlogTokens
        state.backlogHidden.removeAll(keepingCapacity: true)
        state.backlogTokens.removeAll(keepingCapacity: true)

        // The current target carry is trusted and completes the frontier
        // transition. Normalize it exactly once before it enters head history.
        state.roundTrustedHidden.append(targetFinalNorm(hidden))
        state.roundTrustedTokens.append(tokens)
        state.targetHiddenFrontier = nil

        if state.roundTrustedTokens.count == 1 {
            return (state.roundTrustedTokens[0], state.roundTrustedHidden[0])
        }
        let feedTokens = concatenated(state.roundTrustedTokens, axis: 1)
        let feedHidden = concatenated(state.roundTrustedHidden, axis: 1)
        return (feedTokens, feedHidden)
    }

    /// Bounded first draft: all but the final `historyFlushRows + 1` feed rows
    /// enter head K/V in fixed blocks from the start of the feed, so the
    /// remaining last-hidden rectangle never exceeds one block plus its
    /// proposal row. The original feed roots stay retained by the round.
    private func flushLeadingHistory(
        _ feed: (tokens: MLXArray, hidden: MLXArray), state: RequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        guard let rows = historyFlushRows, mtp.layers.count == 1, state.caches.count == 1 else {
            return feed
        }
        let count = feed.tokens.dim(1)
        var start = 0
        while count - start > rows + 1 {
            appendFlushedHistory(
                tokens: feed.tokens[0..., start ..< (start + rows)],
                hidden: feed.hidden[0..., start ..< (start + rows), 0...], state: state)
            start += rows
        }
        guard start > 0 else { return feed }
        return (feed.tokens[0..., start...], feed.hidden[0..., start..., 0...])
    }

    private func appendFlushedHistory(tokens: MLXArray, hidden: MLXArray, state: RequestState) {
        let appended = mtp.appendHistoryKV(
            hidden: hidden, nextTokenIds: tokens, embedTokens: target.model.embedTokens,
            cache: state.caches)
        precondition(appended, "inline Qwen MTP history flush requires its single-layer head")
        eval(state.caches.flatMap { $0.innerState() })
    }

    /// Trusted rows queued for head K/V (inspection: what a request holds
    /// between flushes).
    public func trustedBacklogRows(requestState: any CBv2MTPRequestState) -> Int {
        guard let state = requestState as? RequestState else { return 0 }
        return state.backlogTokens.reduce(0) { $0 + $1.dim(1) }
    }

    /// Move the trusted backlog's leading complete `historyFlushRows` blocks
    /// into head K/V while more than one block of rows stays queued behind
    /// them. These are exactly the blocks the next round's first draft would
    /// flush first (`flushLeadingHistory` cuts blocks from the start of its
    /// feed, which begins with this backlog, and flushes every block more
    /// than `historyFlushRows + 1` rows from the feed's end): same rows, same
    /// shapes, same order, so head history and every later proposal are
    /// unchanged. A long prompt then holds at most about one block plus one
    /// prompt frame of trusted rows, not its whole history, until its first
    /// round. The remaining rows are copied out of the flushed backing. Call
    /// only between rounds, and never before a prompt prefix checkpoint the
    /// caller still has to take. Returns the number of rows moved.
    @discardableResult
    public func flushCompleteHistoryBlocks(requestState: any CBv2MTPRequestState) -> Int {
        guard let state = requestState as? RequestState else {
            preconditionFailure("inline Qwen MTP received foreign request state")
        }
        precondition(!state.isReleased, "inline Qwen MTP flushed released request state")
        precondition(!state.roundInFlight, "inline Qwen MTP flushed history during a round")
        precondition(
            state.backlogHidden.count == state.backlogTokens.count,
            "inline Qwen MTP trusted backlog diverged")
        guard let rows = historyFlushRows, mtp.layers.count == 1, state.caches.count == 1
        else { return 0 }
        let count = state.backlogTokens.reduce(0) { $0 + $1.dim(1) }
        // A block [s, s + rows) is flushed by the first draft whenever the
        // feed (this backlog, any later rows, then the carry) holds more than
        // rows + 1 rows from s on; `count - s > rows` guarantees that.
        guard count > rows else { return 0 }
        let tokens = concatenated(state.backlogTokens, axis: 1)
        let hidden = concatenated(state.backlogHidden, axis: 1)
        var start = 0
        while count - start > rows {
            appendFlushedHistory(
                tokens: tokens[0..., start ..< (start + rows)],
                hidden: hidden[0..., start ..< (start + rows), 0...], state: state)
            start += rows
        }
        let restTokens = contiguous(tokens[0..., start...])
        let restHidden = contiguous(hidden[0..., start..., 0...])
        eval(restTokens, restHidden)
        state.backlogTokens = [restTokens]
        state.backlogHidden = [restHidden]
        return start
    }

    /// Move the complete trusted backlog into head K/V once it holds at least
    /// `historyFlushRows` rows, one evaluated block at a time, so an ordinary
    /// decode stretch cannot accumulate one root per committed token. The
    /// carried frontier stays queued. Call only between rounds, never before a
    /// prompt prefix checkpoint (it carries unflushed history). Returns the
    /// number of rows moved.
    @discardableResult
    public func flushTrustedHistory(requestState: any CBv2MTPRequestState) -> Int {
        guard let state = requestState as? RequestState else {
            preconditionFailure("inline Qwen MTP received foreign request state")
        }
        precondition(!state.isReleased, "inline Qwen MTP flushed released request state")
        precondition(!state.roundInFlight, "inline Qwen MTP flushed history during a round")
        precondition(
            state.backlogHidden.count == state.backlogTokens.count,
            "inline Qwen MTP trusted backlog diverged")
        guard let rows = historyFlushRows, mtp.layers.count == 1, state.caches.count == 1
        else { return 0 }
        let count = state.backlogTokens.reduce(0) { $0 + $1.dim(1) }
        guard count >= rows else { return 0 }
        let tokens = concatenated(state.backlogTokens, axis: 1)
        let hidden = concatenated(state.backlogHidden, axis: 1)
        var start = 0
        while start < count {
            let end = min(count, start + rows)
            appendFlushedHistory(
                tokens: tokens[0..., start ..< end], hidden: hidden[0..., start ..< end, 0...],
                state: state)
            start = end
        }
        state.backlogHidden.removeAll(keepingCapacity: true)
        state.backlogTokens.removeAll(keepingCapacity: true)
        return count
    }

    private func draftToken(hidden: MLXArray, shortlist: MLXArray?) -> MLXArray {
        if allowsShortlist, let shortlist {
            let logits = shortlistLogits(hidden: hidden, ids: shortlist)
            return shortlist[argMax(logits[0..., -1, 0...], axis: -1)]
                .asType(.int32)
        }
        return argMax(headLogits(hidden)[0..., -1, 0...], axis: -1)
            .asType(.int32)
    }

    /// Explicit pre-cutover A/B oracle. It shares production history upkeep,
    /// but stages the proposed draft through a second complete MTP/lm-head
    /// forward and performs the historical blocking mid-round evaluation.
    private func legacyDoubleForwardDraftStep(
        tokens: MLXArray, hidden: MLXArray, state: RequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        precondition(!state.roundInFlight, "inline Qwen MTP round already in flight")
        let feed = beginRound(tokens: tokens, hidden: hidden, state: state)
        let output =
            moduleLastHiddenWithKVOnlyHistory(
                hidden: feed.hidden, tokens: feed.tokens, cache: state.caches)
            ?? moduleForward(
                hidden: feed.hidden, tokens: feed.tokens, cache: state.caches)
        let lastHidden = output[0..., (output.dim(1) - 1)..., 0...]
        let logits = headLogits(lastHidden)
        let draft = argMax(logits[0..., -1, 0...], axis: -1).asType(.int32)
        state.roundRoots.append(contentsOf: [lastHidden, draft])
        state.roundDraftSteps = 1
        state.roundValidHistoryOffset = state.cacheOffset

        eval([draft, lastHidden] + state.caches.flatMap { $0.innerState() })
        _ = forward(
            hidden: lastHidden, tokens: draft.reshaped([1, 1]), cache: state.caches)
        return (draft, lastHidden)
    }

    public func evaluationTargets(
        for requestState: any CBv2MTPRequestState
    ) -> [MLXArray] {
        guard let state = requestState as? RequestState, !state.isReleased else {
            return []
        }
        return state.caches.flatMap { $0.innerState() }
            + state.backlogHidden + state.backlogTokens
            + [state.targetHiddenFrontier].compactMap { $0 }
            + state.roundTrustedHidden + state.roundTrustedTokens + state.roundRoots
    }

    public func finalizeRound(
        requestState: any CBv2MTPRequestState,
        confirmedInputTokens: Int,
        committedDraftTokens: MLXArray,
        committedTargetHidden: MLXArray
    ) {
        guard let state = requestState as? RequestState else {
            preconditionFailure("inline Qwen MTP received foreign request state")
        }
        precondition(!state.isReleased, "inline Qwen MTP finalized released request state")
        precondition(state.roundInFlight, "inline Qwen MTP finalized without a round")
        precondition(
            (0 ... state.roundDraftSteps + 1).contains(confirmedInputTokens),
            "inline Qwen MTP confirmed prefix exceeds the draft round")
        precondition(
            committedDraftTokens.ndim == 2 && committedTargetHidden.ndim == 3
                && committedDraftTokens.dim(0) == 1
                && committedTargetHidden.dim(0) == 1
                && committedDraftTokens.dim(1) == committedTargetHidden.dim(1),
            "inline Qwen MTP committed target rows mismatch")
        let committedDraftCount = committedDraftTokens.dim(1)
        precondition(
            committedDraftCount <= state.roundDraftSteps
                && committedDraftCount <= max(0, confirmedInputTokens - 1),
            "inline Qwen MTP committed drafts exceed confirmed target inputs")

        trim(state: state, to: state.roundValidHistoryOffset)
        if committedDraftCount > 0 {
            // These are pre-final-norm target verify hiddens, never speculative
            // assistant hiddens. Normalize exactly once, retain lazily, and
            // flush them with the next carry.
            state.backlogTokens.append(committedDraftTokens)
            state.backlogHidden.append(targetFinalNorm(committedTargetHidden))
        }
        state.clearRound()
    }

    public func discardRound(requestState: any CBv2MTPRequestState) {
        guard let state = requestState as? RequestState,
            !state.isReleased, state.roundInFlight
        else { return }

        trim(state: state, to: state.roundBaseOffset)
        // Restore all trusted transitions consumed by the abandoned graph in
        // original order. No speculative assistant hidden enters the backlog.
        state.backlogHidden =
            state.roundTrustedHidden + state.backlogHidden
        state.backlogTokens =
            state.roundTrustedTokens + state.backlogTokens
        state.clearRound()
    }

    private func trim(state: RequestState, to offset: Int) {
        let rollback = state.cacheOffset - offset
        precondition(rollback >= 0, "inline Qwen MTP cache checkpoint moved forward")
        guard rollback > 0 else { return }
        for cache in state.caches {
            precondition(cache.trim(rollback) == rollback)
        }
    }

    public func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        guard let state = requestState as? RequestState, !state.isReleased else {
            return
        }
        state.clearAll()
    }
}
