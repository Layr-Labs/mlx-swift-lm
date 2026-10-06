// Copyright © 2026 Eigen Labs / Darkbloom Assignment 1.
//
// Fusion Lightning MTP for `qwen4_exp` (not Qwen 3.5 concat-FC).
// Source of truth: omlx `qwen4_exp/language.py` `Qwen4ExpMTPModule`.
//
// Hidden contract:
//   - Target capture is the pre-mixer residual `[B, T, hc·H]`.
//   - Logits come from the mixed `[B, T, H]` mixer output.
//   - There is no Qwen3.5 `targetFinalNorm` — oQ4e has no final RMSNorm.

import CoreFoundation
import Foundation
import MLX
import MLXLMCommon
import MLXNN

public protocol Qwen4ExpMTPTargeting: AnyObject {
    var qwen4ExpTextTarget: Qwen4ExpTextModel { get }
}

extension Qwen4ExpTextModel: Qwen4ExpMTPTargeting {
    public var qwen4ExpTextTarget: Qwen4ExpTextModel { self }
}

extension Qwen4ExpModel: Qwen4ExpMTPTargeting {
    public var qwen4ExpTextTarget: Qwen4ExpTextModel { languageModel }
}

public enum Qwen4ExpInlineMTPError: Error, LocalizedError, Sendable, Equatable {
    case invalidConfiguration(String)
    case incompatibleTarget(field: String, artifact: Int, target: Int)
    case invalidWeightIndex(String)
    case missingWeights
    case duplicateWeight(String)
    case missingQuantization(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail):
            return "Invalid inline Qwen4 Lightning MTP configuration: \(detail)."
        case .incompatibleTarget(let field, let artifact, let target):
            return "Inline Qwen4 Lightning MTP target mismatch at \(field): artifact=\(artifact), target=\(target)."
        case .invalidWeightIndex(let detail):
            return "Invalid inline Qwen4 Lightning MTP weight index: \(detail)."
        case .missingWeights:
            return "The checkpoint declares Lightning MTP but contains no matching tensors."
        case .duplicateWeight(let key):
            return "The Lightning MTP tensor \(key) appears more than once."
        case .missingQuantization(let path):
            return "The quantized Lightning MTP module \(path) has no matching quantization entry."
        }
    }
}

struct Qwen4ExpInlineMTPMetadata: Sendable {
    let textConfiguration: Qwen4ExpTextConfiguration
    let prefix: String
    let blockSize: Int
    let quantization: BaseConfiguration.PerLayerQuantization?

    func resolvedQuantization(
        for path: String
    ) -> BaseConfiguration.Quantization? {
        quantization?.quantization(layer: path)
    }
}

private func lightningMTPLayerConfiguration(
    _ args: Qwen4ExpTextConfiguration
) -> Qwen4ExpTextConfiguration {
    var copy = args
    copy.hiddenLayers = 1
    copy.layerTypes = ["qwen_sparse_attention"]
    copy.fullAttentionInterval = 1
    copy.pleLayerIds = []
    return copy
}

// MARK: - MTPModule

/// Fusion `Qwen4ExpMTPModule`: project embed and each residual stream,
/// add them, run one QSA decoder layer, mix for logits, keep residual.
final class Qwen4ExpMTPModule: Module {
    let hiddenSize: Int
    let hcCount: Int

    @ModuleInfo(key: "pre_fc_norm_embedding") var preFcNormEmbedding: Qwen4ExpRMSNorm
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFcNormHidden: Qwen4ExpRMSNorm
    @ModuleInfo(key: "fc_embedding") var fcEmbedding: Linear
    @ModuleInfo(key: "fc_hidden") var fcHidden: Linear
    @ModuleInfo(key: "layers") var layers: [Qwen4ExpDecoderLayer]
    @ModuleInfo(key: "hyper_connection_mixer") var hyperConnectionMixer: Qwen4ExpGatedResidual

    init(_ args: Qwen4ExpTextConfiguration) throws {
        self.hiddenSize = args.hiddenSize
        self.hcCount = args.hcCount
        let hcHidden = args.hcCount * args.hiddenSize
        _preFcNormEmbedding.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFcNormHidden.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: hcHidden, eps: args.rmsNormEps)
        _fcEmbedding.wrappedValue = Linear(args.hiddenSize, args.hiddenSize, bias: false)
        _fcHidden.wrappedValue = Linear(args.hiddenSize, args.hiddenSize, bias: false)
        let layerConfig = lightningMTPLayerConfiguration(args)
        let layerCount = max(1, args.mtpNumHiddenLayers)
        _layers.wrappedValue = try (0 ..< layerCount).map { _ in
            try Qwen4ExpDecoderLayer(
                layerConfig, layerIdx: 0, mmapPLE: false, fuseGateUp: true)
        }
        _hyperConnectionMixer.wrappedValue = Qwen4ExpGatedResidual(
            layerConfig, useCombine: false)
        super.init()
    }

    func fuseInputs(tokenEmbeddings: MLXArray, hiddenStates: MLXArray) -> MLXArray {
        let expectedWidth = hcCount * hiddenSize
        var hidden = hiddenStates
        if hidden.ndim == 4 {
            let prefix = Array(hidden.shape.dropLast(2))
            hidden = hidden.reshaped(prefix + [expectedWidth])
        }
        precondition(
            hidden.ndim == 3 && hidden.dim(-1) == expectedWidth,
            "Qwen4 Lightning MTP expects hidden shape [batch, tokens, hc_count * hidden_size]")
        let projectedEmbedding = fcEmbedding(preFcNormEmbedding(tokenEmbeddings))
        let streams = preFcNormHidden(hidden).reshaped(
            hidden.dim(0), hidden.dim(1), hcCount, hiddenSize)
        let projectedHidden = fcHidden(streams)
        return (projectedEmbedding.expandedDimensions(axis: -2) + projectedHidden)
            .reshaped(hidden.shape)
    }

    func callAsFunction(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any CBv2AttendingLayerCache]
    ) -> (mixed: MLXArray, residual: MLXArray) {
        var residual = fuseInputs(
            tokenEmbeddings: embedTokens(nextTokenIds), hiddenStates: hidden)
        precondition(cache.count == layers.count, "Qwen4 Lightning MTP cache count mismatch")
        for (layer, layerCache) in zip(layers, cache) {
            residual = layer.cbv2Forward(
                residual,
                inputIds: nextTokenIds,
                modelLayerIndex: 0,
                attentionCache: layerCache,
                recurrentState: [],
                positionIds: nil)
        }
        return (hyperConnectionMixer.mix(residual), residual)
    }
}

// MARK: - Artifact-scoped assistant

public final class Qwen4ExpInlineMTPAssistant: Module, @unchecked Sendable {
    static let cacheAllocationStep = 256

    private let mtp: Qwen4ExpMTPModule
    private let target: Qwen4ExpTextModel
    private let installedVerificationMode: CBv2MTPVerificationMode
    private let primeChunkTokens: Int
    private let skipColdPromptReplay: Bool
    let skipUnprimedRestoredReplay: Bool

    public let blockSize: Int
    public var targetIdentity: ObjectIdentifier { ObjectIdentifier(target) }

    var prefixCheckpointGeometry: (width: Int, dtype: DType, vocabulary: Int, maximumLength: Int, verification: String) {
        (target.configuration.hcCount * target.configuration.hiddenSize,
         Qwen4ExpActivation.hiddenDType(from: target.model.embedTokens),
         target.vocabularySize, target.configuration.maxPositionEmbeddings,
         installedVerificationMode.rawValue)
    }

    init(
        configuration: Qwen4ExpTextConfiguration,
        blockSize: Int,
        target: Qwen4ExpTextModel,
        verificationMode: CBv2MTPVerificationMode?,
        primeChunkTokens: Int? = nil,
        skipColdPromptReplay: Bool? = nil
    ) throws {
        precondition(
            Self.maximumChainedDraftTokens <= CBv2MTPConfig.testedMaxDraftTokens,
            "Qwen4 Lightning draft depth exceeds CBv2's verified rectangle")
        self.mtp = try Qwen4ExpMTPModule(configuration)
        self.blockSize = blockSize
        self.target = target
        self.primeChunkTokens = Qwen4ExpMTPPriming.validatedChunkTokens(
            primeChunkTokens ?? Qwen4ExpMTPPriming.environmentChunkTokens())
        let replayPolicy = skipColdPromptReplay.map {
            $0 ? Qwen4ExpMTPPriming.ReplayPolicy.coldOnly : .full
        } ?? Qwen4ExpMTPPriming.replayPolicy()
        self.skipColdPromptReplay = replayPolicy != .full
        self.skipUnprimedRestoredReplay = replayPolicy == .unprimed
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

    public func makeCache() -> [any KVCache] {
        mtp.layers.enumerated().map { index, _ in
            let kind = CBv2LayerKind(
                attention: .full,
                headDim: target.configuration.headDim,
                kvHeads: target.configuration.kvHeads,
                queryHeads: target.configuration.attentionHeads,
                modelLayerIndex: 0,
                extraStorageBytesPerToken: Qwen4ExpPrefillMemory.qsaSidecarBytesPerToken(
                    indexerHeadDim: target.configuration.indexerHeadDim,
                    compressRatio: target.configuration.indexerCompressRatio))
            let row = CBv2FullSequenceKV(
                promptLength: 0,
                maxLength: target.configuration.maxPositionEmbeddings,
                kvHeads: target.configuration.kvHeads,
                headDim: target.configuration.headDim)
            return CBv2LayerCache(layerIndex: index, kind: kind, rows: [row])
        }
    }

    public func forward(
        hidden: MLXArray,
        tokens: MLXArray,
        cache: [any KVCache]
    ) -> (logits: MLXArray, hidden: MLXArray) {
        let output = moduleForward(hidden: hidden, tokens: tokens, cache: cache)
        return (headLogits(output.mixed), output.residual)
    }

    func headLogits(_ mixed: MLXArray) -> MLXArray {
        if target.configuration.tieWordEmbeddings {
            return target.model.embedTokens.asLinear(mixed)
        }
        return target.lmHead!(mixed)
    }

    func moduleForward(
        hidden: MLXArray,
        tokens: MLXArray,
        cache: [any KVCache]
    ) -> (mixed: MLXArray, residual: MLXArray) {
        mtp(
            hidden: hidden,
            nextTokenIds: tokens,
            embedTokens: target.model.embedTokens,
            cache: attendingCaches(cache))
    }

    func shortlistLogits(hidden: MLXArray, ids: MLXArray) -> MLXArray {
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

    public static func load(
        from modelDirectory: URL,
        target: any LanguageModel,
        verificationMode: CBv2MTPVerificationMode? = nil
    ) throws -> Qwen4ExpInlineMTPAssistant {
        let target = try qwen4TextTarget(target)
        let metadata = try loadMetadata(from: modelDirectory)
        try validate(metadata.textConfiguration, against: target.configuration)

        var indexed: [String: MLXArray]
        if metadata.prefix.isEmpty {
            indexed = try loadStandaloneWeights(from: modelDirectory)
        } else {
            indexed = try loadIndexedWeights(
                from: modelDirectory, prefix: metadata.prefix)
        }
        let assistant = try Qwen4ExpInlineMTPAssistant(
            configuration: metadata.textConfiguration,
            blockSize: metadata.blockSize,
            target: target,
            verificationMode: verificationMode)

        // Prefix-stripped official Q4 assistant experts retain their stacked
        // gate/up layout. Re-key the packed tensors with their scales/biases;
        // split compatible checkpoints use the same checked fusion helper.
        if metadata.textConfiguration.numExperts > 0 {
            indexed = qwen35FuseSwitchMLPGateUp(
                weights: indexed,
                perLayerQuantization: metadata.quantization,
                setFused: { qwen35SetSwitchGLUGateUpFused($1, at: $0, in: assistant.mtp) })
        }

        let scaledPaths = Set(indexed.keys.compactMap { key -> String? in
            guard key.hasSuffix(".scales") else { return nil }
            return String(key.dropLast(".scales".count))
        })
        for path in scaledPaths
        where metadata.resolvedQuantization(for: path) == nil {
            throw Qwen4ExpInlineMTPError.missingQuantization(path)
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

    private static func qwen4TextTarget(
        _ target: any LanguageModel
    ) throws -> Qwen4ExpTextModel {
        if let targeting = target as? any Qwen4ExpMTPTargeting {
            return targeting.qwen4ExpTextTarget
        }
        throw Qwen4ExpInlineMTPError.invalidConfiguration(
            "target type \(String(describing: type(of: target))) is not Qwen4 Lightning")
    }

    private static let globalQuantizationKeys: Set<String> = [
        "group_size", "bits", "mode"
    ]

    private static let quantizationMetadataKeys =
        globalQuantizationKeys.union([
            "quant_method", "linear_class", "quantization_mode"
        ])

    private static func selectedJSONObject(
        in root: [String: Any],
        primaryKey: String,
        fallbackKey: String
    ) throws -> [String: Any]? {
        for key in [primaryKey, fallbackKey] {
            guard let raw = root[key], !(raw is NSNull) else { continue }
            guard let object = raw as? [String: Any] else {
                throw Qwen4ExpInlineMTPError.invalidConfiguration(
                    "Lightning MTP quantization must be an object")
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
                    throw Qwen4ExpInlineMTPError.invalidConfiguration(
                        "quantization entry \(path) must be false or an object")
                }
                perLayer[path] = .skip
            } else {
                guard let object = raw as? [String: Any] else {
                    throw Qwen4ExpInlineMTPError.invalidConfiguration(
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

    static func loadMetadata(from directory: URL) throws -> Qwen4ExpInlineMTPMetadata {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw Qwen4ExpInlineMTPError.invalidConfiguration("config.json is not an object")
        }
        guard let text = root["text_config"] as? [String: Any] else {
            throw Qwen4ExpInlineMTPError.invalidConfiguration("text_config is required")
        }

        let prefix: String
        let blockSize: Int
        let rawQuantization: [String: Any]?
        if let inline = root["mtplx_mtp"] as? [String: Any],
            inline["included"] as? Bool == true
        {
            prefix = (inline["prefix"] as? String) ?? "mtp."
            guard prefix == "mtp." else {
                throw Qwen4ExpInlineMTPError.invalidConfiguration(
                    "only the mtp. prefix is supported")
            }
            blockSize = (inline["block_size"] as? NSNumber)?.intValue ?? 3
            guard let quantization = root["mtplx_mtp_quantization"] as? [String: Any]
            else {
                throw Qwen4ExpInlineMTPError.invalidConfiguration(
                    "mtplx_mtp_quantization is required")
            }
            rawQuantization = quantization
        } else if let inTree = try huggingFaceInTreeMTP(root: root, directory: directory)
        {
            prefix = inTree.prefix
            blockSize = inTree.blockSize
            rawQuantization = inTree.quantization
        } else {
            throw Qwen4ExpInlineMTPError.invalidConfiguration(
                "Fusion in-tree mtp.* tensors or mtplx_mtp.included=true is required")
        }
        guard (2...8).contains(blockSize) else {
            throw Qwen4ExpInlineMTPError.invalidConfiguration(
                "block_size \(blockSize) is outside 2...8")
        }
        let textData = try JSONSerialization.data(withJSONObject: text)
        let configuration = try JSONDecoder.json5().decode(
            Qwen4ExpTextConfiguration.self, from: textData)
        guard configuration.mtpNumHiddenLayers > 0,
            configuration.mtpNumHiddenLayers <= 4
        else {
            throw Qwen4ExpInlineMTPError.invalidConfiguration(
                "mtp_num_hidden_layers must be within 1...4")
        }

        let quantization: BaseConfiguration.PerLayerQuantization?
        if let rawQuantization, !rawQuantization.isEmpty {
            quantization = try decodeQuantization(rawQuantization)
        } else {
            quantization = nil
        }
        return Qwen4ExpInlineMTPMetadata(
            textConfiguration: configuration,
            prefix: prefix,
            blockSize: blockSize,
            quantization: quantization)
    }

    private static func huggingFaceInTreeMTP(
        root: [String: Any], directory: URL
    ) throws -> (prefix: String, blockSize: Int, quantization: [String: Any]?)? {
        let text = root["text_config"] as? [String: Any]
        let layers = (text?["mtp_num_hidden_layers"] as? NSNumber)?.intValue
            ?? (root["mtp_num_hidden_layers"] as? NSNumber)?.intValue
            ?? 0
        guard (1...4).contains(layers) else { return nil }

        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        guard FileManager.default.fileExists(atPath: indexURL.path),
            let indexData = try? Data(contentsOf: indexURL),
            let index = try? JSONDecoder().decode(WeightIndex.self, from: indexData)
        else { return nil }

        let prefix: String
        if index.weightMap.keys.contains(where: { $0.hasPrefix("language_model.mtp.") }) {
            prefix = "language_model.mtp."
        } else if index.weightMap.keys.contains(where: { $0.hasPrefix("mtp.") }) {
            prefix = "mtp."
        } else {
            return nil
        }

        let blockSize = (root["block_size"] as? NSNumber)?.intValue
            ?? (text?["block_size"] as? NSNumber)?.intValue
            ?? 3
        var quantization = try selectedJSONObject(
            in: root, primaryKey: "mtplx_mtp_quantization", fallbackKey: "quantization")
        if quantization == nil {
            quantization = try selectedJSONObject(
                in: root, primaryKey: "quantization_config", fallbackKey: "quantization_config")
        }
        if let raw = quantization {
            var global: [String: Any] = [:]
            for key in globalQuantizationKeys where raw[key] != nil {
                global[key] = raw[key]
            }
            for (key, value) in raw
            where key.hasPrefix("language_model.mtp.") || key.hasPrefix("mtp.") {
                let stripped = key.hasPrefix("language_model.mtp.")
                    ? String(key.dropFirst("language_model.mtp.".count))
                    : String(key.dropFirst("mtp.".count))
                global[stripped] = value
            }
            quantization = global.isEmpty ? nil : global
        }
        return (prefix, blockSize, quantization)
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
            throw Qwen4ExpInlineMTPError.invalidWeightIndex(String(describing: error))
        }

        var byFile: [String: [(source: String, destination: String)]] = [:]
        for (key, file) in index.weightMap where key.hasPrefix(prefix) {
            guard file == URL(fileURLWithPath: file).lastPathComponent,
                file.hasSuffix(".safetensors")
            else {
                throw Qwen4ExpInlineMTPError.invalidWeightIndex(
                    "unsafe shard path for \(key)")
            }
            let destination = String(key.dropFirst(prefix.count))
            guard !destination.isEmpty else {
                throw Qwen4ExpInlineMTPError.invalidWeightIndex("empty stripped key")
            }
            byFile[file, default: []].append((key, destination))
        }
        guard !byFile.isEmpty else { throw Qwen4ExpInlineMTPError.missingWeights }

        var weights: [String: MLXArray] = [:]
        for file in byFile.keys.sorted() {
            let url = directory.appendingPathComponent(file)
            let (shard, _) = try loadArraysAndMetadata(url: url)
            for entry in byFile[file]! {
                guard let value = shard[entry.source] else {
                    throw Qwen4ExpInlineMTPError.invalidWeightIndex(
                        "indexed tensor \(entry.source) is absent from \(file)")
                }
                guard weights.updateValue(value, forKey: entry.destination) == nil else {
                    throw Qwen4ExpInlineMTPError.duplicateWeight(entry.destination)
                }
            }
        }
        return weights
    }

    static func loadStandaloneWeights(
        from directory: URL
    ) throws -> [String: MLXArray] {
        let urls: [URL]
        do {
            urls = try FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles])
                .filter { $0.pathExtension == "safetensors" }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
        } catch {
            throw Qwen4ExpInlineMTPError.invalidWeightIndex(String(describing: error))
        }
        guard !urls.isEmpty, urls.count <= 64 else {
            throw Qwen4ExpInlineMTPError.missingWeights
        }

        var weights: [String: MLXArray] = [:]
        for url in urls {
            let (shard, _) = try loadArraysAndMetadata(url: url)
            for (key, value) in shard {
                guard !key.isEmpty, key.utf8.count <= 1024 else {
                    throw Qwen4ExpInlineMTPError.invalidWeightIndex(
                        "standalone tensor key is empty or oversized")
                }
                guard weights.updateValue(value, forKey: key) == nil else {
                    throw Qwen4ExpInlineMTPError.duplicateWeight(key)
                }
            }
        }
        guard !weights.isEmpty else { throw Qwen4ExpInlineMTPError.missingWeights }
        return weights
    }

    private static func validate(
        _ artifact: Qwen4ExpTextConfiguration,
        against target: Qwen4ExpTextConfiguration
    ) throws {
        let fields: [(String, Int, Int)] = [
            ("hidden_size", artifact.hiddenSize, target.hiddenSize),
            ("vocab_size", artifact.vocabularySize, target.vocabularySize),
            ("num_attention_heads", artifact.attentionHeads, target.attentionHeads),
            ("num_key_value_heads", artifact.kvHeads, target.kvHeads),
            ("head_dim", artifact.headDim, target.headDim),
            ("num_experts", artifact.numExperts, target.numExperts),
            ("num_experts_per_tok", artifact.numExpertsPerTok, target.numExpertsPerTok),
            ("hc_count", artifact.hcCount, target.hcCount),
        ]
        for (field, artifactValue, targetValue) in fields
        where artifactValue != targetValue {
            throw Qwen4ExpInlineMTPError.incompatibleTarget(
                field: field, artifact: artifactValue, target: targetValue)
        }
    }

    private func attendingCaches(_ cache: [any KVCache]) -> [any CBv2AttendingLayerCache] {
        cache.map { entry in
            guard let attending = entry as? any CBv2AttendingLayerCache else {
                preconditionFailure("Qwen4 Lightning MTP requires CBv2 QSA caches")
            }
            return attending
        }
    }
}

extension Qwen4ExpInlineMTPAssistant: CBv2MTPRequestStatefulDrafter {
    final class RequestState: CBv2MTPRequestState {
        let owner: ObjectIdentifier
        var caches: [any KVCache]
        var backlogHidden: [MLXArray] = []
        var backlogTokens: [MLXArray] = []
        var targetHiddenFrontier: MLXArray?
        var roundBaseOffset = 0
        var roundValidHistoryOffset = 0
        var roundDraftSteps = 0
        var roundSkippedColdInputCount = 0
        var roundInFlight = false
        var isReleased = false
        /// Logical target inputs deliberately absent from the cold assistant
        /// cache. They remain part of the target-authoritative request cursor.
        var logicalInputBase = 0
        /// True only for a process-local, newly-created request state. Restored
        /// prefix/state checkpoints explicitly keep the established replay path.
        var coldPromptReplayEligible: Bool
        var roundTrustedHidden: [MLXArray] = []
        var roundTrustedTokens: [MLXArray] = []
        var roundRoots: [MLXArray] = []

        var cacheOffset: Int {
            guard let first = caches.first else { return 0 }
            precondition(
                caches.dropFirst().allSatisfy { $0.offset == first.offset },
                "Lightning MTP cache offsets diverged")
            return first.offset
        }

        var backlogInputCount: Int {
            backlogTokens.reduce(0) { total, tokens in
                let (next, overflow) = total.addingReportingOverflow(tokens.dim(1))
                return overflow ? Int.max : next
            }
        }

        var committedInputCount: Int {
            let committedCache =
                roundInFlight ? roundValidHistoryOffset : cacheOffset
            let inFlightColdBase = roundInFlight ? roundSkippedColdInputCount : 0
            let (base, baseOverflow) = logicalInputBase.addingReportingOverflow(
                inFlightColdBase)
            let (withCache, cacheOverflow) = base.addingReportingOverflow(
                committedCache)
            let (total, backlogOverflow) = withCache.addingReportingOverflow(
                backlogInputCount)
            return baseOverflow || cacheOverflow || backlogOverflow ? Int.max : total
        }

        var hasPendingPrefillForCostAccounting: Bool {
            Qwen4ExpMTPPriming.isPrefillCost(
                cacheTokens: cacheOffset, backlogTokens: backlogInputCount)
        }

        var stagedInputCount: Int {
            guard roundInFlight else { return 0 }
            return max(0, cacheOffset - roundValidHistoryOffset)
        }

        /// `CBv2LayerCache.innerState()` includes Qwen4 index sidecars only
        /// for multi-row caches. The Lightning assistant is permanently B1,
        /// so its evaluation fence and live-byte accounting must name them
        /// explicitly before a round can finalize or roll back.
        var qwen4Sidecars: [MLXArray] {
            caches.compactMap { $0 as? CBv2LayerCache }.flatMap { layer in
                [layer.qwen4IndexKeys, layer.qwen4IndexPositionIds,
                 layer.qwen4PooledIndexKeys].compactMap { $0 }
            }
        }

        var materializedBytes: Int {
            let arrays =
                caches.flatMap { $0.innerState() }
                + qwen4Sidecars
                + backlogHidden + backlogTokens
                + [targetHiddenFrontier].compactMap { $0 }
                + roundTrustedHidden + roundTrustedTokens + roundRoots
            return arrays.reduce(0) { total, array in
                let (next, overflow) = total.addingReportingOverflow(array.nbytes)
                return overflow ? Int.max : next
            }
        }

        init(owner: ObjectIdentifier, caches: [any KVCache],
             coldPromptReplayEligible: Bool) {
            self.owner = owner
            self.caches = caches
            self.coldPromptReplayEligible = coldPromptReplayEligible
        }

        func clearRound() {
            roundBaseOffset = cacheOffset
            roundValidHistoryOffset = cacheOffset
            roundDraftSteps = 0
            roundSkippedColdInputCount = 0
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
            roundSkippedColdInputCount = 0
            logicalInputBase = 0
            coldPromptReplayEligible = false
            roundInFlight = false
            isReleased = true
        }
    }

    private final class UnusedPreparedCapture: CBv2MTPPreparedCapture {}

    public var mtpTargetIdentity: ObjectIdentifier? { targetIdentity }
    public var requiredVerificationMode: CBv2MTPVerificationMode? {
        installedVerificationMode
    }
    /// Five-step-capable Lightning chain. Production stays at the measured
    /// four-draft ceiling; `DARKBLOOM_QWEN_MTP_MAX_DRAFT=1...5` is the
    /// explicit qualification/rollback control. Verify columns past the
    /// accepted prefix are pure GPU cost on low-acceptance prose.
    static let maximumChainedDraftTokens = 5
    static let defaultMaximumDraftTokens = 4

    static func validatedMaxDraftOverride(
        environment: [String: String]
    ) -> Int? {
        guard let raw = environment["DARKBLOOM_QWEN_MTP_MAX_DRAFT"],
            let value = Int(raw.trimmingCharacters(in: .whitespacesAndNewlines)),
            (1 ... maximumChainedDraftTokens).contains(value)
        else { return nil }
        return value
    }

    static let maxDraftOverride: Int? = {
        validatedMaxDraftOverride(environment: ProcessInfo.processInfo.environment)
    }()

    static func resolvedMaximumDraftTokens(
        forceDoubleForward: Bool, maxDraftOverride: Int?
    ) -> Int {
        forceDoubleForward ? 1 : (maxDraftOverride ?? defaultMaximumDraftTokens)
    }

    public var maximumDraftTokens: Int? {
        Self.resolvedMaximumDraftTokens(
            forceDoubleForward: Self.forceDoubleForward,
            maxDraftOverride: Self.maxDraftOverride)
    }
    public var maximumSpeculativeBatch: Int? { 1 }
    public var supportsTargetPrefixAcceptance: Bool { true }

    static let forceSerialVerification: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN_MTP_SERIAL"]
        else { return false }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()
    static let forceDoubleForward: Bool = {
        guard
            let raw = ProcessInfo.processInfo.environment[
                "DARKBLOOM_QWEN_MTP_DOUBLE_FORWARD"]
        else { return false }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()
    static let shortlistSize: Int? = {
        guard
            let raw = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN_MTP_SHORTLIST"],
            let value = Int(raw), value > 0
        else { return nil }
        return value
    }()

    public var draftShortlistSize: Int? {
        Self.forceDoubleForward ? nil : Self.shortlistSize
    }
    public var requestStateBytesPerToken: Int {
        Self.stateBytesPerToken(
            configuration: target.configuration,
            layerCount: mtp.layers.count,
            cacheElementBytes: mtp.hyperConnectionMixer.hcNorm.weight.dtype.size,
            hiddenElementBytes: mtp.hyperConnectionMixer.hcNorm.weight.dtype.size)
    }
    public var requestStateTokenGranularity: Int { Self.cacheAllocationStep }
    public var requestStateTokenAllocationPadding: Int {
        Self.maximumChainedDraftTokens
    }

    static func cacheBytesPerToken(
        configuration: Qwen4ExpTextConfiguration,
        layerCount: Int,
        elementBytes: Int
    ) -> Int {
        let (headsByDimension, geometryOverflow) = configuration.kvHeads
            .multipliedReportingOverflow(by: configuration.headDim)
        let (kvElements, kvOverflow) = headsByDimension.multipliedReportingOverflow(by: 2)
        let (layerElements, layerOverflow) = kvElements.multipliedReportingOverflow(
            by: layerCount)
        let (bytes, byteOverflow) = layerElements.multipliedReportingOverflow(
            by: elementBytes)
        guard !geometryOverflow, !kvOverflow, !layerOverflow, !byteOverflow else {
            return Int.max
        }
        let sidecar = Qwen4ExpPrefillMemory.qsaSidecarBytesPerToken(
            indexerHeadDim: configuration.indexerHeadDim,
            compressRatio: configuration.indexerCompressRatio)
        let (withSidecar, sidecarOverflow) = bytes.addingReportingOverflow(
            sidecar * layerCount)
        return sidecarOverflow ? Int.max : withSidecar
    }

    static func stateBytesPerToken(
        configuration: Qwen4ExpTextConfiguration,
        layerCount: Int,
        cacheElementBytes: Int,
        hiddenElementBytes: Int
    ) -> Int {
        let cacheBytes = cacheBytesPerToken(
            configuration: configuration, layerCount: layerCount,
            elementBytes: cacheElementBytes)
        let residualWidth = configuration.hcCount * configuration.hiddenSize
        let (hiddenBytes, hiddenOverflow) = residualWidth
            .multipliedReportingOverflow(by: hiddenElementBytes)
        guard cacheBytes != Int.max, !hiddenOverflow else { return Int.max }
        let (withHidden, hiddenAdditionOverflow) = cacheBytes.addingReportingOverflow(
            hiddenBytes)
        let (withToken, tokenAdditionOverflow) = withHidden.addingReportingOverflow(
            MemoryLayout<Int32>.stride)
        return hiddenAdditionOverflow || tokenAdditionOverflow ? Int.max : withToken
    }

    public func makeRequestState() -> any CBv2MTPRequestState {
        RequestState(
            owner: ObjectIdentifier(self), caches: makeCache(),
            coldPromptReplayEligible: skipColdPromptReplay)
    }

    public func snapshotRequestState(
        _ requestState: any CBv2MTPRequestState
    ) throws -> Qwen4ExpMTPStateSnapshot {
        guard let state = requestState as? RequestState else {
            throw Qwen4ExpMTPStateError.incompatible(
                "Qwen4 Lightning received foreign request state")
        }
        guard !state.isReleased, !state.roundInFlight else {
            throw Qwen4ExpMTPStateError.lifecycle(
                "Qwen4 Lightning snapshot requires settled live state")
        }
        guard state.backlogHidden.count == state.backlogTokens.count else {
            throw Qwen4ExpMTPStateError.lifecycle(
                "Qwen4 Lightning backlog arrays diverged")
        }
        var layers: [Qwen4ExpMTPSequenceSnapshot] = []
        layers.reserveCapacity(state.caches.count)
        for (index, cache) in state.caches.enumerated() {
            guard let layer = cache as? CBv2LayerCache, layer.rows.count == 1,
                let row = layer.rows[0] as? CBv2Qwen4IndexerRow
            else {
                throw Qwen4ExpMTPStateError.incompatible(
                    "Qwen4 Lightning cache layer \(index) is not snapshot-capable")
            }
            CBv2Qwen4IndexerBind.harvest(layer, into: row)
            let kv = row.snapshot()
            let qsa = kv.offset > 0 ? try row.snapshotQwen4Indexer() : nil
            layers.append(
                Qwen4ExpMTPSequenceSnapshot(
                    layerIndex: index,
                    keys: kv.keys,
                    values: kv.values,
                    offset: kv.offset,
                    qwen4Indexer: qsa))
        }
        try validateBacklog(
            hidden: state.backlogHidden,
            tokens: state.backlogTokens,
            frontier: state.targetHiddenFrontier)
        let snapshot = Qwen4ExpMTPStateSnapshot(
            cacheLayers: layers,
            backlogHidden: state.backlogHidden,
            backlogTokens: state.backlogTokens,
            targetHiddenFrontier: state.targetHiddenFrontier,
            committedInputCount: state.committedInputCount,
            logicalInputBase: state.logicalInputBase)
        guard snapshot.committedInputCount >= 0 else {
            throw Qwen4ExpMTPStateError.lifecycle(
                "Qwen4 Lightning committed frontier overflowed")
        }
        return snapshot
    }

    public func restoreRequestState(
        from snapshot: Qwen4ExpMTPStateSnapshot
    ) throws -> any CBv2MTPRequestState {
        guard snapshot.cacheLayers.count == mtp.layers.count,
            snapshot.cacheLayers.enumerated().allSatisfy({ $0.offset == $0.element.layerIndex })
        else {
            throw Qwen4ExpMTPStateError.incompatible(
                "Qwen4 Lightning cache layer layout changed")
        }
        try validateBacklog(
            hidden: snapshot.backlogHidden,
            tokens: snapshot.backlogTokens,
            frontier: snapshot.targetHiddenFrontier)

        let caches = makeCache()
        var commonOffset: Int?
        for (layerSnapshot, cache) in zip(snapshot.cacheLayers, caches) {
            guard layerSnapshot.offset >= 0,
                layerSnapshot.offset <= target.configuration.maxPositionEmbeddings,
                layerSnapshot.keys.ndim == 4,
                layerSnapshot.values.ndim == 4,
                layerSnapshot.keys.shape == layerSnapshot.values.shape,
                layerSnapshot.keys.dim(0) == 1,
                layerSnapshot.keys.dim(1) == target.configuration.kvHeads,
                layerSnapshot.keys.dim(2) == layerSnapshot.offset,
                layerSnapshot.keys.dim(3) == target.configuration.headDim,
                let layer = cache as? CBv2LayerCache,
                layer.rows.count == 1,
                let row = layer.rows[0] as? CBv2Qwen4IndexerRow
            else {
                throw Qwen4ExpMTPStateError.incompatible(
                    "Qwen4 Lightning cache tensor geometry changed")
            }
            if let existing = commonOffset, existing != layerSnapshot.offset {
                throw Qwen4ExpMTPStateError.incompatible(
                    "Qwen4 Lightning cache offsets diverged")
            }
            commonOffset = layerSnapshot.offset
            if layerSnapshot.offset > 0 {
                guard let qsa = layerSnapshot.qwen4Indexer else {
                    throw Qwen4ExpMTPStateError.incompatible(
                        "Qwen4 Lightning QSA side-state is missing")
                }
                _ = row.update(keys: layerSnapshot.keys, values: layerSnapshot.values)
                // Direct row rehydration advances the row's host offset but
                // not the layer cache's device-side RoPE offset. Recompose
                // while the QSA sidecar is still empty, then install it.
                layer.setRows(layer.rows)
                try row.restoreQwen4Indexer(qsa)
                guard CBv2Qwen4IndexerBind.restore(layer, from: row) else {
                    throw Qwen4ExpMTPStateError.incompatible(
                        "Qwen4 Lightning QSA side-state could not be rebound")
                }
            } else if layerSnapshot.qwen4Indexer != nil {
                throw Qwen4ExpMTPStateError.incompatible(
                    "Qwen4 Lightning empty cache has QSA side-state")
            }
        }
        let state = RequestState(
            owner: ObjectIdentifier(self), caches: caches,
            coldPromptReplayEligible: skipUnprimedRestoredReplay
                && snapshot.logicalInputBase == 0 && commonOffset == 0)
        state.backlogHidden = snapshot.backlogHidden
        state.backlogTokens = snapshot.backlogTokens
        state.targetHiddenFrontier = snapshot.targetHiddenFrontier
        guard snapshot.logicalInputBase >= 0,
            snapshot.logicalInputBase <= snapshot.committedInputCount,
            snapshot.committedInputCount <= target.configuration.maxPositionEmbeddings
        else {
            throw Qwen4ExpMTPStateError.incompatible(
                "Qwen4 Lightning logical input frontier is invalid")
        }
        state.logicalInputBase = snapshot.logicalInputBase
        guard state.committedInputCount == snapshot.committedInputCount else {
            throw Qwen4ExpMTPStateError.incompatible(
                "Qwen4 Lightning committed input frontier changed")
        }
        return state
    }

    private func validateBacklog(
        hidden: [MLXArray],
        tokens: [MLXArray],
        frontier: MLXArray?
    ) throws {
        let width = target.configuration.hcCount * target.configuration.hiddenSize
        guard hidden.count == tokens.count else {
            throw Qwen4ExpMTPStateError.incompatible(
                "Qwen4 Lightning backlog pair count changed")
        }
        for (hiddenRows, tokenRows) in zip(hidden, tokens) {
            guard hiddenRows.ndim == 3, tokenRows.ndim == 2,
                hiddenRows.dim(0) == 1, tokenRows.dim(0) == 1,
                hiddenRows.dim(1) == tokenRows.dim(1),
                hiddenRows.dim(1) > 0,
                hiddenRows.dim(2) == width
            else {
                throw Qwen4ExpMTPStateError.incompatible(
                    "Qwen4 Lightning backlog tensor geometry changed")
            }
        }
        if let frontier {
            guard frontier.shape == [1, 1, width] else {
                throw Qwen4ExpMTPStateError.incompatible(
                    "Qwen4 Lightning hidden frontier geometry changed")
            }
        }
    }

    public func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState
    ) {
        guard let state = requestState as? RequestState else {
            preconditionFailure("Lightning MTP received foreign request state")
        }
        precondition(!state.isReleased, "Lightning MTP observed released request state")
        precondition(!state.roundInFlight, "Lightning MTP observed target during a round")
        precondition(
            observation.tokens.ndim == 2 && observation.hidden.ndim == 3
                && observation.tokens.dim(0) == 1 && observation.hidden.dim(0) == 1
                && observation.tokens.dim(1) == observation.hidden.dim(1),
            "Lightning MTP target observation shape mismatch")
        precondition(
            observation.hidden.dim(-1) == target.configuration.hcCount
                * target.configuration.hiddenSize,
            "Lightning MTP target hidden must be the pre-mixer residual")

        let count = observation.tokens.dim(1)
        guard count > 0 else { return }

        if let frontier = state.targetHiddenFrontier {
            state.backlogHidden.append(frontier)
            state.backlogTokens.append(observation.tokens[0..., 0 ..< 1])
        }
        if count > 1 {
            state.backlogHidden.append(observation.hidden[0..., 0 ..< count - 1, 0...])
            state.backlogTokens.append(observation.tokens[0..., 1 ..< count])
        }
        state.targetHiddenFrontier =
            observation.hidden[0..., (count - 1) ..< count, 0...]
    }

    public func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        UnusedPreparedCapture()
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure("Lightning MTP requires request-owned assistant state")
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        guard let state = requestState as? RequestState else {
            preconditionFailure("Lightning MTP received foreign request state")
        }
        precondition(!state.isReleased, "Lightning MTP drafted with released request state")
        precondition(
            tokens.ndim == 2 && hidden.ndim == 3
                && tokens.dim(0) == 1 && tokens.dim(1) == 1
                && hidden.dim(0) == 1 && hidden.dim(1) == 1,
            "Lightning MTP draft input shape mismatch")

        if Self.forceDoubleForward {
            return legacyDoubleForwardDraftStep(tokens: tokens, hidden: hidden, state: state)
        }

        let isFirstStep = !state.roundInFlight
        let output: (mixed: MLXArray, residual: MLXArray)
        if isFirstStep {
            let skipsColdHistory = shouldSkipColdPromptReplay(state)
            if !skipsColdHistory && Qwen4ExpMTPPriming.needsChunking(
                backlog: state.backlogTokens, chunkTokens: primeChunkTokens)
            {
                prepareRound(
                    tokens: tokens, hidden: hidden, state: state,
                    skippedColdInputCount: 0)
                output = Qwen4ExpMTPPriming.forward(
                    hidden: state.roundTrustedHidden, tokens: state.roundTrustedTokens,
                    chunkTokens: primeChunkTokens,
                    forward: { hidden, tokens in
                        self.moduleForward(hidden: hidden, tokens: tokens, cache: state.caches)
                    },
                    materialize: { output in
                        // A chunk boundary must drain the graph, including the
                        // sidecars omitted by CBv2LayerCache.innerState(). Original
                        // trusted observations stay owned until round settlement.
                        CBv2DeferredHostFill.resolveBeforeEvaluation()
                        eval([output.mixed, output.residual]
                            + state.caches.flatMap { $0.innerState() }
                            + state.qwen4Sidecars)
                    })
            } else {
                let feed = beginRound(
                    tokens: tokens, hidden: hidden, state: state,
                    skipColdPromptReplay: skipsColdHistory)
                output = moduleForward(
                    hidden: feed.hidden, tokens: feed.tokens, cache: state.caches)
            }
        } else {
            precondition(
                state.roundDraftSteps < Self.maximumChainedDraftTokens,
                "Lightning MTP exceeded its five-step draft chain")
            output = moduleForward(hidden: hidden, tokens: tokens, cache: state.caches)
        }
        let lastResidual = output.residual[0..., (output.residual.dim(1) - 1)..., 0...]
        let lastMixed = output.mixed[0..., (output.mixed.dim(1) - 1)..., 0...]
        let draft = draftToken(hidden: lastMixed, shortlist: shortlist)
        state.roundRoots.append(contentsOf: [lastResidual, lastMixed, draft])
        state.roundDraftSteps += 1

        if isFirstStep {
            state.roundValidHistoryOffset = state.cacheOffset
        }
        return (draft, lastResidual)
    }

    private func beginRound(
        tokens: MLXArray, hidden: MLXArray, state: RequestState,
        skipColdPromptReplay: Bool
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        let skipped = skipColdPromptReplay ? state.backlogInputCount : 0
        prepareRound(
            tokens: tokens, hidden: hidden, state: state,
            skippedColdInputCount: skipped)
        // Match the unprimed cold behavior: seed the empty assistant head with
        // only the current trusted carry. The full backlog remains round-owned
        // until commit so discard can restore it byte-for-byte.
        if skipped > 0 { return (tokens, hidden) }
        if state.roundTrustedTokens.count == 1 {
            return (state.roundTrustedTokens[0], state.roundTrustedHidden[0])
        }
        let feedTokens = concatenated(state.roundTrustedTokens, axis: 1)
        let feedHidden = concatenated(state.roundTrustedHidden, axis: 1)
        return (feedTokens, feedHidden)
    }

    private func prepareRound(
        tokens: MLXArray, hidden: MLXArray, state: RequestState,
        skippedColdInputCount: Int
    ) {
        precondition(!state.roundInFlight, "Lightning MTP round already in flight")
        precondition(
            state.backlogHidden.count == state.backlogTokens.count,
            "Lightning MTP trusted backlog diverged")

        state.roundBaseOffset = state.cacheOffset
        state.roundValidHistoryOffset = state.cacheOffset
        state.roundDraftSteps = 0
        state.roundSkippedColdInputCount = skippedColdInputCount
        state.roundInFlight = true
        state.roundTrustedHidden = state.backlogHidden
        state.roundTrustedTokens = state.backlogTokens
        state.backlogHidden.removeAll(keepingCapacity: true)
        state.backlogTokens.removeAll(keepingCapacity: true)

        state.roundTrustedHidden.append(hidden)
        state.roundTrustedTokens.append(tokens)
        state.targetHiddenFrontier = nil
    }

    private func shouldSkipColdPromptReplay(_ state: RequestState) -> Bool {
        skipColdPromptReplay
            && state.coldPromptReplayEligible
            && state.cacheOffset == 0
            && state.backlogInputCount > 0
    }

    private func draftToken(hidden: MLXArray, shortlist: MLXArray?) -> MLXArray {
        if let shortlist {
            let logits = shortlistLogits(hidden: hidden, ids: shortlist)
            return shortlist[argMax(logits[0..., -1, 0...], axis: -1)]
                .asType(.int32)
        }
        return argMax(headLogits(hidden)[0..., -1, 0...], axis: -1)
            .asType(.int32)
    }

    private func legacyDoubleForwardDraftStep(
        tokens: MLXArray, hidden: MLXArray, state: RequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        precondition(!state.roundInFlight, "Lightning MTP round already in flight")
        let feed = beginRound(
            tokens: tokens, hidden: hidden, state: state,
            skipColdPromptReplay: shouldSkipColdPromptReplay(state))
        let output = moduleForward(
            hidden: feed.hidden, tokens: feed.tokens, cache: state.caches)
        let lastResidual = output.residual[0..., (output.residual.dim(1) - 1)..., 0...]
        let lastMixed = output.mixed[0..., (output.mixed.dim(1) - 1)..., 0...]
        let draft = argMax(headLogits(lastMixed)[0..., -1, 0...], axis: -1).asType(.int32)
        state.roundRoots.append(contentsOf: [lastResidual, lastMixed, draft])
        state.roundDraftSteps = 1
        state.roundValidHistoryOffset = state.cacheOffset

        eval([draft, lastResidual] + state.caches.flatMap { $0.innerState() }
            + state.qwen4Sidecars)
        _ = forward(
            hidden: lastResidual, tokens: draft.reshaped([1, 1]), cache: state.caches)
        return (draft, lastResidual)
    }

    public func evaluationTargets(
        for requestState: any CBv2MTPRequestState
    ) -> [MLXArray] {
        guard let state = requestState as? RequestState, !state.isReleased else {
            return []
        }
        return state.caches.flatMap { $0.innerState() } + state.qwen4Sidecars
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
            preconditionFailure("Lightning MTP received foreign request state")
        }
        precondition(!state.isReleased, "Lightning MTP finalized released request state")
        precondition(state.roundInFlight, "Lightning MTP finalized without a round")
        precondition(
            (0 ... state.roundDraftSteps + 1).contains(confirmedInputTokens),
            "Lightning MTP confirmed prefix exceeds the draft round")
        precondition(
            committedDraftTokens.ndim == 2 && committedTargetHidden.ndim == 3
                && committedDraftTokens.dim(0) == 1
                && committedTargetHidden.dim(0) == 1
                && committedDraftTokens.dim(1) == committedTargetHidden.dim(1),
            "Lightning MTP committed target rows mismatch")
        let committedDraftCount = committedDraftTokens.dim(1)
        precondition(
            committedDraftCount <= state.roundDraftSteps
                && committedDraftCount <= max(0, confirmedInputTokens - 1),
            "Lightning MTP committed drafts exceed confirmed target inputs")

        let (logicalInputBase, overflow) = state.logicalInputBase
            .addingReportingOverflow(state.roundSkippedColdInputCount)
        precondition(!overflow, "Lightning MTP logical input frontier overflowed")
        trim(state: state, to: state.roundValidHistoryOffset)
        if committedDraftCount > 0 {
            state.backlogTokens.append(committedDraftTokens)
            state.backlogHidden.append(committedTargetHidden)
        }
        state.coldPromptReplayEligible = false
        state.clearRound()
        state.logicalInputBase = logicalInputBase
    }

    public func discardRound(requestState: any CBv2MTPRequestState) {
        guard let state = requestState as? RequestState,
            !state.isReleased, state.roundInFlight
        else { return }

        trim(state: state, to: state.roundBaseOffset)
        state.backlogHidden =
            state.roundTrustedHidden + state.backlogHidden
        state.backlogTokens =
            state.roundTrustedTokens + state.backlogTokens
        state.clearRound()
    }

    private func trim(state: RequestState, to offset: Int) {
        let rollback = state.cacheOffset - offset
        precondition(rollback >= 0, "Lightning MTP cache checkpoint moved forward")
        guard rollback > 0 else { return }
        for cache in state.caches {
            guard let layer = cache as? CBv2LayerCache else {
                preconditionFailure("Lightning MTP trim requires CBv2LayerCache")
            }
            let keys = layer.qwen4IndexKeys
            let positions = layer.qwen4IndexPositionIds
            for row in layer.rows {
                row.rollback(rollback)
            }
            layer.setRows(layer.rows)
            if offset > 0 {
                layer.qwen4IndexKeys = Qwen4ExpIndexCapacity.logicalTokens(
                    keys, length: offset)
                layer.qwen4IndexPositionIds = Qwen4ExpIndexCapacity.logicalPositions(
                    positions, length: offset)
            }
        }
    }

    public func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        guard let state = requestState as? RequestState, !state.isReleased else {
            return
        }
        state.clearAll()
    }
}
