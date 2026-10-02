// Copyright © 2026 Eigen Labs.
// Strict in-memory loading for the authenticated native MiMo converted layout.
// No filesystem payload discovery, factory alias or quantization recipe change.
import CryptoKit
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public enum MiMoV26ConvertedLoadError: Error, Equatable, Sendable {
    case invalidConfiguration(String)
    case invalidInventory(String)
    case invalidTensor(String)
    case invalidQuantization(String)
    case incompatibleModule(String)
    case invalidProvenance(String)
}

public enum MiMoV26ConvertedScalarType: String, Codable, Sendable {
    case bfloat16 = "BF16"
    case float16 = "F16"
    case float32 = "F32"
    case uint32 = "U32"
    case uint8 = "U8"
    var dtype: DType {
        switch self {
        case .bfloat16: .bfloat16
        case .float16: .float16
        case .float32: .float32
        case .uint32: .uint32
        case .uint8: .uint8
        }
    }
    var bytes: Int { self == .uint8 ? 1 : (self == .float32 || self == .uint32 ? 4 : 2) }
}

/// Describes a caller-owned tensor/header. It is not a payload-hash assertion.
public struct MiMoV26ConvertedTensorDescriptor: Codable, Equatable, Sendable {
    public let shape: [Int]
    public let dtype: MiMoV26ConvertedScalarType
    public let file: String
    public init(shape: [Int], dtype: MiMoV26ConvertedScalarType, file: String) {
        self.shape = shape
        self.dtype = dtype
        self.file = file
    }
}

/// Descriptive provenance supplied by the owner. Hash-shaped receipt fields
/// bind references only; this seam does not independently verify their content.
public struct MiMoV26ConvertedProvenance: Equatable, Sendable {
    public enum Layout: String, Codable, Sendable { case nativeConversion, mlxVLM }
    public let layout: Layout
    public let artifactID, sourceRepository, sourceRevision: String
    public let conversionManifestSHA256: String
    public let payloadVerificationReceiptSHA256: String?
    public init(
        artifactID: String, sourceRepository: String, sourceRevision: String,
        conversionManifestSHA256: String, payloadVerificationReceiptSHA256: String? = nil,
        layout: Layout = .nativeConversion
    ) {
        self.layout = layout
        self.artifactID = artifactID
        self.sourceRepository = sourceRepository
        self.sourceRevision = sourceRevision
        self.conversionManifestSHA256 = conversionManifestSHA256
        self.payloadVerificationReceiptSHA256 = payloadVerificationReceiptSHA256
    }
}

public enum MiMoV26ConvertedComponent: String, Codable, CaseIterable, Sendable {
    case target, vision, audioPatch, mtp
}

func convertedHash(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}
func convertedHex(_ value: String, count: Int) -> Bool {
    value.utf8.count == count
        && value.utf8.allSatisfy { (48 ... 57).contains($0) || (97 ... 102).contains($0) }
}
func convertedProduct(_ values: [Int], _ name: String) throws -> Int {
    var n = 1
    for value in values {
        let next = n.multipliedReportingOverflow(by: value)
        guard value > 0 && !next.overflow else {
            throw MiMoV26ConvertedLoadError.invalidInventory("overflow/nonpositive " + name)
        }
        n = next.partialValue
    }
    return n
}

struct MiMoV26ConvertedIndex: Decodable {
    struct Metadata: Decodable {
        let totalSize: Int
        enum CodingKeys: String, CodingKey { case totalSize = "total_size" }
    }
    let metadata: Metadata
    let weightMap: [String: String]
    enum CodingKeys: String, CodingKey {
        case metadata
        case weightMap = "weight_map"
    }
}

/// Complete root-bundle plan, made without constructing MLX modules/arrays.
/// Source-layout metadata remains unchanged; descriptors must prove split Q/K/V.
public struct MiMoV26ConvertedLoadPlan: Sendable {
    public let configuration: MiMoV26Configuration
    public let descriptors: [String: MiMoV26ConvertedTensorDescriptor]
    public let components: [MiMoV26ConvertedComponent: Set<String>]
    public let targetExpertModulePaths: Set<String>
    /// Original file names stay in descriptors/components for source integrity.
    /// Only this validated bijection renames handles at component installation.
    public let parameterNames: [String: String]
    let targetQuantizationPolicies: [String: MiMoV26Quantization.Policy]
    let audioQuantizationPolicies: [String: MiMoV26Quantization.Policy]
    let floatingMTP: Bool
    public let configSHA256, indexSHA256, descriptorSHA256: String
    public let rootFiles: Set<String>
    public let tensorBytes: Int
    public let provenance: MiMoV26ConvertedProvenance
    /// Separately packaged trained components are not claimed loaded by this seam.
    public let externalRequiredComponents: [String]
    let floatingType: MiMoV26ConvertedScalarType

    /// Payload already included in tensorBytes, not additional load/serving
    /// credit. Inventory products and the total were overflow-checked by make.
    public func tensorBytes(for component: MiMoV26ConvertedComponent) -> Int {
        (components[component] ?? []).reduce(0) { total, key in
            let descriptor = descriptors[key]!
            return total + descriptor.shape.reduce(descriptor.dtype.bytes, *)
        }
    }

    public static func make(
        configurationData: Data, indexData: Data,
        descriptors: [String: MiMoV26ConvertedTensorDescriptor],
        provenance: MiMoV26ConvertedProvenance
    ) throws -> Self {
        if provenance.layout == .mlxVLM {
            return try MiMoV26MLXVLMPlan.make(
                configurationData: configurationData,
                indexData: indexData, descriptors: descriptors, provenance: provenance)
        }
        let c = try JSONDecoder().decode(MiMoV26Configuration.self, from: configurationData)
        let index = try JSONDecoder().decode(MiMoV26ConvertedIndex.self, from: indexData)
        guard !provenance.artifactID.isEmpty,
            provenance.sourceRepository == "XiaomiMiMo/MiMo-V2.6-Flash-RL",
            convertedHex(provenance.sourceRevision, count: 40),
            convertedHex(provenance.conversionManifestSHA256, count: 64),
            provenance.payloadVerificationReceiptSHA256.map({ convertedHex($0, count: 64) }) ?? true
        else {
            throw MiMoV26ConvertedLoadError.invalidProvenance("artifact/source/receipt identity")
        }
        guard c.numHiddenLayers <= 48, c.hiddenSize <= 4096, c.intermediateSize <= 16384,
            c.moeIntermediateSize <= 2048, c.routedExpertCount <= 256, c.vocabularySize <= 152576,
            c.numNextnPredictLayers == 3, !c.tieWordEmbeddings, !c.attentionBias,
            c.sharedExpertCount == nil || c.sharedExpertCount == 0,
            c.attentionDropout == 0, let vision = c.vision, let audio = c.audio,
            vision.depth <= 28, vision.hiddenSize <= 1280, vision.intermediateSize <= 4608,
            let mtp = c.embeddedMTP, mtp.numLayers == 3,
            let policy = c.quantization.nativeDefault, policy.mode == "mxfp4",
            policy.bits == 4, policy.groupSize == 32
        else {
            throw MiMoV26ConvertedLoadError.invalidConfiguration(
                "requires complete converted target/vision/audio/MTP geometry")
        }
        let dtype: MiMoV26ConvertedScalarType =
            c.dtype == "bfloat16" ? .bfloat16 : (c.dtype == "float16" ? .float16 : .float32)
        var expected: [String: (shape: [Int], dtype: MiMoV26ConvertedScalarType)] = [:]
        var groups = Dictionary(
            uniqueKeysWithValues: MiMoV26ConvertedComponent.allCases.map { ($0, Set<String>()) })
        var experts = Set<String>()
        var mtpProjections = Set<String>()
        func add(
            _ key: String, _ shape: [Int], _ type: MiMoV26ConvertedScalarType,
            _ component: MiMoV26ConvertedComponent
        ) throws {
            guard expected[key] == nil, shape.allSatisfy({ $0 > 0 && $0 <= Int(Int32.max) }) else {
                throw MiMoV26ConvertedLoadError.invalidInventory("duplicate/invalid shape: " + key)
            }
            _ = try convertedProduct(shape + [type.bytes], key)
            expected[key] = (shape, type)
            groups[component, default: []].insert(key)
        }
        try add("model.embed_tokens.weight", [c.vocabularySize, c.hiddenSize], dtype, .target)
        try add("model.norm.weight", [c.hiddenSize], dtype, .target)
        try add("lm_head.weight", [c.vocabularySize, c.hiddenSize], dtype, .target)
        for layer in 0 ..< c.numHiddenLayers {
            let p = "model.layers.\(layer)."
            let g = try c.attentionGeometry(at: layer)
            let q = try convertedProduct([g.queryHeads, g.headDim], "Q width")
            let k = try convertedProduct([g.keyValueHeads, g.headDim], "K width")
            let v = try convertedProduct([g.keyValueHeads, g.valueHeadDim], "V width")
            let out = try convertedProduct([g.queryHeads, g.valueHeadDim], "O width")
            for name in ["input_layernorm", "post_attention_layernorm"] {
                try add(p + name + ".weight", [c.hiddenSize], dtype, .target)
            }
            for (name, width) in [("q_proj", q), ("k_proj", k), ("v_proj", v)] {
                try add(p + "self_attn." + name + ".weight", [width, c.hiddenSize], dtype, .target)
            }
            try add(p + "self_attn.o_proj.weight", [c.hiddenSize, out], dtype, .target)
            if g.hasSinks {
                try add(p + "self_attn.attention_sink_bias", [g.queryHeads], dtype, .target)
            }
            if c.moeLayerFrequency[layer] == 1 {
                try add(
                    p + "mlp.gate.weight", [c.routedExpertCount, c.hiddenSize],
                    c.moeRouterDType == "bfloat16" ? .bfloat16 : .float32, .target)
                try add(
                    p + "mlp.gate.e_score_correction_bias", [c.routedExpertCount], .float32, .target
                )
                for name in ["gate_proj", "up_proj", "down_proj"] {
                    let rows = name == "down_proj" ? c.hiddenSize : c.moeIntermediateSize
                    let cols = name == "down_proj" ? c.moeIntermediateSize : c.hiddenSize
                    guard cols.isMultiple(of: 32) else {
                        throw MiMoV26ConvertedLoadError.invalidQuantization("MXFP4 dimensions")
                    }
                    let module = p + "mlp.switch_mlp." + name
                    experts.insert(module)
                    try add(
                        module + ".weight", [c.routedExpertCount, rows, cols / 8], .uint32, .target)
                    try add(
                        module + ".scales", [c.routedExpertCount, rows, cols / 32], .uint8, .target)
                }
            } else {
                for name in ["gate_proj", "up_proj"] {
                    try add(
                        p + "mlp." + name + ".weight", [c.intermediateSize, c.hiddenSize], dtype,
                        .target)
                }
                try add(
                    p + "mlp.down_proj.weight", [c.hiddenSize, c.intermediateSize], dtype, .target)
            }
        }
        for (key, shape) in try MiMoV26VisionTower.expectedTensorShapes(configuration: vision) {
            try add(key, shape, dtype, .vision)
        }
        for (key, shape) in try MiMoV26AudioPatchEncoder.expectedTensorShapes(configuration: audio)
        { try add(key, shape, dtype, .audioPatch) }
        let g = c.slidingAttention
        for depth in 0 ..< 3 {
            let p = "mtp.layers.\(depth)."
            for norm in [
                "enorm", "hnorm", "input_layernorm", "pre_mlp_layernorm", "final_layernorm",
            ] { try add(p + norm + ".weight", [c.hiddenSize], dtype, .mtp) }
            if g.hasSinks {
                try add(p + "self_attn.attention_sink_bias", [g.queryHeads], dtype, .mtp)
            }
            let matrices = [
                (
                    "eh_proj", c.hiddenSize,
                    try convertedProduct([2, c.hiddenSize], "MTP concatenation")
                ),
                (
                    "self_attn.q_proj", try convertedProduct([g.queryHeads, g.headDim], "MTP Q"),
                    c.hiddenSize
                ),
                (
                    "self_attn.k_proj",
                    try convertedProduct([g.keyValueHeads, g.headDim], "MTP K"), c.hiddenSize
                ),
                (
                    "self_attn.v_proj",
                    try convertedProduct([g.keyValueHeads, g.valueHeadDim], "MTP V"), c.hiddenSize
                ),
                (
                    "self_attn.o_proj", c.hiddenSize,
                    try convertedProduct([g.queryHeads, g.valueHeadDim], "MTP O")
                ),
                ("mlp.gate_proj", c.intermediateSize, c.hiddenSize),
                ("mlp.up_proj", c.intermediateSize, c.hiddenSize),
                ("mlp.down_proj", c.hiddenSize, c.intermediateSize),
            ]
            for (name, rows, cols) in matrices {
                let module = p + name
                mtpProjections.insert(module)
                guard cols.isMultiple(of: 64) else {
                    throw MiMoV26ConvertedLoadError.invalidQuantization("affine4/64 dimensions")
                }
                let a = c.quantization.nativeOverrides[module]
                let b = c.quantization.nativeOverrides["language_model." + module]
                if let a, let b, a != b {
                    throw MiMoV26ConvertedLoadError.invalidQuantization(
                        "conflicting MTP aliases: " + module)
                }
                guard let resolved = a ?? b, case .quantize(let policy) = resolved,
                    policy.mode == "affine", policy.bits == 4, policy.groupSize == 64
                else {
                    throw MiMoV26ConvertedLoadError.invalidQuantization(
                        "missing/wrong explicit MTP policy: " + module)
                }
                try add(module + ".weight", [rows, cols / 8], .uint32, .mtp)
                for suffix in ["scales", "biases"] {
                    try add(module + "." + suffix, [rows, cols / 64], dtype, .mtp)
                }
            }
        }
        // Only aliases of an actual MTP projection or exact target expert path
        // may override quantization. Dense modules never inherit root MXFP4.
        for (path, override) in c.quantization.nativeOverrides {
            let canonical =
                path.hasPrefix("language_model.mtp.")
                ? String(path.dropFirst("language_model.".count)) : path
            if mtpProjections.contains(canonical) { continue }
            guard experts.contains(path), case .quantize(let policy) = override,
                policy.mode == "mxfp4", policy.bits == 4, policy.groupSize == 32
            else {
                throw MiMoV26ConvertedLoadError.invalidQuantization(
                    "unmapped/incompatible override: " + path)
            }
        }
        if case .object(let rawQuantization) = c.rawFields["quantization"] {
            let allowed = Set(["mode", "bits", "group_size"]).union(
                c.quantization.nativeOverrides.keys)
            guard Set(rawQuantization.keys) == allowed else {
                throw MiMoV26ConvertedLoadError.invalidQuantization(
                    "unmapped native quantization metadata")
            }
        }
        guard Set(descriptors.keys) == Set(expected.keys),
            Set(index.weightMap.keys) == Set(expected.keys)
        else {
            throw MiMoV26ConvertedLoadError.invalidInventory("missing/unmapped root tensors")
        }
        var total = 0
        for (key, spec) in expected {
            let actual = descriptors[key]!
            let file = actual.file
            guard actual.shape == spec.shape, actual.dtype == spec.dtype,
                file == index.weightMap[key], !file.isEmpty,
                file == (file as NSString).lastPathComponent,
                !file.contains("\\"), file.hasSuffix(".safetensors"), file != ".safetensors",
                !groups[.mtp]!.contains(key) || file == mtp.file
            else {
                throw MiMoV26ConvertedLoadError.invalidInventory(key)
            }
            let bytes = try convertedProduct(actual.shape + [actual.dtype.bytes], key)
            let sum = total.addingReportingOverflow(bytes)
            guard !sum.overflow else {
                throw MiMoV26ConvertedLoadError.invalidInventory("root byte overflow")
            }
            total = sum.partialValue
        }
        guard total == index.metadata.totalSize else {
            throw MiMoV26ConvertedLoadError.invalidInventory("index tensor byte total")
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return .init(
            configuration: c, descriptors: descriptors, components: groups,
            targetExpertModulePaths: experts,
            parameterNames: Dictionary(uniqueKeysWithValues: expected.keys.map { ($0, $0) }),
            targetQuantizationPolicies: Dictionary(
                uniqueKeysWithValues: experts.map { ($0, c.quantization.nativeDefault!) }),
            audioQuantizationPolicies: [:], floatingMTP: false,
            configSHA256: convertedHash(configurationData), indexSHA256: convertedHash(indexData),
            descriptorSHA256: convertedHash(try encoder.encode(descriptors)),
            rootFiles: Set(index.weightMap.values),
            tensorBytes: total, provenance: provenance,
            externalRequiredComponents: ["audio_tokenizer", "dflash"], floatingType: dtype)
    }
}

/// New components are published together only after complete validation/load.
/// The plan describes their loading event; callers must preserve component
/// ownership and cannot treat later arbitrary Module mutation as still verified.
public struct MiMoV26ConvertedBundle {
    public let target: MiMoV26TextModel
    public let vision: MiMoV26VisionTower
    public let audioPatch: MiMoV26AudioPatchEncoder
    public let mtp: MiMoV26MTP
    public let plan: MiMoV26ConvertedLoadPlan
}

public enum MiMoV26ConvertedWeights {
    /// Caller supplies already-read tensor handles. Metadata inspection never
    /// evaluates them; file streaming and payload digest verification are separate.
    public static func load(
        plan: MiMoV26ConvertedLoadPlan,
        tensors: [String: MLXArray]
    ) throws -> MiMoV26ConvertedBundle {
        guard Set(tensors.keys) == Set(plan.descriptors.keys) else {
            throw MiMoV26ConvertedLoadError.invalidTensor("missing/unmapped root tensor handles")
        }
        for (key, descriptor) in plan.descriptors {
            let array = tensors[key]!
            guard array.shape == descriptor.shape, array.dtype == descriptor.dtype.dtype else {
                throw MiMoV26ConvertedLoadError.invalidTensor(key)
            }
        }
        func component(_ type: MiMoV26ConvertedComponent) -> [String: MLXArray] {
            Dictionary(
                uniqueKeysWithValues: plan.components[type]!.map {
                    (plan.parameterNames[$0]!, tensors[$0]!)
                })
        }
        // Entire request validated before any model constructor/update. Fresh
        // components remain private if construction or a strict load throws.
        let target = try MiMoV26TextModel(plan.configuration)
        try MiMoV26PackedModuleLoader.prepare(target, policies: plan.targetQuantizationPolicies)
        let targetTensors = component(.target)
        let constructed = Dictionary(uniqueKeysWithValues: target.parameters().flattened())
        guard Set(constructed.keys) == Set(targetTensors.keys),
            constructed.allSatisfy({ targetTensors[$0.key]!.shape == $0.value.shape })
        else {
            throw MiMoV26ConvertedLoadError.incompatibleModule("target parameter closure")
        }
        try target.update(parameters: .unflattened(targetTensors), verify: .all)
        let vision = try MiMoV26VisionTower(configuration: plan.configuration.vision!)
        try vision.loadNativeWeights(
            component(.vision), expectedDType: plan.floatingType.dtype,
            flattenedPatchStorage: plan.provenance.layout == .mlxVLM)
        let audio = try MiMoV26AudioPatchEncoder(configuration: plan.configuration.audio!)
        if plan.audioQuantizationPolicies.isEmpty {
            try audio.loadNativeWeights(
                component(.audioPatch), expectedDType: plan.floatingType.dtype)
        } else {
            try MiMoV26PackedModuleLoader.prepare(audio, policies: plan.audioQuantizationPolicies)
            try audio.loadPackedWeights(
                component(.audioPatch), expectedDType: plan.floatingType.dtype)
        }
        let mtp = try MiMoV26MTP(target: target, floatingProjections: plan.floatingMTP)
        try mtp.loadConvertedWeights(component(.mtp))
        return .init(target: target, vision: vision, audioPatch: audio, mtp: mtp, plan: plan)
    }
}
