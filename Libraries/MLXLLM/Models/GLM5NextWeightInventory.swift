// Copyright © 2026 Eigen Labs.
// Source schema: zai-org/GLM-5.3-Flash@91b69215804f2d2036a8867f6ce44848e278dcda.
// Native artifact: Vontra/GLM-5.3-Flash-MLX-4bit-MTP@76add2a341a1cd90ad0e86bb69839ea9c35827c6.

import Foundation
import MLX
import MLXNN

public enum GLM5NextLoadError: Error, Equatable {
    case invalid(String)
    case missing(String)
    case unexpected(String)
}

/// Header-only validation. No MLX arrays are constructed by inventory validation.
public struct GLM5NextWeightInventory {
    public enum Part: String, CaseIterable, Sendable { case text, mtp, vision }
    /// `.affineMixed` keeps the `.vontraQ4` names, stored floating tensors and
    /// Q4/group64 routed experts. Any other packed matrix may instead use its
    /// declared per-module affine group64 override (`mixedOverrideBits`).
    /// `.nvfp4Hybrid` is the `.affineMixed` layout except that the target
    /// dense MLPs and all routed experts are NVIDIA NVFP4: `.weight` U32
    /// words, `.scales` U8 FP8-E4M3 per 16 inputs and an F32 scalar
    /// `.weight_scale_2` (ModelOpt's per-tensor scale, applied to the output).
    public enum Storage: Equatable, Sendable {
        case vontraQ4, float32
        case affineMixed
        case nvfp4Hybrid
        public var isPacked: Bool { self != .float32 }
    }
    /// Declared packing of one matrix, recorded on each of its packed tensors.
    public struct Packing: Hashable, Sendable {
        public let bits: Int
        public let groupSize: Int
        public let mode: QuantizationMode
        public init(bits: Int, groupSize: Int, mode: QuantizationMode = .affine) {
            self.bits = bits
            self.groupSize = groupSize
            self.mode = mode
        }
        static let q4Group64 = Packing(bits: 4, groupSize: 64)
        static let nvfp4 = Packing(bits: 4, groupSize: 16, mode: .nvfp4)
    }
    /// Safetensors dtypes a GLM5Next artifact may store, with their MLX types.
    static let storedTypes: [String: (dtype: DType, bytes: Int)] = [
        "BF16": (.bfloat16, 2), "F32": (.float32, 4), "U32": (.uint32, 4), "U8": (.uint8, 1),
    ]
    static let mixedOverrideBits: Set<Int> = [6, 8]
    public struct Tensor: Equatable, Sendable {
        public let name: String
        public let shape: [Int]
        public let dtype: String
        public let shard: String
        public init(name: String, shape: [Int], dtype: String, shard: String) {
            self.name = name
            self.shape = shape
            self.dtype = dtype
            self.shard = shard
        }
    }
    public struct Entry: Sendable {
        public let tensor: Tensor
        public let part: Part
        public let nativeName: String
        public let bytes: Int
        /// Nil for stored floating tensors.
        public let packing: Packing?
    }
    public let configuration: GLM5NextConfiguration
    public let storage: Storage
    public let entries: [String: Entry]
    public func entries(in part: Part) -> [Entry] {
        entries.values.filter { $0.part == part }.sorted { $0.tensor.name < $1.tensor.name }
    }
    /// The partition's stored bytes (a sum: no name order needed).
    public func bytes(in part: Part) -> Int {
        entries.values.reduce(0) { $1.part == part ? $0 + $1.bytes : $0 }
    }

    /// Source names stay available for the separate MTP and vision owners. Only
    /// individual expert IDs disappear in the native stacked-bank destination.
    public static func nativeName(for source: String) -> String {
        var name = source.replacingOccurrences(of: "model.language_model.", with: "model.")
        if let range = name.range(of: #"\.mlp\.experts\.[0-9]+\."#, options: .regularExpression) {
            name.replaceSubrange(range, with: ".mlp.experts.")
        }
        return name
    }

    /// The only packed storage a converted configuration can select. Selection
    /// is not acceptance: validate() still checks the complete declaration.
    public static func packedStorage(declaredBy configuration: GLM5NextConfiguration) throws
        -> Storage
    {
        guard let policy = configuration.nativeQuantization else {
            throw GLM5NextLoadError.invalid("packed artifact requires native quantization metadata")
        }
        if policy.perModule.values.contains(where: { $0.mode == .nvfp4 }) { return .nvfp4Hybrid }
        return policy.perModule.isEmpty ? .vontraQ4 : .affineMixed
    }

    /// Module keys a hybrid artifact must declare as NVFP4: exactly the target
    /// dense MLP matrices and every routed expert matrix (target and MTP
    /// layers); never shared experts, attention, heads or vision.
    public static func nvfp4Modules(_ c: GLM5NextTextConfiguration) -> Set<String> {
        var modules = Set<String>()
        let projections = ["gate_proj", "up_proj", "down_proj"]
        for index in 0 ..< c.hiddenLayers + c.numNextNPredictLayers {
            let p = "model.language_model.layers.\(index).mlp"
            if index < c.hiddenLayers && c.mlpLayerTypes[index] == .dense {
                for name in projections { modules.insert(p + "." + name) }
            } else {
                for expert in 0 ..< c.nRoutedExperts {
                    for name in projections { modules.insert(p + ".experts.\(expert).\(name)") }
                }
            }
        }
        return modules
    }

    public static func validate(
        configuration: GLM5NextConfiguration, tensors: [Tensor], storage: Storage
    ) throws -> Self {
        if case .vontraQ4 = storage {
            guard let policy = configuration.nativeQuantization,
                policy.defaultQuantization.bits == 4, policy.defaultQuantization.groupSize == 64,
                policy.defaultQuantization.mode == .affine,
                policy.perModule.isEmpty, policy.skippedModules.isEmpty
            else {
                throw GLM5NextLoadError.invalid("Vontra requires exact affine Q4/group64 metadata")
            }
        } else if storage == .float32, configuration.nativeQuantization != nil {
            throw GLM5NextLoadError.invalid("floating fixture cannot declare packed quantization")
        }
        let expected = try schema(configuration: configuration, storage: storage)
        var result: [String: Entry] = [:]
        var total = 0
        for tensor in tensors {
            guard result[tensor.name] == nil else {
                throw GLM5NextLoadError.invalid("duplicate \(tensor.name)")
            }
            guard let spec = expected[tensor.name] else {
                throw GLM5NextLoadError.unexpected(tensor.name)
            }
            guard tensor.shape == spec.shape, tensor.dtype == spec.dtype,
                !tensor.shard.isEmpty, !tensor.shard.contains("/"), !tensor.shard.contains("\\"),
                tensor.shard != ".", tensor.shard != ".."
            else { throw GLM5NextLoadError.invalid("header \(tensor.name)") }
            guard var bytes = storedTypes[tensor.dtype]?.bytes else {
                throw GLM5NextLoadError.invalid("header \(tensor.name)")
            }
            for dimension in tensor.shape {
                let product = bytes.multipliedReportingOverflow(by: dimension)
                guard dimension > 0, !product.overflow else {
                    throw GLM5NextLoadError.invalid("tensor size")
                }
                bytes = product.partialValue
            }
            let sum = total.addingReportingOverflow(bytes)
            guard !sum.overflow else { throw GLM5NextLoadError.invalid("inventory size") }
            total = sum.partialValue
            result[tensor.name] = Entry(
                tensor: tensor, part: spec.part, nativeName: nativeName(for: tensor.name),
                bytes: bytes, packing: spec.packing)
        }
        if let missing = expected.keys.sorted().first(where: { result[$0] == nil }) {
            throw GLM5NextLoadError.missing(missing)
        }
        return Self(configuration: configuration, storage: storage, entries: result)
    }

    /// Complete schema, also useful to a bounded header reader before any payload
    /// access. The floating mode is explicit test/reference storage, not a fallback
    /// that accepts BF16/FP8 source matrices in a Q4 artifact.
    public static func expectedTensors(
        configuration: GLM5NextConfiguration, storage: Storage, shard: String
    ) throws -> [Tensor] {
        try schema(configuration: configuration, storage: storage).map { name, spec in
            Tensor(name: name, shape: spec.shape, dtype: spec.dtype, shard: shard)
        }.sorted { $0.name < $1.name }
    }

    private struct Spec {
        let shape: [Int]
        let dtype: String
        let part: Part
        let packing: Packing?
    }

    /// Mixed policy keys are artifact module prefixes (tensor names without
    /// `.weight`/`.scales`/`.biases`). schema() refuses keys that are not
    /// non-expert packed matrices, so every declaration changes a known shape.
    /// Hybrid storage additionally requires the NVFP4 declarations to be
    /// exactly `nvfp4Modules`; those keys are returned separately, never as
    /// affine overrides.
    private static func mixedOverrides(_ root: GLM5NextConfiguration, storage: Storage) throws
        -> [String: Packing]
    {
        if case .sourceFP8? = root.quantizationConfig {
            throw GLM5NextLoadError.invalid("mixed storage cannot retain source FP8 metadata")
        }
        guard let policy = root.nativeQuantization,
            policy.defaultQuantization.bits == 4, policy.defaultQuantization.groupSize == 64,
            policy.defaultQuantization.mode == .affine, !policy.perModule.isEmpty,
            policy.skippedModules.isEmpty, policy.metadata.isEmpty
        else {
            throw GLM5NextLoadError.invalid(
                "mixed storage requires an affine Q4/group64 default and module overrides only")
        }
        let nvfp4 = Set(policy.perModule.filter { $0.value.mode == .nvfp4 }.keys)
        if storage == .nvfp4Hybrid {
            guard nvfp4 == nvfp4Modules(root.textConfig) else {
                throw GLM5NextLoadError.invalid(
                    "nvfp4 storage requires exactly the target dense MLP and routed expert modules")
            }
        }
        var overrides: [String: Packing] = [:]
        for (module, declared) in policy.perModule.sorted(by: { $0.key < $1.key }) {
            if storage == .nvfp4Hybrid, nvfp4.contains(module) {
                guard declared.bits == 4, declared.groupSize == 16 else {
                    throw GLM5NextLoadError.invalid("quantization override \(module)")
                }
                continue
            }
            guard declared.mode == .affine, declared.groupSize == 64,
                mixedOverrideBits.contains(declared.bits)
            else { throw GLM5NextLoadError.invalid("quantization override \(module)") }
            overrides[module] = Packing(bits: declared.bits, groupSize: declared.groupSize)
        }
        return overrides
    }

    private static func schema(configuration root: GLM5NextConfiguration, storage: Storage) throws
        -> [String: Spec]
    {
        let c = root.textConfig
        let v = root.visionConfig
        guard c.indexKPool == 4, c.indexKPoolCompress, c.indexKPoolAlwaysSelectTail,
            c.indexTopK <= 2048, c.mhc, !c.attentionBias, !c.tieWordEmbeddings,
            c.attentionDropout == 0, c.numNextNPredictLayers <= 1
        else { throw GLM5NextLoadError.invalid("unsupported GLM5Next execution schema") }
        let mixed = storage == .affineMixed || storage == .nvfp4Hybrid
        var overrides = try mixed ? mixedOverrides(root, storage: storage) : [:]
        var specs: [String: Spec] = [:]
        func add(
            _ name: String, _ shape: [Int], _ part: Part, fp32: Bool = false,
            packing: Packing? = nil
        ) {
            specs[name] = Spec(
                shape: shape, dtype: storage == .float32 || fp32 ? "F32" : "BF16", part: part,
                packing: packing)
        }
        func matrix(
            _ name: String, _ output: Int, _ input: Int, _ part: Part, routedExpert: Bool = false,
            nvfp4: Bool = false
        ) throws {
            guard storage.isPacked else {
                add(name + ".weight", [output, input], part)
                return
            }
            if nvfp4 {
                let packing = Packing.nvfp4
                guard input % packing.groupSize == 0 else {
                    throw GLM5NextLoadError.invalid("nvfp4 input width \(name)")
                }
                specs[name + ".weight"] = Spec(
                    shape: [output, input * packing.bits / 32], dtype: "U32", part: part,
                    packing: packing)
                specs[name + ".scales"] = Spec(
                    shape: [output, input / packing.groupSize], dtype: "U8", part: part,
                    packing: packing)
                specs[name + ".weight_scale_2"] = Spec(
                    shape: [], dtype: "F32", part: part, packing: packing)
                return
            }
            let declared = overrides.removeValue(forKey: name)
            // One policy per stacked bank: experts always keep the Q4 default.
            guard !routedExpert || declared == nil else {
                throw GLM5NextLoadError.invalid("routed expert override \(name)")
            }
            let packing = declared ?? .q4Group64
            let packedBits = input.multipliedReportingOverflow(by: packing.bits)
            guard input % packing.groupSize == 0, !packedBits.overflow,
                packedBits.partialValue % 32 == 0
            else {
                throw GLM5NextLoadError.invalid("Q\(packing.bits) input width \(name)")
            }
            specs[name + ".weight"] = Spec(
                shape: [output, packedBits.partialValue / 32], dtype: "U32", part: part,
                packing: packing)
            add(name + ".scales", [output, input / packing.groupSize], part, packing: packing)
            add(name + ".biases", [output, input / packing.groupSize], part, packing: packing)
        }
        func mlp(
            _ prefix: String, _ intermediate: Int, _ part: Part, routedExpert: Bool = false,
            nvfp4: Bool = false
        ) throws {
            let (r, f) = (routedExpert, nvfp4)
            try matrix(
                prefix + ".gate_proj", intermediate, c.hiddenSize, part, routedExpert: r, nvfp4: f)
            try matrix(
                prefix + ".up_proj", intermediate, c.hiddenSize, part, routedExpert: r, nvfp4: f)
            try matrix(
                prefix + ".down_proj", c.hiddenSize, intermediate, part, routedExpert: r, nvfp4: f)
        }
        // mixedOverrides() has already proved the declaration equals this set.
        let nvfp4FeedForward = storage == .nvfp4Hybrid
        func dsa(_ prefix: String, _ part: Part) throws {
            try matrix(prefix + ".q_a_proj", c.qLoraRank, c.hiddenSize, part)
            add(prefix + ".q_a_layernorm.weight", [c.qLoraRank], part)
            try matrix(prefix + ".q_b_proj", c.attentionHeads * c.qkNopeHeadDim, c.qLoraRank, part)
            try matrix(prefix + ".kv_a_proj_with_mqa", c.kvLoraRank, c.hiddenSize, part)
            add(prefix + ".kv_a_layernorm.weight", [c.kvLoraRank], part)
            try matrix(
                prefix + ".kv_b_proj", c.attentionHeads * (c.qkNopeHeadDim + c.vHeadDim),
                c.kvLoraRank, part)
            try matrix(prefix + ".o_proj", c.hiddenSize, c.attentionHeads * c.vHeadDim, part)
            let i = prefix + ".indexer"
            try matrix(i + ".wq_b", c.indexNHeads * c.indexHeadDim, c.qLoraRank, part)
            try matrix(i + ".wk", c.indexHeadDim, c.hiddenSize, part)
            try matrix(i + ".weights_proj", c.indexNHeads, c.hiddenSize, part)
            add(i + ".k_norm.weight", [c.indexHeadDim], part)
            add(i + ".k_norm.bias", [c.indexHeadDim], part)
            add(i + ".index_kpool_compress_ape", [4, c.indexHeadDim], part)
            add(i + ".index_kpool_compress_gate", [c.indexHeadDim, c.hiddenSize], part)
        }
        try matrix("model.language_model.embed_tokens", c.vocabularySize, c.hiddenSize, .text)
        try matrix("lm_head", c.vocabularySize, c.hiddenSize, .text)
        add("model.language_model.norm.weight", [c.hiddenSize], .text)
        for index in 0 ..< c.hiddenLayers + c.numNextNPredictLayers {
            let part: Part = index < c.hiddenLayers ? .text : .mtp
            let p = "model.language_model.layers.\(index)"
            add(p + ".input_layernorm.weight", [c.hiddenSize], part)
            add(p + ".post_attention_layernorm.weight", [c.hiddenSize], part)
            if part == .text {
                for site in ["attn", "ffn"] {
                    add(
                        p + ".hc_\(site)_fn", [(2 + c.hcMult) * c.hcMult, c.hcMult * c.hiddenSize],
                        part)
                    add(p + ".hc_\(site)_base", [(2 + c.hcMult) * c.hcMult], part, fp32: true)
                    add(p + ".hc_\(site)_scale", [3], part, fp32: true)
                }
            } else {
                for name in ["enorm", "hnorm", "shared_head.norm"] {
                    add(p + ".\(name).weight", [c.hiddenSize], part)
                }
                try matrix(p + ".eh_proj", c.hiddenSize, 2 * c.hiddenSize, part)
            }
            if part == .text && c.layerTypes[index] == .linearAttention {
                let a = p + ".self_attn"
                let linear = c.linearAttention
                let width = linear.numHeads * linear.headDim
                for name in ["q", "k", "v"] {
                    try matrix(a + ".\(name)_proj", width, c.hiddenSize, part)
                    add(a + ".\(name)_conv1d.weight", [width, 1, linear.shortConvKernelSize], part)
                }
                for name in ["f", "g"] {
                    try matrix(a + ".\(name)_a_proj", linear.headDim, c.hiddenSize, part)
                    try matrix(a + ".\(name)_b_proj", width, linear.headDim, part)
                }
                try matrix(a + ".b_proj", linear.numHeads, c.hiddenSize, part)
                try matrix(a + ".o_proj", c.hiddenSize, width, part)
                add(a + ".A_log", [linear.numHeads], part, fp32: true)
                add(a + ".dt_bias", [width], part, fp32: true)
                add(a + ".o_norm.weight", [linear.headDim], part)
            } else {
                try dsa(p + ".self_attn", part)
            }
            if part == .text && c.mlpLayerTypes[index] == .dense {
                try mlp(p + ".mlp", c.intermediateSize, part, nvfp4: nvfp4FeedForward)
            } else {
                add(p + ".mlp.gate.weight", [c.nRoutedExperts, c.hiddenSize], part)
                add(p + ".mlp.gate.e_score_correction_bias", [c.nRoutedExperts], part, fp32: true)
                try mlp(p + ".mlp.shared_experts", c.moeIntermediateSize * c.nSharedExperts, part)
                for expert in 0 ..< c.nRoutedExperts {
                    try mlp(
                        p + ".mlp.experts.\(expert)", c.moeIntermediateSize, part,
                        routedExpert: true, nvfp4: nvfp4FeedForward)
                }
            }
        }
        let p = "model.visual"
        let head = v.hiddenSize / v.numHeads
        add(
            p + ".patch_embed.proj.weight",
            [v.hiddenSize, v.inChannels, v.temporalPatchSize, v.patchSize, v.patchSize], .vision)
        add(p + ".patch_embed.proj.bias", [v.hiddenSize], .vision)
        add(p + ".post_layernorm.weight", [v.hiddenSize], .vision)
        add(
            p + ".downsample.weight",
            [v.outHiddenSize, v.hiddenSize, v.spatialMergeSize, v.spatialMergeSize], .vision)
        add(p + ".downsample.bias", [v.outHiddenSize], .vision)
        for index in 0 ..< v.depth {
            let b = p + ".blocks.\(index)"
            for name in ["norm1", "norm2"] { add(b + ".\(name).weight", [v.hiddenSize], .vision) }
            for name in ["q_norm", "k_norm"] { add(b + ".attn.\(name).weight", [head], .vision) }
            for (name, output, input) in [
                ("attn.qkv", 3 * v.hiddenSize, v.hiddenSize),
                ("attn.proj", v.hiddenSize, v.hiddenSize),
                ("mlp.gate_proj", v.intermediateSize, v.hiddenSize),
                ("mlp.up_proj", v.intermediateSize, v.hiddenSize),
                ("mlp.down_proj", v.hiddenSize, v.intermediateSize),
            ] {
                add(b + ".\(name).weight", [output, input], .vision)
                if v.attentionBias { add(b + ".\(name).bias", [output], .vision) }
            }
        }
        for (name, output, input) in [
            ("proj", v.outHiddenSize, v.outHiddenSize),
            ("gate_proj", v.projectionIntermediateSize, v.outHiddenSize),
            ("up_proj", v.projectionIntermediateSize, v.outHiddenSize),
            ("down_proj", v.outHiddenSize, v.projectionIntermediateSize),
        ] { add(p + ".merger.\(name).weight", [output, input], .vision) }
        for suffix in ["weight", "bias"] {
            add(p + ".merger.post_projection_norm.\(suffix)", [v.outHiddenSize], .vision)
        }
        // Includes declarations for stored floating, vision or unknown modules.
        if let unknown = overrides.keys.min() {
            throw GLM5NextLoadError.unexpected("quantization override \(unknown)")
        }
        return specs
    }
}

/// Caller-provided bounded resolver. It must verify shard hashes/completeness
/// externally and must not read partial download files. Each returned array is
/// checked again; an inventory does not authorize mismatching payload geometry.
final class GLM5NextTensorLoader {
    let inventory: GLM5NextWeightInventory
    let resolve: (String) throws -> MLXArray
    let releaseSourceTensors: (([String]) -> Void)?
    let materializeFusedProjections: (([MLXArray]) throws -> Void)?
    /// A tensor-parallel rank keeps only its part of each split tensor; the
    /// full stored shape is still verified against the inventory first.
    var shard: GLM5NextTensorShard?
    /// Payloads read and verified ahead of construction, already in this
    /// rank's shape (`GLM5NextArtifactLoader.prepare`): when set, every
    /// tensor and routed-expert bank comes from here instead of `resolve`,
    /// `shard.slice` and a stack, with the same bytes.
    var prepared: GLM5NextPreparedParts?
    private(set) var consumed: Set<String> = []
    init(
        _ inventory: GLM5NextWeightInventory, _ resolve: @escaping (String) throws -> MLXArray,
        releaseSourceTensors: (([String]) -> Void)? = nil,
        materializeFusedProjections: (([MLXArray]) throws -> Void)? = nil
    ) {
        self.inventory = inventory
        self.resolve = resolve
        self.releaseSourceTensors = releaseSourceTensors
        self.materializeFusedProjections = materializeFusedProjections
    }
    func array(_ name: String, releaseSource: Bool = true) throws -> MLXArray {
        guard let entry = inventory.entries[name] else { throw GLM5NextLoadError.missing(name) }
        if let prepared {
            let part = try prepared.take(name)
            guard part.shape == (try shard?.localShape(name, entry.tensor.shape) ?? entry.tensor.shape),
                part.dtype == GLM5NextWeightInventory.storedTypes[entry.tensor.dtype]?.dtype
            else {
                throw GLM5NextLoadError.invalid("prepared payload \(name)")
            }
            consumed.insert(name)
            return part
        }
        // The returned MLXArray must own its storage independently of the
        // resolver's lookup cache. Eviction drops source-cache references only.
        defer { if releaseSource { releaseSourceTensors?([name]) } }
        let result = try resolve(name)
        guard result.shape == entry.tensor.shape,
            result.dtype == GLM5NextWeightInventory.storedTypes[entry.tensor.dtype]?.dtype
        else {
            throw GLM5NextLoadError.invalid("payload \(name)")
        }
        consumed.insert(name)
        return try shard?.slice(name, result) ?? result
    }
    /// The prepared stack of `names` (one routed expert each, in order), the
    /// array `stacked(names.map { array($0) })` would build; nil when nothing
    /// was prepared.
    func preparedBank(_ names: [String]) throws -> MLXArray? {
        guard let prepared else { return nil }
        guard let first = names.first, let entry = inventory.entries[first] else {
            throw GLM5NextLoadError.missing(names.first ?? "empty expert bank")
        }
        let bank = try prepared.takeBank(names)
        let member = try shard?.localShape(first, entry.tensor.shape) ?? entry.tensor.shape
        let dtype = GLM5NextWeightInventory.storedTypes[entry.tensor.dtype]?.dtype
        guard bank.shape == [names.count] + member, bank.dtype == dtype,
            names.allSatisfy({
                inventory.entries[$0]?.tensor.shape == entry.tensor.shape
                    && inventory.entries[$0]?.tensor.dtype == entry.tensor.dtype
            })
        else {
            throw GLM5NextLoadError.invalid("prepared expert bank \(first)")
        }
        consumed.formUnion(names)
        return bank
    }
    func linear(_ prefix: String) throws -> Linear {
        let weight = try array(prefix + ".weight")
        if weight.dtype == .uint32 {
            let packing = try self.packing(prefix)
            if packing.mode == .nvfp4 {
                return try GLM5NextNVFP4Linear(
                    weight: weight, scales: array(prefix + ".scales"),
                    globalScale: array(prefix + ".weight_scale_2"))
            }
            return QuantizedLinear(
                weight: weight, scales: try array(prefix + ".scales"),
                biases: try array(prefix + ".biases"), groupSize: packing.groupSize,
                bits: packing.bits)
        }
        return Linear(weight: weight)
    }
    func packing(_ prefix: String) throws -> GLM5NextWeightInventory.Packing {
        guard let packing = inventory.entries[prefix + ".weight"]?.packing else {
            throw GLM5NextLoadError.invalid("undeclared packing \(prefix)")
        }
        return packing
    }
    /// A stacked bank has one declaration; nil only when every member is floating.
    func bankPacking(_ prefixes: [String]) throws -> GLM5NextWeightInventory.Packing? {
        let declared = prefixes.first.flatMap { inventory.entries[$0 + ".weight"]?.packing }
        guard prefixes.allSatisfy({ inventory.entries[$0 + ".weight"]?.packing == declared }) else {
            throw GLM5NextLoadError.invalid("non-uniform bank packing \(prefixes.first ?? "")")
        }
        return declared
    }
    func finishText() throws {
        if let missing = inventory.entries(in: .text).first(where: {
            !consumed.contains($0.tensor.name)
        }) {
            throw GLM5NextLoadError.missing("unconsumed text tensor \(missing.tensor.name)")
        }
    }
}
