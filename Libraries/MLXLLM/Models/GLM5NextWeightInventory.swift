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
    public enum Storage: Equatable, Sendable { case vontraQ4, float32 }
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
    }
    public let configuration: GLM5NextConfiguration
    public let storage: Storage
    public let entries: [String: Entry]
    public func entries(in part: Part) -> [Entry] {
        entries.values.filter { $0.part == part }.sorted { $0.tensor.name < $1.tensor.name }
    }
    public func bytes(in part: Part) -> Int { entries(in: part).reduce(0) { $0 + $1.bytes } }

    /// Source names stay available for the separate MTP and vision owners. Only
    /// individual expert IDs disappear in the native stacked-bank destination.
    public static func nativeName(for source: String) -> String {
        var name = source.replacingOccurrences(of: "model.language_model.", with: "model.")
        if let range = name.range(of: #"\.mlp\.experts\.[0-9]+\."#, options: .regularExpression) {
            name.replaceSubrange(range, with: ".mlp.experts.")
        }
        return name
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
        } else if configuration.nativeQuantization != nil {
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
            var bytes = tensor.dtype == "F32" || tensor.dtype == "U32" ? 4 : 2
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
                bytes: bytes)
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
        var specs: [String: Spec] = [:]
        func add(_ name: String, _ shape: [Int], _ part: Part, fp32: Bool = false) {
            specs[name] = Spec(
                shape: shape, dtype: storage == .float32 || fp32 ? "F32" : "BF16", part: part)
        }
        func matrix(_ name: String, _ output: Int, _ input: Int, _ part: Part) throws {
            if storage == .vontraQ4 {
                guard input % 64 == 0 else {
                    throw GLM5NextLoadError.invalid("Q4 input width \(name)")
                }
                specs[name + ".weight"] = Spec(
                    shape: [output, input / 8], dtype: "U32", part: part)
                add(name + ".scales", [output, input / 64], part)
                add(name + ".biases", [output, input / 64], part)
            } else {
                add(name + ".weight", [output, input], part)
            }
        }
        func mlp(_ prefix: String, _ intermediate: Int, _ part: Part) throws {
            try matrix(prefix + ".gate_proj", intermediate, c.hiddenSize, part)
            try matrix(prefix + ".up_proj", intermediate, c.hiddenSize, part)
            try matrix(prefix + ".down_proj", c.hiddenSize, intermediate, part)
        }
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
                try mlp(p + ".mlp", c.intermediateSize, part)
            } else {
                add(p + ".mlp.gate.weight", [c.nRoutedExperts, c.hiddenSize], part)
                add(p + ".mlp.gate.e_score_correction_bias", [c.nRoutedExperts], part, fp32: true)
                try mlp(p + ".mlp.shared_experts", c.moeIntermediateSize * c.nSharedExperts, part)
                for expert in 0 ..< c.nRoutedExperts {
                    try mlp(p + ".mlp.experts.\(expert)", c.moeIntermediateSize, part)
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
    private(set) var consumed: Set<String> = []
    init(
        _ inventory: GLM5NextWeightInventory, _ resolve: @escaping (String) throws -> MLXArray,
        releaseSourceTensors: (([String]) -> Void)? = nil
    ) {
        self.inventory = inventory
        self.resolve = resolve
        self.releaseSourceTensors = releaseSourceTensors
    }
    func array(_ name: String, releaseSource: Bool = true) throws -> MLXArray {
        guard let entry = inventory.entries[name] else { throw GLM5NextLoadError.missing(name) }
        // The returned MLXArray must own its storage independently of the
        // resolver's lookup cache. Eviction drops source-cache references only.
        defer { if releaseSource { releaseSourceTensors?([name]) } }
        let result = try resolve(name)
        let dtype: DType =
            entry.tensor.dtype == "U32"
            ? .uint32 : entry.tensor.dtype == "F32" ? .float32 : .bfloat16
        guard result.shape == entry.tensor.shape, result.dtype == dtype else {
            throw GLM5NextLoadError.invalid("payload \(name)")
        }
        consumed.insert(name)
        return result
    }
    func linear(_ prefix: String) throws -> Linear {
        let weight = try array(prefix + ".weight")
        if weight.dtype == .uint32 {
            return QuantizedLinear(
                weight: weight, scales: try array(prefix + ".scales"),
                biases: try array(prefix + ".biases"), groupSize: 64, bits: 4)
        }
        return Linear(weight: weight)
    }
    func finishText() throws {
        if let missing = inventory.entries(in: .text).first(where: {
            !consumed.contains($0.tensor.name)
        }) {
            throw GLM5NextLoadError.missing("unconsumed text tensor \(missing.tensor.name)")
        }
    }
}
