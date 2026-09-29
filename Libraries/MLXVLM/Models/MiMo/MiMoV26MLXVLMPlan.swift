// Copyright © 2026 Eigen Labs.
// Metadata-only validation of the published MLX-VLM MiMo layout. No arrays,
// artifact rewrite, dequantization or synthetic conversion receipt is involved.
import Foundation
import MLXLLM

enum MiMoV26MLXVLMPlan {
    static func make(
        configurationData: Data, indexData: Data,
        descriptors: [String: MiMoV26ConvertedTensorDescriptor],
        provenance: MiMoV26ConvertedProvenance
    ) throws -> MiMoV26ConvertedLoadPlan {
        let c = try JSONDecoder().decode(MiMoV26Configuration.self, from: configurationData)
        let index = try JSONDecoder().decode(MiMoV26ConvertedIndex.self, from: indexData)
        guard provenance.layout == .mlxVLM, !provenance.artifactID.isEmpty,
            ["XiaomiMiMo/MiMo-V2.6-Flash-MOPD", "XiaomiMiMo/MiMo-V2.6-Flash-RL"].contains(
                provenance.sourceRepository),
            convertedHex(provenance.sourceRevision, count: 40),
            convertedHex(provenance.conversionManifestSHA256, count: 64),
            provenance.payloadVerificationReceiptSHA256.map({ convertedHex($0, count: 64) }) ?? true
        else {
            throw MiMoV26ConvertedLoadError.invalidProvenance("MLX-VLM artifact/source identity")
        }
        guard c.numHiddenLayers <= 48, c.hiddenSize <= 4096, c.intermediateSize <= 16384,
            c.moeIntermediateSize <= 2048, c.routedExpertCount <= 256, c.vocabularySize <= 152576,
            c.numNextnPredictLayers == 3, c.dtype == "bfloat16", !c.tieWordEmbeddings,
            !c.attentionBias, c.attentionDropout == 0,
            c.sharedExpertCount == nil || c.sharedExpertCount == 0,
            let vision = c.vision, let audio = c.audio,
            vision.depth <= 28, vision.hiddenSize <= 1280, vision.intermediateSize <= 4608,
            c.quantization.nativeDefault?.mode == "affine",
            c.quantization.nativeDefault?.bits == 4, c.quantization.nativeDefault?.groupSize == 64,
            c.quantization.sourceMetadata == nil
        else {
            throw MiMoV26ConvertedLoadError.invalidConfiguration(
                "MLX-VLM packed target/audio and floating vision/MTP geometry")
        }
        // Each trained parameter has one source name and one native name. Keep
        // the source namespace in the filesystem plan for full hash/TOCTOU checks.
        var expected: [String: (shape: [Int], dtype: MiMoV26ConvertedScalarType)] = [:]
        var names: [String: String] = [:]
        var canonicalNames = Set<String>()
        var groups = Dictionary(
            uniqueKeysWithValues: MiMoV26ConvertedComponent.allCases.map { ($0, Set<String>()) })
        var targetPolicies: [String: MiMoV26Quantization.Policy] = [:]
        var audioPolicies: [String: MiMoV26Quantization.Policy] = [:]
        var usedPolicies = Set<String>()
        var experts = Set<String>()
        func sourceName(_ key: String, _ component: MiMoV26ConvertedComponent) -> String {
            switch component {
            case .target: return "language_model." + key
            case .vision: return "vision_tower." + key.dropFirst("visual.".count)
            case .audioPatch: return key
            case .mtp: return "language_model.model." + key
            }
        }
        func add(
            _ key: String, _ shape: [Int], _ dtype: MiMoV26ConvertedScalarType,
            _ component: MiMoV26ConvertedComponent
        ) throws {
            let source = sourceName(key, component)
            guard expected[source] == nil, canonicalNames.insert(key).inserted,
                !shape.isEmpty, shape.allSatisfy({ $0 > 0 && $0 <= Int(Int32.max) })
            else {
                throw MiMoV26ConvertedLoadError.invalidInventory(
                    "duplicate/invalid MLX-VLM parameter: " + key)
            }
            _ = try convertedProduct(shape + [dtype.bytes], source)
            expected[source] = (shape, dtype)
            names[source] = key
            groups[component, default: []].insert(source)
        }
        func packed(
            _ module: String, _ shape: [Int], _ component: MiMoV26ConvertedComponent,
            expert: Bool = false
        ) throws {
            let source = sourceName(module, component)
            guard let declared = c.quantization.nativeOverrides[source],
                case .quantize(let policy) = declared,
                policy.mode == (expert ? "mxfp4" : "affine"), policy.bits == (expert ? 4 : 8),
                policy.groupSize == (expert ? 32 : 64),
                let cols = shape.last, cols.isMultiple(of: policy.groupSize),
                shape.count == (expert ? 3 : 2), usedPolicies.insert(source).inserted
            else {
                throw MiMoV26ConvertedLoadError.invalidQuantization(
                    "missing/incompatible explicit MLX-VLM policy: " + source)
            }
            let prefix = Array(shape.dropLast())
            try add(module + ".weight", prefix + [cols / (32 / policy.bits)], .uint32, component)
            try add(
                module + ".scales", prefix + [cols / policy.groupSize], expert ? .uint8 : .bfloat16,
                component)
            if !expert {
                try add(
                    module + ".biases", prefix + [cols / policy.groupSize], .bfloat16, component)
            }
            if component == .target {
                targetPolicies[module] = policy
            } else if component == .audioPatch {
                audioPolicies[module] = policy
            } else {
                throw MiMoV26ConvertedLoadError.invalidQuantization("unexpected packed component")
            }
            if expert { experts.insert(module) }
        }
        try packed("model.embed_tokens", [c.vocabularySize, c.hiddenSize], .target)
        try add("model.norm.weight", [c.hiddenSize], .bfloat16, .target)
        try packed("lm_head", [c.vocabularySize, c.hiddenSize], .target)
        for layer in 0 ..< c.numHiddenLayers {
            let p = "model.layers.\(layer)."
            let g = try c.attentionGeometry(at: layer)
            for norm in ["input_layernorm", "post_attention_layernorm"] {
                try add(p + norm + ".weight", [c.hiddenSize], .bfloat16, .target)
            }
            for (name, width) in [
                ("q_proj", g.queryHeads * g.headDim),
                ("k_proj", g.keyValueHeads * g.headDim),
                ("v_proj", g.keyValueHeads * g.valueHeadDim),
            ] {
                try packed(p + "self_attn." + name, [width, c.hiddenSize], .target)
            }
            try packed(
                p + "self_attn.o_proj", [c.hiddenSize, g.queryHeads * g.valueHeadDim], .target)
            if g.hasSinks {
                try add(p + "self_attn.attention_sink_bias", [g.queryHeads], .bfloat16, .target)
            }
            if c.moeLayerFrequency[layer] == 1 {
                try add(
                    p + "mlp.gate.weight", [c.routedExpertCount, c.hiddenSize],
                    c.moeRouterDType == "bfloat16" ? .bfloat16 : .float32, .target)
                try add(
                    p + "mlp.gate.e_score_correction_bias", [c.routedExpertCount], .float32, .target
                )
                for name in ["gate_proj", "up_proj", "down_proj"] {
                    try packed(
                        p + "mlp.switch_mlp." + name,
                        [
                            c.routedExpertCount,
                            name == "down_proj" ? c.hiddenSize : c.moeIntermediateSize,
                            name == "down_proj" ? c.moeIntermediateSize : c.hiddenSize,
                        ], .target, expert: true)
                }
            } else {
                for name in ["gate_proj", "up_proj"] {
                    try packed(p + "mlp." + name, [c.intermediateSize, c.hiddenSize], .target)
                }
                try packed(p + "mlp.down_proj", [c.hiddenSize, c.intermediateSize], .target)
            }
        }
        for (key, shape) in try MiMoV26VisionTower.expectedTensorShapes(configuration: vision) {
            let stored =
                key == "visual.patch_embed.proj.weight"
                ? [vision.hiddenSize, try convertedProduct(Array(shape.dropFirst()), key)] : shape
            try add(key, stored, .bfloat16, .vision)
        }
        for (key, shape) in try MiMoV26AudioPatchEncoder.expectedTensorShapes(configuration: audio)
        {
            if key.hasSuffix(".weight"), shape.count == 2 {
                try packed(String(key.dropLast(".weight".count)), shape, .audioPatch)
            } else {
                try add(key, shape, .bfloat16, .audioPatch)
            }
        }
        let g = c.slidingAttention
        for depth in 0 ..< c.numNextnPredictLayers {
            let p = "mtp.layers.\(depth)."
            for norm in [
                "enorm", "hnorm", "input_layernorm", "pre_mlp_layernorm", "final_layernorm",
            ] {
                try add(p + norm + ".weight", [c.hiddenSize], .bfloat16, .mtp)
            }
            if g.hasSinks {
                try add(p + "self_attn.attention_sink_bias", [g.queryHeads], .bfloat16, .mtp)
            }
            for (name, rows, cols) in [
                ("eh_proj", c.hiddenSize, 2 * c.hiddenSize),
                ("self_attn.q_proj", g.queryHeads * g.headDim, c.hiddenSize),
                ("self_attn.k_proj", g.keyValueHeads * g.headDim, c.hiddenSize),
                ("self_attn.v_proj", g.keyValueHeads * g.valueHeadDim, c.hiddenSize),
                ("self_attn.o_proj", c.hiddenSize, g.queryHeads * g.valueHeadDim),
                ("mlp.gate_proj", c.intermediateSize, c.hiddenSize),
                ("mlp.up_proj", c.intermediateSize, c.hiddenSize),
                ("mlp.down_proj", c.hiddenSize, c.intermediateSize),
            ] { try add(p + name + ".weight", [rows, cols], .bfloat16, .mtp) }
        }
        guard usedPolicies == Set(c.quantization.nativeOverrides.keys) else {
            throw MiMoV26ConvertedLoadError.invalidQuantization(
                "unmapped MLX-VLM policy; floating MTP/vision must not inherit root default")
        }
        for field in ["quantization", "quantization_config"] {
            if case .object(let raw) = c.rawFields[field],
                Set(raw.keys) != usedPolicies.union(["mode", "bits", "group_size"])
            {
                throw MiMoV26ConvertedLoadError.invalidQuantization(
                    "unmapped native metadata: " + field)
            }
        }
        guard Set(descriptors.keys) == Set(expected.keys),
            Set(index.weightMap.keys) == Set(expected.keys)
        else {
            throw MiMoV26ConvertedLoadError.invalidInventory(
                "missing/unmapped MLX-VLM root tensors")
        }
        var total = 0
        for (key, spec) in expected {
            let actual = descriptors[key]!
            let file = actual.file
            guard actual.shape == spec.shape, actual.dtype == spec.dtype,
                file == index.weightMap[key], !file.isEmpty,
                file == (file as NSString).lastPathComponent,
                !file.contains("\\"), file.hasSuffix(".safetensors"), file != ".safetensors"
            else {
                throw MiMoV26ConvertedLoadError.invalidInventory(key)
            }
            let sum = total.addingReportingOverflow(
                try convertedProduct(actual.shape + [actual.dtype.bytes], key))
            guard !sum.overflow else {
                throw MiMoV26ConvertedLoadError.invalidInventory("MLX-VLM byte overflow")
            }
            total = sum.partialValue
        }
        guard total == index.metadata.totalSize else {
            throw MiMoV26ConvertedLoadError.invalidInventory("index tensor byte total")
        }
        if let embedded = c.embeddedMTP {
            guard groups[.mtp]!.allSatisfy({ descriptors[$0]!.file == embedded.file }) else {
                throw MiMoV26ConvertedLoadError.invalidInventory("embedded MTP file declaration")
            }
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return .init(
            configuration: c, descriptors: descriptors, components: groups,
            targetExpertModulePaths: experts,
            parameterNames: names, targetQuantizationPolicies: targetPolicies,
            audioQuantizationPolicies: audioPolicies, floatingMTP: true,
            configSHA256: convertedHash(configurationData), indexSHA256: convertedHash(indexData),
            descriptorSHA256: convertedHash(try encoder.encode(descriptors)),
            rootFiles: Set(index.weightMap.values),
            tensorBytes: total, provenance: provenance,
            externalRequiredComponents: ["audio_tokenizer"],
            floatingType: .bfloat16)
    }
}
