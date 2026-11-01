import Foundation
import Testing

@testable import MLXLLM

/// Tiny counterpart of the mixed conversion recipe. JSON/headers only, so it
/// is safe in the no-MLX UnitTests suite.
enum GLM5NextMixedInventoryFixture {
    /// KDA q/k/v/o at 6 bits; every other non-expert packed matrix at 8 bits.
    static func recipe(_ c: GLM5NextTextConfiguration) -> [String: Int] {
        var bits = ["lm_head": 8, "model.language_model.embed_tokens": 8]
        for index in 0 ..< c.hiddenLayers + c.numNextNPredictLayers {
            let p = "model.language_model.layers.\(index)"
            let target = index < c.hiddenLayers
            if target && c.layerTypes[index] == .linearAttention {
                for name in ["q", "k", "v", "o"] { bits[p + ".self_attn.\(name)_proj"] = 6 }
                for name in ["f_a", "f_b", "g_a", "g_b", "b"] {
                    bits[p + ".self_attn.\(name)_proj"] = 8
                }
            } else {
                for name in [
                    "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj",
                    "indexer.wq_b", "indexer.wk", "indexer.weights_proj",
                ] {
                    bits[p + ".self_attn." + name] = 8
                }
            }
            let mlp =
                target && c.mlpLayerTypes[index] == .dense ? p + ".mlp" : p + ".mlp.shared_experts"
            for name in ["gate_proj", "up_proj", "down_proj"] { bits[mlp + "." + name] = 8 }
            if !target { bits[p + ".eh_proj"] = 8 }
        }
        return bits
    }

    static func policy(_ overrides: [String: Int], groupSize: Int = 64) -> [String: Any] {
        var policy: [String: Any] = ["bits": 4, "group_size": 64, "mode": "affine"]
        for (module, bits) in overrides {
            policy[module] = ["bits": bits, "group_size": groupSize, "mode": "affine"]
        }
        return policy
    }

    /// The packed fixture with a replaced policy, mirrored into
    /// `quantization_config` unless another value is supplied.
    static func configuration(
        quantization: [String: Any], quantizationConfig: [String: Any]? = nil
    ) throws -> GLM5NextConfiguration {
        let base = try GLM5NextInventoryFixture.configuration(q4: true)
        var root =
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(base)) as! [String: Any]
        root["quantization"] = quantization
        root["quantization_config"] = quantizationConfig ?? quantization
        return try JSONDecoder().decode(
            GLM5NextConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
    }

    static func recipeConfiguration() throws -> GLM5NextConfiguration {
        let text = try GLM5NextInventoryFixture.configuration(q4: true).textConfig
        return try configuration(quantization: policy(recipe(text)))
    }

    static func inventory(
        _ configuration: GLM5NextConfiguration, _ storage: GLM5NextWeightInventory.Storage
    ) throws -> GLM5NextWeightInventory {
        try GLM5NextWeightInventory.validate(
            configuration: configuration,
            tensors: GLM5NextWeightInventory.expectedTensors(
                configuration: configuration, storage: storage, shard: "fixture.safetensors"),
            storage: storage)
    }
}

extension UnitTests {
    @Suite struct GLM5NextMixedInventoryTests {
        private typealias Fixture = GLM5NextMixedInventoryFixture
        private let headPrefix = "model.language_model.layers.0.self_attn"

        @Test func storageSelectionIsClosedOverTheDeclaredPolicy() throws {
            let q4 = try GLM5NextInventoryFixture.configuration(q4: true)
            #expect(try GLM5NextWeightInventory.packedStorage(declaredBy: q4) == .vontraQ4)
            #expect(
                try GLM5NextWeightInventory.packedStorage(
                    declaredBy: Fixture.recipeConfiguration()) == .affineMixed)
            #expect(throws: GLM5NextLoadError.self) {
                try GLM5NextWeightInventory.packedStorage(
                    declaredBy: GLM5NextInventoryFixture.configuration())
            }
            #expect(GLM5NextWeightInventory.Storage.affineMixed.isPacked)
            #expect(GLM5NextWeightInventory.Storage.vontraQ4.isPacked)
            #expect(!GLM5NextWeightInventory.Storage.float32.isPacked)
        }

        @Test func vontraQ4LayoutIsTheFloatingSchemaAtQ4Group64() throws {
            let config = try GLM5NextInventoryFixture.configuration(q4: true)
            try expectPackedLayout(Fixture.inventory(config, .vontraQ4), bits: [:])
        }

        @Test func mixedRecipeDeclaresExactPerModuleShapesAndBytes() throws {
            let config = try Fixture.recipeConfiguration()
            let recipe = Fixture.recipe(config.textConfig)
            #expect(recipe.count == 61)
            let mixed = try Fixture.inventory(config, .affineMixed)
            try expectPackedLayout(mixed, bits: recipe)
            let q4 = try Fixture.inventory(
                GLM5NextInventoryFixture.configuration(q4: true), .vontraQ4)
            #expect(Set(mixed.entries.keys) == Set(q4.entries.keys))

            func entry(_ name: String) throws -> GLM5NextWeightInventory.Entry {
                try #require(mixed.entries[name])
            }
            let embedding = try entry("model.language_model.embed_tokens.weight")
            #expect(embedding.tensor.shape == [32, 16] && embedding.tensor.dtype == "U32")
            #expect(embedding.packing == GLM5NextWeightInventory.Packing(bits: 8, groupSize: 64))
            #expect(try entry("lm_head.weight").tensor.shape == [32, 16])
            #expect(try entry(headPrefix + ".q_proj.weight").tensor.shape == [64, 12])
            #expect(try entry(headPrefix + ".q_proj.scales").tensor.shape == [64, 1])
            #expect(try entry(headPrefix + ".q_proj.biases").packing?.bits == 6)
            #expect(try entry(headPrefix + ".b_proj.weight").tensor.shape == [1, 16])
            let expert = try entry("model.language_model.layers.3.mlp.experts.1.down_proj.weight")
            #expect(expert.tensor.shape == [64, 8] && expert.packing == .q4Group64)
            let projection = try entry("model.language_model.layers.4.eh_proj.weight")
            #expect(projection.tensor.shape == [64, 32] && projection.part == .mtp)
            #expect(
                try entry("model.language_model.layers.4.eh_proj.scales").tensor.shape == [64, 2])
            let router = try entry("model.language_model.layers.3.mlp.gate.weight")
            #expect(router.tensor.dtype == "BF16" && router.packing == nil)
            #expect(try entry("model.language_model.layers.3.hc_attn_base").tensor.dtype == "F32")
            #expect(try entry("model.visual.blocks.0.attn.qkv.weight").packing == nil)

            // Scales and offsets keep their group64 widths: only packed words grow.
            var delta: [GLM5NextWeightInventory.Part: Int] = [:]
            for (module, bits) in recipe {
                let packed = try #require(q4.entries[module + ".weight"])
                let words = packed.tensor.shape[0] * packed.tensor.shape[1] * 8 * (bits - 4) / 8
                delta[packed.part, default: 0] += words
            }
            for part in GLM5NextWeightInventory.Part.allCases {
                #expect(mixed.bytes(in: part) - q4.bytes(in: part) == delta[part, default: 0])
            }
            #expect(delta[.vision] == nil && delta[.mtp, default: 0] > 0)
        }

        @Test func undeclaredUnknownAndExpertOverridesAreRejectedBeforePayload() throws {
            let text = try GLM5NextInventoryFixture.configuration(q4: true).textConfig
            let recipe = Fixture.recipe(text)
            func reject(_ extra: [String: Int], _ error: GLM5NextLoadError) throws {
                let config = try Fixture.configuration(
                    quantization: Fixture.policy(recipe.merging(extra) { $1 }))
                #expect(throws: error) {
                    try GLM5NextWeightInventory.expectedTensors(
                        configuration: config, storage: .affineMixed, shard: "x.safetensors")
                }
                #expect(throws: error) {
                    try GLM5NextWeightInventory.validate(
                        configuration: config, tensors: [], storage: .affineMixed)
                }
            }
            // Keys are artifact module prefixes, not exporter or tensor names.
            for unknown in [
                "language_model.lm_head", "lm_head.weight",
                "model.language_model.layers.0.input_layernorm",
                "model.language_model.layers.3.mlp.gate",
                "model.language_model.layers.3.self_attn.indexer.index_kpool_compress_gate",
                "model.language_model.layers.3.self_attn.indexer.k_norm",
                "model.language_model.layers.99.self_attn.q_proj",
                "model.visual.blocks.0.attn.qkv", "model.visual.merger.proj",
            ] {
                try reject([unknown: 8], .unexpected("quantization override \(unknown)"))
            }
            let expert = "model.language_model.layers.3.mlp.experts.0.gate_proj"
            try reject([expert: 8], .invalid("routed expert override \(expert)"))
            let mtpExpert = "model.language_model.layers.4.mlp.experts.1.down_proj"
            try reject([mtpExpert: 8], .invalid("routed expert override \(mtpExpert)"))
        }

        @Test func wrongWidthsGroupsDefaultsSkipsAndMetadataAreRejected() throws {
            let text = try GLM5NextInventoryFixture.configuration(q4: true).textConfig
            let recipe = Fixture.recipe(text)
            func rejectPolicy(
                _ policy: [String: Any], _ error: GLM5NextLoadError,
                quantizationConfig: [String: Any]? = nil
            ) throws {
                let config = try Fixture.configuration(
                    quantization: policy, quantizationConfig: quantizationConfig)
                #expect(throws: error) {
                    try GLM5NextWeightInventory.expectedTensors(
                        configuration: config, storage: .affineMixed, shard: "x.safetensors")
                }
            }
            // Redundant Q4 declarations are refused too: one encoding per layout.
            for bits in [2, 3, 4, 5] {
                try rejectPolicy(
                    Fixture.policy(recipe.merging(["lm_head": bits]) { $1 }),
                    .invalid("quantization override lm_head"))
            }
            for group in [32, 128] {
                try rejectPolicy(
                    Fixture.policy(["lm_head": 8], groupSize: group),
                    .invalid("quantization override lm_head"))
            }
            let policyError = GLM5NextLoadError.invalid(
                "mixed storage requires an affine Q4/group64 default and module overrides only")
            try rejectPolicy(Fixture.policy([:]), policyError)
            for (bits, group) in [(8, 64), (4, 32), (4, 128), (6, 64)] {
                var policy = Fixture.policy(recipe)
                policy["bits"] = bits
                policy["group_size"] = group
                try rejectPolicy(policy, policyError)
            }
            var skipped = Fixture.policy(recipe)
            skipped["model.language_model.layers.3.mlp.gate"] = false
            try rejectPolicy(skipped, policyError)
            var labelled = Fixture.policy(recipe)
            labelled["quant_method"] = "affine"
            try rejectPolicy(labelled, policyError)
            try rejectPolicy(
                Fixture.policy(recipe),
                .invalid("mixed storage cannot retain source FP8 metadata"),
                quantizationConfig: [
                    "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                    "weight_block_size": [128, 128], "modules_to_not_convert": [String](),
                ])
            // The two namespaces must agree; this is the configuration decoder's check.
            #expect(throws: GLM5NextConfigurationError.self) {
                try Fixture.configuration(
                    quantization: Fixture.policy(recipe),
                    quantizationConfig: Fixture.policy(recipe.merging(["lm_head": 6]) { $1 }))
            }
        }

        @Test func storagesAreMutuallyExclusiveAndHeadersMustMatchTheDeclaration() throws {
            let q4 = try GLM5NextInventoryFixture.configuration(q4: true)
            let mixed = try Fixture.recipeConfiguration()
            let recipe = Fixture.recipe(mixed.textConfig)
            func headers(
                _ config: GLM5NextConfiguration, _ storage: GLM5NextWeightInventory.Storage
            )
                throws -> [GLM5NextWeightInventory.Tensor]
            {
                try GLM5NextWeightInventory.expectedTensors(
                    configuration: config, storage: storage, shard: "x.safetensors")
            }
            let declared = try headers(mixed, .affineMixed)
            #expect(
                throws: GLM5NextLoadError.invalid(
                    "Vontra requires exact affine Q4/group64 metadata")
            ) {
                try GLM5NextWeightInventory.validate(
                    configuration: mixed, tensors: declared, storage: .vontraQ4)
            }
            #expect(
                throws: GLM5NextLoadError.invalid(
                    "floating fixture cannot declare packed quantization")
            ) {
                try GLM5NextWeightInventory.validate(
                    configuration: mixed, tensors: declared, storage: .float32)
            }
            #expect(throws: GLM5NextLoadError.self) {
                try GLM5NextWeightInventory.validate(
                    configuration: q4, tensors: headers(q4, .vontraQ4), storage: .affineMixed)
            }
            // Uniform Q4 words under a mixed declaration are a shape mismatch.
            #expect(throws: GLM5NextLoadError.invalid("header lm_head.weight")) {
                try GLM5NextWeightInventory.validate(
                    configuration: mixed, tensors: headers(q4, .vontraQ4), storage: .affineMixed)
            }
            // A module stored at 8 bits but not declared keeps the Q4 expectation.
            let undeclared = try Fixture.configuration(
                quantization: Fixture.policy(recipe.filter { $0.key != "lm_head" }))
            #expect(throws: GLM5NextLoadError.invalid("header lm_head.weight")) {
                try GLM5NextWeightInventory.validate(
                    configuration: undeclared, tensors: declared, storage: .affineMixed)
            }
            // Declared 8 bits, stored 6 bits.
            let q = headPrefix + ".q_proj"
            let widened = try Fixture.configuration(
                quantization: Fixture.policy(recipe.merging([q: 8]) { $1 }))
            #expect(throws: GLM5NextLoadError.invalid("header \(q).weight")) {
                try GLM5NextWeightInventory.validate(
                    configuration: widened, tensors: declared, storage: .affineMixed)
            }
            #expect(throws: GLM5NextLoadError.missing("\(q).scales")) {
                try GLM5NextWeightInventory.validate(
                    configuration: mixed, tensors: declared.filter { $0.name != q + ".scales" },
                    storage: .affineMixed)
            }
            #expect(throws: GLM5NextLoadError.self) {
                try GLM5NextWeightInventory.validate(
                    configuration: mixed,
                    tensors: declared.map {
                        $0.name == q + ".scales"
                            ? .init(name: $0.name, shape: $0.shape, dtype: "F32", shard: $0.shard)
                            : $0
                    }, storage: .affineMixed)
            }
        }

        /// Each packed matrix is the floating schema's matrix at its declared
        /// width; every other tensor keeps the floating schema's name and shape.
        private func expectPackedLayout(
            _ inventory: GLM5NextWeightInventory, bits declared: [String: Int]
        ) throws {
            let floating = Dictionary(
                uniqueKeysWithValues: try GLM5NextWeightInventory.expectedTensors(
                    configuration: inventory.configuration, storage: .float32,
                    shard: "floating.safetensors"
                ).map { ($0.name, $0) })
            var modules = Set<String>()
            for entry in inventory.entries.values where entry.tensor.dtype == "U32" {
                let module = String(entry.tensor.name.dropLast(".weight".count))
                let source = try #require(floating[module + ".weight"])
                let (output, input) = (source.shape[0], source.shape[1])
                let bits = declared[module] ?? 4
                let packing = GLM5NextWeightInventory.Packing(bits: bits, groupSize: 64)
                #expect(entry.tensor.name.hasSuffix(".weight"))
                #expect(entry.tensor.shape == [output, input * bits / 32])
                for suffix in [".weight", ".scales", ".biases"] {
                    #expect(inventory.entries[module + suffix]?.packing == packing)
                }
                for suffix in [".scales", ".biases"] {
                    let offsets = try #require(inventory.entries[module + suffix]?.tensor)
                    #expect(offsets.shape == [output, input / 64] && offsets.dtype == "BF16")
                }
                modules.insert(module)
            }
            #expect(Set(declared.keys).isSubset(of: modules))
            for entry in inventory.entries.values where entry.packing == nil {
                #expect(floating[entry.tensor.name]?.shape == entry.tensor.shape)
                #expect(["BF16", "F32"].contains(entry.tensor.dtype))
            }
            #expect(inventory.entries.count == floating.count + 2 * modules.count)
        }
    }
}
