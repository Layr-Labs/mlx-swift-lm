import Foundation
import Testing

@testable import MLXLLM

/// Tiny counterpart of the NVFP4 hybrid artifact (`glm53-nvfp4-layout.md`):
/// target dense MLPs and all routed experts NVFP4, everything else affine Q4/g64
/// or stored. JSON/headers only, so it is safe in the no-MLX UnitTests suite.
enum GLM5NextNVFP4InventoryFixture {
    static var nvfp4: [String: Any] { ["bits": 4, "group_size": 16, "mode": "nvfp4"] }

    static func policy(
        _ text: GLM5NextTextConfiguration, nvfp4Modules: Set<String>? = nil,
        affine: [String: Int] = [:]
    ) -> [String: Any] {
        var policy = GLM5NextMixedInventoryFixture.policy(affine)
        for module in nvfp4Modules ?? GLM5NextWeightInventory.nvfp4Modules(text) {
            policy[module] = nvfp4
        }
        return policy
    }

    static func configuration(nvfp4Modules: Set<String>? = nil, affine: [String: Int] = [:])
        throws -> GLM5NextConfiguration
    {
        let text = try GLM5NextInventoryFixture.configuration(q4: true).textConfig
        return try GLM5NextMixedInventoryFixture.configuration(
            quantization: policy(text, nvfp4Modules: nvfp4Modules, affine: affine))
    }

    static func inventory(_ configuration: GLM5NextConfiguration) throws
        -> GLM5NextWeightInventory
    {
        try GLM5NextMixedInventoryFixture.inventory(configuration, .nvfp4Hybrid)
    }
}

extension UnitTests {
    @Suite struct GLM5NextNVFP4InventoryTests {
        private typealias Fixture = GLM5NextNVFP4InventoryFixture
        private let p0 = "model.language_model.layers.0"
        private let p3 = "model.language_model.layers.3"
        private let mtp = "model.language_model.layers.4"

        @Test func declaredModulesAreExactlyTheTargetFeedForwardWeights() throws {
            let text = try GLM5NextInventoryFixture.configuration(q4: true).textConfig
            var expected = Set<String>()
            for name in ["gate_proj", "up_proj", "down_proj"] {
                for layer in 0 ..< 3 { expected.insert("model.language_model.layers.\(layer).mlp.\(name)") }
                for expert in 0 ..< 2 {
                    expected.insert(p3 + ".mlp.experts.\(expert).\(name)")
                    expected.insert(mtp + ".mlp.experts.\(expert).\(name)")
                }
            }
            #expect(GLM5NextWeightInventory.nvfp4Modules(text) == expected)
            let config = try Fixture.configuration()
            #expect(try GLM5NextWeightInventory.packedStorage(declaredBy: config) == .nvfp4Hybrid)
            #expect(GLM5NextWeightInventory.Storage.nvfp4Hybrid.isPacked)
        }

        @Test func hybridSchemaDeclaresExactNVFP4AndAffineTensors() throws {
            let inventory = try Fixture.inventory(Fixture.configuration())
            func entry(_ name: String) throws -> GLM5NextWeightInventory.Entry {
                try #require(inventory.entries[name])
            }
            let nvfp4 = GLM5NextWeightInventory.Packing(bits: 4, groupSize: 16, mode: .nvfp4)
            for module in [
                p0 + ".mlp.down_proj", p3 + ".mlp.experts.1.gate_proj",
                mtp + ".mlp.experts.0.down_proj",
            ] {
                let weight = try entry(module + ".weight")
                #expect(weight.tensor.dtype == "U32" && weight.tensor.shape == [64, 8])
                #expect(weight.packing == nvfp4)
                let scales = try entry(module + ".scales")
                #expect(scales.tensor.dtype == "U8" && scales.tensor.shape == [64, 4])
                #expect(scales.bytes == 256 && scales.packing == nvfp4)
                let global = try entry(module + ".weight_scale_2")
                #expect(global.tensor.dtype == "F32" && global.tensor.shape == [])
                #expect(global.bytes == 4 && global.packing == nvfp4)
                #expect(inventory.entries[module + ".biases"] == nil)
            }
            // Shared experts, attention and heads stay affine Q4/g64.
            for module in [
                mtp + ".mlp.shared_experts.up_proj",
                p3 + ".mlp.shared_experts.gate_proj", p3 + ".self_attn.kv_b_proj",
                p0 + ".self_attn.q_proj", "lm_head", "model.language_model.embed_tokens",
                mtp + ".eh_proj",
            ] {
                #expect(try entry(module + ".weight").packing == .q4Group64, "\(module)")
                #expect(try entry(module + ".biases").tensor.dtype == "BF16", "\(module)")
                #expect(inventory.entries[module + ".weight_scale_2"] == nil)
            }
            #expect(try entry(p3 + ".mlp.gate.weight").packing == nil)
            #expect(try entry("model.visual.blocks.0.attn.qkv.weight").tensor.dtype == "BF16")

            // Same tensor set as the affine layout, except biases -> weight_scale_2,
            // and the same 4.5 bits per weight at these widths plus the scalars.
            let q4 = try GLM5NextMixedInventoryFixture.inventory(
                GLM5NextInventoryFixture.configuration(q4: true), .vontraQ4)
            let renamed = Set(
                q4.entries.keys.map { name -> String in
                    let module = String(name.dropLast(".biases".count))
                    guard name.hasSuffix(".biases"),
                        inventory.entries[module + ".weight_scale_2"] != nil
                    else { return name }
                    return module + ".weight_scale_2"
                })
            #expect(Set(inventory.entries.keys) == renamed)
            let nvfp4Matrices = inventory.entries.values.filter {
                $0.tensor.name.hasSuffix(".weight_scale_2")
            }.count
            #expect(nvfp4Matrices == 9 + 6 + 6)
            // Affine g64 scales+biases (2 x 2 bytes / 64) equal FP8 scales (1 byte / 16).
            #expect(inventory.bytes(in: .text) == q4.bytes(in: .text) + 4 * 15)
            #expect(inventory.bytes(in: .mtp) == q4.bytes(in: .mtp) + 4 * 6)
            #expect(inventory.bytes(in: .vision) == q4.bytes(in: .vision))
        }

        @Test func partialForeignOrMisplacedNVFP4DeclarationsAreRejected() throws {
            let text = try GLM5NextInventoryFixture.configuration(q4: true).textConfig
            let exact = GLM5NextWeightInventory.nvfp4Modules(text)
            let mismatch = GLM5NextLoadError.invalid(
                "nvfp4 storage requires exactly the target dense MLP and routed expert modules")
            for modules in [
                exact.subtracting([p3 + ".mlp.experts.1.down_proj"]),
                exact.subtracting([mtp + ".mlp.experts.0.gate_proj"]),
                exact.subtracting([p0 + ".mlp.up_proj"]),
                exact.union([p3 + ".mlp.experts"]),
                exact.union([p3 + ".mlp.shared_experts.gate_proj"]),
                exact.union([mtp + ".mlp.shared_experts.gate_proj"]),
                exact.union([mtp + ".eh_proj"]),
                exact.union([p0 + ".self_attn.q_proj"]),
                exact.union(["lm_head"]),
            ] {
                let config = try Fixture.configuration(nvfp4Modules: modules)
                #expect(throws: mismatch) { try Fixture.inventory(config) }
            }
            // Affine overrides keep the mixed rule; expert banks cannot be affine-overridden.
            let widened = try Fixture.inventory(
                Fixture.configuration(affine: [p0 + ".self_attn.q_proj": 8]))
            #expect(widened.entries[p0 + ".self_attn.q_proj.weight"]?.tensor.shape == [64, 16])
            let shared = mtp + ".mlp.shared_experts.gate_proj"
            let sharedWide = try Fixture.inventory(Fixture.configuration(affine: [shared: 8]))
            #expect(sharedWide.entries[shared + ".weight"]?.packing?.bits == 8)
            // NVFP4 requires its fixed geometry, at decode time.
            var policy = Fixture.policy(text)
            policy[p3 + ".mlp.experts.0.up_proj"] = ["bits": 4, "group_size": 32, "mode": "nvfp4"]
            #expect(throws: GLM5NextConfigurationError.self) {
                try GLM5NextMixedInventoryFixture.configuration(quantization: policy)
            }
            // Other storages refuse NVFP4 declarations, and the hybrid refuses none.
            let config = try Fixture.configuration()
            let declared = try GLM5NextWeightInventory.expectedTensors(
                configuration: config, storage: .nvfp4Hybrid, shard: "x.safetensors")
            #expect(throws: GLM5NextLoadError.self) {
                try GLM5NextWeightInventory.validate(
                    configuration: config, tensors: declared, storage: .affineMixed)
            }
            #expect(throws: GLM5NextLoadError.self) {
                try GLM5NextWeightInventory.validate(
                    configuration: config, tensors: declared, storage: .vontraQ4)
            }
            let q4 = try GLM5NextInventoryFixture.configuration(q4: true)
            #expect(throws: GLM5NextLoadError.self) {
                try GLM5NextWeightInventory.expectedTensors(
                    configuration: q4, storage: .nvfp4Hybrid, shard: "x.safetensors")
            }
        }

        @Test func headersMustMatchTheNVFP4DeclarationExactly() throws {
            let config = try Fixture.configuration()
            let declared = try GLM5NextWeightInventory.expectedTensors(
                configuration: config, storage: .nvfp4Hybrid, shard: "x.safetensors")
            func validate(_ tensors: [GLM5NextWeightInventory.Tensor]) throws {
                _ = try GLM5NextWeightInventory.validate(
                    configuration: config, tensors: tensors, storage: .nvfp4Hybrid)
            }
            try validate(declared)
            func replacing(
                _ name: String, shape: [Int]? = nil, dtype: String? = nil
            ) -> [GLM5NextWeightInventory.Tensor] {
                declared.map {
                    $0.name == name
                        ? .init(
                            name: name, shape: shape ?? $0.shape, dtype: dtype ?? $0.dtype,
                            shard: $0.shard)
                        : $0
                }
            }
            let module = p3 + ".mlp.experts.0.down_proj"
            for (name, shape, dtype) in [
                (module + ".scales", nil, "BF16"), (module + ".scales", [64, 1], nil),
                (module + ".weight_scale_2", [1], nil), (module + ".weight_scale_2", nil, "BF16"),
                (module + ".weight", [64, 16], nil), (module + ".weight", nil, "U8"),
            ] as [(String, [Int]?, String?)] {
                #expect(throws: GLM5NextLoadError.invalid("header \(name)")) {
                    try validate(replacing(name, shape: shape, dtype: dtype))
                }
            }
            #expect(throws: GLM5NextLoadError.missing(module + ".weight_scale_2")) {
                try validate(declared.filter { $0.name != module + ".weight_scale_2" })
            }
            // NVIDIA's activation scale and affine offsets are not part of the layout.
            for extra in [module + ".input_scale", module + ".biases", module + ".weight_scale"] {
                #expect(throws: GLM5NextLoadError.unexpected(extra)) {
                    try validate(
                        declared + [.init(name: extra, shape: [1], dtype: "F32", shard: "x")])
                }
            }
        }
    }
}
