import CryptoKit
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Tiny on-disk checkpoint whose packed matrices are real MLX affine
/// quantizations, at each stored width, of known FP32 source values. No real
/// artifact paths, downloads, conversion or cache directories are touched.
final class GLM5NextMixedArtifactFixture {
    let directory: URL
    let shards = ["weights-0.safetensors", "weights-1.safetensors"]
    /// Exact stored bytes of every tensor.
    private(set) var payload: [String: Data] = [:]
    /// FP32 source of every packed module, before quantization.
    private(set) var sources: [String: MLXArray] = [:]

    /// config.json holds `declared`; `layout` selects the stored headers.
    init(
        declared: GLM5NextConfiguration,
        layout: (GLM5NextConfiguration, GLM5NextWeightInventory.Storage)? = nil
    ) throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "glm5-next-mixed-\(UUID())")
        let (configuration, storage) = layout ?? (declared, .affineMixed)
        let inventory = try GLM5NextWeightInventory.validate(
            configuration: configuration,
            tensors: GLM5NextWeightInventory.expectedTensors(
                configuration: configuration, storage: storage, shard: shards[0]),
            storage: storage)
        for entry in inventory.entries.values {
            let name = entry.tensor.name
            guard let packing = entry.packing else {
                payload[name] = Self.floating(entry.tensor).asData(access: .copy).data
                continue
            }
            guard name.hasSuffix(".weight") else { continue }
            let module = String(name.dropLast(".weight".count))
            let shape = [entry.tensor.shape[0], entry.tensor.shape[1] * 32 / packing.bits]
            let source = Self.values(module, shape)
            let (words, scales, offsets) = MLX.quantized(
                source, groupSize: packing.groupSize, bits: packing.bits)
            guard let offsets else { throw GLM5NextLoadError.missing(module + ".biases") }
            sources[module] = source
            payload[name] = words.asData(access: .copy).data
            payload[module + ".scales"] = scales.asType(.bfloat16).asData(access: .copy).data
            payload[module + ".biases"] = offsets.asType(.bfloat16).asData(access: .copy).data
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        try JSONEncoder().encode(declared).write(to: url("config.json"))
        let names = inventory.entries.keys.sorted()
        var weightMap: [String: String] = [:]
        var total = 0
        for (index, shard) in shards.enumerated() {
            var header: [String: Any] = ["__metadata__": ["format": "mlx"]]
            var data = Data()
            for name in names.enumerated().filter({ $0.offset % shards.count == index }).map(
                \.element)
            {
                let tensor = inventory.entries[name]!.tensor
                let start = data.count
                data.append(payload[name]!)
                header[name] = [
                    "shape": tensor.shape, "dtype": tensor.dtype,
                    "data_offsets": [start, data.count],
                ]
                weightMap[name] = shard
            }
            total += data.count
            try GLM5NextLocalArtifactFixture.safetensors(header: header, payload: data)
                .write(to: url(shard))
        }
        let index: [String: Any] = ["metadata": ["total_size": total], "weight_map": weightMap]
        try JSONSerialization.data(withJSONObject: index, options: .sortedKeys)
            .write(to: url("model.safetensors.index.json"))
    }

    deinit { try? FileManager.default.removeItem(at: directory) }
    func url(_ name: String) -> URL { directory.appendingPathComponent(name) }

    func loader(
        storage: GLM5NextWeightInventory.Storage, roots: GLM5NextArtifactLoadRoots? = nil
    ) throws -> GLM5NextArtifactLoader {
        let files = try Dictionary(
            uniqueKeysWithValues: (["config.json", "model.safetensors.index.json"] + shards).map {
                name in
                let bytes = try Data(contentsOf: url(name))
                let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
                return (name, GLM5NextArtifactIdentity.File(bytes: bytes.count, sha256: digest))
            })
        return try GLM5NextArtifactLoader(
            directory: directory,
            expectedIdentity: .init(
                repository: "synthetic/glm5-next-mixed",
                revision: String(repeating: "b", count: 40), files: files),
            storage: storage, constructionRoots: roots)
    }

    static func values(_ name: String, _ shape: [Int]) -> MLXArray {
        let salt = name.utf8.reduce(0) { ($0 + Int($1)) % 17 }
        let count = shape.reduce(1, *)
        return MLXArray((0 ..< count).map { Float(($0 * 7 + salt) % 23 - 11) / 88 }, shape)
    }

    private static func floating(_ tensor: GLM5NextWeightInventory.Tensor) -> MLXArray {
        let name = tensor.name
        let constant: Float?
        if name.contains("norm.weight") || name.hasSuffix("_scale") {
            constant = 1
        } else if name.hasSuffix("A_log") {
            constant = -0.7
        } else if name.hasSuffix("dt_bias") {
            constant = -0.2
        } else if name.hasSuffix(".bias") || name.hasSuffix("_base") {
            constant = 0
        } else {
            constant = nil
        }
        let value = constant.map { MLXArray.ones(tensor.shape) * $0 } ?? values(name, tensor.shape)
        return value.asType(tensor.dtype == "F32" ? .float32 : .bfloat16)
    }
}

extension KernelTests {
    @Suite struct GLM5NextMixedArtifactLoaderTests {
        private typealias Mixed = GLM5NextMixedInventoryFixture

        @Test func mixedArtifactRoundTripsDeclaredWidthsWithoutRequantization() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextMixedArtifactFixture(
                    declared: Mixed.recipeConfiguration())
                let loader = try fixture.loader(storage: .affineMixed)
                let resources = loader.resources
                // LOAD accounting is driven by the stored mixed sizes, not a Q4 formula.
                #expect(
                    resources.textBytes + resources.mtpBytes + resources.visionBytes
                        == fixture.payload.values.reduce(0) { $0 + $1.count })
                #expect(resources.largestTensorBytes == fixture.payload.values.map(\.count).max())
                // Two experts' packed words form the largest stacked destination.
                let member = "model.language_model.layers.3.mlp.experts.0.down_proj.weight"
                #expect(resources.largestExpertStackBytes == 2 * fixture.payload[member]!.count)

                let text = try loader.loadTextModel()
                let embedding = try #require(text.model.embedTokens as? GLM5NextPackedEmbedding)
                #expect(embedding.bits == 8 && embedding.groupSize == 64)
                #expect(embedding.shape == (32, 64))
                let embedSource = try #require(fixture.sources["model.language_model.embed_tokens"])
                #expect(
                    embedding.weight.asData(access: .copy).data
                        == fixture.payload["model.language_model.embed_tokens.weight"])
                let rows = MLXArray([Int32(2), 5])
                expectClose(text.embed(rows.reshaped(1, 2))[0], embedSource[rows], bits: 8)

                try expectStored("lm_head", text.lmHead, bits: 8, fixture)
                let kda = try #require(text.model.layers[0].attention as? GLM5NextKDASublayer)
                let p0 = "model.language_model.layers.0"
                try expectStored(p0 + ".self_attn.q_proj", kda.q, bits: 6, fixture)
                try expectStored(p0 + ".self_attn.o_proj", kda.output, bits: 6, fixture)
                try expectStored(p0 + ".self_attn.f_a_proj", kda.fA, bits: 8, fixture)
                let dense = try #require(text.model.layers[0].mlp as? GLM5NextMLP)
                try expectStored(p0 + ".mlp.down_proj", dense.downProj, bits: 8, fixture)
                let p3 = "model.language_model.layers.3"
                let dsa = try #require(text.model.layers[3].sparseAttention)
                try expectStored(p3 + ".self_attn.kv_b_proj", dsa.kvB, bits: 8, fixture)
                try expectStored(
                    p3 + ".self_attn.indexer.weights_proj", dsa.indexer.headWeights, bits: 8,
                    fixture)
                let moe = try #require(text.model.layers[3].mlp as? GLM5NextMoE)
                try expectStored(
                    p3 + ".mlp.shared_experts.up_proj", moe.sharedExperts.upProj, bits: 8, fixture)
                let experts = try #require(moe.experts.gateProj as? QuantizedSwitchLinear)
                #expect(experts.bits == 4 && experts.groupSize == 64 && experts.mode == .affine)
                let members = (0 ..< 2).map { p3 + ".mlp.experts.\($0).gate_proj.weight" }
                #expect(
                    experts.weight.asData(access: .copy).data
                        == members.reduce(Data()) { $0 + fixture.payload[$1]! })

                let output = try text.forward(
                    tokens: MLXArray([Int32(2), 3, 4], [1, 3]), cache: nil)
                #expect(output.logits.shape == [1, 3, 32])
                #expect(
                    output.logits.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite })

                // MTP geometry checks the declared 8-bit eh_proj width.
                #expect(!loader.unresolved(in: .mtp).isEmpty)
                let mtp = try loader.loadMTP(target: text)
                #expect(mtp.targetIdentity == ObjectIdentifier(text))
                #expect(loader.unresolved(in: .mtp).isEmpty)
                #expect(!loader.unresolved(in: .vision).isEmpty)
                try loader.verifyUnchanged()
            }
        }

        @Test func vontraQ4ArtifactStillLoadsUniformQ4Group64() throws {
            try Device.withDefaultDevice(.cpu) {
                let q4 = try GLM5NextInventoryFixture.configuration(q4: true)
                let fixture = try GLM5NextMixedArtifactFixture(
                    declared: q4, layout: (q4, .vontraQ4))
                let loader = try fixture.loader(storage: .vontraQ4)
                let text = try loader.loadTextModel()
                let embedding = try #require(text.model.embedTokens as? GLM5NextPackedEmbedding)
                #expect(embedding.bits == 4 && embedding.groupSize == 64)
                #expect(embedding.shape == (32, 64))
                try expectStored("lm_head", text.lmHead, bits: 4, fixture)
                let kda = try #require(text.model.layers[0].attention as? GLM5NextKDASublayer)
                try expectStored(
                    "model.language_model.layers.0.self_attn.q_proj", kda.q, bits: 4, fixture)
                let moe = try #require(text.model.layers[3].mlp as? GLM5NextMoE)
                #expect((moe.experts.downProj as? QuantizedSwitchLinear)?.bits == 4)
                _ = try loader.loadMTP(target: text)
                #expect(loader.unresolved(in: .mtp).isEmpty)
            }
        }

        @Test func declarationMismatchesFailBeforeAnyTensorIsMaterialized() throws {
            try Device.withDefaultDevice(.cpu) {
                let q4 = try GLM5NextInventoryFixture.configuration(q4: true)
                let mixed = try Mixed.recipeConfiguration()
                let recipe = Mixed.recipe(mixed.textConfig)
                let roots = GLM5NextArtifactLoadRoots()

                // Uniform Q4 words under the mixed declaration.
                let narrow = try GLM5NextMixedArtifactFixture(
                    declared: mixed, layout: (q4, .vontraQ4))
                #expect(throws: GLM5NextLoadError.self) {
                    try narrow.loader(storage: .affineMixed, roots: roots)
                }
                // Exporter-namespace key over otherwise exact headers.
                let foreign = try GLM5NextMixedArtifactFixture(
                    declared: Mixed.configuration(
                        quantization: Mixed.policy(
                            recipe.merging(["language_model.lm_head": 8]) { $1 })),
                    layout: (mixed, .affineMixed))
                #expect(
                    throws: GLM5NextLoadError.unexpected(
                        "quantization override language_model.lm_head")
                ) {
                    try foreign.loader(storage: .affineMixed, roots: roots)
                }
                // Each artifact is accepted by exactly one storage.
                let exact = try GLM5NextMixedArtifactFixture(declared: mixed)
                #expect(
                    throws: GLM5NextLoadError.invalid(
                        "Vontra requires exact affine Q4/group64 metadata")
                ) {
                    try exact.loader(storage: .vontraQ4, roots: roots)
                }
                #expect(
                    throws: GLM5NextLoadError.invalid(
                        "floating fixture cannot declare packed quantization")
                ) {
                    try exact.loader(storage: .float32, roots: roots)
                }
                let uniform = try GLM5NextMixedArtifactFixture(
                    declared: q4, layout: (q4, .vontraQ4))
                #expect(throws: GLM5NextLoadError.self) {
                    try uniform.loader(storage: .affineMixed, roots: roots)
                }
                #expect(roots.retainedArrayCount == 0 && roots.retainedOwnerCount == 0)
                _ = try exact.loader(storage: .affineMixed, roots: roots)
                _ = try uniform.loader(storage: .vontraQ4, roots: roots)
                #expect(roots.retainedArrayCount == 0)
            }
        }

        @Test func mixedConstructionKeepsBoundedMaterializationContract() throws {
            try Device.withDefaultDevice(.cpu) {
                let config = try Mixed.recipeConfiguration()
                let inventory = try Mixed.inventory(config, .affineMixed)
                var reads = 0
                func tensor(_ name: String) throws -> MLXArray {
                    reads += 1
                    let header = try #require(inventory.entries[name]).tensor
                    let dtype: DType =
                        header.dtype == "U32"
                        ? .uint32 : header.dtype == "F32" ? .float32 : .bfloat16
                    let one =
                        name.hasSuffix(".scales") || name.contains("norm.weight")
                        || name.hasSuffix("_scale")
                    return one
                        ? MLXArray.ones(header.shape, dtype: dtype)
                        : MLXArray.zeros(header.shape, dtype: dtype)
                }
                func materialize(_ array: MLXArray) throws {
                    try withError { error in
                        eval(array)
                        try error.check()
                    }
                }
                #expect(throws: GLM5NextLoadError.self) {
                    try GLM5NextTextModel(
                        configuration: config.textConfig, inventory: inventory, tensor: tensor,
                        materializeExpertStack: materialize)
                }
                #expect(throws: GLM5NextLoadError.self) {
                    try GLM5NextDSAAttention(
                        configuration: config.textConfig,
                        sourcePrefix: "model.language_model.layers.3.self_attn",
                        inventory: inventory, tensor: tensor)
                }
                #expect(reads == 0)
                let target = try GLM5NextTextModel(
                    configuration: config.textConfig, inventory: inventory, tensor: tensor,
                    materializeExpertStack: materialize, releaseSourceTensors: { _ in })
                reads = 0
                #expect(throws: GLM5NextMTPError.self) {
                    try GLM5NextMTP(target: target, tensor: tensor, releaseSourceTensors: { _ in })
                }
                #expect(reads == 0)
                var released = Set<String>()
                _ = try GLM5NextMTP(
                    target: target, tensor: tensor, materializeExpertStack: materialize,
                    releaseSourceTensors: { released.formUnion($0) })
                #expect(released == Set(inventory.entries(in: .mtp).map { $0.tensor.name }))
            }
        }

        /// Stored words, scales and offsets reach the module unchanged, and its
        /// declared width reproduces the FP32 source within quantization error.
        private func expectStored(
            _ module: String, _ layer: Linear, bits: Int, _ fixture: GLM5NextMixedArtifactFixture
        ) throws {
            let packed = try #require(layer as? QuantizedLinear)
            #expect(packed.bits == bits && packed.groupSize == 64 && packed.mode == .affine)
            #expect(packed.weight.asData(access: .copy).data == fixture.payload[module + ".weight"])
            #expect(packed.scales.asData(access: .copy).data == fixture.payload[module + ".scales"])
            let offsets = try #require(packed.biases)
            #expect(offsets.asData(access: .copy).data == fixture.payload[module + ".biases"])
            let restored = dequantized(
                packed.weight, scales: packed.scales, biases: offsets,
                groupSize: packed.groupSize, bits: packed.bits)
            expectClose(restored, try #require(fixture.sources[module]), bits: bits)
        }

        private func expectClose(_ actual: MLXArray, _ source: MLXArray, bits: Int) {
            #expect(actual.shape == source.shape)
            let step = 0.25 / Float((1 << bits) - 1)
            let error = abs(actual.asType(.float32) - source).max().item(Float.self)
            #expect(error <= 2 * step + 0.002)
        }
    }
}
