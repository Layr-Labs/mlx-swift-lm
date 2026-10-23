import CryptoKit
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// NVIDIA ModelOpt NVFP4 encoding of `source` through MLX's own quantizer:
/// MLX `nvfp4` words and FP8-E4M3 scales, and ModelOpt's `weight_scale_2`
/// (`amax / (448 * 6)`), so `W ≈ e2m1 * fp8(scales) * weight_scale_2`.
struct GLM5NextNVFP4Matrix {
    let words: MLXArray
    let scales: MLXArray
    /// F32 scalar.
    let weightScale2: MLXArray

    init(_ source: MLXArray) {
        let amax = abs(source).max().asType(.float32).reshaped(1)
        let packed = MLX.quantized(
            source.asType(.float32), groupSize: 16, bits: 4, mode: .nvfp4, globalScale: amax)
        words = packed.wq
        scales = packed.scales
        weightScale2 = amax.reshaped([]) / Float(448 * 6)
        eval(words, scales, weightScale2)
    }

    /// The exact matrix the NVFP4 encoding stores, in FP32.
    var dequantized: MLXArray {
        MLX.dequantized(
            words, scales: scales, biases: nil, groupSize: 16, bits: 4, mode: .nvfp4,
            dtype: .float32) * weightScale2
    }
}

/// Tiny on-disk hybrid checkpoint: real MLX NVFP4 target dense MLPs and routed
/// experts (each expert at its own magnitude, so a wrong per-expert global
/// scale is visible), real affine Q4/g64 for every other packed matrix, and
/// the exact FP32 matrices each stored encoding represents.
final class GLM5NextNVFP4ArtifactFixture {
    let directory: URL
    let declared: GLM5NextConfiguration
    let inventory: GLM5NextWeightInventory
    let shards = ["weights-0.safetensors", "weights-1.safetensors"]
    private(set) var stored: [String: MLXArray] = [:]
    /// Module or floating tensor name -> the FP32 value the artifact encodes.
    private(set) var represented: [String: MLXArray] = [:]

    init(configuration: GLM5NextConfiguration? = nil) throws {
        declared = try configuration ?? GLM5NextNVFP4InventoryFixture.configuration()
        inventory = try GLM5NextNVFP4InventoryFixture.inventory(declared)
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "glm5-next-nvfp4-\(UUID())")
        try Device.withDefaultDevice(.cpu) {
            for entry in inventory.entries.values {
                let name = entry.tensor.name
                guard let packing = entry.packing else {
                    let value = Self.floating(entry.tensor)
                    stored[name] = value
                    represented[name] = value.asType(.float32)
                    continue
                }
                guard name.hasSuffix(".weight") else { continue }
                let module = String(name.dropLast(".weight".count))
                let shape = [entry.tensor.shape[0], entry.tensor.shape[1] * 32 / packing.bits]
                let source = Self.values(module, shape)
                if packing.mode == .nvfp4 {
                    let matrix = GLM5NextNVFP4Matrix(source)
                    stored[name] = matrix.words
                    stored[module + ".scales"] = matrix.scales
                    stored[module + ".weight_scale_2"] = matrix.weightScale2
                    represented[module] = matrix.dequantized
                } else {
                    let (words, scales, offsets) = MLX.quantized(
                        source, groupSize: packing.groupSize, bits: packing.bits)
                    let (s, b) = (scales.asType(.bfloat16), offsets!.asType(.bfloat16))
                    stored[name] = words
                    stored[module + ".scales"] = s
                    stored[module + ".biases"] = b
                    represented[module] = MLX.dequantized(
                        words, scales: s, biases: b, groupSize: packing.groupSize,
                        bits: packing.bits, dtype: .float32)
                }
            }
            eval(Array(stored.values) + Array(represented.values))
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        try JSONEncoder().encode(declared).write(to: url("config.json"))
        var weightMap: [String: String] = [:]
        var total = 0
        let names = inventory.entries.keys.sorted()
        for (index, shard) in shards.enumerated() {
            var header: [String: Any] = ["__metadata__": ["format": "mlx"]]
            var data = Data()
            for name in names.enumerated().filter({ $0.offset % shards.count == index })
                .map(\.element)
            {
                let tensor = inventory.entries[name]!.tensor
                let start = data.count
                data.append(try payload(name))
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
    func payload(_ name: String) throws -> Data {
        guard let value = stored[name] else { throw GLM5NextLoadError.missing(name) }
        return value.asData(access: .copy).data
    }

    func loader(
        storage: GLM5NextWeightInventory.Storage = .nvfp4Hybrid,
        policy: GLM5NextExecutionPolicy = .reference
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
                repository: "synthetic/glm5-next-nvfp4",
                revision: String(repeating: "c", count: 40), files: files),
            storage: storage, executionPolicy: policy)
    }

    /// The same model as unquantized FP32 storage holding exactly the
    /// matrices the hybrid encodes; `swapExpertGlobalScales` gives routed
    /// experts 0 and 1 each other's `weight_scale_2` (a negative control).
    func floatingReference(swapExpertGlobalScales: Bool = false) throws -> GLM5NextTextModel {
        var root =
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(declared))
            as! [String: Any]
        root["quantization"] = nil
        root["quantization_config"] = nil
        let config = try JSONDecoder().decode(
            GLM5NextConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
        let floating = try GLM5NextWeightInventory.validate(
            configuration: config,
            tensors: GLM5NextWeightInventory.expectedTensors(
                configuration: config, storage: .float32, shard: "reference.safetensors"),
            storage: .float32)
        return try GLM5NextTextModel(
            configuration: config.textConfig, inventory: floating,
            tensor: { name in
                if let value = self.represented[name] { return value }
                let module = String(name.dropLast(".weight".count))
                guard let value = self.represented[module] else {
                    throw GLM5NextLoadError.missing(name)
                }
                let swapped = [".experts.0.": ".experts.1.", ".experts.1.": ".experts.0."]
                    .first { module.contains($0.key) }
                if swapExpertGlobalScales, let swapped,
                    let own = self.stored[module + ".weight_scale_2"],
                    let other = self.stored[
                        module.replacingOccurrences(of: swapped.key, with: swapped.value)
                            + ".weight_scale_2"]
                {
                    return value / own * other
                }
                return value
            })
    }

    /// Exactly representable in BF16; each routed expert has its own magnitude.
    static func values(_ name: String, _ shape: [Int]) -> MLXArray {
        let salt = name.utf8.reduce(0) { ($0 + Int($1)) % 17 }
        var magnitude: Float = 1
        if let range = name.range(of: #"\.experts\.[0-9]+\."#, options: .regularExpression) {
            magnitude = Float(Int(name[range].filter(\.isNumber))! + 1) * 0.75
        }
        let count = shape.reduce(1, *)
        return MLXArray(
            (0 ..< count).map { Float(($0 * 7 + salt) % 23 - 11) / 64 * magnitude }, shape)
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
    @Suite(.serialized)
    struct GLM5NextNVFP4Tests {
        private let tokens = MLXArray([Int32(2), 5, 3, 9, 7, 1, 4], [1, 7])
        private let p0 = "model.language_model.layers.0"
        private let p3 = "model.language_model.layers.3"

        private static func maxError(_ actual: MLXArray, _ expected: MLXArray) -> Float {
            abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
        }

        private static func fence(_ arrays: [MLXArray]) throws {
            try withError { error in
                eval(arrays)
                StreamOrDevice.default.stream.synchronize()
                try error.check()
            }
        }

        /// Each expert's output carries its own global scale, through every
        /// gather route (unsorted broadcast and index-aligned sorted rows).
        @Test(arguments: [DeviceType.cpu, .gpu])
        func switchLinearAppliesEachExpertsGlobalScale(device: DeviceType) throws {
            try Device.withDefaultDevice(Device(device)) {
                let (experts, output, input) = (3, 32, 64)
                let matrices = (0 ..< experts).map {
                    GLM5NextNVFP4Matrix(
                        GLM5NextNVFP4ArtifactFixture.values(
                            "bank.experts.\($0).gate_proj", [output, input]))
                }
                let bank = try QuantizedSwitchLinear(
                    nvfp4InputDims: input, outputDims: output, numExperts: experts,
                    weight: stacked(matrices.map(\.words)),
                    scales: stacked(matrices.map(\.scales)),
                    globalScales: stacked(matrices.map(\.weightScale2)))
                #expect(bank.mode == .nvfp4 && bank.biases == nil && bank.groupSize == 16)
                let dense = stacked(matrices.map(\.dequantized))
                let x = GLM5NextNVFP4ArtifactFixture.values("x", [4, input])
                let indices = MLXArray([Int32(2), 0, 1, 2, 0, 0, 1, 2], [4, 2])

                let actual = bank(x.expandedDimensions(axes: [-2, -3]), indices)
                let expected = gatherMM(
                    x.expandedDimensions(axes: [-2, -3]), dense.swappedAxes(-1, -2),
                    rhsIndices: indices)
                try Self.fence([actual, expected])
                #expect(actual.shape == [4, 2, 1, output])
                #expect(Self.maxError(actual, expected) < 1e-4)

                let flat = indices.flattened()
                let order = argSort(flat)
                let rows = x[order.floorDivide(2)].expandedDimensions(axis: -2)
                let sorted = bank(rows, flat[order], sortedIndices: true)
                let reference = gatherMM(
                    rows, dense.swappedAxes(-1, -2), rhsIndices: flat[order])
                try Self.fence([sorted, reference])
                #expect(Self.maxError(sorted, reference) < 1e-4)

                // BF16 activations keep the per-expert scale unrounded.
                let half = bank(x.asType(.bfloat16).expandedDimensions(axes: [-2, -3]), indices)
                #expect(half.dtype == .bfloat16)
                #expect(Self.maxError(half, expected) < 0.05 * abs(expected).max().item(Float.self))
            }
        }

        @Test func packedConstructionRefusesForeignNVFP4Geometry() throws {
            try Device.withDefaultDevice(.cpu) {
                let matrix = GLM5NextNVFP4Matrix(
                    GLM5NextNVFP4ArtifactFixture.values("m", [16, 32]))
                let words = matrix.words.expandedDimensions(axis: 0)
                let scales = matrix.scales.expandedDimensions(axis: 0)
                func make(_ w: MLXArray, _ s: MLXArray, _ g: MLXArray) throws {
                    _ = try QuantizedSwitchLinear(
                        nvfp4InputDims: 32, outputDims: 16, numExperts: 1, weight: w, scales: s,
                        globalScales: g)
                }
                try make(words, scales, matrix.weightScale2.reshaped(1))
                #expect(throws: QuantizedSwitchLinear.LoadError.invalidScales) {
                    try make(words, scales.asType(.bfloat16), matrix.weightScale2.reshaped(1))
                }
                #expect(throws: QuantizedSwitchLinear.LoadError.invalidScales) {
                    try make(words, scales, MLXArray([Float(1), 2]))
                }
                #expect(throws: QuantizedSwitchLinear.LoadError.invalidWeight) {
                    try make(words[0..., 0..., ..<2], scales, matrix.weightScale2.reshaped(1))
                }
                #expect(throws: GLM5NextLoadError.invalid("nvfp4 global scale")) {
                    try GLM5NextNVFP4Linear(
                        weight: matrix.words, scales: matrix.scales,
                        globalScale: MLXArray([Float.nan]))
                }
                #expect(throws: GLM5NextLoadError.invalid("nvfp4 global scale")) {
                    try GLM5NextNVFP4Linear(
                        weight: matrix.words, scales: matrix.scales,
                        globalScale: MLXArray([Float(0)]))
                }
                let linear = try GLM5NextNVFP4Linear(
                    weight: matrix.words, scales: matrix.scales, globalScale: matrix.weightScale2)
                #expect(linear.shape == (16, 32))
                let x = GLM5NextNVFP4ArtifactFixture.values("x", [3, 32])
                #expect(Self.maxError(linear(x), matmul(x, matrix.dequantized.T)) < 1e-4)
                // A fused gate/up buffer would drop the per-matrix scale.
                let fused = try GLM5NextFusedProjection.make([linear, linear])
                #expect(fused == nil)
            }
        }

        @Test func hybridArtifactRoundTripsStoredBytesAndQuotesExactBytes() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextNVFP4ArtifactFixture()
                #expect(throws: GLM5NextLoadError.self) {
                    try fixture.loader(storage: .affineMixed)
                }
                #expect(throws: GLM5NextLoadError.self) { try fixture.loader(storage: .vontraQ4) }
                let loader = try fixture.loader()
                let resources = loader.resources
                let total = try fixture.inventory.entries.keys.reduce(0) {
                    $0 + (try fixture.payload($1).count)
                }
                #expect(resources.textBytes + resources.mtpBytes + resources.visionBytes == total)
                let member = p3 + ".mlp.experts.0.down_proj.weight"
                let memberBytes = try fixture.payload(member).count
                #expect(resources.largestExpertStackBytes == 2 * memberBytes)

                let text = try loader.loadTextModel()
                let dense = try #require(text.model.layers[0].mlp as? GLM5NextMLP)
                for (name, projection) in [
                    ("gate_proj", dense.gateProj), ("up_proj", dense.upProj),
                    ("down_proj", dense.downProj),
                ] {
                    let module = p0 + ".mlp." + name
                    let packed = try #require(projection as? GLM5NextNVFP4Linear)
                    #expect(packed.mode == .nvfp4 && packed.biases == nil)
                    let words = try fixture.payload(module + ".weight")
                    let scales = try fixture.payload(module + ".scales")
                    let global = try fixture.payload(module + ".weight_scale_2")
                    #expect(packed.weight.asData(access: .copy).data == words)
                    #expect(packed.scales.asData(access: .copy).data == scales)
                    #expect(packed.weightScale2.asData(access: .copy).data == global)
                }
                let moe = try #require(text.model.layers[3].mlp as? GLM5NextMoE)
                #expect(!(moe.sharedExperts.gateProj is GLM5NextNVFP4Linear))
                #expect((moe.sharedExperts.gateProj as? QuantizedLinear)?.mode == .affine)
                for (name, bank) in [
                    ("gate_proj", moe.experts.gateProj), ("up_proj", moe.experts.upProj),
                    ("down_proj", moe.experts.downProj),
                ] {
                    let packed = try #require(bank as? QuantizedSwitchLinear)
                    #expect(packed.mode == .nvfp4 && packed.bits == 4 && packed.groupSize == 16)
                    let members = (0 ..< 2).map { p3 + ".mlp.experts.\($0).\(name)" }
                    func joined(_ suffix: String) throws -> Data {
                        try members.reduce(Data()) { $0 + (try fixture.payload($1 + suffix)) }
                    }
                    let (words, scales, global) = try (
                        joined(".weight"), joined(".scales"), joined(".weight_scale_2")
                    )
                    #expect(packed.weight.asData(access: .copy).data == words)
                    #expect(packed.scales.asData(access: .copy).data == scales)
                    let globals = try #require(packed.globalScales)
                    #expect(globals.shape == [2])
                    #expect(globals.asData(access: .copy).data == global)
                }
                let output = try text.forward(tokens: tokens, cache: nil)
                #expect(output.logits.shape == [1, 7, 32])
                #expect(
                    output.logits.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite })

                // The MTP layer's routed experts are NVFP4 too.
                let mtp = try loader.loadMTP(target: text)
                #expect(loader.unresolved(in: .mtp).isEmpty)
                let mtpBanks = mtp.modules().compactMap { $0 as? QuantizedSwitchLinear }
                #expect(mtpBanks.count == 3)
                #expect(mtpBanks.allSatisfy { $0.mode == .nvfp4 && $0.globalScales?.shape == [2] })
                try loader.verifyUnchanged()
            }
        }

        /// The hybrid's feed-forward modules compute the matrices it stores:
        /// each matches an FP32 module holding exactly those matrices to BF16
        /// activation rounding, while exchanging two experts' global scales does
        /// not. (Whole-model logits compound BF16 rounding through four mHC
        /// layers, so the bound is applied per module.)
        @Test func hybridModulesMatchTheFloatingModulesOfTheirEncodedMatrices() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextNVFP4ArtifactFixture()
                let hybrid = try fixture.loader().loadTextModel()
                let reference = try fixture.floatingReference()
                let swapped = try fixture.floatingReference(swapExpertGlobalScales: true)
                let x = GLM5NextNVFP4ArtifactFixture.values("x", [5, 64])
                for index in [0, 1, 2, 3] {
                    func apply(_ model: GLM5NextTextModel, _ input: MLXArray) throws -> MLXArray {
                        guard let layer = model.model.layers[index].mlp as? UnaryLayer else {
                            throw GLM5NextLoadError.invalid("feed-forward layer \(index)")
                        }
                        return layer(input)
                    }
                    let actual = try apply(hybrid, x.asType(.bfloat16))
                    let expected = try apply(reference, x)
                    let control = try apply(swapped, x)
                    try Self.fence([actual, expected, control])
                    let span = abs(expected).max().item(Float.self)
                    let error = Self.maxError(actual, expected)
                    #expect(actual.dtype == .bfloat16)
                    #expect(
                        error.isFinite && error < 0.03 * span, "layer \(index) \(error)/\(span)")
                    if index == 3 {
                        #expect(Self.maxError(control, expected) > 5 * error)
                    }
                }
                let output = try hybrid.forward(tokens: tokens, cache: nil).logits
                #expect(output.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite })
            }
        }

        /// Every serving member runs NVFP4 layers (natively or through the
        /// MLX gather fallback) and agrees with the hybrid's reference graph.
        @Test(arguments: [GLM5NextExecutionPolicy.serving, .init(routedExperts: .fusedDecode)])
        func servingPolicyMatchesReferenceOnGPU(policy: GLM5NextExecutionPolicy) throws {
            try Device.withDefaultDevice(.gpu) {
                let fixture = try GLM5NextNVFP4ArtifactFixture()
                let reference = try fixture.loader().loadTextModel()
                let native = try fixture.loader(policy: policy).loadTextModel()
                #expect(native.executionPolicy == policy)
                let expected = try reference.forward(tokens: tokens, cache: nil)
                let actual = try native.forward(tokens: tokens, cache: nil)
                try Self.fence([expected.logits, actual.logits])
                let span = abs(expected.logits).max().item(Float.self)
                let prefill = Self.maxError(actual.logits, expected.logits)
                #expect(prefill < 0.03 * span, "prefill \(prefill) span \(span)")

                let cache = native.newTextCache()
                let steps = try (0 ..< tokens.dim(1)).map {
                    try native.forward(tokens: tokens[0..., $0 ..< $0 + 1], cache: cache).logits
                }
                let tokenwise = concatenated(steps, axis: 1)
                try Self.fence([tokenwise])
                let decode = Self.maxError(tokenwise, expected.logits)
                #expect(decode < 0.03 * span, "decode \(decode) span \(span)")
            }
        }
    }
}
