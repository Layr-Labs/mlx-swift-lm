// Copyright © 2026 Eigen Labs.

import CryptoKit
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A synthetic sharded checkpoint in the producer's layout: BF16, calibrated
/// FP8 and weight-only NVFP4 projections side by side, plus vision tensors.
/// Bytes are generated on the host; nothing is downloaded.
struct MuseGlimmerTinyCheckpoint {
    struct Tensor {
        var dtype: String
        var shape: [Int]
        var bytes: [UInt8]
    }

    let configuration: MuseGlimmerTextModelConfiguration
    var tensors: [String: Tensor] = [:]
    /// Format per model path, as written.
    var formats: [String: MuseGlimmerInstallation.ProjectionFormat] = [:]

    private static func random(_ count: Int, seed: UInt64, _ map: (UInt64) -> UInt8) -> [UInt8] {
        var state = seed &* 0x9E37_79B9_7F4A_7C15 &+ 1
        return (0 ..< count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return map(state >> 33)
        }
    }

    static func scalar(_ value: Float) -> Tensor {
        Tensor(dtype: "F32", shape: [], bytes: withUnsafeBytes(of: value.bitPattern) { Array($0) })
    }

    static func bf16(_ shape: [Int], seed: UInt64, scale: Float, offset: Float = 0) -> Tensor {
        let values = (MuseGlimmerFixture.hostTensor(shape, seed: seed) * scale + offset)
            .asType(.bfloat16)
        return Tensor(dtype: "BF16", shape: shape, bytes: Array(values.asData(access: .copy).data))
    }

    /// The same values as `bf16`, stored as `dtype` (BF16 or F32).
    static func dense(
        _ shape: [Int], seed: UInt64, scale: Float, offset: Float = 0, dtype: DType
    ) -> Tensor {
        let tensor = bf16(shape, seed: seed, scale: scale, offset: offset)
        guard dtype == .float32 else { return tensor }
        let values = MLXArray(Data(tensor.bytes), shape, dtype: .bfloat16).asType(.float32)
        return Tensor(dtype: "F32", shape: shape, bytes: Array(values.asData(access: .copy).data))
    }

    /// `order` rotates through the projection slots; `head` is the format of
    /// `lm_head` (NVFP4 as published).
    init(
        layers: Int = 3, hidden: Int = 64, intermediate: Int = 128, window: Int = 64,
        order: [MuseGlimmerInstallation.ProjectionFormat] = [.fp8, .nvfp4, .dense],
        head: MuseGlimmerInstallation.ProjectionFormat = .nvfp4, dtype: DType = .bfloat16
    ) throws {
        configuration = try MuseGlimmerFixture.configuration(
            MuseGlimmerFixture.pagedText(
                window: window, layers: layers, hidden: hidden, intermediate: intermediate))
        let c = configuration
        let trunk = "model.language_model."
        tensors[trunk + "embed_tokens.weight"] = Self.dense(
            [c.vocabularySize, c.hiddenSize], seed: 1, scale: 1, dtype: dtype)
        tensors[trunk + "norm.weight"] = Self.dense(
            [c.hiddenSize], seed: 2, scale: 0.1, offset: 1, dtype: dtype)
        var seed: UInt64 = 10
        for layer in 0 ..< c.hiddenLayers {
            for norm in [
                "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ] {
                seed += 1
                tensors[trunk + "layers.\(layer).\(norm).weight"] = Self.dense(
                    [c.hiddenSize], seed: seed, scale: 0.1, dtype: dtype)
            }
        }
        for (index, slot) in MuseGlimmerInstallation.projectionSlots(c).enumerated() {
            seed += 1
            let format = slot.path == "lm_head" ? head : order[index % order.count]
            formats[slot.path] = format
            let prefix = MuseGlimmerInstallation.checkpointKey(slot.path)
            let fanIn = Float(slot.input).squareRoot()
            switch format {
            case .dense:
                tensors[prefix + ".weight"] = Self.dense(
                    [slot.output, slot.input], seed: seed, scale: 1.7 / fanIn, dtype: dtype)
            case .fp8:
                // E4M3 magnitudes in [1/8, 8): exponent field 4...9.
                tensors[prefix + ".weight"] = Tensor(
                    dtype: "F8_E4M3", shape: [slot.output, slot.input],
                    bytes: Self.random(slot.output * slot.input, seed: seed) {
                        UInt8(($0 & 1) << 7 | (4 + ($0 >> 1) % 6) << 3 | ($0 >> 5) & 7)
                    })
                tensors[prefix + ".weight_scale"] = Self.scalar(0.5 / fanIn)
                tensors[prefix + ".input_scale"] = Self.scalar(16.0 / 448)
            case .nvfp4:
                tensors[prefix + ".weight"] = Tensor(
                    dtype: "U8", shape: [slot.output, slot.input / 2],
                    bytes: Self.random(slot.output * slot.input / 2, seed: seed) {
                        UInt8(truncatingIfNeeded: $0)
                    })
                // Block scales in [1/4, 4): exponent field 5...8.
                tensors[prefix + ".weight_scale"] = Tensor(
                    dtype: "F8_E4M3", shape: [slot.output, slot.input / 16],
                    bytes: Self.random(slot.output * slot.input / 16, seed: seed &+ 7919) {
                        UInt8((5 + $0 % 4) << 3 | ($0 >> 4) & 7)
                    })
                tensors[prefix + ".weight_scale_2"] = Self.scalar(0.25 / fanIn)
            }
        }
        tensors["model.vision_projection.weight"] = Self.bf16(
            [c.hiddenSize, 8], seed: 900, scale: 1)
        tensors["model.vision_adapter.fc1.weight"] = Self.bf16([8, 8], seed: 901, scale: 1)
        tensors["model.vision_tower.ln_pre.bias"] = Self.bf16([8], seed: 902, scale: 1)
    }

    /// Two shards by sorted key order, an index, and their hashes.
    struct Written {
        let directory: URL
        let indexData: Data
        let indexSHA256: String
        let shardSHA256: [String: String]

        func files(overriding hashes: [String: String] = [:]) throws -> [String: ModelOptTensorFile]
        {
            var files = [String: ModelOptTensorFile]()
            for (name, hash) in shardSHA256 {
                files[name] = try ModelOptTensorFile(
                    url: directory.appendingPathComponent(name), sha256: hashes[name] ?? hash)
            }
            return files
        }

        func source() throws -> ModelOptIndexedCheckpoint {
            try ModelOptIndexedCheckpoint(
                indexData: indexData, indexSHA256: indexSHA256, files: try files())
        }
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// `unindexed` names tensors written to a shard but left out of the index.
    func write(unindexed: Set<String> = []) throws -> Written {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "muse-glimmer-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let keys = tensors.keys.sorted()
        let split = keys.count / 2
        let names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        var weightMap = [String: String]()
        var hashes = [String: String]()
        for (name, slice) in zip(names, [keys[..<split], keys[split...]]) {
            var header = [String: Any]()
            var payload = Data()
            for key in slice {
                let tensor = tensors[key]!
                header[key] = [
                    "dtype": tensor.dtype, "shape": tensor.shape,
                    "data_offsets": [payload.count, payload.count + tensor.bytes.count],
                ]
                payload.append(contentsOf: tensor.bytes)
                if !unindexed.contains(key) { weightMap[key] = name }
            }
            let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
            var length = UInt64(json.count).littleEndian
            var data = withUnsafeBytes(of: &length) { Data($0) }
            data.append(json)
            data.append(payload)
            try data.write(to: directory.appendingPathComponent(name))
            hashes[name] = Self.sha256(data)
        }
        let index = try JSONSerialization.data(
            withJSONObject: ["metadata": ["format": "pt"], "weight_map": weightMap],
            options: [.sortedKeys])
        try index.write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        return Written(
            directory: directory, indexData: index, indexSHA256: Self.sha256(index),
            shardSHA256: hashes)
    }
}

@Suite(.serialized, .tags(.integration))
struct MuseGlimmerInstallationTests {
    typealias Failure = MuseGlimmerInstallation.Failure

    private func stage(_ checkpoint: MuseGlimmerTinyCheckpoint) throws
        -> MuseGlimmerInstallation.Installed
    {
        let written = try checkpoint.write()
        defer { try? FileManager.default.removeItem(at: written.directory) }
        return try MuseGlimmerInstallation.stage(
            configuration: checkpoint.configuration, source: try written.source(),
            dtype: .bfloat16, activationOverflow: .museGlimmer)
    }

    private func array(_ tensor: MuseGlimmerTinyCheckpoint.Tensor) -> MLXArray {
        let dtype: DType =
            tensor.dtype == "BF16" ? .bfloat16 : tensor.dtype == "F32" ? .float32 : .uint8
        return MLXArray(Data(tensor.bytes), tensor.shape, dtype: dtype)
    }

    private func float(_ tensor: MuseGlimmerTinyCheckpoint.Tensor) -> Float {
        Float(
            bitPattern: tensor.bytes.enumerated().reduce(UInt32(0)) {
                $0 | UInt32($1.element) << (8 * $1.offset)
            })
    }

    @Test func mixedPrecisionCheckpointInstallsEveryProjectionInItsOwnFormat() throws {
        let checkpoint = try MuseGlimmerTinyCheckpoint()
        let installed = try stage(checkpoint)
        let receipt = installed.receipt
        #expect(receipt.plan.projections == checkpoint.formats)
        #expect(receipt.plan.count(.fp8) == 8 && receipt.plan.count(.dense) == 8)
        #expect(receipt.plan.count(.nvfp4) == 9)
        #expect(receipt.activationOverflow == .saturating)
        #expect(
            receipt.plan.visionTensors == [
                "model.vision_adapter.fc1.weight", "model.vision_projection.weight",
                "model.vision_tower.ln_pre.bias",
            ])
        #expect(Set(installed.visionSources.keys) == Set(receipt.plan.visionTensors))
        #expect(receipt.plan.visionPayloadBytes == (64 * 8 + 8 * 8 + 8) * 2)
        let textTensors = checkpoint.tensors.filter { !$0.key.hasPrefix("model.vision") }
        #expect(receipt.loadedTensorCount == textTensors.count)
        #expect(
            receipt.plan.textPayloadBytes == textTensors.values.reduce(0) { $0 + $1.bytes.count })

        // Each slot holds the layer type of its format, with the file's bytes.
        let model = installed.model
        let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        for (path, format) in checkpoint.formats {
            let prefix = MuseGlimmerInstallation.checkpointKey(path)
            let weight = try #require(checkpoint.tensors[prefix + ".weight"])
            switch format {
            case .dense:
                let layer = try #require(leaves[path] as? Linear)
                #expect(type(of: layer) == Linear.self && layer.bias == nil)
                #expect(MuseGlimmerFixture.sameBits(layer.weight, array(weight)))
            case .fp8:
                let layer = try #require(leaves[path] as? ModelOptCalibratedLinear)
                #expect(layer.format == .fp8 && layer.activationOverflow == .saturating)
                #expect(layer.weightDType == .bfloat16)
                let parameters = Dictionary(uniqueKeysWithValues: layer.parameters().flattened())
                #expect(
                    MuseGlimmerFixture.sameBits(try #require(parameters["weight"]), array(weight)))
                #expect(
                    parameters["weight_scale"]?.item(Float.self)
                        == float(try #require(checkpoint.tensors[prefix + ".weight_scale"])))
                #expect(
                    parameters["input_scale"]?.item(Float.self)
                        == float(try #require(checkpoint.tensors[prefix + ".input_scale"])))
            case .nvfp4:
                let layer = try #require(leaves[path] as? ModelOptWeightOnlyNVFP4Linear)
                #expect(
                    MuseGlimmerFixture.sameBits(layer.weight.view(dtype: .uint8), array(weight)))
                #expect(
                    MuseGlimmerFixture.sameBits(
                        layer.weightScale,
                        array(try #require(checkpoint.tensors[prefix + ".weight_scale"]))))
                #expect(
                    layer.weightScale2.item(Float.self)
                        == float(try #require(checkpoint.tensors[prefix + ".weight_scale_2"])))
            }
        }
        #expect(
            MuseGlimmerFixture.sameBits(
                model.tokenEmbeddingTable,
                array(try #require(checkpoint.tensors["model.language_model.embed_tokens.weight"])))
        )
        #expect(model.activationDType == .bfloat16)

        // The mixed tower runs, and its logits track a dense tower holding
        // the decoded weights. FP8 layers also quantize their activations to
        // eight bits, so this is a gross-error check, not parity.
        let tokens = MLXArray(MuseGlimmerFixture.tokens(40, vocabulary: 96)).reshaped(1, -1)
        let logits = model(tokens, cache: nil).asType(.float32)
        #expect(all(isFinite(logits)).item(Bool.self))
        let dense = MuseGlimmerTextModel(checkpoint.configuration)
        var weights = [String: MLXArray]()
        for (name, value) in model.parameters().flattened()
        where !checkpoint.formats.keys.contains(where: { name.hasPrefix($0 + ".") }) {
            weights[name] = value
        }
        for (path, format) in checkpoint.formats {
            switch format {
            case .dense: weights[path + ".weight"] = (leaves[path] as! Linear).weight
            case .fp8:
                let layer = leaves[path] as! ModelOptCalibratedLinear
                let parameters = Dictionary(uniqueKeysWithValues: layer.parameters().flattened())
                let decoded = try ModelOptFP8Weights(
                    e4m3Bytes: parameters["weight"]!,
                    tensorScale: parameters["weight_scale"]!.item(Float.self)
                ).decodedRows(0 ..< layer.outputDimensions, dtype: .float32)
                weights[path + ".weight"] = decoded
            case .nvfp4:
                weights[path + ".weight"] =
                    (leaves[path] as! ModelOptWeightOnlyNVFP4Linear).dequantizedWeight()
            }
        }
        try dense.update(
            parameters: .unflattened(weights.mapValues { $0.asType(.float32) }), verify: [.all])
        let reference = dense(tokens, cache: nil)
        let relative =
            sqrt(((logits - reference) * (logits - reference)).sum())
            / sqrt((reference * reference).sum())
        #expect(relative.item(Float.self) < 0.35)

        // Staging twice gives the same bits.
        let again = try stage(checkpoint)
        #expect(
            MuseGlimmerFixture.sameBits(
                again.model(tokens, cache: nil), model(tokens, cache: nil)))
    }

    @Test func mixedTowerServesThroughThePagedEngine() async throws {
        let model = try stage(try MuseGlimmerTinyCheckpoint()).model
        let prompt = MuseGlimmerFixture.tokens(150, vocabulary: 96)
        let expected = try MuseGlimmerFixture.eagerGreedyWithMargins(
            model, prompt: prompt, chunk: 64, steps: 10)
        let backend = try PagedKVBackend(
            layerKinds: model.cbv2LayerKinds,
            config: .init(
                capacityBytes: 64 << 20, dtype: .bfloat16, maxPrefillChunk: 64,
                nominalMaxSequenceLength: 512, segmentSizeBytes: 1 << 20,
                layerDTypes: model.kvStorageDTypes))
        let engine = EngineV2(
            model: CBv2SteppableLanguageModelAdapter(model), layerKinds: model.cbv2LayerKinds,
            backend: backend, cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 64, prefillChunkSize: 64,
                soloPrefillStripeTokens: 64, maxWaiting: 2, enablePrefixCache: false),
            admissionConfig: .init(watermarkFraction: 0))
        let result = await cbv2SchedCollect(
            try engine.submit(
                .init(
                    id: .init(1), promptTokens: prompt.map(Int.init),
                    sampling: .init(temperature: 0, seed: 0), maxTokens: 10)))
        await engine.shutdown()
        #expect(result.finishReason == .length)
        // A BF16 tower: the two attention kernels may part at a near-tie only.
        MuseGlimmerFixture.expectSameUntilNearTie(result.tokens, expected)
    }

    @Test func wrongHashesAndUnindexedTensorsAreRefusedBeforeAnyLoad() throws {
        let checkpoint = try MuseGlimmerTinyCheckpoint()
        let written = try checkpoint.write()
        defer { try? FileManager.default.removeItem(at: written.directory) }
        let zeros = String(repeating: "0", count: 64)
        #expect(throws: ModelOptIndexedCheckpoint.Failure.identity) {
            try ModelOptIndexedCheckpoint(
                indexData: written.indexData, indexSHA256: zeros, files: try written.files())
        }
        let shard = try #require(written.shardSHA256.keys.sorted().first)
        #expect(throws: ModelOptTensorFile.Failure.self) {
            try written.files(overriding: [shard: zeros])
        }
        // One shard missing from the set the index names.
        var partial = try written.files()
        partial.removeValue(forKey: shard)
        #expect(throws: ModelOptIndexedCheckpoint.Failure.shardSet) {
            try ModelOptIndexedCheckpoint(
                indexData: written.indexData, indexSHA256: written.indexSHA256, files: partial)
        }
        // The pinned entry refuses any artifact but the pinned one.
        #expect(throws: Failure.identity("checkpoint pin")) {
            try MuseGlimmerInstallation.stage(
                source: try written.source(), configuration: checkpoint.configuration)
        }
        #expect(throws: Failure.identity("model.safetensors.index.json")) {
            try MuseGlimmerInstallation.openPinnedSource(directory: written.directory)
        }
        // A tensor present in a shard but absent from the index.
        let hidden = "model.language_model.layers.0.input_layernorm.weight"
        let unindexed = try checkpoint.write(unindexed: [hidden])
        defer { try? FileManager.default.removeItem(at: unindexed.directory) }
        #expect(throws: ModelOptIndexedCheckpoint.Failure.inventory(hidden)) {
            try unindexed.source()
        }
    }

    /// The index, `config.json` and `hf_quant_config.json` are read as shards
    /// are opened: a symlink, a file that is not regular or one over its bound
    /// refuses the load before it is read, and a quantization record that is
    /// present but unreadable refuses it too.
    @Test func smallCheckpointFilesAreReadLikeShards() throws {
        typealias Index = ModelOptIndexedCheckpoint
        let files = FileManager.default
        let written = try MuseGlimmerTinyCheckpoint().write()
        let directory = written.directory
        defer { try? files.removeItem(at: directory) }
        func stage() throws { _ = try MuseGlimmerInstallation.stage(directory: directory) }
        func swapForSymlink(_ url: URL) throws -> URL {
            let target = directory.appendingPathComponent("target-" + url.lastPathComponent)
            try files.moveItem(at: url, to: target)
            try files.createSymbolicLink(at: url, withDestinationURL: target)
            return target
        }
        func restore(_ url: URL, from target: URL) throws {
            try files.removeItem(at: url)
            try files.moveItem(at: target, to: url)
        }
        let config = directory.appendingPathComponent("config.json")
        let configBytes = Data(#"{"text_config": {}}"#.utf8)
        try configBytes.write(to: config)
        // Readable files: the tiny index then fails the pin.
        #expect(throws: Failure.identity("model.safetensors.index.json")) { try stage() }

        let index = directory.appendingPathComponent("model.safetensors.index.json")
        var target = try swapForSymlink(index)
        #expect(throws: Index.Failure.unsafeFile(index.lastPathComponent)) { try stage() }
        #expect(throws: Index.Failure.unsafeFile(index.lastPathComponent)) {
            try MuseGlimmerInstallation.openPinnedSource(directory: directory)
        }
        try restore(index, from: target)

        target = try swapForSymlink(config)
        #expect(throws: Index.Failure.unsafeFile("config.json")) { try stage() }
        try restore(config, from: target)
        try Data(count: MuseGlimmerInstallation.maximumRecordBytes + 1).write(to: config)
        #expect(throws: Index.Failure.unsafeFile("config.json")) { try stage() }
        try configBytes.write(to: config)

        let record = directory.appendingPathComponent("hf_quant_config.json")
        try files.createDirectory(at: record, withIntermediateDirectories: false)
        #expect(throws: Index.Failure.unsafeFile(record.lastPathComponent)) { try stage() }
        try files.removeItem(at: record)
        try files.createSymbolicLink(
            at: record, withDestinationURL: directory.appendingPathComponent("absent.json"))
        #expect(throws: Index.Failure.unsafeFile(record.lastPathComponent)) { try stage() }
        try files.removeItem(at: record)
        try Data("{}".utf8).write(to: record)
        try files.setAttributes([.posixPermissions: 0], ofItemAtPath: record.path)
        #expect(throws: Index.Failure.unsafeFile(record.lastPathComponent)) { try stage() }
        try files.setAttributes([.posixPermissions: 0o600], ofItemAtPath: record.path)
        // A readable record: the load goes on to the pin.
        #expect(throws: Failure.identity("model.safetensors.index.json")) { try stage() }

        #expect(throws: Index.Failure.unsafeFile("config.json")) {
            try Index.readRegularFile(at: config, maximumBytes: configBytes.count - 1)
        }
        #expect(try Index.readRegularFile(at: config, maximumBytes: configBytes.count) == configBytes)
    }

    /// NVFP4's unscaled product can leave the FP16 range, so a tower with an
    /// NVFP4 projection refuses FP16 activations; one without plans in FP16.
    @Test func fp16ActivationsAreRefusedWithNVFP4Projections() throws {
        func plan(_ checkpoint: MuseGlimmerTinyCheckpoint, _ dtype: DType) throws
            -> MuseGlimmerInstallation.Plan
        {
            var checkpoint = checkpoint
            if dtype == .float16 {
                for (key, tensor) in checkpoint.tensors where tensor.dtype == "BF16" {
                    checkpoint.tensors[key]?.dtype = "F16"
                }
            }
            let written = try checkpoint.write()
            defer { try? FileManager.default.removeItem(at: written.directory) }
            return try MuseGlimmerInstallation.plan(
                configuration: checkpoint.configuration, source: try written.source(),
                dtype: dtype)
        }
        let mixed = try MuseGlimmerTinyCheckpoint()
        #expect(try plan(mixed, .bfloat16).count(.nvfp4) == 9)
        #expect(throws: Failure.dtype("F16 activations with NVFP4 projections")) {
            try plan(mixed, .float16)
        }
        let withoutNVFP4 = try MuseGlimmerTinyCheckpoint(order: [.fp8, .dense], head: .dense)
        let half = try plan(withoutNVFP4, .float16)
        #expect(half.count(.nvfp4) == 0 && half.count(.fp8) > 0)
    }

    @Test func missingExtraAndMisshapenTensorsRefuseTheLoad() throws {
        let base = try MuseGlimmerTinyCheckpoint()
        func expect(
            _ failure: Failure, _ mutate: (inout MuseGlimmerTinyCheckpoint) -> Void,
            sourceLocation: SourceLocation = #_sourceLocation
        ) {
            var checkpoint = base
            mutate(&checkpoint)
            #expect(throws: failure, sourceLocation: sourceLocation) { try stage(checkpoint) }
        }
        let fp8 = try #require(base.formats.first { $0.value == .fp8 && $0.key != "lm_head" }?.key)
        let nvfp4 = try #require(
            base.formats.first { $0.value == .nvfp4 && $0.key != "lm_head" }?.key)
        let dense = try #require(base.formats.first { $0.value == .dense }?.key)
        let key = MuseGlimmerInstallation.checkpointKey
        let scalar = MuseGlimmerTinyCheckpoint.scalar(1)

        // Missing sidecars and missing tensors.
        expect(.missing(key(fp8) + ".input_scale")) {
            $0.tensors.removeValue(forKey: key(fp8) + ".input_scale")
        }
        expect(.missing(key(fp8) + ".weight_scale")) {
            $0.tensors.removeValue(forKey: key(fp8) + ".weight_scale")
        }
        expect(.missing(key(nvfp4) + ".weight_scale_2")) {
            $0.tensors.removeValue(forKey: key(nvfp4) + ".weight_scale_2")
        }
        expect(.missing("lm_head.weight")) { $0.tensors.removeValue(forKey: "lm_head.weight") }
        expect(.missing("model.language_model.norm.weight")) {
            $0.tensors.removeValue(forKey: "model.language_model.norm.weight")
        }
        // Activation calibration on a weight-only NVFP4 projection, and other extras.
        expect(.unexpected(key(nvfp4) + ".input_scale")) {
            $0.tensors[key(nvfp4) + ".input_scale"] = scalar
        }
        expect(.unexpected(key(dense) + ".weight_scale")) {
            $0.tensors[key(dense) + ".weight_scale"] = scalar
        }
        expect(.unexpected(key(fp8) + ".bias")) {
            $0.tensors[key(fp8) + ".bias"] = MuseGlimmerTinyCheckpoint.bf16([4], seed: 1, scale: 1)
        }
        expect(.unexpected("model.language_model.layers.0.self_attn.k_scale")) {
            $0.tensors["model.language_model.layers.0.self_attn.k_scale"] = scalar
        }
        expect(.unexpected("model.language_model.layers.3.input_layernorm.weight")) {
            $0.tensors["model.language_model.layers.3.input_layernorm.weight"] =
                MuseGlimmerTinyCheckpoint.bf16([64], seed: 1, scale: 1)
        }
        expect(.unexpected("mtp.fc.weight")) {
            $0.tensors["mtp.fc.weight"] = MuseGlimmerTinyCheckpoint.bf16([4, 4], seed: 1, scale: 1)
        }
        // Wrong shapes.
        expect(.layout("model.language_model.norm.weight")) {
            $0.tensors["model.language_model.norm.weight"] = MuseGlimmerTinyCheckpoint.bf16(
                [32], seed: 1, scale: 1)
        }
        expect(.layout(key(dense) + ".weight")) {
            let shape = $0.tensors[key(dense) + ".weight"]!.shape
            $0.tensors[key(dense) + ".weight"] = MuseGlimmerTinyCheckpoint.bf16(
                [shape[1], shape[0]], seed: 1, scale: 1)
        }
        expect(.layout(key(nvfp4) + ".weight_scale")) {
            var scale = $0.tensors[key(nvfp4) + ".weight_scale"]!
            scale.shape = [scale.shape[0] * 2, scale.shape[1] / 2]
            $0.tensors[key(nvfp4) + ".weight_scale"] = scale
        }
        // Wrong dtypes: a float32 embedding in a BF16 load, a non-float vision tensor.
        expect(.dtype("model.language_model.embed_tokens.weight")) {
            var table = $0.tensors["model.language_model.embed_tokens.weight"]!
            table.dtype = "F32"
            table.bytes += table.bytes
            $0.tensors["model.language_model.embed_tokens.weight"] = table
        }
        expect(.dtype("model.vision_tower.ln_pre.bias")) {
            $0.tensors["model.vision_tower.ln_pre.bias"] = MuseGlimmerTinyCheckpoint.Tensor(
                dtype: "U8", shape: [8], bytes: Array(repeating: 0, count: 8))
        }
    }

    @Test func quantizationRecordIsCrossCheckedButNeverDrivesLoading() throws {
        let checkpoint = try MuseGlimmerTinyCheckpoint()
        let plan = try stage(checkpoint).receipt.plan
        func record(_ formats: [String: MuseGlimmerInstallation.ProjectionFormat]) throws -> Data {
            var layers = [String: Any]()
            for (path, format) in formats where format != .dense {
                layers[MuseGlimmerInstallation.checkpointKey(path)] =
                    format == .fp8
                    ? ["quant_algo": "FP8"] : ["quant_algo": "W4A16_NVFP4", "group_size": 16]
            }
            return try JSONSerialization.data(withJSONObject: [
                "producer": ["name": "modelopt"],
                "quantization": [
                    "quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": NSNull(),
                    "quantized_layers": layers,
                ],
            ])
        }
        try MuseGlimmerInstallation.crossCheck(
            quantizationRecord: record(checkpoint.formats), plan: plan)
        var swapped = checkpoint.formats
        let fp8 = try #require(swapped.first { $0.value == .fp8 }?.key)
        swapped[fp8] = .nvfp4
        #expect(throws: Failure.quantizationRecord(fp8)) {
            try MuseGlimmerInstallation.crossCheck(quantizationRecord: record(swapped), plan: plan)
        }
        var missing = checkpoint.formats
        missing[fp8] = .dense
        #expect(throws: Failure.quantizationRecord(fp8)) {
            try MuseGlimmerInstallation.crossCheck(quantizationRecord: record(missing), plan: plan)
        }
        #expect(throws: Failure.quantizationRecord("unreadable")) {
            try MuseGlimmerInstallation.crossCheck(quantizationRecord: Data("[]".utf8), plan: plan)
        }
    }

    @Test func pinNamesThePublishedArtifactAndArchitecture() throws {
        #expect(MuseGlimmerCheckpointPin.revision == "47818374517751c48c55cde2621594926b1888b6")
        #expect(MuseGlimmerCheckpointPin.shards.count == 3)
        for hash in MuseGlimmerCheckpointPin.shards.values + [MuseGlimmerCheckpointPin.index] {
            #expect(hash.count == 64 && hash.allSatisfy { $0.isHexDigit && !$0.isUppercase })
        }
        let pinned = MuseGlimmerInstallation.pinnedConfiguration
        #expect(pinned.hiddenLayers == 52 && pinned.hiddenSize == 6656)
        // 52 layers of 8 projections and the head: the 180 + 213 + 24 census.
        #expect(MuseGlimmerInstallation.projectionSlots(pinned).count == 417)
        #expect(ModelOptActivationOverflow.museGlimmer == .saturating)
    }
}
