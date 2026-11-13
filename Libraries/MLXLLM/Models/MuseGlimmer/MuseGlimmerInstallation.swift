// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Strict loading of NVIDIA's ModelOpt checkpoint into the native text tower.
///
/// Each projection's format comes from the dtype of its `weight` and the
/// sidecars beside it in the hash-verified shard headers: BF16 stays dense,
/// F8_E4M3 with `weight_scale` and `input_scale` is calibrated FP8 (W8A8), U8
/// with `weight_scale` and `weight_scale_2` is weight-only NVFP4 (W4A16).
/// A missing, extra or wrong-shaped tensor refuses the load. The vision tensors
/// are verified and accounted for but not loaded: this type installs text only.
///
/// This grants no serving, catalog or admission capability, and reserves no
/// memory: the host holds its own load admission around `stage`.
public enum MuseGlimmerInstallation {
    public enum Failure: Error, Equatable {
        /// The index, a shard or the architecture is not the pinned artifact's.
        case identity(String)
        /// A tensor the architecture needs is absent.
        case missing(String)
        /// A tensor is present that the architecture does not have.
        case unexpected(String)
        /// A tensor has the wrong shape or byte count.
        case layout(String)
        /// A tensor has a dtype this loader does not accept in that place.
        case dtype(String)
        /// `hf_quant_config.json` disagrees with the verified headers.
        case quantizationRecord(String)
    }

    public enum ProjectionFormat: String, Sendable, Equatable { case dense, fp8, nvfp4 }

    /// Header-only reading of a checkpoint against one architecture.
    public struct Plan: Sendable, Equatable {
        /// Format by model path, for example `model.layers.0.self_attn.q_proj` or `lm_head`.
        public let projections: [String: ProjectionFormat]
        /// Checkpoint keys of the embedding table and the norm scales.
        public let denseTensors: [String]
        /// Checkpoint keys of the vision tower, adapter and projection: verified, not loaded.
        public let visionTensors: [String]
        public let textPayloadBytes: Int
        public let visionPayloadBytes: Int

        public func count(_ format: ProjectionFormat) -> Int {
            projections.values.filter { $0 == format }.count
        }
    }

    public struct Receipt: Sendable {
        public let indexSHA256: String
        public let shardSHA256: [String: String]
        public let plan: Plan
        public let activationOverflow: ModelOptActivationOverflow
        /// Text tensors read from the shards.
        public let loadedTensorCount: Int
    }

    public final class Installed {
        public let model: MuseGlimmerTextModel
        public let receipt: Receipt
        /// The verified vision tensors, for a later loader of the vision stack.
        public let visionSources: [String: ModelOptTensorSource]

        init(
            model: MuseGlimmerTextModel, receipt: Receipt,
            visionSources: [String: ModelOptTensorSource]
        ) {
            self.model = model
            self.receipt = receipt
            self.visionSources = visionSources
        }
    }

    static let trunkPrefix = "model.language_model."
    static let visionPrefixes = [
        "model.vision_tower.", "model.vision_adapter.", "model.vision_projection.",
    ]

    /// Every projection slot of the tower: model path, input and output width.
    static func projectionSlots(_ c: MuseGlimmerTextModelConfiguration)
        -> [(path: String, input: Int, output: Int)]
    {
        let queryDim = c.attentionHeads * c.headDimensions
        let kvDim = c.kvHeads * c.headDimensions
        var slots: [(String, Int, Int)] = [("lm_head", c.hiddenSize, c.vocabularySize)]
        for layer in 0 ..< c.hiddenLayers {
            let prefix = "model.layers.\(layer)."
            slots += [
                (prefix + "self_attn.q_proj", c.hiddenSize, queryDim),
                (prefix + "self_attn.k_proj", c.hiddenSize, kvDim),
                (prefix + "self_attn.v_proj", c.hiddenSize, kvDim),
                (prefix + "self_attn.gate_proj", c.hiddenSize, queryDim),
                (prefix + "self_attn.o_proj", queryDim, c.hiddenSize),
                (prefix + "mlp.gate_proj", c.hiddenSize, c.intermediateSize),
                (prefix + "mlp.up_proj", c.hiddenSize, c.intermediateSize),
                (prefix + "mlp.down_proj", c.intermediateSize, c.hiddenSize),
            ]
        }
        return slots
    }

    /// `model.layers.0.x` in the tower is `model.language_model.layers.0.x`
    /// in the checkpoint; `lm_head` keeps its name.
    static func checkpointKey(_ modelPath: String) -> String {
        modelPath.hasPrefix("model.")
            ? trunkPrefix + modelPath.dropFirst("model.".count) : modelPath
    }

    static func modelKey(_ checkpointKey: String) -> String {
        checkpointKey.hasPrefix(trunkPrefix)
            ? "model." + checkpointKey.dropFirst(trunkPrefix.count) : checkpointKey
    }

    static func headerName(_ dtype: DType) throws -> String {
        switch dtype {
        case .bfloat16: return "BF16"
        case .float16: return "F16"
        case .float32: return "F32"
        default: throw Failure.dtype("activation dtype \(dtype)")
        }
    }

    // MARK: Plan

    /// Classifies every tensor of `source` from header metadata alone and
    /// checks coverage in both directions. Reads no payload.
    public static func plan(
        configuration c: MuseGlimmerTextModelConfiguration, source: ModelOptIndexedCheckpoint,
        dtype: DType = .bfloat16
    ) throws -> Plan {
        let dense = try headerName(dtype)
        var remaining = Set(source.tensors.keys)
        var textBytes = 0
        var denseTensors = [String]()
        func takeDense(_ key: String, shape: [Int]) throws {
            guard let tensor = source.tensors[key] else { throw Failure.missing(key) }
            guard tensor.layout.dtype == dense else { throw Failure.dtype(key) }
            guard tensor.layout.shape == shape else { throw Failure.layout(key) }
            remaining.remove(key)
            textBytes += tensor.layout.byteCount
            denseTensors.append(key)
        }
        try takeDense(trunkPrefix + "embed_tokens.weight", shape: [c.vocabularySize, c.hiddenSize])
        try takeDense(trunkPrefix + "norm.weight", shape: [c.hiddenSize])
        for layer in 0 ..< c.hiddenLayers {
            for norm in [
                "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ] {
                try takeDense(
                    trunkPrefix + "layers.\(layer).\(norm).weight", shape: [c.hiddenSize])
            }
        }
        var projections = [String: ProjectionFormat]()
        for slot in projectionSlots(c) {
            let prefix = checkpointKey(slot.path)
            let group = source.group(prefix)
            guard let weight = group["weight"] else { throw Failure.missing(prefix + ".weight") }
            let layouts = group.mapValues(\.layout)
            do {
                switch weight.layout.dtype {
                case dense:
                    _ = try ModelOptProjectionLayout(
                        tensors: layouts, inputDimensions: slot.input,
                        outputDimensions: slot.output)
                    projections[slot.path] = .dense
                case "F8_E4M3":
                    _ = try ModelOptProjectionLayout(
                        tensors: layouts, inputDimensions: slot.input,
                        outputDimensions: slot.output)
                    projections[slot.path] = .fp8
                case "U8":
                    _ = try ModelOptWeightOnlyNVFP4Layout(
                        tensors: layouts, inputDimensions: slot.input,
                        outputDimensions: slot.output)
                    projections[slot.path] = .nvfp4
                default: throw Failure.dtype(prefix + ".weight")
                }
            } catch let failure as ModelOptProjectionLayout.Failure {
                switch failure {
                case .missing(let leaf): throw Failure.missing(prefix + "." + leaf)
                case .unexpected(let leaf): throw Failure.unexpected(prefix + "." + leaf)
                case .invalidTensor(let leaf): throw Failure.layout(prefix + "." + leaf)
                case .invalidDimensions: throw Failure.layout(prefix)
                }
            }
            for tensor in group.values {
                remaining.remove(tensor.name)
                textBytes += tensor.layout.byteCount
            }
        }
        var vision = [String]()
        var visionBytes = 0
        for key in remaining.sorted() {
            guard visionPrefixes.contains(where: { key.hasPrefix($0) }) else {
                throw Failure.unexpected(key)
            }
            let layout = source.tensors[key]!.layout
            guard ["BF16", "F16", "F32"].contains(layout.dtype) else { throw Failure.dtype(key) }
            vision.append(key)
            visionBytes += layout.byteCount
        }
        return Plan(
            projections: projections, denseTensors: denseTensors.sorted(), visionTensors: vision,
            textPayloadBytes: textBytes, visionPayloadBytes: visionBytes)
    }

    /// Compares the producer's `hf_quant_config.json` with a plan derived
    /// from the verified headers. The JSON never drives loading; a
    /// disagreement means the artifact is not what was qualified.
    public static func crossCheck(quantizationRecord data: Data, plan: Plan) throws {
        guard
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let quantization = root["quantization"] as? [String: Any],
            let layers = quantization["quantized_layers"] as? [String: [String: Any]]
        else { throw Failure.quantizationRecord("unreadable") }
        guard
            quantization["kv_cache_quant_algo"] == nil
                || quantization["kv_cache_quant_algo"] is NSNull
        else { throw Failure.quantizationRecord("kv_cache_quant_algo") }
        var recorded = [String: ProjectionFormat]()
        for (key, entry) in layers {
            switch entry["quant_algo"] as? String {
            case "FP8": recorded[modelKey(key)] = .fp8
            case "W4A16_NVFP4":
                guard entry["group_size"] as? Int == ModelOptWeightOnlyNVFP4Layout.blockSize
                else { throw Failure.quantizationRecord(key) }
                recorded[modelKey(key)] = .nvfp4
            default: throw Failure.quantizationRecord(key)
            }
        }
        let observed = plan.projections.filter { $0.value != .dense }
        for key in Set(recorded.keys).union(observed.keys).sorted()
        where recorded[key] != observed[key] {
            throw Failure.quantizationRecord(key)
        }
    }

    // MARK: Stage

    /// The published decoder: the architecture the pinned weights belong to.
    public static let pinnedConfiguration: MuseGlimmerTextModelConfiguration = {
        // Every default of the configuration is the published value.
        try! JSONDecoder().decode(MuseGlimmerTextModelConfiguration.self, from: Data("{}".utf8))
    }()

    /// Opens the pinned artifact in `directory`: the index and every shard
    /// must match `MuseGlimmerCheckpointPin`. Hashing reads each shard once.
    public static func openPinnedSource(directory: URL) throws -> ModelOptIndexedCheckpoint {
        let indexData = try Data(
            contentsOf: directory.appendingPathComponent("model.safetensors.index.json"))
        let names: [String]
        do {
            names = try ModelOptIndexedCheckpoint.shardNames(
                indexData: indexData, indexSHA256: MuseGlimmerCheckpointPin.index)
        } catch ModelOptIndexedCheckpoint.Failure.identity {
            throw Failure.identity("model.safetensors.index.json")
        }
        guard Set(names) == Set(MuseGlimmerCheckpointPin.shards.keys) else {
            throw Failure.identity("shard set")
        }
        // One hash pass per shard, side by side.
        let results = ShardResults(count: names.count)
        DispatchQueue.concurrentPerform(iterations: names.count) { index in
            let name = names[index]
            results.set(
                index,
                Result {
                    try ModelOptTensorFile(
                        url: directory.appendingPathComponent(name),
                        sha256: MuseGlimmerCheckpointPin.shards[name]!)
                })
        }
        var files = [String: ModelOptTensorFile]()
        for (name, result) in zip(names, results.values) {
            do { files[name] = try result!.get() } catch ModelOptTensorFile.Failure.checksum {
                throw Failure.identity(name)
            }
        }
        return try ModelOptIndexedCheckpoint(
            indexData: indexData, indexSHA256: MuseGlimmerCheckpointPin.index, files: files)
    }

    private final class ShardResults: @unchecked Sendable {
        private let lock = NSLock()
        private var storage: [Result<ModelOptTensorFile, Error>?]
        init(count: Int) { storage = Array(repeating: nil, count: count) }
        func set(_ index: Int, _ value: Result<ModelOptTensorFile, Error>) {
            lock.lock()
            storage[index] = value
            lock.unlock()
        }
        var values: [Result<ModelOptTensorFile, Error>?] {
            lock.lock()
            defer { lock.unlock() }
            return storage
        }
    }

    /// Production entry: the pinned NVIDIA artifact with the published
    /// architecture, BF16 activations and saturating FP8 activation overflow.
    public static func stage(
        source: ModelOptIndexedCheckpoint, configuration: MuseGlimmerTextModelConfiguration
    ) throws -> Installed {
        guard source.indexSHA256 == MuseGlimmerCheckpointPin.index,
            source.shardSHA256 == MuseGlimmerCheckpointPin.shards
        else { throw Failure.identity("checkpoint pin") }
        guard configuration == pinnedConfiguration else { throw Failure.identity("architecture") }
        return try stage(
            configuration: configuration, source: source, dtype: .bfloat16,
            activationOverflow: .museGlimmer)
    }

    /// The pinned artifact from a local directory in the Hugging Face layout.
    /// `config.json` must describe the published decoder; `hf_quant_config.json`
    /// is cross-checked when present.
    public static func stage(directory: URL) throws -> Installed {
        let configuration = try MuseGlimmerTextModelConfiguration.textConfiguration(
            checkpointConfiguration: Data(
                contentsOf: directory.appendingPathComponent("config.json")))
        let source = try openPinnedSource(directory: directory)
        let record = directory.appendingPathComponent("hf_quant_config.json")
        if let data = try? Data(contentsOf: record) {
            try crossCheck(
                quantizationRecord: data,
                plan: plan(configuration: configuration, source: source))
        }
        return try stage(source: source, configuration: configuration)
    }

    /// Unpinned so tests can stage tiny checkpoints; production goes through
    /// the pinned entries above.
    static func stage(
        configuration c: MuseGlimmerTextModelConfiguration, source: ModelOptIndexedCheckpoint,
        dtype: DType, activationOverflow: ModelOptActivationOverflow
    ) throws -> Installed {
        let plan = try plan(configuration: c, source: source, dtype: dtype)
        let limit = plan.textPayloadBytes
        let model = MuseGlimmerTextModel(c)
        let parents = Dictionary(uniqueKeysWithValues: model.namedModules())
        // Every parameter of the finished tower, by model key.
        var mapped = [String: MLXArray]()
        var loaded = 0
        for slot in projectionSlots(c) {
            try Task.checkCancellation()
            let group = source.group(checkpointKey(slot.path))
            let layer: Module
            switch plan.projections[slot.path]! {
            case .dense:
                layer = Linear(
                    weight: try group["weight"]!.array(maximumBytes: limit), bias: nil)
            case .fp8:
                layer = try ModelOptProjectionLoader.load(
                    sources: group, inputDimensions: slot.input, outputDimensions: slot.output,
                    weightDType: dtype, maximumBytes: limit,
                    activationOverflow: activationOverflow)
            case .nvfp4:
                layer = try ModelOptWeightOnlyNVFP4Linear.load(
                    sources: group, inputDimensions: slot.input, outputDimensions: slot.output,
                    maximumBytes: limit)
            }
            try MLX.withError { errors in
                eval(layer.parameters())
                try errors.check()
            }
            let parts = slot.path.split(separator: ".").map(String.init)
            guard let parent = parents[parts.dropLast().joined(separator: ".")] else {
                throw Failure.unexpected(slot.path)
            }
            try parent.update(modules: .unflattened([(parts.last!, layer)]), verify: .all)
            for (leaf, value) in layer.parameters().flattened() {
                mapped[slot.path + "." + leaf] = value
            }
            loaded += group.count
        }
        for key in plan.denseTensors {
            try Task.checkCancellation()
            mapped[modelKey(key)] = try source.tensors[key]!.array(maximumBytes: limit)
            loaded += 1
        }
        // The module tree and the checkpoint's text keys must name the same
        // parameters: nothing left at its initial value, nothing unplaced.
        let expected = Set(model.parameters().flattened().map(\.0))
        let textKeys = Set(
            source.tensors.keys.filter { key in
                !visionPrefixes.contains(where: { key.hasPrefix($0) })
            }.map(modelKey))
        if let key = expected.subtracting(mapped.keys).sorted().first {
            throw Failure.missing(key)
        }
        if let key = textKeys.symmetricDifference(expected).sorted().first {
            throw Failure.unexpected(key)
        }
        try model.update(parameters: .unflattened(mapped), verify: [.all])
        try MLX.withError { errors in
            eval(model)
            try errors.check()
        }
        let receipt = Receipt(
            indexSHA256: source.indexSHA256, shardSHA256: source.shardSHA256, plan: plan,
            activationOverflow: activationOverflow, loadedTensorCount: loaded)
        var vision = [String: ModelOptTensorSource]()
        for key in plan.visionTensors { vision[key] = source.tensors[key] }
        return Installed(model: model, receipt: receipt, visionSources: vision)
    }
}
