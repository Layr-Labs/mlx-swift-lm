import CryptoKit
import Foundation
import MLX
import MLXNN

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Tiny seeded synthetic Super target with a calibrated FP8 output head and an
/// embedded NVFP4 prediction head loaded through the production loader. Never
/// official weights; numbers from it are synthetic fixture measurements only.
enum NemotronSuperMTPFixture {
    /// `random` exercises every head projection numerically. `copy` makes the
    /// draft reproduce the target's previous prediction (fused = hnorm(hidden),
    /// tiny attention/MoE terms), so greedy repetition yields accepted drafts.
    /// `predictive` drafts from the seed embedding alone (fused = enorm(embed));
    /// with a weak-mixer target it agrees often, but not always, at every depth.
    enum Head { case random, copy, predictive }

    static let vocabulary = 64
    static let hidden = 32

    static func configuration() -> NemotronHConfiguration {
        NemotronHConfiguration(
            vocabSize: vocabulary, hiddenSize: hidden, numHiddenLayers: 3,
            numAttentionHeads: 2, numKeyValueHeads: 1, mambaNumHeads: 2, mambaHeadDim: 16,
            ssmStateSize: 8, convKernel: 4, nGroups: 1, intermediateSize: 32,
            moeIntermediateSize: 32, moeSharedExpertIntermediateSize: 32,
            nRoutedExperts: 4, numExpertsPerTok: 2, hybridOverridePattern: "M*E", headDim: 16,
            nSharedExperts: 1, numNextnPredictLayers: 1,
            mtpLayersBlockType: ["attention", "moe"], moeLatentSize: 16)
    }

    /// Keyed, order-independent initialization; never the shared global RNG.
    static func randomize(_ module: Module, seed: UInt64, scale: Float = 1) throws {
        var updates: [(String, MLXArray)] = []
        for (index, (name, value)) in module.parameters().flattened().enumerated()
        where [.float32, .bfloat16, .float16].contains(value.dtype) {
            let key = MLXRandom.key(seed &* 1_000_003 &+ UInt64(index))
            let noise = MLXRandom.normal(value.shape, key: key)
            let leaf = name.split(separator: ".").last.map(String.init) ?? name
            if ["A_log", "D", "dt_bias"].contains(leaf) { continue }
            let next: MLXArray
            if value.ndim == 1 && name.contains("norm") {
                next = 1 + 0.1 * noise
            } else if value.ndim == 1 {
                next = 0.1 * noise
            } else {
                next = noise * (scale / Float(value.dim(-1)).squareRoot())
            }
            updates.append((name, next.asType(value.dtype)))
        }
        try module.update(
            parameters: .unflattened(updates), verify: [.noUnusedKeys, .shapeMismatch])
        eval(module.parameters())
    }

    /// E4M3 codes excluding both NaN encodings.
    static func fp8Codes(_ shape: [Int], seed: UInt64) -> MLXArray {
        let value = MLXRandom.randInt(Int32(0) ..< 254, shape, key: MLXRandom.key(seed))
        return which(value .>= 127, value + 1, value).asType(.uint8)
    }

    /// `mixerScale` below one shrinks every block's output projection, so the
    /// residual stream (and greedy choice) is dominated by the token embedding
    /// while every Mamba/attention/MoE state still contributes.
    static func target(seed: UInt64 = 7, mixerScale: Float = 1) throws -> NemotronSuperModel {
        let language = NemotronHModel(configuration())
        try randomize(language, seed: seed)
        if mixerScale != 1 {
            let outputs = language.parameters().flattened().filter { name, _ in
                [
                    "mixer.out_proj.weight", "mixer.o_proj.weight", "switch_mlp.fc2.weight",
                    "shared_experts.down_proj.weight",
                ].contains { name.hasSuffix($0) }
            }
            try language.update(
                parameters: .unflattened(outputs.map { ($0.0, $0.1 * mixerScale) }),
                verify: [.noUnusedKeys, .shapeMismatch])
            eval(language.parameters())
        }
        let projection = try ModelOptCalibratedLinear(
            weights: ModelOptFP8Weights(
                e4m3Bytes: fp8Codes([vocabulary, hidden], seed: seed &+ 99),
                tensorScale: 1.0 / 256),
            inputScale: 0.02, weightDType: .float32)
        try language.installModelOptProjections(["lm_head": projection])
        let images = NemotronSuperImageComposition(
            tower: NemotronRadio(
                NemotronRadioDimensions(
                    width: 2, layers: 1, heads: 1,
                    channels: 1, patch: 2, positionSide: 2, prefix: 1, mlpWidth: 3)),
            projector: NemotronSuperVisionProjector(
                visionWidth: 2, hiddenWidth: 3, outputWidth: hidden),
            width: hidden, imageTokenID: 18)
        return try NemotronSuperModel(languageModel: language, imageComposition: images)
    }

    struct Tensor {
        let dtype: String
        let shape: [Int]
        let data: Data
    }

    private static func scalar(_ value: Float) -> Tensor {
        var bits = value.bitPattern.littleEndian
        return Tensor(dtype: "F32", shape: [1], data: withUnsafeBytes(of: &bits) { Data($0) })
    }

    private static func bytes(_ array: MLXArray, dtype: String) -> Tensor {
        let data = array.asData()
        return Tensor(dtype: dtype, shape: data.shape, data: data.data)
    }

    static func headTensors(_ kind: Head, seed: UInt64) throws -> [String: Tensor] {
        let c = configuration()
        let module = try NemotronSuperMTPModule(
            configuration: c, calibration: .init(keyScale: 1, valueScale: 1), dtype: .float32)
        try randomize(module, seed: seed)
        if kind != .random {
            let identity = MLXArray.eye(hidden)
            let zeros = MLXArray.zeros([hidden, hidden])
            let fusion = concatenated(
                kind == .copy ? [zeros, identity] : [identity, zeros], axis: 1)
            var updates: [(String, MLXArray)] = [("layers.0.eh_proj.weight", fusion)]
            for name in [
                "layers.0.enorm.weight", "layers.0.hnorm.weight", "layers.0.norm.weight",
                "layers.1.norm.weight", "layers.1.final_layernorm.weight",
            ] {
                updates.append((name, MLXArray.ones([hidden])))
            }
            let parameters = Dictionary(uniqueKeysWithValues: module.parameters().flattened())
            updates.append(
                (
                    "layers.0.mixer.o_proj.weight",
                    parameters["layers.0.mixer.o_proj.weight"]! * 0.001
                ))
            try module.update(
                parameters: .unflattened(updates), verify: [.noUnusedKeys, .shapeMismatch])
        }
        let prefix = "language_model.mtp."
        var result: [String: Tensor] = [:]
        for (name, value) in module.parameters().flattened()
        where !name.contains(".shared_experts.") && !name.contains(".switch_mlp.") {
            result[prefix + name] = bytes(value.asType(.float32), dtype: "F32")
        }
        let globalScale: Float = kind == .random ? 0.05 : 1e-4
        var expertSeed = seed &+ 1_000
        func expert(_ stem: String, input: Int, output: Int) {
            expertSeed &+= 3
            let packed = MLXRandom.randInt(
                Int32(0) ..< 256, [output, input / 2], key: MLXRandom.key(expertSeed))
            let scales = MLXRandom.randInt(
                Int32(40) ..< 60, [output, input / 16], key: MLXRandom.key(expertSeed &+ 1))
            result[stem + ".weight"] = bytes(packed.asType(.uint8), dtype: "U8")
            result[stem + ".weight_scale"] = bytes(scales.asType(.uint8), dtype: "F8_E4M3")
            result[stem + ".weight_scale_2"] = scalar(globalScale)
            result[stem + ".input_scale"] = scalar(0.01)
        }
        let latent = c.moeLatentSize!
        let mixer = prefix + "layers.1.mixer."
        expert(
            mixer + "shared_experts.up_proj", input: hidden,
            output: c.moeSharedExpertIntermediateSize)
        expert(
            mixer + "shared_experts.down_proj", input: c.moeSharedExpertIntermediateSize,
            output: hidden)
        for index in 0 ..< c.nRoutedExperts {
            expert(mixer + "experts.\(index).up_proj", input: latent, output: c.moeIntermediateSize)
            expert(
                mixer + "experts.\(index).down_proj", input: c.moeIntermediateSize, output: latent)
        }
        result[prefix + "layers.0.mixer.k_proj.k_scale"] = scalar(1)
        result[prefix + "layers.0.mixer.v_proj.v_scale"] = scalar(1)
        return result
    }

    /// Two-shard synthetic safetensors source; files are removed after `body`.
    static func withSource<Value>(
        _ tensors: [String: Tensor], _ body: (ModelOptCheckpointSource) throws -> Value
    ) throws -> Value {
        var files: [String: ModelOptTensorFile] = [:]
        var map: [String: String] = [:]
        var urls: [URL] = []
        defer { for url in urls { try? FileManager.default.removeItem(at: url) } }
        for shard in 0 ..< 2 {
            let name = "fixture-\(shard).safetensors"
            var header: [String: Any] = [:]
            var payload = Data()
            for (index, key) in tensors.keys.sorted().enumerated() where index % 2 == shard {
                let tensor = tensors[key]!
                let start = payload.count
                payload.append(tensor.data)
                header[key] = [
                    "dtype": tensor.dtype, "shape": tensor.shape,
                    "data_offsets": [start, payload.count],
                ]
                map[key] = name
            }
            let headerData = try JSONSerialization.data(
                withJSONObject: header, options: [.sortedKeys])
            var length = UInt64(headerData.count).littleEndian
            var data = withUnsafeBytes(of: &length) { Data($0) }
            data.append(headerData)
            data.append(payload)
            let url = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString)
            urls.append(url)
            try data.write(to: url)
            files[name] = try ModelOptTensorFile(
                url: url,
                sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
        }
        let index = try JSONSerialization.data(
            withJSONObject: ["weight_map": map], options: [.sortedKeys])
        return try body(
            ModelOptCheckpointSource(
                indexData: index,
                indexSHA256: SHA256.hash(data: index).map { String(format: "%02x", $0) }.joined(),
                files: files))
    }

    static func head(
        for target: NemotronHModel, kind: Head, seed: UInt64 = 11,
        construction: NativeConstructionScope = NativeConstructionScope()
    ) throws -> NemotronSuperMTPModule {
        try withSource(headTensors(kind, seed: seed)) { source in
            let quote = try NemotronSuperMTPModule.loadQuote(
                configuration: target.configuration, source: source)
            return try NemotronSuperMTPModule.load(
                for: target, source: source, quote: quote, construction: construction,
                dtype: .float32)
        }
    }

    static func settings(depth: Int = 1, maximumSequenceLength: Int = 128)
        -> NemotronSuperMTPAssistant.Settings
    {
        .init(
            maximumDraftTokens: depth, allowsDeepDrafts: depth > 1,
            maximumSequenceLength: maximumSequenceLength, historyChunkRows: 8,
            backendScratchBytes: 1 << 16)
    }

    /// One fully assembled text-only speculation stack.
    struct Stack {
        let model: NemotronSuperModel
        let target: NemotronSuperMTPTextTarget
        let head: NemotronSuperMTPModule
        let assistant: NemotronSuperMTPAssistant
    }

    static func stack(
        kind: Head, depth: Int = 1, targetSeed: UInt64 = 7, headSeed: UInt64 = 11,
        mixerScale: Float = 1,
        maximumSequenceLength: Int = 128
    ) throws -> Stack {
        let model = try target(seed: targetSeed, mixerScale: mixerScale)
        let target = NemotronSuperMTPTextTarget(model)
        let head = try head(for: model.languageModel, kind: kind, seed: headSeed)
        let assistant = try target.makeAssistant(
            head: head,
            settings: settings(depth: depth, maximumSequenceLength: maximumSequenceLength))
        return Stack(model: model, target: target, head: head, assistant: assistant)
    }
}
