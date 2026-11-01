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
    /// Attention head width; the paged KV kernels take 64, 128, 256 or 512.
    @TaskLocal static var attentionHeadDim = 16

    static func configuration() -> NemotronHConfiguration {
        NemotronHConfiguration(
            vocabSize: vocabulary, hiddenSize: hidden, numHiddenLayers: 3,
            numAttentionHeads: 2, numKeyValueHeads: 1, mambaNumHeads: 2, mambaHeadDim: 16,
            ssmStateSize: 8, convKernel: 4, nGroups: 1, intermediateSize: 32,
            moeIntermediateSize: 32, moeSharedExpertIntermediateSize: 32,
            nRoutedExperts: 4, numExpertsPerTok: 2, hybridOverridePattern: "M*E",
            headDim: attentionHeadDim,
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
    /// while every Mamba/attention/MoE state still contributes. Every ModelOpt
    /// projection quantizes activations under `activationOverflow`, by default
    /// the production Super policy.
    static func target(
        seed: UInt64 = 7, mixerScale: Float = 1, quantized: Bool = false,
        activationOverflow: ModelOptActivationOverflow = .nemotronSuper
    ) throws -> NemotronSuperModel {
        let language = NemotronHModel(configuration())
        try randomize(language, seed: seed)
        if quantized {
            return try compose(
                quantizedLanguage(
                    language, seed: seed, mixerScale: mixerScale,
                    activationOverflow: activationOverflow))
        }
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
            inputScale: 0.02, weightDType: .float32, activationOverflow: activationOverflow)
        try language.installModelOptProjections(["lm_head": projection])
        return try compose(language)
    }

    private static func compose(_ language: NemotronHModel) throws -> NemotronSuperModel {
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

    static func nvfp4Weights(input: Int, output: Int, seed: UInt64, scale: Float) throws
        -> ModelOptNVFP4Weights
    {
        let packed = MLXRandom.randInt(
            Int32(0) ..< 256, [output, input / 2], key: MLXRandom.key(seed))
        let blocks = MLXRandom.randInt(
            Int32(40) ..< 60, [output, input / 16], key: MLXRandom.key(seed &+ 1))
        return try ModelOptNVFP4Weights(
            packedBytes: packed.asType(.uint8), e4m3ScaleBytes: blocks.asType(.uint8),
            tensorScale: scale)
    }

    /// The pinned checkpoint's precision mix on fixture geometry: BF16
    /// activations (float32 `A_log` and router bias), calibrated FP8 Mamba,
    /// attention and output projections, NVFP4 routed and shared experts.
    private static func quantizedLanguage(
        _ language: NemotronHModel, seed: UInt64, mixerScale: Float,
        activationOverflow: ModelOptActivationOverflow
    ) throws -> NemotronHModel {
        let narrowed = language.parameters().flattened().map { name, value -> (String, MLXArray) in
            let keep = name.hasSuffix("A_log") || name.hasSuffix("e_score_correction_bias")
            return (name, keep ? value : value.asType(.bfloat16))
        }
        try language.update(parameters: .unflattened(narrowed), verify: [.noUnusedKeys])
        let c = configuration()
        var projections: [String: ModelOptCalibratedLinear] = [:]
        var experts: [String: ModelOptNVFP4SwitchLinear] = [:]
        var next = seed &* 7_919
        func fp8(_ input: Int, _ output: Int, scale: Float) throws -> ModelOptCalibratedLinear {
            next &+= 1
            return try ModelOptCalibratedLinear(
                weights: ModelOptFP8Weights(
                    e4m3Bytes: fp8Codes([output, input], seed: next), tensorScale: scale),
                inputScale: 0.02, weightDType: .bfloat16, activationOverflow: activationOverflow)
        }
        func nvfp4(_ input: Int, _ output: Int, scale: Float) throws -> ModelOptNVFP4Weights {
            next &+= 2
            return try nvfp4Weights(input: input, output: output, seed: next, scale: scale)
        }
        let mamba = c.mambaNumHeads * c.mambaHeadDim
        let convDim = mamba + 2 * c.nGroups * c.ssmStateSize
        let attention = c.numAttentionHeads * c.headDim!
        let kv = c.numKeyValueHeads * c.headDim!
        let latent = c.moeLatentSize!
        for (layer, kind) in c.hybridOverridePattern.enumerated() {
            let prefix = "backbone.layers.\(layer).mixer."
            switch kind {
            case "M":
                projections[prefix + "in_proj"] = try fp8(
                    hidden, mamba + convDim + c.mambaNumHeads, scale: 1.0 / 512)
                projections[prefix + "out_proj"] = try fp8(mamba, hidden, scale: mixerScale / 512)
            case "*":
                projections[prefix + "q_proj"] = try fp8(hidden, attention, scale: 1.0 / 512)
                projections[prefix + "k_proj"] = try fp8(hidden, kv, scale: 1.0 / 512)
                projections[prefix + "v_proj"] = try fp8(hidden, kv, scale: 1.0 / 512)
                projections[prefix + "o_proj"] = try fp8(attention, hidden, scale: mixerScale / 512)
            case "E":
                let shared = c.moeSharedExpertIntermediateSize
                projections[prefix + "shared_experts.up_proj"] = try ModelOptCalibratedLinear(
                    weights: nvfp4(hidden, shared, scale: 0.05), inputScale: 0.01,
                    weightDType: .bfloat16, activationOverflow: activationOverflow)
                projections[prefix + "shared_experts.down_proj"] = try ModelOptCalibratedLinear(
                    weights: nvfp4(shared, hidden, scale: 0.05 * mixerScale), inputScale: 0.05,
                    weightDType: .bfloat16, activationOverflow: activationOverflow)
                experts[prefix + "switch_mlp.fc1"] = try ModelOptNVFP4SwitchLinear(
                    experts: (0 ..< c.nRoutedExperts).map { _ in
                        try nvfp4(latent, c.moeIntermediateSize, scale: 0.05)
                    },
                    inputScales: Array(repeating: 0.01, count: c.nRoutedExperts),
                    weightDType: .bfloat16, activationOverflow: activationOverflow)
                experts[prefix + "switch_mlp.fc2"] = try ModelOptNVFP4SwitchLinear(
                    experts: (0 ..< c.nRoutedExperts).map { _ in
                        try nvfp4(c.moeIntermediateSize, latent, scale: 0.05 * mixerScale)
                    },
                    inputScales: Array(repeating: 0.05, count: c.nRoutedExperts),
                    weightDType: .bfloat16, activationOverflow: activationOverflow)
            default: break
            }
        }
        projections["lm_head"] = try fp8(hidden, vocabulary, scale: 1.0 / 256)
        try language.installModelOptProjections(projections)
        try language.installModelOptExperts(experts)
        eval(language.parameters())
        return language
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

    static func headTensors(_ kind: Head, seed: UInt64, dtype: DType = .float32) throws
        -> [String: Tensor]
    {
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
            let wide = dtype == .float32 || name.hasSuffix("e_score_correction_bias")
            result[prefix + name] =
                wide
                ? bytes(value.asType(.float32), dtype: "F32")
                : bytes(value.asType(.bfloat16), dtype: "BF16")
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
        construction: NativeConstructionScope = NativeConstructionScope(),
        dtype: DType = .float32,
        activationOverflow: ModelOptActivationOverflow = .nemotronSuper
    ) throws -> NemotronSuperMTPModule {
        try withSource(headTensors(kind, seed: seed, dtype: dtype)) { source in
            let quote = try NemotronSuperMTPModule.loadQuote(
                configuration: target.configuration, source: source)
            return try NemotronSuperMTPModule.load(
                for: target, source: source, quote: quote, construction: construction,
                dtype: dtype, activationOverflow: activationOverflow)
        }
    }

    static func settings(
        depth: Int = 1, maximumSequenceLength: Int = 128,
        verification: NemotronSuperMTPAssistant.Settings.Verification = .serial
    ) -> NemotronSuperMTPAssistant.Settings {
        .init(
            maximumDraftTokens: depth, allowsDeepDrafts: depth > 1,
            maximumSequenceLength: maximumSequenceLength, historyChunkRows: 8,
            backendScratchBytes: 1 << 16, verification: verification)
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
        mixerScale: Float = 1, quantized: Bool = false,
        verification: NemotronSuperMTPAssistant.Settings.Verification = .serial,
        maximumSequenceLength: Int = 128
    ) throws -> Stack {
        let model = try target(seed: targetSeed, mixerScale: mixerScale, quantized: quantized)
        let target =
            verification == .rectangularExact
            ? try NemotronSuperMTPCaptureTarget(capturing: model)
            : NemotronSuperMTPTextTarget(model)
        let head = try head(
            for: model.languageModel, kind: kind, seed: headSeed,
            dtype: quantized ? .bfloat16 : .float32)
        let assistant = try target.makeAssistant(
            head: head,
            settings: settings(
                depth: depth, maximumSequenceLength: maximumSequenceLength,
                verification: verification))
        return Stack(model: model, target: target, head: head, assistant: assistant)
    }
}
