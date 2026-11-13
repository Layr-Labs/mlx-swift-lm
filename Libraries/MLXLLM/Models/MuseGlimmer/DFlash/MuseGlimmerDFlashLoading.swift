// Copyright © 2026 Eigen Labs.

import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// How one projection of a drafter artifact is stored. Read from the verified
/// weight file's header, never from `config.json`.
public enum MuseGlimmerDFlashProjectionFormat: Sendable, Equatable {
    /// `weight` in the artifact's dense dtype, `[out, in]`.
    case dense
    /// MLX affine quantization: `weight` packed as U8 `[out, in * bits / 8]`,
    /// `scales` and `biases` `[out, in / groupSize]`.
    case affine(bits: Int, groupSize: Int)

    public var bitsPerWeight: Double {
        switch self {
        case .dense: 16
        case .affine(let bits, let groupSize): Double(bits) + 32 / Double(groupSize)
        }
    }
}

/// The exact tensor set of a drafter artifact.
public struct MuseGlimmerDFlashInventory: Sendable, Equatable {
    public struct Projection: Sendable, Equatable {
        public let path: String
        public let outputDimensions: Int
        public let inputDimensions: Int
    }

    public let projections: [Projection]
    /// Norm weight name to length.
    public let norms: [String: Int]

    public init(_ c: MuseGlimmerDFlashConfiguration) {
        var projections = [
            Projection(
                path: "encoder.fc", outputDimensions: c.hiddenSize,
                inputDimensions: c.contextWidth)
        ]
        var norms = [
            "encoder.output_norm_enc.weight": c.hiddenSize,
            "norm.weight": c.hiddenSize,
        ]
        let queryWidth = c.attentionHeads * c.headDimensions
        let keyWidth = c.kvHeads * c.headDimensions
        for layer in 0 ..< c.hiddenLayers {
            let root = "layers.\(layer)."
            for (name, output, input) in [
                ("self_attn.q_proj", queryWidth, c.hiddenSize),
                ("self_attn.k_proj", keyWidth, c.hiddenSize),
                ("self_attn.v_proj", keyWidth, c.hiddenSize),
                ("self_attn.o_proj", c.hiddenSize, queryWidth),
                ("mlp.gate_proj", c.intermediateSize, c.hiddenSize),
                ("mlp.up_proj", c.intermediateSize, c.hiddenSize),
                ("mlp.down_proj", c.hiddenSize, c.intermediateSize),
            ] {
                projections.append(
                    Projection(path: root + name, outputDimensions: output, inputDimensions: input))
            }
            norms[root + "input_layernorm.weight"] = c.hiddenSize
            norms[root + "post_attention_layernorm.weight"] = c.hiddenSize
            norms[root + "self_attn.q_norm.weight"] = c.headDimensions
            norms[root + "self_attn.k_norm.weight"] = c.headDimensions
        }
        self.projections = projections
        self.norms = norms
    }

    static let affineBits: Set<Int> = [2, 3, 4, 5, 6, 8]
    static let affineGroupSizes: Set<Int> = [32, 64, 128]
    static let denseDTypes: Set<String> = ["BF16", "F16", "F32"]

    /// Each projection's format, from the header. Throws unless the header
    /// holds exactly this inventory with one dense dtype throughout.
    public func formats(
        header: [String: ModelOptProjectionLayout.Tensor]
    ) throws -> (formats: [String: MuseGlimmerDFlashProjectionFormat], denseDType: String) {
        var expected = Set(norms.keys)
        var formats: [String: MuseGlimmerDFlashProjectionFormat] = [:]
        var dense = Set<String>()
        for (name, length) in norms {
            guard let tensor = header[name] else {
                throw MuseGlimmerDFlashError.inventory(name)
            }
            guard tensor.shape == [length] else { throw MuseGlimmerDFlashError.shape(name) }
            dense.insert(tensor.dtype)
        }
        for projection in projections {
            let weightName = projection.path + ".weight"
            guard let weight = header[weightName] else {
                throw MuseGlimmerDFlashError.inventory(weightName)
            }
            expected.insert(weightName)
            let (output, input) = (projection.outputDimensions, projection.inputDimensions)
            if weight.dtype == "U8" {
                let scalesName = projection.path + ".scales"
                let biasesName = projection.path + ".biases"
                guard let scales = header[scalesName], let biases = header[biasesName] else {
                    throw MuseGlimmerDFlashError.inventory(scalesName)
                }
                expected.formUnion([scalesName, biasesName])
                guard weight.shape.count == 2, weight.shape[0] == output,
                    scales.shape.count == 2, scales.shape[0] == output, scales.shape[1] > 0,
                    biases.shape == scales.shape, scales.dtype == biases.dtype,
                    input % scales.shape[1] == 0, (weight.shape[1] * 8) % input == 0
                else { throw MuseGlimmerDFlashError.shape(weightName) }
                let bits = weight.shape[1] * 8 / input
                let groupSize = input / scales.shape[1]
                guard Self.affineBits.contains(bits), Self.affineGroupSizes.contains(groupSize),
                    weight.shape[1] % 4 == 0
                else { throw MuseGlimmerDFlashError.shape(weightName) }
                dense.insert(scales.dtype)
                formats[projection.path] = .affine(bits: bits, groupSize: groupSize)
            } else {
                guard weight.shape == [output, input] else {
                    throw MuseGlimmerDFlashError.shape(weightName)
                }
                dense.insert(weight.dtype)
                formats[projection.path] = .dense
            }
        }
        guard Set(header.keys) == expected else {
            throw MuseGlimmerDFlashError.inventory("unexpected tensors")
        }
        guard dense.count == 1, let dtype = dense.first, Self.denseDTypes.contains(dtype) else {
            throw MuseGlimmerDFlashError.shape("dense dtype")
        }
        return (formats, dtype)
    }
}

/// What a drafter artifact must be before any of it is used.
public struct MuseGlimmerDFlashArtifactPin: Sendable, Equatable {
    public static let weightsFileName = "model.safetensors"
    public static let configurationFileName = "config.json"

    /// Lower-case hex SHA-256 of `model.safetensors`.
    public var weightsSHA256: String
    /// Lower-case hex SHA-256 of `config.json`.
    public var configurationSHA256: String
    /// The configuration `config.json` must decode to.
    public var configuration: MuseGlimmerDFlashConfiguration

    public init(
        weightsSHA256: String, configurationSHA256: String,
        configuration: MuseGlimmerDFlashConfiguration
    ) {
        self.weightsSHA256 = weightsSHA256
        self.configurationSHA256 = configurationSHA256
        self.configuration = configuration
    }

    /// `meta-models/Muse-Glimmer-30B-assistant` at revision
    /// `c676cce91658f9568aaf264a416211e2782735b7`: 58 BF16 tensors.
    public static let museGlimmer30BBF16 = MuseGlimmerDFlashArtifactPin(
        weightsSHA256: "fd88d337eb84f8d0e6ba33a7684d7efa6722d4460ba4d6badca9699418392a84",
        configurationSHA256: "38915167b64b1e6405492aacae5b1b4511b6431163d2960b9bd25821df6fa30a",
        configuration: .museGlimmer30B)
}

extension MuseGlimmerDFlashDraftModel {
    static let maximumConfigurationBytes = 64 << 10

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Reads and checks `config.json` against the pin.
    static func verifiedConfiguration(
        directory: URL, pin: MuseGlimmerDFlashArtifactPin
    ) throws -> MuseGlimmerDFlashConfiguration {
        let url = directory.appendingPathComponent(
            MuseGlimmerDFlashArtifactPin.configurationFileName)
        let values = try? url.resourceValues(forKeys: [
            .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey,
        ])
        guard let values, values.isRegularFile == true, values.isSymbolicLink != true,
            let size = values.fileSize, size > 0, size <= maximumConfigurationBytes,
            let data = try? Data(contentsOf: url), data.count == size
        else { throw MuseGlimmerDFlashError.artifact("config.json") }
        guard sha256Hex(data) == pin.configurationSHA256 else {
            throw MuseGlimmerDFlashError.checksum("config.json")
        }
        let decoded: MuseGlimmerDFlashConfiguration
        do {
            decoded = try JSONDecoder().decode(MuseGlimmerDFlashConfiguration.self, from: data)
        } catch let error as MuseGlimmerDFlashError {
            throw error
        } catch {
            throw MuseGlimmerDFlashError.configuration("config.json")
        }
        guard decoded == pin.configuration else {
            throw MuseGlimmerDFlashError.configuration("config.json differs from its pin")
        }
        return decoded
    }

    /// Loads a drafter artifact directory holding `config.json` and
    /// `model.safetensors`.
    ///
    /// Both files must hash to the pin. The weight file must hold exactly the
    /// drafter's tensors, each with the shape its configuration gives; each
    /// projection's format (dense or affine) comes from the verified header.
    /// `maximumBytes` bounds the tensor payload; the caller holds admission
    /// for it plus one largest-tensor copy while loading.
    public static func load(
        directory: URL, pin: MuseGlimmerDFlashArtifactPin, maximumBytes: Int
    ) throws -> MuseGlimmerDFlashDraftModel {
        let configuration = try verifiedConfiguration(directory: directory, pin: pin)
        let file: ModelOptTensorFile
        do {
            file = try ModelOptTensorFile(
                url: directory.appendingPathComponent(MuseGlimmerDFlashArtifactPin.weightsFileName),
                sha256: pin.weightsSHA256)
        } catch ModelOptTensorFile.Failure.checksum {
            throw MuseGlimmerDFlashError.checksum("model.safetensors")
        } catch let error as CancellationError {
            throw error
        } catch {
            throw MuseGlimmerDFlashError.artifact("model.safetensors: \(error)")
        }
        return try load(configuration: configuration, file: file, maximumBytes: maximumBytes)
    }

    /// Builds the model from an already verified weight file.
    static func load(
        configuration: MuseGlimmerDFlashConfiguration, file: ModelOptTensorFile, maximumBytes: Int
    ) throws -> MuseGlimmerDFlashDraftModel {
        let inventory = MuseGlimmerDFlashInventory(configuration)
        let header = file.entries.mapValues(\.layout)
        let (formats, denseName) = try inventory.formats(header: header)
        var payload = 0
        for tensor in header.values {
            let (next, overflow) = payload.addingReportingOverflow(tensor.byteCount)
            guard !overflow, maximumBytes > 0, next <= maximumBytes else {
                throw MuseGlimmerDFlashError.budget
            }
            payload = next
        }
        let dtype: DType =
            switch denseName {
            case "F32": .float32
            case "F16": .float16
            default: .bfloat16
            }
        func array(_ name: String) throws -> MLXArray {
            do {
                return try ModelOptTensorSource(file: file, name: name).array(
                    maximumBytes: maximumBytes)
            } catch let error as CancellationError {
                throw error
            } catch {
                throw MuseGlimmerDFlashError.artifact("\(name): \(error)")
            }
        }
        let model = try MuseGlimmerDFlashDraftModel(configuration, dtype: dtype)
        var parameters: [String: MLXArray] = [:]
        var replacements: [(String, Module)] = []
        for name in inventory.norms.keys { parameters[name] = try array(name) }
        for projection in inventory.projections {
            let path = projection.path
            switch formats[path] {
            case .dense?:
                parameters[path + ".weight"] = try array(path + ".weight")
            case .affine(let bits, let groupSize)?:
                let layer = QuantizedLinear(
                    weight: try array(path + ".weight").view(dtype: .uint32), bias: nil,
                    scales: try array(path + ".scales"), biases: try array(path + ".biases"),
                    groupSize: groupSize, bits: bits, mode: .affine)
                guard layer.shape == (projection.outputDimensions, projection.inputDimensions)
                else { throw MuseGlimmerDFlashError.shape(path) }
                layer.freeze()
                replacements.append((path, layer))
            case nil:
                throw MuseGlimmerDFlashError.inventory(path)
            }
        }
        try model.install(projections: replacements)
        do {
            try model.update(
                parameters: ModuleParameters.unflattened(parameters),
                verify: [.noUnusedKeys, .shapeMismatch])
        } catch {
            throw MuseGlimmerDFlashError.inventory("parameters: \(error)")
        }
        // Every placeholder must be gone before the only evaluation.
        let loaded = Set(parameters.keys).union(
            replacements.flatMap { path, layer in
                layer.parameters().flattened().map { path + "." + $0.0 }
            })
        guard loaded == Set(model.parameters().flattened().map(\.0)) else {
            throw MuseGlimmerDFlashError.inventory("incomplete coverage")
        }
        try MLX.withError { errors in
            eval(model.parameters())
            try errors.check()
        }
        return model
    }

    /// Replaces projection modules by path, for example with quantized ones.
    func install(projections: [(String, Module)]) throws {
        let parents = Dictionary(uniqueKeysWithValues: namedModules())
        for (path, replacement) in projections {
            let parts = path.split(separator: ".").map(String.init)
            guard let key = parts.last,
                let parent = parents[parts.dropLast().joined(separator: ".")]
            else { throw MuseGlimmerDFlashError.inventory(path) }
            do {
                try parent.update(modules: .unflattened([(key, replacement)]), verify: .all)
            } catch {
                throw MuseGlimmerDFlashError.inventory("\(path): \(error)")
            }
        }
    }

    /// Format of every projection as loaded.
    public var projectionFormats: [String: MuseGlimmerDFlashProjectionFormat] {
        var formats: [String: MuseGlimmerDFlashProjectionFormat] = [:]
        for (path, module) in leafModules().flattened() {
            if let quantized = module as? QuantizedLinear {
                formats[path] = .affine(bits: quantized.bits, groupSize: quantized.groupSize)
            } else if module is Linear {
                formats[path] = .dense
            }
        }
        return formats
    }
}
