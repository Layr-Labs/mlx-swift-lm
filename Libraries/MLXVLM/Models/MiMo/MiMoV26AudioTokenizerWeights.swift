// Copyright © 2026 Eigen Labs. Input-only strict maps for the retained sidecar.
import Foundation
import MLX
import MLXLLM

public enum MiMoV26AudioTensorType: String, Codable, Sendable {
    case bfloat16 = "BF16"
    case float32 = "F32"
    var dtype: DType { self == .bfloat16 ? .bfloat16 : .float32 }
    var bytes: Int { self == .bfloat16 ? 2 : 4 }
}
public struct MiMoV26AudioTensorDescriptor: Codable, Equatable, Sendable {
    public let shape: [Int]
    public let dtype: MiMoV26AudioTensorType
    public let byteCount: Int
    public init(shape: [Int], dtype: MiMoV26AudioTensorType, byteCount: Int) {
        self.shape = shape
        self.dtype = dtype
        self.byteCount = byteCount
    }
}
public struct MiMoV26AudioSidecarPlan: Sendable {
    public let configuration: MiMoV26AudioInputConfiguration
    public let descriptors: [String: MiMoV26AudioTensorDescriptor]
    public let requiredInputNames: Set<String>
    public let sourcePayloadSHA256: String
    public let inputStoredBytes, retainedUnusedStoredBytes: Int
    fileprivate init(
        configuration: MiMoV26AudioInputConfiguration,
        descriptors: [String: MiMoV26AudioTensorDescriptor],
        requiredInputNames: Set<String>, sourcePayloadSHA256: String, inputStoredBytes: Int,
        retainedUnusedStoredBytes: Int
    ) {
        self.configuration = configuration
        self.descriptors = descriptors
        self.requiredInputNames = requiredInputNames
        self.sourcePayloadSHA256 = sourcePayloadSHA256
        self.inputStoredBytes = inputStoredBytes
        self.retainedUnusedStoredBytes = retainedUnusedStoredBytes
    }
}
public struct MiMoV26AudioInputWeights {
    public let encoder: MiMoV26AudioTokenizerEncoder
    public let quantizer: MiMoV26ResidualVectorQuantizer
    public let configuration: MiMoV26AudioInputConfiguration
    public let sourceIdentity: String
    public let generation: UUID
    /// Transformed owners, NOT a source-file read-order or physical-byte receipt.
    public var materializationRoots: [MLXArray] {
        encoder.parameters().flattened().sorted { $0.0 < $1.0 }.map { $0.1 }
            + quantizer.materializationRoots
    }
    fileprivate init(
        encoder: MiMoV26AudioTokenizerEncoder, quantizer: MiMoV26ResidualVectorQuantizer,
        configuration: MiMoV26AudioInputConfiguration, sourceIdentity: String, generation: UUID
    ) {
        self.encoder = encoder
        self.quantizer = quantizer
        self.configuration = configuration
        self.sourceIdentity = sourceIdentity
        self.generation = generation
    }
}

public enum MiMoV26AudioTokenizerWeights {
    public static let selectedPayloadSHA256 =
        "077033345d80eef3a315e8d394e0589667e80e4cdaba9bc5a7488410c6657265"

    public static func networkSourceShapes(configuration c: MiMoV26AudioInputConfiguration)
        -> [String: [Int]]
    {
        var shapes: [String: [Int]] = [
            "encoder.conv1.weight": [c.hiddenSize, c.melBands, 3],
            "encoder.conv1.bias": [c.hiddenSize],
            "encoder.conv2.weight": [c.hiddenSize, c.hiddenSize, 3],
            "encoder.conv2.bias": [c.hiddenSize],
            "encoder.layer_norm.weight": [c.hiddenSize], "encoder.layer_norm.bias": [c.hiddenSize],
            "encoder.down_sample_layer.0.weight": [c.hiddenSize, c.hiddenSize, 2],
            "encoder.down_sample_norm.weight": [c.hiddenSize],
            "encoder.down_sample_norm.bias": [c.hiddenSize],
        ]
        for i in 0 ..< c.layers {
            layerShapes(
                prefix: "encoder.layers.\(i)", width: c.hiddenSize, ffn: c.ffnSize, into: &shapes)
        }
        return shapes
    }
    public static func inputSourceShapes(configuration c: MiMoV26AudioInputConfiguration)
        -> [String: [Int]]
    {
        var shapes = networkSourceShapes(configuration: c)
        for (i, bins) in c.codebookSizes.enumerated() {
            shapes[tableName(i)] = [bins, c.hiddenSize]
        }
        return shapes
    }
    static func tableName(_ index: Int) -> String {
        "encoder.quantizer.vq.layers.\(index)._codebook.embed"
    }
    private static func layerShapes(
        prefix: String, width: Int, ffn: Int, into shapes: inout [String: [Int]]
    ) {
        for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            shapes[prefix + ".self_attn." + name + ".weight"] = [width, width]
            if name != "k_proj" { shapes[prefix + ".self_attn." + name + ".bias"] = [width] }
        }
        for name in ["self_attn_layer_norm", "final_layer_norm"] {
            shapes[prefix + "." + name + ".weight"] = [width]
            shapes[prefix + "." + name + ".bias"] = [width]
        }
        shapes[prefix + ".fc1.weight"] = [ffn, width]
        shapes[prefix + ".fc1.bias"] = [ffn]
        shapes[prefix + ".fc2.weight"] = [width, ffn]
        shapes[prefix + ".fc2.bias"] = [width]
    }
    private static func retainedShapes(configuration c: MiMoV26AudioInputConfiguration) -> [String:
        [Int]]
    {
        // Exact selected asset closure. These values are not instantiated as
        // input modules or treated as proof of an audio-output capability.
        var shapes: [String: [Int]] = [
            "decoder.dconv1.conv.bias": [1024], "decoder.dconv1.conv.weight": [1024, 1024, 2],
            "decoder.dconv1.norm.bias": [1024], "decoder.dconv1.norm.weight": [1024],
            "decoder.dconv2.conv.bias": [128], "decoder.dconv2.conv.weight": [1024, 128, 3],
            "decoder.dconv2.norm.bias": [128], "decoder.dconv2.norm.weight": [128],
            "decoder.layer_norm.bias": [1024], "decoder.layer_norm.weight": [1024],
            "decoder.vocoder.istft.window": [960], "decoder.vocoder.out.bias": [962],
            "decoder.vocoder.out.weight": [962, 128],
        ]
        for (i, pair) in [(481, 128), (241, 64), (121, 32)].enumerated() {
            shapes["decoder.mel_loss_fn.mel_transforms.\(i).mel_scale.fb"] = [pair.0, pair.1]
            shapes["decoder.mel_loss_fn.mel_transforms.\(i).spectrogram.window"] = [
                (pair.0 - 1) * 2
            ]
        }
        for i in 0 ..< 24 {
            layerShapes(prefix: "decoder.layers.\(i)", width: 1024, ffn: 4096, into: &shapes)
        }
        for (i, bins) in c.codebookSizes.enumerated() {
            let prefix = "encoder.quantizer.vq.layers.\(i)._codebook."
            shapes[prefix + "embed_avg"] = [bins, c.hiddenSize]
            shapes[prefix + "cluster_size"] = [bins]
            shapes[prefix + "inited"] = [1]
        }
        return shapes
    }

    /// The source digest is an owner-supplied reference, not hashing performed
    /// here. Root's authenticated filesystem/load session verifies the bytes.
    public static func preflight(
        descriptors: [String: MiMoV26AudioTensorDescriptor],
        configuration c: MiMoV26AudioInputConfiguration, sourcePayloadSHA256: String
    ) throws -> MiMoV26AudioSidecarPlan {
        guard sourcePayloadSHA256 == selectedPayloadSHA256, c.hiddenSize == 1024, c.layers == 24,
            c.heads == 16, c.ffnSize == 4096, c.melBands == 128,
            c.codebookSizes == [1024, 1024, 256] + Array(repeating: 128, count: 17)
        else {
            throw MiMoV26AudioInputError.weights("selected sidecar/profile identity")
        }
        let input = inputSourceShapes(configuration: c)
        let retained = retainedShapes(configuration: c)
        let expected = input.merging(retained) { _, _ in
            preconditionFailure("duplicate sidecar descriptor")
        }
        guard expected.count == 828, input.count == 389, Set(descriptors.keys) == Set(expected.keys)
        else {
            throw MiMoV26AudioInputError.weights("missing/unclassified sidecar tensors")
        }
        var inputBytes = 0
        var retainedBytes = 0
        for (name, shape) in expected {
            let d = descriptors[name]!
            let type: MiMoV26AudioTensorType =
                name.hasPrefix("encoder.") && !name.hasPrefix("encoder.quantizer.")
                ? .bfloat16 : .float32
            let count = try MiMoV26AudioChecked.product(shape + [type.bytes], name)
            guard d.shape == shape, d.dtype == type, d.byteCount == count else {
                throw MiMoV26AudioInputError.weights(name)
            }
            if input[name] != nil {
                inputBytes = try MiMoV26AudioChecked.add(inputBytes, count, "input weights")
            } else {
                retainedBytes = try MiMoV26AudioChecked.add(
                    retainedBytes, count, "retained asset weights")
            }
        }
        guard inputBytes == 634_204_160, retainedBytes == 1_238_321_176 else {
            throw MiMoV26AudioInputError.weights("sidecar byte closure")
        }
        return .init(
            configuration: c, descriptors: descriptors, requiredInputNames: Set(input.keys),
            sourcePayloadSHA256: sourcePayloadSHA256,
            inputStoredBytes: inputBytes, retainedUnusedStoredBytes: retainedBytes)
    }

    public static func load(plan: MiMoV26AudioSidecarPlan, inputWeights: [String: MLXArray]) throws
        -> MiMoV26AudioInputWeights
    {
        guard Set(inputWeights.keys) == plan.requiredInputNames else {
            throw MiMoV26AudioInputError.weights("input subset closure")
        }
        return try makeBundle(
            configuration: plan.configuration, weights: inputWeights,
            identity: plan.sourcePayloadSHA256)
    }
    static func fixtureBundle(
        configuration: MiMoV26AudioInputConfiguration, weights: [String: MLXArray]
    ) throws -> MiMoV26AudioInputWeights {
        try makeBundle(
            configuration: configuration, weights: weights, identity: "synthetic-hf5711-fixture")
    }
    private static func makeBundle(
        configuration c: MiMoV26AudioInputConfiguration, weights: [String: MLXArray],
        identity: String
    ) throws -> MiMoV26AudioInputWeights {
        let expected = inputSourceShapes(configuration: c)
        guard Set(weights.keys) == Set(expected.keys) else {
            throw MiMoV26AudioInputError.weights("input tensor keys")
        }
        for (name, shape) in expected {
            let type: DType = name.hasPrefix("encoder.quantizer.") ? .float32 : .bfloat16
            guard weights[name]!.shape == shape, weights[name]!.dtype == type else {
                throw MiMoV26AudioInputError.weights(name)
            }
        }
        let encoder = MiMoV26AudioTokenizerEncoder(configuration: c)
        try encoder.loadNativeNetworkWeights(
            weights.filter { !$0.key.hasPrefix("encoder.quantizer.") })
        let quantizer = try MiMoV26ResidualVectorQuantizer(
            configuration: c, sourceTables: c.codebookSizes.indices.map { weights[tableName($0)]! })
        return .init(
            encoder: encoder, quantizer: quantizer, configuration: c, sourceIdentity: identity,
            generation: encoder.loadedGeneration!)
    }
    static func prepareNetwork(
        _ weights: [String: MLXArray], configuration c: MiMoV26AudioInputConfiguration
    ) throws -> [String: MLXArray] {
        let expected = networkSourceShapes(configuration: c)
        guard Set(weights.keys) == Set(expected.keys) else {
            throw MiMoV26AudioInputError.weights("network tensor closure")
        }
        for (name, shape) in expected {
            guard weights[name]!.shape == shape, weights[name]!.dtype == .bfloat16 else {
                throw MiMoV26AudioInputError.weights(name)
            }
        }
        let convolutionNames: Set<String> = [
            "encoder.conv1.weight", "encoder.conv2.weight", "encoder.down_sample_layer.0.weight",
        ]
        return weights.mapValues { $0 }.map { name, value in
            (name, convolutionNames.contains(name) ? contiguous(value.transposed(0, 2, 1)) : value)
        }.reduce(into: [:]) { $0[$1.0] = $1.1 }
    }
}
