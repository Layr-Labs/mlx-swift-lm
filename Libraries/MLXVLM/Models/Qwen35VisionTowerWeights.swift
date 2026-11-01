// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import MLXNN

extension Qwen35VisionTower {
    /// Scalar inventory of the existing unquantized Qwen3.5 visual module.
    /// This validates geometry before constructing random parameter graphs.
    /// Shapes use the published MLX layout, including O,T,H,W,C patch weights.
    public static func expectedTensorShapes(configuration: Qwen35Configuration) throws -> [String:
        [Int]]
    {
        try validate(configuration: configuration, textDType: .float32)
        let v = configuration.visionConfiguration
        let merged = try Qwen4ExpMediaGeometry.product(
            v.hiddenSize, v.spatialMergeSize, v.spatialMergeSize)
        let qkv = try Qwen4ExpMediaGeometry.product(v.hiddenSize, 3)
        var result: [String: [Int]] = [
            "vision_tower.patch_embed.proj.weight": [
                v.hiddenSize, v.temporalPatchSize, v.patchSize, v.patchSize, v.inChannels,
            ],
            "vision_tower.patch_embed.proj.bias": [v.hiddenSize],
            "vision_tower.pos_embed.weight": [v.numPositionEmbeddings, v.hiddenSize],
            "vision_tower.merger.norm.weight": [v.hiddenSize],
            "vision_tower.merger.norm.bias": [v.hiddenSize],
            "vision_tower.merger.linear_fc1.weight": [merged, merged],
            "vision_tower.merger.linear_fc1.bias": [merged],
            "vision_tower.merger.linear_fc2.weight": [v.outHiddenSize, merged],
            "vision_tower.merger.linear_fc2.bias": [v.outHiddenSize],
        ]
        for i in 0 ..< v.depth {
            let root = "vision_tower.blocks.\(i)."
            for norm in ["norm1", "norm2"] {
                result[root + norm + ".weight"] = [v.hiddenSize]
                result[root + norm + ".bias"] = [v.hiddenSize]
            }
            result[root + "attn.qkv.weight"] = [qkv, v.hiddenSize]
            result[root + "attn.qkv.bias"] = [qkv]
            result[root + "attn.proj.weight"] = [v.hiddenSize, v.hiddenSize]
            result[root + "attn.proj.bias"] = [v.hiddenSize]
            result[root + "mlp.linear_fc1.weight"] = [v.intermediateSize, v.hiddenSize]
            result[root + "mlp.linear_fc1.bias"] = [v.intermediateSize]
            result[root + "mlp.linear_fc2.weight"] = [v.hiddenSize, v.intermediateSize]
            result[root + "mlp.linear_fc2.bias"] = [v.hiddenSize]
        }
        // Check every eventual shape product while this is still host metadata.
        for shape in result.values {
            _ = try shape.reduce(1) { try Qwen4ExpMediaGeometry.product($0, $1) }
        }
        return result
    }

    /// One pinned tensor at a time. The caller gates each read and checkpoints
    /// evaluation while retaining this module on any failure until drain.
    /// No conversion of the stored floating arithmetic is performed here.
    public func loadNativeWeightsSerial(
        expectedDType: DType,
        read: (String, [Int]) throws -> MLXArray,
        checkpoint: ([MLXArray]) throws -> Void
    ) throws {
        guard [.bfloat16, .float16, .float32].contains(expectedDType) else {
            throw VLMError.processing("Unsupported Qwen visual storage dtype")
        }
        let expected = try Self.expectedTensorShapes(configuration: configuration)
        let constructed = Dictionary(
            uniqueKeysWithValues: parameters().flattened().map { ($0.0, $0.1.shape) })
        guard constructed == expected else {
            throw VLMError.processing("Qwen visual constructor differs from its admitted inventory")
        }
        for (name, shape) in expected.sorted(by: { $0.key < $1.key }) {
            let value = try read(name, shape)
            guard value.shape == shape, value.dtype == expectedDType else {
                throw VLMError.processing(
                    "Qwen visual tensor differs from its admitted shape or dtype")
            }
            try update(
                parameters: ModuleParameters.unflattened([name: value]),
                verify: [.noUnusedKeys, .shapeMismatch])
            try checkpoint([value])
        }
        freeze()
    }
}
