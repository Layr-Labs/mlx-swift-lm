// Checkpoint mapping follows NVIDIA modeling_radio.py (Apache-2.0),
// revision 7d4bc747b6bdc2cff64c58d65da0d08dd9caa25c. See NemotronRadio.swift.
import MLX
import MLXNN

extension NemotronRadio {
    public enum WeightError: Error {
        case missing(String), unexpected(String), duplicate(String), invalidShape(String), invalidDType(String)
    }

    /// Image-only extraction. Video projection and detached conditioner buffers
    /// are explicitly excluded, never substituted for image weights.
    public func loadImageWeights(from checkpoint: [String: MLXArray]) throws {
        let mapped = try validatedImageWeights(from: checkpoint)
        try update(parameters: ModuleParameters.unflattened(mapped), verify: [.all])
    }

    /// Pure metadata/graph preparation: no module parameter mutation.
    func validatedImageWeights(from checkpoint: [String: MLXArray]) throws -> [String: MLXArray] {
        let root = "vision_model.radio_model.model."
        let expected = Dictionary(uniqueKeysWithValues: parameters().flattened())
        var mapped: [String: MLXArray] = [:]
        func put(_ name: String, _ value: MLXArray) throws {
            guard mapped[name] == nil else { throw WeightError.duplicate(name) }
            guard let parameter = expected[name] else { throw WeightError.unexpected(name) }
            guard parameter.shape == value.shape else { throw WeightError.invalidShape(name) }
            guard [.float16, .bfloat16, .float32].contains(value.dtype) else {
                throw WeightError.invalidDType(name)
            }
            mapped[name] = value
        }
        for (name, tensor) in checkpoint {
            if expected[name] != nil { try put(name, tensor); continue }
            if name.hasPrefix("vision_model."), expected[String(name.dropFirst("vision_model.".count))] != nil {
                try put(String(name.dropFirst("vision_model.".count)), tensor); continue
            }
            if name == "vision_model.radio_model.input_conditioner.norm_mean"
                || name == "vision_model.radio_model.input_conditioner.norm_std"
                || name == "vision_model.summary_idxs"
                || name == root + "patch_generator.video_embedder.weight" { continue }
            guard name.hasPrefix("vision_model.") else { continue }
            guard name.hasPrefix(root) else { throw WeightError.unexpected(name) }
            let local = String(name.dropFirst(root.count))
            switch local {
            case "patch_generator.embedder.weight": try put("embeddings.projection.weight", tensor)
            case "patch_generator.pos_embed": try put("embeddings.position_embedding", tensor)
            case "patch_generator.cls_token.token": try put("embeddings.cls_register_token", tensor)
            default:
                let parts = local.split(separator: ".").map(String.init)
                guard parts.count >= 4, parts[0] == "blocks", let block = Int(parts[1]),
                    block >= 0 && block < dimensions.layers else { throw WeightError.unexpected(name) }
                let prefix = "blocks.\(block)."
                let suffix = parts.dropFirst(2).joined(separator: ".")
                if suffix == "attn.qkv.weight" || suffix == "attn.qkv.bias" {
                    let parameter = suffix.hasSuffix("weight") ? "weight" : "bias"
                    let shape = parameter == "weight" ? [3 * dimensions.width, dimensions.width] : [3 * dimensions.width]
                    guard tensor.shape == shape else { throw WeightError.invalidShape(name) }
                    for (index, destination) in ["query", "key", "value"].enumerated() {
                        let indices = (index * dimensions.width)..<((index + 1) * dimensions.width)
                        let piece = take(tensor, MLXArray(Array(indices)), axis: 0)
                        try put(prefix + destination + "." + parameter, piece)
                    }
                } else {
                    let mappedSuffix = suffix.replacingOccurrences(of: "attn.proj.", with: "output.")
                        .replacingOccurrences(of: "mlp.fc1.", with: "fc1.")
                        .replacingOccurrences(of: "mlp.fc2.", with: "fc2.")
                    try put(prefix + mappedSuffix, tensor)
                }
            }
        }
        for name in expected.keys where mapped[name] == nil { throw WeightError.missing(name) }
        return mapped
    }
}
