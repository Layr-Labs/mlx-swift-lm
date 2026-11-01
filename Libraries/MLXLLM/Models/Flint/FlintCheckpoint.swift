// Official mixed-bit loading adapted from Recursive rec (Apache-2.0).
// BF16 expert stacking adapted from Prince Canuma / mlx-vlm (MIT).
// See ThirdPartyNotices/Flint.md for immutable sources and full notices.
import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum FlintCheckpointError: Error, LocalizedError {
    case invalid(String)
    public var errorDescription: String? {
        switch self {
        case .invalid(let reason): "Invalid Flint checkpoint: \(reason)"
        }
    }
}

extension FlintModel {
    /// Flint's official export stores packed bytes and an INT8 embedding despite
    /// a default INT4 policy. Keep this adaptation model-owned; never infer bits
    /// from arbitrary malformed shapes or change the generic loader's defaults.
    public func loadFlintWeights(from directory: URL) throws {
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw FlintCheckpointError.invalid("no weight shards") }
        var weights = [String: MLXArray]()
        for file in files {
            for (key, value) in try loadArrays(url: file) {
                guard weights.updateValue(value, forKey: key) == nil else {
                    throw FlintCheckpointError.invalid("duplicate tensor \(key)")
                }
            }
        }
        try installFlintWeights(weights)
        eval(self)
    }

    /// Also used by bounded fixture tests. Validation occurs before strict module
    /// update, so missing experts and conflicting aliases cannot be synthesized.
    public func installFlintWeights(_ raw: [String: MLXArray]) throws {
        var weights = try normalizedFlintWeights(raw)
        let isExport = configuration.mlxExport != nil
        var expected = [String: WeightRequirement]()
        var quantizedPaths = Set<String>()
        for (name, parameter) in parameters().flattened() {
            let shape = parameter.shape
            if isExport, name.hasSuffix(".weight"), shape.count >= 2,
                !name.hasSuffix(".mlp.gate.weight")
            {
                let path = String(name.dropLast(".weight".count))
                let bits = path == "model.embed_tokens" ? 8 : 4
                guard let width = shape.last, width % 64 == 0 else {
                    throw FlintCheckpointError.invalid("expected group-64 packed weight at \(name)")
                }
                var packedShape = shape
                packedShape[packedShape.count - 1] = width / 32 * bits
                var scaleShape = shape
                scaleShape[scaleShape.count - 1] = width / 64
                expected[name] = WeightRequirement(shape: packedShape, dtypes: [.uint32])
                expected[path + ".scales"] = WeightRequirement(
                    shape: scaleShape, dtypes: [.bfloat16])
                expected[path + ".biases"] = WeightRequirement(
                    shape: scaleShape, dtypes: [.bfloat16])
                quantizedPaths.insert(path)
            } else {
                let dtypes: [DType] =
                    !isExport || name.hasSuffix(".expert_bias")
                    ? [.bfloat16, .float32] : [.bfloat16]
                expected[name] = WeightRequirement(shape: shape, dtypes: dtypes)
            }
        }
        guard Set(expected.keys) == Set(weights.keys) else {
            let missing = Set(expected.keys).subtracting(weights.keys).sorted().prefix(8)
            let extra = Set(weights.keys).subtracting(expected.keys).sorted().prefix(8)
            throw FlintCheckpointError.invalid(
                "missing \(Array(missing)); unexpected \(Array(extra))")
        }
        for (name, requirement) in expected {
            guard let value = weights[name], value.shape == requirement.shape else {
                throw FlintCheckpointError.invalid("incorrect shape at \(name)")
            }
            guard requirement.dtypes.contains(value.dtype) else {
                throw FlintCheckpointError.invalid("unsupported dtype at \(name)")
            }
            if name.hasSuffix(".softmax_offset") { weights[name] = value.asType(.float32) }
        }
        // Validate the entire checkpoint before replacing any module. A rejected
        // export must not leave the model partially quantized and unusable on retry.
        if isExport {
            quantize(model: self) { path, _ in
                guard quantizedPaths.contains(path) else { return nil }
                return (groupSize: 64, bits: path == "model.embed_tokens" ? 8 : 4, mode: .affine)
            }
        }
        try update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        // Exported (quantized) layers then share matmuls across projections of
        // one input; the checkpoint's tensors are validated above unchanged.
        if isExport { try fuseProjections() }
    }

    private struct WeightRequirement {
        let shape: [Int]
        let dtypes: [DType]
    }

    func normalizedFlintWeights(_ raw: [String: MLXArray]) throws -> [String: MLXArray] {
        var weights = [String: MLXArray]()
        for (rawKey, value) in raw {
            let key =
                rawKey.hasPrefix("language_model.")
                ? String(rawKey.dropFirst("language_model.".count)) : rawKey
            guard weights[key] == nil else {
                throw FlintCheckpointError.invalid("conflicting alias \(key)")
            }
            if value.dtype == .uint8 {
                guard configuration.mlxExport != nil, key.hasSuffix(".weight"),
                    value.ndim >= 2, value.dim(-1) % 4 == 0
                else {
                    throw FlintCheckpointError.invalid("unexpected packed bytes at \(key)")
                }
                weights[key] = value.view(dtype: .uint32)
            } else {
                weights[key] = value
            }
        }
        for index in 0 ..< configuration.hiddenLayers
        where !configuration.mlpOnlyLayers.contains(index) {
            let prefix = "model.layers.\(index).mlp."
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let source = (0 ..< configuration.numExperts).map {
                    prefix + "experts.\($0).\(projection).weight"
                }
                if source.contains(where: { weights[$0] != nil }) {
                    let destination = prefix + "switch_mlp.\(projection).weight"
                    guard configuration.mlxExport == nil, weights[destination] == nil,
                        source.allSatisfy({ weights[$0] != nil })
                    else {
                        throw FlintCheckpointError.invalid(
                            "incomplete or conflicting experts at \(destination)")
                    }
                    let expectedShape =
                        projection == "down_proj"
                        ? [configuration.hiddenSize, configuration.moeIntermediateSize]
                        : [configuration.moeIntermediateSize, configuration.hiddenSize]
                    let experts = source.compactMap { weights[$0] }
                    guard
                        experts.allSatisfy({
                            $0.shape == expectedShape
                                && ($0.dtype == .bfloat16 || $0.dtype == .float32)
                        })
                    else {
                        throw FlintCheckpointError.invalid(
                            "invalid expert shape or dtype at \(destination)")
                    }
                    weights[destination] = stacked(experts)
                    for key in source { weights.removeValue(forKey: key) }
                }
            }
            let bias = prefix + "expert_bias"
            for alias in [prefix + "gate.expert_bias", prefix + "gate.e_score_correction_bias"] {
                if let value = weights.removeValue(forKey: alias) {
                    guard weights[bias] == nil else {
                        throw FlintCheckpointError.invalid("conflicting router bias")
                    }
                    weights[bias] = value
                }
            }
            // The published release has no expert correction bias. Only this
            // documented optional parameter may be supplied by the loader.
            if weights[bias] == nil { weights[bias] = MLXArray.zeros([configuration.numExperts]) }
        }
        return weights
    }
}
