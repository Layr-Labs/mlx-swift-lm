// Copyright © 2026 Eigen Labs.
import MLXLLM
import MLXLMCommon
import MLXNN

enum MiMoV26PackedModuleLoader {
    /// Constructs packed module shells only. Their lazy initializer parameters
    /// are replaced by authenticated checkpoint arrays before any evaluation.
    static func prepare(_ model: Module, policies: [String: MiMoV26Quantization.Policy]) throws {
        let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        var parents = Dictionary(uniqueKeysWithValues: model.namedModules())
        parents[""] = model
        var replacements: [String: [(String, Module)]] = [:]
        for path in policies.keys.sorted() {
            let policy = policies[path]!
            guard let leaf = leaves[path], !(leaf is Quantized),
                leaf is Linear || leaf is Embedding || leaf is SwitchLinear,
                let replacement = quantizeSingle(
                    layer: leaf, groupSize: policy.groupSize, bits: policy.bits,
                    mode: policy.mode == "mxfp4" ? .mxfp4 : .affine)
            else {
                throw MiMoV26ConvertedLoadError.incompatibleModule(path)
            }
            // Start at the real containing Module, not a sparse root update
            // through layers[0] when only later MoE layers change. Arrays such
            // as all speech embeddings are grouped under their actual owner.
            var pieces = path.split(separator: ".").map(String.init)
            pieces.removeLast()
            while !pieces.isEmpty && parents[pieces.joined(separator: ".")] == nil {
                pieces.removeLast()
            }
            let parent = pieces.joined(separator: ".")
            let relative = parent.isEmpty ? path : String(path.dropFirst(parent.count + 1))
            replacements[parent, default: []].append((relative, replacement))
        }
        for parent in replacements.keys.sorted() {
            do {
                try parents[parent]!.update(
                    modules: .unflattened(replacements[parent]!), verify: .noUnusedKeys)
            } catch { throw MiMoV26ConvertedLoadError.incompatibleModule(parent) }
        }
        let installed = Set(
            model.leafModules().flattened().compactMap { path, module in
                module is Quantized ? path : nil
            })
        guard installed == Set(policies.keys) else {
            throw MiMoV26ConvertedLoadError.incompatibleModule("quantized module closure")
        }
    }
}
