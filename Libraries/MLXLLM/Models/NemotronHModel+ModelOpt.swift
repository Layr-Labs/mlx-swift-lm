import MLX
import MLXLMCommon
import MLXNN

extension NemotronHModel {
    public enum ModelOptInstallationError: Error {
        case unsupportedModel
        case unsupportedProjection(String)
        case incompatibleProjection(String)
    }

    /// Installs a validated batch into explicitly supported bias-free projection
    /// slots. This is not a checkpoint loader: callers still verify payloads,
    /// completeness, model identity and admission, and serialize against inference.
    /// Lightning/MTP subclasses are excluded; their capabilities are not qualified
    /// for these eager calibrated projections. No mutation occurs on preflight error.
    public func installModelOptProjections(_ replacements: [String: ModelOptCalibratedLinear])
        throws
    {
        guard type(of: self) == NemotronHModel.self else {
            throw ModelOptInstallationError.unsupportedModel
        }
        let leaves = Dictionary(uniqueKeysWithValues: leafModules().flattened())
        var allowed: Set<String> = ["lm_head"]
        for (layer, kind) in configuration.hybridOverridePattern.enumerated() {
            let prefix = "backbone.layers.\(layer).mixer."
            let projections: [String]
            switch kind {
            case "M": projections = ["in_proj", "out_proj"]
            case "*": projections = ["q_proj", "k_proj", "v_proj", "o_proj"]
            case "-": projections = ["up_proj", "down_proj"]
            case "E": projections = ["shared_experts.up_proj", "shared_experts.down_proj"]
            default: projections = []
            }
            allowed.formUnion(projections.map { prefix + $0 })
        }
        for (path, replacement) in replacements {
            guard allowed.contains(path), let original = leaves[path] as? Linear,
                type(of: original) == Linear.self
            else {
                throw ModelOptInstallationError.unsupportedProjection(path)
            }
            let (output, input) = original.shape
            guard original.bias == nil, replacement.inputDimensions == input,
                replacement.outputDimensions == output,
                replacement.weightDType == original.weight.dtype
            else {
                throw ModelOptInstallationError.incompatibleProjection(path)
            }
        }
        // Update the actual parent module rather than unflattening a sparse
        // layers array: MLX's array updater requires contiguous dictionary nodes.
        let parents = Dictionary(uniqueKeysWithValues: namedModules())
        let updates = try replacements.map { path, value -> (Module, String, Module) in
            let parts = path.split(separator: ".").map(String.init)
            guard let key = parts.last,
                let parent = parents[parts.dropLast().joined(separator: ".")]
            else {
                throw ModelOptInstallationError.unsupportedProjection(path)
            }
            return (parent, key, value as Module)
        }
        for (parent, key, value) in updates {
            try parent.update(modules: .unflattened([(key, value)]), verify: .all)
        }
    }
}

extension NemotronHModel {
    /// Atomic preflight for routed packed experts. The two projections may be
    /// staged together; complete-checkpoint admission remains the caller's job.
    public func installModelOptExperts(_ replacements: [String: ModelOptNVFP4SwitchLinear]) throws {
        guard type(of: self) == NemotronHModel.self else {
            throw ModelOptInstallationError.unsupportedModel
        }
        let leaves = Dictionary(uniqueKeysWithValues: leafModules().flattened())
        var allowed = Set<String>()
        for (layer, kind) in configuration.hybridOverridePattern.enumerated() where kind == "E" {
            for projection in ["fc1", "fc2"] {
                allowed.insert("backbone.layers.\(layer).mixer.switch_mlp.\(projection)")
            }
        }
        for (path, replacement) in replacements {
            guard allowed.contains(path), let original = leaves[path] as? SwitchLinear,
                type(of: original) == SwitchLinear.self
            else { throw ModelOptInstallationError.unsupportedProjection(path) }
            let parameters = Dictionary(uniqueKeysWithValues: original.parameters().flattened())
            guard parameters["bias"] == nil,
                parameters["weight"]?.shape == replacement.logicalShape,
                replacement.originalWeightDType == parameters["weight"]?.dtype
            else {
                throw ModelOptInstallationError.incompatibleProjection(path)
            }
        }
        // Update the actual parent module rather than unflattening a sparse
        // layers array: MLX's array updater requires contiguous dictionary nodes.
        let parents = Dictionary(uniqueKeysWithValues: namedModules())
        let updates = try replacements.map { path, value -> (Module, String, Module) in
            let parts = path.split(separator: ".").map(String.init)
            guard let key = parts.last,
                let parent = parents[parts.dropLast().joined(separator: ".")]
            else {
                throw ModelOptInstallationError.unsupportedProjection(path)
            }
            return (parent, key, value as Module)
        }
        for (parent, key, value) in updates {
            try parent.update(modules: .unflattened([(key, value)]), verify: .all)
        }
    }
}

extension NemotronHModel {
    /// Serial native prefill bound for the calibrated correctness baseline.
    /// Mirrors explicit projection workspace limits; it is not a provider
    /// activation-floor estimate. The output head sees only the final position.
    public var calibratedPrefillTokenLimit: Int? {
        var found = false
        var limit = 32
        for (path, module) in leafModules().flattened() {
            if let projection = module as? ModelOptCalibratedLinear {
                found = true
                if path != "lm_head" {
                    let remaining = (64 << 20) / 32 - projection.inputDimensions
                    limit = min(
                        limit,
                        max(
                            0,
                            remaining / (projection.inputDimensions + projection.outputDimensions)))
                }
            }
            if let experts = module as? ModelOptNVFP4SwitchLinear {
                found = true
                let shape = experts.logicalShape
                let bytesPerAssignment = (shape[2] * 8 + shape[1] * 2) * 4
                limit = min(
                    limit, (64 << 20) / bytesPerAssignment / max(1, configuration.numExpertsPerTok))
                limit = min(limit, 8192 / max(1, configuration.numExpertsPerTok))
            }
        }
        return found ? limit : nil
    }
}
