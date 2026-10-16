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
    /// Largest graph prefill chunk; the CBv2 engine may choose smaller ones.
    public static let modelOptGraphPrefillTokenCap = 2048
    /// Explicit array workspace of one graph prefill chunk's ModelOpt calls,
    /// sized for production chunks. The owner admits it with the chunk.
    public static let modelOptGraphPrefillWorkspaceLimit = 2 << 30

    /// Every installed ModelOpt module selects `execution`; non-ModelOpt
    /// layers are untouched. Returns the number of modules switched.
    @discardableResult
    public func setModelOptExecution(_ execution: ModelOptExecution) -> Int {
        var count = 0
        for (_, module) in leafModules().flattened() {
            if let projection = module as? ModelOptCalibratedLinear {
                projection.execution = execution
                count += 1
            } else if let experts = module as? ModelOptNVFP4SwitchLinear {
                experts.execution = execution
                count += 1
            }
        }
        return count
    }

    /// Opt-in batched decode and verify windows: with 2 to 16 rows, every
    /// ModelOpt projection, router and latent projection gives each row the
    /// bits of a one-row call, reading each weight once. Routed experts already
    /// run one single-vector product per row under `.rowwise`.
    public func setModelOptRowExactBatches(_ enabled: Bool) {
        for (_, module) in namedModules() {
            (module as? ModelOptCalibratedLinear)?.rowExactBatches = enabled
            (module as? NemotronHMoE)?.rowExactBatches = enabled
            (module as? NemotronHMoEGate)?.rowExactBatches = enabled
        }
    }

    /// Selects the accumulation of graph products with many rows (prefill).
    public func setModelOptAccumulation(_ accumulation: ModelOptAccumulation) {
        for (_, module) in leafModules().flattened() {
            (module as? ModelOptCalibratedLinear)?.accumulation = accumulation
            (module as? ModelOptNVFP4SwitchLinear)?.accumulation = accumulation
        }
    }

    /// Explicit ModelOpt array bytes of the heaviest layer for one prefill
    /// chunk of `tokens`, all of the layer's calls assumed live together:
    /// each projection's float32 activation, float32 sums, output and FP8
    /// decoded slice, plus the routed input/output copies of sorted experts.
    /// nil unless every module runs the graph path. The head sees one row.
    public func modelOptGraphPrefillWorkspaceBytes(tokens: Int) -> Int? {
        guard tokens > 0 else { return nil }
        var layers: [String: Int] = [:]
        var found = false
        let topK = max(1, configuration.numExpertsPerTok)
        for (path, module) in leafModules().flattened() {
            let layer = path.split(separator: ".").prefix(3).joined(separator: ".")
            let bytes: Int?
            if let projection = module as? ModelOptCalibratedLinear {
                guard projection.execution == .graph else { return nil }
                found = true
                bytes = try? projection.graphWorkspaceBytes(rows: path == "lm_head" ? 1 : tokens)
            } else if let experts = module as? ModelOptNVFP4SwitchLinear {
                guard experts.execution == .graph else { return nil }
                found = true
                let rows = tokens.multipliedReportingOverflow(by: topK)
                let copies = 2 * experts.logicalShape[2] * (experts.originalWeightDType?.size ?? 4)
                bytes =
                    rows.overflow
                    ? nil
                    : (try? experts.graphWorkspaceBytes(rows: rows.partialValue))
                        .map { $0 + rows.partialValue * copies }
            } else {
                continue
            }
            guard let bytes else { return nil }
            layers[layer, default: 0] += bytes
        }
        return found ? layers.values.max() ?? 0 : nil
    }

    /// Serial native prefill bound. The eager baseline mirrors its 64 MiB
    /// projection workspace (at most 32 tokens). The graph path admits
    /// `modelOptGraphPrefillWorkspaceLimit` bytes of
    /// `modelOptGraphPrefillWorkspaceBytes(tokens:)` for one chunk, up to
    /// `modelOptGraphPrefillTokenCap` tokens. Neither is an activation floor.
    public var calibratedPrefillTokenLimit: Int? {
        if let one = modelOptGraphPrefillWorkspaceBytes(tokens: 1) {
            // Nondecreasing in the chunk length, so bisect for the largest fit.
            guard one <= Self.modelOptGraphPrefillWorkspaceLimit else { return 1 }
            var low = 1
            var high = Self.modelOptGraphPrefillTokenCap
            while low < high {
                let middle = (low + high + 1) / 2
                if let bytes = modelOptGraphPrefillWorkspaceBytes(tokens: middle),
                    bytes <= Self.modelOptGraphPrefillWorkspaceLimit
                {
                    low = middle
                } else {
                    high = middle - 1
                }
            }
            return low
        }
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
