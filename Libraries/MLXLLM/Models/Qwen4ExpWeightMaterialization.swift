import MLX
import MLXLMCommon

extension Qwen4ExpTextModel: IncrementalCheckpointMaterializing {
    public var needsIncrementalCheckpointMaterialization: Bool {
        namedModules().contains { ($0.1 as? SwitchGLU)?.hasFusedGateUp == true }
    }

    public func materializeCheckpointWeightsIncrementally() throws {
        // The loader relinquishes both staging owners before this hook.
        // Retire one split gate/up input pair before the next fused copy;
        // no quantization or projection arithmetic changes here.
        for (_, module) in namedModules() {
            guard let glu = module as? SwitchGLU, glu.hasFusedGateUp else { continue }
            try MLX.checkedEval(glu)
            Memory.clearCache()
        }
    }
}

extension Qwen4ExpModel: IncrementalCheckpointMaterializing {
    public var needsIncrementalCheckpointMaterialization: Bool {
        languageModel.needsIncrementalCheckpointMaterialization
    }

    public func materializeCheckpointWeightsIncrementally() throws {
        try languageModel.materializeCheckpointWeightsIncrementally()
    }
}
