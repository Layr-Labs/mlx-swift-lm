import MLX

/// A model that evaluates plain decode steps (every row one token, no
/// speculative round) with larger Metal command buffers than MLX's per-device
/// defaults, so the CPU encodes further ahead of the GPU. Such steps keep few
/// and small temporaries alive per buffer; prefill chunks (which may decode
/// weight tiles) and speculative rounds keep MLX's defaults and their memory.
public protocol CBv2DecodeCommandBufferLimitsProviding: AnyObject {
    var decodeCommandBufferLimits: GPU.CommandBufferLimits? { get }
}

extension EngineLoopV2 {
    /// Runs one step's graph construction and evaluation with the model's
    /// decode command-buffer limits when `plan` is a plain decode step.
    func withDecodeCommandBuffers<R>(
        _ plan: CBv2StepPlan, speculative: Bool = false, _ body: () throws -> R
    ) rethrows -> R {
        guard !speculative,
            let limits = (model as? CBv2DecodeCommandBufferLimitsProviding)?
                .decodeCommandBufferLimits,
            !plan.assignments.isEmpty, plan.assignments.allSatisfy({ $0.numTokens == 1 })
        else { return try body() }
        return try GPU.withCommandBufferLimits(limits, body)
    }
}
