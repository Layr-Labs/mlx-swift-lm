import MLX

/// A model that evaluates plain decode steps (every row one token, no
/// speculative round) with larger Metal command buffers than MLX's per-device
/// defaults, so the CPU encodes further ahead of the GPU. Such steps keep few
/// and small temporaries alive per buffer; prefill chunks (which may decode
/// weight tiles) and speculative rounds keep MLX's defaults and their memory
/// unless the model also declares `speculativeCommandBufferLimits`.
public protocol CBv2DecodeCommandBufferLimitsProviding: AnyObject {
    var decodeCommandBufferLimits: GPU.CommandBufferLimits? { get }
    /// Limits for speculative (MTP) rounds whose rows are all decode or
    /// verification rows (at most `EngineLoopV2.speculativeCommandBufferRowTokens`
    /// tokens); rounds that carry a prompt chunk keep MLX's defaults.
    var speculativeCommandBufferLimits: GPU.CommandBufferLimits? { get }
}

extension CBv2DecodeCommandBufferLimitsProviding {
    public var speculativeCommandBufferLimits: GPU.CommandBufferLimits? { nil }
}

extension EngineLoopV2 {
    /// Longest row of a speculative round that takes the speculative limits.
    static let speculativeCommandBufferRowTokens = 8

    /// Runs one step's graph construction and evaluation with the model's
    /// decode command-buffer limits when `plan` is a plain decode step, or its
    /// speculative limits when `plan` is a speculative round of decode and
    /// verification rows.
    func withDecodeCommandBuffers<R>(
        _ plan: CBv2StepPlan, speculative: Bool = false, _ body: () throws -> R
    ) rethrows -> R {
        guard let provider = model as? CBv2DecodeCommandBufferLimitsProviding,
            !plan.assignments.isEmpty
        else { return try body() }
        let limits: GPU.CommandBufferLimits?
        if speculative {
            limits =
                plan.assignments.allSatisfy({
                    $0.numTokens <= Self.speculativeCommandBufferRowTokens
                }) ? provider.speculativeCommandBufferLimits : nil
        } else {
            limits =
                plan.assignments.allSatisfy({ $0.numTokens == 1 })
                ? provider.decodeCommandBufferLimits : nil
        }
        guard let limits else { return try body() }
        return try GPU.withCommandBufferLimits(limits, body)
    }
}
