import Foundation
import MLX

/// A model that evaluates plain decode steps (every row one token, no
/// speculative round) with its own Metal command-buffer limits instead of
/// `CBv2DecodeCommandBufferDefault`; larger buffers let the CPU encode further
/// ahead of the GPU. Such steps keep few
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

/// Command-buffer limits for the plain decode steps of a model that declares
/// none: the value the families that declare limits measured (100 operations
/// per buffer; the input bound never triggers). Partitioning never changes a
/// result. A process that sets `MLX_MAX_OPS_PER_BUFFER` or
/// `MLX_MAX_MB_PER_BUFFER` (the cluster ranks pin 16 operations) keeps its own
/// limits, and `DARKBLOOM_CBV2_DECODE_COMMAND_BUFFERS=0` turns the default off.
/// A model that wants MLX's device defaults declares operations and megabytes 0.
public enum CBv2DecodeCommandBufferDefault {
    public static let limits = GPU.CommandBufferLimits(operations: 100, megabytes: 1 << 20)
    public static let enabled = isEnabled(environment: ProcessInfo.processInfo.environment)

    static func isEnabled(environment: [String: String]) -> Bool {
        if environment["DARKBLOOM_CBV2_DECODE_COMMAND_BUFFERS"] == "0" { return false }
        return environment["MLX_MAX_OPS_PER_BUFFER"] == nil
            && environment["MLX_MAX_MB_PER_BUFFER"] == nil
    }

    /// A plain decode step's limits: the model's own, else the default.
    static func decodeLimits(
        declared: GPU.CommandBufferLimits?, defaultEnabled: Bool = enabled
    ) -> GPU.CommandBufferLimits? {
        declared ?? (defaultEnabled ? limits : nil)
    }
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
        guard !plan.assignments.isEmpty else { return try body() }
        let provider = model as? CBv2DecodeCommandBufferLimitsProviding
        let limits: GPU.CommandBufferLimits?
        if speculative {
            limits =
                plan.assignments.allSatisfy({
                    $0.numTokens <= Self.speculativeCommandBufferRowTokens
                }) ? provider?.speculativeCommandBufferLimits : nil
        } else {
            limits =
                plan.assignments.allSatisfy({ $0.numTokens == 1 })
                ? CBv2DecodeCommandBufferDefault.decodeLimits(
                    declared: provider?.decodeCommandBufferLimits) : nil
        }
        guard let limits else { return try body() }
        return try GPU.withCommandBufferLimits(limits, body)
    }
}
