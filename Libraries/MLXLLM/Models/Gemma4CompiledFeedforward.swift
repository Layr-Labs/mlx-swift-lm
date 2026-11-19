import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The supported FFN's current arrays in a fixed order. This includes neither
/// attention parameters nor any KV, prefix, position, or persistence state.
struct Gemma4FeedforwardCompilationState {
    let parameters: [MLXArray]
    let modules: [Module]
    let topK: Int
    let suppliesHostRowIndices: Bool
}

private final class Gemma4WeakFeedforwardState: Updatable {
    weak var owner: Gemma4DecoderLayer?

    init(_ owner: Gemma4DecoderLayer) { self.owner = owner }

    func innerState() -> [MLXArray] {
        guard let state = owner?.feedforwardCompilationState() else {
            preconditionFailure("Gemma feed-forward topology changed during a compiled call")
        }
        return state.parameters
    }
}

/// Opaque holder so Module reflection does not expose a second parameter tree.
/// Weak state and weak closure captures avoid retaining the owning decoder.
final class Gemma4CompiledFeedforwardCache: CompiledTraceInvalidating {
    private struct Entry {
        // Retaining these objects prevents identity reuse after replacement.
        // They are the original child modules, not derived copies of weights.
        let modules: [Module]
        let suppliesHostRowIndices: Bool
        let forward: (MLXArray) -> MLXArray
    }

    private let lock = NSLock()
    private var entry: Entry?

    private static let enabled: Bool = {
        guard
            let raw = ProcessInfo.processInfo.environment[
                "DARKBLOOM_GEMMA4_COMPILED_FEEDFORWARD"]
        else { return true }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()

    /// Steps whose top-K assignments reach 64 (at top-8: 8 to 16 decode rows,
    /// or 4 to 8 verification rows of two tokens) take the sorted expert path;
    /// they compile unless `DARKBLOOM_GEMMA4_COMPILED_FEEDFORWARD_SORTED` is set
    /// to something other than 1, true, yes or on.
    static let sortedEnabled = sortedEnabled(
        ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_COMPILED_FEEDFORWARD_SORTED"])

    static func sortedEnabled(_ raw: String?) -> Bool {
        guard let raw else { return true }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }

    /// Tests only: replaces the environment's choice for sorted rows (nil: the
    /// environment's) in the calling task.
    @TaskLocal static var sortedOverrideForTesting: Bool?

    /// Whether `[rows, width]` decode or verification input may take the trace.
    /// Up to 7 rows below 64 assignments keep the unsorted expert path. Up to 16
    /// tokens at 64 or more assignments sort them inside the trace exactly as
    /// the eager graph does (16 covers c = 8 verification and c = 16 decode).
    /// A scoped batch-decode or prompt projection keeps those sorted sizes eager.
    static func admitsShape(rows: Int, width: Int, topK: Int) -> Bool {
        guard topK > 0, (1 ... 2).contains(width), rows >= 1 else { return false }
        let assignments = rows * width * topK
        if rows <= 7 && assignments < 64 { return true }
        return (sortedOverrideForTesting ?? sortedEnabled)
            && assignments >= 64 && rows * width <= 16
            && !Gemma4BatchDecodeProjection.active && !Gemma4PrefillProjection.active
    }

    var isCompiled: Bool { lock.withLock { entry != nil } }

    func invalidate() {
        let retired = lock.withLock {
            let retired = entry
            entry = nil
            return retired
        }
        // CompiledFunction.deinit acquires MLX's eval lock. Retire outside
        // this holder's lock to preserve the eval-lock -> holder-lock order.
        withExtendedLifetime(retired) {}
    }

    func callAsFunction(
        _ owner: Gemma4DecoderLayer, _ input: MLXArray, isExpertPrefill: Bool
    ) -> MLXArray? {
        guard Self.enabled, MLXHardwareInfo.isCompiledDecodeSupported else { return nil }
        // Structural fallback must also retire an older trace. Otherwise an
        // adapter or unsupported replacement would retain the old weight bank.
        guard let state = owner.feedforwardCompilationState() else {
            invalidate()
            return nil
        }
        let canCompile =
            Device.defaultDevice().deviceType == .gpu
            && !isExpertPrefill && input.ndim == 3 && input.dtype == .bfloat16
            && input.dim(2) == owner.config.hiddenSize
            && Self.admitsShape(rows: input.dim(0), width: input.dim(1), topK: state.topK)
            // Preserve existing per-invocation component receipts.
            && !CBv2ForwardShapeObservation.isActive

        let result: (forward: ((MLXArray) -> MLXArray)?, retired: Entry?) = lock.withLock {
            var retired: Entry?
            if let current = entry,
                current.suppliesHostRowIndices != state.suppliesHostRowIndices
                    || current.modules.count != state.modules.count
                    || !zip(current.modules, state.modules).allSatisfy({ $0 === $1 })
            {
                retired = current
                entry = nil
            }
            // A changed topology is retired even during prefill, training,
            // CPU execution or another call that must keep the eager path.
            guard canCompile else { return (nil, retired) }
            if let entry { return (entry.forward, retired) }
            let weights = Gemma4WeakFeedforwardState(owner)
            let created = compile(inputs: [weights], shapeless: false) {
                [weak owner] (input: MLXArray) -> MLXArray in
                guard let owner else {
                    preconditionFailure("Gemma feed-forward trace outlived its decoder")
                }
                return owner.feedforward(input, isExpertPrefill: false)
            }
            entry = Entry(
                modules: state.modules,
                suppliesHostRowIndices: state.suppliesHostRowIndices,
                forward: created)
            return (created, retired)
        }
        withExtendedLifetime(result.retired) {}
        guard let forward = result.forward else { return nil }
        // Enter MLX's eval/compile locks only after releasing the holder lock.
        return forward(input)
    }
}
