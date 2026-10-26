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
            && !isExpertPrefill && input.ndim == 3 && (1 ... 2).contains(input.dim(1))
            && (1 ... 7).contains(input.dim(0)) && input.dtype == .bfloat16
            && input.dim(2) == owner.config.hiddenSize
            && state.topK > 0 && input.dim(0) * input.dim(1) * state.topK < 64
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
