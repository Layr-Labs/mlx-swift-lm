import Foundation
import MLX
import MLXFast

/// Effective-selection count for the direct sorted-expert reduction. Benchmark
/// callers arm this after warmup and snapshot it only after the engine is idle.
/// The unarmed hot path reads one plain Bool and performs no atomic operation,
/// locking, allocation, or clock access.
public struct WeightedExpertUnsortStats: Sendable, Equatable {
    public let effectiveCalls: Int
}

/// Benchmark-facing requested/effective contract for one measured scope.
public struct WeightedExpertUnsortProvenance: Sendable, Equatable {
    public let requested: Bool
    public let effectiveCalls: Int

    public var engaged: Bool { effectiveCalls > 0 }
    public var missingExpectedEngagement: Bool { requested && !engaged }
}

private final class WeightedExpertUnsortProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var effectiveCalls = 0
    // Benchmark boundaries guarantee no engine work is in flight while this
    // plain flag changes. Concurrent recorders only read it while armed.
    private var enabled = false

    @inline(__always)
    func recordEffective() {
        guard enabled else { return }
        lock.lock()
        defer { lock.unlock() }
        // Defensively close a recorder/snapshot lock handoff. The idle-boundary
        // contract prevents a concurrent unsynchronized flag mutation.
        guard enabled else { return }
        effectiveCalls += 1
    }

    func snapshot() -> WeightedExpertUnsortStats {
        lock.lock()
        enabled = false
        defer { lock.unlock() }
        return WeightedExpertUnsortStats(effectiveCalls: effectiveCalls)
    }

    func reset() {
        lock.lock()
        effectiveCalls = 0
        enabled = true
        lock.unlock()
    }
}

private let weightedExpertUnsortProbe = WeightedExpertUnsortProbe()

/// Process-wide provenance snapshot for the weighted expert unsort experiment.
public func weightedExpertUnsortStats() -> WeightedExpertUnsortStats {
    weightedExpertUnsortProbe.snapshot()
}

/// Disarm and snapshot one benchmark scope with its resolved request state.
public func weightedExpertUnsortProvenance(
    requested: Bool
) -> WeightedExpertUnsortProvenance {
    let stats = weightedExpertUnsortStats()
    return WeightedExpertUnsortProvenance(
        requested: requested,
        effectiveCalls: stats.effectiveCalls)
}

/// Reset the provenance counters before a benchmark cell.
public func resetWeightedExpertUnsortStats() {
    weightedExpertUnsortProbe.reset()
}

/// Fused inverse-permutation + weighted reduction for the sorted MoE prefill path.
///
/// `SwitchGLU` sorts expert assignments before its gathered matrix multiplies.
/// The regular path restores `[tokens, topK, hidden]` and then reduces it with
/// ``weightedExpertSum``. This kernel reads those sorted rows through the inverse
/// permutation and writes `[tokens, hidden]` directly, avoiding that full
/// `[tokens, topK, hidden]` intermediate.
private let weightedExpertUnsortKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
    name: "weighted_expert_unsort",
    inputNames: ["sorted_outputs", "inverse_order", "weights"],
    outputNames: ["output"],
    source: """
        uint feature = thread_position_in_grid.x;
        uint token = thread_position_in_grid.y;

        T accumulator = (T)0;
        const uint assignment_base = token * (uint)K;
        for (uint slot = 0; slot < (uint)K; ++slot) {
            const uint assignment = assignment_base + slot;
            const uint sorted_row = (uint)inverse_order[assignment];
            // Preserve the legacy bfloat16 multiply-then-reduce rounding.
            const T weighted = (T)(
                (float)sorted_outputs[sorted_row * threads_per_grid.x + feature]
                * (float)weights[assignment]);
            accumulator = accumulator + weighted;
        }
        output[token * threads_per_grid.x + feature] = accumulator;
    """,
    ensureRowContiguous: true
)

/// Consume production-shaped sorted expert rows through their inverse
/// permutation and reduce original top-K slots into `[tokens, hidden]`.
///
/// Accepted layouts are bfloat16 `[tokens * K, hidden]` with `hidden % 64 == 0`,
/// uint32 inverse order, and bfloat16 `[tokens, K]` with `K >= 1` and at least
/// 64 assignments. Gemma 4 uses K=8/hidden=2816; native Qwen4 uses K=10/hidden=2560.
/// Callers retain the legacy scatter + weighted sum for every other layout.
public func weightedExpertUnsort(
    sortedOutputs: MLXArray,
    inverseOrder: MLXArray,
    weights: MLXArray
) -> MLXArray {
    weightedExpertUnsortOnStream(sortedOutputs: sortedOutputs, inverseOrder: inverseOrder,
                                weights: weights, stream: .default)
}

/// Internal stream-explicit twin for a forward-local deferred result. The
/// public API and all arithmetic, admission and effective-count behavior stay
/// unchanged; only the original construction stream is carried explicitly.
func weightedExpertUnsortOnStream(
    sortedOutputs: MLXArray, inverseOrder: MLXArray, weights: MLXArray,
    stream: StreamOrDevice
) -> MLXArray {
    let hidden = sortedOutputs.dim(1)
    let topK = weights.ndim == 2 ? weights.dim(1) : 0
    precondition(
        sortedOutputs.ndim == 2 && (hidden % 64 == 0)
            && sortedOutputs.dtype == .bfloat16,
        "weightedExpertUnsort outputs must be bfloat16 [assignments, hidden] with hidden % 64 == 0")
    precondition(
        inverseOrder.ndim == 1 && inverseOrder.dtype == .uint32,
        "weightedExpertUnsort inverse order must be flat uint32")
    precondition(
        weights.ndim == 2 && topK >= 1 && weights.size >= 64
            && weights.dtype == .bfloat16,
        "weightedExpertUnsort weights must be sorted-prefill bfloat16 [tokens, K]")
    precondition(
        sortedOutputs.dim(0) == weights.size && inverseOrder.size == weights.size,
        "weightedExpertUnsort assignment counts must match")

    let tokens = weights.dim(0)
    weightedExpertUnsortProbe.recordEffective()
    return weightedExpertUnsortKernel(
        [sortedOutputs, inverseOrder, weights],
        template: [
            ("T", sortedOutputs.dtype),
            ("K", topK),
        ],
        grid: (hidden, tokens, 1),
        threadGroup: (64, 4, 1),
        outputShapes: [[tokens, hidden]],
        outputDTypes: [.bfloat16],
        stream: stream
    )[0]
}


/// Record the same effective reduction when it is performed inside a fused
/// Gemma tail. Call only after that consumer has taken its pending projection.
func recordFusedWeightedExpertUnsort() {
    weightedExpertUnsortProbe.recordEffective()
}
