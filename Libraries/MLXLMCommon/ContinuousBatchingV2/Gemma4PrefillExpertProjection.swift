// Copyright © 2026 Eigen Labs.
import Cmlx
import MLX

/// A synchronous, forward-local projection result. Only the SDK producer can
/// construct it. It either resolves through the original weighted reducer or
/// is consumed once by the fused tail; it is intentionally not Sendable.
public final class Gemma4PrefillExpertProjection {
    struct Pending {
        let sorted: MLXArray
        let inverse: MLXArray
        let rows: Int
        let weights: MLXArray
    }
    private var state: Gemma4DeferredExpertState<Pending, MLXArray>
    let stream: StreamOrDevice

    init(resolved: MLXArray) {
        state = .init(resolved: resolved)
        stream = .default
    }

    init?(sorted: MLXArray, order: Gemma4PrefillExpertOrder, weights: MLXArray) {
        guard sorted.shape == [order.rows * 8, 2816], sorted.dtype == .bfloat16,
            order.inverseOrder.shape == [order.rows * 8], order.inverseOrder.dtype == .uint32,
            weights.shape == [order.rows, 8], weights.dtype == .bfloat16,
            order.rows >= 8,
            let sortedSnapshot = Self.snapshot(sorted), let inverseSnapshot = Self.snapshot(order.inverseOrder),
            let weightSnapshot = Self.snapshot(weights) else { return nil }
        state = .init(pending: Pending(sorted: sortedSnapshot, inverse: inverseSnapshot,
                                      rows: order.rows, weights: weightSnapshot))
        stream = .default
    }

    // Defer the operation, not the caller's mutable Swift wrappers. Match the
    // descriptor-capture boundary of constructing the ordinary reducer now.
    private static func snapshot(_ array: MLXArray) -> MLXArray? {
        var context = mlx_array_new()
        guard mlx_array_set(&context, array.ctx) == 0 else {
            mlx_array_free(context)
            return nil
        }
        return MLXArray(context)
    }

    var pendingForFusion: Pending? { state.pending }

    func consumeForFusion() {
        precondition(state.consumePending() != nil, "Gemma expert projection was already consumed")
        recordFusedWeightedExpertUnsort()
    }

    /// Materialize at most once with the unchanged original reducer. Resolving
    /// after successful fusion is a caller error, not a second reduction.
    public func resolve() -> MLXArray {
        let constructionStream = stream
        guard let output = state.resolve(using: { pending in
            weightedExpertUnsortOnStream(sortedOutputs: pending.sorted,
                inverseOrder: pending.inverse, weights: pending.weights, stream: constructionStream)
        }) else { preconditionFailure("Gemma expert projection was consumed by a fused tail") }
        return output
    }
}
