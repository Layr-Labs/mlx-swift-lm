// Copyright © 2026 Eigen Labs.
import Cmlx
import MLX

/// Only this score-derived producer can supply routes to the fixed-size kernels.
public final class Gemma4B8ExpertRouting {
    public let indices: MLXArray
    public let weights: MLXArray
    let rowOrder: MLXArray
    let sortedKeys: MLXArray
    let executionKeys: MLXArray
    let usesPrefixBounds: Bool
    let inverseOrder: MLXArray
    let reductionWeights: MLXArray
    let stream: StreamOrDevice

    private init(indices: MLXArray, weights: MLXArray, indexSnapshot: MLXArray,
                 weightSnapshot: MLXArray, stream: StreamOrDevice,
                 policy: Gemma4B8RoutePolicy, products: [MLXArray]?) {
        self.indices = indices
        self.weights = weights
        self.stream = stream
        let products = products ?? (policy.rank
            ? Gemma4B8RouteKernels.rank(indexSnapshot, policy: policy, stream: stream) : nil)
        if let products {
            rowOrder = products[0]
            executionKeys = products[1]
            inverseOrder = products[2]
            usesPrefixBounds = policy.prefixBounds
        } else {
            let flat = indexSnapshot.flattened()
            let order = argSort(flat, stream: stream)
            rowOrder = order.floorDivide(8, stream: stream)
            executionKeys = flat[order]
            inverseOrder = argSort(order, stream: stream)
            usesPrefixBounds = false
        }
        // Generic gather/bias APIs never receive tagged words.
        sortedKeys = usesPrefixBounds ? bitwiseAnd(executionKeys, UInt32(255), stream: stream) : executionKeys
        reductionWeights = weightSnapshot.reshaped(8, 8)
    }

    private static func snapshot(_ x: MLXArray) -> MLXArray? {
        var context = mlx_array_new()
        guard mlx_array_set(&context, x.ctx) == 0 else { mlx_array_free(context); return nil }
        return MLXArray(context)
    }

    public static func make(scores: MLXArray, perExpertScale: MLXArray,
                            finalists: Gemma4RouterFinalistsPolicy.Plan? = nil,
                            policy: Gemma4B8RoutePolicy = .process) -> Self? {
        let stream = StreamOrDevice.default
        guard stream == .gpu, scores.shape == [8, 1, 128], scores.dtype == .bfloat16,
            perExpertScale.shape == [128], perExpertScale.dtype == .bfloat16 else { return nil }
        let indices: MLXArray
        let weights: MLXArray
        var products: [MLXArray]?
        if policy.fold, let finalists, finalists.scoreShape == scores.shape, finalists.rows == 8 {
            let outputs = Gemma4B8RouteKernels.fold(scores: scores, scale: perExpertScale,
                policy: policy, stream: stream)
            indices = outputs[0]
            weights = outputs[1]
            products = Array(outputs[2..<5])
        } else if let finalists, let selected = Gemma4RouterFinalistsV1.apply(scores: scores,
            perExpertScale: perExpertScale, plan: finalists, stream: stream) {
            indices = selected.indices
            weights = selected.weights
        } else {
            let selected = argPartition(scores, kth: 120, axis: -1)[.ellipsis, 120...]
            var selectedWeights = takeAlong(scores, selected, axis: -1)
            selectedWeights = softmax(selectedWeights, axis: -1, precise: true)
            selectedWeights = selectedWeights * perExpertScale[selected]
            indices = selected
            weights = selectedWeights
        }
        guard indices.dtype == .uint32, let indexSnapshot = snapshot(indices),
            let weightSnapshot = snapshot(weights) else { return nil }
        return Self(indices: indices, weights: weights, indexSnapshot: indexSnapshot,
                    weightSnapshot: weightSnapshot, stream: stream, policy: policy, products: products)
    }
}
