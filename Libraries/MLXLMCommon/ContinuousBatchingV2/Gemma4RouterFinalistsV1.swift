// Copyright © 2026 Eigen Labs.
import MLX
import MLXFast

/// Explicit score-derived selection. Every output index is an original expert
/// index in0..<128. The ordinary weight chain remains the selection-only fallback.
public enum Gemma4RouterFinalistsV1 {
    private static let selection = MLXFast.metalKernel(name: "db_gemma4_finalists_selection_v1",
        inputNames: ["scores"], outputNames: ["indices"], source: Gemma4RouterFinalistsSources.selection,
        header: Gemma4RouterFinalistsSources.selectionHeader, ensureRowContiguous: true)
    private static let nativeWeights = MLXFast.metalKernel(name: "db_gemma4_finalists_native_weights_v1",
        inputNames: ["scores", "pes"], outputNames: ["indices", "weights"], source: Gemma4RouterFinalistsSources.nativeKeysWeights,
        header: Gemma4RouterFinalistsSources.weightsHeader(orderKeysEnabled: true), ensureRowContiguous: true)
    private static let bitonicWeights = MLXFast.metalKernel(name: "db_gemma4_finalists_bitonic_weights_v1",
        inputNames: ["scores", "pes"], outputNames: ["indices", "weights"], source: Gemma4RouterFinalistsSources.bitonicWeights,
        header: Gemma4RouterFinalistsSources.weightsHeader(orderKeysEnabled: false), ensureRowContiguous: true)

    public static func apply(scores: MLXArray, perExpertScale: MLXArray,
        plan: Gemma4RouterFinalistsPolicy.Plan, stream: StreamOrDevice = .default)
        -> (indices: MLXArray, weights: MLXArray)? {
        guard scores.shape == plan.scoreShape, scores.dtype == .bfloat16,
            perExpertScale.shape == [128], Gemma4PrefillGlueV1.gpuStream(stream) else { return nil }
        var shape = scores.shape
        shape[2] = 8
        if plan.fusedWeights {
            guard perExpertScale.dtype == .bfloat16 else { return nil }
            let result = (plan.nativeOrderKeys ? nativeWeights : bitonicWeights)([scores, perExpertScale],
                template: [("T", DType.bfloat16)], grid: (plan.rows * plan.threads, 1, 1),
                threadGroup: (plan.threads, 1, 1), outputShapes: [shape, shape],
                outputDTypes: [.uint32, .bfloat16], stream: stream)
            return (result[0], result[1])
        }
        let indices = selection([scores], grid: (plan.rows * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [shape], outputDTypes: [.uint32], stream: stream)[0]
        var weights = takeAlong(scores, indices, axis: -1, stream: stream)
        weights = softmax(weights, axis: -1, precise: true, stream: stream)
        weights = multiply(weights, take(perExpertScale, indices, axis: 0, stream: stream), stream: stream)
        return (indices, weights)
    }
}
