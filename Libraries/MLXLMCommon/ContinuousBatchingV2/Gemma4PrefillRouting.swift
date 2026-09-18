// Copyright © 2026 Eigen Labs.
import Cmlx
import MLX
import MLXFast

/// Bounded routes derived from a 128-column score tensor, not a claim attached
/// to arbitrary indices. Public views are separate from the descriptor snapshots
/// used by the specialized consumer. The object is confined to one forward.
public final class Gemma4PrefillRouting {
    public let indices: MLXArray
    public let weights: MLXArray
    let flatIndices: MLXArray
    let flatWeights: MLXArray
    let rowOrder: MLXArray
    let sortedIndices: MLXArray
    let inverseOrder: MLXArray
    let tokenShape: [Int]
    let rows: Int
    let stream: StreamOrDevice

    private init(indices: MLXArray, weights: MLXArray, flatIndices: MLXArray, flatWeights: MLXArray,
                 sorted: (MLXArray, MLXArray, MLXArray), tokenShape: [Int], rows: Int, stream: StreamOrDevice) {
        self.indices = indices
        self.weights = weights
        self.flatIndices = flatIndices
        self.flatWeights = flatWeights
        self.rowOrder = sorted.0
        self.sortedIndices = sorted.1
        self.inverseOrder = sorted.2
        self.tokenShape = tokenShape
        self.rows = rows
        self.stream = stream
    }

    private static func snapshot(_ x: MLXArray) -> MLXArray? {
        var context = mlx_array_new()
        guard mlx_array_set(&context, x.ctx) == 0 else { mlx_array_free(context); return nil }
        return MLXArray(context)
    }

    /// The exact original router selection: argPartition, last eight indices,
    /// takeAlong, precise softmax and per-expert scale, without a dtype change.
    public static func make(scores: MLXArray, perExpertScale: MLXArray,
                            context: Gemma4PrefillGluePolicy.Context,
                            finalists: Gemma4RouterFinalistsPolicy.Plan? = nil) -> Self? {
        let stream = StreamOrDevice.default
        guard let plan = context.routePlan(scoreShape: scores.shape), perExpertScale.shape == [128],
            Gemma4PrefillGlueV1.gpuStream(stream) else { return nil }
        let indices: MLXArray
        let weights: MLXArray
        if let finalists, let selected = Gemma4RouterFinalistsV1.apply(scores: scores,
            perExpertScale: perExpertScale, plan: finalists, stream: stream) {
            // This producer also emits only original expert IDs in0..<128.
            indices = selected.indices
            weights = selected.weights
        } else {
            var selected = argPartition(scores, kth: 120, axis: -1)
            selected = selected[.ellipsis, 120...]
            var selectedWeights = takeAlong(scores, selected, axis: -1)
            selectedWeights = softmax(selectedWeights, axis: -1, precise: true)
            selectedWeights = selectedWeights * perExpertScale[selected]
            indices = selected
            weights = selectedWeights
        }
        guard indices.dtype == .uint32, let indexSnapshot = snapshot(indices),
            let weightSnapshot = snapshot(weights) else { return nil }
        let flatIndices = indexSnapshot.reshaped(plan.rows, 8)
        let flatWeights = weightSnapshot.reshaped(plan.rows, 8)
        let sorted = Gemma4RouteSort.sortBounded(flatIndices.flattened(), plan: plan, stream: stream)
        return Self(indices: indices, weights: weights, flatIndices: flatIndices, flatWeights: flatWeights,
            sorted: sorted, tokenShape: Array(scores.shape.dropLast()), rows: plan.rows, stream: stream)
    }
}

/// Private to the score-derived producer above: every key is an index into its
/// validated 128-column scores. No unbounded public/raw-index call reaches it.
private enum Gemma4RouteSort {
    private static let histogram = make("histogram", inputs: ["keys"], outputs: ["block_hist"], source: Gemma4RouteSortSources.histogram)
    private static let scan = make("scan", inputs: ["block_hist"], outputs: ["block_offset"], source: Gemma4RouteSortSources.scan)
    private static let parallelScan = make("parallel_scan", inputs: ["block_hist"], outputs: ["block_offset"], source: Gemma4RouteSortSources.parallelScan)
    private static let scatter = make("scatter", inputs: ["keys", "block_offset"], outputs: ["row_order", "sorted_keys", "inverse_order"], source: Gemma4RouteSortSources.scatter)
    private static let bitsetScatter = make("bitset_scatter", inputs: ["keys", "block_offset"], outputs: ["row_order", "sorted_keys", "inverse_order"], source: Gemma4RouteSortSources.bitsetScatter)

    private static func make(_ name: String, inputs: [String], outputs: [String], source: String) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(name: "db_gemma4_route_\(name)_v1", inputNames: inputs,
            outputNames: outputs, source: source, ensureRowContiguous: true)
    }

    static func sortBounded(_ keys: MLXArray, plan: Gemma4PrefillGluePolicy.Context.RoutePlan,
                            stream: StreamOrDevice) -> (MLXArray, MLXArray, MLXArray) {
        let hist = histogram([keys], grid: (plan.blocks * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[plan.blocks, 256]], outputDTypes: [.uint32], stream: stream)[0]
        let scanKernel = plan.parallelScan ? parallelScan : scan
        let threads = plan.parallelScan ? 1024 : 256
        let offsets = scanKernel([hist], template: [("NE", 128)], grid: (threads, 1, 1),
            threadGroup: (threads, 1, 1), outputShapes: [[plan.blocks, 256]], outputDTypes: [.uint32], stream: stream)[0]
        let scatterKernel = plan.bitset ? bitsetScatter : scatter
        let outputs = scatterKernel([keys, offsets], template: [("M", 8), ("NE", 128)],
            grid: (plan.blocks * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[plan.assignments], [plan.assignments], [plan.assignments]],
            outputDTypes: [.uint32, .uint32, .uint32], stream: stream)
        return (outputs[0], outputs[1], outputs[2])
    }
}
