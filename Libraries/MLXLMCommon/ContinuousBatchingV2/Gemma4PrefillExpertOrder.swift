// Copyright © 2026 Eigen Labs.
import MLX

/// An owned sort/inverse pair, produced exactly as gatherSort does. The
/// scatter kernel never accepts an arbitrary externally supplied permutation.
/// No host readback, index mutation or caching across projections.
struct Gemma4PrefillExpertOrder {
    let sortedIndices: MLXArray
    let inverseOrder: MLXArray
    let rows: Int
    private let gatherRows: MLXArray

    private init(sortedIndices: MLXArray, inverseOrder: MLXArray, rows: Int, gatherRows: MLXArray) {
        self.sortedIndices = sortedIndices
        self.inverseOrder = inverseOrder
        self.rows = rows
        self.gatherRows = gatherRows
    }

    static func make(indices: MLXArray, rows: Int, context: Gemma4PrefillGluePolicy.Context)
        -> Self? {
        guard let assignments = context.expertAssignments(rows: rows, indexShape: indices.shape,
            indicesUInt32: indices.dtype == .uint32) else { return nil }
        let flat = indices.flattened()
        let order = argSort(flat)
        let inverse = argSort(order)
        guard inverse.dtype == .uint32, inverse.shape == [assignments] else { return nil }
        return Self(sortedIndices: flat[order], inverseOrder: inverse, rows: rows, gatherRows: order.floorDivide(8))
    }

    static func fromRouting(_ routing: Gemma4PrefillRouting) -> Self {
        Self(sortedIndices: routing.sortedIndices, inverseOrder: routing.inverseOrder,
             rows: routing.rows, gatherRows: routing.rowOrder)
    }

    /// Exactly the original gatherSort input gather, using this same order.
    func gatherNormalized(_ x: MLXArray) -> MLXArray {
        x.reshaped(rows, 1, 2816)[gatherRows]
    }
}
