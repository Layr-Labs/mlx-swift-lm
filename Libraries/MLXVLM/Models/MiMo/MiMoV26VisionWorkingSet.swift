import MLXLLM

/// Source-derived temporary allocation bound for forwardBounded. Every block
/// and temporal grid is synchronously evaluated before its successor. Decoded
/// pixels, patch backing and accumulated output features are charged separately.
enum MiMoV26VisionWorkingSet {
    static func frameBytes(
        _ geometry: MiMoV26MediaGeometry.Plan, configuration c: MiMoV26VisionConfiguration
    ) throws -> Int {
        func mul(_ values: Int...) throws -> Int {
            try MiMoV26AudioChecked.product(values, "managed vision frame")
        }
        func add(_ a: Int, _ b: Int) throws -> Int {
            try MiMoV26AudioChecked.add(a, b, "managed vision frame")
        }
        let n = try mul(geometry.gridH, geometry.gridW)
        // Covers projections, rotary tables, normalizations, gated MLP, merger,
        // old/new residuals and layout indices, with FP32 widths and slack.
        let widths = try add(mul(c.hiddenSize, 64), mul(c.intermediateSize, 16))
        let activations = try mul(n, widths, 4)
        // Keep full attention as the peak even when local blocks use tiles.
        // This also covers kernels that materialize the head dimension.
        let scores = try scoreBytes(geometry, queryHeads: c.queryHeads)
        let tiles = try add(n, 127) / 128
        let nodes = try add(64, add(256, mul(96, tiles)))
        return try add(add(activations, scores), mul(nodes, 16_384))
    }

    static func scoreBytes(_ geometry: MiMoV26MediaGeometry.Plan, queryHeads: Int) throws -> Int {
        try MiMoV26AudioChecked.product(
            [geometry.gridH, geometry.gridW, geometry.gridH, geometry.gridW, queryHeads, 16],
            "managed vision frame scores")
    }
}
