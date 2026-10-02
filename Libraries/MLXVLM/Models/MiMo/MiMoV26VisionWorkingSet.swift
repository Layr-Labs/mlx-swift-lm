import MLX
import MLXLLM

/// Source-derived temporary allocation bound for forwardBounded. Every block
/// and temporal grid is synchronously evaluated before its successor. Decoded
/// pixels, patch backing and accumulated output features are charged separately.
enum MiMoV26VisionWorkingSet {
    static func frameBytes(
        _ geometry: MiMoV26MediaGeometry.Plan, configuration c: MiMoV26VisionConfiguration,
        stream: StreamOrDevice = .default
    ) throws -> Int {
        func mul(_ values: Int...) throws -> Int {
            try MiMoV26AudioChecked.product(values, "managed vision frame")
        }
        func add(_ a: Int, _ b: Int) throws -> Int {
            try MiMoV26AudioChecked.add(a, b, "managed vision frame")
        }
        let n = try mul(geometry.gridH, geometry.gridW)
        let shape = try MiMoV26VisionShape(c)
        let tiles = try add(n, 127) / 128
        let nodes = try add(64, add(256, mul(96, tiles)))
        let allocationSlack = try mul(nodes, 16_384)
        if usesFusedAttention(shape, stream: stream) {
            // Count live tensor families from VisionAttention/MLP, not all
            // layers or a score matrix the selected Metal kernel never makes.
            // All widths use FP32, including BF16 projections. Rotary keeps
            // FP32 input/rotation/products/result and the cast; the MLP keeps
            // gate/up/activation/product. Include both residuals, norm/proj
            // outputs, layout copies and attention output/concatenation.
            let residuals = try mul(c.hiddenSize, 8)
            let rotary = try add(mul(add(shape.qWidth, shape.kvWidth), 6), mul(shape.headDim, 4))
            let attention = try add(shape.fusedWidth, mul(shape.qWidth, 2))
            let mlp = try mul(c.intermediateSize, 4)
            let widths = try add(add(residuals, rotary), add(attention, mlp))
            let tensors = try mul(n, widths, 4)
            // Local masks have at most 128 queries x (128 + 2*window) keys;
            // charge every tile's Int32 positions/distance and boolean masks.
            // Full attention has no mask and neither path allocates N*N scores.
            let maskWidth = min(n, try add(128, mul(shape.window, 2)))
            let masks = try mul(n, maskWidth, 16)
            // Merger's MLP can be wider than the block for other configurations.
            let merger = try mul(n, add(mul(shape.mergeWidth, 4), c.outputHiddenSize), 4)
            return try add(max(try add(tensors, masks), merger), allocationSlack)
        }
        // Covers projections, rotary tables, normalizations, gated MLP, merger,
        // old/new residuals and layout indices, with FP32 widths and slack.
        let widths = try add(mul(c.hiddenSize, 64), mul(c.intermediateSize, 16))
        let activations = try mul(n, widths, 4)
        // Keep full attention as the peak even when local blocks use tiles.
        // This also covers kernels that materialize the head dimension.
        let scores = try scoreBytes(geometry, queryHeads: c.queryHeads)
        return try add(add(activations, scores), allocationSlack)
    }

    /// Mirror only the pinned Metal kernel case used by the published MiMo
    /// artifact. qk_channels defaults to 64, NOT hiddenSize/queryHeads (40).
    /// Full queries use fused SDPA; local 128-query tiles and <=8-query tails
    /// also fuse (GQA*tail <=32). Local keys <=256 never use two-pass scratch.
    /// CPU/custom streams, other head geometry and wider local windows retain
    /// the full-score allowance. Managed preparation quotes and executes in
    /// one synchronous engine-queue scope, without a stream-changing await.
    static func usesFusedAttention(
        _ shape: MiMoV26VisionShape, stream: StreamOrDevice = .default
    ) -> Bool {
        #if os(macOS)
            return stream == .gpu && shape.headDim == 64
                && shape.config.queryHeads / shape.config.keyValueHeads <= 4
                && shape.window <= 64
        #else
            return false
        #endif
    }

    static func scoreBytes(_ geometry: MiMoV26MediaGeometry.Plan, queryHeads: Int) throws -> Int {
        try MiMoV26AudioChecked.product(
            [geometry.gridH, geometry.gridW, geometry.gridH, geometry.gridW, queryHeads, 16],
            "managed vision frame scores")
    }
}
