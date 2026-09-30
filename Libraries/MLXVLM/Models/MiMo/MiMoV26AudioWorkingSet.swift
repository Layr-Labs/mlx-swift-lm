import MLX
import MLXLLM

/// Scratch for the owned PCM -> bounded encoder -> bounded RVQ path. Each
/// encoder block and quantizer step completes before the next is constructed.
/// Codec weights are charged by the separate authenticated load reservation.
enum MiMoV26AudioWorkingSet {
    static func bytes(_ plan: MiMoV26AudioInputPlan) throws -> Int {
        let c = plan.configuration
        func add(_ a: Int, _ b: Int) throws -> Int {
            try MiMoV26AudioChecked.add(a, b, "bounded audio work")
        }
        func mul(_ values: Int...) throws -> Int {
            try MiMoV26AudioChecked.product(values, "bounded audio work")
        }
        let fused: Bool
        #if os(macOS)
            fused = StreamOrDevice.default == .gpu && c.headDim == 64
        #else
            fused = false
        #endif
        let melFrames = try plan.melFrameCounts.reduce(0, add)
        let codeFrames = plan.totalCodeFrames
        let retained = try add(mul(melFrames, c.melBands, 4), mul(codeFrames, c.hiddenSize, 8))
        var encoder = 0
        for group in plan.groups {
            let segments = group.segmentIndices.map { plan.segments[$0] }
            let frames = try segments.reduce(0) { try add($0, $1.convFrames) }
            // Keep padded conv1/conv2, packed input, saved skip and output
            // features in addition to the one current transformer's graph.
            let convolutions = try mul(group.paddedMelFrames, c.hiddenSize, 8, 4)
            let transforms = try mul(frames, add(mul(c.hiddenSize, 24), mul(c.ffnSize, 4)), 4)
            var masks = 0
            for segment in segments {
                // Encoder really builds a dense causal/local mask. Fused SDPA
                // avoids per-head scores, not this shared mask.
                let width = try add(16, fused ? 0 : mul(c.heads, 16))
                masks = try add(masks, mul(segment.convFrames, segment.convFrames, width))
            }
            encoder = max(encoder, try add(convolutions, add(transforms, masks)))
        }
        let tile = min(codeFrames, plan.limits.rvqTileFrames)
        let bins = c.codebookSizes.max() ?? 0
        // One actual tile/codebook at a time: distance, finite/tie reductions,
        // selected indices, old/new residual and temporary table squares.
        let distances = try mul(tile, bins, 6, 4)
        let residuals = try mul(tile, c.hiddenSize, 6, 4)
        let tableWork = try mul(bins, c.hiddenSize, 2, 4)
        let rvq = try add(add(distances, residuals), tableWork)
        let front = try mul(plan.frontendWorkingElementUpperBound, 4)
        return try add(retained, max(front, max(encoder, rvq)))
    }
}
