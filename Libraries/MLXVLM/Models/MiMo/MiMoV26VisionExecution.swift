import MLX

extension MiMoV26VisionTower {
    /// Managed serving completes each frame and transformer block before
    /// constructing the next graph. The caller owns the native lane, tracks
    /// every live root, and must synchronously evaluate each checkpoint.
    /// Admission's frame working-set quote is valid only for this path.
    func forwardBounded(
        patches: MLXArray, grids: [MiMoV26VisionGrid], limits: MiMoV26VisionLimits,
        checkpoint: ([MLXArray]) throws -> Void
    ) throws -> MLXArray {
        guard patches.ndim == 2 else {
            throw MiMoV26VisionError.invalidInput("native flattened patch geometry")
        }
        let layout = try MiMoV26VisionLayout.make(
            grids: grids, mergeSize: configuration.spatialMergeSize,
            queryHeads: configuration.queryHeads, limits: limits)
        guard patches.dim(0) == layout.patchCount else {
            throw MiMoV26VisionError.invalidInput("patch count does not match grids")
        }
        var outputs: [MLXArray] = []
        var start = 0
        for grid in grids {
            // Layout validated these products and their aggregate above.
            let count = grid.height * grid.width
            for _ in 0 ..< grid.temporal {
                let frame = patches[start ..< (start + count)]
                let output = try forwardWithCheckpoints(
                    patches: frame,
                    grids: [.init(temporal: 1, height: grid.height, width: grid.width)],
                    limits: limits,
                    checkpoint: { values in try checkpoint([patches] + outputs + values) })
                outputs.append(output)
                start += count
            }
        }
        let result = concatenated(outputs, axis: 0)
        try checkpoint([patches] + outputs + [result])
        return result
    }
}
