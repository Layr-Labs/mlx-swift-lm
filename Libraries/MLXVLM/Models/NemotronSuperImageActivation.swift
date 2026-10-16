// Copyright © 2026 Eigen Labs.
import MLX

/// Upper bound on the MLX bytes that evaluating one image's tower and
/// projector graph allocates beyond its retained normalized input. Images are
/// evaluated one at a time, so a request needs the bound of its largest image.
/// The bound sums three phase bounds (embeddings, one transformer block, the
/// projector) rather than relying on MLX releasing a phase before the next.
/// It is derived from shapes, not measured; the global activation reserve is
/// a separate safeguard. Grids beyond 13312 patches return `Int.max`.
extension NemotronSuperImageComposition {
    public func activationBytes(patchRows rows: Int, patchColumns columns: Int) -> Int {
        Self.activationBytes(
            rows: rows, columns: columns, tower: tower.dimensions,
            projectorHidden: projector.projection.linear1.weight.dim(0),
            projectorOutput: projector.projection.linear2.weight.dim(0),
            elementBytes: visionDType.size)
    }

    /// The bound for the pinned composition every staged installation uses:
    /// C-RADIO (1280 wide, 16 heads, 5120 MLP, 10 prefix tokens) in bfloat16
    /// and the 5120 -> 20480 -> 4096 projector.
    public static func pinnedActivationBytes(patchRows rows: Int, patchColumns columns: Int) -> Int
    {
        activationBytes(
            rows: rows, columns: columns, tower: .superVL,
            projectorHidden: 20480, projectorOutput: 4096, elementBytes: DType.bfloat16.size)
    }

    static func activationBytes(
        rows: Int, columns: Int, tower d: NemotronRadioDimensions,
        projectorHidden: Int, projectorOutput: Int, elementBytes e: Int
    ) -> Int {
        // Larger grids are refused by preparation; they are never admissible.
        guard rows > 0, columns > 0, rows <= 13312, columns <= 13312, rows * columns <= 13312 else {
            return .max
        }
        let patches = rows * columns
        let tokens = patches + d.prefix
        let samples = patches * d.channels * d.patch * d.patch
        let float32 = 4
        // Pixel dtype cast and patch rearrangement; the FP32 position table and
        // three live FP32 bilinear terms; patch, position and prefix rows.
        let embeddings =
            2 * samples * e + d.positionSide * d.positionSide * d.width * float32
            + 3 * patches * d.width * float32 + 3 * tokens * d.width * e
        // Fourteen token-width tensors (residuals, norms, q/k/v and contiguous
        // copies, scaled queries, attention output and its transpose, the output
        // projection, fc2) and three MLP-width tensors (fc1 and GELU temporaries).
        var block = 14 * tokens * d.width * e + 3 * tokens * d.mlpWidth * e
        // MLX's unfused fallback materializes the scores and their softmax.
        if !fusedAttention(headDimension: d.width / d.heads, queryLength: tokens) {
            block += 2 * d.heads * tokens * tokens * e
        }
        // Final norm and two pixel-shuffle copies; the FP32 projection norm
        // (three live terms) and its cast; linear1, ReLU and square at the
        // hidden width; the projected output.
        let merged = patches / 4
        let projector =
            4 * patches * d.width * e + 3 * patches * d.width * float32
            + 3 * merged * projectorHidden * e + merged * projectorOutput * e
        return embeddings + block + projector
    }

    /// Mirrors `ScaledDotProductAttention::use_fallback` in the pinned MLX for
    /// unmasked inference on a GPU stream: whether the fused kernel runs.
    static func fusedAttention(headDimension: Int, queryLength: Int) -> Bool {
        if queryLength > 8 { return [64, 72, 80, 96, 128].contains(headDimension) }
        return [64, 96, 128, 256].contains(headDimension)
    }
}
