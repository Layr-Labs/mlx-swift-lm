import MLX

/// Explicit FP32 power contract of the pinned MLX vision reference. Generic
/// Metal `pow` depends on library/compiler math-function defaults; multiplication
/// and a compiled x**2 are not bit-identical substitutes for this operation.
/// Scope this to the new architecture, preserving all existing model numerics.
enum DiffusionGemmaVisionMath {
    private static let squareKernel = MLXFast.metalKernel(
        name: "diffusion_gemma_vision_precise_square",
        inputNames: ["x"], outputNames: ["out"],
        source: """
            uint i = thread_position_in_grid.x;
            out[i] = metal::precise::powr(metal::abs(x[i]), 2.0f);
            """)

    static func square(_ input: MLXArray) -> MLXArray {
        let x = input.asType(.float32)
        return squareKernel(
            [x], grid: (x.size, 1, 1), threadGroup: (min(256, x.size), 1, 1),
            outputShapes: [x.shape], outputDTypes: [.float32])[0]
    }

    private static let powerKernel = MLXFast.metalKernel(
        name: "diffusion_gemma_vision_precise_positive_power",
        inputNames: ["exponents", "base"], outputNames: ["out"],
        source: """
            uint i = thread_position_in_grid.x;
            out[i] = metal::precise::powr(base, exponents[i]);
            """)

    static func positivePower(base: Float, exponents: MLXArray) -> MLXArray {
        let x = exponents.asType(.float32)
        return powerKernel(
            [x, MLXArray(base)], grid: (x.size, 1, 1), threadGroup: (min(256, x.size), 1, 1),
            outputShapes: [x.shape], outputDTypes: [.float32])[0]
    }
}
