// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN

/// One NVIDIA ModelOpt NVFP4 matrix in MLX `nvfp4` layout: E2M1 words, one
/// FP8-E4M3 scale per 16 inputs and the per-tensor `weight_scale_2`, with
/// `W = e2m1 * fp8(scales) * weight_scale_2`. MLX's weight-only matmul takes
/// no global scale, so the scalar is applied to the output in FP32. The
/// unscaled product is `y / weight_scale_2`, which can exceed FP16 range:
/// callers run BF16 or FP32 activations.
final class GLM5NextNVFP4Linear: QuantizedLinear {
    /// F32 `[1]` (stored as a scalar), NVIDIA's `weight_scale_2` verbatim (not MLX's
    /// amax-style `globalScale`, which `QuantizedLinear` now declares and this type leaves nil).
    let weightScale2: MLXArray

    init(weight: MLXArray, scales: MLXArray, globalScale: MLXArray) throws {
        guard weight.ndim == 2, weight.dtype == .uint32, scales.dtype == .uint8,
            scales.shape == [weight.dim(0), weight.dim(1) / 2], weight.dim(1) % 2 == 0,
            globalScale.size == 1, globalScale.ndim <= 1, globalScale.dtype == .float32
        else { throw GLM5NextLoadError.invalid("nvfp4 matrix geometry") }
        try glm5NextRequirePositiveFinite(globalScale)
        self.weightScale2 = globalScale.reshaped(1)
        super.init(
            weight: weight, bias: nil, scales: scales, biases: nil, groupSize: 16, bits: 4,
            mode: .nvfp4)
        freeze()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let output = super.callAsFunction(x)
        return (output.asType(.float32) * weightScale2).asType(output.dtype)
    }
}

/// Refuses a zero, negative or non-finite stored NVFP4 global scale.
func glm5NextRequirePositiveFinite(_ scales: MLXArray) throws {
    let valid = try MLX.withError { error in
        let value = all(scales .> 0 .&& isFinite(scales)).item(Bool.self)
        try error.check()
        return value
    }
    guard valid else { throw GLM5NextLoadError.invalid("nvfp4 global scale") }
}
