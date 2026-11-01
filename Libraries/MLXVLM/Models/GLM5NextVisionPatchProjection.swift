// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import MLX

/// The source patch Conv3d's CPU contraction: 64 independent FP32 FMA lanes,
/// folded by halves to four lanes, then summed left to right before bias.
/// An explicit order avoids architecture-dependent BLAS vector reductions.
/// The caller bounds frame size; Metal continues to use its native matmul.
func glm5NextVisionCPUPatchProjection(
    patches: MLXArray, weight: MLXArray, bias: MLXArray?, dtype: DType,
    evaluate: ([MLXArray]) throws -> Void
) throws -> MLXArray {
    guard patches.ndim == 2, weight.ndim == 2, patches.dim(1) == weight.dim(1),
        bias == nil || bias!.shape == [weight.dim(0)]
    else { throw GLM5NextVisionError.invalidInput("CPU patch projection geometry") }
    let inputs = patches.asType(dtype).asType(.float32)
    let matrix = weight.asType(.float32)
    let offsets = bias?.asType(.float32)
    return try MLX.withError { error in
        try Task.checkCancellation()
        // The tower's owner may refuse before any tensor is evaluated or
        // copied to host storage. Keep its checkpoint on this eager CPU path.
        try evaluate([inputs, matrix] + (offsets.map { [$0] } ?? []))
        Stream.cpu.synchronize()
        try error.check()
        let x = inputs.asArray(Float.self)
        let w = matrix.asArray(Float.self)
        let b = offsets?.asArray(Float.self)
        try error.check()
        let rows = patches.dim(0)
        let columns = patches.dim(1)
        let channels = weight.dim(0)
        var output = [Float](repeating: 0, count: rows * channels)
        for row in 0 ..< rows {
            try Task.checkCancellation()
            let inputStart = row * columns
            for channel in 0 ..< channels {
                let weightStart = channel * columns
                var lanes = [SIMD4<Float>](repeating: .zero, count: 16)
                var column = 0
                while column + 4 <= columns {
                    let xi = inputStart + column
                    let wi = weightStart + column
                    let xv = SIMD4<Float>(x[xi], x[xi + 1], x[xi + 2], x[xi + 3])
                    let wv = SIMD4<Float>(w[wi], w[wi + 1], w[wi + 2], w[wi + 3])
                    let lane = (column / 4) % 16
                    lanes[lane] = lanes[lane].addingProduct(xv, wv)
                    column += 4
                }
                while column < columns {
                    let lane = column % 64
                    lanes[lane / 4][lane % 4] = lanes[lane / 4][lane % 4].addingProduct(
                        x[inputStart + column], w[weightStart + column])
                    column += 1
                }
                for width in [8, 4, 2, 1] {
                    for lane in 0 ..< width { lanes[lane] += lanes[lane + width] }
                }
                let total = ((lanes[0][0] + lanes[0][1]) + lanes[0][2]) + lanes[0][3]
                output[row * channels + channel] = total + (b?[channel] ?? 0)
            }
        }
        return MLXArray(output, [rows, channels]).asType(dtype)
    }
}
