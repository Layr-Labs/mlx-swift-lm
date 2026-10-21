// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0
// Chunk form of `recurrent_kimi_delta_attention` in Transformers
// f0705f991a71f05dad9569336b181119f5546f93; the WY/UT decomposition follows
// flash-linear-attention's `naive_chunk_kda`.

import MLX

/// Chunk-parallel KDA recurrence over the same precomputed FP32 inputs as
/// `GLM5NextLinearKernel.apply`. Within a chunk of C tokens every quantity is a
/// batched matrix product; only the C-token chunk boundaries are sequential.
///
/// With Γ_t = exp(G_t), G the inclusive in-chunk cumulative log decay and S the
/// state entering the chunk, the per-token recurrence unrolls exactly to
///   δ = (I + M)⁻¹ diag(β) (V − (Γ⊙K) S),  M[t,j] = β_t k_t·(Γ_t/Γ_j ⊙ k_j), j < t
///   O = (Γ⊙Q) S + P δ,                     P[t,j] = q_t·(Γ_t/Γ_j ⊙ k_j), j ≤ t
///   S' = diag(Γ_C) S + (Γ_C/Γ ⊙ K)ᵀ δ.
/// Γ_t/Γ_j is factored through a mid-chunk pivot r as exp(G_t−G_r)·exp(G_r−G_j);
/// the admitted chunk size keeps both exponents within ±`maximumExponent`, so
/// neither factor overflows. (Products of the factors, and exp(G) itself, may
/// underflow to subnormals or zero, as the decayed terms do in the recurrence.)
/// (I + M)⁻¹ is solved by forward substitution, as in FLA's `naive_chunk_kda`:
/// a truncated Neumann product grows like (1 + a)^(C−2) when keys are nearly
/// parallel and beta is near one.
///
/// Mathematically equal to the recurrence, not bitwise: products are
/// reassociated. The substitution uses elementwise FP32 multiply and sum, but
/// the remaining matrix products are GEMMs, which MLX may run at TF32
/// precision on Neural-Accelerator GPUs while `MLX_ENABLE_TF32` is enabled
/// (its default).
enum GLM5NextLinearChunkedRecurrence {
    static let maximumChunkSize = 64
    static let maximumExponent: Float = 80

    /// Every log decay lies in (gateLowerBound, 0); the softplus gate has no
    /// lower bound and is never admitted.
    static func supports(chunkSize: Int, gateLowerBound: Float?) -> Bool {
        guard (1 ... maximumChunkSize).contains(chunkSize), let gateLowerBound,
            gateLowerBound.isFinite, gateLowerBound < 0
        else { return false }
        return -gateLowerBound * Float(chunkSize / 2) <= maximumExponent
    }

    /// q, k, logDecay [B,T,H,K]; v [B,T,H,V]; beta [B,T,H]; state [B,H,K,V];
    /// all FP32. Returns FP32 values [B,T,H,V] and the next state. A token with
    /// zero log decay and zero beta leaves the state unchanged, which is also
    /// how the tail of the final chunk is padded.
    static func apply(
        query: MLXArray, key: MLXArray, value: MLXArray,
        logDecay: MLXArray, beta: MLXArray, state: MLXArray, chunkSize: Int
    ) -> (values: MLXArray, state: MLXArray) {
        let batch = query.dim(0)
        let length = query.dim(1)
        let heads = query.dim(2)
        let valueDimension = value.dim(3)
        let chunks = (length + chunkSize - 1) / chunkSize
        let padding = chunks * chunkSize - length
        // [B,T,H,D] -> [B,H,N,C,D]
        func blocked(_ x: MLXArray) -> MLXArray {
            let width = x.dim(3)
            let padded =
                padding == 0
                ? x
                : concatenated(
                    [x, MLXArray.zeros([batch, padding, heads, width], dtype: .float32)], axis: 1)
            return padded.reshaped(batch, chunks, chunkSize, heads, width)
                .transposed(0, 3, 1, 2, 4)
        }
        let q = blocked(query)
        let k = blocked(key)
        let v = blocked(value)
        let b = blocked(beta.expandedDimensions(axis: -1))
        let cumulative = cumsum(blocked(logDecay), axis: 3)
        let pivot = cumulative[0..., 0..., 0..., (chunkSize / 2) ..< (chunkSize / 2 + 1), 0...]
        let rising = exp(cumulative - pivot)
        let falling = k * exp(pivot - cumulative)
        let positions = MLXArray((0 ..< chunkSize).map { Int32($0) })
        let rows = positions.reshaped(chunkSize, 1)
        let columns = positions.reshaped(1, chunkSize)
        // Selected, not multiplied: unselected upper entries may be infinite.
        let strict =
            MLX.where(
                rows .> columns, matmul(k * rising, falling.swappedAxes(-1, -2)), 0) * b
        let inclusive = MLX.where(
            rows .>= columns, matmul(q * rising, falling.swappedAxes(-1, -2)), 0)

        // Row i of A = (I + M)⁻¹ − I is −M[i,:] + Σ_{j<i} −M[i,j]·A[j,:], with
        // rows j < i already solved. Elementwise, never a GEMM, so TF32 cannot
        // round it. Entries at or above the diagonal stay zero.
        var solved = -strict
        for row in 1 ..< chunkSize {
            let coefficients = solved[0..., 0..., 0..., row, ..<row]
            let update =
                (coefficients.expandedDimensions(axis: -1)
                * solved[0..., 0..., 0..., ..<row, 0...]).sum(axis: -2)
            solved = MLX.where(
                rows .== Int32(row),
                (solved[0..., 0..., 0..., row] + update).expandedDimensions(axis: -2), solved)
        }
        let inverse = identity(chunkSize, dtype: .float32) + solved
        let decay = exp(cumulative)
        let u = matmul(inverse, v * b)
        let w = matmul(inverse, k * decay * b)
        let decayedQuery = q * decay
        let last = cumulative[0..., 0..., 0..., (chunkSize - 1)..., 0...]
        let tail = (k * exp(last - cumulative)).swappedAxes(-1, -2)
        let chunkDecay = exp(last).swappedAxes(-1, -2)

        var current = state
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(chunks)
        for chunk in 0 ..< chunks {
            let delta = u[0..., 0..., chunk] - matmul(w[0..., 0..., chunk], current)
            outputs.append(
                matmul(decayedQuery[0..., 0..., chunk], current)
                    + matmul(inclusive[0..., 0..., chunk], delta))
            current =
                chunkDecay[0..., 0..., chunk] * current + matmul(tail[0..., 0..., chunk], delta)
        }
        let values = stacked(outputs, axis: 2)
            .transposed(0, 2, 3, 1, 4)
            .reshaped(batch, chunks * chunkSize, heads, valueDimension)
        return (padding == 0 ? values : values[0..., ..<length], current)
    }
}
