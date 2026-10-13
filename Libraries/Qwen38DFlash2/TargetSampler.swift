// Copyright 2026 Youssof Altoukhi
// SPDX-License-Identifier: Apache-2.0
// Swift port of the Qwen 3.8 target sampler at the revision in NOTICE.

import MLX

/// The target posterior sampler used by the measured Qwen 3.8 DFlash2 lane.
/// Draft proposals remain greedy; only authoritative target rows use this
/// fixed temperature/top-p/top-k distribution.
public final class Qwen38TargetSampler: @unchecked Sendable {
    public static let temperature: Float = 1
    public static let topP: Float = 0.95
    public static let topK = 20
    public static let seed: UInt64 = 42

    private let randomState: MLXRandom.RandomState

    public init(seed: UInt64 = Qwen38TargetSampler.seed) {
        randomState = MLXRandom.RandomState(seed: seed)
    }

    public func sample(logits: MLXArray) -> MLXArray {
        let rows = logits.asType(.float32) / Self.temperature
        let count = min(Self.topK, rows.dim(-1))
        var (indices, values) = qwen38OrderedTopKSupport(rows, count: count)

        let total = rows.logSumExp(axis: -1, keepDims: true)
        let probabilities = exp(values - total)
        let higherMass = probabilities.cumsum(axis: -1) - probabilities
        let first = MLXArray(Int32(0) ..< Int32(count)) .== 0
        let keep = (higherMass .< Self.topP) | first
        values = MLX.where(keep, values, MLXArray(-Float.infinity))

        // Spell out the source PR's MLX 0.32.0 categorical implementation.
        // MLX 0.32.2 otherwise selects a new inverse-CDF path for width one,
        // changing the seeded benchmark stream only at that runtime shape.
        let offsets = (values + MLXRandom.gumbel(values.shape, key: randomState))
            .argMax(axis: -1)
        indices = takeAlong(indices, offsets[.ellipsis, .newAxis], axis: -1)
        return indices[.ellipsis, 0].asType(.int32)
    }
}

func qwen38DeterministicTopKSupport(
    _ logits: MLXArray,
    count: Int
) -> (indices: MLXArray, values: MLXArray) {
    precondition(count > 0 && count <= logits.dim(-1))
    let provisional = argPartition(-logits, kth: count - 1, axis: -1)[
        .ellipsis, ..<count]
    let provisionalValues = takeAlong(logits, provisional, axis: -1)
    let cutoff = provisionalValues.min(axis: -1, keepDims: true)
    let higher = logits .> cutoff
    let tied = logits .== cutoff
    let higherCount = higher.asType(.int32).sum(axis: -1, keepDims: true)
    let tiedRank = tied.asType(.int32).cumsum(axis: -1)
    let chosen = higher | (tied & (tiedRank .<= (count - higherCount)))
    let selected = MLX.where(chosen, logits, MLXArray(-Float.infinity))
    let indices = argPartition(-selected, kth: count - 1, axis: -1)[
        .ellipsis, ..<count]
    return (indices, takeAlong(logits, indices, axis: -1))
}

func qwen38OrderedTopKSupport(
    _ logits: MLXArray,
    count: Int
) -> (indices: MLXArray, values: MLXArray) {
    let (indices, values) = qwen38DeterministicTopKSupport(logits, count: count)
    let candidateValues = values[.ellipsis, 0..., .newAxis]
    let otherValues = values[.ellipsis, .newAxis, 0...]
    let candidateIDs = indices[.ellipsis, 0..., .newAxis]
    let otherIDs = indices[.ellipsis, .newAxis, 0...]
    let rank =
        ((otherValues .> candidateValues)
        | ((otherValues .== candidateValues) & (otherIDs .< candidateIDs)))
        .asType(.int32)
        .sum(axis: -1)
    let order = argSort(rank, axis: -1)
    return (
        takeAlong(indices, order, axis: -1),
        takeAlong(values, order, axis: -1)
    )
}
