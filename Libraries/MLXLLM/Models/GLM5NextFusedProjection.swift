// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import MLX
import MLXNN

/// Bias-free projections that read the same input, stored once as their row
/// concatenation and run as one matmul, for
/// `GLM5NextExecutionPolicy.Projections.fusedInputs`.
///
/// Quantization groups run along the input axis, so concatenating output rows
/// keeps every packed row, scale and offset unchanged. The members returned by
/// `make` are zero-copy row views of the one buffer, so the checkpoint's module
/// tree and resident bytes are unchanged. Matmul kernels choose their
/// reduction split by output width, so results match to rounding, not bitwise.
struct GLM5NextFusedProjection {
    let linear: Linear
    /// Output width of each member, in order.
    let widths: [Int]

    /// nil when the members cannot share one buffer (mixed storage, packing,
    /// dtype, input width, a bias, or NVFP4). An artifact materializer retains
    /// the group before admission/evaluation; nil preserves checked local eval.
    static func make(
        _ members: [Linear], materialize: (([MLXArray]) throws -> Void)? = nil
    ) throws -> (fused: Self, members: [Linear])? {
        // NVFP4 members carry a per-matrix output scale a shared buffer would drop.
        guard members.count > 1, members.allSatisfy({ $0.bias == nil }),
            !members.contains(where: { $0 is GLM5NextNVFP4Linear })
        else { return nil }
        let inputWidth = members[0].shape.1
        guard members.allSatisfy({ $0.shape.1 == inputWidth }) else { return nil }
        let widths = members.map(\.shape.0)
        let fused: Linear
        if let packed = members as? [QuantizedLinear] {
            let first = packed[0]
            guard
                packed.allSatisfy({
                    $0.groupSize == first.groupSize && $0.bits == first.bits
                        && $0.mode == first.mode && $0.scales.dtype == first.scales.dtype
                        && $0.weight.dtype == first.weight.dtype
                        && ($0.biases == nil) == (first.biases == nil)
                })
            else { return nil }
            let biases = first.biases.map { _ in concatenated(packed.map { $0.biases! }, axis: 0) }
            fused = QuantizedLinear(
                weight: concatenated(packed.map(\.weight), axis: 0),
                scales: concatenated(packed.map(\.scales), axis: 0), biases: biases,
                groupSize: first.groupSize, bits: first.bits, mode: first.mode)
        } else {
            guard !members.contains(where: { $0 is QuantizedLinear }),
                members.allSatisfy({ $0.weight.dtype == members[0].weight.dtype })
            else { return nil }
            fused = Linear(weight: concatenated(members.map(\.weight), axis: 0))
        }
        var views: [Linear] = []
        var start = 0
        for width in widths {
            let rows = start ..< start + width
            if let packed = fused as? QuantizedLinear {
                views.append(
                    QuantizedLinear(
                        weight: packed.weight[rows], scales: packed.scales[rows],
                        biases: packed.biases?[rows], groupSize: packed.groupSize,
                        bits: packed.bits, mode: packed.mode))
            } else {
                views.append(Linear(weight: fused.weight[rows]))
            }
            start += width
        }
        // Materialize the buffer and its views now so the separately loaded
        // members are released and the views share the fused storage.
        let arrays =
            (fused.parameters().flattened() + views.flatMap { $0.parameters().flattened() })
            .map(\.1)
        try MLX.withError { error in
            if let materialize { try materialize(arrays) } else { eval(arrays) }
            try error.check()
        }
        return (Self(linear: fused, widths: widths), views)
    }

    /// First output column of each member.
    var columns: [Int] {
        var starts: [Int] = []
        var total = 0
        for width in widths {
            starts.append(total)
            total += width
        }
        return starts
    }

    /// One matmul, split back into the members' outputs.
    func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        split(linear(x), indices: Array(columns.dropFirst()), axis: -1)
    }

    /// The unsplit `[..., ΣN]` output (members at `columns`) and its member
    /// views, with prefill-sized steps routed by `prefill`.
    func project(_ x: MLXArray, prefill: GLM5NextExecutionPolicy.PrefillProjections) throws
        -> (output: MLXArray, members: [MLXArray])
    {
        let output = try prefill.project(linear, x)
        return (output, split(output, indices: Array(columns.dropFirst()), axis: -1))
    }
}
