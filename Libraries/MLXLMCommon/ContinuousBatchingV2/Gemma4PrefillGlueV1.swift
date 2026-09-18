// Copyright © 2026 Eigen Labs.
// Adapted from David Tai's SDK port and final Gemma MLXFast challenge 27c821c4.
import Cmlx
import MLX
import MLXFast

/// Scheduled-prefill-only normalization/residual fusion. Every tensor is an
/// explicit kernel input; no cached model parameters or request-owned buffers.
/// Source compilation is not shader/numerical/performance qualification.
public enum Gemma4PrefillGlueV1 {
    public typealias Context = Gemma4PrefillGluePolicy.Context
    private static let threads = 704

    private struct Kernels: Sendable {
        let normResidual: MLXFast.MLXFastKernel
        let preNorm: MLXFast.MLXFastKernel
        let dualPreNorm: MLXFast.MLXFastKernel
        let branchTail: MLXFast.MLXFastKernel
        let branchTailChained: MLXFast.MLXFastKernel
        let attentionBranchPrefix: MLXFast.MLXFastKernel
        let preNormScatter: MLXFast.MLXFastKernel
        let expertTailChained: MLXFast.MLXFastKernel

        init(vectorized: Bool) {
            let suffix = vectorized ? "vec4" : "scalar"
            let header = Gemma4PrefillGlueSources.header(vectorized: vectorized)
            func make(_ name: String, _ inputs: [String], _ outputs: [String], _ source: String)
                -> MLXFast.MLXFastKernel {
                MLXFast.metalKernel(name: "db_gemma4_prefill_\(name)_v1_\(suffix)",
                    inputNames: inputs, outputNames: outputs, source: source,
                    header: header, ensureRowContiguous: true)
            }
            normResidual = make("norm_residual", ["x", "w", "res"], ["out"],
                                Gemma4PrefillGlueSources.normResidual)
            preNorm = make("prenorm", ["x", "w"], ["out"], Gemma4PrefillGlueSources.preNorm)
            dualPreNorm = make("dual_prenorm", ["x", "w1", "w2"], ["out1", "out2"],
                               Gemma4PrefillGlueSources.dualPreNorm)
            branchTail = make("branch_tail", ["h1", "h2", "w1", "w2", "w3", "res2"], ["out"],
                              Gemma4PrefillGlueSources.branchTail)
            branchTailChained = make("branch_tail_chain",
                ["h1", "h2", "w1", "w2", "w3", "res2", "s", "wn"], ["out", "normed"],
                Gemma4PrefillGlueSources.branchTailChained)
            attentionBranchPrefix = make("attention_branch_prefix", ["x", "w", "res", "wd", "wr"],
                ["out", "dense", "router"], Gemma4PrefillGlueSources.attentionBranchPrefix)
            preNormScatter = make("prenorm_scatter_tgcache", ["x", "w", "inverse"], ["out"],
                Gemma4PrefillGlueSources.preNormScatter)
            expertTailChained = make("expert_tail_chain",
                ["sorted", "inverse_order", "route_weights", "h1", "w1", "w2", "w3", "res2", "s", "wn"],
                ["out", "normed"], vectorized ? Gemma4PrefillGlueSources.expertTailChainedVector
                    : Gemma4PrefillGlueSources.expertTailChainedScalar)
        }
    }

    // Immutable kernel definitions, initialized only after admitted invocation.
    // These contain no arrays, model identity, request state or mutable scratch.
    private static let scalar = Kernels(vectorized: false)
    private static let vector = Kernels(vectorized: true)

    static func gpuStream(_ stream: StreamOrDevice) -> Bool {
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        var device = mlx_device_new()
        defer { mlx_device_free(device) }
        var type = MLX_CPU
        return mlx_stream_get_device(&device, stream.ctx) == 0
            && mlx_device_get_type(&type, device) == 0 && type == MLX_GPU
        #else
        return false
        #endif
    }

    private static func aligned(_ array: MLXArray) -> Bool {
        var available = false, rowContiguous = false, unique = false
        var allocated: UInt = 0, offset: UInt = 0, elements: UInt = 0
        guard mlx_array_get_buffer_info(&available, &allocated, &offset, &elements,
            &rowContiguous, &unique, array.ctx) == 0 else { return false }
        return Gemma4PrefillGluePolicy.permitsVectorLoad(available: available,
            rowContiguous: rowContiguous, byteOffset: offset, allocatedBytes: allocated)
    }

    static func usesVectorLoads(_ context: Context, activations: [MLXArray]) -> Bool {
        context.vectorized && activations.allSatisfy(aligned)
    }

    static func alignedActivations(_ activations: [MLXArray]) -> Bool {
        activations.allSatisfy(aligned)
    }

    private static func selected(_ context: Context, activations: [MLXArray]) -> Kernels {
        // Never force evaluation to establish alignment. Lazy/unaligned inputs
        // retain scalar accessors with the same fused arithmetic.
        usesVectorLoads(context, activations: activations) ? vector : scalar
    }

    private static func rows(_ x: MLXArray, weight: MLXArray, eps: Float,
                             context: Context, stream: StreamOrDevice) -> Int? {
        guard let count = context.rows(shape: x.shape, inputBF16: x.dtype == .bfloat16,
            weightShape: weight.shape, weightBF16: weight.dtype == .bfloat16, eps: eps),
            gpuStream(stream) else { return nil }
        return count
    }

    public struct AttentionBranchPrefix {
        public let out: MLXArray
        public let denseNorm: MLXArray
        public let routerNorm: MLXArray
    }

    public static func attentionBranchPrefix(attn x: MLXArray, residual: MLXArray,
        wPostAttn: MLXArray, wDense: MLXArray, wRouter: MLXArray, eps: Float,
        context: Context, stream: StreamOrDevice = .default) -> AttentionBranchPrefix? {
        guard context.branchPrefix,
            let count = rows(x, weight: wPostAttn, eps: eps, context: context, stream: stream),
            residual.shape == x.shape, residual.dtype == x.dtype,
            wDense.shape == wPostAttn.shape, wDense.dtype == wPostAttn.dtype,
            wRouter.shape == wPostAttn.shape, wRouter.dtype == wPostAttn.dtype else { return nil }
        let output = selected(context, activations: [x, residual]).attentionBranchPrefix(
            [x, wPostAttn, residual, wDense, wRouter], template: [("T", x.dtype)],
            grid: (threads, count, 1), threadGroup: (threads, 1, 1),
            outputShapes: [x.shape, x.shape, x.shape], outputDTypes: [x.dtype, x.dtype, x.dtype],
            stream: stream)
        return AttentionBranchPrefix(out: output[0], denseNorm: output[1], routerNorm: output[2])
    }

    /// Internal producer-bound path: only the sort owner can supply the inverse
    /// permutation. Every output row has exactly one writer, even for tied IDs.
    static func preNormScatter(x: MLXArray, weight: MLXArray, order: Gemma4PrefillExpertOrder,
        eps: Float, context: Context, stream: StreamOrDevice = .default) -> MLXArray? {
        guard context.scatter,
            let count = rows(x, weight: weight, eps: eps, context: context, stream: stream),
            order.rows == count,
            let assignments = context.scatterAssignments(rows: count,
                indexShape: [count, 8], indicesUInt32: order.inverseOrder.dtype == .uint32),
            order.inverseOrder.shape == [assignments], order.sortedIndices.shape == [assignments]
        else { return nil }
        return selected(context, activations: [x]).preNormScatter(
            [x, weight, order.inverseOrder], template: [("T", x.dtype)],
            grid: (threads, count, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[assignments, 1, 2816]], outputDTypes: [x.dtype], stream: stream)[0]
    }

    public static func normResidual(x: MLXArray, weight: MLXArray, residual: MLXArray,
        eps: Float, context: Context, stream: StreamOrDevice = .default) -> MLXArray? {
        guard let count = rows(x, weight: weight, eps: eps, context: context, stream: stream),
            residual.shape == x.shape, residual.dtype == x.dtype else { return nil }
        return selected(context, activations: [x, residual]).normResidual(
            [x, weight, residual], template: [("T", x.dtype)], grid: (threads, count, 1),
            threadGroup: (threads, 1, 1), outputShapes: [x.shape], outputDTypes: [x.dtype],
            stream: stream)[0]
    }

    public static func preNorm(x: MLXArray, weight: MLXArray, eps: Float,
        context: Context, stream: StreamOrDevice = .default) -> MLXArray? {
        guard let count = rows(x, weight: weight, eps: eps, context: context, stream: stream) else { return nil }
        return selected(context, activations: [x]).preNorm([x, weight], template: [("T", x.dtype)],
            grid: (threads, count, 1), threadGroup: (threads, 1, 1),
            outputShapes: [x.shape], outputDTypes: [x.dtype], stream: stream)[0]
    }

    public static func dualPreNorm(x: MLXArray, w1: MLXArray, w2: MLXArray,
        eps: Float, context: Context, stream: StreamOrDevice = .default) -> (MLXArray, MLXArray)? {
        guard let count = rows(x, weight: w1, eps: eps, context: context, stream: stream),
            w2.shape == w1.shape, w2.dtype == w1.dtype else { return nil }
        let values = selected(context, activations: [x]).dualPreNorm(
            [x, w1, w2], template: [("T", x.dtype)], grid: (threads, count, 1),
            threadGroup: (threads, 1, 1), outputShapes: [x.shape, x.shape],
            outputDTypes: [x.dtype, x.dtype], stream: stream)
        return (values[0], values[1])
    }

    public static func branchTail(h1: MLXArray, h2: MLXArray, w1: MLXArray, w2: MLXArray,
        w3: MLXArray, residual2: MLXArray, eps: Float, context: Context,
        stream: StreamOrDevice = .default) -> MLXArray? {
        guard let count = rows(h1, weight: w1, eps: eps, context: context, stream: stream),
            h2.shape == h1.shape, h2.dtype == h1.dtype,
            residual2.shape == h1.shape, residual2.dtype == h1.dtype,
            w2.shape == w1.shape, w2.dtype == w1.dtype,
            w3.shape == w1.shape, w3.dtype == w1.dtype else { return nil }
        return selected(context, activations: [h1, h2, residual2]).branchTail(
            [h1, h2, w1, w2, w3, residual2], template: [("T", h1.dtype)],
            grid: (threads, count, 1), threadGroup: (threads, 1, 1),
            outputShapes: [h1.shape], outputDTypes: [h1.dtype], stream: stream)[0]
    }

    public static func branchTailChained(h1: MLXArray, h2: MLXArray, w1: MLXArray, w2: MLXArray,
        w3: MLXArray, residual2: MLXArray, layerScalar: MLXArray, nextInputNormWeight: MLXArray,
        eps: Float, context: Context, stream: StreamOrDevice = .default)
        -> (out: MLXArray, normedNext: MLXArray)? {
        guard context.chained,
            let count = rows(h1, weight: w1, eps: eps, context: context, stream: stream),
            h2.shape == h1.shape, h2.dtype == h1.dtype,
            residual2.shape == h1.shape, residual2.dtype == h1.dtype,
            w2.shape == w1.shape, w2.dtype == w1.dtype,
            w3.shape == w1.shape, w3.dtype == w1.dtype,
            layerScalar.size == 1, layerScalar.dtype == h1.dtype,
            nextInputNormWeight.shape == w1.shape, nextInputNormWeight.dtype == w1.dtype
        else { return nil }
        // A rank-zero custom-kernel input is scalar-bound, while s[0] in
        // the retained body requires a buffer. Reshape only; no scalar cast.
        let scalarBuffer = layerScalar.reshaped([1], stream: stream)
        let values = selected(context, activations: [h1, h2, residual2]).branchTailChained(
            [h1, h2, w1, w2, w3, residual2, scalarBuffer, nextInputNormWeight],
            template: [("T", h1.dtype)], grid: (threads, count, 1), threadGroup: (threads, 1, 1),
            outputShapes: [h1.shape, h1.shape], outputDTypes: [h1.dtype, h1.dtype], stream: stream)
        return (values[0], values[1])
    }

    /// Consume a producer-owned pending reduction only after every tail gate
    /// passes. A rejected attempt leaves it available for the original reducer.
    public static func branchTailChainedUnsort(h1: MLXArray, expert: Gemma4PrefillExpertProjection,
        w1: MLXArray, w2: MLXArray, w3: MLXArray, residual2: MLXArray,
        layerScalar: MLXArray, nextInputNormWeight: MLXArray, eps: Float,
        context: Context, stream: StreamOrDevice = .default)
        -> (out: MLXArray, normedNext: MLXArray)? {
        guard context.expertTail, context.chained, expert.stream == stream,
            let count = rows(h1, weight: w1, eps: eps, context: context, stream: stream),
            let pending = expert.pendingForFusion,
            pending.rows == count, pending.sorted.shape == [count * 8, 2816],
            pending.inverse.shape == [count * 8], pending.inverse.dtype == .uint32,
            pending.sorted.dtype == h1.dtype, pending.weights.shape == [count, 8],
            pending.weights.dtype == h1.dtype,
            residual2.shape == h1.shape, residual2.dtype == h1.dtype,
            w2.shape == w1.shape, w2.dtype == w1.dtype,
            w3.shape == w1.shape, w3.dtype == w1.dtype,
            layerScalar.size == 1, layerScalar.dtype == h1.dtype,
            nextInputNormWeight.shape == w1.shape, nextInputNormWeight.dtype == w1.dtype
        else { return nil }
        let scalarBuffer = layerScalar.reshaped([1], stream: stream)
        let values = selected(context, activations: [h1, residual2]).expertTailChained(
            [pending.sorted, pending.inverse, pending.weights, h1,
             w1, w2, w3, residual2, scalarBuffer, nextInputNormWeight],
            template: [("T", h1.dtype)], grid: (threads, count, 1), threadGroup: (threads, 1, 1),
            outputShapes: [h1.shape, h1.shape], outputDTypes: [h1.dtype, h1.dtype], stream: stream)
        expert.consumeForFusion()
        return (values[0], values[1])
    }
}
