// Copyright © 2026 Eigen Labs.
import MLX
import MLXFast

/// Default-off per-row decode glue. No B8 fixed-stride sum tables, global
/// parameter caches or MTP controller changes. Every tensor is explicit.
public enum Gemma4DecodeGlueV1 {
    public typealias Context = Gemma4DecodeGluePolicy.Context
    private static func make(_ name: String, inputs: [String], outputs: [String], source: String) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(name: "db_gemma4_decode_\(name)_v1", inputNames: inputs,
            outputNames: outputs, source: source, ensureRowContiguous: true)
    }
    private static let norm = make("norm_residual", inputs: ["x", "res", "w"], outputs: ["out"], source: Gemma4DecodeGlueSources.normResidual)
    private static let assistantNorm = make("assistant_norm_residual", inputs: ["x", "res", "w"], outputs: ["out"], source: Gemma4DecodeGlueSources.normResidual1024)
    private static let dual = make("dual_prenorm", inputs: ["x", "w1", "w2"], outputs: ["out1", "out2"], source: Gemma4DecodeGlueSources.dualPreNorm)
    private static let tail = make("tail", inputs: ["a", "b", "res", "w1", "w2", "w3", "s"], outputs: ["out"], source: Gemma4DecodeGlueSources.tail)
    private static let pairedTail = make("tail_paired", inputs: ["a", "b", "res", "w1", "w2", "w3", "s"], outputs: ["out"], source: Gemma4DecodeGlueSources.pairedRmsTailSource(Gemma4DecodeGlueSources.tail))
    private static let chain = make("tail_chain", inputs: ["a", "b", "res", "w1", "w2", "w3", "s", "wn"], outputs: ["out", "normed"], source: Gemma4DecodeGlueSources.tailChained)
    private static let pairedChain = make("tail_chain_paired", inputs: ["a", "b", "res", "w1", "w2", "w3", "s", "wn"], outputs: ["out", "normed"], source: Gemma4DecodeGlueSources.pairedRmsTailSource(Gemma4DecodeGlueSources.tailChained))

    private static func localBroadcast(_ name: String, inputs: [String], outputs: [String], source: String) -> MLXFast.MLXFastKernel? {
        guard let transformed = Gemma4RMSBroadcastSources.transform(source) else { return nil }
        return make(name + "_local_broadcast", inputs: inputs, outputs: outputs, source: transformed)
    }
    private static let localNorm = localBroadcast("norm_residual", inputs: ["x", "res", "w"], outputs: ["out"], source: Gemma4DecodeGlueSources.normResidual)
    private static let localDual = localBroadcast("dual_prenorm", inputs: ["x", "w1", "w2"], outputs: ["out1", "out2"], source: Gemma4DecodeGlueSources.dualPreNorm)
    private static let localTail = localBroadcast("tail", inputs: ["a", "b", "res", "w1", "w2", "w3", "s"], outputs: ["out"], source: Gemma4DecodeGlueSources.tail)
    private static let localPairedTail = localBroadcast("tail_paired", inputs: ["a", "b", "res", "w1", "w2", "w3", "s"], outputs: ["out"], source: Gemma4DecodeGlueSources.pairedRmsTailSource(Gemma4DecodeGlueSources.tail))
    private static let localChain = localBroadcast("tail_chain", inputs: ["a", "b", "res", "w1", "w2", "w3", "s", "wn"], outputs: ["out", "normed"], source: Gemma4DecodeGlueSources.tailChained)
    private static let localPairedChain = localBroadcast("tail_chain_paired", inputs: ["a", "b", "res", "w1", "w2", "w3", "s", "wn"], outputs: ["out", "normed"], source: Gemma4DecodeGlueSources.pairedRmsTailSource(Gemma4DecodeGlueSources.tailChained))

    private static func rows(_ x: MLXArray, weight: MLXArray, eps: Float, context: Context, stream: StreamOrDevice) -> Int? {
        guard let rows = context.rows(shape: x.shape, inputBF16: x.dtype == .bfloat16,
            weightShape: weight.shape, weightBF16: weight.dtype == .bfloat16, eps: eps),
            Gemma4PrefillGlueV1.gpuStream(stream) else { return nil }
        return rows
    }

    public static func normResidual(x: MLXArray, residual: MLXArray, weight: MLXArray, eps: Float,
        context: Context, stream: StreamOrDevice = .default) -> MLXArray? {
        guard let count = rows(x, weight: weight, eps: eps, context: context, stream: stream),
            residual.shape == x.shape, residual.dtype == x.dtype else { return nil }
        let kernel = context.axis == 1024 ? assistantNorm : (context.localRMSBroadcast ? localNorm ?? norm : norm)
        return kernel([x, residual, weight], template: [("T", x.dtype)],
            grid: (count * (context.axis / 4), 1, 1), threadGroup: (context.axis / 4, 1, 1),
            outputShapes: [x.shape], outputDTypes: [x.dtype], stream: stream)[0]
    }

    public static func dualPreNorm(x: MLXArray, w1: MLXArray, w2: MLXArray, eps: Float,
        context: Context, stream: StreamOrDevice = .default) -> (MLXArray, MLXArray)? {
        guard context.axis == 2816,
            let count = rows(x, weight: w1, eps: eps, context: context, stream: stream),
            w2.shape == w1.shape, w2.dtype == w1.dtype else { return nil }
        let kernel = context.localRMSBroadcast ? localDual ?? dual : dual
        let output = kernel([x, w1, w2], template: [("T", x.dtype)], grid: (count * 704, 1, 1),
            threadGroup: (704, 1, 1), outputShapes: [x.shape, x.shape],
            outputDTypes: [x.dtype, x.dtype], stream: stream)
        return (output[0], output[1])
    }

    public static func branchTail(h1: MLXArray, h2: MLXArray, residual: MLXArray,
        w1: MLXArray, w2: MLXArray, w3: MLXArray, layerScalar: MLXArray, eps: Float,
        context: Context, stream: StreamOrDevice = .default) -> MLXArray? {
        guard context.axis == 2816,
            let count = rows(h1, weight: w1, eps: eps, context: context, stream: stream),
            h2.shape == h1.shape, h2.dtype == h1.dtype,
            residual.shape == h1.shape, residual.dtype == h1.dtype,
            w2.shape == w1.shape, w2.dtype == w1.dtype,
            w3.shape == w1.shape, w3.dtype == w1.dtype,
            layerScalar.size == 1, layerScalar.dtype == h1.dtype else { return nil }
        // metalKernel binds rank-zero arrays as scalar references; these
        // retained MSL bodies use s[0]. Give either scalar shape a one-item
        // buffer without changing its dtype, value, stream or arithmetic.
        let scalarBuffer = layerScalar.reshaped([1], stream: stream)
        let original = context.paired ? pairedTail : tail
        let kernel = context.localRMSBroadcast ? (context.paired ? localPairedTail : localTail) ?? original : original
        return kernel([h1, h2, residual, w1, w2, w3, scalarBuffer],
            template: [("T", h1.dtype)], grid: (count * 704, 1, 1), threadGroup: (704, 1, 1),
            outputShapes: [h1.shape], outputDTypes: [h1.dtype], stream: stream)[0]
    }

    public static func branchTailChained(h1: MLXArray, h2: MLXArray, residual: MLXArray,
        w1: MLXArray, w2: MLXArray, w3: MLXArray, layerScalar: MLXArray, nextWeight: MLXArray,
        eps: Float, context: Context, stream: StreamOrDevice = .default)
        -> (out: MLXArray, normalized: MLXArray)? {
        guard context.axis == 2816, context.chained,
            let count = rows(h1, weight: w1, eps: eps, context: context, stream: stream),
            h2.shape == h1.shape, h2.dtype == h1.dtype,
            residual.shape == h1.shape, residual.dtype == h1.dtype,
            w2.shape == w1.shape, w2.dtype == w1.dtype,
            w3.shape == w1.shape, w3.dtype == w1.dtype,
            nextWeight.shape == w1.shape, nextWeight.dtype == w1.dtype,
            layerScalar.size == 1, layerScalar.dtype == h1.dtype else { return nil }
        let scalarBuffer = layerScalar.reshaped([1], stream: stream)
        let original = context.paired ? pairedChain : chain
        let kernel = context.localRMSBroadcast ? (context.paired ? localPairedChain : localChain) ?? original : original
        let output = kernel([h1, h2, residual, w1, w2, w3, scalarBuffer, nextWeight],
            template: [("T", h1.dtype)], grid: (count * 704, 1, 1), threadGroup: (704, 1, 1),
            outputShapes: [h1.shape, h1.shape], outputDTypes: [h1.dtype, h1.dtype], stream: stream)
        return (output[0], output[1])
    }
}
