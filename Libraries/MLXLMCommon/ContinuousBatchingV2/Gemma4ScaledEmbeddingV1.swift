// Copyright © 2026 Eigen Labs.
import MLX
import MLXFast
import MLXNN

/// No re-quantization or cached parameters. The original embedding's valid-token
/// domain applies (-vocab <= token < vocab); out-of-domain gather is not redefined.
public enum Gemma4ScaledEmbeddingV1 {
    private static let kernel = MLXFast.metalKernel(name: "db_gemma4_scaled_embedding_q4g64_v1",
        inputNames: ["tokens", "w", "scales", "biases", "embed_scale"], outputNames: ["out"],
        source: Gemma4ScaledEmbeddingSources.source, ensureRowContiguous: true)

    public static func apply(tokens: MLXArray, embedding: Embedding, embedScale: Float,
        hidden: Int, targetEligible: Bool, policy: Gemma4ScaledEmbeddingPolicy,
        stream: StreamOrDevice = .default) -> MLXArray? {
        guard policy.enabled, targetEligible, let quantized = embedding as? QuantizedEmbedding,
            type(of: quantized) == QuantizedEmbedding.self, quantized.mode == .affine,
            let biases = quantized.biases, quantized.weight.ndim == 2,
            let plan = policy.plan(targetEligible: targetEligible, tokenShape: tokens.shape,
                tokensInt32: tokens.dtype == .int32, vocab: quantized.weight.dim(0), hidden: hidden,
                bits: quantized.bits, groupSize: quantized.groupSize),
            quantized.weight.dtype == .uint32,
            quantized.weight.dim(1) == hidden / 8,
            quantized.scales.dtype == .bfloat16, biases.dtype == .bfloat16,
            quantized.scales.shape == [quantized.weight.dim(0), hidden / 64],
            biases.shape == quantized.scales.shape, Gemma4PrefillGlueV1.gpuStream(stream) else { return nil }
        return kernel([tokens, quantized.weight, quantized.scales, biases,
                       embedScale.asMLXArray(dtype: .bfloat16)],
            template: [("T", DType.bfloat16)], grid: (hidden / 8, plan.rows, 1), threadGroup: (32, 8, 1),
            outputShapes: [plan.outputShape], outputDTypes: [.bfloat16], stream: stream)[0]
    }
}
