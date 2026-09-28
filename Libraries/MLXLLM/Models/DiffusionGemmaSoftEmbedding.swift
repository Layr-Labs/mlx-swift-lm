// DiffusionGemma soft conditioning, using the packaged upstream MLX affine math.
import Cmlx
import Foundation
import MLX
import MLXLMCommon

enum DiffusionGemmaSoftEmbedding {
    // Experimental until full-model qualification. Removing the switch is rollback.
    static let enabled = enabled(
        from: ProcessInfo.processInfo.environment["DARKBLOOM_DIFFUSION_SOFT_EMBEDDING"])

    static func enabled(from value: String?) -> Bool {
        value.map { ["1", "true", "yes", "on"].contains($0.lowercased()) } ?? false
    }

    // This existing non-evaluating C API rejects compile/grad/vmap tracing and
    // retained graphs. No identities or arrays are retained by this projection.
    static func outsideTransform(_ arrays: [MLXArray]) -> Bool {
        arrays.allSatisfy { array in
            var identity: UInt = 0
            var ordinary = false
            return _mlx_array_constant_cache_identity(&identity, &ordinary, array.ctx) == 0
                && ordinary
        }
    }

    static func eligible(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode, inference: Bool, enabled: Bool
    ) -> Bool {
        guard enabled, inference, Device.defaultDevice().deviceType == .gpu,
            StreamOrDevice.default.stream == Stream.gpu,
            x.shape == [1, 256, 262144], x.dtype == .bfloat16,
            weight.shape == [262144, 704], weight.dtype == .uint32,
            scales.shape == [262144, 44], scales.dtype == .bfloat16,
            let biases, biases.shape == scales.shape, biases.dtype == .bfloat16,
            groupSize == 64, bits == 8, mode == .affine
        else { return false }
        return outsideTransform([x, weight, scales, biases])
    }

    static func accelerated(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode, inference: Bool, enabled: Bool
    ) -> MLXArray? {
        guard eligible(
            x, weight: weight, scales: scales, biases: biases, groupSize: groupSize,
            bits: bits, mode: mode, inference: inference, enabled: enabled),
            let biases, let kernel
        else { return nil }
        // Same K traversal, dequantization and FP32 accumulation as qmm_n.
        // Only BM changes from 32 to 64. The native canvas and sampler do not.
        // Keep safe contiguous preparation for arbitrary input views/offsets.
        let result = kernel(
            [weight, scales, biases, x], template: [("T", DType.bfloat16)],
            grid: (88 * 32, 4 * 2, 2), threadGroup: (32, 2, 2),
            outputShapes: [[1, 256, 2816]], outputDTypes: [.bfloat16],
            stream: .default)[0]
        DiffusionGemmaSoftEmbeddingDiagnostics.recordDispatch()
        return result
    }

    static func project(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode, inference: Bool, enabled: Bool
    ) -> MLXArray {
        accelerated(
            x, weight: weight, scales: scales, biases: biases, groupSize: groupSize,
            bits: bits, mode: mode, inference: inference, enabled: enabled)
            ?? quantizedMM(
                x, weight, scales: scales, biases: biases, transpose: false,
                groupSize: groupSize, bits: bits, mode: mode)
    }

    private static let kernel: MLXFast.MLXFastKernel? = {
        guard (try? Qwen4ExpMetalHeaders.validateResources()) != nil else { return nil }
        var quantized = Qwen4ExpMetalHeaders.quantized
        guard let start = quantized.range(
            of: "\ntemplate <typename T, int group_size, int bits, int D, bool batched>\n[[kernel]]")
                ?? quantized.range(of: "\n[[kernel]]")
        else { return nil }
        quantized = String(quantized[..<start.lowerBound])
        let header = Qwen4ExpMetalHeaders.gemm + "\n"
            + Qwen4ExpMetalHeaders.quantizedUtils + "\n" + quantized + """

            constant int diffusion_soft_K = 262144;
            constant int diffusion_soft_N = 2816;
            constant int diffusion_soft_M = 256;
            """
        return MLXFast.metalKernel(
            name: "diffusion_soft_embedding_bm64",
            inputNames: ["w", "scales", "biases", "x"], outputNames: ["y"],
            source: """
                threadgroup T Xs[64 * (32 + 16 / sizeof(T))];
                threadgroup T Ws[32 * (32 + 16 / sizeof(T))];
                qmm_n_impl<T, 64, 8, 64, 32, 32>(
                    w, scales, biases, x, y, Xs, Ws,
                    diffusion_soft_K, diffusion_soft_N, diffusion_soft_M,
                    threadgroup_position_in_grid, thread_index_in_threadgroup,
                    simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
                """,
            header: header, ensureRowContiguous: true)
    }()
}
