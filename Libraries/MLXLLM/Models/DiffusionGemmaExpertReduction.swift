// Native DiffusionGemma expert-output reduction; existing SDK kernel and order.
import Foundation
import MLX
import MLXLMCommon

enum DiffusionGemmaExpertReduction {
    /// Inference default. Explicit false restores the original graph.
    /// Latched before inference; never mutate the environment during a request.
    static let enabled = enabled(
        from: ProcessInfo.processInfo.environment[
            "DARKBLOOM_DIFFUSION_EXPERT_UNSORT"])

    static func enabled(from value: String?) -> Bool {
        guard let value else { return true }
        return ["1", "true", "yes", "on"].contains(value.lowercased())
    }

    /// The reused shader computes flattened addresses with uint arithmetic.
    /// Keep larger logical tensors on the original 64-bit-aware graph without
    /// imposing a smaller model context or constructing a large test buffer.
    static func shaderIndexFits(assignmentCount: Int, hidden: Int) -> Bool {
        guard assignmentCount > 0, hidden > 0 else { return false }
        let (elements, overflow) = assignmentCount.multipliedReportingOverflow(by: hidden)
        return !overflow && elements <= Int(UInt32.max)
    }

    static func eligible(
        _ outputs: MLXArray, inverse: MLXArray?, weights: MLXArray, enabled: Bool
    ) -> Bool {
        guard enabled, Device.defaultDevice().deviceType == .gpu,
            StreamOrDevice.default.stream == Stream.gpu,
            outputs.ndim == 3, outputs.dim(1) == 1, outputs.dim(2) == 2816,
            shaderIndexFits(assignmentCount: outputs.dim(0), hidden: outputs.dim(2)),
            outputs.dtype == .bfloat16,
            weights.ndim == 2, weights.dim(1) == 8, weights.size >= 64,
            weights.dtype == .bfloat16, outputs.dim(0) == weights.size,
            let inverse, inverse.ndim == 1, inverse.dtype == .uint32,
            inverse.size == weights.size
        else { return false }
        return true
    }

    static func reduce(
        _ outputs: MLXArray, inverse: MLXArray?, indicesShape: [Int], weights: MLXArray,
        enabled: Bool
    ) -> MLXArray {
        if eligible(outputs, inverse: inverse, weights: weights, enabled: enabled),
            indicesShape == weights.shape, let inverse
        {
            return weightedExpertUnsort(
                sortedOutputs: outputs.squeezed(axis: -2), inverseOrder: inverse,
                weights: weights)
        }
        // Preserve every legacy operation, including BF16 multiply rounding and
        // reduction order, for training, CPU/custom streams and other geometries.
        var restored = outputs
        if let inverse {
            restored = scatterUnsort(x: restored, invOrder: inverse, shape: indicesShape)
        }
        restored = restored.squeezed(axis: -2)
        return (restored * expandedDimensions(weights, axis: -1)).sum(axis: -2)
    }
}
