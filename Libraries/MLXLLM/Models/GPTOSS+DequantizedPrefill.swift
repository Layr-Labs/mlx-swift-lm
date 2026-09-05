import Foundation
import MLX
import MLXLMCommon

/// Experimental per-call expansion, never an expanded-weight cache. The
/// caller enables it only for large sorted GPTOSS-20B prompt rectangles.
enum GPTOSSDequantizedPrefill {
    static let enabled =
        ProcessInfo.processInfo.environment["DARKBLOOM_GPTOSS_PREFILL_DEQUANT"] == "1"

    static func eligible(inputDims: Int, hiddenDims: Int, experts: Int,
                         assignments: Int, sequenceLength: Int, sorted: Bool) -> Bool {
        inputDims == 2880 && hiddenDims == 2880 && experts == 32
            && assignments >= 2048 && sequenceLength > 1 && sorted
    }

    /// Returns nil for non-MXFP4 or non-index-aligned geometry. This helper
    /// remains callable on tiny test modules; the production caller applies
    /// the exact-model and minimum-work gates above before reaching it.
    static func project(_ module: SwitchLinear, _ x: MLXArray, _ indices: MLXArray,
                        sortedIndices: Bool) -> MLXArray? {
        guard let quantized = module as? QuantizedSwitchLinear,
              quantized.mode == .mxfp4, quantized.groupSize == 32, quantized.bits == 4,
              sortedIndices, x.ndim >= 2, x.dim(-2) == 1,
              x.dtype == .float32 || x.dtype == .bfloat16,
              x.size == indices.size * x.dim(-1) else { return nil }
        let parameters = Dictionary(uniqueKeysWithValues: module.parameters().flattened())
        guard let weight = parameters["weight"], let scales = parameters["scales"],
              weight.ndim == 3, scales.ndim == 3,
              weight.dtype == .uint32, scales.dtype == .uint8,
              weight.dim(-1) * 8 == x.dim(-1) else { return nil }
        // Match MXFP4 gather-QMM's output/weight-loader type exactly. In
        // particular, FP32 activations never use a BF16 expansion shortcut.
        let expanded = dequantized(weight, scales: scales, biases: nil,
                                   groupSize: 32, bits: 4, mode: .mxfp4, dtype: x.dtype)
        var output = gatherMM(x, expanded.swappedAxes(-1, -2),
                              rhsIndices: indices, sortedIndices: true)
        if let bias = parameters["bias"] {
            output = output + MLX.expandedDimensions(bias[indices], axis: -2)
        }
        return output
    }
}
