import MLX
import MLXNN

/// Explicit registered-stage prompt policy. Each affine projection keeps the
/// singleton token reduction even when the prompt is evaluated in chunks.
/// Ordinary SDK callers, independent decode lanes and normalization are unchanged.
public enum Gemma4PrefillProjectionScope {
    public static func withPrefillFrame<T>(
        tokens: MLXArray, enabled: Bool,
        _ body: () throws -> T
    ) rethrows -> T {
        let prefill =
            enabled && tokens.ndim == 2
            && tokens.dim(0) == 1 && tokens.dim(1) > 1
        return try Gemma4PrefillProjection.$active.withValue(prefill, operation: body)
    }
}

enum Gemma4PrefillProjection {
    @TaskLocal static var active = false

    static func applies(_ input: MLXArray) -> Bool {
        active && input.ndim == 3 && input.dim(0) == 1 && input.dim(1) > 1
    }

    static func appliesExperts(
        _ input: MLXArray, indices: MLXArray,
        weights: MLXArray, isExpertPrefill: Bool
    ) -> Bool {
        // Stage admission already pins the model and packed banks. This gate
        // keeps verification, other dtypes and incompatible routing shapes on
        // their original path, including explicitly scoped SDK callers.
        applies(input) && isExpertPrefill && input.dtype == .bfloat16
            && indices.ndim == 3 && indices.dim(0) == 1
            && indices.dim(1) == input.dim(1) && indices.dim(2) > 0
            && indices.dtype == .uint32 && weights.shape == indices.shape
            && weights.dtype == .bfloat16
    }

    static func linear(_ linear: Linear, _ input: MLXArray) -> MLXArray {
        // Scope only the registered BF16 affine geometry. Custom module
        // overrides and other quantizers retain their actual forward method.
        guard applies(input), type(of: linear) == QuantizedLinear.self,
            let packed = linear as? QuantizedLinear,
            supports(
                packed.mode, bits: packed.bits, groupSize: packed.groupSize,
                input: input, scales: packed.scales)
        else { return linear(input) }
        var result = project(
            input, weight: packed.weight, scales: packed.scales,
            biases: packed.biases, groupSize: packed.groupSize, bits: packed.bits)
        if let bias = packed.bias { result = result + bias }
        return result
    }

    private static func supports(
        _ mode: QuantizationMode, bits: Int, groupSize: Int,
        input: MLXArray, scales: MLXArray
    ) -> Bool {
        mode == .affine && [4, 8].contains(bits) && groupSize == 64
            && input.dtype == .bfloat16 && scales.dtype == .bfloat16
    }

    private static func project(
        _ input: MLXArray, weight: MLXArray, scales: MLXArray,
        biases: MLXArray?, groupSize: Int, bits: Int
    ) -> MLXArray {
        // Views only: leading broadcast weights keep native M=1 for each token
        // instead of dispatching the sequence rectangle to a chip-specific QMM.
        let tokens = input.dim(1)
        return Gemma4BatchDecodeProjection.project(
            input.reshaped([tokens, 1, input.dim(2)]), weight: weight,
            scales: scales, biases: biases, groupSize: groupSize, bits: bits
        )
        .reshaped([1, tokens, weight.dim(0)])
    }
}
