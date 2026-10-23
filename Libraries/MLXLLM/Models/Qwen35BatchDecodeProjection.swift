import MLX
import MLXNN

/// A cluster decode frame may request singleton projection arithmetic for
/// independent Qwen 3.5 rows. The default, prefill and speculative paths do not
/// opt in. This scope changes no MLX dispatcher and carries no request state.
public enum Qwen35BatchDecodeScope {
    @TaskLocal public static var enabled = false
}

enum Qwen35BatchDecodeProjection {
    /// Set only by Qwen35TextModel's two ordinary CBv2 decode entry points.
    /// Other models sharing Qwen layers cannot inherit the cluster request.
    @TaskLocal static var active = false

    static func applies(_ input: MLXArray) -> Bool {
        active && input.ndim == 3 && (2 ... 8).contains(input.dim(0)) && input.dim(1) == 1
    }

    static func withModel<T>(_ tokens: MLXArray, _ body: () throws -> T) rethrows -> T {
        let decode =
            Qwen35BatchDecodeScope.enabled && tokens.ndim == 2
            && (2 ... 8).contains(tokens.dim(0)) && tokens.dim(1) == 1
        return try $active.withValue(decode, operation: body)
    }

    static func linear(_ linear: Linear, _ input: MLXArray) -> MLXArray {
        guard applies(input) else { return linear(input) }
        // Subclasses can rotate inputs, add adapters, or have another numeric
        // contract. Preserve their actual forward method instead of reading
        // base-class arrays and bypassing the override.
        guard type(of: linear) == QuantizedLinear.self,
            let packed = linear as? QuantizedLinear,
            supports(
                packed.mode, bits: packed.bits,
                groupSize: packed.groupSize, input: input, scales: packed.scales)
        else { return perRow(input, linear.callAsFunction) }
        var result = project(
            input, weight: packed.weight, scales: packed.scales,
            biases: packed.biases, groupSize: packed.groupSize, bits: packed.bits)
        if let bias = packed.bias { result = result + bias }
        return result
    }

    static func embedding(_ embedding: Embedding, _ input: MLXArray) -> MLXArray {
        guard applies(input) else { return embedding.asLinear(input) }
        guard type(of: embedding) == QuantizedEmbedding.self,
            let packed = embedding as? QuantizedEmbedding,
            supports(
                packed.mode, bits: packed.bits, groupSize: packed.groupSize,
                input: input, scales: packed.scales)
        else { return perRow(input, embedding.asLinear) }
        return project(
            input, weight: packed.weight, scales: packed.scales,
            biases: packed.biases, groupSize: packed.groupSize, bits: packed.bits)
    }

    /// Keep unqualified quantizers and dtype combinations on the original
    /// module. The supported path has native-byte synthetic evidence on M3/M5.
    static func supports(
        _ mode: QuantizationMode, bits: Int, groupSize: Int,
        input: MLXArray, scales: MLXArray
    ) -> Bool {
        guard mode == .affine, [2, 3, 4, 8].contains(bits), [64, 128].contains(groupSize) else {
            return false
        }
        switch (input.dtype, scales.dtype) {
        case (.bfloat16, .bfloat16), (.float16, .float16),
            (.float32, .float16), (.float32, .bfloat16), (.float32, .float32):
            return true
        default:
            return false
        }
    }

    private static func project(
        _ input: MLXArray, weight: MLXArray, scales: MLXArray,
        biases: MLXArray?, groupSize: Int, bits: Int
    ) -> MLXArray {
        // A leading broadcast dimension keeps native matmul M=1, B=rows.
        // 2D weights would flatten B into M and select qmv_wide/QMM. These are
        // views, not duplicated weights. Native affine promotion is unchanged.
        quantizedMM(
            input, weight.expandedDimensions(axis: 0),
            scales: scales.expandedDimensions(axis: 0),
            biases: biases.map { $0.expandedDimensions(axis: 0) },
            transpose: true, groupSize: groupSize, bits: bits)
    }

    static func perRow(_ input: MLXArray, _ forward: (MLXArray) -> MLXArray) -> MLXArray {
        concatenated((0 ..< input.dim(0)).map { forward(input[$0 ..< $0 + 1]) }, axis: 0)
    }
}
