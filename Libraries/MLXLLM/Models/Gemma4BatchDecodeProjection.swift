import MLX
import MLXNN

/// Explicit ordinary Gemma decode scope. A caller must bind independent CBv2
/// request caches; prefill, speculative verification and the default model API
/// do not opt in. The scope owns no request state and changes no MLX dispatcher.
public enum Gemma4BatchDecodeScope {
    public static func withDecodeFrame<T>(
        tokens: MLXArray, enabled: Bool,
        _ body: () throws -> T
    ) rethrows -> T {
        let decode =
            enabled && tokens.ndim == 2
            && (2 ... 8).contains(tokens.dim(0)) && tokens.dim(1) == 1
        return try Gemma4BatchDecodeProjection.$active.withValue(decode, operation: body)
    }

    /// Stage wrappers own their tied output embedding outside the text model.
    public static func projectHead(_ embedding: Embedding, _ hidden: MLXArray) -> MLXArray {
        Gemma4BatchDecodeProjection.embedding(embedding, hidden)
    }
}

enum Gemma4BatchDecodeProjection {
    @TaskLocal static var active = false

    static func applies(_ input: MLXArray) -> Bool {
        active && input.ndim == 3 && (2 ... 8).contains(input.dim(0)) && input.dim(1) == 1
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
    /// module. Registered Gemma uses affine 4/8-bit group64 and BF16 constants.
    static func supports(
        _ mode: QuantizationMode, bits: Int, groupSize: Int,
        input: MLXArray, scales: MLXArray
    ) -> Bool {
        guard mode == .affine, [4, 8].contains(bits), groupSize == 64 else {
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
