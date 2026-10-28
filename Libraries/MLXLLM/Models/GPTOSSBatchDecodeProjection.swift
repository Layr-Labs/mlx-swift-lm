import MLX
import MLXNN

/// Explicit GPT-OSS independent-history decode arithmetic. Ordinary forward,
/// prefill and CBv2 entry points do not call this helper. Stored affine8
/// weights remain shared views, while each native matmul keeps M=1.
public enum GPTOSSBatchDecodeProjection {
    public static func linear(_ layer: Linear, _ input: MLXArray) -> MLXArray {
        guard input.ndim == 3, (2 ... 8).contains(input.dim(0)), input.dim(1) == 1 else {
            return layer(input)
        }
        guard type(of: layer) == QuantizedLinear.self, let q = layer as? QuantizedLinear,
            q.mode == .affine, q.bits == 8, q.groupSize == 64,
            input.dtype == .bfloat16, q.scales.dtype == .bfloat16
        else {
            return rows(input, layer.callAsFunction)
        }
        var output = quantizedMM(
            input, q.weight.expandedDimensions(axis: 0),
            scales: q.scales.expandedDimensions(axis: 0),
            biases: q.biases.map { $0.expandedDimensions(axis: 0) }, transpose: true,
            groupSize: 64, bits: 8)
        if let bias = q.bias { output = output + bias }
        return output
    }

    static func rows(_ input: MLXArray, _ forward: (MLXArray) -> MLXArray) -> MLXArray {
        guard input.dim(0) > 1 else { return forward(input) }
        return concatenated((0 ..< input.dim(0)).map { forward(input[$0 ..< $0 + 1]) }, axis: 0)
    }
}
