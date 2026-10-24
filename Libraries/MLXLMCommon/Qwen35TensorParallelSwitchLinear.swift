import MLX
import MLXNN

/// A channel-sharded Qwen expert down projection. The collective receives
/// unrounded FP32 expert outputs, before the owning SwitchGLU restores route
/// order and applies router weights. Shared-expert gating stays downstream.
public final class Qwen35TensorParallelSwitchLinear: QuantizedSwitchLinear {
    private final class Reduction {
        let sum: (MLXArray) -> MLXArray
        init(_ sum: @escaping (MLXArray) -> MLXArray) { self.sum = sum }
    }

    private let reduction: Reduction

    public enum InstallationError: Error {
        case unsupportedProjection
    }

    /// Retains the already cut packed arrays. No full bank is dequantized or
    /// replicated. Only the registered affine, bias-free expert contract is
    /// accepted; adding a replicated bias before all-sum would count it twice.
    public init(
        _ source: QuantizedSwitchLinear,
        allSum: @escaping (MLXArray) -> MLXArray
    ) throws {
        guard type(of: source) == QuantizedSwitchLinear.self,
            source.mode == .affine, source.bias == nil,
            [4, 8].contains(source.bits), let offsets = source.biases,
            source.globalScales == nil
        else {
            throw InstallationError.unsupportedProjection
        }
        reduction = Reduction(allSum)
        try super.init(
            inputDims: source.shape.inputDims,
            outputDims: source.shape.outputDims,
            numExperts: source.shape.numExperts,
            weight: source.weight, scales: source.scales, biases: offsets,
            groupSize: source.groupSize, bits: source.bits)
    }

    public override func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let partial = super.callAsFunction(
            x.asType(.float32), indices,
            sortedIndices: sortedIndices)
        return reduction.sum(partial).asType(x.dtype)
    }
}
