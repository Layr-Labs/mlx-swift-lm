import MLX

/// Preserves ModelOpt 0.37 NVFP4 bytes and both scales without requantization.
/// Producer reference: NVIDIA/Model-Optimizer df0882a63c39a001a1be384b680166a41259045c.
/// This tensor adapter is not a model loader or an activation-quantization policy.
public struct ModelOptNVFP4Weights {
    public enum ValidationError: Error {
        case invalidPackedLayout, invalidScaleLayout, invalidGlobalScale
        case invalidInputShape, invalidInputDType, invalidInputScale
    }

    public let packedWeights: MLXArray
    public let blockScales: MLXArray
    public let tensorScale: Float
    public let inputDimensions: Int
    public let outputDimensions: Int

    /// The caller must establish safetensors provenance: weights are U8 and
    /// scales are F8_E4M3, which MLX loads as their raw UInt8 bit patterns.
    /// Accepts ordinary row-major ModelOpt layout, not flattened CUTLASS scales.
    public init(packedBytes: MLXArray, e4m3ScaleBytes: MLXArray, tensorScale: Float) throws {
        guard packedBytes.dtype == .uint8, packedBytes.ndim == 2,
            packedBytes.dim(0) > 0, packedBytes.dim(1) > 0,
            packedBytes.dim(1) % 8 == 0
        else { throw ValidationError.invalidPackedLayout }
        guard e4m3ScaleBytes.dtype == .uint8,
            e4m3ScaleBytes.shape == [packedBytes.dim(0), packedBytes.dim(1) / 8]
        else { throw ValidationError.invalidScaleLayout }
        guard tensorScale.isFinite, tensorScale > 0,
            tensorScale <= Float.greatestFiniteMagnitude / (448 * 6)
        else { throw ValidationError.invalidGlobalScale }
        self.packedWeights = packedBytes.view(dtype: .uint32)
        self.blockScales = e4m3ScaleBytes
        self.tensorScale = tensorScale
        self.inputDimensions = packedBytes.dim(1) * 2
        self.outputDimensions = packedBytes.dim(0)
    }

    /// Candidate calibrated per-matrix operation; not a gathered expert route.
    /// `inputScale` is ModelOpt's exported direct multiplier, not MLX's amax.
    /// Shape and metadata validation is synchronous; MLX execution is lazy.
    /// Numerical parity (including scale overflow) must be qualified separately.
    public func calibratedMatmul(_ input: MLXArray, inputScale: Float) throws -> MLXArray {
        guard input.ndim == 2, input.dim(0) > 0, input.dim(1) == inputDimensions else {
            throw ValidationError.invalidInputShape
        }
        guard [.float16, .bfloat16, .float32].contains(input.dtype) else {
            throw ValidationError.invalidInputDType
        }
        guard inputScale.isFinite, inputScale > 0,
            inputScale <= Float.greatestFiniteMagnitude / (448 * 6),
            (1 / inputScale).isFinite,
            ((448 * 6) / (inputScale * (448 * 6))).isFinite else {
            throw ValidationError.invalidInputScale
        }
        let result = MLX.quantizedQuantizedMM(
            input, packedWeights, scales: blockScales,
            groupSize: 16, bits: 4, mode: .nvfp4,
            globalScaleX: MLXArray(inputScale * (448 * 6)),
            globalScaleW: MLXArray(tensorScale * (448 * 6)))
        // ModelOpt's E4M3FN cast returns NaN beyond the 464 rounding boundary;
        // native Metal saturates instead. One NaN block contaminates every dot
        // product in its row, including zero weights. Restore this reference
        // behavior without a host synchronization or dense weight expansion.
        let promoted = input.asType(.float32)
        let rawScale = abs(promoted).max(axis: -1, keepDims: true) / (6 * inputScale)
        let invalid = logicalOr(rawScale .> 464,
            isNaN(promoted).any(axis: -1, keepDims: true))
        return which(invalid, MLXArray(Float.nan).asType(result.dtype), result)
    }

    /// Bounded reference/development operation. Allocates dense output; do not
    /// call across a full checkpoint as an implicit conversion/loading strategy.
    public func dequantized(dtype: DType = .bfloat16) -> MLXArray {
        // ModelOpt exports a direct multiplier amax/(448*6). MLX takes amax.
        MLX.dequantized(
            packedWeights, scales: blockScales, biases: nil,
            groupSize: 16, bits: 4, mode: .nvfp4,
            globalScale: MLXArray(tensorScale * (448 * 6)), dtype: dtype)
    }
}
