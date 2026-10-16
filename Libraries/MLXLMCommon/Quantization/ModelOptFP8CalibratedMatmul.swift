// Reference algorithm: NVIDIA Model-Optimizer, Apache-2.0,
// df0882a63c39a001a1be384b680166a41259045c,
// modelopt/torch/quantization/tensor_quant.py::_fp8_eager.
import Cmlx
import MLX

extension ModelOptFP8Weights {
    /// Bounded per-tensor W8A8 candidate against ModelOpt's eager reference.
    /// `inputScale` is the exported direct input multiplier (amax / 448).
    /// `weightDType` is the original weight dtype from checkpoint provenance;
    /// the official Super checkpoint declares BF16. It is not inferred from
    /// activation dtype. Mixed activation/weight dtypes remain unqualified.
    /// Computes only the requested output-weight rows; no bias or expert gather.
    /// Native CUDA-extension/model parity and performance are not established.
    ///
    /// The 64 MiB conservative estimate covers explicit array intermediates,
    /// not backend matmul scratch or concurrent calls. The caller owns admission
    /// and evaluation lifetime; do not queue unbounded lazy row slices.
    public func calibratedMatmul(
        _ input: MLXArray, inputScale: Float, weightDType: DType, outputRows: Range<Int>,
        activationOverflow: ModelOptActivationOverflow = .eagerNaN
    ) throws -> MLXArray {
        guard input.ndim == 2, input.dim(0) > 0, input.dim(1) == bytes.dim(1) else {
            throw ValidationError.invalidInputShape
        }
        guard [.float16, .bfloat16, .float32].contains(weightDType),
            input.dtype == weightDType
        else {
            throw ValidationError.invalidDType
        }
        guard !outputRows.isEmpty, outputRows.lowerBound >= 0,
            outputRows.upperBound <= bytes.dim(0)
        else { throw ValidationError.invalidRows }
        let amax = inputScale * 448
        let scale = 448 / amax
        let inverseScale = 1 / scale
        guard inputScale.isFinite, inputScale > 0, amax.isFinite, amax > 0,
            scale.isFinite, scale > 0, inverseScale.isFinite, inverseScale > 0
        else {
            throw ValidationError.invalidInputScale
        }
        // Bound each product before summing, avoiding overflow in allocation
        // estimates. 32 bytes/element covers conversion/mask/scaling temporaries.
        let limit = (64 << 20) / 32
        let shapes = [
            (input.dim(0), bytes.dim(1)),
            (outputRows.count, bytes.dim(1)), (input.dim(0), outputRows.count),
        ]
        var elements = 0
        for (rows, columns) in shapes {
            let (count, overflow) = rows.multipliedReportingOverflow(by: columns)
            guard !overflow, count <= limit - elements else { throw ValidationError.workspaceLimit }
            elements += count
        }
        let scaled = input.asType(.float32) * scale
        var packedHandle = mlx_array_new()
        let status = mlx_to_fp8(&packedHandle, scaled.ctx, StreamOrDevice.default.ctx)
        guard status == 0 else {
            mlx_array_free(packedHandle)
            throw ValidationError.nativeConversion
        }
        let packed = MLXArray(packedHandle)
        // Native to_fp8 saturates and native from_fp8 decodes NaN codes as480.
        // Route through the decoder's E4M3FN correction and restore overflow/
        // NaN semantics of the unclamped PyTorch eager cast explicitly; the
        // saturating policy keeps the native saturation and restores only NaN.
        let activation = try ModelOptFP8Weights(e4m3Bytes: packed, tensorScale: inverseScale)
            .decodedRows(0 ..< input.dim(0), dtype: input.dtype)
        let invalid: MLXArray
        switch activationOverflow {
        case .eagerNaN: invalid = logicalOr(abs(scaled) .> 464, isNaN(scaled))
        case .saturating: invalid = isNaN(scaled)
        }
        let calibrated = which(invalid, MLXArray(Float.nan).asType(input.dtype), activation)
        let weights = try decodedRows(outputRows, dtype: weightDType)
        return matmul(calibrated, weights.transposed())
    }
}
