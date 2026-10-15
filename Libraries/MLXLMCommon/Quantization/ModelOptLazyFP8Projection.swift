import Cmlx
import MLX

/// Explicit graph-only counterpart to the bounded eager FP8 baseline. No model
/// or sampler selects it automatically. The caller owns the quoted array budget,
/// separate backend scratch, and the final evaluation/validation fence.
public struct ModelOptLazyFP8Projection {
    public let value: MLXArray
    /// Must evaluate and check at the caller's normal fence BEFORE consuming
    /// value. Invalid mutable calibration also poisons value with NaN.
    public let calibrationValid: MLXArray
    public let arrayWorkspaceBytes: Int

    public static func requiredArrayWorkspaceBytes(
        rows: Int, inputDimensions: Int,
        outputDimensions: Int
    ) throws -> Int {
        guard rows > 0, rows <= 8192, inputDimensions > 0, outputDimensions > 0 else {
            throw ModelOptCalibratedLinear.Failure.invalidInput
        }
        // 32 bytes per element conservatively covers registered/input storage,
        // native decode, masks, FP32 multiplication and dtype casts. Dense
        // weights remain live for this lazy graph: no eager tile retirement.
        var total = 256
        for (a, b) in [
            (rows, inputDimensions), (outputDimensions, inputDimensions), (rows, outputDimensions),
        ] {
            let (elements, overflow) = a.multipliedReportingOverflow(by: b)
            let (bytes, bytesOverflow) = elements.multipliedReportingOverflow(by: 32)
            let (next, sumOverflow) = total.addingReportingOverflow(bytes)
            guard !overflow, !bytesOverflow, !sumOverflow else {
                throw ModelOptCalibratedLinear.Failure.workspaceLimit
            }
            total = next
        }
        return total
    }
}

extension ModelOptCalibratedLinear {
    /// Uses current registered scalar tensors, never host-cached calibration.
    /// Shape and budget rejection precede construction of any new MLX operation.
    /// Runtime scalar validity is device evidence returned for the owner's fence;
    /// this is deliberately not a drop-in UnaryLayer that could discard it.
    public func lazyFP8Forward(_ input: MLXArray, admittedArrayWorkspaceBytes: Int) throws
        -> ModelOptLazyFP8Projection
    {
        guard case .fp8 = format, weight.dtype == .uint8,
            weight.shape == [outputDimensions, inputDimensions], weightScale2 == nil,
            weightScale.dtype == .float32, inputScale.dtype == .float32,
            weightScale.shape.isEmpty || weightScale.shape == [1],
            inputScale.shape.isEmpty || inputScale.shape == [1]
        else { throw Failure.invalidMetadata }
        guard input.ndim > 0, input.dim(-1) == inputDimensions, input.dtype == weightDType else {
            throw Failure.invalidInput
        }
        var rows = 1
        for dimension in input.shape.dropLast() {
            guard dimension > 0, dimension <= 8192, rows <= 8192 / dimension else {
                throw Failure.workspaceLimit
            }
            rows *= dimension
        }
        let quote = try ModelOptLazyFP8Projection.requiredArrayWorkspaceBytes(
            rows: rows,
            inputDimensions: inputDimensions, outputDimensions: outputDimensions)
        guard admittedArrayWorkspaceBytes >= quote else { throw Failure.workspaceLimit }
        let inputMultiplier = inputScale.reshaped([])
        let weightMultiplier = weightScale.reshaped([])
        let amax = inputMultiplier * Float(448)
        let multiplier = Float(448) / amax
        let inverse = Float(1) / multiplier
        func positiveFinite(_ value: MLXArray) -> MLXArray {
            logicalAnd(isFinite(value), value .> 0)
        }
        let valid = [
            inputMultiplier, weightMultiplier, amax, multiplier, inverse,
            Float(1) / inputMultiplier,
        ]
        .map(positiveFinite).reduce(MLXArray(true)) { logicalAnd($0, $1) }
        func decode(_ bytes: MLXArray) throws -> MLXArray {
            var handle = mlx_array_new()
            guard
                mlx_from_fp8(
                    &handle, bytes.ctx, DType.float32.cmlxDtype, StreamOrDevice.default.ctx) == 0
            else {
                mlx_array_free(handle)
                throw ModelOptFP8Weights.ValidationError.nativeConversion
            }
            return which(
                logicalOr(bytes .== 127, bytes .== 255), MLXArray(Float.nan), MLXArray(handle))
        }
        let scaled = input.reshaped(rows, inputDimensions).asType(.float32) * multiplier
        var handle = mlx_array_new()
        guard mlx_to_fp8(&handle, scaled.ctx, StreamOrDevice.default.ctx) == 0 else {
            mlx_array_free(handle)
            throw ModelOptFP8Weights.ValidationError.nativeConversion
        }
        let packed = MLXArray(handle)
        // Match ModelOpt _fp8_eager, including its non-SATFINITE >464/NaN rule.
        // This is FP8 projection calibration, NOT the CUDA FP8 KV-cache codec.
        let activation = (try decode(packed) * inverse).asType(weightDType)
        let invalid = logicalOr(abs(scaled) .> 464, isNaN(scaled))
        let calibrated = which(invalid, MLXArray(Float.nan).asType(weightDType), activation)
        let dense = (try decode(weight) * weightMultiplier).asType(weightDType)
        let result = matmul(calibrated, dense.transposed())
        let checked = which(valid, result, MLXArray(Float.nan).asType(weightDType))
            .reshaped(Array(input.shape.dropLast()) + [outputDimensions])
        return ModelOptLazyFP8Projection(
            value: checked, calibrationValid: valid, arrayWorkspaceBytes: quote)
    }
}
