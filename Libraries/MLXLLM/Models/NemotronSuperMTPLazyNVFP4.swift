import MLX
import MLXLMCommon

/// Graph-only counterpart of `ModelOptCalibratedLinear.checkedForward` for the
/// NVFP4 format, built from the layer's registered tensors instead of host
/// scalar reads. It issues the same `quantizedQuantizedMM` call and the same
/// ModelOpt E4M3 overflow/NaN rule as `ModelOptNVFP4Weights.calibratedMatmul`,
/// so finite results are bit-identical to the eager path. Scalar validity is
/// returned for the owner's completion fence and also poisons the value.
/// Local workaround: the fastpath-owned ModelOpt files have no lazy NVFP4
/// projection yet.
enum NemotronSuperMTPLazyNVFP4 {
    enum Failure: Error { case format, metadata, input, workspace }

    /// Explicit arrays at 32 bytes per element: input promotion, overflow
    /// mask and output. Excludes the resident packed weights and backend scratch.
    static func arrayBytes(rows: Int, inputDimensions: Int, outputDimensions: Int) throws -> Int {
        guard rows > 0, rows <= 8192, inputDimensions > 0, outputDimensions > 0 else {
            throw Failure.workspace
        }
        let (sum, o1) = inputDimensions.addingReportingOverflow(outputDimensions)
        let (elements, o2) = sum.multipliedReportingOverflow(by: rows)
        let (bytes, o3) = elements.multipliedReportingOverflow(by: 32)
        let (total, o4) = bytes.addingReportingOverflow(1024)
        guard !o1, !o2, !o3, !o4 else { throw Failure.workspace }
        return total
    }

    struct Tensors {
        let weight: MLXArray
        let blockScales: MLXArray
        let tensorScale: MLXArray
        let inputScale: MLXArray
    }

    /// Every shape, dtype and budget refusal, before any MLX operation.
    static func validate(
        _ projection: ModelOptCalibratedLinear, inputShape: [Int], inputDType: DType,
        admittedArrayBytes: Int
    ) throws -> Tensors {
        guard case .nvfp4 = projection.format else { throw Failure.format }
        let inputs = projection.inputDimensions
        let outputs = projection.outputDimensions
        let tensors = Dictionary(uniqueKeysWithValues: projection.parameters().flattened())
        guard let weight = tensors["weight"], let blockScales = tensors["weight_scale"],
            let global = tensors["weight_scale_2"], let inputScale = tensors["input_scale"],
            weight.dtype == .uint32, weight.shape == [outputs, inputs / 8],
            blockScales.dtype == .uint8, blockScales.shape == [outputs, inputs / 16],
            global.dtype == .float32, global.shape.isEmpty || global.shape == [1],
            inputScale.dtype == .float32, inputScale.shape.isEmpty || inputScale.shape == [1]
        else { throw Failure.metadata }
        guard !inputShape.isEmpty, inputShape.last == inputs, inputDType == projection.weightDType
        else { throw Failure.input }
        var rows = 1
        for dimension in inputShape.dropLast() {
            guard dimension > 0, dimension <= 8192, rows <= 8192 / dimension else {
                throw Failure.workspace
            }
            rows *= dimension
        }
        let quote = try arrayBytes(rows: rows, inputDimensions: inputs, outputDimensions: outputs)
        guard admittedArrayBytes >= quote else { throw Failure.workspace }
        return Tensors(
            weight: weight, blockScales: blockScales, tensorScale: global.reshaped([]),
            inputScale: inputScale.reshaped([]))
    }

    static func forward(
        _ projection: ModelOptCalibratedLinear, _ input: MLXArray, admittedArrayBytes: Int
    ) throws -> (value: MLXArray, valid: MLXArray) {
        let tensors = try validate(
            projection, inputShape: input.shape, inputDType: input.dtype,
            admittedArrayBytes: admittedArrayBytes)
        let inputs = projection.inputDimensions
        let outputs = projection.outputDimensions
        let rows = input.size / inputs
        let weight = tensors.weight
        let blockScales = tensors.blockScales
        let scale = tensors.inputScale
        let tensorScale = tensors.tensorScale
        let bound = Float(448 * 6)
        func positiveFinite(_ value: MLXArray) -> MLXArray {
            logicalAnd(isFinite(value), value .> 0)
        }
        // Union of ModelOptCalibratedLinear.validate, ModelOptNVFP4Weights.init
        // and calibratedMatmul's host checks, evaluated on device.
        let valid = [
            positiveFinite(scale), positiveFinite(scale * bound),
            positiveFinite(bound / (scale * bound)), isFinite(Float(1) / scale),
            scale .<= Float.greatestFiniteMagnitude / bound,
            positiveFinite(tensorScale), tensorScale .<= Float.greatestFiniteMagnitude / bound,
        ].reduce(MLXArray(true)) { logicalAnd($0, $1) }
        let x = input.reshaped(rows, inputs)
        let result = quantizedQuantizedMM(
            x, weight, scales: blockScales, groupSize: 16, bits: 4, mode: .nvfp4,
            globalScaleX: scale * bound, globalScaleW: tensorScale * bound)
        let promoted = x.asType(.float32)
        let rawScale = abs(promoted).max(axis: -1, keepDims: true) / (scale * Float(6))
        let invalid = logicalOr(
            rawScale .> 464, isNaN(promoted).any(axis: -1, keepDims: true))
        let nan = MLXArray(Float.nan).asType(result.dtype)
        let checked = which(valid, which(invalid, nan, result), nan)
        return (checked.reshaped(Array(input.shape.dropLast()) + [outputs]), valid)
    }
}
