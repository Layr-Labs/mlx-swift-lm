import Foundation
import MLX

/// Per-expert calibration as float32 device tables, built once from the
/// init-validated host scalars with the eager reference's host arithmetic.
struct ModelOptNVFP4RouteTables {
    /// MLX global activation scale, `inputScale * 2688`.
    let globalScale: MLXArray
    /// `6 * inputScale` under the eager rule, +infinity when saturating.
    let overflowDivisor: MLXArray
    /// MLX's in-kernel `(tensorScale * 2688) / 2688`.
    let inverseWeightScale: MLXArray

    init(
        inputScales: [Float], tensorScales: [Float], activationOverflow: ModelOptActivationOverflow
    ) {
        globalScale = MLXArray(inputScales.map { $0 * (448 * 6) })
        overflowDivisor = MLXArray(
            inputScales.map { activationOverflow.nvfp4OverflowDivisor(inputScale: $0) })
        inverseWeightScale = ModelOptActivationKernels.inverseWeightScale(
            MLXArray(tensorScales.map { $0 * (448 * 6) }))
        eval(globalScale, overflowDivisor, inverseWeightScale)
    }
}

/// Host-built int32 maps from routed rows to input rows, keyed by shape.
/// Building them on the host needs no device values, so no synchronization.
final class ModelOptRowSourceCache: @unchecked Sendable {
    private let lock = NSLock()
    private var entries: [[Int]: MLXArray] = [:]

    /// `[0, rows)` as uint32: explicit gather row indices, so MLX does not
    /// build its default `arange` on the device for every call.
    func identity(rows: Int) -> MLXArray {
        let key = [-2, rows]
        if let cached = lock.withLock({ entries[key] }) { return cached }
        let result = MLXArray((0 ..< rows).map { UInt32($0) })
        lock.withLock {
            if entries.count >= 16 { entries.removeAll() }
            entries[key] = result
        }
        return result
    }

    func sources(inputBatch: [Int], routedBatch: [Int], rowsPerAssignment m: Int) -> MLXArray {
        let key = inputBatch + [-1] + routedBatch + [m]
        if let cached = lock.withLock({ entries[key] }) { return cached }
        let rank = routedBatch.count
        let padded = Array(repeating: 1, count: rank - inputBatch.count) + inputBatch
        var strides = Array(repeating: 0, count: rank)
        var running = 1
        for axis in stride(from: rank - 1, through: 0, by: -1) {
            strides[axis] = padded[axis] == 1 ? 0 : running
            running *= padded[axis]
        }
        let assignments = routedBatch.reduce(1, *)
        var values = [Int32](repeating: 0, count: assignments * m)
        var coordinate = Array(repeating: 0, count: rank)
        for assignment in 0 ..< assignments {
            var source = 0
            for axis in 0 ..< rank { source += coordinate[axis] * strides[axis] }
            for row in 0 ..< m { values[assignment * m + row] = Int32(source * m + row) }
            var axis = rank - 1
            while axis >= 0 {
                coordinate[axis] += 1
                if coordinate[axis] < routedBatch[axis] { break }
                coordinate[axis] = 0
                axis -= 1
            }
        }
        let result = MLXArray(values)
        lock.withLock {
            if entries.count >= 16 { entries.removeAll() }
            entries[key] = result
        }
        return result
    }
}

extension ModelOptNVFP4SwitchLinear {
    public enum GraphFailure: Error { case invalidExperts, invalidInput, invalidIndices, workspace }

    /// Largest routed row count of one graph call (8192 tokens x 22 experts fit).
    public static let maximumGraphRows = 1 << 18

    /// Explicit arrays of one graph call with `rows` routed rows: float32
    /// activation, uint32 IDs, row scales, float32 sums and the output. Excludes
    /// the resident bank, caller inputs and MLX backend scratch.
    public func graphWorkspaceBytes(rows: Int, outputBytesPerElement: Int = 4) throws -> Int {
        guard rows > 0, rows <= Self.maximumGraphRows, outputBytesPerElement > 0 else {
            throw GraphFailure.workspace
        }
        let perRow = 4 * inputDims + 4 * outputDims + outputBytesPerElement * outputDims + 12
        let (bytes, overflow) = rows.multipliedReportingOverflow(by: perRow)
        guard !overflow else { throw GraphFailure.workspace }
        return bytes
    }

    /// Device-resident route: per-row exact activation quantization with the
    /// row's expert scale, one gathered packed NVFP4 product and one weight-scale
    /// rounding. Throws only on static shape/dtype errors; never reads device
    /// values. Invalid IDs yield NaN rows and `validIndices == false`.
    /// `.rowwise` accumulation is bit-identical to `checkedForward`.
    /// `squaredReLU` returns NemotronH's `relu2` of the result, fused into the
    /// final rounding kernel (bit-identical to applying `relu2` afterwards).
    public func graphForward(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false,
        squaredReLU: Bool = false
    ) throws -> ModelOptGraphOutput {
        guard weight.dtype == .uint32, weight.shape == [numExperts, outputDims, inputDims / 8],
            blockScales.dtype == .uint8,
            blockScales.shape == [numExperts, outputDims, inputDims / 16]
        else { throw GraphFailure.invalidExperts }
        guard x.ndim >= 2, x.dim(-1) == inputDims, x.dim(-2) > 0,
            [.float16, .bfloat16, .float32].contains(x.dtype),
            originalWeightDType == nil || x.dtype == originalWeightDType
        else { throw GraphFailure.invalidInput }
        guard [.int32, .uint32].contains(indices.dtype), indices.size > 0 else {
            throw GraphFailure.invalidIndices
        }
        let inputBatch = Array(x.shape.dropLast(2))
        let rank = max(inputBatch.count, indices.ndim)
        let lhs = Array(repeating: 1, count: rank - inputBatch.count) + inputBatch
        let rhs = Array(repeating: 1, count: rank - indices.ndim) + indices.shape
        var shape: [Int] = []
        var assignments = 1
        for (a, b) in zip(lhs, rhs) {
            guard a > 0, b > 0, a == b || a == 1 || b == 1 else { throw GraphFailure.invalidInput }
            let dimension = max(a, b)
            guard assignments <= Self.maximumGraphRows / dimension else {
                throw GraphFailure.workspace
            }
            assignments *= dimension
            shape.append(dimension)
        }
        let m = x.dim(-2)
        guard assignments <= Self.maximumGraphRows / m else { throw GraphFailure.workspace }
        let rows = assignments * m
        _ = try graphWorkspaceBytes(rows: rows, outputBytesPerElement: x.dtype.size)

        let inputRows = x.reshaped(inputBatch.reduce(1, *) * m, inputDims)
        let sources = rowSources.sources(
            inputBatch: inputBatch, routedBatch: shape, rowsPerAssignment: m)
        // int32 or uint32: the kernel reads either; out-of-range IDs of both
        // (including uint32 values that wrap negative) are invalid rows.
        var experts = broadcast(indices, to: shape).flattened()
        if m > 1 { experts = repeated(experts, count: m, axis: 0) }
        // MLX's sorted matrix route needs sorted rows, at least 16, and on
        // average 4 per expert; below that it uses the per-row kernel anyway.
        let tiled =
            accumulation == .tiled && sortedIndices && rows >= 16 && rows >= 4 * numExperts
        let quantized = ModelOptActivationKernels.nvfp4Routed(
            inputRows, sources: sources, experts: experts,
            globalScale: routeTables.globalScale, overflowDivisor: routeTables.overflowDivisor,
            inverseWeightScale: routeTables.inverseWeightScale,
            activationDType: tiled ? x.dtype : .float32)
        // Without a global scale, MLX returns the unscaled float32 sum for each
        // row; scaledCast then rounds once, exactly like the global-scale kernel.
        // Rowwise stays on MLX's gathered single-vector kernel: exact multi-row
        // variants measured no faster, as the per-row reduction is ALU-bound.
        if tiled, ModelOptNVFP4GroupedGEMM.isEnabled,
            ModelOptNVFP4GroupedGEMM.supports(inputs: inputDims)
        {
            let value = ModelOptNVFP4GroupedGEMM.product(
                quantized.activation, weight: weight, scales: blockScales,
                experts: quantized.safeExperts, expertScale: routeTables.inverseWeightScale,
                squaredReLU: squaredReLU)
            let signed = experts.dtype == .int32 ? experts : experts.asType(.int64)
            return ModelOptGraphOutput(
                value: value.reshaped(shape + [m, outputDims]),
                validIndices: logicalAnd(signed .>= 0, signed .< numExperts).all())
        }
        // MLX takes its sorted (matrix) route only without explicit row indices.
        let sums = gatherQuantizedMM(
            quantized.activation.reshaped(rows, 1, inputDims), weight,
            scales: blockScales, biases: nil,
            lhsIndices: tiled ? nil : rowSources.identity(rows: rows),
            rhsIndices: quantized.safeExperts, transpose: true, groupSize: 16, bits: 4,
            mode: .nvfp4, sortedIndices: tiled)
        let value = ModelOptActivationKernels.scaledCast(
            sums.reshaped(rows, outputDims), scale: quantized.rowScale, dtype: x.dtype,
            squaredReLU: squaredReLU)
        let signed = experts.dtype == .int32 ? experts : experts.asType(.int64)
        let valid = logicalAnd(signed .>= 0, signed .< numExperts).all()
        return ModelOptGraphOutput(
            value: value.reshaped(shape + [m, outputDims]), validIndices: valid)
    }
}
