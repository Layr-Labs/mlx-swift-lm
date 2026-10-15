import Cmlx
import Foundation
import MLX

/// Host-validated calibration of one projection as device constants.
struct ModelOptCalibratedGraphConstants {
    enum Storage {
        /// `globalScale` and `overflowDivisor` are `[1]`; see the routed tables.
        case nvfp4(globalScale: MLXArray, overflowDivisor: MLXArray)
        /// `multipliers` is `[448 / amax, 1 / (448 / amax)]`. `unitScales` is the
        /// shared E8M0 1.0 plane for MXFP8 products, nil when K % 32 != 0.
        /// `invalidOutputs` marks rows of E4M3FN NaN-code weights, when present.
        case fp8(multipliers: MLXArray, unitScales: MLXArray?, invalidOutputs: MLXArray?)
    }
    let storage: Storage
    /// `[1]` float32 applied after the packed product: the NVFP4 inverse
    /// global weight scale, or the FP8 per-tensor weight scale.
    let outputScale: MLXArray
}

/// Read-only E8M0 planes of exponent 127 (1.0), one per `[N, K/32]` shape and
/// shared by every live projection of that shape. With unit block scales an
/// MXFP8 product reads the unmodified per-tensor E4M3 bytes; the producer
/// per-tensor scale is applied to the float32 sums afterwards.
final class ModelOptUnitScalePlanes: @unchecked Sendable {
    static let shared = ModelOptUnitScalePlanes()
    private final class Entry { weak var plane: MLXArray? }
    private let lock = NSLock()
    private var entries: [[Int]: Entry] = [:]

    func plane(rows: Int, groups: Int) -> MLXArray {
        lock.withLock {
            let key = [rows, groups]
            if let plane = entries[key]?.plane { return plane }
            let plane = MLXArray.full(
                [rows, groups], values: MLXArray(UInt8(127)), type: UInt8.self)
            eval(plane)
            let entry = Entry()
            entry.plane = plane
            entries = entries.filter { $0.value.plane != nil }
            entries[key] = entry
            return plane
        }
    }
}

extension ModelOptCalibratedLinear {
    /// Resident bytes of the shared unit-scale plane a `.packed` FP8
    /// projection of this shape holds (one plane per shape across all such
    /// projections); zero for NVFP4 and for `.exactReference`.
    public var graphResidentBytes: Int {
        guard format == .fp8, fp8Product == .packed, inputDimensions % 32 == 0 else { return 0 }
        return outputDimensions * (inputDimensions / 32)
    }

    /// Explicit arrays of one graph call with `rows` input rows: float32
    /// activation, float32 sums and the output, plus for FP8 one decoded
    /// weight slice (exact product: at most 32 Mi elements; dense fallback for
    /// K % 32 != 0: the whole matrix in float32). Excludes resident weights,
    /// the caller's input and MLX backend scratch.
    public func graphWorkspaceBytes(rows: Int) throws -> Int {
        guard rows > 0, rows <= 1 << 16 else { throw Failure.workspaceLimit }
        var perRow = 4 * inputDimensions + 4 * outputDimensions
        perRow += weightDType.size * outputDimensions
        var (bytes, overflow) = rows.multipliedReportingOverflow(by: perRow)
        if format == .fp8 {
            let elements = outputDimensions.multipliedReportingOverflow(by: inputDimensions)
            var slice = 0
            if inputDimensions % 32 != 0 {
                slice = elements.partialValue.multipliedReportingOverflow(by: 8).partialValue
            } else if fp8Product == .exactReference {
                slice = min(elements.partialValue, 1 << 25) * weightDType.size
            }
            let sum = bytes.addingReportingOverflow(slice)
            overflow = overflow || elements.overflow || sum.overflow
            bytes = sum.partialValue
        }
        guard !overflow else { throw Failure.workspaceLimit }
        return bytes
    }

    /// Host reads of the registered scalars with the eager per-call checks.
    /// Runs at init and after each parameter update, never inside a forward.
    func makeGraphConstants() throws -> ModelOptCalibratedGraphConstants {
        let input = try scalar(inputScale)
        try Self.validate(input, dtype: weightDType, format: format)
        switch format {
        case .nvfp4:
            guard let global = weightScale2 else { throw Failure.invalidMetadata }
            let tensorScale = try scalar(global)
            guard tensorScale.isFinite, tensorScale > 0,
                tensorScale <= Float.greatestFiniteMagnitude / (448 * 6)
            else { throw Failure.invalidMetadata }
            let inverse = ModelOptActivationKernels.inverseWeightScale(
                MLXArray([tensorScale * (448 * 6)]))
            let storage = ModelOptCalibratedGraphConstants.Storage.nvfp4(
                globalScale: MLXArray([input * (448 * 6)]),
                overflowDivisor: MLXArray([6 * input]))
            eval(inverse)
            return ModelOptCalibratedGraphConstants(storage: storage, outputScale: inverse)
        case .fp8:
            let tensorScale = try scalar(weightScale)
            let amax = input * 448
            let multiplier = 448 / amax
            let inverse = 1 / multiplier
            guard tensorScale.isFinite, tensorScale > 0, amax.isFinite, amax > 0,
                multiplier.isFinite, multiplier > 0, inverse.isFinite, inverse > 0,
                weight.dtype == .uint8, weight.shape == [outputDimensions, inputDimensions]
            else { throw Failure.invalidMetadata }
            // E4M3FN reserves 0x7f/0xff for NaN; the eager decoder turns every
            // output that reads one into NaN. Scan once; mask only when present.
            let nanCodes = (weight & UInt8(0x7F)) .== UInt8(0x7F)
            let invalidRows = nanCodes.any(axis: 1)
            let hasNaNCodes = invalidRows.any().item(Bool.self)
            let unitScales =
                fp8Product == .packed && inputDimensions % 32 == 0
                ? ModelOptUnitScalePlanes.shared.plane(
                    rows: outputDimensions, groups: inputDimensions / 32) : nil
            if hasNaNCodes { eval(invalidRows) }
            return ModelOptCalibratedGraphConstants(
                storage: .fp8(
                    multipliers: MLXArray([multiplier, inverse]), unitScales: unitScales,
                    invalidOutputs: hasNaNCodes ? invalidRows : nil),
                outputScale: MLXArray([tensorScale]))
        }
    }

    /// Device-resident projection: exact activation quantization, a packed
    /// product on the producer bytes and one rounding by the producer weight
    /// scale. Never reads device values; throws only on static errors.
    /// NVFP4 `.rowwise` is bit-identical to `checkedForward`. FP8 products sum
    /// E4M3 weights exactly in float32 before the per-tensor scale; the eager
    /// reference first rounds every scaled weight to `weightDType`.
    public func graphForward(_ input: MLXArray) throws -> MLXArray {
        guard let constants = graphConstants else { throw Failure.invalidMetadata }
        guard input.ndim > 0, input.dim(-1) == inputDimensions, input.dtype == weightDType else {
            throw Failure.invalidInput
        }
        var rows = 1
        for dimension in input.shape.dropLast() {
            guard dimension > 0, rows <= (1 << 16) / dimension else {
                throw Failure.workspaceLimit
            }
            rows *= dimension
        }
        _ = try graphWorkspaceBytes(rows: rows)
        let flat = input.reshaped(rows, inputDimensions)
        let result: MLXArray
        switch constants.storage {
        case .nvfp4(let globalScale, let overflowDivisor):
            guard weight.dtype == .uint32,
                weight.shape == [outputDimensions, inputDimensions / 8],
                weightScale.dtype == .uint8,
                weightScale.shape == [outputDimensions, inputDimensions / 16]
            else { throw Failure.invalidMetadata }
            let activation = ModelOptActivationKernels.nvfp4Dense(
                flat, globalScale: globalScale, overflowDivisor: overflowDivisor)
            let sums: MLXArray
            // MLX's single-vector product without a global scale is the same
            // kernel as the eager global-scale one, except K of 64/128 (quad).
            if accumulation == .tiled || (rows == 1 && ![64, 128].contains(inputDimensions)) {
                sums = quantizedMM(
                    activation, weight, scales: weightScale, biases: nil,
                    transpose: true, groupSize: 16, bits: 4, mode: .nvfp4)
            } else {
                sums = gatherQuantizedMM(
                    activation.reshaped(rows, 1, inputDimensions),
                    weight.reshaped(1, outputDimensions, inputDimensions / 8),
                    scales: weightScale.reshaped(1, outputDimensions, inputDimensions / 16),
                    biases: nil, rhsIndices: MLXArray.zeros([rows], type: UInt32.self),
                    transpose: true, groupSize: 16, bits: 4, mode: .nvfp4
                ).reshaped(rows, outputDimensions)
            }
            result = ModelOptActivationKernels.scaledCast(
                sums, scale: constants.outputScale, dtype: weightDType)
        case .fp8(let multipliers, let unitScales, let invalidOutputs):
            guard weight.dtype == .uint8, weight.shape == [outputDimensions, inputDimensions]
            else { throw Failure.invalidMetadata }
            let activation = ModelOptActivationKernels.fp8(flat, multipliers: multipliers)
            var value: MLXArray
            if fp8Product == .exactReference {
                value = exactFP8(
                    activation.asType(weightDType), rows: rows, scale: constants.outputScale)
            } else if let unitScales {
                let sums = quantizedMM(
                    activation, weight.view(dtype: .uint32), scales: unitScales, biases: nil,
                    transpose: true, groupSize: 32, bits: 8, mode: .mxfp8)
                value = ModelOptActivationKernels.scaledCast(
                    sums, scale: constants.outputScale, dtype: weightDType)
            } else {
                value = try denseFP8(activation.asType(weightDType), scale: constants.outputScale)
            }
            if let invalidOutputs {
                value = which(invalidOutputs, MLXArray(Float.nan).asType(weightDType), value)
            }
            result = value
        }
        return result.reshaped(Array(input.shape.dropLast()) + [outputDimensions])
    }

    /// The eager reference's product without its host synchronization: the
    /// same per-weight rounding and output-row tiles (row-count dependent below
    /// its 64 MiB bound), so the result is bit-identical. Beyond the eager
    /// bound, tiles keep each decoded slice under 64 MiB.
    private func exactFP8(_ activation: MLXArray, rows: Int, scale: MLXArray) -> MLXArray {
        let limit = (64 << 20) / 32
        let live = rows * inputDimensions + rows * outputDimensions
        let remaining = live <= limit ? limit - live : limit
        let tileRows = max(1, min(outputDimensions, remaining / inputDimensions))
        var tiles = stride(from: 0, to: outputDimensions, by: tileRows).map {
            $0 ..< min(outputDimensions, $0 + tileRows)
        }
        if rows == 1 { tiles = Self.mergedVectorTiles(tiles, inputs: inputDimensions) }
        let pieces = tiles.map { tile in
            matmul(
                activation,
                ModelOptActivationKernels.fp8DenseWeight(
                    weight[tile, 0...], scale: scale, dtype: weightDType
                ).transposed())
        }
        return pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: -1)
    }

    /// MLX's single-vector product (`gemv`, mlx cb77239b matmul.cpp) reduces
    /// each output over K in an order fixed by K alone, except that it splits K
    /// across 8 simdgroups when K >= 16 * outputs (K > 64). Adjacent eager
    /// tiles that agree on that split, merged up to 32 Mi decoded elements,
    /// therefore give bit-identical outputs with far fewer dispatches.
    static func mergedVectorTiles(_ tiles: [Range<Int>], inputs: Int) -> [Range<Int>] {
        func splitsK(_ outputs: Int) -> Bool { inputs > 64 && inputs >= 16 * outputs }
        func kind(_ outputs: Int) -> [Bool] { [splitsK(outputs), outputs < 4] }
        var merged: [Range<Int>] = []
        for tile in tiles {
            if let last = merged.last, kind(last.count) == kind(tile.count),
                kind(last.count + tile.count) == kind(tile.count),
                (last.count + tile.count) * inputs <= 1 << 25
            {
                merged[merged.count - 1] = last.lowerBound ..< tile.upperBound
            } else {
                merged.append(tile)
            }
        }
        return merged
    }

    /// Eager-equivalent decode for K % 32 != 0, which MXFP8 groups cannot tile.
    private func denseFP8(_ activation: MLXArray, scale: MLXArray) throws -> MLXArray {
        var handle = mlx_array_new()
        guard
            mlx_from_fp8(&handle, weight.ctx, DType.float32.cmlxDtype, StreamOrDevice.default.ctx)
                == 0
        else {
            mlx_array_free(handle)
            throw ModelOptFP8Weights.ValidationError.nativeConversion
        }
        let decoded = which(
            logicalOr(weight .== UInt8(127), weight .== UInt8(255)), MLXArray(Float.nan),
            MLXArray(handle))
        let dense = (decoded * scale.reshaped([])).asType(weightDType)
        return matmul(activation, dense.transposed())
    }
}
