import MLX

/// Host-validated calibration of one projection as device constants.
struct ModelOptCalibratedGraphConstants {
    enum Storage {
        /// `globalScale` and `overflowDivisor` are `[1]`; see the routed tables.
        case nvfp4(globalScale: MLXArray, overflowDivisor: MLXArray)
        /// `multipliers` is `[448 / amax, 1 / (448 / amax)]`.
        case fp8(multipliers: MLXArray)
    }
    let storage: Storage
    /// `[1]` float32 applied after the packed product: the NVFP4 inverse
    /// global weight scale, or the FP8 per-tensor weight scale.
    let outputScale: MLXArray
}

extension ModelOptCalibratedLinear {
    /// Explicit arrays of one graph call with `rows` input rows: float32
    /// activation, float32 sums and the output, plus for FP8 with several rows
    /// one decoded weight slice of at most 32 Mi elements (a single row decodes
    /// inside the vector kernel). Excludes resident weights, the caller's input
    /// and MLX backend scratch.
    public func graphWorkspaceBytes(rows: Int) throws -> Int {
        guard rows > 0, rows <= 1 << 16 else { throw Failure.workspaceLimit }
        var perRow = 4 * inputDimensions + 4 * outputDimensions
        perRow += weightDType.size * outputDimensions
        var (bytes, overflow) = rows.multipliedReportingOverflow(by: perRow)
        if format == .fp8, rows > 1 || inputDimensions <= 64 {
            let elements = outputDimensions.multipliedReportingOverflow(by: inputDimensions)
            let sum = bytes.addingReportingOverflow(
                min(elements.partialValue, 1 << 25) * weightDType.size)
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
            return ModelOptCalibratedGraphConstants(
                storage: .fp8(multipliers: MLXArray([multiplier, inverse])),
                outputScale: MLXArray([tensorScale]))
        }
    }

    /// Device-resident projection: exact activation quantization, then for
    /// NVFP4 a packed product on the producer bytes and one rounding by the
    /// weight scale, for FP8 the eager per-weight decode on the device. Never
    /// reads device values; throws only on static errors. NVFP4 `.rowwise` and
    /// FP8 are bit-identical to `checkedForward`.
    /// `squaredReLU` returns NemotronH's `relu2` of the result (fused into the
    /// final NVFP4 rounding kernel; bit-identical to applying it afterwards).
    public func graphForward(_ input: MLXArray, squaredReLU: Bool = false) throws -> MLXArray {
        try graphForward(input, squaredReLU: squaredReLU, accumulation: accumulation)
    }

    private func graphForward(
        _ input: MLXArray, squaredReLU: Bool, accumulation: ModelOptAccumulation
    ) throws -> MLXArray {
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
                sums, scale: constants.outputScale, dtype: weightDType, squaredReLU: squaredReLU)
        case .fp8(let multipliers):
            guard weight.dtype == .uint8, weight.shape == [outputDimensions, inputDimensions]
            else { throw Failure.invalidMetadata }
            let activation = ModelOptActivationKernels.fp8(flat, multipliers: multipliers)
            let value = exactFP8(activation, rows: rows, scale: constants.outputScale)
            if squaredReLU {
                let r = maximum(value, MLXArray(0))
                result = r * r
            } else {
                result = value
            }
        }
        return result.reshaped(Array(input.shape.dropLast()) + [outputDimensions])
    }

    /// The eager reference's product without its host synchronization: the
    /// same per-weight rounding and output-row tiles (row-count dependent below
    /// its 64 MiB bound), so the result is bit-identical; one row runs the
    /// fused vector kernel instead of decoding a slice.
    private func exactFP8(_ activation: MLXArray, rows: Int, scale: MLXArray) -> MLXArray {
        let limit = (64 << 20) / 32
        let live = rows * inputDimensions + rows * outputDimensions
        // Beyond the eager bound there is no reference tiling to reproduce;
        // larger slices (at most 32 Mi decoded elements) keep products efficient.
        let tileElements = live <= limit ? limit - live : 1 << 25
        let tileRows = max(1, min(outputDimensions, tileElements / inputDimensions))
        var tiles = stride(from: 0, to: outputDimensions, by: tileRows).map {
            $0 ..< min(outputDimensions, $0 + tileRows)
        }
        let pieces: [MLXArray]
        if rows == 1 && inputDimensions > 64 {
            pieces = vectorRuns(tiles).map { run in
                ModelOptRowExact.product(
                    weight[run.rows, 0...], scale: scale, vectors: activation,
                    splitsK: run.splitsK)
            }
        } else {
            if rows == 1 { tiles = Self.mergedVectorTiles(tiles, inputs: inputDimensions) }
            pieces = tiles.map { tile in
                matmul(
                    activation,
                    ModelOptActivationKernels.fp8DenseWeight(
                        weight[tile, 0...], scale: scale, dtype: weightDType
                    ).transposed())
            }
        }
        return pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: -1)
    }

    /// One dispatch per run of eager one-row tiles that share MLX's K split.
    private func vectorRuns(_ tiles: [Range<Int>]) -> [(rows: Range<Int>, splitsK: Bool)] {
        var runs: [(rows: Range<Int>, splitsK: Bool)] = []
        for tile in tiles {
            let splitsK = inputDimensions >= 16 * tile.count
            if let last = runs.last, last.splitsK == splitsK {
                runs[runs.count - 1].rows = last.rows.lowerBound ..< tile.upperBound
            } else {
                runs.append((tile, splitsK))
            }
        }
        return runs
    }

    /// Eager tiles of a one-row call: `(64 MiB / 32 - K - N) / K` output rows.
    private var oneRowTiles: [Range<Int>] {
        let limit = (64 << 20) / 32
        let live = inputDimensions + outputDimensions
        let tileElements = live <= limit ? limit - live : 1 << 25
        let tileRows = max(1, min(outputDimensions, tileElements / inputDimensions))
        return stride(from: 0, to: outputDimensions, by: tileRows).map {
            $0 ..< min(outputDimensions, $0 + tileRows)
        }
    }

    /// Up to 16 rows, each bit-identical to `graphForward` of that row alone
    /// (batch-invariant decode), with each weight read once for all rows.
    /// NVFP4 already runs one single-vector product per row; FP8 uses
    /// ``ModelOptRowExact`` over the one-row eager tiles.
    public func rowExactForward(_ input: MLXArray, squaredReLU: Bool = false) throws -> MLXArray {
        guard let constants = graphConstants else { throw Failure.invalidMetadata }
        guard input.ndim > 0, input.dim(-1) == inputDimensions, input.dtype == weightDType else {
            throw Failure.invalidInput
        }
        let rows = input.size / inputDimensions
        guard case .fp8(let multipliers) = constants.storage, rows > 1,
            inputDimensions > 64
        else {
            guard rows <= ModelOptRowExact.maximumRows else { throw Failure.workspaceLimit }
            return try graphForward(input, squaredReLU: squaredReLU, accumulation: .rowwise)
        }
        guard rows <= ModelOptRowExact.maximumRows,
            weight.dtype == .uint8, weight.shape == [outputDimensions, inputDimensions]
        else { throw Failure.workspaceLimit }
        let activation = ModelOptActivationKernels.fp8(
            input.reshaped(rows, inputDimensions), multipliers: multipliers)
        let pieces = vectorRuns(oneRowTiles).map { run in
            ModelOptRowExact.product(
                weight[run.rows, 0...], scale: constants.outputScale, vectors: activation,
                splitsK: run.splitsK)
        }
        var result = pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: -1)
        if squaredReLU {
            let r = maximum(result, MLXArray(0))
            result = r * r
        }
        return result.reshaped(Array(input.shape.dropLast()) + [outputDimensions])
    }

    /// MLX's single-vector product (`gemv`, mlx cb77239b matmul.cpp) reduces
    /// each output over K in an order fixed by K alone, except that it splits K
    /// across 8 simdgroups when K >= 16 * outputs (K > 64). Adjacent eager
    /// tiles that agree on that split, merged up to 32 Mi decoded elements,
    /// therefore give bit-identical outputs with fewer dispatches.
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
}
