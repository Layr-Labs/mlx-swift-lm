import MLX
import MLXNN

/// Routed ModelOpt NVFP4 experts. `callAsFunction` runs the device-resident
/// graph route by default (`graphForward`); `checkedForward` is the bounded
/// eager reference, which synchronizes routing IDs to the host, groups by
/// expert and restores assignment order. Both preserve the producer bytes and
/// scales; neither uses affine QMM.
public final class ModelOptNVFP4SwitchLinear: SwitchLinear {
    public enum Failure: Error {
        case invalidExperts, invalidIndices, invalidInput, invalidBroadcast, workspaceLimit
    }
    @ModuleInfo(key: "block_scales") var blockScales: MLXArray
    public let originalWeightDType: DType?
    private let tensorScales: [Float]
    private let inputScales: [Float]
    private static let maximumAssignments = 8192
    private static let maximumScratchBytes = 64 * 1024 * 1024
    public var execution: ModelOptExecution = .graph
    public var accumulation: ModelOptAccumulation = .tiled
    /// Device copies of the immutable, init-validated producer calibration.
    let routeTables: ModelOptNVFP4RouteTables
    let rowSources = ModelOptRowSourceCache()

    public init(experts: [ModelOptNVFP4Weights], inputScales: [Float], weightDType: DType? = nil)
        throws
    {
        if let weightDType, ![DType.float16, .bfloat16, .float32].contains(weightDType) {
            throw Failure.invalidExperts
        }
        self.originalWeightDType = weightDType
        guard let first = experts.first, experts.count <= 4096,
            experts.count == inputScales.count,
            experts.allSatisfy({
                $0.inputDimensions == first.inputDimensions
                    && $0.outputDimensions == first.outputDimensions
            })
        else { throw Failure.invalidExperts }
        for scale in inputScales {
            guard scale.isFinite, scale > 0, scale <= Float.greatestFiniteMagnitude / (448 * 6),
                (1 / scale).isFinite, ((448 * 6) / (scale * (448 * 6))).isFinite
            else { throw Failure.invalidExperts }
        }
        self.tensorScales = experts.map(\.tensorScale)
        self.inputScales = inputScales
        self.routeTables = ModelOptNVFP4RouteTables(
            inputScales: inputScales, tensorScales: experts.map(\.tensorScale))
        self._blockScales.wrappedValue = stacked(experts.map(\.blockScales))
        super.init(
            inputDims: first.inputDimensions, outputDims: first.outputDimensions,
            numExperts: experts.count, weight: stacked(experts.map(\.packedWeights)))
    }

    public var logicalShape: [Int] { [numExperts, outputDims, inputDims] }

    public override func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode)
        -> Module
    {
        // Already owns exact producer quantization; generic quantize must not reinterpret it.
        self
    }

    /// Graph mode records routing validity in ``ModelOptGraphValidity/current``
    /// when an owner has bound one; invalid IDs yield NaN rows either way.
    public override func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray,
        sortedIndices: Bool = false
    ) -> MLXArray {
        do {
            switch execution {
            case .eagerReference:
                return try checkedForward(x, indices, sortedIndices: sortedIndices)
            case .graph:
                let output = try graphForward(x, indices, sortedIndices: sortedIndices)
                ModelOptGraphValidity.current?.record(output.validIndices)
                return output.value
            }
        } catch {
            preconditionFailure("Invalid ModelOpt NVFP4 expert call: \(error)")
        }
    }

    /// `relu2(self(x, indices))` (NemotronH's squared ReLU). Graph mode fuses
    /// it into the final rounding kernel, bit-identical to applying it after.
    public func squaredReLU(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        guard execution == .graph else {
            let r = maximum(self(x, indices, sortedIndices: sortedIndices), MLXArray(0))
            return r * r
        }
        do {
            let output = try graphForward(
                x, indices, sortedIndices: sortedIndices, squaredReLU: true)
            ModelOptGraphValidity.current?.record(output.validIndices)
            return output.value
        } catch {
            preconditionFailure("Invalid ModelOpt NVFP4 expert call: \(error)")
        }
    }

    /// Per-call explicit-array allowance; excludes resident expert bank, backend
    /// scratch, other layers and overlapping calls. Graph units are conservative
    /// operation allowances, not an introspected MLX graph-node count.
    public struct LazyDecodeQuote: Sendable {
        public let arrayBytes: Int
        public let graphUnits: Int
    }

    public func lazyDecodeQuote(assignments: Int) throws -> LazyDecodeQuote {
        guard assignments > 0, assignments <= 64 else { throw Failure.workspaceLimit }
        func product(_ a: Int, _ b: Int) throws -> Int {
            let (v, overflow) = a.multipliedReportingOverflow(by: b)
            guard !overflow else { throw Failure.workspaceLimit }
            return v
        }
        func sum(_ a: Int, _ b: Int) throws -> Int {
            let (v, overflow) = a.addingReportingOverflow(b)
            guard !overflow else { throw Failure.workspaceLimit }
            return v
        }
        // Every selected packed matrix and E4M3 block-scale matrix remains live
        // in the lazy graph, including duplicate expert selections. Double the
        // gather allowance for backend contiguous copies; reserve FP32 input,
        // calibration, masks and output intermediates at 32 bytes per element.
        let gathered = try product(outputDims, try sum(inputDims / 2, inputDims / 16))
        let perAssignment = try sum(
            try product(gathered, 2),
            try sum(try product(try sum(inputDims, outputDims), 32), 1024))
        let bytes = try sum(try product(assignments, perAssignment), try product(numExperts, 8))
        return LazyDecodeQuote(arrayBytes: bytes, graphUnits: assignments * 128 + 32)
    }

    /// Opt-in single-token route. Never synchronizes routing values to the host.
    /// Inputs are [A,1,K] or [1,1,K], IDs [A]; invalid IDs yield all-NaN rows
    /// after safe clamping, NOT a thrown content-validation error. Callers must
    /// establish router provenance and check results at their normal fence.
    /// Admission must cover the sum of all outstanding calls plus backend scratch.
    /// Existing checkedForward remains the host-validated general implementation.
    public func lazyDecode(
        _ x: MLXArray, _ indices: MLXArray,
        admittedArrayBytes: Int, admittedGraphUnits: Int
    ) throws -> MLXArray {
        guard indices.ndim == 1, [.int32, .uint32].contains(indices.dtype) else {
            throw Failure.invalidIndices
        }
        let count = indices.size
        let quote = try lazyDecodeQuote(assignments: count)
        guard admittedArrayBytes >= quote.arrayBytes, admittedGraphUnits >= quote.graphUnits else {
            throw Failure.workspaceLimit
        }
        guard x.ndim == 3, x.dim(1) == 1, x.dim(2) == inputDims,
            x.dim(0) == 1 || x.dim(0) == count,
            [.float16, .bfloat16, .float32].contains(x.dtype),
            originalWeightDType == nil || originalWeightDType == x.dtype
        else { throw Failure.invalidInput }
        guard weight.dtype == .uint32, weight.shape == [numExperts, outputDims, inputDims / 8],
            blockScales.dtype == .uint8,
            blockScales.shape == [numExperts, outputDims, inputDims / 16]
        else { throw Failure.invalidExperts }
        // Calibrations originate in validated immutable Swift arrays. Recreate
        // constants rather than caching values from mutable Module parameters.
        let activation = MLXArray(inputScales)
        let global = MLXArray(tensorScales)
        let valid = logicalAnd(indices .>= 0, indices .< numExperts)
        let safe = clip(indices, min: 0, max: numExperts - 1).asType(.int32)
        var outputs: [MLXArray] = []
        for assignment in 0 ..< count {
            let id = safe[assignment]
            let packed = take(weight, id, axis: 0)
            let scales = take(blockScales, id, axis: 0)
            let inputScale = take(activation, id, axis: 0)
            let weightScale = take(global, id, axis: 0)
            let row = x[x.dim(0) == 1 ? 0 : assignment]
            let result = quantizedQuantizedMM(
                row, packed, scales: scales,
                groupSize: 16, bits: 4, mode: .nvfp4,
                globalScaleX: inputScale * (448 * 6), globalScaleW: weightScale * (448 * 6))
            let promoted = row.asType(.float32)
            let overflow = abs(promoted).max(axis: -1, keepDims: true) / (6 * inputScale) .> 464
            let invalid = logicalOr(
                logicalNot(valid[assignment]),
                logicalOr(overflow, isNaN(promoted).any(axis: -1, keepDims: true)))
            outputs.append(which(invalid, MLXArray(Float.nan).asType(result.dtype), result))
        }
        return stacked(outputs)
    }

    public func checkedForward(
        _ x: MLXArray, _ indices: MLXArray,
        sortedIndices: Bool = false
    ) throws -> MLXArray {
        guard weight.dtype == .uint32, weight.shape == [numExperts, outputDims, inputDims / 8],
            blockScales.dtype == .uint8,
            blockScales.shape == [numExperts, outputDims, inputDims / 16]
        else { throw Failure.invalidExperts }
        if let originalWeightDType, x.dtype != originalWeightDType { throw Failure.invalidInput }
        guard x.ndim >= 2, x.dim(-1) == inputDims, x.dim(-2) > 0,
            [.float16, .bfloat16, .float32].contains(x.dtype)
        else { throw Failure.invalidInput }
        guard [.int32, .uint32].contains(indices.dtype), indices.size > 0 else {
            throw Failure.invalidIndices
        }
        let xBatch = Array(x.shape.dropLast(2))
        let rank = max(x.ndim - 2, indices.ndim)
        let lhs = Array(repeating: 1, count: rank - xBatch.count) + xBatch
        let rhs = Array(repeating: 1, count: rank - indices.ndim) + indices.shape
        var shape: [Int] = []
        var assignments = 1
        for (a, b) in zip(lhs, rhs) {
            guard a > 0, b > 0, a == b || a == 1 || b == 1 else { throw Failure.invalidBroadcast }
            let dimension = max(a, b)
            guard dimension <= Self.maximumAssignments,
                assignments <= Self.maximumAssignments / dimension
            else { throw Failure.workspaceLimit }
            assignments *= dimension
            shape.append(dimension)
        }
        let m = x.dim(-2)
        guard m <= Self.maximumAssignments, assignments <= Self.maximumAssignments / m else {
            throw Failure.workspaceLimit
        }
        let rows = assignments * m
        // Expanded input, grouped input, grouped output and restored output,
        // plus FP32 calibration intermediates. Conservative even for BF16.
        let (width, overflow) = inputDims.multipliedReportingOverflow(by: 8)
        let (outWidth, outOverflow) = outputDims.multipliedReportingOverflow(by: 2)
        let (totalWidth, sumOverflow) = width.addingReportingOverflow(outWidth)
        guard !overflow, !outOverflow, !sumOverflow,
            totalWidth <= Self.maximumScratchBytes / 4 / rows
        else { throw Failure.workspaceLimit }
        // Every bound above is checked before the sole host synchronization.
        let selected = broadcast(indices, to: shape).asType(.int32).asArray(Int32.self)
        guard selected.allSatisfy({ $0 >= 0 && Int($0) < numExperts }) else {
            throw Failure.invalidIndices
        }
        var groups = [[Int32]](repeating: [], count: numExperts)
        for (assignment, expert) in selected.enumerated() {
            for row in 0 ..< m { groups[Int(expert)].append(Int32(assignment * m + row)) }
        }
        let expanded = broadcast(x, to: shape + [m, inputDims]).reshaped(rows, inputDims)
        var outputs: [MLXArray] = []
        var order: [Int32] = []
        for expert in groups.indices where !groups[expert].isEmpty {
            let positions = groups[expert]
            let matrix = try ModelOptNVFP4Weights(
                packedBytes: weight[expert].view(dtype: .uint8),
                e4m3ScaleBytes: blockScales[expert], tensorScale: tensorScales[expert])
            outputs.append(
                try matrix.calibratedMatmul(
                    take(expanded, MLXArray(positions), axis: 0),
                    inputScale: inputScales[expert]))
            order += positions
        }
        var inverse = [Int32](repeating: 0, count: rows)
        for (grouped, original) in order.enumerated() { inverse[Int(original)] = Int32(grouped) }
        return take(concatenated(outputs, axis: 0), MLXArray(inverse), axis: 0)
            .reshaped(shape + [m, outputDims])
    }
}
