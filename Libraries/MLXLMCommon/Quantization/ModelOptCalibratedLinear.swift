import MLX
import MLXNN

/// Native calibrated projection primitive, deliberately not a `Linear` subclass
/// or generic `Quantized` layer. ModelOpt per-tensor FP8 is not MXFP8. Existing
/// Linear-typed projection slots need explicit adaptation before installation.
/// No dense initialization, bias, automatic conversion or checkpoint registration.
/// Calibration tensors remain registered for strict Module parameter updates.
/// `callAsFunction` runs the graph path (`graphForward`) by default; its scalar
/// calibration is host-validated at init and after every parameter update.
public final class ModelOptCalibratedLinear: Module, UnaryLayer {
    public enum Format: Sendable { case nvfp4, fp8 }
    public enum Failure: Error { case invalidMetadata, invalidInput, workspaceLimit }
    public let format: Format
    public let weightDType: DType
    public let inputDimensions: Int
    public let outputDimensions: Int
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "weight_scale") var weightScale: MLXArray
    @ModuleInfo(key: "weight_scale_2") var weightScale2: MLXArray?
    @ModuleInfo(key: "input_scale") var inputScale: MLXArray
    public var execution: ModelOptExecution = .graph
    public var accumulation: ModelOptAccumulation = .rowwise
    /// Selecting `.packed` allocates the shared unit-scale plane here, at
    /// selection time, never inside a forward graph.
    public var fp8Product: ModelOptFP8Product = .exactReference {
        didSet {
            if format == .fp8, oldValue != fp8Product { graphConstants = try? makeGraphConstants() }
        }
    }
    /// Nil after a failed validation: the graph path then refuses every call.
    var graphConstants: ModelOptCalibratedGraphConstants?

    /// Host result of the last calibration validation (init or parameter
    /// update). When false, `graphForward` throws and `callAsFunction` traps,
    /// as the eager path does; no device read or probe graph is needed.
    public var graphCalibrationValid: Bool { graphConstants != nil }

    public init(weights: ModelOptNVFP4Weights, inputScale: Float, weightDType: DType) throws {
        try Self.validate(inputScale, dtype: weightDType, format: .nvfp4)
        format = .nvfp4
        self.weightDType = weightDType
        inputDimensions = weights.inputDimensions
        outputDimensions = weights.outputDimensions
        _weight.wrappedValue = weights.packedWeights
        _weightScale.wrappedValue = weights.blockScales
        _weightScale2.wrappedValue = MLXArray(weights.tensorScale)
        _inputScale.wrappedValue = MLXArray(inputScale)
        super.init()
        graphConstants = try makeGraphConstants()
    }

    public init(weights: ModelOptFP8Weights, inputScale: Float, weightDType: DType) throws {
        try Self.validate(inputScale, dtype: weightDType, format: .fp8)
        format = .fp8
        self.weightDType = weightDType
        inputDimensions = weights.bytes.dim(1)
        outputDimensions = weights.bytes.dim(0)
        _weight.wrappedValue = weights.bytes
        _weightScale.wrappedValue = MLXArray(weights.tensorScale)
        _weightScale2.wrappedValue = nil
        _inputScale.wrappedValue = MLXArray(inputScale)
        super.init()
        graphConstants = try makeGraphConstants()
    }

    /// Parameter updates are the only way registered calibration changes, so
    /// the eager per-call host check moves here, outside every forward graph.
    /// Invalid calibration is still accepted, as before; the graph path then
    /// refuses every call, as the eager path does.
    @discardableResult
    public override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        graphConstants = nil
        try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
        graphConstants = try? makeGraphConstants()
        return self
    }

    static func validate(_ scale: Float, dtype: DType, format: Format) throws {
        let bound: Float = format == .nvfp4 ? 448 * 6 : 448
        guard [.float16, .bfloat16, .float32].contains(dtype), scale.isFinite, scale > 0,
            (scale * bound).isFinite, scale * bound > 0,
            (bound / (scale * bound)).isFinite, bound / (scale * bound) > 0,
            (1 / scale).isFinite
        else { throw Failure.invalidMetadata }
    }

    func scalar(_ value: MLXArray) throws -> Float {
        guard value.dtype == .float32, value.shape.isEmpty || value.shape == [1] else {
            throw Failure.invalidMetadata
        }
        return value.item(Float.self)
    }

    /// Scalar calibration validation synchronizes to host. The 64 MiB estimate
    /// bounds explicit intermediates, not backend scratch/concurrent calls.
    /// FP8 output-row tiles execute eagerly; callers must not compile this method.
    /// Mixed activation/original-weight dtypes are deliberately rejected.
    public func checkedForward(_ input: MLXArray) throws -> MLXArray {
        guard input.ndim > 0, input.dim(-1) == inputDimensions,
            input.dtype == weightDType
        else { throw Failure.invalidInput }
        var rows = 1
        for dimension in input.shape.dropLast() {
            guard dimension > 0, dimension <= 8192, rows <= 8192 / dimension else {
                throw Failure.workspaceLimit
            }
            rows *= dimension
        }
        // Reserve live input/output first. FP8 weights are decoded one output-row
        // tile at a time; NVFP4 remains packed. The factor32 conservatively covers
        // FP32 calibration temporaries, not just the BF16 result.
        let limit = (64 << 20) / 32
        var remaining = limit
        let sizes = [(rows, inputDimensions), (rows, outputDimensions)]
        for (a, b) in sizes {
            let (count, overflow) = a.multipliedReportingOverflow(by: b)
            guard !overflow, count <= remaining else { throw Failure.workspaceLimit }
            remaining -= count
        }
        let result: MLXArray
        switch format {
        case .nvfp4:
            guard weight.dtype == .uint32,
                weight.shape == [outputDimensions, inputDimensions / 8],
                weightScale.dtype == .uint8,
                weightScale.shape == [outputDimensions, inputDimensions / 16],
                let global = weightScale2
            else { throw Failure.invalidMetadata }
            let scale = try scalar(inputScale)
            try Self.validate(scale, dtype: weightDType, format: format)
            let matrix = try ModelOptNVFP4Weights(
                packedBytes: weight.view(dtype: .uint8),
                e4m3ScaleBytes: weightScale, tensorScale: scalar(global))
            result = try matrix.calibratedMatmul(
                input.reshaped(rows, inputDimensions), inputScale: scale)
        case .fp8:
            guard weight.dtype == .uint8, weight.shape == [outputDimensions, inputDimensions],
                weightScale2 == nil
            else { throw Failure.invalidMetadata }
            let scale = try scalar(inputScale)
            try Self.validate(scale, dtype: weightDType, format: format)
            let matrix = try ModelOptFP8Weights(e4m3Bytes: weight, tensorScale: scalar(weightScale))
            let tileRows = min(outputDimensions, remaining / inputDimensions)
            guard tileRows > 0 else { throw Failure.workspaceLimit }
            var pieces: [MLXArray] = []
            for start in stride(from: 0, to: outputDimensions, by: tileRows) {
                let end = min(outputDimensions, start + tileRows)
                let piece = try matrix.calibratedMatmul(
                    input.reshaped(rows, inputDimensions),
                    inputScale: scale, weightDType: weightDType, outputRows: start ..< end)
                // Evaluate before the next tile to retire its decoded weight and
                // calibration graph. Output pieces stay live and were budgeted
                // above. This is a serial eager baseline, never a compiled graph.
                eval(piece)
                pieces.append(piece)
            }
            result = pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: -1)
        }
        return result.reshaped(Array(input.shape.dropLast()) + [outputDimensions])
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        do {
            switch execution {
            case .graph: return try graphForward(input)
            case .eagerReference: return try checkedForward(input)
            }
        } catch {
            preconditionFailure("Invalid calibrated ModelOpt projection: \(error)")
        }
    }
}
