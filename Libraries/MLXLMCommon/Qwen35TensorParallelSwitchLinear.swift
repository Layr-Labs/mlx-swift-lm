import MLX
import MLXNN

/// A row-parallel projection of a channel-sharded Qwen MoE layer whose
/// unrounded FP32 partial can travel in the routed experts' collective
/// (`Qwen35TensorParallelSwitchLinear.carrying`): the shared expert's down
/// projection. The partial must be exactly the one the projection's own
/// forward would sum across ranks.
public protocol Qwen35TensorParallelPackedPartial: AnyObject {
    /// This rank's unrounded FP32 partial of `x`, without any collective.
    func qwen35TensorParallelPartial(_ x: MLXArray) -> MLXArray
}

/// A channel-sharded Qwen expert down projection. The collective receives
/// unrounded FP32 expert outputs, before the owning SwitchGLU restores route
/// order and applies router weights. Shared-expert gating stays downstream.
public final class Qwen35TensorParallelSwitchLinear: QuantizedSwitchLinear {
    private final class Reduction {
        let sum: (MLXArray) -> MLXArray
        init(_ sum: @escaping (MLXArray) -> MLXArray) { self.sum = sum }
    }

    /// Rows another projection packs into one call's collective, and their
    /// sum once that collective is built. Scoped to one `carrying` call and
    /// touched only by the thread that bound it, while that call runs.
    private final class Packing: @unchecked Sendable {
        let owner: ObjectIdentifier
        let rows: MLXArray
        var sum: MLXArray?
        init(owner: ObjectIdentifier, rows: MLXArray) {
            self.owner = owner
            self.rows = rows
        }
    }

    @TaskLocal private static var packing: Packing?
    /// The projection whose call inside `withoutCollective` returns its
    /// partial unsummed. Scoped to that call, as `packing` is.
    @TaskLocal private static var deferred: ObjectIdentifier?

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

    /// Runs `body`, the owning SwitchGLU forward, which calls this projection
    /// exactly once, with `rows` (another projection's FP32 partial
    /// `[..., outputDims]`) packed after that call's expert rows in its one
    /// collective. Returns `body`'s result and the FP32 cross-rank sum of
    /// `rows`. The collective adds every element across the ranks on its own
    /// (two ranks: one commutative add), so packing leaves each expert value
    /// and each packed value bit-identical to two separate collectives.
    public func carrying(
        _ rows: MLXArray, through body: () -> MLXArray
    ) -> (output: MLXArray, rowsSum: MLXArray) {
        precondition(
            rows.dtype == .float32 && rows.ndim >= 1 && rows.dim(-1) == shape.outputDims,
            "Packed rows are FP32 partials of the expert output width")
        let packing = Packing(owner: ObjectIdentifier(self), rows: rows)
        let output = Self.$packing.withValue(packing, operation: body)
        guard let sum = packing.sum else {
            preconditionFailure(
                "The routed expert down projection did not run; packed rows were never summed")
        }
        return (output, sum)
    }

    /// Runs `body`, the owning SwitchGLU forward, which calls this projection
    /// exactly once, without that call's collective: the projection returns
    /// its unrounded FP32 partial (`[..., outputDims]`), for an owner that
    /// sums several independent rows' partials in one collective (`sum`)
    /// and rounds each value as that call's own collective would.
    public func withoutCollective(_ body: () -> MLXArray) -> MLXArray {
        Self.$deferred.withValue(ObjectIdentifier(self), operation: body)
    }

    /// This projection's collective over FP32 partial rows
    /// (`withoutCollective`): each element's cross-rank sum, unrounded.
    public func sum(_ rows: MLXArray) -> MLXArray {
        precondition(
            rows.dtype == .float32 && rows.dim(-1) == shape.outputDims,
            "Summed rows are FP32 partials of the expert output width")
        return reduction.sum(rows)
    }

    public override func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let partial = super.callAsFunction(
            x.asType(.float32), indices,
            sortedIndices: sortedIndices)
        if Self.deferred == ObjectIdentifier(self) { return partial }
        guard let packing = Self.packing, packing.owner == ObjectIdentifier(self) else {
            return reduction.sum(partial).asType(x.dtype)
        }
        precondition(packing.sum == nil, "One collective carries the packed rows")
        let width = partial.dim(-1)
        let routed = partial.size / width
        let summed = reduction.sum(
            concatenated(
                [partial.reshaped([routed, width]), packing.rows.reshaped([-1, width])], axis: 0))
        packing.sum = summed[routed...].reshaped(packing.rows.shape)
        return summed[..<routed].reshaped(partial.shape).asType(x.dtype)
    }
}
