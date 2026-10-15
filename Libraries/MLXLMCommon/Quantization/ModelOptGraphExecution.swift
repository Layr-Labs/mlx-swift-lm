import Foundation
import MLX

/// Selects how a ModelOpt module executes `callAsFunction`.
public enum ModelOptExecution: Sendable {
    /// Device-resident graph: no host synchronization, the default.
    case graph
    /// The bounded eager baseline (`checkedForward`), kept for reference tests.
    case eagerReference
}

/// Accumulation shape of graph products with many rows.
public enum ModelOptAccumulation: Sendable {
    /// One matrix-vector product per row, bit-identical to the eager reference.
    case rowwise
    /// MLX matrix kernels for large row counts. Producer bits and scales are
    /// unchanged; only the float32 summation order differs from the reference.
    case tiled
}

/// Product used by graph FP8 projections.
public enum ModelOptFP8Product: Sendable {
    /// MXFP8 kernels with unit block scales read the E4M3 bytes directly and
    /// sum exact weights in float32; the per-tensor scale is applied once.
    case packed
    /// Bit-identical to `checkedForward`: per call, each scaled weight is
    /// rounded to the weight dtype in the eager reference's output-row tiles.
    case exactReference
}

/// Graph result whose routing validity the caller checks at its own fence.
public struct ModelOptGraphOutput {
    public let value: MLXArray
    /// Boolean scalar: every expert ID was in range. Rows with invalid IDs are
    /// NaN in `value`; evaluate and check this before consuming `value`.
    public let validIndices: MLXArray
}

/// Collects device validity evidence while a forward graph is built, for one
/// owner to check at its completion fence. Graph code never reads it on the host.
public final class ModelOptGraphValidity: @unchecked Sendable {
    public enum Failure: Error, Equatable { case invalidExpertIndices }

    @TaskLocal public static var current: ModelOptGraphValidity?

    private let lock = NSLock()
    private var combined: MLXArray?

    public init() {}

    public func record(_ flag: MLXArray) {
        lock.withLock { combined = combined.map { logicalAnd($0, flag) } ?? flag }
    }

    /// Include in the owner's evaluation of the forward outputs.
    public var evaluationRoots: [MLXArray] { lock.withLock { combined.map { [$0] } ?? [] } }

    /// Host read at the owner's fence, after the forward has been evaluated.
    public func check() throws {
        guard let flag = lock.withLock({ combined }) else { return }
        guard flag.item(Bool.self) else { throw Failure.invalidExpertIndices }
    }
}
