import Foundation
import MLX

/// Shared, immutable device arrays for small host-scalar kernel arguments
/// (norm epsilons, clip limits, scaling factors). An array built from host
/// values allocates and fills a buffer on every call (about 3 µs on an M3
/// Ultra), and a decode step passes such scalars to hundreds of fused kernels.
/// Entries are keyed by the values' bit patterns, so a cached array holds
/// exactly the bits a fresh `MLXArray(values)` would. The arrays are never
/// written and never donated (the cache keeps a reference).
public enum HostConstantArrays {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var float32Arrays: [[UInt32]: MLXArray] = [:]

    /// `MLXArray(values)`: a float32 vector of `values`.
    public static func float32(_ values: [Float]) -> MLXArray {
        let key = values.map(\.bitPattern)
        lock.lock()
        defer { lock.unlock() }
        if let cached = float32Arrays[key] { return cached }
        let array = MLXArray(values)
        // A bounded set: models pass a handful of distinct scalars.
        if float32Arrays.count >= 256 { float32Arrays.removeAll() }
        float32Arrays[key] = array
        return array
    }

    /// `MLXArray(true)`: a boolean scalar.
    nonisolated(unsafe) public static let trueScalar = MLXArray(true)

    /// `MLXArray([Float(1)])`.
    nonisolated(unsafe) public static let floatOne = MLXArray([Float(1)])
}
