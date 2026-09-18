// Copyright © 2026 Eigen Labs.
import Cmlx
import MLX

/// One-forward, one-consumer carry with descriptor identity, not just Swift
/// wrapper identity. Capture hooks and Module.update may replace a weight's
/// descriptor without replacing its wrapper. No cross-call cache or global state.
public final class Gemma4DecodeNormalizationCarry {
    private struct Pending {
        let source: MLXArray
        let weight: MLXArray
        let normalized: MLXArray
        let sourceID: UInt
        let weightID: UInt
        let eps: Float
        let stream: StreamOrDevice
    }
    private var pending: Pending?

    private init(_ pending: Pending) { self.pending = pending }

    private static func identity(_ x: MLXArray) -> UInt? {
        var id: UInt = 0
        var allowed = false
        guard _mlx_array_constant_cache_identity(&id, &allowed, x.ctx) == 0, allowed else { return nil }
        return id
    }
    private static func snapshot(_ x: MLXArray) -> MLXArray? {
        var context = mlx_array_new()
        guard mlx_array_set(&context, x.ctx) == 0 else { mlx_array_free(context); return nil }
        return MLXArray(context)
    }

    public static func capture(source: MLXArray, normalized: MLXArray, weight: MLXArray, eps: Float) -> Self? {
        guard source.shape == normalized.shape, source.dtype == normalized.dtype,
            let source = snapshot(source), let weight = snapshot(weight),
            let sourceID = identity(source), let weightID = identity(weight) else { return nil }
        return Self(Pending(source: source, weight: weight, normalized: normalized,
            sourceID: sourceID, weightID: weightID, eps: eps, stream: .default))
    }

    public func take(source: MLXArray, weight: MLXArray, eps: Float) -> MLXArray? {
        let value = pending
        pending = nil
        guard let value, value.eps == eps, value.stream == StreamOrDevice.default,
            Self.identity(source) == value.sourceID, Self.identity(weight) == value.weightID,
            source.shape == value.normalized.shape, source.dtype == value.normalized.dtype else { return nil }
        return value.normalized
    }
}
