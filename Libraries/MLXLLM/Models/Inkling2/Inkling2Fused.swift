// Compiled elementwise chains for Inkling 2. Each closure traces the same ops,
// dtypes and order as the plain path in Inkling2Math and the conv modules, so
// results are unchanged; compilation only stops materializing intermediates
// and merges their launches.
import Foundation
import MLX
import MLXLMCommon

enum Inkling2Fused {
    /// `DARKBLOOM_INKLING2_FUSED=0` restores the plain op-by-op graphs.
    static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_FUSED"]?.lowercased()
        return raw.map { !["0", "false", "off", "no"].contains($0) } ?? true
    }()

    /// `DARKBLOOM_INKLING2_CONCAT_PROJECTIONS=0` keeps the separate attention
    /// projections at `prepare()` (no concatenated row-exact buffers).
    static let concatenatedProjections: Bool = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_INKLING2_CONCAT_PROJECTIONS"]?
            .lowercased()
        return raw.map { !["0", "false", "off", "no"].contains($0) } ?? true
    }()

    /// Test hook: overrides `concatenatedProjections` at run time for the
    /// current task (the buffers themselves are built by `prepare()`).
    @TaskLocal static var concatenationOverride: Bool? = nil
    /// Whether prepared concatenated products are used. Independent of
    /// `active`: concatenation changes which products run, compilation only
    /// how elementwise chains are launched.
    static var concatenationActive: Bool { concatenationOverride ?? concatenatedProjections }

    /// Test hook: overrides `enabled` for the current task.
    @TaskLocal static var override: Bool? = nil
    static var active: Bool { override ?? enabled }

    typealias Function = @Sendable ([MLXArray]) -> [MLXArray]

    private final class Cache: @unchecked Sendable {
        let lock = NSLock()
        var functions: [String: Function] = [:]
        func function(_ key: String, _ make: () -> Function) -> Function {
            lock.withLock {
                if let f = functions[key] { return f }
                let f = make()
                functions[key] = f
                return f
            }
        }
    }
    private static let cache = Cache()

    static func call(
        _ key: String, shapeless: Bool, _ inputs: [MLXArray], _ body: @escaping Function
    )
        -> [MLXArray]
    {
        cache.function(key) { MLX.compile(shapeless: shapeless, body) }(inputs)
    }

    static func unweightedRMSNorm(_ x: MLXArray, eps: Float) -> MLXArray {
        call("unorm:\(eps)", shapeless: true, [x]) { a in
            [Inkling2Math.plainUnweightedRMSNorm(a[0], eps: eps)]
        }[0]
    }

    static func weightedRMSNorm(_ x: MLXArray, weight: MLXArray, eps: Float) -> MLXArray {
        call("wnorm:\(eps)", shapeless: true, [x, weight]) { a in
            [a[1].asType(.float32) * Inkling2Math.plainUnweightedRMSNorm(a[0], eps: eps)]
        }[0]
    }

    static func l2Clip(_ x: MLXArray, threshold: Float) -> MLXArray {
        call("clip:\(threshold)", shapeless: true, [x]) { a in
            [Inkling2Math.plainL2Clip(a[0], threshold: threshold)]
        }[0]
    }

    /// `clip(x.f32 * scale.f32).asType(x.dtype)`.
    static func scaledClip(_ x: MLXArray, scale: MLXArray, threshold: Float) -> MLXArray {
        call("sclip:\(threshold):\(x.dtype)", shapeless: true, [x, scale]) { a in
            [
                Inkling2Math.plainL2Clip(
                    a[0].asType(.float32) * a[1].asType(.float32), threshold: threshold
                ).asType(a[0].dtype)
            ]
        }[0]
    }

    /// Static short convolution: `(y, next)` for `previous` `[B, k-1, C]`,
    /// `x` `[B, L, C]` and FP32 `taps` `[C, k]`.
    static func shortConvolution(_ previous: MLXArray, _ x: MLXArray, taps: MLXArray)
        -> (MLXArray, MLXArray)
    {
        let r = call("sconv:\(x.dtype)", shapeless: false, [previous, x, taps]) { a in
            let (previous, x, taps) = (a[0], a[1], a[2])
            let (l, kernel) = (x.dim(1), taps.dim(1))
            let context = concatenated(
                [previous.asType(.float32), x.asType(.float32)], axis: 1)
            var out = context[0..., 0 ..< l, 0...] * taps[0..., 0]
            for j in 1 ..< kernel {
                out = out + context[0..., j ..< (j + l), 0...] * taps[0..., j]
            }
            let y = (out + x.asType(.float32)) / sqrt(Float(kernel + 1))
            let next = concatenated([previous, x], axis: 1)[0..., l ..< (l + kernel - 1), 0...]
            return [y.asType(x.dtype), next]
        }
        return (r[0], r[1])
    }

    /// Dynamic short convolution tail: `taps` `[B, L, H, C/H, k]` FP32.
    static func dynamicConvolution(_ previous: MLXArray, _ x: MLXArray, taps: MLXArray)
        -> (MLXArray, MLXArray)
    {
        let r = call("dconv:\(x.dtype)", shapeless: false, [previous, x, taps]) { a in
            let (previous, x, taps) = (a[0], a[1], a[2])
            let (b, l, c) = (x.dim(0), x.dim(1), x.dim(2))
            let kernel = taps.dim(-1)
            let context = concatenated(
                [previous.asType(.float32), x.asType(.float32)], axis: 1)
            var out: MLXArray? = nil
            for lag in 0 ..< kernel {
                let start = kernel - 1 - lag
                let window = context[0..., start ..< (start + l), 0...]
                let term = window * taps[.ellipsis, lag].reshaped(b, l, c)
                out = out.map { $0 + term } ?? term
            }
            let y = (out! + x.asType(.float32)) / sqrt(Float(2 * kernel + 1))
            let next = concatenated([previous, x], axis: 1)[0..., l ..< (l + kernel - 1), 0...]
            return [y.asType(x.dtype), next]
        }
        return (r[0], r[1])
    }

    /// `(1 - tanh(f)) * h + tanh(g) * m`, FP32.
    static func messageGateTail(_ h: MLXArray, _ m: MLXArray, f: MLXArray, g: MLXArray)
        -> MLXArray
    {
        call("mgate", shapeless: true, [h, m, f, g]) { a in
            [(1 - tanh(a[2])) * a[0] + tanh(a[3]) * a[1]]
        }[0]
    }

    /// `(attended.f32 * sigmoid(gate.f32)).asType(attended.dtype)`.
    static func outputGate(_ attended: MLXArray, gate: MLXArray) -> MLXArray {
        call("ogate:\(attended.dtype)", shapeless: true, [attended, gate]) { a in
            [(a[0].asType(.float32) * sigmoid(a[1].asType(.float32))).asType(a[0].dtype)]
        }[0]
    }

    /// Softcapped SwiGLU on interleaved gate/up rows, cast to `dtype`.
    static func expertActivation(
        _ gateUp: MLXArray, gateCap: Float, upCap: Float, dtype: DType
    ) -> MLXArray {
        call("act:\(gateCap):\(upCap):\(dtype)", shapeless: false, [gateUp]) { a in
            let x = a[0].asType(.float32)
            let gate = x[.ellipsis, .stride(from: 0, by: 2)]
            let up = x[.ellipsis, .stride(from: 1, by: 2)]
            return [
                (Inkling2Math.softcap(gate, gateCap) * sigmoid(gate)
                    * Inkling2Math.softcap(up, upCap)).asType(dtype)
            ]
        }[0]
    }

    /// Block attention residual over `blocks ++ [partial]` (see `plainAttentionResidual`).
    static func attentionResidual(
        queries: MLXArray, blocks: [MLXArray], partial: MLXArray
    ) -> MLXArray {
        let count = blocks.count
        return call("mix:\(count)", shapeless: false, [queries] + blocks + [partial]) { a in
            [
                plainAttentionResidual(
                    queries: a[0], blocks: Array(a[1 ..< (1 + count)]), partial: a[1 + count])
            ]
        }[0]
    }

    /// FP32 weighted sum over the expert axis, cast to `dtype`.
    static func expertCombine(_ perExpert: MLXArray, weights: MLXArray, dtype: DType) -> MLXArray {
        call("combine:\(dtype)", shapeless: true, [perExpert, weights]) { a in
            [
                (a[0].asType(.float32) * a[1].expandedDimensions(axis: -1)).sum(axis: -2).asType(
                    dtype)
            ]
        }[0]
    }
}
