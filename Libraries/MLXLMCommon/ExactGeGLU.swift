// Copyright © 2026 Eigen Labs / Darkbloom.
//
// `compiledGeGLU` (the approximate-GELU product
// `(0.5 * g * (1 + tanh(sqrt(2 / pi) * (g + 0.044715 * g * g * g)))) * up`)
// as one kernel. MLX's compile splits that expression into two kernels (its
// fusion stops at depth 11), and each op rounds to the activation dtype either
// way, so one kernel that runs the same ops in the same grouping writes the
// same bits. The constants are the Float scalars Swift computes, cast to the
// activation dtype as MLX casts them. `ExactGeGLU.verified` checks every bf16
// or f16 gate value against `compiledGeGLU` once per process before the kernel
// is used.

import Foundation
import MLX

public enum ExactGeGLU {
    /// Diagnostic only: `DARKBLOOM_EXACT_GEGLU=0` keeps `compiledGeGLU`.
    public static let enabledByEnvironment: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_EXACT_GEGLU"] != "0"
    @TaskLocal public static var enabledOverride: Bool?
    public static var isEnabled: Bool { enabledOverride ?? enabledByEnvironment }

    /// The widest call covered (decode-shaped rows: 16 tokens of a dense MLP
    /// or of 8 experts). Prompt chunks keep `compiledGeGLU`.
    public static let maximumElements = 131_072

    /// Checks every gate value of `dtype` against `compiledGeGLU` once per
    /// process (outside any compiled trace); returns whether they matched.
    @discardableResult
    public static func prepare(_ dtype: DType) -> Bool {
        isEnabled && [.bfloat16, .float16].contains(dtype)
            && Verification.shared.verified(dtype)
    }

    /// Whether the kernel covers `gate` and `up`: same shape and dtype, at
    /// most `maximumElements`, and `prepare` already matched this dtype. Never
    /// measures, so it is safe inside a compiled trace.
    public static func supports(_ gate: MLXArray, _ up: MLXArray) -> Bool {
        guard isEnabled, gate.shape == up.shape, gate.dtype == up.dtype,
            gate.size <= maximumElements
        else { return false }
        return Verification.shared.known(gate.dtype) == true
    }

    /// `compiledGeGLU(gate, up)`'s values; call only where `supports` holds.
    public static func apply(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
        let constants = Constants.shared.values
        return kernel(
            [gate, up, constants],
            template: [("T", gate.dtype)],
            grid: (gate.size, 1, 1), threadGroup: (min(256, max(1, gate.size)), 1, 1),
            outputShapes: [gate.shape], outputDTypes: [gate.dtype])[0]
    }

    /// `[0.5, 1, sqrt(2 / pi), 0.044715]` as the Float scalars Swift computes.
    final class Constants: @unchecked Sendable {
        static let shared = Constants()
        let values: MLXArray = {
            let v = MLXArray([Float(0.5), Float(1), sqrt(2 / Float.pi), Float(0.044715)])
            eval(v)
            return v
        }()
    }

    final class Verification: @unchecked Sendable {
        static let shared = Verification()
        private let lock = NSLock()
        private var results: [DType: Bool] = [:]

        func known(_ dtype: DType) -> Bool? { lock.withLock { results[dtype] } }

        func verified(_ dtype: DType) -> Bool {
            if let known = lock.withLock({ results[dtype] }) { return known }
            let ok = Self.check(dtype)
            lock.withLock { results[dtype] = ok }
            return ok
        }

        /// Every 16-bit pattern as the gate, against several `up` rows.
        static func check(_ dtype: DType) -> Bool {
            guard Device.defaultDevice().deviceType == .gpu,
                ResidualRMSNorm.verifiedFamilies.contains(AffineQuantizedVectors.family())
            else { return false }
            // Every 16-bit pattern, built through Float (exact for every
            // finite value and the infinities).
            let values: [Float] = (0 ..< 65_536).map { bits in
                dtype == .bfloat16
                    ? Float(bitPattern: UInt32(bits) << 16)
                    : Float(Float16(bitPattern: UInt16(bits)))
            }
            let gate = MLXArray(values).asType(dtype)
            var state: UInt64 = 0xA076_1D64_78BD_642F
            for _ in 0 ..< 3 {
                let ups: [Float] = (0 ..< 65_536).map { _ in
                    state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                    return Float(Int((state >> 33) % 4097) - 2048) / 256
                }
                let up = MLXArray(ups).asType(dtype)
                let expected = compiledGeGLU(gate, up)
                let actual = apply(gate, up)
                eval(expected, actual)
                if expected.asData(access: .copy).data != actual.asData(access: .copy).data {
                    return false
                }
            }
            return true
        }
    }

    // compiledGeGLU's ops in its grouping, each rounded to T.
    private static let kernel = MLXFast.metalKernel(
        name: "exact_geglu", inputNames: ["gate", "up", "c"], outputNames: ["out"],
        source: """
            const uint i = thread_position_in_grid.x;
            const T c_half = static_cast<T>(c[0]);
            const T c_one = static_cast<T>(c[1]);
            const T c_k = static_cast<T>(c[2]);
            const T c_cubic = static_cast<T>(c[3]);
            const T x = gate[i];
            const T t1 = c_cubic * x;
            const T t2 = t1 * x;
            const T t3 = t2 * x;
            const T s = x + t3;
            const T u = c_k * s;
            const T th = static_cast<T>(metal::precise::tanh(static_cast<float>(u)));
            const T w = c_one + th;
            const T y = c_half * x;
            const T z = y * w;
            out[i] = z * up[i];
            """)
}
