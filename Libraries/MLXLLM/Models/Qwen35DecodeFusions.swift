// Copyright © 2026 Eigen Labs / Darkbloom.
//
// Compiled elementwise chains for the dense and routed Qwen 3.5 family
// (qwen3_5: the 9B, the 27B, the 35B A3B) — the per-token floor.
//
// Batch-1 decode of these models is bound by the GPU's kernel chain, not by
// reading the weights: one 27B decode step is ~3,200 graph primitives (~2,200
// kernels), and half the weights (one tensor-parallel shard) still takes
// ~85 % of the whole step. Two chains per linear-attention layer are pure
// elementwise sequences that MLX `compile` fuses into one kernel each while
// keeping every op's traced dtype, so the fused result is the unfused chain's
// bits (the same construction as `Qwen4ExpFusions`, which serves qwen4):
//
// - the Gated DeltaNet gates: beta = sigmoid(b) in b's dtype then float32;
//   g = exp(-exp(float32(A_log)) * softplus(a + dt_bias)) — nine kernels;
// - the gated output norm's tail after the fast RMSNorm kernel:
//   (silu(float32(gate)) * float32(normed)) rounded to the activation dtype —
//   five kernels.
//
// On (default) when compiled decode is supported on the hardware; off with
// DARKBLOOM_QWEN35_FUSE_ELEMENTWISE=0. Each chain is the exact expression of
// the unfused path (`computeGatedDeltaG`, `Qwen3NextRMSNormGated`).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum Qwen35DecodeFusions: Sendable {
    static let envFlag = "DARKBLOOM_QWEN35_FUSE_ELEMENTWISE"

    /// Whether an environment enables the fusions (absent: on).
    static func enabled(environment: [String: String], compiledDecodeSupported: Bool) -> Bool {
        guard compiledDecodeSupported else { return false }
        guard let raw = environment[envFlag]?.lowercased() else { return true }
        return !["0", "false", "off", "no"].contains(raw)
    }

    static let isEnabled: Bool = enabled(
        environment: Qwen4ExpEnvironment.snapshot,
        compiledDecodeSupported: MLXHardwareInfo.isCompiledDecodeSupported)

    /// The SwiGLU product and the cached Gated DeltaNet scale constants (on
    /// with the fusions above; diagnostic only:
    /// `DARKBLOOM_QWEN35_DECODE_FUSIONS_V2=0` keeps their unfused forms).
    static var isEnabledV2: Bool { v2Override ?? isEnabledV2ByEnvironment }
    static let isEnabledV2ByEnvironment: Bool =
        isEnabled && Qwen4ExpEnvironment.snapshot["DARKBLOOM_QWEN35_DECODE_FUSIONS_V2"] != "0"
    /// A test or bench's own choice (nil: the environment's); the same values either way.
    @TaskLocal static var v2Override: Bool?

    // MARK: Gated DeltaNet gates (the expressions of the unfused path)

    static func gdnGates(a: MLXArray, b: MLXArray, aLog: MLXArray, dtBias: MLXArray)
        -> (g: MLXArray, beta: MLXArray)
    {
        let out = gdnGatesCompiled([a, b, aLog, dtBias])
        return (out[0], out[1])
    }

    private static let gdnGatesCompiled: @Sendable ([MLXArray]) -> [MLXArray] = compile(
        shapeless: true
    ) { args in
        let beta = sigmoid(args[1]).asType(.float32)
        let g = computeGatedDeltaG(args[2], args[0], args[3])
        return [g, beta]
    }

    // MARK: SwiGLU (the dense MLP's `silu(gate) * up`)

    /// `silu(gate) * up` as one compiled kernel: `silu` keeps its own traced
    /// expression (`x * sigmoid(x)`, rounded to the activation dtype at each
    /// op as the uncompiled chain is), then the product with `up`.
    static func siluMultiply(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
        siluMultiplyCompiled([gate, up])[0]
    }

    private static let siluMultiplyCompiled: @Sendable ([MLXArray]) -> [MLXArray] = compile(
        shapeless: true
    ) { args in
        [args[0] * sigmoid(args[0]) * args[1]]
    }

    /// The attention output gate `x * sigmoid(gate)` (`sigmoidMultiply`) as
    /// one compiled kernel, each op at its traced dtype: the same values.
    static func sigmoidGate(_ x: MLXArray, _ gate: MLXArray) -> MLXArray {
        sigmoidGateCompiled([x, gate])[0]
    }

    private static let sigmoidGateCompiled: @Sendable ([MLXArray]) -> [MLXArray] = compile(
        shapeless: true
    ) { args in
        [args[0] * sigmoid(args[1])]
    }

    /// The Gated DeltaNet's q/k scale constants (`1/d` and `1/sqrt(d)` in the
    /// activation dtype), created once per head width and dtype instead of
    /// per forward: each forward otherwise widened two fresh scalars on the
    /// GPU (96 cast kernels a 27B step). The same values.
    static func qkScales(headDim: Int, dtype: DType) -> (q: MLXArray, k: MLXArray) {
        scaleLock.lock()
        defer { scaleLock.unlock() }
        let key = "\(headDim)-\(dtype)"
        if let cached = scales[key] { return cached }
        let invScale = pow(Float(headDim), -0.5)
        let made = (MLXArray(pow(invScale, 2)).asType(dtype), MLXArray(invScale).asType(dtype))
        eval(made.0, made.1)
        scales[key] = made
        return made
    }

    private static let scaleLock = NSLock()
    nonisolated(unsafe) private static var scales: [String: (q: MLXArray, k: MLXArray)] = [:]

    // MARK: gated RMSNorm tail, silu gate

    /// `(silu(float32(gate)) * float32(normed))` rounded to `dtype`, as
    /// `Qwen3NextRMSNormGated` computes it after `MLXFast.rmsNorm`.
    static func gatedSiluNormTail(normed: MLXArray, gate: MLXArray, dtype: DType) -> MLXArray {
        tail(dtype)([normed, gate])[0]
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var tails: [DType: @Sendable ([MLXArray]) -> [MLXArray]] =
        [:]

    /// One compiled tail per output dtype (the cast is part of the trace).
    private static func tail(_ dtype: DType) -> @Sendable ([MLXArray]) -> [MLXArray] {
        lock.lock()
        defer { lock.unlock() }
        if let cached = tails[dtype] { return cached }
        let made: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { args in
            let g = silu(args[1].asType(.float32))
            return [(g * args[0].asType(.float32)).asType(dtype)]
        }
        tails[dtype] = made
        return made
    }
}
