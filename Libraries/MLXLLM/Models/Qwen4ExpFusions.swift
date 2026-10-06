// Copyright © 2026 Eigen Labs / Darkbloom Assignment 1.
//
// Compiled elementwise chains for Flash-Next (qwen4_exp) decode.
//
// Serial decode on this model is bound by MLX graph-node count (~7,800 nodes
// per token, ~157 Metal command buffers), not by kernel speed: the host
// spends ~24 ms encoding and the GPU is idle a quarter of the time. Every
// chain here is a sequence of elementwise ops (plus, for the L2 QK norm, one
// reduction) that MLX `compile` fuses into a single kernel while preserving
// each op's traced dtype, so the fused result is bit-identical to the
// unfused chain. Shapeless traces serve T=1 decode and every Lightning verify
// width from the same kernel; `hyperInjection` reshapes to the residual
// shape and is traced per shape.
//
// Kill switch: DARKBLOOM_QWEN4_FUSE_ELEMENTWISE=0 (also off when compiled
// decode is unsupported on the hardware).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum Qwen4ExpFusions: Sendable {
    static let envFlag = "DARKBLOOM_QWEN4_FUSE_ELEMENTWISE"

    static let isEnabled: Bool = {
        guard MLXHardwareInfo.isCompiledDecodeSupported else { return false }
        guard let raw = Qwen4ExpEnvironment.snapshot[envFlag]?.lowercased() else { return true }
        return !["0", "false", "off", "no"].contains(raw)
    }()

    // MARK: GDN gates: beta = sigmoid(b) (f32); g = exp(-exp(A_log) * softplus(a + dt_bias))

    static func gdnGates(a: MLXArray, b: MLXArray, aLog: MLXArray, dtBias: MLXArray)
        -> (g: MLXArray, beta: MLXArray)
    {
        let out = gdnGatesCompiled([a, b, aLog, dtBias])
        return (out[0], out[1])
    }

    private static let gdnGatesCompiled: @Sendable ([MLXArray]) -> [MLXArray] = compile(
        shapeless: true
    ) { args in
        let a = args[0]
        let b = args[1]
        let aLog = args[2]
        let dtBias = args[3]
        let beta = sigmoid(b).asType(.float32)
        let g = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
        return [g, beta]
    }

    // MARK: qwen4 L2 QK normalisation

    static func l2QK(q: MLXArray, k: MLXArray, scale: MLXArray) -> (MLXArray, MLXArray) {
        let out = l2QKCompiled([q, k, scale])
        return (out[0], out[1])
    }

    private static let l2QKCompiled: @Sendable ([MLXArray]) -> [MLXArray] = compile(
        shapeless: true
    ) { args in
        let q = args[0]
        let k = args[1]
        let scale = args[2]
        let qNormed = q * rsqrt(q.square().sum(axis: -1, keepDims: true) + 1e-6) * scale
        let kNormed = k * rsqrt(k.square().sum(axis: -1, keepDims: true) + 1e-6)
        return [qNormed, kNormed]
    }

    // MARK: gated RMSNorm tail: (sigmoid(gate) * normed) in f32, back to normed dtype

    static let gatedNormFinish: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
        shapeless: true
    ) { normed, gate in
        (sigmoid(gate.asType(.float32)) * normed.asType(.float32)).asType(normed.dtype)
    }

    // MARK: silu after the depthwise conv

    static let silu: @Sendable (MLXArray) -> MLXArray = compile(shapeless: true) { x in
        MLXNN.silu(x)
    }

    // MARK: hyper-connection injection: keep(hyper + branch[..., None, :] * injection[..., None])

    static func hyperInjection(branch: MLXArray, hyper: MLXArray, injection: MLXArray) -> MLXArray {
        hyperInjectionCompiled([branch, hyper, injection])[0]
    }

    private static let hyperInjectionCompiled: @Sendable ([MLXArray]) -> [MLXArray] = compile(
        shapeless: false
    ) { args in
        let branch = args[0]
        let hyper = args[1]
        let injection = args[2]
        let weighted = branch.expandedDimensions(axis: -2) * injection.expandedDimensions(axis: -1)
        return [Qwen4ExpActivation.keep(hyper + weighted.reshaped(hyper.shape))]
    }

    // MARK: MoE shared-expert combine: combined + sigmoid(gate) * sharedY

    static let sharedExpertCombine: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = compile(
        shapeless: true
    ) { combined, sharedGate, sharedY in
        combined + sigmoid(sharedGate) * sharedY
    }
}
