// Copyright © 2026 Eigen Labs.
//
// Qwen4Exp gated-deltanet: the decay/beta gates as one compiled chain.
//
// WHY ONLY THIS. Lever (2) of the decode-speed program tried three custom
// Metal kernels here (concat+conv+silu+split, both q/k norms, the gated
// output norm). Measured in release on a real-shaped model they did not
// pay: an MLXFast.metalKernel launch costs ~0.13–0.2 ms of CPU on this
// MLX (the custom-kernel primitive is built and hashed per call), which is
// more than the built-in ops it replaced — the fused conv path ran at
// ~0.47 ms against ~0.22 ms for the unfused chain, and the gate kernel tied
// its chain. A custom kernel only pays when it replaces MANY heavy ops,
// which the recurrence kernel does and these did not. The kernels were
// removed; what remains is the elementwise gate chain as a `compile`d
// function (one fused kernel for exp/softplus/sigmoid), which does pay at
// ~6 µs per elementwise node removed. MLXLM_GDN_FUSION=0 keeps the unfused
// chain in the same binary for A/B.

import Foundation
import MLX
import MLXNN

/// A/B switch, read once.
let qwen4ExpGDNFusionEnabled: Bool =
    ProcessInfo.processInfo.environment["MLXLM_GDN_FUSION"] != "0"

enum Qwen4ExpGDNKernels {
    /// `[decay, beta]` from `(a, b, A_log, dt_bias)`, one fused kernel:
    /// decay = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b),
    /// both float32 as `gatedDeltaUpdate` expects.
    static let gates: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) {
        (inputs: [MLXArray]) -> [MLXArray] in
        let a = inputs[0]
        let b = inputs[1]
        let aLog = inputs[2]
        let dtBias = inputs[3]
        let decay = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
        return [decay, sigmoid(b).asType(.float32)]
    }
}
