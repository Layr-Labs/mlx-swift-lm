import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class ZZScratchKernelCostTests: XCTestCase {
    func testKernelVsChain() throws {
        let (Hk, Dk, Hv, Dv, K) = (16, 128, 48, 128, 4)
        let keyDim = Hk * Dk
        let C = 2 * keyDim + Hv * Dv
        MLXRandom.seed(1)
        let w = (MLXRandom.normal([C, K, 1]) * 0.3).asType(.bfloat16)
        let state = MLXRandom.normal([1, K - 1, C]).asType(.bfloat16)
        let x = MLXRandom.normal([1, 1, C]).asType(.bfloat16)
        let norm = Qwen4ExpRMSNormGated(dimensions: Dv, eps: 1e-6, activation: "silu")
        let out = MLXRandom.normal([1, 1, Hv, Dv]).asType(.bfloat16)
        let z = MLXRandom.normal([1, 1, Hv, Dv]).asType(.bfloat16)
        eval(w, state, x, norm, out, z)
        func bench(_ name: String, _ reps: Int, _ body: () -> [MLXArray]) {
            for _ in 0 ..< 5 { eval(body()) }
            var t: [Double] = []
            for _ in 0 ..< 30 {
                let s = DispatchTime.now().uptimeNanoseconds
                var outs: [MLXArray] = []
                for _ in 0 ..< reps { outs += body() }
                eval(outs)
                t.append(Double(DispatchTime.now().uptimeNanoseconds - s) / 1e6)
            }
            t.sort(); print(String(format: "KCOST %-46@ x%d min %.3f ms  (%.1f us per rep)", name, reps, t[0], t[0] / Double(reps) * 1000))
        }
        let invScale = Foundation.pow(Float(Dk), -0.5)
        for reps in [1, 20] {
            bench("chain: concat+conv+silu+split+2norm+2mul", reps) {
                let ci = concatenated([state, x], axis: 1)
                let ns = contiguous(ci[0..., (1 - K)..., 0...])
                let co = silu(MLX.conv1d(ci, w, groups: C))
                let p = MLX.split(co, indices: [keyDim, 2 * keyDim], axis: -1)
                let q = MLXArray(invScale * invScale).asType(.bfloat16) * MLXFast.rmsNorm(p[0].reshaped(1, 1, Hk, Dk), weight: MLXArray.mlxNone, eps: 1e-6)
                let k = MLXArray(invScale).asType(.bfloat16) * MLXFast.rmsNorm(p[1].reshaped(1, 1, Hk, Dk), weight: MLXArray.mlxNone, eps: 1e-6)
                return [q, k, p[2], ns]
            }
            bench("kernels: gdn_conv + gdn_qk_norm", reps) {
                let f = Qwen4ExpGDNScratchKernels.conv(state: state, x: x, weight: w, kernelSize: K, keyDim: keyDim, valueHeads: Hv, valueHeadDim: Dv)
                let n = Qwen4ExpGDNScratchKernels.qkNorm(qk: f.qk, keyHeads: Hk, keyHeadDim: Dk, eps: 1e-6, scales: MLXArray([invScale * invScale, invScale]))
                return [n.q, n.k, f.v, f.newState]
            }
            bench("chain: gated norm (rmsNorm+silu+mul+casts)", reps) { [norm(out, gate: z)] }
            bench("kernel: gdn_gate", reps) { [Qwen4ExpGDNScratchKernels.gate(x: out, gate: z, weight: norm.weight, eps: 1e-6, sigmoid: false)] }
        }
    }
}
