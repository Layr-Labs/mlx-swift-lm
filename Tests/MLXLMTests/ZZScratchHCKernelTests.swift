import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class ZZScratchHCKernelTests: XCTestCase {
    func testHCMixerKernelsVsChain() throws {
        let (hc, D, lowrank) = (4, 2560, 320)
        MLXRandom.seed(2)
        var json = Qwen4ExpFixture.configurationJSON
        json = json.replacingOccurrences(of: "\"hidden_size\": 32", with: "\"hidden_size\": \(D)")
            .replacingOccurrences(of: "\"hc_count\": 2", with: "\"hc_count\": \(hc)")
            .replacingOccurrences(of: "\"hc_lowrank\": 8", with: "\"hc_lowrank\": \(lowrank)")
        let config = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self, from: Data(json.utf8))
        let mixer = Qwen4ExpGatedResidual(config, useInject: true)
        let replaced = mixer.parameters().flattened().map { ($0.0, (MLXRandom.normal($0.1.shape) * 0.3).asType(.bfloat16)) }
        mixer.update(parameters: ModuleParameters.unflattened(replaced)); eval(mixer)
        quantize(model: mixer) { _, m in (m as? Linear).map { $0.weight.dim(-1) % 64 == 0 ? (64, 4, .affine) : nil } ?? nil }
        eval(mixer)
        let hyper = MLXRandom.normal([1, 1, hc * D]).asType(.bfloat16)
        let blockOut = MLXRandom.normal([1, 1, D]).asType(.bfloat16)
        eval(hyper, blockOut)
        let inv = 1 / Float(hc)
        func chain() -> MLXArray {
            let (input, residual, inject) = mixer.mixWithInject(hyper)
            _ = input
            return qwen4ExpInject(residual: residual, output: blockOut, inject: inject)
        }
        func kernels() -> MLXArray {
            let normed = Qwen4ExpHCScratchKernels.norm(hyper, weight: mixer.hcNorm.weight, hc: hc, dims: D, eps: config.rmsNormEps, offsetZero: mixer.hcNorm.weightOffset == 0)
            let w = Qwen4ExpHCScratchKernels.scaledSilu(mixer.mixDown(normed), scale: inv)
            let mixed = Qwen4ExpHCScratchKernels.mix(mixer.mixUp(w), normed: normed, hc: hc, dims: D)
            _ = mixed
            return Qwen4ExpHCScratchKernels.inject(residual: hyper, output: blockOut, logit: mixer.blockInject!(normed), scale: inv, hc: hc, dims: D)
        }
        // exactness (activation precision)
        let a = chain(), b = kernels(); eval(a, b)
        let delta = MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
        let scale = MLX.abs(a.asType(.float32)).max().item(Float.self)
        print(String(format: "HCK max|delta| %.4f on |max| %.3f", delta, scale))
        func bench(_ name: String, _ reps: Int, _ body: () -> MLXArray) {
            for _ in 0 ..< 5 { eval(body()) }
            var t: [Double] = []
            for _ in 0 ..< 30 {
                let s = DispatchTime.now().uptimeNanoseconds
                var outs: [MLXArray] = []
                for _ in 0 ..< reps { outs.append(body()) }
                eval(outs)
                t.append(Double(DispatchTime.now().uptimeNanoseconds - s) / 1e6)
            }
            t.sort(); print(String(format: "HCK %-34@ x%d min %.3f ms (%.1f us per rep)", name, reps, t[0], t[0] / Double(reps) * 1000))
        }
        for reps in [1, 20] {
            bench("chain: mixWithInject + inject", reps, chain)
            bench("kernels: norm+silu+mix+inject", reps, kernels)
        }
    }
}
