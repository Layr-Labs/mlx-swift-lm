import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// Regression tests for the hyper connection of DeepSeek V4 (issue #194,
/// defect 5).
///
/// `hcPre` has two paths: the fused Sinkhorn Metal kernel on the GPU, and an
/// ops path on the CPU. Both must compute `pre = sigmoid(pre) + hc_eps` with no
/// normalization, as mlx-lm `deepseek_v41.py:241` does, and so give the same
/// collapsed output on the same input.
///
/// The test `hyperConnectionKernelMatchesTheOpsPath` is copied from
/// `DeepseekV4ForwardPassTests` of PR #184, without its `withKnownIssue`
/// block, and runs on the hyper connections of two layers.
@Suite
struct DeepseekV4HyperConnectionTests {

    typealias Tiny = DeepseekV4TinyModel

    /// The fused Sinkhorn kernel of the hyper connection must give the same
    /// result as the ops path that `hcPre` uses on the CPU.
    @Test(arguments: [(0, "attn"), (2, "ffn")])
    func hyperConnectionKernelMatchesTheOpsPath(layer: Int, sublayer: String) throws {
        let model = try Tiny.make()
        let block = model.model.layers[layer]
        let params = sublayer == "attn" ? block.attn_hc : block.ffn_hc
        let x = MLXRandom.normal([2, 3, 4, 32], key: MLXRandom.key(9))
        eval(x)
        let (gpuY, gpuPost, gpuComb) = hcPre(
            x: x, hcFn: params.fn, hcScale: params.scale, hcBase: params.base, hcMult: 4,
            sinkhornIters: 3, eps: 1e-6)
        eval(gpuY, gpuPost, gpuComb)
        let (cpuY, cpuPost, cpuComb) = Device.withDefaultDevice(.cpu) {
            let result = hcPre(
                x: x, hcFn: params.fn, hcScale: params.scale, hcBase: params.base,
                hcMult: 4, sinkhornIters: 3, eps: 1e-6)
            eval(result.0, result.1, result.2)
            return result
        }
        // The Metal kernel and the CPU ops sum in a different order. `post`
        // and `comb` are small values after sigmoid and Sinkhorn: 1e-5. The
        // collapsed output sums larger values over the streams: 1e-4.
        #expect(Tiny.maxAbsDifference(gpuPost, cpuPost) <= 1e-5, "post")
        #expect(Tiny.maxAbsDifference(gpuComb, cpuComb) <= 1e-5, "comb")
        #expect(Tiny.maxAbsDifference(gpuY, cpuY) <= 1e-4, "collapsed")
    }
}
