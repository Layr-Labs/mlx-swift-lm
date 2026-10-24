import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {
    /// The small-batch NAX projection against MLX's quantized matmul. The GPU
    /// cases need Neural Accelerators and are skipped elsewhere.
    @Suite(.serialized)
    struct Qwen35SmallBatchLinearTests {
        private func layer(inputs: Int = 2048, outputs: Int = 96) -> QuantizedLinear {
            MLXRandom.seed(7)
            let weight = (MLXRandom.normal([outputs, inputs]) * 0.02).asType(.bfloat16)
            let stock = QuantizedLinear(weight: weight, bias: nil, groupSize: 64, bits: 4)
            eval(stock)
            return stock
        }

        /// Weights round to BF16 before the multiply, as in MLX's matrix kernels.
        @Test(.enabled(if: Qwen35SmallBatchLinear.available))
        func matchesQuantizedMatmulFromFourToSixteenRows() throws {
            let stock = layer()
            let fast = try #require(Qwen35SmallBatchLinear.make(from: stock))
            for shape in [[4, 2048], [8, 2048], [13, 2048], [16, 2048], [2, 4, 2048]] {
                let x = MLXRandom.normal(shape).asType(.bfloat16)
                let y = fast(x)
                #expect(y.shape == shape.dropLast() + [96])
                #expect(allClose(y, stock(x), rtol: 2e-2, atol: 2e-2).item(Bool.self))
            }
        }

        @Test(.enabled(if: Qwen35SmallBatchLinear.available))
        func otherRowCountsAndShapesKeepMLXsKernel() throws {
            let stock = layer()
            let fast = try #require(Qwen35SmallBatchLinear.make(from: stock))
            for rows in [1, 3, 17] {
                let x = MLXRandom.normal([rows, 2048]).asType(.bfloat16)
                #expect(arrayEqual(fast(x), stock(x)).item(Bool.self))
            }
            // K must be a multiple of 1,024 and N of 32.
            let narrow = layer(inputs: 1536, outputs: 48)
            let fastNarrow = try #require(Qwen35SmallBatchLinear.make(from: narrow))
            let x = MLXRandom.normal([8, 1536]).asType(.bfloat16)
            #expect(arrayEqual(fastNarrow(x), narrow(x)).item(Bool.self))
        }

        @Test func installKeepsTheGatedDeltaInputProjections() {
            final class Attention: Module {
                @ModuleInfo(key: "in_proj_qkv") var inProjQKV = Linear(2048, 96, bias: false)
            }
            final class Layer: Module {
                @ModuleInfo(key: "linear_attn") var linearAttn = Attention()
                @ModuleInfo(key: "down_proj") var downProj = Linear(2048, 96, bias: false)
            }
            let model = Layer()
            quantize(model: model, groupSize: 64, bits: 4)
            Qwen35SmallBatchLinear.install(in: model)
            #expect(type(of: model.linearAttn.inProjQKV) == QuantizedLinear.self)
            #expect((model.downProj is Qwen35SmallBatchLinear) == Qwen35SmallBatchLinear.available)
        }
    }
}
