import MLX
import MLXNN
import Testing

@testable import MLXLLM

@Suite("Qwen 3.8 fused QK RMSNorm and RoPE")
struct Qwen38DFlashQKKernelTests {
    @Test("BF16 H256 R64 fused outputs match composed operations")
    func numericParity() {
        MLXRandom.seed(38_021)
        let q = MLXRandom.normal([1, 4, 24, 256]).asType(.bfloat16)
        let k = MLXRandom.normal([1, 4, 4, 256]).asType(.bfloat16)
        let qw = MLXRandom.uniform(low: 0.8, high: 1.2, [256]).asType(.bfloat16)
        let kw = MLXRandom.uniform(low: 0.8, high: 1.2, [256]).asType(.bfloat16)
        let eps: Float = 1e-6
        let offset = 37
        let rope = RoPE(dimensions: 64, traditional: false, base: 10_000_000)

        func expected(_ x: MLXArray, weight: MLXArray) -> MLXArray {
            let normalized = x * rsqrt(mean(x * x, axis: -1, keepDims: true) + eps) * weight
            return rope(normalized.transposed(0, 2, 1, 3), offset: offset)
        }
        let expectedQ = expected(q, weight: qw)
        let expectedK = expected(k, weight: kw)
        let (actualQ, actualK) = qwen38FusedQKRMSRoPE(
            queries: q, keys: k, queryWeight: qw, keyWeight: kw,
            epsilon: eps, offset: offset)
        eval(expectedQ, expectedK, actualQ, actualK)
        #expect(actualQ.shape == expectedQ.shape)
        #expect(actualK.shape == expectedK.shape)
        #expect(
            abs(actualQ.asType(.float32) - expectedQ.asType(.float32)).max().item(Float.self) < 0.04
        )
        #expect(
            abs(actualK.asType(.float32) - expectedK.asType(.float32)).max().item(Float.self) < 0.04
        )
    }
}
