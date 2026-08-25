import MLX
import Testing

@testable import MLXLLM

@Suite("Qwen 3.8 fused residual boundary")
struct Qwen38DFlashBoundaryKernelTests {
    @Test("BF16 add and RMSNorm preserve the rounded boundary")
    func numericParity() {
        MLXRandom.seed(38_048)
        let base = MLXRandom.normal([4, 5_120]).asType(.bfloat16)
        let delta = MLXRandom.normal([4, 5_120]).asType(.bfloat16)
        let weight = MLXRandom.uniform(low: 0.8, high: 1.2, [5_120]).asType(.bfloat16)
        let epsilon: Float = 1e-6
        let expectedHidden = base + delta
        let expectedNorm = expectedHidden
            * rsqrt(mean(expectedHidden * expectedHidden, axis: -1, keepDims: true) + epsilon)
            * weight
        let (hidden, normed) = qwen38FusedAddRMSNorm(
            base: base, delta: delta, weight: weight, epsilon: epsilon)
        eval(expectedHidden, expectedNorm, hidden, normed)
        #expect(arrayEqual(hidden, expectedHidden).item(Bool.self))
        #expect(
            abs(normed.asType(.float32) - expectedNorm.asType(.float32))
                .max().item(Float.self) < 0.04)
    }
}
