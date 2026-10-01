import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// Regression tests for issue #202: the dynamic NTK RoPE of InternLM2 took
/// the sequence length from `x.dim(1)`. The attention gives it queries and
/// keys as `[B, heads, L, D]`, so `x.dim(1)` is the head count, and a long
/// prompt never got the scaled base.
///
/// The first test is copied from `internLM2DynamicRopeScalesByTheSequenceLength`
/// in the kernel forward-pass tests of PR #185, without the known issue.
@Suite
struct Internlm2DynamicRoPETests {

    /// Dynamic NTK scaling raises the RoPE base when the sequence is longer
    /// than `max_position_embeddings`.
    @Test func dynamicRopeScalesByTheSequenceLength() {
        let rope = Internlm2DynamicNTKScalingRoPE(
            dims: 8, maxPositionEmbeddings: 8, base: 10000, scale: 2)
        // 4 heads, 12 positions.
        let x = MLXRandom.normal([1, 4, 12, 8], key: MLXRandom.key(2))
        // seq_len 12 > 8: base * (2 * 12 / 8 - 1) ^ (8 / 6)
        let base = 10000 * pow(Float(2 * 12) / 8 - 1, Float(8) / 6)
        // `scale: 2` pins the current behavior: the Swift class passes its
        // `scale` to `MLXFast.RoPE`. The reference passes 1.0 and uses the
        // factor only in the base. See #232.
        let expected = MLXFast.RoPE(
            x, dimensions: 8, traditional: false, base: base, scale: 2, offset: 0)
        let output = rope(x, offset: 0)
        // Tolerance 1e-5: both sides call the same kernel with the same base,
        // so only float32 rounding of the base can differ.
        #expect(maxAbsDifference(output, expected) <= 1e-5, "scaled base")
    }

    /// A sequence not longer than `max_position_embeddings` keeps the base,
    /// also when the head count is larger than `max_position_embeddings`.
    @Test func shortSequenceKeepsTheBase() {
        let rope = Internlm2DynamicNTKScalingRoPE(
            dims: 8, maxPositionEmbeddings: 8, base: 10000, scale: 2)
        // 12 heads, 4 positions.
        let x = MLXRandom.normal([1, 12, 4, 8], key: MLXRandom.key(3))
        // `scale: 2` pins the current behavior; the reference passes 1.0.
        // See #232.
        let expected = MLXFast.RoPE(
            x, dimensions: 8, traditional: false, base: 10000, scale: 2, offset: 0)
        let output = rope(x, offset: 0)
        // Tolerance 1e-5: both sides call the same kernel with the same base.
        #expect(maxAbsDifference(output, expected) <= 1e-5, "original base")
    }

    private func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }
}
