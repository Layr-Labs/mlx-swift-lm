import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// Regression tests for the RoPE of InternLM2.
///
/// Issue #202: the dynamic NTK RoPE took the sequence length from `x.dim(1)`.
/// The attention gives it queries and keys as `[B, heads, L, D]`, so
/// `x.dim(1)` is the head count, and a long prompt never got the scaled base.
///
/// Issue #232: the attention never read `factor` for type `dynamic`, it used
/// the dynamic NTK RoPE for every `rope_scaling` type, and the RoPE passed its
/// scale to `MLXFast.RoPE`. The reference (mlx-lm rope_utils.py
/// `initialize_rope` and `DynamicNTKScalingRoPE`) uses the dynamic NTK RoPE
/// only for type `dynamic`, uses the factor only in the base, and always
/// passes scale 1.0.
///
/// The first test is copied from `internLM2DynamicRopeScalesByTheSequenceLength`
/// in the kernel forward-pass tests of PR #185, without the known issue.
@Suite
struct Internlm2DynamicRoPETests {

    /// Dynamic NTK scaling raises the RoPE base when the sequence is longer
    /// than `max_position_embeddings`.
    @Test func dynamicRopeScalesByTheSequenceLength() {
        let rope = Internlm2DynamicNTKScalingRoPE(
            dims: 8, maxPositionEmbeddings: 8, base: 10000, factor: 2)
        // 4 heads, 12 positions.
        let x = MLXRandom.normal([1, 4, 12, 8], key: MLXRandom.key(2))
        // seq_len 12 > 8: base * (2 * 12 / 8 - 1) ^ (8 / 6)
        let base = 10000 * pow(Float(2 * 12) / 8 - 1, Float(8) / 6)
        // The reference passes scale 1.0 and uses the factor only in the base.
        let expected = MLXFast.RoPE(
            x, dimensions: 8, traditional: false, base: base, scale: 1, offset: 0)
        let output = rope(x, offset: 0)
        // Tolerance 1e-5: both sides call the same kernel with the same base,
        // so only float32 rounding of the base can differ.
        #expect(maxAbsDifference(output, expected) <= 1e-5, "scaled base")
    }

    /// A sequence not longer than `max_position_embeddings` keeps the base,
    /// also when the head count is larger than `max_position_embeddings`.
    @Test func shortSequenceKeepsTheBase() {
        let rope = Internlm2DynamicNTKScalingRoPE(
            dims: 8, maxPositionEmbeddings: 8, base: 10000, factor: 2)
        // 12 heads, 4 positions.
        let x = MLXRandom.normal([1, 12, 4, 8], key: MLXRandom.key(3))
        let expected = MLXFast.RoPE(
            x, dimensions: 8, traditional: false, base: 10000, scale: 1, offset: 0)
        let output = rope(x, offset: 0)
        // Tolerance 1e-5: both sides call the same kernel with the same base.
        #expect(maxAbsDifference(output, expected) <= 1e-5, "original base")
    }

    /// `rope_scaling` type `dynamic` with factor 2: the attention reads the
    /// factor, uses it in the base, and rotates with scale 1.
    @Test func dynamicScalingReadsTheFactor() throws {
        let rope = try attentionRope(ropeScaling: #"{"type": "dynamic", "factor": 2.0}"#)
        let x = MLXRandom.normal([1, 4, 12, 8], key: MLXRandom.key(4))
        let base = 10000 * pow(Float(2 * 12) / 8 - 1, Float(8) / 6)
        let expected = MLXFast.RoPE(
            x, dimensions: 8, traditional: false, base: base, scale: 1, offset: 0)
        let output = applyRotaryPosition(rope, to: x, cache: nil)
        #expect(maxAbsDifference(output, expected) <= 1e-5, "dynamic factor 2")
    }

    /// `rope_scaling` type `linear` with factor 2: a plain RoPE with scale
    /// 1 / 2. The base does not change for a long sequence.
    @Test func linearScalingUsesAPlainRope() throws {
        let rope = try attentionRope(ropeScaling: #"{"type": "linear", "factor": 2.0}"#)
        let x = MLXRandom.normal([1, 4, 12, 8], key: MLXRandom.key(5))
        let expected = MLXFast.RoPE(
            x, dimensions: 8, traditional: false, base: 10000, scale: 0.5, offset: 0)
        let output = applyRotaryPosition(rope, to: x, cache: nil)
        #expect(maxAbsDifference(output, expected) <= 1e-5, "linear factor 2")
    }

    /// No `rope_scaling`: a plain RoPE with scale 1. The base does not change
    /// for a sequence longer than `max_position_embeddings`.
    @Test func noScalingUsesAPlainRope() throws {
        let rope = try attentionRope(ropeScaling: nil)
        let x = MLXRandom.normal([1, 4, 12, 8], key: MLXRandom.key(6))
        let expected = MLXFast.RoPE(
            x, dimensions: 8, traditional: false, base: 10000, scale: 1, offset: 0)
        let output = applyRotaryPosition(rope, to: x, cache: nil)
        #expect(maxAbsDifference(output, expected) <= 1e-5, "no scaling")
    }

    /// The RoPE of an attention layer with head dimension 8 and
    /// `max_position_embeddings` 8.
    private func attentionRope(ropeScaling: String?) throws -> RoPELayer {
        let scaling = ropeScaling.map { #", "rope_scaling": \#($0)"# } ?? ""
        let json = """
            {"hidden_size": 32, "num_hidden_layers": 1, "intermediate_size": 48,
             "num_attention_heads": 4, "num_key_value_heads": 4, "rms_norm_eps": 1e-6,
             "vocab_size": 64, "max_position_embeddings": 8, "rope_theta": 10000\(scaling)}
            """
        let args = try JSONDecoder().decode(
            InternLM2Configuration.self, from: Data(json.utf8))
        return Internlm2Attention(args).rope
    }

    private func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }
}
