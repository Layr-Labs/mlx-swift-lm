import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// Regression tests for the causal attention mask of `FalconH1Model`
/// (issue #199).
///
/// The tests use a tiny FalconH1 model with seeded random weights: hidden
/// size 32, 2 layers, 4 query heads over 2 key heads, vocabulary 64. They use
/// no real weights and no network.
///
/// The two tests are copies of the FalconH1 cases of
/// `aLaterTokenDoesNotChangeEarlierLogits` and
/// `cachedDecodeMatchesTheFullForwardPass` in PR #185, without the
/// `withKnownIssue` block. The helpers are small copies of `SyntheticModel`
/// and `ForwardPassChecks` from PR #183. They are private to this suite, so
/// that they do not collide with those types.
@Suite
struct FalconH1CausalMaskTests {

    static let vocabularySize = 64

    // Tolerance of the float32 comparisons: the cached path and the full path
    // differ in the order of the attention sums, and the SSM layers run chunk
    // by chunk. The differences are near 1e-6.
    static let tolerance: Float = 1e-4

    static var configuration: [String: Any] {
        [
            "hidden_size": 32, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "intermediate_size": 48, "num_hidden_layers": 2,
            "mamba_d_ssm": 32, "mamba_n_heads": 4, "mamba_d_head": 8,
            "mamba_d_state": 16, "mamba_n_groups": 1, "mamba_d_conv": 4,
            "vocab_size": vocabularySize,
        ]
    }

    /// A change to the token at position 6 must not change the logits of
    /// positions 0 to 5, and must change the logits at position 6.
    @Test func aLaterTokenDoesNotChangeEarlierLogits() throws {
        let model = try Self.makeModel(seed: 1)
        let position = 6
        let row = Self.tokens(count: 11, seed: 1)
        var changed = row
        changed[position] = (row[position] + 1) % Self.vocabularySize
        let original = Self.logits(model, [row])
        let modified = Self.logits(model, [changed])

        let before = Self.maxAbsDifference(
            original[0..., ..<position], modified[0..., ..<position])
        let after = Self.maxAbsDifference(
            original[0..., position...], modified[0..., position...])
        #expect(before <= Self.tolerance, "positions before \(position) changed by \(before)")
        #expect(after > 1e-3, "the change at \(position) must change its own logits")
    }

    /// A prompt in chunks and decode steps with the model's own cache must
    /// give the logits of the full pass without a cache.
    @Test func cachedDecodeMatchesTheFullForwardPass() throws {
        let model = try Self.makeModel(seed: 1)
        let rowA = Self.tokens(count: 11, seed: 1)
        let rowB = Self.tokens(count: 11, seed: 2)
        for rows in [[rowA], [rowA, rowB]] {
            let full = Self.logits(model, rows)
            let cache = model.newCache(parameters: nil)
            var start = 0
            for chunk in [5, 3, 1, 1, 1] {
                let part = rows.map { Array($0[start ..< start + chunk]) }
                let stepped = Self.logits(model, part, cache: cache)
                let difference = Self.maxAbsDifference(
                    stepped, full[0..., start ..< start + chunk, 0...])
                #expect(
                    difference <= Self.tolerance,
                    "rows \(rows.count), positions \(start) ..< \(start + chunk): cached logits differ by \(difference)"
                )
                start += chunk
            }
        }
    }

    // MARK: - Helpers (copied from PR #183 Kernel/Support/SyntheticModel.swift)

    /// Builds the tiny model and gives each floating-point parameter seeded
    /// random values: a norm scale near 1, another 1-D parameter near 0, and
    /// a matrix with a standard deviation of `1 / sqrt(fan-in)`.
    private static func makeModel(seed: UInt64) throws -> FalconH1Model {
        let data = try JSONSerialization.data(withJSONObject: configuration)
        let model = FalconH1Model(
            try JSONDecoder().decode(FalconH1Configuration.self, from: data))
        let parameters = model.parameters().flattened().sorted { $0.0 < $1.0 }
        var updated: [(String, MLXArray)] = []
        for (index, (name, value)) in parameters.enumerated() {
            guard value.dtype.isFloatingPoint else { continue }
            let key = MLXRandom.key(seed &* 1_000_003 &+ UInt64(index))
            let noise = MLXRandom.normal(value.shape, key: key)
            let random: MLXArray
            if value.ndim <= 1 {
                random = name.hasSuffix("weight") ? 1 + 0.1 * noise : 0.1 * noise
            } else {
                let fanIn =
                    name.contains("conv")
                    ? value.shape.dropFirst().reduce(1, *) : value.dim(-1)
                random = noise * (1 / Float(fanIn).squareRoot())
            }
            let dtype: DType = value.dtype == .float64 ? .float32 : value.dtype
            updated.append((name, random.asType(dtype)))
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
        eval(model)
        return model
    }

    /// Token IDs from a fixed linear congruential generator.
    private static func tokens(count: Int, seed: Int) -> [Int] {
        var state = UInt64(truncatingIfNeeded: seed) &+ 0x9E37_79B9
        return (0 ..< count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int((state >> 33) % UInt64(vocabularySize))
        }
    }

    private static func logits(
        _ model: FalconH1Model, _ rows: [[Int]], cache: [KVCache]? = nil
    ) -> MLXArray {
        let input = MLXArray(rows.flatMap { $0.map { Int32($0) } })
            .reshaped(rows.count, rows[0].count)
        let output = model(input, cache: cache)
        eval(output)
        return output
    }

    private static func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
}
