import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// Regression tests for issue #198: without a cache, the attention mask of
/// Granite MoE hybrid was `.none`, so the full pass was not causal.
///
/// The model is tiny and has seeded random weights: hidden size 32, a Mamba
/// layer and an attention layer, each with 4 experts and top-2 routing. The
/// prompt has 11 tokens. The checks are copied from the "GraniteMoeHybrid"
/// case of the kernel forward-pass tests in PR #185, without the known
/// issue.
///
/// Tolerance 1e-4: the cached pass and the full pass differ in the order of
/// the attention sums, and the Mamba layer runs chunk by chunk. The
/// differences are near 1e-6.
@Suite
struct GraniteMoeHybridCausalMaskTests {

    static let vocabularySize = 64
    static let tolerance: Float = 1e-4

    static func model(seed: UInt64) throws -> GraniteMoeHybridModel {
        let configuration = try TinyModel.configuration(
            GraniteMoeHybridConfiguration.self,
            [
                "vocab_size": vocabularySize, "hidden_size": 32, "intermediate_size": 48,
                "num_hidden_layers": 2, "max_position_embeddings": 256,
                "num_attention_heads": 4, "num_key_value_heads": 2,
                "attention_bias": false, "embedding_multiplier": 1.0,
                "attention_multiplier": 0.125, "logits_scaling": 1.0,
                "residual_multiplier": 1.0, "layer_types": ["mamba", "attention"],
                "rms_norm_eps": 1e-6, "rope_theta": 10000, "num_local_experts": 4,
                "num_experts_per_tok": 2, "shared_intermediate_size": 16,
                "mamba_n_heads": 4, "mamba_d_head": 8, "mamba_proj_bias": false,
                "mamba_d_state": 16, "mamba_d_conv": 4, "mamba_n_groups": 1,
                "mamba_conv_bias": true, "tie_word_embeddings": false,
            ])
        let model = GraniteMoeHybridModel(configuration)
        TinyModel.randomize(model, seed: seed)
        return model
    }

    static func row(_ seed: Int) -> [Int] {
        TinyModel.tokens(count: 11, vocabularySize: vocabularySize, seed: seed)
    }

    @Test func aLaterTokenDoesNotChangeEarlierLogits() throws {
        TinyModel.checkCausality(
            try Self.model(seed: 1), row: Self.row(1), position: 6,
            vocabularySize: Self.vocabularySize, tolerance: Self.tolerance)
    }

    @Test func cachedDecodeMatchesTheFullForwardPass() throws {
        let model = try Self.model(seed: 1)
        for rows in [[Self.row(1)], [Self.row(1), Self.row(2)]] {
            TinyModel.checkCacheConsistency(
                model, rows: rows, chunks: [5, 3, 1, 1, 1], tolerance: Self.tolerance)
        }
    }
}

/// Tiny-model helpers for this file. They are copied from the kernel test
/// support of PR #183 (`Tests/MLXLMTests/Kernel/Support/SyntheticModel.swift`)
/// and are private, so that this file does not depend on that PR.
private enum TinyModel {

    /// Decodes a model configuration from a JSON dictionary.
    static func configuration<C: Decodable>(_ type: C.Type, _ values: [String: Any]) throws -> C {
        try JSONDecoder().decode(C.self, from: JSONSerialization.data(withJSONObject: values))
    }

    /// Replaces every floating-point parameter with seeded random values:
    /// norm scales near 1, other 1-D values near 0, and matrices with a
    /// standard deviation of `1 / sqrt(fan-in)`.
    static func randomize(_ model: Module, seed: UInt64) {
        let parameters = model.parameters().flattened().sorted { $0.0 < $1.0 }
        var updated: [(String, MLXArray)] = []
        for (index, (name, value)) in parameters.enumerated() where value.dtype.isFloatingPoint {
            let noise = MLXRandom.normal(
                value.shape, key: MLXRandom.key(seed &* 1_000_003 &+ UInt64(index)))
            let random: MLXArray
            if value.ndim <= 1 {
                random = name.hasSuffix("weight") ? 1 + 0.1 * noise : 0.1 * noise
            } else {
                let fanIn =
                    name.contains("conv") ? value.shape.dropFirst().reduce(1, *) : value.dim(-1)
                random = noise * (1 / Float(fanIn).squareRoot())
            }
            updated.append((name, random.asType(value.dtype)))
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
        eval(model)
    }

    /// Token IDs from a fixed linear congruential generator.
    static func tokens(count: Int, vocabularySize: Int, seed: Int) -> [Int] {
        var state = UInt64(truncatingIfNeeded: seed) &+ 0x9E37_79B9
        return (0 ..< count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int((state >> 33) % UInt64(vocabularySize))
        }
    }

    /// A `[rows, length]` int32 array of token IDs.
    static func batch(_ rows: [[Int]]) -> MLXArray {
        MLXArray(rows.flatMap { $0.map { Int32($0) } }).reshaped(rows.count, rows[0].count)
    }

    static func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// Runs the model on `rows` and returns the logits.
    static func logits(
        _ model: any LanguageModel, _ rows: [[Int]], cache: [KVCache]? = nil
    ) -> MLXArray {
        let output = model(batch(rows), cache: cache)
        eval(output)
        return output
    }
}

extension TinyModel {

    /// A change to the token at `position` must not change the logits
    /// before `position`, and must change the logits at `position`.
    static func checkCausality(
        _ model: any LanguageModel, row: [Int], position: Int, vocabularySize: Int,
        tolerance: Float, sourceLocation: SourceLocation = #_sourceLocation
    ) {
        var changed = row
        changed[position] = (row[position] + 1) % vocabularySize
        let original = logits(model, [row])
        let modified = logits(model, [changed])
        let before = maxAbsDifference(original[0..., ..<position], modified[0..., ..<position])
        let after = maxAbsDifference(original[0..., position...], modified[0..., position...])
        #expect(
            before <= tolerance, "positions before \(position) changed by \(before)",
            sourceLocation: sourceLocation)
        #expect(
            after > 1e-3, "the change at \(position) must change its own logits",
            sourceLocation: sourceLocation)
    }

    /// The logits of the sequence run in `chunks` with a new cache must
    /// match the logits of one full pass without a cache. A chunk of 1 is a
    /// decode step.
    static func checkCacheConsistency(
        _ model: any LanguageModel, rows: [[Int]], chunks: [Int], tolerance: Float,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let full = logits(model, rows)
        let cache = model.newCache(parameters: nil)
        var start = 0
        for chunk in chunks {
            let part = rows.map { Array($0[start ..< start + chunk]) }
            let difference = maxAbsDifference(
                logits(model, part, cache: cache), full[0..., start ..< start + chunk, 0...])
            #expect(
                difference <= tolerance,
                "positions \(start) ..< \(start + chunk): cached logits differ by \(difference)",
                sourceLocation: sourceLocation)
            start += chunk
        }
    }
}
