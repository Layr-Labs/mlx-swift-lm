import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// Regression tests for issue #255: Exaone 4 made one mask from the first
/// layer cache, without the sliding window, for all layers.
///
/// The model is tiny and has seeded random weights: hidden size 32, one
/// local ("L") layer with a window of 4 and one global ("G") layer, 4 query
/// heads of size 8 over 2 key heads, vocabulary 64. The prompt has 11
/// tokens, longer than the window. The check is copied from
/// `exaone4SlidingLayersIgnoreTheWindowWithoutACache` of the kernel
/// forward-pass tests in PR #235, without the known issue. It also runs
/// with the local layer first ("LG"), where the global layer got the mask
/// of the rotating cache.
///
/// Tolerance 1e-4: the cached pass and the full pass differ in the order of
/// the attention sums. The differences are near 1e-6.
@Suite
struct Exaone4SlidingMaskTests {

    static let vocabularySize = 64
    static let tolerance: Float = 1e-4

    static func model(pattern: String, seed: UInt64) throws -> Exaone4Model {
        let configuration = try TinyModel.configuration(
            Exaone4Configuration.self,
            [
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                "rms_norm_eps": 1e-6, "vocab_size": vocabularySize,
                "max_position_embeddings": 256, "rope_theta": 10000,
                "tie_word_embeddings": true,
            ], overrides: ["sliding_window": 4, "sliding_window_pattern": pattern])
        let model = Exaone4Model(configuration)
        TinyModel.randomize(model, seed: seed)
        return model
    }

    /// Each local layer gets a `RotatingKVCache` with the window, and each
    /// global layer a `KVCacheSimple`. The full pass must apply the same
    /// window as the cache, for batch 1 and batch 2.
    @Test(arguments: ["GL", "LG"]) func slidingLayersApplyTheWindowWithoutACache(pattern: String)
        throws
    {
        let model = try Self.model(pattern: pattern, seed: 1)
        let cache = model.newCache(parameters: nil)
        let local = try #require(Array(pattern).firstIndex(of: "L"))
        #expect(cache.count == 2)
        #expect(cache[1 - local] is KVCacheSimple)
        #expect((cache[local] as? RotatingKVCache)?.maxSize == 4)

        let rows = [1, 2].map {
            TinyModel.tokens(count: 11, vocabularySize: Self.vocabularySize, seed: $0)
        }
        TinyModel.checkCacheConsistency(
            model, rows: [rows[0]], chunks: [5, 3, 1, 1, 1], tolerance: Self.tolerance,
            cache: cache)
        TinyModel.checkCacheConsistency(
            model, rows: rows, chunks: [5, 3, 1, 1, 1], tolerance: Self.tolerance)
    }
}

/// Tiny-model helpers for this file. They are copied from the kernel test
/// support of PR #183 (`Tests/MLXLMTests/Kernel/Support/SyntheticModel.swift`)
/// and are private, so that this file does not depend on that PR.
private enum TinyModel {

    /// Decodes a model configuration from a JSON dictionary. A key in
    /// `overrides` replaces the same key in `base`.
    static func configuration<C: Decodable>(
        _ type: C.Type, _ base: [String: Any], overrides: [String: Any] = [:]
    ) throws -> C {
        let merged = base.merging(overrides) { _, new in new }
        return try JSONDecoder().decode(
            C.self, from: JSONSerialization.data(withJSONObject: merged))
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

    /// The logits of the sequence run in `chunks` with `cache`, or with a
    /// new cache from `model.newCache(parameters: nil)`, must match the
    /// logits of one full pass without a cache. A chunk of 1 is a decode
    /// step.
    static func checkCacheConsistency(
        _ model: any LanguageModel, rows: [[Int]], chunks: [Int], tolerance: Float,
        cache: [KVCache]? = nil, sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let full = logits(model, rows)
        let cache = cache ?? model.newCache(parameters: nil)
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
