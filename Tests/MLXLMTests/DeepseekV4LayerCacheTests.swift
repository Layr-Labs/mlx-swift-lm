import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// Regression tests for `DeepseekV4LayerCache` (issue #238).
///
/// 1. The cache must not be trimmable. A trim moved back only the rotating
///    window and left the pooled state (closed windows and buffered rows)
///    of the `PoolingCache` objects. mlx-lm deepseek_v41.py returns `False`
///    from `is_trimmable` for this cache.
/// 2. `copy()` must copy each `PoolingCache`. The old copy shared them, so
///    an update of the copy changed the original.
///
/// The tests use the tiny random model of `DeepseekV4TinyModel` (no real
/// weights). A prompt of 6 tokens is shorter than the window of 8, so the
/// rotating caches alone would be trimmable. In the ratio 4 layer it closes
/// one pooled window and leaves 2 buffered rows.
@Suite
struct DeepseekV4LayerCacheTests {

    typealias Tiny = DeepseekV4TinyModel

    /// Runs a 6-token prompt and returns the model and its filled caches.
    static func filledCache() throws -> (DeepseekV4Model, [KVCache]) {
        let model = try Tiny.make()
        let cache = model.newCache(parameters: nil)
        _ = Tiny.logits(model, [Tiny.row(1, count: 6)], cache: cache)
        return (model, cache)
    }

    /// The layer caches report themselves as not trimmable, and
    /// `trimPromptCache` trims nothing.
    @Test func layerCacheIsNotTrimmable() throws {
        let (_, cache) = try Self.filledCache()
        let layer = try #require(cache[1] as? DeepseekV4LayerCache)
        #expect(layer.pooling[0].pooledCount == 1, "pooled windows before the trim")
        #expect(layer.pooling[0].bufKV?.dim(1) == 2, "buffered rows before the trim")

        #expect(!layer.isTrimmable, "ratio 4 layer trimmable")
        #expect(!(cache[2] as! DeepseekV4LayerCache).isTrimmable, "ratio 128 layer trimmable")
        #expect(!canTrimPromptCache(cache), "prompt cache trimmable")
        #expect(trimPromptCache(cache, numTokens: 4) == 0, "trimmed tokens")
        #expect(cache.map(\.offset) == [6, 6, 6], "offsets after the trim")
        #expect(layer.trim(4) == 0, "direct trim")
        #expect(layer.offset == 6, "offset after the direct trim")
    }

    /// A copy has its own pooled caches. A decode step on the copy does not
    /// change the pooled state of the original, and the same step on the
    /// original then gives the same logits as on the copy.
    @Test func copyDoesNotShareThePooledState() throws {
        let (model, cache) = try Self.filledCache()
        let copies = cache.map { $0.copy() }
        let layer = try #require(cache[1] as? DeepseekV4LayerCache)
        let copy = try #require(copies[1] as? DeepseekV4LayerCache)
        #expect(zip(layer.pooling, copy.pooling).allSatisfy { $0 !== $1 }, "shared objects")

        let next = [[Tiny.row(2, count: 1)[0]]]
        let fromCopy = Tiny.logits(model, next, cache: copies)
        #expect(copy.pooling[0].bufKV?.dim(1) == 3, "buffered rows of the copy")
        #expect(layer.pooling[0].bufKV?.dim(1) == 2, "buffered rows of the original")
        #expect(layer.pooling[0].pooledCount == 1, "pooled windows of the original")
        #expect(
            (cache[2] as! DeepseekV4LayerCache).pooling[0].bufKV?.dim(1) == 6,
            "buffered rows of the original ratio 128 layer")

        let fromOriginal = Tiny.logits(model, next, cache: cache)
        #expect(Tiny.maxAbsDifference(fromOriginal, fromCopy) == 0, "logits of the same step")
    }
}
