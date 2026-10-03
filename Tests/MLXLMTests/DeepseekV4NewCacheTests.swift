import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// Regression tests for the caches of DeepSeek V4 (issue #194, defect 3).
///
/// The generation code calls `newCache(parameters:)`. It must give the caches
/// of `makeCache(parameters:)`: a rotating window for each layer, and the
/// pooled caches for the compressed attention layers.
///
/// The test `newCacheGivesTheCompressedCaches` is copied from
/// `DeepseekV4ForwardPassTests` of PR #184, without its `withKnownIssue`
/// block.
@Suite
struct DeepseekV4NewCacheTests {

    typealias Tiny = DeepseekV4TinyModel

    /// `newCache(parameters:)` is what the generation code calls. It must give
    /// the caches that the attention layers need.
    @Test func newCacheGivesTheCompressedCaches() throws {
        let model = try Tiny.make()
        let cache = model.newCache(parameters: nil)
        #expect(cache[0] is RotatingKVCache, "layer 0 cache")
        #expect(cache[1] is DeepseekV4LayerCache, "layer 1 cache")
        #expect(cache[2] is DeepseekV4LayerCache, "layer 2 cache")
    }

    /// Each cache has the sliding window of the configuration (8), and each
    /// compressed layer has the pooled caches for its compress ratio: 2 for
    /// ratio 4 (the attention and the indexer), 1 for ratio 128.
    @Test func newCacheHasTheWindowAndThePooledCaches() throws {
        let model = try Tiny.make()
        let cache = model.newCache(parameters: GenerateParameters())
        #expect(cache.count == 3)
        #expect(cache[0].maxSize == 8, "layer 0 window")
        let sparse = try #require(cache[1] as? DeepseekV4LayerCache)
        #expect(sparse.maxSize == 8, "layer 1 window")
        #expect(sparse.pooling.map(\.ratio) == [4, 4], "layer 1 pooled caches")
        let compressed = try #require(cache[2] as? DeepseekV4LayerCache)
        #expect(compressed.maxSize == 8, "layer 2 window")
        #expect(compressed.pooling.map(\.ratio) == [128], "layer 2 pooled caches")
    }

    /// A prompt in chunks and decode steps with the caches of
    /// `newCache(parameters:)` give the same logits as with the caches of
    /// `makeCache(parameters:)`.
    @Test func newCacheGivesTheLogitsOfMakeCache() throws {
        let model = try Tiny.make()
        let row = Tiny.row(1, count: 136)
        let chunks = [64, 66, 1, 1, 1, 1, 1, 1]
        let fromNew = model.newCache(parameters: nil)
        let fromMake = model.makeCache(parameters: GenerateParameters())
        var start = 0
        for chunk in chunks {
            let part = [Array(row[start ..< start + chunk])]
            let difference = Tiny.maxAbsDifference(
                Tiny.logits(model, part, cache: fromNew),
                Tiny.logits(model, part, cache: fromMake))
            #expect(difference == 0, "positions \(start) ..< \(start + chunk): \(difference)")
            start += chunk
        }
    }
}
