import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Regression tests for issue #239.
///
/// 1. A saved and loaded `ChunkedKVCache` must have the same `offset` and
///    start position as the saved cache. The old code took the offset from
///    the buffer length, which is smaller than the offset after
///    `maybeTrimFront()`.
/// 2. `MambaCache.extract(_:)` must keep every slot in its position, also
///    the empty ones. The old code went through `state`, which drops the
///    empty slots.
///
/// The saved format does not change: two arrays (keys, values) and the meta
/// state `[chunkSize, startPosition]`. `oldFormatFileLoadsWithTheRightOffset`
/// writes such a file by hand, as the old code wrote it, and loads it.
@Suite
struct PromptCacheChunkedMambaTests {

    static let heads = 2
    static let headDim = 8

    /// Random keys and values for `count` new tokens.
    static func tokens(_ count: Int, seed: UInt64) -> (keys: MLXArray, values: MLXArray) {
        let shape = [1, heads, count, headDim]
        let keys = MLXRandom.normal(shape, key: MLXRandom.key(seed * 3))
        let values = MLXRandom.normal(shape, key: MLXRandom.key(seed * 3 + 1))
        eval(keys, values)
        return (keys, values)
    }

    static func equal(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.shape == b.shape && (a .== b).all().item(Bool.self)
    }

    static func tempURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("prompt-cache-\(UUID().uuidString)")
            .appendingPathExtension("safetensors")
    }

    static func saveAndLoad(_ cache: KVCache) throws -> KVCache {
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [cache])
        let (loaded, _) = try loadPromptCache(url: url)
        #expect(loaded.count == 1, "loaded cache count")
        return loaded[0]
    }

    /// The loaded cache must continue like the original: the same next
    /// offset, keys and values.
    static func expectSameNextUpdate(_ cache: KVCache, _ loaded: KVCache, seed: UInt64) {
        let next = tokens(1, seed: seed)
        let (keysA, valuesA) = cache.update(keys: next.keys, values: next.values)
        let (keysB, valuesB) = loaded.update(keys: next.keys, values: next.values)
        #expect(loaded.offset == cache.offset, "next offset")
        #expect(equal(keysB, keysA), "next keys")
        #expect(equal(valuesB, valuesA), "next values")
    }

    /// 10 tokens, chunk size 8, step 4, `maybeTrimFront()`: the cache has
    /// offset 10 and start position 2. The loaded cache must have the same.
    @Test func chunkedCacheLoadKeepsTheOffsetAfterTrimFront() throws {
        let cache = ChunkedKVCache(chunkSize: 8)
        cache.step = 4
        let first = Self.tokens(10, seed: 1)
        _ = cache.update(keys: first.keys, values: first.values)
        cache.maybeTrimFront()
        #expect(cache.offset == 10 && cache.metaState == ["8", "2"], "cache before save")

        let loaded = try Self.saveAndLoad(cache)
        #expect(loaded is ChunkedKVCache, "loaded type")
        #expect(loaded.offset == 10, "loaded offset")
        #expect(loaded.metaState == ["8", "2"], "loaded meta state")
        Self.expectSameNextUpdate(cache, loaded, seed: 2)
    }

    /// One more token after the trim: the buffer now has unused rows. The
    /// saved state must hold only the 9 valid tokens, and the loaded cache
    /// must have offset 11.
    @Test func chunkedCacheSavesOnlyTheValidTokens() throws {
        let cache = ChunkedKVCache(chunkSize: 8)
        cache.step = 4
        let first = Self.tokens(10, seed: 3)
        _ = cache.update(keys: first.keys, values: first.values)
        cache.maybeTrimFront()
        let second = Self.tokens(1, seed: 4)
        _ = cache.update(keys: second.keys, values: second.values)
        let expectedKeys = concatenated(
            [first.keys[.ellipsis, 2 ..< 10, 0...], second.keys], axis: 2)
        #expect(Self.equal(cache.state[0], expectedKeys), "state keys")

        let loaded = try Self.saveAndLoad(cache)
        #expect(loaded.offset == 11, "loaded offset")
        #expect(loaded.metaState == ["8", "2"], "loaded meta state")
        #expect(Self.equal(loaded.state[0], expectedKeys), "loaded keys")
        Self.expectSameNextUpdate(cache, loaded, seed: 5)
    }

    /// A file in the format that the old code wrote: arrays `0.0` and `0.1`
    /// with the 8 kept tokens, meta state `["8", "2"]` and class
    /// `ChunkedKVCache`. The load must give offset 10 (2 + 8).
    @Test func oldFormatFileLoadsWithTheRightOffset() throws {
        let kept = Self.tokens(8, seed: 6)
        let url = Self.tempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try MLX.save(
            arrays: ["0.0": kept.keys, "0.1": kept.values],
            metadata: ["0.0.0": "8", "0.0.1": "2", "2.0": "ChunkedKVCache"],
            url: url)

        let (loaded, _) = try loadPromptCache(url: url)
        #expect(loaded.count == 1, "loaded cache count")
        let cache = try #require(loaded.first as? ChunkedKVCache)
        #expect(cache.offset == 10, "loaded offset")
        #expect(cache.metaState == ["8", "2"], "loaded meta state")
        #expect(Self.equal(cache.state[0], kept.keys), "loaded keys")
        #expect(Self.equal(cache.state[1], kept.values), "loaded values")
    }

    /// The order of the setters does not change the result: `metaState`
    /// first and `state` second gives the same offset as the load order.
    @Test func chunkedCacheSettersWorkInEitherOrder() {
        let kept = Self.tokens(8, seed: 7)
        let cache = ChunkedKVCache()
        cache.metaState = ["8", "2"]
        cache.state = [kept.keys, kept.values]
        #expect(cache.offset == 10, "offset")
        #expect(cache.metaState == ["8", "2"], "meta state")
    }

    /// `extract(_:)` keeps an empty slot 0 and the state in slot 1.
    @Test func mambaCacheExtractKeepsSlotPositions() {
        let cache = MambaCache()
        let slot = MLXRandom.normal([2, 3, 4], key: MLXRandom.key(8))
        cache[1] = slot
        let extracted = cache.extract(0)
        #expect(extracted is MambaCache, "extracted type")
        #expect(extracted.slotCount == 2, "slot count")
        #expect(extracted.presentSlotIndices == [1], "present slots")
        #expect(
            extracted[1].map { Self.equal($0, slot[0 ..< 1]) } == true, "slot 1 values")
    }
}
