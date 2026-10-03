import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite struct PromptCacheValidationTests {
    static func rows(_ count: Int, start: Int = 0) -> MLXArray {
        MLXArray((start ..< (start + count)).map(Float.init)).reshaped(1, 1, count, 1)
    }

    static func tempURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("cache-validation-\(UUID().uuidString).safetensors")
    }

    @Test func unpopulatedCachesRoundTrip() throws {
        let caches: [KVCache] = [
            KVCacheSimple(), ChunkedKVCache(), RotatingKVCache(maxSize: 8),
            QuantizedKVCache(), ArraysCache(size: 3), MambaCache(), CacheList(caches: []),
        ]
        for cache in caches {
            let url = Self.tempURL()
            defer { try? FileManager.default.removeItem(at: url) }
            try savePromptCache(url: url, cache: [cache])
            let (loaded, _) = try loadPromptCache(url: url)
            #expect(loaded.count == 1)
            #expect(type(of: loaded[0]) == type(of: cache))
            #expect(loaded[0].state.isEmpty)
            #expect(loaded[0].metaState == cache.metaState)
            #expect(loaded[0].offset == 0)
        }
        let url = Self.tempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [], metadata: ["model.name": "tiny"])
        let (loaded, metadata) = try loadPromptCache(url: url)
        #expect(loaded.isEmpty)
        #expect(metadata == ["model.name": "tiny"])
    }

    @Test func mixedEmptyLayersAndNestedCacheListRoundTrip() throws {
        let populated = ChunkedKVCache(chunkSize: 2)
        populated.step = 4
        _ = populated.update(keys: Self.rows(3), values: Self.rows(3, start: 100))
        populated.maybeTrimFront()
        let mamba = MambaCache()
        mamba[1] = MLXArray([1, 2, 3, 4] as [Float]).reshaped(2, 2)
        let list = CacheList(caches: [KVCacheSimple(), populated, mamba, ChunkedKVCache()])
        let url = Self.tempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: [KVCacheSimple(), list, MambaCache()])
        let (loaded, _) = try loadPromptCache(url: url)
        #expect(loaded.count == 3)
        #expect(loaded[0].state.isEmpty && loaded[2].state.isEmpty)
        let restored = try #require(loaded[1] as? CacheList)
        #expect(restored.children.count == 4)
        #expect(restored.children[0].state.isEmpty && restored.children[3].state.isEmpty)
        #expect(restored.children[1].offset == 3)
        #expect(restored.children[1].state[0].asArray(Float.self) == [1, 2])
        let restoredMamba = try #require(restored.children[2] as? MambaCache)
        #expect(restoredMamba.presentSlotIndices == [1])
        #expect(restoredMamba.extract(1)[1]?.asArray(Float.self) == [3, 4])
    }

    @Test func malformedChunkedMetadataThrows() throws {
        for meta in [
            ["8"], ["8", "0", "extra"], ["bad", "0"], ["8", "bad"],
            ["-1", "0"], ["8", "-1"], ["8", String(Int.max)],
        ] {
            let url = Self.tempURL()
            defer { try? FileManager.default.removeItem(at: url) }
            var metadata = ["2.0": "ChunkedKVCache"]
            for (i, value) in meta.enumerated() { metadata["0.0.\(i)"] = value }
            try save(
                arrays: ["0.0": Self.rows(1), "0.1": Self.rows(1)], metadata: metadata, url: url)
            #expect(throws: (any Error).self) { _ = try loadPromptCache(url: url) }
        }
    }

    @Test func nonnumericChunkedMetadataDoesNotSilentlyFallBack() throws {
        for meta in [["bad", "0"], ["8", "bad"]] {
            let url = Self.tempURL()
            defer { try? FileManager.default.removeItem(at: url) }
            try save(
                arrays: ["0.0": Self.rows(1), "0.1": Self.rows(1)],
                metadata: ["2.0": "ChunkedKVCache", "0.0.0": meta[0], "0.0.1": meta[1]],
                url: url)
            #expect(throws: (any Error).self) { _ = try loadPromptCache(url: url) }
        }
    }

    @Test func malformedStateAndMetadataInventoriesThrow() throws {
        let cases: [(String, [String], [String: MLXArray])] = [
            ("KVCache", [""], ["0.0": Self.rows(1)]),
            ("RotatingKVCache", ["0", "8", "256", "0"], [:]),
            ("RotatingKVCache", ["bad", "8", "256", "0", "0"], [:]),
            ("QuantizedKVCache", ["256", "bad", "64", "8"], [:]),
            ("QuantizedKVCache", ["256", "0", "64"], [:]),
            ("ArraysCache", ["2", "bad"], [:]),
            ("MambaCache", ["2", "-1"], ["0.0": Self.rows(1)]),
            ("CacheList", ["-1"], [:]),
            ("CacheList", ["1", "ChunkedKVCache", "0", "1", "8"], [:]),
            ("CacheList", ["1", "KVCache", "-1", "1", ""], [:]),
            ("CacheList", ["1", "KVCache", "0", String(Int.max)], [:]),
        ]
        for (name, meta, arrays) in cases {
            let url = Self.tempURL()
            defer { try? FileManager.default.removeItem(at: url) }
            var metadata = ["2.0": name]
            for (i, value) in meta.enumerated() { metadata["0.0.\(i)"] = value }
            try save(arrays: arrays, metadata: metadata, url: url)
            #expect(throws: (any Error).self) { _ = try loadPromptCache(url: url) }
        }
        for key in ["0.-1.0", "0.0.-1", "2.-1", "2.\(Int.max)"] {
            let url = Self.tempURL()
            defer { try? FileManager.default.removeItem(at: url) }
            try save(arrays: [:], metadata: ["2.0": "KVCache", key: "0"], url: url)
            #expect(throws: (any Error).self) { _ = try loadPromptCache(url: url) }
        }
    }

    @Test func legacyCompactMambaFileStillLoads() throws {
        let url = Self.tempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try save(
            arrays: ["0.0": Self.rows(2), "0.1": Self.rows(2, start: 10)],
            metadata: ["0.0.0": "", "2.0": "MambaCache"], url: url)
        let (loaded, _) = try loadPromptCache(url: url)
        let cache = try #require(loaded[0] as? MambaCache)
        #expect(cache.slotCount == 2 && cache.presentSlotIndices == [0, 1])
        #expect(cache[1]?.asArray(Float.self) == [10, 11])
    }

    @Test func chunkedBoundaryContinuationAndSetterOrders() throws {
        for chunk in [0, 1, 8] {
            for step in [1, 4, 256] {
                for count in [0, 1, 8, 10] {
                    let cache = ChunkedKVCache(chunkSize: chunk)
                    cache.step = step
                    _ = cache.update(keys: Self.rows(count), values: Self.rows(count, start: 100))
                    cache.maybeTrimFront()
                    let expected = cache.state
                    #expect(expected[0].dim(2) == min(count, chunk))
                    for reverse in [false, true] {
                        let restored = ChunkedKVCache()
                        if reverse { restored.metaState = cache.metaState }
                        restored.state = expected
                        if !reverse { restored.metaState = cache.metaState }
                        restored.metaState = cache.metaState
                        #expect(restored.offset == cache.offset)
                        let copy = cache.copy()
                        let next = Self.rows(1, start: cache.offset)
                        let (a, _) = restored.update(keys: next, values: next)
                        let (b, _) = copy.update(keys: next, values: next)
                        #expect(a.asArray(Float.self) == b.asArray(Float.self))
                    }
                    let url = Self.tempURL()
                    defer { try? FileManager.default.removeItem(at: url) }
                    try savePromptCache(url: url, cache: [cache])
                    let (loaded, _) = try loadPromptCache(url: url)
                    #expect(loaded[0].offset == cache.offset)
                    #expect(loaded[0].state[0].shape == expected[0].shape)
                    #expect(
                        expected[0].size == 0
                            || loaded[0].state[0].asArray(Float.self)
                                == expected[0].asArray(Float.self)
                    )
                }
            }
        }
    }

    @Test func mambaExtractionPreservesEveryOptionalSlotAndBatchBoundary() {
        for mask in 0 ..< 4 {
            let cache = MambaCache()
            for slot in 0 ..< 2 where mask & (1 << slot) != 0 {
                cache[slot] = MLXArray((0 ..< 12).map { Float($0 + slot * 100) }).reshaped(3, 2, 2)
            }
            for index in 0 ..< 3 {
                let extracted = cache.extract(index)
                #expect(extracted is MambaCache)
                #expect(extracted.slotCount == 2)
                #expect(extracted.presentSlotIndices == cache.presentSlotIndices)
                for slot in cache.presentSlotIndices {
                    #expect(extracted[slot]?.shape == [1, 2, 2])
                    #expect(
                        extracted[slot]?.asArray(Float.self)
                            == cache[slot]?[index ..< (index + 1)].asArray(Float.self))
                    let before = cache[slot]!.asArray(Float.self)
                    extracted[slot]![0, 0, 0] = MLXArray(Float(-1))
                    #expect(cache[slot]!.asArray(Float.self) == before)
                }
            }
        }
    }
}
