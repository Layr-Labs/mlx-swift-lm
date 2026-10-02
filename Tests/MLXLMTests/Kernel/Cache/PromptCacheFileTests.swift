import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

extension KernelTests {

    /// Tests of the prompt cache file format, the cache state setters, the
    /// cache list and the quantized attention paths in `KVCache.swift` that
    /// `KVCacheTests` and `KVCacheBehaviorTests` do not run.
    ///
    /// The files are written to a new temporary folder and deleted at the
    /// end of each test.
    @Suite
    struct PromptCacheFileTests {

        static let heads = 2

        static func tokens(_ count: Int, seed: UInt64, headDim: Int = 8)
            -> (keys: MLXArray, values: MLXArray)
        {
            let shape = [1, heads, count, headDim]
            let keys = MLXRandom.normal(shape, key: MLXRandom.key(seed * 2))
            let values = MLXRandom.normal(shape, key: MLXRandom.key(seed * 2 + 1))
            eval(keys, values)
            return (keys, values)
        }

        /// Runs `body` with the URL of a `.safetensors` file in a new
        /// temporary folder, and deletes the folder after it.
        static func withFile(_ body: (URL) throws -> Void) throws {
            let folder = FileManager.default.temporaryDirectory
                .appendingPathComponent("prompt-cache-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: folder) }
            try body(folder.appendingPathComponent("cache.safetensors"))
        }

        // MARK: - File format

        /// A chunked cache without a chunk size, a ring that wrapped, and
        /// user metadata with dots in the keys keep their values through a
        /// save and a load.
        @Test func saveAndLoadKeepTheSettingsAndTheUserMetadata() throws {
            let chunked = ChunkedKVCache()
            let first = Self.tokens(3, seed: 1)
            _ = chunked.update(keys: first.keys, values: first.values)

            let rotating = RotatingKVCache(maxSize: 4, keep: 1, step: 2)
            _ = rotating.update(keys: first.keys, values: first.values)
            for step in 0 ..< 3 {
                let new = Self.tokens(1, seed: UInt64(10 + step))
                _ = rotating.update(keys: new.keys, values: new.values)
            }
            #expect(rotating.metaState == ["1", "4", "2", "6", "3"])

            try Self.withFile { url in
                try savePromptCache(
                    url: url, cache: [chunked, rotating],
                    metadata: ["model.name": "tiny", "tokens": "6"])
                let (loaded, metadata) = try loadPromptCache(url: url)
                #expect(metadata == ["model.name": "tiny", "tokens": "6"])
                #expect(loaded.count == 2)

                let loadedChunked = try #require(loaded[0] as? ChunkedKVCache)
                #expect(loadedChunked.metaState == ["None", "0"])
                #expect(loadedChunked.offset == 3)
                #expect(
                    SyntheticModel.maxAbsDifference(loadedChunked.state[0], first.keys) == 0)

                let loadedRotating = try #require(loaded[1] as? RotatingKVCache)
                #expect(loadedRotating.metaState == rotating.metaState)
                #expect(loadedRotating.debugDescription.contains("offset: 6, maxSize: 4"))
                // The loaded ring continues like the original one.
                let next = Self.tokens(1, seed: 20)
                let (keysA, _) = rotating.update(keys: next.keys, values: next.values)
                let (keysB, _) = loadedRotating.update(keys: next.keys, values: next.values)
                #expect(SyntheticModel.maxAbsDifference(keysA, keysB) == 0)
            }
        }

        /// A file can leave out a cache index, have keys that are not
        /// `i.j`, and use the legacy one-value meta state of `ArraysCache`.
        @Test func loadFillsMissingIndicesAndReadsTheLegacyArraysFormat() throws {
            let kv = Self.tokens(2, seed: 3)
            let slotA = MLXArray([1, 2] as [Float]).reshaped(1, 2)
            let slotB = MLXArray([3, 4, 5] as [Float]).reshaped(1, 3)
            try Self.withFile { url in
                try MLX.save(
                    arrays: [
                        "1.0": kv.keys, "1.1": kv.values,
                        "2.0": slotA, "2.1": slotB,
                        "junk": slotA,
                    ],
                    metadata: [
                        "0.0.0": "2", "0.0.1": "",
                        "0.1.0": "",
                        "0.2.0": "",
                        "2.0": "ArraysCache", "2.1": "KVCache", "2.2": "ArraysCache",
                    ],
                    url: url)
                let (loaded, metadata) = try loadPromptCache(url: url)
                #expect(metadata.isEmpty)
                #expect(loaded.count == 3)

                // Cache 0 has no arrays in the file: two empty slots.
                let empty = try #require(loaded[0] as? ArraysCache)
                #expect(empty.slotCount == 2)
                #expect(empty.state.isEmpty)

                let simple = try #require(loaded[1] as? KVCacheSimple)
                #expect(simple.offset == 2)
                #expect(SyntheticModel.maxAbsDifference(simple.state[1], kv.values) == 0)

                // The legacy format keeps the arrays in order.
                let legacy = try #require(loaded[2] as? ArraysCache)
                #expect(legacy.slotCount == 2)
                #expect(legacy[0]?.asArray(Float.self) == [1, 2])
                #expect(legacy[1]?.asArray(Float.self) == [3, 4, 5])
            }
        }

        /// Each bad file gives an error, not a crash.
        @Test func loadRejectsBadFiles() throws {
            let array = MLXArray([1, 2] as [Float])
            let badFiles: [(String, [String: MLXArray], [String: String])] = [
                ("unknown class", ["0.0": array], ["0.0.0": "", "2.0": "Bogus"]),
                (
                    "class count does not match", ["0.0": array, "1.0": array],
                    ["0.0.0": "", "0.1.0": "", "2.0": "MambaCache"]
                ),
                (
                    "rotating meta state too short", ["0.0": array],
                    ["0.0.0": "0", "2.0": "RotatingKVCache"]
                ),
                (
                    "rotating cache without a size", ["0.0": array],
                    [
                        "0.0.0": "0", "0.0.1": "None", "0.0.2": "256", "0.0.3": "0",
                        "0.0.4": "0", "2.0": "RotatingKVCache",
                    ]
                ),
                (
                    "rotating size is not a number", ["0.0": array],
                    [
                        "0.0.0": "0", "0.0.1": "four", "0.0.2": "256", "0.0.3": "0",
                        "0.0.4": "0", "2.0": "RotatingKVCache",
                    ]
                ),
            ]
            for (name, arrays, metadata) in badFiles {
                try Self.withFile { url in
                    try MLX.save(arrays: arrays, metadata: metadata, url: url)
                    #expect(throws: (any Error).self, "\(name)") {
                        _ = try loadPromptCache(url: url)
                    }
                }
            }
        }

        // MARK: - CacheList

        @Test func cacheListStateAndMetaState() throws {
            let a = KVCacheSimple()
            let b = KVCacheSimple()
            let first = Self.tokens(2, seed: 4)
            _ = a.update(keys: first.keys, values: first.values)
            _ = b.update(keys: first.keys, values: first.values)
            let list = CacheList(a, b)
            #expect((list[1] as? KVCacheSimple) === b)
            #expect(list.children.count == 2)
            #expect(list.metaState == ["2", "KVCache", "2", "1", "", "KVCache", "2", "1", ""])

            // The state setter gives each child its own part.
            let x = Self.tokens(3, seed: 5)
            let y = Self.tokens(4, seed: 6)
            list.state = [x.keys, x.values, y.keys, y.values]
            #expect(a.offset == 3)
            #expect(b.offset == 4)
            #expect(SyntheticModel.maxAbsDifference(b.state[1], y.values) == 0)

            let rebuilt = try CacheList.fromState(state: list.state, metaState: list.metaState)
            #expect(rebuilt.children.map(\.offset) == [3, 4])
            #expect(rebuilt.metaState == list.metaState)
        }

        @Test func cacheListFromStateRejectsBadMetaState() {
            let bad: [(String, [String])] = [
                ("no child count", []),
                ("truncated", ["1", "KVCache"]),
                ("state count is not a number", ["1", "KVCache", "x", "0"]),
                ("meta state count is not a number", ["1", "KVCache", "0", "y"]),
                ("unknown child class", ["1", "Bogus", "0", "0"]),
            ]
            for (name, metaState) in bad {
                #expect(throws: (any Error).self, "\(name)") {
                    _ = try CacheList.fromState(state: [], metaState: metaState)
                }
            }
        }

        // MARK: - QuantizedKVCache state

        /// A quantization without biases stores 4 arrays. The meta state
        /// setter restores the offset, the group size and the bits.
        @Test func quantizedCacheStateWithoutBiases() {
            let source = QuantizedKVCache(groupSize: 32, bits: 8)
            let new = Self.tokens(4, seed: 7, headDim: 32)
            _ = source.updateQuantized(keys: new.keys, values: new.values)
            let full = source.state
            #expect(full.count == 6)

            let cache = QuantizedKVCache()
            cache.state = [full[0], full[1], full[3], full[4]]
            cache.metaState = ["256", "4", "32", "8"]
            #expect(cache.offset == 4)
            #expect(cache.groupSize == 32)
            #expect(cache.bits == 8)
            #expect(cache.state.count == 4)
            #expect(cache.innerState().count == 4)
            #expect(cache.getQuantizedState()?.0.2 == nil)
            #expect(cache.metaState == ["256", "4", "32", "8"])

            #expect(cache.trim(1) == 1)
            #expect(cache.state.count == 4)
            #expect(cache.state[0].dim(2) == 3, "the state follows the offset")
            #expect(SyntheticModel.maxAbsDifference(cache.state[0], full[0][0..., 0..., ..<3]) == 0)
        }

        // MARK: - ArraysCache transient state

        @Test func arraysCacheClearsTheSpeculativeState() {
            func tape() -> ArraysCache.PrefixReplayTape {
                let x = MLXArray.zeros([1, 2])
                return ArraysCache.PrefixReplayTape(
                    convInput: x, q: x, k: x, v: x, a: x, b: x, ssmPre: nil, mask: nil,
                    rowCount: 2, convStateRows: 3)
            }
            let cache = ArraysCache(size: 1)
            let saved = tape()
            #expect(saved.rowCount == 2)
            #expect(saved.convStateRows == 3)
            #expect(saved.ssmPre == nil)

            func arm() {
                cache.rollbackState = (MLXArray.zeros([1]), MLXArray.zeros([1]))
                cache.prefixReplayTape = tape()
            }
            arm()
            cache.clearMTPTransientState()
            #expect(cache.rollbackState == nil && cache.prefixReplayTape == nil, "clear")

            arm()
            cache.state = [MLXArray.ones([1, 2])]
            #expect(cache.rollbackState == nil && cache.prefixReplayTape == nil, "state set")

            arm()
            cache.filter(batchIndices: MLXArray([Int32(0)]))
            #expect(cache.rollbackState == nil && cache.prefixReplayTape == nil, "filter")

            arm()
            cache.extend(other: ArraysCache(size: 1))
            #expect(cache.rollbackState == nil && cache.prefixReplayTape == nil, "extend")
        }

        /// `extend` takes the batch size of a cache with no state from its
        /// left padding or from its lengths, and fills the missing rows with
        /// zeros.
        @Test func arraysCacheExtendUsesPaddingAndLengthsForTheBatchSize() {
            let padded = ArraysCache(size: 1, leftPadding: [1])
            padded[0] = MLXArray.ones([1, 3])
            #expect(padded.metaState == ["1", "0", "1"])
            let other = ArraysCache(size: 1, leftPadding: [2, 0])
            padded.extend(other: other)
            #expect(padded[0]?.shape == [3, 3])
            #expect(padded[0]?.asArray(Float.self) == [1, 1, 1, 0, 0, 0, 0, 0, 0])
            #expect(padded.leftPaddingValues == [1, 2, 0])

            let byLength = ArraysCache(size: 1)
            byLength.prepare(lengths: [4])
            let filled = ArraysCache(size: 1)
            filled[0] = MLXArray.ones([2, 3])
            byLength.extend(other: filled)
            #expect(byLength[0]?.asArray(Float.self) == [0, 0, 0, 1, 1, 1, 1, 1, 1])
            #expect(byLength.lengths?.asArray(Int32.self) == [4, 0, 0])
        }

        // MARK: - Masks and quantized attention

        @Test func cacheMaskOptionsAndDescriptions() {
            let simple = KVCacheSimple()
            let new = Self.tokens(2, seed: 8)
            _ = simple.update(keys: new.keys, values: new.values)
            #expect(simple.debugDescription.contains("offset: 2, step: 256"))
            if case .array(let mask) = simple.makeMask(n: 3, windowSize: nil, returnArray: true) {
                #expect(
                    mask.asArray(Bool.self)
                        == createCausalMask(n: 3, offset: 2).asArray(Bool.self))
                #expect(mask.shape == [3, 5])
            } else {
                Issue.record("returnArray gives an array mask")
            }

            let rotating = RotatingKVCache(maxSize: 4)
            _ = rotating.update(keys: new.keys, values: new.values)
            #expect(rotating.debugDescription.contains("offset: 2, maxSize: 4, keep: 0, idx: 2"))
            if case .none = rotating.makeMask(n: 1, windowSize: nil, returnArray: false) {
            } else {
                Issue.record("one token without a window needs no mask")
            }
            if case .array(let mask) = createAttentionMask(
                h: MLXArray.zeros([1, 3, 4]), cache: rotating, windowSize: 4, returnArray: false)
            {
                // The offset 2 plus 3 tokens is longer than the window.
                #expect(mask.shape == [3, 5])
            } else {
                Issue.record("a chunk past the window needs an array mask")
            }
        }

        /// With as many query heads as key heads, the quantized attention
        /// takes an additive float mask and the first array of `.arrays`.
        @Test func quantizedAttentionWithoutGroupedHeadsTakesEveryMaskKind() {
            let headDim = 32
            let cache = QuantizedKVCache(groupSize: 32, bits: 8)
            let new = Self.tokens(4, seed: 9, headDim: headDim)
            let queries = MLXRandom.normal([1, Self.heads, 4, headDim], key: MLXRandom.key(99))
            let (qKeys, qValues) = cache.updateQuantized(keys: new.keys, values: new.values)
            let keys = dequantized(
                qKeys.0, scales: qKeys.1, biases: qKeys.2, groupSize: 32, bits: 8)
            let values = dequantized(
                qValues.0, scales: qValues.1, biases: qValues.2, groupSize: 32, bits: 8)

            let causal = createCausalMask(n: 4, offset: 0)
            let additive = MLX.where(causal, MLXArray(Float(0)), MLXArray(Float(-1e9)))
            let scale = 1 / Float(headDim).squareRoot()
            let scores = matmul(queries * scale, keys.transposed(0, 1, 3, 2)) + additive
            let expected = matmul(softmax(scores, axis: -1), values)

            // Tolerance 1e-5: the same float32 attention over the same
            // dequantized keys, with the sums in another order.
            let masks: [(String, MLXFast.ScaledDotProductAttentionMaskMode)] = [
                ("additive", .array(additive)),
                ("arrays", .arrays([causal])),
                ("causal", .causal),
            ]
            for (name, mask) in masks {
                let output = quantizedScaledDotProductAttention(
                    queries: queries, quantizedKeys: qKeys, quantizedValues: qValues,
                    scale: scale, mask: mask, groupSize: 32, bits: 8)
                #expect(output.shape == [1, Self.heads, 4, headDim])
                #expect(
                    SyntheticModel.maxAbsDifference(output, expected) <= 1e-5, "\(name) mask")
            }
        }
    }
}
