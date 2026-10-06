import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

extension KernelTests {

    /// Behavior tests of the caches, the masks and the quantized attention
    /// in `KVCache.swift`.
    ///
    /// The attention tests feed the same random keys and values to a cache
    /// and to a reference that keeps every token. They compare attention
    /// outputs, so the order of the tokens in a ring buffer does not matter,
    /// but a lost, stale or extra token does.
    @Suite
    struct KVCacheBehaviorTests {

        static let heads = 2
        static let headDim = 8

        /// Random keys, values and queries for `count` new tokens.
        static func tokens(_ count: Int, seed: UInt64, batch: Int = 1, headDim: Int = headDim)
            -> (keys: MLXArray, values: MLXArray, queries: MLXArray)
        {
            let shape = [batch, heads, count, headDim]
            let keys = MLXRandom.normal(shape, key: MLXRandom.key(seed * 3))
            let values = MLXRandom.normal(shape, key: MLXRandom.key(seed * 3 + 1))
            let queries = MLXRandom.normal(shape, key: MLXRandom.key(seed * 3 + 2))
            eval(keys, values, queries)
            return (keys, values, queries)
        }

        static func attention(
            _ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode
        ) -> MLXArray {
            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values,
                scale: 1 / Float(queries.dim(-1)).squareRoot(), mask: mask)
            eval(output)
            return output
        }

        // Tolerance of the attention comparisons: both sides run the same
        // float32 attention over the same keys; only the order and the
        // number of masked entries differ. The error is near 1e-7 for
        // outputs of size 1.
        static let tolerance: Float = 1e-5

        // MARK: - KVCacheSimple

        @Test func simpleCacheKeepsEveryTokenAcrossStepBoundaries() {
            let cache = KVCacheSimple()
            cache.step = 4
            var allKeys: [MLXArray] = []
            var allValues: [MLXArray] = []
            for (index, count) in [3, 2, 4, 1, 5].enumerated() {
                let new = Self.tokens(count, seed: UInt64(index + 1))
                allKeys.append(new.keys)
                allValues.append(new.values)
                let (keys, values) = cache.update(keys: new.keys, values: new.values)
                let expectedKeys = concatenated(allKeys, axis: 2)
                #expect(keys.shape == expectedKeys.shape)
                #expect(SyntheticModel.maxAbsDifference(keys, expectedKeys) == 0)
                #expect(
                    SyntheticModel.maxAbsDifference(values, concatenated(allValues, axis: 2)) == 0)
            }
            #expect(cache.offset == 15)
            #expect(cache.state[0].dim(2) == 15)

            // Trimming 4 tokens and adding 2 overwrites the last 4.
            #expect(cache.isTrimmable)
            #expect(cache.trim(4) == 4)
            let new = Self.tokens(2, seed: 9)
            let (keys, _) = cache.update(keys: new.keys, values: new.values)
            let kept = concatenated(allKeys, axis: 2)[0..., 0..., ..<11]
            #expect(
                SyntheticModel.maxAbsDifference(keys, concatenated([kept, new.keys], axis: 2))
                    == 0)
            #expect(cache.offset == 13)
            // Trimming more than the offset trims the offset only.
            #expect(cache.trim(100) == 13)
            #expect(cache.offset == 0)
        }

        @Test func simpleCacheMasks() {
            let cache = KVCacheSimple()
            _ = cache.update(
                keys: Self.tokens(3, seed: 1).keys, values: Self.tokens(3, seed: 1).values)
            if case .none = cache.makeMask(n: 1, windowSize: nil, returnArray: false) {
            } else {
                Issue.record("one token needs no mask")
            }
            if case .causal = cache.makeMask(n: 4, windowSize: nil, returnArray: false) {
            } else {
                Issue.record("a prompt without a window uses the causal mask")
            }
            guard case .array(let mask) = cache.makeMask(n: 4, windowSize: 2, returnArray: false)
            else {
                Issue.record("a prompt longer than the window needs an array mask")
                return
            }
            let expected = createCausalMask(n: 4, offset: 3, windowSize: 2)
            #expect(mask.shape == [4, 7])
            #expect((mask .== expected).all().item(Bool.self))
        }

        // MARK: - RotatingKVCache

        /// Runs the chunks through a rotating cache and through a reference
        /// that keeps every token, with the mask each would give a model, and
        /// compares the attention outputs.
        static func checkRotatingAttention(
            maxSize: Int, window: Int, chunks: [Int], step: Int = 256,
            sourceLocation: SourceLocation = #_sourceLocation
        ) {
            let cache = RotatingKVCache(maxSize: maxSize, step: step)
            var allKeys: MLXArray? = nil
            var allValues: MLXArray? = nil
            for (index, count) in chunks.enumerated() {
                let new = Self.tokens(count, seed: UInt64(100 + index))
                let previous = cache.offset

                let mask = cache.makeMask(n: count, windowSize: window, returnArray: false)
                let (keys, values) = cache.update(keys: new.keys, values: new.values)
                let output = Self.attention(new.queries, keys, values, mask: mask)

                allKeys = allKeys.map { concatenated([$0, new.keys], axis: 2) } ?? new.keys
                allValues = allValues.map { concatenated([$0, new.values], axis: 2) } ?? new.values
                let referenceMask = createCausalMask(
                    n: count, offset: previous, windowSize: window)
                let expected = Self.attention(
                    new.queries, allKeys!, allValues!, mask: .array(referenceMask))

                let difference = SyntheticModel.maxAbsDifference(output, expected)
                #expect(
                    difference <= Self.tolerance,
                    "chunk \(index) at offset \(previous) differs by \(difference)",
                    sourceLocation: sourceLocation)
                #expect(cache.offset == previous + count, sourceLocation: sourceLocation)
            }
        }

        @Test func rotatingCacheMatchesSlidingWindowAttention() {
            // A prompt longer than the window, decode steps that wrap the
            // ring twice, then a second prompt chunk after the wrap.
            Self.checkRotatingAttention(
                maxSize: 4, window: 4, chunks: [6, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1])
            // A ring that grows in steps of 2 before it is full.
            Self.checkRotatingAttention(
                maxSize: 5, window: 5, chunks: [1, 1, 1, 1, 1, 1, 1, 2, 1], step: 2)
        }

        /// A ring larger than the attention window: a decode step needs the
        /// rolled array mask of `RotatingKVCache.makeMask`.
        @Test func rotatingCacheLargerThanTheWindowMasksOldTokens() {
            Self.checkRotatingAttention(
                maxSize: 6, window: 4, chunks: [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            Self.checkRotatingAttention(
                maxSize: 6, window: 4, chunks: [7, 1, 1, 1, 1, 1, 1, 1])
        }

        /// With `keep`, the first tokens stay in the ring for good and the
        /// other slots hold the most recent tokens.
        @Test func rotatingCacheKeepsTheFirstTokens() {
            let maxSize = 5
            let keep = 2
            let cache = RotatingKVCache(maxSize: maxSize, keep: keep)
            var allKeys: [MLXArray] = []
            var allValues: [MLXArray] = []
            for (index, count) in [3, 1, 1, 1, 1, 1, 1, 1, 1].enumerated() {
                let new = Self.tokens(count, seed: UInt64(200 + index))
                let previous = cache.offset
                let mask = cache.makeMask(n: count, windowSize: nil, returnArray: false)
                let (keys, values) = cache.update(keys: new.keys, values: new.values)
                let output = Self.attention(new.queries, keys, values, mask: mask)

                allKeys.append(new.keys)
                allValues.append(new.values)
                let length = previous + count
                // Allowed keys: positions before `keep`, and the last
                // maxSize - keep positions. The first chunk fits in the ring,
                // so it uses the plain causal mask.
                var allowed: [Bool] = []
                for query in previous ..< length {
                    for key in 0 ..< length {
                        let recent = key > length - 1 - (maxSize - keep)
                        allowed.append(key <= query && (key < keep || recent || length <= maxSize))
                    }
                }
                let referenceMask = MLXArray(allowed).reshaped(count, length)
                let expected = Self.attention(
                    new.queries, concatenated(allKeys, axis: 2), concatenated(allValues, axis: 2),
                    mask: .array(referenceMask))
                #expect(
                    SyntheticModel.maxAbsDifference(output, expected) <= Self.tolerance,
                    "chunk \(index) at offset \(previous)")
            }
            #expect(cache.metaState == ["2", "5", "256", "11", "5"])
        }

        @Test func rotatingCacheCopyContinuesLikeTheOriginal() {
            let cache = RotatingKVCache(maxSize: 4)
            for (index, count) in [5, 1, 1, 1].enumerated() {
                let new = Self.tokens(count, seed: UInt64(300 + index))
                _ = cache.update(keys: new.keys, values: new.values)
            }
            #expect(cache.isTrimmable == false, "a full ring cannot be trimmed")
            let copy = cache.copy()
            #expect(copy.metaState == cache.metaState)

            for index in 0 ..< 3 {
                let new = Self.tokens(1, seed: UInt64(400 + index))
                let (keysA, valuesA) = cache.update(keys: new.keys, values: new.values)
                let (keysB, valuesB) = copy.update(keys: new.keys, values: new.values)
                let outputA = Self.attention(new.queries, keysA, valuesA, mask: .none)
                let outputB = Self.attention(new.queries, keysB, valuesB, mask: .none)
                #expect(SyntheticModel.maxAbsDifference(outputA, outputB) == 0, "step \(index)")
            }
        }

        @Test func rotatingCacheTrimBeforeTheRingIsFull() {
            let cache = RotatingKVCache(maxSize: 8)
            let first = Self.tokens(5, seed: 1)
            _ = cache.update(keys: first.keys, values: first.values)
            #expect(cache.isTrimmable)
            #expect(cache.trim(2) == 2)
            #expect(cache.offset == 3)

            let next = Self.tokens(1, seed: 2)
            let (keys, _) = cache.update(keys: next.keys, values: next.values)
            let expected = concatenated([first.keys[0..., 0..., ..<3], next.keys], axis: 2)
            #expect(keys.shape == expected.shape)
            #expect(SyntheticModel.maxAbsDifference(keys, expected) == 0)
        }

        // MARK: - QuantizedKVCache

        static func dequantize(
            _ quantized: (MLXArray, MLXArray, MLXArray?), groupSize: Int, bits: Int
        ) -> MLXArray {
            dequantized(
                quantized.0, scales: quantized.1, biases: quantized.2, groupSize: groupSize,
                bits: bits)
        }

        /// Attention with the sink as one extra score column, in plain ops.
        static func referenceAttention(
            queries: MLXArray, keys: MLXArray, values: MLXArray, mask: MLXArray?,
            sinks: MLXArray?
        ) -> MLXArray {
            let repeats = queries.dim(1) / keys.dim(1)
            let keys = repeated(keys, count: repeats, axis: 1)
            let values = repeated(values, count: repeats, axis: 1)
            let scale = 1 / Float(queries.dim(-1)).squareRoot()
            var scores = matmul(queries * scale, keys.transposed(0, 1, 3, 2))
            if let mask {
                scores = MLX.where(mask, scores, MLXArray(-Float.infinity))
            }
            if let sinks {
                let (b, h, l) = (scores.dim(0), scores.dim(1), scores.dim(2))
                let column = broadcast(sinks.reshaped(1, h, 1, 1), to: [b, h, l, 1])
                let weights = softmax(concatenated([scores, column], axis: -1), axis: -1)
                return matmul(weights[.ellipsis, ..<scores.dim(-1)], values)
            }
            return matmul(softmax(scores, axis: -1), values)
        }

        /// The quantized attention matches float attention over the
        /// dequantized keys and values, with grouped-query heads, the causal
        /// mask and decode steps.
        @Test(arguments: [false, true])
        func quantizedAttentionMatchesDequantizedAttention(withSinks: Bool) {
            let headDim = 32
            let cache = QuantizedKVCache(groupSize: 32, bits: 8)
            let sinks = withSinks ? MLXArray([0.5, -1.0, 2.0, 0.0] as [Float]) : nil
            var fullKeys: [MLXArray] = []
            for (index, count) in [5, 1, 1, 3].enumerated() {
                let new = Self.tokens(count, seed: UInt64(500 + index), headDim: headDim)
                fullKeys.append(new.keys)
                // 4 query heads over 2 key heads.
                let queries = concatenated([new.queries, new.queries * 0.5], axis: 1)
                let (qKeys, qValues) = cache.updateQuantized(keys: new.keys, values: new.values)
                let mask: MLXFast.ScaledDotProductAttentionMaskMode = count > 1 ? .causal : .none
                let output = quantizedScaledDotProductAttention(
                    queries: queries, quantizedKeys: qKeys, quantizedValues: qValues,
                    scale: 1 / Float(headDim).squareRoot(), mask: mask, groupSize: 32, bits: 8,
                    sinks: sinks)

                let keys = Self.dequantize(qKeys, groupSize: 32, bits: 8)
                let values = Self.dequantize(qValues, groupSize: 32, bits: 8)
                let length = keys.dim(2)
                let referenceMask =
                    count > 1 ? createCausalMask(n: count, offset: length - count) : nil
                let expected = Self.referenceAttention(
                    queries: queries, keys: keys, values: values, mask: referenceMask, sinks: sinks)
                #expect(
                    SyntheticModel.maxAbsDifference(output, expected) <= Self.tolerance,
                    "chunk \(index)")
                // 8-bit keys stay close to the float keys.
                #expect(
                    SyntheticModel.maxAbsDifference(keys, concatenated(fullKeys, axis: 2)) <= 0.05)
            }
            #expect(cache.offset == 10)
        }

        /// With 2 rows the mask has a batch axis. The scores are reshaped to
        /// `[B, heads, L, kL]` before the mask is applied.
        @Test func quantizedAttentionAppliesABatchedArrayMask() {
            let headDim = 32
            let cache = QuantizedKVCache(groupSize: 32, bits: 8)
            let new = Self.tokens(4, seed: 600, batch: 2, headDim: headDim)
            let queries = concatenated([new.queries, new.queries], axis: 1)
            let (qKeys, qValues) = cache.updateQuantized(keys: new.keys, values: new.values)
            // Row 1 has 2 tokens of left padding.
            let mask = createCausalMask(n: 4, offset: 0, leftPadding: MLXArray([0, 2] as [Int32]))
            #expect(mask.shape == [2, 1, 4, 4])
            let output = quantizedScaledDotProductAttention(
                queries: queries, quantizedKeys: qKeys, quantizedValues: qValues,
                scale: 1 / Float(headDim).squareRoot(), mask: .array(mask), groupSize: 32, bits: 8)
            let expected = Self.referenceAttention(
                queries: queries, keys: Self.dequantize(qKeys, groupSize: 32, bits: 8),
                values: Self.dequantize(qValues, groupSize: 32, bits: 8), mask: mask, sinks: nil)
            // Rows 0 and 1 of the padded row mask every key, so compare the
            // rows that attend to at least one key.
            #expect(
                SyntheticModel.maxAbsDifference(output[0 ..< 1], expected[0 ..< 1])
                    <= Self.tolerance)
            #expect(
                SyntheticModel.maxAbsDifference(
                    output[1 ..< 2, 0..., 2...], expected[1 ..< 2, 0..., 2...]) <= Self.tolerance)
            // A fully masked query row gives zeros, not NaN.
            #expect(SyntheticModel.maxAbs(output[1 ..< 2, 0..., ..<2]) == 0)
        }

        @Test func quantizedCacheGrowsPastOneStepAndKeepsItsSettings() {
            let cache = QuantizedKVCache(groupSize: 64, bits: 4)
            let first = Self.tokens(200, seed: 1, headDim: 32)
            let second = Self.tokens(100, seed: 2, headDim: 32)
            _ = cache.updateQuantized(keys: first.keys, values: first.values)
            // Head dimension 32 is not a multiple of 64. The empty cache
            // takes the group size 32 instead.
            #expect(cache.groupSize == 32)
            let (qKeys, _) = cache.updateQuantized(keys: second.keys, values: second.values)
            #expect(qKeys.0.dim(2) == 300)
            let keys = Self.dequantize(qKeys, groupSize: 32, bits: 4)
            // 4-bit affine quantization of normal values in groups of 32:
            // the error stays below 1/15 of the group range (about 6).
            #expect(
                SyntheticModel.maxAbsDifference(
                    keys, concatenated([first.keys, second.keys], axis: 2)) <= 0.5)

            let copy = cache.copy() as! QuantizedKVCache
            #expect(copy.groupSize == 32)
            #expect(copy.bits == 4)
            #expect(copy.offset == 300)
            #expect(copy.metaState == cache.metaState)

            #expect(cache.trim(50) == 50)
            #expect(cache.getQuantizedState()?.0.0.dim(2) == 250)
            let unquantized = cache.toUnquantized()
            #expect(unquantized.offset == 250)
            #expect(unquantized.state[0].shape == [1, 2, 250, 32])
        }

        @Test func simpleCacheQuantizesAndDequantizes() {
            let cache = KVCacheSimple()
            let new = Self.tokens(6, seed: 7, headDim: 32)
            _ = cache.update(keys: new.keys, values: new.values)
            let quantized = cache.toQuantized(groupSize: 64, bits: 8)
            #expect(quantized.groupSize == 32)
            #expect(quantized.offset == 6)
            let back = quantized.toUnquantized()
            #expect(back.offset == 6)
            #expect(SyntheticModel.maxAbsDifference(back.state[0], new.keys) <= 0.05)
            #expect(SyntheticModel.maxAbsDifference(back.state[1], new.values) <= 0.05)

            let empty = KVCacheSimple().toQuantized(groupSize: 64, bits: 4)
            #expect(empty.offset == 0)
            #expect(empty.state.isEmpty)
        }

        @Test func maybeQuantizeConvertsOnlyAttentionCaches() {
            func makeCaches(headDim: Int, tokens: Int) -> [KVCache] {
                let mamba = MambaCache()
                mamba[0] = MLXArray.zeros([1, 3, 4])
                let simple = KVCacheSimple()
                let new = Self.tokens(tokens, seed: 1, headDim: headDim)
                _ = simple.update(keys: new.keys, values: new.values)
                return [mamba, simple]
            }

            var caches = makeCaches(headDim: 32, tokens: 4)
            maybeQuantizeKVCache(cache: &caches, kvBits: 8, kvGroupSize: 32, quantizedKVStart: 0)
            #expect(caches[0] is MambaCache)
            #expect((caches[1] as? QuantizedKVCache)?.bits == 8)
            #expect(caches[1].offset == 4)

            // At the start threshold, without bits, or with a head dimension
            // that no group size divides, the caches stay as they are.
            caches = makeCaches(headDim: 32, tokens: 4)
            maybeQuantizeKVCache(cache: &caches, kvBits: 8, kvGroupSize: 32, quantizedKVStart: 4)
            #expect(!(caches[1] is QuantizedKVCache))
            maybeQuantizeKVCache(cache: &caches, kvBits: nil)
            #expect(!(caches[1] is QuantizedKVCache))
            caches = makeCaches(headDim: 24, tokens: 4)
            maybeQuantizeKVCache(cache: &caches, kvBits: 8, kvGroupSize: 32, quantizedKVStart: 0)
            #expect(!(caches[1] is QuantizedKVCache))
        }

        // MARK: - ChunkedKVCache

        @Test func chunkedCacheDropsTheFrontAndKeepsTheRecentTokens() {
            let cache = ChunkedKVCache(chunkSize: 8)
            cache.step = 4
            let first = Self.tokens(6, seed: 1)
            let second = Self.tokens(4, seed: 2)
            _ = cache.update(keys: first.keys, values: first.values)
            _ = cache.update(keys: second.keys, values: second.values)
            let all = concatenated([first.keys, second.keys], axis: 2)

            cache.maybeTrimFront()
            #expect(cache.metaState == ["8", "2"])
            let third = Self.tokens(1, seed: 3)
            let (keys, _) = cache.update(keys: third.keys, values: third.values)
            let expected = concatenated([all[0..., 0..., 2...], third.keys], axis: 2)
            #expect(keys.shape == expected.shape)
            #expect(SyntheticModel.maxAbsDifference(keys, expected) == 0)
            #expect(cache.offset == 11)

            // trim cannot remove tokens before the start position.
            #expect(cache.trim(100) == 9)
            #expect(cache.offset == 2)
        }

        /// A copy of a chunked cache after `maybeTrimFront()` must continue
        /// like the original.
        @Test func chunkedCacheCopyAfterTrimFrontContinuesLikeTheOriginal() {
            let cache = ChunkedKVCache(chunkSize: 8)
            cache.step = 4
            let first = Self.tokens(10, seed: 1)
            _ = cache.update(keys: first.keys, values: first.values)
            cache.maybeTrimFront()

            let copy = cache.copy()
            let next = Self.tokens(1, seed: 2)
            let (keysA, _) = cache.update(keys: next.keys, values: next.values)
            let (keysB, _) = copy.update(keys: next.keys, values: next.values)
            #expect(copy.offset == cache.offset, "copy offset")
            #expect(keysB.shape == keysA.shape, "copy keys shape")
        }

        // MARK: - ArraysCache and MambaCache

        @Test func arraysCacheFiltersExtendsAndExtractsRows() {
            let cache = ArraysCache(size: 2)
            cache[0] = MLXArray(0 ..< 6).reshaped(3, 2).asType(.float32)
            cache[1] = MLXArray(10 ..< 13).reshaped(3, 1).asType(.float32)

            cache.filter(batchIndices: MLXArray([2, 0] as [Int32]))
            #expect(cache[0]!.asArray(Float.self) == [4, 5, 0, 1])
            #expect(cache[1]!.asArray(Float.self) == [12, 10])

            let other = ArraysCache(size: 2)
            other[0] = MLXArray([7, 8] as [Float]).reshaped(1, 2)
            cache.extend(other: other)
            #expect(cache[0]!.asArray(Float.self) == [4, 5, 0, 1, 7, 8])
            // The other cache has no slot 1, so its row there is zero.
            #expect(cache[1]!.asArray(Float.self) == [12, 10, 0])

            let row = cache.extract(1)
            #expect(row[0]!.asArray(Float.self) == [0, 1])
            #expect(row[1]!.asArray(Float.self) == [10])

            let mamba = MambaCache()
            mamba[0] = MLXArray(0 ..< 4).reshaped(2, 2).asType(.float32)
            mamba[1] = MLXArray(4 ..< 8).reshaped(2, 2).asType(.float32)
            let extracted = mamba.extract(1)
            #expect(extracted is MambaCache)
            #expect(extracted[0]!.asArray(Float.self) == [2, 3])
            #expect(extracted[1]!.asArray(Float.self) == [6, 7])
        }

        @Test func arraysCacheMasksFollowPaddingAndLengths() {
            let cache = ArraysCache(size: 2, leftPadding: [0, 2])
            #expect(
                cache.makeMask(N: 4)!.asArray(Bool.self)
                    == [true, true, true, true, false, false, true, true])
            cache.advance(1)
            #expect(
                cache.makeMask(N: 3)!.asArray(Bool.self)
                    == [true, true, true, false, true, true])
            cache.finalize()
            #expect(cache.makeMask(N: 3) == nil)

            cache.prepare(lengths: [3, 1])
            #expect(
                cache.makeMask(N: 3)!.asArray(Bool.self)
                    == [true, true, true, true, false, false])
            #expect(createSSMMask(h: MLXArray.zeros([2, 3, 4]), cache: nil) == nil)
        }

        /// A copy keeps each state in its slot, also when an earlier slot is
        /// empty.
        @Test func arraysCacheCopyKeepsSlotPositions() {
            let cache = ArraysCache(size: 3)
            cache[1] = MLXArray([1, 2] as [Float])
            let copy = cache.copy() as! ArraysCache
            #expect(copy.slotCount == 3, "slot count")
            #expect(copy.presentSlotIndices == [1], "present slots")
        }

        // MARK: - Masks and prompt cache helpers

        @Test func causalMaskFollowsWindowLengthsAndLeftPadding() {
            func expected(
                n: Int, offset: Int, window: Int?, length: Int? = nil, padding: Int? = nil
            ) -> [Bool] {
                var result: [Bool] = []
                for query in offset ..< offset + n {
                    for key in 0 ..< offset + n {
                        var allowed = key <= query
                        if let window { allowed = allowed && query < key + window }
                        if let length { allowed = allowed && key < length }
                        if let padding { allowed = allowed && key >= padding }
                        result.append(allowed)
                    }
                }
                return result
            }
            #expect(
                createCausalMask(n: 3, offset: 4, windowSize: 3).asArray(Bool.self)
                    == expected(n: 3, offset: 4, window: 3))
            let byLength = createCausalMask(
                n: 3, offset: 2, lengths: MLXArray([5, 3] as [Int32]))
            #expect(byLength.shape == [2, 1, 3, 5])
            #expect(
                byLength.asArray(Bool.self)
                    == expected(n: 3, offset: 2, window: nil, length: 5)
                    + expected(n: 3, offset: 2, window: nil, length: 3))
            let byPadding = createCausalMask(
                n: 3, offset: 0, windowSize: 2, leftPadding: MLXArray([0, 1] as [Int32]))
            #expect(
                byPadding.asArray(Bool.self)
                    == expected(n: 3, offset: 0, window: 2, padding: 0)
                    + expected(n: 3, offset: 0, window: 2, padding: 1))
        }

        @Test func attentionMaskHelpersWithoutACache() {
            let prompt = MLXArray.zeros([1, 5, 4])
            let token = MLXArray.zeros([1, 1, 4])
            if case .none = makeAttentionMask(n: 1, cache: nil) {
            } else {
                Issue.record("one token needs no mask")
            }
            if case .causal = makeAttentionMask(n: 5, cache: nil) {
            } else {
                Issue.record("a prompt uses the causal mask")
            }
            if case .array(let mask) = makeAttentionMask(n: 5, cache: nil, windowSize: 2) {
                #expect(mask.shape == [5, 5])
            } else {
                Issue.record("a prompt longer than the window needs an array mask")
            }
            if case .array(let mask) = createAttentionMask(
                h: prompt, cache: nil as KVCache?, returnArray: true)
            {
                #expect(mask.shape == [5, 5])
            } else {
                Issue.record("returnArray gives an array mask")
            }
            if case .none = createAttentionMask(h: token, cache: nil as KVCache?) {
            } else {
                Issue.record("one token needs no mask")
            }
            let rotating = RotatingKVCache(maxSize: 4)
            _ = rotating.update(
                keys: Self.tokens(2, seed: 1).keys, values: Self.tokens(2, seed: 1).values)
            if case .array(let mask) = makeAttentionMask(n: 3, cache: rotating) {
                #expect(mask.shape == [3, 5])
            } else {
                Issue.record("a prompt that fills the ring needs an array mask")
            }

            // The array form counts the offset of the first cache.
            let simple = KVCacheSimple()
            _ = simple.update(
                keys: Self.tokens(2, seed: 1).keys, values: Self.tokens(2, seed: 1).values)
            let arrayMask: MLXArray? = createAttentionMask(h: prompt, cache: [simple])
            #expect(arrayMask?.shape == [5, 7])
            let noMask: MLXArray? = createAttentionMask(h: token, cache: [simple])
            #expect(noMask == nil)
        }

        @Test func promptCacheHelpersTrimOnlyTrimmableCaches() {
            let rotating = makePromptCacheWithLayerCount(numLayers: 2, maxKVSize: 8)
            #expect(rotating.count == 2)
            #expect(rotating.allSatisfy { $0 is RotatingKVCache })
            #expect(rotating[0].metaState[0] == "4", "keep 4 tokens")
            #expect(rotating[0].maxSize == 8)
            let simple = makePromptCacheWithLayerCount(numLayers: 3)
            #expect(simple.allSatisfy { type(of: $0) == KVCacheSimple.self })

            for cache in simple {
                _ = cache.update(
                    keys: Self.tokens(5, seed: 1).keys, values: Self.tokens(5, seed: 1).values)
            }
            #expect(canTrimPromptCache(simple))
            #expect(trimPromptCache(simple, numTokens: 3) == 3)
            #expect(simple.allSatisfy { $0.offset == 2 })

            let mixed: [KVCache] = [KVCacheSimple(), MambaCache()]
            _ = mixed[0].update(
                keys: Self.tokens(5, seed: 1).keys, values: Self.tokens(5, seed: 1).values)
            #expect(!canTrimPromptCache(mixed))
            #expect(trimPromptCache(mixed, numTokens: 3) == 0)
            #expect(mixed[0].offset == 5)
            #expect(trimPromptCache([], numTokens: 3) == 0)
        }

        @Test func cacheListTrimsAndReportsEveryChild() {
            let a = KVCacheSimple()
            let b = KVCacheSimple()
            _ = a.update(keys: Self.tokens(4, seed: 1).keys, values: Self.tokens(4, seed: 1).values)
            _ = b.update(keys: Self.tokens(4, seed: 2).keys, values: Self.tokens(4, seed: 2).values)
            let list = CacheList(a, b)
            #expect(list.isTrimmable)
            #expect(list.trim(3) == 3)
            #expect(a.offset == 1)
            #expect(b.offset == 1)
            #expect(list.innerState().count == 4)
            #expect(CacheList(a, MambaCache()).isTrimmable == false)
        }
    }
}
