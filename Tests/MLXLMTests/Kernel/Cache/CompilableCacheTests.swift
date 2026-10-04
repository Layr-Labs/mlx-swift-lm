import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

extension KernelTests {

    /// Tests of the fixed-size caches for compiled decode:
    /// `CompilableRotatingKVCache`, `CompilableKVCache`, and the dynamic slice
    /// helpers in `DynamicSlice.swift` that they use.
    ///
    /// The compilable rotating cache must write the same ring as
    /// `RotatingKVCache` on the same inputs. The buffers hold copies of the
    /// same values, so these comparisons use tolerance 0.
    @Suite
    struct CompilableCacheTests {

        static let heads = 2
        static let headDim = 8

        /// Random keys, values and queries for `count` new tokens.
        static func tokens(_ count: Int, seed: UInt64)
            -> (keys: MLXArray, values: MLXArray, queries: MLXArray)
        {
            let shape = [1, heads, count, headDim]
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

        /// The boolean values of an array mask, or nil for a symbolic mask.
        static func maskValues(_ mode: MLXFast.ScaledDotProductAttentionMaskMode) -> [Bool]? {
            if case .array(let mask) = mode {
                return mask.asArray(Bool.self)
            }
            return nil
        }

        // Tolerance of the attention comparisons: the two sides attend over
        // the same keys. The compilable side also has masked slots, which
        // add exact zeros to the softmax sums. The error is near 1e-7.
        static let tolerance: Float = 1e-5

        // MARK: - CompilableRotatingKVCache

        /// After a prefill in a `RotatingKVCache`, the promoted cache and the
        /// source cache take the same decode steps. Each step must write the
        /// same physical ring slots, keep the same counters, and give the
        /// same attention output. The steps wrap the ring two times.
        @Test(arguments: [0, 2])
        func promotedRingMatchesRotatingCache(keep: Int) {
            let maxSize = 5
            let rotating = RotatingKVCache(maxSize: maxSize, keep: keep)
            let prompt = Self.tokens(3, seed: 1)
            _ = rotating.update(keys: prompt.keys, values: prompt.values)

            let compilable = CompilableRotatingKVCache(from: rotating)
            #expect(compilable.keys?.shape == [1, Self.heads, maxSize, Self.headDim])
            #expect(compilable.idxArray.item(Int32.self) == 3)
            #expect(compilable.offsetArray.item(Int32.self) == 3)
            #expect(compilable.offset == 3)
            #expect(compilable.keep == keep)
            #expect(compilable.maxSize == maxSize)
            #expect(compilable.innerState().count == 4)

            for step in 0 ..< 9 {
                let new = Self.tokens(1, seed: UInt64(10 + step))
                let mask = compilable.makeMask(n: 1, windowSize: nil, returnArray: false)
                let (rotatingKeys, rotatingValues) = rotating.update(
                    keys: new.keys, values: new.values)
                let (keys, values) = compilable.update(keys: new.keys, values: new.values)

                #expect(keys.shape == [1, Self.heads, maxSize, Self.headDim], "step \(step)")
                let valid = rotatingKeys.dim(2)
                #expect(
                    SyntheticModel.maxAbsDifference(keys[0..., 0..., ..<valid], rotatingKeys) == 0,
                    "step \(step): keys")
                #expect(
                    SyntheticModel.maxAbsDifference(values[0..., 0..., ..<valid], rotatingValues)
                        == 0, "step \(step): values")

                // RotatingKVCache moves its write index back to `keep` just
                // before the next write. The compilable cache wraps at once.
                let expectedIndex = rotating.idx == maxSize ? keep : rotating.idx
                #expect(
                    compilable.idxArray.item(Int32.self) == Int32(expectedIndex),
                    "step \(step): write index")
                #expect(
                    compilable.offsetArray.item(Int32.self) == Int32(rotating.offset),
                    "step \(step): offset")

                let output = Self.attention(new.queries, keys, values, mask: mask)
                let expected = Self.attention(
                    new.queries, rotatingKeys, rotatingValues, mask: .none)
                #expect(
                    SyntheticModel.maxAbsDifference(output, expected) <= Self.tolerance,
                    "step \(step): attention")
            }
        }

        /// An empty cache allocates the full ring at the first update. The
        /// mask follows the causal rule in the linear phase and allows every
        /// slot after the ring is full.
        @Test func emptyRingAllocatesAndMasks() {
            let maxSize = 6
            let cache = CompilableRotatingKVCache(maxSize: maxSize)
            #expect(cache.innerState().count == 2, "only the two counters")
            let first = Self.tokens(2, seed: 1)
            let (keys, _) = cache.update(keys: first.keys, values: first.values)
            #expect(keys.shape == [1, Self.heads, maxSize, Self.headDim])
            #expect(SyntheticModel.maxAbsDifference(keys[0..., 0..., ..<2], first.keys) == 0)
            #expect(SyntheticModel.maxAbs(keys[0..., 0..., 2...]) == 0, "unwritten slots stay zero")
            #expect(cache.idxArray.item(Int32.self) == 2)

            // Decode query at position 2: slots 0 ... 2.
            #expect(
                Self.maskValues(cache.makeMask(n: 1, windowSize: nil, returnArray: false))
                    == [true, true, true, false, false, false])
            // Three queries at positions 2, 3 and 4.
            #expect(
                Self.maskValues(cache.makeMask(n: 3, windowSize: nil, returnArray: false))
                    == [
                        true, true, true, false, false, false,
                        true, true, true, true, false, false,
                        true, true, true, true, true, false,
                    ])

            for step in 0 ..< 4 {
                let new = Self.tokens(1, seed: UInt64(20 + step))
                _ = cache.update(keys: new.keys, values: new.values)
            }
            #expect(cache.offsetArray.item(Int32.self) == 6)
            #expect(cache.idxArray.item(Int32.self) == 0, "the ring wraps to slot 0")
            #expect(
                Self.maskValues(cache.makeMask(n: 1, windowSize: nil, returnArray: false))
                    == Array(repeating: true, count: maxSize))
        }

        /// A sliding window must keep the current query position and the
        /// `window - 1` positions before it, like `createCausalMask` and the
        /// `RotatingKVCache` mask.
        @Test func windowedMaskKeepsTheCurrentPosition() {
            let cache = CompilableRotatingKVCache(maxSize: 8)
            let prompt = Self.tokens(5, seed: 3)
            _ = cache.update(keys: prompt.keys, values: prompt.values)
            let values = Self.maskValues(cache.makeMask(n: 1, windowSize: 3, returnArray: false))
            #expect(values?.count == 8)
            // The query at position 5 attends to positions 3, 4 and 5.
            let reference = createCausalMask(n: 1, offset: 5, windowSize: 3).asArray(Bool.self)
            #expect(reference == [false, false, false, true, true, true])
            #expect(
                values == reference + [false, false],
                "window keeps the current position")
        }

        /// `promote(from:maxLength:)` is the same as `init(from:)`. An empty
        /// source gives an empty cache that allocates at the first update.
        @Test func promoteFromAnEmptyRotatingCache() {
            let source = RotatingKVCache(maxSize: 4, keep: 1, step: 2)
            let cache = CompilableRotatingKVCache.promote(from: source, maxLength: 99)
            #expect(cache.keys == nil)
            #expect(cache.maxSize == 4)
            #expect(cache.keep == 1)
            #expect(cache.step == 2)
            #expect(cache.idxArray.item(Int32.self) == 0)
            let new = Self.tokens(1, seed: 4)
            let (keys, values) = cache.update(keys: new.keys, values: new.values)
            #expect(keys.shape == [1, Self.heads, 4, Self.headDim])
            #expect(values.shape == [1, Self.heads, 4, Self.headDim])
            #expect(cache.offsetArray.item(Int32.self) == 1)
        }

        /// A prompt longer than the ring leaves `RotatingKVCache` with more
        /// than `maxSize` slots until the next update trims them. The
        /// promoted cache must still get a buffer of `maxSize` slots and a
        /// write index inside the ring.
        @Test func promotionAfterAPromptLongerThanTheRing() {
            let rotating = RotatingKVCache(maxSize: 4)
            let prompt = Self.tokens(6, seed: 5)
            _ = rotating.update(keys: prompt.keys, values: prompt.values)
            let cache = CompilableRotatingKVCache(from: rotating)
            #expect(cache.offsetArray.item(Int32.self) == 6)
            #expect(
                cache.keys?.dim(2) == 4, "promoted buffer has maxCacheSize slots")
            #expect(
                cache.idxArray.item(Int32.self) < 4, "promoted write index is inside the ring")
        }

        // MARK: - CompilableKVCache

        @Test func compilableCacheStateOffsetAndTrim() {
            let cache = CompilableKVCache(maxLength: 8)
            #expect(cache.state.isEmpty)
            #expect(cache.innerState().count == 1, "only the offset")
            #expect(cache.isTrimmable)

            let prompt = Self.tokens(5, seed: 1)
            _ = cache.update(keys: prompt.keys, values: prompt.values)
            #expect(cache.offset == 5)
            #expect(cache.innerState().count == 3)
            #expect(cache.state.count == 2)
            #expect(SyntheticModel.maxAbsDifference(cache.state[0], prompt.keys) == 0)
            #expect(SyntheticModel.maxAbsDifference(cache.state[1], prompt.values) == 0)
            #expect(cache.debugDescription.contains("offset=5, maxLength=8"))

            // trim and a later update match KVCacheSimple.
            let simple = KVCacheSimple()
            _ = simple.update(keys: prompt.keys, values: prompt.values)
            #expect(cache.trim(2) == 2)
            #expect(simple.trim(2) == 2)
            #expect(cache.offset == 3)
            let next = Self.tokens(1, seed: 2)
            let (keys, values) = cache.update(keys: next.keys, values: next.values)
            let (simpleKeys, simpleValues) = simple.update(keys: next.keys, values: next.values)
            #expect(keys.shape == [1, Self.heads, 8, Self.headDim])
            #expect(SyntheticModel.maxAbsDifference(keys[0..., 0..., ..<4], simpleKeys) == 0)
            #expect(SyntheticModel.maxAbsDifference(values[0..., 0..., ..<4], simpleValues) == 0)
            #expect(cache.trim(10) == 4, "trim stops at offset 0")
            #expect(cache.offset == 0)

            // The offset setter replaces the counter.
            cache.offset = 2
            #expect(cache.offsetArray.item(Int32.self) == 2)
            #expect(cache.state[0].dim(2) == 2)
        }

        @Test func compilableCacheStateSetterAndFullBuffer() {
            let prompt = Self.tokens(3, seed: 7)
            let cache = CompilableKVCache(maxLength: 6)
            cache.state = [prompt.keys]
            #expect(cache.state.isEmpty, "a state without two arrays is ignored")

            cache.state = [prompt.keys, prompt.values]
            #expect(cache.offset == 3)
            #expect(cache.keys?.shape == [1, Self.heads, 6, Self.headDim])
            #expect(SyntheticModel.maxAbsDifference(cache.state[0], prompt.keys) == 0)
            #expect(SyntheticModel.maxAbsDifference(cache.state[1], prompt.values) == 0)

            // When the buffer is full, the state is the whole buffer.
            let rest = Self.tokens(3, seed: 8)
            _ = cache.update(keys: rest.keys, values: rest.values)
            #expect(cache.offset == 6)
            #expect(cache.state[0].dim(2) == 6)
            #expect(
                SyntheticModel.maxAbsDifference(
                    cache.state[0], concatenated([prompt.keys, rest.keys], axis: 2)) == 0)
        }

        @Test func compilableCachePromotion() {
            let empty = CompilableKVCache(from: KVCacheSimple(), maxLength: 4)
            #expect(empty.keys == nil)
            #expect(empty.offset == 0)

            let simple = KVCacheSimple()
            let prompt = Self.tokens(2, seed: 9)
            _ = simple.update(keys: prompt.keys, values: prompt.values)
            let promoted = CompilableKVCache.promote(from: simple, maxLength: 16)
            #expect(promoted.maxLength == 16)
            #expect(promoted.offset == 2)
            #expect(promoted.keys?.dim(2) == 16)
            #expect(SyntheticModel.maxAbsDifference(promoted.state[0], prompt.keys) == 0)
        }

        /// A copy must be independent of the original, like the copies of
        /// the other caches.
        @Test func compilableCacheCopyIsIndependent() {
            let cache = CompilableKVCache(maxLength: 8, step: 4)
            let prompt = Self.tokens(3, seed: 11)
            _ = cache.update(keys: prompt.keys, values: prompt.values)
            let copy = cache.copy() as! CompilableKVCache
            #expect(copy.maxLength == 8)
            #expect(copy.step == 4)
            #expect(copy.offset == 3)

            let next = Self.tokens(1, seed: 12)
            _ = copy.update(keys: next.keys, values: next.values)
            #expect(copy.offset == 4)
            withKnownIssue(
                """
                CompilableKVCache.copy() shares the keys, values and offsetArray objects \
                with the original. update() changes them in place with _updateInternal, \
                so an update of the copy also changes the original.
                """
            ) {
                #expect(cache.offset == 3, "original offset after copy update")
                #expect(
                    SyntheticModel.maxAbs(cache.keys![0..., 0..., 3 ..< 4]) == 0,
                    "original keys after copy update")
            } matching: {
                $0.isFailedExpectation([
                    "original offset after copy update", "original keys after copy update",
                ])
            }
        }

        // MARK: - Dynamic slices

        @Test func dynamicSliceReadsAndWritesAtAnArrayStart() {
            let source = MLXArray(Int32(0) ..< Int32(12)).reshaped(3, 4)
            let slice = dynamicSlice(
                source, start: MLXArray([Int32(1)]), axes: [1], sliceSize: [3, 2])
            #expect(slice.shape == [3, 2])
            #expect(slice.asArray(Int32.self) == [1, 2, 5, 6, 9, 10])

            let rows = dynamicSlice(
                source, start: MLXArray([Int32(2)]), axes: [0], sliceSize: [1, 4])
            #expect(rows.asArray(Int32.self) == [8, 9, 10, 11])

            let update = MLXArray([Int32(-1), -2, -3]).reshaped(3, 1)
            let written = dynamicSliceUpdate(
                source, update: update, start: MLXArray([Int32(3)]), axes: [1])
            #expect(written.shape == [3, 4])
            #expect(
                written.asArray(Int32.self) == [0, 1, 2, -1, 4, 5, 6, -2, 8, 9, 10, -3])
            #expect(
                source.asArray(Int32.self) == (0 ..< 12).map { Int32($0) },
                "the source is not changed")
        }

        /// In a compiled function the start is a graph input, so one compiled
        /// function reads a different slice for each start value.
        @Test func dynamicSliceFollowsTheStartInACompiledFunction() {
            let read = compile { (source: MLXArray, start: MLXArray) -> MLXArray in
                dynamicSlice(source, start: start, axes: [0], sliceSize: [2])
            }
            let source = MLXArray(Int32(10) ..< Int32(16))
            for start in 0 ..< 4 {
                let result = read(source, MLXArray([Int32(start)]))
                #expect(
                    result.asArray(Int32.self) == [Int32(10 + start), Int32(11 + start)],
                    "start \(start)")
            }
        }
    }
}
