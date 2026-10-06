import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

/// Regression tests for the window mask and the promotion of
/// `CompilableRotatingKVCache`.
///
/// - The window mask must keep the slot of the current token. It must not
///   keep one token too old.
/// - The promotion of a source buffer that is longer than the ring must give
///   a buffer of `maxSize` slots and a write index inside the ring.
///
/// The first two tests are copied from `CompilableCacheTests` of PR #240,
/// without the known issues.
@Suite
struct CompilableRotatingKVCacheRingTests {

    static let heads = 2
    static let headDim = 8

    // Tolerance of the attention comparisons: the two sides attend over the
    // same keys. The compilable side also has masked slots, which add exact
    // zeros to the softmax sums. The error is near 1e-7.
    static let tolerance: Float = 1e-5

    /// A sliding window must keep the current query position and the
    /// `window - 1` positions before it, like `createCausalMask` and the
    /// `RotatingKVCache` mask.
    @Test func windowedMaskKeepsTheCurrentPosition() {
        let cache = CompilableRotatingKVCache(maxSize: 8)
        let prompt = RingHelpers.tokens(5, seed: 3)
        _ = cache.update(keys: prompt.keys, values: prompt.values)
        let values = RingHelpers.maskValues(
            cache.makeMask(n: 1, windowSize: 3, returnArray: false))
        #expect(values?.count == 8)
        // The query at position 5 attends to positions 3, 4 and 5.
        let reference = createCausalMask(n: 1, offset: 5, windowSize: 3).asArray(Bool.self)
        #expect(reference == [false, false, false, true, true, true])
        #expect(values == reference + [false, false], "window keeps the current position")
    }

    /// A prompt longer than the ring leaves `RotatingKVCache` with more than
    /// `maxSize` slots until the next update trims them. The promoted cache
    /// must still get a buffer of `maxSize` slots and a write index inside
    /// the ring.
    @Test func promotionAfterAPromptLongerThanTheRing() {
        let rotating = RotatingKVCache(maxSize: 4)
        let prompt = RingHelpers.tokens(6, seed: 5)
        _ = rotating.update(keys: prompt.keys, values: prompt.values)
        let cache = CompilableRotatingKVCache(from: rotating)
        #expect(cache.offsetArray.item(Int32.self) == 6)
        #expect(cache.keys?.dim(2) == 4, "promoted buffer has maxCacheSize slots")
        #expect(
            cache.idxArray.item(Int32.self) < 4, "promoted write index is inside the ring")
    }

    /// After the promotion of a prompt longer than the ring, the promoted
    /// cache and the source cache take the same decode steps. Each step must
    /// write the same ring slots and give the same attention output.
    @Test(arguments: [0, 2])
    func longPromptPromotionDecodesLikeTheRotatingCache(keep: Int) {
        let maxSize = 5
        let rotating = RotatingKVCache(maxSize: maxSize, keep: keep)
        let prompt = RingHelpers.tokens(8, seed: 6)
        _ = rotating.update(keys: prompt.keys, values: prompt.values)
        let compilable = CompilableRotatingKVCache(from: rotating)
        #expect(compilable.keys?.shape == [1, Self.heads, maxSize, Self.headDim])
        #expect(compilable.idxArray.item(Int32.self) == Int32(keep))

        for step in 0 ..< 7 {
            let new = RingHelpers.tokens(1, seed: UInt64(30 + step))
            let mask = compilable.makeMask(n: 1, windowSize: nil, returnArray: false)
            let (rotatingKeys, rotatingValues) = rotating.update(
                keys: new.keys, values: new.values)
            let (keys, values) = compilable.update(keys: new.keys, values: new.values)

            #expect(
                RingHelpers.maxAbsDifference(keys, rotatingKeys) == 0, "step \(step): keys")
            #expect(
                RingHelpers.maxAbsDifference(values, rotatingValues) == 0,
                "step \(step): values")
            #expect(
                compilable.offsetArray.item(Int32.self) == Int32(rotating.offset),
                "step \(step): offset")

            let output = RingHelpers.attention(new.queries, keys, values, mask: mask)
            let expected = RingHelpers.attention(
                new.queries, rotatingKeys, rotatingValues, mask: .none)
            #expect(
                RingHelpers.maxAbsDifference(output, expected) <= Self.tolerance,
                "step \(step): attention")
        }
    }

    /// With a window smaller than the ring, each decode step must attend to
    /// the same tokens as `RotatingKVCache` with its window mask, before and
    /// after the ring wraps.
    @Test func windowedDecodeMatchesTheRotatingCache() {
        let maxSize = 6
        let window = 3
        let rotating = RotatingKVCache(maxSize: maxSize)
        let prompt = RingHelpers.tokens(2, seed: 7)
        _ = rotating.update(keys: prompt.keys, values: prompt.values)
        let compilable = CompilableRotatingKVCache(from: rotating)

        for step in 0 ..< 10 {
            let new = RingHelpers.tokens(1, seed: UInt64(40 + step))
            let mask = compilable.makeMask(n: 1, windowSize: window, returnArray: false)
            let rotatingMask = rotating.makeMask(n: 1, windowSize: window, returnArray: false)
            let (rotatingKeys, rotatingValues) = rotating.update(
                keys: new.keys, values: new.values)
            let (keys, values) = compilable.update(keys: new.keys, values: new.values)

            let output = RingHelpers.attention(new.queries, keys, values, mask: mask)
            let expected = RingHelpers.attention(
                new.queries, rotatingKeys, rotatingValues, mask: rotatingMask)
            #expect(
                RingHelpers.maxAbsDifference(output, expected) <= Self.tolerance,
                "step \(step): attention")
        }
    }
}

/// Helpers for this file. They are copied from `CompilableCacheTests` and
/// `SyntheticModel` of PR #240 and are private, so that this file does not
/// depend on that PR.
private enum RingHelpers {

    /// Random keys, values and queries for `count` new tokens.
    static func tokens(_ count: Int, seed: UInt64)
        -> (keys: MLXArray, values: MLXArray, queries: MLXArray)
    {
        let shape = [
            1, CompilableRotatingKVCacheRingTests.heads, count,
            CompilableRotatingKVCacheRingTests.headDim,
        ]
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

    /// The largest absolute difference between two arrays, as a float.
    static func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
}
