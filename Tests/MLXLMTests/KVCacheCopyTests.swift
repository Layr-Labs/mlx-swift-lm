import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Regression tests for `copy()` of `ChunkedKVCache`, `ArraysCache` and
/// `MambaCache` (issue #193).
///
/// The tests are copies of `chunkedCacheCopyAfterTrimFrontContinuesLikeTheOriginal`
/// and `arraysCacheCopyKeepsSlotPositions` of `KVCacheBehaviorTests` in
/// PR #183, without the `withKnownIssue` blocks. This file adds the same
/// slot check for `MambaCache`, and checks the values of the copies. The
/// `tokens` helper is a copy of the one in `KVCacheBehaviorTests`.
@Suite
struct KVCacheCopyTests {

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

    /// A copy of a chunked cache after `maybeTrimFront()` must continue like
    /// the original.
    @Test func chunkedCacheCopyAfterTrimFrontContinuesLikeTheOriginal() {
        let cache = ChunkedKVCache(chunkSize: 8)
        cache.step = 4
        let first = Self.tokens(10, seed: 1)
        _ = cache.update(keys: first.keys, values: first.values)
        cache.maybeTrimFront()

        let copy = cache.copy()
        #expect(copy.metaState == cache.metaState, "copy meta state")
        let next = Self.tokens(1, seed: 2)
        let (keysA, valuesA) = cache.update(keys: next.keys, values: next.values)
        let (keysB, valuesB) = copy.update(keys: next.keys, values: next.values)
        #expect(copy.offset == cache.offset, "copy offset")
        #expect(keysB.shape == keysA.shape, "copy keys shape")
        #expect(keysB.shape == keysA.shape && (keysA .== keysB).all().item(Bool.self), "copy keys")
        #expect(
            valuesB.shape == valuesA.shape && (valuesA .== valuesB).all().item(Bool.self),
            "copy values")
    }

    /// A copy keeps each state in its slot, also when an earlier slot is
    /// empty.
    @Test func arraysCacheCopyKeepsSlotPositions() {
        let cache = ArraysCache(size: 3)
        cache[1] = MLXArray([1, 2] as [Float])
        let copy = cache.copy() as! ArraysCache
        #expect(copy.slotCount == 3, "slot count")
        #expect(copy.presentSlotIndices == [1], "present slots")
        #expect(copy[1]?.asArray(Float.self) == [1, 2], "slot 1 values")
    }

    /// The same for `MambaCache`: a state in slot 1 stays in slot 1.
    @Test func mambaCacheCopyKeepsSlotPositions() {
        let cache = MambaCache()
        cache[1] = MLXArray([3, 4] as [Float])
        let copy = cache.copy()
        #expect(copy is MambaCache, "copy type")
        let mamba = copy as! ArraysCache
        #expect(mamba.slotCount == 2, "slot count")
        #expect(mamba.presentSlotIndices == [1], "present slots")
        #expect(mamba[1]?.asArray(Float.self) == [3, 4], "slot 1 values")
    }
}
