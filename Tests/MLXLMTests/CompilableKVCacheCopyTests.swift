import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

/// Regression tests for `CompilableKVCache.copy()`: the copy shared the
/// keys, values and offset arrays with the original, and `update()` changes
/// them in place, so an update of the copy also changed the original.
///
/// The first test is copied from `CompilableCacheTests` of PR #240, without
/// the known issue.
@Suite
struct CompilableKVCacheCopyTests {

    static let heads = 2
    static let headDim = 8

    /// A copy must be independent of the original, like the copies of the
    /// other caches.
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
        #expect(cache.offset == 3, "original offset after copy update")
        #expect(
            Self.maxAbs(cache.keys![0..., 0..., 3 ..< 4]) == 0,
            "original keys after copy update")
    }

    /// An update of the original must not change the copy. The copy keeps
    /// the values of the prompt.
    @Test func originalUpdateDoesNotChangeTheCopy() {
        let cache = CompilableKVCache(maxLength: 8)
        let prompt = Self.tokens(3, seed: 13)
        _ = cache.update(keys: prompt.keys, values: prompt.values)
        let copy = cache.copy() as! CompilableKVCache

        let next = Self.tokens(2, seed: 14)
        _ = cache.update(keys: next.keys, values: next.values)
        #expect(cache.offset == 5)
        #expect(copy.offset == 3, "copy offset after original update")
        #expect(
            Self.maxAbs(copy.values![0..., 0..., 3 ..< 5]) == 0,
            "copy values after original update")
        #expect(
            Self.maxAbsDifference(copy.state[0], prompt.keys) == 0, "copy keeps the prompt keys")
    }

    /// Random keys and values for `count` new tokens.
    /// Copied from `CompilableCacheTests.tokens` of PR #240.
    private static func tokens(_ count: Int, seed: UInt64) -> (keys: MLXArray, values: MLXArray) {
        let shape = [1, heads, count, headDim]
        let keys = MLXRandom.normal(shape, key: MLXRandom.key(seed * 3))
        let values = MLXRandom.normal(shape, key: MLXRandom.key(seed * 3 + 1))
        eval(keys, values)
        return (keys, values)
    }

    /// The largest absolute value of an array, as a float.
    /// Copied from `SyntheticModel.maxAbs` of PR #240.
    private static func maxAbs(_ a: MLXArray) -> Float {
        abs(a.asType(.float32)).max().item(Float.self)
    }

    /// The largest absolute difference between two arrays, as a float.
    /// Copied from `SyntheticModel.maxAbsDifference` of PR #240.
    private static func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
}
