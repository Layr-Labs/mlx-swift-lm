import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Regression tests for `ChunkedKVCache.maybeTrimFront()` (issue #234).
///
/// The reference (mlx-lm cache.py `ChunkedKVCache.maybe_trim_front`) counts
/// the valid tokens, `offset - start_position`. When there are more than
/// `chunk_size`, it keeps the newest `chunk_size` of them. The old Swift code
/// compared the buffer length with the chunk size and kept the last
/// `chunkSize` buffer rows. The buffer grows in steps, so its last rows can
/// be unused, and the old code kept fewer valid tokens than the reference.
@Suite
struct ChunkedKVCacheTrimTests {

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

    /// 10 tokens with chunk size 8 and step 4: the buffer has 12 rows, and
    /// the last 2 are unused. The trim must keep tokens 2 ..< 10, so the next
    /// update returns those 8 tokens and the new one.
    @Test func trimKeepsTheNewestChunkOfValidTokens() {
        let cache = ChunkedKVCache(chunkSize: 8)
        cache.step = 4
        let first = Self.tokens(10, seed: 1)
        _ = cache.update(keys: first.keys, values: first.values)
        cache.maybeTrimFront()
        #expect(cache.metaState == ["8", "2"], "start position after the trim")
        #expect(cache.offset == 10, "offset after the trim")

        let next = Self.tokens(1, seed: 2)
        let (keys, values) = cache.update(keys: next.keys, values: next.values)
        let expectedKeys = concatenated(
            [first.keys[.ellipsis, 2 ..< 10, 0...], next.keys], axis: 2)
        let expectedValues = concatenated(
            [first.values[.ellipsis, 2 ..< 10, 0...], next.values], axis: 2)
        #expect(Self.equal(keys, expectedKeys), "kept keys")
        #expect(Self.equal(values, expectedValues), "kept values")
    }

    /// 6 tokens with chunk size 8 and the default step of 256: there are
    /// fewer valid tokens than the chunk size, so the trim must do nothing,
    /// although the buffer has 256 rows.
    @Test func trimKeepsAShortCache() {
        let cache = ChunkedKVCache(chunkSize: 8)
        let first = Self.tokens(6, seed: 3)
        _ = cache.update(keys: first.keys, values: first.values)
        cache.maybeTrimFront()
        #expect(cache.metaState == ["8", "0"], "start position after the trim")
        #expect(cache.offset == 6, "offset after the trim")
        #expect(Self.equal(cache.state[0], first.keys), "kept keys")
        #expect(Self.equal(cache.state[1], first.values), "kept values")
    }
}
