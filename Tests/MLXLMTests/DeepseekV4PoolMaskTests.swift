import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// Regression tests for the pool mask of DeepSeek V4 (issue #194, defects 1
/// and 2).
///
/// A query must not see a pooled window that ends after it. This must hold
/// with a cache (the prompt in one chunk or in many) and without a cache (the
/// full pass). The reference is mlx-lm `deepseek_v41.py:450`.
///
/// The tests `compressedCacheMatchesTheFullForwardPass` and
/// `aLaterTokenDoesNotChangeEarlierLogits` are copied from
/// `DeepseekV4ForwardPassTests` of PR #184, without their `withKnownIssue`
/// blocks.
@Suite
struct DeepseekV4PoolMaskTests {

    typealias Tiny = DeepseekV4TinyModel

    // The paths run the same float32 math, with sums in another order. A mask
    // fault gives differences above 1e-2.
    static let tolerance: Float = 1e-4

    static func compressedCache(_ model: DeepseekV4Model) -> [KVCache] {
        model.makeCache(parameters: GenerateParameters())
    }

    /// Runs `rows` in `chunks` through `cache` and returns the logits of every
    /// position.
    static func chunkedLogits(
        _ model: DeepseekV4Model, rows: [[Int]], chunks: [Int], cache: [KVCache]
    ) -> MLXArray {
        var start = 0
        var parts: [MLXArray] = []
        for chunk in chunks {
            let part = rows.map { Array($0[start ..< start + chunk]) }
            parts.append(Tiny.logits(model, part, cache: cache))
            start += chunk
        }
        return concatenated(parts, axis: 1)
    }

    /// Changes the token at `position` and checks that the logits before
    /// `position` do not change, and that the logits at `position` change.
    static func checkCausality(
        _ model: DeepseekV4Model, row: [Int], position: Int,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        var changed = row
        changed[position] = (row[position] + 1) % Tiny.vocabularySize
        let original = Tiny.logits(model, [row])
        let modified = Tiny.logits(model, [changed])
        let before = Tiny.maxAbsDifference(
            original[0..., ..<position], modified[0..., ..<position])
        let after = Tiny.maxAbsDifference(
            original[0..., position...], modified[0..., position...])
        #expect(
            before <= tolerance, "positions before \(position) changed by \(before)",
            sourceLocation: sourceLocation)
        #expect(
            after > 1e-3, "the change at \(position) must change its own logits",
            sourceLocation: sourceLocation)
    }

    /// The mask uses floor division. Positions 2 to 6 with ratio 4: position 2
    /// sees no window, because window 0 ends at position 3. Positions 3 to 6
    /// see window 0 only. True division gives position 2 window 0 too.
    @Test func pooledWindowMaskUsesFloorDivision() {
        let mask = pooledWindowMask(queryCount: 5, offset: 2, poolCount: 3, ratio: 4)
        let expected: [Bool] = [
            false, false, false,
            true, false, false,
            true, false, false,
            true, false, false,
            true, false, false,
        ]
        #expect(mask.shape == [5, 3])
        #expect(mask.asArray(Bool.self) == expected)
    }

    /// The full pass without a cache matches the pass with the compressed
    /// caches of `makeCache(parameters:)`.
    @Test func compressedCacheMatchesTheFullForwardPass() throws {
        let model = try Tiny.make()
        let rows = [Tiny.row(1, count: 136)]
        let full = Tiny.logits(model, rows)
        // The first queries see no pooled window, on the sparse top-k path.
        // Their pooled scores are all -inf, which must not give NaN.
        #expect(isFinite(full).all().item(Bool.self), "a logit of the full pass is not finite")
        let cached = Self.chunkedLogits(
            model, rows: rows, chunks: [136], cache: Self.compressedCache(model))
        #expect(isFinite(cached).all().item(Bool.self), "a cached logit is not finite")
        let difference = Tiny.maxAbsDifference(cached, full)
        #expect(difference <= Self.tolerance, "differs by \(difference)")
    }

    /// The cache consistency check that the mask defect blocked, on the
    /// compressed layer of ratio 128 (no overlap): each chunk with the
    /// compressed caches gives the logits of the same positions in the full
    /// pass without a cache. A chunk of 1 is a decode step. The second chunk
    /// holds positions 64 to 129, and the pooled window 0 closes at position
    /// 127, so the queries before it must not see it.
    ///
    /// The same check on the full model (with the ratio-4 layer) also needs
    /// the fix of the compressor state across calls (a separate pull request).
    @Test(arguments: [[64, 66, 1, 1, 1, 1, 1, 1], [8, 120, 4, 1, 1, 1, 1]])
    func eachChunkMatchesTheFullForwardPass(chunks: [Int]) throws {
        let model = try Tiny.make(["compress_ratios": [0, 128, 0]])
        let rows = [Tiny.row(1, count: 136)]
        let full = Tiny.logits(model, rows)
        let cache = Self.compressedCache(model)
        var start = 0
        for chunk in chunks {
            let part = rows.map { Array($0[start ..< start + chunk]) }
            let stepped = Tiny.logits(model, part, cache: cache)
            let difference = Tiny.maxAbsDifference(
                stepped, full[0..., start ..< start + chunk, 0...])
            #expect(
                difference <= Self.tolerance,
                "positions \(start) ..< \(start + chunk): cached logits differ by \(difference)")
            start += chunk
        }
    }

    @Test func aLaterTokenDoesNotChangeEarlierLogits() throws {
        let model = try Tiny.make()
        // 3 tokens give no pooled window. 12 tokens give 3 windows of 4.
        Self.checkCausality(model, row: Tiny.row(1, count: 3), position: 2)
        Self.checkCausality(model, row: Tiny.row(1, count: 12), position: 9)
    }

    /// The causality check for the ratio-128 layer: position 100 is in the
    /// first window of 128 tokens, which the queries before position 127 must
    /// not see.
    @Test func aLaterTokenDoesNotChangeEarlierLogitsThroughTheLongWindow() throws {
        let model = try Tiny.make()
        Self.checkCausality(model, row: Tiny.row(1, count: 136), position: 100)
    }
}
