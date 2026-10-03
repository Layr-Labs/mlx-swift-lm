import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// Regression tests for the compressor state of DeepSeek V4 across calls.
///
/// A prompt in chunks and decode steps must give the logits of the full pass.
/// The pooling cache must keep two things between calls: the buffered rows
/// of a window that a chunk does not fill, and, for the overlap compressor of
/// ratio 4, the last full window, which the next window mixes in.
///
/// The test `chunkedPrefillAndDecodeMatchOnePrefill` is copied from
/// `DeepseekV4ForwardPassTests` of PR #184, without its `withKnownIssue`
/// block. It also needs the pool mask fix of issue #194.
@Suite
struct DeepseekV4CompressorStateTests {

    typealias Tiny = DeepseekV4TinyModel

    // The paths run the same float32 math, with sums in another order. A
    // state fault gives differences above 1e-2.
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

    /// The compressed caches of `makeCache(parameters:)`: a prompt in chunks
    /// and decode steps give the same logits as the same prompt in one chunk
    /// with a new cache.
    @Test func chunkedPrefillAndDecodeMatchOnePrefill() throws {
        let model = try Tiny.make()
        let rows = [Tiny.row(1, count: 136)]
        let whole = Self.chunkedLogits(
            model, rows: rows, chunks: [136], cache: Self.compressedCache(model))
        let chunked = Self.chunkedLogits(
            model, rows: rows, chunks: [64, 66, 1, 1, 1, 1, 1, 1],
            cache: Self.compressedCache(model))
        let difference = Tiny.maxAbsDifference(chunked, whole)
        #expect(difference <= Self.tolerance, "differs by \(difference)")
    }

    /// Each chunk with the compressed caches gives the logits of the same
    /// positions in the full pass without a cache. A chunk of 1 is a decode
    /// step.
    ///
    /// - ratios `[0, 4, 0]`, chunks `[64, 72]`: the first window of the second
    ///   chunk must mix in the last window of the first chunk (overlap).
    /// - ratios `[0, 128, 0]`, chunks `[3, 5, 128]`: the chunk of 5 does not
    ///   fill a window, so the buffer must keep the 3 earlier rows.
    /// - the full model with both.
    @Test(arguments: [
        ([0, 4, 0], [64, 72]),
        ([0, 128, 0], [3, 5, 128]),
        ([0, 4, 128], [64, 66, 1, 1, 1, 1, 1, 1]),
        ([0, 4, 128], [3, 5, 120, 4, 1, 1, 1, 1]),
    ])
    func eachChunkMatchesTheFullForwardPass(ratios: [Int], chunks: [Int]) throws {
        let model = try Tiny.make(["compress_ratios": ratios])
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
}
