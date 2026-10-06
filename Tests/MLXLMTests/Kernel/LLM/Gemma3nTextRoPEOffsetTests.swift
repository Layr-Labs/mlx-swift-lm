import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// Regression tests for issue #195, defects 1 and 2: the queries and the
/// keys of one forward pass must use the same RoPE offset.
///
/// Copied from `Gemma3nTextForwardPassTests` in PR #184 without the
/// `withKnownIssue` blocks.
@Suite
struct Gemma3nTextRoPEOffsetTests {

    typealias Tiny = Gemma3nTextTinyModel

    /// Defect 1. Without KV sharing, and with a sequence that fits in the
    /// sliding window of 16, a prompt in chunks and decode steps must give
    /// the logits of the full forward pass.
    @Test func cachedDecodeMatchesTheFullForwardPass() throws {
        let model = try Tiny.make(["num_kv_shared_layers": 0])
        let rows = [Tiny.row(1)]
        let full = Tiny.logits(model, rows)
        let cache = model.newCache(parameters: nil)
        var start = 0
        for chunk in [5, 3, 1, 1, 1] {
            let stepped = Tiny.logits(
                model, rows.map { Array($0[start ..< start + chunk]) }, cache: cache)
            let difference = Tiny.maxAbsDifference(
                stepped, full[0..., start ..< start + chunk, 0...])
            #expect(
                difference <= Tiny.tolerance,
                "positions \(start) ..< \(start + chunk): cached logits differ by \(difference)")
            start += chunk
        }
    }

    /// Defect 2. With KV sharing, the last 2 layers read the caches of
    /// layers 0 and 1. A prompt in chunks and decode steps must give the
    /// same logits as the same prompt in one chunk.
    @Test func kvSharedLayersGiveTheSameLogitsInChunks() throws {
        let model = try Tiny.make()
        let rows = [Tiny.row(1)]
        let difference = Tiny.maxAbsDifference(
            Tiny.chunkedLogits(model, rows, chunks: [5, 3, 1, 1, 1]),
            Tiny.chunkedLogits(model, rows, chunks: [11]))
        #expect(difference <= Tiny.tolerance, "differs by \(difference)")
    }
}
