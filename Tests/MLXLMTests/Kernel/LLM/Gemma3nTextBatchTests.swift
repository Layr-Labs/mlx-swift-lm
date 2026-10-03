import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// Regression tests for issue #195, defect 5: the model must run with a
/// batch of 2 rows.
///
/// Before the fix, `Gemma3nAltUp.correct` stopped the process with a
/// broadcast error for 2 rows, so PR #184 ran batch size 1 only. These are
/// the batch-2 shape and cache checks that PR #184 could not run.
@Suite
struct Gemma3nTextBatchTests {

    typealias Tiny = Gemma3nTextTinyModel

    @Test func batchOfTwoHasTheExpectedShapeAndIsFinite() throws {
        let model = try Tiny.make()
        let logits = Tiny.logits(model, [Tiny.row(1, count: 7), Tiny.row(2, count: 7)])
        #expect(logits.shape == [2, 7, Tiny.vocabularySize])
        #expect(logits.dtype == .float32)
        #expect(isFinite(logits).all().item(Bool.self), "a logit is not finite")
    }

    /// Each row of a batch of 2 gives the same logits as the row alone.
    @Test func eachRowOfABatchMatchesTheRowAlone() throws {
        let model = try Tiny.make()
        let rowA = Tiny.row(1)
        let rowB = Tiny.row(2)
        let batched = Tiny.logits(model, [rowA, rowB])
        let aloneA = Tiny.logits(model, [rowA])
        let aloneB = Tiny.logits(model, [rowB])
        let differenceA = Tiny.maxAbsDifference(batched[0 ..< 1], aloneA)
        let differenceB = Tiny.maxAbsDifference(batched[1 ..< 2], aloneB)
        #expect(differenceA <= Tiny.tolerance, "row 0 differs by \(differenceA)")
        #expect(differenceB <= Tiny.tolerance, "row 1 differs by \(differenceB)")
        #expect(
            Tiny.maxAbsDifference(aloneA, aloneB) > 1e-3,
            "the two rows must give different logits")
    }

    /// With a cache, a prompt in chunks and decode steps for a batch of 2
    /// give the same logits as the same steps for each row alone. The KV
    /// caches and the KV-shared layers take the batch of 2.
    @Test func cachedStepsOfABatchMatchTheRowsAlone() throws {
        let model = try Tiny.make()
        let rowA = Tiny.row(1)
        let rowB = Tiny.row(2)
        let chunks = [5, 3, 1, 1, 1]
        let batched = Tiny.chunkedLogits(model, [rowA, rowB], chunks: chunks)
        #expect(batched.shape == [2, 11, Tiny.vocabularySize])
        let differenceA = Tiny.maxAbsDifference(
            batched[0 ..< 1], Tiny.chunkedLogits(model, [rowA], chunks: chunks))
        let differenceB = Tiny.maxAbsDifference(
            batched[1 ..< 2], Tiny.chunkedLogits(model, [rowB], chunks: chunks))
        #expect(differenceA <= Tiny.tolerance, "row 0 differs by \(differenceA)")
        #expect(differenceB <= Tiny.tolerance, "row 1 differs by \(differenceB)")
    }
}
