import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// Regression test for issue #195, defect 3: a prompt longer than the
/// sliding window must stay causal.
///
/// Copied from `Gemma3nTextForwardPassTests` in PR #184 without the
/// `withKnownIssue` block.
@Suite
struct Gemma3nTextSlidingMaskTests {

    typealias Tiny = Gemma3nTextTinyModel

    /// The sliding window is 4, shorter than the prompt of 11 tokens, so the
    /// sliding layers get an array mask. A change to the token at position 6
    /// must not change the logits before position 6, and must change the
    /// logits at position 6.
    @Test func promptLongerThanTheWindowStaysCausal() throws {
        let model = try Tiny.make(["sliding_window": 4])
        let row = Tiny.row(1)
        let position = 6
        var changed = row
        changed[position] = (row[position] + 1) % Tiny.vocabularySize
        let original = Tiny.logits(model, [row])
        let modified = Tiny.logits(model, [changed])

        let before = Tiny.maxAbsDifference(
            original[0..., ..<position], modified[0..., ..<position])
        let after = Tiny.maxAbsDifference(
            original[0..., position...], modified[0..., position...])
        #expect(before <= Tiny.tolerance, "positions before \(position) changed by \(before)")
        #expect(after > 1e-3, "the change at \(position) must change its own logits")
    }

    /// The decode steps read a rotating cache of 4 slots, so they see only
    /// the last 4 positions. The full pass must mask the prompt to the same
    /// window, and so give the same logits.
    @Test func promptLongerThanTheWindowMatchesTheCachedPass() throws {
        let model = try Tiny.make(["sliding_window": 4])
        let rows = [Tiny.row(1)]
        let difference = Tiny.maxAbsDifference(
            Tiny.chunkedLogits(model, rows, chunks: [5, 3, 1, 1, 1]),
            Tiny.logits(model, rows))
        #expect(difference <= Tiny.tolerance, "differs by \(difference)")
    }
}
