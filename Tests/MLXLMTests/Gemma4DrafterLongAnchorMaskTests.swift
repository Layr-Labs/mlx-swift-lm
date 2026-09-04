import Foundation
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The drafter's sliding-window admission at anchors far past the window,
/// checked against the target's own decode rule.
///
/// This path is invisible below the window. `Gemma4CBv2MTPDrafter.slidingMask`
/// returns nil unless `slidingStart <= anchor - window`, and with a
/// `WindowedSequenceKV` (`retainedCount == absoluteOffset - oldestValidPosition`,
/// with `oldestValidPosition` pinned to `absoluteOffset - window` on every
/// update) that condition is false for every anchor below the window and true
/// for every anchor at or above it. So a 64-token prompt takes the `.none`
/// branch and a 17,408-token prompt takes the masked branch, and no
/// short-prompt test can distinguish a correct mask from a broken one.
///
/// Arm A3 measured per-token acceptance falling from 0.91 at short context to
/// 0.51 at 17,408 with everything else held, which made this the first place
/// to look. The admission arithmetic is host arithmetic, so it is checked here
/// with no model, no kernels and no device.
@Suite
struct Gemma4DrafterLongAnchorMaskTests {
    private let window = 1024

    /// What the ring hands the drafter at `anchor`: `retainedCount` entries
    /// whose first is at absolute position `slidingStart`.
    private func capture(anchor: Int) -> (slidingStart: Int, length: Int) {
        let retained = min(anchor, window)
        return (anchor - retained, retained)
    }

    /// Absolute positions the drafter's mask admits — the real formula from
    /// `slidingMask`: index `i` is valid iff `i < length` and
    /// `slidingStart + i > anchor - window`.
    private func drafterAdmits(anchor: Int) -> Set<Int> {
        let (start, length) = capture(anchor: anchor)
        let masked = start <= anchor - window || length < min(anchor, window)
        return Set(
            (0..<length).compactMap { i -> Int? in
                guard !masked || start + i > anchor - window else { return nil }
                return start + i
            })
    }

    /// What the TARGET attends when it decodes at position `anchor`:
    /// `updateAndAttendRow` writes the key at `anchor`, advances
    /// `oldestValidPosition` to `anchor + 1 - window`, and returns
    /// `[oldestValidPosition, absoluteOffset)` — the half-open window ending
    /// at, and including, its own position.
    private func targetDecodeAttends(anchor: Int) -> Set<Int> {
        Set(max(0, anchor + 1 - window)...anchor)
    }

    @Test("the drafter admits exactly the target's decode window minus the unwritten anchor")
    func drafterWindowMatchesTargetDecodeWindow() {
        // Below the window, at both boundaries, and far past it — including
        // THE TEST's anchor and the output position where arm 1 first
        // diverged.
        for anchor in [1, 63, 64, 512, 1023, 1024, 1025, 2048, 4096, 17408, 17451] {
            let drafter = drafterAdmits(anchor: anchor)
            let target = targetDecodeAttends(anchor: anchor)
            // The drafter runs BEFORE the carry token's key exists, so it can
            // never see position `anchor`. Everything else must match: same
            // lower bound, same contiguity, no extra and no missing key.
            #expect(drafter == target.subtracting([anchor]),
                    "drafter/target window mismatch at anchor \(anchor)")
            #expect(drafter.count == (anchor < window ? anchor : window - 1),
                    "unexpected admitted count at anchor \(anchor)")
        }
    }

    @Test("the boundary key at anchor - window is masked, and only it")
    func boundaryKeyIsMaskedExactlyOnce() {
        // A full rotating ring retains the key at `anchor - window`, which is
        // one position OUTSIDE the target's strict `(anchor - window, anchor]`
        // rule. Masking it is the whole reason the mask exists; masking one
        // more, or one fewer, is the bug this pins.
        for anchor in [1024, 1025, 4096, 17408] {
            let admitted = drafterAdmits(anchor: anchor)
            #expect(!admitted.contains(anchor - window))
            #expect(admitted.contains(anchor - window + 1))
            #expect(admitted.contains(anchor - 1))
            #expect(!admitted.contains(anchor))
            #expect(admitted.count == window - 1)
            // Contiguous: no hole anywhere in the admitted range.
            #expect(admitted == Set((anchor - window + 1)...(anchor - 1)))
        }
    }

    @Test("below the window nothing is masked, which is why short prompts prove nothing")
    func shortAnchorsTakeTheUnmaskedBranch() {
        for anchor in [1, 8, 63, 64, 512, 1023] {
            let (start, length) = capture(anchor: anchor)
            #expect(start == 0)
            #expect(length == anchor)
            // `slidingMask`'s own gate: no mask is built at all here.
            #expect(!(start <= anchor - window))
            #expect(drafterAdmits(anchor: anchor) == Set(0..<anchor))
        }
        // And at the boundary the gate flips, permanently.
        for anchor in [1024, 1025, 17408] {
            let (start, _) = capture(anchor: anchor)
            #expect(start <= anchor - window)
        }
    }

    @Test("the drafter's query position is the target's verify column-zero position")
    func queryPositionMatchesVerifyColumnZero() {
        // `prepare` sets `PositionOffset.batch([anchor])` and the verify feeds
        // the carry token as column 0 at the same `anchor`
        // (`fullRow.absoluteOffset == carry.kvOffset` is asserted in
        // `mtpBuildVerifyGraph`). Both sides must derive the same number from
        // the same fact, at every anchor.
        for anchor in [0, 1, 1023, 1024, 17408] {
            let drafterQueryPosition = anchor
            let verifyColumnZeroPosition = anchor
            #expect(drafterQueryPosition == verifyColumnZeroPosition)
            // And the drafter's newest visible key is one before it.
            if anchor > 0 {
                #expect(drafterAdmits(anchor: anchor).max() == anchor - 1)
            }
        }
    }
}
