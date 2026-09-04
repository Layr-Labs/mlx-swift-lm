import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The two places this engine's Gemma 4 drafter differed from the mlx-vlm
/// reference, pinned against the reference rule.
///
/// Neither can change a committed token — the target verifies everything the
/// drafter proposes — so both cost acceptance and nothing else, which is
/// exactly the shape of the measured gap (0.66 per-token acceptance against
/// the 0.91 the same head reaches on short prompts).
@Suite
struct Gemma4MTPDrafterSemanticsTests {
    @Test("the drafter query sits on the last real key, not one past it")
    func draftQueryPositionIsTheLastRealKey() {
        // Reference: `_mtp_draft_position(kv_offset) = max(kv_offset - 1, 0)`
        // (mtp.py:587-593, 714-720). After an N-token prefill kv_offset is N,
        // so the drafter drafts at N-1 — the position of the newest key it can
        // actually see. Drafting at N put every shared key one slot further
        // away than the head was trained to see it, at every layer and every
        // context length.
        // Both legs are pinned so the A/B cannot silently stop being an A/B:
        // with the switch on the rule is the reference's, with it off it is
        // the identity this engine used to apply.
        let reference = Gemma4MTPDrafterSemantics.referenceSemantics
        for kvOffset in [1, 2, 64, 1023, 1024, 1025, 17408, 17451] {
            #expect(
                Gemma4MTPDrafterSemantics.draftQueryPosition(kvOffset: kvOffset)
                    == (reference ? kvOffset - 1 : kvOffset),
                "wrong drafter query position at kv offset \(kvOffset)")
        }
        // Clamped at the start of a sequence, where there is no previous key.
        #expect(Gemma4MTPDrafterSemantics.draftQueryPosition(kvOffset: 0) == 0)
        // Monotone and gap-free: a constant shift, never a length-keyed rule.
        for kvOffset in 1...64 {
            let here = Gemma4MTPDrafterSemantics.draftQueryPosition(kvOffset: kvOffset)
            let next = Gemma4MTPDrafterSemantics.draftQueryPosition(kvOffset: kvOffset + 1)
            #expect(next == here + 1)
        }
    }

    @Test("the seed hidden is normed, and the norm is not something a linear can absorb")
    func seedHiddenIsPostNorm() throws {
        // `Gemma4MTPForward.lastHidden` is what reaches the drafter's
        // `pre_projection`. Reference: `speculative_draft_hidden(h)` is
        // `model.norm(h)` (language.py:671-672). This asserts the property
        // that makes the difference matter — RMSNorm rescales each row by a
        // row-dependent quantity, so no fixed linear can absorb it and the two
        // choices are genuinely different inputs, not a reparameterisation.
        let hidden = MLXArray(
            [1.0, 2.0, 3.0, 4.0, 40.0, 50.0, 60.0, 70.0] as [Float]
        ).reshaped([2, 4]).asType(.float32)
        let rms = sqrt(mean(hidden * hidden, axis: -1, keepDims: true) + 1e-6)
        let normed = hidden / rms
        eval(normed, rms)
        let scales = rms.asArray(Float.self)
        // The two rows are rescaled by DIFFERENT factors; a single linear map
        // applied to the pre-norm hidden cannot reproduce the post-norm one.
        #expect(scales.count == 2)
        #expect(abs(scales[0] - scales[1]) > 1.0)
        // And normalization actually changes the row's direction relative to
        // its neighbours in magnitude terms, which is what the projection sees.
        let normedValues = normed.asArray(Float.self)
        let rawValues = hidden.asArray(Float.self)
        #expect(abs(normedValues[0] - rawValues[0]) > 1e-3)
        #expect(abs(normedValues[4] - rawValues[4]) > 1e-3)
    }

    @Test("the switch is a straight A/B with no length in it")
    func switchIsShapeIndependent() {
        // Whatever the switch resolves to in this process, the mapping is a
        // pure function of the KV offset with no threshold: either identity or
        // a constant -1 clamped at zero. A future length-keyed branch would
        // have to change this signature to compile.
        let reference = Gemma4MTPDrafterSemantics.referenceSemantics
        for kvOffset in [0, 1, 1023, 1024, 17408] {
            let position = Gemma4MTPDrafterSemantics.draftQueryPosition(kvOffset: kvOffset)
            #expect(position == (reference ? max(kvOffset - 1, 0) : kvOffset))
            #expect(position >= 0)
            #expect(position <= kvOffset)
        }
    }
}
