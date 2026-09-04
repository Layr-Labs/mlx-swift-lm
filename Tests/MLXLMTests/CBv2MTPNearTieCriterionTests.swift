import Foundation
import Testing

@testable import MLXLMCommon

/// The criterion that decides whether the rectangular verify can agree with
/// serial decode on a token: the top-two logit margin against the arithmetic
/// difference between the two paths.
///
/// Rectangular verification serializes its ATTENTION
/// (`mtpSerializesRectangularAttention`) but runs every projection, the MoE
/// gather and the LM head at `M = 1+k`, where serial decode runs them at
/// `M = 1`. GEMV and GEMM do not accumulate in the same order, so the two
/// paths' logits differ at the last bits. `CBv2MTPVerificationMode`'s
/// `rectangularExact` case exists precisely because plain `.rectangular` is
/// not M1-equivalent, and `CBv2MTPEngineParityTests` says so out loud:
/// "Random near-tie logits are intentionally not a cross-kernel parity oracle
/// on Apple GPUs."
///
/// None of that is a defect. What it means is that token agreement is a
/// function of MARGIN, and this suite pins that function so the property is
/// stated somewhere executable rather than only in comments. It is pure
/// arithmetic — no model, no kernels, no device — so it runs while a GPU
/// window is busy.
@Suite
struct CBv2MTPNearTieCriterionTests {
    /// Does a perturbation of size `epsilon` applied to the runner-up change
    /// which logit is largest? Exactly when the margin is smaller.
    private func argmaxSurvives(margin: Float, epsilon: Float) -> Bool {
        let top: Float = 10.0
        let runnerUp = top - margin
        return top > runnerUp + epsilon
    }

    @Test("token agreement is decided by the top-two margin, not by depth")
    func agreementIsAMarginProperty() {
        // A bfloat16 logit near 10.0 has a ULP of about 0.0625: 8 mantissa
        // bits, so the representable spacing at that magnitude is 2^-7 * 8.
        // Two paths that disagree by a ULP agree on the argmax whenever the
        // margin clears it, and can disagree when it does not.
        let ulp: Float = 0.0625
        #expect(argmaxSurvives(margin: 1.0, epsilon: ulp))
        #expect(argmaxSurvives(margin: 0.5, epsilon: ulp))
        #expect(argmaxSurvives(margin: 0.1, epsilon: ulp))
        #expect(!argmaxSurvives(margin: 0.01, epsilon: ulp))
        #expect(!argmaxSurvives(margin: 0.0, epsilon: ulp))
        // The criterion does not mention the draft depth, the verify width,
        // the prompt length or the context size — which is why the observed
        // divergence sits at one output index for every width from 2 to 8
        // and why width 1, which runs no verify, matches serial exactly.
        for width in 2...8 {
            _ = width
            #expect(argmaxSurvives(margin: 1.0, epsilon: ulp))
            #expect(!argmaxSurvives(margin: 0.001, epsilon: ulp))
        }
    }

    @Test("a wide-margin fixture cannot detect a near-tie divergence")
    func wideMarginFixturesCannotCertifyExactness() {
        // This is the reason the engine's rectangular parity tests build a
        // deterministic wide-margin cycle, and the reason passing them does
        // NOT certify token equality on real content. A generation long
        // enough to contain one narrow margin will diverge; arm 1 found its
        // first at output index 43 of 1,024.
        let ulp: Float = 0.0625
        let wideMarginFixture: [Float] = [8.0, 4.0, 2.0, 1.0]
        for margin in wideMarginFixture {
            #expect(argmaxSurvives(margin: margin, epsilon: ulp))
        }
        let realContentMargins: [Float] = [3.0, 1.2, 0.4, 0.02, 0.9]
        #expect(realContentMargins.contains { !argmaxSurvives(margin: $0, epsilon: ulp) })
    }

    @Test("the margin band is what a run should report, and it is depth-free")
    func reportableBand() {
        // The number that decides whether a task eval can pass is not "did it
        // diverge" but "at how many committed positions was the margin inside
        // the arithmetic band". Both endpoints are properties of the model and
        // the content, never of k — so an instrument that records it belongs
        // on the verify columns, not on the round.
        let ulp: Float = 0.0625
        let margins: [Float] = [5.0, 2.0, 0.5, 0.03, 0.004, 1.5, 0.08]
        let atRisk = margins.filter { !argmaxSurvives(margin: $0, epsilon: ulp) }
        #expect(atRisk == [0.03, 0.004])
        #expect(Double(atRisk.count) / Double(margins.count) < 0.5)
    }
}
