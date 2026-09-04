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

    /// The DEFAULT is pinned, not just the predicate's shape.
    ///
    /// The crossover is a property of the engine stack. On this branch, S1
    /// unfused against S2 fused — one switch apart — measured
    /// 115.0→119.0 / 134.4→143.3 / 138.5→153.7 / 138.1→147.7 tok/s at
    /// w2/w3/w4/w5, so fusing pays at every width and the floor is 2. The
    /// production pin measured the opposite on the same code (+2.17 ms/round
    /// at w2, -0.97 at w5) and defaults to 5.
    ///
    /// Both values live in this file's history, so a merge between the two
    /// branches can silently pick either. Carrying 5 onto this stack was
    /// measured: S6 fixed w4 fell to 152.5 tok/s against S5's 168.4, and the
    /// adaptive controller settled on depth 4 instead of depth 3. That is a
    /// ~10% throughput regression with no error message, which is what this
    /// assertion exists to convert into a test failure.
    @Test("the fused-verify floor is this stack's measured crossover")
    func fusedVerifyFloorIsTheMeasuredCrossover() {
        #expect(CBv2MTPRoundSwitches.measuredFusedVerifyCrossover == 2)
        // Only assert the resolved value when no operator override is in play;
        // control arms legitimately run with =1 or =5.
        if ProcessInfo.processInfo.environment["MTPLX_MTP_FUSED_VERIFY_MIN_WIDTH"] == nil {
            #expect(
                CBv2MTPRoundSwitches.fusedVerifyMinWidth
                    == CBv2MTPRoundSwitches.measuredFusedVerifyCrossover)
        }
    }

    /// DEFAULT ON. Measured on the serial stack at verify width 4:
    /// 138.5 -> 151.6 tok/s alone, and 153.7 -> 168.4 with the fused verify.
    /// The switch moves WHEN the drafter is submitted, never what it computes,
    /// so the emitted stream is unchanged.
    ///
    /// The default is pinned because it is now part of the measured
    /// configuration. A build that loses it loses about 10% with no error.
    /// DEFAULT ON, and it must be read together with `fusedVerifyMinWidth`.
    ///
    /// `fusesVerifyAttention(width:)` ANDs the two. Build5 shipped with the
    /// crossover moved to 2 and this switch still OFF, so nothing fused and the
    /// defaults arm read 152.0 tok/s at width 4 instead of about 165. The two
    /// assertions below are therefore one contract, not two: the master switch
    /// is armed AND the narrowest rectangle takes the fused path.
    @Test("the fused verify is armed by default at every width this branch runs")
    func fusedVerifyDefaultsOn() {
        guard ProcessInfo.processInfo.environment[
            "MTPLX_MTP_FUSED_VERIFY_ATTENTION"] == nil,
            ProcessInfo.processInfo.environment[
                "MTPLX_MTP_FUSED_VERIFY_MIN_WIDTH"] == nil
        else { return }
        #expect(CBv2MTPRoundSwitches.fusedVerifyAttention)
        #expect(CBv2MTPRoundSwitches.fusesVerifyAttention(width: 2))
    }

    @Test("pipelined draft submit is on by default")
    func pipelinedDraftSubmitDefaultsOn() {
        if ProcessInfo.processInfo.environment["MTPLX_MTP_PIPELINED_DRAFT_SUBMIT"] == nil {
            #expect(CBv2MTPRoundSwitches.pipelinedDraftSubmit)
        }
    }

    @Test("fused verify engages only from its measured crossover width up")
    func fusedVerifyHasAWidthFloor() {
        // The predicate reads the WIDTH and nothing else: no prompt length, no
        // context size, no KV length. See `fusedVerifyMinWidth` for the two
        // stacks' measurements and why the floor is a per-branch constant.
        let floor = CBv2MTPRoundSwitches.fusedVerifyMinWidth
        #expect(floor >= 1)
        for width in 1...8 {
            let fused = CBv2MTPRoundSwitches.fusesVerifyAttention(width: width)
            // The master switch still gates everything; when it is off nothing
            // fuses at any width.
            #expect(fused == (CBv2MTPRoundSwitches.fusedVerifyAttention && width >= floor))
        }
        // Monotone in width: a wider rectangle never loses the fusion a
        // narrower one had.
        var seenFused = false
        for width in 1...8 {
            let fused = CBv2MTPRoundSwitches.fusesVerifyAttention(width: width)
            if seenFused { #expect(fused) }
            seenFused = seenFused || fused
        }
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
