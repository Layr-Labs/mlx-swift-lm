// CBv2MeasuredStackDefaultsTests.swift
//
// The EMPTY environment must equal the measured stack.
//
// This file is the add/add resolution of two versions of the same idea, and
// the resolution is their UNION, not a choice between them.
//
// The base PR ships the trimmed shape: every REJECTED port switch had its code
// deleted, and what remains are the KEPT levers, each a kill switch that
// DEFAULTS ON. Its suite pins those.
//
// PR 2 (MTP on the serial path) started from the untrimmed tree, where the
// rejected candidates were still readable switches flipped to their measured
// value; each flip needed a pin so a build could not lose it in silence. On
// this stacked branch those readers are GONE, so sixteen of those pins went
// with them: a switch that does not exist cannot be set wrong, and a pin
// asserting that a variable nobody reads is still unread is a test that can
// never fail. What survives is the second suite below — the MTP switches this
// stack genuinely ships ON or OFF by measurement rather than by absence.
//
// Each assertion is skipped when its variable is set, so a control arm that
// disarms one switch does not fail the suite.
//
// None of these keys exists on upstream mlx-swift-lm `main`.

import Foundation
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Assert a default only when the operator has not overridden it.
private func pinDefault(
    _ key: String, _ actual: @autoclosure () -> Bool, _ expected: Bool = true,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    guard ProcessInfo.processInfo.environment[key] == nil else { return }
    #expect(actual() == expected, "\(key)", sourceLocation: sourceLocation)
}

@Suite("The empty environment equals the measured Gemma 4 serial stack")
struct CBv2MeasuredStackDefaultsTests {

    // MARK: - Serial levers folded in from feat/gemma4-serial-levers-ringfold

    @Test("the ring-read fold and the two dense/residency levers are on by default")
    func serialLeverDefaults() {
        // RING-READ-FOLD-B1: +3.0% decode on the serial binary, and +2.04% on
        // the MTP binary re-measured as a one-variable A/B with the engage
        // mark present in the ON arm and absent in the OFF arm.
        pinDefault("DARKBLOOM_CBV2_RING_READ_FOLD_B1", CBv2RingReadFoldB1.enabled)
        // GATEUP-DENSE-CONCAT (+0.56% decode).
        pinDefault("DARKBLOOM_GEMMA4_DENSE_GATEUP_CONCAT", gemma4DenseGateUpConcatEnabled)
        // RESIDENCY-001 (+0.43% decode): unset resolves to the device maximum.
        guard ProcessInfo.processInfo.environment["DARKBLOOM_METAL_RESIDENCY_SET"] == nil
        else { return }
        #expect(CBv2MetalResidencySetV1.setting(from: nil) == .deviceMaximum)
    }

    // MARK: - Decode-side levers that carry the measured +8.6%

    @Test("the decode any-rows / any-batch levers are on by default")
    func decodeLeverDefaults() {
        // FUSED-LAYER-GLUE-ANY-ROWS (+6.2%): the largest decode lever.
        pinDefault(
            "DARKBLOOM_GEMMA4_FUSED_LAYER_GLUE_ANY_ROWS", Gemma4FusedLayerGlue.anyRowsEnabled)
        // QKV-NORM-ANY-BATCH + ROUTER-ANY-ROWS (+1.1% together).
        pinDefault("DARKBLOOM_GEMMA4_QKV_NORM_ANY_BATCH", gemma4QKVNormAnyBatchEnabled)
        pinDefault("DARKBLOOM_GEMMA4_ROUTER_ANY_ROWS", Gemma4RouterFinalistsV1.anyRowsEnabled)
        // COMPACT-DECODE-ROOTS (+1.7%).
        pinDefault(
            "DARKBLOOM_CBV2_COMPACT_DECODE_ROOTS", resolveCBv2CompactDecodeRootsEnabled(nil))
        // DECODE-LADDER (submission overlap): base admission and any-batch lift.
        pinDefault(
            "DARKBLOOM_GEMMA4_DECODE_ASYNC_EVAL_LADDER",
            resolveGemma4DecodeAsyncEvalLadderEnabled(nil))
        pinDefault(
            "DARKBLOOM_GEMMA4_DECODE_LADDER_ANY_BATCH",
            resolveGemma4DecodeLadderAnyBatchEnabled(nil))
    }

    // MARK: - Prefill masters (batch-agnostic composite, kept as engaged)

    @Test("the prefill fusion masters are on by default")
    func prefillMasterDefaults() {
        pinDefault("DARKBLOOM_CBV2_PREFILL_SDPA_COMPOSE", CBv2ComposedPrefillSDPAV1.enabled)
        pinDefault("DARKBLOOM_GEMMA4_PREFILL_GLUE", Gemma4PrefillGlueV1.enabled)
    }
}

@Suite("The empty environment equals the measured MTP stack")
struct CBv2MeasuredMTPStackDefaultsTests {

    /// The MTP switches that ship ON, each measured at THE TEST.
    ///
    /// The ring-read fold is the third one, and it is pinned by the serial
    /// suite above rather than twice here: it is a serial-path lever that the
    /// MTP binary re-measured, not an MTP switch.
    @Test("the measured MTP switches are on by default")
    func mtpDefaults() {
        // Compact verify roots: +1.2% at width 4, 7 of 7 completions
        // token-identical.
        pinDefault(
            "DARKBLOOM_CBV2_MTP_COMPACT_ROOTS",
            resolveCBv2MTPCompactRootsEnabled(nil), true)
        // Verify-work lever 2: byte-identical digest, +0.5% decode.
        pinDefault(
            "DARKBLOOM_GEMMA4_DRAFTER_NORM_RESIDUAL_FUSE",
            gemma4DrafterNormResidualFuseEnabled, true)
    }

    /// The fused verify is two switches that must be read together.
    ///
    /// `fusesVerifyAttention(width:)` ANDs the master switch with the crossover
    /// width. A build that moved the crossover to 2 and left the master switch
    /// OFF never fused, and its defaults arm read 152.0 tok/s at width 4
    /// instead of about 165. Pinning one half would permit that state again.
    @Test("the fused verify is armed at every width this branch runs")
    func fusedVerifyDefaults() {
        guard ProcessInfo.processInfo.environment[
            "MTPLX_MTP_FUSED_VERIFY_ATTENTION"] == nil,
            ProcessInfo.processInfo.environment[
                "MTPLX_MTP_FUSED_VERIFY_MIN_WIDTH"] == nil
        else { return }
        #expect(CBv2MTPRoundSwitches.fusedVerifyAttention)
        #expect(CBv2MTPRoundSwitches.fusesVerifyAttention(width: 2))
    }

    /// Pipelined draft submit: 138.5 -> 151.6 tok/s alone at width 4, and
    /// 153.7 -> 168.4 with the fused verify.
    @Test("pipelined draft submit is on by default")
    func pipelinedDraftSubmitDefaultsOn() {
        if ProcessInfo.processInfo.environment["MTPLX_MTP_PIPELINED_DRAFT_SUBMIT"] == nil {
            #expect(CBv2MTPRoundSwitches.pipelinedDraftSubmit)
        }
    }

    /// The union rule is OFF, and this one is not a throughput decision.
    ///
    /// It changes WHICH experts the verify rectangle gathers, so it changes the
    /// arithmetic. Every number this stack reports was measured with the rule
    /// off. "Flat" was a throughput result and says nothing about output
    /// equivalence, so it cannot license shipping a different gather than the
    /// one whose tokens were checked.
    @Test("the union verify rule is off by default")
    func unionVerifyDefaultsOff() {
        pinDefault(
            "MTPLX_MTP_UNION_VERIFY",
            SwitchGLUExpertGrouping.unionAcrossRows, false)
    }

    /// The one rejected port candidate PR 2 KEPT as a switch rather than
    /// deleting. It ships OFF, which is the value the measured stack ran.
    @Test("the logitless greedy head is off by default")
    func logitlessGreedyHeadDefaultsOff() {
        pinDefault(
            "DARKBLOOM_CBV2_LOGITSLESS_GREEDY_HEAD",
            CBv2TiedLMHeadArgmaxB1V1.enabled, false)
    }
}
