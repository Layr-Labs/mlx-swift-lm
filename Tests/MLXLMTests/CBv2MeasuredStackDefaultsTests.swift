// CBv2MeasuredStackDefaultsTests.swift
//
// The EMPTY environment must equal the measured Gemma 4 serial stack.
//
// PR 1 ships the trimmed shape: every REJECTED port switch had its code
// deleted, so there is no reader left to disagree with the measured stack.
// What remains are the KEPT levers, each a kill switch that DEFAULTS ON — an
// empty environment is the measured configuration and `=0` restores the stock
// path. These assertions pin those defaults, so a silent flip to OFF (which
// produces no error and no log line, only a slower number that looks like
// noise) becomes a red test.
//
// Each assertion is skipped when its variable is set, so a control arm that
// disarms one switch does not fail the suite.
//
// None of these keys exists on upstream mlx-swift-lm `main`.

import Foundation
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("The empty environment equals the measured Gemma 4 serial stack")
struct CBv2MeasuredStackDefaultsTests {

    /// Assert a default only when the operator has not overridden it.
    private func pin(
        _ key: String, _ actual: @autoclosure () -> Bool, _ expected: Bool = true,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        guard ProcessInfo.processInfo.environment[key] == nil else { return }
        #expect(actual() == expected, "\(key)", sourceLocation: sourceLocation)
    }

    // MARK: - Serial levers folded in from feat/gemma4-serial-levers-ringfold

    @Test("the ring-read fold and the two dense/residency levers are on by default")
    func serialLeverDefaults() {
        // RING-READ-FOLD-B1 (+3.0% decode): the whole point of the fold.
        pin("DARKBLOOM_CBV2_RING_READ_FOLD_B1", CBv2RingReadFoldB1.enabled)
        // GATEUP-DENSE-CONCAT (+0.56% decode).
        pin("DARKBLOOM_GEMMA4_DENSE_GATEUP_CONCAT", gemma4DenseGateUpConcatEnabled)
        // RESIDENCY-001 (+0.43% decode): unset resolves to the device maximum.
        guard ProcessInfo.processInfo.environment["DARKBLOOM_METAL_RESIDENCY_SET"] == nil
        else { return }
        #expect(CBv2MetalResidencySetV1.setting(from: nil) == .deviceMaximum)
    }

    // MARK: - Decode-side levers that carry the measured +8.6%

    @Test("the decode any-rows / any-batch levers are on by default")
    func decodeLeverDefaults() {
        // FUSED-LAYER-GLUE-ANY-ROWS (+6.2%): the largest decode lever.
        pin("DARKBLOOM_GEMMA4_FUSED_LAYER_GLUE_ANY_ROWS", Gemma4FusedLayerGlue.anyRowsEnabled)
        // QKV-NORM-ANY-BATCH + ROUTER-ANY-ROWS (+1.1% together).
        pin("DARKBLOOM_GEMMA4_QKV_NORM_ANY_BATCH", gemma4QKVNormAnyBatchEnabled)
        pin("DARKBLOOM_GEMMA4_ROUTER_ANY_ROWS", Gemma4RouterFinalistsV1.anyRowsEnabled)
        // COMPACT-DECODE-ROOTS (+1.7%).
        pin("DARKBLOOM_CBV2_COMPACT_DECODE_ROOTS", resolveCBv2CompactDecodeRootsEnabled(nil))
        // DECODE-LADDER (submission overlap): base admission and any-batch lift.
        pin("DARKBLOOM_GEMMA4_DECODE_ASYNC_EVAL_LADDER", resolveGemma4DecodeAsyncEvalLadderEnabled(nil))
        pin("DARKBLOOM_GEMMA4_DECODE_LADDER_ANY_BATCH", resolveGemma4DecodeLadderAnyBatchEnabled(nil))
    }

    // MARK: - Prefill masters (batch-agnostic composite, kept as engaged)

    @Test("the prefill fusion masters are on by default")
    func prefillMasterDefaults() {
        pin("DARKBLOOM_CBV2_PREFILL_SDPA_COMPOSE", CBv2ComposedPrefillSDPAV1.enabled)
        pin("DARKBLOOM_GEMMA4_PREFILL_GLUE", Gemma4PrefillGlueV1.enabled)
    }
}
