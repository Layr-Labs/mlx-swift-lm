// CBv2MeasuredStackDefaultsTests.swift
//
// The EMPTY environment must equal the measured stack.
//
// Every switch below is port-origin: none of them exists on upstream
// mlx-swift-lm `main` (checked with `git log -S<KEY> origin/main` and a direct
// `git grep <KEY> origin/main` — zero hits for all 17). Each one was measured at
// THE TEST on the Gemma 4 serial stack and REJECTED, so the shipping default is
// the measured value and the variable re-arms it for a control arm.
//
// The reason these are pinned at all: a default that disagrees with the
// measured stack is a silent regression. It produces no error, no log line and
// no failing test — only a slower number that looks like noise. Build5 shipped
// exactly that way, with the fused-verify crossover moved to 2 while the master
// switch stayed off, and the defaults arm read 152.0 tok/s instead of about 165.
// These assertions convert that class of mistake into a red test.
//
// Each assertion is skipped when its variable is set, so a control arm that
// disarms one switch does not fail the suite.

import Foundation
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("The empty environment equals the measured stack")
struct CBv2MeasuredStackDefaultsTests {

    /// Assert a default only when the operator has not overridden it.
    private func pin(
        _ key: String, _ actual: @autoclosure () -> Bool, _ expected: Bool,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        guard ProcessInfo.processInfo.environment[key] == nil else { return }
        #expect(actual() == expected, "\(key)", sourceLocation: sourceLocation)
    }

    // MARK: - Prefill fusions, all measured rejected

    @Test("the composed prefill SDPA fusions are off by default")
    func composedPrefillDefaults() {
        pin(
            "DARKBLOOM_CBV2_PREFILL_MASK_FUSE",
            CBv2ComposedPrefillSDPAV1.maskFuseEnabled, false)
        pin(
            "DARKBLOOM_CBV2_PREFILL_MASK_SYNTH",
            CBv2ComposedPrefillSDPAV1.maskSynthEnabled, false)
        pin(
            "DARKBLOOM_CBV2_PREFILL_SOFTMAX_VEC",
            CBv2PrefillSoftmaxVecV1.enabled, false)
        pin(
            "DARKBLOOM_CBV2_PREFILL_TOKENMAJOR_JOIN",
            CBv2AttentionV1.tokenMajorJoinEnabled, false)
    }

    @Test("the Gemma 4 prefill glue fusions are off by default")
    func gemma4PrefillDefaults() {
        pin(
            "DARKBLOOM_GEMMA4_PREFILL_BRANCH_PREFIX",
            Gemma4PrefillGlueV1.branchPrefixEnabled, false)
        pin(
            "DARKBLOOM_GEMMA4_PREFILL_PRENORM_GATHER",
            Gemma4PrefillGlueV1.prenormGatherEnabled, false)
        pin(
            "DARKBLOOM_GEMMA4_QKV_NORM_PREFILL",
            gemma4QKVNormPrefillEnabled, false)
        pin(
            "DARKBLOOM_GEMMA4_PREFILL_DEQ_CACHE",
            Gemma4PrefillDeqGEMMV1.cacheEnabled, false)
        pin(
            "DARKBLOOM_GEMMA4_PREFILL_GATEUP_FUSE",
            switchGateUpFusePrefillEnabled, false)
    }

    /// An Int cadence, not a switch. 0 disables the long-prefill chunked eval.
    @Test("the long-prefill chunked eval cadence is 0 by default")
    func longPrefillChunkEvalDefault() {
        guard ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PREFILL_CHUNK_EVAL_LONG"] == nil
        else { return }
        #expect(gemma4LongPrefillChunkEvalLayers == 0)
    }

    // MARK: - Decode-side levers, all measured rejected

    @Test("the decode-side levers are off by default")
    func decodeDefaults() {
        pin(
            "DARKBLOOM_GEMMA4_DECODE_GATEUP_FUSE",
            switchGateUpFuseDecodeEnabled, false)
        pin(
            "DARKBLOOM_GEMMA4_LASTQ_D512",
            CBv2RaggedComposedD512DecodeAttentionV1.lastQueryPrefillEnabled, false)
        pin(
            "DARKBLOOM_CBV2_LOGITSLESS_GREEDY_HEAD",
            CBv2TiedLMHeadArgmaxB1V1.enabled, false)
        pin(
            "DARKBLOOM_CBV2_PARALLEL_ARGMAX",
            CBv2ParallelArgMaxV1.enabled, false)
    }

    /// The q4 KV mirror's two write paths. Both are inert at the single-prompt
    /// shape, because the mirror is never allocated while served concurrency is
    /// below `mirrorReaderBatch`. They are pinned anyway: the measured stack set
    /// them to 0, and a default that only happens to be harmless is still a
    /// default that disagrees with what was measured.
    @Test("the KV mirror write paths are off by default")
    func kvMirrorDefaults() {
        pin("MLX_KV_QUANT_GPUPACK", CBv2WindowedSequenceKV.gpuPackEnabled, false)
        pin(
            "MLX_KV_QUANT_PAIRWRITE",
            CBv2WindowedSequenceKV.pairedMirrorWriteEnabled, false)
    }

    /// The admission coalescing window stays at 3 ms, unlike the sixteen port
    /// candidates around it.
    ///
    /// It is a CONCURRENT-SERVING knob, not a port candidate, and the arm that
    /// set it to 0 did not measure it doing harm. Read the gate
    /// (`EngineLoopV2`, ADMIT-COALESCE-001): the wait applies only when
    /// `runningCount == 0`, `waitingCount > 0` and
    /// `waitingCount < maxConcurrentRequests`. At the single-prompt B=1 shape
    /// that fires exactly once, before the only request starts, and costs at
    /// most 3 ms of a roughly 6.4 s run -- 0.05%, once, never touching decode.
    /// A live decode is never delayed by a late arrival.
    ///
    /// What it buys at concurrency is real: without it a plan taken the instant
    /// one submission lands admits only the submissions that won that race, and
    /// the rest pay a second prefill pass and a below-capacity mixed step.
    /// Deleting that to save 3 ms once at B=1 would be optimising the benchmark
    /// against the product.
    @Test("the admission coalescing window stays 3 ms by default")
    func admitCoalesceDefault() {
        guard ProcessInfo.processInfo.environment["DARKBLOOM_ADMIT_COALESCE_MS"] == nil
        else { return }
        #expect(EngineLoopV2.admitCoalesceWindowMS == 3)
    }

    // MARK: - Kept ON, and measured that way

    /// These three were measured and KEPT. They are pinned in the same place as
    /// the rejected ones so the whole stack reads from one file.
    @Test("the kept switches stay on by default")
    func keptDefaults() {
        pin("MLX_KV_QUANT", CBv2WindowedSequenceKV.quantEnabled, true)
        pin(
            "MLX_KV_QUANT_CONSUMER_GATE",
            CBv2WindowedSequenceKV.consumerGateEnabled, true)
        pin(
            "DARKBLOOM_CBV2_MTP_COMPACT_ROOTS",
            resolveCBv2MTPCompactRootsEnabled(nil), true)
        // The ring fold's reader arrived with the ringfold merge. Before that
        // this key was a NO-READER on the shipping branch, so setting it to 1
        // did nothing at all — the merge is what makes the approved value
        // reachable, and this pin is what says it stayed reachable.
        pin(
            "DARKBLOOM_CBV2_RING_READ_FOLD_B1",
            CBv2RingReadFoldB1.enabled, true)
        // Verify-work lever 2 (kept): the hidden-1024 drafter fuses
        // post-attention norm+residual through a 1024-shaped kernel. Measured
        // bit-exact (identical opaqueTokenDigest vs control) and decode-positive
        // (+0.5%, -0.1 ms/round) at THE TEST. Default ON; a control arm re-arms.
        pin(
            "DARKBLOOM_GEMMA4_DRAFTER_NORM_RESIDUAL_FUSE",
            gemma4DrafterNormResidualFuseEnabled, true)
    }

    /// The union rule is OFF, and this one is not a throughput decision.
    ///
    /// It changes WHICH experts the verify rectangle gathers, so it changes the
    /// arithmetic. Every number this stack reports, including the 165.2 tok/s
    /// arm and all of its parity data, was measured with the rule off. "Flat"
    /// was a throughput result; it says nothing about output equivalence, so it
    /// cannot license shipping a different gather than the one whose tokens
    /// were checked.
    @Test("the union verify rule is off by default")
    func unionVerifyDefaultsOff() {
        pin(
            "MTPLX_MTP_UNION_VERIFY",
            SwitchGLUExpertGrouping.unionAcrossRows, false)
    }
}
