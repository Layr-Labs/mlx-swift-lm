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

    /// The admission coalescing window, in milliseconds. The measured stack ran
    /// 0, so a request is admitted when it arrives rather than after a 3 ms
    /// wait for company.
    @Test("the admission coalescing window is 0 ms by default")
    func admitCoalesceDefault() {
        guard ProcessInfo.processInfo.environment["DARKBLOOM_ADMIT_COALESCE_MS"] == nil
        else { return }
        #expect(EngineLoopV2.admitCoalesceWindowMS == 0)
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
    }
}
