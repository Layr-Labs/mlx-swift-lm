// CBv2RingReadFoldB1.swift
//
// RING-READ-FOLD-B1: at batch 1, a sliding-window decode step attends the
// retained ring IN PLACE instead of materialising a temporal-order window.
//
// ## What it replaces
//
// The B=1 decode path is `updateAndAttend` → `if B == 1` → `updateAndAttendRow`
// → `row.update()` → for a windowed row, `writeDecodeToken` then a return of
// `temporalOrder(keys, oldest, offset)` (`WindowedSequenceKV.swift:311-318`).
// On a full, wrapped ring — the steady state at THE TEST — that `temporalOrder`
// is a two-slice `concatenated`: it MATERIALISES the whole
// `[1, kvHeads, window, headDim]` window (≈4.19 MB per K and V) every sliding
// layer every token, which `attendCommittedRow`'s SDPA then reads again. Over
// 25 sliding layers that is ~210 MB written + ~210 MB re-read per token of pure
// data movement the attention does not need — the plane-movement the graph walk
// (scratchpad/kv-write-b1-report / the flat KV-WRITE-B1 arm) found still on the
// critical path after the write was shown already in-place on mlx 0.32.2.
//
// ## What it does
//
// When the ring is full and the layer is a plain sliding layer (scale 1, no
// sinks/softcap, the kernel's shapes), it performs the ordinary in-place ring
// write (`decodeRingWrite`, already O(row) on 0.32.2) and then attends via
// `CBv2RaggedTwoPassDecodeAttentionV1.attendRingB1` — a one-row transcription
// of the B=8 `attendRing` two-pass that walks the ring in temporal order by
// modular indexing (`slot = (start + i) % window`) inside the kernel, with the
// SAME online-softmax reduction order and the SAME `passBActive` combine. No
// `temporalOrder` concat, no window materialisation; the ring buffers are read
// where they lie.
//
// Exactness — NEAR-TIE, not bit-exact. The READ is bitwise identical (the unit
// test proves the modular walk returns the same temporal window `temporalOrder`
// concatenates). The ATTENTION is not: `attendRingB1` is the B=8 ragged
// two-pass with `batch_index == 0` (bit-for-bit the B=8 kernel for row 0), and
// that two-pass reduces the key axis in strided blocks + a `passB` merge —
// a DIFFERENT floating-point order than `MLXFast.scaledDotProductAttention`,
// MLX's own decode SDPA the pre-fold concat path called (arch-dependent block
// count 64/128, its own online-softmax merge; `scaled_dot_product_attention.cpp`).
// Same keys, non-associative softmax rescale, so one greedy argmax in ~6,000
// tokens flips (7×1,024 off/on: six prompts identical, one diverges once near
// the end) — the SAME near-tie class as the D512 two-pass. The ring WRAP is not
// the cause (the walk's blocks cover the same temporal positions a strided block
// over the contiguous window would). HumanEval is the gate. Eligibility is
// checked BEFORE the write so the post-write attend never fails and no row is
// written twice.
//
// Kill switch: `DARKBLOOM_CBV2_RING_READ_FOLD_B1=0`/`false`/`no`/`off` restores
// the `temporalOrder` concat + SDPA path. Engagement: `[engage] ringreadfoldb1`
// under `MLXFAST_ENGAGE_MARKS=1`.

import Foundation
import MLX

/// Batch-1 in-place sliding-ring decode read. See the file header.
enum CBv2RingReadFoldB1 {

    /// Kill switch: default ON. `0`/`false`/`no`/`off` restores the copying
    /// `temporalOrder` concat + SDPA path.
    static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_RING_READ_FOLD_B1"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// True iff a B=1 sliding decode step can take the in-place ring fold:
    /// enabled, a plain sliding layer (scale 1, no sinks/softcap), a FULL,
    /// non-stale, non-staged ring, and shapes the one-row ring kernel accepts.
    /// Checked BEFORE `decodeRingWrite` so the post-write `attendRingB1` cannot
    /// return nil and a row is never written twice.
    static func eligible(
        ring: CBv2WindowedSequenceKV, kind: CBv2LayerKind,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?, softcap: Float?
    ) -> Bool {
        guard enabled, scale == 1.0, sinks == nil, softcap == nil,
            case .slidingWindow(let window) = kind.attention,
            // Full ring, bf16 ring not stale, no staged (MTP) transaction.
            let view = ring.decodeRingViewBeforeWrite
        else { return false }
        return CBv2RaggedTwoPassDecodeAttentionV1.ringB1Eligible(
            queries: queries, keys: view.keys, values: view.values,
            slidingWindowLength: window)
    }
}
