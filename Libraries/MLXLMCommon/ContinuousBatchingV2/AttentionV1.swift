// AttentionV1.swift
//
// The v1 attention dispatch used by `CBv2LayerCache.updateAndAttend`:
// per-row `MLXFast.scaledDotProductAttention` against each sequence's own
// contiguous KV. The paged backend (workstream C) replaces this behind the
// same `CBv2AttendingLayerCache` protocol.
//
// ## Why this path cannot have the left-padding bug class
// - Decode is rectangular [B, 1]: each row attends EXACTLY its own KV, with
//   no mask at all — a fully-masked row cannot exist by construction, so
//   NaN poisoning (ee2a921) is impossible.
// - Prefill is per-request [1, chunk]: masks are per-request, derived only
//   from that request's own lengths — batch composition cannot influence
//   them.
// - The mask mode is a PURE FUNCTION of (L, returned KV length, window):
//   never data-dependent, never drifting across steps for the same logical
//   computation (MLX #3384 / report 10 §4 invariant 5).

import Foundation
import MLX

/// Namespace for the v1 (per-row SDPA) attention dispatch.
enum CBv2AttentionV1 {

    /// Mask mode for a single-request attention call.
    ///
    /// - `L == 1` (decode): `.none`. The row's retained KV IS its window —
    ///   sliding-window eviction already dropped everything outside it.
    /// - `L > 1` (prefill chunk) against `kL` returned KV entries:
    ///   - `.causal` when no window is configured, or when `kL <= window`
    ///     (the window cannot bind: the oldest returned entry is inside
    ///     every query's window).
    ///   - causal ∧ window ARRAY mask when `kL > window` (a windowed layer's
    ///     multi-token update returned pre-eviction history so early chunk
    ///     tokens see their full window; later tokens must not over-attend).
    ///
    /// Pure in (L, kL, window): the same request produces the same mask mode
    /// at the same point in its lifetime regardless of batchmates.
    static func maskMode(L: Int, kL: Int, window: Int?)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        if L == 1 { return .none }
        if let window, kL > window {
            // Relative coordinates: keys span [0, kL), queries are the last
            // L positions. Window comparisons are translation-invariant, so
            // the absolute offset is irrelevant.
            return .array(createCausalMask(n: L, offset: kL - L, windowSize: window))
        }
        return .causal
    }

    /// Update each row with this step's K/V and attend.
    ///
    /// - queries/keys/values: `[B, heads, L, headDim]` with `L == 1` for
    ///   decode (B == rows.count) or `B == 1` for a prefill chunk.
    /// - Returns `[B, queryHeads, L, headDim]`.
    static func updateAndAttend(
        rows: [CBv2SequenceKV], kind: CBv2LayerKind,
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        let B = queries.dim(0)
        let L = queries.dim(2)
        precondition(!rows.isEmpty, "CBv2AttentionV1: no rows")
        precondition(
            B == rows.count,
            "CBv2AttentionV1: batch \(B) != rows \(rows.count) — prefill must run per-request [1, chunk]"
        )
        precondition(
            B == 1 || L == 1,
            "CBv2AttentionV1: ragged shapes are impossible in v2 — decode is [B, 1], prefill is [1, chunk]"
        )
        let effectiveSinks = kind.hasSinks ? sinks : nil

        if B == 1 {
            let (cachedKeys, cachedValues) = rows[0].update(keys: keys, values: values)
            return sdpa(
                queries: queries, keys: cachedKeys, values: cachedValues, scale: scale,
                mask: maskMode(L: L, kL: cachedKeys.dim(2), window: window(of: kind)),
                sinks: effectiveSinks)
        }

        // Batched decode: split queries per row, per-row update + SDPA
        // against that row's own KV, then concatenate. No masks — each row
        // sees exactly its own KV, so batch-composition invariance holds by
        // construction and fully-masked rows cannot exist.
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(B)
        for (index, row) in rows.enumerated() {
            let (cachedKeys, cachedValues) = row.update(
                keys: keys[index ..< (index + 1)],
                values: values[index ..< (index + 1)])
            outputs.append(
                sdpa(
                    queries: queries[index ..< (index + 1)],
                    keys: cachedKeys, values: cachedValues, scale: scale,
                    mask: .none, sinks: effectiveSinks))
        }
        return concatenated(outputs, axis: 0)
    }

    /// Attend against `sourceRows`' KV WITHOUT updating (Gemma-4 cross-layer
    /// KV sharing: shared layers project Q only and borrow the source
    /// layer's K/V — the source layer already appended this step's tokens
    /// earlier in the forward pass).
    static func attendBorrowing(
        sourceRows: [CBv2SequenceKV], sourceKind: CBv2LayerKind, kind: CBv2LayerKind,
        queries: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        let B = queries.dim(0)
        let L = queries.dim(2)
        precondition(!sourceRows.isEmpty, "CBv2AttentionV1: no source rows to borrow from")
        precondition(
            B == sourceRows.count,
            "CBv2AttentionV1: batch \(B) != source rows \(sourceRows.count)")
        precondition(
            B == 1 || L == 1,
            "CBv2AttentionV1: ragged shapes are impossible in v2 — decode is [B, 1], prefill is [1, chunk]"
        )
        let effectiveSinks = kind.hasSinks ? sinks : nil

        if B == 1 {
            let (cachedKeys, cachedValues, _) = sourceRows[0].snapshot()
            return sdpa(
                queries: queries, keys: cachedKeys, values: cachedValues, scale: scale,
                mask: maskMode(L: L, kL: cachedKeys.dim(2), window: window(of: sourceKind)),
                sinks: effectiveSinks)
        }

        var outputs: [MLXArray] = []
        outputs.reserveCapacity(B)
        for (index, row) in sourceRows.enumerated() {
            let (cachedKeys, cachedValues, _) = row.snapshot()
            outputs.append(
                sdpa(
                    queries: queries[index ..< (index + 1)],
                    keys: cachedKeys, values: cachedValues, scale: scale,
                    mask: .none, sinks: effectiveSinks))
        }
        return concatenated(outputs, axis: 0)
    }

    // MARK: - Private

    private static func window(of kind: CBv2LayerKind) -> Int? {
        switch kind.attention {
        case .full: return nil
        case .slidingWindow(let window): return window
        }
    }

    private static func sdpa(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, sinks: MLXArray?
    ) -> MLXArray {
        MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale,
            mask: mask, sinks: sinks)
    }
}
