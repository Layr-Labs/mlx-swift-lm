// PagedSeamContract.swift
//
// The frozen seam between the three hottest files of the paged backend:
//
//     PagedKVPool  <->  PagedSequenceKV  <->  PagedLayerCache
//
// Five work items want `PagedLayerCache`, five want `PagedKVPool`, and three
// want `PagedSequenceKV`. Those tracks are developed concurrently, so the
// signatures and shared constants they depend on are declared HERE, once,
// before any of them starts. A track that invents its own version of anything
// in this file will collide at integration.
//
// This file is DECLARATIONS AND CONSTANTS ONLY. It adds no behaviour and
// changes none. Every `TODO(track X)` below names the track that supplies the
// implementation; until then the protocol simply has no paged conformer,
// which is legal and compiles.

import Foundation
import MLX

// MARK: - Speculative span

/// The maximum number of token positions a single speculative round can write
/// past a row's confirmed frontier.
///
/// **This constant exists so that ring sizing and headroom checking cannot
/// disagree.** Two work items consume it and they live in different files:
///
///  - `PagedKVPool.ringPageCount` must reserve space for it, so that a
///    windowed ring can never alias a live in-window entry during a round
///    (WS-3.1). Today `ringPageCount` is `window + maxPrefillChunk`, which
///    happens to leave ~528 tokens of slack for gemma-4 — an accident, not an
///    invariant, and one that WS-1.2's ring shrink deliberately removes.
///  - `PagedSequenceKV.supportsSpeculativeWrites` must test against it rather
///    than returning a blanket `windowSize == nil` (WS-3.3).
///
/// Derivation: a round writes the target column plus `maxDraftTokens` drafted
/// columns, so the span is `CBv2MTPConfig.testedMaxDraftTokens + 1`. It is
/// stated as a literal with a static assertion rather than computed, so that
/// raising the MTP bound cannot silently under-size a ring that was built
/// before the raise.
public enum CBv2PagedSpeculation {
    /// Worst-case positions written beyond the confirmed frontier in one round.
    public static let maxSpeculativeSpan = 8

    /// Fails the build if the MTP draft bound outgrows the reserved span.
    /// `maxDraftTokens` drafted columns plus one target column.
    static func assertSpanCoversMTPBound() {
        assert(
            maxSpeculativeSpan >= CBv2MTPConfig.testedMaxDraftTokens + 1,
            """
            CBv2PagedSpeculation.maxSpeculativeSpan (\(maxSpeculativeSpan)) no longer \
            covers CBv2MTPConfig.testedMaxDraftTokens + 1 \
            (\(CBv2MTPConfig.testedMaxDraftTokens + 1)). Raise the span and re-check \
            PagedKVPool.ringPageCount before raising the MTP draft bound.
            """)
    }
}

// MARK: - Rectangular MTP verification

/// Opt-in marker for a layer cache that can have its attention serialised
/// per query during MTP rectangular verification.
///
/// **Why this protocol exists.** `EngineLoopV2+MTPTargetVerification` currently
/// reaches the `mtpSerializesRectangularAttention` flag through
/// `as? CBv2LayerCache`, and traps with `preconditionFailure` when the cast
/// fails. `CBv2LayerCache` is `final` and `PagedLayerCache` is a *sibling*
/// conformer of `CBv2AttendingLayerCache`, not a subclass, so the cast can
/// never succeed for a paged bank. That trap is a `fatalError`: it kills the
/// provider daemon, taking every co-resident model's in-flight requests with
/// it, and it emits no telemetry.
///
/// It is unreachable today only because gemma-4's windowed paged rows fail the
/// storage-eligibility gate first and no round is ever built. WS-3.3 removes
/// that gate — so WS-3.3 without WS-3.4 converts a silent no-op into a process
/// abort. They must land together.
///
/// **Contract.** Setting the flag to `true` obliges the cache to attend one
/// query position at a time for the duration of the round, so that each
/// column is bit-identical to that column run as a standalone `L == 1` decode.
/// Rectangular verification does NOT require batched multi-query attention:
/// the contiguous implementation already serialises attention and batches only
/// the weight-bound model body across the `1 + k` columns. A paged conformer is
/// therefore a column loop over the existing decode dispatch, not a new kernel.
///
/// Callers MUST degrade to serial verification for a cache that does not
/// conform, and MUST NOT trap.
protocol CBv2MTPRectangularSerializing: AnyObject {
    /// While `true`, attention is computed one query position at a time.
    /// Set for the duration of a rectangular verification round and cleared
    /// in a `defer`.
    var mtpSerializesRectangularAttention: Bool { get set }
}

/// The contiguous cache already owns the stored flag (`LayerCacheV2.swift`),
/// so conformance is declaration-only.
extension CBv2LayerCache: CBv2MTPRectangularSerializing {}

// TODO(track L, WS-3.4): conform `PagedLayerCache` once `updateAndAttend`
// honours the flag with a per-column loop, replacing the
// `precondition(b == 1, "prefill chunks are per-request [1, chunk]")` guard.
// `attendBorrowing` needs the same loop plus an explicit per-column `qPos` on
// `dispatchDecode`, because the source layer has already advanced
// `absoluteOffset` by L and `decodeAttendRange` would otherwise resolve only
// the final column.

// MARK: - Row-side speculative transaction

/// A sequence row that can stage and roll back speculative writes.
///
/// The contiguous windowed ring already implements this behaviour behind
/// `CBv2SequenceKV`'s `beginSpeculativeWrite()` / `commitSpeculativeWrite()`
/// (which are protocol requirements with default no-ops, so a paged row
/// silently inherits the no-ops today). This protocol adds only the part that
/// does not exist anywhere yet: a way to ASK a row how much speculative room
/// it has, rather than assuming.
///
/// **Paged rows need no data staging.** A windowed paged ring aliases at
/// `ringPages * pageSize`, not at `window`, so a rolled-back speculative write
/// destroys only entries already outside every attendable range — provided the
/// ring reserves `CBv2PagedSpeculation.maxSpeculativeSpan`. The transaction is
/// therefore pure bookkeeping: a speculative base, a tightened rollback
/// precondition, deferred page frees, and restoration of the retained-count
/// input.
protocol CBv2PagedSpeculativeRow: AnyObject {
    /// Positions this row can write past its confirmed frontier without
    /// destroying an entry that is still attendable.
    ///
    /// Windowed rows: `ringPages * pageSize - window`. Full rows: unbounded in
    /// practice, reported as `Int.max`. A row is speculation-eligible when this
    /// is `>= CBv2PagedSpeculation.maxSpeculativeSpan`.
    var speculativeHeadroom: Int { get }
}

// TODO(track R, WS-3.2/3.3): conform `PagedSequenceKV`, then redefine
// `supportsSpeculativeWrites` as
// `speculativeHeadroom >= CBv2PagedSpeculation.maxSpeculativeSpan`
// and delete the comment above it claiming the ring aliases within the window.
// That comment is false, contradicts this file's own header 97 lines above it,
// and contradicts `pagedattention.metal`'s "Windowed rings cannot alias"
// assertion — the shader is the correct one.

// MARK: - Frozen signatures for the tracks

// The remaining seam members are added to the CONCRETE types by their owning
// track. They are listed here so three concurrent authors agree on names,
// argument labels and return shapes before writing any of them. Changing an
// entry below requires updating this file first, which produces a review
// conflict rather than a silent integration failure.
//
//   PagedKVPool                                     owner: track P
//   -----------------------------------------------------------------
//   static func ringPageCount(window:config:) -> Int
//       WS-1.2 + WS-3.1. Becomes
//         ceil(window / pageSize) + ceil(maxSpeculativeSpan / pageSize)
//       replacing `window + maxPrefillChunk`, and gains a construction guard
//       `ringPages * pageSize >= window + maxSpeculativeSpan`.
//       MUST NOT land before WS-3.5 — shrinking the ring collapses the margin
//       that currently keeps the MTP lazy-gather hazard latent.
//   func drainDeferredFrees()
//       WS-3.2c. Releases pages queued during a speculative transaction.
//       Called from the row's `commitSpeculativeWrite()`, never inline in
//       `rollback`, so a page cannot be recycled to another row while a
//       round's captures still name it.
//
//   PagedSequenceKV                                 owner: track R
//   -----------------------------------------------------------------
//   func windowSnapshot() -> (keys: MLXArray, values: MLXArray, base: Int)?
//       WS-4.1. The sliding window's contents plus the absolute position of
//       its first token. `nil` for full-attention rows. This is what lets an
//       adopter restore a donor's window instead of replaying it, which is the
//       change that takes gemma-4's replay bound from 25,600 tokens to 0.
//   func restoreWindow(keys:values:base:)
//       WS-4.1. Inverse of `windowSnapshot()`. Page-aligned by construction:
//       `blockSize 256 % pageSize 16 == 0`, so post-adoption writes start at
//       slot 0 and adoption is a pointer swap, never a byte copy. WS-0.6 must
//       assert that divisibility first — it currently holds by coincidence
//       across two files with no cross-reference.
//   func installShared(_ pages: [Int32], upTo boundary: Int)
//       WS-4.1. Adopts donor pages under refcount. With the window restored
//       there is no second cursor, so `pagedHybridRequiresDualCursor`
//       evaporates rather than being solved.
//
//   PagedLayerCache                                 owner: track L
//   -----------------------------------------------------------------
//   prefillAttend                                   WS-0.2p
//       Gains an internal query-block loop. Three constraints, each of which a
//       naive port gets wrong:
//         1. Do NOT call `CBv2AttentionV1.attendQueryBlocks`. It is `private`,
//            and its `maskMode` returns symbolic `.causal`, which violates the
//            pinned-path contract in this file's header ("always `.array`").
//         2. Keep the gather HOISTED. Per-block visible spans overlap by
//            `window - 1`, so gathering per block is a 3x pessimisation on
//            sliding layers and 4x on full layers.
//         3. Build the position vectors ONCE per chunk and slice per block.
//            Rebuilding them inside the loop regresses host `arange` work 4x
//            on full layers, because each block's span is nearly the whole
//            history.
//       Reuse `CBv2AttentionV1.queryBlockSize` and `.shouldBlockQueries`
//       (both already internal) so one kill switch,
//       `DARKBLOOM_CBV2_ATTN_QUERY_BLOCK`, covers both backends.
//       MUST land before the `b == 1` precondition is lifted: unblocked packed
//       prefill puts B gathers and B score tensors live simultaneously behind
//       `concatenated(axis: 0)`, which exceeds the activation reserve on a
//       single layer at B=8.
//   func bindSpanContext(_ context: CBv2SpanChunkContext?)
//       WS-2.2. Paged already builds its mask in absolute coordinates, so the
//       vision span overlay composes onto it more directly than it does onto
//       contiguous's symbolic mode. Note the span path stays UNBLOCKED for
//       now, so it does not inherit WS-0.2p's memory bound.
