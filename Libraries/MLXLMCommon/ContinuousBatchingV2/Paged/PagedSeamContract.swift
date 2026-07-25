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
// This file is DECLARATIONS, CONSTANTS AND ADMISSION RULES. It adds no
// backend behaviour and changes none. The only executable code here is the
// arithmetic and the refusals the seam is DEFINED by, which live in one place
// precisely so that two tracks cannot end up maintaining two versions of them:
//
//   * `CBv2PagedSpeculation.maxSpeculativeSpan` — validated against the MTP
//     draft bound on first read, in every build configuration;
//   * `CBv2PagedRingGeometry` — the windowed ring formula, which
//     `PagedKVPool.ringPageCount` must reproduce exactly;
//   * `CBv2PagedWindowSnapshot` — a donated sliding window and the single
//     absolute boundary at which it may be installed.
//
// Every `TODO(track X)` below names the track that supplies the
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
///    (WS-3.1). It does: the landed formula's second term is
///    `ceil(maxSpeculativeSpan / pageSize)` (`PagedKVPool.swift:625-630`),
///    and the construction guard at `PagedKVPool.swift:564-571` refuses to
///    build a pool whose ring cannot hold the attendable span plus one round.
///  - `PagedSequenceKV.supportsSpeculativeWrites` must test against it rather
///    than returning a blanket `windowSize == nil` (WS-3.3). It does, at
///    `PagedSequenceKV.swift:169-171`.
///
/// Derivation: a round writes the target column plus `maxDraftTokens` drafted
/// columns, so the span is `CBv2MTPConfig.testedMaxDraftTokens + 1`. It is
/// stated as a LITERAL rather than computed from the MTP bound, so that
/// raising that bound cannot silently re-size every windowed ring in the
/// process — the memory consequence has to be looked at by a human. The
/// literal is then mechanically checked; see `maxSpeculativeSpan`.
public enum CBv2PagedSpeculation {

    /// The frozen literal. Consumers read `maxSpeculativeSpan`, never this —
    /// that is the property carrying the validation.
    static let declaredSpan = 8

    /// Worst-case positions written beyond the confirmed frontier in one
    /// round.
    ///
    /// **The MTP relationship is enforced HERE, on the shipping path**
    /// (PR#86 review, `:50`). The check used to live in a
    /// `assertSpanCoversMTPBound()` that nothing called, and `assert` is
    /// compiled out under `-O`, so raising `CBv2MTPConfig.testedMaxDraftTokens`
    /// past 7 would have under-sized every windowed ring in a release build
    /// with no diagnostic anywhere. Now the validation runs the first time
    /// ANY ring is sized, because sizing reads this property.
    ///
    /// Swift cannot make it a build-time error. The two operands are separate
    /// `static let`s and the mandatory constant folder does not propagate
    /// across global initialisers, so the usual "force a `UInt` underflow in
    /// a constant expression" trick compiles clean at both `-Onone` and `-O`
    /// (measured on this toolchain, both configurations, with the bound
    /// deliberately violated). A `precondition` in this initialiser is the
    /// strongest mechanism the language actually offers: unlike `assert` it
    /// is live in every configuration except `-Ounchecked`.
    /// `CBv2PagedSeamContractTests.speculativeSpanCoversMTPDraftBound` is the
    /// always-run CI copy, so drift is caught in a test run rather than by a
    /// daemon trapping in the field.
    public static let maxSpeculativeSpan: Int = {
        assertSpanCoversMTPBound()
        return declaredSpan
    }()

    /// `true` while the reserved span still covers a whole MTP round:
    /// `maxDraftTokens` drafted columns plus one target column.
    ///
    /// Reads `declaredSpan`, not `maxSpeculativeSpan`, so the validation can
    /// call it without recursing through the property it validates.
    static var spanCoversMTPBound: Bool {
        declaredSpan >= CBv2MTPConfig.testedMaxDraftTokens + 1
    }

    /// Traps if the MTP draft bound has outgrown the reserved span.
    /// `maxDraftTokens` drafted columns plus one target column.
    ///
    /// A `precondition`, not an `assert`: an under-sized ring is silent KV
    /// corruption, and the only way to reach this condition is a source
    /// change to one of two constants, which CI catches first.
    static func assertSpanCoversMTPBound() {
        precondition(spanCoversMTPBound, spanDriftMessage)
    }

    /// The diagnostic, shared by the runtime check and its test copy so the
    /// two cannot describe the same drift differently.
    static var spanDriftMessage: String {
        """
        CBv2PagedSpeculation.maxSpeculativeSpan (\(declaredSpan)) no longer \
        covers CBv2MTPConfig.testedMaxDraftTokens + 1 \
        (\(CBv2MTPConfig.testedMaxDraftTokens + 1)). Raise the span and re-check \
        PagedKVPool.ringPageCount before raising the MTP draft bound.
        """
    }
}

// MARK: - Windowed ring geometry

/// The windowed ring formula, in code.
///
/// `PagedKVPool.ringPageCount(window:config:)` (`PagedKVPool.swift:625-630`)
/// is the shipping implementation; this is the contract's statement of the
/// same rule. They are bound by
/// `CBv2PagedSeamContractTests.ringFormulaMatchesPagedKVPool`, which compares
/// them across a matrix of windows, chunks and page sizes and fails on any
/// divergence — so the frozen contract and the shipping code cannot drift
/// apart silently, which is exactly what they had done (PR#86 review, `:158`).
///
/// The pool keeps its own copy rather than calling this one because its
/// construction path (`PagedKVPool.checkedRingPageCount`) has to
/// overflow-check every intermediate and turn a hostile operator config into
/// `CBv2KVError.backendIneligible` instead of a trap. This copy is the plain
/// arithmetic that the checked one must agree with.
public enum CBv2PagedRingGeometry {

    /// The widest range a windowed row can be asked to gather at once.
    ///
    /// `PagedSequenceKV.retainedCount` is
    /// `min(written, window - 1 + lastUpdateTokens)`
    /// (`PagedSequenceKV.swift:90-94`), and a prefill chunk of `n` tokens
    /// leaves `lastUpdateTokens == n`. So the widest exposure is
    /// `window - 1 + maxPrefillChunk`, NOT `window`: the chunk's earliest
    /// query at absolute `q` still has to see `[q - window + 1, q]`, and the
    /// range ending at the chunk's LAST position therefore spans
    /// `window - 1 + n`.
    public static func attendableTokens(window: Int, maxPrefillChunk: Int) -> Int {
        window - 1 + maxPrefillChunk
    }

    /// Tokens the ring MUST cover: the attendable range plus one speculative
    /// round. `PagedKVPool.checkedRingPageCount` refuses to build a pool that
    /// does not reach this (`PagedKVPool.swift:564-571`).
    public static func requiredTokens(window: Int, maxPrefillChunk: Int) -> Int {
        attendableTokens(window: window, maxPrefillChunk: maxPrefillChunk)
            + CBv2PagedSpeculation.maxSpeculativeSpan
    }

    /// Ring length in pages:
    ///
    ///     ceil((window - 1 + maxPrefillChunk) / pageSize)
    ///         + ceil(maxSpeculativeSpan / pageSize)
    ///
    /// gemma-4 (window 1,024, chunk 512, pageSize 16, span 8):
    /// `ceil(1535 / 16) + ceil(8 / 16) == 96 + 1 == 97` pages == 1,552 tokens.
    public static func ringPageCount(window: Int, pageSize: Int, maxPrefillChunk: Int) -> Int {
        let attendable = attendableTokens(window: window, maxPrefillChunk: maxPrefillChunk)
        let spanPages = (CBv2PagedSpeculation.maxSpeculativeSpan + pageSize - 1) / pageSize
        return (attendable + pageSize - 1) / pageSize + spanPages
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
// That comment is false, contradicts this file's own header, and contradicts
// `pagedattention.metal`'s "Windowed rings cannot alias" assertion — the
// shader is the correct one.

// MARK: - Windowed prefix adoption (WS-4.1)

/// Why a donated sliding window was refused at a matched boundary. Every case
/// is a normal, expected outcome whose handling is "fall back to replay" —
/// never a trap, never a partial install.
public enum CBv2PagedWindowRestoreRefusal: Error, Equatable, CustomStringConvertible {

    /// The snapshot ends at a different absolute position than the boundary
    /// being adopted. This is the common case and the reason this type
    /// exists: `PrefixCacheV2` can return a boundary far behind the donation
    /// endpoint the window was taken at.
    case boundaryMismatch(snapshotEnd: Int, requested: Int)

    /// Right boundary, wrong extent. A window shorter than
    /// `min(matchedBoundary, window)` is missing its oldest entries, and
    /// those are invisible to attention rather than recoverable by a short
    /// replay; a longer one is not a window at all.
    case inexactWindow(tokens: Int, required: Int)

    /// A full-attention row has no window to restore. Full layers come back
    /// through `PrefixCacheV2`'s per-layer snapshots, not through this seam.
    case notWindowed(requested: Int)

    public var description: String {
        switch self {
        case .boundaryMismatch(let snapshotEnd, let requested):
            return
                "windowed prefix refused: snapshot ends at absolute \(snapshotEnd) but the "
                + "adoption boundary is \(requested); installing it would place the donor's "
                + "keys at the wrong absolute positions"
        case .inexactWindow(let tokens, let required):
            return
                "windowed prefix refused: snapshot carries \(tokens) positions, the boundary "
                + "needs exactly \(required); a partial window is not an exact restore"
        case .notWindowed(let requested):
            return "windowed prefix refused: row at boundary \(requested) has no sliding window"
        }
    }
}

/// A donated sliding window, together with the ONE absolute boundary at which
/// it may be installed.
///
/// **Why this is a type and not three loose arguments** (PR#86 review, `:173`).
/// `PrefixCacheV2` indexes EVERY whole-block boundary of a donation — "Every
/// whole-block boundary of this entry is indexed, so shorter prefixes of a
/// long donation still hit" (`PrefixCacheV2.swift:124-126`) — and its lookup
/// scans longest-to-shortest and returns `matched = k * blockSize` for any `k`
/// up to the entry's block count (`PrefixCacheV2.swift:216-231`). A finished
/// paged row, by contrast, retains only the last `retainedCount` positions
/// ending at its own `absoluteOffset`. So a donation that ended near token
/// 4,096 is ALSO indexed at 1,024, and the trailing window it carries cannot
/// serve that hit.
///
/// Under the previous frozen signature — `restoreWindow(keys:values:base:)`,
/// with `base` an argument taken on trust and no boundary to check it against
/// — an adopter had exactly two ways to proceed and both were wrong:
///
///  * install the payload anyway, writing the donor's positions
///    `[3072, 4096)` into the adopter's `[0, 1024)`. `PagedSequenceKV` stores
///    by absolute position (`gatherRange` maps `p` to ring slot
///    `(p / pageSize) % ringPages`), so nothing downstream can notice: silent
///    wrong answers, no trap, no telemetry; or
///  * replay, which contradicts the same seam entry's "replay bound 25,600
///    tokens -> 0" claim.
///
/// The resolution is the one WS-4.2 reached independently on the provider side
/// (`SSDWindowSidecar.swift`, provider PR#588): the persisted form is
/// PER-BLOCK and content-addressed off the same chain hash as the
/// full-attention block, the window at boundary `M` is assembled from the
/// `W / blockSize` sidecars tiling `[M - W, M)`, and a boundary whose tiling
/// is incomplete is REFUSED rather than partially restored — "a PARTIAL
/// window restore is NOT exact (the missing oldest entries are invisible to
/// attention and cannot be recovered by a short replay ...), so a boundary is
/// adoptable only when EVERY tiling block is present"
/// (`SSDWindowSidecar.swift:61-66`, enforced at `:318` and `:364`, with the
/// authenticated `windowBase` anti-splice check at `:292-303`).
///
/// This type is the engine-side statement of the same rule, and it makes the
/// wrong-absolute-position outcome unrepresentable rather than merely
/// discouraged:
///
///  * `tokens` is read off `keys`, never supplied, so a snapshot cannot claim
///    an extent it does not carry;
///  * `endBoundary` is derived from `base + tokens`, so the position a payload
///    belongs at is a property OF the payload, not of the call;
///  * every install goes through `requireAdmissible(at:window:)`, which
///    refuses any boundary but that one.
public struct CBv2PagedWindowSnapshot {

    /// `[1, kvHeads, tokens, headDim]`, oldest position first.
    public let keys: MLXArray
    /// `[1, kvHeads, tokens, headDim]`, oldest position first.
    public let values: MLXArray
    /// Absolute position of the FIRST retained token.
    public let base: Int
    /// Retained positions. Read off `keys.dim(2)`; never a caller's claim.
    public let tokens: Int

    /// Absolute position one past the last retained token — the only boundary
    /// this payload may be installed at.
    public var endBoundary: Int { base + tokens }

    /// `nil` for anything that is not a well-formed window.
    ///
    /// Deliberately failable rather than trapping: a mis-shaped or corrupt
    /// donation must degrade to replay, which is always safe, and a cache
    /// read is not a place to abort a multi-tenant daemon.
    ///
    /// The argument list is exactly WS-4.1's `windowSnapshot()` tuple, which
    /// is also the provider's `SSDWindowSnapshotting` return type and
    /// `SSDWindowSidecar.Window` (`SSDWindowSidecar.swift:205`, `:392-394`),
    /// so bridging a donor snapshot is
    /// `CBv2PagedWindowSnapshot(keys: w.keys, values: w.values, base: w.base)`.
    public init?(keys: MLXArray, values: MLXArray, base: Int) {
        guard base >= 0,
            keys.ndim == 4, values.ndim == 4,
            keys.dim(0) == 1, values.dim(0) == 1,
            keys.dim(1) == values.dim(1),
            keys.dim(2) == values.dim(2),
            keys.dim(3) == values.dim(3),
            keys.dim(2) > 0
        else { return nil }
        self.keys = keys
        self.values = values
        self.base = base
        self.tokens = keys.dim(2)
    }

    /// Throws unless this payload is an EXACT window for `matchedBoundary` on
    /// a row whose sliding window is `window` (`nil` for full attention).
    ///
    /// Admissible means both of:
    ///
    ///  * `endBoundary == matchedBoundary` — the window was taken at exactly
    ///    the boundary being adopted;
    ///  * `tokens == min(matchedBoundary, window)` — it is the whole window,
    ///    or, for a boundary shorter than one window, the whole history. The
    ///    two together pin `base == matchedBoundary - min(matchedBoundary,
    ///    window)`, so an admissible base is page-aligned whenever
    ///    `matchedBoundary` is, which WS-0.6 guarantees for a matched block
    ///    boundary (`PagedKVPool.swift:340-361`).
    ///
    /// `restoreWindow` MUST call this first and propagate the refusal.
    public func requireAdmissible(at matchedBoundary: Int, window: Int?) throws {
        guard let window, window > 0 else {
            throw CBv2PagedWindowRestoreRefusal.notWindowed(requested: matchedBoundary)
        }
        guard endBoundary == matchedBoundary else {
            throw CBv2PagedWindowRestoreRefusal.boundaryMismatch(
                snapshotEnd: endBoundary, requested: matchedBoundary)
        }
        let required = min(matchedBoundary, window)
        guard tokens == required else {
            throw CBv2PagedWindowRestoreRefusal.inexactWindow(tokens: tokens, required: required)
        }
    }
}

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
//       WS-1.2 + WS-3.1. LANDED, as
//         ceil((window - 1 + maxPrefillChunk) / pageSize)
//             + ceil(maxSpeculativeSpan / pageSize)
//       (`PagedKVPool.swift:625-630`), with the construction guard
//       `ringTokens >= (window - 1 + maxPrefillChunk) + maxSpeculativeSpan`
//       at `PagedKVPool.swift:564-571`. `CBv2PagedRingGeometry` above is this
//       file's copy of the same arithmetic and a test binds the two.
//
//       REJECTED ALTERNATIVE. This entry used to declare
//         ceil(window / pageSize) + ceil(maxSpeculativeSpan / pageSize)
//       with the guard `ringPages * pageSize >= window + maxSpeculativeSpan`,
//       and the shipping code never matched it. It is 65 pages (1,040 tokens)
//       for gemma-4 against an attendable range of 1,535, and it makes a
//       DAEMON ABORT reachable from ordinary prefill — not from MTP, not from
//       an edge case. `PagedSequenceKV.update` writes the chunk and then
//       calls `attendableViews()` (`PagedSequenceKV.swift:100-110`), which
//       gathers `retainedCount == window - 1 + n` ending at the new frontier;
//       `gatherRange` trips
//       `precondition(start >= absoluteOffset - ring, "gather of evicted
//       window range")` (`PagedSequenceKV.swift:401-407`) as soon as that
//       range exceeds the ring, i.e. for any windowed chunk past
//       `pageSize + 1` tokens. `snapshot()` (`:112-115`) and
//       `guardSpeculativeSpan`'s LIVE bound (`:245-262`) read the same
//       `retainedCount` and fail the same way. Reproduced by track R.
//
//       ONE CORRECTION to the review that raised this, recorded so it is not
//       "fixed" back: the mechanism it named — "`PagedLayerCache
//       .updateAndAttend` writes the entire chunk before gathering" — is no
//       longer true. WS-1.2's other half landed: `prefillKV` assembles
//       `gather([q - window + 1, q)) ++ chunk` BEFORE `row.write`
//       (`PagedLayerCache.swift:196-217`, `:510-544`), so the layer's own
//       prefill path needs only `window - 1`. The `maxPrefillChunk` term
//       survives anyway, because the ROW-level `retainedCount` surface above
//       is still post-write. The ring may drop to
//       `ceil(window / pageSize) + ceil(span / pageSize)` only once every
//       consumer of `retainedCount` is pre-write too; until then the two
//       halves are one work item and shipping one alone is the trap.
//       `perSequenceTokenDemand` and `pageDemand` are both
//       `min(..., ringPageCount(...))` and inherit any such reduction for
//       free, so there is nothing to collect at those two sites either.
//   func drainDeferredFrees()
//       WS-3.2c. Releases pages queued during a speculative transaction.
//       Called from the row's `commitSpeculativeWrite()`, never inline in
//       `rollback`, so a page cannot be recycled to another row while a
//       round's captures still name it.
//
//   PagedSequenceKV                                 owner: track R
//   -----------------------------------------------------------------
//   func windowSnapshot() -> (keys: MLXArray, values: MLXArray, base: Int)?
//       WS-4.1. SHAPE UNCHANGED — the provider's `SSDWindowSnapshotting`
//       (PR#588, `SSDWindowSidecar.swift:392-394`) is declared against this
//       exact tuple and conforms `PagedSequenceKV` retroactively, so the
//       return type is load-bearing across two repositories.
//
//       What changed is the CLAIM attached to it. This is the window at the
//       row's OWN donation endpoint: `base` is
//       `absoluteOffset - retainedCount`, and the result is `nil` for
//       full-attention rows. It restores exactly one boundary, `base +
//       tokens`, and no other. It does NOT on its own take gemma-4's replay
//       bound from 25,600 tokens to 0 for an arbitrary hit — it does so for a
//       hit AT the donation endpoint. Coverage of the earlier boundaries
//       `PrefixCacheV2` will actually return comes from WS-4.2's per-block
//       sidecars, whose tiling across successive donations is the whole point
//       of their granularity (`SSDWindowSidecar.swift:39-66`).
//   func restoreWindow(_ snapshot: CBv2PagedWindowSnapshot, at matchedBoundary: Int) throws
//       WS-4.1. Inverse of `windowSnapshot()`, KEYED BY THE BOUNDARY BEING
//       ADOPTED. MUST begin with
//       `try snapshot.requireAdmissible(at: matchedBoundary, window: windowSize)`
//       and propagate the refusal so the caller degrades to replay; it MUST
//       NOT trap and MUST NOT install a partial window. Replaces
//       `restoreWindow(keys:values:base:)`, which took `base` on trust — see
//       `CBv2PagedWindowSnapshot` above for why that was unsound.
//
//       Adoption stays a pointer swap, never a byte copy: an admissible base
//       is `matchedBoundary - min(matchedBoundary, window)`, page-aligned
//       whenever `matchedBoundary` is, and `blockSize 256 % pageSize 16 == 0`
//       makes every matched block boundary a page boundary. That divisibility
//       is no longer a coincidence across two files with no cross-reference:
//       WS-0.6 asserts it unconditionally at pool construction
//       (`PagedKVPool.swift:340-361`).
//   func installShared(_ pages: [Int32], upTo boundary: Int)
//       WS-4.1. Adopts donor pages under refcount. `boundary` is the SAME
//       value passed to `restoreWindow`; a row that restores its window at M
//       and installs shared pages to some other boundary has reintroduced the
//       second cursor this seam exists to remove. With the window restored at
//       M there is no second cursor, so `pagedHybridRequiresDualCursor`
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
