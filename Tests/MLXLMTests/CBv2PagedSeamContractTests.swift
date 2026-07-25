// CBv2PagedSeamContractTests.swift
//
// The mechanical half of `PagedSeamContract.swift`. That file is the frozen
// seam three concurrent tracks build against; this file is what stops it
// drifting away from the code it describes.
//
// Two of these suites exist because the frozen contract HAD drifted (PR#86
// review):
//
//  * `:158` — the contract declared a ring formula
//    (`ceil(window / pageSize) + ceil(span / pageSize)`, 65 pages for
//    gemma-4) that `PagedKVPool` never implemented and that makes an
//    ordinary-prefill daemon abort reachable. Nothing failed when they
//    disagreed. `RingFormula` below fails.
//  * `:173` — the contract froze `restoreWindow(keys:values:base:)`, which
//    cannot tell whether the window it is handed belongs at the boundary
//    being adopted. `WindowSnapshotBoundary` below pins the refusal.
//
// The third (`:50`) is the always-run copy of an invariant that used to be
// guarded by an `assert` inside a function nothing called.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

// MARK: - Ring formula

@Suite("CBv2 paged seam contract: ring formula")
struct CBv2PagedSeamContractRingFormulaTests {

    /// Windows, page sizes and chunks wide enough to cross every rounding
    /// boundary in either expression: exact multiples, one-off-either-side,
    /// a page larger than the window, and a chunk larger than the window.
    private static let windows = [1, 15, 16, 17, 32, 128, 512, 1024, 4096]
    private static let pageSizes = [1, 8, 16, 64, 256]
    private static let chunks = [1, 8, 15, 16, 17, 64, 512, 4096]

    private func config(pageSize: Int, maxPrefillChunk: Int) -> PagedKVPoolConfig {
        // `ringPageCount` reads only `pageSize` and `maxPrefillChunk`; the
        // rest is inert here, and `maxBufferLength` is passed explicitly so
        // the matrix never touches the GPU.
        PagedKVPoolConfig(
            pageSize: pageSize,
            capacityBytes: 1 << 20,
            maxPrefillChunk: maxPrefillChunk,
            nominalMaxSequenceLength: 4096,
            maxBufferLength: 1 << 40)
    }

    /// THE binding test. `CBv2PagedRingGeometry.ringPageCount` is the
    /// contract's statement of the ring formula and
    /// `PagedKVPool.ringPageCount` is the shipping one; if either moves
    /// without the other, this fails.
    @Test func ringFormulaMatchesPagedKVPool() {
        for window in Self.windows {
            for pageSize in Self.pageSizes {
                for chunk in Self.chunks {
                    let cfg = config(pageSize: pageSize, maxPrefillChunk: chunk)
                    let contract = CBv2PagedRingGeometry.ringPageCount(
                        window: window, pageSize: pageSize, maxPrefillChunk: chunk)
                    let shipping = PagedKVPool.ringPageCount(window: window, config: cfg)
                    #expect(
                        contract == shipping,
                        """
                        window \(window), pageSize \(pageSize), chunk \(chunk): the frozen \
                        contract says \(contract) pages, PagedKVPool.ringPageCount computes \
                        \(shipping). PagedSeamContract.swift and PagedKVPool.swift must state \
                        one formula — update both, or neither.
                        """)
                }
            }
        }
    }

    /// The contract's formula must always satisfy the contract's own
    /// construction floor, which is the guard `PagedKVPool
    /// .checkedRingPageCount` enforces at pool build. A formula that can
    /// under-run its own floor would make every pool with that geometry
    /// `backendIneligible` instead of merely under-sized.
    @Test func ringAlwaysCoversAttendableRangePlusOneRound() {
        for window in Self.windows {
            for pageSize in Self.pageSizes {
                for chunk in Self.chunks {
                    let tokens =
                        CBv2PagedRingGeometry.ringPageCount(
                            window: window, pageSize: pageSize, maxPrefillChunk: chunk) * pageSize
                    let required = CBv2PagedRingGeometry.requiredTokens(
                        window: window, maxPrefillChunk: chunk)
                    #expect(
                        tokens >= required,
                        """
                        window \(window), pageSize \(pageSize), chunk \(chunk): ring holds \
                        \(tokens) tokens but must cover \(required) (attendable \
                        \(CBv2PagedRingGeometry.attendableTokens(
                            window: window, maxPrefillChunk: chunk)) + span \
                        \(CBv2PagedSpeculation.maxSpeculativeSpan))
                        """)
                }
            }
        }
    }

    /// The widest range a windowed row exposes is `retainedCount` right after
    /// a bulk write — `window - 1 + maxPrefillChunk`, not `window`. This is
    /// the term the rejected formula dropped, so it is pinned separately from
    /// the page arithmetic that consumes it.
    @Test func attendableSpanCarriesTheWholePrefillChunk() {
        #expect(CBv2PagedRingGeometry.attendableTokens(window: 1024, maxPrefillChunk: 512) == 1535)
        #expect(CBv2PagedRingGeometry.attendableTokens(window: 1, maxPrefillChunk: 1) == 1)
        #expect(
            CBv2PagedRingGeometry.requiredTokens(window: 1024, maxPrefillChunk: 512)
                == 1535 + CBv2PagedSpeculation.maxSpeculativeSpan)
    }

    /// gemma-4's geometry, and the arithmetic that rejected the smaller
    /// formula the contract used to declare.
    ///
    /// 25 of gemma-4's 30 layers are sliding at window 1,024 with pageSize 16
    /// and the default 512-token prefill chunk. The landed ring is 97 pages;
    /// the rejected `ceil(window / pageSize) + ceil(span / pageSize)` is 65.
    /// 65 pages is 1,040 tokens against an attendable range of 1,535, so a
    /// row that writes a full chunk and then gathers `retainedCount` asks
    /// `gatherRange` for a range the ring has already lapped — the "gather of
    /// evicted window range" precondition, i.e. a daemon abort on ordinary
    /// prefill.
    @Test func gemma4RingIsNinetySevenPagesAndTheRejectedFormulaIsTooSmall() {
        let window = 1024
        let pageSize = 16
        let chunk = 512

        let landed = CBv2PagedRingGeometry.ringPageCount(
            window: window, pageSize: pageSize, maxPrefillChunk: chunk)
        #expect(landed == 97, "gemma-4's windowed layers ring at 97 pages (1,552 tokens)")
        #expect(landed * pageSize == 1552)

        // The formula PagedSeamContract.swift:154-158 used to declare.
        let rejected =
            (window + pageSize - 1) / pageSize
            + (CBv2PagedSpeculation.maxSpeculativeSpan + pageSize - 1) / pageSize
        #expect(rejected == 65, "the rejected formula sizes gemma-4's ring at 65 pages")
        #expect(
            rejected * pageSize
                < CBv2PagedRingGeometry.attendableTokens(
                    window: window, maxPrefillChunk: chunk),
            """
            the rejected ring (\(rejected * pageSize) tokens) must be shown SMALLER than the \
            range a row can be asked to gather (\(CBv2PagedRingGeometry.attendableTokens(
                window: window, maxPrefillChunk: chunk))) — that gap is why it was rejected
            """)

        // And it is not a gemma-4 quirk: the rejected formula is too small
        // wherever the chunk outruns one page, which is every default config.
        for chunk in [17, 64, 512, 4096] {
            let small = (window + pageSize - 1) / pageSize + 1
            #expect(
                small * pageSize
                    < CBv2PagedRingGeometry.requiredTokens(
                        window: window, maxPrefillChunk: chunk),
                "chunk \(chunk): the window-only ring cannot hold the attendable range")
        }
    }
}

// MARK: - Speculative span (PR#86 review, PagedSeamContract.swift:50)

@Suite("CBv2 paged seam contract: speculative span")
struct CBv2PagedSeamContractSpanTests {

    /// The invariant the old `assertSpanCoversMTPBound()` was supposed to
    /// hold and could not: nothing called it, and `assert` is compiled out
    /// under `-O`. This runs in every CI configuration.
    @Test func speculativeSpanCoversMTPDraftBound() {
        let drift = Comment(rawValue: CBv2PagedSpeculation.spanDriftMessage)
        #expect(CBv2PagedSpeculation.spanCoversMTPBound, drift)
        #expect(
            CBv2PagedSpeculation.maxSpeculativeSpan >= CBv2MTPConfig.testedMaxDraftTokens + 1,
            drift)
        // Exercise the shipping validation itself, so the function is not
        // dead again the moment someone deletes its one caller.
        CBv2PagedSpeculation.assertSpanCoversMTPBound()
    }

    /// The validated property must expose the frozen literal unchanged —
    /// otherwise every consumer that reads `maxSpeculativeSpan` (ring sizing,
    /// `speculativeHeadroom`, `supportsSpeculativeWrites`) would be sizing
    /// against something the contract never declared.
    @Test func validatedSpanIsTheDeclaredLiteral() {
        #expect(CBv2PagedSpeculation.maxSpeculativeSpan == CBv2PagedSpeculation.declaredSpan)
        #expect(CBv2PagedSpeculation.maxSpeculativeSpan == 8)
    }

    /// The drift message must name both constants, because the person who
    /// trips it is raising one of them and needs to know which other one to
    /// re-check.
    @Test func driftMessageNamesBothConstants() {
        let message = CBv2PagedSpeculation.spanDriftMessage
        #expect(message.contains("maxSpeculativeSpan"))
        #expect(message.contains("testedMaxDraftTokens"))
        #expect(message.contains("ringPageCount"))
    }
}

// MARK: - Window snapshot boundary (PR#86 review, PagedSeamContract.swift:173)

@Suite("CBv2 paged seam contract: window snapshot boundary")
struct CBv2PagedSeamContractWindowSnapshotTests {

    private static let kvHeads = 2
    private static let headDim = 8

    private func snapshot(base: Int, tokens: Int) -> CBv2PagedWindowSnapshot {
        let shape = [1, Self.kvHeads, tokens, Self.headDim]
        let snapshot = CBv2PagedWindowSnapshot(
            keys: MLXArray.zeros(shape, dtype: .float16),
            values: MLXArray.zeros(shape, dtype: .float16),
            base: base)
        // A well-formed payload must construct; a nil here is a test bug.
        return snapshot!
    }

    /// THE regression. A donation that ended near token 4,096 is indexed by
    /// `PrefixCacheV2` at every whole-block boundary it covers, so a later
    /// lookup can legitimately match at 1,024. The trailing window that row
    /// retained holds absolute positions [3072, 4096); installing it at 1,024
    /// would place those keys at [0, 1024) — silently wrong answers, since
    /// paged storage is indexed by absolute position and nothing downstream
    /// can notice. It must be refused.
    @Test func windowSnapshotIsRefusedAtEveryBoundaryButItsOwn() throws {
        let window = 1024
        let donated = snapshot(base: 3072, tokens: window)
        #expect(donated.endBoundary == 4096)

        // Its own boundary: admissible.
        try donated.requireAdmissible(at: 4096, window: window)

        // The earlier boundary the prefix cache can return: refused, naming
        // both positions.
        #expect(
            throws: CBv2PagedWindowRestoreRefusal.boundaryMismatch(
                snapshotEnd: 4096, requested: 1024)
        ) {
            try donated.requireAdmissible(at: 1024, window: window)
        }

        // Every other boundary too, including the snapshot's own base and
        // off-by-ones on either side.
        for boundary in [1, 256, 3072, 4095, 4097, 8192] {
            #expect(
                throws: CBv2PagedWindowRestoreRefusal.boundaryMismatch(
                    snapshotEnd: 4096, requested: boundary)
            ) {
                try donated.requireAdmissible(at: boundary, window: window)
            }
        }
    }

    /// A partial window is refused rather than restored. The missing oldest
    /// entries are invisible to attention — they cannot be recovered by a
    /// short replay — so a half window at the right boundary is still a wrong
    /// answer. This is the rule WS-4.2 enforces on the provider side by
    /// requiring every tiling block to be present.
    @Test func partialWindowIsRefusedAtItsOwnBoundary() {
        let window = 1024
        let partial = snapshot(base: 3584, tokens: 512)
        #expect(partial.endBoundary == 4096)
        #expect(
            throws: CBv2PagedWindowRestoreRefusal.inexactWindow(tokens: 512, required: 1024)
        ) {
            try partial.requireAdmissible(at: 4096, window: window)
        }
    }

    /// An over-long payload is refused for the same reason in reverse: it is
    /// not a window, and installing it would write positions the row's ring
    /// cannot hold at that boundary.
    @Test func overlongWindowIsRefused() {
        let window = 256
        let overlong = snapshot(base: 3072, tokens: 1024)
        #expect(
            throws: CBv2PagedWindowRestoreRefusal.inexactWindow(tokens: 1024, required: 256)
        ) {
            try overlong.requireAdmissible(at: 4096, window: window)
        }
    }

    /// A boundary shorter than one window: the row's whole history IS its
    /// window, so a snapshot based at 0 covering all of it is exact.
    @Test func historyShorterThanOneWindowIsAWholeWindow() throws {
        let window = 1024
        let short = snapshot(base: 0, tokens: 512)
        try short.requireAdmissible(at: 512, window: window)

        // Still keyed by the boundary: the same payload is not admissible
        // anywhere else, and a hole at the front is still refused.
        #expect(
            throws: CBv2PagedWindowRestoreRefusal.boundaryMismatch(
                snapshotEnd: 512, requested: 256)
        ) {
            try short.requireAdmissible(at: 256, window: window)
        }
        let holed = snapshot(base: 8, tokens: 504)
        #expect(
            throws: CBv2PagedWindowRestoreRefusal.inexactWindow(tokens: 504, required: 512)
        ) {
            try holed.requireAdmissible(at: 512, window: window)
        }
    }

    /// Full-attention rows do not come back through this seam at all; their
    /// K/V is restored from `PrefixCacheV2`'s per-layer snapshots. A window
    /// offered to one is refused rather than quietly ignored.
    @Test func fullAttentionRowsHaveNoWindowToRestore() {
        let donated = snapshot(base: 3072, tokens: 1024)
        #expect(throws: CBv2PagedWindowRestoreRefusal.notWindowed(requested: 4096)) {
            try donated.requireAdmissible(at: 4096, window: nil)
        }
        #expect(throws: CBv2PagedWindowRestoreRefusal.notWindowed(requested: 4096)) {
            try donated.requireAdmissible(at: 4096, window: 0)
        }
    }

    /// The extent is read off the payload, never supplied, so a caller cannot
    /// claim a window it does not carry — which is what makes the
    /// wrong-position install unrepresentable rather than merely checked.
    @Test func extentAndBoundaryAreDerivedFromThePayload() {
        for (base, tokens) in [(0, 1), (16, 256), (3072, 1024)] {
            let s = snapshot(base: base, tokens: tokens)
            #expect(s.tokens == tokens)
            #expect(s.keys.dim(2) == tokens)
            #expect(s.endBoundary == base + tokens)
        }
    }

    /// A mis-shaped or corrupt donation yields `nil` instead of trapping: a
    /// cache read must degrade to replay, never abort a multi-tenant daemon.
    @Test func malformedPayloadsAreRefusedWithoutTrapping() {
        let heads = Self.kvHeads
        let dim = Self.headDim
        let ok = MLXArray.zeros([1, heads, 32, dim], dtype: .float16)

        // Negative base.
        #expect(CBv2PagedWindowSnapshot(keys: ok, values: ok, base: -1) == nil)
        // Not 4-D.
        let threeD = MLXArray.zeros([heads, 32, dim], dtype: .float16)
        #expect(CBv2PagedWindowSnapshot(keys: threeD, values: threeD, base: 0) == nil)
        // Batched (a window is one row).
        let batched = MLXArray.zeros([2, heads, 32, dim], dtype: .float16)
        #expect(CBv2PagedWindowSnapshot(keys: batched, values: batched, base: 0) == nil)
        // Keys and values disagreeing on the token extent.
        let shorter = MLXArray.zeros([1, heads, 16, dim], dtype: .float16)
        #expect(CBv2PagedWindowSnapshot(keys: ok, values: shorter, base: 0) == nil)
        // Keys and values disagreeing on head geometry.
        let wideHead = MLXArray.zeros([1, heads, 32, dim * 2], dtype: .float16)
        #expect(CBv2PagedWindowSnapshot(keys: ok, values: wideHead, base: 0) == nil)
        // Empty window.
        let empty = MLXArray.zeros([1, heads, 0, dim], dtype: .float16)
        #expect(CBv2PagedWindowSnapshot(keys: empty, values: empty, base: 0) == nil)
    }
}
