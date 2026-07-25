// PagedSequenceKV.swift
//
// Per-sequence, per-layer KV state for the paged backend (WS-C).
//
// A `PagedSequenceKV` owns a page TABLE (`[Int32]` physical page ids into
// its group's slabs) plus its absolute position counter. There is no shared
// frontier: joining a batch adds an object, leaving drops it, and nothing a
// batchmate does can move this sequence's positions or evict its pages.
//
// Windowed layers use a RING of pages: token at absolute position `p` lives
// in table slot `(p / pageSize) % ringPages`, slot `p % pageSize`. Eviction
// is therefore implicit overwrite keyed to absolute positions — the recent
// end always survives, no window masks over shared buffers exist, and the
// ring is sized (`PagedKVPool.ringPageCount`) to cover everything the row
// can be asked for at once: the trailing window, the extra `chunk - 1`
// positions a prefill chunk's earliest query still needs (see
// `retainedCount`), and one speculative span (`CBv2PagedSpeculation
// .maxSpeculativeSpan`) so an MTP round overwrites only already-evicted
// slots.

import Foundation
import MLX

public final class PagedSequenceKV: CBv2SequenceKV, CBv2PagedSpeculativeRow {
    let pool: PagedKVPool
    let groupKey: PagedKVGroupKey
    /// Pool-issued monotonic identity (never reused, unlike a heap address).
    /// Device block-table caches fingerprint rows by (serial, tableVersion)
    /// so a new row can never alias a released one.
    let serial: UInt64
    /// Sliding window in tokens (nil == full attention).
    public let windowSize: Int?
    /// Ring capacity in pages for windowed layers (nil for full).
    let ringPages: Int?
    /// Worst-case pages this sequence may allocate (reserved at admission;
    /// the pool guarantees allocations up to this count cannot fail).
    let reservedPages: Int
    /// Hard bound on `absoluteOffset` (from the request's maxLength).
    public let maxLength: Int

    /// Physical page ids. Full: index == logical page. Windowed: index ==
    /// logical page % ringPages.
    private(set) var table: [Int32] = []
    /// Bumped whenever `table` changes — lets the layer cache reuse device
    /// block tables across steps that only append tokens within a page.
    private(set) var tableVersion: Int = 0

    public private(set) var absoluteOffset: Int = 0
    /// Absolute position before which nothing was ever written here
    /// (nonzero only after `fastForward(to:)` for prefix-adopted rows).
    private(set) var baseOffset: Int = 0
    /// Size of the most recent update — governs how much trailing context
    /// `retainedCount` exposes for windowed layers (a chunk's earliest query
    /// must see its full window).
    private var lastUpdateTokens: Int = 1

    /// Absolute frontier as of `beginSpeculativeWrite()`; `nil` outside a
    /// speculative transaction. `rollback` may not cross it.
    private(set) var speculativeBase: Int?
    /// `lastUpdateTokens` as of `beginSpeculativeWrite()`, restored when a
    /// round leaves no surviving write.
    private var speculativeUpdateTokens: Int = 1

    private var released = false

    init(
        pool: PagedKVPool, kind: CBv2LayerKind, maxLength: Int, reservedPages: Int
    ) {
        self.pool = pool
        self.groupKey = PagedKVGroupKey(kind)
        self.serial = pool.nextRowSerial()
        switch kind.attention {
        case .full:
            self.windowSize = nil
            self.ringPages = nil
        case .slidingWindow(let window):
            self.windowSize = window
            self.ringPages = PagedKVPool.ringPageCount(window: window, config: pool.config)
        }
        self.maxLength = maxLength
        self.reservedPages = reservedPages
    }

    deinit {
        assert(released, "[PagedSequenceKV] leaked without release — pages still allocated")
    }

    // MARK: - CBv2SequenceKV

    public var retainedCount: Int {
        let written = absoluteOffset - baseOffset
        guard let window = windowSize else { return written }
        return min(written, window - 1 + lastUpdateTokens)
    }

    public var byteCount: Int {
        table.count * pool.group(groupKey).pageBytes
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        var k = keys
        var v = values
        if k.ndim == 4 {
            precondition(k.dim(0) == 1, "per-sequence update requires batch dim 1")
            k = k.squeezed(axis: 0)
            v = v.squeezed(axis: 0)
        }
        write(keys: k, values: v)
        return attendableViews()
    }

    public func snapshot() -> (keys: MLXArray, values: MLXArray, offset: Int) {
        let (k, v) = attendableViews()
        return (k, v, absoluteOffset)
    }

    // MARK: - Speculative (MTP) transaction

    /// Positions this row can write past its confirmed frontier before a
    /// rollback would stop being value-exact.
    ///
    /// WINDOWED rows: the page ring aliases at `ringPages * pageSize`, which
    /// is strictly GREATER than `window` — `PagedKVPool.ringPageCount`
    /// reserves the window plus a speculative span (and must also cover a
    /// prefill chunk; see the caveat below). A write past the frontier
    /// therefore overwrites slots whose positions were `ringPages * pageSize`
    /// behind it: already evicted, outside every attendable window,
    /// unreachable by any later read. The slack between that alias distance
    /// and the window IS the headroom. Same argument as this file's header
    /// and as `pagedattention.metal` ("Windowed rings cannot alias: the
    /// overwritten slot held a position >= ring*S older than the query,
    /// outside every attendable window").
    ///
    /// CAVEAT — this is the STATIC headroom, measured against `window`, and
    /// it is the formula frozen in `CBv2PagedSpeculativeRow`. The range a row
    /// actually exposes is `retainedCount`, which is `window - 1 +
    /// lastUpdateTokens` right after a bulk write, so the LIVE headroom can
    /// be smaller. `guardSpeculativeSpan` checks the live figure at each
    /// write; this property answers the planning-time question "is this row
    /// speculation-capable at all", which is a property of the ring, not of
    /// the row's current phase.
    ///
    /// FULL rows: no alias distance exists at all — every logical page owns
    /// its own table slot, and `rollback` frees the pages past the new
    /// frontier while every read stays bounded by `absoluteOffset`.
    public var speculativeHeadroom: Int {
        guard let window = windowSize, let ring = ringPages else { return .max }
        return ring * pool.config.pageSize - window
    }

    /// Eligibility is DERIVED from `speculativeHeadroom` — never from the
    /// attention kind.
    ///
    /// The gate this replaced was `windowSize == nil`, justified by a claim
    /// that the paged ring aliases within its window and so destroys the
    /// oldest in-window entries. That claim was false. It transplanted the
    /// CONTIGUOUS ring's problem (`CBv2WindowedSequenceKV` holds exactly
    /// `window` slots, so it genuinely aliases at `window` and genuinely
    /// needs data staging) onto the paged ring, which holds
    /// `ringPages * pageSize` slots — always at least a whole speculative
    /// span more than the window. Nothing a round overwrites is still
    /// readable, so the paged transaction is bookkeeping only: no staging
    /// buffer exists or is needed.
    ///
    /// Comparing against `CBv2PagedSpeculation.maxSpeculativeSpan` rather
    /// than against zero is what keeps the ring sizing and the MTP draft
    /// bound coupled: raise the span past what the ring reserves and rows go
    /// INELIGIBLE, instead of silently corrupting confirmed history.
    public var supportsSpeculativeWrites: Bool {
        speculativeHeadroom >= CBv2PagedSpeculation.maxSpeculativeSpan
    }

    public func beginSpeculativeWrite() {
        precondition(
            speculativeBase == nil,
            "[PagedSequenceKV] beginSpeculativeWrite while already armed")
        speculativeBase = absoluteOffset
        speculativeUpdateTokens = lastUpdateTokens
    }

    public func commitSpeculativeWrite() {
        guard let base = speculativeBase else { return }
        speculativeBase = nil
        // Pages `rollback` took out of the table were only QUEUED: the
        // round's gathers are lazy and still name those physical pages, so
        // handing them back before the round closes lets another row
        // allocate and rewrite them underneath a live capture. This is the
        // first moment no capture can reach them.
        pool.drainDeferredFrees()
        settleUpdateTokens(base: base)
    }

    /// Restore the `retainedCount` input at the end of a round.
    ///
    /// `retainedCount` is `min(written, window - 1 + lastUpdateTokens)`. A
    /// bulk `write()` inside a transaction legitimately raises the input to
    /// `n` so the chunk's earliest query still sees its full window, but
    /// leaving it there afterwards over-exposes the row by `n - 1` against
    /// the plain run, which produced the same confirmed tokens one decode at
    /// a time. So: `1` when any write of this round survived (decode-shaped,
    /// exposing exactly `window`), otherwise the value the round opened with.
    private func settleUpdateTokens(base: Int) {
        lastUpdateTokens = absoluteOffset > base ? 1 : speculativeUpdateTokens
    }

    /// Trap before a transaction writes further past its base than the ring
    /// can absorb.
    ///
    /// Two bounds, because `speculativeHeadroom` is measured against `window`
    /// while the range the row actually exposes is `retainedCount`, which is
    /// `window - 1 + lastUpdateTokens` immediately after a bulk write:
    ///
    ///  - STATIC: the span must fit the ring's slack over the window. This is
    ///    the eligibility bound, and a round already passed it at planning
    ///    time; it fires only for an over-long round.
    ///  - LIVE: the span must not reach the oldest position the row can still
    ///    be asked for. Writing position `p` destroys whatever held
    ///    `p - ringPages * pageSize`, so the round may not extend past
    ///    `absoluteOffset - retainedCount + ringPages * pageSize`. Under a
    ///    ring that covers `window - 1 + maxPrefillChunk + span` this cannot
    ///    fire; under an under-sized one it fires HERE, naming the cause,
    ///    instead of surfacing later as a wrong-KV answer or as a bare
    ///    "gather of evicted window range" from `gatherRange`.
    ///
    /// The LIVE bound is not belt-and-braces, and the residual slack is much
    /// thinner than it looks. `ringPageCount` reserves
    /// `ceil((window - 1 + maxPrefillChunk) / pageSize) + ceil(span / pageSize)`
    /// pages, so for gemma-4 (window 1024, chunk 512, page 16) the ring is
    /// 1,552 tokens and the margin over the LIVE attendable range depends
    /// entirely on what the row last did:
    ///
    ///      at MTP time      retainedCount 1,024  ->  margin 528
    ///      during a chunk   retainedCount 1,535  ->  margin  17
    ///
    /// The 17 is the number that matters: prefill is the path that runs
    /// constantly, and it is where the ring is nearly exact. Neither figure
    /// is asserted anywhere — both fall out of chunk sizing and page
    /// rounding, so a change to `maxPrefillChunk`, `pageSize` or the window
    /// moves them silently. The scheduled `gather(ring) ++ chunk` rework
    /// takes the ring down to the window proper, at which point the 528
    /// disappears too. Treat a violation here as a real defect, never as
    /// something the slack can absorb.
    private func guardSpeculativeSpan(adding n: Int) {
        guard let base = speculativeBase else { return }
        let span = absoluteOffset + n - base
        precondition(
            span <= speculativeHeadroom,
            "[PagedSequenceKV] speculative span \(span) exceeds headroom "
                + "\(speculativeHeadroom) — rollback would not be value-exact")
        guard let ring = ringPages else { return }
        let ringTokens = ring * pool.config.pageSize
        let oldestAttendable = absoluteOffset - retainedCount
        precondition(
            absoluteOffset + n - ringTokens <= oldestAttendable,
            "[PagedSequenceKV] speculative write to \(absoluteOffset + n - 1) would evict "
                + "position \(absoluteOffset + n - 1 - ringTokens), still attendable from "
                + "\(oldestAttendable) — ring \(ringTokens) does not cover retainedCount "
                + "\(retainedCount) plus the round. The ring holds the attendable range "
                + "plus one speculative span. Any apparent slack is an accident of chunk "
                + "sizing that nothing asserts — on gemma-4 it is 528 tokens at MTP time "
                + "but only 17 during a prefill chunk, and the scheduled ring rework "
                + "removes both. Re-size the ring or shorten the round; do not assume "
                + "there is room to spare.")
    }

    public func rollback(_ n: Int) {
        precondition(n >= 0 && n <= absoluteOffset - baseOffset, "rollback past written tokens")
        if let base = speculativeBase {
            // Confirmed history is NOT recoverable by rewinding the counter:
            // the round's writes already overwrote the ring slots behind the
            // base, and a full row's pages past its frontier are queued for
            // release. Crossing the base would re-expose those slots as if
            // they still held their original positions.
            precondition(
                absoluteOffset - n >= base,
                "[PagedSequenceKV] rollback into confirmed history "
                    + "(\(absoluteOffset) - \(n) < speculative base \(base))")
        }
        guard n > 0 else { return }
        absoluteOffset -= n
        if ringPages == nil {
            // Full layer: release pages past the new frontier. Freed pages
            // may hold stale bytes; they are only ever re-read by their NEXT
            // owner after that owner writes them (all reads are bounded by
            // the owner's absoluteOffset), so no scrubbing pass is needed.
            let keepPages = (absoluteOffset + pool.config.pageSize - 1) / pool.config.pageSize
            if keepPages < table.count {
                if speculativeBase == nil {
                    pool.freePages(group: groupKey, pages: table[keepPages...])
                } else {
                    // Inside a round the release is DEFERRED to commit; see
                    // `commitSpeculativeWrite`. The pages stay out of the
                    // free list until then, so `ensurePage` will draw fresh
                    // ones — which cannot breach the reservation, because a
                    // transaction never writes after its rollback (the engine
                    // rolls back at finalize, EngineLoopV2+MTPFinalize).
                    pool.deferFreePages(group: groupKey, pages: table[keepPages...])
                }
                table.removeSubrange(keepPages...)
                tableVersion += 1
            }
        }
        // Windowed rings keep their pages; the rolled-back slots are
        // overwritten before they can re-enter any attendable range.
        if let base = speculativeBase {
            settleUpdateTokens(base: base)
        } else {
            lastUpdateTokens = 1
        }
    }

    // MARK: - Writes

    /// Append `keys`/`values` (`[kvHeads, n, headDim]`) at the current
    /// frontier via ONE in-place write-kernel dispatch. Never throws: the
    /// pool guarantees pages up to the admission-time reservation.
    func write(keys: MLXArray, values: MLXArray) {
        let n = keys.dim(1)
        guard n > 0 else { return }
        precondition(
            absoluteOffset + n <= maxLength,
            "write past maxLength (\(absoluteOffset) + \(n) > \(maxLength))")
        if windowSize != nil {
            precondition(
                n <= pool.config.maxPrefillChunk,
                "windowed update of \(n) tokens exceeds maxPrefillChunk "
                    + "\(pool.config.maxPrefillChunk) — the ring cannot hold it")
        }
        guardSpeculativeSpan(adding: n)
        let s = pool.config.pageSize

        // Physical destination (page * pageSize + slot) per token.
        var slots = [Int32]()
        slots.reserveCapacity(n)
        for i in 0 ..< n {
            let pos = absoluteOffset + i
            let physical = ensurePage(logicalPage: pos / s)
            slots.append(physical * Int32(s) + Int32(pos % s))
        }
        pool.writeTokens(group: groupKey, slots: slots, keys: keys, values: values)

        absoluteOffset += n
        lastUpdateTokens = n
    }

    /// Reserve the destination for ONE decode token and advance the
    /// frontier WITHOUT touching storage — the paged decode kernel writes
    /// the tile in place (fused write, see PagedAttentionKernel.decode).
    /// Host Int math only; never a device sync.
    func prepareDecodeWrite() -> (page: Int32, slot: Int) {
        precondition(
            absoluteOffset + 1 <= maxLength,
            "write past maxLength (\(absoluteOffset) + 1 > \(maxLength))")
        guardSpeculativeSpan(adding: 1)
        let s = pool.config.pageSize
        let page = ensurePage(logicalPage: absoluteOffset / s)
        let slot = absoluteOffset % s
        absoluteOffset += 1
        lastUpdateTokens = 1
        return (page, slot)
    }

    private func ensurePage(logicalPage lp: Int) -> Int32 {
        let slotIndex: Int
        if let ring = ringPages {
            slotIndex = lp % ring
        } else {
            slotIndex = lp
        }
        while table.count <= slotIndex {
            precondition(
                table.count < reservedPages,
                "[PagedSequenceKV] allocation past reservation (\(reservedPages) pages) — "
                    + "admission accounting bug")
            table.append(pool.allocatePage(group: groupKey))
            tableVersion += 1
        }
        return table[slotIndex]
    }

    // MARK: - Reads

    /// Gathered `[1, kvHeads, retainedCount, headDim]` views over the
    /// attendable range, oldest to newest, in the pool dtype. The views
    /// are lazy reads of the live slabs — consume them within the current
    /// engine step and drop them: the slabs are mutated in place, so a
    /// stale unevaluated gather could observe recycled pages.
    func attendableViews() -> (keys: MLXArray, values: MLXArray) {
        let retained = retainedCount
        let start = absoluteOffset - retained
        return gatherRange(start: start, count: retained)
    }

    /// Gather an arbitrary written range (used by prefill fallback and
    /// snapshots). `start` is an absolute position.
    func gatherRange(start: Int, count: Int) -> (keys: MLXArray, values: MLXArray) {
        guard count > 0 else {
            return pool.gather(group: groupKey, pages: [], firstSlot: 0, count: 0)
        }
        precondition(start >= baseOffset && start + count <= absoluteOffset)
        if let window = windowSize {
            // The ring only holds the recent end; older positions are gone.
            let ring = ringPages! * pool.config.pageSize
            precondition(
                start >= absoluteOffset - ring,
                "gather of evicted window range (window \(window))")
        }
        let s = pool.config.pageSize
        let lpFirst = start / s
        let lpLast = (start + count - 1) / s
        var pages: [Int32] = []
        pages.reserveCapacity(lpLast - lpFirst + 1)
        for lp in lpFirst ... lpLast {
            let idx = ringPages.map { lp % $0 } ?? lp
            pages.append(table[idx])
        }
        return pool.gather(
            group: groupKey, pages: pages, firstSlot: start % s, count: count)
    }

    /// Modular table length the decode kernel divides by to resolve a
    /// logical page to a physical one (`phys = table[logicalPage % len]`).
    ///
    /// WINDOWED rows MUST report the FULL ring length, never `table.count`.
    /// Pages are placed at `logicalPage % ringPages` (`ensurePage`), but the
    /// physical table is grown LAZILY only up to the slots the writes/replay
    /// touched. A prefix-adopted row (`fastForward` then a trailing replay
    /// that covers less than the ring) can leave `table.count < ringPages`;
    /// feeding that as the divisor makes the kernel wrap at the wrong length
    /// and alias the WRONG physical pages during decode. The ring length is
    /// the divisor the writes used, so it is the only correct divisor.
    /// Every position the decode actually attends was written, so its ring
    /// slot (`< table.count`) is allocated — the larger divisor never indexes
    /// an unallocated slot.
    ///
    /// FULL rows keep `table.count` (identity modulo: `table.count` already
    /// exceeds every logical page a full row can reach).
    var decodeTableLength: Int { ringPages ?? table.count }

    /// Kernel-facing row descriptor for decode: the absolute range the
    /// current query may attend to, plus the (modular) table length.
    /// All plain Swift Int math — never a device sync.
    var decodeAttendRange: (start: Int, length: Int) {
        let qPos = absoluteOffset - 1
        precondition(qPos >= baseOffset, "decode before any token was written")
        var start = baseOffset
        if let window = windowSize {
            start = max(start, qPos - window + 1)
        }
        return (start, qPos - start + 1)
    }

    // MARK: - Lifecycle

    /// Advance the position counter without writing storage. Only valid on
    /// a fresh WINDOWED state during prefix-cache adoption: the engine
    /// recomputes the trailing window tokens for windowed layers, and those
    /// recomputed tokens must land at their true absolute positions.
    public func fastForward(to offset: Int) {
        precondition(windowSize != nil, "fastForward is only for windowed layers")
        precondition(
            table.isEmpty && absoluteOffset == 0 && baseOffset == 0,
            "fastForward requires a fresh state")
        precondition(offset >= 0 && offset <= maxLength)
        absoluteOffset = offset
        baseOffset = offset
    }

    /// Return every page to the pool and release the reservation.
    /// O(pages) metadata, no device work. Idempotent.
    ///
    /// RECYCLE INVARIANT: the freed pages may be handed to a new row whose
    /// bulk writes have no graph edge to THIS row's in-flight reads (the
    /// no-write kernel variants and gathers consume fences but never
    /// advance them). Callers must therefore only release a row after its
    /// last consuming step has been host-synced (the engine's finalize
    /// discipline — releases happen on the engine thread between steps).
    /// A release-without-host-sync fast path would race silently.
    func releaseStorage() {
        guard !released else { return }
        released = true
        if speculativeBase != nil {
            // Released mid-round: the row finished in flight, so the deferred
            // release fence tears it down and `commitSpeculativeWrite` never
            // runs. Draining here is what keeps its queued pages from being
            // orphaned, and it is safe under the RECYCLE INVARIANT above —
            // releases only happen between host-synced steps, so no lazy
            // capture can still name a queued page.
            speculativeBase = nil
            pool.drainDeferredFrees()
        }
        pool.freePages(group: groupKey, pages: table)
        table.removeAll()
        tableVersion += 1
        pool.unreserve([groupKey: reservedPages])
    }
}
